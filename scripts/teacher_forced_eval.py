"""Teacher-forced regeneration of the first seconds of a dataset with a trained model, and MSS / KAD against the originals.

Generic over codecs and both generation modes: the codec, the mode and the geometry are read from
<model-dir>/checkpoints/inference.gin. Every unit of the model (an atom, or an AR block) is generated from the REAL past
(the stored latents of the original) and the REAL CLAP embedding (the stored one, aligned to the unit as in training), so
no generated latent is fed back. Atoms are rendered with the same overlap-add as normal generation, AR blocks go through
the codec's stateful streaming decoder.

Usage (from SCAPES/):
    ../venv_SCAPES/bin/python scripts/teacher_forced_eval.py --model-dir ../runs/microtex_encodec48k/model \
        --dataset ../datasets/microtex --name microtex_encodec48k_atoms --out ../runs/eval_tf --codec-roundtrip on
    ../venv_SCAPES/bin/python scripts/teacher_forced_eval.py --out ../runs/eval_tf --summarize-only

Outputs (see README.md written to --out): <out>/original_10s/*.wav, <out>/<name>/{generated,codec_roundtrip}/*.wav,
<out>/<name>/{metrics.json,per_file.csv}, <out>/summary.{md,json}, <out>/README.md.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # the SCAPES package
import eval_metrics as em

SECONDS = 10.0
DELAY_SECONDS = 0.5
NOISE_SEED_BASE = 1000
N_SEGMENTS = 5
CLAP_WINDOW_SECONDS = 8.0     # the CLAP wrapper pads or trims every input to 8 s


# ─── units, real past and real CLAP ───

# Start frames (codec frames) of the units of the model that fit in the first `seconds`: an atom [t, t+F) with t = k H,
# an AR block [t, t+B) with t = k B.
def unit_starts(geom, seconds: float) -> list:
    total = int(np.floor(seconds * geom.frame_rate + 1e-6))
    step = geom.unit_hop_frames
    return [k * step for k in range((total - geom.unit_frames) // step + 1)] if total >= geom.unit_frames else []


# Real units of one file as the timeline wants them (model units, [1, frame_dim, U] each) and the real CLAP embedding of
# every unit. latents: the stored dict of the representation; clap: the stored [G, 1024] array of the grid
# g * stride_frames; standardise: engine._standardise (identity when the model is not standardised).
def real_units(geom, latents: dict, clap: torch.Tensor, starts: list, standardise=lambda z: z):
    if geom.stride_frames != geom.unit_hop_frames:
        raise NotImplementedError("the CLAP grid has a stride different from the unit hop: not supported by this tool")
    units, claps = [], []
    for t in starts:
        k = t // geom.unit_hop_frames
        if geom.mode == "autoregressive":
            z = latents["latent"][:, t:t + geom.unit_frames].float().unsqueeze(0)
            z = standardise(z)
        elif geom.encoding == "whole":
            # atoms of a codec stored as one whole latent [d, T] (e.g. CoDiCodec): the atom is the slice [t, t + F)
            z = standardise(latents["latent"][:, t:t + geom.unit_frames].float().unsqueeze(0))
        else:
            z = standardise(latents["latent"][k].float().unsqueeze(0))
            if geom.has_scale:
                scale = latents["scale"][k].float().reshape(1, 1, 1).expand(1, 1, geom.unit_frames)
                z = torch.cat([z, scale], dim=1)
        units.append(z)
        claps.append(clap[t // geom.stride_frames].float())
    return units, claps


# Teacher-forced generation: every step has the real unit as its past (TF=True) and its real CLAP; the model draws the
# unit (nfe, cfg) from noise seeded once with `seed`. Returns the audio [C, N] at the codec rate.
def generate_teacher_forced(engine, units, claps, nfe: int, cfg: float, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    engine.decode_seed = seed
    timeline = engine.build_base_timeline(units, claps, None, default_TF=True)
    timeline = engine.generate(timeline, NFE=nfe, cfg_scale=cfg)
    return engine.decode_timeline(timeline)


# Codec round trip of the same real units: the codec decoding of the stored encoding, with the same render path
# (overlap-add with the raw stored scales for atoms, the streaming decoder for AR).
@torch.no_grad()
def codec_roundtrip(engine, units) -> torch.Tensor:
    from SCAPES.inference.render import StreamRenderer, render_atoms
    g = engine.geom
    if g.mode == "autoregressive":
        renderer = StreamRenderer(engine.codec)
        for z in units:
            renderer.push(engine._unstandardise(z[:, :g.latent_dim, :].to(engine.device)))
        return renderer.finish().cpu()
    units = [u.to(engine.device) for u in units]
    return render_atoms(engine.codec, units, g, engine.mu, engine.std, base_seed=0, smooth=[False] * len(units)).cpu()


# ─── audio files ───

def write_wav(path: Path, audio, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    sf.write(str(path), a.T, sr, subtype="PCM_16")


def file_sha(path: Path, n: int = 12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()[:n]


# ─── CLAP embeddings for KAD ───

# CLAP embedding (unit norm, deterministic: no random extension) of each mono segment [n_i] of equal length.
@torch.no_grad()
def clap_embed(clap_model, segments: list, device: str) -> np.ndarray:
    x = torch.stack([torch.from_numpy(np.asarray(s, dtype=np.float32)) for s in segments]).unsqueeze(1)   # [B, 1, n]
    out = clap_model.compute_embedding(x.to(device), og_sr=em.SR, random_extension=False)
    return out.float().cpu().numpy().astype(np.float64)


# Clip-level embedding of the region [a, e) of a signal: the mean of the CLAP embeddings of its first and last 8 s
# windows (the wrapper trims to 8 s, so two windows cover a region of up to 16 s), renormalised to unit norm.
def clip_embedding(clap_model, x: np.ndarray, a: int, e: int, device: str) -> np.ndarray:
    w = int(CLAP_WINDOW_SECONDS * em.SR)
    starts = sorted({a, max(a, e - w)})
    emb = clap_embed(clap_model, [x[s:s + w] for s in starts], device).mean(axis=0)
    return emb / np.linalg.norm(emb)


# Segment-level embeddings: the region [a, e) cut in N_SEGMENTS equal non-overlapping segments (about 1.9 s)
def segment_embeddings(clap_model, x: np.ndarray, a: int, e: int, device: str) -> np.ndarray:
    n = (e - a) // N_SEGMENTS
    return clap_embed(clap_model, [x[a + i * n:a + (i + 1) * n] for i in range(N_SEGMENTS)], device)


# ─── evaluation of one model ───

CSV_FIELDS = ["file", "row"] + [f"{m}_{n}" for n in em.MSS_FFT_SIZES for m in ("log_l1", "spec_conv")] + \
             ["mean_log_l1", "mean_spec_conv", "lag_samples", "peak"]


def mss_row_values(res: dict) -> dict:
    out = {}
    for n, v in res["scales"].items():
        out[f"log_l1_{n}"], out[f"spec_conv_{n}"] = v["log_l1"], v["spec_conv"]
    out["mean_log_l1"], out["mean_spec_conv"] = res["mean_log_l1"], res["mean_spec_conv"]
    return out


def mean_over_files(per_file: dict) -> dict:
    files = list(per_file)
    keys = per_file[files[0]].keys()
    return {k: float(np.mean([per_file[f][k] for f in files])) for k in keys}


def run_model(args):
    from SCAPES.auxiliar.clap_wrapper import CLAPWrapper
    from SCAPES.data.dataprep.semantic import load_audio_48k
    from SCAPES.data.representations import RepresentationStore
    from SCAPES.inference.FlowInference import FlowInference

    out_root = Path(args.out)
    model_out = out_root / args.name
    if args.name in ("original_10s", "summary", "README"):
        raise ValueError(f"--name {args.name!r} is reserved")
    model_out.mkdir(parents=True, exist_ok=True)
    dataset = Path(args.dataset)
    model_dir = Path(args.model_dir)
    t0 = time.time()

    # ─── the model: best checkpoint, geometry from inference.gin ───
    engine = FlowInference(model_dir, device=args.device, checkpoint=args.checkpoint)
    geom, info = engine.geom, engine.info
    ckpt_names = engine._resolve_checkpoint_names(args.checkpoint, model_dir)
    ckpt_paths = [model_dir / "checkpoints" / n for n in ckpt_names]
    print(f"model {args.name}: codec {geom.codec}, mode {geom.mode}, unit {geom.unit_frames} frames, hop "
          f"{geom.unit_hop_frames}, checkpoint {ckpt_names[0]} (sha256 {file_sha(ckpt_paths[0])})", flush=True)

    store = RepresentationStore(dataset, geom, info)
    store.ensure()      # raises if the stored representation was built with another codec or geometry
    clap_key = store.clap_key("2023", geom.context_seconds, engine.semantic_random_extension)
    files = sorted((dataset / "raw").glob("*.wav"))
    if args.max_files:
        files = files[:args.max_files]
    starts = unit_starts(geom, SECONDS)
    print(f"{len(files)} files, {len(starts)} units of {geom.unit_frames} frames per file, CLAP key {clap_key}", flush=True)

    sr_codec = info.sample_rate
    a = int(em.EXCLUDE_SECONDS * em.SR)
    ref, ref_next, gen, rt = {}, {}, {}, {}
    peaks, lags = {}, {}
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    # ─── generation, file by file ───
    for i, path in enumerate(files):
        stem = path.stem
        audio = load_audio_48k(path).numpy()
        n10 = int(SECONDS * em.SR)
        write_wav(out_root / "original_10s" / f"{stem}.wav", audio[:, :n10], em.SR)
        ref[stem] = em.to_mono(audio[:, :n10])
        ref_next[stem] = em.to_mono(audio[:, n10:2 * n10])

        latents = store.load_latents(path)
        clap = store.load_grid_array("clap", clap_key, path)
        units, claps = real_units(geom, latents, clap, starts, engine._standardise)
        t1 = time.time()
        g_audio = generate_teacher_forced(engine, units, claps, args.nfe, args.cfg, args.seed)
        write_wav(model_out / "generated" / f"{stem}.wav", g_audio.numpy(), sr_codec)
        peaks[(stem, "generated")] = float(g_audio.abs().max())
        gen[stem] = em.to_mono(em.resample_to_48k(g_audio.numpy(), sr_codec))
        msg = f"  [{i + 1}/{len(files)}] {stem}: generated {g_audio.shape[-1] / sr_codec:.2f} s in {time.time() - t1:.1f} s"
        if args.codec_roundtrip == "on":
            r_audio = codec_roundtrip(engine, units)
            write_wav(model_out / "codec_roundtrip" / f"{stem}.wav", r_audio.numpy(), sr_codec)
            peaks[(stem, "codec_roundtrip")] = float(r_audio.abs().max())
            rt[stem] = em.to_mono(em.resample_to_48k(r_audio.numpy(), sr_codec))
            msg += f", round trip {r_audio.shape[-1] / sr_codec:.2f} s"
        print(msg, flush=True)

    gen_gpu_peak_gb = torch.cuda.max_memory_allocated() / 1e9 if str(args.device).startswith("cuda") else None
    del engine
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()

    # ─── comparison region: [0.4 s, common end) of every signal ───
    stems = list(ref)
    e = min([len(v) for v in ref.values()] + [len(v) for v in gen.values()] + [len(v) for v in rt.values()])
    # noise of length e (indexed like the other signals) with the RMS of the compared region of the original
    noise = {s: em.white_noise_like(ref[s][a:e], NOISE_SEED_BASE + i, n=e) for i, s in enumerate(stems)}
    d = int(DELAY_SECONDS * em.SR)
    rows = {"generated": gen}
    if rt:
        rows["codec_roundtrip"] = rt
    print(f"comparison region: [{a / em.SR:.2f}, {e / em.SR:.3f}) s ({(e - a) / em.SR:.3f} s)", flush=True)

    per_file = {name: {} for name in list(rows) + ["white_noise", "delayed_0.5s"]}
    csv_rows = []
    for s in stems:
        for name, sig in rows.items():
            res = em.mss(ref[s][a:e], sig[s][a:e])
            lag = em.estimate_lag(ref[s][a:e], sig[s][a:e])
            lags[(s, name)] = lag
            per_file[name][s] = mss_row_values(res)
            csv_rows.append({"file": s, "row": name, **per_file[name][s], "lag_samples": lag,
                             "peak": peaks[(s, name)]})
        res = em.mss(ref[s][a:e], noise[s][a:e])
        per_file["white_noise"][s] = mss_row_values(res)
        csv_rows.append({"file": s, "row": "white_noise", **per_file["white_noise"][s], "lag_samples": "", "peak": ""})
        # the original delayed by 0.5 s against itself, without padding: reference x(t), t in [a + d, e); delayed x(t - d)
        res = em.mss(ref[s][a + d:e], ref[s][a:e - d])
        per_file["delayed_0.5s"][s] = mss_row_values(res)
        csv_rows.append({"file": s, "row": "delayed_0.5s", **per_file["delayed_0.5s"][s], "lag_samples": "", "peak": ""})

    mss_summary = {name: mean_over_files(v) for name, v in per_file.items()}
    mss_summary["delayed_0.5s"]["region_start_s"] = (a + d) / em.SR
    with open(model_out / "per_file.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(csv_rows)

    # ─── KAD: CLAP embeddings of the sets ───
    clap_model = CLAPWrapper(version="2023", use_cuda=str(args.device).startswith("cuda"))
    sets = {"generated": gen, "floor": {s: ref_next[s] for s in stems}, "white_noise": noise}
    if rt:
        sets["codec_roundtrip"] = rt
    # (the floor signals are the seconds 10-20 of the files: sample 0 of ref_next is second 10; the same region is cut)
    emb_clip, emb_seg = {}, {}
    for name, sig in [("reference", ref)] + list(sets.items()):
        emb_clip[name] = np.stack([clip_embedding(clap_model, sig[s], a, e, args.device) for s in stems])
        emb_seg[name] = np.concatenate([segment_embeddings(clap_model, sig[s], a, e, args.device) for s in stems])
        print(f"  CLAP embeddings: {name}", flush=True)
    del clap_model
    evals_clip = {k: v for k, v in emb_clip.items() if k != "reference"}
    evals_seg = {k: v for k, v in emb_seg.items() if k != "reference"}
    kad_clip = em.kad_with_bootstrap(emb_clip["reference"], evals_clip, np.arange(len(stems)), args.n_boot, 0, "floor")
    kad_seg = em.kad_with_bootstrap(emb_seg["reference"], evals_seg, np.repeat(np.arange(len(stems)), N_SEGMENTS),
                                    args.n_boot, 0, "floor")

    metrics = {
        "name": args.name, "note": args.note, "codec": geom.codec, "mode": geom.mode,
        "checkpoint": {"flow": str(ckpt_paths[0]), "local_encoder": str(ckpt_paths[1]), "which": args.checkpoint,
                       "flow_sha256_12": file_sha(ckpt_paths[0])},
        "geometry": {"unit_frames": geom.unit_frames, "unit_hop_frames": geom.unit_hop_frames,
                     "memory_units": geom.memory_units, "frame_rate": geom.frame_rate,
                     "codec_sample_rate": sr_codec, "units_per_file": len(starts)},
        "generation": {"seed": args.seed, "nfe": args.nfe, "cfg": args.cfg, "files": stems,
                       "dataset": str(dataset), "clap_key": clap_key, "seconds": SECONDS,
                       "resampled_to_48k": sr_codec != em.SR, "peak_gpu_gb": gen_gpu_peak_gb},
        "region": {"start_s": a / em.SR, "end_s": e / em.SR, "mono": True, "sample_rate": em.SR},
        "mss": {"fft_sizes": list(em.MSS_FFT_SIZES), "eps": em.LOG_EPS, "rows": mss_summary, "per_file": per_file},
        "lag_samples": {f"{s}/{name}": v for (s, name), v in lags.items()},
        "kad": {"n_files": len(stems), "n_boot": args.n_boot, "segments_per_clip": N_SEGMENTS,
                "segment_seconds": ((e - a) // N_SEGMENTS) / em.SR, "clip_level": kad_clip, "segment_level": kad_seg},
        "wall_seconds": time.time() - t0,
    }
    (model_out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"done in {time.time() - t0:.0f} s -> {model_out}", flush=True)
    return metrics


# ─── summary over all models ───

def fmt(x, nd=3):
    return f"{x:.{nd}f}"


def mss_table(models: dict) -> str:
    sizes = [str(n) for n in em.MSS_FFT_SIZES]
    secs = [f"{int(n) / em.SR:.2f} s" for n in sizes]
    head = ["row"] + [f"L1 {s}" for s in secs] + ["L1 mean"] + [f"SC {s}" for s in secs] + ["SC mean"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, m in models.items():
        for row, label in [("generated", "generated (teacher forced)"), ("codec_roundtrip", "codec round trip"),
                           ("white_noise", "white noise (RMS matched)"), ("delayed_0.5s", "original delayed 0.5 s")]:
            r = m["mss"]["rows"].get(row)
            if r is None:
                continue
            cells = [f"{name}: {label}"] + [fmt(r[f"log_l1_{n}"]) for n in sizes] + [fmt(r["mean_log_l1"])] + \
                    [fmt(r[f"spec_conv_{n}"]) for n in sizes] + [fmt(r["mean_spec_conv"])]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def kad_cell(v: dict) -> str:
    return f"{v['kad']:.2f} (boot std {v['boot_std']:.2f}, 95% [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}])"


def kad_table(models: dict, level: str, title: str) -> str:
    lines = [f"**{title}**", "", "| row | KAD (point, bootstrap std, 95% interval) | generated/row minus floor (paired 95% interval) |",
             "|---|---|---|"]
    for name, m in models.items():
        k = m["kad"][level]["sets"]
        for row, label in [("generated", "generated (teacher forced)"), ("floor", "floor: original 0-10 s vs 10-20 s"),
                           ("codec_roundtrip", "codec round trip"), ("white_noise", "white noise")]:
            if row not in k:
                continue
            diff = k[row].get("minus_floor")
            dcell = "-" if diff is None else f"{diff['mean']:.2f} [{diff['ci95'][0]:.2f}, {diff['ci95'][1]:.2f}]" + \
                (" (contains 0)" if diff["contains_zero"] else "")
            lines.append(f"| {name}: {label} | {kad_cell(k[row])} | {dcell} |")
    return "\n".join(lines)


def write_summary(out_root: Path):
    models = {}
    for p in sorted(out_root.glob("*/metrics.json")):
        models[p.parent.name] = json.loads(p.read_text())
    if not models:
        print("no metrics.json found")
        return
    (out_root / "summary.json").write_text(json.dumps(models, indent=1))
    first = next(iter(models.values()))
    nf = first["kad"]["n_files"]
    md = ["# Teacher-forced evaluation: summary", "",
          "First 10 s of the 16 files of the training dataset (train region: this measures fit, not generalisation). "
          "Protocol and caveats: see README.md. Lower is better everywhere. MSS: natural-log magnitude L1 and spectral "
          "convergence (SC) at three big windows, mean over files; region 0.4 s to the end of the shortest signal, mono 48 kHz.", "",
          "## Models", ""]
    for name, m in models.items():
        md.append(f"- {name}: codec {m['codec']}, mode {m['mode']}, checkpoint {Path(m['checkpoint']['flow']).name} "
                  f"(sha256 {m['checkpoint']['flow_sha256_12']}...), seed {m['generation']['seed']}, nfe {m['generation']['nfe']}, "
                  f"cfg {m['generation']['cfg']}, region [{m['region']['start_s']:.1f}, {m['region']['end_s']:.2f}) s"
                  + (f". {m['note']}" if m.get("note") else ""))
    md += ["", "## MSS (big windows)", "", mss_table(models), "",
           "## KAD (CLAP embedding, 100 x unbiased MMD^2, RBF kernel, sigma = median distance of the reference)", "",
           f"Reference set: the {first['kad']['n_files']} original clips. Bootstrap: {first['kad']['n_boot']} resamples of the clips "
           f"with replacement (paired across rows). The clip-level estimate is {nf} vs {nf} items, far below the N >= 100 the "
           "KAD paper tested: it is noisy and can be negative.", "",
           kad_table(models, "clip_level", f"Clip level (one embedding per clip, {nf} vs {nf})"), "",
           kad_table(models, "segment_level", f"Segment level (extra: {first['kad']['segments_per_clip']} segments of "
                     f"{first['kad']['segment_seconds']:.2f} s per clip, {nf * first['kad']['segments_per_clip']} vs "
                     f"{nf * first['kad']['segments_per_clip']}; segments of one clip are not independent)"), ""]
    (out_root / "summary.md").write_text("\n".join(md))
    write_readme(out_root, models)
    print(f"summary written to {out_root}/summary.md ({len(models)} models)")


# One line per model: codec, mode, output rate and the files to open, in the order to listen
def listening_order(models: dict) -> str:
    lines = []
    for i, (n, m) in enumerate(models.items(), 1):
        g = m["geometry"]
        lines.append(f"{i}. `{n}` ({m['codec']}, {m['mode']}, {g['codec_sample_rate']} Hz, unit hop "
                     f"{g['unit_hop_frames']} frames at {g['frame_rate']:.2f} Hz, compared region up to {m['region']['end_s']:.2f} s): "
                     f"`{n}/generated/<file>.wav`, then `{n}/codec_roundtrip/<file>.wav`")
    return "\n".join(lines)


def write_readme(out_root: Path, models: dict):
    root = out_root.resolve()
    lines = [f"""# Teacher-forced evaluation (first 10 s of the microtex files)

Everything lives under `{root}`.

## Where the audio is

- `original_10s/<file>.wav`: the first 10 s of each of the 16 files (48 kHz, stereo, 16-bit). The reference for all models.
- `<model>/generated/<file>.wav`: the model's teacher-forced regeneration (codec native rate, 16-bit).
- `<model>/codec_roundtrip/<file>.wav`: the real audio encoded and decoded by the codec, no model (the ceiling the model cannot beat).

Models found: {", ".join(f"`{n}`" for n in models)}.

How to listen, per file: (1) `original_10s/<file>.wav`, (2) `<model>/generated/<file>.wav`, (3) `<model>/codec_roundtrip/<file>.wav`.
Example: `original_10s/rain.wav`, `{next(iter(models))}/generated/rain.wav`, `{next(iter(models))}/codec_roundtrip/rain.wav`.
The first 0.2 s of the atoms output are silent by design (see below).

## Listening order (per file: original, then each model's generated, then its codec round trip)

{listening_order(models)}

## What was generated (teacher forcing, semantic reconstruction)

For every unit of the model (an atom with its hop, or an AR block) over the first 10 s: the past is the REAL latent of the
original (the stored codec encoding in `datasets/microtex/representations/<codec>__<encoding>/`), the CLAP condition is the
REAL audio's stored CLAP embedding aligned to the unit exactly as in training (1 s window starting at the target), and the
unit is drawn from noise with seed 0 (reset at the start of each file), nfe 16, CFG 3.0. No generated latent is fed back.
Atoms are rendered with the overlap-add of normal generation (per-atom scale taken from the generated scale channel with V1's
smoothing). AR blocks go through the codec's stateful streaming decoder, with no overlap-add. The best checkpoint
(best_flow_model.pt, best_local_encoder.pt) is used, never the last.

## Comparison protocol

- Mono (mean of the channels) at 48 kHz for original and generated (generated audio of a lower-rate codec is resampled up
  with torchaudio's sinc resampler). Both are cut to the same length (generated audio can be shorter than 10 s because only
  whole units fit).
- The first 0.4 s are excluded from every metric for every model, so atoms and AR are treated the same: the atoms output is
  silent until about 0.2 s by design (an atom only contributes its last hop; V1 'silent first 30 frames'). The start offset
  is checked with the cross-correlation lag between original and output (`lag_samples` in per_file.csv and metrics.json;
  0 = no offset; a noisy generated texture does not correlate with the original, so for the generated rows the lag is a
  weak check and the codec round trip is the reliable one).
- MSS: |STFT| with Hann windows of 32768, 65536 and 131072 samples (0.68, 1.37, 2.73 s), hop = window/4, no centre padding.
  Per scale: (a) mean |ln(|S|+1e-7) - ln(|S'|+1e-7)|, (b) spectral convergence ||S-S'||_F / ||S||_F (S = original). Mean over
  scales, then mean over files. Windows this long compare the long-term spectral envelope of a texture, not its fine detail.
- Reference rows: codec round trip, white Gaussian noise with the RMS of the original (seed 1000+file index), and the
  original delayed by 0.5 s (compared on [0.9 s, end): x(t) against x(t-0.5 s), no padding).
- KAD: own implementation (kadtk pins python<3.12, numpy<2, torch<2.6 and cannot be installed here). KAD = 100 x unbiased
  MMD^2 (diagonals removed) with the Gaussian RBF kernel exp(-||x-y||^2 / (2 sigma^2)), sigma = median pairwise distance of
  the reference set (Chung et al. 2025). Embedding: the project's CLAP (msclap 2023) wrapper, unit-norm 1024-d audio
  embedding, no random extension (deterministic). Clip level: mean of the embeddings of the first and last 8 s windows of the
  compared region (the wrapper trims every input to 8 s), renormalised. Segment level (extra): the compared region cut in 5
  equal segments (about 1.9 s), each padded by the wrapper to 8 s by repetition with crossfades. The wrapper peak-normalises
  its input, so KAD ignores overall level.
- KAD reference set: the 16 original clips. Floor: KAD(original 0-10 s, the same files' seconds 10-20), i.e. other real audio
  of the same files. Bootstrap: 200 resamples of the clips with replacement (same resample for all rows, so differences
  between rows are paired); sigma is fixed to the full-reference median. Duplicated clips count as distinct items and raise
  the within-set terms, so the bootstrap mean sits above the point estimate: read the interval as a spread.

## Caveats

- This is the TRAIN region: it measures how well the model fits what it saw, not generalisation. The reserve region
  [108, 120) s and the validation region are not used.
- With teacher forcing the past is always perfect; free-running generation is harder and will score worse.
- Clip-level KAD has 16 vs 16 items, far below the N >= 100 the KAD paper tested; the estimate is noisy and can be negative.
  Segment-level KAD has 80 vs 80 items but the 5 segments of a clip are correlated, so it is not 80 independent samples.
- Every evaluation set is paired with the reference by file (the same 16 files), while the white noise is not. The pairing
  raises the cross term of the MMD, which is why the floor and the codec round trip can come out negative: compare rows with
  each other (the paired differences in the summary), not with zero.
- A sound's generated version is a new draw, not a copy: the spectral metrics at long windows are insensitive to the fine
  structure, and MSS and KAD are not perceptual scores. Listen.
- The original reference WAVs are 16-bit renderings of the 24-bit files; the metrics use the float signal.
"""]
    (out_root / "README.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", help="trained model folder (with checkpoints/inference.gin)")
    ap.add_argument("--dataset", help="dataset folder (raw/ and representations/)")
    ap.add_argument("--name", help="name of this model in the outputs")
    ap.add_argument("--out", default="../runs/eval_tf", help="output root")
    ap.add_argument("--codec-roundtrip", choices=["on", "off"], default="on")
    ap.add_argument("--checkpoint", default="best", help="'best' (default) or 'last'")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--nfe", type=int, default=16)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--note", default="", help="free text kept in metrics.json and shown in the summary")
    ap.add_argument("--max-files", type=int, default=0, help="debug: only the first N files")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    if not args.summarize_only:
        if not (args.model_dir and args.dataset and args.name):
            ap.error("--model-dir, --dataset and --name are required")
        run_model(args)
    write_summary(Path(args.out))


if __name__ == "__main__":
    main()
