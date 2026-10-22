"""Slice B11: S5 generation and diagnostics (design S5, Metrics D-a..D-e).

For every trained run <runs>/<name>/model (the codec name is read from its inference.gin):
  - generates --seconds of audio from one prompt per file x each seed (the prompt is the stored CLAP of the first sample
    of the file's --prompt-region), AR through the streaming decoder, atoms with overlap-add;
  - production checks (S5 ii) with SCAPES.inference.diagnostics.production_checks;
  - D-a codec round trip MSS of the real region (ceiling), D-b CLAP adherence, D-c AR seam ratio, D-d memorisation,
    D-e efficiency; each mean has a bootstrap interval over the prompts (files).
The reserve region must only be used in the final acceptance (B12): earlier runs use --prompt-region val.

Usage (from SCAPES/):
    ../venv_SCAPES/bin/python scripts/s5_eval.py --runs ../runs/s5 --dataset ../datasets/microtex --seeds 0 1 2 \
        --seconds 10 --prompt-region val --out ../runs/s5_val

Outputs: <out>/s5_eval.json, s5_eval.png, README.md, original/, <codec>/{generated,roundtrip}/*.wav, <codec>/s5_codec.json.
Each codec runs in its own process (CoDiCodec flips global torch flags at import).
"""
import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import eval_metrics as em

BAND_HZ = 12000.0          # band-limited MSS (the 24 kHz codecs have no content above)
EXCLUDE_SECONDS = 0.4      # head excluded from the MSS, as in the teacher-forced evaluation
MAX_TRAIN_UNITS = 20000    # D-d: cap on the training units searched
MAX_VAL_UNITS = 5000       # D-d: cap on the reference val units


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def write_json(path, obj):
    Path(path).write_text(json.dumps(clean(obj), indent=1))


def write_wav(path: Path, audio, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    sf.write(str(path), a.T, sr, subtype="PCM_16")


# Mean of per-prompt values with a bootstrap interval over the prompts (percentiles 2.5 and 97.5)
def stat(values, n_boot: int = 10000, seed: int = 0) -> dict:
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=np.float64)
    if len(v) == 0:
        return {"mean": None, "ci95": None, "n": 0}
    means = v[np.random.RandomState(seed).randint(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return {"mean": float(v.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "n": int(len(v))}


# FFT sizes of the MSS that fit in n samples; for very short signals the largest power of two that does
def fft_sizes_for(n: int) -> tuple:
    sizes = tuple(s for s in em.MSS_FFT_SIZES if s <= n)
    return sizes if sizes else (max(16, 1 << int(math.floor(math.log2(max(n, 32)))) ),)


def mss_pair(ref, sig) -> dict:
    n = min(len(ref), len(sig))
    sizes = fft_sizes_for(n)
    full = em.mss(ref[:n], sig[:n], fft_sizes=sizes)
    band = em.mss(ref[:n], sig[:n], fft_sizes=sizes, max_hz=BAND_HZ)
    return {"full_log_l1": full["mean_log_l1"], "full_spec_conv": full["mean_spec_conv"],
            "band_log_l1": band["mean_log_l1"], "band_spec_conv": band["mean_spec_conv"], "fft_sizes": list(sizes)}


# ─── discovery ───

def codec_name_of(model_dir: Path):
    gin = model_dir / "checkpoints" / "inference.gin"
    if not gin.exists():
        return None
    m = re.search(r'^\s*codec\.name\s*=\s*"([^"]+)"', gin.read_text(), re.M)
    return m.group(1) if m else None


def discover(runs: Path) -> list:
    found = []
    for model in sorted(runs.glob("*/model")):
        name = codec_name_of(model)
        if name is not None:
            found.append((name, model.parent))
    return found


# ─── one codec ───

# Units [ld * F] (flattened, model units, latent channels only) of the given start frames of one file
def unit_matrix(engine, dataset, fname, ts):
    units = real_units(engine, dataset, fname, ts)
    ld = engine.geom.latent_dim
    return torch.stack([u[0, :ld].reshape(-1) for u in units]) if units else torch.zeros(0, 0)


# Real units [1, frame_dim, F] of one file at the given start frames, in model units (standardised if the model is):
# a slice of the whole latent, or the stored atom k = t // H of a per-atom representation
def real_units(engine, dataset, fname, starts):
    g = engine.geom
    latent, scale = dataset._latent[fname], dataset._scale[fname]
    units = []
    for t in starts:
        if dataset.per_atom:
            k = t // g.unit_hop_frames
            z = engine._standardise(latent[k].float().unsqueeze(0))
            if g.has_scale:
                z = torch.cat([z, scale[k].float().reshape(1, 1, 1).expand(1, 1, g.unit_frames)], dim=1)
        else:
            z = engine._standardise(latent[:, t:t + g.unit_frames].float().unsqueeze(0))
        units.append(z)
    return units


# Distance of every row of q to its nearest row of x (L2), in chunks
def nearest_distance(q: torch.Tensor, x: torch.Tensor, device: str, chunk: int = 256) -> torch.Tensor:
    x = x.to(device)
    out = []
    for i in range(0, len(q), chunk):
        out.append(torch.cdist(q[i:i + chunk].to(device), x).min(dim=1).values.cpu())
    return torch.cat(out)


def evaluate_codec(args, name: str, run_dir: Path) -> dict:
    from audio_codec_wrappers.audio import adapt_audio
    from SCAPES.data.config_loader import load_dataprep_config
    from SCAPES.data.dataset import AtomSequenceDataset
    from SCAPES.data.dataprep.semantic import load_audio_48k, make_clap_embedder
    from SCAPES.data.representations import frames_to_samples_48k
    from SCAPES.inference.FlowInference import FlowInference
    from SCAPES.inference.diagnostics import production_checks, seam_ratio
    from SCAPES.pipeline import units_for_seconds
    import teacher_forced_eval as tf

    device = args.device
    out_dir = Path(args.out)
    region = args.prompt_region
    model_dir = run_dir / "model"
    cfg_dir = run_dir / "config"
    if not (cfg_dir / "dataprep.gin").exists():
        cfg_dir = Path(args.dataset) / "config"
    engine = FlowInference(model_dir, device=device, checkpoint=args.checkpoint)
    geom = engine.geom
    cfg = load_dataprep_config(cfg_dir / "dataprep.gin")
    dataset = AtomSequenceDataset(args.dataset, cfg, requested_keys={"target_semantic"}, device="cpu", verbose=False)
    if (dataset.geom.mode, dataset.geom.codec, dataset.geom.unit_frames, dataset.geom.unit_hop_frames,
            dataset.geom.memory_frames) != (geom.mode, geom.codec, geom.unit_frames, geom.unit_hop_frames,
                                            geom.memory_frames):
        raise RuntimeError(f"{name}: the dataset config {cfg_dir} and the model geometry differ")
    sr = engine.sr
    fr = geom.frame_rate
    ld = geom.latent_dim
    unit_seconds = geom.unit_frames / fr
    print(f"[{name}] mode {geom.mode}, unit {geom.unit_frames} frames, checkpoint {args.checkpoint}, prompt region {region}",
          flush=True)

    # samples of the prompt region per file (start frames on the stride grid)
    by_region = {}
    for (fname, t), r in zip(dataset.all_indices, dataset._regions):
        by_region.setdefault((fname, r), []).append(t)
    files = [f for f in dataset.filenames if (f, region) in by_region]
    if args.max_files:
        files = files[:args.max_files]
    if not files:
        raise RuntimeError(f"{name}: no file has samples in the {region} region")
    ts = {f: sorted(by_region[(f, region)]) for f in files}
    # prompt: the stored CLAP at the first grid point of the region start (not the first training sample)
    prompt_k, prompts = {}, {}
    for f in files:
        lo, hi = dataset._bounds[f][region]
        k = math.ceil(lo * fr / geom.stride_frames - 1e-6)
        assert k * geom.stride_frames / fr + geom.context_seconds <= hi + 1e-6, f"{f}: prompt window leaves the region"
        prompt_k[f] = k
        prompts[f] = dataset._clap[f][k].reshape(-1).float()
    prompt_start = {f: prompt_k[f] * geom.stride_frames / fr for f in files}
    n_units = units_for_seconds(geom, args.seconds)

    # ─── generation (timed), with an untimed warm-up that loads the codec ───
    def generate(prompt, seed, units):
        torch.manual_seed(seed)
        engine.decode_seed = seed
        timeline = engine.build_base_timeline([None] * units, [prompt] * units, None, default_TF=False)
        timeline = engine.generate(timeline, NFE=args.nfe, cfg_scale=args.cfg)
        return timeline, engine.decode_timeline(timeline)

    generate(prompts[files[0]], 0, min(2, n_units))
    cuda = str(device).startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    gens = {}
    for f in files:
        for seed in args.seeds:
            if cuda:
                torch.cuda.synchronize()
            t0 = time.time()
            timeline, audio = generate(prompts[f], seed, n_units)
            if cuda:
                torch.cuda.synchronize()
            dt = time.time() - t0
            z = torch.cat([s["atom_generated"][0, :ld].cpu() for s in timeline], dim=-1) if geom.mode == "autoregressive" else None
            units = [s["atom_generated"][0, :ld].cpu() for s in timeline]
            gens[(f, seed)] = {"audio": audio, "seconds": dt, "z": z, "units": units}
            print(f"[{name}] {f} seed {seed}: {audio.shape[-1] / sr:.2f} s in {dt:.1f} s", flush=True)
    peak_gen = int(torch.cuda.max_memory_allocated()) if cuda else 0

    # ─── originals of the prompt region, production checks, wavs ───
    audio48 = {}
    ref_rms, bounds = {}, {}
    for f in files:
        audio48[f] = load_audio_48k(dataset.audio_path(f))
        lo, hi = dataset._bounds[f][region]
        seg = audio48[f][:, int(lo * 48000):int(hi * 48000)].mean(dim=0)
        ref_rms[f] = float(seg.double().pow(2).mean().sqrt())
        bounds[f] = (lo, hi)
    clips = []
    for (f, seed), g in gens.items():
        stem = Path(f).stem
        chk = production_checks(g["audio"], sr, ref_rms[f], args.seconds, unit_seconds)
        path = out_dir / name / "generated" / f"{stem}_seed{seed}.wav"
        write_wav(path, g["audio"].numpy(), sr)
        torch.save({"z": g["z"], "units": g["units"]}, path.with_suffix(".pt"))
        g["checks"] = chk
        clips.append({"file": f, "seed": seed, **chk, "ref_rms": ref_rms[f], "gen_seconds": g["seconds"],
                      "rtf": g["seconds"] / (g["audio"].shape[-1] / sr), "wav": str(path.relative_to(out_dir))})

    # ─── D-a: codec round trip of the real region (ceiling) ───
    d_a_files = {}
    a_skip = int(EXCLUDE_SECONDS * em.SR)
    for f in files:
        H = geom.unit_hop_frames
        t0 = -(-ts[f][0] // H) * H          # first unit start inside the region on the unit grid
        t_end = min(t0 + int(args.seconds * fr), ts[f][-1] + geom.unit_frames)
        count = (t_end - t0 - geom.unit_frames) // H + 1
        starts = [t0 + k * H for k in range(max(count, 0))]
        if not starts:
            continue
        units = real_units(engine, dataset, f, starts)
        rt = tf.codec_roundtrip(engine, units)
        a0 = int(frames_to_samples_48k(t0, geom))
        n_ref = int(args.seconds * 48000)
        ref_clip = audio48[f][:, a0:a0 + n_ref]
        write_wav(out_dir / "original" / f"{Path(f).stem}.wav", ref_clip.numpy(), 48000)
        write_wav(out_dir / name / "roundtrip" / f"{Path(f).stem}.wav", rt.numpy(), sr)
        ref = em.to_mono(ref_clip.numpy())
        sig = em.to_mono(em.resample_to_48k(rt.numpy(), sr))
        e = min(len(ref), len(sig))
        a = min(a_skip, max(e - 64, 0))
        noise = em.white_noise_like(ref[a:e], 1000 + len(d_a_files), n=e)
        d_a_files[f] = {"roundtrip": mss_pair(ref[a:e], sig[a:e]), "white_noise": mss_pair(ref[a:e], noise[a:e]),
                        "seconds": (e - a) / em.SR}
    d_a = {"files": d_a_files, "note": "MSS of the codec round trip of the real region against the original, "
                                       "[0.4 s, end); white noise of matched RMS is the floor-free reference",
           "band_hz": BAND_HZ}
    for key in ("full_log_l1", "full_spec_conv", "band_log_l1", "band_spec_conv"):
        d_a[f"roundtrip_{key}"] = stat([v["roundtrip"][key] for v in d_a_files.values()])
        d_a[f"white_noise_{key}"] = stat([v["white_noise"][key] for v in d_a_files.values()])

    # ─── D-b: CLAP adherence ───
    window = int(round(geom.context_seconds * 48000))
    embed = make_clap_embedder(device, engine.semantic_random_extension)
    torch.manual_seed(0)

    def embed_windows(x48):          # x48 [2, N] at 48 kHz -> [W, 1024] of non-overlapping windows
        n = x48.shape[-1] // window
        if n == 0:
            return torch.zeros(0, 1024)
        batch = torch.stack([x48[:, i * window:(i + 1) * window] for i in range(n)]).to(device)
        with torch.no_grad():
            return embed(batch, 48000).detach().float().cpu()

    def cosine(emb, prompt):
        return torch.nn.functional.cosine_similarity(emb, prompt.view(1, -1), dim=1)

    gen_emb = {}
    for (f, seed), g in gens.items():
        if g["checks"]["finite"]:
            gen_emb[(f, seed)] = embed_windows(adapt_audio(g["audio"], sr, 48000, 2)[0])
    d_b_files = {}
    ctx_frames = int(round(geom.context_seconds * fr))
    for i, f in enumerate(files):
        other = prompts[files[(i + 1) % len(files)]]
        mine = [gen_emb[(f, s)] for s in args.seeds if (f, s) in gen_emb]
        if not mine:
            continue
        mine = torch.cat(mine)
        # real windows of the same region, after the prompt window, non-overlapping
        real_ts, nxt = [], (prompt_k[f] * geom.stride_frames) + ctx_frames
        for t in ts[f]:
            if t >= nxt:
                real_ts.append(t)
                nxt = t + ctx_frames
        real = torch.stack([dataset._clap[f][t // geom.stride_frames].reshape(-1).float() for t in real_ts]) if real_ts else None
        d_b_files[f] = {"generated": float(cosine(mine, prompts[f]).mean()),
                        "other_file_prompt": float(cosine(mine, other).mean()) if len(files) > 1 else None,
                        "real_floor": float(cosine(real, prompts[f]).mean()) if real is not None else None,
                        "n_windows": int(len(mine)), "n_real_windows": len(real_ts)}
    d_b = {"files": d_b_files, "window_seconds": geom.context_seconds,
           "generated": stat([v["generated"] for v in d_b_files.values()]),
           "real_floor": stat([v["real_floor"] for v in d_b_files.values()]),
           "other_file_prompt": stat([v["other_file_prompt"] for v in d_b_files.values()]),
           "generated_minus_other": stat([v["generated"] - v["other_file_prompt"] for v in d_b_files.values()
                                          if v["other_file_prompt"] is not None]),
           "note": "mean cosine of CLAP(generated 1 s windows) with the file's prompt; real_floor: real windows of the "
                   "same region after the prompt (the 'floor' of the design: real audio of the same file); "
                   "other_file_prompt: the same generated windows against the prompt of the next file"}
    del embed

    # ─── D-c: AR seam ratio ───
    if geom.mode == "autoregressive":
        B = geom.unit_frames
        rho = {}
        for i, f in enumerate(files):
            t0, t_last = ts[f][0], ts[f][-1]
            z_real = dataset._latent[f][:ld, t0:t_last + B].float()
            z_real = (z_real - engine.mu.cpu().view(-1, 1)) / engine.std.cpu().view(-1, 1) if engine.mu is not None else z_real
            gen_rho = [seam_ratio(gens[(f, s)]["z"], B, phase=0) for s in args.seeds if gens[(f, s)]["z"].shape[-1] > B]
            if gen_rho:
                rho[f] = {"generated": float(np.mean(gen_rho)), "real": seam_ratio(z_real, B, phase=t0 % B)}
        d_c = {"applicable": True, "block_frames": B, "files": rho,
               "generated": stat([v["generated"] for v in rho.values()]),
               "real": stat([v["real"] for v in rho.values()]),
               "generated_over_real": stat([v["generated"] / v["real"] for v in rho.values()]),
               "note": "1 = block boundaries are as smooth as the rest; the senior rule of E3b is generated <= 2 x real"}
    else:
        d_c = {"applicable": False, "note": "block seams exist only in autoregressive mode"}

    # ─── D-d: memorisation (nearest training unit, standardised latents, L2) ───
    rs = np.random.RandomState(0)
    train_ts = [(f, t) for (f, t), r in zip(dataset.all_indices, dataset._regions) if r == "train"]
    val_ts = [(f, t) for (f, t), r in zip(dataset.all_indices, dataset._regions) if r == "val"]
    if len(train_ts) > MAX_TRAIN_UNITS:
        train_ts = [train_ts[i] for i in sorted(rs.choice(len(train_ts), MAX_TRAIN_UNITS, replace=False))]
    if len(val_ts) > MAX_VAL_UNITS:
        val_ts = [val_ts[i] for i in sorted(rs.choice(len(val_ts), MAX_VAL_UNITS, replace=False))]

    def matrix(pairs):
        rows = []
        for f in dict.fromkeys(p[0] for p in pairs):
            rows.append(unit_matrix(engine, dataset, f, [t for g_, t in pairs if g_ == f]))
        return torch.cat(rows) if rows else torch.zeros(0, 0)

    x_train, x_val = matrix(train_ts), matrix(val_ts)
    val_pairs = [p for f in dict.fromkeys(q[0] for q in val_ts) for p in val_ts if p[0] == f]   # row order of x_val
    val_nn = nearest_distance(x_val, x_train, device)
    # the design threshold (1st percentile of val-to-train distances) is 0 when more than 1% of the val units have an
    # exact copy in the train region (a repeated segment of a file); the headline threshold then leaves those units out
    dup = val_nn < 1e-3
    threshold_raw = float(np.percentile(val_nn.numpy(), 1.0))
    threshold = float(np.percentile(val_nn[~dup].numpy(), 1.0)) if bool((~dup).any()) else threshold_raw
    d_d_files = {}
    all_gen = []
    for f in files:
        q = torch.stack([u.reshape(-1) for s in args.seeds for u in gens[(f, s)]["units"]])
        nn = nearest_distance(q, x_train, device)
        all_gen.append(nn)
        d_d_files[f] = {"fraction_below_threshold": float((nn < threshold).float().mean()),
                        "fraction_below_threshold_raw": float((nn < threshold_raw).float().mean()),
                        "fraction_exact_copy": float((nn < 1e-3).float().mean()),
                        "median_nn_distance": float(nn.median()), "n_units": int(len(nn))}
    gen_nn = torch.cat(all_gen)
    d_d = {"files": d_d_files, "threshold_1st_percentile_val_to_train": threshold,
           "threshold_raw_1st_percentile": threshold_raw, "val_units_with_exact_train_copy": float(dup.float().mean()),
           "val_files_with_exact_copy": sorted({p[0] for p, d_ in zip(val_pairs, dup.tolist()) if d_}),
           "fraction_exact_copy": stat([v["fraction_exact_copy"] for v in d_d_files.values()]),
           "fraction_below_threshold_raw": stat([v["fraction_below_threshold_raw"] for v in d_d_files.values()]),
           "n_train_units": int(len(x_train)), "n_val_units": int(len(x_val)),
           "val_to_train_median": float(val_nn.median()), "generated_to_train_median": float(gen_nn.median()),
           "fraction_below_threshold": stat([v["fraction_below_threshold"] for v in d_d_files.values()]),
           "unit_dim": int(x_train.shape[1]),
           "note": "fraction of generated units whose nearest training unit is closer than the 1st percentile of the "
                   "nearest-train distances of held-out val units (about 0.01 expected for a model that does not copy), "
                   "val units with an exact train copy (distance < 1e-3) left out of the threshold; "
                   "fraction_below_threshold_raw uses the plain 1st percentile"}
    # the same check for the prompt region's own units (val or reserve) against the train region
    reg_pairs = [(f, t) for f in files for t in ts[f]]
    if len(reg_pairs) > MAX_VAL_UNITS:
        reg_pairs = [reg_pairs[i] for i in sorted(rs.choice(len(reg_pairs), MAX_VAL_UNITS, replace=False))]
    x_reg = matrix(reg_pairs)
    reg_rows = [p_ for f in dict.fromkeys(q[0] for q in reg_pairs) for p_ in reg_pairs if p_[0] == f]
    reg_nn = nearest_distance(x_reg, x_train, device)
    reg_dup = reg_nn < 1e-3
    reg_raw = float(np.percentile(reg_nn.numpy(), 1.0))
    reg_thr = float(np.percentile(reg_nn[~reg_dup].numpy(), 1.0)) if bool((~reg_dup).any()) else reg_raw
    reg_file_dup, reg_file_below = {}, {}
    for f in files:
        mask = torch.tensor([p_[0] == f for p_ in reg_rows])
        reg_file_dup[f] = float(reg_dup[mask].float().mean()) if bool(mask.any()) else None
    for f in files:
        nn = torch.cat([nearest_distance(torch.stack([u.reshape(-1) for s in args.seeds for u in gens[(f, s)]["units"]]), x_train, device)])
        reg_file_below[f] = float((nn < reg_thr).float().mean())
    d_d.update({"region": region, "threshold_1st_percentile_region_to_train": reg_thr,
                "threshold_region_raw": reg_raw, "region_units_with_exact_train_copy": float(reg_dup.float().mean()),
                "region_files_with_exact_copy": sorted({p_[0] for p_, d_ in zip(reg_rows, reg_dup.tolist()) if d_}),
                "region_exact_copy_fraction": reg_file_dup, "fraction_below_threshold_region_per_file": reg_file_below,
                "fraction_below_threshold_region": stat(list(reg_file_below.values())), "n_region_units": int(len(x_reg))})
    del x_train

    # ─── D-e: efficiency ───
    rtf = [c["rtf"] for c in clips]
    n_params = sum(p.numel() for p in engine.model.parameters()) + sum(p.numel() for p in engine.local_encoder.parameters())
    d_e = {"parameters": int(n_params), "peak_gpu_bytes_generation": peak_gen,
           "rtf_median": float(np.median(rtf)), "rtf_min": float(np.min(rtf)), "rtf_max": float(np.max(rtf)),
           "rtf_per_clip_includes_decode": True,
           "note": "batch 1, NFE %d, CFG %.1f, after an untimed warm-up; sequential in one process (not the S3 protocol)"
                   % (args.nfe, args.cfg)}

    production = {"n_clips": len(clips), "n_pass": int(sum(c["pass"] for c in clips))}
    for k in ("finite", "duration_ok", "rms_ok", "rms_first_ok", "clip_ok"):
        production[k] = int(sum(c[k] for c in clips))
    production["pass_rate"] = production["n_pass"] / max(len(clips), 1)
    meta = {"codec": name, "mode": geom.mode, "checkpoint": args.checkpoint, "prompt_region": region,
            "seeds": list(args.seeds), "seconds": args.seconds, "nfe": args.nfe, "cfg": args.cfg, "sample_rate": sr,
            "unit_frames": geom.unit_frames, "unit_seconds": unit_seconds, "n_units": n_units, "n_files": len(files),
            "run_dir": str(run_dir), "has_latent_stats": engine.mu is not None,
            "prompt_start_seconds": prompt_start, "region_bounds_seconds": bounds}
    return {"clips": clips, "production": production,
            "diagnostics": {"D-a": d_a, "D-b": d_b, "D-c": d_c, "D-d": d_d, "D-e": d_e}, "meta": meta}


# ─── figures and README ───

def make_figure(out_dir: Path, result: dict):
    from matplotlib.figure import Figure
    names = list(result)
    colors = {n: c for n, c in zip(names, ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"])}
    fig = Figure(figsize=(13, 8))
    ax = fig.subplots(2, 2)
    x = np.arange(len(names))

    def bars(a, key_means, title, ylabel, hline=None):
        for i, n in enumerate(names):
            m, ci = key_means(result[n])
            if m is None:
                continue
            a.bar(i, m, color=colors[n], yerr=[[m - ci[0]], [ci[1] - m]] if ci else None, capsize=3)
        a.set_xticks(x)
        a.set_xticklabels(names, rotation=20)
        a.set_title(title)
        a.set_ylabel(ylabel)
        a.grid(True, axis="y", linestyle="--", alpha=0.6)
        if hline is not None:
            a.axhline(hline, color="k", linestyle="--", linewidth=1)

    def pick(d, key):
        s = d.get(key) or {}
        return s.get("mean"), s.get("ci95")

    bars(ax[0, 0], lambda r: pick(r["diagnostics"]["D-a"], "roundtrip_band_log_l1"),
         "D-a codec round trip, band-limited log-L1 (lower is better)", "log-L1")
    bars(ax[0, 1], lambda r: pick(r["diagnostics"]["D-b"], "generated"), "D-b CLAP cosine, generated vs prompt", "cosine")
    for i, n in enumerate(names):
        for key, style in (("real_floor", "k-"), ("other_file_prompt", "r-")):
            s = result[n]["diagnostics"]["D-b"].get(key) or {}
            if s.get("mean") is not None:
                ax[0, 1].hlines(s["mean"], i - 0.4, i + 0.4, colors="k" if key == "real_floor" else "r", linewidth=2)
    ax[0, 1].set_title("D-b CLAP cosine: bar generated, black real windows, red other-file prompt")
    bars(ax[1, 0], lambda r: pick(r["diagnostics"]["D-c"], "generated") if r["diagnostics"]["D-c"].get("applicable") else (None, None),
         "D-c seam ratio of generated latents (AR only; 1 = no seams)", "ratio", hline=1.0)
    bars(ax[1, 1], lambda r: pick(r["diagnostics"]["D-d"], "fraction_below_threshold"),
         "D-d fraction of generated units closer to train than val 1st percentile", "fraction", hline=0.01)
    fig.tight_layout()
    fig.savefig(out_dir / "s5_eval.png", dpi=120)


def fmt(s, digits=3):
    if not s or s.get("mean") is None:
        return "n/a"
    lo, hi = s["ci95"]
    return f"{s['mean']:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def write_readme(out_dir: Path, result: dict, args):
    names = list(result)
    files = sorted({c["file"] for n in names for c in result[n]["clips"]})
    L = ["# S5 generation and diagnostics", "",
         f"Prompt region: {args.prompt_region}. Checkpoint: {args.checkpoint}. Seeds {list(args.seeds)}, "
         f"{args.seconds} s per clip, NFE {args.nfe}, CFG {args.cfg}. Metrics and clips are in s5_eval.json; figure s5_eval.png.", "",
         "## Production checks (finite, duration, RMS within [-20, +10] dB of the region, whole clip and first second, clipping <= 0.1%)", "",
         "| codec | clips | pass | finite | duration | rms | rms first 1 s | clip |", "|---|---|---|---|---|---|---|---|"]
    for n in names:
        p = result[n]["production"]
        L.append(f"| {n} | {p['n_clips']} | {p['n_pass']} | {p['finite']} | {p['duration_ok']} | {p['rms_ok']} | "
                 f"{p['rms_first_ok']} | {p['clip_ok']} |")
    L += ["", "## Diagnostics (mean over prompts with 95% bootstrap interval)", "",
          "| codec | D-a round trip band log-L1 | D-a noise band log-L1 | D-b generated | D-b real floor | D-b other-file prompt | "
          "D-c seam gen | D-c seam real | D-d fraction | D-e params (M) | D-e peak GPU (GB) | D-e RTF |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        d = result[n]["diagnostics"]
        c = d["D-c"]
        L.append(f"| {n} | {fmt(d['D-a']['roundtrip_band_log_l1'])} | {fmt(d['D-a']['white_noise_band_log_l1'])} | "
                 f"{fmt(d['D-b']['generated'])} | {fmt(d['D-b']['real_floor'])} | {fmt(d['D-b']['other_file_prompt'])} | "
                 f"{fmt(c.get('generated')) if c['applicable'] else 'n/a'} | {fmt(c.get('real')) if c['applicable'] else 'n/a'} | "
                 f"{fmt(d['D-d']['fraction_below_threshold'])} | {d['D-e']['parameters'] / 1e6:.1f} | "
                 f"{d['D-e']['peak_gpu_bytes_generation'] / 1e9:.2f} | {d['D-e']['rtf_median']:.2f} |")
    L += ["", "D-d threshold (1st percentile of val-to-train nearest distances, exact copies left out) and val units with an exact train copy:", ""]
    for n in names:
        d = result[n]["diagnostics"]["D-d"]
        L.append(f"- {n}: threshold {d['threshold_1st_percentile_val_to_train']:.2f} (plain 1st percentile "
                 f"{d['threshold_raw_1st_percentile']:.2f}); val units with an exact train copy "
                 f"{d['val_units_with_exact_train_copy']:.3f} in files {d['val_files_with_exact_copy']}; "
                 f"generated units that are exact copies {fmt(d['fraction_exact_copy'])}")
    L += ["", "## Listening order", "",
          "1. `original/<file>.wav` is the real audio of the file from the start of the prompt region (48 kHz).",
          "2. `<codec>/roundtrip/<file>.wav` is the codec's own reconstruction of that audio: the best any model of that codec can do.",
          "3. `<codec>/generated/<file>_seed<k>.wav` is the generated clip from that region's first CLAP window.",
          "For each file listen to the original, then each codec's round trip, then its generated clips. Start with the files "
          f"{', '.join(files[:3])}; then go through the rest in alphabetical order.", ""]
    for f in files:
        stem = Path(f).stem
        L.append(f"### {stem}")
        L.append(f"- original/{stem}.wav")
        for n in names:
            L.append(f"- {n}/roundtrip/{stem}.wav")
            for s in args.seeds:
                L.append(f"- {n}/generated/{stem}_seed{s}.wav")
        L.append("")
    (out_dir / "README.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--prompt-region", choices=["val", "reserve"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default="best")
    ap.add_argument("--nfe", type=int, default=16)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--only", default=None, help="internal: evaluate this run folder name in this process")
    args = ap.parse_args()
    out_dir = Path(args.out).resolve()
    runs = Path(args.runs).resolve()
    args.dataset = str(Path(args.dataset).resolve())
    args.out = str(out_dir)

    if args.only is not None:
        run_dir = runs / args.only
        name = codec_name_of(run_dir / "model")
        res = evaluate_codec(args, name, run_dir)
        write_json(out_dir / name / "s5_codec.json", res)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, run_dir in discover(runs):
        cmd = [sys.executable, str(Path(__file__).resolve()), "--runs", str(runs), "--dataset", args.dataset,
               "--seeds", *map(str, args.seeds), "--seconds", str(args.seconds), "--prompt-region", args.prompt_region,
               "--out", str(out_dir), "--checkpoint", args.checkpoint, "--nfe", str(args.nfe), "--cfg", str(args.cfg),
               "--device", args.device, "--max-files", str(args.max_files), "--only", run_dir.name]
        log = out_dir / f"{name}.log"
        with open(log, "w") as fh:
            code = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
        if code != 0:
            raise RuntimeError(f"{name}: worker failed (exit {code}), see {log}\n" + log.read_text()[-3000:])
        result[name] = json.loads((out_dir / name / "s5_codec.json").read_text())
        print(f"{name}: {result[name]['production']}", flush=True)
    write_json(out_dir / "s5_eval.json", result)
    make_figure(out_dir, result)
    write_readme(out_dir, result, args)


if __name__ == "__main__":
    main()
