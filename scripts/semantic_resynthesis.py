"""Semantic resynthesis of the first 10 s of the microtex files with the S5 models (repo's run_resynthesis_pipeline).

For every unit of the model the CLAP embedding is the stored one of the SOURCE audio at that unit (time-varying, aligned as
in teacher_forced_eval.py), and every unit is generated from noise with NO real past (TF=False, cold start), then decoded:
AR models through the stream decoder, atoms with overlap-add. Seed 0, NFE 16, CFG 3.0, best checkpoint. The excerpts are
TRAIN-region audio (as in the teacher-forced evaluation): this measures fit, not generalisation. No reserve audio is used.

Three things should not be confused: prompt synthesis (s5_eval: one CLAP held for all units, free past), teacher forcing
(real past, real CLAP per unit) and this one (real CLAP per unit, generated past).

Usage (from SCAPES/):
    ../venv_SCAPES/bin/python scripts/semantic_resynthesis.py --runs ../runs/s5 --dataset ../datasets/microtex \
        --eval-tf ../runs/eval_tf --out ../runs/semantic_resynth
Outputs: <out>/<codec>/<file>.wav, <out>/<codec>/metrics.json, <out>/summary.{md,json}, <out>/README.md.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import eval_metrics as em
import teacher_forced_eval as tf

CODECS = {"encodec24k": "microtex_encodec24k_ar", "spectrostream": "microtex_spectrostream_ar",
          "encodec48k": "microtex_encodec48k_atoms", "codicodec": "microtex_codicodec_atoms"}
BAND_HZ = 12000.0


# Cold-start generation: real per-unit CLAP, no real past (TF=False); returns the audio [C, N] at the codec rate
def semantic_resynthesis(engine, claps, nfe: int, cfg: float, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    engine.decode_seed = seed
    timeline = engine.build_base_timeline([None] * len(claps), claps, None, default_TF=False)
    timeline = engine.generate(timeline, NFE=nfe, cfg_scale=cfg)
    return engine.decode_timeline(timeline)


# RMS (dB) of a mono signal in consecutive windows of `win` samples
def rms_db_track(x: np.ndarray, win: int) -> list:
    n = len(x) // win
    return [float(20 * np.log10(np.sqrt(np.mean(x[i * win:(i + 1) * win] ** 2)) + 1e-12)) for i in range(n)]


def read_wav_mono_48k(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return em.to_mono(em.resample_to_48k(x.T, sr))


def run_codec(args, codec: str):
    from SCAPES.auxiliar.clap_wrapper import CLAPWrapper
    from SCAPES.data.dataprep.semantic import load_audio_48k
    from SCAPES.data.representations import RepresentationStore
    from SCAPES.inference.FlowInference import FlowInference

    out = Path(args.out) / codec
    out.mkdir(parents=True, exist_ok=True)
    tf_dir = Path(args.eval_tf) / CODECS[codec]
    engine = FlowInference(Path(args.runs) / codec / "model", device=args.device, checkpoint="best")
    geom, info = engine.geom, engine.info
    store = RepresentationStore(args.dataset, geom, info)
    store.ensure()
    clap_key = store.clap_key("2023", geom.context_seconds, engine.semantic_random_extension)
    files = sorted((Path(args.dataset) / "raw").glob("*.wav"))
    if args.max_files:
        files = files[:args.max_files]
    starts = tf.unit_starts(geom, tf.SECONDS)
    sr_codec = info.sample_rate
    ref, ref_next, gen, tfg, rt, per_clip = {}, {}, {}, {}, {}, {}
    t0 = time.time()
    for path in files:
        stem = path.stem
        audio = load_audio_48k(path).numpy()
        n10 = int(tf.SECONDS * em.SR)
        ref[stem] = em.to_mono(audio[:, :n10])
        ref_next[stem] = em.to_mono(audio[:, n10:2 * n10])
        latents = store.load_latents(path)
        clap = store.load_grid_array("clap", clap_key, path)
        _, claps = tf.real_units(geom, latents, clap, starts, engine._standardise)
        t1 = time.time()
        audio_g = semantic_resynthesis(engine, claps, args.nfe, args.cfg, args.seed)
        tf.write_wav(out / f"{stem}.wav", audio_g.numpy(), sr_codec)
        gen[stem] = em.to_mono(em.resample_to_48k(audio_g.numpy(), sr_codec))
        g = audio_g.numpy()
        track = rms_db_track(gen[stem], em.SR)
        per_clip[stem] = {"seconds": g.shape[-1] / sr_codec, "finite": bool(np.isfinite(g).all()), "peak": float(np.abs(g).max()),
                          "gen_seconds": time.time() - t1, "rms_db_per_second": track,
                          "rms_db_first5": float(np.mean(track[:5])), "rms_db_last5": float(np.mean(track[-5:])),
                          "ref_rms_db_first5": float(np.mean(rms_db_track(ref[stem], em.SR)[:5])),
                          "ref_rms_db_last5": float(np.mean(rms_db_track(ref[stem], em.SR)[-5:]))}
        tfg[stem] = read_wav_mono_48k(tf_dir / "generated" / f"{stem}.wav")
        rt[stem] = read_wav_mono_48k(tf_dir / "codec_roundtrip" / f"{stem}.wav")
        print(f"[{codec}] {stem}: {per_clip[stem]['seconds']:.2f} s in {per_clip[stem]['gen_seconds']:.1f} s", flush=True)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if str(args.device).startswith("cuda") else None
    del engine
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()

    # metrics on [0.4 s, common end), as the teacher-forced evaluation
    stems = list(ref)
    a = int(em.EXCLUDE_SECONDS * em.SR)
    e = min([len(v) for d in (ref, gen, tfg, rt) for v in d.values()])
    noise = {s: em.white_noise_like(ref[s][a:e], tf.NOISE_SEED_BASE + i, n=e) for i, s in enumerate(stems)}
    rows = {"semantic_resynthesis": gen, "teacher_forced": tfg, "codec_roundtrip": rt, "white_noise": noise}
    mss = {}
    for name, sig in rows.items():
        full = [em.mss(ref[s][a:e], sig[s][a:e]) for s in stems]
        band = [em.mss(ref[s][a:e], sig[s][a:e], max_hz=BAND_HZ) for s in stems]
        mss[name] = {"full_log_l1": float(np.mean([r["mean_log_l1"] for r in full])),
                     "full_spec_conv": float(np.mean([r["mean_spec_conv"] for r in full])),
                     "band_log_l1": float(np.mean([r["mean_log_l1"] for r in band])),
                     "band_spec_conv": float(np.mean([r["mean_spec_conv"] for r in band])),
                     "per_file_full_log_l1": {s: r["mean_log_l1"] for s, r in zip(stems, full)},
                     "per_file_band_log_l1": {s: r["mean_log_l1"] for s, r in zip(stems, band)}}
    clap_model = CLAPWrapper(version="2023", use_cuda=str(args.device).startswith("cuda"))
    sets = dict(rows)
    sets["floor"] = {s: ref_next[s] for s in stems}
    emb_clip, emb_seg = {}, {}
    for name, sig in [("reference", ref)] + list(sets.items()):
        emb_clip[name] = np.stack([tf.clip_embedding(clap_model, sig[s], a, e, args.device) for s in stems])
        emb_seg[name] = np.concatenate([tf.segment_embeddings(clap_model, sig[s], a, e, args.device) for s in stems])
    del clap_model
    ev_c = {k: v for k, v in emb_clip.items() if k != "reference"}
    ev_s = {k: v for k, v in emb_seg.items() if k != "reference"}
    kad_clip = em.kad_with_bootstrap(emb_clip["reference"], ev_c, np.arange(len(stems)), args.n_boot, 0, "floor")
    kad_seg = em.kad_with_bootstrap(emb_seg["reference"], ev_s, np.repeat(np.arange(len(stems)), tf.N_SEGMENTS), args.n_boot, 0, "floor")
    metrics = {"codec": codec, "mode": geom.mode, "checkpoint": "best", "seed": args.seed, "nfe": args.nfe, "cfg": args.cfg,
               "region": {"start_s": a / em.SR, "end_s": e / em.SR}, "files": stems, "mss": mss, "per_clip": per_clip,
               "kad": {"clip_level": kad_clip, "segment_level": kad_seg, "n_boot": args.n_boot},
               "peak_gpu_gb": peak_gb, "wall_seconds": time.time() - t0, "units_per_file": len(starts)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))


LABEL = {"semantic_resynthesis": "semantic resynthesis (cold start, real CLAP per unit)", "teacher_forced": "teacher forced (real past)",
         "codec_roundtrip": "codec round trip", "white_noise": "white noise"}


def write_summary(out_root: Path):
    models = {c: json.loads((out_root / c / "metrics.json").read_text()) for c in CODECS if (out_root / c / "metrics.json").exists()}
    (out_root / "summary.json").write_text(json.dumps(models, indent=1))
    md = ["# Semantic resynthesis: summary", "",
          "First 10 s of the 16 microtex files: TRAIN-region audio (fit, not generalisation; no reserve use). Best checkpoints, seed 0, NFE 16, "
          "CFG 3.0. Mono 48 kHz, region 0.4 s to the common end; MSS = mean of three big windows (log-L1 / spectral convergence), band = bins below 12 kHz. "
          "The teacher-forced and round-trip rows are re-read from the 16-bit wavs of runs/eval_tf, the semantic row is computed from the float output, "
          "so full-band log-L1 can differ by about 0.01-0.09 from the eval_tf summary (quantisation floor in near-empty bins).", "",
          "## MSS", "", "| model | row | full L1 | full SC | band L1 | band SC |", "|---|---|---|---|---|---|"]
    for c, m in models.items():
        for r in ("semantic_resynthesis", "teacher_forced", "codec_roundtrip", "white_noise"):
            x = m["mss"][r]
            md.append(f"| {c} | {LABEL[r]} | {x['full_log_l1']:.3f} | {x['full_spec_conv']:.3f} | {x['band_log_l1']:.3f} | {x['band_spec_conv']:.3f} |")
    for level, title in (("clip_level", "clip level (16 vs 16)"), ("segment_level", "segment level (80 vs 80)")):
        md += ["", f"## KAD, {title}: point [95% bootstrap interval]; last column is the paired difference to the floor (original 0-10 s vs 10-20 s)", "",
               "| model | row | KAD | minus floor |", "|---|---|---|---|"]
        for c, m in models.items():
            k = m["kad"][level]["sets"]
            for r in ("semantic_resynthesis", "teacher_forced", "codec_roundtrip", "white_noise", "floor"):
                d = k[r].get("minus_floor")
                md.append(f"| {c} | {LABEL.get(r, 'floor')} | {k[r]['kad']:.2f} [{k[r]['ci95'][0]:.2f}, {k[r]['ci95'][1]:.2f}] | "
                          + ("-" if d is None else f"{d['mean']:.2f} [{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}]") + " |")
    md += ["", "## Drift check: mean RMS (dB) of the first 5 s and of the last 5 s, mean over files (generated vs original)", "",
           "| model | gen first 5 s | gen last 5 s | original first 5 s | original last 5 s | non-finite clips | max peak | seconds per clip |", "|---|---|---|---|---|---|---|---|"]
    for c, m in models.items():
        pc = m["per_clip"].values()
        md.append(f"| {c} | {np.mean([p['rms_db_first5'] for p in pc]):.1f} | {np.mean([p['rms_db_last5'] for p in pc]):.1f} | "
                  f"{np.mean([p['ref_rms_db_first5'] for p in pc]):.1f} | {np.mean([p['ref_rms_db_last5'] for p in pc]):.1f} | "
                  f"{sum(not p['finite'] for p in pc)} | {max(p['peak'] for p in pc):.2f} | {np.mean([p['gen_seconds'] for p in pc]):.1f} |")
    (out_root / "summary.md").write_text("\n".join(md) + "\n")
    # README with the listening order
    stems = next(iter(models.values()))["files"] if models else []
    R = ["# Semantic resynthesis: listening folder", "",
         "Each model generates the first 10 s of every file from the real per-unit CLAP of that audio and a generated (not real) past. "
         "Train-region excerpts. Metrics in summary.md.", "",
         "Listening order per file: 1 original, 2 semantic resynthesis (this folder), 3 teacher forced (real past), 4 codec round trip "
         "(the limit of the codec). Paths are relative to this folder.", ""]
    for s in stems:
        R.append(f"### {s}")
        R.append(f"1. ../eval_tf/original_10s/{s}.wav")
        for c in models:
            R += [f"- {c}: 2. {c}/{s}.wav, 3. ../eval_tf/{CODECS[c]}/generated/{s}.wav, 4. ../eval_tf/{CODECS[c]}/codec_roundtrip/{s}.wav"]
        R.append("")
    (out_root / "README.md").write_text("\n".join(R))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="../runs/s5")
    ap.add_argument("--dataset", default="../datasets/microtex")
    ap.add_argument("--eval-tf", default="../runs/eval_tf")
    ap.add_argument("--out", default="../runs/semantic_resynth")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--nfe", type=int, default=16)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--only", default=None, help="internal: run this codec in this process")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    for k in ("runs", "dataset", "eval_tf", "out"):
        setattr(args, k, str(Path(getattr(args, k)).resolve()))
    if args.only:
        run_codec(args, args.only)
        return
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        for codec in CODECS:      # one process per codec (CoDiCodec flips global torch flags at import)
            cmd = [sys.executable, str(Path(__file__).resolve()), "--only", codec] + \
                  [x for k in ("runs", "dataset", "eval_tf", "out", "device", "nfe", "cfg", "seed", "n_boot", "max_files")
                   for x in (f"--{k.replace('_', '-')}", str(getattr(args, k)))]
            with open(Path(args.out) / f"{codec}.log", "w") as fh:
                code = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
            if code != 0:
                raise RuntimeError(f"{codec} failed, see {Path(args.out) / (codec + '.log')}")
    write_summary(Path(args.out))


if __name__ == "__main__":
    main()
