"""Extra analysis of the teacher-forced evaluation, computed from the saved wavs (no model needed).

For each model folder in <out>/ with generated/ and codec_roundtrip/:
  - <model>/mss_band.json: MSS (log-L1 and spectral convergence) restricted to the bins below --max-hz, for the rows
    generated, codec_roundtrip, white_noise, delayed_0.5s, per file and averaged (same signals, region, noise seeds and
    delay as teacher_forced_eval.py; the wavs are 16-bit and clipped, so the full-band numbers recomputed here are
    compared with the stored ones as a check).
  - <model>/clap_per_file.json: per file, the cosine distance between the CLAP clip embedding of the original and of
    the generated / round trip audio (a per-file proxy: KAD itself needs a set).
Usage (from SCAPES/): ../venv_SCAPES/bin/python scripts/eval_extra.py --out ../runs/eval_tf
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_metrics as em
import teacher_forced_eval as tf


def read_mono(path: Path, sr_target=em.SR):
    x, sr = sf.read(str(path), always_2d=True)
    return em.to_mono(x.T), sr


def load_signals(out: Path, model: Path):
    ref, gen, rt = {}, {}, {}
    for p in sorted((out / "original_10s").glob("*.wav")):
        ref[p.stem] = read_mono(p)[0]
        for name, store in (("generated", gen), ("codec_roundtrip", rt)):
            x, sr = read_mono(model / name / p.name)
            store[p.stem] = em.to_mono(em.resample_to_48k(x[None, :], sr))
    return ref, gen, rt


def mean_rows(per_file):
    keys = next(iter(per_file.values())).keys()
    return {k: float(np.mean([v[k] for v in per_file.values()])) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../runs/eval_tf")
    ap.add_argument("--max-hz", type=float, default=12000.0)
    ap.add_argument("--no-clap", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    clap_model = None
    if not args.no_clap:
        from SCAPES.auxiliar.clap_wrapper import CLAPWrapper
        import torch
        clap_model = CLAPWrapper(version="2023", use_cuda=torch.cuda.is_available())
    a = int(em.EXCLUDE_SECONDS * em.SR)
    d = int(tf.DELAY_SECONDS * em.SR)
    for mp in sorted(out.glob("*/metrics.json")):
        model = mp.parent
        stored = json.loads(mp.read_text())
        ref, gen, rt = load_signals(out, model)
        stems = list(ref)
        e = min([len(v) for v in ref.values()] + [len(v) for v in gen.values()] + [len(v) for v in rt.values()])
        noise = {s: em.white_noise_like(ref[s][a:e], tf.NOISE_SEED_BASE + i, n=e) for i, s in enumerate(stems)}
        res = {"max_hz": args.max_hz, "region_end_s": e / em.SR, "full_band_check": {}, "per_file": {}, "rows": {}}
        for band, mh in (("full", None), ("band", args.max_hz)):
            pf = {r: {} for r in ("generated", "codec_roundtrip", "white_noise", "delayed_0.5s")}
            for s in stems:
                for name, sig in (("generated", gen), ("codec_roundtrip", rt), ("white_noise", noise)):
                    pf[name][s] = tf.mss_row_values(em.mss(ref[s][a:e], sig[s][a:e], max_hz=mh))
                pf["delayed_0.5s"][s] = tf.mss_row_values(em.mss(ref[s][a + d:e], ref[s][a:e - d], max_hz=mh))
            rows = {r: mean_rows(v) for r, v in pf.items()}
            if band == "band":
                res["per_file"], res["rows"] = pf, rows
            else:
                res["full_band_check"] = {r: {"recomputed_L1": rows[r]["mean_log_l1"], "stored_L1": stored["mss"]["rows"][r]["mean_log_l1"],
                                              "recomputed_SC": rows[r]["mean_spec_conv"], "stored_SC": stored["mss"]["rows"][r]["mean_spec_conv"]}
                                          for r in rows}
        (model / "mss_band.json").write_text(json.dumps(res, indent=1))
        print(model.name, {r: (round(v["mean_log_l1"], 3), round(v["mean_spec_conv"], 3)) for r, v in res["rows"].items()}, flush=True)
        if clap_model is not None:
            dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            dist = {"generated": {}, "codec_roundtrip": {}}
            for s in stems:
                er = tf.clip_embedding(clap_model, ref[s], a, e, dev)
                for name, sig in (("generated", gen), ("codec_roundtrip", rt)):
                    dist[name][s] = float(1.0 - np.dot(er, tf.clip_embedding(clap_model, sig[s], a, e, dev)))
            (model / "clap_per_file.json").write_text(json.dumps(dist, indent=1))


if __name__ == "__main__":
    main()
