"""Codec-independent audio check for repeated material between the train region and a held-out region.

Per file (mono, 48 kHz, no resampling): train = [0, 0.8 D); 12 non-overlapping 1 s windows of the region (val or
reserve) are searched in the train audio by normalised cross-correlation (FFT, sliding energy from a cumulative sum).
Reports the best NCC and its lag per window and, per file, the fraction of windows with NCC >= 0.99 (exact repeat)
and >= 0.9 (near repeat). It only reads data.

    ../venv_SCAPES/bin/python scripts/region_dup_check.py --dataset ../datasets/microtex --region val --out FILE
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

N_WINDOWS = 12
TRAIN_END = 0.8
BOUNDS = {"val": (0.8, 0.9), "reserve": (0.9, 1.0)}


# Best normalised cross-correlation of window w inside signal x: returns (ncc, lag index of the match start)
def best_ncc(x: np.ndarray, w: np.ndarray):
    n = len(w)
    if len(x) < n:
        return float("nan"), -1
    w = w - w.mean()
    wn = np.sqrt((w ** 2).sum())
    if wn == 0:
        return float("nan"), -1
    cross = fftconvolve(x, w[::-1], mode="valid")                 # sum x[i+j] * w[j] (w has zero mean)
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x ** 2)])
    s1 = c1[n:] - c1[:-n]
    s2 = c2[n:] - c2[:-n]
    var = np.maximum(s2 - s1 ** 2 / n, 1e-12)
    ncc = cross / (np.sqrt(var) * wn)
    i = int(np.argmax(ncc))
    return float(ncc[i]), i


def check_file(path: Path, region: str) -> dict:
    x, sr = sf.read(str(path), always_2d=True, dtype="float64")
    x = x.mean(axis=1)
    d = len(x)
    lo, hi = int(BOUNDS[region][0] * d), int(BOUNDS[region][1] * d)
    train = x[:int(TRAIN_END * d)]
    # 12 non-overlapping 1 s windows spread evenly over the region
    starts = np.linspace(lo, hi - sr, N_WINDOWS).astype(int) if hi - lo >= N_WINDOWS * sr else \
        np.arange(lo, hi - sr + 1, sr)[:N_WINDOWS]
    wins = []
    for s in starts:
        ncc, lag = best_ncc(train, x[s:s + sr])
        wins.append({"start_s": s / sr, "ncc": ncc, "match_start_s": lag / sr if lag >= 0 else None,
                     "lag_s": (s - lag) / sr if lag >= 0 else None})
    v = np.array([w["ncc"] for w in wins], dtype=float)
    return {"windows": wins, "n_windows": len(wins), "fraction_exact_ncc_ge_0.99": float(np.mean(v >= 0.99)),
            "fraction_near_ncc_ge_0.9": float(np.mean(v >= 0.9)), "max_ncc": float(np.nanmax(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--region", choices=["val", "reserve"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for p in sorted((Path(a.dataset) / "raw").glob("*.wav")):
        res[p.name] = check_file(p, a.region)
        print(p.name, res[p.name]["fraction_exact_ncc_ge_0.99"], res[p.name]["fraction_near_ncc_ge_0.9"], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"region": a.region, "files": res}, indent=1))


if __name__ == "__main__":
    main()
