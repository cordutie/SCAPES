"""Two B12 figures from s5_eval.json (read-only): RMS deviation per clip with the [-20, +10] dB band, and the per-file
reconstruction ceiling (codec round trip MSS) with one colour per file.

    ../venv_SCAPES/bin/python scripts/b12_figures.py --eval ../runs/s5_eval
"""
import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    a = ap.parse_args()
    out = Path(a.eval)
    r = json.loads((out / "s5_eval.json").read_text())
    names = list(r)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    # RMS deviation per clip (dB relative to the region RMS of the file), failures marked
    fig = Figure(figsize=(13, 4.5))
    ax = fig.subplots()
    files = sorted({c["file"] for n in names for c in r[n]["clips"]})
    for i, n in enumerate(names):
        for c in r[n]["clips"]:
            x = files.index(c["file"]) + (i - 1.5) * 0.18
            ok = c["pass"]
            ax.scatter(x, c["rms_db"], color=colors[i], s=22, marker="o" if ok else "X", label=n if c is r[n]["clips"][0] else None)
    ax.axhspan(-20, 10, color="grey", alpha=0.12, label="allowed [-20, +10] dB")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(range(len(files)))
    ax.set_xticklabels([f[:-4] for f in files], rotation=45, ha="right")
    ax.set_ylabel("clip RMS relative to the reserve region (dB)")
    ax.set_title("RMS deviation per clip (X: a production check failed on this clip, any item)")
    ax.legend(ncol=5, fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out / "rms_deviation_per_clip.png", dpi=120)

    # reconstruction ceiling per file: band log-L1 of the codec round trip, 16 colours
    fig = Figure(figsize=(13, 4.8))
    ax = fig.subplots()
    cmap = colormaps["tab20"]
    w = 0.8 / len(files)
    for j, f in enumerate(files):
        vals = [r[n]["diagnostics"]["D-a"]["files"][f]["roundtrip"]["band_log_l1"] if f in r[n]["diagnostics"]["D-a"]["files"] else np.nan for n in names]
        ax.bar(np.arange(len(names)) + (j - 7.5) * w, vals, width=w, color=cmap(j / 20), label=f[:-4])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("band (<12 kHz) log-L1 of the codec round trip")
    ax.set_title("Reconstruction ceiling M(x, D(E(x))) per reserve file (lower is better)")
    ax.legend(ncol=8, fontsize=7)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out / "reconstruction_ceiling_per_file.png", dpi=120)


if __name__ == "__main__":
    main()
