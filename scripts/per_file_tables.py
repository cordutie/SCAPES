"""Per-file tables (band-limited MSS and CLAP cosine distance to the original) of all models, from eval_extra.py outputs.
Usage (from SCAPES/): ../venv_SCAPES/bin/python scripts/per_file_tables.py --out ../runs/eval_tf
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../runs/eval_tf")
    out = Path(ap.parse_args().out)
    ms = [p.parent.name for p in sorted(out.glob("*/mss_band.json"))]
    D = {m: json.loads((out / m / "mss_band.json").read_text()) for m in ms}
    C = {m: json.loads((out / m / "clap_per_file.json").read_text()) for m in ms}
    files = list(D[ms[0]]["per_file"]["generated"])
    lines = ["# Per-file tables (MSS below 12 kHz, mean over the three windows; CLAP cosine distance of the clip embedding to the original)", ""]

    def tab(title, fn, fmt):
        lines.extend([f"**{title}**", "", "| file | " + " | ".join(m.replace("microtex_", "") for m in ms) + " |", "|---" * (len(ms) + 1) + "|"])
        for f in files:
            lines.append(f"| {f} | " + " | ".join(fmt % fn(m, f) for m in ms) + " |")
        lines.extend([f"| mean | " + " | ".join(fmt % np.mean([fn(m, f) for f in files]) for m in ms) + " |", ""])

    for row in ("generated", "codec_roundtrip"):
        tab(f"log-L1 below 12 kHz, {row}", lambda m, f: D[m]["per_file"][row][f]["mean_log_l1"], "%.2f")
        tab(f"spectral convergence below 12 kHz, {row}", lambda m, f: D[m]["per_file"][row][f]["mean_spec_conv"], "%.2f")
        tab(f"CLAP cosine distance to the original, {row}", lambda m, f, row=row: C[m][row][f], "%.3f")
    (out / "per_file_tables.md").write_text("\n".join(lines))
    print("written", out / "per_file_tables.md")


if __name__ == "__main__":
    main()
