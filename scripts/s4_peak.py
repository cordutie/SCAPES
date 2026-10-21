"""Slice B9: peak GPU memory of the dataprep of one codec, measured on a scratch copy of the dataset (raw files linked).

Usage (from SCAPES/), one fresh process per codec:
    ../venv_SCAPES/bin/python scripts/s4_peak.py --dataset ../datasets/microtex --config ../runs/s5/<codec>/config \
        --scratch <empty folder> --out ../runs/s4/peak_<codec>.json
The scratch folder gets raw/ (symlink) and config/ (copy); the real dataset is not touched. Delete the scratch after.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SCAPES.pipeline import run_dataprep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "raw").symlink_to(Path(args.dataset).resolve() / "raw")
    shutil.copytree(args.config, scratch / "config")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    store = run_dataprep(scratch, config_dir=scratch / "config")
    result = {"rep_key": store.key, "seconds": time.time() - t0,
              "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
