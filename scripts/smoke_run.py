"""Smoke test of a prepared run: N real training steps, then one validation pass, with the numbers needed to project a run.

Usage (from SCAPES/):
    ../venv_SCAPES/bin/python scripts/smoke_run.py --dataset ../datasets/microtex --config ../runs/X/config --out ../runs/X_smoke
Prints seconds/step (after 3 warm-up steps), peak GPU memory, steps per epoch, validation pass time, and the projected
time per epoch and for the epoch cap. Writes <out>/smoke.json. Audio is never generated.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from SCAPES.data.config_loader import load_training_config
from SCAPES.pipeline import build_trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_training_config(Path(args.config) / "training.gin")
    trainer = build_trainer(args.dataset, out / "model", config_dir=args.config)
    steps_per_epoch = len(trainer.train_loader)
    val_batches = len(trainer.val_loader) if trainer.val_loader is not None else 0

    # warm-up (3 steps), then the timed steps; train_epoch stops at max_steps
    trainer.max_steps = 3
    trainer.train_epoch()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.max_steps = 3 + args.steps
    trainer.train_epoch()
    torch.cuda.synchronize()
    sec_step = (time.time() - t0) / args.steps
    peak_train = torch.cuda.max_memory_allocated() / 2**30

    # one deterministic validation pass
    t0 = time.time()
    trainer.val_epoch()
    torch.cuda.synchronize()
    val_s = time.time() - t0
    peak_all = torch.cuda.max_memory_allocated() / 2**30

    epoch_s = sec_step * steps_per_epoch + val_s
    result = {
        "steps": args.steps, "seconds_per_step": sec_step, "peak_gpu_gib_train": peak_train, "peak_gpu_gib_with_val": peak_all,
        "steps_per_epoch": steps_per_epoch, "val_batches": val_batches, "val_pass_seconds": val_s,
        "epoch_seconds": epoch_s, "epoch_cap": cfg.epochs, "cap_hours": epoch_s * cfg.epochs / 3600,
        "train_samples": len(trainer.train_loader.dataset), "val_samples": len(trainer.val_loader.dataset),
    }
    print(json.dumps(result, indent=1))
    (out / "smoke.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
