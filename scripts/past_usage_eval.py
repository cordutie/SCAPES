"""Deterministic val loss with the real past and with the null past, for a trained model's best (or last) checkpoint.

Builds the trainer in a scratch folder (the model folder is only read), loads the saved weights, checks that the latent
statistics match the saved ones, and calls trainer.evaluate_past_usage on the val split.
Usage (from SCAPES/): ../venv_SCAPES/bin/python scripts/past_usage_eval.py --model-dir ../runs/s5/encodec24k/model \
    --config-dir ../runs/s5/encodec24k/config --dataset ../datasets/microtex --scratch /path/scratch --out out.json
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--config-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--checkpoint", default="best")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from SCAPES.pipeline import build_trainer
    scratch = Path(args.scratch)
    shutil.rmtree(scratch, ignore_errors=True)
    trainer = build_trainer(args.dataset, scratch, config_dir=args.config_dir)
    ck = Path(args.model_dir) / "checkpoints"
    trainer.model.load_state_dict(torch.load(ck / f"{args.checkpoint}_flow_model.pt", map_location=trainer.device)["model_state_dict"])
    trainer.local_encoder.load_state_dict(
        torch.load(ck / f"{args.checkpoint}_local_encoder.pt", map_location=trainer.device)["model_state_dict"])
    saved = torch.load(ck / "latent_stats.pt", map_location="cpu", weights_only=True)
    stats_match = bool(torch.allclose(saved["mu"].cpu(), trainer.mu.cpu(), atol=1e-5) and
                       torch.allclose(saved["std"].cpu(), trainer.std.cpu(), atol=1e-5)) if trainer.mu is not None else None
    val = trainer.val_loader.dataset
    r = trainer.evaluate_past_usage(val)
    r.update({"n_val_samples": len(val), "gain_percent": 100 * (1 - r["ratio"]), "latent_stats_match": stats_match,
              "checkpoint": args.checkpoint, "model_dir": args.model_dir})
    Path(args.out).write_text(json.dumps(r, indent=1))
    print(json.dumps(r))
    shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
