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


CLEAN = ["forest.wav", "park.wav", "rain.wav", "tokio.wav"]


# Per-file val loss with the real past and with the null past (same protocol as evaluate_past_usage), the gain per file, its
# mean over files with a bootstrap interval over files, and the same for the clean files (no audio repeat of the train region)
def per_file_split(trainer, subset, n_boot=10000):
    import numpy as np
    from torch.utils.data import DataLoader
    loader = DataLoader(subset, batch_size=trainer.batch_size, shuffle=False)
    real = trainer._eval_loss(loader)[3]
    null = trainer._eval_loss(loader, null_past=True)[3]
    files = sorted(real)
    gain = {f: 100 * (1 - real[f] / null[f]) for f in files}

    def boot(v):
        v = np.array(v)
        m = v[np.random.RandomState(0).randint(0, len(v), (n_boot, len(v)))].mean(axis=1)
        return {"mean": float(v.mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], "n": len(v)}
    clean = [f for f in files if f in CLEAN]
    out = {"per_file_L_real": real, "per_file_L_null": null, "per_file_gain_percent": gain,
           "gain_percent_mean_over_files": boot([gain[f] for f in files])}
    if clean:
        out["clean_files"] = clean
        out["gain_percent_clean_files"] = boot([gain[f] for f in clean])
        out["L_real_clean"] = float(np.mean([real[f] for f in clean]))
        out["L_null_clean"] = float(np.mean([null[f] for f in clean]))
        rep = [f for f in files if f not in CLEAN]
        out["gain_percent_repeat_files"] = boot([gain[f] for f in rep])
    return out


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
    # A run trained without standardisation (training.standardize_latents False, e.g. the V1-style encodec48k run) has no
    # latent_stats.pt: the trainer then feeds the raw latents (and the raw per-atom scale channel) to the model, and the trainer
    # built above applies exactly that in _prepare_batch, so nothing has to be loaded here.
    stats_path = ck / "latent_stats.pt"
    if trainer.mu is None:
        if stats_path.exists():
            raise RuntimeError(f"{stats_path} exists but the config has training.standardize_latents False")
        stats_match = None
    else:
        saved = torch.load(stats_path, map_location="cpu", weights_only=True)
        stats_match = bool(torch.allclose(saved["mu"].cpu(), trainer.mu.cpu(), atol=1e-5) and
                           torch.allclose(saved["std"].cpu(), trainer.std.cpu(), atol=1e-5))
    val = trainer.val_loader.dataset
    r = trainer.evaluate_past_usage(val)
    r.update({"n_val_samples": len(val), "gain_percent": 100 * (1 - r["ratio"]), "latent_stats_match": stats_match,
              "standardized": trainer.mu is not None, "region": "val", "checkpoint": args.checkpoint, "model_dir": args.model_dir})
    r.update(per_file_split(trainer, val))
    Path(args.out).write_text(json.dumps(r, indent=1))
    print(json.dumps(r))
    shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
