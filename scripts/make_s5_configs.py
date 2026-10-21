"""Slice B9: writes the S5 run configs (runs/s5/<codec>/config/{dataprep,training}.gin) with pipeline.build_run_config.

Usage (from SCAPES/):
    ../venv_SCAPES/bin/python scripts/make_s5_configs.py --dataset ../datasets/microtex --out ../runs/s5
Early stopping decides the length of every run: no training.max_steps (the trainer disables patience when it is set), a
very large training.epochs that is never reached, patience 10 epochs on the deterministic validation loss, validation
every epoch, no validation audio, checkpoints best and last only. encodec48k is already trained
(runs/microtex_encodec48k): its config is copied unchanged and only its dataprep is checked.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SCAPES.pipeline import build_run_config

SNIPPETS = {"encodec24k": "encodec24k_ar.gin", "spectrostream": "spectrostream_ar.gin", "codicodec": "codicodec_atoms.gin"}
AR_CODECS = ("encodec24k", "spectrostream")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--decision", default=str(REPO.parent / "runs" / "s3b_decision.json"))
    ap.add_argument("--encodec48k-config", default=str(REPO.parent / "runs" / "microtex_encodec48k" / "config"))
    args = ap.parse_args()

    prefix = json.loads(Path(args.decision).read_text())["ar_prefix_seconds"]
    out = Path(args.out)
    for codec, snippet in SNIPPETS.items():
        overrides = [
            "splits.val_split_ratio = 0.1",
            "splits.reserve_ratio = 0.1",
            "training.epochs = 100000   # never reached: patience (10 epochs, deterministic val) ends the run",
            "training.save_resume_states = True   # a crash can be continued; delete last_trainer_state.pt after the run",
            "training.audio_val_freq = 100000000   # no validation audio",
            "training.log_every = 50",
            "training.seed = 0",
        ]
        if codec in AR_CODECS:
            overrides.append(f"ar.prefix_seconds = {prefix}")
        build_run_config(Path(args.dataset) / "config", [REPO / "SCAPES" / "config" / "codecs" / snippet], overrides,
                         out / codec / "config")
        print("wrote", out / codec / "config")

    # EnCodec 48k: the config of the finished run, copied (same representation key, nothing re-encoded)
    dst = out / "encodec48k" / "config"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("dataprep.gin", "training.gin"):
        shutil.copy2(Path(args.encodec48k_config) / name, dst / name)
    print("copied", dst)


if __name__ == "__main__":
    main()
