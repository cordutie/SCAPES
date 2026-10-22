"""
SCAPES Training Pipeline

Trains a FlowModel + LocalEncoder on a preprocessed dataset (thin wrapper over SCAPES.pipeline.run_training).

Examples:
    python scripts/train.py --path ../datasets/microtex --save_path ../models/SCAPES
    python scripts/train.py --path ../datasets/micro_microtex --save_path ../models/SCAPES_test --config_dir /path/to/run_config --max_steps 2000

All settings come from <config_dir>/training.gin and dataprep.gin (default: <dataset>/config).
Can be run from any directory.
"""

import sys
import pathlib
import argparse

repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from SCAPES.pipeline import run_training


def main():
    parser = argparse.ArgumentParser(
        description="SCAPES Training — reads all settings from <config_dir>/training.gin and dataprep.gin"
    )
    parser.add_argument("--path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--save_path", type=str, required=True, help="Where to save model checkpoints")
    parser.add_argument("--config_dir", type=str, default=None, help="Folder with dataprep.gin and training.gin (default: <path>/config)")
    parser.add_argument("--max_steps", type=int, default=None, help="Stop after this many optimiser steps (overrides training.max_steps)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from 'latest' or 'best'")
    args = parser.parse_args()

    run_training(args.path, args.save_path, config_dir=args.config_dir, resume=args.resume, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
