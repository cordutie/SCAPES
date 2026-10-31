"""
SCAPES Dataprep Pipeline

Encodes the raw audio with the codec of dataprep.gin (the codec of generation.gin and its mode), precomputes semantic (CLAP) and
structure (audio features) annotations. Thin wrapper over SCAPES.pipeline.run_dataprep.

Examples:
    python scripts/dataprep.py --path ../datasets/microtex
    python scripts/dataprep.py --path ../datasets/FSDE

All settings come from <dataset_name>/config/dataprep.gin.
Can be run from any directory.
"""

import sys
import pathlib
import argparse

repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from SCAPES.pipeline import run_dataprep


def main():
    parser = argparse.ArgumentParser(
        description="SCAPES Dataprep — reads all settings from <config_dir>/dataprep.gin (default <dataset>/config)"
    )
    parser.add_argument("--path", "-p", type=str, required=True,
                        help="Path to the dataset")
    parser.add_argument("--config_dir", type=str, default=None,
                        help="Folder with dataprep.gin (default: <path>/config)")
    args = parser.parse_args()

    dataset_path = pathlib.Path(args.path).resolve()
    config_dir = pathlib.Path(args.config_dir).resolve() if args.config_dir else dataset_path / "config"
    if not (config_dir / "dataprep.gin").exists():
        print(f"Config not found: {config_dir / 'dataprep.gin'}")
        sys.exit(1)

    run_dataprep(dataset_path, config_dir=config_dir)


if __name__ == "__main__":
    main()
