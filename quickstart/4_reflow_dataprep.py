"""
SCAPES ReFlow Dataprep Pipeline

Generates a ReFlow dataset from a pretrained model: runs autoregressive
generation over the original dataset and saves (noise, atom) pairs.

Examples:
    python 4_reflow_dataprep.py -d ../../datasets/microtex -m ../../models/microtex -o ../../datasets/microtex_reflow -n 2

Run this from the `quickstart/` directory.
"""

import sys
import pathlib
import argparse

repo_root = pathlib.Path.cwd().parent
sys.path.insert(0, str(repo_root))

from SCAPES.reflow.data_generation import generate_reflow_dataset


def main():
    parser = argparse.ArgumentParser(
        description="SCAPES ReFlow Dataprep — generates (noise, atom) pairs from a pretrained model"
    )
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="Path to the original dataset")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Path to the pretrained model directory")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Path for the output ReFlow dataset (default: <dataset>_reflow)")
    parser.add_argument("-n", "--n_runs", type=int, default=2,
                        help="Number of generation runs per file (default: 2)")
    parser.add_argument("--nfe", type=int, default=32,
                        help="Number of ODE function evaluations (default: 32)")
    parser.add_argument("--cfg_scale", type=float, default=3.0,
                        help="CFG scale (default: 3.0)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (default: auto)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress detailed progress output")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.dataset.rstrip("/") + "_reflow"

    generate_reflow_dataset(
        dataset_path=args.dataset,
        model_dir=args.model,
        output_path=args.output,
        n_runs=args.n_runs,
        NFE=args.nfe,
        cfg_scale=args.cfg_scale,
        device=args.device,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
