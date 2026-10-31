"""Write one run config per SPEC = codec:rate:memory (memory optional), from the config of a dataset.

Each config merges <dataset>/config (without its geometry lines), config/generation.gin and the three keys of the SPEC.
Output: <out>/<codec>-<rate>hz-<memory>s/{dataprep,training}.gin (memory is "default", with no "s", when omitted).

Examples (from SCAPES/):
    python scripts/make_run_configs.py --dataset ../datasets/microtex --out ../runs/cfg music2latent:10 encodec24:20:1.0
"""
import sys
import argparse
import pathlib

repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from SCAPES.pipeline import build_run_config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="dataset folder (its config/ holds dataprep.gin, training.gin)")
    parser.add_argument("--out", required=True, help="folder that receives one sub-folder per SPEC")
    parser.add_argument("specs", nargs="+", help="codec:rate[:memory], e.g. music2latent:10:0.5")
    args = parser.parse_args()

    generation_gin = repo_root / "SCAPES" / "config" / "generation.gin"
    for spec in args.specs:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            parser.error(f"bad SPEC '{spec}', expected codec:rate[:memory]")
        codec, rate = parts[0], float(parts[1])
        overrides = [f'codec = "{codec}"', f"conditioning_rate = {rate}"]
        memory = "default"
        suffix = ""
        if len(parts) == 3:
            memory = parts[2]
            suffix = "s"
            overrides.append(f"conditioning_memory_seconds = {float(memory)}")
        out_dir = pathlib.Path(args.out) / f"{codec}-{rate:g}hz-{memory}{suffix}" / "config"
        build_run_config(pathlib.Path(args.dataset) / "config", [generation_gin], overrides, out_dir)
        print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
