"""Writes the run configs of the sample-level atom variants (runs/v2_atoms/<name>/config/{dataprep,training}.gin) with
pipeline.build_run_config, or a chain launcher for some of the runs.

Usage (from SCAPES/):
    ../venv_SCAPES/bin/python scripts/make_atoms_configs.py --dataset ../datasets/microtex_v2 --out ../runs/v2_atoms [--variants NAME ...] [--seeds 0]
    ../venv_SCAPES/bin/python scripts/make_atoms_configs.py --out ../runs/v2_atoms --chain launch_chain.sh --chain-names RUN [RUN ...]
Phase 1: the four consistent sizes (c4 is the template, the others change atoms.seconds only). The independent variants are
added in phase 2. Every run trains until patience fires (no step budget), best and last checkpoints only.
Chain mode writes no config: it checks that every named run has config/ and writes the launcher into --out.
"""
import argparse
import stat
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SCAPES.pipeline import build_run_config

CODECS = REPO / "SCAPES" / "config" / "codecs"
# name -> (snippet, overrides)
VARIANTS = {
    "m2l-cons-c2": ("music2latent_consistent.gin", ['atoms.seconds = 0.18']),
    "m2l-cons-c4": ("music2latent_consistent.gin", []),
    "m2l-cons-c6": ("music2latent_consistent.gin", ['atoms.seconds = 0.55']),
    "m2l-cons-c8": ("music2latent_consistent.gin", ['atoms.seconds = 0.74']),
}
# the same for every run: early stopping decides the length
COMMON = [
    "splits.val_split_ratio = 0.1",
    "splits.reserve_ratio = 0.1",
    "training.epochs = 100000   # never reached: patience (10 validations of 100 steps, deterministic val) ends the run",
    "training.save_resume_states = True   # a crash can be continued; the launcher deletes last_trainer_state.pt after the run",
    "training.audio_val_freq = 100000000   # no validation audio",
    "training.log_every = 50",
]


def write_configs(args):
    names = args.variants or list(VARIANTS)
    for seed in args.seeds:
        for name in names:
            snippet, overrides = VARIANTS[name]
            run = name if seed == 0 else f"{name}-s{seed}"
            lines = COMMON + [f"training.seed = {seed}"] + overrides
            build_run_config(Path(args.dataset) / "config", [CODECS / snippet], lines, Path(args.out) / run / "config")
            print("wrote", Path(args.out) / run / "config")


LAUNCHER = """#!/usr/bin/env bash
# Sequential chain of atom runs: {names}
# Start with:  cd {here} && setsid nohup ./{file} > chain.log 2>&1 < /dev/null & echo $! > chain.pid
# Each run trains until patience fires (no step budget), saves best + last, then the resume state is deleted.
# Refuses to start if a model folder already exists. The watchdog (watch_chain.py) applies the abort rules.
ROOT={root}
HERE={here}
PY=$ROOT/venv_SCAPES/bin/python
RUNS="{names}"
cd "$ROOT/SCAPES" || exit 1
for c in $RUNS; do
  if [ -e "$HERE/$c/model" ]; then echo "refusing to start: $HERE/$c/model exists"; exit 2; fi
done
sleep 3   # chain.pid is written by the launcher line right after this script starts
$PY -u scripts/watch_chain.py "$HERE" $RUNS --step-rule --max-hours 4 > "$HERE/watch.log" 2>&1 &
for c in $RUNS; do
  run=$HERE/$c
  echo "$(date +%F_%T) start $c"
  $PY -u -c "
import os
open('$run/pid','w').write(str(os.getpid()))
from SCAPES.pipeline import run_training
run_training('$ROOT/datasets/microtex_v2', '$run/model', config_dir='$run/config')
" > "$run/train.log" 2>&1
  code=$?
  echo $code > "$run/exit_code"
  echo "$(date +%F_%T) end $c exit $code"
  [ "$code" = 0 ] && rm -f "$run/model/checkpoints/last_trainer_state.pt"
  du -sh "$run/model"
done
echo "$(date +%F_%T) chain done"
"""


def write_chain(args):
    out = Path(args.out).resolve()
    for name in args.chain_names:
        if not (out / name / "config").is_dir():
            raise SystemExit(f"{out / name / 'config'} does not exist: write the configs first")
    path = out / args.chain
    path.write_text(LAUNCHER.format(names=" ".join(args.chain_names), here=out, file=args.chain, root=REPO.parent))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="../datasets/microtex_v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--chain", help="write this launcher into --out instead of the configs")
    ap.add_argument("--chain-names", nargs="+", default=[])
    args = ap.parse_args()
    if args.chain:
        if not args.chain_names:
            ap.error("--chain needs --chain-names")
        write_chain(args)
    else:
        write_configs(args)


if __name__ == "__main__":
    main()
