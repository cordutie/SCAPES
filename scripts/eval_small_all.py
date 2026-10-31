"""One command for the evaluation of the small-transformer runs (B14), steps a to e in order. NO reserve use.

  a  teacher-forced eval of runs/s5_small/<codec>/model -> runs/eval_tf_small (MSS, KAD, baselines, band-limited MSS, per-file
     CLAP, per-file tables, past-usage gain)
  b  s5_eval.py on the val region -> runs/s5_val_small (production checks, D-a..D-e)
  c  semantic resynthesis -> runs/semantic_resynth_small
  d  val per-file split (clean vs repeat files), steps and wall time -> runs/s5_small/val_split.json
  e  compare_small_large.py -> runs/small_vs_large.{md,json}
The large results (runs/s5, eval_tf, s5_val, semantic_resynth, ...) are only read.

    cd SCAPES && ../venv_SCAPES/bin/python scripts/eval_small_all.py            # runs everything
    ../venv_SCAPES/bin/python scripts/eval_small_all.py --dry                    # prints the commands only
    ../venv_SCAPES/bin/python scripts/eval_small_all.py --only d e               # a subset
Refuses to start while the chain has not finished (no 'chain done' in runs/s5_small/chain.log) unless --force; refuses if an
output folder exists (so nothing is overwritten) unless --overwrite.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO.parent / "runs"
PY = sys.executable
CODECS = {"encodec24k": "microtex_encodec24k_ar", "spectrostream": "microtex_spectrostream_ar",
          "encodec48k": "microtex_encodec48k_atoms"}


def commands(a):
    small, ds = Path(a.small_runs), Path(a.dataset)
    tf_out, val_out, sem_out = Path(a.eval_tf_out), Path(a.val_out), Path(a.sem_out)
    s = "scripts/"
    c = {"a": [], "b": [], "c": [], "d": [], "e": []}
    for codec, name in CODECS.items():
        c["a"].append([PY, s + "teacher_forced_eval.py", "--model-dir", str(small / codec / "model"), "--dataset", str(ds),
                       "--name", name, "--out", str(tf_out), "--codec-roundtrip", "on",
                       "--note", f"small transformer (runs/s5_small/{codec}), best checkpoint, train-region excerpts"])
    c["a"].append([PY, s + "eval_extra.py", "--out", str(tf_out)])
    c["a"].append([PY, s + "teacher_forced_eval.py", "--out", str(tf_out), "--summarize-only"])
    c["a"].append([PY, s + "per_file_tables.py", "--out", str(tf_out)])
    for codec in CODECS:
        c["a"].append([PY, s + "past_usage_eval.py", "--model-dir", str(small / codec / "model"), "--config-dir",
                       str(small / codec / "config"), "--dataset", str(ds), "--scratch", str(tf_out / f"_scratch_{codec}"),
                       "--out", str(tf_out / f"past_usage_{codec}.json")])
    c["b"].append([PY, s + "s5_eval.py", "--runs", str(small), "--dataset", str(ds), "--seeds", "0", "1", "2", "--seconds", "10",
                   "--prompt-region", "val", "--out", str(val_out)])
    c["c"].append([PY, s + "semantic_resynthesis.py", "--runs", str(small), "--dataset", str(ds), "--eval-tf", str(tf_out),
                   "--out", str(sem_out)])
    c["d"].append([PY, s + "val_split_report.py", "--runs", str(small), "--out", str(small / "val_split.json")])
    c["e"].append([PY, s + "compare_small_large.py", "--out", str(a.compare_out), "--small-tf", str(tf_out), "--small-val", str(val_out),
                   "--small-sem", str(sem_out), "--small-runs", str(small)])
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small-runs", default=str(RUNS / "s5_small"))
    ap.add_argument("--dataset", default=str(REPO.parent / "datasets" / "microtex"))
    ap.add_argument("--eval-tf-out", default=str(RUNS / "eval_tf_small"))
    ap.add_argument("--val-out", default=str(RUNS / "s5_val_small"))
    ap.add_argument("--sem-out", default=str(RUNS / "semantic_resynth_small"))
    ap.add_argument("--compare-out", default=str(RUNS / "small_vs_large"))
    ap.add_argument("--only", nargs="+", choices=list("abcde"), default=list("abcde"))
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--force", action="store_true", help="start although the chain has not finished")
    ap.add_argument("--overwrite", action="store_true", help="allow existing output folders")
    a = ap.parse_args()
    cmds = commands(a)
    if not a.dry:
        log = Path(a.small_runs) / "chain.log"
        if not a.force and not (log.exists() and "chain done" in log.read_text()):
            sys.exit("the chain has not finished (no 'chain done' in chain.log); use --force to override")
        outs = {"a": a.eval_tf_out, "b": a.val_out, "c": a.sem_out}
        for k in a.only:
            if k in outs and Path(outs[k]).exists() and not a.overwrite:
                sys.exit(f"{outs[k]} exists; use --overwrite")
    for k in a.only:
        for cmd in cmds[k]:
            print(f"[{k}] " + " ".join(cmd), flush=True)
            if a.dry:
                continue
            code = subprocess.run(cmd, cwd=REPO).returncode
            if code != 0:
                print(f"[{k}] exit {code}: continuing with the next command", flush=True)
        for codec in CODECS:                      # scratch of the past-usage runs
            shutil.rmtree(Path(a.eval_tf_out) / f"_scratch_{codec}", ignore_errors=True)


if __name__ == "__main__":
    main()
