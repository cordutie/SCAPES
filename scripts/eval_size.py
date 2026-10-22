"""Evaluation of the trained runs of ONE model size (same protocol as eval_small_all.py, without touching other outputs). NO reserve use.

  a  teacher-forced eval (MSS, KAD, baselines, band MSS, per-file CLAP, per-file tables)   -> runs/eval_tf<suffix>
  b  s5_eval.py on the val region (production checks, D-a..D-e)                              -> runs/s5_val<suffix>
  c  semantic resynthesis                                                                    -> runs/semantic_resynth<suffix>
  d  val per-file split (clean vs repeat files), steps and wall time                         -> <runs>/val_split.json
  u  past usage (real vs null past, per-file split, CI) for all four codecs incl. encodec48k  -> runs/past_usage_all/<size>/<codec>.json

    ../venv_SCAPES/bin/python scripts/eval_size.py --size petite --runs ../runs/s5_petite --out-suffix _petite
    ../venv_SCAPES/bin/python scripts/eval_size.py --size large --runs ../runs/s5 --only u      # past usage only, for an old run
    ../venv_SCAPES/bin/python scripts/eval_size.py --size petite --runs ../runs/s5_petite --out-suffix _petite --dry
Refuses to start while the run folder has no chain_done_marker (written by launch_chain_petite_tiny.sh) unless --force, and refuses if an
output folder exists unless --overwrite. Steps a, b, c need an idle GPU for clean timings (b records RTF).
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
          "encodec48k": "microtex_encodec48k_atoms", "codicodec": "microtex_codicodec_atoms"}


def commands(a):
    runs, ds = Path(a.runs), Path(a.dataset)
    tf_out, val_out, sem_out = RUNS / f"eval_tf{a.out_suffix}", RUNS / f"s5_val{a.out_suffix}", RUNS / f"semantic_resynth{a.out_suffix}"
    use_out = RUNS / "past_usage_all" / a.size
    s = "scripts/"
    c = {k: [] for k in "abcdu"}
    for codec, name in CODECS.items():
        c["a"].append([PY, s + "teacher_forced_eval.py", "--model-dir", str(runs / codec / "model"), "--dataset", str(ds), "--name", name,
                       "--out", str(tf_out), "--codec-roundtrip", "on", "--note", f"size {a.size} ({runs.name}/{codec}), best checkpoint, train-region excerpts"])
    c["a"] += [[PY, s + "eval_extra.py", "--out", str(tf_out)], [PY, s + "teacher_forced_eval.py", "--out", str(tf_out), "--summarize-only"],
               [PY, s + "per_file_tables.py", "--out", str(tf_out)]]
    c["b"].append([PY, s + "s5_eval.py", "--runs", str(runs), "--dataset", str(ds), "--seeds", "0", "1", "2", "--seconds", "10",
                   "--prompt-region", "val", "--out", str(val_out)])
    c["c"].append([PY, s + "semantic_resynthesis.py", "--runs", str(runs), "--dataset", str(ds), "--eval-tf", str(tf_out), "--out", str(sem_out)])
    c["d"].append([PY, s + "val_split_report.py", "--runs", str(runs), "--out", str(runs / "val_split.json")])
    for codec in CODECS:
        c["u"].append([PY, s + "past_usage_eval.py", "--model-dir", str(runs / codec / "model"), "--config-dir", str(runs / codec / "config"),
                       "--dataset", str(ds), "--scratch", str(use_out / f"_scratch_{codec}"), "--out", str(use_out / f"{codec}.json")])
    return c, {"a": tf_out, "b": val_out, "c": sem_out, "u": use_out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, help="size name for labels and the past-usage folder (petite, tiny, small, large ...)")
    ap.add_argument("--runs", required=True, help="folder with <codec>/model and <codec>/config, e.g. ../runs/s5_petite")
    ap.add_argument("--out-suffix", default=None, help="suffix of the output folders (default _<size>)")
    ap.add_argument("--dataset", default=str(REPO.parent / "datasets" / "microtex"))
    ap.add_argument("--only", nargs="+", choices=list("abcdu"), default=list("abcdu"))
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    a.runs = str(Path(a.runs).resolve())
    a.out_suffix = f"_{a.size}" if a.out_suffix is None else a.out_suffix
    cmds, outs = commands(a)
    if not a.dry:
        if not a.force and not (Path(a.runs) / "chain_done_marker").exists():
            sys.exit(f"{a.runs}/chain_done_marker is missing (the chain has not finished); use --force to override")
        for k in a.only:
            if k in outs and outs[k].exists() and not a.overwrite and k != "u":
                sys.exit(f"{outs[k]} exists; use --overwrite")
    for k in a.only:
        for cmd in cmds[k]:
            print(f"[{k}] " + " ".join(cmd), flush=True)
            if a.dry:
                continue
            if k == "u":
                outs["u"].mkdir(parents=True, exist_ok=True)
            code = subprocess.run(cmd, cwd=REPO).returncode
            if code != 0:
                print(f"[{k}] exit {code}: continuing", flush=True)
    for codec in CODECS:
        shutil.rmtree(outs["u"] / f"_scratch_{codec}", ignore_errors=True)


if __name__ == "__main__":
    main()
