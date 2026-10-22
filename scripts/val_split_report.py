"""Per-file validation curves of trained runs split into the clean files (no audio repeat of the train region) and the
repeat files, plus steps-to-best and wall time. Read-only.

    ../venv_SCAPES/bin/python scripts/val_split_report.py --runs ../runs/s5_small --out ../runs/s5_small/val_split.json
Per run <runs>/<codec>/model it reads logs/val_per_file.json, logs/history_steps.json and loss/loss_history.json.
Clean files: forest, park, rain, tokio (audio exact-repeat fraction 0 in runs/s5_val/duplication_audio.json); the other 12 repeat.
Selection of the best checkpoint is by the total validation loss (all files); the clean-only minimum is reported next to it.
"""
import argparse
import json
from pathlib import Path

import numpy as np

CLEAN = ["forest.wav", "park.wav", "rain.wav", "tokio.wav"]
CODECS = ["encodec24k", "spectrostream", "encodec48k", "codicodec"]


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


# Log folders of a run in training order: logs_part1, logs_part2, ... (earlier parts of a resumed run), then logs.
# The step counter continues across parts, the wall clock ("time") restarts at each part.
def log_dirs(model):
    model = Path(model)
    parts = sorted(model.glob("logs_part*"), key=lambda p: int(p.name.replace("logs_part", "")))
    return parts + [model / "logs"]


def history(model):
    out, offset = [], 0.0
    for d in log_dirs(model):
        h = load(d / "history_steps.json")
        h = h if isinstance(h, list) else (h or {}).get("entries", [])
        out += [dict(e, time=e["time"] + offset) for e in h]
        if h:
            offset += h[-1]["time"]
    return out


# per-file validation entries of all parts, in training order
def val_per_file(model):
    out = []
    for d in log_dirs(model):
        out += load(d / "val_per_file.json") or []
    return out


# total validation loss per validation, all parts: loss/loss_history.json of a resumed run already holds the earlier parts
# when its length matches the number of validations; otherwise the parts are concatenated
def val_totals(model, n):
    model = Path(model)
    lh = load(model / "loss" / "loss_history.json") or {}
    v = (lh.get("val") or {}).get("total") or []
    if len(v) == n:
        return v
    out = []
    for d in sorted(model.glob("loss_part*"), key=lambda p: int(p.name.replace("loss_part", ""))) + [model / "loss"]:
        out += ((load(d / "loss_history.json") or {}).get("val") or {}).get("total") or []
    return out


def report(run: Path) -> dict:
    model = run / "model"
    vpf = val_per_file(model)
    hist = history(model)
    out = {"codec": run.name}
    if hist:
        out["train_steps_logged"] = hist[-1]["step"]
        out["wall_seconds_at_last_log"] = hist[-1]["time"]
        out["s_per_step_mean"] = hist[-1]["time"] / hist[-1]["step"]
        out["n_log_parts"] = len(log_dirs(model)) if any((d / "history_steps.json").exists() for d in log_dirs(model)[:-1]) else 1
    val_total = val_totals(model, len(vpf))
    if not vpf:
        return out
    steps = [e["step"] for e in vpf]
    files = list(vpf[0]["per_file"])
    clean = [f for f in files if f in CLEAN]
    repeat = [f for f in files if f not in CLEAN]
    curve = {"step": steps,
             "all": [float(np.mean([e["per_file"][f] for f in files])) for e in vpf],
             "clean": [float(np.mean([e["per_file"][f] for f in clean])) for e in vpf],
             "repeat": [float(np.mean([e["per_file"][f] for f in repeat])) for e in vpf]}
    b_total = int(np.argmin(val_total)) if len(val_total) == len(vpf) else int(np.argmin(curve["all"]))
    b_clean = int(np.argmin(curve["clean"]))
    out.update({
        "n_validations": len(vpf), "clean_files": clean, "repeat_files": repeat,
        "best_by_total": {"index": b_total + 1, "step": steps[b_total], "val_total": val_total[b_total] if val_total else None,
                          "mean_all_files": curve["all"][b_total], "mean_clean": curve["clean"][b_total],
                          "mean_repeat": curve["repeat"][b_total]},
        "min_clean": {"index": b_clean + 1, "step": steps[b_clean], "mean_clean": curve["clean"][b_clean],
                      "mean_repeat_at_that_step": curve["repeat"][b_clean]},
        "last": {"step": steps[-1], "mean_all_files": curve["all"][-1], "mean_clean": curve["clean"][-1], "mean_repeat": curve["repeat"][-1]},
        "wall_seconds_to_best_step": (hist[min(range(len(hist)), key=lambda i: abs(hist[i]["step"] - steps[b_total]))]["time"] if hist else None),
        "curves": curve,
        "train_curve": {"step": [h["step"] for h in hist], "loss": [h["loss"] for h in hist]} if hist else None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for c in CODECS:
        if (Path(a.runs) / c / "model").exists():
            res[c] = report(Path(a.runs) / c)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for c, r in res.items():
        if "best_by_total" in r:
            print(c, "best step", r["best_by_total"]["step"], "min clean step", r["min_clean"]["step"],
                  "clean at best %.3f / at its min %.3f" % (r["best_by_total"]["mean_clean"], r["min_clean"]["mean_clean"]))


if __name__ == "__main__":
    main()
