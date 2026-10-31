"""Watchdog for a sequential training chain (senior abort rules: non-finite, traceback/OOM, ratio rule).
Usage: watch_chain.py <runs dir> <name> [<name> ...] [--step-rule] [--max-hours H]; the runs dir holds chain.pid and
<name>/{pid,train.log,model}; writes <name>/watch_summary.json.
--step-rule: replaces the epoch-ratio rule; once history_steps.json has entries with step >= 1000, kill if
  mean(loss, steps 900-1000) / mean(loss, steps 50-150) > 0.9.
--max-hours H: kill a run whose wall time exceeds H hours. A kill writes "KILLED: <why>" into watch_summary.json."""
import argparse, json, math, os, signal, time, pathlib, re
import sys
ap = argparse.ArgumentParser()
ap.add_argument("runs")
ap.add_argument("names", nargs="+")
ap.add_argument("--step-rule", action="store_true")
ap.add_argument("--max-hours", type=float, default=None)
ARGS = ap.parse_args()
P = pathlib.Path(ARGS.runs).resolve()
CODECS = ARGS.names
CHAIN = int(open(P / "chain.pid").read())

def alive(pid):
    try:
        os.kill(pid, 0)
        return "Z" not in open(f"/proc/{pid}/stat").read().split(") ")[-1][:2]
    except OSError:
        return False

def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None

def entries(h):
    return h if isinstance(h, list) else (h or {}).get("entries", (h or {}).get("history", []))

# mean loss over steps 900-1000 divided by mean loss over steps 50-150; None until an entry with step >= 1000 exists
def step_ratio(es):
    es = [e for e in es if isinstance(e, dict) and "step" in e and "loss" in e]
    if not any(e["step"] >= 1000 for e in es):
        return None
    late = [e["loss"] for e in es if 900 <= e["step"] <= 1000]
    early = [e["loss"] for e in es if 50 <= e["step"] <= 150]
    if not late or not early:
        return None
    return (sum(late) / len(late)) / (sum(early) / len(early))

def watch(c):
    r = P / c
    while not (r / "pid").exists():
        if not alive(CHAIN): return
        time.sleep(20)
    pid = int(open(r / "pid").read()); notes = []; done_ratio = False
    t_start = time.time()
    def kill(why):
        notes.append("KILLED: " + why)
        try: os.kill(pid, signal.SIGTERM)
        except OSError: pass
    while alive(pid):
        time.sleep(45)
        log = ""
        try: log = open(r / "train.log", errors="ignore").read()[-200000:]
        except Exception: pass
        if "Traceback" in log or "out of memory" in log.lower() or "non-finite loss or grad norm" in log:
            notes.append("log flag: " + ("Traceback" if "Traceback" in log else "OOM/non-finite")); 
            if "Traceback" in log or "out of memory" in log.lower(): kill("traceback/OOM in train.log")
        for e in entries(load(r / "model/logs/history_steps.json"))[-5:]:
            if isinstance(e, dict):
                for k in ("loss", "total", "grad_norm"):
                    v = e.get(k)
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): kill(f"non-finite {k}")
        if re.search(r"Val \|[^\n]*nan", log, re.I): kill("nan on Val line")
        lh = load(r / "model/loss/loss_history.json") or {}
        tr = (lh.get("train") or {}).get("total") or []
        if ARGS.max_hours is not None and (time.time() - t_start) / 3600 > ARGS.max_hours:
            kill(f"wall time above {ARGS.max_hours} h")
        if ARGS.step_rule and not done_ratio:
            ratio = step_ratio(entries(load(r / "model/logs/history_steps.json")))
            if ratio is not None:
                done_ratio = True
                notes.append(f"step ratio (loss 900-1000 / loss 50-150) {ratio:.3f} (kill > 0.9)")
                if ratio > 0.9: kill(f"ratio {ratio:.3f} > 0.9")
        if not ARGS.step_rule and c != "codicodec" and not done_ratio and len(tr) >= 28:
            ratio = tr[27] / tr[4]; done_ratio = True
            notes.append(f"epoch28/epoch5 train ratio {ratio:.3f} (kill > 0.9)")
            if ratio > 0.9: kill(f"ratio {ratio:.3f} > 0.9")
        if any(n.startswith("KILLED") for n in notes): break
    lh = load(r / "model/loss/loss_history.json") or {}
    tr = (lh.get("train") or {}).get("total") or []; va = (lh.get("val") or {}).get("total") or []
    s = {"codec": c, "notes": notes, "train_points": len(tr), "val_points": len(va)}
    if va:
        b = min(range(len(va)), key=lambda i: va[i]); s.update(best_val_index=b + 1, best_val=va[b], last_val=va[-1], since_best=len(va) - 1 - b)
    if tr: s.update(train_first=tr[0], train_last=tr[-1], first_train_below_1=next((i + 1 for i, v in enumerate(tr) if v < 1), None))
    s["exit_code"] = (r / "exit_code").read_text().strip() if (r / "exit_code").exists() else None
    json.dump(s, open(r / "watch_summary.json", "w"), indent=1); print(json.dumps(s), flush=True)
    return notes

for c in CODECS:
    n = watch(c)
    if n and any(x.startswith("KILLED") for x in n): print("stopped on kill; chain continues with next codec unless killed", flush=True)
