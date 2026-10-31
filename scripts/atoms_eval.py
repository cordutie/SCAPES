"""E5 driver: teacher-forced evaluation, speed and training record of the trained atom variants.

Usage (from SCAPES/):
    ../venv_SCAPES/bin/python scripts/atoms_eval.py --runs ../runs/v2_atoms --dataset ../datasets/microtex_v2 --out ../runs/v2_atoms_eval \
        [--variants ...] --seeds 0 1 --nfe 16 --cfg 1.0 [--start-seconds 0 --floor-start-seconds 10] [--rtf on|off]
For each variant with model/checkpoints/best_flow_model.pt and each seed, a subprocess runs teacher_forced_eval.py (wavs only
for the first seed). Then RTF (rtf_bench.bench_model) and the training record. Writes <out>/{summary.json,summary.md}, the
figures mss_vs_size / kad_vs_size / rtf_vs_size.png, <out>/tf/<variant>__s<seed>/, <out>/rtf/<variant>.json and
<out>/listen/<file>/{original,<variant>,<variant>__roundtrip}.wav for applause, keyboard, orchestra and wind (first seed).
"""
import argparse
import json
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

VARIANTS = ["m2l-cons-c2", "m2l-cons-c4", "m2l-cons-c6", "m2l-cons-c8"]
LISTEN = ("applause", "keyboard", "orchestra", "wind")
REFERENCE_SOURCE = REPO.parent / "old_stuff" / "scapes_v2" / "final" / "data" / "metrics.md"
REFERENCE_ROWS = (("codicodec", "small (mp)"), ("encodec48k", "small (mp)"))


def minmax(v):
    return [float(min(v)), float(max(v))]


def mean_mm(v):
    return {"mean": float(np.mean(v)), "range": minmax(v)}


def run_tf(args, variant, seed, out_tf):
    cmd = [sys.executable, str(HERE / "teacher_forced_eval.py"), "--model-dir", str(Path(args.runs) / variant / "model"),
           "--dataset", args.dataset, "--name", f"{variant}__s{seed}", "--out", str(out_tf), "--seed", str(seed),
           "--nfe", str(args.nfe), "--cfg", str(args.cfg), "--codec-roundtrip", "on",
           "--start-seconds", str(args.start_seconds)]
    if args.floor_start_seconds is not None:
        cmd += ["--floor-start-seconds", str(args.floor_start_seconds)]
    if seed != args.seeds[0]:
        cmd.append("--no-wav")
    subprocess.run(cmd, cwd=REPO, check=True, env={**__import__("os").environ, "PYTHONPATH": str(REPO)})


def copy_listen(out, variant, seed):
    tf = out / "tf"
    for wav in sorted((tf / "original_10s").glob("*.wav")):
        if not any(w in wav.stem for w in LISTEN):
            continue
        d = out / "listen" / wav.stem
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wav, d / "original.wav")
        for sub, suffix in (("generated", ""), ("codec_roundtrip", "__roundtrip")):
            src = tf / f"{variant}__s{seed}" / sub / wav.name
            if src.exists():
                shutil.copy2(src, d / f"{variant}{suffix}.wav")


# Training record from model/logs: steps, best step (lowest mean validation loss), wall minutes, peak GB, seconds per step
def training_record(run):
    logs = Path(run) / "model" / "logs"
    rec = {}
    if (logs / "history_steps.json").exists():
        h = json.loads((logs / "history_steps.json").read_text())
        rec["steps"] = h[-1]["step"]
        rec["wall_minutes"] = h[-1]["time"] / 60
        rec["s_per_step"] = h[-1]["time"] / h[-1]["step"]
    if (logs / "val_per_file.json").exists():
        v = json.loads((logs / "val_per_file.json").read_text())
        means = [(e["step"], float(np.mean(list(e["per_file"].values())))) for e in v]
        rec["best_step"] = min(means, key=lambda t: t[1])[0]
        rec["best_val"] = min(m for _, m in means)
    if (logs / "peak_memory.json").exists():
        rec["peak_gb"] = json.loads((logs / "peak_memory.json").read_text())["max_memory_allocated_bytes"] / 1e9
    return rec


def collect(args, variant, out):
    tf_dir = out / "tf"
    ms = [json.loads((tf_dir / f"{variant}__s{s}" / "metrics.json").read_text()) for s in args.seeds]
    g = ms[0]["geometry"]
    gen = [m["mss"]["rows"]["generated"] for m in ms]
    rt = ms[0]["mss"]["rows"]["codec_roundtrip"]
    noise = ms[0]["mss"]["rows"]["white_noise"]
    kc = [m["kad"]["clip_level"]["sets"] for m in ms]
    ks = [m["kad"]["segment_level"]["sets"] for m in ms]
    row = {
        "variant": variant, "type": g["atom_type"], "F": g["unit_frames"], "C_s": g["atom_samples"] / g["codec_sample_rate"],
        "H_s": g["hop_samples"] / g["codec_sample_rate"], "rate_hz": g["codec_sample_rate"] / g["hop_samples"],
        "overlap_s": (g["atom_samples"] - g["hop_samples"]) / g["codec_sample_rate"],
        "X_s": g["crossfade_samples"] / g["codec_sample_rate"], "M": g["memory_units"], "T_M_s": g["memory_seconds"],
        "window": g["window"],
        "mss_l1": mean_mm([x["mean_log_l1"] for x in gen]), "mss_sc": mean_mm([x["mean_spec_conv"] for x in gen]),
        "kad_clip": mean_mm([k["generated"]["kad"] for k in kc]), "kad_segment": mean_mm([k["generated"]["kad"] for k in ks]),
        "roundtrip_mss_l1": rt["mean_log_l1"], "roundtrip_kad": kc[0]["codec_roundtrip"]["kad"],
        "noise_mss_l1": noise["mean_log_l1"], "floor_kad": kc[0]["floor"]["kad"],
        "duration_check_ok": bool(all(d["ok"] for m in ms for d in m["duration_check"].values())),
        "seeds": args.seeds, "nfe": args.nfe, "cfg": args.cfg,
    }
    row["very_bad"] = bool(row["mss_l1"]["mean"] > 0.5 * (row["roundtrip_mss_l1"] + row["noise_mss_l1"]))     # MSS only (D18)
    row.update(training_record(Path(args.runs) / variant))
    return row


def reference_rows():
    if not REFERENCE_SOURCE.exists():
        return []
    out = []
    for line in REFERENCE_SOURCE.read_text().splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) > 3 and (cells[0], cells[1]) in REFERENCE_ROWS and cells[2] == "16":
            out.append({"source": str(REFERENCE_SOURCE), "codec": cells[0], "size": cells[1], "nfe": 16, "cells": cells[3:]})
    return out


def run_rtf(args, variant, out):
    import rtf_bench
    ns = Namespace(device="cuda", calls=20, warmup=3, configs=["nfe16_cfgoff", "nfe16_cfg3"], no_bf16=True, batch=1)
    res = rtf_bench.bench_model(ns, variant, str(Path(args.runs) / variant / "model"))
    (out / "rtf").mkdir(parents=True, exist_ok=True)
    (out / "rtf" / f"{variant}.json").write_text(json.dumps(res, indent=1))
    return {k: res["configs"][k]["rtf"]["mean"] for k in res["configs"]}


def write_md(path, rows, refs, args):
    region = "held-out reserve region" if args.start_seconds >= 108 else "train region: a fit measure"
    L = [f"# Atoms evaluation (teacher forced, {region})", "",
         f"Seeds {args.seeds}, nfe {args.nfe}, cfg {args.cfg}, start {args.start_seconds} s. MSS: natural-log magnitude L1 and "
         "spectral convergence (mean of three big windows, mono 48 kHz, region [0.4 s, end)); mean over seeds [min, max]. "
         "KAD x100, clip and segment level. 'very bad' flag: MSS L1 above the midpoint of the round trip and white noise (MSS only).", "",
         "| variant | type | F | C (s) | H (s) | rate (Hz) | overlap (s) | X (s) | M | T_M (s) | window | steps | best step | wall (min) | peak GB | s/step "
         "| MSS L1 | MSS SC | KAD clip | KAD segment | round trip MSS L1 | round trip KAD | noise MSS L1 | floor KAD | RTF_new cfg off | RTF_new cfg 3 | duration ok | very bad |",
         "|" + "---|" * 28]

    def f(x, d=3):
        return "" if x is None else f"{x:.{d}f}"

    def mm(v, d=3):
        return f"{v['mean']:.{d}f} [{v['range'][0]:.{d}f}, {v['range'][1]:.{d}f}]"
    for r in rows:
        rtf = r.get("rtf", {})
        L.append(f"| {r['variant']} | {r['type']} | {r['F']} | {r['C_s']:.3f} | {r['H_s']:.3f} | {r['rate_hz']:.2f} | {r['overlap_s']:.3f} | {r['X_s']:.4f} "
                 f"| {r['M']} | {r['T_M_s']:.3f} | {r['window']} | {r.get('steps', '')} | {r.get('best_step', '')} | {f(r.get('wall_minutes'), 1)} "
                 f"| {f(r.get('peak_gb'), 2)} | {f(r.get('s_per_step'), 3)} | {mm(r['mss_l1'])} | {mm(r['mss_sc'])} | {mm(r['kad_clip'], 2)} | {mm(r['kad_segment'], 2)} "
                 f"| {r['roundtrip_mss_l1']:.3f} | {r['roundtrip_kad']:.2f} | {r['noise_mss_l1']:.3f} | {r['floor_kad']:.2f} "
                 f"| {f(rtf.get('nfe16_cfgoff'))} | {f(rtf.get('nfe16_cfg3'))} | {r['duration_check_ok']} | {r['very_bad']} |")
    if refs:
        L += ["", "Reference rows (legacy frame atoms, copied from " + refs[0]["source"] + "; columns: MSS L1, MSS SC, MSS band L1, MSS band SC, KAD clip, KAD floor, round trip):", ""]
        L += [f"- {r['codec']} {r['size']}, nfe 16, cfg off: {' | '.join(r['cells'])}" for r in refs]
    path.write_text("\n".join(L) + "\n")


def write_figures(out, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["F"] for r in rows]

    def fig(key, ylabel, name, ref=None):
        f, ax = plt.subplots(figsize=(5.5, 3.8))
        y = [r[key]["mean"] for r in rows]
        err = [[r[key]["mean"] - r[key]["range"][0] for r in rows], [r[key]["range"][1] - r[key]["mean"] for r in rows]]
        ax.errorbar(x, y, yerr=err, marker="o", color="#2a78d6", capsize=3, label="generated (2 seeds: min-max)")
        if ref:
            ax.plot(x, [r[ref] for r in rows], marker="s", color="#eb6834", markerfacecolor="none", label="codec round trip")
        ax.set_xlabel("latents per atom")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.legend(fontsize=8)
        f.tight_layout()
        f.savefig(out / name, dpi=130)
        plt.close(f)
    fig("mss_l1", "MSS L1 (log magnitude)", "mss_vs_size.png", "roundtrip_mss_l1")
    fig("kad_clip", "KAD clip (x100)", "kad_vs_size.png", "roundtrip_kad")
    if all("rtf" in r for r in rows):
        f, ax = plt.subplots(figsize=(5.5, 3.8))
        for label, colour in (("nfe16_cfgoff", "#2a78d6"), ("nfe16_cfg3", "#eb6834")):
            ax.plot(x, [r["rtf"][label] for r in rows], marker="o", color=colour, label=label)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="RTF = 1")
        ax.set_xlabel("latents per atom")
        ax.set_ylabel("RTF_new")
        ax.set_xticks(x)
        ax.legend(fontsize=8)
        f.tight_layout()
        f.savefig(out / "rtf_vs_size.png", dpi=130)
        plt.close(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--nfe", type=int, default=16)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--start-seconds", type=float, default=0.0)
    ap.add_argument("--floor-start-seconds", type=float, default=None)
    ap.add_argument("--rtf", choices=["on", "off"], default="on")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = args.variants or VARIANTS
    done = [v for v in names if (Path(args.runs) / v / "model" / "checkpoints" / "best_flow_model.pt").exists()]
    for v in set(names) - set(done):
        print(f"skipping {v}: no best_flow_model.pt", flush=True)

    rows = []
    for v in done:
        for s in args.seeds:
            run_tf(args, v, s, out / "tf")
        copy_listen(out, v, args.seeds[0])
        row = collect(args, v, out)
        if args.rtf == "on":
            row["rtf"] = run_rtf(args, v, out)
        rows.append(row)
    (out / "summary.json").write_text(json.dumps({"rows": rows, "references": reference_rows(), "args": vars(args)}, indent=1))
    write_md(out / "summary.md", rows, reference_rows(), args)
    if rows:
        write_figures(out, rows)
    print("wrote", out / "summary.md")


if __name__ == "__main__":
    main()
