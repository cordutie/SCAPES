"""Side-by-side of the large (S5) and small-transformer runs for every metric. Read-only; writes <out>.md and <out>.json.

    ../venv_SCAPES/bin/python scripts/compare_small_large.py --out ../runs/small_vs_large
All paths have defaults (large: runs/eval_tf, runs/s5_val, runs/semantic_resynth, runs/diag_s5, runs/s5; small: runs/eval_tf_small,
runs/s5_val_small, runs/semantic_resynth_small, runs/s5_small); every one can be overridden (a self-comparison is a valid test).
Missing inputs give n/a, not an error.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import val_split_report as vsr

CODECS = ["encodec24k", "spectrostream", "encodec48k", "codicodec"]
TF_NAME = {"encodec24k": "microtex_encodec24k_ar", "spectrostream": "microtex_spectrostream_ar",
           "encodec48k": "microtex_encodec48k_atoms", "codicodec": "microtex_codicodec_atoms"}

CAVEATS = [
    "lr 1e-4 and batch 128 were tuned for neither size: the comparison is of two untuned configurations, not of the best small against the best large.",
    "The best checkpoint is selected by the total validation loss, which is driven by the 12 files that repeat the train audio; the clean-only (forest, park, rain, tokio) and repeat numbers are reported next to it (val_split section).",
    "The reserve is spent: this is a paired comparison on the val region and on train-region excerpts (teacher forcing, semantic resynthesis), not an acceptance test.",
    "Random numbers and data order differ with model size (initialisation draws and the batch order change), and each size was trained once (seed 0): differences smaller than the seed-to-seed spread, which was not measured, are not evidence.",
]


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def get(d, *keys):
    for k in keys:
        if d is None:
            return None
        try:
            d = d[k]
        except (KeyError, IndexError, TypeError):
            return None
    return d


def fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def collect(tf_dir, val_dir, sem_dir, usage_dir, runs_dir, usage_pattern, usage_fallback_dir=None):
    tfs = load(Path(tf_dir) / "summary.json") or {}
    val = load(Path(val_dir) / "s5_eval.json") or {}
    sem = load(Path(sem_dir) / "summary.json") or {}
    out = {}
    for c in CODECS:
        m = tfs.get(TF_NAME[c])
        v = val.get(c)
        s = sem.get(c)
        usage = load(Path(usage_dir) / usage_pattern.format(codec=c))
        if usage is None and usage_fallback_dir:        # e.g. the large encodec48k result lives in eval_tf
            usage = load(Path(usage_fallback_dir) / f"past_usage_{c}.json")
        hist = vsr.history(Path(runs_dir) / c / "model")
        r = {}
        for row in ("generated", "codec_roundtrip"):
            r[f"mss_{row}_full_L1"] = get(m, "mss", "rows", row, "mean_log_l1")
            r[f"mss_{row}_full_SC"] = get(m, "mss", "rows", row, "mean_spec_conv")
            r[f"mss_{row}_band_L1"] = get(m, "mss_band", "rows", row, "mean_log_l1")
            r[f"mss_{row}_band_SC"] = get(m, "mss_band", "rows", row, "mean_spec_conv")
        r["kad_clip_generated"] = get(m, "kad", "clip_level", "sets", "generated", "kad")
        r["kad_clip_generated_ci95"] = get(m, "kad", "clip_level", "sets", "generated", "ci95")
        r["kad_clip_roundtrip"] = get(m, "kad", "clip_level", "sets", "codec_roundtrip", "kad")
        r["past_usage_gain_percent"] = get(usage, "gain_percent")
        r["past_usage_L_real"] = get(usage, "L_real")
        r["past_usage_L_null"] = get(usage, "L_null")
        r["past_usage_gain_percent_clean_files"] = get(usage, "gain_percent_clean_files", "mean")
        r["past_usage_gain_percent_repeat_files"] = get(usage, "gain_percent_repeat_files", "mean")
        p = get(v, "production")
        r["production_pass"] = f"{p['n_pass']}/{p['n_clips']}" if p else None
        for k, key in (("D_b_generated", "generated"), ("D_b_other_file", "other_file_prompt"), ("D_b_real_floor", "real_floor")):
            r[k] = get(v, "diagnostics", "D-b", key, "mean")
        r["D_b_generated_ci95"] = get(v, "diagnostics", "D-b", "generated", "ci95")
        r["D_c_seam_generated"] = get(v, "diagnostics", "D-c", "generated", "mean")
        r["D_c_seam_real"] = get(v, "diagnostics", "D-c", "real", "mean")
        r["D_d_fraction_below_threshold"] = get(v, "diagnostics", "D-d", "fraction_below_threshold", "mean")
        r["D_a_roundtrip_band_L1"] = get(v, "diagnostics", "D-a", "roundtrip_band_log_l1", "mean")
        r["params_total_M"] = (get(v, "diagnostics", "D-e", "parameters") or 0) / 1e6 or None
        r["RTF_batch1_generation_plus_decode"] = get(v, "diagnostics", "D-e", "rtf_median")
        r["peak_gpu_generation_GB"] = (get(v, "diagnostics", "D-e", "peak_gpu_bytes_generation") or 0) / 1e9 or None
        r["semantic_band_L1"] = get(s, "mss", "semantic_resynthesis", "band_log_l1")
        r["semantic_band_SC"] = get(s, "mss", "semantic_resynthesis", "band_spec_conv")
        r["semantic_kad_clip"] = get(s, "kad", "clip_level", "sets", "semantic_resynthesis", "kad")
        if hist:
            r["train_s_per_step"] = hist[-1]["time"] / hist[-1]["step"]
            r["train_steps_logged"] = hist[-1]["step"]
            r["train_wall_minutes"] = hist[-1]["time"] / 60
        vs = vsr.report(Path(runs_dir) / c) if (Path(runs_dir) / c / "model").exists() else {}
        r["best_step"] = get(vs, "best_by_total", "step")
        r["val_best_total"] = get(vs, "best_by_total", "val_total")
        r["val_clean_at_best"] = get(vs, "best_by_total", "mean_clean")
        r["val_repeat_at_best"] = get(vs, "best_by_total", "mean_repeat")
        r["val_clean_min"] = get(vs, "min_clean", "mean_clean")
        r["val_clean_min_step"] = get(vs, "min_clean", "step")
        out[c] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    R = HERE.parent.parent / "runs"
    ap.add_argument("--out", default=str(R / "small_vs_large"))
    ap.add_argument("--large-tf", default=str(R / "eval_tf"))
    ap.add_argument("--small-tf", default=str(R / "eval_tf_small"))
    ap.add_argument("--large-val", default=str(R / "s5_val"))
    ap.add_argument("--small-val", default=str(R / "s5_val_small"))
    ap.add_argument("--large-sem", default=str(R / "semantic_resynth"))
    ap.add_argument("--small-sem", default=str(R / "semantic_resynth_small"))
    ap.add_argument("--large-runs", default=str(R / "s5"))
    ap.add_argument("--small-runs", default=str(R / "s5_small"))
    ap.add_argument("--large-usage-dir", default=str(R / "diag_s5"))
    ap.add_argument("--small-usage-dir", default=None, help="default: the small eval_tf folder")
    a = ap.parse_args()
    large = collect(a.large_tf, a.large_val, a.large_sem, a.large_usage_dir, a.large_runs, "past_usage_{codec}_best.json", usage_fallback_dir=a.large_tf)
    small = collect(a.small_tf, a.small_val, a.small_sem, a.small_usage_dir or a.small_tf, a.small_runs, "past_usage_{codec}.json")
    Path(a.out + ".json").write_text(json.dumps({"large": large, "small": small, "caveats": CAVEATS}, indent=1))
    L = ["# Small (37 M) against large (233 M) transformer, same data and recipe", "",
         "Caveats (from the senior):", ""] + [f"{i + 1}. {c}" for i, c in enumerate(CAVEATS)]
    L += ["", "Lower is better for MSS, KAD, D-a, D-d, RTF and times; higher for D-b generated, past-usage gain and production pass. "
          "Teacher-forced and semantic numbers are on train-region excerpts; production, D-a..D-e on the val region (prompts from val).", ""]
    keys = list(large[CODECS[0]])
    for c in CODECS:
        L += [f"## {c}", "", "| metric | large | small | small - large |", "|---|---|---|---|"]
        for k in keys:
            lv, sv = large[c][k], small[c][k]
            d = (sv - lv) if isinstance(lv, (int, float)) and isinstance(sv, (int, float)) else None
            L.append(f"| {k} | {fmt(lv)} | {fmt(sv)} | {fmt(d)} |")
        L.append("")
    Path(a.out + ".md").write_text("\n".join(L))
    print("wrote", a.out + ".md")


if __name__ == "__main__":
    main()
