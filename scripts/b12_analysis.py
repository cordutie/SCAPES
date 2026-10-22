"""Read-only analysis of the B12 outputs (S1 reserve, S5 reserve, duplication, S3 final). Writes <out>.md and <out>.json.

    ../venv_SCAPES/bin/python scripts/b12_analysis.py --runs ../runs --out ../runs/b12_analysis
It changes nothing in the run folders (the listening pointers go to a separate file with --listening).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def load(p):
    return json.loads(Path(p).read_text())


def stat(values, n_boot=10000, seed=0):
    v = np.array([x for x in values if x is not None and math.isfinite(x)], float)
    if len(v) == 0:
        return None
    m = v[np.random.RandomState(seed).randint(0, len(v), (n_boot, len(v)))].mean(1)
    return {"mean": float(v.mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], "n": len(v)}


def f(s, d=3):
    return "n/a" if not s else f"{s['mean']:.{d}f} [{s['ci95'][0]:.{d}f}, {s['ci95'][1]:.{d}f}] (n={s['n']})"


def s1(runs, L, out):
    tau = load(runs.parent / "audio_codec_wrappers" / "s1_thresholds.json")["tau"]
    rows, fails, n_rt = [], [], 0
    for codec in ["encodec24k", "spectrostream", "encodec48k", "codicodec"]:
        for clip_dir in sorted((runs / "s1_reserve" / codec).iterdir()):
            r = load(clip_dir / "report.json")
            n_rt += 1
            st = r["checks"].get("streaming", {})
            st_ok = None if not st else all(v for v in st.values() if v is not None)
            for seed, sd in r["seeds"].items():
                c = r["checks"][seed]
                row = {"codec": codec, "clip": clip_dir.name, "seed": seed, "rho_nu": sd["rho_nu"], "tau": tau[codec],
                       "lag": sd["lag"], "M_hat": sd["M_hat"], "M_noise": sd["M_noise"], "nonstationary": sd["nonstationary"],
                       "c_hat": sd["c_hat"], "c_delayed": sd["c_delayed"], "F1": c["F1"], "F2": c["F2"], "F3": c["F3"],
                       "F3_strict": c["F3_strict"], "lag_ok": c["lag"], "length": c["length"], "streaming": st_ok}
                rows.append(row)
                for k in ("F1", "F2", "F3", "F3_strict", "lag_ok", "length"):
                    if row[k] is False:
                        fails.append((codec, clip_dir.name, seed, k))
                if st_ok is False:
                    fails.append((codec, clip_dir.name, seed, "streaming"))
    out["s1"] = {"round_trips": n_rt, "rows": rows, "failures": fails}
    L += ["## 1. S1 reserve (frozen thresholds, no retuning)", "",
          f"{n_rt} round trips (4 codecs x 5 clips; CoDiCodec has 3 seeds, so {len(rows)} codec-clip-seed rows). Failures by item: "
          f"{fails if fails else 'none'}.", "",
          "| codec | clip | seed | rho_nu | tau | F1 | F2 | F3 (tie 0.005) | F3 strict | lag | length | streaming |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['codec']} | {r['clip']} | {r['seed']} | {r['rho_nu']:.3f} | {r['tau']:.3f} | {r['F1']} | {r['F2']} | {r['F3']} | "
                 f"{r['F3_strict']} | {r['lag']} | {r['length']} | {r['streaming']} |")
    L.append("")


def s5(runs, L, out):
    e = load(runs / "s5_eval" / "s5_eval.json")
    dup = load(runs / "s5_eval" / "duplication_audio.json")["files"]
    dupv = load(runs / "s5_val" / "duplication_audio.json")["files"]
    names = list(e)
    out["s5"] = {}
    L += ["## 2. S5 reserve: production checks (4 x 48 clips)", "",
          "| codec | pass | finite | duration | rms | rms first 1 s | clip | prompt start (s, min-max) |", "|---|---|---|---|---|---|---|---|"]
    fails = []
    for n in names:
        p = e[n]["production"]
        ps = list(e[n]["meta"]["prompt_start_seconds"].values())
        L.append(f"| {n} | {p['n_pass']}/{p['n_clips']} | {p['finite']} | {p['duration_ok']} | {p['rms_ok']} | {p['rms_first_ok']} | "
                 f"{p['clip_ok']} | {min(ps):.2f}-{max(ps):.2f} |")
        for c in e[n]["clips"]:
            if not c["pass"]:
                bad = [k for k in ("finite", "duration_ok", "rms_ok", "rms_first_ok", "clip_ok") if not c[k]]
                fails.append((n, c["file"], c["seed"], bad, c["rms_db"], c["rms_first_db"], c["clipped_fraction"]))
    L += ["", "Failures:", "", "| codec | file | seed | failed item | rms dB | first-1s rms dB | clipped fraction |", "|---|---|---|---|---|---|---|"]
    for x in fails:
        L.append(f"| {x[0]} | {x[1]} | {x[2]} | {', '.join(x[3])} | {x[4]:.1f} | {x[5]:.1f} | {x[6]:.5f} |")
    out["s5"]["production_failures"] = fails
    L += ["", "### D-a..D-e (mean over 16 prompts [95% bootstrap over prompts])", "",
          "| codec | D-a rt band log-L1 | D-a rt band SC | D-a noise band log-L1 | D-b generated | D-b real floor | D-b other-file | D-c gen | D-c real | D-d frac (region thr) | params M | peak GPU GB | RTF median |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        d = e[n]["diagnostics"]
        c = d["D-c"]
        L.append(f"| {n} | {f(d['D-a']['roundtrip_band_log_l1'])} | {f(d['D-a']['roundtrip_band_spec_conv'])} | {f(d['D-a']['white_noise_band_log_l1'])} | "
                 f"{f(d['D-b']['generated'])} | {f(d['D-b']['real_floor'])} | {f(d['D-b']['other_file_prompt'])} | "
                 f"{f(c.get('generated')) if c['applicable'] else 'n/a'} | {f(c.get('real')) if c['applicable'] else 'n/a'} | "
                 f"{f(d['D-d']['fraction_below_threshold_region'])} | {d['D-e']['parameters'] / 1e6:.1f} | "
                 f"{d['D-e']['peak_gpu_bytes_generation'] / 1e9:.2f} | {d['D-e']['rtf_median']:.2f} |")
    L += ["", "D-e RTF: batch 1, sequential in one process, includes decode, after an untimed warm-up. It is NOT the ABBA protocol of S3.", ""]

    # duplication
    files = sorted(dup)
    clean = [x for x in files if dup[x]["fraction_exact_ncc_ge_0.99"] == 0.0]
    out["s5"]["clean_files"] = clean
    L += ["## 3. Duplication (declared before the run)", "",
          "Audio: fraction of 12 one-second reserve windows (resp. val windows) with NCC >= 0.99 against the train audio. Latent: fraction of "
          "reserve units (stored latents) with an exact train copy (distance < 1e-3). Production: clips passing out of 3 seeds.", "",
          "| file | audio exact (reserve) | audio near >=0.9 (reserve) | audio exact (val) | " + " | ".join(f"latent exact {n}" for n in names) + " | " +
          " | ".join(f"pass {n}" for n in names) + " |", "|" + "---|" * (4 + 2 * len(names))]
    for x in files:
        lat = [e[n]["diagnostics"]["D-d"]["region_exact_copy_fraction"].get(x) for n in names]
        pas = [sum(c["pass"] for c in e[n]["clips"] if c["file"] == x) for n in names]
        L.append(f"| {x} | {dup[x]['fraction_exact_ncc_ge_0.99']:.2f} | {dup[x]['fraction_near_ncc_ge_0.9']:.2f} | {dupv[x]['fraction_exact_ncc_ge_0.99']:.2f} | " +
                 " | ".join("n/a" if v is None else f"{v:.2f}" for v in lat) + " | " + " | ".join(f"{p}/3" for p in pas) + " |")
    n_exact_r = sum(dup[x]["fraction_exact_ncc_ge_0.99"] > 0 for x in files)
    n_exact_v = sum(dupv[x]["fraction_exact_ncc_ge_0.99"] > 0 for x in files)
    L += ["", f"Files with any exact-repeat window: reserve {n_exact_r}/16, val {n_exact_v}/16. Clean files (audio exact-repeat fraction = 0 in the reserve): {clean}.", ""]

    L += ["### D-b and D-d on all files and on the clean files (stratification declared before the run, not a criterion)", "",
          "| codec | D-b generated (all) | D-b generated (clean) | D-b real floor (clean) | D-b other-file (clean) | gen - other (clean) | D-d frac below region thr (all) | (clean) | generated exact train copies (all, mean per file) | (clean) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    out["s5"]["stratified"] = {}
    for n in names:
        d = e[n]["diagnostics"]
        b = d["D-b"]["files"]
        dd = d["D-d"]
        sel = lambda key: [b[x][key] for x in clean if x in b and b[x][key] is not None]
        gen_other = [b[x]["generated"] - b[x]["other_file_prompt"] for x in clean if x in b]
        below = dd["fraction_below_threshold_region_per_file"]
        ex = {x: v["fraction_exact_copy"] for x, v in dd["files"].items()}
        row = {"gen_clean": stat(sel("generated")), "floor_clean": stat(sel("real_floor")), "other_clean": stat(sel("other_file_prompt")),
               "gen_minus_other_clean": stat(gen_other), "dd_clean": stat([below[x] for x in clean if x in below]),
               "dd_all": dd["fraction_below_threshold_region"], "exact_gen_all": stat(list(ex.values())),
               "exact_gen_clean": stat([ex[x] for x in clean if x in ex])}
        out["s5"]["stratified"][n] = row
        L.append(f"| {n} | {f(d['D-b']['generated'])} | {f(row['gen_clean'])} | {f(row['floor_clean'])} | {f(row['other_clean'])} | {f(row['gen_minus_other_clean'])} | "
                 f"{f(row['dd_all'])} | {f(row['dd_clean'])} | {f(row['exact_gen_all'])} | {f(row['exact_gen_clean'])} |")
    L += ["", "D-d region thresholds (1st percentile of reserve-to-train nearest distance, exact copies left out) and plain value:", ""]
    for n in names:
        dd = e[n]["diagnostics"]["D-d"]
        L.append(f"- {n}: {dd['threshold_1st_percentile_region_to_train']:.3f} (plain {dd['threshold_region_raw']:.3f}); reserve units with exact train copy "
                 f"{dd['region_units_with_exact_train_copy']:.3f} in {dd['region_files_with_exact_copy']}")
    L.append("")


def s3(runs, L, out):
    r = load(runs / "s3_final" / "rtf.json")
    procs = r["processes"]
    sm = [v for p in procs for v in (p["gpu_before"]["sm_clock_mhz"], p["gpu_after"]["sm_clock_mhz"])]
    temp = [v for p in procs for v in (p["gpu_before"]["temp_c"], p["gpu_after"]["temp_c"])]
    rtfs = [x for p in procs for x in p["rtf"]]
    ab = {k: v for k, v in r["ratio_per_pair"].items()}
    out["s3"] = {k: r[k] for k in ("speedup", "speedup_ci95", "margin", "outcome", "ratio_per_pair", "pairs_above_1", "ci95_log")}
    out["s3"]["sm_clock_range"], out["s3"]["temp_range"], out["s3"]["rtf_range"] = [min(sm), max(sm)], [min(temp), max(temp)], [min(rtfs), max(rtfs)]
    L += ["## 4. S3 final (encodec24k, large, random init, 10 AB pairs in ABBA order, 5 A/A pairs per mode, seed base 1000)", "",
          f"- speedup = exp(mean log(median RTF atoms / median RTF AR)); above 1 means AR is cheaper: {r['speedup']:.4f}, 95% CI [{r['speedup_ci95'][0]:.4f}, {r['speedup_ci95'][1]:.4f}]",
          f"- A/A margin (log): {r['margin']:.4f} (= factor {math.exp(r['margin']):.3f}); CI of the log ratio {r['ci95_log']}",
          f"- outcome: **{r['outcome']}**; pairs with ratio above 1: {r['pairs_above_1']}/10",
          "- per-pair ratios: " + ", ".join(f"{k}: {v:.3f}" for k, v in ab.items()),
          f"- verdict against the plan: {'faster' if r['outcome'] == 'faster' else 'NOT faster: report plainly as **not cheaper at 10 Hz**'}.",
          f"- contamination conditions: SM clock {min(sm):.0f}-{max(sm):.0f} MHz over {len(procs)} processes (before and after; expected 2580-2715), GPU temperature {min(temp):.0f}-{max(temp):.0f} C, "
          f"per-rep RTF {min(rtfs):.3f}-{max(rtfs):.3f} (expected < 1.5), CPU governor {r['cpu_governor']}.",
          "- reading: the CI of the ratio lies below 1 (AR about 1-5% slower than atoms) but inside the A/A margin factor, so by the pre-declared rule it is 'indistinguishable'; AR is not cheaper.",
          "- idle check: runs/s3_gpu_before.txt and s3_gpu_after.txt list no compute apps (the pid header is the only line); GPU 0 % utilisation, 439/440 MiB used. At the end the SM clock reads 210 MHz (idle).",
          "- caveat (seed overlap with B6): the timed repetitions use noise seeds process_seed + 1000 (rep+1); B12 process seeds are 1000-1039 (rep seeds 2000-4039) and B6 process seeds 0-~30 (rep seeds 1000-~3030), so the noise seeds 2000+slot and 3000+slot overlap between B6 and B12. The model weights differ (model seed = seed_base + slot), so inputs are not identical, and RTF does not depend on the noise values; I did not test that.", ""]


def listening(runs, path):
    e = load(runs / "s5_eval" / "s5_eval.json")
    dup = load(runs / "s5_eval" / "duplication_audio.json")["files"]
    clean = {x for x in dup if dup[x]["fraction_exact_ncc_ge_0.99"] == 0.0}
    L = ["", "## Pointers to the cleanest and the worst examples (added after the run, read from s5_eval.json and duplication_audio.json)", "",
         "Ranking is by D-b CLAP cosine of the generated clips with their prompt (mean over seeds), within the 4 clean files (no exact audio repeat of the train region) and within all 16. "
         "The round-trip worst is by band log-L1 of the codec round trip (D-a), which is the limit of the codec, not of the model.", ""]
    for n, v in e.items():
        b = v["diagnostics"]["D-b"]["files"]
        a = v["diagnostics"]["D-a"]["files"]
        allr = sorted(b, key=lambda x: -b[x]["generated"])
        cl = [x for x in allr if x in clean]
        rtw = sorted(a, key=lambda x: -a[x]["roundtrip"]["band_log_l1"])[:3]
        stem = lambda x: x[:-4]
        L += [f"- {n}: best CLAP among clean files: {stem(cl[0])} ({b[cl[0]]['generated']:.3f}); worst among clean files: {stem(cl[-1])} ({b[cl[-1]]['generated']:.3f}); "
              f"best of all 16: {stem(allr[0])} ({b[allr[0]]['generated']:.3f}); worst of all: {stem(allr[-1])} ({b[allr[-1]]['generated']:.3f}); "
              f"worst codec round trips: {', '.join(stem(x) for x in rtw)}. Clips: {n}/generated/<file>_seed1000.wav, with {n}/roundtrip/<file>.wav and original/<file>.wav",
              ]
    L += ["", "Listen first to the clean files (forest, park, rain, tokio): only there a generated clip cannot be a copy of material seen in training. "
          "For the other 12 files the reserve region repeats train audio, so a good-sounding clip is not evidence of generalisation.", ""]
    with open(path, "a") as fh:
        fh.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--listening", action="store_true", help="append the pointers to runs/s5_eval/README.md")
    a = ap.parse_args()
    runs = Path(a.runs).resolve()
    L, out = ["# B12 analysis (read-only)", ""], {}
    s1(runs, L, out)
    s5(runs, L, out)
    s3(runs, L, out)
    Path(a.out + ".md").write_text("\n".join(L))
    Path(a.out + ".json").write_text(json.dumps(out, indent=1, default=str))
    if a.listening:
        listening(runs, runs / "s5_eval" / "README.md")


if __name__ == "__main__":
    main()
