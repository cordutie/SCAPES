"""E2 checks of the dataprep of the sample-level atom variants (design S5, S6, D23).

Usage (from SCAPES/):
    ../venv_SCAPES/bin/python scripts/atoms_check.py --dataset ../datasets/microtex_v2 --runs ../runs/v2_atoms --out ../runs/v2_atoms_check [--variants ...]
Per variant config (runs/<name>/config/dataprep.gin):
  - geometry against the design table (F, C, H, X, zeroed part, key, window equal_power);
  - per file, index.json num_atoms == atom_count(5292000, C, H);
  - codec round trip of the stored atoms with k H + C <= 10 r (seed k), duration check, envelope lag of the round trip
    against the source on [w, w + 1) s for w in 1, 3, 5, 7, and two positive controls on the same windows:
      1. the source delayed by 2048 samples against the source: lag 2048 +- 256 at every window;
      2. the same atoms laid with the hop shortened by SHORTEN samples ("simulated time loss"): |lag| grows with w;
  - big-window MSS of the round trip against the 48 kHz original.
Writes atoms_check.json, atoms_check.md, lag.png, peak_memory.json, env.json. Exits 1 if a count, geometry or length check
fails, or if control 1 fails (the diagnostic would not see a delay).
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import eval_metrics as em
from SCAPES.codecs import get_codec, get_codec_info
from SCAPES.data.config_loader import load_dataprep_config
from SCAPES.data.geometry import atom_count, resolve_geometry, window_for
from SCAPES.data.representations import RepresentationStore
from SCAPES.inference.render import ola_render
from SCAPES.training.run_logging import environment_record
from teacher_forced_eval import duration_check

N_SOURCE = 5292000          # samples of a 120 s file at 44.1 kHz
SECONDS = 10
WINDOWS = (1, 3, 5, 7)
DELAY = 2048
DELAY_TOL = 256
LAG_LIMIT = 512  # S6 criterion, reported not gating
SHORTEN = 64    # 256 saturates the 0.2 s max lag from w = 3 s on (shift 256 * floor(w r / H) > 8820)
# design table: variant -> (F, C, zeroed); H = 4410 and X = 3782 for all four
EXPECTED = {"m2l-cons-c2": (2, 8192, 0), "m2l-cons-c4": (4, 16384, 8192),
            "m2l-cons-c6": (6, 24576, 16384), "m2l-cons-c8": (8, 32768, 24576)}
H_EXPECTED, X_EXPECTED = 4410, 3782


def envelope_lag(x, y, r):
    from audio_codec_wrappers.metrics import envelope_lag as lag
    return int(lag(x, y, r, max_lag_s=0.2))


def mono(a):
    return a.mean(axis=0) if a.ndim == 2 else a


# Decodes the first K10 stored atoms of a file (seed k), returns the list of audios [ch, C] on the cpu
@torch.no_grad()
def decode_stored(codec, data, geom, k10):
    audios = []
    for k in range(k10):
        z = data["latent"][k].float().unsqueeze(0).to(codec.device if hasattr(codec, "device") else "cuda")
        scale = data["scale"][k:k + 1].float().reshape(-1).to(z.device) if geom.has_scale else None
        audios.append(codec.decode(z, scale, k)[0].float().cpu())
    return audios


def check_variant(name, run_dir, dataset, codec_cache, device, out_dir):
    cfg = load_dataprep_config(run_dir / "config" / "dataprep.gin")
    info = get_codec_info(cfg.codec_name)
    geom = resolve_geometry(cfg, info)
    store = RepresentationStore(dataset, geom, info)
    store.ensure()
    r = geom.sample_rate
    res = {"variant": name, "key": store.key, "checks": {}}

    # geometry against the table
    F, C, zeroed = EXPECTED[name]
    window = window_for(geom, info)
    geo = {"F": geom.unit_frames, "C": geom.atom_samples, "H": geom.hop_samples, "X": geom.crossfade_samples,
           "zeroed": geom.atom_samples - geom.hop_samples - geom.crossfade_samples, "key": store.key, "window": window,
           "atom_type": geom.atom_type}
    res["geometry"] = geo
    res["checks"]["geometry"] = bool(geo["F"] == F and geo["C"] == C and geo["H"] == H_EXPECTED and geo["X"] == X_EXPECTED
                                     and geo["zeroed"] == zeroed and geo["key"] == f"music2latent__atoms-c{C}-h{H_EXPECTED}"
                                     and window == "equal_power" and geom.atom_type == "consistent")

    # atom counts
    index = store.read_index()
    want = atom_count(N_SOURCE, geom.atom_samples, geom.hop_samples)
    counts = {s: e.get("num_atoms") for s, e in index.items()}
    res["num_atoms"] = counts
    res["num_atoms_expected"] = want
    res["checks"]["counts"] = bool(len(counts) > 0 and all(v == want for v in counts.values()))

    # round trip and lags
    from audio_codec_wrappers.audio import load_audio
    codec = get_codec(cfg.codec_name, device)
    k10 = atom_count(SECONDS * r, geom.atom_samples, geom.hop_samples)
    res["k10"] = k10
    short = dataclasses.replace(geom, hop_samples=geom.hop_samples - SHORTEN)
    files = sorted((dataset / "raw").glob("*.wav"))
    per_file, durations = {}, {}
    for path in files:
        stem = path.stem
        data = store.load_latents(path)
        audio, sr = load_audio(path)
        x = codec.prepare(audio, sr)[0].float().cpu().numpy()           # [ch, N] at the codec rate
        audios = decode_stored(codec, data, geom, k10)
        y = ola_render(audios, geom, window).numpy()
        y_short = ola_render(audios, short, window).numpy()
        durations[stem] = duration_check(geom, k10, y.shape[-1], SECONDS)
        xm, ym, ysm = mono(x), mono(y), mono(y_short)
        delayed = np.concatenate([np.zeros(DELAY), xm])[:len(xm)]
        lags = {"roundtrip": [], "delayed": [], "short_hop": []}
        for w in WINDOWS:
            a, b = int(w * r), int((w + 1) * r)
            lags["roundtrip"].append(envelope_lag(xm[a:b], ym[a:b], r))
            lags["delayed"].append(envelope_lag(xm[a:b], delayed[a:b], r))
            lags["short_hop"].append(envelope_lag(xm[a:b], ysm[a:b], r))
        # big-window MSS against the 48 kHz original, region [0.4 s, common end)
        ref = em.to_mono(em.resample_to_48k(x[:, :SECONDS * r], r))
        sig = em.to_mono(em.resample_to_48k(y, r))
        a0, e = int(em.EXCLUDE_SECONDS * em.SR), min(len(ref), len(sig))
        mss = em.mss(ref[a0:e], sig[a0:e])
        per_file[stem] = {"lags": lags, "mss": {"mean_log_l1": float(mss["mean_log_l1"]), "mean_spec_conv": float(mss["mean_spec_conv"])}}
        print(f"  {name} {stem}: lags {lags['roundtrip']}, delayed {lags['delayed']}, short {lags['short_hop']}", flush=True)
    res["files"] = per_file
    res["duration_check"] = durations
    res["checks"]["lengths"] = bool(all(d["ok"] for d in durations.values()))

    # controls (D23)
    delayed_ok = all(abs(l - DELAY) <= DELAY_TOL for f in per_file.values() for l in f["lags"]["delayed"])
    need = SHORTEN * (6 * r // geom.hop_samples) / 2
    if need <= 0.2 * r:
        grow = [abs(f["lags"]["short_hop"][-1]) - abs(f["lags"]["short_hop"][0]) for f in per_file.values()]
        short_ok = bool(all(g >= need for g in grow))
        res["control_short_hop"] = {"need_growth_samples": need, "growth": grow, "ok": short_ok}
    else:
        res["control_short_hop"] = {"need_growth_samples": need, "ok": None, "note": "beyond max lag"}
    res["control_delayed"] = {"ok": bool(delayed_ok)}
    res["checks"]["control_delayed"] = bool(delayed_ok)
    res["roundtrip_lag_max_abs"] = int(max(abs(l) for f in per_file.values() for l in f["lags"]["roundtrip"]))
    res["roundtrip_mss_l1_mean"] = float(np.mean([f["mss"]["mean_log_l1"] for f in per_file.values()]))
    del codec
    return res


def write_md(path, results):
    L = ["# Atoms check (E2)", ""]
    L += ["| variant | key | F | C | H | X | zeroed | atoms/file | geometry | counts | lengths | control delay | control short hop | max abs round-trip lag | round-trip MSS L1 (vs 44.1 kHz-prepared source) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        g, c = r["geometry"], r["checks"]
        ctl = r["control_short_hop"]
        if ctl["ok"] is None:
            ctl_s = "beyond max lag"
        elif ctl["ok"]:
            ctl_s = "ok"
        else:
            bad = [s for s, g in zip(r["files"], ctl["growth"]) if g < ctl["need_growth_samples"]]
            ctl_s = "non-gating; failing files: " + ", ".join(bad)
        L.append(f"| {r['variant']} | {r['key']} | {g['F']} | {g['C']} | {g['H']} | {g['X']} | {g['zeroed']} | {r['num_atoms_expected']} "
                 f"| {c['geometry']} | {c['counts']} | {c['lengths']} | {c['control_delayed']} | {ctl_s} "
                 f"| {r['roundtrip_lag_max_abs']} | {r['roundtrip_mss_l1_mean']:.3f} |")
    L += ["", f"Lag in samples at the codec rate, windows w = {list(WINDOWS)} s, positive = the round trip is later than the source.", ""]
    L += [f"S6 lag verdict: the criterion |round-trip lag| <= {LAG_LIMIT} samples is reported, not gating (design S6). "
          "Files over the limit, per variant:", ""]
    for r in results:
        over = [s for s, f in r["files"].items() if max(abs(l) for l in f["lags"]["roundtrip"]) > LAG_LIMIT]
        L.append(f"- {r['variant']}: " + (", ".join(over) if over else "none"))
    L += ["", f"SHORTEN = {SHORTEN} samples (not the plan's 256): 256 saturates the 0.2 s max lag (8820 samples at 44.1 kHz), "
          "because the shift to measure at w = 7 s is 17920 > 8820. The short-hop control is non-gating.", ""]
    L += ["Evidence (senior's probe, for the P1 report): the orchestra source envelope has low contrast (std 1.25 vs rain 1.79 "
          "vs helicopter 5.68); its Pearson correlation is 0.34 at lag 0 and 0.49 at lag -1792 (broad, weak peak); the readings "
          "do not drift with w and the lengths are exact, so the round-trip lag is not accumulated time loss. The helicopter "
          "envelope is periodic (autocorrelation 0.62 at 14 frames = 3584 samples, rotor at about 12 Hz), so its short-hop lag "
          "can only be read modulo about 3584 samples.", ""]
    for r in results:
        L.append(f"## {r['variant']}: lag per file (round trip | delayed control | short-hop control)")
        for s, f in r["files"].items():
            L.append(f"- {s}: {f['lags']['roundtrip']} | {f['lags']['delayed']} | {f['lags']['short_hop']}")
        L.append("")
    path.write_text("\n".join(L))


def write_figure(path, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for c, r in zip(colours, results):
        arr = np.array([f["lags"]["roundtrip"] for f in r["files"].values()])
        ax.errorbar(WINDOWS, arr.mean(0), yerr=[arr.mean(0) - arr.min(0), arr.max(0) - arr.mean(0)], color=c, marker="o",
                    capsize=3, label=r["variant"])
    for v in (-512, 512):
        ax.axhline(v, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("window start (s)")
    ax.set_ylabel("envelope lag of the round trip (samples)")
    ax.set_title("round trip against the source (dashed: +-512)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="+")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = Path(args.runs)
    names = args.variants or [n for n in EXPECTED if (runs / n / "config").is_dir()]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    results = []
    for name in names:
        print(f"checking {name}", flush=True)
        results.append(check_variant(name, runs / name, Path(args.dataset), None, args.device, out))
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    (out / "atoms_check.json").write_text(json.dumps(results, indent=1))
    (out / "peak_memory.json").write_text(json.dumps({"peak_gpu_gb": peak}))
    (out / "env.json").write_text(json.dumps(environment_record(), indent=1))
    write_md(out / "atoms_check.md", results)
    write_figure(out / "lag.png", results)
    failed = [(r["variant"], k) for r in results for k, v in r["checks"].items() if not v]
    print("failed checks:", failed or "none")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
