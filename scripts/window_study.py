"""Window study (design D10, D25): which overlap-add window keeps the level and the spectrum at the joins of atoms.

Each file's first seconds are segmented into atoms (C, H, X of a configuration), encoded in batches, decoded with several
base seeds (atom k with seed b + k), and overlap-added with each window of geometry.WINDOWS. Measures: rho_pair and rho_seed
(coherence of neighbouring decodes and of two seeds), the level at the joins (window_metrics), the join-masked MSS and the
big-window MSS against the source, plus two reference rows (whole-file round trip, white noise). No model is trained.
Phase 1 wires the Music2Latent consistent configurations; the independent and EnCodec 48k ones are phase 2.

Usage (from SCAPES/):
    ../venv_SCAPES/bin/python scripts/window_study.py --dataset ../datasets/microtex_v2 --out ../runs/window_study \
        --codecs music2latent --configs cons-c2 cons-c4 cons-c6 cons-c8 --start 0 --seconds 10 --base-seeds 0 10000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_metrics as em  # noqa: E402
import window_metrics as wm  # noqa: E402
from SCAPES.data.config_loader import DataprepConfig  # noqa: E402
from SCAPES.data.geometry import WINDOWS, crossfade_halves, resolve_geometry  # noqa: E402

BATCH = 64
REGION_SECONDS = (1.0, 9.0)       # the MSS region
G_RANGE = (0.25, 4.0)
CROSSOVER = 2 ** 0.5 - 1          # rho at which Hann and equal power deviate equally (design)
COLORS = {"hann": "tab:blue", "equal_power": "tab:orange"}
LISTEN_DEFAULT = ["applause", "keyboard", "orchestra", "wind"]

# Phase 1: Music2Latent consistent atoms (atoms.seconds, hop 0.10 s, crossfade 0.02 s, as the template of slice 8)
CONFIGS = {
    "music2latent": {"cons-c2": 0.18, "cons-c4": 0.37, "cons-c6": 0.55, "cons-c8": 0.74},
}
HOP_SECONDS, CROSSFADE_SECONDS = 0.10, 0.02


# ─── helpers ───

def build_geometry(codec_name, seconds, info):
    cfg = DataprepConfig(codec_name=codec_name, generation_mode="atoms", atoms_type="consistent", atoms_seconds=seconds,
                         atoms_hop_seconds=HOP_SECONDS, atoms_crossfade_seconds=CROSSFADE_SECONDS, memory_buffer_atoms=3)
    return resolve_geometry(cfg, info)


def minmax(values):
    return [float(min(values)), float(max(values))]


def mean(values):
    return float(np.mean(values))


# Encodes atoms xs [K, ch, C] in batches; returns z [K, d, F] and scale [K] or None, on the codec device
def encode_atoms(codec, xs, r):
    zs, scales = [], []
    for i in range(0, xs.shape[0], BATCH):
        enc = codec.encode(xs[i: i + BATCH].float(), r)
        zs.append(enc.z)
        scales.append(enc.scale)
    return torch.cat(zs), (torch.cat(scales) if scales[0] is not None else None)


# Decodes atom k with seed base_seed + k; returns [K, ch, C] float32 on the cpu
@torch.no_grad()
def decode_atoms(codec, z, scale, base_seed):
    out = []
    for k in range(z.shape[0]):
        s = scale[k: k + 1] if scale is not None else None
        out.append(codec.decode(z[k: k + 1], s, base_seed + k)[0].float().cpu())
    return torch.stack(out)


def to_48k_mono(sig, r):
    return em.resample_to_48k(em.to_mono(sig.numpy() if torch.is_tensor(sig) else sig)[None].astype(np.float32), r)[0]


def write_wav(path, sig, sr):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(np.asarray(sig).T, -1.0, 1.0), sr, subtype="PCM_16")


def new_cfg_acc(geom):
    return {"geom": {"C": geom.atom_samples, "H": geom.hop_samples, "X": geom.crossfade_samples, "F": geom.unit_frames},
            "files": {}, "seeds": {}, "flags": [], "overlap_add_length": None}


def seed_acc(acc, seed):
    if seed not in acc["seeds"]:
        acc["seeds"][seed] = {"Y2": 0.0, "X2": 0.0, "Px": torch.zeros(wm.N_BINS, dtype=torch.float64),
                              "Py": {w: torch.zeros(wm.N_BINS, dtype=torch.float64) for w in WINDOWS},
                              "join_mss": {w: [] for w in WINDOWS}, "big_mss": {w: [] for w in WINDOWS}}
    return acc["seeds"][seed]


# ─── one file, one configuration ───

def study_file(codec, info, geom, acc, name, x, orig48, seeds, args, listen_dir, codec_name, config):
    C, H, X, r = geom.atom_samples, geom.hop_samples, geom.crossfade_samples, info.sample_rate
    xs = wm.segment(x.cpu().double(), C, H)
    K = xs.shape[0]
    z, scale = encode_atoms(codec, xs.to(x.device), r)
    starts = wm.join_starts(K, C, H, X)
    intervals = [(int(round(s * 48000 / r)), int(round((s + X) * 48000 / r))) for s in starts]
    region = (int(REGION_SECONDS[0] * 48000), int(REGION_SECONDS[1] * 48000))
    ys = {}
    file_acc = {"rho_pair": {}, "G": {}}
    px = wm.bin_powers(x.cpu(), starts, X)
    for seed in seeds:
        y = decode_atoms(codec, z, scale, seed)
        if not torch.isfinite(y).all():
            raise RuntimeError(f"non-finite decode: {codec_name} {config} {name} seed {seed}")
        ys[seed] = y
        yd = y.double()
        # rho_pair: tail of y_{k-1} against head of y_k on the crossfade interval
        a = yd[:-1][:, :, C - X:]
        b = yd[1:][:, :, C - H - X: C - H]
        sab, saa, sbb = wm.pooled_coherence(a, b)
        file_acc["rho_pair"][seed] = sab / np.sqrt(saa * sbb)
        G = wm.atom_gain(yd, xs)
        file_acc["G"][seed] = G
        if not G_RANGE[0] <= G <= G_RANGE[1]:
            raise RuntimeError(f"G {G:.3f} outside {G_RANGE}: {codec_name} {config} {name} seed {seed}")
        sa = seed_acc(acc, seed)
        sa["Y2"] += float((yd ** 2).sum())
        sa["X2"] += float((xs ** 2).sum())
        sa["Px"] += px
        for win in WINDOWS:
            out = wm.overlap_add(yd, H, wm.window(C, H, X, win))
            acc["overlap_add_length"] = int(out.shape[-1])
            sa["Py"][win] += wm.bin_powers(out, starts, X)
            y48 = to_48k_mono(out.float(), r)
            sa["join_mss"][win].append(wm.join_mss(orig48, y48, intervals, region))
            sl = slice(region[0], region[1])
            sa["big_mss"][win].append(em.mss(orig48[sl], y48[sl])["mean_log_l1"])
            if seed == seeds[0] and listen_dir is not None and name in args.listen_files:
                write_wav(listen_dir / codec_name / config / f"{name}__{win}.wav", out.float().numpy(), r)
    # rho_seed: the same atom decoded with the two first seeds, on the rise and fall regions of the window
    if len(seeds) >= 2:
        ya, yb = ys[seeds[0]].double(), ys[seeds[1]].double()
        sab = saa = sbb = 0.0
        for sl in (slice(C - H - X, C - H), slice(C - X, C)):
            p, q, s = wm.pooled_coherence(ya[:, :, sl], yb[:, :, sl])
            sab, saa, sbb = sab + p, saa + q, sbb + s
        file_acc["rho_seed"] = sab / np.sqrt(saa * sbb)
    acc["files"][name] = file_acc
    return intervals, region


# Reference rows of one file: whole-file codec round trip and RMS-matched white noise, big MSS and join MSS per configuration
def reference_file(codec, info, x, orig48, intervals_by_config, region):
    r = info.sample_rate
    enc = codec.encode(x[None].float(), r)
    with torch.no_grad():
        rt = codec.decode(enc.z, enc.scale, 0)[0].float().cpu()
    rt48 = to_48k_mono(rt, r)
    noise48 = em.white_noise_like(orig48, 0, len(rt48))
    sl = slice(region[0], region[1])
    out = {}
    for label, sig in (("roundtrip", rt48), ("noise", noise48)):
        out[label] = {"big_mss": em.mss(orig48[sl], sig[sl])["mean_log_l1"],
                      "join_mss": {c: wm.join_mss(orig48, sig, iv, region) for c, iv in intervals_by_config.items()}}
    return out


# ─── summaries ───

def summarise_config(acc, seeds):
    out = {"geom": acc["geom"], "flags": acc["flags"], "overlap_add_length": acc["overlap_add_length"]}
    files = acc["files"]
    out["rho_pair"] = {"mean": mean([mean(list(f["rho_pair"].values())) for f in files.values()]),
                       "minmax": minmax([mean(list(f["rho_pair"].values())) for f in files.values()]),
                       "per_file": {n: mean(list(f["rho_pair"].values())) for n, f in files.items()}}
    if all("rho_seed" in f for f in files.values()):
        vals = [f["rho_seed"] for f in files.values()]
        out["rho_seed"] = {"mean": mean(vals), "minmax": minmax(vals), "per_file": {n: f["rho_seed"] for n, f in files.items()}}
    else:
        out["rho_seed"] = None
    allG = [g for f in files.values() for g in f["G"].values()]
    Y2 = sum(s["Y2"] for s in acc["seeds"].values())
    X2 = sum(s["X2"] for s in acc["seeds"].values())
    out["G"] = {"pooled": Y2 / X2, "minmax_files": minmax(allG)}
    Px = sum(s["Px"] for s in acc["seeds"].values())
    out["windows"] = {}
    for win in WINDOWS:
        Py = sum(s["Py"][win] for s in acc["seeds"].values())
        level = wm.level_db(Py, Px, Y2 / X2)
        summary = wm.level_summary(level)
        per_seed = {}
        for seed, s in acc["seeds"].items():
            per_seed[str(seed)] = wm.level_summary(wm.level_db(s["Py"][win], s["Px"], s["Y2"] / s["X2"]))
        join = {str(seed): mean(s["join_mss"][win]) for seed, s in acc["seeds"].items()}
        big = {str(seed): mean(s["big_mss"][win]) for seed, s in acc["seeds"].items()}
        out["windows"][win] = {
            "level_db": [float(v) for v in level], **summary,
            "per_seed": per_seed,
            "minmax_seeds": {k: minmax([p[k] for p in per_seed.values()]) for k in ("delta", "centre", "edge")},
            "join_mss_by_seed": join, "big_mss_by_seed": big,
            "join_mss": mean(list(join.values())), "big_mss": mean(list(big.values())),
            "join_mss_per_file_seed": {str(seed): s["join_mss"][win] for seed, s in acc["seeds"].items()},
        }
    return out


def proposal(codec_summary, configs):
    worst = {w: max(codec_summary[c]["windows"][w]["delta"] for c in configs) for w in WINDOWS}
    seeds = list(codec_summary[configs[0]]["windows"]["hann"]["join_mss_by_seed"])
    mss, rng = {}, {}
    for w in WINDOWS:
        by_seed = [mean([codec_summary[c]["windows"][w]["join_mss_by_seed"][s] for c in configs]) for s in seeds]
        mss[w] = mean(by_seed)
        rng[w] = float(max(by_seed) - min(by_seed))
    result = wm.propose_window(worst, mss, rng)
    result.update({"worst": worst, "mss": mss, "mss_seed_range": rng, "configs": list(configs)})
    return result


# ─── outputs ───

def fmt_mm(v, d=3):
    return f"{v[0]:.{d}f} to {v[1]:.{d}f}"


def write_markdown(path, results, proposals, refs, args):
    lines = ["# Window study", "",
             f"files: {len(args.files_used)}; first {args.seconds} s from {args.start} s; base seeds {args.base_seeds}; "
             f"tolerance {wm.TOL_DB} dB", ""]
    for codec_name, summary in results.items():
        lines += [f"## {codec_name}", "",
                  "| config | C | H | X | rho_pair (min-max files) | rho_seed (min-max files) | G |", "|---|---|---|---|---|---|---|"]
        for cfg, s in summary.items():
            g = s["geom"]
            rs = f"{s['rho_seed']['mean']:.3f} ({fmt_mm(s['rho_seed']['minmax'])})" if s["rho_seed"] else "n/a"
            lines.append(f"| {cfg} | {g['C']} | {g['H']} | {g['X']} | {s['rho_pair']['mean']:.3f} ({fmt_mm(s['rho_pair']['minmax'])}) "
                         f"| {rs} | {s['G']['pooled']:.3f} |")
        lines += ["", "| config | window | delta dB (min-max seeds) | centre dB (min-max seeds) | edge dB (min-max seeds) | join MSS | big MSS |",
                  "|---|---|---|---|---|---|---|"]
        for cfg, s in summary.items():
            for win in WINDOWS:
                w = s["windows"][win]
                mm = w["minmax_seeds"]
                lines.append(f"| {cfg} | {win} | {w['delta']:.2f} ({fmt_mm(mm['delta'], 2)}) | {w['centre']:+.2f} ({fmt_mm(mm['centre'], 2)}) "
                             f"| {w['edge']:+.2f} ({fmt_mm(mm['edge'], 2)}) | {w['join_mss']:.4f} | {w['big_mss']:.4f} |")
        p = proposals[codec_name]
        lines += ["", f"Proposal: **{p['window']}** by {p['by']}. worst delta {p['worst']}, join MSS {p['mss']}, "
                      f"seed range {p['mss_seed_range']}.",
                  f"H1 (equal power keeps the level better, margin > {wm.TOL_DB} dB): "
                  f"{'supported' if p['h1']['supported'] else 'not supported'}, margin {p['h1']['margin_db']:.2f} dB "
                  f"(hann worst minus equal_power worst).", ""]
        ref = refs[codec_name]
        lines += ["Reference rows (mean over files; big MSS, join MSS per configuration):", "",
                  "| row | big MSS | " + " | ".join(f"join MSS {c}" for c in summary) + " |",
                  "|---|---|" + "---|" * len(summary)]
        for label in ("roundtrip", "noise"):
            lines.append(f"| {label} | {ref[label]['big_mss']:.4f} | " + " | ".join(f"{ref[label]['join_mss'][c]:.4f}" for c in summary) + " |")
        lines.append("")
    path.write_text("\n".join(lines))


def make_figures(out_dir, results):
    for codec_name, summary in results.items():
        cfgs = list(summary)
        # delta, grouped bars
        fig, ax = plt.subplots(figsize=(1.8 * len(cfgs) + 2, 4))
        xpos = np.arange(len(cfgs))
        for i, win in enumerate(WINDOWS):
            ax.bar(xpos + (i - 0.5) * 0.38, [summary[c]["windows"][win]["delta"] for c in cfgs], 0.38, label=win, color=COLORS[win])
        ax.axhline(wm.TOL_DB, color="k", linestyle="--", label=f"tolerance {wm.TOL_DB} dB")
        ax.set_xticks(xpos, cfgs)
        ax.set_ylabel("delta (dB)")
        ax.set_title(f"{codec_name}: worst bin deviation at the joins")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"delta_{codec_name}.png", dpi=120)
        plt.close(fig)
        # rho, with min-max over files
        fig, ax = plt.subplots(figsize=(1.8 * len(cfgs) + 2, 4))
        for i, (key, color) in enumerate((("rho_pair", "tab:green"), ("rho_seed", "tab:purple"))):
            vals = [summary[c][key] for c in cfgs]
            if any(v is None for v in vals):
                continue
            m = np.array([v["mean"] for v in vals])
            lo = np.array([v["minmax"][0] for v in vals])
            hi = np.array([v["minmax"][1] for v in vals])
            ax.bar(xpos + (i - 0.5) * 0.38, m, 0.38, yerr=[m - lo, hi - m], capsize=3, label=key, color=color)
        ax.axhline(CROSSOVER, color="k", linestyle="--", label=f"crossover {CROSSOVER:.2f}")
        ax.set_xticks(xpos, cfgs)
        ax.set_ylabel("rho")
        ax.set_title(f"{codec_name}: coherence (bars: min-max over files)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"rho_{codec_name}.png", dpi=120)
        plt.close(fig)
        # level profile per configuration
        for cfg in cfgs:
            s = summary[cfg]
            fig, ax = plt.subplots(figsize=(6, 4))
            u_bins = (np.arange(wm.N_BINS) + 0.5) / wm.N_BINS
            rho = s["rho_pair"]["mean"]
            for win in WINDOWS:
                ax.plot(u_bins, s["windows"][win]["level_db"], "o-", color=COLORS[win], label=f"{win} measured")
                rise, fall = (t.double().numpy() for t in crossfade_halves(1024, win))
                phi = rise ** 2 + fall ** 2 + 2 * rho * rise * fall
                ax.plot((np.arange(1024) + 0.5) / 1024, 10 * np.log10(phi), "--", color=COLORS[win], label=f"{win} theory, rho={rho:.2f}")
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.set_xlabel("u (position in the crossfade)")
            ax.set_ylabel("level (dB)")
            ax.set_title(f"{codec_name} {cfg}: level at the joins")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(out_dir / f"profile_{codec_name}_{cfg}.png", dpi=120)
            plt.close(fig)


# ─── main ───

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--codecs", nargs="+", default=["music2latent"])
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--base-seeds", nargs="+", type=int, default=[0, 10000])
    p.add_argument("--files", nargs="+", default=None, help="wav names in <dataset>/raw (default all)")
    p.add_argument("--configs", nargs="+", default=None, help="configuration names (default all wired for the codec)")
    p.add_argument("--listen-files", nargs="+", default=LISTEN_DEFAULT, help="file names (no .wav) written as audio")
    args = p.parse_args()

    from audio_codec_wrappers.audio import adapt_audio, load_audio
    from SCAPES.codecs import get_codec, get_codec_info
    from SCAPES.training.run_logging import environment_record

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    listen_dir = out_dir / "listen"
    raw = Path(args.dataset) / "raw"
    paths = [raw / f for f in args.files] if args.files else sorted(raw.glob("*.wav"))
    args.files_used = [q.name for q in paths]
    args.listen_files = [f[:-4] if f.endswith(".wav") else f for f in args.listen_files]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = list(args.base_seeds)
    (out_dir / "env.json").write_text(json.dumps(environment_record(), indent=1))

    audios = {}
    for q in paths:
        a, sr = load_audio(q)
        a = adapt_audio(a, sr, 48000, 2)[0]
        audios[q.stem] = a[:, int(args.start * 48000): int((args.start + args.seconds) * 48000)]
    for stem in args.listen_files:
        if stem in audios:
            write_wav(listen_dir / "original" / f"{stem}.wav", audios[stem].numpy(), 48000)

    results, proposals, refs = {}, {}, {}
    t0 = time.time()
    for codec_name in args.codecs:
        if codec_name not in CONFIGS:
            raise NotImplementedError(f"{codec_name}: its configurations are phase 2 (P2-1); phase 1 wires {list(CONFIGS)}")
        wired = CONFIGS[codec_name]
        names = args.configs if args.configs else list(wired)
        for c in names:
            if c not in wired:
                raise NotImplementedError(f"configuration {c!r} of {codec_name} is not wired in phase 1: {list(wired)}")
        info = get_codec_info(codec_name)
        codec = get_codec(codec_name, device)
        geoms = {c: build_geometry(codec_name, wired[c], info) for c in names}
        accs = {c: new_cfg_acc(g) for c, g in geoms.items()}
        ref_files = []
        for stem, audio in audios.items():
            x = codec.prepare(audio, 48000)[0]
            orig48 = em.to_mono(audio.numpy())
            intervals_by_config = {}
            for c in names:
                intervals, region = study_file(codec, info, geoms[c], accs[c], stem, x, orig48, seeds, args,
                                               listen_dir, codec_name, c)
                intervals_by_config[c] = intervals
            ref_files.append(reference_file(codec, info, x, orig48, intervals_by_config, region))
            print(f"[{time.time() - t0:7.1f}s] {codec_name} {stem} done", flush=True)
        results[codec_name] = {c: summarise_config(accs[c], seeds) for c in names}
        proposals[codec_name] = proposal(results[codec_name], names)
        refs[codec_name] = {label: {"big_mss": mean([r[label]["big_mss"] for r in ref_files]),
                                    "join_mss": {c: mean([r[label]["join_mss"][c] for r in ref_files]) for c in names}}
                            for label in ("roundtrip", "noise")}
        for c in names:
            s = results[codec_name][c]
            print(f"{codec_name} {c}: rho_pair {s['rho_pair']['mean']:.3f} G {s['G']['pooled']:.3f} "
                  + " ".join(f"{w} delta {s['windows'][w]['delta']:.2f} join_mss {s['windows'][w]['join_mss']:.4f}" for w in WINDOWS),
                  flush=True)
        print(f"{codec_name} proposal: {proposals[codec_name]['window']} by {proposals[codec_name]['by']}", flush=True)

    (out_dir / "window_study.json").write_text(json.dumps(
        {"args": {k: v for k, v in vars(args).items()}, "results": results, "proposals": proposals, "references": refs,
         "seconds_total": time.time() - t0}, indent=1))
    write_markdown(out_dir / "window_study.md", results, proposals, refs, args)
    make_figures(out_dir, results)
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    (out_dir / "peak_memory.json").write_text(json.dumps({"peak_gpu_bytes": peak}))
    print(f"done in {time.time() - t0:.0f}s, peak GPU {peak / 2 ** 20:.0f} MiB, outputs in {out_dir}")


if __name__ == "__main__":
    main()
