"""Diagrams of the overlap-add geometry, drawn from the real geometry code (SCAPES.data.geometry).

Usage (from SCAPES/):  python scripts/make_overlap_diagrams.py   -> docs/overlap_add/*.png
"""
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SCAPES.codecs import get_codec_info  # noqa: E402
from SCAPES.data.config_loader import load_dataprep_config  # noqa: E402
from SCAPES.data.geometry import crossfade_law_for, ola_window, resolve_geometry  # noqa: E402

OUT = REPO / "docs" / "overlap_add"
CONFIG = REPO / "SCAPES" / "config" / "codecs"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
ATOM_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # categorical slots 1-4, in order
N_ATOMS = 4


def geometry(template, **overrides):
    lines = [l for l in (CONFIG / template).read_text().splitlines() if not any(l.startswith(k + " ") for k in overrides)]
    lines += [f"{k} = {v}" for k, v in overrides.items() if v is not None]
    path = Path(tempfile.mkdtemp()) / "a.gin"
    path.write_text("\n".join(lines) + "\n")
    cfg = load_dataprep_config(path)
    info = get_codec_info(cfg.codec_name)
    return resolve_geometry(cfg, info), info


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def draw_case(ax_t, ax_g, g, info, title):
    law = crossfade_law_for(g, info)
    sr, h = g.sample_rate, g.hop_length
    F, H, X = g.unit_frames, g.unit_hop_frames, g.crossfade_samples
    G = g.grain_overlap_samples
    atom_s, hop_s = F * h / sr, g.audio_hop_samples / sr
    used_from = 0.0 if G else ((F - H) * h - X) / sr   # window is zero before this point of the atom
    w = ola_window(g, law).double().numpy()
    t_local = np.arange(len(w)) / sr
    total = np.zeros(int((N_ATOMS - 1) * g.audio_hop_samples + len(w)))
    t_total = np.arange(len(total)) / sr
    for k in range(N_ATOMS):
        c = ATOM_COLOURS[k]
        s = k * hop_s
        ax_t.barh(k, atom_s, left=s, height=0.62, color=c, alpha=0.18, edgecolor=c, linewidth=0.8)
        ax_t.barh(k, atom_s - used_from, left=s + used_from, height=0.62, color=c, alpha=0.8, linewidth=0)
        ax_t.text(s + 0.5 * atom_s, k, f"atom {k + 1}", ha="center", va="center", fontsize=7.5, color=INK)
        ax_g.plot(s + t_local, w, color=c, linewidth=1.6)
        i0 = k * g.audio_hop_samples
        total[i0:i0 + len(w)] += w ** 2 if law == "equal_power" else w
    for k in range(N_ATOMS - 1):                     # overlap of atom k with atom k + 1
        lo, hi = (k + 1) * hop_s, k * hop_s + atom_s
        for ax in (ax_t, ax_g):
            ax.axvspan(lo, hi, color="#8d8c86", alpha=0.10, linewidth=0)
    ax_g.plot(t_total, total, color=INK, linewidth=1.0, linestyle=(0, (4, 2)))
    ax_t.set_ylim(N_ATOMS - 0.4, -0.7)
    ax_t.set_yticks([])
    ax_t.set_title(title, fontsize=9, color=INK, loc="left", pad=22)
    overlap_s = G / sr if G else (F - H) * h / sr
    fade_s = G / sr if G else X / sr
    ax_t.text(0.0, 1.03, f"atom {atom_s * 1000:.0f} ms, hop {hop_s * 1000:.0f} ms, overlap {overlap_s * 1000:.0f} ms "
              f"({overlap_s / atom_s:.0%}), fade {fade_s * 1000:.0f} ms" + (" (grains)" if G else ""),
              transform=ax_t.transAxes, fontsize=7.5, color=INK2)
    ax_g.set_ylim(0, 1.25)
    ax_g.set_xlim(*ax_t.get_xlim())
    ax_g.set_xlabel("time (s)", fontsize=8, color=INK2)
    ax_g.set_ylabel("window gain" if law == "hann" else "window gain", fontsize=8, color=INK2)
    ax_g.text(0.99, 0.04, "dashed: sum of gains" if law == "hann" else "dashed: sum of squared gains (power)",
              transform=ax_g.transAxes, ha="right", fontsize=7.5, color=INK2)
    ax_g.text(0.01, 1.02, f"{'Hann' if law == 'hann' else 'equal power'}", transform=ax_g.transAxes, fontsize=8, color=INK2)
    style(ax_t)
    style(ax_g)


def grid(cases, path, suptitle):
    n = len(cases)
    fig, axes = plt.subplots(2, n, figsize=(max(4.6 * n, 9.0), 5.2), gridspec_kw={"height_ratios": [1.0, 0.9]}, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for j, (g, info, title) in enumerate(cases):
        draw_case(axes[0, j], axes[1, j], g, info, title)
    fig.suptitle(suptitle, fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.915, "faint part of a bar = zero gain (discarded); grey band = overlap with the next atom", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_laws():
    x = np.linspace(0, 1, 400)
    rise, fall = np.sin(np.pi / 2 * x) ** 2, np.cos(np.pi / 2 * x) ** 2          # Hann halves
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    fig.patch.set_facecolor(SURFACE)
    series = {"Hann": ("#2a78d6", lambda v: v), "equal power": ("#eb6834", np.sqrt)}
    for ax, kind in zip(axes, ("coherent", "incoherent")):
        for name, (c, f) in series.items():
            a, b = f(rise), f(fall)
            if kind == "coherent":              # the same waveform in both atoms: amplitudes add
                level = 20 * np.log10(a + b)
            else:                               # independent noise realisations: powers add
                level = 10 * np.log10(a ** 2 + b ** 2)
            ax.plot(x, level, color=c, linewidth=2, label=name)
        ax.axhline(0, color=INK2, linewidth=0.6)
        ax.set_ylim(-6, 4)
        ax.set_xlabel("position in the crossfade", fontsize=8.5, color=INK2)
        ax.set_ylabel("level of the sum (dB)", fontsize=8.5, color=INK2)
        ax.set_title("same waveform in both atoms (deterministic decoder)" if kind == "coherent"
                     else "independent decodes (stochastic decoder)", fontsize=9, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=8.5, loc="lower center")
        style(ax)
    axes[0].text(0.5, 3.1, "equal power: +3 dB bump", color="#eb6834", fontsize=8.5, ha="center")
    axes[1].text(0.5, -3.7, "Hann: -3 dB dip", color="#2a78d6", fontsize=8.5, ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "fig_laws.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_laws()

    g, info = geometry("encodec48k_atoms.gin")
    grid([(g, info, "EnCodec 48k (deterministic decoder)")], OUT / "fig_encodec48k.png",
         "EnCodec 48k: long overlap, Hann crossfade")

    cases = []
    for mode in ("normal", "strong", "brute"):
        g, info = geometry("music2latent_atoms.gin", **{"atoms.seconds": 0.37, "atoms.overlap_mode": f'"{mode}"'})
        cases.append((g, info, f"Music2Latent, overlap_mode = {mode}"))
    grid(cases, OUT / "fig_music2latent.png", "Music2Latent (4-frame atoms): the hop follows from the overlap mode, equal-power crossfade")

    cases = []
    g, info = geometry("codicodec_atoms.gin")
    cases.append((g, info, "CoDiCodec, trained V2 geometry (2 chunks)"))
    for mode in ("normal", "strong", "brute"):
        g, info = geometry("codicodec_grains.gin", **{"atoms.seconds": 0.74, "atoms.overlap_mode": f'"{mode}"'})
        cases.append((g, info, f"CoDiCodec grains, overlap_mode = {mode}"))
    grid(cases, OUT / "fig_codicodec.png",
         "CoDiCodec: the trained models regenerate a chunk (50 % overlap); grains generate one atom at a time and overlap in the audio only")
    print("written", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
