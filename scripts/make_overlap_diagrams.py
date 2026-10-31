"""Diagrams of the overlap-add geometry, drawn from the real geometry code (SCAPES.data.geometry).

Usage (from SCAPES/):  python scripts/make_overlap_diagrams.py   -> docs/overlap_add/{fig_windows,fig_consistent,fig_encodec48k}.png
Phase 1: Hann and equal-power windows, the four consistent sizes, EnCodec 48k (Hann). The independent figures come in phase 2.
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
from SCAPES.data.geometry import crossfade_halves, ola_window, resolve_geometry, window_for  # noqa: E402

OUT = REPO / "docs" / "overlap_add"
CONFIG = REPO / "SCAPES" / "config" / "codecs"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
ATOM_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # categorical slots 1-4, in order
N_ATOMS = 4


# Geometry of a template with some gin lines replaced
def geometry(template, **overrides):
    lines = [l for l in (CONFIG / template).read_text().splitlines() if not any(l.startswith(k + " ") for k in overrides)]
    lines += [f"{k} = {v}" for k, v in overrides.items()]
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


# One case: the atoms on a timeline (top) and their windows with the summed gain (bottom)
def draw_case(ax_t, ax_g, g, info, title):
    window = window_for(g, info)
    sr = g.sample_rate
    C, H, X = g.atom_samples, g.hop_samples, g.crossfade_samples
    used_from = (C - H - X) / sr                         # the window is zero before this point of the atom
    w = ola_window(g, window).double().numpy()
    t_local = np.arange(len(w)) / sr
    total = np.zeros((N_ATOMS - 1) * H + C)
    t_total = np.arange(len(total)) / sr
    for k in range(N_ATOMS):
        c = ATOM_COLOURS[k]
        s = k * H / sr
        ax_t.barh(k, C / sr, left=s, height=0.62, color=c, alpha=0.18, edgecolor=c, linewidth=0.8)
        ax_t.barh(k, C / sr - used_from, left=s + used_from, height=0.62, color=c, alpha=0.8, linewidth=0)
        ax_t.text(s + 0.5 * C / sr, k, f"atom {k + 1}", ha="center", va="center", fontsize=7.5, color=INK)
        ax_g.plot(s + t_local, w, color=c, linewidth=1.6)
        total[k * H:k * H + C] += w ** 2 if window == "equal_power" else w
    for k in range(N_ATOMS - 1):                         # overlap of atom k with atom k + 1
        for ax in (ax_t, ax_g):
            ax.axvspan((k + 1) * H / sr, (k * H + C) / sr, color="#8d8c86", alpha=0.10, linewidth=0)
    ax_g.plot(t_total, total, color=INK, linewidth=1.0, linestyle=(0, (4, 2)))
    ax_t.set_ylim(N_ATOMS - 0.4, -0.7)
    ax_t.set_yticks([])
    ax_t.set_title(title, fontsize=9, color=INK, loc="left", pad=22)
    ax_t.text(0.0, 1.03, f"atom {C / sr * 1000:.0f} ms, hop {H / sr * 1000:.0f} ms, overlap {(C - H) / sr * 1000:.0f} ms, "
              f"fade {X / sr * 1000:.0f} ms ({X} samples)", transform=ax_t.transAxes, fontsize=7.5, color=INK2)
    ax_g.set_ylim(0, 1.25)
    ax_g.set_xlim(*ax_t.get_xlim())
    ax_g.set_xlabel("time (s)", fontsize=8, color=INK2)
    ax_g.set_ylabel("window gain", fontsize=8, color=INK2)
    ax_g.text(0.99, 0.04, "dashed: sum of squared gains (power)" if window == "equal_power" else "dashed: sum of gains",
              transform=ax_g.transAxes, ha="right", fontsize=7.5, color=INK2)
    ax_g.text(0.01, 1.02, "equal power" if window == "equal_power" else "Hann", transform=ax_g.transAxes, fontsize=8, color=INK2)
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


# The two windows (halves) and the summed level of two decodes for correlation rho = 0 and 1
def fig_windows():
    n = 400
    u = (np.arange(n) + 0.5) / n
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    fig.patch.set_facecolor(SURFACE)
    series = {"Hann": ("#2a78d6", "hann"), "equal power": ("#eb6834", "equal_power")}
    for name, (c, win) in series.items():
        rise, fall = crossfade_halves(n, win)
        axes[0].plot(u, rise.numpy(), color=c, linewidth=2, label=f"{name}: rise")
        axes[0].plot(u, fall.numpy(), color=c, linewidth=2, linestyle="--", label=f"{name}: fall")
        for ax, rho in zip(axes[1:], (0.0, 1.0)):
            # power of the sum of two decodes with correlation rho: a^2 + b^2 + 2 rho a b
            power = rise.numpy() ** 2 + fall.numpy() ** 2 + 2 * rho * rise.numpy() * fall.numpy()
            ax.plot(u, 10 * np.log10(power), color=c, linewidth=2, label=name)
    axes[0].set_title("window halves", fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("gain", fontsize=8, color=INK2)
    axes[1].set_title("level of two decodes, rho = 0 (independent)", fontsize=9, color=INK, loc="left")
    axes[2].set_title("level of two decodes, rho = 1 (the same waveform)", fontsize=9, color=INK, loc="left")
    for ax in axes[1:]:
        ax.axhline(0, color=INK2, linewidth=0.6)
        ax.set_ylim(-6, 4)
        ax.set_ylabel("dB", fontsize=8, color=INK2)
    for ax in axes:
        ax.set_xlabel("position in the crossfade", fontsize=8, color=INK2)
        ax.legend(fontsize=7.5, frameon=False)
        style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_windows.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_consistent():
    cases = []
    for name, seconds in (("c2", 0.18), ("c4", 0.37), ("c6", 0.55), ("c8", 0.74)):
        g, info = geometry("music2latent_consistent.gin", **{"atoms.seconds": seconds})
        cases.append((g, info, f"Music2Latent consistent {name}: {g.unit_frames} latents"))
    grid(cases, OUT / "fig_consistent.png", "Consistent atoms: four sizes, one hop, one crossfade")


def fig_encodec48k():
    g, info = geometry("encodec48k_atoms.gin")
    grid([(g, info, "EnCodec 48k (deterministic decoder, Hann)")], OUT / "fig_encodec48k.png", "EnCodec 48k atoms")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_windows()
    fig_consistent()
    fig_encodec48k()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
