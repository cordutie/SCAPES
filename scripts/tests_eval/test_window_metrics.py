"""Known-answer tests of scripts/window_metrics.py: coherence, join level, join-masked MSS, the two-way window proposal
(design D10, D25). Phase 1: the window check of gate W (consistent atoms) uses them. CPU only. Run from SCAPES/: ../venv_SCAPES/bin/python -m pytest scripts/tests_eval -q
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import window_metrics as wm  # noqa: E402


def test_segment():
    from SCAPES.data.geometry import atom_count
    x = torch.arange(1000, dtype=torch.float64).view(1, -1)
    seg = wm.segment(x, 128, 40)
    K = atom_count(1000, 128, 40)
    assert seg.shape == (K, 1, 128)
    for k in range(K):
        assert seg[k, 0, 0] == 40 * k and seg[k, 0, -1] == 40 * k + 127


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_pooled_coherence_known(rho):
    g = torch.Generator().manual_seed(0)
    s, n = torch.randn(2, 200000, generator=g, dtype=torch.float64)
    a, b = s, rho * s + math.sqrt(1 - rho ** 2) * n
    sab, saa, sbb = wm.pooled_coherence(a.view(1, -1), b.view(1, -1))
    assert abs(sab / math.sqrt(saa * sbb) - rho) <= 0.01


def _decodes(rho, C=4096, H=2048, K=300, seed=0, gain=1.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, (K - 1) * H + C, generator=g, dtype=torch.float64)
    xs = wm.segment(x, C, H)
    noise = torch.randn(xs.shape, generator=g, dtype=torch.float64)
    y = gain * (math.sqrt(rho) * xs + math.sqrt(1 - rho) * noise)
    return x, xs, y


def _binned_expectation(name, X, rho, n_bins=16):
    from SCAPES.data.geometry import crossfade_halves
    r, f = (t.double() for t in crossfade_halves(X, name))
    phi = r ** 2 + f ** 2 + 2 * rho * r * f                    # power of two unit-power decodes of correlation rho
    b = (torch.arange(X) * n_bins) // X
    return torch.stack([phi[b == i].mean() for i in range(n_bins)])


# (C, H, X): a whole-overlap crossfade (independent brute shape) and the consistent c2 geometry of gate W (zeroed part 2900)
GEOMS = [(4096, 2048, 2048), (8192, 4410, 882)]


@pytest.mark.parametrize("rho", [0.0, 1.0])
@pytest.mark.parametrize("name", ["hann", "equal_power"])
@pytest.mark.parametrize("C,H,X", GEOMS)
def test_level_known(rho, name, C, H, X):
    x, xs, y = _decodes(rho, C, H)
    out = wm.overlap_add(y, H, wm.window(C, H, X, name))
    starts = wm.join_starts(y.shape[0], C, H, X)
    G = wm.atom_gain(y, xs)
    level = wm.level_db(wm.bin_powers(out, starts, X), wm.bin_powers(x, starts, X), G)
    summary = wm.level_summary(level)
    expect = 10 * torch.log10(_binned_expectation(name, X, rho))
    assert abs(summary["centre"] - float(expect[list(wm.CENTRE_BINS)].mean())) <= 0.2
    assert abs(summary["delta"] - float(expect.abs().max())) <= 0.2
    nominal = {("hann", 1.0): 0.0, ("equal_power", 1.0): 3.0, ("hann", 0.0): -3.0, ("equal_power", 0.0): 0.0}[(name, rho)]
    # the centre bins average over u in [1/4, 3/4), so the nominal u = 1/2 value is approached, not reached
    assert abs(summary["centre"] - nominal) <= 1.0
    if nominal != 0.0:                 # a coherent equal-power join bumps, an incoherent Hann join dips
        assert np.sign(summary["centre"]) == np.sign(nominal) and abs(summary["centre"]) >= 1.0


def test_gain_invariance():
    C, H = 4096, 2048
    X = C - H
    x, xs, y = _decodes(0.3, C, H, K=60)
    w = wm.window(C, H, X, "hann")
    starts = wm.join_starts(y.shape[0], C, H, X)
    G1, G2 = wm.atom_gain(y, xs), wm.atom_gain(2 * y, xs)
    assert abs(G2 / G1 - 4.0) <= 1e-9
    l1 = wm.level_db(wm.bin_powers(wm.overlap_add(y, H, w), starts, X), wm.bin_powers(x, starts, X), G1)
    l2 = wm.level_db(wm.bin_powers(wm.overlap_add(2 * y, H, w), starts, X), wm.bin_powers(x, starts, X), G2)
    assert float((torch.as_tensor(l1) - torch.as_tensor(l2)).abs().max()) <= 1e-9


def test_join_frame_mask():
    n, n_fft = 48000, 1024
    hop = n_fft // 4
    m = wm.join_frame_mask(n, n_fft, [(10000, 12000)], (0, n))
    n_frames = 1 + (n - n_fft) // hop
    centres = np.arange(n_frames) * hop + n_fft // 2
    assert m.shape == (n_frames,)
    assert np.array_equal(m, (centres >= 10000) & (centres < 12000))
    assert not wm.join_frame_mask(n, n_fft, [(10000, 12000)], (20000, n)).any()


def test_join_mss():
    x = np.random.RandomState(0).randn(48000 * 3) * 0.1
    intervals, region = [(48000, 50000)], (24000, 120000)
    assert wm.join_mss(x, x.copy(), intervals, region) == 0.0
    y_in = x.copy()
    y_in[48000:50000] *= 0.5
    assert wm.join_mss(x, y_in, intervals, region) > 0.05
    y_out = x.copy()
    y_out[100000:102000] *= 0.5       # far from every masked frame (the largest frame around the join ends near 52048)
    assert abs(wm.join_mss(x, y_out, intervals, region)) <= 1e-9


def test_propose_window_cases():
    r = wm.propose_window({"hann": 2.0, "equal_power": 0.4}, {"hann": 0.5, "equal_power": 0.6},
                          {"hann": 0.02, "equal_power": 0.03})
    assert (r["window"], r["by"], r["h1"]["supported"]) == ("equal_power", "level", True)
    r = wm.propose_window({"hann": 1.0, "equal_power": 0.8}, {"hann": 0.50, "equal_power": 0.60},
                          {"hann": 0.02, "equal_power": 0.03})
    assert (r["window"], r["by"], r["h1"]["supported"]) == ("hann", "mss", False)
    r = wm.propose_window({"hann": 1.0, "equal_power": 0.8}, {"hann": 0.50, "equal_power": 0.51},
                          {"hann": 0.02, "equal_power": 0.03})
    assert r["window"] is None and r["by"] == "no metric preference"
    r = wm.propose_window({"hann": 0.4, "equal_power": 2.0}, {"hann": 0.5, "equal_power": 0.5},
                          {"hann": 0.0, "equal_power": 0.0})
    assert (r["window"], r["by"], r["h1"]["supported"]) == ("hann", "level", False)
    assert abs(r["h1"]["margin_db"] - (0.4 - 2.0)) <= 1e-12
