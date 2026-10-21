"""Known-answer tests of scripts/eval_metrics.py (run from SCAPES/: ../venv_SCAPES/bin/python -m pytest scripts/tests_eval -q)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_metrics as em


def noise(n=200000, seed=0):
    return np.random.RandomState(seed).randn(n) * 0.1


def test_mss_identical_is_zero():
    x = noise()
    r = em.mss(x, x.copy())
    assert r["mean_log_l1"] == 0.0 and r["mean_spec_conv"] == 0.0


def test_mss_scaling_by_two_shifts_log_by_ln2():
    x = noise()
    r = em.mss(x, 2.0 * x, eps=1e-12)
    for v in r["scales"].values():
        assert v["log_l1"] == pytest.approx(np.log(2.0), abs=1e-6)
        assert v["spec_conv"] == pytest.approx(1.0, abs=1e-9)   # ||S - 2S|| / ||S|| = 1


def test_mss_independent_noise_known_value():
    x = noise(seed=0)
    assert em.mss(x, noise(seed=1))["mean_log_l1"] > 0.1
    # magnitudes of independent noise are Rayleigh: E|a-b|^2 / E a^2 = 2 var / E a^2 = 2 - pi/2
    assert em.mss(x, noise(seed=1))["mean_spec_conv"] == pytest.approx(np.sqrt(2.0 - np.pi / 2.0), abs=0.03)


def test_mss_too_short_raises():
    with pytest.raises(ValueError):
        em.mss(noise(1000), noise(1000))


def test_white_noise_like_matches_rms():
    x = np.sin(np.linspace(0, 400, 100000)) * 0.3
    n = em.white_noise_like(x, 0)
    assert np.sqrt(np.mean(n ** 2)) == pytest.approx(np.sqrt(np.mean(x ** 2)), rel=0.02)


def test_estimate_lag():
    x = noise(50000)
    y = np.concatenate([np.zeros(123), x])[:50000]
    assert em.estimate_lag(x, y) == 123
    assert em.estimate_lag(y, x) == -123
    assert em.estimate_lag(x, x) == 0


def test_to_mono_and_resample():
    st = np.stack([np.ones(10), 3 * np.ones(10)])
    assert np.allclose(em.to_mono(st), 2.0)
    t = np.sin(2 * np.pi * 440 * np.arange(24000) / 24000)[None]
    r = em.resample_to_48k(t, 24000)
    assert r.shape[-1] == 48000


def test_kad_identical_matches_closed_form():
    # X == Y: the unbiased estimate is (2/n)(mean off-diagonal k - 1) * 100, small and negative
    x = np.random.RandomState(0).randn(400, 8)
    sigma = em.median_pairwise_distance(x)
    k = em.rbf_gram(x, x, sigma)
    n = len(x)
    mean_off = (k.sum() - n) / (n * (n - 1))
    assert em.kad(x, x.copy()) == pytest.approx(100 * 2 / n * (mean_off - 1), rel=1e-9)
    assert abs(em.kad(x, x.copy())) < 0.5


def test_kad_same_distribution_is_near_zero_on_average():
    vals = []
    for s in range(30):
        r = np.random.RandomState(s)
        vals.append(em.kad(r.randn(60, 8), r.randn(60, 8), sigma=4.0))
    assert abs(np.mean(vals)) < 0.1
    assert min(vals) < 0 < max(vals)      # unbiased: it can be negative


def test_kad_monotone_in_shift():
    r = np.random.RandomState(0)
    x = r.randn(300, 8)
    vals = [em.kad(x, r.randn(300, 8) + shift, sigma=4.0) for shift in (0.0, 0.5, 1.0, 2.0, 4.0)]
    assert vals[0] < vals[1] < vals[2] < vals[3]
    assert vals[1] > 0
    assert vals[4] > vals[2]


def test_kad_bootstrap_shapes_and_paired_difference():
    r = np.random.RandomState(0)
    ref = r.randn(16, 8)
    evals = {"near": ref + 0.05 * r.randn(16, 8), "far": ref + 3.0, "same": ref.copy()}
    out = em.kad_with_bootstrap(ref, evals, np.arange(16), n_boot=50, seed=0, baseline="near")
    assert out["sets"]["far"]["kad"] > out["sets"]["near"]["kad"]
    assert out["sets"]["far"]["minus_near"]["contains_zero"] is False
    assert len(out["sets"]["far"]["ci95"]) == 2
    # segment level: 5 items per clip, same clips drawn for all items of a clip
    ref5 = r.randn(80, 8)
    out5 = em.kad_with_bootstrap(ref5, {"a": ref5 + 1.0}, np.repeat(np.arange(16), 5), n_boot=20)
    assert out5["sets"]["a"]["kad"] > 0
