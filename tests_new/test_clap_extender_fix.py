"""S1: the CLAP audio_extender first junction crossfades two pieces of written audio (design D19, plan Tests).

Before the fix, the first segment advanced `pos` by its whole length, so the first crossfade blended the new segment's head
with silence (a fade-in from 0). After the fix `pos = use_length - overlap`, so the written region always ends at
pos + overlap and every junction blends written audio.

Lives outside tests/ because that folder is locked. Run from SCAPES/: ../venv_SCAPES/bin/python -m pytest tests_new -q
"""
import threading

import numpy as np
import pytest
import torch

from SCAPES.auxiliar.clap_wrapper import audio_extender

SR, DUR, RATIO = 1000, 1.0, 0.1          # expected length 1000, overlap 100
O = 100


def _extend(x, random_extension=False):
    return audio_extender(x, random_extension=random_extension, sample_rate=SR, duration=DUR, overlap_ratio=RATIO)


@pytest.mark.parametrize("random_extension", [False, True])
def test_constant_input_stays_constant(random_extension):
    # linear fades sum to 1, so a constant input must stay constant everywhere (the old code ramps 0 -> 1 on [300, 400))
    np.random.seed(0)
    y = _extend(torch.ones(1, 1, 300), random_extension)
    assert y.shape == (1, 1, 1000)
    assert float((y - 1.0).abs().max()) <= 1e-6


def test_periodic_structure():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(300, generator=g) * 2.0 - 1.0
    x[50] = 1.0                        # convex crossfades -> the peak is exactly this sample, y needs no rescaling
    y = _extend(x.view(1, 1, -1), False)[0, 0]
    fade_in, fade_out = torch.linspace(0, 1, O), torch.linspace(1, 0, O)
    junction = x[200:300] * fade_out + x[0:100] * fade_in
    assert float((y[0:200] - x[0:200]).abs().max()) <= 1e-6, "first segment, copied"
    assert float((y[200:300] - junction).abs().max()) <= 1e-6, "first junction must blend written audio"
    assert float((y[300:400] - x[100:200]).abs().max()) <= 1e-6, "second segment, copied"
    assert float((y[400:500] - junction).abs().max()) <= 1e-6, "second junction"


def test_random_segments_longer_than_overlap():
    """Guard min_length = max(T // 2, overlap + 1) (design D19). Without it, random mode on short rows draws segments no
    longer than the overlap: they stall the loop for an iteration and re-crossfade the same region (probe 2026-10-30: the
    loop always ends, there is no hang). For T = 2 (overlap 1) the guard leaves only the whole row as a segment, so random
    mode must equal the non-random extension for every seed; the old code differs on 6 of these 10 seeds."""
    x = torch.tensor([[[1.0, -0.4]]])
    ref = _extend(x, False)
    for seed in range(10):
        np.random.seed(seed)
        y = _extend(x, True)
        assert float((y - ref).abs().max()) <= 1e-6, f"seed {seed}"


@pytest.mark.parametrize("T", [1, 3, 10, 201])
def test_random_mode_short_input_returns(T):
    """Regression guard: random mode on rows of at most 2 * overlap + 1 samples returns quickly, with shape, finite values and
    peak 1 (it already does before the guard)."""
    g = torch.Generator().manual_seed(T)
    x = torch.rand(1, 1, T, generator=g) * 2.0 - 1.0
    x[0, 0, 0] = 1.0
    for seed in range(5):
        out = {}

        def run():
            np.random.seed(seed)
            out["y"] = _extend(x, True)

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(5.0)
        assert not t.is_alive(), f"audio_extender did not return (T={T}, seed {seed})"
        y = out["y"]
        assert y.shape == (1, 1, 1000) and torch.isfinite(y).all()
        assert abs(float(y.abs().max()) - 1.0) <= 1e-6


def test_one_sample_input():
    out = {}

    def run():
        out["y"] = _extend(torch.tensor([[[0.7]]]), False)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(5.0)
    assert not t.is_alive(), "audio_extender did not return within 5 s for a 1-sample input"
    y = out["y"]
    assert y.shape == (1, 1, 1000)
    assert float((y - 1.0).abs().max()) <= 1e-6
