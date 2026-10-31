"""D8: the one crossfade window of SCAPES (SCAPES/data/geometry.py: crossfade_window), equal power, used by ola_render.
Geometries are duck-typed (atom_samples, hop_samples, crossfade_samples): the window only reads those.
Run from SCAPES/: ../venv_SCAPES/bin/python -m pytest tests_v3/test_window.py -q
"""
import inspect
from types import SimpleNamespace

import pytest
import torch


def geo(n, x, o, h):
    return SimpleNamespace(atom_samples=(o + x + n) * h, hop_samples=n * h, crossfade_samples=x * h)


# (n, x, o, h): toy atoms with x < n, x = n, x = n/2; the encodec48 and music2latent defaults (design: Worked examples)
GEOS = {"toy_n16_x4": (16, 4, 16, 8), "toy_n1_x1": (1, 1, 10, 8), "toy_n8_x4": (8, 4, 8, 8),
        "encodec48_default": (15, 8, 15, 320), "music2latent_default": (1, 1, 4, 4096)}


@pytest.mark.parametrize("name", list(GEOS))
def test_unit_power(name):
    from SCAPES.data.geometry import crossfade_window
    g = geo(*GEOS[name])
    C, H, X = g.atom_samples, g.hop_samples, g.crossfade_samples
    O = C - H - X
    w = crossfade_window(g).double()
    assert w.shape == (C,)
    assert torch.equal(w[:O], torch.zeros(O, dtype=torch.float64))
    assert torch.equal(w[O + X:O + H], torch.ones(H - X, dtype=torch.float64))
    rise, fall = w[O:O + X], w[O + H:]
    hann = torch.hann_window(2 * X, dtype=torch.float64)             # equal power = sqrt of the periodic Hann halves
    assert float((rise - hann[:X].sqrt()).abs().max()) <= 1e-6
    assert float((fall - hann[X:].sqrt()).abs().max()) <= 1e-6
    assert float((rise ** 2 + fall ** 2 - 1).abs().max()) <= 1e-6
    K = 12                                                            # overlap-add of the squared windows = 1 inside
    acc = torch.zeros((K - 1) * H + C, dtype=torch.float64)
    for k in range(K):
        acc[k * H:k * H + C] += w ** 2
    lo, hi = C - H, (K - 1) * H + C - X
    assert float((acc[lo:hi] - 1).abs().max()) <= 1e-6


def test_one_window_function():
    import SCAPES.data.geometry as geometry
    from SCAPES.inference.render import ola_render, render_atoms
    assert callable(geometry.crossfade_window)
    for gone in ("crossfade_halves", "ola_window", "window_for", "WINDOWS"):
        assert not hasattr(geometry, gone), gone
    assert "window" not in inspect.signature(ola_render).parameters
    assert "window" not in inspect.signature(render_atoms).parameters
    assert "change only this function" in (geometry.crossfade_window.__doc__ or "")


def test_ola_render_uses_it():
    from SCAPES.data.geometry import crossfade_window
    from SCAPES.inference.render import ola_render
    g = geo(16, 4, 16, 8)
    C, H = g.atom_samples, g.hop_samples
    gen = torch.Generator().manual_seed(0)
    audios = [torch.randn(2, C, generator=gen) for _ in range(5)]
    w = crossfade_window(g)
    ref = torch.zeros(2, 4 * H + C)
    for k, a in enumerate(audios):
        ref[:, k * H:k * H + C] += a * w
    assert torch.equal(ola_render(audios, g), ref)
