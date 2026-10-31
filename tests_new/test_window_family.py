"""S4 windows and S2: Hann and equal power on the unified window layout [0_{C-H-X}, rise, 1_{H-X}, fall] (design D3, D4).

Lives outside tests/ because that folder is locked. Run from SCAPES/: ../venv_SCAPES/bin/python -m pytest tests_new -q
"""
import math
import warnings

import pytest
import torch

from _atoms_toy import TOY_GEOM, phase2, toy_geometry
from conftest import CONFIG, TOY_COMMON, resolve_lines

KINDS = ["consistent", pytest.param("brute", marks=phase2)]   # C = 128 with (H, X) = (36, 16) and (64, 64)


def _ola(w: torch.Tensor, hop: int, k: int) -> torch.Tensor:
    out = torch.zeros((k - 1) * hop + w.shape[0], dtype=torch.float64)
    for i in range(k):
        out[i * hop: i * hop + w.shape[0]] += w.double()
    return out


def _template(tmp_path, stem, extra=()):
    lines = (CONFIG / "dataprep.gin").read_text().splitlines() + \
        (CONFIG / "codecs" / f"{stem}.gin").read_text().splitlines() + list(extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return resolve_lines(tmp_path, lines, f"t_{stem}")[1]


def test_halves():
    from SCAPES.data.config_loader import ConfigError
    from SCAPES.data.geometry import crossfade_halves
    for n in (16, 882):
        hann = torch.hann_window(2 * n)
        rise, fall = crossfade_halves(n, "hann")
        assert torch.equal(rise, hann[:n]) and torch.equal(fall, hann[n:])
        r2, f2 = crossfade_halves(n, "equal_power")
        assert float((r2 - hann[:n].sqrt()).abs().max()) <= 1e-7
        assert float((f2 - hann[n:].sqrt()).abs().max()) <= 1e-7
    with pytest.raises(ConfigError):
        crossfade_halves(16, "linear")


@pytest.mark.parametrize("kind", KINDS)
def test_partition_hann(tmp_path, kind):
    from SCAPES.inference.render import ola_render
    g = toy_geometry(tmp_path, kind)
    C, H, X = TOY_GEOM[kind]
    assert (g.atom_samples, g.hop_samples, g.crossfade_samples) == (C, H, X)
    K = 12
    out = ola_render([torch.ones(1, C) for _ in range(K)], g, window="hann")
    assert out.shape == (1, (K - 1) * H + C)
    interior = out[0, C - H:(K - 1) * H + C - X]
    assert interior.numel() > 0
    assert float((interior - 1).abs().max()) <= 1e-6


@pytest.mark.parametrize("kind", KINDS)
def test_equal_power_powers_add(tmp_path, kind):
    from SCAPES.data.geometry import ola_window
    g = toy_geometry(tmp_path, kind)
    C, H, X = TOY_GEOM[kind]
    K = 12
    w = ola_window(g, "equal_power")
    power = _ola(w ** 2, H, K)[C - H:(K - 1) * H + C - X]
    assert float((power - 1).abs().max()) <= 1e-6
    amp = _ola(w, H, K)
    for k in range(1, K):
        centre = k * H + C - H - X + X // 2          # u = 1/2 of join k: sin(pi/4) + cos(pi/4)
        assert abs(float(amp[centre]) - math.sqrt(2.0)) <= 1e-3


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", ["hann", "equal_power"])
def test_window_layout(tmp_path, kind, name):
    from SCAPES.data.geometry import crossfade_halves, ola_window
    g = toy_geometry(tmp_path, kind)
    C, H, X = TOY_GEOM[kind]
    w = ola_window(g, name)
    rise, fall = crossfade_halves(X, name)
    assert w.shape == (C,)
    assert torch.equal(w[:C - H - X], torch.zeros(C - H - X))
    assert torch.equal(w[C - H - X:C - H], rise)
    assert torch.equal(w[C - H:C - X], torch.ones(H - X))
    assert torch.equal(w[C - X:], fall)


@pytest.mark.parametrize("kind", KINDS)
def test_render_placement(tmp_path, kind):
    from SCAPES.data.geometry import ola_window
    from SCAPES.inference.render import ola_render
    g = toy_geometry(tmp_path, kind)
    C, H, X = TOY_GEOM[kind]
    K, j = 5, C - X - 3
    atoms = []
    for _ in range(K):
        a = torch.zeros(1, C)
        a[0, j] = 1.0
        atoms.append(a)
    out = ola_render(atoms, g, window="equal_power")[0]
    w = ola_window(g, "equal_power")
    assert out.shape == ((K - 1) * H + C,)
    want = torch.zeros_like(out)
    for k in range(K):
        want[k * H + j] += w[j]
    assert float((out - want).abs().max()) <= 1e-7


def test_window_for(tmp_path):
    from SCAPES.codecs import get_codec_info
    from SCAPES.data.geometry import ola_window, window_for
    g = _template(tmp_path, "encodec48k_atoms")
    info = get_codec_info("encodec48k")
    assert window_for(g, info) == "hann"
    F, H, Xf, spf = 48, 15, 3, 320            # V1 FlowInference._build_ola_window
    hann = torch.hann_window(Xf * spf * 2)
    v1 = torch.cat([torch.zeros((F - H - Xf) * spf), hann[: Xf * spf], torch.ones((H - Xf) * spf), hann[Xf * spf:]])
    assert torch.equal(ola_window(g, window_for(g, info)).float().cpu(), v1)
    gc = _template(tmp_path, "codicodec_atoms")
    assert window_for(gc, get_codec_info("codicodec")) == "equal_power"     # today's by-codec default, kept
    m2l = get_codec_info("music2latent")
    base = ['codec.name = "music2latent"', 'generation.mode = "atoms"', 'atoms.type = "consistent"', "atoms.seconds = 0.37",
            "atoms.hop_seconds = 0.10", "atoms.crossfade_seconds = 0.02", "dataset.memory_buffer_atoms = 3"] + TOY_COMMON
    gh = resolve_lines(tmp_path, base + ['atoms.window = "hann"'], "m2l_hann")[1]
    assert gh.window == "hann" and window_for(gh, m2l) == "hann"
    gd = resolve_lines(tmp_path, base, "m2l_default")[1]
    assert window_for(gd, m2l) == "equal_power"
