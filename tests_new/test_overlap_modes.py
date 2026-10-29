"""Overlap modes (normal / strong / brute) of the stochastic decoders and the crossfade laws (Hann, equal power)."""
import tempfile
from pathlib import Path

import pytest
import torch

from SCAPES.codecs import get_codec_info
from SCAPES.data.config_loader import ConfigError, load_dataprep_config
from SCAPES.data.geometry import crossfade_law_for, geometry_from_inference_gin, ola_window, resolve_geometry
from SCAPES.inference.render import ola_render
from SCAPES.data.config_loader import parse_gin_config

CONFIG = Path(__file__).resolve().parents[1] / "SCAPES" / "config"


def _geom(codec_gin, **overrides):
    lines = [l for l in (CONFIG / "codecs" / codec_gin).read_text().splitlines()
             if not any(l.startswith(k + " ") for k in overrides)]
    lines += [f"{k} = {v}" for k, v in overrides.items() if v is not None]    # None removes the key
    path = Path(tempfile.mkdtemp()) / "a.gin"
    path.write_text("\n".join(lines) + "\n")
    cfg = load_dataprep_config(path)
    return resolve_geometry(cfg, get_codec_info(cfg.codec_name))


def _fh(g):
    return g.unit_frames, g.unit_hop_frames, g.crossfade_samples


# Music2Latent: 4096-sample frames, quantum 1. Atom of 4 frames, 8 frames.
@pytest.mark.parametrize("seconds,mode,expected", [
    (0.37, "normal", (4, 3, 2205)),     # overlap 1 frame, short fade
    (0.37, "strong", (4, 3, 4096)),     # overlap 1 frame = 25 %, whole overlap crossfaded
    (0.37, "brute", (4, 2, 8192)),      # overlap 2 frames = 50 %
    (0.74, "normal", (8, 7, 2205)),
    (0.74, "strong", (8, 6, 8192)),
    (0.74, "brute", (8, 4, 16384)),
])
def test_music2latent_hop_follows_the_mode(seconds, mode, expected):
    g = _geom("music2latent_atoms.gin", **{"atoms.seconds": seconds, "atoms.overlap_mode": f'"{mode}"'})
    assert _fh(g) == expected and g.overlap_mode == mode


def test_music2latent_template():
    g = _geom("music2latent_atoms.gin")
    assert (g.unit_frames, g.unit_hop_frames, g.latent_dim, g.has_scale, g.encoding) == (4, 3, 128, False, "whole")


# CoDiCodec: grains. One atom (whole 0.743 s chunks) at a time, no shared latent frames, overlapped in the audio only.
def _grains(seconds, mode):
    return _geom("codicodec_grains.gin", **{"atoms.seconds": seconds, "atoms.overlap_mode": f'"{mode}"'})


@pytest.mark.parametrize("mode,overlap", [("normal", 2205), ("strong", 8192), ("brute", 16384)])
def test_codicodec_grains(mode, overlap):
    g = _grains(0.74, mode)
    assert (g.unit_frames, g.unit_hop_frames, g.crossfade_samples, g.grain_overlap_samples) == (8, 8, 0, overlap)
    assert g.audio_hop_samples == 8 * 4096 - overlap
    assert g.stride_frames == 8                      # training windows and the CLAP grid do not overlap
    assert g.overlap_mode == mode


def test_codicodec_two_chunk_grains():
    g = _grains(1.48, "strong")
    assert (g.unit_frames, g.grain_overlap_samples) == (16, round(0.25 * 16 * 4096))


def test_trained_codicodec_template_is_unchanged():
    g = _geom("codicodec_atoms.gin")
    assert (g.unit_frames, g.unit_hop_frames, g.grain_overlap_samples, g.overlap_mode) == (16, 8, 0, None)


def test_grain_output_length_and_units():
    from SCAPES.pipeline import units_for_seconds
    g = _grains(0.74, "brute")
    n = g.unit_frames * g.hop_length
    out = ola_render([torch.ones(1, n) for _ in range(5)], g, "equal_power")
    assert out.shape[-1] == 4 * g.audio_hop_samples + n
    k = units_for_seconds(g, 10.0)
    assert (k - 1) * g.audio_hop_samples + n >= 10.0 * g.sample_rate > (k - 2) * g.audio_hop_samples + n


def test_overlap_mode_is_refused_for_deterministic_decoders():
    lines = (CONFIG / "codecs" / "encodec48k_atoms.gin").read_text().splitlines() + ['atoms.overlap_mode = "strong"']
    path = Path(tempfile.mkdtemp()) / "a.gin"
    path.write_text("\n".join(lines))
    with pytest.raises(ConfigError, match="stochastic decoder"):
        resolve_geometry(load_dataprep_config(path), get_codec_info("encodec48k"))


def test_invalid_mode_and_conflicting_keys():
    with pytest.raises(ConfigError, match="invalid"):
        _geom("music2latent_atoms.gin", **{"atoms.overlap_mode": '"soft"'})
    with pytest.raises(ConfigError, match="derived"):
        _geom("music2latent_atoms.gin", **{"atoms.hop_seconds": 0.2})
    with pytest.raises(ConfigError, match="whole overlap"):
        _geom("music2latent_atoms.gin", **{"atoms.crossfade_seconds": 0.01})
    with pytest.raises(ConfigError, match="longer"):
        _geom("music2latent_atoms.gin", **{"atoms.seconds": 0.05, "atoms.overlap_mode": '"brute"'})


def test_law_by_codec_and_override():
    m2l, enc = get_codec_info("music2latent"), get_codec_info("encodec48k")
    assert crossfade_law_for(_geom("music2latent_atoms.gin"), m2l) == "equal_power"
    assert crossfade_law_for(_geom("encodec48k_atoms.gin"), enc) == "hann"
    assert crossfade_law_for(_geom("music2latent_atoms.gin", **{"atoms.crossfade_law": '"hann"'}), m2l) == "hann"
    assert crossfade_law_for(_geom("encodec48k_atoms.gin", **{"atoms.crossfade_law": '"equal_power"'}), enc) == "equal_power"
    with pytest.raises(ConfigError, match="invalid"):
        _geom("music2latent_atoms.gin", **{"atoms.crossfade_law": '"linear"'})


def test_geometry_round_trips_through_inference_gin(tmp_path):
    g = _geom("music2latent_atoms.gin", **{"atoms.overlap_mode": '"brute"', "atoms.crossfade_law": '"hann"'})
    path = tmp_path / "inference.gin"
    path.write_text("\n".join(g.to_gin_lines()) + "\n")
    back = geometry_from_inference_gin(parse_gin_config(path), get_codec_info("music2latent"))
    assert (_fh(back), back.overlap_mode, back.crossfade_law) == (_fh(g), "brute", "hann")


@pytest.mark.parametrize("mode", ["normal", "strong", "brute"])
def test_equal_power_window(mode):
    g = _geom("music2latent_atoms.gin", **{"atoms.seconds": 0.74, "atoms.overlap_mode": f'"{mode}"'})
    w = ola_window(g, "equal_power")
    assert w.shape[0] == g.unit_frames * g.hop_length
    H = g.unit_hop_frames * g.hop_length
    total = torch.zeros(w.shape[0] + 6 * H, dtype=torch.float64)
    for k in range(7):
        total[k * H:k * H + w.shape[0]] += w.double() ** 2
    interior = total[w.shape[0]:6 * H]
    assert float((interior - 1).abs().max()) <= 1e-5          # powers add up to 1
    h = ola_window(g, "hann").double()
    assert torch.allclose(w.double() ** 2, h, atol=1e-6)      # equal power is the square root of Hann


@pytest.mark.parametrize("mode", ["normal", "strong", "brute"])
def test_equal_power_grain_window(mode):
    g = _grains(0.74, mode)
    w = ola_window(g, "equal_power").double()
    n, step = w.shape[0], g.audio_hop_samples
    assert n == g.unit_frames * g.hop_length
    total = torch.zeros(n + 6 * step, dtype=torch.float64)
    for k in range(7):
        total[k * step:k * step + n] += w ** 2
    interior = total[n:6 * step]
    assert float((interior - 1).abs().max()) <= 1e-5


def test_incoherent_audio_keeps_its_power_under_equal_power_but_dips_under_hann():
    g = _geom("music2latent_atoms.gin", **{"atoms.seconds": 0.74, "atoms.overlap_mode": '"brute"'})
    n = g.unit_frames * g.hop_length
    torch.manual_seed(0)
    atoms = [torch.randn(1, n) for _ in range(40)]
    H = g.unit_hop_frames * g.hop_length

    def power_at_joins(law):
        y = ola_render(atoms, g, law)
        # the middle of each overlap region (where both atoms have weight 0.5 under Hann)
        mids = [k * H + n - g.crossfade_samples // 2 for k in range(1, 38)]
        return torch.stack([y[0, m - 200:m + 200].pow(2).mean() for m in mids]).mean().item()

    assert abs(power_at_joins("equal_power") - 1.0) < 0.15
    assert power_at_joins("hann") < 0.65                       # about -3 dB at the middle of the join
