"""Overlap-add modes for atoms (normal / strong / brute) and the Music2Latent geometry."""
from pathlib import Path

import pytest
import torch

from SCAPES.codecs import get_codec_info
from SCAPES.data.config_loader import ConfigError, load_dataprep_config
from SCAPES.data.geometry import OVERLAP_MODES, ola_window, resolve_geometry, with_overlap_mode
from SCAPES.inference.render import ola_render

CONFIG = Path(__file__).resolve().parents[1] / "SCAPES" / "config"


def _geom(codec_gin):
    cfg = load_dataprep_config(CONFIG / "codecs" / codec_gin)
    return resolve_geometry(cfg, get_codec_info(cfg.codec_name))


def test_music2latent_geometry():
    g = _geom("music2latent_atoms.gin")
    assert (g.unit_frames, g.unit_hop_frames, g.latent_dim, g.has_scale, g.encoding) == (8, 6, 128, False, "whole")
    assert g.hop_length == 4096 and g.sample_rate == 44100


def test_normal_keeps_the_geometry():
    g = _geom("codicodec_atoms.gin")
    assert with_overlap_mode(g, get_codec_info("codicodec"), "normal") is g


def test_strong_and_brute_fractions():
    g, info = _geom("codicodec_atoms.gin"), get_codec_info("codicodec")
    atom = g.unit_frames * g.hop_length
    strong, brute = with_overlap_mode(g, info, "strong"), with_overlap_mode(g, info, "brute")
    assert strong.crossfade_samples == round(0.25 * atom)
    assert brute.crossfade_samples == round(0.5 * atom)
    for h in (strong, brute):
        w = ola_window(h)
        assert w.shape[0] == atom and float(w.max()) <= 1.0


def test_music2latent_overlaps_only_by_the_crossfade():
    g, info = _geom("music2latent_atoms.gin"), get_codec_info("music2latent")
    assert (g.unit_frames - g.unit_hop_frames) * g.hop_length == with_overlap_mode(g, info, "strong").crossfade_samples
    with pytest.raises(ConfigError, match="does not fit"):
        with_overlap_mode(g, info, "brute")


def test_brute_is_a_plain_hann_overlap_add():
    """With the atom twice the hop and X = half the atom, the window is a full Hann and the overlap-add has constant gain."""
    g = with_overlap_mode(_geom("codicodec_atoms.gin"), get_codec_info("codicodec"), "brute")
    atom = g.unit_frames * g.hop_length
    out = ola_render([torch.ones(2, atom) for _ in range(5)], g)
    steady = out[:, atom // 2: -atom // 2]
    assert torch.allclose(steady, torch.ones_like(steady), atol=1e-5)


def test_crossfade_gain_is_one_for_every_mode():
    g0 = _geom("codicodec_atoms.gin")
    for mode in OVERLAP_MODES:
        g = with_overlap_mode(g0, get_codec_info("codicodec"), mode)
        atom = g.unit_frames * g.hop_length
        out = ola_render([torch.ones(1, atom) for _ in range(4)], g)
        assert torch.allclose(out[:, atom:-atom], torch.ones_like(out[:, atom:-atom]), atol=1e-5), mode


def test_brute_does_not_fit_encodec48k_atoms():
    g = _geom("encodec48k_atoms.gin")     # 0.32 s atoms, 0.1 s hop: the crossfade cannot exceed the hop
    with pytest.raises(ConfigError, match="does not fit"):
        with_overlap_mode(g, get_codec_info("encodec48k"), "brute")
    with_overlap_mode(g, get_codec_info("encodec48k"), "strong")   # 25 % of 0.32 s = 0.08 s fits


def test_invalid_mode_and_ar_geometry():
    g = _geom("codicodec_atoms.gin")
    with pytest.raises(ConfigError, match="invalid"):
        with_overlap_mode(g, get_codec_info("codicodec"), "soft")
    ar = _geom("encodec24k_ar.gin")
    assert with_overlap_mode(ar, get_codec_info("encodec24k"), "normal") is ar
    with pytest.raises(ConfigError, match="atoms mode"):
        with_overlap_mode(ar, get_codec_info("encodec24k"), "strong")
