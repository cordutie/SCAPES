"""Atoms in samples (V3): hop counts, tick properties and the inference.gin round trip on the toy geometries.

The resolution of the three keys (table, rules, rejections) is tested in tests_v3/. Lives outside tests/ because that
folder is locked. Run from SCAPES/: ../venv_SCAPES/bin/python -m pytest tests_new -q
"""
import pytest

from _atoms_toy import TOY_GEOM, toy_geometry
from conftest import TOY_AR, TOY_COMMON, resolve_lines

# (C, H, K over 120 s at 44.1 kHz): music2latent consistent atoms of 2, 4, 6 and 8 latents, hop 0.1 s
COUNTS = {2: (8192, 4410, 1199), 4: (16384, 4410, 1197), 6: (24576, 4410, 1195), 8: (32768, 4410, 1193)}
N_120 = 5292000


@pytest.mark.parametrize("n_latents", sorted(COUNTS))
def test_atom_count(n_latents):
    from SCAPES.data.geometry import atom_count
    C, H, K = COUNTS[n_latents]
    assert atom_count(N_120, C, H) == K
    assert atom_count(C - 1, C, H) == 0
    assert atom_count(C, C, H) == 1


def test_encoding_key(tmp_path):
    from SCAPES.data.geometry import representation_key
    g = toy_geometry(tmp_path)
    C, H, _ = TOY_GEOM["consistent"]
    assert representation_key(g) == f"toy_noncausal__atoms-c{C}-h{H}" and g.encoding == f"atoms-c{C}-h{H}"


def test_tick_properties(tmp_path):
    g = toy_geometry(tmp_path)
    C, H, X = TOY_GEOM["consistent"]
    assert (g.atom_samples, g.hop_samples, g.crossfade_samples) == (C, H, X)
    assert (g.tick_samples, g.hop_ticks, g.stride_ticks, g.past_ticks, g.stride_samples) == (H, 1, 1, 3, H)
    assert (g.unit_hop_frames, g.stride_frames) == (0, 0)
    with pytest.raises(ValueError):
        _ = g.past_frames
    a = resolve_lines(tmp_path, TOY_AR + TOY_COMMON, "toy_ar")[1]
    assert a.atom_type == "frames"
    assert (a.tick_samples, a.hop_ticks, a.stride_ticks, a.past_ticks) == (8, 32, 32, 96)
    assert a.past_frames == 96


def test_inference_gin_round_trip(tmp_path):
    from SCAPES.codecs import get_codec_info
    from SCAPES.data.config_loader import parse_gin_config
    from SCAPES.data.geometry import geometry_from_inference_gin
    cases = [toy_geometry(tmp_path), resolve_lines(tmp_path, TOY_AR + TOY_COMMON, "toy_ar")[1]]
    for i, g in enumerate(cases):
        path = tmp_path / f"inference_{g.codec}_{i}.gin"
        path.write_text("\n".join(g.to_gin_lines()) + "\n")
        back = geometry_from_inference_gin(parse_gin_config(path), get_codec_info(g.codec))
        assert back == g, g.codec
