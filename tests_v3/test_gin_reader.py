"""S5, D3, D6, D7: the unified gin reader, canonical names, geometry from the resolution, inference.gin (V3 only).
Run from SCAPES/: ../venv_SCAPES/bin/python -m pytest tests_v3/test_gin_reader.py -q
"""
import pytest

from v3_helpers import TOY_COMMON, resolve_v3

USER = ["encodec24", "spectrostream", "encodec48", "music2latent"]


def _gin_round_trip(tmp_path, geom, name):
    from SCAPES.codecs import get_codec_info
    from SCAPES.data.config_loader import parse_gin_config
    from SCAPES.data.geometry import geometry_from_inference_gin
    p = tmp_path / f"{name}.gin"
    p.write_text("\n".join(geom.to_gin_lines()) + "\n")
    return geometry_from_inference_gin(parse_gin_config(p), get_codec_info(geom.codec))


@pytest.mark.parametrize("codec", USER)
def test_three_keys_resolve(tmp_path, codec):
    from SCAPES.generation import CODECS, resolve_generation
    _, g = resolve_v3(tmp_path, [f'codec = "{codec}"'] + TOY_COMMON, codec)
    assert g.mode == CODECS[codec].mode
    assert g.codec == codec
    assert g.resolution == resolve_generation(codec)


@pytest.mark.parametrize("order", ["old_first", "new_first"])
def test_rejection_on_written_keys(tmp_path, order):
    from SCAPES.data.config_loader import ConfigError
    lines = ['codec.name = "encodec48k"', 'codec = "encodec48"']
    if order == "new_first":
        lines = lines[::-1]
    with pytest.raises(ConfigError, match=r"codec =|renamed"):     # the rejection of codec.name, not another error
        resolve_v3(tmp_path, lines + TOY_COMMON, order)


OLD = ['codec.name = "encodec48k"', 'generation.mode = "atoms"', "atoms.seconds = 0.32", "atoms.hop_seconds = 0.1",
       "atoms.crossfade_seconds = 0.02", 'atoms.type = "consistent"', 'atoms.window = "hann"', 'atoms.overlap = "brute"',
       "dataset.memory_buffer_atoms = 3", "ar.block_seconds = 0.1", "ar.memory_seconds = 1.6", "ar.prefix_seconds = 0.01"]


@pytest.mark.parametrize("line", OLD)
def test_old_keys_rejected(tmp_path, line):
    from SCAPES.data.config_loader import ConfigError
    with pytest.raises(ConfigError, match=r"conditioning_rate|codec =|follows from the codec"):
        resolve_v3(tmp_path, ['codec = "encodec48"', line] + TOY_COMMON, line.split()[0])


def test_wrapper_name_rejected_in_gin(tmp_path):
    from SCAPES.data.config_loader import ConfigError
    with pytest.raises(ConfigError, match="encodec48"):
        resolve_v3(tmp_path, ['codec = "encodec48k"'] + TOY_COMMON, "wrapper_name")
    with pytest.raises(ConfigError) as e:
        resolve_v3(tmp_path, ['codec = "codicodec"'] + TOY_COMMON, "removed_codec")
    msg = str(e.value)
    for name in USER:
        assert name in msg, name
    assert "toy_" not in msg
    with pytest.raises(ConfigError):
        resolve_v3(tmp_path, TOY_COMMON, "no_codec")


def test_derived_fields(tmp_path):
    from SCAPES.data.geometry import representation_key
    _, g = resolve_v3(tmp_path, ['codec = "encodec48"', "conditioning_rate = 12", "conditioning_memory_seconds = 0.5"]
                      + TOY_COMMON, "e48")
    assert (g.mode, g.unit_frames, g.hop_samples, g.crossfade_samples, g.memory_units, g.atom_samples) == \
        ("atoms", 33, 4160, 2240, 6, 10560)
    assert not hasattr(g, "window")                     # one window function, no per-geometry window (D8)
    assert representation_key(g) == "encodec48__atoms-c10560-h4160"
    assert g.hop_samples % g.hop_length == 0
    _, a = resolve_v3(tmp_path, ['codec = "encodec24"'] + TOY_COMMON, "e24")
    assert (a.mode, a.unit_frames, a.unit_hop_frames, a.stride_frames, a.memory_frames, a.prefix_frames, a.decode) == \
        ("autoregressive", 8, 8, 8, 75, 0, "stream")


CASES = [(c, []) for c in USER] + [("encodec48", ["conditioning_rate = 12", "conditioning_memory_seconds = 0.5"]),
                                    ("music2latent", ["conditioning_rate = 3", "conditioning_memory_seconds = 1"])]


@pytest.mark.parametrize("codec,extra", CASES, ids=[f"{c}-{len(e)}" for c, e in CASES])
def test_inference_gin_round_trip(tmp_path, codec, extra):
    _, g = resolve_v3(tmp_path, [f'codec = "{codec}"'] + extra + TOY_COMMON, f"rt_{codec}_{len(extra)}")
    back = _gin_round_trip(tmp_path, g, f"inference_{codec}_{len(extra)}")
    assert back == g
    lines = g.to_gin_lines()
    assert any(ln.startswith("resolved.new_latents") for ln in lines)
    assert any(ln.startswith("conditioning_rate") for ln in lines)


def test_inference_gin_table_guard(tmp_path):
    from SCAPES.codecs import get_codec_info
    from SCAPES.data.config_loader import ConfigError, parse_gin_config
    from SCAPES.data.geometry import geometry_from_inference_gin
    _, g = resolve_v3(tmp_path, ['codec = "encodec48"'] + TOY_COMMON, "guard")
    lines = []
    for ln in g.to_gin_lines():
        if ln.startswith("resolved.new_latents"):
            key, value = ln.split("=", 1)
            ln = f"{key}= {int(value.split('#')[0]) + 1}"
        lines.append(ln)
    p = tmp_path / "edited.gin"
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(ConfigError, match="codec table"):
        geometry_from_inference_gin(parse_gin_config(p), get_codec_info("encodec48"))


OLD_INFERENCE = {  # written inline: no file of models/ is read (D7, no backward compatibility)
    "v1": ["model.d_model = 1024", "atoms.frames = 48", "atoms.hop_frames = 15", "atoms.crossfade_frames = 3",
           "dataset.memory_buffer_atoms = 1", "dataset.context_seconds = 1.0"],
    "phase1": ['codec.name = "music2latent"', 'generation.mode = "atoms"', 'atoms.type = "consistent"', "atoms.frames = 4",
               "atoms.chunk_samples = 16384", "atoms.hop_samples = 4410", "atoms.crossfade_samples = 3782",
               "dataset.memory_buffer_atoms = 3", "dataset.hop_atoms = 1", "dataset.context_seconds = 1.0"],
}


@pytest.mark.parametrize("kind", list(OLD_INFERENCE))
def test_old_inference_gin_rejected(tmp_path, kind):
    from SCAPES.codecs import get_codec_info
    from SCAPES.data.config_loader import ConfigError, parse_gin_config
    from SCAPES.data.geometry import geometry_from_inference_gin
    p = tmp_path / "inference.gin"
    p.write_text("\n".join(OLD_INFERENCE[kind]) + "\n")
    with pytest.raises(ConfigError, match=r"conditioning_rate|codec =|follows from the codec"):
        geometry_from_inference_gin(parse_gin_config(p), get_codec_info("music2latent"))


def test_buffer_span_renamed(tmp_path):
    from SCAPES.data.geometry import Geometry
    _, g = resolve_v3(tmp_path, ['codec = "music2latent"'] + TOY_COMMON, "span")
    assert not hasattr(Geometry, "memory_seconds")
    C, H, M = g.atom_samples, g.hop_samples, g.memory_units
    assert (C, H, M) == (24576, 4096, 5)
    assert abs(g.buffer_span_seconds - (C + (M - 1) * H) / g.sample_rate) <= 1e-12
    assert g.resolution.covered_latents == 6 and g.resolution.buffer_span_latents == 10
