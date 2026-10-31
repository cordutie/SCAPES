"""S5 geometry and S9 keys: consistent and independent atoms in samples (design Formulation, D1-D7; hop_table.html).

The variants are built from explicit gin lines (no template), so these tests do not depend on the templates.
Phase 1 (consistent, 2/4/6/8 latents) runs by default; the independent variants are phase 2 (skipped unless SCAPES_PHASE=2).
The rejection test below necessarily names the removed keys atoms.overlap_mode and atoms.crossfade_law: this file is one
of the exemptions of tests_new/test_old_code_removed.py.

Lives outside tests/ because that folder is locked. Run from SCAPES/: ../venv_SCAPES/bin/python -m pytest tests_new -q
"""
import warnings

import pytest

from _atoms_toy import phase1_only, phase2
from conftest import CONFIG, TOY_COMMON, resolve_lines

R = 44100
N_120 = 5292000
SECONDS = {2: 0.18, 4: 0.37, 6: 0.55, 8: 0.74}

# name -> (codec, type, F, overlap, C, H, X, rate Hz, T_M s (M = 3), zeroed s, K over 120 s)   (design S5 table, plan slice 9)
VARIANTS = {
    "m2l-cons-c2": ("music2latent", "consistent", 2, None, 8192, 4410, 882, 10.00, 0.39, 0.07, 1199),
    "m2l-cons-c4": ("music2latent", "consistent", 4, None, 16384, 4410, 882, 10.00, 0.57, 0.25, 1197),
    "m2l-cons-c6": ("music2latent", "consistent", 6, None, 24576, 4410, 882, 10.00, 0.76, 0.44, 1195),   # phase 1 (design P-6)
    "m2l-cons-c8": ("music2latent", "consistent", 8, None, 32768, 4410, 882, 10.00, 0.94, 0.62, 1193),
    "m2l-ind-c2-normal": ("music2latent", "independent", 2, "normal", 8192, 7168, 1024, 6.15, 0.51, 0.0, 738),
    "m2l-ind-c2-strong": ("music2latent", "independent", 2, "strong", 8192, 6144, 2048, 7.18, 0.46, 0.0, 860),
    "m2l-ind-c2-brute": ("music2latent", "independent", 2, "brute", 8192, 4096, 4096, 10.77, 0.37, 0.0, 1290),
    "m2l-ind-c4-normal": ("music2latent", "independent", 4, "normal", 16384, 14336, 2048, 3.08, 1.02, 0.0, 368),
    "m2l-ind-c4-strong": ("music2latent", "independent", 4, "strong", 16384, 12288, 4096, 3.59, 0.93, 0.0, 430),
    "m2l-ind-c4-brute": ("music2latent", "independent", 4, "brute", 16384, 8192, 8192, 5.38, 0.74, 0.0, 644),
    "m2l-ind-c6-normal": ("music2latent", "independent", 6, "normal", 24576, 21504, 3072, 2.05, 1.53, 0.0, 245),
    "m2l-ind-c6-strong": ("music2latent", "independent", 6, "strong", 24576, 18432, 6144, 2.39, 1.39, 0.0, 286),
    "m2l-ind-c6-brute": ("music2latent", "independent", 6, "brute", 24576, 12288, 12288, 3.59, 1.11, 0.0, 429),
    "cdc-ind-c8-normal": ("codicodec", "independent", 8, "normal", 32768, 28672, 4096, 1.54, 2.04, 0.0, 184),
    "cdc-ind-c8-strong": ("codicodec", "independent", 8, "strong", 32768, 24576, 8192, 1.79, 1.86, 0.0, 214),
    "cdc-ind-c8-brute": ("codicodec", "independent", 8, "brute", 32768, 16384, 16384, 2.69, 1.49, 0.0, 321),
}


CONSISTENT = [n for n, v in VARIANTS.items() if v[1] == "consistent"]
INDEPENDENT = [n for n, v in VARIANTS.items() if v[1] == "independent"]
ALL_PARAMS = CONSISTENT + [pytest.param(n, marks=phase2) for n in INDEPENDENT]


def lines_of(name, window="hann", M=3, extra=()):
    codec, kind, F, ov = VARIANTS[name][:4]
    lines = [f'codec.name = "{codec}"', 'generation.mode = "atoms"', f'atoms.type = "{kind}"',
             f"atoms.seconds = {SECONDS[F]}", f"dataset.memory_buffer_atoms = {M}"]
    lines += ["atoms.hop_seconds = 0.10", "atoms.crossfade_seconds = 0.02"] if kind == "consistent" else [f'atoms.overlap = "{ov}"']
    if window is not None:
        lines.append(f'atoms.window = "{window}"')
    return lines + list(extra) + TOY_COMMON


def geom_of(tmp_path, name, **kw):
    return resolve_lines(tmp_path, lines_of(name, **kw), name + "_" + str(abs(hash(str(kw)))))[1]


def _template(tmp_path, stem, extra=()):
    lines = (CONFIG / "dataprep.gin").read_text().splitlines() + \
        (CONFIG / "codecs" / f"{stem}.gin").read_text().splitlines() + list(extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return resolve_lines(tmp_path, lines, f"t_{stem}_{len(extra)}")[1]


@pytest.mark.parametrize("name", ALL_PARAMS)
def test_hop_table(tmp_path, name):
    codec, kind, F, ov, C, H, X, rate, tm, zeroed, _ = VARIANTS[name]
    g = geom_of(tmp_path, name)
    assert (g.codec, g.atom_type, g.unit_frames, g.atom_samples, g.hop_samples, g.crossfade_samples) == \
        (codec, kind, F, C, H, X), name
    assert g.atom_samples == F * g.hop_length
    assert g.overlap == ov, name
    assert round(R / g.hop_samples, 2) == rate, name
    assert round(g.memory_seconds, 2) == tm, name
    assert round((C - H - X) / R, 2) == zeroed, name
    assert (g.memory_units, g.memory_frames, g.decode) == (3, F, "ola"), name


def _keys_distinct(tmp_path, names):
    from SCAPES.data.geometry import representation_key
    keys = []
    for name in names:
        codec, C, H = VARIANTS[name][0], VARIANTS[name][4], VARIANTS[name][5]
        g = geom_of(tmp_path, name)
        assert representation_key(g) == f"{codec}__atoms-c{C}-h{H}", name
        assert g.encoding == f"atoms-c{C}-h{H}"
        keys.append(representation_key(g))
    assert len(set(keys)) == len(keys)


def test_encoding_keys(tmp_path):
    _keys_distinct(tmp_path, CONSISTENT)


@phase2
def test_encoding_keys_all(tmp_path):
    _keys_distinct(tmp_path, CONSISTENT + INDEPENDENT)


def test_constraints(tmp_path):
    from SCAPES.data.config_loader import ConfigError
    cons = lines_of("m2l-cons-c2")
    bad = {
        "xfade_gt_overlap": cons + ["atoms.crossfade_seconds = 0.09"],          # 3969 > C - H = 3782
        "hop_ge_chunk": cons + ["atoms.hop_seconds = 0.19"],                   # 8379 >= 8192
        "no_xfade": [ln for ln in cons if not ln.startswith("atoms.crossfade_seconds")],
        "bad_window": cons + ['atoms.window = "linear"'],
        "bad_type": [ln if not ln.startswith("atoms.type") else 'atoms.type = "grains"' for ln in cons],
    }
    for case, lines in bad.items():
        with pytest.raises(ConfigError):
            resolve_lines(tmp_path, lines, case)
    resolve_lines(tmp_path, cons, "ok_cons")                                    # the valid neighbour resolves


@phase1_only
def test_independent_is_phase2_in_phase1(tmp_path):
    from SCAPES.data.config_loader import ConfigError
    with pytest.raises(ConfigError, match="phase 2"):
        resolve_lines(tmp_path, lines_of("m2l-ind-c2-normal"), "ind_phase1")


@phase2
def test_constraints_independent(tmp_path):
    from SCAPES.data.config_loader import ConfigError
    from SCAPES.data.geometry import independent_split
    ind = lines_of("m2l-ind-c2-normal")
    for case, lines in {"ind_with_hop": ind + ["atoms.hop_seconds = 0.10"],
                        "ind_bad_overlap": ind + ['atoms.overlap = "soft"']}.items():
        with pytest.raises(ConfigError):
            resolve_lines(tmp_path, lines, case)
    with pytest.raises(ConfigError):
        independent_split(12, "normal")                                      # 12 / 8 = 1.5 is not an integer
    assert independent_split(12, "brute") == (6, 6)
    assert independent_split(8192, "normal") == (7168, 1024)
    resolve_lines(tmp_path, ind, "ok_ind")


def test_rejected_old_keys(tmp_path):
    from SCAPES.data.config_loader import ConfigError
    with pytest.raises(ConfigError, match=r'atoms\.type = "independent"'):
        resolve_lines(tmp_path, lines_of("m2l-cons-c4") + ['atoms.overlap_mode = "strong"'], "old_overlap_mode")
    with pytest.raises(ConfigError, match=r"atoms\.window"):
        resolve_lines(tmp_path, lines_of("m2l-cons-c4") + ['atoms.crossfade_law = "hann"'], "old_crossfade_law")


def test_legacy_rejects_new_keys(tmp_path):
    from SCAPES.data.config_loader import ConfigError
    with pytest.raises(ConfigError):
        _template(tmp_path, "encodec48k_atoms", ['atoms.window = "equal_power"'])
    with pytest.raises(ConfigError):
        _template(tmp_path, "encodec48k_atoms", ['atoms.overlap = "brute"'])


def _round_trip(tmp_path, cases):
    from SCAPES.codecs import get_codec_info
    from SCAPES.data.config_loader import parse_gin_config
    from SCAPES.data.geometry import geometry_from_inference_gin
    for i, g in enumerate(cases):
        path = tmp_path / f"inference_{g.codec}_{g.atom_type}_{i}.gin"
        path.write_text("\n".join(g.to_gin_lines()) + "\n")
        back = geometry_from_inference_gin(parse_gin_config(path), get_codec_info(g.codec))
        assert back == g, (g.codec, g.atom_type)


def test_inference_gin_round_trip(tmp_path):
    _round_trip(tmp_path, [geom_of(tmp_path, "m2l-cons-c2", window="hann"),
                           geom_of(tmp_path, "m2l-cons-c8", window=None),
                           _template(tmp_path, "encodec48k_atoms")])


@phase2
def test_inference_gin_round_trip_independent(tmp_path):
    _round_trip(tmp_path, [geom_of(tmp_path, "m2l-ind-c6-brute", window=None),
                           geom_of(tmp_path, "cdc-ind-c8-normal", window="equal_power")])


def test_tick_properties(tmp_path):
    g = geom_of(tmp_path, "m2l-cons-c4")
    assert (g.tick_samples, g.hop_ticks, g.stride_ticks, g.past_ticks, g.stride_samples) == (4410, 1, 1, 3, 4410)
    assert (g.unit_hop_frames, g.stride_frames) == (0, 0)
    with pytest.raises(ValueError):
        _ = g.past_frames
    e = _template(tmp_path, "encodec48k_atoms")
    assert e.atom_type == "frames"
    assert (e.tick_samples, e.hop_ticks, e.stride_ticks, e.past_ticks) == (320, 15, 15, 45)
    assert (e.atom_samples, e.hop_samples, e.stride_samples) == (15360, 4800, 4800)
    assert e.past_frames == 45


@pytest.mark.parametrize("name", ALL_PARAMS)
def test_atom_count(name):
    from SCAPES.data.geometry import atom_count
    C, H, K = VARIANTS[name][4], VARIANTS[name][5], VARIANTS[name][10]
    assert atom_count(N_120, C, H) == K, name
    assert atom_count(C - 1, C, H) == 0
    assert atom_count(C, C, H) == 1
