"""teacher_forced_eval.py for sample-level atoms: unit ticks, real units, the duration check (S6, S7 tool).
Geometry only (codec info, no codec load). Phase 1: m2l-cons-c4; phase 2 (SCAPES_PHASE=2): m2l-ind-c4-strong as well.
Run from SCAPES/: ../venv_SCAPES/bin/python -m pytest scripts/tests_eval -q
"""
import math
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[2] / "tests_new"))
import teacher_forced_eval as tf
from _atoms_toy import phase2

R = 44100
GEOMS = {   # name -> (geometry lines, C, H, K over 10 s)
    "m2l-cons-c4": (['atoms.type = "consistent"', "atoms.seconds = 0.37", "atoms.hop_seconds = 0.10",
                     "atoms.crossfade_seconds = 0.02"], 16384, 4410, 97),
    "m2l-ind-c4-strong": (['atoms.type = "independent"', "atoms.seconds = 0.37", 'atoms.overlap = "strong"'], 16384, 12288, 35),
}
NAMES = ["m2l-cons-c4", pytest.param("m2l-ind-c4-strong", marks=phase2)]


def _geom(tmp_path, name):
    from SCAPES.codecs import get_codec_info
    from SCAPES.data.config_loader import load_dataprep_config
    from SCAPES.data.geometry import resolve_geometry
    lines, C, H, _ = GEOMS[name]
    p = tmp_path / f"{name}.gin"
    p.write_text("\n".join(['codec.name = "music2latent"', 'generation.mode = "atoms"', 'atoms.window = "hann"',
                            "dataset.memory_buffer_atoms = 3", "semantic.context_seconds = 1.0",
                            "semantic.random_extension = False", "structure.features = []", "splits.val_split_ratio = 0.1",
                            "splits.reserve_ratio = 0.1", "dataset.hop_atoms = 1", "seed = 0"] + lines) + "\n")
    g = resolve_geometry(load_dataprep_config(p), get_codec_info("music2latent"))
    assert (g.atom_samples, g.hop_samples) == (C, H)
    return g


@pytest.mark.parametrize("name", NAMES)
def test_unit_starts_new_types(tmp_path, name):
    g = _geom(tmp_path, name)
    _, C, H, K10 = GEOMS[name]
    assert tf.unit_starts(g, 10.0) == list(range(K10))
    late = tf.unit_starts(g, 10.0, start_seconds=108.0)
    assert late[0] == math.ceil(108 * R / H - 1e-9)          # 1080 for the 0.1 s hop, 388 for H = 12288
    assert late == list(range(late[0], late[-1] + 1))
    assert late[-1] * H + C <= 118 * R < (late[-1] + 1) * H + C


@pytest.mark.parametrize("name", NAMES)
def test_real_units_new_type(tmp_path, name):
    g = _geom(tmp_path, name)
    K = 40
    latents = {"latent": torch.arange(K * 128 * 4, dtype=torch.float32).view(K, 128, 4).half(), "scale": torch.ones(K)}
    clap = torch.randn(K, 1024)
    starts = [0, 5, 34]
    units, claps = tf.real_units(g, latents, clap, starts)
    assert len(units) == len(claps) == 3
    for u, c, k in zip(units, claps, starts):
        assert u.shape == (1, 128, 4)
        assert torch.equal(u, latents["latent"][k].float().unsqueeze(0))
        assert torch.equal(c, clap[k].float())


@pytest.mark.parametrize("name", NAMES)
def test_duration_check(tmp_path, name):
    g = _geom(tmp_path, name)
    _, C, H, K10 = GEOMS[name]
    good = (K10 - 1) * H + C                                   # 439744 (consistent c4), 434176 (independent c4 strong)
    r = tf.duration_check(g, K10, good, 10.0)
    assert r["ok"] and r["expected"] == good and r["shortfall_samples"] == 10 * R - good
    assert not tf.duration_check(g, K10, good - 1, 10.0)["ok"]
    assert not tf.duration_check(g, K10, good, 11.0)["ok"]   # shortfall >= H: atoms are missing
