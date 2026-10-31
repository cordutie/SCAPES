"""teacher_forced_eval.py for sample-level atoms: unit ticks, real units, the duration check (S6, S7 tool).
Geometry only (codec info, no codec load), music2latent at the table defaults.
Run from SCAPES/: ../venv_SCAPES/bin/python -m pytest scripts/tests_eval -q
"""
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[2] / "tests_new"))
import teacher_forced_eval as tf

R = 44100


def _geom(tmp_path, name="music2latent"):
    """V3 geometry of music2latent at the table defaults: only the codec line (geometry only, no codec load)."""
    from SCAPES.codecs import get_codec_info
    from SCAPES.data.config_loader import load_dataprep_config
    from SCAPES.data.geometry import resolve_geometry
    p = tmp_path / f"{name}.gin"
    p.write_text("\n".join([f'codec = "{name}"', "semantic.context_seconds = 1.0", "semantic.random_extension = False",
                            "structure.features = []", "splits.val_split_ratio = 0.1", "splits.reserve_ratio = 0.1",
                            "dataset.hop_atoms = 1", "seed = 0"]) + "\n")
    g = resolve_geometry(load_dataprep_config(p), get_codec_info(name))
    assert g.atom_type == "consistent"
    return g


def _k10(g):
    return (10 * R - g.atom_samples) // g.hop_samples + 1


def test_unit_starts(tmp_path):
    g = _geom(tmp_path)
    C, H, K10 = g.atom_samples, g.hop_samples, _k10(g)
    assert tf.unit_starts(g, 10.0) == list(range(K10))
    late = tf.unit_starts(g, 10.0, start_seconds=108.0)
    assert late[0] == math.ceil(108 * R / H - 1e-9)          
    assert late == list(range(late[0], late[-1] + 1))
    assert late[-1] * H + C <= 118 * R < (late[-1] + 1) * H + C


def test_real_units(tmp_path):
    g = _geom(tmp_path)
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


def test_duration_check(tmp_path):
    g = _geom(tmp_path)
    C, H, K10 = g.atom_samples, g.hop_samples, _k10(g)
    good = (K10 - 1) * H + C
    r = tf.duration_check(g, K10, good, 10.0)
    assert r["ok"] and r["expected"] == good and r["shortfall_samples"] == 10 * R - good
    assert not tf.duration_check(g, K10, good - 1, 10.0)["ok"]
    assert not tf.duration_check(g, K10, good, 11.0)["ok"]   # shortfall >= H: atoms are missing
