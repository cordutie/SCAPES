"""Toy CPU tests of the generation side of scripts/teacher_forced_eval.py (random-init models, no GPU).

Run from SCAPES/: ../venv_SCAPES/bin/python -m pytest scripts/tests_eval -q
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import teacher_forced_eval as tf


def engine_of(codec, geometry=None):
    from SCAPES.inference.FlowInference import FlowInference
    e = FlowInference.from_config(codec, size="small", geometry=geometry, device="cpu")
    e.model.eval()
    return e


def test_unit_starts_atoms_and_ar():
    e = engine_of("encodec48k")
    g = e.geom
    starts = tf.unit_starts(g, 10.0)
    sr = g.sample_rate
    assert starts == list(range(len(starts)))        # atoms: the unit is one hop tick
    assert starts[-1] * g.hop_samples + g.atom_samples <= 10.0 * sr < (starts[-1] + 1) * g.hop_samples + g.atom_samples
    ar = engine_of("encodec24k")
    s2 = tf.unit_starts(ar.geom, 10.0)
    assert s2[-1] + ar.geom.unit_frames <= 10.0 * ar.geom.frame_rate
    assert s2[-1] + 2 * ar.geom.unit_frames > 10.0 * ar.geom.frame_rate


def test_real_units_atoms_use_stored_past_and_clap():
    g = engine_of("encodec48k").geom
    K = 40
    lat = {"latent": torch.randn(K, g.latent_dim, g.unit_frames).half(), "scale": torch.rand(K)}
    clap = torch.randn(K, 1024)
    starts = tf.unit_starts(g, 4.0)
    units, claps = tf.real_units(g, lat, clap, starts)
    k = 7
    assert units[k].shape == (1, g.frame_dim, g.unit_frames)
    assert torch.equal(units[k][0, :g.latent_dim], lat["latent"][k].float())
    assert torch.allclose(units[k][0, g.latent_dim], lat["scale"][k].expand(g.unit_frames))
    assert torch.equal(claps[k], clap[k])


def test_real_units_ar_blocks_are_consecutive_slices():
    g = engine_of("encodec24k").geom
    T = 1000
    lat = {"latent": torch.randn(g.latent_dim, T)}
    clap = torch.randn(T // g.stride_frames + 1, 1024)
    starts = tf.unit_starts(g, 3.0)
    units, claps = tf.real_units(g, lat, clap, starts)
    assert torch.equal(torch.cat(units, dim=-1)[0], lat["latent"][:, :len(units) * g.unit_frames])
    assert torch.equal(claps[2], clap[2])


def test_teacher_forced_atoms_generation_and_roundtrip():
    e = engine_of("encodec48k")
    g = e.geom
    K = 6
    lat = {"latent": torch.randn(K, g.latent_dim, g.unit_frames).half() * 0.5, "scale": torch.rand(K) + 0.5}
    units, claps = tf.real_units(g, lat, torch.randn(K, 1024), list(range(K)))
    a1 = tf.generate_teacher_forced(e, units, claps, nfe=2, cfg=3.0, seed=0)
    a2 = tf.generate_teacher_forced(e, units, claps, nfe=2, cfg=3.0, seed=0)
    assert torch.equal(a1, a2)                         # fixed seed: reproducible
    assert a1.shape[-1] == (K - 1) * g.hop_samples + g.atom_samples
    assert all(t["TF"] for t in e.timeline)
    # the past of step t is the stored unit, not the generated one
    assert torch.equal(e._unit_at(e.timeline, 2), units[2].to(e.device))
    rt = tf.codec_roundtrip(e, units)
    assert rt.shape == a1.shape
    # silent head by design: nothing before the first atom's active part
    head = g.atom_samples - g.hop_samples - g.crossfade_samples   # C - H - X samples are zeroed by the window
    assert head > 0 and a1[..., :head].abs().max() == 0


def test_teacher_forced_ar_generation_and_roundtrip():
    e = engine_of("encodec24k")
    g = e.geom
    T = g.unit_frames * 4
    units, claps = tf.real_units(g, {"latent": torch.randn(g.latent_dim, T) * 0.5}, torch.randn(8, 1024),
                                 tf.unit_starts(g, T / g.frame_rate))
    a = tf.generate_teacher_forced(e, units, claps, nfe=2, cfg=3.0, seed=0)
    assert a.shape[-1] == len(units) * g.unit_frames * g.hop_length
    rt = tf.codec_roundtrip(e, units)
    assert rt.shape == a.shape
