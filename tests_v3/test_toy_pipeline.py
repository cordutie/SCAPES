"""S4, S6 on toys: the startup print, dataset items from the resolution, representations keyed by the geometry,
training defaults from the codec table, a few training steps and a generation (toy codecs, stub CLAP, GPU).
Run from SCAPES/: ../venv_SCAPES/bin/python -m pytest tests_v3/test_toy_pipeline.py -q
"""
import json
import math

import numpy as np
import pytest
import torch

from v3_helpers import (LOG_EVERY_STEP, TOY_AR_V3, TOY_ATOMS_SNAP, TOY_ATOMS_V3, TOY_ATOMS_XN, load_dataset_v3,
                      make_toy_v3, prepare_toy_v3)

KINDS = {"ar": TOY_AR_V3, "atoms": TOY_ATOMS_V3, "atoms_xn": TOY_ATOMS_XN, "atoms_snap": TOY_ATOMS_SNAP}


def _geometry_lines(text):
    lines = text.splitlines()
    return [ln for ln in lines if ln.startswith("[SCAPES] toy_")], [ln for ln in lines if ln.startswith("[SCAPES] warning:")]


def test_training_defaults_from_table(tmp_path):
    from SCAPES.generation import CODECS
    from SCAPES.pipeline import build_trainer
    root = tmp_path / "toy"
    cfg_dir, _ = prepare_toy_v3(root, TOY_ATOMS_V3)
    trainer = build_trainer(root, root / "model", config_dir=cfg_dir)
    spec = CODECS["toy_noncausal"]
    assert trainer.standardize == spec.standardize_latents and trainer.past_noise == spec.past_noise
    cfg2 = make_toy_v3(root, TOY_ATOMS_V3, extra_training=["training.past_noise = 0.2"], config_name="config2",
                       write_wavs=False)
    trainer2 = build_trainer(root, root / "model2", config_dir=cfg2)
    assert trainer2.past_noise == 0.2


@pytest.mark.parametrize("kind", ["ar", "atoms_snap"])
def test_startup_print_once(tmp_path, capsys, kind):
    from SCAPES.inference.FlowInference import FlowInference
    from SCAPES.pipeline import build_trainer, run_training
    root = tmp_path / "toy"
    capsys.readouterr()
    cfg_dir, store = prepare_toy_v3(root, KINDS[kind], extra_training=LOG_EVERY_STEP)
    notes = store.geom.resolution.notes
    geo, warn = _geometry_lines(capsys.readouterr().out)
    assert len(geo) == 1 and len(warn) == len(notes), ("run_dataprep", geo, warn)
    if kind == "atoms_snap":
        assert notes == ("snapped",)
    build_trainer(root, root / "m0", config_dir=cfg_dir)
    geo, warn = _geometry_lines(capsys.readouterr().out)
    assert len(geo) == 1 and len(warn) == len(notes), ("build_trainer", geo, warn)
    model_dir = run_training(root, root / "model", config_dir=cfg_dir, max_steps=1)
    capsys.readouterr()
    FlowInference(model_dir, device="cuda", checkpoint="last")
    geo, warn = _geometry_lines(capsys.readouterr().out)
    assert len(geo) == 1 and len(warn) == len(notes), ("FlowInference", geo, warn)


def _reference_indices(ds, res):
    """All (file, tick) of the dataset, written from the resolved counts (design: Dataset indexing; V2 frames for AR)."""
    from SCAPES.data.splits import assign_region, region_bounds
    g = ds.geom
    r, h = g.sample_rate, g.hop_length
    ctx48 = int(round(g.context_seconds * 48000))
    out = []
    for fname in ds.filenames:
        e = ds._entries[fname]
        n48, dur = e["num_samples_48k"], e["duration_s"]
        bounds = region_bounds(dur, ds.config.val_split_ratio, ds.config.reserve_split_ratio)
        if res.mode == "atoms":
            H, C, M, K = res.n * h, res.atom * h, res.memory_atoms, e["num_atoms"]
            gi = 0
            while int(np.rint(gi * H * 48000 / r)) + ctx48 <= n48:
                k = gi
                gi += 1
                if k < M:
                    continue
                if k > K - 1:
                    break
                span = ((k - M) * H / r, max((k * H + C) / r, k * H / r + g.context_seconds))
                if assign_region(span, bounds) is not None:
                    out.append((fname, k))
        else:
            B, W, T, f = res.n, res.memory_window, e["num_frames"], r / h
            gi = 0
            while int(np.rint(gi * B * h * 48000 / r)) + ctx48 <= n48:
                t = gi * B
                gi += 1
                if t < W:
                    continue
                if t + B > T:
                    break
                span = ((t - W) / f, max((t + B) / f, t / f + g.context_seconds))
                if assign_region(span, bounds) is not None:
                    out.append((fname, t))
    return out


@pytest.mark.parametrize("kind", ["ar", "atoms"])
def test_dataset_items_follow_resolution(tmp_path, kind):
    root = tmp_path / "toy"
    cfg_dir, store = prepare_toy_v3(root, KINDS[kind])
    ds = load_dataset_v3(root, cfg_dir)
    res = ds.geom.resolution
    d = ds.geom.latent_dim
    assert [(f, int(t)) for f, t in ds.all_indices] == _reference_indices(ds, res)
    item = ds[len(ds) // 2]
    if kind == "atoms":
        assert tuple(item["memory_buffer_latent"].shape) == (res.memory_atoms, d, res.atom)
        assert tuple(item["target_latent"].shape) == (d, res.atom)
    else:
        assert tuple(item["memory_buffer_latent"].shape) == (1, d, res.memory_window)
        assert tuple(item["target_latent"].shape) == (d, res.n)


def test_representations_keyed(tmp_path):
    root = tmp_path / "toy"
    prepare_toy_v3(root, TOY_ATOMS_V3, config_name="c16")                                       # n 16: A 36
    prepare_toy_v3(root, ['codec = "toy_noncausal"', "conditioning_rate = 125", "conditioning_memory_seconds = 0.05"],
                   config_name="c8", write_wavs=False)                                         # n 8: A 20
    reps = root / "representations"
    a, b = reps / "toy_noncausal__atoms-c288-h128", reps / "toy_noncausal__atoms-c160-h64"
    assert a.is_dir() and b.is_dir()
    before = {p: p.stat().st_mtime_ns for p in a.rglob("*") if p.is_file()}
    prepare_toy_v3(root, TOY_ATOMS_V3, config_name="c16", write_wavs=False)
    assert {p: p.stat().st_mtime_ns for p in a.rglob("*") if p.is_file()} == before
    n_folders = len([p for p in reps.iterdir() if p.is_dir()])
    prepare_toy_v3(root, ['codec = "toy_noncausal"', "conditioning_rate = 62.5", "conditioning_memory_seconds = 0.1"],
                   config_name="c16_mem", write_wavs=False)                                    # same n, other memory
    assert len([p for p in reps.iterdir() if p.is_dir()]) == n_folders


@pytest.mark.parametrize("kind", ["ar", "atoms", "atoms_xn"])
def test_train_and_generate(tmp_path, kind):
    from SCAPES.data.config_loader import parse_gin_config
    from SCAPES.inference.FlowInference import FlowInference
    from SCAPES.pipeline import run_training
    root = tmp_path / "toy"
    cfg_dir, store = prepare_toy_v3(root, KINDS[kind], extra_training=LOG_EVERY_STEP)
    model_dir = run_training(root, root / "model", config_dir=cfg_dir, max_steps=3)
    hist = json.loads((model_dir / "logs" / "history_steps.json").read_text())
    assert hist and all(math.isfinite(h["loss"]) for h in hist)
    gin = parse_gin_config(model_dir / "checkpoints" / "inference.gin")
    assert "resolved" in gin and gin["resolved"]["new_latents"] == store.geom.resolution.n
    engine = FlowInference(model_dir, device="cuda", checkpoint="last")
    assert engine.geom == store.geom
    g = torch.Generator().manual_seed(0)
    clap = [torch.nn.functional.normalize(torch.randn(1024, generator=g), dim=0) for _ in range(4)]
    audio = engine.generate_soundscape(clap, nfe=2, cfg_scale=1.0, seed=0)
    assert torch.isfinite(audio).all()
    geom = engine.geom
    want = 3 * geom.hop_samples + geom.atom_samples if geom.mode == "atoms" else 4 * geom.unit_frames * geom.hop_length
    assert audio.shape[-1] == want
