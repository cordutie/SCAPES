"""Training, inference.gin, generation and validation audio for the atoms of the V3 geometry (toy, GPU).

test_overfit_one_batch_new_type replaces the plan's weaker "loss decreases" check: it is the locked one-batch overfit
protocol of tests/test_trainer.py (1000 steps on one fixed batch, last-20 mean <= 0.35 x first-5 mean; the 0.35 bound was
calibrated there, probe P-12 of the earlier project), applied to the new atom geometry.

Lives outside tests/ because that folder is locked. Run from SCAPES/: ../venv_SCAPES/bin/python -m pytest tests_new -q
"""
import json
import math

import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Subset

from _atoms_toy import RATE, TOY_GEOM, prepare_new_toy, toy_geometry

KINDS = ["consistent"]
NO_REG = ["training.past_dropout = 0.0", "training.conditioning_dropout = 0.0", "training.past_noise = 0.0"]


def _geom(config_dir):
    from SCAPES.codecs import get_codec_info
    from SCAPES.data.config_loader import load_dataprep_config
    from SCAPES.data.geometry import resolve_geometry
    cfg = load_dataprep_config(config_dir / "dataprep.gin")
    return resolve_geometry(cfg, get_codec_info(cfg.codec_name))


@pytest.mark.parametrize("kind", KINDS)
def test_train_and_generate(tmp_path, kind):
    from SCAPES.data.config_loader import parse_gin_config
    from SCAPES.inference.FlowInference import FlowInference
    from SCAPES.pipeline import run_training
    root = tmp_path / "toy"
    cfg_dir, _ = prepare_new_toy(root, kind, extra_training=["training.log_every = 1"])
    model_dir = run_training(root, root / "model", config_dir=cfg_dir, max_steps=5)
    hist = json.loads((model_dir / "logs" / "history_steps.json").read_text())
    assert hist and all(math.isfinite(h["loss"]) for h in hist)
    C, H, X = TOY_GEOM[kind]
    gin = parse_gin_config(model_dir / "checkpoints" / "inference.gin")
    res = gin["resolved"]
    assert (res["atom_latents"], res["new_latents"], res["crossfade_latents"]) == (C // 8, H // 8, X // 8)
    engine = FlowInference(model_dir, device="cuda", checkpoint="last")
    assert engine.geom == _geom(cfg_dir)
    g = torch.Generator().manual_seed(0)
    clap = [torch.nn.functional.normalize(torch.randn(1024, generator=g), dim=0) for _ in range(5)]
    audio = engine.generate_soundscape(clap, nfe=2, cfg_scale=1.0, seed=0)
    assert torch.isfinite(audio).all()
    assert audio.shape[-1] == 4 * H + C


def test_overfit_one_batch_new_type(tmp_path):
    from SCAPES.pipeline import build_trainer
    root = tmp_path / "toy"
    steps = 1000
    cfg_dir, _ = prepare_new_toy(root, "consistent",
                                 extra_training=NO_REG + [f"training.max_steps = {steps}", "training.log_every = 1"])
    trainer = build_trainer(root, root / "model", config_dir=cfg_dir)
    g = trainer.dataset.geom                      # the run must really use the sample-level geometry, not a legacy one
    assert (g.atom_type, g.atom_samples, g.hop_samples, g.crossfade_samples) == ("consistent",) + TOY_GEOM["consistent"]
    train, _ = trainer.dataset.get_splits()
    batch = next(iter(DataLoader(Subset(train, list(range(8))), batch_size=8, shuffle=False)))
    trainer.train_loader = [batch] * steps
    trainer.val_loader = None
    trainer.train(epochs=1, audio_val_freq=10 ** 9, val_nfe=4, patience=10)
    losses = [h["loss"] for h in json.loads((trainer.model_path / "logs" / "history_steps.json").read_text())]
    assert len(losses) == steps
    first, last = sum(losses[:5]) / 5, sum(losses[-20:]) / 20
    assert last <= 0.35 * first, f"{first:.3f} -> {last:.3f}"


@pytest.mark.parametrize("kind", ["consistent"])
def test_units_for_seconds_new_types(tmp_path, kind):
    from SCAPES.pipeline import units_for_seconds
    g = toy_geometry(tmp_path, kind)
    C, H, _ = TOY_GEOM[kind]
    for s in (0.0, 0.1, 1.0):
        K = units_for_seconds(g, s)
        assert K >= 1 and (K - 1) * H + C >= s * RATE - 1e-9
        assert K == 1 or (K - 2) * H + C < s * RATE, (kind, s, K)


def test_validation_audio_new_type(tmp_path):
    from SCAPES.pipeline import build_trainer
    root = tmp_path / "toy"
    cfg_dir, _ = prepare_new_toy(root, "consistent",
                                 extra_training=['training.val_files = ["a.wav"]', "training.val_duration = 0.1"])
    trainer = build_trainer(root, root / "model", config_dir=cfg_dir)
    trainer.generate_validation_audio(epoch=1, NFE=2)
    C, H, _ = TOY_GEOM["consistent"]
    n = max(1, int(0.1 * RATE // H))
    y, sr = sf.read(str(trainer.val_dir / "epoch_1_a_TF.wav"), always_2d=True)
    assert sr == RATE
    assert y.shape[0] == (n - 1) * H + C
