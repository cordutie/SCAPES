"""Slice B9c: last_* only after a validation (val_every_steps mode) and the non-finite step guard.

Lives outside tests/ because that folder is locked. Run from SCAPES/: ../venv_SCAPES/bin/python -m pytest tests_new -q
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from conftest import prepare_toy

NO_REG = ["training.past_dropout = 0.0", "training.conditioning_dropout = 0.0", "training.past_noise = 0.0"]


def _build(root, extra=(), resume=None):
    from SCAPES.pipeline import build_trainer
    prepare_toy(root, "autoregressive", [], NO_REG + list(extra))
    return build_trainer(root, root / "model", config_dir=root / "config", resume=resume)


def _train(trainer, epochs, patience=10):
    trainer.train(epochs=epochs, audio_val_freq=10 ** 9, val_nfe=4, patience=patience)


def _short(trainer, n):
    trainer.train_loader = list(trainer.train_loader)[:n]


def _mtime(trainer, name):
    f = trainer.ckpt_dir / name
    return f.stat().st_mtime_ns if f.exists() else None


def test_last_only_in_validation_epochs(tmp_path):
    trainer = _build(tmp_path / "toy", ["training.val_every_steps = 6", "training.save_resume_states = True"])
    _short(trainer, 4)   # epochs of 4 steps: validations at 6 (epoch 2), 12 (epoch 3), 18 (epoch 5)
    saved = []
    orig = trainer._epoch_bookkeeping

    def spy(global_epoch, audio_val_freq, val_nfe, save_last=True):
        orig(global_epoch, audio_val_freq, val_nfe, save_last=save_last)
        saved.append((global_epoch, save_last, _mtime(trainer, "last_flow_model.pt"),
                      _mtime(trainer, "last_trainer_state.pt")))

    trainer._epoch_bookkeeping = spy
    _train(trainer, epochs=5)
    flags = [s[1] for s in saved]
    assert flags == [False, True, True, False, True]
    assert saved[0][2] is None and saved[0][3] is None            # nothing written in epoch 1
    assert saved[3][2] == saved[2][2] and saved[3][3] == saved[2][3]   # epoch 4 left the files alone
    assert saved[4][2] > saved[3][2]
    # the state on disk resumes: it holds the step and the validations of epoch 5
    resumed = _build(tmp_path / "toy", ["training.val_every_steps = 6", "training.save_resume_states = True"], resume="latest")
    assert resumed.global_step == 20 and len(resumed.val_losses["total"]) == 3


def test_default_mode_saves_last_every_epoch(tmp_path):
    trainer = _build(tmp_path / "toy", ["training.save_resume_states = True"])
    _short(trainer, 3)
    flags = []
    orig = trainer._epoch_bookkeeping
    trainer._epoch_bookkeeping = lambda *a, save_last=True: (flags.append(save_last), orig(*a, save_last=save_last))
    _train(trainer, epochs=3)
    assert flags == [True, True, True]


def _nan_loss(monkeypatch, bad_calls):
    import SCAPES.training.FlowModel_trainer as ft
    orig = ft.flow_matching_loss
    state = {"n": 0}

    def fake(*a, **k):
        state["n"] += 1
        out = orig(*a, **k)
        if state["n"] in bad_calls:
            return (out[0] * float("nan"),) + tuple(out[1:])
        return out

    monkeypatch.setattr(ft, "flow_matching_loss", fake)


def test_isolated_nonfinite_steps_are_skipped(tmp_path, monkeypatch, capsys):
    trainer = _build(tmp_path / "toy")
    _short(trainer, 6)
    _nan_loss(monkeypatch, {2, 4})
    before = [p.detach().clone() for p in trainer.model.parameters()]
    _train(trainer, epochs=1)
    out = capsys.readouterr().out
    assert out.count("non-finite") == 2
    assert trainer.nonfinite_steps == 2
    assert trainer.global_step == 4   # 6 batches, 2 skipped
    after = list(trainer.model.parameters())
    assert all(torch.isfinite(p).all() for p in after)
    assert any(not torch.equal(a, b) for a, b in zip(after, before))   # the good steps trained
    assert all(torch.isfinite(torch.tensor(v)) for v in trainer.train_losses["total"])


def test_consecutive_nonfinite_aborts(tmp_path, monkeypatch):
    trainer = _build(tmp_path / "toy", ["training.val_every_steps = 3"])
    _short(trainer, 20)
    _nan_loss(monkeypatch, set(range(3, 100)))
    with pytest.raises(RuntimeError, match="non-finite"):
        _train(trainer, epochs=1)
    assert trainer.nonfinite_steps == 5
    assert trainer.global_step == 2
    assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
