"""Slice B9b: training.val_every_steps (validate every N steps, patience counted in validations).

Default None keeps the V1 behaviour: one validation per epoch, patience in epochs.
Lives outside tests/ because that folder is locked. Run from SCAPES/: ../venv_SCAPES/bin/python -m pytest tests_new -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from conftest import prepare_toy

NO_REG = ["training.past_dropout = 0.0", "training.conditioning_dropout = 0.0", "training.past_noise = 0.0"]


def _build(root, extra_training=(), resume=None):
    from SCAPES.pipeline import build_trainer
    prepare_toy(root, "autoregressive", [], list(NO_REG) + list(extra_training))
    return build_trainer(root, root / "model", config_dir=root / "config", resume=resume)


# val_epoch that returns the given losses in order (the last one repeats); records the global step of every call
def _fake_val(trainer, losses):
    calls = []

    def val_epoch():
        calls.append(trainer.global_step)
        i = min(len(calls) - 1, len(losses) - 1)
        return losses[i], losses[i], 0.0

    trainer.val_epoch = val_epoch
    return calls


def _train(trainer, epochs, patience=10):
    trainer.train(epochs=epochs, audio_val_freq=10 ** 9, val_nfe=4, patience=patience)


def test_default_validates_once_per_epoch(tmp_path):
    trainer = _build(tmp_path / "toy")
    assert trainer.val_every_steps is None
    trainer.train_loader = list(trainer.train_loader)[:4]
    calls = _fake_val(trainer, [1.0])
    _train(trainer, epochs=3)
    assert calls == [4, 8, 12]


def test_validates_every_n_steps_across_epochs(tmp_path):
    trainer = _build(tmp_path / "toy", ["training.val_every_steps = 5"])
    trainer.train_loader = list(trainer.train_loader)[:4]   # epochs of 4 steps: 5 does not divide 4
    calls = _fake_val(trainer, [1.0, 0.9, 0.8, 0.7])
    _train(trainer, epochs=6)   # 24 steps
    assert calls == [5, 10, 15, 20]
    assert len(trainer.val_losses["total"]) == 4


def test_patience_counts_validations(tmp_path):
    trainer = _build(tmp_path / "toy", ["training.val_every_steps = 3"])
    trainer.train_loader = list(trainer.train_loader)[:2]
    # best at the 1st validation, then 4 validations without improvement = patience 4
    calls = _fake_val(trainer, [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    _train(trainer, epochs=100, patience=4)
    assert calls == [3, 6, 9, 12, 15]
    assert (trainer.ckpt_dir / "best_flow_model.pt").exists()
    assert (trainer.ckpt_dir / "last_flow_model.pt").exists()
    assert not list(trainer.ckpt_dir.glob("epoch_*"))   # only best and last


def test_improvement_resets_patience(tmp_path):
    trainer = _build(tmp_path / "toy", ["training.val_every_steps = 3"])
    trainer.train_loader = list(trainer.train_loader)[:2]
    calls = _fake_val(trainer, [1.0, 2.0, 2.0, 0.5, 2.0, 2.0, 2.0])
    _train(trainer, epochs=100, patience=3)
    assert len(calls) == 7   # best at the 4th, then 3 without improvement


def test_resume_keeps_patience_counter(tmp_path):
    root = tmp_path / "toy"
    extra = ["training.val_every_steps = 3", "training.save_resume_states = True"]
    trainer = _build(root, extra)
    trainer.train_loader = list(trainer.train_loader)[:4]
    _fake_val(trainer, [1.0, 2.0, 2.0])   # best at validation 1, then 2 without improvement
    _train(trainer, epochs=3)             # 12 steps: validations at 3, 6, 9, 12
    n_val = len(trainer.val_losses["total"])
    assert n_val == 4

    resumed = _build(root, extra, resume="latest")
    assert resumed.resume_no_improve == n_val - 1   # best was the first validation
    assert resumed.global_step == trainer.global_step
    assert resumed.val_every_steps == 3
