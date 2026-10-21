"""Slice B4: trainer, run logging, past dropout/noise, deterministic validation, prefix fallback (design Model, D7, D17).

Interfaces used (plan "Interfaces", Phase B):
- pipeline.build_trainer(dataset_path, save_path, config_dir=None) -> FlowTrainer (run_training = build + train);
- FlowTrainer.train(epochs, audio_val_freq, val_nfe, patience) (V1 signature), FlowTrainer.val_epoch() -> (total, ...);
- FlowModel_trainer calls flow_matching_loss through its module-level name (spied here);
- FlowModel_trainer.past_dropout_mask(batch_size, M, p, device="cpu") -> bool [B, M] (atoms);
- FlowModel_trainer.past_dropout_frame_mask(batch_size, W, B, p, device="cpu") -> bool [batch, W] (AR, one window);
- FlowModel_trainer.apply_past_noise(past, gamma) -> tensor (not in place);
- logs in <save_path>/logs/: history_steps.json, val_per_file.json, peak_memory.json, env.json.
"""
import json

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from conftest import load_toy_dataset, make_toy_dataset, max_abs, prepare_toy, stub_clap

NO_REG = ["training.past_dropout = 0.0", "training.conditioning_dropout = 0.0", "training.past_noise = 0.0"]
OVERFIT_STEPS = 1000
OVERFIT_RATIO = 0.35  # probe P-12: informative conditioning 0.21-0.25, conditioning removed 0.56-0.58 (3 seeds)


def _build(root, mode="autoregressive", extra_dataprep=(), extra_training=()):
    from SCAPES.pipeline import build_trainer
    prepare_toy(root, mode, extra_dataprep, extra_training)
    return build_trainer(root, root / "model", config_dir=root / "config")


def _fixed_batch(trainer, n=8):
    train, _ = trainer.dataset.get_splits()
    return next(iter(DataLoader(Subset(train, list(range(n))), batch_size=n, shuffle=False)))


def _history(trainer):
    return json.loads((trainer.model_path / "logs" / "history_steps.json").read_text())


def _overfit(tmp_path, mode, extra_dataprep=()):
    extra = NO_REG + [f"training.max_steps = {OVERFIT_STEPS}", "training.log_every = 1"]
    trainer = _build(tmp_path / "toy", mode, extra_dataprep, extra)
    batch = _fixed_batch(trainer)
    trainer.train_loader = [batch] * OVERFIT_STEPS
    trainer.val_loader = None
    trainer.train(epochs=1, audio_val_freq=10 ** 9, val_nfe=4, patience=10)
    losses = [h["loss"] for h in _history(trainer)]
    assert len(losses) == OVERFIT_STEPS
    first, last = sum(losses[:5]) / 5, sum(losses[-20:]) / 20
    assert last <= OVERFIT_RATIO * first, f"{mode}: {first:.3f} -> {last:.3f}"


def test_overfit_ar_toy(tmp_path):
    _overfit(tmp_path, "autoregressive")


def test_overfit_atoms_toy(tmp_path):
    _overfit(tmp_path, "atoms")


def test_overfit_ar_prefix_toy(tmp_path):
    _overfit(tmp_path, "autoregressive", ["ar.prefix_seconds = 0.002"])


def test_max_steps_and_logs(tmp_path):
    trainer = _build(tmp_path / "toy", extra_training=["training.max_steps = 37", "training.log_every = 10"])
    assert len(trainer.train_loader) > 37  # the stop happens mid-epoch
    calls = {"n": 0}
    orig = trainer.optimizer.step

    def counting_step(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    trainer.optimizer.step = counting_step
    trainer.train(epochs=1000, audio_val_freq=10 ** 9, val_nfe=4, patience=10)
    assert calls["n"] == 37
    hist = _history(trainer)
    assert [h["step"] for h in hist] == [10, 20, 30]
    for h in hist:
        assert set(h) >= {"step", "loss", "loss_latent", "loss_scale", "grad_norm", "lr", "time"}
    logs = trainer.model_path / "logs"
    peak = json.loads((logs / "peak_memory.json").read_text())
    assert peak["max_memory_allocated_bytes"] > 0
    env = json.loads((logs / "env.json").read_text())
    assert set(env) >= {"torch", "cuda", "cudnn", "jax", "numpy", "gpu", "driver", "git", "requirements_lock"}
    for repo in ("SCAPES", "audio_codec_wrappers"):
        assert set(env["git"][repo]) >= {"head", "dirty"}


def test_past_dropout_prefix():
    from SCAPES.training.FlowModel_trainer import past_dropout_mask
    torch.manual_seed(0)
    m = past_dropout_mask(10000, 3, 0.25)
    assert m.shape == (10000, 3) and m.dtype == torch.bool
    rows = m.any(dim=1)
    assert abs(rows.float().mean().item() - 0.25) <= 0.02
    j = m.sum(dim=1)
    for b in torch.nonzero(rows).flatten().tolist():
        k = int(j[b])
        assert m[b, :k].all() and not m[b, k:].any()  # oldest units first, a prefix from index 0
    counts = torch.bincount(j[rows], minlength=4)[1:].float() / rows.sum()
    assert float((counts - 1 / 3).abs().max()) <= 0.05  # j ~ U{1..M}


def test_past_dropout_frame_mask():
    """AR (one window): masks the oldest W - k*B frames, k ~ U{0..ceil(W/B)-1}, i.e. exactly the histories that exist at
    generation start (design Model, Esteban review)."""
    from SCAPES.training.FlowModel_trainer import past_dropout_frame_mask
    for W, B, lengths in ((96, 32, {96, 64, 32}), (80, 32, {80, 48, 16})):
        torch.manual_seed(0)
        m = past_dropout_frame_mask(10000, W, B, 0.25)
        assert m.shape == (10000, W) and m.dtype == torch.bool
        rows = m.any(dim=1)
        assert abs(rows.float().mean().item() - 0.25) <= 0.02
        n = m.sum(dim=1)
        for b in torch.nonzero(rows).flatten().tolist():
            k = int(n[b])
            assert m[b, :k].all() and not m[b, k:].any()  # a prefix from the oldest frame
        got = set(int(v) for v in n[rows].tolist())
        assert got == lengths, (W, B, got)
        for L in lengths:
            frac = float((n[rows] == L).float().mean())
            assert abs(frac - 1 / len(lengths)) <= 0.05


def test_past_noise():
    from SCAPES.training.FlowModel_trainer import apply_past_noise
    torch.manual_seed(0)
    past = torch.zeros(2000, 3, 16, 32)
    assert torch.equal(apply_past_noise(past, 0.0), past)
    out = apply_past_noise(past, 0.1)
    assert torch.count_nonzero(past) == 0  # not in place
    per_sample = (out - past).flatten(1).std(dim=1)
    assert float(per_sample.max()) <= 0.1 * 1.1
    assert 0.04 <= float(per_sample.mean()) <= 0.06  # alpha ~ U[0, 0.1]: E[alpha] = 0.05


def test_inference_gin_contents(tmp_path):
    from SCAPES.data.config_loader import parse_gin_config
    from SCAPES.pipeline import run_training
    root = tmp_path / "toy"
    prepare_toy(root, "autoregressive")
    model_dir = run_training(root, root / "model", config_dir=root / "config", max_steps=5)
    gin = parse_gin_config(model_dir / "checkpoints" / "inference.gin")
    assert gin["codec"]["name"] == "toy_causal"
    assert gin["generation"]["mode"] == "autoregressive"
    assert (gin["ar"]["block_frames"], gin["ar"]["memory_frames"], gin["ar"]["stride_frames"],
            gin["ar"]["prefix_frames"]) == (32, 96, 16, 0)
    assert "memory_buffer_atoms" not in gin.get("dataset", {})  # AR has one window, no atom count
    assert gin["training"]["standardize_latents"] is True
    stats = torch.load(model_dir / "checkpoints" / "latent_stats.pt", map_location="cpu", weights_only=True)
    assert stats["mu"].shape == (16,) and stats["std"].shape == (16,)


def test_deterministic_val(tmp_path, monkeypatch):
    import SCAPES.training.FlowModel_trainer as T
    trainer = _build(tmp_path / "toy")
    seen = []
    orig = T.flow_matching_loss

    def spy(*a, **k):
        if k.get("s") is not None:
            seen.append(k["s"].detach().flatten().cpu())
        return orig(*a, **k)

    monkeypatch.setattr(T, "flow_matching_loss", spy)
    cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
    a = trainer.val_epoch()[0]
    b = trainer.val_epoch()[0]
    assert torch.equal(torch.get_rng_state(), cpu) and torch.equal(torch.cuda.get_rng_state(), cuda)
    assert abs(a - b) <= 1e-6 * abs(a)
    s = torch.cat(seen)
    assert set(round(v, 6) for v in s.tolist()) == {0.125, 0.375, 0.625, 0.875}
    n_val = len(trainer.dataset.get_splits()[1])
    assert s.numel() == 2 * 4 * n_val  # two calls x 4 stratified s per val sample


def test_prefix_masking(tmp_path, monkeypatch):
    """o = 2 (ar.prefix_seconds = 0.002 at 1000 Hz): the prefix given to the loss is the last o past frames, clean."""
    import SCAPES.training.FlowModel_trainer as T
    extra_t = NO_REG + ["training.standardize_latents = False", "training.max_steps = 1", "training.log_every = 1"]
    trainer = _build(tmp_path / "toy", extra_dataprep=["ar.prefix_seconds = 0.002"], extra_training=extra_t)
    assert trainer.dataset.geom.prefix_frames == 2
    seen = {}
    orig = T.flow_matching_loss

    def spy(*a, **k):
        seen["prefix"] = k.get("prefix")
        return orig(*a, **k)

    monkeypatch.setattr(T, "flow_matching_loss", spy)
    batch = _fixed_batch(trainer)
    trainer.train_loader = [batch]
    trainer.val_loader = None
    trainer.train(epochs=1, audio_val_freq=10 ** 9, val_nfe=4, patience=10)
    p = seen["prefix"]
    assert p is not None and p.shape == (8, 2, 16)
    want = batch["memory_buffer_latent"][:, -1, :, -2:].transpose(1, 2)
    assert max_abs(p, want) == 0.0

    # R2-C5: with past noise the prefix is taken from the past AFTER apply_past_noise (same perturbation as the memory)
    extra_n = [l for l in extra_t if not l.startswith("training.past_noise")] + ["training.past_noise = 0.1"]
    trainer = _build(tmp_path / "toy_noise", extra_dataprep=["ar.prefix_seconds = 0.002"], extra_training=extra_n)
    noised = {}
    orig_noise = T.apply_past_noise

    def spy_noise(past, gamma):
        out = orig_noise(past, gamma)
        noised["out"] = out.detach().clone()
        return out

    monkeypatch.setattr(T, "apply_past_noise", spy_noise)
    batch = _fixed_batch(trainer)
    trainer.train_loader = [batch]
    trainer.val_loader = None
    trainer.train(epochs=1, audio_val_freq=10 ** 9, val_nfe=4, patience=10)
    p = seen["prefix"]
    want = noised["out"][:, -1, :, -2:].transpose(1, 2)
    assert max_abs(p, want) == 0.0
    clean = batch["memory_buffer_latent"][:, -1, :, -2:].transpose(1, 2)
    assert max_abs(p, clean) > 0.0  # non-vacuous: the noise did change it


def test_past_usage_eval(tmp_path):
    trainer = _build(tmp_path / "toy")
    _, val = trainer.dataset.get_splits()
    torch.manual_seed(0)
    torch.nn.init.normal_(trainer.model.transformer.output_proj.weight, std=0.05)  # non-zero outputs
    r = trainer.evaluate_past_usage(val)
    assert set(r) >= {"L_real", "L_null", "ratio"}
    assert abs(r["ratio"] - r["L_real"] / r["L_null"]) <= 1e-9
    assert abs(r["ratio"] - 1.0) > 1e-6  # with a memory-using model the swap matters
    orig_forward = trainer.model.forward

    def ignores_memory(x_t, s, context_vector, encoded_past, structure_vector=None):
        return orig_forward(x_t, s, context_vector, torch.zeros_like(encoded_past), structure_vector)

    trainer.model.forward = ignores_memory
    r = trainer.evaluate_past_usage(val)
    assert abs(r["ratio"] - 1.0) <= 1e-6


def test_unsupported_features(tmp_path):
    from SCAPES.pipeline import build_trainer
    root = tmp_path / "toy"
    prepare_toy(root, "autoregressive")
    for line in ("training.spectral_representation = True", "training.use_discriminator = True",
                 "local_encoder.temporal_compression = 3"):  # R2-C4: AR needs one encoded frame per latent frame
        tr = (root / "config" / "training.gin").read_text()
        cfg = root / f"cfg_{line.split('.')[1].split()[0]}"
        cfg.mkdir()
        (cfg / "dataprep.gin").write_text((root / "config" / "dataprep.gin").read_text())
        (cfg / "training.gin").write_text(tr + line + "\n")
        with pytest.raises(NotImplementedError):
            build_trainer(root, root / f"model_{cfg.name}", config_dir=cfg)
