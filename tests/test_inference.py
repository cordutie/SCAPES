"""Slice B5: generation in both modes on toy models, and lazy random-init engines for S3 (design "Inference", D15).

Interfaces used: FlowInference(model_dir, device, checkpoint), generate_soundscape(clap, nfe, cfg_scale, seed),
FlowModel.generate(..., prefix=None), engine.model / engine.timeline ("atom_generated" in model units, [1, p, U]).
"""
import subprocess
import sys

import pytest
import torch

from conftest import SCAPES_REPO, TOY_HOP, train_toy


def _claps(n):
    out = []
    for i in range(n):
        c = torch.zeros(1024)
        c[0] = 0.05 * (i + 1)
        out.append(c)
    return out


def _count_generate(monkeypatch):
    from SCAPES.models.flow.FlowModel import FlowModel
    calls = {"n": 0}
    orig = FlowModel.generate

    def spy(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(FlowModel, "generate", spy)
    return calls


def test_ar_generation_toy(tmp_path, monkeypatch):
    from SCAPES.inference.FlowInference import FlowInference
    model_dir = train_toy(tmp_path / "toy", "autoregressive", steps=5)
    engine = FlowInference(model_dir, device="cuda", checkpoint="last")
    calls = _count_generate(monkeypatch)
    y = engine.generate_soundscape(_claps(7), nfe=2, cfg_scale=3.0, seed=0)
    assert y.shape == (1, 7 * 32 * TOY_HOP)
    assert torch.isfinite(y).all()
    assert calls["n"] == 7


def test_ar_prefix_generation_toy(tmp_path, monkeypatch):
    from SCAPES.inference.FlowInference import FlowInference
    from SCAPES.models.flow.FlowModel import FlowModel
    model_dir = train_toy(tmp_path / "toy", "autoregressive", steps=5, extra_dataprep=["ar.prefix_seconds = 0.002"])
    engine = FlowInference(model_dir, device="cuda", checkpoint="last")
    state = {"block": -1, "seen": []}
    orig = FlowModel.generate

    def spy(self, *a, **k):
        state["block"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(FlowModel, "generate", spy)

    def hook(module, args):
        state["seen"].append((state["block"], args[0].shape[1], args[0][:, :2].detach().clone()))

    handle = engine.model.transformer.register_forward_pre_hook(hook)
    y = engine.generate_soundscape(_claps(7), nfe=2, cfg_scale=3.0, seed=0)
    handle.remove()
    assert y.shape == (1, 7 * 32 * TOY_HOP)
    assert state["block"] == 6 and state["seen"]
    for k, n_pos, prefix in state["seen"]:
        assert n_pos == 2 + 32
        if k == 0:
            assert torch.count_nonzero(prefix) == 0  # no frame exists yet: zero (mean) prefix
        else:
            prev = engine.timeline[k - 1]["atom_generated"][0, :, -2:].T.unsqueeze(0)  # [1, 2, p]
            assert torch.equal(prefix.to(prev.device), prev.expand_as(prefix.to(prev.device)))


def test_atoms_generation_toy(tmp_path, monkeypatch):
    from SCAPES.inference.FlowInference import FlowInference
    model_dir = train_toy(tmp_path / "toy", "atoms", steps=5)
    engine = FlowInference(model_dir, device="cuda", checkpoint="last")
    calls = _count_generate(monkeypatch)
    K, F, H = 5, 48, 16
    y = engine.generate_soundscape(_claps(K), nfe=2, cfg_scale=3.0, seed=0)
    assert y.shape == (1, ((K - 1) * H + F) * TOY_HOP)
    assert torch.isfinite(y).all()
    assert calls["n"] == K


@pytest.mark.parametrize("codec,mode", [("encodec24k", "autoregressive"), ("spectrostream", "autoregressive"),
                                        ("encodec48k", "atoms"), ("codicodec", "atoms")])
def test_from_config_lazy(codec, mode):
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(SCAPES_REPO)!r})\n"
        "import audio_codec_wrappers, audio_codec_wrappers.registry as R\n"
        "def forbidden(*a, **k):\n"
        "    raise AssertionError('codec weights loaded by from_config')\n"
        "audio_codec_wrappers.load_codec = forbidden\n"
        "R.load_codec = forbidden\n"
        "import SCAPES.codecs as SC\n"
        "SC.get_codec = forbidden\n"
        "from SCAPES.inference.FlowInference import FlowInference\n"
        f"eng = FlowInference.from_config({codec!r}, {mode!r}, size='small', device='cpu', seed=0)\n"
        "assert eng.model is not None\n"
        f"if {codec!r} != 'spectrostream':\n"
        "    assert 'jax' not in sys.modules, 'jax imported'\n"
        "assert 'codicodec' not in sys.modules, 'codicodec imported'\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=SCAPES_REPO, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.strip().splitlines()[-1] == "ok"
