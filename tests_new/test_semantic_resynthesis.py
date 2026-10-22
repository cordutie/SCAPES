"""Semantic resynthesis: cold start with per-unit CLAP gives finite audio of the right length, deterministic for a seed,
and does not read the given (real) units. Run from SCAPES/: ../venv_SCAPES/bin/python -m pytest tests_new -q"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from conftest import SCRIPTS, train_toy


def _load():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("semantic_resynthesis", SCRIPTS / "semantic_resynthesis.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_semantic_resynthesis_toy(tmp_path):
    from SCAPES.inference.FlowInference import FlowInference
    mod = _load()
    model = train_toy(tmp_path / "toy", "autoregressive", steps=10)
    engine = FlowInference(model, device="cuda" if torch.cuda.is_available() else "cpu")
    claps = [torch.randn(1024) for _ in range(6)]
    a = mod.semantic_resynthesis(engine, claps, nfe=2, cfg=3.0, seed=0)
    b = mod.semantic_resynthesis(engine, claps, nfe=2, cfg=3.0, seed=0)
    c = mod.semantic_resynthesis(engine, claps, nfe=2, cfg=3.0, seed=1)
    assert torch.isfinite(a).all() and a.shape[-1] > 0
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert not any(s["TF"] for s in engine.timeline)       # cold start: nothing is teacher-forced


def test_rms_db_track():
    mod = _load()
    x = np.concatenate([np.ones(100), 0.1 * np.ones(100)])
    t = mod.rms_db_track(x, 100)
    assert abs(t[0]) < 1e-6 and abs(t[1] + 20) < 1e-6
