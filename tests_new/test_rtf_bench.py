"""rtf_bench on tiny toy models (CPU timing): both modes give finite audio of the right length, per-block timing records, flow +
decode = total, and the guards refuse to run while the GPU is in use / the chain is unfinished."""
import importlib.util
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from conftest import SCRIPTS, train_toy

sys.path.insert(0, str(SCRIPTS))
import rtf_bench


def _bench(model, name):
    args = Namespace(device="cpu", calls=3, warmup=1, no_bf16=True)
    return rtf_bench.bench_model(args, name, str(model))


def test_bench_ar_and_atoms_toy(tmp_path):
    for mode in ("autoregressive", "atoms"):
        model = train_toy(tmp_path / mode, mode, steps=5)
        r = _bench(model, mode)
        c = r["configs"]["nfe16_cfg3"]
        g_unit = r["unit_frames"] if mode == "autoregressive" else r["unit_hop_frames"]
        assert abs(c["new_audio_seconds"] - g_unit * 8 / 8000) < 1e-9          # toy codecs: hop 8 samples at 8 kHz
        assert abs(c["generated_audio_seconds"] - r["unit_frames"] * 8 / 8000) < 1e-9
        assert abs(c["rtf_generated"]["mean"] * c["generated_audio_seconds"] - c["rtf"]["mean"] * c["new_audio_seconds"]) < 1e-9
        assert abs(c["overlap_seconds"] - (c["generated_audio_seconds"] - c["new_audio_seconds"])) < 1e-12
        if mode == "autoregressive":
            assert c["rtf_generated_over_rtf_new"] == 1.0 and c["overlap_seconds"] == 0.0
        else:
            assert c["rtf_generated_over_rtf_new"] < 1.0 and c["overlap_seconds"] > 0
        assert c["finite"] and c["calls"] == 3 and c["rtf"]["mean"] > 0
        assert abs(c["per_block_seconds"]["mean"] - c["flow_seconds"]["mean"] - c["decode_seconds"]["mean"]) < 1e-9
        assert set(r["configs"]) == {"nfe32_cfg3", "nfe16_cfg3", "nfe16_cfgoff"}


def test_bench_guards(tmp_path):
    cmd = [sys.executable, str(SCRIPTS / "rtf_bench.py"), "--small-runs", str(tmp_path), "--out", str(tmp_path / "o")]
    out = subprocess.run(cmd, capture_output=True, text=True)
    assert out.returncode != 0 and ("GPU" in out.stderr or "chain" in out.stderr)
    dry = subprocess.run([sys.executable, str(SCRIPTS / "rtf_bench.py"), "--dry"], capture_output=True, text=True)
    assert dry.returncode == 0 and sum(l.split()[0].endswith("_large") for l in dry.stdout.splitlines()[:8]) == 4 and sum(l.split()[0].endswith("_small") for l in dry.stdout.splitlines()[:8]) == 4
