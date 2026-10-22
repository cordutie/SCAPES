"""region_dup_check: known answers on synthetic audio (a looped file repeats, a noise file does not)."""
import importlib.util
from pathlib import Path

import numpy as np
import soundfile as sf

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "region_dup_check.py"


def load():
    spec = importlib.util.spec_from_file_location("region_dup_check", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_loop_is_exact_and_noise_is_not(tmp_path):
    mod = load()
    sr = 8000
    rng = np.random.RandomState(0)
    loop = np.tile(rng.randn(10 * sr), 20)          # 200 s, period 10 s
    noise = rng.randn(200 * sr)
    sf.write(str(tmp_path / "loop.wav"), loop.astype(np.float32), sr, subtype="FLOAT")
    sf.write(str(tmp_path / "noise.wav"), noise.astype(np.float32), sr, subtype="FLOAT")
    a = mod.check_file(tmp_path / "loop.wav", "val")
    b = mod.check_file(tmp_path / "noise.wav", "val")
    assert a["fraction_exact_ncc_ge_0.99"] == 1.0
    assert all(abs((w["lag_s"] / 10) - round(w["lag_s"] / 10)) < 1e-3 for w in a["windows"])   # lag is a multiple of the period
    assert b["fraction_near_ncc_ge_0.9"] == 0.0 and b["max_ncc"] < 0.2
