"""Slices B6, B8, B11: tools for S3 (RTF), the AR falsifier E3b and the S5 evaluation (design Metrics, R1-C2, R1-C5).

Interfaces used (plan, slices B6/B8/B11):
- scripts/measure_rtf.py: CLI (--gin-dir, --size custom, --pairs, --aa-pairs, --seed-base, --session-warmup) and functions
  paired_log_ratios, aa_log_ratios, aa_margin, bootstrap_mean_ci, classify (R2-C3);
- scripts/s5_eval.py: CLI;
- SCAPES.inference.diagnostics: seam_ratio(z [d, T], block, phase=0), production_checks(y, sample_rate, ref_rms,
  target_seconds, unit_seconds).
"""
import importlib.util
import json
import math
import subprocess
import sys

import torch

from conftest import (SCRIPTS, SCAPES_REPO, train_toy)


def _script(name):
    spec = importlib.util.spec_from_file_location(f"scapes_script_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(args, timeout=1800):
    out = subprocess.run([sys.executable, *map(str, args)], cwd=SCAPES_REPO, capture_output=True, text=True,
                         timeout=timeout)
    assert out.returncode == 0, out.stderr[-4000:]
    return out


def test_s3_statistics_known():
    """R2-C3: paired log-ratios by time slot, A/A margin, bootstrap CI of the mean, three outcomes."""
    import statistics
    m = _script("measure_rtf")
    lo, hi = m.bootstrap_mean_ci([0.3, 0.3, 0.3], n_resamples=10000, seed=0)
    assert lo == hi == 0.3
    procs = [
        {"pair": 0, "slot": 0, "kind": "ab", "mode": "atoms", "rtf": [1.9, 2.0, 2.1]},
        {"pair": 0, "slot": 1, "kind": "ab", "mode": "autoregressive", "rtf": [1.0, 1.0, 1.0]},
        {"pair": 1, "slot": 2, "kind": "ab", "mode": "autoregressive", "rtf": [1.0, 1.0, 1.0]},
        {"pair": 1, "slot": 3, "kind": "ab", "mode": "atoms", "rtf": [3.5, 4.0, 4.5]},
        {"pair": 2, "slot": 4, "kind": "aa", "mode": "atoms", "rtf": [1.0]},
        {"pair": 2, "slot": 5, "kind": "aa", "mode": "atoms", "rtf": [1.1]},
        {"pair": 3, "slot": 6, "kind": "aa", "mode": "autoregressive", "rtf": [1.0]},
        {"pair": 3, "slot": 7, "kind": "aa", "mode": "autoregressive", "rtf": [0.9]},
    ]
    lr = {int(k): v for k, v in m.paired_log_ratios(procs).items()}
    assert set(lr) == {0, 1}
    assert abs(lr[0] - math.log(2.0)) <= 1e-12 and abs(lr[1] - math.log(4.0)) <= 1e-12  # medians, paired by slot
    aa = sorted(m.aa_log_ratios(procs))
    assert len(aa) == 2
    assert abs(aa[0] - math.log(1.0 / 1.1)) <= 1e-12 and abs(aa[1] - math.log(1.0 / 0.9)) <= 1e-12
    assert abs(m.aa_margin(aa) - max(0.01, 2 * statistics.stdev(aa))) <= 1e-12
    assert m.aa_margin([]) == 0.01 and m.aa_margin([0.0, 0.0]) == 0.01
    # resampling the 2 pairs: bootstrap means take only log2, 1.5*log2, log4
    lo, hi = m.bootstrap_mean_ci(list(lr.values()), n_resamples=10000, seed=0)
    assert abs(lo - math.log(2.0)) <= 1e-12 and abs(hi - math.log(4.0)) <= 1e-12
    assert m.classify((0.05, 0.2), 0.01) == "faster"
    assert m.classify((-0.2, -0.05), 0.01) == "slower"
    assert m.classify((-0.005, 0.02), 0.01) == "indistinguishable"
    assert m.classify((0.005, 0.02), 0.01) == "indistinguishable"  # above 0 but inside the noise margin


def test_seam_ratio_known():
    from SCAPES.inference.diagnostics import seam_ratio
    T, Bk = 320, 32
    t = torch.arange(T, dtype=torch.float64)
    direction = torch.randn(16, 1, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    direction = direction / direction.norm()
    linear = direction * t  # every step has norm 1
    assert abs(seam_ratio(linear, Bk) - 1.0) <= 1e-6
    jumps = direction * (t + 5.0 * torch.div(t, Bk, rounding_mode="floor"))  # seams have norm 6
    assert abs(seam_ratio(jumps, Bk) - 6.0) <= 1e-6


def test_s5_eval_toy(tmp_path):
    from SCAPES.inference.diagnostics import production_checks
    runs = tmp_path / "runs"
    root = tmp_path / "toy"
    train_toy(root, "autoregressive", steps=20, save_name="model")
    (runs / "toy_causal").mkdir(parents=True)
    (runs / "toy_causal" / "model").symlink_to(root / "model", target_is_directory=True)
    out_dir = tmp_path / "s5"
    _run([SCRIPTS / "s5_eval.py", "--runs", runs, "--dataset", root, "--seeds", "0", "--seconds", "1",
          "--prompt-region", "val", "--out", out_dir])
    r = json.loads((out_dir / "s5_eval.json").read_text())
    clips = r["toy_causal"]["clips"]
    assert len(clips) == 2  # 2 files x 1 seed
    for c in clips:
        assert set(c) >= {"file", "seed", "finite", "duration_ok", "rms_ok", "rms_first_ok", "clip_ok", "pass"}
    assert set(r["toy_causal"]["diagnostics"]) >= {"D-a", "D-b", "D-c", "D-d", "D-e"}

    # production checks on known signals (metric gaming: silence must fail)
    sr, ref = 8000, 0.1
    g = torch.Generator().manual_seed(0)
    good = 0.1 * torch.randn(1, 10 * sr, generator=g)
    assert production_checks(good, sr, ref, 10.0, 0.032)["pass"] is True
    silent = torch.zeros(1, 10 * sr)
    c = production_checks(silent, sr, ref, 10.0, 0.032)
    assert c["rms_ok"] is False and c["pass"] is False
    late = good.clone()
    late[:, :sr] = 0.0
    c = production_checks(late, sr, ref, 10.0, 0.032)
    assert c["rms_ok"] is True and c["rms_first_ok"] is False and c["pass"] is False
    clipped = good.clone()
    clipped[:, ::50] = 1.0  # 2% of samples at full scale
    assert production_checks(clipped, sr, ref, 10.0, 0.032)["clip_ok"] is False
    nan = good.clone()
    nan[0, 5] = float("nan")
    assert production_checks(nan, sr, ref, 10.0, 0.032)["finite"] is False
    assert production_checks(good[:, : 9 * sr], sr, ref, 10.0, 0.032)["duration_ok"] is False
