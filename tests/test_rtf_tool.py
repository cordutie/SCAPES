"""Slices B6, B8, B11: tools for S3 (RTF), the AR falsifier E3b and the S5 evaluation (design Metrics, R1-C2, R1-C5).

Interfaces used (plan, slices B6/B8/B11):
- scripts/measure_rtf.py: CLI (--gin-dir, --size custom, --pairs, --aa-pairs, --seed-base, --session-warmup) and functions
  paired_log_ratios, aa_log_ratios, aa_margin, bootstrap_mean_ci, classify (R2-C3);
- scripts/ar_falsifier.py: CLI (--config-dir, --gen-seconds, --gen-seeds);
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

from conftest import (SCRIPTS, SCAPES_REPO, TINY_TRAINING, TOY_AR, TOY_COMMON, prepare_toy, train_toy)


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


def test_rtf_json_toy(tmp_path):
    gin_dir = tmp_path / "gins"
    gin_dir.mkdir()
    atoms = ['codec.name = "toy_causal"', 'generation.mode = "atoms"', "atoms.seconds = 0.048",
             "atoms.hop_seconds = 0.016", "atoms.crossfade_seconds = 0.004", "dataset.memory_buffer_atoms = 1"]
    (gin_dir / "toy_causal_atoms.gin").write_text("\n".join(atoms + TOY_COMMON + TINY_TRAINING) + "\n")
    (gin_dir / "toy_causal_ar.gin").write_text("\n".join(TOY_AR + TOY_COMMON + TINY_TRAINING) + "\n")
    out_dir = tmp_path / "rtf"
    _run([SCRIPTS / "measure_rtf.py", "--codec", "toy_causal", "--modes", "atoms", "autoregressive",
          "--gin-dir", gin_dir, "--size", "custom", "--seconds", "1", "--nfe", "2", "--cfg", "3.0",
          "--pairs", "2", "--aa-pairs", "1", "--seed-base", "0", "--reps", "2", "--session-warmup", "0",
          "--random-init", "--out", out_dir])
    r = json.loads((out_dir / "rtf.json").read_text())
    m = _script("measure_rtf")
    procs = r["processes"]
    ab = [p for p in procs if p["kind"] == "ab"]
    aa = [p for p in procs if p["kind"] == "aa"]
    assert len(ab) == 4 and len(aa) == 4  # 2 AB pairs; 1 A/A pair per mode
    assert len({p["pid"] for p in procs}) == 8  # one fresh subprocess per process slot
    assert sorted(p["slot"] for p in procs) == list(range(8))
    # ABBA: pair 0 runs atoms first, pair 1 runs AR first
    for pair, first in ((0, "atoms"), (1, "autoregressive")):
        two = sorted((p for p in ab if p["pair"] == pair), key=lambda p: p["slot"])
        assert [p["mode"] for p in two][0] == first and {p["mode"] for p in two} == {"atoms", "autoregressive"}
    assert sorted(p["mode"] for p in aa) == ["atoms", "atoms", "autoregressive", "autoregressive"]
    flags = [json.dumps(p["flags"], sort_keys=True) for p in procs]
    assert len(set(flags)) == 1  # identical numeric flags in every process (R2-C8)
    assert set(procs[0]["flags"]) >= {"matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_benchmark",
                                      "float32_matmul_precision"}
    for p in procs:
        assert p["warmup_runs"] >= 1 and p["gpu_before"] and p["gpu_after"]
        assert len(p["rtf"]) == 2 and all(v > 0 for v in p["rtf"])
        for k in ("median_rtf", "solves_per_s", "s_per_solve", "decode_s_per_audio_s"):
            assert math.isfinite(p[k]) and p[k] > 0, k
    lr = {int(k): v for k, v in r["log_ratio_per_pair"].items()}
    assert set(lr) == {0, 1}
    assert {int(k): v for k, v in m.paired_log_ratios(procs).items()} == lr
    assert r["pairs_above_1"] == sum(v > 0 for v in lr.values())
    assert r["margin"] == m.aa_margin(m.aa_log_ratios(procs)) and r["margin"] >= 0.01
    assert abs(r["speedup"] - math.exp(sum(lr.values()) / 2)) <= 1e-9
    assert r["outcome"] == m.classify(r["ci95_log"], r["margin"])
    assert r["outcome"] in {"faster", "slower", "indistinguishable"}
    assert "cpu_governor" in r and "session_warmup_s" in r


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


def test_ar_falsifier_toy(tmp_path):
    root, _ = prepare_toy(tmp_path / "toy", "autoregressive")
    out_dir = tmp_path / "falsifier"
    _run([SCRIPTS / "ar_falsifier.py", "--dataset", root, "--codec", "toy_causal", "--prefix-seconds", "0",
          "--steps", "30", "--config-dir", root / "config", "--gen-seconds", "0.5", "--gen-seeds", "0", "1",
          "--out", out_dir])
    r = json.loads((out_dir / "falsifier.json").read_text())
    assert set(r) >= {"L_real", "L_null", "rho_seam_gen", "rho_seam_real", "pass"}
    for k in ("L_real", "L_null", "rho_seam_gen", "rho_seam_real"):
        assert math.isfinite(r[k]) and r[k] > 0, k
    assert r["pass"] == (r["L_real"] < 0.98 * r["L_null"] and r["rho_seam_gen"] <= 2 * r["rho_seam_real"])


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
