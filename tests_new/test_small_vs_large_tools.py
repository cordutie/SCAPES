"""val_split_report and eval_small_all on fake data (CPU only, no model)."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import val_split_report as vsr


def test_val_split_report(tmp_path):
    model = tmp_path / "encodec24k" / "model"
    (model / "logs").mkdir(parents=True)
    (model / "loss").mkdir()
    files = ["forest.wav", "park.wav", "rain.wav", "tokio.wav", "applause.wav", "boiling.wav"]
    # clean files are best at validation 1, repeat files at validation 3; the total (mean of all) is best at 3
    vpf = [{"step": 10, "per_file": {f: (1.0 if f in vsr.CLEAN else 3.0) for f in files}},
           {"step": 20, "per_file": {f: (1.5 if f in vsr.CLEAN else 2.0) for f in files}},
           {"step": 30, "per_file": {f: (2.0 if f in vsr.CLEAN else 1.0) for f in files}}]
    (model / "logs" / "val_per_file.json").write_text(json.dumps(vpf))
    (model / "logs" / "history_steps.json").write_text(json.dumps([{"step": 10, "loss": 2.0, "time": 5.0}, {"step": 30, "loss": 1.0, "time": 15.0}]))
    (model / "loss" / "loss_history.json").write_text(json.dumps({"train": {"total": [2, 1]}, "val": {"total": [2.3, 1.9, 1.7]}}))
    r = vsr.report(tmp_path / "encodec24k")
    assert r["best_by_total"]["step"] == 30 and r["min_clean"]["step"] == 10
    assert abs(r["best_by_total"]["mean_clean"] - 2.0) < 1e-9 and abs(r["best_by_total"]["mean_repeat"] - 1.0) < 1e-9
    assert abs(r["s_per_step_mean"] - 0.5) < 1e-9


def test_eval_small_all_dry_and_guard(tmp_path):
    cmd = [sys.executable, str(REPO / "scripts" / "eval_small_all.py"), "--small-runs", str(tmp_path)]
    out = subprocess.run(cmd + ["--dry"], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.count("[a]") == 11 and "[e]" in out.stdout and "--prompt-region val" in out.stdout
    assert "reserve" not in out.stdout
    guard = subprocess.run(cmd + ["--only", "d"], capture_output=True, text=True)
    assert guard.returncode != 0 and "chain has not finished" in (guard.stderr + guard.stdout)


def test_size_aliases_resolve_to_the_same_config():
    from SCAPES.data.config_loader import _SIZE_TABLE
    groups = {"petite": "pp", "tiny": "p", "small": "mp", "medium": "mf", "large": "f", "extra_large": "ff"}
    for name, alias in groups.items():
        assert _SIZE_TABLE[name] == _SIZE_TABLE[alias], name
    assert _SIZE_TABLE["petite"] == {"d_model": 256, "num_layers": 4, "nhead": 4, "dim_feedforward": 1024}
    assert _SIZE_TABLE["tiny"] == {"d_model": 384, "num_layers": 5, "nhead": 6, "dim_feedforward": 1536}
    assert "ppp" not in _SIZE_TABLE
    assert "audiobox" in _SIZE_TABLE and all(v["d_model"] // v["nhead"] == 64 for v in _SIZE_TABLE.values())
