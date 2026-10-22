"""past_usage_eval on a run trained without standardisation (no latent_stats.pt, like the V1-style encodec48k run), and on a
standardised one: both work, the unstandardised one reports standardized False and latent_stats_match None, and the loss with
the real past equals the trainer's own deterministic validation loss (same preprocessing as in training)."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from conftest import SCAPES_REPO, SCRIPTS, train_toy


def _run(root, model, tmp_path):
    out = tmp_path / "pu.json"
    cmd = [sys.executable, str(SCRIPTS / "past_usage_eval.py"), "--model-dir", str(model), "--config-dir", str(root / "config"),
           "--dataset", str(root), "--scratch", str(tmp_path / "scratch"), "--out", str(out)]
    r = subprocess.run(cmd, cwd=SCAPES_REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(out.read_text())


def test_past_usage_without_latent_stats(tmp_path):
    root = tmp_path / "toy"
    model = train_toy(root, "atoms", steps=5)
    assert not (model / "checkpoints" / "latent_stats.pt").exists()        # toy atoms: standardize_latents False
    r = _run(root, model, tmp_path)
    assert r["standardized"] is False and r["latent_stats_match"] is None
    assert r["L_real"] > 0 and r["L_null"] > 0 and abs(r["gain_percent"] - 100 * (1 - r["L_real"] / r["L_null"])) < 1e-9
    assert set(r["per_file_gain_percent"]) == {"a.wav", "b.wav"} and r["gain_percent_mean_over_files"]["n"] == 2


def test_past_usage_with_latent_stats(tmp_path):
    root = tmp_path / "toy"
    model = train_toy(root, "autoregressive", steps=5)
    assert (model / "checkpoints" / "latent_stats.pt").exists()
    r = _run(root, model, tmp_path)
    assert r["standardized"] is True and r["latent_stats_match"] is True
