"""S6 (slow, GPU): the four codecs end to end on datasets/microtex_smoke (2 x 30 s of microtex): dataprep, dataset items,
inference geometry, 5 training steps, a 2-unit generation, each codec at its defaults, run once (plan slice 6). It runs
scripts/s2_smoke.py, one process per codec stage, and reads its s2_result.json files.

Contract with s2_smoke.py (plan, Interfaces): out/<codec>/s2_result.json has
  stages_ok (int, 3 when dataprep, training and generation succeeded), resolutions_equal (bool),
  startup_lines ({stage: number of "[SCAPES] <codec> (" lines in its log}), item_shapes_ok (bool),
  generation_length_ok (bool), train.all_finite (bool), peak_gpu_bytes ({stage: int}),
  resolution (the dataprep stage's Resolution as a dict, dataclasses.asdict).
Run from SCAPES/: ../venv_SCAPES/bin/python -m pytest tests_v3/test_smoke_codecs.py -q -m slow
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DATA = REPO.parent / "datasets" / "microtex_smoke"
CODECS = ["encodec24", "spectrostream", "encodec48", "music2latent"]

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not (DATA / "raw").is_dir(), reason="datasets/microtex_smoke/raw is missing (plan slice 6)")]


def _run(out, codecs, extra=()):
    cmd = [sys.executable, str(REPO / "scripts" / "s2_smoke.py"), "--dataset", str(DATA), "--out", str(out),
           "--codecs", *codecs, "--max-steps", "5", *extra]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO)})
    assert r.returncode == 0, r.stderr[-3000:]


def _check(out, codec):
    res = json.loads((out / codec / "s2_result.json").read_text())
    assert res["stages_ok"] == 3, res
    assert res["resolutions_equal"] is True
    assert set(res["startup_lines"]) == {"dataprep", "training", "generation"}
    assert all(v == 1 for v in res["startup_lines"].values()), res["startup_lines"]
    assert res["item_shapes_ok"] is True and res["generation_length_ok"] is True
    assert res["train"]["all_finite"] is True
    assert max(res["peak_gpu_bytes"].values()) < 20 * 2 ** 30
    return res


def test_four_codecs(tmp_path):
    out = tmp_path / "s2"
    _run(out, CODECS)
    for codec in CODECS:
        _check(out, codec)

