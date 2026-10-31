"""S1: the regenerated CLAP fixture keeps the V1 audio and records why it changed (design D20, plan slice 2).

The locked tests/test_env_scapes.py::test_clap_fixture checks the embedding itself; this test checks the provenance.
Lives outside tests/ because that folder is locked. Run from SCAPES/: ../venv_SCAPES/bin/python -m pytest tests_new -q
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "fixtures" / "v1"
sys.path.insert(0, str(REPO / "scripts"))


def test_fixture_audio_unchanged_and_hash():
    import make_v1_fixtures
    ref = torch.load(FIXTURES / "clap_ref.pt", map_location="cpu", weights_only=True)
    assert torch.equal(ref["audio"], make_v1_fixtures.load_water(48000)), "the fixture audio must be the V1 audio"
    sums = dict(reversed(line.split()) for line in (FIXTURES / "SHA256SUMS").read_text().splitlines() if line.strip())
    assert sums["clap_ref.pt"] == hashlib.sha256((FIXTURES / "clap_ref.pt").read_bytes()).hexdigest()
    meta = json.loads((FIXTURES / "meta.json").read_text())
    rec = meta["clap_ref_regenerated"]
    assert "audio_extender" in rec["reason"]
    assert 0.9 < float(rec["cos_old_new"]) < 1.0
    assert rec.get("git_head") and rec.get("date")
