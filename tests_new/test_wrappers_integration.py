"""audio_codec_wrappers and SCAPES in one environment: the pinned checkout, the four codecs next to CLAP in one process,
and the whole pipeline (dataprep, training, generation) for every codec on the small dataset."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
CODECS = ["encodec24k", "spectrostream", "encodec48k", "codicodec", "music2latent"]
PIPELINE_CASES = CODECS + ["codicodec_grains"]
SMOKE_DATASET = REPO.parent / "datasets" / "micro_microtex"


def _wrappers_root():
    import audio_codec_wrappers
    return Path(audio_codec_wrappers.__file__).resolve().parents[1]


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def test_wrappers_checkout_has_what_the_codecs_read():
    root = _wrappers_root()
    assert (root / "weights.lock.json").is_file()
    assert (root / "scripts" / "fetch_weights.py").is_file()
    assert _git(root, "rev-parse", "HEAD").returncode == 0


def test_installed_wrappers_are_the_submodule():
    """The environment imports the wrappers from the submodule, at the commit the submodule pointer records."""
    assert _wrappers_root() == (REPO / "external" / "audio_codec_wrappers").resolve()
    pointer = _git(REPO, "ls-tree", "HEAD", "external/audio_codec_wrappers").stdout.split()
    assert pointer and pointer[1] == "commit", "external/audio_codec_wrappers is not a submodule of this repository"
    assert _git(_wrappers_root(), "merge-base", "--is-ancestor", pointer[2], "HEAD").returncode == 0, \
        f"submodule checkout does not contain the committed pointer {pointer[2]}"


def test_wrapper_registry_matches_scapes_codecs():
    from SCAPES.codecs import get_codec_info
    for name in CODECS:
        info = get_codec_info(name)
        assert info.sample_rate > 0 and info.latent_dim > 0 and info.hop_length > 0


@pytest.mark.parametrize("name", CODECS)
def test_codec_round_trip_next_to_clap(name):
    """All four codecs are loaded one after the other in this process, with CLAP loaded once (torch and jax together)."""
    from SCAPES.codecs import get_codec, get_codec_info
    info = get_codec_info(name)
    codec = get_codec(name, "cuda")
    g = torch.Generator().manual_seed(0)
    x = 0.1 * torch.randn(1, info.channels, 2 * info.sample_rate, generator=g)
    enc = codec.encode(x, info.sample_rate)
    assert enc.z.shape[:2] == (1, info.latent_dim) and torch.isfinite(enc.z).all()
    y = codec.decode(enc.z, enc.scale, seed=0)
    assert y.shape == (1, info.channels, enc.z.shape[-1] * info.hop_length) and torch.isfinite(y).all()
    clap = _clap()
    emb = clap.compute_embedding(x[:, :1].cuda(), og_sr=info.sample_rate, random_extension=False)
    assert emb.shape == (1, 1024) and torch.isfinite(emb).all()


_CLAP = []


def _clap():
    if not _CLAP:
        from SCAPES.auxiliar.clap_wrapper import CLAPWrapper
        _CLAP.append(CLAPWrapper(version="2023", use_cuda=True))
    return _CLAP[0]


@pytest.mark.skipif(not (SMOKE_DATASET / "raw").is_dir(), reason="datasets/micro_microtex not present")
def test_pipeline_runs_for_every_codec(tmp_path):
    """Dataprep, 20 training steps and 10 s of generation per codec, each stage in a fresh process (scripts/s2_smoke.py)."""
    out = tmp_path / "s2"
    proc = subprocess.run([sys.executable, str(REPO / "scripts" / "s2_smoke.py"), "--dataset", str(SMOKE_DATASET),
                           "--out", str(out), "--codecs", *PIPELINE_CASES, "--max-steps", "20", "--no-spectral-check"],
                          cwd=REPO, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO)})
    assert proc.returncode == 0, proc.stderr[-2000:]
    for name in PIPELINE_CASES:
        res = json.loads((out / name / "s2_result.json").read_text())
        assert res["stages_ok"] == 3, f"{name}: {res['stages']}"
        art = res["artefacts"]
        assert art["best_flow_model"] and art["inference_gin"] and art["wav_10s"], f"{name}: {art}"
        assert res["train"]["all_finite"], name
        if name in ("encodec48k", "codicodec", "music2latent", "codicodec_grains"):
            _check_atoms_generation(out / name / "model", name)


def _check_atoms_generation(model_dir, name):
    """Generation with the trained atoms model: finite audio of the length the geometry gives, with the crossfade law of the codec."""
    from SCAPES.data.geometry import crossfade_law_for
    from SCAPES.inference.FlowInference import FlowInference
    g = torch.Generator().manual_seed(0)
    clap = [torch.nn.functional.normalize(torch.randn(1024, generator=g), dim=0) for _ in range(5)]
    engine = FlowInference(model_dir, device="cuda", checkpoint="last")
    audio = engine.generate_soundscape(clap, nfe=2, cfg_scale=1.0, seed=0)
    geom = engine.geom
    assert torch.isfinite(audio).all()
    assert audio.shape[-1] == (len(clap) - 1) * geom.audio_hop_samples + geom.unit_frames * geom.hop_length
    assert crossfade_law_for(geom, engine.info) == ("hann" if name == "encodec48k" else "equal_power")
