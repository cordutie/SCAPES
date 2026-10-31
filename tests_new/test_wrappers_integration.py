"""audio_codec_wrappers and SCAPES in one environment: the pinned checkout and the codecs next to CLAP in one process.
The whole pipeline for every codec is the slow run of tests_v3/test_smoke_codecs.py."""
import subprocess
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
CODECS = ["encodec24k", "spectrostream", "encodec48k", "music2latent"]


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
