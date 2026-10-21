"""Slice B1: venv_SCAPES after the numpy-2 migration (design D1, A3, D14)."""
import torch

from conftest import FIXTURES, max_abs


def test_versions():
    import jax
    import numpy as np
    import audio_codec_wrappers  # noqa: F401
    assert int(np.__version__.split(".")[0]) == 2
    assert torch.__version__ == "2.10.0+cu128"
    assert jax.__version__ == "0.10.1"


def test_clap_fixture():
    from SCAPES.auxiliar.clap_wrapper import CLAPWrapper
    ref = torch.load(FIXTURES / "clap_ref.pt", map_location="cpu", weights_only=True)
    clap = CLAPWrapper(version="2023", use_cuda=True)
    emb = clap.compute_embedding(ref["audio"], og_sr=48000, random_extension=False)[0]
    cos = torch.nn.functional.cosine_similarity(emb.float().cpu().flatten(), ref["embedding"].float().flatten(), dim=0)
    assert float(cos) >= 1 - 1e-5


def test_encodec48_fixture():
    from audio_codec_wrappers import load_codec
    ref = torch.load(FIXTURES / "encodec48_ref.pt", map_location="cpu", weights_only=True)
    enc = load_codec("encodec48k", device="cuda").encode(ref["audio"], 48000)
    assert max_abs(enc.z, ref["latent"]) <= 1e-5
    assert max_abs(enc.scale.reshape(-1), ref["scale"].reshape(-1)) <= 1e-5
