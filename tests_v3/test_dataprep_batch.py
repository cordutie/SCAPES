"""D9: dataprep encodes segments in batches sized by a sample budget (2**22 samples per encode call).
Run from SCAPES/: ../venv_SCAPES/bin/python -m pytest tests_v3/test_dataprep_batch.py -q
"""
from types import SimpleNamespace

import pytest
import torch


class StubCodec:
    """Records the batch size of every encode call; z = per-frame mean of the first channel (deterministic)."""

    def __init__(self, h):
        self.h, self.calls = h, []

    def encode(self, audio, sr):
        from audio_codec_wrappers import Encoded
        self.calls.append(audio.shape[0])
        B, C, M = audio.shape
        z = audio[:, :1, :].reshape(B, 1, M // self.h, self.h).mean(-1)
        return Encoded(z=z, scale=None, num_samples=M)


@pytest.mark.parametrize("C,H,K,batches", [(2 ** 20, 2 ** 18, 10, [4, 4, 2]), (1024, 512, 100, [100])])
def test_batch_from_sample_budget(C, H, K, batches):
    from SCAPES.data.dataprep.atoms import _encode_segments
    h = 1024 if C >= 1024 else 8
    geom = SimpleNamespace(sample_rate=44100, atom_samples=C, hop_samples=H, latent_dim=1, unit_frames=C // h, hop_length=h)
    x = torch.randn(1, 2, C + (K - 1) * H, generator=torch.Generator().manual_seed(0))
    codec = StubCodec(h)
    lat, scale = _encode_segments(codec, x, geom)
    assert codec.calls == batches
    assert lat.shape == (K, 1, C // h) and scale.shape == (K,)
    one = StubCodec(h)
    lat1, _ = _encode_segments(one, x, geom, batch=1)
    assert one.calls == [1] * K
    assert torch.equal(lat, lat1)
