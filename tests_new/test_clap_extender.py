"""audio_extender: batch rows are independent (own random cuts, own peak normalisation)."""
import numpy as np
import torch

from SCAPES.auxiliar.clap_wrapper import audio_extender

SR, DUR = 1000, 1.0


def _call(x, random_extension):
    return audio_extender(x, random_extension=random_extension, sample_rate=SR, duration=DUR, overlap_ratio=0.1)


def test_row_in_batch_equals_single_call_without_randomness():
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(1, 1, 300, generator=g), 3 * torch.randn(1, 1, 300, generator=g)
    single = _call(a, False)
    batch = _call(torch.cat([a, b]), False)
    assert torch.equal(batch[0:1], single)


def test_each_row_is_peak_normalised():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(3, 1, 300, generator=g) * torch.tensor([1.0, 5.0, 0.1]).view(3, 1, 1)
    for random_extension in (False, True):
        np.random.seed(0)
        y = _call(x, random_extension)
        assert y.shape == (3, 1, int(SR * DUR))
        assert torch.allclose(y.abs().amax(dim=(1, 2)), torch.ones(3))


def test_random_cuts_differ_between_rows():
    x = torch.arange(300, dtype=torch.float32).view(1, 1, -1).repeat(4, 1, 1) + 1.0
    np.random.seed(0)
    y = _call(x, True)
    assert not all(torch.equal(y[0], y[i]) for i in range(1, 4))


def test_two_dim_input_keeps_old_shape():
    assert _call(torch.randn(2, 300), False).shape == (1, int(SR * DUR))
