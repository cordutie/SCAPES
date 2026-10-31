"""rtf_bench.Decoder (the real-time decode path) equals the offline render for consistent / independent atoms (S7 tool).
Toy codec on the GPU. Run from SCAPES/: ../venv_SCAPES/bin/python -m pytest scripts/tests_eval -q
"""
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[2] / "tests_new"))
import rtf_bench
from _atoms_toy import TOY_GEOM, phase2, toy_geometry


class _Engine:
    def __init__(self, geom, codec, info):
        self.geom, self.codec, self.info = geom, codec, info
        self.sr = info.sample_rate
        self.device = codec.device

    def _unstandardise(self, z):
        return z


@pytest.mark.parametrize("kind", ["consistent", pytest.param("independent", marks=phase2)])
def test_decoder_stream_equals_ola_render(tmp_path, kind):
    from audio_codec_wrappers import load_codec
    from SCAPES.codecs import get_codec_info
    from SCAPES.inference.render import render_atoms
    g = toy_geometry(tmp_path, kind, window="equal_power")
    C, H, _ = TOY_GEOM[kind]
    codec = load_codec("toy_noncausal", device="cuda")
    engine = _Engine(g, codec, get_codec_info("toy_noncausal"))
    gen = torch.Generator().manual_seed(0)
    K, F = 8, C // 8
    atoms = []
    for _ in range(K):
        z = torch.randn(1, 16, F, generator=gen)
        scale = (0.5 + torch.rand(1, generator=gen)).view(1, 1, 1).expand(1, 1, F)
        atoms.append(torch.cat([z, scale], dim=1).cuda())
    dec = rtf_bench.Decoder(engine, 1)
    streamed = torch.cat([dec.push(a) for a in atoms], dim=-1).cpu()
    offline = render_atoms(codec, atoms, g, base_seed=0, smooth=[False] * K, window="equal_power").cpu()
    assert streamed.shape[-1] == K * H
    assert float((streamed - offline[..., :K * H]).abs().max()) <= 1e-6
    assert dec.new_seconds == H / 8000
