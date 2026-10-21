"""Slice B5: rendering, atoms OLA and AR streaming (design "Inference")."""
import warnings

import torch

from conftest import CONFIG, TOY_D, TOY_HOP, max_abs, resolve_lines


def _v1_geom(tmp_path):
    lines = (CONFIG / "dataprep.gin").read_text().splitlines() + \
        (CONFIG / "codecs" / "encodec48k_atoms.gin").read_text().splitlines()
    return resolve_lines(tmp_path, lines, "v1")[1]


def test_ola_ones(tmp_path):
    from SCAPES.inference.render import ola_render
    g = _v1_geom(tmp_path)
    F, H, h, K = g.unit_frames, g.unit_hop_frames, g.hop_length, 6
    y = ola_render([torch.ones(2, F * h) for _ in range(K)], g)
    assert y.shape == (2, ((K - 1) * H + F) * h)
    interior = y[:, F * h:(K - 1) * H * h]
    assert interior.numel() > 0
    assert float((interior - 1).abs().max()) <= 1e-6


def _blocks(k=5, b=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, TOY_D, b, generator=g).cuda() for _ in range(k)]


def test_stream_renderer_exact(monkeypatch):
    from audio_codec_wrappers import load_codec
    from SCAPES.inference.render import StreamRenderer
    codec = load_codec("toy_causal", device="cuda")
    pushes = {"n": 0}
    orig_factory = codec.streaming_decoder

    def spy_factory(*a, **k):
        dec = orig_factory(*a, **k)
        orig_push = dec.push

        def counting_push(z):
            pushes["n"] += 1
            return orig_push(z)

        dec.push = counting_push
        return dec

    monkeypatch.setattr(codec, "streaming_decoder", spy_factory)
    blocks = _blocks()
    r = StreamRenderer(codec)
    for z in blocks:
        r.push(z)
    y = r.finish()
    want = codec.decode(torch.cat(blocks, dim=-1))[0]
    assert y.shape == (1, 5 * 32 * TOY_HOP)
    assert max_abs(y, want) <= 1e-6
    assert pushes["n"] == 5



class _DelayedToy:
    """A causal codec whose streaming output is delayed by one frame: latency L = h."""

    def __init__(self, toy):
        self.toy = toy
        self.info = toy.info
        self.device = toy.device

    def decode(self, z, scale=None, seed=0):
        return self.toy.decode(z, scale, seed)

    def streaming_decoder(self, batch_size=1):
        toy = self.toy

        class Dec:
            latency_samples = TOY_HOP

            def __init__(self):
                self.reset()

            def reset(self):
                self.tail = torch.zeros(batch_size, 1, TOY_HOP, device=toy.device)

            def push(self, z):
                y = torch.cat([self.tail, toy.decode(z)], dim=-1)
                self.tail = y[..., -TOY_HOP:]
                return y[..., :-TOY_HOP]

            def flush(self):
                out, self.tail = self.tail, torch.zeros_like(self.tail)
                return out

        return Dec()


def test_stream_renderer_latency():
    from audio_codec_wrappers import load_codec
    from SCAPES.inference.render import StreamRenderer
    toy = load_codec("toy_causal", device="cuda")
    fake = _DelayedToy(toy)
    blocks = _blocks(seed=1)
    r = StreamRenderer(fake)
    for z in blocks:
        r.push(z)
    y = r.finish()
    want = toy.decode(torch.cat(blocks, dim=-1))[0]
    assert y.shape == want.shape
    assert max_abs(y, want) <= 1e-6
