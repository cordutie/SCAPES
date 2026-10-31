"""S7: an exact-inverse stand-in with the real geometry of encodec48 and music2latent goes through dataprep (per-segment
encoding on the latent grid) and the overlap-add with the one window (crossfade_window), and gives back g(t) * source,
with g the overlap-add of the window (equal power: g = 1 outside the crossfades).
Run from SCAPES/: ../venv_SCAPES/bin/python -m pytest tests_v3/test_identity_geometry.py -q
"""
import pytest
import torch

from v3_helpers import (TINY_TRAINING, TOY_COMMON, install_identity_codecs, stub_clap, write_config, write_tone)

CASES = [  # (codec, c, m_s, relation of x to n); x <= n always under the rule (design P-13)
    ("encodec48", 12, 0.5, "x<n"),        # n 13, x 7, o 13, A 33
    ("music2latent", 10, 0.5, "x=n"),     # n 1, x 1, o 4, A 6 (ones part of length 0)
    ("music2latent", 3, 1, "x<n"),        # n 4, x 2, o 4, A 10
]


def _units(data, geom, K):
    F = geom.unit_frames
    out = []
    for k in range(K):
        z = data["latent"][k].float()[None]
        if geom.has_scale:
            z = torch.cat([z, data["scale"][k].float().view(1, 1, 1).expand(1, 1, F)], dim=1)
        out.append(z.cuda())
    return out


@pytest.mark.parametrize("codec,c,ms,rel", CASES, ids=[f"{a}-{b}-{d}" for a, b, _, d in CASES])
def test_identity_codec_reconstruction(tmp_path, monkeypatch, codec, c, ms, rel):
    from SCAPES.data.geometry import atom_count
    from SCAPES.data.geometry import crossfade_window
    from SCAPES.inference.render import render_atoms
    from SCAPES.pipeline import run_dataprep
    install_identity_codecs(monkeypatch)
    from SCAPES.codecs import get_codec
    root = tmp_path / "ds"
    write_tone(root / "raw" / "tone.wav")
    cfg_dir = root / "config"
    write_config(cfg_dir, [f'codec = "{codec}"', f"conditioning_rate = {c}", f"conditioning_memory_seconds = {ms}"]
                 + TOY_COMMON, TINY_TRAINING)
    store = run_dataprep(root, config_dir=cfg_dir, device="cuda", clap_embedder=stub_clap)
    geom = store.geom
    res = geom.resolution
    n, x = res.n, res.crossfade
    assert {"x=n": n == x, "x<n": x < n}[rel], (n, x)
    C, H, X = geom.atom_samples, geom.hop_samples, geom.crossfade_samples
    assert H % geom.hop_length == 0 and H == n * geom.hop_length and X == x * geom.hop_length
    assert C == res.atom * geom.hop_length

    codec_obj = get_codec(codec, "cuda")
    import soundfile as sf
    audio, sr = sf.read(str(root / "raw" / "tone.wav"), dtype="float32", always_2d=True)
    xsrc = codec_obj.prepare(torch.from_numpy(audio.T.copy()), sr)            # [1, ch, N] at the codec rate
    N = xsrc.shape[-1]
    data = store.load_latents(root / "raw" / "tone.wav")
    K = data["latent"].shape[0]
    assert K == atom_count(N, C, H)
    peak = float(xsrc.abs().max())
    for k in (0, 1, K - 1):
        want = codec_obj.encode(xsrc[..., k * H:k * H + C], geom.sample_rate).z[0].float().cpu()
        assert float((data["latent"][k].float() - want).abs().max()) <= 2e-3 * float(want.abs().max()) + 1e-6, k

    units = _units(data, geom, K)
    src = xsrc[0].cpu()
    O = C - H - X
    lo, hi = O + X, (K - 1) * H + C - X
    y = render_atoms(codec_obj, units, geom, smooth=[False] * K).cpu()
    assert y.shape[-1] == (K - 1) * H + C and 0 <= N - y.shape[-1] < H
    w = crossfade_window(geom).double()
    gain = torch.zeros((K - 1) * H + C, dtype=torch.float64)          # g(t), written here, not through ola_render
    for k in range(K):
        gain[k * H:k * H + C] += w
    plateau = (gain[lo:hi] - 1).abs() <= 1e-9
    assert bool(plateau.any()) and float(gain[lo:hi].max()) <= 2 ** 0.5 + 1e-6
    want = gain[lo:hi].float() * src[:, lo:hi]
    assert float((y[:, lo:hi] - want).abs().max()) <= 2e-3 * peak * float(gain.max()), "equal power: g(t) * source"


def test_scale_smoothing_known_values(tmp_path, monkeypatch):
    """V1 scale smoothing (render.py:44-50) on 3 atoms with raw scales 1.0, 2.0, 0.5: smoothed 1.0, 1.09, 0.9919."""
    from SCAPES.inference.render import render_atoms
    from v3_helpers import resolve_v3
    _, geom = resolve_v3(tmp_path, ['codec = "encodec48"'] + TOY_COMMON, "e48")
    assert geom.has_scale

    class Spy:
        def __init__(self):
            self.scales = []

        def decode(self, z, scale=None, seed=0):
            self.scales.append(float(scale.reshape(-1)[0]))
            return torch.zeros(z.shape[0], 2, z.shape[-1] * geom.hop_length, device=z.device)

    F, d = geom.unit_frames, geom.latent_dim
    atoms = [torch.cat([torch.zeros(1, d, F), torch.full((1, 1, F), s)], dim=1) for s in (1.0, 2.0, 0.5)]
    spy = Spy()
    render_atoms(spy, atoms, geom, smooth=None)
    assert [round(s, 4) for s in spy.scales] == [1.0, 1.09, 0.9919]
    spy = Spy()
    render_atoms(spy, atoms, geom, smooth=[False] * 3)
    assert spy.scales == [1.0, 2.0, 0.5]
