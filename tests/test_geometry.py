"""Slice B2: geometry in seconds (ceiling), OLA window, frame_dim and representation keys
(design "Geometry in seconds", revised after Esteban's review)."""
import warnings

import pytest
import torch

from conftest import CONFIG, TOY_COMMON, resolve_lines

# config: (mode, F|B, H|B, X|0, M, memory_frames (F|W), stride, frame_dim, has_scale, decode, key)
DEFAULTS = {
    "encodec48k_atoms": ("atoms", 48, 15, 960, 3, 48, 15, 129, True, "ola", "encodec48k__atoms-f48-h15"),
    "codicodec_atoms": ("atoms", 16, 8, 2205, 1, 16, 8, 64, False, "ola", "codicodec__whole"),
    "encodec24k_ar": ("autoregressive", 8, 8, 0, 1, 120, 8, 128, False, "stream", "encodec24k__whole"),
    "spectrostream_ar": ("autoregressive", 3, 3, 0, 1, 40, 3, 256, False, "stream", "spectrostream__whole"),
    "encodec24k_atoms": ("atoms", 24, 8, 480, 3, 24, 8, 128, False, "ola", "encodec24k__whole"),
}


def _template_geom(tmp_path, stem):
    lines = (CONFIG / "dataprep.gin").read_text().splitlines() + (CONFIG / "codecs" / f"{stem}.gin").read_text().splitlines()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return resolve_lines(tmp_path, lines, stem)[1]


def test_default_table(tmp_path):
    for stem, (mode, unit, hop, x, m, mem, stride, fd, scale, dec, _) in DEFAULTS.items():
        g = _template_geom(tmp_path, stem)
        got = (g.mode, g.unit_frames, g.unit_hop_frames, g.crossfade_samples, g.memory_units, g.memory_frames,
               g.stride_frames, g.frame_dim, g.has_scale, g.decode)
        assert got == (mode, unit, hop, x, m, mem, stride, fd, scale, dec), stem
        assert g.prefix_frames == 0
        assert g.past_frames == (m * hop if mode == "atoms" else mem), stem


def test_rounding():
    """Ceiling to the codec quantum, robust to float error (design Phi)."""
    from SCAPES.data.geometry import seconds_to_frames, seconds_to_samples
    f_cd = 44100 / 4096
    assert f_cd == 10.7666015625
    assert seconds_to_frames(0.10, 75, 1) == 8        # 7.5 -> 8
    assert seconds_to_frames(0.10, 25, 1) == 3        # 2.5 -> 3
    assert seconds_to_frames(0.10, 150, 1) == 15      # exact, not 16
    assert seconds_to_frames(0.32, 150, 1) == 48
    assert seconds_to_frames(1.6, 75, 1) == 120
    assert seconds_to_frames(1.6, 25, 1) == 40
    assert seconds_to_frames(0.07, 100, 1) == 7       # 7.000000000000001 in float must not become 8
    assert seconds_to_frames(0.071, 100, 1) == 8      # ceiling, not rounding
    assert seconds_to_frames(0.10, f_cd, 8) == 8      # one CoDiCodec chunk
    assert seconds_to_frames(0.32, f_cd, 8) == 8
    assert seconds_to_frames(1.0, f_cd, 8) == 16
    assert seconds_to_frames(1.486, f_cd, 8) == 16
    assert seconds_to_frames(1.5, f_cd, 8) == 24
    assert seconds_to_frames(0.0, 75, 1) == 1
    assert seconds_to_samples(0.02, 48000) == 960
    assert seconds_to_samples(0.05, 44100) == 2205
    assert seconds_to_samples(0.0, 48000) == 0


def test_constraints(tmp_path):
    from SCAPES.data.config_loader import ConfigError
    base = ['codec.name = "toy_noncausal"', 'generation.mode = "atoms"', "dataset.memory_buffer_atoms = 1"] + TOY_COMMON
    bad_atoms = {
        "h_gt_f": ["atoms.seconds = 0.048", "atoms.hop_seconds = 0.064", "atoms.crossfade_seconds = 0.001"],
        "x_gt_overlap": ["atoms.seconds = 0.048", "atoms.hop_seconds = 0.040", "atoms.crossfade_seconds = 0.010"],
        "x_gt_hop": ["atoms.seconds = 0.048", "atoms.hop_seconds = 0.008", "atoms.crossfade_seconds = 0.009"],
    }
    for name, lines in bad_atoms.items():
        with pytest.raises(ConfigError):
            resolve_lines(tmp_path, base + lines, name)
    # CoDiCodec: 0.32 s and 0.1 s both ceil to one chunk, so F == H and a crossfade is impossible
    with pytest.raises(ConfigError):
        resolve_lines(tmp_path, ['codec.name = "codicodec"', 'generation.mode = "atoms"', "atoms.seconds = 0.32",
                                 "atoms.hop_seconds = 0.10", "atoms.crossfade_seconds = 0.05",
                                 "dataset.memory_buffer_atoms = 1"] + TOY_COMMON, "codicodec_f_eq_h")
    ar = ['codec.name = "toy_causal"', 'generation.mode = "autoregressive"', "ar.block_seconds = 0.032",
          "ar.stride_seconds = 0.016"] + TOY_COMMON
    with pytest.raises(ConfigError):
        resolve_lines(tmp_path, ar + ["ar.memory_seconds = 0.016"], "w_lt_b")
    with pytest.raises(ConfigError):
        resolve_lines(tmp_path, ar + ["ar.memory_seconds = 0.096", "ar.prefix_seconds = 0.2"], "o_gt_w")
    # the valid neighbours resolve; W need not be a multiple of B (one window, no atom count)
    resolve_lines(tmp_path, base + ["atoms.seconds = 0.048", "atoms.hop_seconds = 0.016",
                                    "atoms.crossfade_seconds = 0.004"], "ok_atoms")
    g = resolve_lines(tmp_path, ar + ["ar.memory_seconds = 0.080"], "ok_ar")[1]
    assert (g.memory_units, g.memory_frames, g.unit_frames, g.past_frames) == (1, 80, 32, 80)


def _ola_sum(w: torch.Tensor, hop: int, k: int) -> torch.Tensor:
    out = torch.zeros((k - 1) * hop + w.shape[0], dtype=torch.float64)
    for i in range(k):
        out[i * hop: i * hop + w.shape[0]] += w.double()
    return out


def test_ola_partition_of_unity(tmp_path):
    from SCAPES.data.geometry import ola_window
    for stem in ("encodec48k_atoms", "codicodec_atoms"):
        g = _template_geom(tmp_path, stem)
        h = g.hop_length
        w = ola_window(g)
        assert w.shape == (g.unit_frames * h,)
        k = 10
        s = _ola_sum(w, g.unit_hop_frames * h, k)
        interior = s[g.unit_frames * h: (k - 1) * g.unit_hop_frames * h]
        assert interior.numel() > 0
        assert float((interior - 1).abs().max()) <= 1e-6, stem


def test_ola_equals_v1(tmp_path):
    """V1 FlowInference._build_ola_window, reimplemented: F=48, H=15, crossfade 3 frames, 320 samples/frame."""
    from SCAPES.data.geometry import ola_window
    F, H, Xf, spf = 48, 15, 3, 320
    hann = torch.hann_window(Xf * spf * 2)
    v1 = torch.cat([torch.zeros((F - H - Xf) * spf), hann[: Xf * spf], torch.ones((H - Xf) * spf), hann[Xf * spf:]])
    g = _template_geom(tmp_path, "encodec48k_atoms")
    assert torch.equal(ola_window(g).float().cpu(), v1)


def test_frame_dim_and_key(tmp_path):
    from SCAPES.data.geometry import representation_key
    for stem, (*_, fd, scale, _dec, key) in DEFAULTS.items():
        g = _template_geom(tmp_path, stem)
        assert g.frame_dim == fd
        assert g.frame_dim == g.latent_dim + int(g.has_scale)
        assert representation_key(g) == key
        assert g.encoding == key.split("__", 1)[1]
