"""Shared helpers for the V3 tests (pipeline/plan.md, Tests). This folder is unlocked; tests/, tests_new/ and
scripts/tests_eval/ are locked and only read here.

The toy helpers of the locked tests/conftest.py are reused through a separate module name, so that this file does not
shadow it (it is not named conftest). Toy codecs: 8 kHz, hop 8, d = 16, latent rate 1000 Hz.
"""
import dataclasses
import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location("scapes_locked_conftest", REPO / "tests" / "conftest.py")
locked = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(locked)
write_toy_wavs, stub_clap, max_abs = locked.write_toy_wavs, locked.stub_clap, locked.max_abs
TINY_TRAINING, TOY_COMMON, write_config = locked.TINY_TRAINING, locked.TOY_COMMON, locked.write_config

# Toy generation lines with the three V3 keys (design: Per-codec table; toy rows)
TOY_AR_V3 = ['codec = "toy_causal"', "conditioning_rate = 31.25", "conditioning_memory_seconds = 0.096"]        # n 32, W 96
TOY_ATOMS_V3 = ['codec = "toy_noncausal"', "conditioning_rate = 62.5", "conditioning_memory_seconds = 0.05"]    # n 16, x 4, o 16, A 36, K 3
TOY_ATOMS_XN = ['codec = "toy_noncausal"', "conditioning_rate = 1000", "conditioning_memory_seconds = 0.01"]   # n 1 = x 1, o 10, A 12, K 9
TOY_ATOMS_SNAP = ['codec = "toy_noncausal"', "conditioning_rate = 60", "conditioning_memory_seconds = 0.05"]    # snapped to n 17: x 4, o 17, A 38
LOG_EVERY_STEP = ["training.log_every = 1"]


def resolve_v3(tmp_path: Path, lines, name: str = "cfg"):
    """Writes the lines as dataprep.gin, loads and resolves them with the V3 reader. Returns (cfg, geometry)."""
    from SCAPES.codecs import get_codec_info
    from SCAPES.data.config_loader import load_dataprep_config
    from SCAPES.data.geometry import resolve_geometry
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "dataprep.gin").write_text("\n".join(list(lines)) + "\n")
    cfg = load_dataprep_config(d / "dataprep.gin")
    return cfg, resolve_geometry(cfg, get_codec_info(cfg.codec_name))


def make_toy_v3(root: Path, lines, extra_training=(), config_name: str = "config", write_wavs: bool = True) -> Path:
    if write_wavs:
        write_toy_wavs(root)
    cfg_dir = root / config_name
    write_config(cfg_dir, list(lines) + list(TOY_COMMON), list(TINY_TRAINING) + list(extra_training))
    return cfg_dir


def prepare_toy_v3(root: Path, lines, extra_training=(), config_name: str = "config", write_wavs: bool = True):
    """Toy wavs, config and dataprep with the stub CLAP. Returns (config_dir, store)."""
    from SCAPES.pipeline import run_dataprep
    cfg_dir = make_toy_v3(root, lines, extra_training, config_name, write_wavs)
    store = run_dataprep(root, config_dir=cfg_dir, device="cuda", clap_embedder=stub_clap)
    return cfg_dir, store


def load_dataset_v3(root: Path, config_dir: Path, requested_keys=None):
    from SCAPES.data.config_loader import load_dataprep_config
    from SCAPES.data.dataset import AtomSequenceDataset
    cfg = load_dataprep_config(config_dir / "dataprep.gin")
    return AtomSequenceDataset(root, cfg, requested_keys=requested_keys, device="cpu", verbose=False)


# ─── identity stand-in codec (S7) ───

class IdentityCodec:
    """Exact-inverse stand-in with the real sample rate, hop, channels and scale flag of a codec; latent_dim = h * channels.
    encode: per-frame reshape (divided by the mono RMS when the codec normalises its input); decode: the exact inverse."""

    def __new__(cls, info, device="cpu"):
        from audio_codec_wrappers import CodecWrapper

        class _Identity(CodecWrapper):
            def __init__(self, info, device):
                self.info = info
                self.device = torch.device(device)

            def _encode(self, audio):
                B, C, M = audio.shape
                h = self.info.hop_length
                scale = None
                if self.info.normalizes_input:
                    scale = audio.mean(dim=1).pow(2).mean(dim=-1).sqrt() + 1e-8
                    audio = audio / scale[:, None, None]
                z = audio.reshape(B, C, M // h, h).permute(0, 1, 3, 2).reshape(B, C * h, M // h)
                return z, scale

            def decode(self, z, scale=None, seed=0):
                z = z.to(self.device).float()
                B, D, T = z.shape
                h, C = self.info.hop_length, self.info.channels
                x = z.reshape(B, C, h, T).permute(0, 1, 3, 2).reshape(B, C, T * h)
                if self.info.normalizes_input:
                    x = x * scale.to(self.device).reshape(-1, 1, 1)
                return x

        return _Identity(info, device)


def install_identity_codecs(monkeypatch, wrappers=("encodec48k", "music2latent")):
    """Replaces get_codec / get_codec_info for the given wrapper names in SCAPES.codecs and in every loaded SCAPES module
    that imported them by name. Returns the dict of created codecs."""
    import SCAPES.codecs as sc
    import SCAPES.data.dataprep.atoms  # noqa: F401  (load the consumers before patching their names)
    import SCAPES.data.dataset  # noqa: F401
    import SCAPES.inference.FlowInference  # noqa: F401
    import SCAPES.pipeline  # noqa: F401
    orig_info, orig_codec = sc.get_codec_info, sc.get_codec
    made = {}

    def fake_info(name):
        info = orig_info(name)
        if info.name in wrappers:
            return dataclasses.replace(info, latent_dim=info.hop_length * info.channels)
        return info

    def fake_codec(name, device):
        info = fake_info(name)
        if info.name in wrappers:
            key = (info.name, str(device))
            if key not in made:
                made[key] = IdentityCodec(info, device)
            return made[key]
        return orig_codec(name, device)

    for mod in list(sys.modules.values()):
        modname = getattr(mod, "__name__", "") or ""
        if modname == "SCAPES" or modname.startswith("SCAPES."):
            if getattr(mod, "get_codec_info", None) is orig_info:
                monkeypatch.setattr(mod, "get_codec_info", fake_info)
            if getattr(mod, "get_codec", None) is orig_codec:
                monkeypatch.setattr(mod, "get_codec", fake_codec)
    return made


def write_tone(path: Path, seconds: float = 4.0, sr: int = 48000, seed: int = 0):
    """Stereo 48 kHz test signal: a 440 Hz sine (0.3) plus Gaussian noise (0.05)."""
    import numpy as np
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed)
    x = 0.3 * np.sin(2 * np.pi * 440.0 * t)[:, None] + 0.05 * rng.standard_normal((n, 2))
    sf.write(str(path), x.astype("float32"), sr, subtype="FLOAT")


def atom_geometry_ref(n: int, x_default: int, a_min: int):
    """Esteban's rule, written independently of the code: x = min(ceil(n/2), x_def), o = max(n, A_min - x - n), A = o + x + n
    (design: Summary). Returns (x, o, A)."""
    x = min(-(-n // 2), x_default)
    o = max(n, a_min - x - n)
    return x, o, o + x + n


def m_latents(rate: float, seconds: float) -> int:
    return math.ceil(rate * seconds - 1e-9)
