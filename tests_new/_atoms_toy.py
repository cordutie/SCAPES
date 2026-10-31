"""Shared helpers for the consistent / independent atom tests (plan.md, Tests). Not a test module.

Toy codecs (audio_codec_wrappers): 8 kHz, hop 8, d = 16, frame rate 1000 Hz. The toy dataset of tests/conftest.py: two 10 s
48 kHz stereo wavs, a.wav (440 Hz sine + noise) and b.wav (ramp t/20). At 8 kHz a file has N = 80000 samples.

Toy geometries (design.md, plan.md Tests):
  consistent:  atoms.seconds 0.016 -> F = 16, C = 128; hop 0.0045 s -> H = 36 (off the 8-sample frame grid, so the
               legacy frame geometry cannot reproduce it); crossfade 0.002 s -> X = 16
  independent: atoms.seconds 0.016 -> C = 128; overlap normal -> X = 16, H = 112; brute -> X = 64, H = 64
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from conftest import (TINY_TRAINING, TOY_ATOMS_TRAINING, TOY_COMMON, resolve_lines, stub_clap,  # noqa: E402
                      write_config, write_toy_wavs)

# Phases (design D24): phase 1 = consistent atoms; phase 2 = independent atoms, postponed by Esteban (2026-10-30).
# Phase-2 tests are skipped unless SCAPES_PHASE=2 is set; at phase-2 checkpoints every one of them must pass.
PHASE = int(os.environ.get("SCAPES_PHASE", "1"))
phase2 = pytest.mark.skipif(PHASE < 2, reason="phase 2: independent atoms, postponed by Esteban (2026-10-30)")
phase1_only = pytest.mark.skipif(PHASE >= 2, reason="phase-1 guard: independent atoms are implemented in phase 2")

RATE = 8000
N_TOY = 80000                      # 10 s at 8 kHz
TOY_GEOM = {                       # kind -> (C, H, X)
    "consistent": (128, 36, 16),
    "independent": (128, 112, 16),
    "brute": (128, 64, 64),
}


def new_type_lines(kind: str, codec: str = "toy_noncausal", M: int = 2, window: str = "hann") -> list:
    """Dataprep lines of a toy new-type geometry (kind: consistent | independent (normal) | brute)."""
    lines = [f'codec.name = "{codec}"', 'generation.mode = "atoms"', f"dataset.memory_buffer_atoms = {M}",
             "atoms.seconds = 0.016"]
    if kind == "consistent":
        lines += ['atoms.type = "consistent"', "atoms.hop_seconds = 0.0045", "atoms.crossfade_seconds = 0.002"]
    elif kind == "independent":
        lines += ['atoms.type = "independent"', 'atoms.overlap = "normal"']
    elif kind == "brute":
        lines += ['atoms.type = "independent"', 'atoms.overlap = "brute"']
    else:
        raise ValueError(kind)
    if window is not None:
        lines.append(f'atoms.window = "{window}"')
    return lines


def toy_geometry(tmp_path, kind: str, codec: str = "toy_noncausal", M: int = 2, window: str = "hann", name=None):
    """Resolved Geometry of a toy new-type configuration."""
    return resolve_lines(tmp_path, new_type_lines(kind, codec, M, window) + TOY_COMMON, name or f"geom_{kind}_{codec}")[1]


def make_new_toy(root: Path, kind: str, codec: str = "toy_noncausal", M: int = 2, extra_dataprep=(), extra_training=(),
                 config_name: str = "config", write_wavs: bool = True) -> Path:
    """<root>/raw/{a,b}.wav and <root>/<config_name>/{dataprep,training}.gin for a new-type toy geometry."""
    if write_wavs:
        write_toy_wavs(root)
    dp = new_type_lines(kind, codec, M) + TOY_COMMON + list(extra_dataprep)
    tr = TINY_TRAINING + TOY_ATOMS_TRAINING + list(extra_training)
    write_config(root / config_name, dp, tr)
    return root / config_name


def prepare_new_toy(root: Path, kind: str, codec: str = "toy_noncausal", M: int = 2, extra_dataprep=(), extra_training=(),
                    config_name: str = "config", write_wavs: bool = True):
    """make_new_toy + dataprep with the stub CLAP. Returns (config_dir, store)."""
    from SCAPES.pipeline import run_dataprep
    cfg_dir = make_new_toy(root, kind, codec, M, extra_dataprep, extra_training, config_name, write_wavs)
    store = run_dataprep(root, config_dir=cfg_dir, device="cuda", clap_embedder=stub_clap)
    return cfg_dir, store


def load_dataset(root: Path, config_dir: Path, requested_keys=None):
    from SCAPES.data.config_loader import load_dataprep_config
    from SCAPES.data.dataset import AtomSequenceDataset
    cfg = load_dataprep_config(config_dir / "dataprep.gin")
    return AtomSequenceDataset(root, cfg, requested_keys=requested_keys, device="cpu", verbose=False)


def prepared_audio(codec, path):
    """The file as the codec sees it: codec.prepare of the raw audio, [1, ch, N] at the codec rate (as latents_maker)."""
    from audio_codec_wrappers.audio import load_audio
    audio, sr = load_audio(path)
    return codec.prepare(audio, sr)
