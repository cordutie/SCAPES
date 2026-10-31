"""Shared helpers for the atom tests (not a test module). V3: a toy geometry is given by the three keys only.

Toy codecs (audio_codec_wrappers): 8 kHz, hop 8, d = 16, frame rate 1000 Hz. The toy dataset of tests/conftest.py: two 10 s
48 kHz stereo wavs, a.wav (440 Hz sine + noise) and b.wav (ramp t/20). At 8 kHz a file has N = 80000 samples.

Toy atoms (tests/conftest.py TOY_ATOMS: conditioning_rate 62.5, conditioning_memory_seconds 0.05): n = 16 latents,
crossfade x = 4, overlapped o = 16, atom A = 36 latents, K = 3 memory atoms. In samples: C = 288, H = 128, X = 32.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from conftest import (TINY_TRAINING, TOY_ATOMS, TOY_COMMON, resolve_lines, stub_clap,  # noqa: E402
                      write_config, write_toy_wavs)

RATE = 8000
N_TOY = 80000                      # 10 s at 8 kHz
M_TOY = 3                          # memory atoms of the toy geometry
TOY_GEOM = {"consistent": (288, 128, 32)}      # kind -> (C, H, X) in samples


def new_type_lines(kind: str = "consistent", codec: str = "toy_noncausal", M: int = M_TOY, window=None) -> list:
    """Dataprep lines of the toy atoms geometry for the given codec (kind, M and window are kept for the callers)."""
    return [f'codec = "{codec}"'] + TOY_ATOMS[1:]


def toy_geometry(tmp_path, kind: str = "consistent", codec: str = "toy_noncausal", M: int = M_TOY, window=None, name=None):
    """Resolved Geometry of the toy atoms configuration."""
    return resolve_lines(tmp_path, new_type_lines(kind, codec) + TOY_COMMON, name or f"geom_{kind}_{codec}")[1]


def make_new_toy(root: Path, kind: str, codec: str = "toy_noncausal", M: int = M_TOY, extra_dataprep=(), extra_training=(),
                 config_name: str = "config", write_wavs: bool = True) -> Path:
    """<root>/raw/{a,b}.wav and <root>/<config_name>/{dataprep,training}.gin for a new-type toy geometry."""
    if write_wavs:
        write_toy_wavs(root)
    dp = new_type_lines(kind, codec, M) + TOY_COMMON + list(extra_dataprep)
    tr = TINY_TRAINING + ["training.standardize_latents = False", "training.past_noise = 0.0"] + list(extra_training)
    write_config(root / config_name, dp, tr)
    return root / config_name


def prepare_new_toy(root: Path, kind: str, codec: str = "toy_noncausal", M: int = M_TOY, extra_dataprep=(), extra_training=(),
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
