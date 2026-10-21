"""Shared fixtures for the SCAPES V2 tests (plan.md, Phase B).

Toy dataset: two 10 s, 48 kHz stereo wavs:
- a.wav: a 440 Hz sine (0.3) plus Gaussian noise (0.05, seed 0);
- b.wav: a ramp x(t) = t/20 (max 0.5) on both channels, for the CLAP alignment test.
Toy codecs (audio_codec_wrappers): 8 kHz, hop 8, d = 16, frame rate 1000 Hz.
Stub CLAP embedder: [mean of the window, 0, ..., 0] (1024-d).
"""
import os
import pathlib
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pytest
import soundfile as sf
import torch

SCAPES_REPO = pathlib.Path(__file__).resolve().parents[1]
PROJECT = SCAPES_REPO.parent
FIXTURES = SCAPES_REPO / "fixtures" / "v1"          # written by scripts/make_v1_fixtures.py (slice B0)
V1_MODEL = PROJECT / "models" / "micro_water"
CONFIG = SCAPES_REPO / "SCAPES" / "config"
SCRIPTS = SCAPES_REPO / "scripts"

if str(SCAPES_REPO) not in sys.path:
    sys.path.insert(0, str(SCAPES_REPO))

SR = 48000
DURATION = 10.0
TOY_RATE = 8000
TOY_HOP = 8
TOY_D = 16

TOY_AR = [
    'codec.name = "toy_causal"',
    'generation.mode = "autoregressive"',
    "ar.block_seconds = 0.032",     # B = 32 frames (present block)
    "ar.memory_seconds = 0.096",    # W = 96 frames: ONE past window, cross-attended (Esteban review)
    "ar.stride_seconds = 0.016",    # S = 16
]
TOY_ATOMS = [
    'codec.name = "toy_noncausal"',
    'generation.mode = "atoms"',
    "atoms.seconds = 0.048",            # F = 48
    "atoms.hop_seconds = 0.016",        # H = 16
    "atoms.crossfade_seconds = 0.004",  # X = 32 samples
    "dataset.memory_buffer_atoms = 1",
]
TOY_COMMON = [
    "semantic.context_seconds = 0.5",
    "semantic.random_extension = False",
    "structure.features = []",
    "precompute.batch_size = 64",
    "splits.val_split_ratio = 0.1",
    "splits.reserve_ratio = 0.1",
    "dataset.hop_atoms = 1",
    "seed = 0",
    'device = "cuda"',
]
TINY_TRAINING = [
    "model.d_model = 64",
    "model.nhead = 4",
    "model.num_layers = 2",
    "model.dim_feedforward = 128",
    "local_encoder.hidden_dim = 32",
    "local_encoder.time_entanglement = True",
    "local_encoder.temporal_compression = 1",
    "training.epochs = 1000",
    "training.batch_size = 8",
    "training.learning_rate = 1e-3",
    "training.past_dropout = 0.25",
    "training.conditioning_dropout = 0.2",
    "training.audio_val_freq = 1000000",
    "training.val_nfe = 4",
    "training.checkpoint_freq = 0",
    "training.regularizers_and_weights = []",
    "training.log_every = 10",
    "training.seed = 0",
    "inference.cfg_scale = 3.0",
]
TOY_AR_TRAINING = ["training.standardize_latents = True", "training.past_noise = 0.1"]
TOY_ATOMS_TRAINING = ["training.standardize_latents = False", "training.past_noise = 0.0"]


def stub_clap(audio: torch.Tensor, sr: int) -> torch.Tensor:
    """[B, 2, n] at 48 kHz -> [B, 1024], dim 0 = mean of the window over channels and samples."""
    out = torch.zeros(audio.shape[0], 1024, device=audio.device, dtype=torch.float32)
    out[:, 0] = audio.float().mean(dim=(1, 2))
    return out


def write_toy_wavs(dataset: pathlib.Path) -> None:
    raw = dataset / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    n = int(DURATION * SR)
    t = np.arange(n, dtype=np.float64) / SR
    rng = np.random.default_rng(0)
    a = 0.3 * np.sin(2 * np.pi * 440.0 * t)[:, None] + 0.05 * rng.standard_normal((n, 2))
    b = np.repeat((t / 20.0)[:, None], 2, axis=1)
    sf.write(str(raw / "a.wav"), a.astype(np.float32), SR, subtype="FLOAT")
    sf.write(str(raw / "b.wav"), b.astype(np.float32), SR, subtype="FLOAT")


def write_config(config_dir: pathlib.Path, dataprep_lines, training_lines) -> pathlib.Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "dataprep.gin").write_text("\n".join(dataprep_lines) + "\n")
    (config_dir / "training.gin").write_text("\n".join(training_lines) + "\n")
    return config_dir


def make_toy_dataset(root: pathlib.Path, mode: str = "autoregressive", extra_dataprep=(), extra_training=(),
                     config_name: str = "config", write_wavs: bool = True) -> pathlib.Path:
    """Creates <root>/raw/{a,b}.wav and <root>/<config_name>/{dataprep,training}.gin. Returns root."""
    if write_wavs:
        write_toy_wavs(root)
    if mode == "autoregressive":
        dp = TOY_AR + TOY_COMMON + list(extra_dataprep)
        tr = TINY_TRAINING + TOY_AR_TRAINING + list(extra_training)
    else:
        dp = TOY_ATOMS + TOY_COMMON + list(extra_dataprep)
        tr = TINY_TRAINING + TOY_ATOMS_TRAINING + list(extra_training)
    write_config(root / config_name, dp, tr)
    return root


def prepare_toy(root: pathlib.Path, mode: str = "autoregressive", extra_dataprep=(), extra_training=()):
    """Toy dataset plus dataprep with the stub CLAP. Returns (dataset_path, store)."""
    from SCAPES.pipeline import run_dataprep
    make_toy_dataset(root, mode, extra_dataprep, extra_training)
    store = run_dataprep(root, config_dir=root / "config", device="cuda", clap_embedder=stub_clap)
    return root, store


def load_toy_dataset(root: pathlib.Path, requested_keys=None, config_dir=None):
    from SCAPES.data.config_loader import load_dataprep_config
    from SCAPES.data.dataset import AtomSequenceDataset
    cfg = load_dataprep_config((config_dir or root / "config") / "dataprep.gin")
    return AtomSequenceDataset(root, cfg, requested_keys=requested_keys, device="cpu", verbose=False)


def resolve_lines(tmp_path: pathlib.Path, lines, name: str = "cfg"):
    """Writes the lines as dataprep.gin, loads and resolves them. Returns (cfg, geometry)."""
    from SCAPES.codecs import get_codec_info
    from SCAPES.data.config_loader import load_dataprep_config
    from SCAPES.data.geometry import resolve_geometry
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "dataprep.gin").write_text("\n".join(lines) + "\n")
    cfg = load_dataprep_config(d / "dataprep.gin")
    return cfg, resolve_geometry(cfg, get_codec_info(cfg.codec_name))


def train_toy(root: pathlib.Path, mode: str = "autoregressive", steps: int = 5, extra_dataprep=(), extra_training=(),
              save_name: str = "model") -> pathlib.Path:
    """Toy dataset + dataprep + a short run_training. Returns the model directory."""
    from SCAPES.pipeline import run_training
    prepare_toy(root, mode, extra_dataprep, extra_training)
    return run_training(root, root / save_name, config_dir=root / "config", max_steps=steps)


def max_abs(a, b) -> float:
    a = torch.as_tensor(a).detach().double().cpu()
    b = torch.as_tensor(b).detach().double().cpu()
    assert a.shape == b.shape, f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"
    return float((a - b).abs().max()) if a.numel() else 0.0


@pytest.fixture
def toy_ar(tmp_path):
    return prepare_toy(tmp_path / "toy_ar", "autoregressive")


@pytest.fixture
def toy_atoms(tmp_path):
    return prepare_toy(tmp_path / "toy_atoms", "atoms")
