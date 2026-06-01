import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DataprepConfig:
    atoms_frames: int = 48
    atoms_hop_frames: int = 15
    atoms_crossfade_frames: int = 3
    semantic_context_seconds: float = 1.0
    semantic_random_extension: bool = True
    structure_features: list = field(default_factory=list)
    precompute_batch_size: int = 128
    val_split_ratio: float = 0.1
    train_split: Optional[str] = None
    val_split: Optional[str] = None
    memory_buffer_atoms: int = 3
    hop_atoms: int = 1
    seed: int = 42
    device: str = "auto"


_GIN_MAP = [
    (["atoms", "frames"], "atoms_frames"),
    (["atoms", "hop_frames"], "atoms_hop_frames"),
    (["atoms", "crossfade_frames"], "atoms_crossfade_frames"),
    (["semantic", "context_seconds"], "semantic_context_seconds"),
    (["semantic", "random_extension"], "semantic_random_extension"),
    (["structure", "features"], "structure_features"),
    (["precompute", "batch_size"], "precompute_batch_size"),
    (["splits", "val_split_ratio"], "val_split_ratio"),
    (["splits", "train_split"], "train_split"),
    (["splits", "val_split"], "val_split"),
    (["dataset", "memory_buffer_atoms"], "memory_buffer_atoms"),
    (["dataset", "hop_atoms"], "hop_atoms"),
    (["seed"], "seed"),
    (["device"], "device"),
]


def _set_nested(config, dotted_key, value):
    keys = [k for k in dotted_key.split(".") if k]
    current = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    if keys:
        current[keys[-1]] = value


def _parse_gin_value(raw_value):
    value = raw_value.strip()
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ["none", "null"]:
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _strip_inline_comment(line):
    result = []
    in_single = False
    in_double = False
    escape = False
    for ch in line:
        if escape:
            result.append(ch)
            escape = False
            continue
        if ch == "\\":
            result.append(ch)
            escape = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        if ch == "#" and not in_single and not in_double:
            break
        result.append(ch)
    return "".join(result).strip()


def _is_value_complete(value):
    stack = []
    in_single = False
    in_double = False
    escape = False
    for ch in value:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if in_single or in_double:
            continue
        if ch in "[({":
            stack.append(ch)
        elif ch in "])}":
            if stack:
                stack.pop()
    return not stack and not in_single and not in_double


def parse_gin_config(path: Path):
    config = {}
    current_key = None
    value_parts = []
    with open(path, "r") as f:
        for raw_line in f:
            line = _strip_inline_comment(raw_line)
            if not line:
                continue

            if current_key is None:
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key:
                    continue
                current_key = key
                value_parts = [value]
            else:
                value_parts.append(line.strip())

            value_str = " ".join(value_parts)
            if _is_value_complete(value_str):
                _set_nested(config, current_key, _parse_gin_value(value_str))
                current_key = None
                value_parts = []

    if current_key is not None and value_parts:
        value_str = " ".join(value_parts)
        _set_nested(config, current_key, _parse_gin_value(value_str))

    return config


def load_config(gin_path: Path):
    if gin_path.is_dir():
        gin_path = gin_path / "dataprep.gin"
    if not gin_path.exists():
        return {}, None
    return parse_gin_config(gin_path), gin_path


def load_dataprep_config(gin_path: Path) -> DataprepConfig:
    path = Path(gin_path)
    if path.is_dir():
        candidate = path / "config" / "dataprep.gin"
        if candidate.exists():
            path = candidate
        else:
            path = path / "dataprep.gin"
    if not path.exists():
        raise FileNotFoundError(f"Gin config not found: {path}")

    raw = parse_gin_config(path)
    cfg = DataprepConfig()
    for keys, attr in _GIN_MAP:
        val = _get_nested(raw, keys, None)
        if val is not None:
            setattr(cfg, attr, val)
    return cfg


def _get_nested(config, keys, default=None):
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


# ─── Training Config ────────────────────────────────────────────────────────────

_SIZE_TABLE = {
    "small":       {"d_model": 512,  "num_layers": 6,  "nhead": 8,  "dim_feedforward": 2048},
    "medium":      {"d_model": 768,  "num_layers": 8,  "nhead": 12, "dim_feedforward": 2048},
    "large":       {"d_model": 1024, "num_layers": 12, "nhead": 16, "dim_feedforward": 2048},
    "extra_large": {"d_model": 1280, "num_layers": 16, "nhead": 20, "dim_feedforward": 2048},
    "audiobox":    {"d_model": 1024, "num_layers": 24, "nhead": 16, "dim_feedforward": 4096},
}


@dataclass
class TrainingConfig:
    # Model size — picks from the size table (small/medium/large/extra_large/audiobox).
    # Either 'size' OR all of (d_model, nhead, num_layers, dim_feedforward) must be set.
    size: Optional[str] = None

    # FlowModel architecture (architecture only — data-derived params like
    # frame_dim, frames_per_atom, num_past_atoms, structure_dim etc. are
    # read from the dataset at runtime)
    d_model: Optional[int] = None
    nhead: Optional[int] = None
    num_layers: Optional[int] = None
    dim_feedforward: Optional[int] = None

    # LocalEncoder architecture
    local_encoder_hidden_dim: int = 256
    local_encoder_time_entanglement: bool = True
    local_encoder_temporal_compression: int = 1

    # Training loop
    epochs: int = 75
    batch_size: int = 128
    learning_rate: float = 1e-4
    past_dropout: float = 0.25
    conditioning_dropout: float = 0.2
    audio_val_freq: int = 15
    val_nfe: int = 16
    context_source: str = "clap"

    # Checkpointing
    checkpoint_freq: int = 15       # 0 = only best/last, no epoch checkpoints
    save_resume_states: bool = False  # whether to save optimizer/trainer states

    # Validation audio
    val_files: Optional[list] = None  # filenames, ["all"], or ["random=N"]; None = skip
    val_duration: float = 5.0

    # Inference defaults
    cfg_scale: float = 3.0

    # Adversarial discriminator (True = enable 2-stage training)
    use_discriminator: bool = False
    disc_epochs: int = 10
    stage2_epochs: int = 10

    # Regularizers: list of [name, weight] pairs, e.g. [["time_phase", 0.1], ["fft_phase", 0.05]]
    regularizers_and_weights: Optional[list] = None


_TRAINING_GIN_MAP = [
    (["model", "size"], "size"),
    (["model", "d_model"], "d_model"),
    (["model", "nhead"], "nhead"),
    (["model", "num_layers"], "num_layers"),
    (["model", "dim_feedforward"], "dim_feedforward"),
    (["local_encoder", "hidden_dim"], "local_encoder_hidden_dim"),
    (["local_encoder", "time_entanglement"], "local_encoder_time_entanglement"),
    (["local_encoder", "temporal_compression"], "local_encoder_temporal_compression"),
    (["training", "epochs"], "epochs"),
    (["training", "batch_size"], "batch_size"),
    (["training", "learning_rate"], "learning_rate"),
    (["training", "past_dropout"], "past_dropout"),
    (["training", "conditioning_dropout"], "conditioning_dropout"),
    (["training", "audio_val_freq"], "audio_val_freq"),
    (["training", "val_nfe"], "val_nfe"),
    (["training", "context_source"], "context_source"),
    (["training", "checkpoint_freq"], "checkpoint_freq"),
    (["training", "save_resume_states"], "save_resume_states"),
    (["training", "val_files"], "val_files"),
    (["training", "val_duration"], "val_duration"),
    (["inference", "cfg_scale"], "cfg_scale"),
    (["training", "use_discriminator"], "use_discriminator"),
    (["training", "stage2_epochs"], "stage2_epochs"),
    (["training", "regularizers_and_weights"], "regularizers_and_weights"),
]


def _resolve_size_table(cfg: TrainingConfig) -> TrainingConfig:
    explicit = all(v is not None for v in [cfg.d_model, cfg.nhead, cfg.num_layers, cfg.dim_feedforward])

    if cfg.size is not None:
        if explicit:
            raise ValueError(
                "Provide either 'model.size' OR explicit arch params (d_model, nhead, num_layers, dim_feedforward), not both."
            )
        if cfg.size not in _SIZE_TABLE:
            raise ValueError(f"Invalid size '{cfg.size}'. Choose from: {list(_SIZE_TABLE.keys())}")
        table = _SIZE_TABLE[cfg.size]
        cfg.d_model = table["d_model"]
        cfg.nhead = table["nhead"]
        cfg.num_layers = table["num_layers"]
        cfg.dim_feedforward = table["dim_feedforward"]
    else:
        if not explicit:
            raise ValueError(
                "Either 'model.size' or all of (d_model, nhead, num_layers, dim_feedforward) must be provided."
            )

    return cfg


def load_training_config(gin_path: Path) -> TrainingConfig:
    path = Path(gin_path)
    if path.is_dir():
        candidate = path / "config" / "training.gin"
        if candidate.exists():
            path = candidate
        else:
            path = path / "training.gin"
    if not path.exists():
        raise FileNotFoundError(f"Gin config not found: {path}")

    raw = parse_gin_config(path)
    cfg = TrainingConfig()
    for keys, attr in _TRAINING_GIN_MAP:
        val = _get_nested(raw, keys, None)
        if val is not None:
            setattr(cfg, attr, val)
    _resolve_size_table(cfg)
    return cfg
