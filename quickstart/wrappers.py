"""Two calls for the quickstart notebook: data_preparation and training_loop.

Both merge <dataset>/config with the three keys of SCAPES/config/generation.gin (codec, conditioning_rate,
conditioning_memory_seconds) into <dataset>/run_config/, and use that. Pass the keys to change them.
"""
import shutil
from pathlib import Path

from SCAPES.pipeline import build_run_config, run_dataprep, run_training

GENERATION_GIN = Path(__file__).resolve().parent.parent / "SCAPES" / "config" / "generation.gin"


# The dataset's config plus generation.gin and the overrides given, written to <dataset>/run_config
def _run_config(dataset_path, codec, conditioning_rate, conditioning_memory_seconds):
    overrides = []
    if codec is not None:
        overrides.append(f'codec = "{codec}"')
    if conditioning_rate is not None:
        overrides.append(f"conditioning_rate = {conditioning_rate}")
    if conditioning_memory_seconds is not None:
        overrides.append(f"conditioning_memory_seconds = {conditioning_memory_seconds}")
    dataset_path = Path(dataset_path)
    out = build_run_config(dataset_path / "config", [GENERATION_GIN], overrides, dataset_path / "run_config")
    cherry = dataset_path / "config" / "cherry_picking.csv"
    if cherry.exists():
        shutil.copy(cherry, out / cherry.name)
    return out


def data_preparation(dataset_path, codec=None, conditioning_rate=None, conditioning_memory_seconds=None):
    config_dir = _run_config(dataset_path, codec, conditioning_rate, conditioning_memory_seconds)
    return run_dataprep(dataset_path, config_dir=config_dir)


def training_loop(dataset_path, model_path, codec=None, conditioning_rate=None, conditioning_memory_seconds=None):
    config_dir = _run_config(dataset_path, codec, conditioning_rate, conditioning_memory_seconds)
    return run_training(dataset_path, model_path, config_dir=config_dir)
