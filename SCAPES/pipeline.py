"""Entry points of the V2 pipeline. build_run_config and run_dataprep so far; run_training and run_generation come later."""
import random
from pathlib import Path

import numpy as np
import torch

# gin keys owned by the codec snippet: a dataset's own lines in these namespaces are dropped
_GEOMETRY_PREFIXES = ("atoms.", "ar.", "codec.", "generation.", "dataset.memory_buffer_atoms")


def _is_geometry_line(line: str) -> bool:
    key = line.split("=", 1)[0].strip() if "=" in line else ""
    return key.startswith(_GEOMETRY_PREFIXES)


# Writes out_dir/{dataprep,training}.gin: the dataset's files without their geometry lines, then each snippet, then the
# overrides (the last occurrence of a key wins). The dataset's own files are never edited.
def build_run_config(dataset_config_dir, snippets: list[Path], overrides: list[str], out_dir) -> Path:
    src, out = Path(dataset_config_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tail = []
    for snippet in snippets:
        tail += Path(snippet).read_text().splitlines()
    tail += list(overrides)
    for name in ("dataprep.gin", "training.gin"):
        kept = [ln for ln in (src / name).read_text().splitlines() if not _is_geometry_line(ln)]
        (out / name).write_text("\n".join(kept + [""] + tail) + "\n")
    return out


# Dataprep: latents, CLAP and structure arrays of every raw file into <dataset>/representations/<key>/.
# config_dir defaults to <dataset>/config and must hold dataprep.gin. clap_embedder replaces the CLAP model (tests).
# Returns the RepresentationStore.
def run_dataprep(dataset_path, config_dir=None, device=None, clap_embedder=None):
    from SCAPES.data.config_loader import load_dataprep_config
    from SCAPES.data.dataprep import latents_maker, precompute_semantic_annotations, precompute_structure_annotations
    from SCAPES.data.dataset import AtomSequenceDataset

    dataset_path = Path(dataset_path)
    config_dir = Path(config_dir) if config_dir is not None else dataset_path / "config"
    config = load_dataprep_config(config_dir / "dataprep.gin")
    if device is None or device == "auto":
        device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # seed python, numpy and torch before CLAP (random extension of short windows)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    print(f"Dataset: {dataset_path}\nConfig:  {config_dir}\nDevice:  {device}")
    print("\n[1/3] Latents...")
    store = latents_maker(dataset_path, config, device=device)

    dataset = AtomSequenceDataset(dataset_path, config, requested_keys=["index"], device="cpu", verbose=True)
    print(f"\n[2/3] CLAP annotations (batch_size={config.precompute_batch_size})...")
    precompute_semantic_annotations(dataset, embedder=clap_embedder, batch_size=config.precompute_batch_size,
                                    device=device)
    print("\n[3/3] Structure annotations...")
    precompute_structure_annotations(dataset, batch_size=config.precompute_batch_size, device=device)
    print("\nDone!")
    return store
