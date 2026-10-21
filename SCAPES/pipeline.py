"""Entry points of the V2 pipeline. build_run_config, run_dataprep, build_trainer and run_training so far; run_generation comes later."""
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


# Dataset, loaders, models, optimiser and FlowTrainer, as run_training uses them. config_dir defaults to <dataset>/config
# and must hold dataprep.gin and training.gin. resume: None, "latest", "best" or an epoch number.
def build_trainer(dataset_path, save_path, config_dir=None, resume=None):
    import random
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from SCAPES.codecs import get_codec_info
    from SCAPES.data.config_loader import load_dataprep_config, load_training_config
    from SCAPES.data.dataset import AtomSequenceDataset
    from SCAPES.data.geometry import resolve_geometry
    from SCAPES.models.factorization import LocalEncoder
    from SCAPES.models.flow import FlowModel
    from SCAPES.training.FlowModel_trainer import FlowTrainer

    dataset_path = Path(dataset_path)
    config_dir = Path(config_dir) if config_dir is not None else dataset_path / "config"
    dataprep_config = load_dataprep_config(config_dir / "dataprep.gin")
    training_config = load_training_config(config_dir / "training.gin")
    geom = resolve_geometry(dataprep_config, get_codec_info(dataprep_config.codec_name))

    # Features that only exist for encodec48k (D13) and what the AR window assumes (R2-C4)
    if geom.codec != "encodec48k" and (training_config.spectral_representation or training_config.use_discriminator):
        raise NotImplementedError("spectral_representation and use_discriminator are only available for encodec48k (D13)")
    if geom.mode == "autoregressive" and training_config.local_encoder_temporal_compression != 1:
        raise NotImplementedError("autoregressive mode needs local_encoder.temporal_compression = 1 "
                                  "(one encoded frame per latent frame, R2-C4)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\nDataset: {dataset_path}\nCodec: {geom.codec} ({geom.mode})")

    seed = training_config.seed
    loader_generator = None
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        loader_generator = torch.Generator().manual_seed(seed)

    # Dataset (CPU, the trainer moves the batches to the GPU)
    requested_keys = ["memory_buffer_latent", "target_latent", "memory_buffer_scale", "target_scale",
                      "target_semantic", "index"]
    if dataprep_config.structure_features:
        requested_keys.insert(-1, "target_structure")
    dataset = AtomSequenceDataset(dataset_path=dataset_path, config=dataprep_config, requested_keys=requested_keys,
                                  device="cpu", verbose=True)
    train_split, val_split = dataset.get_splits()
    train_loader = DataLoader(train_split, batch_size=training_config.batch_size, shuffle=True, drop_last=True,
                              generator=loader_generator)
    val_loader = DataLoader(val_split, batch_size=training_config.batch_size, shuffle=False) if len(val_split) > 0 else None
    if val_loader is None:
        print("No validation split: training without validation.")

    # Models. AR: one past window (num_past_atoms = 1, memory of W frames), target of B + o frames (R2-C4).
    if training_config.spectral_representation:
        frame_dim, frames_per_atom = 385, geom.unit_frames // 2 + 1
    else:
        frame_dim, frames_per_atom = geom.frame_dim, geom.unit_frames + geom.prefix_frames
    local_encoder = LocalEncoder(config=training_config, in_channels=frame_dim)
    flow_model = FlowModel(
        config=training_config,
        frame_dim=frame_dim,
        context_vector_dim=1024,
        num_past_atoms=geom.memory_units,
        frames_per_atom=frames_per_atom,
        memory_frames=geom.memory_frames if geom.mode == "autoregressive" else None,
        structure_dim=len(dataset.structure_feature_names),
        device=device,
    )

    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"LocalEncoder params: {count_params(local_encoder):,}")
    print(f"FlowModel params:    {count_params(flow_model):,}")

    optimizer = AdamW(list(flow_model.parameters()) + list(local_encoder.parameters()), lr=training_config.learning_rate)

    return FlowTrainer(
        model=flow_model,
        local_encoder=local_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        dataset=dataset,
        processor=None,
        optimizer=optimizer,
        config=training_config,
        model_path=save_path,
        resume_from=resume,
    )


# build_trainer + train. max_steps overrides training.max_steps. Returns save_path.
def run_training(dataset_path, save_path, config_dir=None, resume=None, max_steps=None) -> Path:
    from SCAPES.data.dataprep.semantic import build_semantics_folder
    from SCAPES.data.config_loader import load_training_config

    dataset_path = Path(dataset_path)
    config_dir = Path(config_dir) if config_dir is not None else dataset_path / "config"
    trainer = build_trainer(dataset_path, save_path, config_dir=config_dir, resume=resume)
    if max_steps is not None:
        trainer.max_steps = max_steps

    # Semantics folder from the cherry-picking CSV (for the demo)
    cherry_csv = config_dir / "cherry_picking.csv"
    if cherry_csv.exists():
        build_semantics_folder(csv_path=cherry_csv, output_dir=Path(save_path) / "semantics", dataset=trainer.dataset)

    cfg = load_training_config(config_dir / "training.gin")
    trainer.train(epochs=cfg.epochs, audio_val_freq=cfg.audio_val_freq, val_nfe=cfg.val_nfe)
    print("Training complete!")
    return Path(save_path)
