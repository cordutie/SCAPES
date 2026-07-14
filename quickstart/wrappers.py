# DATAPREP ----------------------------------------------------------------------------------------------------

import sys, pathlib, torch
from SCAPES.data.config_loader import load_dataprep_config
from SCAPES.data.dataprep import atoms_maker, precompute_semantic_annotations, precompute_structure_annotations
from SCAPES.data.dataset import AtomSequenceDataset

def data_preparation(DATASET_PATH):
    dataset_path = pathlib.Path(DATASET_PATH).resolve()
    config = load_dataprep_config(dataset_path / "config" / "dataprep.gin")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_structure = bool(config.structure_features)

    print(f"Dataset: {dataset_path}, Device: {device}")
    print(f"Atoms frames: {config.atoms_frames}, hop: {config.atoms_hop_frames}")
    print(f"Structure: {'enabled' if use_structure else 'disabled'}")

    # Step 1: Extract EnCodec atoms from raw audio
    atoms_maker(str(dataset_path), config=config)
    print("Atoms extracted.")

    # Step 2: Initialize dataset and create train/validation split
    dataset = AtomSequenceDataset(
        dataset_path=str(dataset_path),
        config=config,
        requested_keys=[
            "memory_buffer_latent", "target_latent",
            "memory_buffer_scale", "target_scale", "index"
        ],
        verbose=True
    )
    dataset.make_split(val_split=config.val_split_ratio, overwrite=True)
    train_split, val_split = dataset.get_splits()
    print(f"Train: {len(train_split)}, Val: {len(val_split)}")

    # Step 3: Precompute CLAP semantic annotations
    precompute_semantic_annotations(
        dataset=dataset,
        batch_size=config.precompute_batch_size,
        device=device
    )
    print("Semantic annotations done.")

    # Step 4: Visualize semantic embeddings
    from SCAPES.data.visualization import LatentSpaceExplorer

    viz_dataset = AtomSequenceDataset(
        dataset_path=str(dataset_path),
        config=config,
        requested_keys=["target_semantic", "index"],
        device="cpu"
    )
    explorer = LatentSpaceExplorer(viz_dataset, max_samples_per_file=100)
    explorer.plot_semantic(method="pca")
    explorer.plot_semantic(method="tsne")

# Training

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from pathlib import Path

from SCAPES.data.dataset import AtomSequenceDataset
from SCAPES.data.config_loader import load_dataprep_config, load_training_config
from SCAPES.data.dataprep.semantic import build_semantics_folder
from SCAPES.auxiliar.encodec_wrapper import EncodecProcessor
from SCAPES.training.FlowModel_trainer import FlowTrainer
from SCAPES.models.factorization import LocalEncoder
from SCAPES.models.flow import FlowModel

def training_loop(DATASET_PATH, MODEL_PATH):
    # DATASET INITIALIZATION ---------------------------------------------------------------------------

    resume_from = None  # or "latest", "best"

    dataprep_config = load_dataprep_config(DATASET_PATH)
    training_config = load_training_config(DATASET_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    has_structure = (Path(DATASET_PATH) / "annotations" / "structure").exists()
    requested_keys = [
        "memory_buffer_latent", "target_latent",
        "memory_buffer_scale", "target_scale",
        "target_semantic", "index"
    ]
    if has_structure:
        requested_keys.insert(-1, "target_structure")

    dataset = AtomSequenceDataset(
        dataset_path=DATASET_PATH,
        config=dataprep_config,
        requested_keys=requested_keys,
        device="cpu",
        verbose=True
    )

    train_split, val_split = dataset.get_splits()
    train_loader = DataLoader(train_split, batch_size=training_config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_split, batch_size=training_config.batch_size, shuffle=False) if len(val_split) > 0 else None

    print(f"Device: {device}, Model size: {training_config.size.upper()}")
    print(f"Epochs: {training_config.epochs}, Batch: {training_config.batch_size}, LR: {training_config.learning_rate}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader) if val_loader else 0}")

    # SEMANTIC CARD BUILDING ---------------------------------------------------------------------------
    cherry_csv = Path(DATASET_PATH) / "config" / "cherry_picking.csv"
    if cherry_csv.exists():
        build_semantics_folder(
            csv_path=cherry_csv,
            semantic_dir=Path(DATASET_PATH) / "annotations" / "semantic",
            output_dir=Path(MODEL_PATH) / "semantics",
            dataset=dataset,
        )

    # Data-derived constants
    if training_config.spectral_representation:
        frame_dim = 385
        frames_per_atom = dataset.atoms_frames // 2 + 1
    else:
        frame_dim = 129
        frames_per_atom = dataset.atoms_frames
    context_vector_dim = 1024

    print(f"Frame dim: {frame_dim}, Frames/atom: {frames_per_atom}")
    print(f"Memory buffer: {dataset.memory_buffer_atoms} atoms")
    print(f"Structure dim: {dataset.structure_feature_dimension}")

    # Initialize models
    local_encoder = LocalEncoder(config=training_config, in_channels=frame_dim)
    flow_model = FlowModel(
        config=training_config,
        frame_dim=frame_dim,
        context_vector_dim=context_vector_dim,
        num_past_atoms=dataset.memory_buffer_atoms,
        frames_per_atom=frames_per_atom,
        structure_dim=dataset.structure_feature_dimension,
        device=device
    )

    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"LocalEncoder params: {count_params(local_encoder):,}")
    print(f"FlowModel params:    {count_params(flow_model):,}")

    optimizer = AdamW(
        list(flow_model.parameters()) + list(local_encoder.parameters()),
        lr=training_config.learning_rate
    )
    processor_48k = EncodecProcessor(sr=48000, streamable=True, device=device)

    # Train the model -------------------------------------------------------------------------------
    # Train
    trainer = FlowTrainer(
        model=flow_model,
        local_encoder=local_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        dataset=dataset,
        processor=processor_48k,
        optimizer=optimizer,
        config=training_config,
        model_path=MODEL_PATH,
        resume_from=resume_from
    )

    trainer.train(
        epochs=training_config.epochs,
        audio_val_freq=training_config.audio_val_freq,
        val_nfe=training_config.val_nfe
    )
    print("Training complete!")

    # Save the final model

    zip_path = f"{MODEL_PATH.rstrip('/')}.zip"
    # zip the file
    !zip -r "{zip_path}" "{MODEL_PATH}"
    print(f"Created: {zip_path}")

