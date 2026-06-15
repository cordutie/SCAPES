"""
SCAPES ReFlow Training Pipeline

Trains a FlowModel + LocalEncoder on a ReFlow dataset (with paired noise).

Examples:
    python 5_reflow_training.py --path ../../datasets/microtex_reflow --save_path ../../models/microtex_reflow

All settings come from <dataset>/config/training.gin.
Run this from the `quickstart/` directory.
"""

import sys
import pathlib
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

repo_root = pathlib.Path.cwd().parent
sys.path.insert(0, str(repo_root))

from SCAPES.data.dataset import AtomSequenceDataset
from SCAPES.data.config_loader import load_dataprep_config, load_training_config
from SCAPES.auxiliar.encodec_wrapper import EncodecProcessor
from SCAPES.models.factorization import LocalEncoder
from SCAPES.models.flow import FlowModel
from SCAPES.reflow.reflow_trainer import ReFlowTrainer


def main():
    parser = argparse.ArgumentParser(
        description="SCAPES ReFlow Training — reads all settings from <dataset>/config/training.gin"
    )
    parser.add_argument("--path", type=str, required=True, help="Path to ReFlow dataset")
    parser.add_argument("--save_path", type=str, required=True, help="Where to save model checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Resume from 'latest' or 'best'")
    args = parser.parse_args()

    dataprep_config = load_dataprep_config(args.path)
    training_config = load_training_config(args.path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Dataset: {args.path}")
    print(f"Model size: {training_config.size.upper()}")
    print("Mode: ReFlow (paired noise from dataset)")

    has_structure = (pathlib.Path(args.path) / "annotations" / "structure").exists()
    requested_keys = [
        "memory_buffer_latent", "target_latent",
        "memory_buffer_scale", "target_scale",
        "target_semantic", "target_noise", "index",
    ]
    if has_structure:
        requested_keys.insert(-1, "target_structure")

    dataset = AtomSequenceDataset(
        dataset_path=args.path,
        config=dataprep_config,
        requested_keys=requested_keys,
        device="cpu",
        verbose=True,
    )
    train_split, val_split = dataset.get_splits()
    train_loader = DataLoader(
        train_split, batch_size=training_config.batch_size,
        shuffle=True, drop_last=True,
    )
    val_loader = (
        DataLoader(val_split, batch_size=training_config.batch_size, shuffle=False)
        if len(val_split) > 0 else None
    )

    if val_loader is None:
        print("No validation split — training without validation.")

    frame_dim = 129
    context_vector_dim = 1024

    local_encoder = LocalEncoder(
        config=training_config,
        in_channels=frame_dim,
    )
    flow_model = FlowModel(
        config=training_config,
        frame_dim=frame_dim,
        context_vector_dim=context_vector_dim,
        num_past_atoms=dataset.memory_buffer_atoms,
        frames_per_atom=dataset.atoms_frames,
        structure_dim=dataset.structure_feature_dimension,
        device=device,
    )

    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print(f"LocalEncoder params: {count_params(local_encoder):,}")
    print(f"FlowModel params:    {count_params(flow_model):,}")

    optimizer = AdamW(
        list(flow_model.parameters()) + list(local_encoder.parameters()),
        lr=training_config.learning_rate,
    )

    processor_48k = EncodecProcessor(sr=48000, streamable=True, device=device)

    trainer = ReFlowTrainer(
        model=flow_model,
        local_encoder=local_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        dataset=dataset,
        processor=processor_48k,
        optimizer=optimizer,
        config=training_config,
        model_path=args.save_path,
        resume_from=args.resume,
    )

    trainer.train(
        epochs=training_config.epochs,
        audio_val_freq=training_config.audio_val_freq,
        val_nfe=training_config.val_nfe,
    )
    print("ReFlow training complete!")


if __name__ == "__main__":
    main()
