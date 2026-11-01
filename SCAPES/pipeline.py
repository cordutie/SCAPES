"""Entry points of the V2 pipeline: build_run_config, run_dataprep, build_trainer, run_training and run_generation."""
import dataclasses
import math
import random
from pathlib import Path

import numpy as np
import torch

# gin keys owned by the codec snippet: a dataset's own lines in these namespaces are dropped
_GEOMETRY_KEYS = ("codec", "conditioning_rate", "conditioning_memory_seconds")
_GEOMETRY_PREFIXES = ("atoms.", "ar.", "codec.", "generation.", "resolved.", "dataset.memory_buffer_atoms")  # noqa: rejected key kept so old lines are dropped


def _is_geometry_line(line: str) -> bool:
    key = line.split("=", 1)[0].strip() if "=" in line else ""
    return key in _GEOMETRY_KEYS or key.startswith(_GEOMETRY_PREFIXES)


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


# Dataprep: latents and CLAP arrays of every raw file into <dataset>/representations/<key>/.
# config_dir defaults to <dataset>/config and must hold dataprep.gin. clap_embedder replaces the CLAP model (tests).
# Returns the RepresentationStore.
def run_dataprep(dataset_path, config_dir=None, device=None, clap_embedder=None):
    from SCAPES.data.config_loader import load_dataprep_config
    from SCAPES.data.dataprep import latents_maker, precompute_semantic_annotations
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
    print("\n[1/2] Latents...")
    store = latents_maker(dataset_path, config, device=device)

    dataset = AtomSequenceDataset(dataset_path, config, requested_keys=["index"], device="cpu", verbose=True)
    print(f"\n[2/2] CLAP annotations (batch_size={config.precompute_batch_size})...")
    precompute_semantic_annotations(dataset, embedder=clap_embedder, batch_size=config.precompute_batch_size,
                                    device=device)
    print("\nDone!")
    return store


# Dataset, loaders, models, optimiser and FlowTrainer, as run_training uses them. config_dir defaults to <dataset>/config
# and must hold dataprep.gin and training.gin. resume: None, "latest", "best" or an epoch number.
def build_trainer(dataset_path, save_path, config_dir=None, resume=None):
    import random
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from SCAPES.codecs import get_codec_info
    from SCAPES.generation import CODECS, print_resolution
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
    print_resolution(geom.resolution)

    # standardize_latents and past_noise absent from training.gin: the codec's default from the table
    spec = CODECS[geom.codec]
    if training_config.standardize_latents is None:
        training_config = dataclasses.replace(training_config, standardize_latents=spec.standardize_latents)
    if training_config.past_noise is None:
        training_config = dataclasses.replace(training_config, past_noise=spec.past_noise)

    # Features that only exist for encodec48k (D13) and what the AR window assumes (R2-C4)
    if geom.codec != "encodec48" and training_config.use_discriminator:
        raise NotImplementedError("use_discriminator is only available for encodec48k (D13)")
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
    dataset = AtomSequenceDataset(dataset_path=dataset_path, config=dataprep_config, requested_keys=requested_keys,
                                  device="cpu", verbose=True)
    train_split, val_split = dataset.get_splits()
    # "auto" starts at 128 (the value without CUDA) and is replaced by the measured batch after the trainer is built
    auto_batch_size = training_config.batch_size == "auto"

    def make_loaders(batch):
        train = DataLoader(train_split, batch_size=batch, shuffle=True, drop_last=True, generator=loader_generator)
        val = DataLoader(val_split, batch_size=batch, shuffle=False) if len(val_split) > 0 else None
        return train, val

    train_loader, val_loader = make_loaders(128 if auto_batch_size else training_config.batch_size)
    if val_loader is None:
        print("No validation split: training without validation.")

    # Models. AR: one past window (num_past_atoms = 1, memory of W frames), target of B + o frames (R2-C4).
    frame_dim, frames_per_atom = geom.frame_dim, geom.unit_frames + geom.prefix_frames
    local_encoder = LocalEncoder(config=training_config, in_channels=frame_dim)
    flow_model = FlowModel(
        config=training_config,
        frame_dim=frame_dim,
        context_vector_dim=1024,
        num_past_atoms=geom.memory_units,
        frames_per_atom=frames_per_atom,
        memory_frames=geom.memory_frames if geom.mode == "autoregressive" else None,
        device=device,
    )

    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"LocalEncoder params: {count_params(local_encoder):,}")
    print(f"FlowModel params:    {count_params(flow_model):,}")

    optimizer = AdamW(list(flow_model.parameters()) + list(local_encoder.parameters()), lr=training_config.learning_rate)

    trainer = FlowTrainer(
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
    if auto_batch_size:
        if torch.cuda.is_available():
            trainer.batch_size = _auto_training_batch(trainer, train_split)
            trainer.train_loader, trainer.val_loader = make_loaders(trainer.batch_size)
        else:
            trainer.batch_size = 128   # the loaders already use 128
    return trainer


# Items per probe unit: step(b) runs PROBE_UNIT * b items. At 2 to 4 items the peak is dominated by fixed allocations
# (measured on the smoke dataset: flat at 311 MiB, against about 3.4 MiB per item from 128 to 486 items).
PROBE_UNIT = 64


# One training step (loss and backward, no optimizer step); the gradients are cleared after it.
# The same loss as FlowTrainer.train, for one batch.
def _training_probe(trainer, split):
    from torch.utils.data import default_collate
    from SCAPES.auxiliar.losses_flow import flow_matching_loss

    def step(b):
        batch = default_collate([split[i % len(split)] for i in range(PROBE_UNIT * b)])
        past, present, context = trainer._prepare_batch(batch)
        encoded_past, prefix = trainer._encode_past(past, train=True)
        noise = torch.randn_like(present)
        loss = flow_matching_loss(trainer.model, noise, present, context, encoded_past,
                                  n_latent=trainer.n_latent, has_scale=trainer.has_scale, prefix=prefix)[0]
        loss.backward()
        trainer.optimizer.zero_grad(set_to_none=True)
    return step


# Batch size of "auto" on a CUDA device. Prints the choice; the random state is restored, so the seed still applies.
def _auto_training_batch(trainer, split):
    from SCAPES.auxiliar.gpu_batch import auto_batch
    # AdamW keeps two float states per parameter; they are allocated at the first optimizer step, not by the probe
    params = [p for group in trainer.optimizer.param_groups for p in group["params"]]
    adam_bytes = 2 * sum(p.numel() * p.element_size() for p in params)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        # capped at 128, and at the split: a larger batch gives no step at all (drop_last);
        # floor 2: a batch of 1 is not a useful probe of the memory
        batch, free, fixed, per_item = auto_batch(_training_probe(trainer, split), cap=min(128, len(split)),
                                                  extra_fixed_bytes=adam_bytes, items_per_call=PROBE_UNIT, floor=2)
    print(f"batch_size auto -> {batch} (free {free / 2**30:.2f} GiB, fixed {fixed / 2**20:.0f} MiB, "
          f"per item {per_item / 2**20:.1f} MiB)")
    return batch


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
    # with a step budget the epoch count must not cut the run short: every epoch has at least one step
    epochs = cfg.epochs if trainer.max_steps is None else max(cfg.epochs, trainer.max_steps)
    trainer.train(epochs=epochs, audio_val_freq=cfg.audio_val_freq, val_nfe=cfg.val_nfe)
    print("Training complete!")
    return Path(save_path)


# Number of generation units that cover `seconds` of audio: AR blocks of B frames, or atoms placed H frames apart
def units_for_seconds(geom, seconds: float) -> int:
    frames = seconds * geom.frame_rate
    if geom.mode == "autoregressive":
        return max(1, math.ceil(frames / geom.unit_frames - 1e-6))
    return max(1, math.ceil((seconds * geom.sample_rate - geom.atom_samples) / geom.hop_samples - 1e-6) + 1)


# Generates audio from CLAP prompts with a trained model and writes a wav (codec rate). One prompt per unit: a single
# prompt is held for every unit; fewer prompts than units hold the last one. The number of units is unit_count, or
# enough for `seconds`, or len(prompts). out_path without a suffix is a folder and the file is generation.wav in it.
# checkpoint "best" falls back to "last" when training had no validation split. Returns the path of the wav.
def run_generation(model_dir, out_path, prompts: list, unit_count=None, seconds=None, seed: int = 0, nfe: int = 16,
                   cfg_scale: float = 1.0, checkpoint="best") -> Path:
    import soundfile as sf
    from SCAPES.inference.FlowInference import FlowInference

    if not prompts:
        raise ValueError("run_generation needs at least one prompt")
    engine = FlowInference(model_dir, checkpoint=checkpoint)
    if unit_count is None:
        unit_count = units_for_seconds(engine.geom, seconds) if seconds is not None else len(prompts)
    clap = [torch.as_tensor(prompts[min(k, len(prompts) - 1)]).reshape(-1).float() for k in range(unit_count)]

    out_path = Path(out_path)
    if out_path.suffix == "":
        out_path = out_path / "generation.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio = engine.generate_soundscape(clap, nfe=nfe, cfg_scale=cfg_scale, seed=seed)
    sf.write(str(out_path), audio.transpose(0, 1).numpy(), engine.sr)
    print(f"Generated {unit_count} units ({audio.shape[-1] / engine.sr:.2f} s) -> {out_path}")
    return out_path
