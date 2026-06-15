"""
data_generation.py — Generate a ReFlow dataset from a pretrained SCAPES model.

For each audio file in the original dataset, runs N autoregressive generations
using the pretrained flow model. Each generation step saves both the generated
atom (x1) and the white noise (x0) used as the ODE starting point.

The resulting dataset is structured identically to a regular SCAPES dataset
but with an additional `noise/` directory parallel to `atoms/`.

Usage:
    python data_generation.py \\
        --dataset /path/to/original_dataset \\
        --model /path/to/pretrained_model \\
        --output /path/to/reflow_dataset \\
        --n_runs 2

Run from anywhere (handles its own path setup).
"""

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import soundfile as sf
from tqdm import tqdm

_script_dir = Path(__file__).resolve().parent
_git_root = _script_dir.parent.parent
if str(_git_root) not in sys.path:
    sys.path.insert(0, str(_git_root))

from SCAPES.data.dataset import AtomSequenceDataset
from SCAPES.data.config_loader import load_dataprep_config
from SCAPES.inference.FlowInference import FlowInference


def build_embedding_map(
    dataset: AtomSequenceDataset,
    device: str = "cpu",
) -> Dict[str, Dict[int, torch.Tensor]]:
    """Build a mapping from filename -> target_atom_index -> CLAP embedding.

    For sequences without a precomputed embedding (first M atoms and tail),
    fills with the nearest available embedding.
    """
    M = dataset.memory_buffer_atoms
    embedding_map: Dict[str, Dict[int, torch.Tensor]] = defaultdict(dict)
    semantic_dir = dataset.annotations_dir / "semantic"

    for global_idx, (fname, start_idx) in enumerate(dataset.all_indices):
        target_atom = start_idx + M
        emb_path = semantic_dir / f"semantic_{global_idx}.pt"
        if emb_path.exists():
            embedding_map[fname][target_atom] = torch.load(
                emb_path, weights_only=True, map_location=device
            )

    for fname in dataset.filenames:
        fmap = embedding_map[fname]
        sorted_atoms = sorted(fmap.keys())
        if not sorted_atoms:
            continue
        first_atom = sorted_atoms[0]
        last_atom = sorted_atoms[-1]
        count = dataset.manifest[fname]["atoms_count"]
        for ai in range(count):
            if ai not in fmap:
                if ai < first_atom:
                    fmap[ai] = fmap[first_atom]
                elif ai > last_atom:
                    fmap[ai] = fmap[last_atom]
                else:
                    nearest = min(sorted_atoms, key=lambda x: abs(x - ai))
                    fmap[ai] = fmap[nearest]

    return embedding_map


def load_atoms_129d(dataset: AtomSequenceDataset, fname: str) -> List[torch.Tensor]:
    """Returns list of [1, 129, frames] tensors, one per atom."""
    count = dataset.manifest[fname]["atoms_count"]
    atoms = []
    for ai in range(count):
        path = dataset._get_atom_path(fname, ai)
        atom = torch.load(path, weights_only=True, map_location="cpu")
        latent = atom["latent"].float()
        scale = atom["scale"].float()
        scale_exp = scale.unsqueeze(-1).expand(-1, -1, dataset.atoms_frames)
        atoms.append(torch.cat([latent, scale_exp], dim=1))
    return atoms


@torch.no_grad()
def autoregressive_generate(
    engine: FlowInference,
    original_atoms: List[torch.Tensor],
    contexts: List[torch.Tensor],
    structures: Optional[List[torch.Tensor]],
    NFE: int = 32,
    cfg_scale: float = 3.0,
    verbose: bool = False,
):
    """Fully autoregressive generation with noise capture.

    For each atom position t:
      1. Build past buffer from generated atoms (teacher-force first M steps)
      2. Encode past -> LocalEncoder
      3. Sample noise x0 ~ N(0, I)
      4. ODE solve: x1 = model.generate(x0, encoded_past, clap_context, ...)
      5. Record (x0, x1)

    Returns two lists of tensors:
      generated_atoms: list of [1, 129, frames]  (x1, transposed)
      noises:          list of [1, frames, 129]   (x0, as used by model)
    """
    M = engine.segment_length
    total_steps = len(contexts)
    device = engine.device
    frame_dim = 129
    dummy_atom = torch.zeros(1, frame_dim, engine.atoms_frames, device=device)

    timeline = engine.build_base_timeline(
        atoms_129D=original_atoms,
        context_embeddings=contexts,
        structure_embeddings=structures,
        default_TF=False,
    )

    for t in range(min(M, total_steps)):
        timeline[t]["TF"] = True

    generated = []
    noises = []

    for t in tqdm(range(total_steps), desc="Solving ODE", disable=not verbose):
        past_atoms = []
        for i in range(t - M, t):
            if i < 0:
                past_atoms.append(dummy_atom)
            else:
                step = timeline[i]
                if step["TF"]:
                    past_atoms.append(step["atom_given"].to(device))
                else:
                    past_atoms.append(step["atom_generated"].to(device))

        past_buffer = torch.cat(past_atoms, dim=0).unsqueeze(0)
        encoded_past = engine.local_encoder(past_buffer)

        num_nulls = max(0, M - t)
        if num_nulls > 0:
            encoded_past[:, :num_nulls, :, :] = engine.model.null_past_embed

        context = timeline[t]["context_embedding"].to(device)
        if context.dim() == 1:
            context = context.unsqueeze(0)

        structure = timeline[t].get("structure_embedding")
        if structure is not None:
            structure = structure.to(device)
            if structure.dim() == 1:
                structure = structure.unsqueeze(0)

        x0 = torch.randn(1, engine.atoms_frames, frame_dim, device=device)
        pred = engine.model.generate(
            x0=x0,
            encoded_past=encoded_past,
            clap_context=context,
            structure_vector=structure,
            max_nfe=NFE,
            cfg_scale=cfg_scale,
        )

        x1_transposed = pred.transpose(1, 2)
        timeline[t]["atom_generated"] = x1_transposed

        generated.append(x1_transposed.cpu())
        noises.append(x0.cpu())

    return generated, noises


def decode_atoms_to_audio(
    engine: FlowInference,
    generated_atoms: List[torch.Tensor],
    atoms_frames: int,
    atoms_hop_frames: int,
    crossfade_frames: int,
    sr: int = 48000,
) -> torch.Tensor:
    """Decode a list of [1, 129, frames] atoms into a 2-channel audio tensor."""
    samples_per_frame = sr // 150
    segment_samples = atoms_frames * samples_per_frame
    hop_samples = atoms_hop_frames * samples_per_frame
    total_steps = len(generated_atoms)
    total_samples = (total_steps - 1) * hop_samples + segment_samples
    output_buffer = torch.zeros(1, 2, total_samples)

    macro_overlap_frames = atoms_frames - atoms_hop_frames
    crossfade_samples = crossfade_frames * samples_per_frame
    zeros_frames = macro_overlap_frames - crossfade_frames
    zeros = torch.zeros(zeros_frames * samples_per_frame)
    hann = torch.hann_window(crossfade_samples * 2)
    ones_frames = atoms_hop_frames - crossfade_frames
    ones = torch.ones(ones_frames * samples_per_frame)
    ola_window = torch.cat([zeros, hann[:crossfade_samples], ones, hann[crossfade_samples:]])
    ola_window = ola_window.view(1, 1, -1)

    alpha_smooth = 0.6
    max_jump = 1.15
    max_drop = 0.85
    prev_scale = None

    for t in range(total_steps):
        atom_129d = generated_atoms[t].to(engine.device)
        raw_scale = torch.abs(atom_129d[:, 128, :]).mean(dim=-1, keepdim=True)

        if prev_scale is None:
            smoothed_scale = raw_scale
        else:
            target_scale = torch.clamp(raw_scale, prev_scale * max_drop, prev_scale * max_jump)
            smoothed_scale = alpha_smooth * target_scale + (1.0 - alpha_smooth) * prev_scale

        prev_scale = smoothed_scale

        latent = atom_129d[:, :128, :]
        metadata = {
            "audio_scales": [smoothed_scale.squeeze(0).float()],
            "padding_mask": torch.ones(
                (1, latent.shape[-1] * samples_per_frame),
                dtype=torch.bool,
                device=engine.device,
            ),
        }
        audio = engine.processor.decode_latents_audio(latent, metadata=metadata)
        audio = (audio * ola_window.to(engine.device)).cpu()

        start = t * hop_samples
        end = start + segment_samples
        output_buffer[:, :, start:end] += audio

    return output_buffer.squeeze(0)


def duplicate_annotations(
    src_annotations_dir: Path,
    dst_annotations_dir: Path,
    n_seqs_original: int,
    n_runs: int,
    annotation_type: str,
):
    """Duplicate semantic or structure annotations for multiple runs.

    For each run r, copies semantic/structure_{i}.pt to semantic/structure_{r*N+i}.pt.
    """
    src_dir = src_annotations_dir / annotation_type
    dst_dir = dst_annotations_dir / annotation_type
    if not src_dir.exists():
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    for run in range(n_runs):
        for seq_idx in range(n_seqs_original):
            src_path = src_dir / f"{annotation_type}_{seq_idx}.pt"
            if not src_path.exists():
                continue
            dst_idx = run * n_seqs_original + seq_idx
            dst_path = dst_dir / f"{annotation_type}_{dst_idx}.pt"
            if not dst_path.exists():
                try:
                    dst_path.symlink_to(src_path.resolve())
                except OSError:
                    shutil.copy2(src_path, dst_path)


def generate_reflow_dataset(
    dataset_path: str,
    model_dir: str,
    output_path: str,
    n_runs: int = 2,
    NFE: int = 32,
    cfg_scale: float = 3.0,
    device: Optional[str] = None,
    verbose: bool = True,
):
    """Main entry point: generate a complete ReFlow dataset."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_path = Path(dataset_path).resolve()
    output_path = Path(output_path).resolve()
    model_dir = Path(model_dir).resolve()

    print(f"Original dataset: {dataset_path}")
    print(f"Output dataset:   {output_path}")
    print(f"Model:            {model_dir}")
    print(f"Device:           {device}")
    print(f"Runs per file:    {n_runs}")

    # ─── 1. Load original dataset ───
    config = load_dataprep_config(dataset_path)
    dataset = AtomSequenceDataset(
        dataset_path=str(dataset_path),
        config=config,
        requested_keys=["index"],
        device="cpu",
        verbose=verbose,
    )
    n_seqs_original = len(dataset)
    print(f"\nOriginal dataset: {len(dataset.filenames)} files, {n_seqs_original} sequences")

    # ─── 2. Load FlowInference engine ───
    engine = FlowInference(
        model_dir=str(model_dir),
        device=device,
        verbose=verbose,
        checkpoint="best",
    )

    # ─── 3. Build CLAP embedding map ───
    print("\nBuilding CLAP embedding map...")
    embedding_map = build_embedding_map(dataset, device=device)
    total_embs = sum(len(m) for m in embedding_map.values())
    print(f"Embedding coverage: {total_embs} slots across {len(embedding_map)} files")

    # ─── 4. Prepare output directories ───
    atoms_out = output_path / "atoms"
    noise_out = output_path / "noise"
    raw_out = output_path / "raw"
    annotations_out = output_path / "annotations"
    config_out = output_path / "config"

    atoms_out.mkdir(parents=True, exist_ok=True)
    noise_out.mkdir(parents=True, exist_ok=True)
    raw_out.mkdir(parents=True, exist_ok=True)
    annotations_out.mkdir(parents=True, exist_ok=True)
    config_out.mkdir(parents=True, exist_ok=True)

    # Copy config files
    for cfg_name in ["dataprep.gin", "training.gin"]:
        src = dataset_path / "config" / cfg_name
        if src.exists():
            shutil.copy2(src, config_out / cfg_name)

    # ─── 5. Generate per-file, per-run ───
    output_manifest: Dict = {
        "config_version": {
            "atoms_frames": dataset.atoms_frames,
            "atoms_hop_frames": dataset.atoms_hop_frames,
            "atoms_crossfade_frames": dataset.crossfade_frames,
            "semantic_context_seconds": dataset.context_seconds,
        },
        "files": {},
    }

    for fname in tqdm(dataset.filenames, desc="Processing files"):
        stem = Path(fname).stem
        manifest_entry = dataset.manifest[fname]
        n_atoms = manifest_entry["atoms_count"]
        raw_path = Path(manifest_entry["path"])

        original_atoms = load_atoms_129d(dataset, fname)
        fmap = embedding_map[fname]
        contexts = [fmap[ai].to(device) for ai in range(n_atoms)]

        structures = None
        if engine.use_structure:
            structure_dir = dataset.annotations_dir / "structure"
            structures = []
            for global_idx, (ff, start_idx) in enumerate(dataset.all_indices):
                if ff == fname:
                    s_path = structure_dir / f"structure_{global_idx}.pt"
                    if s_path.exists():
                        s = torch.load(s_path, weights_only=True, map_location=device)
                    else:
                        s = torch.zeros(engine.training_config.structure_dim, device=device)
                    structures.append(s)

        for run in range(n_runs):
            run_suffix = f"run_{run}"
            run_name = f"{stem}_{run_suffix}.wav"

            if verbose:
                print(f"\n--- {run_name}: {n_atoms} atoms ---")

            generated, noises = autoregressive_generate(
                engine=engine,
                original_atoms=original_atoms,
                contexts=contexts,
                structures=structures,
                NFE=NFE,
                cfg_scale=cfg_scale,
                verbose=verbose,
            )

            # Save atoms
            run_atoms_dir = atoms_out / f"{stem}_{run_suffix}"
            run_atoms_dir.mkdir(parents=True, exist_ok=True)
            for t, atom in enumerate(generated):
                gen_sq = atom.squeeze(0)
                latent = gen_sq[:128, :].unsqueeze(0).half()
                scale = gen_sq[128:129, :].mean(dim=-1, keepdim=True).half()
                save_path = run_atoms_dir / f"{stem}_{run_suffix}_atom_{t}.pt"
                torch.save({"latent": latent, "scale": scale}, save_path)

            # Save noise
            run_noise_dir = noise_out / f"{stem}_{run_suffix}"
            run_noise_dir.mkdir(parents=True, exist_ok=True)
            for t, noise in enumerate(noises):
                save_path = run_noise_dir / f"{stem}_{run_suffix}_atom_{t}.pt"
                torch.save(noise.squeeze(0), save_path)

            # Decode and save audio
            final_audio = decode_atoms_to_audio(
                engine,
                generated,
                atoms_frames=dataset.atoms_frames,
                atoms_hop_frames=dataset.atoms_hop_frames,
                crossfade_frames=dataset.crossfade_frames,
            )
            sf.write(
                str(raw_out / run_name),
                final_audio.transpose(0, 1).numpy(),
                engine.sr,
            )

            # Record in manifest
            val_info = manifest_entry.get("validation", False)
            val_starts = manifest_entry.get("val_starts", None)
            out_entry = {
                "path": str((raw_out / run_name).resolve()),
                "atoms_count": n_atoms,
                "validation": val_info,
            }
            if val_starts is not None:
                out_entry["val_starts"] = val_starts
            output_manifest["files"][run_name] = out_entry

        # Incremental manifest save
        with open(atoms_out / "manifest.json", "w") as f:
            json.dump(output_manifest, f, indent=4)

    # ─── 6. Duplicate annotations ───
    print("\nDuplicating semantic annotations...")
    duplicate_annotations(
        dataset.annotations_dir, annotations_out,
        n_seqs_original, n_runs, "semantic",
    )
    if engine.use_structure:
        print("Duplicating structure annotations...")
        duplicate_annotations(
            dataset.annotations_dir, annotations_out,
            n_seqs_original, n_runs, "structure",
        )

    # Final manifest save
    with open(atoms_out / "manifest.json", "w") as f:
        json.dump(output_manifest, f, indent=4)

    total_atoms = sum(e["atoms_count"] for e in output_manifest["files"].values())
    total_files = len(output_manifest["files"])
    print(f"\n{'='*60}")
    print(f"✅ ReFlow dataset created: {output_path}")
    print(f"   {total_files} files, {total_atoms} atoms")
    print(f"   Generated with {n_runs} run(s) per original file")
    print(f"{'='*60}")



