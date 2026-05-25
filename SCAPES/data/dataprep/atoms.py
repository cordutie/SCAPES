from pathlib import Path
from typing import Optional

import librosa
import torch
import json

from SCAPES.auxiliar.encodec_wrapper import EncodecProcessor
from SCAPES.data.config_loader import DataprepConfig


def _compute_atom_counts(audio_len_samples, segment_samples, hop_samples, context_samples=None):
    if audio_len_samples < segment_samples:
        return 0, 0

    full_count = 1 + (audio_len_samples - segment_samples) // hop_samples

    if context_samples is None:
        return full_count, full_count

    if audio_len_samples < context_samples:
        return full_count, 0

    allowed_count = 1 + (audio_len_samples - context_samples) // hop_samples
    allowed_count = min(full_count, allowed_count)
    return full_count, allowed_count


def to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    elif isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_cpu(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(to_cpu(v) for v in obj)
    else:
        return obj


def extractor_atoms(
    audio_path,
    processor,
    segment_frames=48,
    hop_frames=15,
    context_samples=None,
    return_stats=False,
):
    sr = processor.sample_rate
    frame_rate = processor.frame_rate
    samples_per_frame = sr // frame_rate

    audio_input, _ = librosa.load(audio_path, sr=sr, mono=False)
    audio_input = torch.tensor(audio_input).unsqueeze(0)

    if audio_input.dim() == 2:
        audio_input = audio_input.unsqueeze(1).repeat(1, 2, 1)
    elif audio_input.shape[1] > 2:
        audio_input = audio_input[:, :2, :]

    audio_input = audio_input.to(processor.device)
    audio_input_sample = audio_input.shape[-1]

    segment_samples = segment_frames * samples_per_frame
    hop_samples = hop_frames * samples_per_frame

    full_count, allowed_count = _compute_atom_counts(
        audio_input_sample,
        segment_samples,
        hop_samples,
        context_samples=context_samples,
    )

    segments = []
    for i in range(allowed_count):
        start = i * hop_samples
        end = start + segment_samples
        segment = audio_input[:, :, start:end]
        if segment.shape[-1] < segment_samples:
            break
        segments.append(segment)

    atoms = []
    for segment in segments:
        latent, metadata = processor.audio_to_latents(segment, sr)
        atom_local = {
            "latent": torch.cat(latent, dim=-1).half(),
            "scale": metadata["audio_scales"][0].half(),
        }
        atoms.append(to_cpu(atom_local))

    if return_stats:
        return atoms, {
            "full_count": full_count,
            "kept_count": len(atoms),
            "removed_count": max(0, full_count - len(atoms)),
        }

    return atoms


def make_atom_path(audio_path, atom_index):
    audio_path = Path(audio_path)

    parts = audio_path.parts
    raw_index = parts.index("raw")

    relative_after_raw = Path(*parts[raw_index + 1:])

    stem = audio_path.stem

    base = Path(*parts[:raw_index]) / "atoms"

    parent_after_raw = relative_after_raw.parent

    atom_folder = parent_after_raw / stem

    atom_filename = stem + f"_atom_{atom_index}.pt"

    return base / atom_folder / atom_filename


def torch_save_atoms(atoms, audio_path):
    for i, atom in enumerate(atoms):
        save_path = make_atom_path(audio_path, i)

        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(atom, save_path)

    print(f"Saved {len(atoms)} atoms for {audio_path} at {save_path.parent}")
    return len(atoms)


def atoms_maker(
    dataset_path: str,
    config: Optional[DataprepConfig] = None,
    *,
    frames: Optional[int] = None,
    hop_frames: Optional[int] = None,
    context_seconds: Optional[float] = None,
    device: Optional[str] = None,
):
    if config is not None:
        frames = config.atoms_frames
        hop_frames = config.atoms_hop_frames
        context_seconds = config.semantic_context_seconds
        device = config.device

    if frames is None or hop_frames is None:
        raise TypeError("atoms_maker requires 'frames' and 'hop_frames' (via config or explicit)")

    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    sr = 48000
    processor_48k_streamable = EncodecProcessor(sr=sr, streamable=True, device=device)

    dataset_path = Path(dataset_path)
    atoms_base_path = dataset_path / "atoms"
    manifest_path = atoms_base_path / "manifest.json"
    legacy_version_file = atoms_base_path / "atoms_config_version.json"

    samples_per_frame = sr // processor_48k_streamable.frame_rate
    segment_frames = frames
    hop_frames_val = hop_frames

    active_config = {
        "atoms_frames": segment_frames,
        "atoms_hop_frames": hop_frames_val,
        "semantic_context_seconds": context_seconds,
    }

    # --- SAFETY HANDSHAKE ---
    if atoms_base_path.exists() and any(atoms_base_path.iterdir()):
        existing_config = {}
        if manifest_path.exists():
            with open(manifest_path) as f:
                existing = json.load(f)
                existing_config = existing.get("config_version", {})
        elif legacy_version_file.exists():
            with open(legacy_version_file) as f:
                existing_config = json.load(f)

        if not existing_config:
            raise RuntimeError(
                f"Existing atoms found at {atoms_base_path}, but no version file exists.\n"
                "Cannot verify compatibility. Please delete the 'atoms' folder manually."
            )

        if (existing_config.get("atoms_frames") != active_config.get("atoms_frames") or
            existing_config.get("atoms_hop_frames") != active_config.get("atoms_hop_frames")):
            raise RuntimeError(
                "CONFIG MISMATCH!\n"
                f"Current config:  {active_config}\n"
                f"Existing atoms:  {existing_config}\n"
                "Atoms at this location are incompatible. Delete the 'atoms' folder to re-generate."
            )

        existing_context = existing_config.get("semantic_context_seconds")
        if existing_context not in [None, active_config.get("semantic_context_seconds")]:
            raise RuntimeError(
                "CONFIG MISMATCH (semantic context)!\n"
                f"Current config:  {active_config}\n"
                f"Existing atoms:  {existing_config}\n"
                "Atoms at this location are incompatible. Delete the 'atoms' folder to re-generate."
            )

        if existing_context is None and context_seconds is not None:
            print("⚠️ Existing atoms were built without semantic context trimming. Tail atoms will be removed.")
        else:
            print("✅ Existing atoms match current configuration. Skipping generation for existing files.")
    else:
        atoms_base_path.mkdir(parents=True, exist_ok=True)
        print(f"Created atom storage at {atoms_base_path}")

    # List Audio Files
    dataset_raw_path = dataset_path / "raw"
    audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.ogg"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(list(dataset_raw_path.rglob(ext)))

    filenames = [f.name for f in audio_files]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Duplicate filenames detected. Ensure all names are unique across subfolders.")

    # Process Loop
    dataset_manifest = {}
    total_removed = 0
    total_full = 0
    files_trimmed = 0
    segment_samples = segment_frames * samples_per_frame
    hop_samples = hop_frames_val * samples_per_frame
    for audio_file in audio_files:
        atom_folder = make_atom_path(audio_file, 0).parent

        context_samples = None
        if context_seconds is not None:
            context_samples = int(round(context_seconds * sr))

        if atom_folder.exists() and any(atom_folder.glob("*_atom_*.pt")):
            atom_files = sorted(atom_folder.glob("*_atom_*.pt"))
            existing_count = len(atom_files)

            if context_samples is not None:
                duration = librosa.get_duration(path=audio_file, sr=sr)
                audio_len_samples = int(round(duration * sr))
                full_count, allowed_count = _compute_atom_counts(
                    audio_len_samples,
                    segment_samples,
                    hop_samples,
                    context_samples=context_samples,
                )
                total_full += full_count

                if allowed_count < existing_count:
                    for idx in range(allowed_count, existing_count):
                        atom_path = make_atom_path(audio_file, idx)
                        if atom_path.exists():
                            atom_path.unlink()
                    removed = existing_count - allowed_count
                    total_removed += removed
                    files_trimmed += 1
                    count = allowed_count
                else:
                    count = existing_count
            else:
                count = existing_count
        else:
            print(f"Processing {audio_file.name}...")
            atoms, stats = extractor_atoms(
                audio_file,
                processor_48k_streamable,
                segment_frames,
                hop_frames_val,
                context_samples=context_samples,
                return_stats=True,
            )
            torch_save_atoms(atoms, audio_file)
            count = len(atoms)
            total_full += stats["full_count"]
            total_removed += stats["removed_count"]
            if stats["removed_count"] > 0:
                files_trimmed += 1

        dataset_manifest[audio_file.name] = {
            "path": str(audio_file.resolve()),
            "atoms_count": count,
        }

    # Save Merged Manifest
    merged = {
        "config_version": {
            "atoms_frames": segment_frames,
            "atoms_hop_frames": hop_frames_val,
            "semantic_context_seconds": context_seconds,
        },
        "files": dataset_manifest,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(merged, f, indent=4)

    # Clean up legacy version file if it exists
    if legacy_version_file.exists():
        legacy_version_file.unlink()

    print(f"--- Extraction Complete ---\nManifest saved to: {manifest_path}")

    if context_seconds is not None:
        if total_removed > 0:
            print(
                "✅ Trimmed tail atoms to guarantee full semantic context windows. "
                f"Removed {total_removed} of {total_full} atoms across {files_trimmed} files "
                f"(context_seconds={context_seconds})."
            )
        else:
            print(
                "✅ All atoms already have full semantic context windows. "
                f"No trimming needed (context_seconds={context_seconds})."
            )
