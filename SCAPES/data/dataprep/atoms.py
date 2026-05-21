from pathlib import Path

import librosa
import torch
import json

from SCAPES.auxiliar.encodec_wrapper import EncodecProcessor
from SCAPES.data.config_loader import load_config

def _get_config_value(config, keys, default=None):
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


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


# Function to send metadata to CPU (to save things in cpu()
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


# Takes a file and makes a list of its atoms
def extractor_atoms(
    audio_path,
    processor,
    segment_frames=48,
    hop_frames=15,
    context_samples=None,
    return_stats=False,
):
    # Use sample rate from processor
    sr = processor.sample_rate
    frame_rate = processor.frame_rate
    samples_per_frame = sr // frame_rate

    # Load audio
    audio_input, _ = librosa.load(audio_path, sr=sr, mono=False)
    audio_input = torch.tensor(audio_input).unsqueeze(0)  # Add batch dimension

    # If mono, make it stereo by duplicating the mono channel to create a stereo signal (2 channels)
    if audio_input.dim() == 2:
        audio_input = audio_input.unsqueeze(1).repeat(1, 2, 1)
    # If it has more than 2 channels, take only the first 2 channels
    elif audio_input.shape[1] > 2:
        audio_input = audio_input[:, :2, :]

    audio_input = audio_input.to(processor.device)
    audio_input_sample = audio_input.shape[-1]

    # Calculate segment and hop directly in samples
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

    # Atoms extraction
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


# Makes the path for the atoms to be saved, based on the original audio path and the atom index. It replaces
# "raw" with "atoms" and adds "_atom_{index}" to the filename.
def make_atom_path(audio_path, atom_index):
    audio_path = Path(audio_path)

    parts = audio_path.parts
    raw_index = parts.index("raw")

    # Path after raw/
    relative_after_raw = Path(*parts[raw_index + 1 :])

    # Original filename without extension
    stem = audio_path.stem

    # Build new structure:
    # Replace raw -> atoms
    base = Path(*parts[:raw_index]) / "atoms"

    # Remove original filename from relative path
    parent_after_raw = relative_after_raw.parent

    # Create folder named after original filename
    atom_folder = parent_after_raw / stem

    # New atom filename
    atom_filename = stem + f"_atom_{atom_index}.pt"

    return base / atom_folder / atom_filename


# Saves the atoms to disk using torch.save, in the path defined by make_atom_path.
def torch_save_atoms(atoms, audio_path):
    for i, atom in enumerate(atoms):
        save_path = make_atom_path(audio_path, i)

        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save
        torch.save(atom, save_path)

    print(f"Saved {len(atoms)} atoms for {audio_path} at {save_path.parent}")
    return len(atoms)


# Full pipeline to compute the atoms for a dataset. It takes the path to the dataset, processes all audio files
# in the "raw" folder, and saves the atoms in the "atoms" folder. It also saves a dataset.json file with the
# path and number of atoms for each audio file.
def atoms_maker(dataset_path):
    # Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize Processor
    sr = 48000
    processor_48k_streamable = EncodecProcessor(sr=sr, streamable=True, device=device)

    dataset_path = Path(dataset_path)
    atoms_base_path = dataset_path / "atoms"
    version_file_path = atoms_base_path / "atoms_config_version.json"

    # 1. Load current target config from gin-style config (mirroring dataset.py logic)
    config_dir = dataset_path / "config"
    current_config, _ = load_config(config_dir)

    atoms_config = _get_config_value(current_config, ["atoms"], {})
    semantic_config = _get_config_value(current_config, ["semantic"], {})

    segment_frames = atoms_config.get("frames", 48)
    hop_frames = atoms_config.get("hop_frames", 15)

    context_seconds = semantic_config.get("context_seconds")
    if context_seconds is None:
        context_seconds = semantic_config.get("context_time", 1.0)

    # Standardize current config for comparison
    active_config = {
        "atoms_frames": segment_frames,
        "atoms_hop_frames": hop_frames,
        "semantic_context_seconds": context_seconds,
    }

    # 2. --- SAFETY HANDSHAKE ---
    if atoms_base_path.exists() and any(atoms_base_path.iterdir()):
        if not version_file_path.exists():
            # Atoms exist but no version file? Dangerous state.
            raise RuntimeError(
                f"Existing atoms found at {atoms_base_path}, but no version file exists.\n"
                "Cannot verify compatibility. Please delete the 'atoms' folder manually."
            )

        with open(version_file_path, "r") as f:
            existing_config = json.load(f)

        if existing_config.get("atoms_frames") != active_config.get("atoms_frames") or existing_config.get(
            "atoms_hop_frames"
        ) != active_config.get("atoms_hop_frames"):
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
            with open(version_file_path, "w") as f:
                json.dump(active_config, f, indent=4)
        else:
            print("✅ Existing atoms match current configuration. Skipping generation for existing files.")
    else:
        # Folder is new or empty, create version file for the new run
        atoms_base_path.mkdir(parents=True, exist_ok=True)
        with open(version_file_path, "w") as f:
            json.dump(active_config, f, indent=4)
        print(f"Created version stamp at {version_file_path}")

    # 3. List Audio Files
    dataset_raw_path = dataset_path / "raw"
    audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.ogg"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(list(dataset_raw_path.rglob(ext)))

    # Unique Filename Check
    filenames = [f.name for f in audio_files]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Duplicate filenames detected. Ensure all names are unique across subfolders.")

    # 4. Process Loop
    dataset_manifest = {}
    total_removed = 0
    total_full = 0
    files_trimmed = 0
    samples_per_frame = sr // processor_48k_streamable.frame_rate
    segment_samples = segment_frames * samples_per_frame
    hop_samples = hop_frames * samples_per_frame
    for audio_file in audio_files:
        # Check if this specific audio already has its atoms folder
        # We use your existing 'make_atom_path' logic to find the folder
        atom_folder = make_atom_path(audio_file, 0).parent

        context_samples = None
        if context_seconds is not None:
            context_samples = int(round(context_seconds * sr))

        if atom_folder.exists() and any(atom_folder.glob("*_atom_*.pt")):
            # Still need to count them for the manifest
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
                hop_frames,
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

    # 5. Save Manifest
    manifest_save_path = dataset_path / "config" / "manifest.json"
    manifest_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_save_path, "w") as f:
        json.dump(dataset_manifest, f, indent=4)

    print(f"--- Extraction Complete ---\nManifest saved to: {manifest_save_path}")

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
