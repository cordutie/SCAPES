import torch
import librosa
from pathlib import Path
import json

# Encoder wrapper
from auxiliar.encodec_wrapper import EncodecProcessor

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
def extractor_atoms(audio_path, processor, segment_frames=21, overlap_frames=3):
    # Use sample rate from processor
    sr         = processor.sample_rate
    frame_rate = processor.frame_rate
    samples_per_frame = sr // frame_rate

    # Load audio
    audio_input, _ = librosa.load(audio_path, sr=sr, mono=False)
    audio_input = torch.tensor(audio_input).unsqueeze(0) # Add batch dimension
    # If mono, make it stereo by duplicating the mono channel to create a stereo signal (2 channels)
    if audio_input.dim() == 2:
        audio_input = audio_input.unsqueeze(1).repeat(1, 2, 1)
    # If it has more than 2 channels, take only the first 2 channels
    elif audio_input.shape[1] > 2:
        audio_input = audio_input[:, :2, :]
    audio_input = audio_input.to(processor.device) # Move to the same device as the processor
    audio_input_sample = audio_input.shape[-1]

    # Calculate segment and overlap in samples
    segment_samples = segment_frames * samples_per_frame
    overlap_samples = overlap_frames * samples_per_frame
    hop_samples     = segment_samples - overlap_samples

    # Segmentation
    segments = []
    for start in range(0, audio_input_sample, hop_samples):
        end = start + segment_samples
        segment = audio_input[:, :, start:end]
        # Skip last segment if it's too short 
        if segment.shape[-1] < segment_samples:
            break
        segments.append(segment)

    # Atoms extraction
    atoms = []
    for i in range(0, len(segments)):
        latent, metadata = processor.audio_to_latents(segments[i], sr)
        atom_local = {
            "latent": torch.cat(latent, dim=-1).half(),
            "scale":  metadata["audio_scales"][0].half(),
        }
        atoms.append(to_cpu(atom_local))

    return atoms

# ATOM STRUCTURE:
# Each atom is a dictionary with the following structure:
# {
#     "latent": tensor of shape (1, 128, segment_frames), # concatenated latents of all layers
#     "scale":  tensor of shape (1,1), # audio scale of the atom
# }
# Note: all saved in 16 bit for compression. 
# REMEMBER LOADING AS 32 bit using .float()

# Makes the path for the atoms to be saved, based on the original audio path and the atom index. It replaces "raw" with "atoms" and adds "_atom_{index}" to the filename.
def make_atom_path(audio_path, atom_index):
    audio_path = Path(audio_path)

    parts = audio_path.parts
    raw_index = parts.index("raw")

    # Path after raw/
    relative_after_raw = Path(*parts[raw_index + 1:])

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
    atom_filename = stem+f"_atom_{atom_index}.pt"

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

# Full pipeline to compute the atoms for a dataset. It takes the path to the dataset, processes all audio files in the "raw" folder, and saves the atoms in the "atoms" folder. It also saves a dataset.json file with the path and number of atoms for each audio file.
def atoms_maker(dataset_path):
    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize 48kHz processor
    sr = 48000
    processor_48k_streamable = EncodecProcessor(sr=sr, streamable=True, device=device)

    # Dataset paths
    dataset_path = Path(dataset_path)

    # Config file
    config_path = dataset_path / "config" / "dataprep.json"
    print(f"Looking for config at {config_path}...")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        segment_frames = config.get("atoms_frames")
        overlap_frames = config.get("atoms_overlap_frames")
    else:
        segment_frames = 21
        overlap_frames = 3

    # List all audio files in dataset_raw_path recursively
    dataset_raw_path = dataset_path / "raw"
    audio_files  = list(dataset_raw_path.rglob("*.wav"))
    audio_files += list(dataset_raw_path.rglob("*.mp3"))
    audio_files += list(dataset_raw_path.rglob("*.flac"))
    audio_files += list(dataset_raw_path.rglob("*.ogg"))

    # Check that all files are called uniquely (no duplicates in the filename, even if they are in different folders)
    filenames = [f.name for f in audio_files]
    if len(filenames) != len(set(filenames)):
        raise ValueError("There are duplicate filenames in the dataset. Please ensure all audio files have unique names, even if they are in different folders.")

    # Process each audio file and save how many atoms were generated
    dataset = {}
    for audio_file in audio_files:
        print(f"Processing {audio_file}...")
        atom_folder = make_atom_path(audio_file, 0).parent
        if not atom_folder.exists() or len(list(atom_folder.glob("*_atom_*.pt"))) == 0:
            atoms = extractor_atoms(audio_file, processor_48k_streamable, segment_frames, overlap_frames)
            torch_save_atoms(atoms, audio_file)
        else:
            print(f"Skipping {audio_file}, atoms already exist.")
            atoms = list(atom_folder.glob("*_atom_*.pt"))
        filename = audio_file.name
        full_path = audio_file.resolve()
        dataset[filename] = {
            "path": str(full_path),       # JSON-safe
            "atoms_count": len(atoms)     # If atoms is list of Path objects, len() works
        }

    # Save manifest json
    count_save_path = dataset_path / "config" / "manifest.json"
    with open(count_save_path, 'w') as f:
        json.dump({str(k): v for k, v in dataset.items()}, f, indent=4)
    print(f"Saved dataset to {count_save_path}")

# def atom_loader(atom_path):
#     return torch.load(atom_path)

# def atom_to_decoder_input(atom):
#     latent_cont = atom["latent"]
#     length = latent_cont.shape[-1]
#     metadata_cont = {
#         "audio_scales": [atom["scale"]],
#         "padding_mask":  torch.ones((1, length*320), dtype=torch.bool, device=latent_cont.device)
#     }
#     return latent_cont, metadata_cont