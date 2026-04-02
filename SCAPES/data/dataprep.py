import torch
import librosa
from pathlib import Path
import json

import os
from tqdm import tqdm

# Encoder wrapper
from SCAPES.auxiliar.encodec_wrapper import EncodecProcessor

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
# Takes a file and makes a list of its atoms
def extractor_atoms(audio_path, processor, segment_frames=39, hop_frames=18):
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
        
    audio_input = audio_input.to(processor.device) 
    audio_input_sample = audio_input.shape[-1]

    # Calculate segment and hop directly in samples
    segment_samples = segment_frames * samples_per_frame
    hop_samples     = hop_frames * samples_per_frame

    # Segmentation
    segments = []
    # Notice we step by hop_samples now!
    for start in range(0, audio_input_sample, hop_samples):
        end = start + segment_samples
        segment = audio_input[:, :, start:end]
        
        # Skip last segment if it's too short (must be exactly 39 frames)
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
import json
import torch
import os
from pathlib import Path

def atoms_maker(dataset_path):
    # Setup Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize Processor
    sr = 48000
    processor_48k_streamable = EncodecProcessor(sr=sr, streamable=True, device=device)

    dataset_path = Path(dataset_path)
    atoms_base_path = dataset_path / "atoms"
    version_file_path = atoms_base_path / "atoms_config_version.json"

    # 1. Load current target config from dataprep.json
    config_path = dataset_path / "config" / "dataprep.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            current_config = json.load(f)
        segment_frames = current_config.get("atoms_frames", 39)
        hop_frames = current_config.get("atoms_hop_frames", 18)
    else:
        segment_frames = 39
        hop_frames = 18
    
    # Standardize current config for comparison
    active_config = {
        "atoms_frames": segment_frames,
        "atoms_hop_frames": hop_frames
    }

    # 2. --- SAFETY HANDSHAKE ---
    if atoms_base_path.exists() and any(atoms_base_path.iterdir()):
        if not version_file_path.exists():
            # Atoms exist but no version file? Dangerous state.
            raise RuntimeError(
                f"Existing atoms found at {atoms_base_path}, but no version file exists.\n"
                "Cannot verify compatibility. Please delete the 'atoms' folder manually."
            )
        
        with open(version_file_path, 'r') as f:
            existing_config = json.load(f)
        
        # Compare configurations
        if existing_config != active_config:
            raise RuntimeError(
                f"CONFIG MISMATCH!\n"
                f"Current config:  {active_config}\n"
                f"Existing atoms:  {existing_config}\n"
                f"Atoms at this location are incompatible. Delete the 'atoms' folder to re-generate."
            )
        else:
            print("✅ Existing atoms match current configuration. Skipping generation for existing files.")
    else:
        # Folder is new or empty, create version file for the new run
        atoms_base_path.mkdir(parents=True, exist_ok=True)
        with open(version_file_path, 'w') as f:
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
    for audio_file in audio_files:
        # Check if this specific audio already has its atoms folder
        # We use your existing 'make_atom_path' logic to find the folder
        atom_folder = make_atom_path(audio_file, 0).parent
        
        if atom_folder.exists() and any(atom_folder.glob("*_atom_*.pt")):
            print(f"Skipping {audio_file.name}, atoms already exist and are verified compatible.")
            # Still need to count them for the manifest
            atom_files = list(atom_folder.glob("*_atom_*.pt"))
            count = len(atom_files)
        else:
            print(f"Processing {audio_file.name}...")
            # Using hop_frames instead of overlap_frames
            atoms = extractor_atoms(audio_file, processor_48k_streamable, segment_frames, hop_frames)
            torch_save_atoms(atoms, audio_file)
            count = len(atoms)

        dataset_manifest[audio_file.name] = {
            "path": str(audio_file.resolve()),
            "atoms_count": count
        }

    # 5. Save Manifest
    manifest_save_path = dataset_path / "config" / "manifest.json"
    manifest_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_save_path, 'w') as f:
        json.dump(dataset_manifest, f, indent=4)
    
    print(f"--- Extraction Complete ---\nManifest saved to: {manifest_save_path}")

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

import os
import torch
import json
from tqdm import tqdm
from pathlib import Path

from SCAPES.models.factorization import GlobalEncoder

def precompute_annotations(
    dataset, 
    annotation_type="clap", # "clap" or "custom"
    time_part="context_win", # "past" or "context_win"
    model=None, 
    batch_size=32,
    device="cuda"
):
    """
    Computes either ground-truth CLAP embeddings (from audio) or 
    Custom-trained proxy embeddings (from latents) and saves them to disk.
    
    The files are saved under:
    annotations_dir / config_folder / [clap|ctx] / [past|context_win]
    """
    if annotation_type not in ["clap", "custom"]:
        raise ValueError("annotation_type must be 'clap' or 'custom'")
    if time_part not in ["past", "context_win"]:
        raise ValueError("time_part must be 'past' or 'context_win'")
        
    # 1. Path construction
    if dataset.annotations_dir is None:
        raise ValueError("Dataset annotations_dir is not set.")
        
    base_dir = dataset.annotations_dir / dataset.config_folder_name
    
    # Map internal logic 'custom' to folder name 'ctx' as per your dataset refactor
    folder_cat = "clap" if annotation_type == "clap" else "ctx"
    save_dir = base_dir / folder_cat / time_part
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n--- Starting Pre-computation ---")
    print(f"Annotation Type: {annotation_type.upper()} (Saving to folder: {folder_cat})")
    print(f"Time Window:     {time_part.upper()}")
    print(f"Save Path:       {save_dir}")
    
    # Map 'context_win' string back to the dataset's internal loader logic ('context')
    loader_part = "context" if time_part == "context_win" else "past"

    if annotation_type=="custom":
        model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc=f"Processing {annotation_type}"):
            batch_inputs = []
            batch_scales = [] 
            indices_to_compute = []
            
            # Check for existing files to allow resuming
            for j in range(batch_size):
                idx = i + j
                if idx >= len(dataset):
                    break
                    
                save_path = save_dir / f"emb_{idx}.pt"
                if save_path.exists():
                    continue  
                    
                indices_to_compute.append(idx)
                
                if annotation_type == "clap":
                    # CLAP uses raw audio
                    raw_audio = dataset.get_raw_audio(idx, part=loader_part)
                    batch_inputs.append(raw_audio)
                    
                elif annotation_type == "custom":
                    # Custom uses EnCodec latents + scales
                    filename, seq_start_idx = dataset.all_indices[idx]
                    atom_start_idx, atom_count = dataset._get_part_indices(seq_start_idx, part=loader_part)
                    
                    latents, scales = dataset._load_atom_sequence(filename, atom_start_idx, atom_count)
                    batch_inputs.append(latents)
                    batch_scales.append(scales)

            if not indices_to_compute:
                continue 

            # Compute embeddings
            if annotation_type == "clap":
                batched_audio = torch.stack(batch_inputs).to(device)
                embedding = model.compute_embedding(
                    batched_audio,
                    og_sr=48000,
                    random_extension=True
                )
                
            elif annotation_type == "custom":
                batched_latents = torch.stack(batch_inputs).to(device) 
                batched_scales = torch.stack(batch_scales).to(device)  
                embedding = model(batched_latents, batched_scales) 
            
            embedding = embedding.detach().cpu()

            # Save per index
            for k, idx in enumerate(indices_to_compute):
                single_embedding = embedding[k]
                save_path = save_dir / f"emb_{idx}.pt"
                torch.save(single_embedding, save_path)

    print(f"✅ Success: {annotation_type}/{time_part} pre-computed and saved in {folder_cat} folder.")
