import torch
from torch.utils.data import Dataset, DataLoader, Subset
import json
from pathlib import Path
import random
import librosa
import warnings

def batch_from_latents_to_audio(batch, dataset, processor, mode="decoded"):
    """
    Converts a DataLoader batch (with latent + scale + index) to audio.

    Args:
        batch: dict with keys ["latent", "scale", "index"]
        dataset: AtomSequenceDataset instance
        processor: EncodecProcessor
        mode: "raw" or "decoded"
    
    Returns:
        Tensor: [Batch, Channels, Samples]
    """
    audios = []
    indices = batch["index"]

    for idx in indices:
        if mode == "raw":
            audio = dataset.get_raw_audio(idx)
        elif mode == "decoded":
            audio = dataset.get_decoded_audio(idx, processor)
        else:
            raise ValueError(f"Unknown mode {mode}")
        audios.append(audio)
    
    return torch.stack(audios, dim=0)

class AtomSequenceDataset(Dataset):
    def __init__(self, dataset_path, clap_emb_dir=None, segment_length=20, hop_size=10, sr=48000, frame_rate=150, device='cpu'):
        self.dataset_path = Path(dataset_path)
        self.clap_emb_dir = Path(clap_emb_dir) if clap_emb_dir is not None else None
        self.segment_length = segment_length
        self.hop_size = hop_size
        self.device = device
        
        # Parameters for audio math
        self.sr = sr
        self.frame_rate = frame_rate
        self.samples_per_frame = sr // frame_rate
        
        # Load the manifest
        json_path = self.dataset_path / "config" / "manifest.json"
        with open(json_path, 'r') as f:
            self.manifest = json.load(f)
            
        # Reading dataprep.json for segment/overlap frames and split names
        dataprep_config_path = self.dataset_path / "config" / "dataprep.json"

        if dataprep_config_path.exists():
            with open(dataprep_config_path, 'r') as f:
                dataprep_config = json.load(f)

            self.segment_frames = dataprep_config.get("atoms_frames")
            self.overlap_frames = dataprep_config.get("atoms_overlap_frames")

            self.train_split_key = dataprep_config.get("train_split")
            self.val_split_key = dataprep_config.get("val_split")

        else:
            self.segment_frames = 21
            self.overlap_frames = 3
            self.train_split_key = None
            self.val_split_key = None

        print(f"Dataset initialized with segment_length={self.segment_length}, hop_size={self.hop_size}, ")
        
        
        # Calculate samples logic
        self.segment_samples = self.segment_frames * self.samples_per_frame
        self.overlap_samples = self.overlap_frames * self.samples_per_frame
        self.hop_samples     = self.segment_samples - self.overlap_samples

        self.filenames = sorted(list(self.manifest.keys()))
        self.all_indices = self._build_mapping(self.filenames)

        # Pre-compute Overlap-Add Window (Optimization)
        self.window = self._build_ola_window()

    def _build_ola_window(self):
        """Creates the Hann-faded window used for Overlap-Add."""
        window = torch.ones(self.segment_samples - 2 * self.overlap_samples)
        hann_window = torch.hann_window(self.overlap_samples * 2)
        left_hann = hann_window[:self.overlap_samples]
        right_hann = hann_window[self.overlap_samples:]
        return torch.cat([left_hann, window, right_hann])


    def _build_mapping(self, filenames):
        mapping = []
        for fname in filenames:
            count = self.manifest[fname]["atoms_count"]
            if count >= self.segment_length:
                for start in range(0, count - self.segment_length + 1, self.hop_size):
                    mapping.append((fname, start))
        return mapping

    def get_raw_audio(self, idx):
        """
        Loads the raw audio corresponding exactly to sequence idx.
        """
        filename, start_atom_idx = self.all_indices[idx]
        audio_path = self.manifest[filename]["path"]

        # Calculate exact sample range
        # Start of the first atom in the sequence
        start_sample = start_atom_idx * self.hop_samples
        
        # End of the last atom in the sequence
        # (Start of last atom) + (length of one atom)
        last_atom_in_seq_idx = start_atom_idx + (self.segment_length - 1)
        end_sample = (last_atom_in_seq_idx * self.hop_samples) + self.segment_samples

        # Load only the required duration to save RAM/Time
        duration_samples = end_sample - start_sample
        offset_seconds = start_sample / self.sr
        duration_seconds = duration_samples / self.sr

        # Load audio chunk
        audio_input, _ = librosa.load(
            audio_path, 
            sr=self.sr, 
            mono=False
            # offset=offset_seconds, 
            # duration=duration_seconds
        )
        
        # Convert to tensor and fix channels (Stereo logic from your script)
        audio_input = torch.tensor(audio_input).unsqueeze(0) # [1, Channels, Samples]
        
        if audio_input.dim() == 2: # [Channels, Samples] -> [1, 2, Samples]
            audio_input = audio_input.unsqueeze(1).repeat(1, 2, 1)
        elif audio_input.shape[1] > 2:
            audio_input = audio_input[:, :2, :]
            
        audio_input = audio_input[:, :2, start_sample:start_sample+duration_samples] # Ensure we only take the needed samples

        return audio_input.squeeze(0).to(self.device) # Shape: [2, total_samples]

    def get_decoded_audio(self, idx, processor):
        """
        Decodes a sequence atom-by-atom and performs Overlap-Add.
        Requires the EncodecProcessor instance.
        """
        # 1. Load the sequence atoms
        filename, start_idx = self.all_indices[idx]
        
        # Calculate output length
        total_samples = (self.segment_length - 1) * self.hop_samples + self.segment_samples
        # Buffer: [Batch, Channels, Samples] -> Encodec usually returns [1, 2, Samples]
        out_audio = torch.zeros((1, 2, total_samples), device=processor.device)
        
        window = self.window.to(processor.device)

        for i in range(self.segment_length):
            # Load atom
            atom_path = self._get_atom_path(filename, start_idx + i)
            atom = torch.load(atom_path, weights_only=True, map_location=processor.device)
            
            # Convert to decoder input
            latent_cont = atom["latent"].float()
            length = latent_cont.shape[-1]
            metadata = {
                "audio_scales": [atom["scale"].float()],
                "padding_mask": torch.ones((1, length * self.samples_per_frame), 
                                         dtype=torch.bool, device=processor.device)
            }
            
            # Decode
            with torch.no_grad():
                decoded_chunk = processor.decode_latents_audio(latent_cont, metadata=metadata)
            
            # Overlap and Add
            start_s = i * self.hop_samples
            end_s = start_s + self.segment_samples
            
            # Apply window and accumulate
            out_audio[:, :, start_s:end_s] += decoded_chunk[:, :, :self.segment_samples] * window
            
        return out_audio.squeeze(0) # Shape: [Channels, total_samples]

    def make_split(self, val_split=None, seed=42, overwrite=False):

        json_path = self.dataset_path / "config" / "manifest.json"

        # ------------------------------------------------
        # SPLIT FROM DIRECTORY STRUCTURE
        # ------------------------------------------------
        if self.train_split_key or self.val_split_key:

            if val_split is not None:
                raise ValueError(
                    "val_split was provided but dataprep.json defines "
                    "train_split/val_split directories."
                )

            if not (self.train_split_key and self.val_split_key):
                raise ValueError(
                    "dataprep.json must define BOTH 'train_split' and 'val_split'."
                )

            print(
                f"Creating split from folder structure: "
                f"{self.train_split_key} / {self.val_split_key}"
            )

            train_files = []
            val_files = []

            for fname in self.filenames:

                entry = self.manifest[fname]
                path = Path(entry["path"])

                parts = path.parts

                if self.val_split_key in parts:
                    entry["validation"] = True
                    val_files.append(fname)

                elif self.train_split_key in parts:
                    entry["validation"] = False
                    train_files.append(fname)

                else:
                    raise ValueError(
                        f"File {path} is not inside "
                        f"{self.train_split_key} or {self.val_split_key}"
                    )

            with open(json_path, "w") as f:
                json.dump(self.manifest, f, indent=4)

            print(
                f"Split created from directory structure: "
                f"{len(train_files)} train files, {len(val_files)} val files."
            )

            return

        # ------------------------------------------------
        # RANDOM SPLIT (fallback)
        # ------------------------------------------------

        if val_split is None:
            raise ValueError(
                "val_split must be provided because dataprep.json "
                "does not define train_split/val_split."
            )

        rng = random.Random(seed)

        already_split = all("validation" in self.manifest[f] for f in self.filenames)

        if already_split and not overwrite:
            warnings.warn(
                "Split already exists in manifest.json. "
                "Use overwrite=True to regenerate it."
            )
            return

        shuffled_files = list(self.filenames)
        rng.shuffle(shuffled_files)

        split_idx = int(len(shuffled_files) * (1 - val_split))

        train_files = set(shuffled_files[:split_idx])
        val_files = set(shuffled_files[split_idx:])

        for f in self.filenames:
            self.manifest[f]["validation"] = f in val_files

        with open(json_path, "w") as f:
            json.dump(self.manifest, f, indent=4)

        print(
            f"Random split created: {len(train_files)} train files, "
            f"{len(val_files)} val files."
        )

    def get_splits(self):
        """
        Builds (train_subset, val_subset) based on 'validation' field
        stored in manifest.json.

        Raises warning if split has not been created.
        """

        # Check if split exists
        if not all("validation" in self.manifest[f] for f in self.filenames):
            warnings.warn(
                "No split found in manifest.json. "
                "Run dataset.make_split(val_split=...) first."
            )
            return None, None

        train_indices = []
        val_indices = []

        for i, (f, _) in enumerate(self.all_indices):
            if self.manifest[f]["validation"]:
                val_indices.append(i)
            else:
                train_indices.append(i)

        train_subset = Subset(self, train_indices)
        val_subset = Subset(self, val_indices)

        print(f"Loaded split: {len(train_indices)} train sequences, "
            f"{len(val_indices)} val sequences.")

        return train_subset, val_subset

    def _get_atom_path(self, original_filename, atom_index):
        original_path = Path(self.manifest[original_filename]["path"])
        stem = original_path.stem
        parts = list(original_path.parts)
        try:
            raw_idx = parts.index("raw")
            relative_parent = Path(*parts[raw_idx + 1 : -1])
        except ValueError:
            relative_parent = Path("")

        atom_filename = f"{stem}_atom_{atom_index}.pt"
        return self.dataset_path / "atoms" / relative_parent / stem / atom_filename

    def __len__(self):
        return len(self.all_indices)

    def __getitem__(self, idx):
        filename, start_idx = self.all_indices[idx]
        latents, scales = [], []
        for i in range(start_idx, start_idx + self.segment_length):
            atom_path = self._get_atom_path(filename, i)
            atom_data = torch.load(atom_path, weights_only=True, map_location='cpu')
            latents.append(atom_data["latent"].squeeze(0).float())
            scales.append(atom_data["scale"].squeeze(0).float())
        
        batch_dict = {
            "latent": torch.stack(latents, dim=0).to(self.device),
            "scale":  torch.stack(scales, dim=0).to(self.device),
            "index": idx  
        }

        # <--- Add this block --->
        if self.clap_emb_dir is not None:
            emb_path = self.clap_emb_dir / f"clap_emb_{idx}.pt"
            clap_emb = torch.load(emb_path, weights_only=True, map_location='cpu')
            batch_dict["clap_emb"] = clap_emb.to(self.device)

        return batch_dict
    
# class AtomDataset(AtomSequenceDataset):
#     def __init__(self, dataset_path, device='cpu'):
#         super().__init__(
#             dataset_path=dataset_path,
#             segment_length=1,
#             hop_size=1,
#             device=device
#         )

#     def __getitem__(self, idx):
#         filename, atom_idx = self.all_indices[idx]
#         atom_path = self._get_atom_path(filename, atom_idx)
#         atom_data = torch.load(atom_path, weights_only=True, map_location='cpu')

#         return {
#             "latent": atom_data["latent"].squeeze(0).float().to(self.device),
#             "scale":  atom_data["scale"].squeeze(0).float().to(self.device),
#         }