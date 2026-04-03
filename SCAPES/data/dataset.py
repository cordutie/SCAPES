import torch
from torch.utils.data import Dataset, Subset
import json
from pathlib import Path
import random
import librosa
import warnings

def batch_from_latents_to_audio(batch, dataset, processor, mode="decoded", part="past"):
    """
    Converts a DataLoader batch to audio.
    """
    audios = []
    indices = batch["index"]

    for idx in indices:
        if mode == "raw":
            audio = dataset.get_raw_audio(idx, part=part)
        elif mode == "decoded":
            audio = dataset.get_decoded_audio(idx, processor, part=part)
        else:
            raise ValueError(f"Unknown mode {mode}")
        audios.append(audio)
    
    return torch.stack(audios, dim=0)

class AtomSequenceDataset(Dataset):
    def __init__(
        self, 
        dataset_path, 
        requested_keys=None,
        segment_length=20, 
        context_length=10, 
        hop_size=10, 
        sr=48000, 
        frame_rate=150, 
        device='cpu',
        verbose=False
    ):
        self.dataset_path    = Path(dataset_path)
        self.annotations_dir = self.dataset_path / "annotations"
        
        # Sliding Window Config
        self.segment_length = segment_length
        self.context_length = context_length
        self.context_shift  = segment_length 
        self.hop_size = hop_size
        
        self.total_length = max(self.segment_length + 1, self.context_shift + self.context_length)
        self.config_folder_name = f"seg_{self.segment_length}_ctx_{self.context_length}_shift_{self.context_shift}_hop_{self.hop_size}"
        
        if requested_keys is None:
            self.requested_keys = ["latent_past", "scale_past", "index"]
        else:
            valid_keys = {"latent_past", "latent_present", "latent_context_win", 
                        "scale_past", "scale_present", "scale_context_win", 
                        "ctx_emb_past", "ctx_emb_context_win", "clap_past", "clap_context_win",
                        "index"}
            if not all(key in valid_keys for key in requested_keys):
                raise ValueError(f"Invalid keys in requested_keys. Valid keys are: {valid_keys}")
            self.requested_keys = requested_keys

        self.device = device
        
        # Parameters for audio math
        self.sr = sr
        self.frame_rate = frame_rate
        self.samples_per_frame = sr // frame_rate
        
        # Load the manifest
        json_path = self.dataset_path / "config" / "manifest.json"
        with open(json_path, 'r') as f:
            self.manifest = json.load(f)
            
        # --- NEW: Reading dataprep.json with Prefix Padding Logic ---
        dataprep_config_path = self.dataset_path / "config" / "dataprep.json"
        if dataprep_config_path.exists():
            with open(dataprep_config_path, 'r') as f:
                dataprep_config = json.load(f)

            self.atoms_frames = dataprep_config.get("atoms_frames", 48)
            self.atoms_hop_frames = dataprep_config.get("atoms_hop_frames", 15)
            self.crossfade_frames = dataprep_config.get("crossfade_frames", 3) # The acoustic fade
            
            self.train_split_key = dataprep_config.get("train_split")
            self.val_split_key = dataprep_config.get("val_split")
        else:
            # create the dataprep.json with default values for the Prefix Padding geometry and save it
            dataprep_config = {
                "atoms_frames": 48,
                "atoms_hop_frames": 15,
                "crossfade_frames": 3,
                "train_split": None,
                "val_split": None
            }
            # save the json file 
            with open(dataprep_config_path, 'w') as f:
                json.dump(dataprep_config, f, indent=4)
            self.atoms_frames = 48
            self.atoms_hop_frames = 15
            self.crossfade_frames = 3
            self.train_split_key = None
            self.val_split_key = None

        # --- Math for the Asymmetric Geometry ---
        self.macro_overlap_frames = self.atoms_frames - self.atoms_hop_frames # Total baked past (e.g. 21)
        
        self.atoms_samples = self.atoms_frames * self.samples_per_frame
        self.hop_samples = self.atoms_hop_frames * self.samples_per_frame
        self.crossfade_samples = self.crossfade_frames * self.samples_per_frame
        self.macro_overlap_samples = self.macro_overlap_frames * self.samples_per_frame

        self.filenames = sorted(list(self.manifest.keys()))
        self.all_indices = self._build_mapping(self.filenames)

        # Pre-compute Overlap-Add Window (Now Asymmetric!)
        self.window = self._build_ola_window()

        self.file_id_lookup = {fname: i for i, fname in enumerate(self.filenames)}
        self.sequence_to_file_id = [self.file_id_lookup[fname] for fname, _ in self.all_indices]

        if verbose:
            atoms_hop_time = self.atoms_hop_frames / self.frame_rate
            control_rate = 1 / atoms_hop_time
            macro_overlap_time = self.macro_overlap_frames / self.frame_rate
            print("\n\033[1mDataset Summary:\033[0m ---------------------------------------------------------------------------")
            print(f"    Your dataset is made of {len(self.filenames)} audio files")
            print(f"    A total of {self.count_atoms()} atoms are in your dataset")
            print(f"    Atoms are {self.atoms_frames} frames long, hopping forward by {self.atoms_hop_frames} frames.")
            print(f"    This implies that the dataset is made of {atoms_hop_time*self.count_atoms() + len(self.filenames)*macro_overlap_time:.1f} seconds of audio in total (taking overlap into account).")
            print(f"    Atoms are overlapped using a MACRO overlap of {self.macro_overlap_frames} frames acting as context history.")
            print(f"    During audio rendering, a MICRO crossfade of {self.crossfade_frames} frames is used to stitch seams.")
            print(f"    This implies the temporal control rate of the model is {control_rate:.2f} Hz ({atoms_hop_time*1000:.1f} ms steps).")
            print(f"    Your dataset has {len(self.all_indices)} sequences in total.")
            print(f"    Requested keys: {self.requested_keys}")
            self.check_if_manifest_has_splits()
            self.check_annotations_exist()

    def count_atoms(self):
        count = 0
        for fname in self.filenames:
            count += self.manifest[fname]["atoms_count"]
        return count

    def check_if_manifest_has_splits(self):
        has_split = all("validation" in self.manifest[f] for f in self.filenames)
        if not has_split:
            print("    No complete split found in manifest.json.")
            return False
            
        train_count = sum(1 for f in self.filenames if self.manifest[f]["validation"] is False)
        val_count = sum(1 for f in self.filenames if self.manifest[f]["validation"] is True)
        partial_count = sum(1 for f in self.filenames if self.manifest[f]["validation"] == "partial")
        
        if partial_count > 0:
            print(f"    Manifest has a chronological split: {partial_count} files split across train/val.")
        else:
            print(f"    Manifest has an existing full-file split: {train_count} train files, {val_count} val files.")
        return True

    def get_splits(self):
        """
        Builds (train_subset, val_subset) based on the 'validation' field
        stored in manifest.json. Supports both full-file and partial-file splits.
        """
        if not all("validation" in self.manifest[f] for f in self.filenames):
            warnings.warn(
                "No split found in manifest.json. "
                "Run dataset.make_split(val_split=...) first."
            )
            return None, None

        train_indices = []
        val_indices = []

        for i, (f, start) in enumerate(self.all_indices):
            val_flag = self.manifest[f].get("validation")
            
            # Handle the new intra-file splitting
            if val_flag == "partial":
                if start in self.manifest[f].get("val_starts", []):
                    val_indices.append(i)
                else:
                    train_indices.append(i)
                    
            # Handle old full-file validation
            elif val_flag is True:
                val_indices.append(i)
                
            # Handle old full-file training
            else:
                train_indices.append(i)

        train_subset = Subset(self, train_indices)
        val_subset = Subset(self, val_indices)

        print(f"Loaded split: {len(train_indices)} train sequences, "
              f"{len(val_indices)} val sequences.")

        return train_subset, val_subset

    def check_annotations_exist(self): 
        base_anno_path = self.annotations_dir / self.config_folder_name
        if not base_anno_path.exists():
            print(f"    There are not annotations for your settings (they should be in {base_anno_path}).")
            return False
        
        print(f"    Annotations found in {base_anno_path}:")
        any_emb = False
        for cat in ["ctx", "clap"]:
            for time_part in ["past", "context_win"]:
                path = base_anno_path / cat / time_part
                if not path.exists():
                    print(f"    ✗ {cat}_{time_part} (should be in {path})")
                else:
                    num_files = len(list(path.glob("emb_*.pt")))
                    print(f"    ✓ {cat}_{time_part}: Found {num_files} files in {path}")
                    any_emb = True
        return any_emb

    def _build_ola_window(self):
        """
        Builds an asymmetric window for Prefix Padding geometry:
        [ Zeros (discard redundant past) | Hann Fade In | Ones (new audio) | Hann Fade Out ]
        """
        zeros_frames = self.macro_overlap_frames - self.crossfade_frames
        zeros = torch.zeros(zeros_frames * self.samples_per_frame)
        
        hann_window = torch.hann_window(self.crossfade_samples * 2)
        left_hann = hann_window[:self.crossfade_samples]
        right_hann = hann_window[self.crossfade_samples:]
        
        ones_frames = self.atoms_hop_frames - self.crossfade_frames
        ones = torch.ones(ones_frames * self.samples_per_frame)
        
        # Final window is exactly `atoms_samples` long
        window = torch.cat([zeros, left_hann, ones, right_hann])
        return window

    def _build_mapping(self, filenames):
        mapping = []
        for fname in filenames:
            count = self.manifest[fname]["atoms_count"]
            if count >= self.total_length:
                for start in range(0, count - self.total_length + 1, self.hop_size):
                    mapping.append((fname, start))
        return mapping

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
        
    def _get_part_indices(self, start_idx, part):
        if part == "past":
            return start_idx, self.segment_length
        elif part == "context":
            return start_idx + self.context_shift, self.context_length
        elif part == "full":
            return start_idx, self.total_length
        else:
            raise ValueError("part must be 'past', 'context', or 'full'")

    def get_raw_audio(self, idx, part="past"):
        filename, seq_start_idx = self.all_indices[idx]
        atom_start_idx, atom_count = self._get_part_indices(seq_start_idx, part)
        
        audio_path = self.manifest[filename]["path"]

        # Calculate absolute bounds
        start_sample = atom_start_idx * self.hop_samples
        # The total length stretches to the end of the final atom
        duration_samples = ((atom_count - 1) * self.hop_samples) + self.atoms_samples

        audio_input, _ = librosa.load(audio_path, sr=self.sr, mono=False)
        audio_input = torch.tensor(audio_input).unsqueeze(0) 
        
        if audio_input.dim() == 2:
            audio_input = audio_input.unsqueeze(1).repeat(1, 2, 1)
        elif audio_input.shape[1] > 2:
            audio_input = audio_input[:, :2, :]
            
        audio_input = audio_input[:, :2, start_sample:start_sample+duration_samples]
        return audio_input.squeeze(0).to(self.device)

    def get_decoded_audio(self, idx, processor, part="past"):
        filename, seq_start_idx = self.all_indices[idx]
        atom_start_idx, atom_count = self._get_part_indices(seq_start_idx, part)
        
        total_samples = (atom_count - 1) * self.hop_samples + self.atoms_samples
        out_audio = torch.zeros((1, 2, total_samples), device=processor.device)
        window = self.window.to(processor.device).view(1, 1, -1)

        for i in range(atom_count):
            atom_path = self._get_atom_path(filename, atom_start_idx + i)
            atom = torch.load(atom_path, weights_only=True, map_location=processor.device)
            
            latent_cont = atom["latent"].float()
            length = latent_cont.shape[-1]
            metadata = {
                "audio_scales": [atom["scale"].float()],
                "padding_mask": torch.ones((1, length * self.samples_per_frame), 
                                         dtype=torch.bool, device=processor.device)
            }
            
            with torch.no_grad():
                decoded_chunk = processor.decode_latents_audio(latent_cont, metadata=metadata)
            
            start_s = i * self.hop_samples
            end_s = start_s + self.atoms_samples
            
            # Apply the asymmetric prefix-padding mask
            out_audio[:, :, start_s:end_s] += decoded_chunk[:, :, :self.atoms_samples] * window
        
        return out_audio.squeeze(0)

    def make_split(self, val_split="dataprep", seed=42, overwrite=False):
        if val_split == "dataprep":
            val_split = None  # This will trigger the directory-based split logic

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
        # PER-FILE CHRONOLOGICAL SPLIT (Texture Setup)
        # ------------------------------------------------

        if val_split is None:
            raise ValueError(
                "val_split must be provided because dataprep.json "
                "does not define train_split/val_split."
            )

        already_split = all("validation" in self.manifest[f] for f in self.filenames)

        if already_split and not overwrite:
            warnings.warn(
                "Split already exists in manifest.json. "
                "Use overwrite=True to regenerate it."
            )
            return

        total_train_seqs = 0
        total_val_seqs = 0

        for f in self.filenames:
            # Figure out all valid start indices for this specific file
            count = self.manifest[f]["atoms_count"]
            starts = list(range(0, count - self.total_length + 1, self.hop_size))
            
            # Chronological split to prevent overlap leakage!
            split_idx = int(len(starts) * (1 - val_split))
            
            val_starts = starts[split_idx:]
            
            # Mark the file as 'partial' to tell get_splits to look at the exact indices
            self.manifest[f]["validation"] = "partial" 
            self.manifest[f]["val_starts"] = val_starts
            
            total_train_seqs += split_idx
            total_val_seqs += len(val_starts)

        with open(json_path, "w") as f:
            json.dump(self.manifest, f, indent=4)

        print(
            f"Per-file chronological split created: {total_train_seqs} train sequences, "
            f"{total_val_seqs} val sequences."
        )

    def __len__(self):
        return len(self.all_indices)

    def _load_atom_sequence(self, filename, start_idx, count):
        latents, scales = [], []
        for i in range(start_idx, start_idx + count):
            atom_path = self._get_atom_path(filename, i)
            atom_data = torch.load(atom_path, weights_only=True, map_location='cpu')
            latents.append(atom_data["latent"].squeeze(0).float())
            scales.append(atom_data["scale"].squeeze(0).float())
        return torch.stack(latents, dim=0), torch.stack(scales, dim=0)

    def __getitem__(self, idx):
        filename, start_idx = self.all_indices[idx]
        batch_dict = {
            "index": idx,
            "label": filename  # This is the unique file key from the manifest
        }
        req = self.requested_keys

        # --- 1. Load Past Latents (Segment) ---
        if any(k in req for k in ["latent_past", "latent_present", "scale_past", "scale_present"]):
            past_latents, past_scales = self._load_atom_sequence(filename, start_idx, self.segment_length)
            
            if "latent_past" in req:
                batch_dict["latent_past"] = past_latents.to(self.device)
            if "scale_past" in req:
                batch_dict["scale_past"] = past_scales.to(self.device)
                
            # Present is defined as the single atom immediately following the segment
            if "latent_present" in req or "scale_present" in req:
                present_path = self._get_atom_path(filename, start_idx + self.segment_length)
                present_atom = torch.load(present_path, weights_only=True, map_location='cpu')
                if "latent_present" in req:
                    batch_dict["latent_present"] = present_atom["latent"].squeeze(0).float().to(self.device)
                if "scale_present" in req:
                    batch_dict["scale_present"] = present_atom["scale"].squeeze(0).float().to(self.device)

        # --- 2. Load Context Window Latents ---
        if "latent_context_win" in req or "scale_context_win" in req:
            ctx_start = start_idx + self.context_shift
            ctx_latents, ctx_scales = self._load_atom_sequence(filename, ctx_start, self.context_length)
            
            if "latent_context_win" in req:
                batch_dict["latent_context_win"] = ctx_latents.to(self.device)
            if "scale_context_win" in req:
                batch_dict["scale_context_win"] = ctx_scales.to(self.device)

        # --- 3. Pre-computed Embeddings ---
        if self.annotations_dir is not None:
            base_anno_path = self.annotations_dir / self.config_folder_name
            
            # Helper to load a specific file
            def load_emb(cat, time_part):
                path = base_anno_path / cat / time_part / f"emb_{idx}.pt"
                if not path.exists():
                    raise FileNotFoundError(f"Requested {cat}_{time_part} but file missing: {path}")
                return torch.load(path, weights_only=True, map_location='cpu').to(self.device)

            if "ctx_emb_past" in req:
                batch_dict["ctx_emb_past"] = load_emb("ctx", "past")
            if "ctx_emb_context_win" in req:
                batch_dict["ctx_emb_context_win"] = load_emb("ctx", "context_win")
                
            if "clap_past" in req:
                batch_dict["clap_past"] = load_emb("clap", "past")
            if "clap_context_win" in req:
                batch_dict["clap_context_win"] = load_emb("clap", "context_win")

        return batch_dict