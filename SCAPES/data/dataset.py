import torch
from torch.utils.data import Dataset, DataLoader, Subset
import json
from pathlib import Path
import random
import librosa
import warnings

def batch_from_latents_to_audio(batch, dataset, processor, mode="decoded", part="past"):
    """
    Converts a DataLoader batch to audio.

    Args:
        batch: dict containing at least "index"
        dataset: AtomSequenceDataset instance
        processor: EncodecProcessor
        mode: "raw" or "decoded"
        part: "past", "context", or "full"
    
    Returns:
        Tensor: [Batch, Channels, Samples]
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
        # Hardcoded to always live inside the dataset root
        self.annotations_dir = self.dataset_path / "annotations"
        
        # Sliding Window Config
        self.segment_length = segment_length
        self.context_length = context_length
        self.context_shift  = segment_length # By default, context window starts right after the past segment
        self.hop_size = hop_size
        
        # Total required atoms to satisfy the furthest reaching window
        # print("segment_length:", segment_length)
        # print("context_length:", context_length)
        # print("context_shift:", context_shift)
        self.total_length = max(self.segment_length + 1, self.context_shift + self.context_length)
        
        # The rigid version-control folder name
        self.config_folder_name = f"seg_{self.segment_length}_ctx_{self.context_length}_shift_{self.context_shift}_hop_{self.hop_size}"
        
        # Default requested keys if none provided
        if requested_keys is None:
            self.requested_keys = ["latent_past", "scale_past", "index"]
        else:
            # check that requested_keys is a list of valid strings
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
            
        # Reading dataprep.json
        dataprep_config_path = self.dataset_path / "config" / "dataprep.json"
        if dataprep_config_path.exists():
            with open(dataprep_config_path, 'r') as f:
                dataprep_config = json.load(f)

            self.atoms_frames = dataprep_config.get("atoms_frames", 21)
            self.atoms_overlap_frames = dataprep_config.get("atoms_overlap_frames", 3)
            self.train_split_key = dataprep_config.get("train_split")
            self.val_split_key = dataprep_config.get("val_split")
        else:
            self.atoms_frames = 21
            self.atoms_overlap_frames = 3
            self.train_split_key = None
            self.val_split_key = None

        # print(f"Dataset initialized with segment={self.segment_length}, context={self.context_length}, shift={self.context_shift}, hop={self.hop_size}")
        # print(f"Requested Keys: {self.requested_keys}")
        # if self.annotations_dir:
        #     print(f"Annotations target: {self.annotations_dir.name}/{self.config_folder_name}/...")
        
        # Calculate samples logic
        self.segment_samples = self.atoms_frames * self.samples_per_frame
        self.overlap_samples = self.atoms_overlap_frames * self.samples_per_frame
        self.hop_samples     = self.segment_samples - self.overlap_samples

        self.filenames = sorted(list(self.manifest.keys()))
        self.all_indices = self._build_mapping(self.filenames)

        # Pre-compute Overlap-Add Window
        self.window = self._build_ola_window()

        self.file_id_lookup = {fname: i for i, fname in enumerate(self.filenames)}
        
        # 2. Map every sequence to its File ID (integer is much smaller than string)
        self.sequence_to_file_id = [self.file_id_lookup[fname] for fname, _ in self.all_indices]

        if verbose:
            # print in bold Dataset Summary:
            print("\n\033[1mDataset Summary:\033[0m ---------------------------------------------------------------------------")
            print(f"    Your dataset is made of {len(self.filenames)} audio files")
            print(f"    Atoms were extracted using {self.atoms_frames} frames and they overlap with each other in {self.atoms_overlap_frames} frames.")
            time_per_atom = self.atoms_frames * self.samples_per_frame / self.sr
            time_per_overlap = self.atoms_overlap_frames * self.samples_per_frame / self.sr
            unique_time = self.segment_samples / self.sr - 2 * self.overlap_samples / self.sr
            print(f"    With this setting every atom is {1000*time_per_atom} ms long and its overlapped on each side by {1000*time_per_overlap} ms.")
            print(f"    This imply that all atoms carry exactly {1000*unique_time} ms of unique information.")
            print(f"    Your atoms are then grouped together in sequences of {self.segment_length} atoms and a new sequence is created every {self.hop_size} atoms.")
            print(f"    With this settings each sequence carry {self.segment_length*unique_time} seconds of unique information and is created every {self.hop_size*unique_time} seconds.")
            print(f"    Your dataset has {len(self.all_indices)} sequences in total.")
            print(f"    For each sequence, other two sequences are created automatically: present and context window")
            print(f"    Present correspond to the atom that goes exactly after the main (past) sequence ends. This is used as target when training.")
            print(f"    Context window is another sequence that start {self.context_shift} atoms after the main (past) sequence ends and it is {self.context_length} atoms long.")
            print(f"    Context window is used to compute context embeddings that are used to help predicting the present atom.")
            print(f"    Requested keys: {self.requested_keys}")
            self.check_if_manifest_has_splits()
            self.check_annotations_exist()
        else:
            print("\n\033[1mDataset Summary:\033[0m ---------------------------------------------------------------------------")
            print(f"    Audio files: {len(self.filenames)}")
            print(f"    Atoms: {self.atoms_frames} frames, {self.atoms_overlap_frames} frames overlap")
            print(f"    Sequence: {self.segment_length} atoms, hop every {self.hop_size} atoms")
            print(f"    Context window: {self.context_length} atoms, shifted from past by {self.context_shift} atoms")
            print(f"    Total sequences: {len(self.all_indices)}")
            print(f"    Requested keys: {self.requested_keys}")
            self.check_if_manifest_has_splits()
            self.check_annotations_exist()

    def check_if_manifest_has_splits(self):
        """
        Checks if the manifest has a complete train/val split and prints the file counts.
        Returns True if a valid split exists, False otherwise.
        """
        # 1. Ensure every file has the 'validation' key
        has_split = all("validation" in self.manifest[f] for f in self.filenames)
        
        if not has_split:
            print("    No complete split found in manifest.json.")
            print("    → Run dataset.make_split(val_split=0.1) to create a random split with 10% validation data or organize your files into train/val folders and run dataset.make_split() to create a split from directory structure.")
            return False
            
        # 2. Count the files in each split
        train_count = sum(1 for f in self.filenames if self.manifest[f]["validation"] is False)
        val_count = sum(1 for f in self.filenames if self.manifest[f]["validation"] is True)
        
        print(f"    Manifest has an existing split: {train_count} train files, {val_count} val files.")
        return True
    
    def check_annotations_exist(self): 
        base_anno_path = self.annotations_dir / self.config_folder_name
        if not base_anno_path.exists():
            print(f"    There are not annotations for your settings (they should be in {base_anno_path}).")
            return False
        
        print(f"    Annotations found in {base_anno_path}:")
        # list recursive subdirectories
        any_emb = False
        for cat in ["ctx", "clap"]:
            for time_part in ["past", "context_win"]:
                path = base_anno_path / cat / time_part
                # print(f"    Checking {cat}_{time_part} in {path}...")
                if not path.exists():
                    print(f"    ✗ {cat}_{time_part} (should be in {path})")
                else:
                    num_files = len(list(path.glob("emb_*.pt")))
                    print(f"    ✓ {cat}_{time_part}: Found {num_files} files in {path}")
                    any_emb = True
        return any_emb

    def _build_ola_window(self):
        window = torch.ones(self.segment_samples - 2 * self.overlap_samples)
        hann_window = torch.hann_window(self.overlap_samples * 2)
        left_hann = hann_window[:self.overlap_samples]
        right_hann = hann_window[self.overlap_samples:]
        return torch.cat([left_hann, window, right_hann])

    def _build_mapping(self, filenames):
        mapping = []
        for fname in filenames:
            count = self.manifest[fname]["atoms_count"]
            # Check against the new max boundary (self.total_length)
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
        """Helper to get start and count for audio loaders."""
        if part == "past":
            return start_idx, self.segment_length
        elif part == "context":
            return start_idx + self.context_shift, self.context_length
        elif part == "full":
            return start_idx, self.total_length
        else:
            raise ValueError("part must be 'past', 'context', or 'full'")

    def get_raw_audio(self, idx, part="past"):
        """Loads raw audio. Part can be 'past', 'context', or 'full'."""
        filename, seq_start_idx = self.all_indices[idx]
        atom_start_idx, atom_count = self._get_part_indices(seq_start_idx, part)
        
        audio_path = self.manifest[filename]["path"]

        start_sample = atom_start_idx * self.hop_samples
        last_atom_idx = atom_start_idx + (atom_count - 1)
        end_sample = (last_atom_idx * self.hop_samples) + self.segment_samples
        duration_samples = end_sample - start_sample

        audio_input, _ = librosa.load(audio_path, sr=self.sr, mono=False)
        audio_input = torch.tensor(audio_input).unsqueeze(0) 
        
        if audio_input.dim() == 2:
            audio_input = audio_input.unsqueeze(1).repeat(1, 2, 1)
        elif audio_input.shape[1] > 2:
            audio_input = audio_input[:, :2, :]
            
        audio_input = audio_input[:, :2, start_sample:start_sample+duration_samples]
        return audio_input.squeeze(0).to(self.device)

    def get_decoded_audio(self, idx, processor, part="past"):
        """Decodes OLA audio. Part can be 'past', 'context', or 'full'."""
        filename, seq_start_idx = self.all_indices[idx]
        atom_start_idx, atom_count = self._get_part_indices(seq_start_idx, part)
        
        total_samples = (atom_count - 1) * self.hop_samples + self.segment_samples
        out_audio = torch.zeros((1, 2, total_samples), device=processor.device)
        window = self.window.to(processor.device)

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
            end_s = start_s + self.segment_samples
            out_audio[:, :, start_s:end_s] += decoded_chunk[:, :, :self.segment_samples] * window
            
        return out_audio.squeeze(0)

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