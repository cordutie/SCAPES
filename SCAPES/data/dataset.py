import math
import random
import warnings
from pathlib import Path
from typing import Optional

import json
import librosa
import torch
from torch.utils.data import Dataset, Subset

from SCAPES.data.config_loader import DataprepConfig


def _auto_device():
    if hasattr(torch, "hpu") and torch.hpu.is_available():
        return "hpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_device(requested):
    if requested in [None, "auto"]:
        return _auto_device()
    if requested == "hpu" and not (hasattr(torch, "hpu") and torch.hpu.is_available()):
        warnings.warn("Requested device 'hpu' but it is not available. Falling back to auto.")
        return _auto_device()
    if requested == "cuda" and not torch.cuda.is_available():
        warnings.warn("Requested device 'cuda' but it is not available. Falling back to cpu.")
        return "cpu"
    return requested


def batch_from_latents_to_audio(batch, dataset, processor, mode="decoded", part="past"):
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
        config: Optional[DataprepConfig] = None,
        *,
        requested_keys=None,
        verbose=True,
        memory_buffer_atoms=None,
        hop_atoms=None,
        context_seconds=None,
        atoms_frames=None,
        atoms_hop_frames=None,
        crossfade_frames=None,
        seed=None,
        device=None,
        train_split=None,
        val_split=None,
        val_split_ratio=None,
        structure_features=None,
        semantic_random_extension=None,
    ):
        if config is not None:
            memory_buffer_atoms = config.memory_buffer_atoms
            hop_atoms = config.hop_atoms
            context_seconds = config.semantic_context_seconds
            atoms_frames = config.atoms_frames
            atoms_hop_frames = config.atoms_hop_frames
            crossfade_frames = config.atoms_crossfade_frames
            seed = config.seed
            device = config.device
            train_split = config.train_split
            val_split = config.val_split
            val_split_ratio = config.val_split_ratio
            structure_features = config.structure_features
            semantic_random_extension = config.semantic_random_extension

        missing = [k for k, v in [
            ("memory_buffer_atoms", memory_buffer_atoms),
            ("hop_atoms", hop_atoms),
            ("context_seconds", context_seconds),
            ("atoms_frames", atoms_frames),
            ("atoms_hop_frames", atoms_hop_frames),
            ("crossfade_frames", crossfade_frames),
            ("seed", seed),
            ("val_split_ratio", val_split_ratio),
            ("structure_features", structure_features),
            ("semantic_random_extension", semantic_random_extension),
        ] if v is None]
        if missing:
            raise TypeError(
                f"AtomSequenceDataset missing required arguments (provide 'config' or set explicitly): "
                f"{', '.join(missing)}"
            )

        if device is None or device == "auto":
            device = _auto_device()
        device = _resolve_device(device)

        self.dataset_path = Path(dataset_path)
        self.annotations_dir = self.dataset_path / "annotations"

        self.memory_buffer_atoms = memory_buffer_atoms
        self.hop_atoms = hop_atoms
        self.context_seconds = context_seconds
        self.atoms_frames = atoms_frames
        self.atoms_hop_frames = atoms_hop_frames
        self.crossfade_frames = crossfade_frames
        self.seed = seed
        self.device = device
        self.train_split_key = train_split
        self.val_split_key = val_split
        self.val_split_ratio = val_split_ratio
        self.semantic_random_extension = semantic_random_extension

        valid_keys = {
            "memory_buffer_latent", "target_latent",
            "memory_buffer_scale", "target_scale",
            "target_semantic", "target_structure",
            "target_audio", "target_context_audio",
            "index"
        }
        if requested_keys is None:
            self.requested_keys = valid_keys
            print(f"No requested_keys provided; defaulting to all: {self.requested_keys}")
        else:
            if not all(key in valid_keys for key in requested_keys):
                raise ValueError(f"Invalid keys in requested_keys. Valid keys are: {valid_keys}")
            self.requested_keys = requested_keys

        # Hardcoded codec constants
        self.sr = 48000
        self.frame_rate = 150
        self.samples_per_frame = self.sr // self.frame_rate

        # Load manifest (new location with fallback)
        self.atoms_manifest_path = self.dataset_path / "atoms" / "manifest.json"
        manifest_path = self.atoms_manifest_path
        if not manifest_path.exists():
            manifest_path = self.dataset_path / "config" / "manifest.json"
        with open(manifest_path, 'r') as f:
            raw = json.load(f)
        if "files" in raw:
            self.manifest = raw["files"]
            self.manifest_config_version = raw.get("config_version", {})
        else:
            self.manifest = raw
            self.manifest_config_version = {}

        # Structure feature config (from gin/config or explicit)
        self.structure_feature_names = structure_features
        self.structure_feature_index = {
            name: i for i, name in enumerate(self.structure_feature_names)
        }
        self.structure_feature_dimension = (
            len(self.structure_feature_names) if "target_structure" in self.requested_keys else 0
        )

        # Hardcoded structure params (removed from gin)
        self.structure_mean_pooling = True
        self.structure_hop_length = self.samples_per_frame
        self.structure_n_fft = max(256, self.structure_hop_length * 4)

        # --- Math for the Asymmetric Geometry ---
        self.macro_overlap_frames = self.atoms_frames - self.atoms_hop_frames

        self.atoms_samples = self.atoms_frames * self.samples_per_frame
        self.hop_samples = self.atoms_hop_frames * self.samples_per_frame
        self.crossfade_samples = self.crossfade_frames * self.samples_per_frame
        self.macro_overlap_samples = self.macro_overlap_frames * self.samples_per_frame

        # --- Context Window (Target-aligned, time-based) ---
        self.context_samples = int(round(self.context_seconds * self.sr))

        if self.context_samples <= self.atoms_samples:
            self.context_atoms_required = 1
        else:
            self.context_atoms_required = int(
                math.ceil((self.context_samples - self.atoms_samples) / self.hop_samples)
            ) + 1

        self.total_length = max(
            self.memory_buffer_atoms + 1,
            self.memory_buffer_atoms + self.context_atoms_required
        )

        self.filenames = sorted(list(self.manifest.keys()))
        self.all_indices = self._build_mapping(self.filenames)

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

            if val_flag == "partial":
                if start in self.manifest[f].get("val_starts", []):
                    val_indices.append(i)
                else:
                    train_indices.append(i)

            elif val_flag is True:
                val_indices.append(i)

            else:
                train_indices.append(i)

        train_subset = Subset(self, train_indices)
        val_subset = Subset(self, val_indices)

        print(f"Loaded split: {len(train_indices)} train sequences, "
              f"{len(val_indices)} val sequences.")

        return train_subset, val_subset

    def check_annotations_exist(self):
        if self.annotations_dir is None:
            return False

        any_found = False

        if "target_semantic" in self.requested_keys:
            semantic_dir = self.annotations_dir / "semantic"
            if not semantic_dir.exists():
                print(f"    ✗ semantic (should be in {semantic_dir})")
            else:
                num_files = len(list(semantic_dir.glob("semantic_*.pt")))
                print(f"    ✓ semantic: Found {num_files} files in {semantic_dir}")
                any_found = True

        if "target_structure" in self.requested_keys:
            structure_dir = self.annotations_dir / "structure"
            if not structure_dir.exists():
                print(f"    ✗ structure (should be in {structure_dir})")
            else:
                num_files = len(list(structure_dir.glob("structure_*.pt")))
                print(f"    ✓ structure: Found {num_files} files in {structure_dir}")
                any_found = True

        return any_found

    def _build_ola_window(self):
        zeros_frames = self.macro_overlap_frames - self.crossfade_frames
        zeros = torch.zeros(zeros_frames * self.samples_per_frame)

        hann_window = torch.hann_window(self.crossfade_samples * 2)
        left_hann = hann_window[:self.crossfade_samples]
        right_hann = hann_window[self.crossfade_samples:]

        ones_frames = self.atoms_hop_frames - self.crossfade_frames
        ones = torch.ones(ones_frames * self.samples_per_frame)

        window = torch.cat([zeros, left_hann, ones, right_hann])
        return window

    def _build_mapping(self, filenames):
        mapping = []
        for fname in filenames:
            count = self.manifest[fname]["atoms_count"]
            if count >= self.total_length:
                for start in range(0, count - self.total_length + 1, self.hop_atoms):
                    mapping.append((fname, start))
        return mapping

    def _get_atom_path(self, original_filename, atom_index):
        original_path = Path(self.manifest[original_filename]["path"])
        stem = original_path.stem
        parts = list(original_path.parts)
        try:
            raw_idx = parts.index("raw")
            relative_parent = Path(*parts[raw_idx + 1: -1])
        except ValueError:
            relative_parent = Path("")

        atom_filename = f"{stem}_atom_{atom_index}.pt"
        return self.dataset_path / "atoms" / relative_parent / stem / atom_filename

    def _get_part_indices(self, start_idx, part):
        if part == "past":
            return start_idx, self.memory_buffer_atoms
        elif part == "target":
            return start_idx + self.memory_buffer_atoms, 1
        elif part == "context":
            return start_idx + self.memory_buffer_atoms, self.context_atoms_required
        elif part == "full":
            return start_idx, self.total_length
        else:
            raise ValueError("part must be 'past', 'target', 'context', or 'full'")

    def _load_raw_audio_file(self, audio_path):
        audio_input, _ = librosa.load(audio_path, sr=self.sr, mono=False)
        audio_input = torch.tensor(audio_input)

        if audio_input.dim() == 1:
            audio_input = audio_input.unsqueeze(0)

        if audio_input.shape[0] == 1:
            audio_input = audio_input.repeat(2, 1)
        elif audio_input.shape[0] > 2:
            audio_input = audio_input[:2, :]

        return audio_input.to(self.device)

    def _slice_audio(self, audio, start_sample, duration_samples):
        end_sample = start_sample + duration_samples
        if end_sample > audio.shape[-1]:
            pad_len = end_sample - audio.shape[-1]
            pad = torch.zeros((audio.shape[0], pad_len), device=audio.device, dtype=audio.dtype)
            audio = torch.cat([audio, pad], dim=-1)
        return audio[:, start_sample:end_sample]

    def get_raw_audio(self, idx, part="past"):
        filename, seq_start_idx = self.all_indices[idx]
        audio_path = self.manifest[filename]["path"]
        audio_input = self._load_raw_audio_file(audio_path)

        if part in ["context", "target_context"]:
            target_start = (seq_start_idx + self.memory_buffer_atoms) * self.hop_samples
            return self._slice_audio(audio_input, target_start, self.context_samples)

        if part == "target":
            target_start = (seq_start_idx + self.memory_buffer_atoms) * self.hop_samples
            return self._slice_audio(audio_input, target_start, self.atoms_samples)

        atom_start_idx, atom_count = self._get_part_indices(seq_start_idx, part)
        start_sample = atom_start_idx * self.hop_samples
        duration_samples = ((atom_count - 1) * self.hop_samples) + self.atoms_samples

        return self._slice_audio(audio_input, start_sample, duration_samples)

    def get_decoded_audio(self, idx, processor, part="past"):
        filename, seq_start_idx = self.all_indices[idx]

        if part in ["context", "target_context"]:
            atom_start_idx = seq_start_idx + self.memory_buffer_atoms
            atom_count = self.context_atoms_required
        elif part == "target":
            atom_start_idx = seq_start_idx + self.memory_buffer_atoms
            atom_count = 1
        else:
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

            out_audio[:, :, start_s:end_s] += decoded_chunk[:, :, :self.atoms_samples] * window

        if part in ["context", "target_context"]:
            out_audio = out_audio[:, :, :self.context_samples]

        return out_audio.squeeze(0)

    def _save_manifest(self):
        path = self.atoms_manifest_path
        path.parent.mkdir(parents=True, exist_ok=True)
        full = {
            "config_version": self.manifest_config_version,
            "files": self.manifest,
        }
        with open(path, "w") as f:
            json.dump(full, f, indent=4)

    def make_split(self, val_split=None, seed=None, overwrite=False):
        if seed is None:
            seed = self.seed
        if seed is not None:
            random.seed(seed)

        if val_split is None:
            val_split = self.val_split_ratio
        if val_split == "dataprep":
            val_split = None

        if self.train_split_key or self.val_split_key:
            if val_split is not None:
                raise ValueError(
                    "val_split was provided but config defines train_split/val_split directories."
                )

            if not (self.train_split_key and self.val_split_key):
                raise ValueError(
                    "config must define BOTH 'train_split' and 'val_split'."
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

            self._save_manifest()

            print(
                f"Split created from directory structure: "
                f"{len(train_files)} train files, {len(val_files)} val files."
            )

            return

        if val_split is None:
            raise ValueError(
                "val_split_ratio must be set in config or passed to make_split()."
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
            count = self.manifest[f]["atoms_count"]
            starts = list(range(0, count - self.total_length + 1, self.hop_atoms))

            split_idx = int(len(starts) * (1 - val_split))

            val_starts = starts[split_idx:]

            self.manifest[f]["validation"] = "partial"
            self.manifest[f]["val_starts"] = val_starts

            total_train_seqs += split_idx
            total_val_seqs += len(val_starts)

        self._save_manifest()

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
            "label": filename
        }
        req = self.requested_keys

        if any(k in req for k in ["memory_buffer_latent", "memory_buffer_scale", "target_latent", "target_scale"]):
            past_latents, past_scales = self._load_atom_sequence(filename, start_idx, self.memory_buffer_atoms)

            if "memory_buffer_latent" in req:
                batch_dict["memory_buffer_latent"] = past_latents.to(self.device)
            if "memory_buffer_scale" in req:
                batch_dict["memory_buffer_scale"] = past_scales.to(self.device)

            if "target_latent" in req or "target_scale" in req:
                present_path = self._get_atom_path(filename, start_idx + self.memory_buffer_atoms)
                present_atom = torch.load(present_path, weights_only=True, map_location='cpu')
                if "target_latent" in req:
                    batch_dict["target_latent"] = present_atom["latent"].squeeze(0).float().to(self.device)
                if "target_scale" in req:
                    batch_dict["target_scale"] = present_atom["scale"].squeeze(0).float().to(self.device)

        if "target_audio" in req or "target_context_audio" in req:
            audio_path = self.manifest[filename]["path"]
            audio_input = self._load_raw_audio_file(audio_path)
            target_start = (start_idx + self.memory_buffer_atoms) * self.hop_samples

            if "target_audio" in req:
                batch_dict["target_audio"] = self._slice_audio(audio_input, target_start, self.atoms_samples)
            if "target_context_audio" in req:
                batch_dict["target_context_audio"] = self._slice_audio(audio_input, target_start, self.context_samples)

        if self.annotations_dir is not None:
            semantic_dir = self.annotations_dir / "semantic"
            structure_dir = self.annotations_dir / "structure"

            if "target_semantic" in req:
                path = semantic_dir / f"semantic_{idx}.pt"
                if not path.exists():
                    raise FileNotFoundError(f"Requested target_semantic but file missing: {path}")
                batch_dict["target_semantic"] = torch.load(path, weights_only=True, map_location='cpu').to(self.device)

            if "target_structure" in req:
                path = structure_dir / f"structure_{idx}.pt"
                if not path.exists():
                    raise FileNotFoundError(f"Requested target_structure but file missing: {path}")
                batch_dict["target_structure"] = torch.load(path, weights_only=True, map_location='cpu').to(self.device)

        return batch_dict
