import math
import warnings
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
from torch.utils.data import Dataset, Subset

from SCAPES.codecs import get_codec_info
from SCAPES.data.config_loader import DataprepConfig
from SCAPES.data.dataprep.semantic import CLAP_VERSION, load_audio_48k
from SCAPES.data.dataprep.structure import structure_key
from SCAPES.data.geometry import resolve_geometry
from SCAPES.data.representations import RepresentationStore, frames_to_samples_48k, grid_starts_48k, ticks_to_samples_48k
from SCAPES.data.splits import assign_region, region_bounds
from audio_codec_wrappers.audio import adapt_audio

SR = 48000
VALID_KEYS = {
    "memory_buffer_latent", "target_latent",
    "memory_buffer_scale", "target_scale",
    "target_semantic", "target_structure",
    "target_audio", "target_context_audio",
    "index", "label", "start_frame",
}


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


# A Subset that remembers its time region ("train" | "val" | "reserve")
class RegionSubset(Subset):
    def __init__(self, dataset, indices, region):
        super().__init__(dataset, indices)
        self.region = region


class AtomSequenceDataset(Dataset):
    """Training samples of both generation modes, read from the representation store (design: Training samples)."""

    def __init__(self, dataset_path, config: DataprepConfig, requested_keys=None, device="cpu", verbose=False):
        self.dataset_path = Path(dataset_path)
        self.config = config
        self.device = _resolve_device(device)
        self.info = get_codec_info(config.codec_name)
        self.geom = geom = resolve_geometry(config, self.info)
        self.store = RepresentationStore(self.dataset_path, geom, self.info)
        self.store.ensure()

        if requested_keys is None:
            self.requested_keys = set(VALID_KEYS)
        else:
            bad = [k for k in requested_keys if k not in VALID_KEYS]
            if bad:
                raise ValueError(f"Invalid keys {bad} in requested_keys. Valid keys are: {sorted(VALID_KEYS)}")
            self.requested_keys = set(requested_keys)

        self.has_scale = geom.has_scale
        self.frame_dim = geom.frame_dim
        self.latent_dim = geom.latent_dim
        self.per_atom = geom.encoding != "whole"
        self.semantic_random_extension = config.semantic_random_extension
        self.structure_feature_names = list(config.structure_features or [])
        self.context_samples = int(round(geom.context_seconds * SR))
        self.is_atoms = geom.atom_type != "frames"
        if self.is_atoms:
            self.target_samples = int(round(geom.atom_samples * SR / geom.sample_rate))
        else:
            self.target_samples = int(frames_to_samples_48k(geom.unit_frames, geom))
        self.clap_key = self.store.clap_key(CLAP_VERSION, geom.context_seconds, config.semantic_random_extension)
        self.structure_key = (structure_key(self.structure_feature_names, geom)
                              if self.structure_feature_names else None)

        index = self.store.read_index()
        if not index:
            raise FileNotFoundError(f"No index.json in {self.store.root}: run dataprep first")
        self._entries = {Path(e["path"]).name: e for e in index.values()}
        self.filenames = sorted(self._entries)

        # All latents of the representation are held in RAM
        self._latent, self._scale, self._units = {}, {}, {}
        for fname in self.filenames:
            data = self.store.load_latents(self.audio_path(fname))
            self._latent[fname] = data["latent"]
            self._scale[fname] = data.get("scale")
            self._units[fname] = data["latent"].shape[0] if self.per_atom else data["latent"].shape[-1]

        self._bounds = {f: region_bounds(self._entries[f]["duration_s"], config.val_split_ratio,
                                         config.reserve_split_ratio) for f in self.filenames}
        self.all_indices, self._regions = self._enumerate()

        # CLAP and structure arrays, only for the keys that are requested
        self._clap, self._structure = {}, {}
        if "target_semantic" in self.requested_keys:
            for fname in self.filenames:
                self._clap[fname] = self.store.load_grid_array("clap", self.clap_key, self.audio_path(fname))
        if "target_structure" in self.requested_keys and self.structure_key:
            for fname in self.filenames:
                self._structure[fname] = self.store.load_grid_array("structure", self.structure_key,
                                                                    self.audio_path(fname))
        self._audio_cache = {}

        if verbose:
            self._print_summary()

    def audio_path(self, fname) -> Path:
        return self.dataset_path / self._entries[fname]["path"]

    # Valid target start frames t = g * stride: a full past, a target inside the file, a CLAP window that fits, and the
    # whole span [past start, max(target end, CLAP window end)] inside one region.
    def _enumerate(self):
        if self.is_atoms:
            return self._enumerate_new()
        g = self.geom
        f = g.frame_rate
        past = g.past_frames
        indices, regions = [], []
        for fname in self.filenames:
            entry = self._entries[fname]
            n48 = entry["num_samples_48k"]
            count = len(grid_starts_48k(n48, g, self.context_samples))
            if self.structure_feature_names:
                count = min(count, len(grid_starts_48k(n48, g, self.target_samples)))
            units = self._units[fname]
            for gi in range(count):
                t = gi * g.stride_frames
                if t < past:
                    continue
                if self.per_atom:
                    if t // g.unit_hop_frames > units - 1:
                        break
                elif t + g.unit_frames > units:
                    break
                span = ((t - past) / f, max((t + g.unit_frames) / f, t / f + g.context_seconds))
                region = assign_region(span, self._bounds[fname])
                if region is None:
                    continue
                indices.append((fname, t))
                regions.append(region)
        return indices, regions

    # Sample-level atoms: grid start g is the atom k = g * stride_ticks. Needs k >= M (a full past) and k <= K - 1, a CLAP
    # window that fits, and the span [(k - M) H, max(k H + C, k H + ctx)] inside one region.
    def _enumerate_new(self):
        g = self.geom
        r, C, H, M = g.sample_rate, g.atom_samples, g.hop_samples, g.memory_units
        ctx = g.context_seconds
        indices, regions = [], []
        for fname in self.filenames:
            n48 = self._entries[fname]["num_samples_48k"]
            count = len(grid_starts_48k(n48, g, self.context_samples))
            if self.structure_feature_names:
                count = min(count, len(grid_starts_48k(n48, g, self.target_samples)))
            units = self._units[fname]
            for gi in range(count):
                k = gi * g.stride_ticks
                if k < M:
                    continue
                if k > units - 1:
                    break
                span = ((k - M) * H / r, max((k * H + C) / r, k * H / r + ctx))
                region = assign_region(span, self._bounds[fname])
                if region is None:
                    continue
                indices.append((fname, k))
                regions.append(region)
        return indices, regions

    def _print_summary(self):
        g = self.geom
        print("\nDataset summary ---------------------------------------------------------------")
        print(f"    {len(self.filenames)} files, representation {self.store.key}")
        if self.is_atoms:
            counts = {r: self._regions.count(r) for r in ("train", "val", "reserve")}
            print(f"    atom {g.atom_samples} samples, hop {g.hop_samples}, stride {g.stride_ticks} hops, past {g.memory_units} atoms")
            print(f"    {len(self.all_indices)} samples: {counts}")
            print(f"    Requested keys: {sorted(self.requested_keys)}")
            return
        print(f"    unit {g.unit_frames} frames ({g.unit_frames / g.frame_rate:.4f} s), stride {g.stride_frames} frames, "
              f"past {g.past_frames} frames, conditioning rate {g.frame_rate / g.stride_frames:.2f} Hz")
        counts = {r: self._regions.count(r) for r in ("train", "val", "reserve")}
        print(f"    {len(self.all_indices)} samples: {counts}")
        print(f"    Requested keys: {sorted(self.requested_keys)}")

    def __len__(self):
        return len(self.all_indices)

    def _subset(self, region):
        idx = [i for i, r in enumerate(self._regions) if r == region]
        return RegionSubset(self, idx, region)

    def get_splits(self):
        train, val = self._subset("train"), self._subset("val")
        print(f"Time split: {len(train)} train samples, {len(val)} val samples, {len(self._subset('reserve'))} reserve.")
        return train, val

    def get_reserve(self):
        return self._subset("reserve")

    # Start of the target in seconds
    def target_time(self, idx) -> float:
        if self.is_atoms:
            return self.all_indices[idx][1] * self.geom.hop_samples / self.geom.sample_rate
        return self.all_indices[idx][1] / self.geom.frame_rate

    # Per-channel mean and population std of the frames inside the region of the subset, over all files.
    # Frame t is in [a, b) iff a <= t/f and (t+1)/f <= b. Per-atom representations are joined by taking the first
    # H frames of every atom and the whole last atom.
    def latent_stats(self, subset):
        region = getattr(subset, "region", None)
        if region is None:
            raise ValueError("latent_stats needs a subset from get_splits() or get_reserve()")
        if self.is_atoms:
            return self._latent_stats_new(region)
        f, H = self.geom.frame_rate, self.geom.unit_hop_frames
        total = torch.zeros(self.latent_dim, dtype=torch.float64)
        total_sq = torch.zeros(self.latent_dim, dtype=torch.float64)
        n = 0
        for fname in self.filenames:
            u = self._latent[fname]
            if self.per_atom:
                if u.shape[0] == 0:
                    continue
                u = torch.cat([u[:-1, :, :H].permute(1, 0, 2).reshape(self.latent_dim, -1), u[-1].float()], dim=1)
            lo, hi = self._bounds[fname][region]
            t0 = max(0, math.ceil(lo * f - 1e-9))
            t1 = min(u.shape[-1], math.floor(hi * f + 1e-9))
            if t1 <= t0:
                continue
            x = u[:, t0:t1].double()
            total += x.sum(dim=1)
            total_sq += (x * x).sum(dim=1)
            n += x.shape[1]
        if n == 0:
            raise ValueError(f"no frames in region {region!r}")
        mu = total / n
        var = (total_sq / n - mu * mu).clamp_min(0.0)
        return mu.float(), var.sqrt().clamp_min(1e-5).float()

    # Sample-level atoms: every frame of every atom k with [kH/r, (kH+C)/r] inside the region (D9).
    def _latent_stats_new(self, region):
        g = self.geom
        r, C, H = g.sample_rate, g.atom_samples, g.hop_samples
        total = torch.zeros(self.latent_dim, dtype=torch.float64)
        total_sq = torch.zeros(self.latent_dim, dtype=torch.float64)
        n = 0
        for fname in self.filenames:
            u = self._latent[fname]
            lo, hi = self._bounds[fname][region]
            keep = [k for k in range(u.shape[0]) if lo - 1e-9 <= k * H / r and (k * H + C) / r <= hi + 1e-9]
            if not keep:
                continue
            x = u[keep].double().permute(1, 0, 2).reshape(self.latent_dim, -1)
            total += x.sum(dim=1)
            total_sq += (x * x).sum(dim=1)
            n += x.shape[1]
        if n == 0:
            raise ValueError(f"no frames in region {region!r}")
        mu = total / n
        var = (total_sq / n - mu * mu).clamp_min(0.0)
        return mu.float(), var.sqrt().clamp_min(1e-5).float()

    # 48 kHz stereo slice [2, n] of a file starting at `start`, zero-padded past the end
    def _raw_slice(self, fname, start, n):
        path = self.audio_path(fname)
        native = sf.info(str(path)).samplerate
        if native == SR:
            data, _ = sf.read(str(path), start=start, stop=start + n, dtype="float32", always_2d=True)
            x = adapt_audio(torch.from_numpy(data.T.copy()), SR, SR, 2)[0]
        else:
            if fname not in self._audio_cache:
                self._audio_cache = {fname: load_audio_48k(path)}
            x = self._audio_cache[fname][:, start:start + n]
        if x.shape[-1] < n:
            x = torch.cat([x, torch.zeros(2, n - x.shape[-1])], dim=-1)
        return x

    def __getitem__(self, idx):
        fname, t = self.all_indices[idx]
        g = self.geom
        req = self.requested_keys
        item = {"index": idx, "label": fname, "start_frame": t}
        u, scale = self._latent[fname], self._scale[fname]
        U, F, H, M = g.unit_frames, g.memory_frames, g.unit_hop_frames, g.memory_units
        k = t if self.is_atoms else t // H

        if "memory_buffer_latent" in req or "memory_buffer_scale" in req:
            if g.mode == "autoregressive":
                mem = u[:, t - F:t].unsqueeze(0)
            elif self.per_atom:
                mem = u[k - M:k].float()
            else:
                mem = torch.stack([u[:, t - (M - j) * H: t - (M - j) * H + F] for j in range(M)])
            if "memory_buffer_latent" in req:
                item["memory_buffer_latent"] = mem.float().to(self.device)
            if "memory_buffer_scale" in req and self.has_scale:
                item["memory_buffer_scale"] = scale[k - M:k].float().unsqueeze(1).to(self.device)
        if "target_latent" in req:
            tgt = u[k] if self.per_atom else u[:, t:t + U]
            item["target_latent"] = tgt.float().to(self.device)
        if "target_scale" in req and self.has_scale:
            item["target_scale"] = scale[k:k + 1].float().to(self.device)

        if "target_audio" in req or "target_context_audio" in req:
            start = int(ticks_to_samples_48k(t, g)) if self.is_atoms else int(frames_to_samples_48k(t, g))
            if "target_audio" in req:
                item["target_audio"] = self._raw_slice(fname, start, self.target_samples).to(self.device)
            if "target_context_audio" in req:
                item["target_context_audio"] = self._raw_slice(fname, start, self.context_samples).to(self.device)

        gi = t // g.stride_ticks if self.is_atoms else t // g.stride_frames
        if "target_semantic" in req:
            item["target_semantic"] = self._clap[fname][gi].to(self.device)
        if "target_structure" in req:
            if self.structure_key:
                item["target_structure"] = self._structure[fname][gi].to(self.device)
            else:
                item["target_structure"] = torch.zeros(0, device=self.device)
        return item
