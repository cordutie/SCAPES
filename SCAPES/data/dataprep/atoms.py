"""Latent extraction: one encoding of every raw file, stored in the representation store (design D8, D9)."""
from pathlib import Path

import soundfile as sf
import torch

from SCAPES.codecs import get_codec, get_codec_info
from SCAPES.data.config_loader import DataprepConfig
from SCAPES.data.geometry import atom_count, resolve_geometry
from SCAPES.generation import print_resolution
from SCAPES.data.representations import RepresentationStore
from audio_codec_wrappers.audio import load_audio

AUDIO_EXTENSIONS = ("*.wav", "*.mp3", "*.flac", "*.ogg")


# V1 atom counts: how many atoms of segment_samples fit, and how many keep a full context window after their start.
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


def list_audio_files(dataset_path) -> list[Path]:
    raw = Path(dataset_path) / "raw"
    files = sorted(p for ext in AUDIO_EXTENSIONS for p in raw.rglob(ext))
    stems = [p.stem for p in files]
    if len(stems) != len(set(stems)):
        raise ValueError("Duplicate file stems detected. Ensure all names are unique across subfolders.")
    return files


# Per-atom encoding (V1 logic): each atom of F frames is encoded alone, so the codec's own normalisation applies per
# atom. Returns (latent [K, d, F] float32, scale [K] float32) on the cpu.
def _encode_atoms(codec, x, geom, context_seconds):
    r, h = geom.sample_rate, geom.hop_length
    seg, hop = geom.unit_frames * h, geom.unit_hop_frames * h
    ctx = int(round(context_seconds * r))
    _, count = _compute_atom_counts(x.shape[-1], seg, hop, context_samples=ctx)
    latents, scales = [], []
    for i in range(count):
        enc = codec.encode(x[..., i * hop: i * hop + seg], r)
        latents.append(enc.z[0].cpu())
        scales.append(enc.scale.reshape(-1)[:1].cpu() if enc.scale is not None else torch.ones(1))
    if not latents:
        return torch.zeros(0, geom.latent_dim, geom.unit_frames), torch.zeros(0)
    return torch.stack(latents), torch.cat(scales)


# Sample-level atoms: segment k is x[0, :, kH:kH+C], encoded on its own (so the codec normalises each segment itself).
# Segments go to the codec in batches. Returns (latent [K, d, F] float32, scale [K] float32) on the cpu.
# batch None: segments per encode call from a sample budget, max(1, 2**22 // C) (design D9).
def _encode_segments(codec, x, geom, batch=None):
    r, C, H = geom.sample_rate, geom.atom_samples, geom.hop_samples
    if batch is None:
        batch = max(1, 2 ** 22 // C)
    K = atom_count(x.shape[-1], C, H)
    latents, scales = [], []
    for b0 in range(0, K, batch):
        segs = torch.stack([x[0, :, k * H: k * H + C] for k in range(b0, min(K, b0 + batch))])
        enc = codec.encode(segs, r)
        z = enc.z.float().cpu()
        latents.append(z)
        if enc.scale is not None:
            scales.append(enc.scale.reshape(z.shape[0], -1)[:, 0].float().cpu())
        else:
            scales.append(torch.ones(z.shape[0]))
    if not latents:
        return torch.zeros(0, geom.latent_dim, geom.unit_frames), torch.zeros(0)
    return torch.cat(latents), torch.cat(scales)


# Segments per encode call on the GPU: the largest batch whose measured memory fits in the free memory (TODO 2).
# Segment size depends on the codec only, not on the file, so one measurement serves the whole run.
def _auto_encode_batch(codec, x, geom):
    from SCAPES.auxiliar.gpu_batch import auto_batch
    seg = x[0, :, :geom.atom_samples]

    def step(b):
        with torch.no_grad():
            codec.encode(torch.stack([seg] * b), geom.sample_rate)

    batch, free, fixed, per_item = auto_batch(step)
    print(f"encode batch auto -> {batch} segments (free {free / 2**30:.2f} GiB, fixed {fixed / 2**20:.0f} MiB, "
          f"per segment {per_item / 2**20:.1f} MiB)")
    return batch


# Encodes every raw file that has no latents yet and writes index.json. Returns the store.
def latents_maker(dataset_path, config: DataprepConfig, device="auto", codec=None) -> RepresentationStore:
    dataset_path = Path(dataset_path)
    if device in (None, "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    info = get_codec_info(config.codec_name)
    geom = resolve_geometry(config, info)
    store = RepresentationStore(dataset_path, geom, info)
    store.ensure()
    print(f"Representation {store.key} at {store.root}")
    print_resolution(geom.resolution)

    files = list_audio_files(dataset_path)
    index = store.read_index()
    per_atom = geom.encoding != "whole"
    encode_batch = None   # on CUDA: segments per encode call, measured once on the first file long enough
    for path in files:
        stem = path.stem
        entry = index.get(stem)
        done = store.has_latents(path) and entry is not None
        if done:
            continue
        # duration and the sample count at 48 kHz come from the file header
        header = sf.info(str(path))
        entry = {"path": str(path.relative_to(dataset_path)),
                 "duration_s": header.frames / header.samplerate,
                 "num_samples_48k": int(round(header.frames * 48000 / header.samplerate))}
        if store.has_latents(path):
            data = store.load_latents(path)
        else:
            print(f"Encoding {path.name}...")
            if codec is None:
                codec = get_codec(config.codec_name, device)
            audio, sr = load_audio(path)
            if per_atom:
                x = codec.prepare(audio, sr)
                if geom.atom_type != "frames":
                    if device == "cuda" and encode_batch is None and x.shape[-1] >= geom.atom_samples:
                        encode_batch = _auto_encode_batch(codec, x, geom)
                    latent, scale = _encode_segments(codec, x, geom, batch=encode_batch)
                else:
                    latent, scale = _encode_atoms(codec, x, geom, config.semantic_context_seconds)
                store.save_latents(path, latent, scale)
            else:
                latent = codec.encode(audio, sr).z[0]
                store.save_latents(path, latent)
            data = store.load_latents(path)
        if per_atom:
            entry["num_atoms"] = int(data["latent"].shape[0])
        else:
            entry["num_frames"] = int(data["latent"].shape[-1])
        index[stem] = entry
    # drop entries of files that are gone
    index = {s: e for s, e in index.items() if s in {p.stem for p in files}}
    store.write_index(dict(sorted(index.items())))
    return store
