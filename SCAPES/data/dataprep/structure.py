import math
from typing import List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm


def _ensure_mono(audio: torch.Tensor) -> torch.Tensor:
    if audio.dim() == 1:
        return audio.unsqueeze(0)
    if audio.dim() == 2:
        return audio.mean(dim=0, keepdim=True) if audio.shape[0] > 1 else audio
    if audio.dim() == 3:
        return audio.mean(dim=1)
    raise ValueError(f"Unexpected audio shape: {audio.shape}")


def _frame_audio(audio: torch.Tensor, frame_length: int, hop_length: int) -> torch.Tensor:
    if audio.dim() == 2:
        audio = audio.unsqueeze(0)
    if audio.shape[-1] < frame_length:
        pad_len = frame_length - audio.shape[-1]
        audio = F.pad(audio, (0, pad_len))
    return audio.unfold(-1, frame_length, hop_length)


def _prepare_stft(
    audio: torch.Tensor,
    atoms_frames: int,
    n_fft: int,
    hop_length: Optional[int],
):
    audio_mono = _ensure_mono(audio).to(torch.float32)
    if audio_mono.dim() == 1:
        audio_mono = audio_mono.unsqueeze(0)

    audio_len = audio_mono.shape[-1]
    if atoms_frames <= 0:
        raise ValueError("atoms_frames must be > 0")

    if hop_length is None:
        if atoms_frames == 1:
            hop_length = max(1, audio_len)
        else:
            if audio_len <= n_fft:
                hop_length = 1
            else:
                hop_length = max(1, int(round((audio_len - n_fft) / (atoms_frames - 1))))

    required_len = n_fft + hop_length * (atoms_frames - 1)
    if audio_len < required_len:
        audio_mono = F.pad(audio_mono, (0, required_len - audio_len))
    elif audio_len > required_len:
        audio_mono = audio_mono[..., :required_len]

    window = torch.hann_window(n_fft, device=audio_mono.device)
    stft = torch.stft(
        audio_mono,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=False,
        return_complex=True,
    )

    mag = stft.abs().clamp_min(1e-8)
    if mag.shape[-1] != atoms_frames:
        if mag.shape[-1] < atoms_frames:
            mag = F.pad(mag, (0, atoms_frames - mag.shape[-1]))
        else:
            mag = mag[..., :atoms_frames]

    return audio_mono, mag, hop_length

def _spectral_centroid(mag: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    mag_sum = mag.sum(dim=1).clamp_min(1e-8)
    return torch.log2((mag * freqs).sum(dim=1) / mag_sum)

def _spectral_bandwidth(mag: torch.Tensor, freqs: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
    mag_sum = mag.sum(dim=1).clamp_min(1e-8)
    centroid_exp = centroid.unsqueeze(1)
    return torch.log2(torch.sqrt(((freqs - centroid_exp) ** 2 * mag).sum(dim=1) / mag_sum))

def _spectral_flatness(mag: torch.Tensor) -> torch.Tensor:
    geo_mean = torch.exp(torch.mean(torch.log(mag.clamp_min(1e-8)), dim=1))
    arith_mean = mag.mean(dim=1).clamp_min(1e-8)
    return geo_mean / arith_mean

def _spectral_entropy(mag: torch.Tensor) -> torch.Tensor:
    power = (mag ** 2).clamp_min(1e-8)
    p = power / power.sum(dim=1, keepdim=True).clamp_min(1e-8)
    entropy = -(p * torch.log(p)).sum(dim=1)
    norm = math.log(power.shape[1])
    if norm > 0:
        entropy = entropy / norm
    return entropy

def _acoustic_complexity(mag: torch.Tensor) -> torch.Tensor:
    diff = mag[:, 1:, :] - mag[:, :-1, :]
    denom = mag[:, 1:, :].clamp_min(1e-8)
    complexity = (diff.abs() / denom).sum(dim=1)
    complexity = torch.log2(complexity + 1.0)
    return complexity

def _spectral_flux(mag: torch.Tensor) -> torch.Tensor:
    log_mag = torch.log(mag.clamp_min(1e-8))
    diff = log_mag[:, :, 1:] - log_mag[:, :, :-1]
    flux = F.relu(diff).sum(dim=1)
    return F.pad(flux, (1, 0))

def _transient_density(mag: torch.Tensor) -> torch.Tensor:
    flux = _spectral_flux(mag)
    norm = flux.mean(dim=1, keepdim=True).clamp_min(1e-8)
    return torch.log2((flux/norm)+1)

def _rms(frames: torch.Tensor) -> torch.Tensor:
    return 10*torch.sqrt(torch.mean(frames ** 2, dim=-1).clamp_min(1e-8))

def _reshaper(feature: torch.Tensor, target_frames: int) -> torch.Tensor:
    # if 4d squeeze it so it is 3d
    if feature.dim() == 4:
        feature = feature.squeeze(0)
    # if 3d squeeze it so it is 2d
    if feature.dim() == 3:
        feature = feature.squeeze(0)
    # if last dim size is not target_frames, resample it using linear interpolation
    if feature.shape[-1] != target_frames:
        feature = F.interpolate(feature.unsqueeze(0), size=target_frames, mode='linear', align_corners=False).squeeze(0)
    return feature

def _compute_structure_features(
    audio: torch.Tensor,
    sr: int,
    atoms_frames: int,
    n_fft: int,
    hop_length: Optional[int],
    feature_names: Optional[List[str]] = None,
    mean_pooling: bool = True,
) -> torch.Tensor:
    if feature_names is None:
        feature_names = [
            "acoustic_complexity",
            "spectral_entropy",
            "transient_density",
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_flatness",
            "rms",
        ]

    audio_mono, mag, hop_length = _prepare_stft(
        audio=audio,
        atoms_frames=atoms_frames,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    freqs = torch.linspace(0.0, sr / 2.0, mag.shape[1], device=mag.device).view(1, -1, 1)
    frames = _frame_audio(audio_mono, frame_length=n_fft, hop_length=hop_length)
    if frames.shape[1] != atoms_frames:
        if frames.shape[1] < atoms_frames:
            pad_len = atoms_frames - frames.shape[1]
            frames = F.pad(frames, (0, 0, 0, pad_len))
        else:
            frames = frames[:, :atoms_frames, :]

    centroid = _spectral_centroid(mag, freqs)
    bandwidth = _spectral_bandwidth(mag, freqs, centroid)

    feature_map = {
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_flatness": _spectral_flatness(mag),
        "spectral_entropy": _spectral_entropy(mag),
        "acoustic_complexity": _acoustic_complexity(mag),
        "transient_density": _transient_density(mag),
        "rms": _rms(frames),
    }

    unknown = [name for name in feature_names if name not in feature_map]
    if unknown:
        raise ValueError(f"Unknown structure feature(s): {unknown}")

    stacked = torch.stack([_reshaper(feature_map[name], atoms_frames) for name in feature_names], dim=1).squeeze(0)
    # print(f"Computed structure features with shape: {stacked.shape} (features x frames)")
    if mean_pooling:
        return (stacked.mean(dim=-1)) # [features]
    return stacked

def precompute_structure_annotations(
    dataset,
    batch_size: int = 1,
    device: str = "auto",
    n_fft: Optional[int] = None,
    hop_length: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
):
    """
    Computes placeholder DSP features from target_audio and saves them to:
    annotations/structure/structure_<idx>.pt
    """
    mean_pooling = dataset.structure_mean_pooling

    if dataset.annotations_dir is None:
        raise ValueError("Dataset annotations_dir is not set.")

    if device in [None, "auto"]:
        device = getattr(dataset, "device", "cpu")

    if n_fft is None:
        n_fft = getattr(dataset, "structure_n_fft", max(256, dataset.samples_per_frame * 4))
    if hop_length is None:
        hop_length = getattr(dataset, "structure_hop_length", dataset.samples_per_frame)
    if feature_names is None:
        feature_names = getattr(dataset, "structure_feature_names", None)
    save_dir = dataset.annotations_dir / "structure"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Starting Structure Pre-computation ---")
    print(f"Save Path: {save_dir}")

    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing structure"):
        for j in range(batch_size):
            idx = i + j
            if idx >= len(dataset):
                break

            save_path = save_dir / f"structure_{idx}.pt"
            if save_path.exists():
                continue

            raw_audio = dataset.get_raw_audio(idx, part="target").to(device)
            features = _compute_structure_features(
                raw_audio,
                sr=dataset.sr,
                atoms_frames=dataset.atoms_frames,
                n_fft=n_fft,
                hop_length=hop_length,
                feature_names=feature_names,
                mean_pooling=mean_pooling,
            )
            torch.save(features.detach().cpu(), save_path)

    print("✅ Structure annotations saved.")
