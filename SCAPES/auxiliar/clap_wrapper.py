import librosa
import torch
import numpy as np
import torch.nn.functional as F

from msclap.CLAPWrapper import CLAPWrapper

# Clap wrapper with custom audio processing for embedding computation of short audio signals
class CLAPWrapper(CLAPWrapper):
    def __init__(self, version="2023", use_cuda=False):
        super().__init__(version=version, use_cuda=use_cuda)

        self.name = "CLAPWrapper"

    def compute_embedding(self, audio, og_sr=48000, random_extension=True):
        # Check that dimensions are B x C x T
        if audio.dim() != 3:
            raise ValueError(f"Expected audio tensor to have 3 dimensions (Batch, Channels, Time), but got {audio.shape}")
        clap_sr = self.args.sampling_rate
        # Resampling + mono if needed
        audio = resample_and_mono_audio(audio, og_sr, clap_sr, mono=True)
        # Extend/trim to 8 seconds
        audio = audio_extender(audio, 
                            random_extension=random_extension,
                            sample_rate=clap_sr, 
                            duration=8.0, 
                            overlap_ratio=0.1) # [B, 1, 8*48000]
        embedding = self._get_audio_embeddings(audio) # [B, 1024]

        # Normalize embeddings to unit norm
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

def resample_and_mono_audio(audio, og_sr, target_sr, mono=True):
    """
    Resample audio from og_sr → target_sr.

    Args:
        audio: Tensor, shape (C, T) or (B, C, T)
        og_sr: original sample rate
        target_sr: target sample rate
        batch_added: bool, if True, squeeze batch dim at the end

    Returns:
        resampled audio tensor
        shape: (C, T_new) or (B, C, T_new) depending on batch_added
    """

    # --- Ensure batch dimension ---
    if audio.dim() == 2:
        audio = audio.unsqueeze(0)  # (1, C, T)
        added_batch = True
    else:
        added_batch = False

    # --- Convert to mono if needed ---
    if mono and audio.shape[1] > 1:
        audio = audio.mean(dim=1, keepdim=True)  # (B, 1, T)

    B, C, T = audio.shape
    T_new = int(T * target_sr / og_sr)

    # Interpolate along time dimension
    # F.interpolate expects (B, C, L) with float, uses mode='linear'
    audio = audio.float()
    resampled = F.interpolate(audio, size=T_new, mode='linear', align_corners=False)

    # Remove batch if needed
    if added_batch:
        resampled = resampled.squeeze(0)

    return resampled

def audio_extender(input_audio, random_extension=False, 
                   sample_rate=48000, duration=7.0, overlap_ratio=0.1):
    """
    Extends or trims audio to a fixed duration with optional random segmenting.

    Args:
        input_audio: Tensor (C, T) or (B, C, T)
        random_extension: bool
        sample_rate: int
        duration: float (seconds)
        overlap_ratio: float (default 0.1 → 100ms at 48kHz)

    Returns:
        Tensor of shape (B, 1, expected_length)
    """

    # --- Ensure batch dimension ---
    if input_audio.dim() == 2:
        input_audio = input_audio.unsqueeze(0)  # (1, C, T)
        batch_added = True
    else:        
        batch_added = False

    B, C, T = input_audio.shape

    # --- Convert to mono ---
    audio = input_audio.mean(dim=1, keepdim=True)  # (B, 1, T)

    expected_length = int(sample_rate * duration)
    overlap = int(overlap_ratio * sample_rate)

    # --- Adjust overlap if audio is shorter than overlap window ---
    overlap = min(overlap, T // 2)  # ensure safe crossfade

    # --- If longer → trim ---
    if T >= expected_length:
        return audio[:, :, :expected_length]

    # --- Create output tensor ---
    output = torch.zeros((B, 1, expected_length), device=audio.device)

    # --- Crossfade windows ---
    fade_out = torch.linspace(1, 0, overlap, device=audio.device)
    fade_in  = torch.linspace(0, 1, overlap, device=audio.device)

    pos = 0
    first_segment = True

    while pos < expected_length:
        remaining = expected_length - pos

        # === NON RANDOM MODE ===
        if not random_extension:
            segment = audio
        # === RANDOM MODE ===
        else:
            min_length = T // 2
            max_length = T
            segment_length = np.random.randint(min_length, max_length + 1)

            if T > segment_length:
                start_pos = np.random.randint(0, T - segment_length + 1)
            else:
                start_pos = 0
                segment_length = T

            segment = audio[:, :, start_pos:start_pos + segment_length]

        seg_len = segment.shape[-1]

        # --- First segment: no fade-in ---
        if first_segment:
            use_length = min(seg_len, remaining)
            output[:, :, pos:pos + use_length] = segment[:, :, :use_length]
            pos += use_length
            first_segment = False
            continue

        # --- Crossfade segment ---
        if remaining <= overlap:
            # Last tiny bit
            output[:, :, pos:pos + remaining] = (
                output[:, :, pos:pos + remaining] * fade_out[:remaining] +
                segment[:, :, :remaining] * fade_in[:remaining]
            )
            break

        # Overlap region
        output[:, :, pos:pos + overlap] = (
            output[:, :, pos:pos + overlap] * fade_out +
            segment[:, :, :overlap] * fade_in
        )

        # Non-overlap region
        non_overlap = min(seg_len - overlap, remaining - overlap)
        if non_overlap > 0:
            output[:, :, pos + overlap:pos + overlap + non_overlap] = \
                segment[:, :, overlap:overlap + non_overlap]

        pos += non_overlap

    # --- Normalize ---
    max_val = torch.max(torch.abs(output))
    if max_val > 0:
        output = output / max_val

    if batch_added:
        output = output.squeeze(0)  # (1, 1, T) → (1, T)

    return output
