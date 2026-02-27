"""
EnCodec Audio Processor Class

A simple wrapper class for EnCodec models (24kHz and 48kHz) that provides easy access to:
- Continuous embeddings (pre-quantization)
- Quantized embeddings 
- Audio reconstruction from both types

This class handles the conversion between audio tensors and EnCodec representations,
supporting both the continuous path (used by AudioBox) and quantized path (standard EnCodec).
"""

import torch
import torchaudio
# Use the standalone `encodec` package for the original torchaudio-style EncodecModel
try:
    from encodec import EncodecModel as TorchEncodecModel
except Exception:
    TorchEncodecModel = None

from transformers import EncodecModel as HFEncodecModel, AutoProcessor
import numpy as np
from typing import Optional, Tuple, Union


class EncodecProcessor:
    """
    A simple processor for EnCodec models with support for both continuous and quantized embeddings.
    
    This class provides an easy-to-use interface for:
    - Converting audio to continuous embeddings (AudioBox-style)
    - Converting audio to quantized embeddings (standard EnCodec)
    - Reconstructing audio from both embedding types
    - Support for both 24kHz and 48kHz models
    """
    
    def __init__(self, sr: int = 24000, device: Optional[str] = None, streamable: bool = False):
        """
        Initialize the EnCodec processor.
        
        Args:
            sr: Sample rate (24000 for 24kHz model, 48000 for 48kHz model)
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detect)
            streamable: If True, disable chunking for streamable processing.
                       For 24kHz: already streamable by default
                       For 48kHz: disables 1-second chunks, processes entire audio at once
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_sr = sr
        self.streamable = streamable
        
        print(f"Initializing EnCodec processor on device: {self.device}")
        print(f"Streamable mode: {streamable}")
        
        if sr == 24000:
            # Load EnCodec 24kHz model (local `encodec` package)
            if TorchEncodecModel is None:
                raise ImportError("The 'encodec' package is required for 24kHz EnCodec support. Install it with `pip install encodec`.")
            self.model = TorchEncodecModel.encodec_model_24khz()
            self.model.eval()
            self.model = self.model.to(self.device)
            self.sample_rate = self.model.sample_rate  # 24000 Hz
            self.frame_rate = self.model.frame_rate    # ~75 Hz
            self.model_type = "24kHz"
            self.is_hf_model = False
            print(f"✓ EnCodec 24kHz model loaded (torchaudio)")
            
        elif sr == 48000:
            # Load EnCodec 48kHz model (HuggingFace Transformers version)
            self.model = HFEncodecModel.from_pretrained("facebook/encodec_48khz")
            self.processor = AutoProcessor.from_pretrained("facebook/encodec_48khz")
            
            # Enable streamable mode if requested
            if streamable:
                print("  → Enabling streamable mode (disabling chunking)")
                self.model.config.chunk_length_s = None
                self.model.config.overlap = None
                # Also update the processor to not add chunking padding
                self.processor.chunk_length_s = None
                self.processor.overlap = None
            
            self.model.eval()
            self.model = self.model.to(self.device)
            self.sample_rate = 48000
            self.frame_rate = 150  # Approximate frame rate for 48kHz model
            self.model_type = "48kHz"
            self.is_hf_model = True
            mode = "streamable" if streamable else "chunked (1s chunks, 1% overlap)"
            print(f"✓ EnCodec 48kHz model loaded (HuggingFace) - {mode}")
            
        else:
            raise ValueError(f"Unsupported sample rate: {sr}. Supported rates: 24000, 48000")
        
        print(f"✓ Sample rate: {self.sample_rate} Hz")
        print(f"✓ Frame rate: {self.frame_rate} Hz")
    
    def _prepare_audio_tensor(self, audio: Union[torch.Tensor, np.ndarray], sample_rate: Optional[int] = None) -> torch.Tensor:
        """
        Prepare audio tensor for EnCodec processing.
        
        Args:
            audio: Audio data as tensor or numpy array
            sample_rate: Sample rate of input audio (will resample if needed)
            
        Returns:
            Audio tensor of shape [batch, channels, time] ready for EnCodec
        """
        # Convert to tensor if needed
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)
        
        # Ensure correct shape based on model type
        if self.is_hf_model:
            # HuggingFace 48kHz model expects stereo: [channels, time] -> [2, time]
            if audio.dim() == 1:
                audio = torch.stack([audio, audio])  # [time] -> [2, time] (stereo)
            elif audio.dim() == 2:
                if audio.shape[0] == 1:
                    # [1, time] -> [2, time] (duplicate to stereo)
                    audio = torch.cat([audio, audio], dim=0)
                elif audio.shape[1] == 1:
                    # [time, 1] -> [2, time]
                    audio = audio.squeeze(1)
                    audio = torch.stack([audio, audio])
                elif audio.shape[0] > audio.shape[1]:
                    # Likely [time, channels] -> transpose to [channels, time]
                    audio = audio.T
                    if audio.shape[0] == 1:
                        audio = torch.cat([audio, audio], dim=0)
            elif audio.dim() == 3:
                # [batch, channels, time] -> [channels, time] (take first batch)
                audio = audio[0]
                if audio.shape[0] == 1:
                    audio = torch.cat([audio, audio], dim=0)
        else:
            # Torchaudio 24kHz model expects: [batch, channels, time]
            if audio.dim() == 1:
                audio = audio.unsqueeze(0).unsqueeze(0)  # [time] -> [1, 1, time]
            elif audio.dim() == 2:
                if audio.shape[0] > audio.shape[1]:
                    # Likely [time, channels] -> transpose to [channels, time]
                    audio = audio.T.unsqueeze(0)  # -> [1, channels, time]
                else:
                    # Likely [channels, time] -> add batch dimension
                    audio = audio.unsqueeze(0)  # -> [1, channels, time]
        
        # Move to device
        audio = audio.to(self.device)
        
        # Resample if needed
        if sample_rate is not None and sample_rate != self.sample_rate:
            print(f"Resampling from {sample_rate}Hz to {self.sample_rate}Hz")
            if self.is_hf_model:
                # For HF model, resample each channel separately
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate).to(self.device)
                audio = resampler(audio)
            else:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate).to(self.device)
                audio = resampler(audio)
        
        return audio
    
    def encode_audio_emb(self, audio: Union[torch.Tensor, np.ndarray], sample_rate: Optional[int] = None) -> torch.Tensor:
        """
        Encode audio to continuous embeddings (pre-quantization).
        This is what AudioBox uses for training and generation.
        
        Args:
            audio: Audio tensor of shape [time] or [channels, time] or [batch, channels, time]
            sample_rate: Sample rate of input audio (will resample if needed)
            
        Returns:
            Continuous embeddings tensor of shape [batch, channels, time_frames]
        """
        # New: implement as composition via audio_to_latents and return latents only
        latents, metadata = self.audio_to_latents(audio, sample_rate)
        # audio_to_latents returns list of chunk embeddings for HF and a single-item list for 24k
        # Concatenate chunk embeddings along time to produce same shape as original API
        if isinstance(latents, list):
            embeddings = torch.cat([l for l in latents], dim=-1)
        else:
            embeddings = latents
        return embeddings

    def audio_to_latents(self, audio: Union[torch.Tensor, np.ndarray], sample_rate: Optional[int] = None):
        """
        Convert audio to continuous (pre-quantization) latent embeddings.

        Returns:
            (latents_list, metadata) where latents_list is a list of tensors (one per chunk)
            and metadata contains any auxiliary information (e.g. padding_mask, audio_scales placeholder).
        """
        with torch.no_grad():
            audio_tensor = self._prepare_audio_tensor(audio, sample_rate)

            if self.is_hf_model:
                # For HF model we must mimic EncodecModel.encode chunking logic so that later
                # quantization/decoding can operate on the same chunk boundaries.
                audio_cpu = audio_tensor.cpu().numpy()
                inputs = self.processor(raw_audio=audio_cpu, sampling_rate=self.sample_rate, return_tensors="pt")
                input_values = inputs["input_values"].to(self.device)
                padding_mask = inputs.get("padding_mask", None)

                cfg = self.model.config
                chunk_length = cfg.chunk_length if cfg.chunk_length is not None else input_values.shape[-1]
                stride = cfg.chunk_stride if cfg.chunk_length is not None else input_values.shape[-1]

                embeddings_chunks = []
                audio_scales = []
                for offset in range(0, input_values.shape[-1], stride):
                    frame = input_values[..., offset: offset + chunk_length]
                    # compute scale and normalize per-frame if needed
                    if cfg.normalize:
                        mono = torch.sum(frame, 1, keepdim=True) / frame.shape[1]
                        scale = mono.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
                        norm_frame = frame / scale
                        audio_scales.append(scale.view(-1, 1))
                    else:
                        norm_frame = frame
                        audio_scales.append(None)

                    emb = self.model.encoder(norm_frame)
                    embeddings_chunks.append(emb)

                metadata = {
                    'padding_mask': padding_mask,
                    'audio_scales': audio_scales,
                }
                return embeddings_chunks, metadata

            else:
                # Torchaudio 24kHz model: encoder expects [batch, channels, time] and no chunking
                embeddings = self.model.encoder(audio_tensor)
                return [embeddings], None
    
    def encode_audio_codes(self, audio: Union[torch.Tensor, np.ndarray], 
                     kbps: float = 6.0, sample_rate: Optional[int] = None) -> torch.Tensor:
        """
        Encode audio to quantized codes (standard EnCodec with discrete codes).
        
        Args:
            audio: Audio tensor of shape [time] or [channels, time] or [batch, channels, time]
            kbps: Target bandwidth in kbps. Supported: 1.5, 3.0, 6.0, 12.0, 24.0
            sample_rate: Sample rate of input audio (will resample if needed)
            
        Returns:
            Discrete codes tensor of shape [batch, num_quantizers, time_frames]
        """
        # New implementation: compose audio_to_latents -> latents_to_codes
        latents_list, latents_meta = self.audio_to_latents(audio, sample_rate)
        codes, metadata = self.latents_to_codes(latents_list, kbps, latents_meta)
        return codes, metadata

    def latents_to_codes(self, latents_list, kbps: float = 6.0, latents_meta: Optional[dict] = None):
        """
        Convert continuous latent embeddings (list of chunk tensors) to discrete codes.

        Returns:
            codes tensor shaped [nb_frames, batch, nb_quantizers, frame_len], metadata dict for decoding
        """
        with torch.no_grad():
            if self.is_hf_model:
                # HF quantizer.encode(embeddings, bandwidth) returns [num_quantizers, batch, time_frames]
                codes_chunks = []
                audio_scales = []
                for i, emb in enumerate(latents_list):
                    codes_chunk = self.model.quantizer.encode(emb, kbps)
                    # transpose to [batch, num_quantizers, time_frames]
                    codes_chunk = codes_chunk.transpose(0, 1)
                    codes_chunks.append(codes_chunk)
                    # audio_scales should be provided by latents_meta if available
                    if latents_meta is not None:
                        audio_scales.append(latents_meta.get('audio_scales', [None] * len(latents_list))[i])
                    else:
                        audio_scales.append(None)

                # pad last chunk to match first chunk length if necessary (mimic model.encode behaviour)
                if len(codes_chunks) > 1:
                    first_len = codes_chunks[0].shape[-1]
                    last_len = codes_chunks[-1].shape[-1]
                    if last_len < first_len:
                        pad = first_len - last_len
                        codes_chunks[-1] = torch.nn.functional.pad(codes_chunks[-1], (0, pad), value=0)

                # Stack to [nb_frames, batch, num_quantizers, frame_len]
                codes = torch.stack(codes_chunks, dim=0)
                metadata = {
                    'audio_scales': audio_scales,
                    'padding_mask': latents_meta.get('padding_mask') if latents_meta else None,
                    'last_frame_pad_length': (first_len - last_len) if len(codes_chunks) > 1 else 0,
                }
                return codes, metadata

            else:
                # Torchaudio model expects full embeddings tensor
                embeddings = latents_list[0]
                # torchaudio quantizer.encode signature: encode(embeddings, frame_rate, bandwidth)
                codes = self.model.quantizer.encode(embeddings, self.frame_rate, kbps)
                return codes, None
        
    def decode_codes_emb(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete audio codes back to quantized embeddings.
        
        Args:
            codes: Discrete codes tensor 
            
        Returns:
            Quantized embeddings tensor of shape [batch, channels, time_frames]
        """
        # Implement as composition: codes -> latents
        return self.codes_to_latents(codes)

    def codes_to_latents(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete codes to quantized embeddings (latents).
        Accepts codes shaped [nb_frames, batch, nb_quantizers, frame_len] or other shapes produced by model APIs.
        Returns a single concatenated embeddings tensor [batch, channels, total_time_frames].
        """
        with torch.no_grad():
            if self.is_hf_model:
                # Expect [chunks, batch, num_quantizers, time_frames]
                if codes.dim() == 4:
                    chunks, batch_size, num_quantizers, time_frames = codes.shape
                    chunk_embeddings = []
                    for i in range(chunks):
                        chunk_codes = codes[i]  # [batch, num_quantizers, time_frames]
                        chunk_codes = chunk_codes.transpose(0, 1)  # -> [num_quantizers, batch, time_frames]
                        chunk_emb = self.model.quantizer.decode(chunk_codes)  # [batch, channels, time_frames]
                        chunk_embeddings.append(chunk_emb)
                    quantized_embeddings = torch.cat(chunk_embeddings, dim=2)
                else:
                    raise ValueError(f"Unexpected codes dimensions: {codes.shape}. Should be [chunks, batch, num_quantizers, time_frames]")
            else:
                # Torchaudio: codes expected as [num_quantizers, batch, time_frames]
                quantized_embeddings = self.model.quantizer.decode(codes)
            return quantized_embeddings

    def decode_codes_audio(self, codes: torch.Tensor, metadata: Optional[dict] = None) -> torch.Tensor:
        """
        Decode discrete audio codes directly to audio using the complete quantized pipeline.
        This is the proper way to reconstruct audio from quantized codes.
        
        Args:
            codes: Discrete codes tensor 
            metadata: For HF model, dict with 'audio_scales' and 'padding_mask'
            
        Returns:
            Reconstructed audio tensor
        """
        # Implement as composition: codes -> latents -> audio
        latents = self.codes_to_latents(codes)
        audio = self.decode_latents_audio(latents, metadata)
        return audio

    def decode_latents_audio(self, embeddings: torch.Tensor, metadata: Optional[dict] = None) -> torch.Tensor:
        """
        Decode embeddings back to audio.
        Works for both continuous and quantized embeddings.
        
        Args:
            embeddings: Tensor of shape [batch, channels, time_frames]
            metadata: Optional dict with 'audio_scales' and 'padding_mask'.
                     If audio_scales is a list, we'll apply chunked decoding with overlap-add.
            
        Returns:
            Reconstructed audio tensor of shape [batch, channels, time]
        """
        with torch.no_grad():
            if self.is_hf_model:
                # HuggingFace 48kHz model
                if metadata is not None and 'audio_scales' in metadata:
                    audio_scales = metadata.get('audio_scales', None)
                    padding_mask = metadata.get('padding_mask', None)
                    
                    # Check if we need chunked decoding (multiple scales means multiple chunks)
                    if isinstance(audio_scales, list) and len(audio_scales) > 1 and all(s is not None for s in audio_scales):
                        # CHUNKED DECODING PATH (for quantized embeddings from codes)
                        # We need to know the expected chunk size from the model config
                        cfg = self.model.config
                        chunk_length = cfg.chunk_length
                        stride = cfg.chunk_stride
                        
                        # Calculate expected embedding chunk size
                        # The encoder downsamples by hop_length (typically 320 for 48kHz)
                        hop_length = np.prod([r for r in cfg.upsampling_ratios])  # typically 320
                        expected_emb_chunk_size = chunk_length // hop_length  # e.g., 30720 / 320 = 96
                        
                        total_frames = embeddings.shape[-1]
                        num_chunks = len(audio_scales)
                        
                        # Split embeddings into chunks based on expected chunk size
                        chunk_audio = []
                        for i in range(num_chunks):
                            start_idx = i * expected_emb_chunk_size
                            end_idx = min(start_idx + expected_emb_chunk_size, total_frames)
                            chunk_emb = embeddings[..., start_idx:end_idx]
                            scale = audio_scales[i]
                            
                            # Decode chunk to audio first, then apply scale
                            audio = self.model.decoder(chunk_emb)
                            if scale is not None:
                                audio = audio * scale.view(-1, 1, 1)
                            chunk_audio.append(audio)
                        
                        # Use overlap-add to reconstruct full audio
                        reconstructed_audio = self.model._linear_overlap_add(chunk_audio, stride)
                        
                        # Remove padding if mask is present
                        if padding_mask is not None and padding_mask.shape[-1] < reconstructed_audio.shape[-1]:
                            reconstructed_audio = reconstructed_audio[..., :padding_mask.shape[-1]]
                    else:
                        # SINGLE-CHUNK or CONTINUOUS EMBEDDINGS PATH
                        # Just decode directly without chunking/overlap-add
                        reconstructed_audio = self.model.decoder(embeddings)
                        
                        # Apply single scale if available
                        if isinstance(audio_scales, list) and len(audio_scales) > 0 and audio_scales[0] is not None:
                            scale = audio_scales[0]
                            reconstructed_audio = reconstructed_audio * scale.view(-1, 1, 1)
                        elif audio_scales is not None and not isinstance(audio_scales, list):
                            reconstructed_audio = reconstructed_audio * audio_scales.view(-1, 1, 1)
                        
                        # Trim to padding mask if present
                        if padding_mask is not None and padding_mask.shape[-1] < reconstructed_audio.shape[-1]:
                            reconstructed_audio = reconstructed_audio[..., :padding_mask.shape[-1]]
                else:
                    # No metadata - just decode directly
                    reconstructed_audio = self.model.decoder(embeddings)
            else:
                # Torchaudio 24kHz model - simple decode
                reconstructed_audio = self.model.decoder(embeddings)
            
            return reconstructed_audio
    
    def get_compression_ratio(self, audio_length: int) -> float:
        """
        Calculate the compression ratio for a given audio length.
        
        Args:
            audio_length: Length of audio in samples
            
        Returns:
            Compression ratio (audio_samples / embedding_frames)
        """
        embedding_frames = audio_length // (self.sample_rate // self.frame_rate)
        return audio_length / embedding_frames if embedding_frames > 0 else 0
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': f'EnCodec {self.model_type}',
            'sample_rate': self.sample_rate,
            'frame_rate': self.frame_rate,
            'device': self.device,
            'is_hf_model': self.is_hf_model,
            'encoder_type': type(self.model.encoder).__name__,
            'decoder_type': type(self.model.decoder).__name__,
            'quantizer_type': type(self.model.quantizer).__name__
        }


if __name__ == "__main__":
    # Simple test if run directly
    print("Testing EncodecProcessor...")
    
    # Test both 24kHz and 48kHz models
    for sr in [24000, 48000]:
        print(f"\n{'='*50}")
        print(f"Testing {sr}Hz model")
        print(f"{'='*50}")
        
        processor = EncodecProcessor(sr=sr)
        
        # Create test signal
        duration = 2.0
        t = torch.linspace(0, duration, int(processor.sample_rate * duration))
        test_audio = 0.5 * torch.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
        
        print(f"\nTest audio: {duration}s sine wave at 440Hz")
        print(f"Audio shape: {test_audio.shape}")
        
        # Test continuous encoding/decoding
        print("\n=== Testing Continuous Path ===")
        cont_embeddings = processor.encode_audio_emb(test_audio)
        cont_reconstructed = processor.decode_latents_audio(cont_embeddings)
        
        print(f"Continuous embeddings shape: {cont_embeddings.shape}")
        print(f"Continuous reconstructed shape: {cont_reconstructed.shape}")
        
        # Test quantized encoding/decoding
        print("\n=== Testing Quantized Path ===")
        codes_result = processor.encode_audio_codes(test_audio, kbps=6.0)
        if processor.is_hf_model:
            codes, metadata = codes_result
            # Test the full pipeline: codes -> embeddings -> audio
            quant_embeddings = processor.decode_codes_emb(codes, metadata)
            quant_reconstructed_from_emb = processor.decode_latents_audio(quant_embeddings)
            # Also test direct codes -> audio
            quant_reconstructed = processor.decode_codes_audio(codes, metadata)
            print(f"Quantized embeddings shape: {quant_embeddings.shape}")
            print(f"Quantized reconstructed from embeddings shape: {quant_reconstructed_from_emb.shape}")
        else:
            codes, _ = codes_result
            quant_embeddings = processor.decode_codes_emb(codes)
            quant_reconstructed = processor.decode_latents_audio(quant_embeddings)
        
        print(f"Audio codes shape: {codes.shape}")
        print(f"Quantized reconstructed shape: {quant_reconstructed.shape}")
        
        print(f"\n=== Model Info ===")
        info = processor.get_model_info()
        for key, value in info.items():
            print(f"{key}: {value}")
    
    print("\n✓ All tests completed successfully!")
