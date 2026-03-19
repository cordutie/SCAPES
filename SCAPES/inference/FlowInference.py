import soundfile as sf
import librosa
import torch
import torch.nn as nn
import math
from typing import List, Dict, Any
from tqdm import tqdm

class FlowInference:
    def __init__(
        self,
        model: nn.Module,              # The trained FlowModel
        local_encoder: nn.Module,      # The trained LocalEncoder
        processor,                     # The EncodecProcessor
        context_model: nn.Module,      # CLAP or your lightweight Proxy
        segment_length: int = 5,       # M: Number of past atoms for memory
        context_length: int = 10,      # Number of atoms in the context window
        # context_shift: int = 5,        # How many atoms ahead the context starts
        atoms_frames: int = 21,        # Frames per atom
        atoms_overlap_frames: int = 3, # Overlapping frames between consecutive atoms
        sr: int = 48000,               # Audio sample rate
        frame_rate: int = 150,         # EnCodec frame rate
        device: str = "cuda",
        verbose = False
    ):
        """
        Initializes the Programmable Inference Engine for Flow Matching Audio.
        """
        self.device = device
        self.verbose = verbose

        # 1. Load and lock models in eval mode
        self.model = model.to(self.device).eval()
        self.local_encoder = local_encoder.to(self.device).eval()
        self.context_model = context_model.to(self.device).eval()
        self.processor = processor # Assumes processor handles its own device internally
        
        # 2. Structural Hyperparameters
        self.segment_length = segment_length
        self.context_length = context_length
        self.context_shift = segment_length
        self.atoms_frames = atoms_frames
        self.atoms_overlap_frames = atoms_overlap_frames
        
        # 3. Audio Math (Matching AtomSequenceDataset)
        self.sr = sr
        self.frame_rate = frame_rate
        self.samples_per_frame = sr // frame_rate
        
        self.segment_samples = self.atoms_frames * self.samples_per_frame
        self.overlap_samples = self.atoms_overlap_frames * self.samples_per_frame
        self.hop_samples = self.segment_samples - self.overlap_samples
        
        # 4. Pre-compute the Overlap-Add (OLA) Hann Window for the final audio rendering
        self.ola_window = self._build_ola_window().to(self.device)
        
        # 5. The Director's Script
        self.timeline: List[Dict[str, Any]] = []

    def _build_ola_window(self):
        """Builds the exact Hann window used during training decoding."""
        window = torch.ones(self.segment_samples - 2 * self.overlap_samples)
        hann_window = torch.hann_window(self.overlap_samples * 2)
        left_hann = hann_window[:self.overlap_samples]
        right_hann = hann_window[self.overlap_samples:]
        return torch.cat([left_hann, window, right_hann])
    
    def load_audio_to_tensor(self, audio_path: str) -> torch.Tensor:
        """
        Loads an audio file from disk, ensures it is stereo [1, 2, T], 
        and moves it to the correct device.
        """
        # Load audio using librosa
        audio_input, _ = librosa.load(audio_path, sr=self.sr, mono=False)
        audio_tensor = torch.tensor(audio_input).unsqueeze(0) # [1, Channels, T]
        
        # If mono, duplicate to stereo
        if audio_tensor.dim() == 2 or audio_tensor.shape[1] == 1:
            # If it was loaded as [1, T], make it [1, 1, T] first
            if audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.unsqueeze(1)
            audio_tensor = audio_tensor.repeat(1, 2, 1)
            
        # If more than 2 channels, truncate
        elif audio_tensor.shape[1] > 2:
            audio_tensor = audio_tensor[:, :2, :]
            
        return audio_tensor.to(self.device).float()

    @torch.no_grad()
    def encode_audio_to_atoms(self, audio_tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Takes a raw stereo waveform tensor [1, 2, T], chops it into overlapping segments, 
        and extracts the unified 129-D latents [1, 129, 21].
        """
        # Ensure tensor is on the right device and shape
        audio_tensor = audio_tensor.to(self.device)
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0) # [1, 2, T]
            
        total_samples = audio_tensor.shape[-1]
        atoms_129D = []
        
        # Sliding window over the audio
        for start in range(0, total_samples, self.hop_samples):
            end = start + self.segment_samples
            segment = audio_tensor[:, :, start:end]
            
            # Skip the last segment if it doesn't have the full 21 frames worth of samples
            if segment.shape[-1] < self.segment_samples:
                break
                
            # Extract EnCodec Latent and Scale
            latent_list, metadata = self.processor.audio_to_latents(segment, self.sr)
            
            latent = torch.cat(latent_list, dim=-1) # Shape: [1, 128, 21]
            scale = metadata["audio_scales"][0]     # Shape: [1, 1]
            
            # The 129th Dimension Trick: Expand scale and concatenate
            scale_expanded = scale.unsqueeze(-1).expand(-1, -1, self.atoms_frames) # [1, 1, 21]
            atom_combined = torch.cat([latent, scale_expanded], dim=1)             # [1, 129, 21]
            
            atoms_129D.append(atom_combined)
            
        return atoms_129D
    
    @torch.no_grad()
    def compute_context_track(self, atoms_129D: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Computes 1024-D context embeddings using ONLY valid, contiguous atoms.
        If the window exceeds the file length, it holds the last valid embedding.
        """
        context_embeddings = []
        total_atoms = len(atoms_129D)
        
        last_valid_emb = None
        
        for t in range(total_atoms):
            # Assuming context_shift is relative to the target atom t
            start_idx = t + self.context_shift 
            end_idx = start_idx + self.context_length
            
            # Check if we have enough real atoms to form a valid window
            if end_idx <= total_atoms:
                # 1. We are safe. Grab the real sequence.
                window_atoms = atoms_129D[start_idx:end_idx]
                
                # Stack into batch sequence: [1, N, 129, 21]
                chunk_129D = torch.cat(window_atoms, dim=0).unsqueeze(0) 
                
                # Split for GlobalEncoder
                latent = chunk_129D[:, :, :128, :] # [1, N, 128, 21]
                scale = chunk_129D[:, :, 128, 0:1] # [1, N, 1]
                
                # Compute embedding
                emb = self.context_model(latent, scale).squeeze(0) # [1024]
                context_embeddings.append(emb)
                
                # Update our fallback
                last_valid_emb = emb 
                
            else:
                # 2. We hit the edge of the file! 
                # The GlobalEncoder would need padding here, which is bad.
                # Instead, we just reuse the last valid acoustic goal.
                if last_valid_emb is not None:
                    context_embeddings.append(last_valid_emb)
                else:
                    # Extreme edge case: the entire audio file is shorter than context_length
                    raise ValueError("Audio file is too short to compute even one context window!")
                    
        return context_embeddings
    
    def build_base_timeline(
        self, 
        atoms_129D: List[torch.Tensor], 
        context_embeddings: List[torch.Tensor], 
        default_TF: bool = False, 
        default_AF: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Zips the extracted atoms and computed contexts into the programmable timeline.
        """
        if len(atoms_129D) != len(context_embeddings):
            raise ValueError(
                f"Length mismatch! Atoms: {len(atoms_129D)}, Contexts: {len(context_embeddings)}. "
                "Ensure compute_context_track processed the same list."
            )
            
        timeline = []
        for t in range(len(atoms_129D)):
            step_dict = {
                "step": t,
                "atom_given": atoms_129D[t],           # [1, 129, 21]
                "context_embedding": context_embeddings[t], # [1024]
                "atom_generated": None,                # Will be filled by FlowModel
                "TF": default_TF,                      # True = Use given for memory
                "AF": default_AF                       # 1.0 = Render given audio, 0.0 = Render generated
            }
            timeline.append(step_dict)
            
        # Store it in the class state for the generation loop to use
        self.timeline = timeline 
        return timeline
    
    @torch.no_grad()
    def generate(self, timeline: List[Dict[str, Any]], NFE: int = 32) -> List[Dict[str, Any]]:
        """
        Runs the Flow Matching ODE solver over the provided programmable timeline.
        Dynamically constructs its sliding memory window based on the TF flag.
        """
        self.model.eval()
        self.local_encoder.eval()

        if not timeline:
            raise ValueError("Timeline is empty! Provide a valid timeline list.")

        M = self.segment_length
        total_steps = len(timeline)
        
        # Dummy atom for Cold Start padding
        dummy_atom = torch.zeros(1, 129, self.atoms_frames, device=self.device)

        if self.verbose:
            print(f"\n--- Starting Generation over {total_steps} steps (NFE={NFE}) ---")

        for t in tqdm(range(total_steps), desc="Solving ODE", disable=not self.verbose):
            # ---------------------------------------------------------
            # 1. Assemble the Memory Buffer (The t-M to t-1 lookback)
            # ---------------------------------------------------------
            past_atoms = []
            
            for i in range(t - M, t):
                if i < 0:
                    past_atoms.append(dummy_atom)
                else:
                    step_dict = timeline[i]
                    if step_dict["TF"]:
                        past_atoms.append(step_dict["atom_given"].to(self.device))
                    else:
                        past_atoms.append(step_dict["atom_generated"].to(self.device))
                        
            past_buffer = torch.cat(past_atoms, dim=0).unsqueeze(0) # [1, M, 129, 21]

            # ---------------------------------------------------------
            # 2. Encode and Inject Cold-Start <NULL> Tokens
            # ---------------------------------------------------------
            encoded_past = self.local_encoder(past_buffer) 
            
            num_nulls = max(0, M - t)
            if num_nulls > 0:
                encoded_past[:, :num_nulls, :, :] = self.model.null_past_embed

            # ---------------------------------------------------------
            # 3. Fetch Context and Generate
            # ---------------------------------------------------------
            context = timeline[t]["context_embedding"].to(self.device)
            if context.dim() == 1:
                context = context.unsqueeze(0) 
                
            x0 = torch.randn(1, self.atoms_frames, 129, device=self.device)
            pred = self.model.generate(x0, encoded_past, context, max_nfe=NFE) 
            
            # ---------------------------------------------------------
            # 4. Save Output back to Timeline
            # ---------------------------------------------------------
            timeline[t]["atom_generated"] = pred.transpose(1, 2)

        if self.verbose:
            print("✅ Generation Complete!")
        return timeline
    
    def _decode_single_atom(self, atom_129D: torch.Tensor) -> torch.Tensor:
        """Helper to safely decode a 129-D tensor back to 48kHz audio."""
        latent = atom_129D[:, :128, :] # [1, 128, 21]
        raw_scale = atom_129D[:, 128, :] # [1, 21]
        scale = torch.abs(raw_scale).mean(dim=-1, keepdim=True) # [1, 1]      

        metadata = {
            "audio_scales": [scale.squeeze(0).float()],
            "padding_mask": torch.ones(
                (1, latent.shape[-1] * self.samples_per_frame), 
                dtype=torch.bool, device=self.device
            )
        }
        
        # We rely strictly on the OLA window to stitch the audio. 
        # If the wrapper has a state reset, it would go here to prevent bleed,
        # but relying purely on the audio-domain OLA window is the safest real-time bet.
        audio = self.processor.decode_latents_audio(latent, metadata=metadata)
        return audio.cpu()

    @torch.no_grad()
    def decode_timeline(self, timeline: List[Dict[str, Any]], output_path: str = None):
        """
        REAL-TIME RENDERER:
        Decodes atom-by-atom and applies the OLA window immediately.
        """
        if not timeline:
            raise ValueError("Timeline is empty!")
        
        if self.verbose:
            print("\n--- Rendering Audio Timeline (Real-Time OLA) ---")
        
        total_steps = len(timeline)
        total_samples = (total_steps - 1) * self.hop_samples + self.segment_samples
        output_buffer = torch.zeros(1, 2, total_samples) # [1, Stereo, T]
        
        ola_window = self.ola_window.view(1, 1, -1).cpu()

        for t in tqdm(range(total_steps), desc="Mixing Audio", disable=not self.verbose):
            step_dict = timeline[t]
            AF = float(step_dict.get("AF", 0.0))
            AF = max(0.0, min(1.0, AF))
            
            audio_mix = torch.zeros(1, 2, self.segment_samples)
            
            # --- Selective Decoding to prevent unnecessary cache thrashing ---
            # If AF is 1.0, we ONLY decode the given audio.
            # If AF is 0.0, we ONLY decode the generated audio.
            
            if AF > 0.0:
                audio_given = self._decode_single_atom(step_dict["atom_given"])
                audio_mix += math.sqrt(AF) * audio_given
                
            if AF < 1.0:
                audio_generated = self._decode_single_atom(step_dict["atom_generated"])
                audio_mix += math.sqrt(1.0 - AF) * audio_generated
                
            # Apply the OLA Hann Window
            audio_mix = audio_mix * ola_window
            
            # Add to the global timeline buffer
            start_sample = t * self.hop_samples
            end_sample = start_sample + self.segment_samples
            output_buffer[:, :, start_sample:end_sample] += audio_mix
            
        final_audio_tensor = output_buffer.squeeze(0)
        
        if output_path:
            sf_audio = final_audio_tensor.transpose(0, 1).numpy()
            sf.write(output_path, sf_audio, self.sr)
            print(f"✅ Audio perfectly rendered and saved to: {output_path}")
            
        return final_audio_tensor