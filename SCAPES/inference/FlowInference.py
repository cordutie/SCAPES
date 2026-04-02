import soundfile as sf
import librosa
import torch
import torch.nn as nn
import math
from typing import List, Dict, Any
from tqdm import tqdm
from IPython.display import Audio, display
from pathlib import Path
import matplotlib.pyplot as plt

# audio to atoms and context that can be used for conditioning
def load_and_encode(engine, audio_path, max_duration=None):
    audio_tensor = engine.load_audio_to_tensor(audio_path)
    if max_duration != None and audio_tensor.shape[-1] > engine.sr * max_duration:
        audio_tensor = audio_tensor[:,:,:48000*max_duration]
    print((f"--- Encoding audio: {audio_path}"))
    atoms = engine.encode_audio_to_atoms(audio_tensor)
    print((f"--- Computing context for audio: {audio_path}"))
    contexts = engine.compute_context_track(atoms)
    # print((len(atoms), "atoms extracted, ", len(contexts), "context embeddings computed."))
    return atoms, contexts

# Resynthesis task
def run_resynthesis_pipeline(
    engine,
    audio_path,
    duration=60,
    play=True,
    save_path=None,
    TF=False, # True = looks at the data atoms; False = purely autoregressive; "partial" = warm start with 5 steps of TF=True, then drops to False for the rest of the timeline.
    NFE = 32,
    context_static=False,  # If True, uses the first context embedding for the entire timeline
    decode_method="ola_smooth"
):
    atoms_src, contexts_src = load_and_encode(engine, audio_path, max_duration=duration)

    atoms    = atoms_src
    contexts = contexts_src

    if context_static==True:  
        c0 = contexts_src[0]
        contexts = [c0 for _ in range(len(atoms))]    

    cold_start = True
    if TF!=True and TF!=False:
        cold_start = False
        TF = False

    timeline = engine.build_base_timeline(
        atoms_129D=atoms,
        context_embeddings=contexts,
        default_TF=TF,
        default_AF=0.0
    )

    # If cold_start is False, we set the first 5 steps to TF=True to provide a warm launch for the generation.
    if cold_start==False:
        for t in range(0, 5):
            timeline[t]["TF"] = True

    completed_timeline = engine.generate(timeline, NFE=NFE)
    final_wav = engine.decode_timeline(completed_timeline, output_path=None, method=decode_method)

    if play:
        # pick filename without extension for display
        filename = Path(audio_path).stem
        print("Resynthesis: ", filename)
        display(Audio(final_wav, rate=engine.sr))

    if save_path != None:
        sf_audio = final_wav.transpose(0, 1).numpy()
        sf.write(save_path, sf_audio, engine.sr)
        print(f"✅ Resynthesized audio saved to: {save_path}")
    return final_wav

# Stickiness curve for interpolation alpha values
def sticky_curve_torch(n_points=100, stickiness=1.0):
    # stickiness must be positive, if not error:
    if stickiness <= 0:
        raise ValueError("Stickiness must be a positive value greater than 0.")
    
    stickiness = 1/stickiness

    alpha_linear = torch.linspace(0, 1, n_points)

    eps = 1e-8
    alpha_linear = alpha_linear.clamp(eps, 1 - eps)

    alpha_sticky = alpha_linear.pow(stickiness) / (
        alpha_linear.pow(stickiness) + (1 - alpha_linear).pow(stickiness)
    )

    return alpha_sticky

# Simplest low-pass filter
def low_pass_filter(signal, alpha=0.5):
    filtered = torch.zeros_like(signal)
    filtered[0] = signal[0]
    # first pass
    for t in range(1, signal.shape[0]):
        filtered[t] = alpha * signal[t] + (1 - alpha) * filtered[t-1]
    # 9 passes more
    for i in range(9):
        for t in range(1, signal.shape[0]):
            filtered[t] = alpha * filtered[t] + (1 - alpha) * filtered[t-1]
    return filtered

# Spherical interpolation for context embeddings
def slerp(v0, v1, alpha, eps=1e-7):
    """
    Spherical linear interpolation between two normalized vectors.
    v0, v1: (..., D)
    alpha: scalar in [0,1]
    """
    v0 = v0 / v0.norm(p=2)
    v1 = v1 / v1.norm(p=2)

    dot = torch.clamp(torch.dot(v0, v1), -1.0 + eps, 1.0 - eps)
    theta = torch.acos(dot)

    if theta < eps:
        # vectors are almost identical; fall back to lerp
        return (1 - alpha) * v0 + alpha * v1

    sin_theta = torch.sin(theta)
    w0 = torch.sin((1 - alpha) * theta) / sin_theta
    w1 = torch.sin(alpha * theta) / sin_theta

    return w0 * v0 + w1 * v1

# Run interpolation between two audios
def run_interpolation_pipeline(
    engine,
    audio_path_1,
    audio_path_2,
    timeline_size=200,
    stay_time=1,
    stickyness = 1.0,
    plot_stickyness_curve=False,
    play=True,
    save_path=None,
    NFE = 32,
    context_static=True,  # If True, uses the first context embedding for each audio only
    decode_method="ola_smooth"
):
    # Generate interpolation alpha values with stickiness and smooth them with a low-pass filter
    alpha_values      = sticky_curve_torch(n_points=timeline_size - 2 * stay_time, stickiness=stickyness)
    alpha_values_full = torch.cat([torch.zeros(stay_time), alpha_values, torch.ones(stay_time)])
    alpha_values_full = low_pass_filter(alpha_values_full, alpha=0.5)

    # Optionally plot the stickiness curve
    if plot_stickyness_curve:
        plt.figure(figsize=(10, 4))
        plt.plot(alpha_values_full.detach().cpu().numpy())
        plt.title(f'Interpolation Values with Stickiness={stickyness}')
        plt.grid()
        plt.show()

    # Load both audios
    atoms_1, contexts_1 = load_and_encode(engine, audio_path_1, max_duration=30)
    atoms_2, contexts_2 = load_and_encode(engine, audio_path_2, max_duration=30)

    # stay time must be an integer bigger than 0
    if stay_time < 0 or not isinstance(stay_time, int):
        raise ValueError("Stay time must be a non-negative integer.")

    # if context is not static ,each context should have at least timeline size number of embeddings, if not error:
    if context_static==False:
        if len(contexts_1) < timeline_size:
            raise ValueError(f"Audio 1 does not have enough context embeddings for the timeline size. Required: {timeline_size}, Available: {len(contexts_1)}")
        if len(contexts_2) < timeline_size:
            raise ValueError(f"Audio 2 does not have enough context embeddings for the timeline size. Required: {timeline_size}, Available: {len(contexts_2)}")
        contexts_1 = contexts_1[:timeline_size]
        contexts_2 = contexts_2[:timeline_size]

    c0 = contexts_1[0]
    c1 = contexts_2[0]

    atoms = [None] * timeline_size
    contexts = [] 

    # Send alpha_values_full to the same device as the context embeddings
    alpha_values_full = alpha_values_full.to(c0.device)

    for t in range(timeline_size):
        alpha = alpha_values_full[t]
        if context_static==True:
            ctx = slerp(c0, c1, alpha)
        else:
            ctx = slerp(contexts_1[t], contexts_2[t], alpha)
        contexts.append(ctx)

    # Build timeline
    timeline = engine.build_base_timeline(
        atoms_129D=atoms,
        context_embeddings=contexts,
        default_TF=False,
        default_AF=0.0
    )

    # Generate
    completed_timeline = engine.generate(timeline, NFE=32)

    # Decode
    final_wav = engine.decode_timeline(completed_timeline, output_path=None, method=decode_method)

    if play:
        print(f"Interpolation: {Path(audio_path_1).stem} -> {Path(audio_path_2).stem}")
        display(Audio(final_wav, rate=engine.sr))

    if save_path != None:
        sf_audio = final_wav.transpose(0, 1).numpy()
        sf.write(save_path, sf_audio, engine.sr)
        print(f"✅ Interpolated audio saved to: {save_path}")

    return final_wav

# FlowInference class encapsulates the entire inference pipeline for Flow Matching Audio, including loading, encoding, context computation, generation, and decoding.
class FlowInference:
    def __init__(
        self,
        model: nn.Module,              # The trained FlowModel
        local_encoder: nn.Module,      # The trained LocalEncoder
        processor,                     # The EncodecProcessor
        context_model: nn.Module,      # CLAP or your lightweight Proxy
        segment_length: int = 5,       # M: Number of past atoms for memory
        context_length: int = 10,      # Number of atoms in the context window
        atoms_frames: int = 39,        # <-- NEW: Total frames per atom
        atoms_hop_frames: int = 18,    # <-- NEW: How far we step forward
        crossfade_frames: int = 3,     # <-- NEW: Acoustic blending overlap
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

        self.context_model_type = context_model.name 

        # If GlobalEncoder is used, put in eval mode.
        if self.context_model_type == "GlobalEncoder":
            self.context_model = context_model.to(self.device).eval()
        else:
            self.context_model = context_model 
        
        self.processor = processor 
        
        # 2. Structural Hyperparameters
        self.segment_length = segment_length
        self.context_length = context_length
        self.context_shift = segment_length
        
        self.atoms_frames = atoms_frames
        self.atoms_hop_frames = atoms_hop_frames
        self.crossfade_frames = crossfade_frames
        
        # The amount of past context baked into the front of every atom
        self.macro_overlap_frames = self.atoms_frames - self.atoms_hop_frames
        
        # 3. Audio Math (Matching AtomSequenceDataset)
        self.sr = sr
        self.frame_rate = frame_rate
        self.samples_per_frame = sr // frame_rate
        
        self.segment_samples = self.atoms_frames * self.samples_per_frame
        self.hop_samples = self.atoms_hop_frames * self.samples_per_frame
        self.crossfade_samples = self.crossfade_frames * self.samples_per_frame
        self.macro_overlap_samples = self.macro_overlap_frames * self.samples_per_frame
        
        # 4. Pre-compute the Asymmetric Overlap-Add (OLA) Window
        self.ola_window = self._build_ola_window().to(self.device)
        
        # 5. The Director's Script
        self.timeline: List[Dict[str, Any]] = []

    def _build_ola_window(self):
        """Builds the asymmetric Prefix Padding window."""
        zeros_frames = self.macro_overlap_frames - self.crossfade_frames
        zeros = torch.zeros(zeros_frames * self.samples_per_frame)
        
        hann_window = torch.hann_window(self.crossfade_samples * 2)
        left_hann = hann_window[:self.crossfade_samples]
        right_hann = hann_window[self.crossfade_samples:]
        
        ones_frames = self.atoms_hop_frames - self.crossfade_frames
        ones = torch.ones(ones_frames * self.samples_per_frame)
        
        return torch.cat([zeros, left_hann, ones, right_hann])
    
    def load_audio_to_tensor(self, audio_path: str) -> torch.Tensor:
        """
        Loads an audio file from disk, ensures it is stereo [1, 2, T], 
        and moves it to the correct device.
        """
        audio_input, _ = librosa.load(audio_path, sr=self.sr, mono=False)
        audio_tensor = torch.tensor(audio_input).unsqueeze(0) 
        
        if audio_tensor.dim() == 2 or audio_tensor.shape[1] == 1:
            if audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.unsqueeze(1)
            audio_tensor = audio_tensor.repeat(1, 2, 1)
            
        elif audio_tensor.shape[1] > 2:
            audio_tensor = audio_tensor[:, :2, :]
            
        return audio_tensor.to(self.device).float()

    @torch.no_grad()
    def encode_audio_to_atoms(self, audio_tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Takes a raw stereo waveform tensor [1, 2, T], chops it into prefix-padded segments, 
        and extracts the unified 129-D latents.
        """
        audio_tensor = audio_tensor.to(self.device)
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0) 
            
        total_samples = audio_tensor.shape[-1]
        atoms_129D = []
        
        # Sliding window using the new asymmetric hop
        for start in range(0, total_samples, self.hop_samples):
            end = start + self.segment_samples
            segment = audio_tensor[:, :, start:end]
            
            # Skip the last segment if it doesn't have the full frames
            if segment.shape[-1] < self.segment_samples:
                break
                
            latent_list, metadata = self.processor.audio_to_latents(segment, self.sr)
            latent = torch.cat(latent_list, dim=-1) 
            scale = metadata["audio_scales"][0]     
            
            scale_expanded = scale.unsqueeze(-1).expand(-1, -1, self.atoms_frames) 
            atom_combined = torch.cat([latent, scale_expanded], dim=1)            
            
            atoms_129D.append(atom_combined)
            
        return atoms_129D
    
    @torch.no_grad()
    def compute_context_track(self, atoms_129D: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Computes 1024-D context embeddings using ONLY valid, contiguous atoms.
        """
        context_embeddings = []
        total_atoms = len(atoms_129D)
        last_valid_emb = None
        ola_window_local = self.ola_window.view(1, 1, -1)
        
        for t in range(total_atoms):
            start_idx = t + self.context_shift 
            end_idx = start_idx + self.context_length
            
            if end_idx <= total_atoms:
                window_atoms = atoms_129D[start_idx:end_idx]
                
                if self.context_model_type == "GlobalEncoder":
                    chunk_129D = torch.cat(window_atoms, dim=0).unsqueeze(0) 
                    latent = chunk_129D[:, :, :128, :] 
                    scale = chunk_129D[:, :, 128, 0:1] 
                    emb = self.context_model(latent, scale).squeeze(0) 
                    
                else:
                    N = len(window_atoms)
                    total_samples = (N - 1) * self.hop_samples + self.segment_samples
                    window_audio = torch.zeros(1, 2, total_samples, device=self.device)
                    
                    for i, atom in enumerate(window_atoms):
                        atom_audio = self._decode_single_atom(atom).to(self.device)
                        atom_audio = atom_audio * ola_window_local
                        
                        start_sample = i * self.hop_samples
                        end_sample = start_sample + self.segment_samples
                        window_audio[:, :, start_sample:end_sample] += atom_audio
                        
                    emb = self.context_model.compute_embedding(
                        window_audio, 
                        og_sr=self.sr, 
                        random_extension=True
                    ).squeeze(0) 

                context_embeddings.append(emb)
                last_valid_emb = emb 
                
            else:
                if last_valid_emb is not None:
                    context_embeddings.append(last_valid_emb)
                else:
                    raise ValueError("Audio file is too short to compute even one context window!")
                    
        return context_embeddings
    
    def build_base_timeline(self, atoms_129D, context_embeddings, default_TF=False, default_AF=0.0):
        if len(atoms_129D) != len(context_embeddings):
            raise ValueError("Length mismatch!")
            
        timeline = []
        for t in range(len(atoms_129D)):
            step_dict = {
                "step": t,
                "atom_given": atoms_129D[t],           
                "context_embedding": context_embeddings[t], 
                "atom_generated": None,                
                "TF": default_TF,                      
                "AF": default_AF                       
            }
            timeline.append(step_dict)
            
        self.timeline = timeline 
        return timeline
    
    @torch.no_grad()
    def generate(self, timeline: List[Dict[str, Any]], NFE: int = 32) -> List[Dict[str, Any]]:
        self.model.eval()
        self.local_encoder.eval()

        if not timeline:
            raise ValueError("Timeline is empty!")

        M = self.segment_length
        total_steps = len(timeline)
        dummy_atom = torch.zeros(1, 129, self.atoms_frames, device=self.device)

        if self.verbose:
            print(f"\n--- Starting Generation over {total_steps} steps (NFE={NFE}) ---")

        for t in tqdm(range(total_steps), desc="Solving ODE", disable=not self.verbose):
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
                        
            past_buffer = torch.cat(past_atoms, dim=0).unsqueeze(0) 

            encoded_past = self.local_encoder(past_buffer) 
            
            num_nulls = max(0, M - t)
            if num_nulls > 0:
                encoded_past[:, :num_nulls, :, :] = self.model.null_past_embed

            context = timeline[t]["context_embedding"].to(self.device)
            if context.dim() == 1:
                context = context.unsqueeze(0) 
                
            x0 = torch.randn(1, self.atoms_frames, 129, device=self.device)
            pred = self.model.generate(x0, encoded_past, context, max_nfe=NFE) 
            
            timeline[t]["atom_generated"] = pred.transpose(1, 2)

        if self.verbose:
            print("✅ Generation Complete!")
        return timeline
    
    def _decode_single_atom(self, atom_129D: torch.Tensor) -> torch.Tensor:
        latent = atom_129D[:, :128, :] 
        raw_scale = atom_129D[:, 128, :] 
        scale = torch.abs(raw_scale).mean(dim=-1, keepdim=True)       

        metadata = {
            "audio_scales": [scale.squeeze(0).float()],
            "padding_mask": torch.ones(
                (1, latent.shape[-1] * self.samples_per_frame), 
                dtype=torch.bool, device=self.device
            )
        }
        audio = self.processor.decode_latents_audio(latent, metadata=metadata)
        return audio.cpu()

    @torch.no_grad()
    def decode_timeline(self, timeline: List[Dict[str, Any]], output_path: str = None, method: str = "ola"):
        if not timeline:
            raise ValueError("Timeline is empty!")
            
        if method == "latent_stitch":
            if self.verbose:
                print("\n--- Rendering Audio Timeline (Prefix Latent Stitching Mode) ---")
                
            gen_chunks = []
            given_chunks = []
            dummy_atom = torch.zeros(1, 129, self.atoms_frames, device=self.device)
            
            # --- THE LATENT STITCH FIX ---
            # Overlap is now on the LEFT. So for t > 0, we discard the first 21 frames.
            for t, step_dict in enumerate(timeline):
                a_gen = step_dict.get("atom_generated")
                a_giv = step_dict.get("atom_given")
                
                if a_gen is None: a_gen = dummy_atom
                if a_giv is None: a_giv = dummy_atom
                
                if t == 0:
                    # Keep the entire first atom
                    gen_chunks.append(a_gen)
                    given_chunks.append(a_giv)
                else:
                    # Discard the redundant past memory, keep only the new frames
                    gen_chunks.append(a_gen[:, :, self.macro_overlap_frames:])
                    given_chunks.append(a_giv[:, :, self.macro_overlap_frames:])
                    
            stitched_gen = torch.cat(gen_chunks, dim=-1)
            stitched_given = torch.cat(given_chunks, dim=-1)
            
            audio_gen = self._decode_single_atom(stitched_gen).to(self.device)
            audio_given = self._decode_single_atom(stitched_given).to(self.device)
            
            # Update AF Envelope math to match Prefix geometry
            total_samples = audio_gen.shape[-1]
            AF_envelope = torch.zeros(1, 1, total_samples, device=self.device)
            
            for t in range(len(timeline)):
                AF = float(timeline[t].get("AF", 0.0))
                AF = max(0.0, min(1.0, AF))
                
                if t == 0:
                    start_sample = 0
                    end_sample = self.segment_samples
                else:
                    start_sample = self.segment_samples + (t - 1) * self.hop_samples
                    end_sample = start_sample + self.hop_samples
                    
                AF_envelope[:, :, start_sample:end_sample] = AF
                
            final_audio = torch.sqrt(AF_envelope) * audio_given + torch.sqrt(1.0 - AF_envelope) * audio_gen
            final_audio_tensor = final_audio.squeeze(0).cpu()

        elif method == "ola":
            if self.verbose:
                print("\n--- Rendering Audio Timeline (Real-Time Asymmetric OLA Mode) ---")
                
            total_steps = len(timeline)
            total_samples = (total_steps - 1) * self.hop_samples + self.segment_samples
            output_buffer = torch.zeros(1, 2, total_samples) 
            ola_window = self.ola_window.view(1, 1, -1).cpu()

            for t in tqdm(range(total_steps), desc="Mixing Audio", disable=not self.verbose):
                step_dict = timeline[t]
                AF = float(step_dict.get("AF", 0.0))
                AF = max(0.0, min(1.0, AF))
                
                audio_mix = torch.zeros(1, 2, self.segment_samples)
                
                if AF > 0.0 and step_dict.get("atom_given") is not None:
                    audio_given = self._decode_single_atom(step_dict["atom_given"])
                    audio_mix += math.sqrt(AF) * audio_given
                    
                if AF < 1.0 and step_dict.get("atom_generated") is not None:
                    audio_generated = self._decode_single_atom(step_dict["atom_generated"])
                    audio_mix += math.sqrt(1.0 - AF) * audio_generated
                    
                audio_mix = audio_mix * ola_window
                
                start_sample = t * self.hop_samples
                end_sample = start_sample + self.segment_samples
                output_buffer[:, :, start_sample:end_sample] += audio_mix
                
            final_audio_tensor = output_buffer.squeeze(0)

        elif method == "ola_smooth":
            if self.verbose:
                print("\n--- Rendering Audio Timeline (Real-Time Asymmetric OLA Smooth Mode) ---")
                
            total_steps = len(timeline)
            total_samples = (total_steps - 1) * self.hop_samples + self.segment_samples
            output_buffer = torch.zeros(1, 2, total_samples) 
            ola_window = self.ola_window.view(1, 1, -1).cpu()

            # --- Smoothing Hyperparameters ---
            alpha = 0.6      # EMA factor (0.0 = completely flat/frozen, 1.0 = no smoothing)
            max_jump = 1.15  # Scale cannot grow by more than 15% per step
            max_drop = 0.85  # Scale cannot drop by more than 15% per step
            
            prev_scale = None

            for t in tqdm(range(total_steps), desc="Mixing Audio (Smooth)", disable=not self.verbose):
                step_dict = timeline[t]
                AF = float(step_dict.get("AF", 0.0))
                AF = max(0.0, min(1.0, AF))
                
                audio_mix = torch.zeros(1, 2, self.segment_samples)
                
                # --- 1. Process & Smooth Generated Atom ---
                atom_gen = step_dict.get("atom_generated")
                if atom_gen is not None:
                    # Clone to prevent permanently altering the timeline dictionary
                    atom_gen_smooth = atom_gen.clone()
                    
                    # Extract the raw scalar value
                    raw_scale = torch.abs(atom_gen[:, 128, :]).mean(dim=-1, keepdim=True)
                    
                    if prev_scale is None:
                        smoothed_scale = raw_scale
                    else:
                        # Rate limit the jump
                        target_scale = torch.clamp(raw_scale, prev_scale * max_drop, prev_scale * max_jump)
                        # Apply EMA blend
                        smoothed_scale = (alpha * target_scale) + ((1.0 - alpha) * prev_scale)
                        
                    prev_scale = smoothed_scale
                    
                    # Inject smoothed scale back into the 129th dimension
                    atom_gen_smooth[:, 128, :] = smoothed_scale.expand_as(atom_gen_smooth[:, 128, :])
                else:
                    atom_gen_smooth = None

                # --- 2. Standard OLA Decoding ---
                if AF > 0.0 and step_dict.get("atom_given") is not None:
                    audio_given = self._decode_single_atom(step_dict["atom_given"])
                    audio_mix += math.sqrt(AF) * audio_given
                    
                if AF < 1.0 and atom_gen_smooth is not None:
                    # Pass the *smoothed* tensor to the internal decoder
                    audio_generated = self._decode_single_atom(atom_gen_smooth)
                    audio_mix += math.sqrt(1.0 - AF) * audio_generated
                    
                audio_mix = audio_mix * ola_window
                
                start_sample = t * self.hop_samples
                end_sample = start_sample + self.segment_samples
                output_buffer[:, :, start_sample:end_sample] += audio_mix
                
            final_audio_tensor = output_buffer.squeeze(0)

        else:
            raise ValueError(f"Unknown rendering method: {method}")

        if output_path:
            sf_audio = final_audio_tensor.transpose(0, 1).numpy()
            sf.write(output_path, sf_audio, self.sr)
            if self.verbose:
                print(f"✅ Audio perfectly rendered and saved to: {output_path}")
            
        return final_audio_tensor