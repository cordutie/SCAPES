import soundfile as sf
import librosa
import torch
import torch.nn as nn
import math
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from IPython.display import Audio, display
from pathlib import Path
import matplotlib.pyplot as plt

from SCAPES.data.config_loader import parse_gin_config, _get_nested, _SIZE_TABLE, _resolve_size_table, TrainingConfig
from SCAPES.models.flow.FlowModel import FlowModel
from SCAPES.models.factorization.LocalEncoder import LocalEncoder
from SCAPES.auxiliar.encodec_wrapper import EncodecProcessor
from SCAPES.auxiliar.clap_wrapper import CLAPWrapper
from SCAPES.data.dataprep.structure import _compute_structure_features

# ==========================================
# PIPELINE HELPER FUNCTIONS
# ==========================================

def load_and_encode(engine, audio_path, max_duration=None):
    audio_tensor = engine.load_audio_to_tensor(audio_path)
    if max_duration != None and audio_tensor.shape[-1] > engine.sr * max_duration:
        audio_tensor = audio_tensor[:,:,:48000*max_duration]
        
    print(f"--- Encoding audio: {audio_path}")
    atoms = engine.encode_audio_to_atoms(audio_tensor)
    
    print(f"--- Computing context for audio: {audio_path}")
    contexts = engine.compute_context_track(audio_tensor)
    
    structures = None
    if engine.use_structure:
        print(f"--- Computing structure for audio: {audio_path}")
        structures = engine.compute_structure_track(audio_tensor)
        
    return atoms, contexts, structures

def run_resynthesis_pipeline(
    engine,
    audio_path,
    duration=60,
    play=True,
    save_path=None,
    TF=False, 
    NFE = 32,
    cfg_scale = 3.0,       
    context_static=False,  
    decode_method="ola_smooth"
):
    atoms_src, contexts_src, structures_src = load_and_encode(engine, audio_path, max_duration=duration)

    atoms = atoms_src
    contexts = contexts_src
    structures = structures_src

    if context_static:  
        contexts = [contexts_src[0] for _ in range(len(atoms))]    
        if structures is not None:
            structures = [structures_src[0] for _ in range(len(atoms))]

    cold_start = True
    if TF != True and TF != False:
        cold_start = False
        TF = False

    timeline = engine.build_base_timeline(
        atoms_129D=atoms,
        context_embeddings=contexts,
        structure_embeddings=structures, 
        default_TF=TF
    )

    if not cold_start:
        for t in range(0, 5):
            timeline[t]["TF"] = True

    completed_timeline = engine.generate(timeline, NFE=NFE, cfg_scale=cfg_scale)
    final_wav = engine.decode_timeline(completed_timeline, output_path=None, method=decode_method)

    if play:
        filename = Path(audio_path).stem
        print("Resynthesis: ", filename)
        display(Audio(final_wav, rate=engine.sr))

    if save_path != None:
        sf_audio = final_wav.transpose(0, 1).numpy()
        sf.write(save_path, sf_audio, engine.sr)
        print(f"✅ Resynthesized audio saved to: {save_path}")
    return final_wav

def sticky_curve_torch(n_points=100, stickiness=1.0):
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

def low_pass_filter(signal, alpha=0.5):
    filtered = torch.zeros_like(signal)
    filtered[0] = signal[0]
    for t in range(1, signal.shape[0]):
        filtered[t] = alpha * signal[t] + (1 - alpha) * filtered[t-1]
    for i in range(9):
        for t in range(1, signal.shape[0]):
            filtered[t] = alpha * filtered[t] + (1 - alpha) * filtered[t-1]
    return filtered

def slerp(v0, v1, alpha, eps=1e-7):
    v0 = v0 / v0.norm(p=2)
    v1 = v1 / v1.norm(p=2)
    dot = torch.clamp(torch.dot(v0, v1), -1.0 + eps, 1.0 - eps)
    theta = torch.acos(dot)
    if theta < eps:
        return (1 - alpha) * v0 + alpha * v1
    sin_theta = torch.sin(theta)
    w0 = torch.sin((1 - alpha) * theta) / sin_theta
    w1 = torch.sin(alpha * theta) / sin_theta
    return w0 * v0 + w1 * v1

def run_interpolation_pipeline(
    engine,
    audio_path_1,
    audio_path_2,
    timeline_size=300,
    stay_time=1,
    stickyness = 1.0,
    plot_stickyness_curve=False,
    play=True,
    save_path=None,
    NFE = 32,
    cfg_scale = 3.0,       
    context_static=True,  
    decode_method="ola_smooth",
    cache=True
):
    alpha_values      = sticky_curve_torch(n_points=timeline_size - 2 * stay_time, stickiness=stickyness)
    alpha_values_full = torch.cat([torch.zeros(stay_time), alpha_values, torch.ones(stay_time)])
    alpha_values_full = low_pass_filter(alpha_values_full, alpha=0.5)

    if plot_stickyness_curve:
        plt.figure(figsize=(10, 4))
        plt.plot(alpha_values_full.detach().cpu().numpy())
        plt.title(f'Interpolation Values with Stickiness={stickyness}')
        plt.grid()
        plt.show()

    cache_1_found = False
    cache_2_found = False
    
    filename_1 = Path(audio_path_1).stem
    filename_2 = Path(audio_path_2).stem
    
    atoms_1_path    = filename_1 + "_atoms.pt"
    contexts_1_path = filename_1 + "_contexts.pt"
    struct_1_path   = filename_1 + "_structures.pt"
    
    atoms_2_path    = filename_2 + "_atoms.pt"
    contexts_2_path = filename_2 + "_contexts.pt"
    struct_2_path   = filename_2 + "_structures.pt"

    if cache:
        if Path(atoms_1_path).exists() and Path(contexts_1_path).exists():
            print(f"Loading cached encodings for {audio_path_1}...")
            atoms_1    = torch.load(atoms_1_path)
            contexts_1 = torch.load(contexts_1_path)
            structures_1 = torch.load(struct_1_path) if Path(struct_1_path).exists() else None
            cache_1_found = True

        if Path(atoms_2_path).exists() and Path(contexts_2_path).exists():
            print(f"Loading cached encodings for {audio_path_2}...")
            atoms_2    = torch.load(atoms_2_path)
            contexts_2 = torch.load(contexts_2_path)
            structures_2 = torch.load(struct_2_path) if Path(struct_2_path).exists() else None
            cache_2_found = True

    if not cache_1_found:
        atoms_1, contexts_1, structures_1 = load_and_encode(engine, audio_path_1, max_duration=31)
    if not cache_2_found:
        atoms_2, contexts_2, structures_2 = load_and_encode(engine, audio_path_2, max_duration=31)

    if cache:
        if not cache_1_found:
            torch.save(atoms_1, atoms_1_path)
            torch.save(contexts_1, contexts_1_path)
            if structures_1 is not None: torch.save(structures_1, struct_1_path)
        if not cache_2_found:
            torch.save(atoms_2, atoms_2_path)
            torch.save(contexts_2, contexts_2_path)
            if structures_2 is not None: torch.save(structures_2, struct_2_path)

    if stay_time < 0 or not isinstance(stay_time, int):
        raise ValueError("Stay time must be a non-negative integer.")

    if not context_static:
        if len(contexts_1) < timeline_size or len(contexts_2) < timeline_size:
            raise ValueError("Audio does not have enough context embeddings for the timeline size.")
        contexts_1 = contexts_1[:timeline_size]
        contexts_2 = contexts_2[:timeline_size]
        
        if engine.use_structure:
            structures_1 = structures_1[:timeline_size]
            structures_2 = structures_2[:timeline_size]

    c0 = contexts_1[0]
    c1 = contexts_2[0]

    atoms = [None] * timeline_size
    contexts = [] 
    structures = [] if engine.use_structure else None

    alpha_values_full = alpha_values_full.to(c0.device)

    for t in range(timeline_size):
        alpha = alpha_values_full[t]
        
        # 1. Slerp CLAP Contexts
        if context_static:
            ctx = slerp(c0, c1, alpha)
        else:
            ctx = slerp(contexts_1[t], contexts_2[t], alpha)
        contexts.append(ctx)
        
        # 2. Lerp Structure Embeddings (Scalars interpolate linearly)
        if engine.use_structure:
            if context_static:
                s_interp = (1 - alpha) * structures_1[0] + alpha * structures_2[0]
            else:
                s_interp = (1 - alpha) * structures_1[t] + alpha * structures_2[t]
            structures.append(s_interp)

    timeline = engine.build_base_timeline(
        atoms_129D=atoms,
        context_embeddings=contexts,
        structure_embeddings=structures,
        default_TF=False
    )

    completed_timeline = engine.generate(timeline, NFE=NFE, cfg_scale=cfg_scale)
    final_wav = engine.decode_timeline(completed_timeline, output_path=None, method=decode_method)

    if play:
        print(f"Interpolation: {Path(audio_path_1).stem} -> {Path(audio_path_2).stem}")
        display(Audio(final_wav, rate=engine.sr))

    if save_path != None:
        sf_audio = final_wav.transpose(0, 1).numpy()
        sf.write(save_path, sf_audio, engine.sr)
        print(f"✅ Interpolated audio saved to: {save_path}")

    return final_wav

# ==========================================
# FLOW INFERENCE ENGINE
# ==========================================

class FlowInference:
    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
        verbose: bool = False,
        checkpoint: str = "best",
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.verbose = verbose

        model_dir = Path(model_dir)
        gin_path = model_dir / "checkpoints" / "inference.gin"
        if not gin_path.exists():
            raise FileNotFoundError(
                f"Expected inference config at {gin_path}. "
                "Run training first to generate this file."
            )

        config = parse_gin_config(gin_path)

        # ─── Data geometry ───
        frame_dim = 129
        context_vector_dim = 1024
        sr = 48000
        frame_rate = 150

        self.atoms_frames = int(_get_nested(config, ["atoms", "frames"], 48))
        self.atoms_hop_frames = int(_get_nested(config, ["atoms", "hop_frames"], 15))
        self.crossfade_frames = int(_get_nested(config, ["atoms", "crossfade_frames"], 3))
        self.memory_buffer_atoms = int(_get_nested(config, ["dataset", "memory_buffer_atoms"], 3))

        self.context_seconds = float(_get_nested(config, ["dataset", "context_seconds"], 1.0))
        self.semantic_random_extension = _get_nested(config, ["dataset", "semantic_random_extension"], True)
        self.structure_feature_names = _get_nested(config, ["structure", "features"], None)

        # ─── Model architecture ───
        raw_size = _get_nested(config, ["model", "size"], None)
        raw_d_model = _get_nested(config, ["model", "d_model"], None)
        raw_nhead = _get_nested(config, ["model", "nhead"], None)
        raw_num_layers = _get_nested(config, ["model", "num_layers"], None)
        raw_dim_feedforward = _get_nested(config, ["model", "dim_feedforward"], None)

        pseudo_config = TrainingConfig(
            size=raw_size,
            d_model=raw_d_model,
            nhead=raw_nhead,
            num_layers=raw_num_layers,
            dim_feedforward=raw_dim_feedforward,
            local_encoder_hidden_dim=int(_get_nested(config, ["local_encoder", "hidden_dim"], 256)),
            local_encoder_time_entanglement=_get_nested(config, ["local_encoder", "time_entanglement"], True),
            local_encoder_temporal_compression=int(_get_nested(config, ["local_encoder", "temporal_compression"], 1)),
            cfg_scale=float(_get_nested(config, ["inference", "cfg_scale"], 3.0)),
        )
        pseudo_config = _resolve_size_table(pseudo_config)
        self.training_config = pseudo_config

        structure_dim = len(self.structure_feature_names) if self.structure_feature_names else 0

        # ─── Build models ───
        self.local_encoder = LocalEncoder(
            in_channels=frame_dim,
            hidden_dim=pseudo_config.local_encoder_hidden_dim,
            out_channels=pseudo_config.d_model,
            time_entanglement=pseudo_config.local_encoder_time_entanglement,
            temporal_compression=pseudo_config.local_encoder_temporal_compression,
        ).to(self.device).eval()

        self.model = FlowModel(
            frame_dim=frame_dim,
            context_vector_dim=context_vector_dim,
            num_past_atoms=self.memory_buffer_atoms,
            frames_per_atom=self.atoms_frames,
            d_model=pseudo_config.d_model,
            nhead=pseudo_config.nhead,
            num_layers=pseudo_config.num_layers,
            dim_feedforward=pseudo_config.dim_feedforward,
            structure_dim=structure_dim,
            device=self.device,
        ).to(self.device).eval()

        # ─── Load checkpoint weights ───
        flow_name, local_name = self._resolve_checkpoint_names(checkpoint, model_dir)

        flow_ckpt = model_dir / "checkpoints" / flow_name
        local_ckpt = model_dir / "checkpoints" / local_name

        for ckpt_path, target_model, label in [
            (flow_ckpt, self.model, "flow model"),
            (local_ckpt, self.local_encoder, "local encoder"),
        ]:
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found at {ckpt_path}. "
                    f"Train a model first or point model_dir to a valid checkpoint directory."
                )
            state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            if isinstance(state, dict) and 'model_state_dict' in state:
                target_model.load_state_dict(state['model_state_dict'])
            else:
                target_model.load_state_dict(state)
            if self.verbose:
                print(f"  Loaded {label} from {ckpt_path.name}")

        if self.verbose:
            for m in [self.local_encoder, self.model]:
                n = sum(p.numel() for p in m.parameters() if p.requires_grad)
                print(f"  {m.__class__.__name__}: {n:,} params")

        # ─── Set up components ───
        self.processor = EncodecProcessor(sr=sr, streamable=True, device=self.device)
        self.context_model = CLAPWrapper(version="2023", use_cuda=(self.device == "cuda"))

        self.use_structure = structure_dim > 0
        self.segment_length = self.memory_buffer_atoms

        self.sr = sr
        self.frame_rate = frame_rate
        self.samples_per_frame = sr // frame_rate
        self.macro_overlap_frames = self.atoms_frames - self.atoms_hop_frames

        self.segment_samples = self.atoms_frames * self.samples_per_frame
        self.hop_samples = self.atoms_hop_frames * self.samples_per_frame
        self.crossfade_samples = self.crossfade_frames * self.samples_per_frame
        self.macro_overlap_samples = self.macro_overlap_frames * self.samples_per_frame

        # Context window: time-based (same logic as dataset.py)
        self.context_samples = int(round(self.context_seconds * self.sr))

        self.ola_window = self._build_ola_window().to(self.device)
        self.timeline: List[Dict[str, Any]] = []

    @staticmethod
    def _resolve_checkpoint_names(checkpoint, model_dir):
        if checkpoint == "best":
            return "best_flow_model.pt", "best_local_encoder.pt"
        elif checkpoint == "last":
            return "last_flow_model.pt", "last_local_encoder.pt"
        elif isinstance(checkpoint, int):
            return f"epoch_{checkpoint}_flow_model.pt", f"epoch_{checkpoint}_local_encoder.pt"
        elif isinstance(checkpoint, str) and checkpoint.isdigit():
            return f"epoch_{checkpoint}_flow_model.pt", f"epoch_{checkpoint}_local_encoder.pt"
        else:
            raise ValueError(f"checkpoint must be 'best', 'last', or an int (epoch number), got {checkpoint}")

    def _build_ola_window(self):
        zeros_frames = self.macro_overlap_frames - self.crossfade_frames
        zeros = torch.zeros(zeros_frames * self.samples_per_frame)
        
        hann_window = torch.hann_window(self.crossfade_samples * 2)
        left_hann = hann_window[:self.crossfade_samples]
        right_hann = hann_window[self.crossfade_samples:]
        
        ones_frames = self.atoms_hop_frames - self.crossfade_frames
        ones = torch.ones(ones_frames * self.samples_per_frame)
        
        return torch.cat([zeros, left_hann, ones, right_hann])
    
    def load_audio_to_tensor(self, audio_path: str) -> torch.Tensor:
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
        audio_tensor = audio_tensor.to(self.device)
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0) 
            
        total_samples = audio_tensor.shape[-1]
        atoms_129D = []
        
        for start in range(0, total_samples, self.hop_samples):
            end = start + self.segment_samples
            segment = audio_tensor[:, :, start:end]
            
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
    def compute_context_track(self, audio_tensor: torch.Tensor) -> List[torch.Tensor]:
        context_embeddings = []
        total_samples = audio_tensor.shape[-1]
        last_valid_emb = None

        total_atoms = 0
        for start in range(0, total_samples, self.hop_samples):
            end = start + self.segment_samples
            if end <= total_samples:
                total_atoms += 1
            else:
                break

        for t in range(total_atoms):
            start_sample = t * self.hop_samples
            end_sample = start_sample + self.context_samples

            if end_sample <= total_samples:
                window_audio = audio_tensor[:, :, start_sample:end_sample]

                emb = self.context_model.compute_embedding(
                    window_audio,
                    og_sr=self.sr,
                    random_extension=self.semantic_random_extension,
                ).squeeze(0)

                context_embeddings.append(emb)
                last_valid_emb = emb

            else:
                if last_valid_emb is not None:
                    context_embeddings.append(last_valid_emb)
                else:
                    raise ValueError("Audio file is too short to compute even one context window!")

        return context_embeddings

    @torch.no_grad()
    def compute_structure_track(self, audio_tensor: torch.Tensor) -> List[torch.Tensor]:
        """Extracts 1D Structure Features using the exact matching Sliding Window."""
        audio_tensor = audio_tensor.to(self.device)
        total_samples = audio_tensor.shape[-1]
        structures = []
        
        n_fft = max(256, self.samples_per_frame * 4)
        hop_length = self.samples_per_frame
        
        for start in range(0, total_samples, self.hop_samples):
            end = start + self.segment_samples
            segment = audio_tensor[:, :, start:end]
            
            if segment.shape[-1] < self.segment_samples:
                break
                
            struct = _compute_structure_features(
                audio=segment,
                sr=self.sr,
                atoms_frames=self.atoms_frames,
                n_fft=n_fft,
                hop_length=hop_length,
                feature_names=self.structure_feature_names, # <--- NEW: Force exact match
                mean_pooling=True 
            )
            structures.append(struct.to(self.device))
            
        return structures
    
    def build_base_timeline(self, atoms_129D, context_embeddings, structure_embeddings=None, default_TF=False):
        if len(atoms_129D) != len(context_embeddings):
            raise ValueError("Length mismatch!")
            
        timeline = []
        for t in range(len(atoms_129D)):
            step_dict = {
                "step": t,
                "atom_given": atoms_129D[t],           
                "context_embedding": context_embeddings[t], 
                "structure_embedding": structure_embeddings[t] if structure_embeddings else None,
                "atom_generated": None,                
                "TF": default_TF                       
            }
            timeline.append(step_dict)
            
        self.timeline = timeline 
        return timeline
    
    @torch.no_grad()
    def generate(self, timeline: List[Dict[str, Any]], NFE: int = 32, cfg_scale: float = 3.0) -> List[Dict[str, Any]]:
        self.model.eval()
        self.local_encoder.eval()

        if not timeline:
            raise ValueError("Timeline is empty!")

        M = self.segment_length
        total_steps = len(timeline)
        dummy_atom = torch.zeros(1, 129, self.atoms_frames, device=self.device)

        if self.verbose:
            print(f"\n--- Starting Generation over {total_steps} steps (NFE={NFE}, CFG={cfg_scale}) ---")

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

            structure = timeline[t].get("structure_embedding")
            if structure is not None:
                structure = structure.to(self.device)
                if structure.dim() == 1:
                    structure = structure.unsqueeze(0)
                
            x0 = torch.randn(1, self.atoms_frames, 129, device=self.device)
            
            pred = self.model.generate(
                x0=x0, 
                encoded_past=encoded_past, 
                clap_context=context, 
                structure_vector=structure,
                max_nfe=NFE,
                cfg_scale=cfg_scale
            ) 
            
            timeline[t]["atom_generated"] = pred.transpose(1, 2)

        if self.verbose:
            print("✅ Generation Complete!")
        return timeline
    
    def _decode_single_atom(self, atom_129D: torch.Tensor, override_scale=None) -> torch.Tensor:
        latent = atom_129D[:, :128, :] 
        if override_scale is not None:
            scale_val = override_scale
        else:
            scale_val = torch.abs(atom_129D[:, 128, :]).mean(dim=-1, keepdim=True)       

        metadata = {
            "audio_scales": [scale_val.squeeze(0).float()],
            "padding_mask": torch.ones(
                (1, latent.shape[-1] * self.samples_per_frame), 
                dtype=torch.bool, device=self.device
            )
        }
        audio = self.processor.decode_latents_audio(latent, metadata=metadata)
        return audio.cpu()

    @torch.no_grad()
    def decode_timeline(self, timeline: List[Dict[str, Any]], output_path: str = None, method: str = "ola_smooth"):
        if not timeline:
            raise ValueError("Timeline is empty!")
            
        total_steps = len(timeline)
        total_samples = (total_steps - 1) * self.hop_samples + self.segment_samples
        output_buffer = torch.zeros(1, 2, total_samples) 
        ola_window = self.ola_window.view(1, 1, -1).cpu()

        if method == "ola_smooth":
            if self.verbose:
                print("\n--- Rendering Audio Timeline (Smooth Mode) ---")
                
            alpha_smooth = 0.6      
            max_jump = 1.15  
            max_drop = 0.85  
            prev_scale = None

            for t in tqdm(range(total_steps), desc="Mixing Audio", disable=not self.verbose):
                step_dict = timeline[t]
                
                atom_to_decode = step_dict.get("atom_generated")
                is_generated = True
                
                if atom_to_decode is None:
                    atom_to_decode = step_dict.get("atom_given")
                    is_generated = False
                    
                if atom_to_decode is None:
                    continue

                if is_generated:
                    raw_scale = torch.abs(atom_to_decode[:, 128, :]).mean(dim=-1, keepdim=True)
                    if prev_scale is None:
                        smoothed_scale = raw_scale
                    else:
                        target_scale = torch.clamp(raw_scale, prev_scale * max_drop, prev_scale * max_jump)
                        smoothed_scale = (alpha_smooth * target_scale) + ((1.0 - alpha_smooth) * prev_scale)
                        
                    prev_scale = smoothed_scale
                    audio = self._decode_single_atom(atom_to_decode, override_scale=smoothed_scale)
                else:
                    audio = self._decode_single_atom(atom_to_decode)
                    
                audio = audio * ola_window
                
                start_sample = t * self.hop_samples
                end_sample = start_sample + self.segment_samples
                output_buffer[:, :, start_sample:end_sample] += audio
                
        else:
            raise ValueError(f"Method '{method}' removed/unsupported. Please use 'ola_smooth'.")

        final_audio_tensor = output_buffer.squeeze(0)

        if output_path:
            sf_audio = final_audio_tensor.transpose(0, 1).numpy()
            sf.write(output_path, sf_audio, self.sr)
            if self.verbose:
                print(f"✅ Audio perfectly rendered and saved to: {output_path}")
            
        return final_audio_tensor