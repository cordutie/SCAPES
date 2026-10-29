import math
import tempfile
import soundfile as sf
import librosa
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from IPython.display import Audio, display
from pathlib import Path
import matplotlib.pyplot as plt

from SCAPES.data.config_loader import (parse_gin_config, _get_nested, _SIZE_TABLE, ConfigError, load_dataprep_config)
from SCAPES.data.geometry import geometry_from_inference_gin, resolve_geometry, with_overlap_mode
from SCAPES.data.representations import frames_to_samples_48k
from SCAPES.models.flow.FlowModel import FlowModel
from SCAPES.models.factorization.LocalEncoder import LocalEncoder
from SCAPES.data.dataprep.structure import _compute_structure_features
from SCAPES.auxiliar.losses_flow import time_to_spectral, spectral_to_time
from SCAPES.inference.render import StreamRenderer, render_atoms
from SCAPES import codecs as scapes_codecs

# ==========================================
# PIPELINE HELPER FUNCTIONS
# ==========================================

def load_and_encode(engine, audio_path, max_duration=None):
    audio_tensor = engine.load_audio_to_tensor(audio_path)
    if max_duration != None and audio_tensor.shape[-1] > engine.load_sr * max_duration:
        audio_tensor = audio_tensor[:,:,:engine.load_sr*max_duration]
        
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
    duration=None,
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

def run_batch_resynthesis_pipeline(
    engine,
    input_dir,
    output_dir="reconstructed",
    duration=None,
    NFE=32,
    cfg_scale=3.0,
    TF=False,
    decode_method="ola_smooth",
    extensions=(".wav", ".flac", ".mp3"),
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    audio_files = []
    for ext in extensions:
        audio_files.extend(sorted(input_path.rglob(f"*{ext}")))
    audio_files.sort(key=lambda p: p.name)

    if not audio_files:
        print(f"No audio files found in {input_dir} with extensions {extensions}")
        return []

    print(f"Found {len(audio_files)} audio file(s) in {input_dir}")
    saved_paths = []

    for audio_path in tqdm(audio_files, desc="Batch Resynthesis"):
        save_path = output_path / f"{audio_path.stem}_reconstructed.wav"
        atoms_src, contexts_src, structures_src = load_and_encode(engine, str(audio_path), max_duration=duration)

        atoms = atoms_src
        contexts = contexts_src
        structures = structures_src

        cold_start = True
        if TF is not True and TF is not False:
            cold_start = False
            tf_actual = False
        else:
            tf_actual = TF

        timeline = engine.build_base_timeline(
            atoms_129D=atoms,
            context_embeddings=contexts,
            structure_embeddings=structures,
            default_TF=tf_actual
        )

        if not cold_start:
            for t in range(0, 5):
                timeline[t]["TF"] = True

        completed_timeline = engine.generate(timeline, NFE=NFE, cfg_scale=cfg_scale)
        final_wav = engine.decode_timeline(completed_timeline, output_path=None, method=decode_method)

        sf_audio = final_wav.transpose(0, 1).numpy()
        sf.write(str(save_path), sf_audio, engine.sr)
        saved_paths.append(str(save_path))

    print(f"\n✅ Batch resynthesis complete — {len(saved_paths)} file(s) saved to {output_dir}/")
    return saved_paths

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

_CODEC_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "codecs"


def _gin_value(v):
    return f'"{v}"' if isinstance(v, str) else str(v)


class FlowInference:
    """
    Generation from a trained model directory, in both modes (design: Inference).
    The codec and the geometry come from <model_dir>/checkpoints/inference.gin (V1 files need the two lines
    codec.name and generation.mode). Latents inside the timeline are in model units (standardised if the model was).
    """

    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
        verbose: bool = False,
        checkpoint: str = "best",
        overlap_mode: str = "normal",
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model_dir = Path(model_dir)
        gin_path = model_dir / "checkpoints" / "inference.gin"
        if not gin_path.exists():
            raise FileNotFoundError(
                f"Expected inference config at {gin_path}. "
                "Run training first to generate this file."
            )
        raw = parse_gin_config(gin_path)
        codec_name = _get_nested(raw, ["codec", "name"], None)
        if codec_name is None or _get_nested(raw, ["generation", "mode"], None) is None:
            raise ConfigError(f"{gin_path}: codec.name and generation.mode are required "
                              f"(a V1 inference.gin works after adding these two lines)")
        info = scapes_codecs.get_codec_info(codec_name)
        geom = geometry_from_inference_gin(raw, info)

        arch = {
            "d_model": _get_nested(raw, ["model", "d_model"], None),
            "nhead": _get_nested(raw, ["model", "nhead"], None),
            "num_layers": _get_nested(raw, ["model", "num_layers"], None),
            "dim_feedforward": _get_nested(raw, ["model", "dim_feedforward"], None),
        }
        if any(v is None for v in arch.values()):
            size = _get_nested(raw, ["model", "size"], None)
            if size not in _SIZE_TABLE:
                raise ConfigError(f"{gin_path}: model architecture is missing (model.size or the four model.* lines)")
            arch = dict(_SIZE_TABLE[size])
        self._setup(raw, info, geom, arch, device, verbose)
        self.set_overlap_mode(overlap_mode)

        # ─── Load checkpoint weights ───
        flow_name, local_name = self._resolve_checkpoint_names(checkpoint, model_dir)
        for name, target_model, label in [(flow_name, self.model, "flow model"),
                                          (local_name, self.local_encoder, "local encoder")]:
            ckpt_path = model_dir / "checkpoints" / name
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

        # ─── Latent statistics of standardised models ───
        if self.standardize:
            stats_path = model_dir / "checkpoints" / "latent_stats.pt"
            if not stats_path.exists():
                raise FileNotFoundError(f"{stats_path} is missing but training.standardize_latents is True")
            stats = torch.load(stats_path, map_location=self.device, weights_only=True)
            self.mu, self.std = stats["mu"].float(), stats["std"].float()

    # Atoms: "normal" = the crossfade of the model's config, "strong" = 25 % of the atom, "brute" = 50 % (see geometry.py).
    # Raises ConfigError if the geometry cannot hold that crossfade (it must not exceed the hop or the overlap).
    def set_overlap_mode(self, mode: str) -> None:
        with_overlap_mode(self.geom, self.info, mode)
        self.overlap_mode = mode

    # Random-init engine with the template geometry of <codec>_<mode>.gin (S3 timing); weights are not loaded and the codec
    # is only loaded at the first decode. geometry: optional gin overrides, e.g. {"ar.block_seconds": 0.32};
    # size "custom" reads the model.* lines from the template.
    @classmethod
    def from_config(cls, codec: str, mode: str, size: str = "large", geometry: Optional[dict] = None,
                    device: Optional[str] = None, seed: int = 0):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        suffix = "ar" if mode == "autoregressive" else "atoms"
        lines = (_CODEC_CONFIG_DIR / f"{codec}_{suffix}.gin").read_text().splitlines()
        lines += [f"{k} = {_gin_value(v)}" for k, v in (geometry or {}).items()]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataprep.gin"
            path.write_text("\n".join(lines) + "\n")
            raw = parse_gin_config(path)
            info = scapes_codecs.get_codec_info(codec)
            geom = resolve_geometry(load_dataprep_config(path), info)
        if size == "custom":
            arch = {k: _get_nested(raw, ["model", k], None) for k in ("d_model", "nhead", "num_layers", "dim_feedforward")}
            if any(v is None for v in arch.values()):
                raise ConfigError("size 'custom' needs model.d_model, nhead, num_layers and dim_feedforward in the gin lines")
        else:
            arch = dict(_SIZE_TABLE[size])
        torch.manual_seed(seed)
        obj = cls.__new__(cls)
        obj._setup(raw, info, geom, arch, device, False)
        return obj

    def _setup(self, raw, info, geom, arch, device, verbose):
        self.device = device
        self.verbose = verbose
        self.info = info
        self.geom = geom
        self.mode = geom.mode
        self.sr = info.sample_rate            # output rate (codec rate)
        self.load_sr = 48000                  # rate of the audio given to CLAP, structure and the helper pipelines
        self.frame_rate = info.frame_rate
        self.latent_dim = geom.latent_dim
        self._codec = None
        self._context_model = None
        self.mu = self.std = None
        self.standardize = bool(_get_nested(raw, ["training", "standardize_latents"], False))

        # ─── Geometry in frames ───
        self.spectral_mode = bool(_get_nested(raw, ["training", "spectral_representation"], False))
        if self.spectral_mode and geom.codec != "encodec48k":
            raise NotImplementedError("spectral_representation is only available for encodec48k (D13)")
        if self.spectral_mode:
            frame_dim = 385  # 128*3 + 1
            self.spectral_frames = int(_get_nested(raw, ["atoms", "spectral_frames"], geom.unit_frames // 2 + 1))
            model_frames = self.spectral_frames
        else:
            frame_dim = geom.frame_dim
            self.spectral_frames = None
            model_frames = geom.unit_frames + geom.prefix_frames
        self.unit_frames = geom.unit_frames
        self.memory_buffer_atoms = geom.memory_units
        self.segment_length = geom.memory_units
        self.context_seconds = geom.context_seconds
        self.context_samples = int(round(self.context_seconds * self.load_sr))
        self.semantic_random_extension = _get_nested(raw, ["dataset", "semantic_random_extension"], True)
        self.structure_feature_names = _get_nested(raw, ["structure", "features"], None)
        structure_dim = len(self.structure_feature_names) if self.structure_feature_names else 0
        self.use_structure = structure_dim > 0

        # ─── Models ───
        hidden = int(_get_nested(raw, ["local_encoder", "hidden_dim"], 256))
        entangle = _get_nested(raw, ["local_encoder", "time_entanglement"], True)
        compression = int(_get_nested(raw, ["local_encoder", "temporal_compression"], 1))
        if geom.mode == "autoregressive" and compression != 1:
            raise NotImplementedError("autoregressive mode needs local_encoder.temporal_compression = 1")
        self.local_encoder = LocalEncoder(
            in_channels=frame_dim,
            hidden_dim=hidden,
            out_channels=arch["d_model"],
            time_entanglement=entangle,
            temporal_compression=compression,
        ).to(device).eval()
        self.model = FlowModel(
            frame_dim=frame_dim,
            context_vector_dim=1024,
            num_past_atoms=geom.memory_units,
            frames_per_atom=model_frames,
            d_model=arch["d_model"],
            nhead=arch["nhead"],
            num_layers=arch["num_layers"],
            dim_feedforward=arch["dim_feedforward"],
            structure_dim=structure_dim,
            memory_frames=geom.memory_frames if geom.mode == "autoregressive" else None,
            device=device,
        ).to(device).eval()
        if self.verbose:
            for m in [self.local_encoder, self.model]:
                n = sum(p.numel() for p in m.parameters() if p.requires_grad)
                print(f"  {m.__class__.__name__}: {n:,} params")

        self.timeline: List[Dict[str, Any]] = []

    # The codec and CLAP are loaded at first use
    @property
    def codec(self):
        if self._codec is None:
            self._codec = scapes_codecs.get_codec(self.info.name, self.device)
        return self._codec

    @property
    def context_model(self):
        if self._context_model is None:
            from SCAPES.auxiliar.clap_wrapper import CLAPWrapper
            self._context_model = CLAPWrapper(version="2023", use_cuda=(str(self.device).startswith("cuda")))
        return self._context_model

    # "best" falls back to "last" when training had no validation split (no best checkpoint is written then)
    @staticmethod
    def _resolve_checkpoint_names(checkpoint, model_dir):
        if checkpoint == "best":
            if not (Path(model_dir) / "checkpoints" / "best_flow_model.pt").exists():
                print("No best checkpoint (training had no validation split): using the last one.")
                return "last_flow_model.pt", "last_local_encoder.pt"
            return "best_flow_model.pt", "best_local_encoder.pt"
        elif checkpoint == "last":
            return "last_flow_model.pt", "last_local_encoder.pt"
        elif isinstance(checkpoint, int):
            return f"epoch_{checkpoint}_flow_model.pt", f"epoch_{checkpoint}_local_encoder.pt"
        elif isinstance(checkpoint, str) and checkpoint.isdigit():
            return f"epoch_{checkpoint}_flow_model.pt", f"epoch_{checkpoint}_local_encoder.pt"
        else:
            raise ValueError(f"checkpoint must be 'best', 'last', or an int (epoch number), got {checkpoint}")

    def load_audio_to_tensor(self, audio_path: str) -> torch.Tensor:
        audio_input, _ = librosa.load(audio_path, sr=self.load_sr, mono=False)
        audio_tensor = torch.tensor(audio_input).unsqueeze(0)

        if audio_tensor.dim() == 2 or audio_tensor.shape[1] == 1:
            if audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.unsqueeze(1)
            audio_tensor = audio_tensor.repeat(1, 2, 1)

        elif audio_tensor.shape[1] > 2:
            audio_tensor = audio_tensor[:, :2, :]

        return audio_tensor.to(self.device).float()

    # Start samples (48 kHz) of the generation units that fit in `total_samples`, and the unit length in 48 kHz samples
    def _unit_starts(self, total_samples):
        g = self.geom
        seg = int(frames_to_samples_48k(g.unit_frames, g))
        starts = []
        k = 0
        while True:
            start = int(frames_to_samples_48k(k * g.unit_hop_frames, g))
            if start + seg > total_samples:
                break
            starts.append(start)
            k += 1
        return starts, seg

    @torch.no_grad()
    def encode_audio_to_atoms(self, audio_tensor: torch.Tensor) -> List[torch.Tensor]:
        """Units [1, frame_dim, U] in model units (standardised latents, plus the scale channel if the codec has one)."""
        g = self.geom
        audio_tensor = audio_tensor.to(self.device)
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)
        starts, seg = self._unit_starts(audio_tensor.shape[-1])
        units = []
        if g.mode == "autoregressive":
            enc = self.codec.encode(audio_tensor, self.load_sr)
            z = self._standardise(enc.z)
            for k in range(len(starts)):
                units.append(z[:, :, k * g.unit_frames:(k + 1) * g.unit_frames])
            return units
        for start in starts:
            enc = self.codec.encode(audio_tensor[:, :, start:start + seg], self.load_sr)
            z = self._standardise(enc.z)[:, :, :g.unit_frames]
            if g.has_scale:
                scale = enc.scale.reshape(-1, 1, 1).expand(-1, 1, g.unit_frames)
                z = torch.cat([z, scale], dim=1)
            units.append(z)
        return units

    def _standardise(self, z):
        if self.mu is None:
            return z
        return (z - self.mu.view(1, -1, 1).to(z.device)) / self.std.view(1, -1, 1).to(z.device)

    @torch.no_grad()
    def compute_context_track(self, audio_tensor: torch.Tensor) -> List[torch.Tensor]:
        total_samples = audio_tensor.shape[-1]
        starts, _ = self._unit_starts(total_samples)
        context_embeddings = []
        last_valid_emb = None
        for start in starts:
            end = start + self.context_samples
            if end <= total_samples:
                emb = self.context_model.compute_embedding(
                    audio_tensor[:, :, start:end],
                    og_sr=self.load_sr,
                    random_extension=self.semantic_random_extension,
                ).squeeze(0)
                context_embeddings.append(emb)
                last_valid_emb = emb
            elif last_valid_emb is not None:
                context_embeddings.append(last_valid_emb)
            else:
                raise ValueError("Audio file is too short to compute even one context window!")
        return context_embeddings

    @torch.no_grad()
    def compute_structure_track(self, audio_tensor: torch.Tensor) -> List[torch.Tensor]:
        """Extracts 1D Structure Features using the exact matching Sliding Window."""
        audio_tensor = audio_tensor.to(self.device)
        starts, seg = self._unit_starts(audio_tensor.shape[-1])
        hop48 = int(round(self.info.hop_length * self.load_sr / self.info.sample_rate))
        n_fft = max(256, hop48 * 4)
        structures = []
        for start in starts:
            struct = _compute_structure_features(
                audio=audio_tensor[:, :, start:start + seg],
                sr=self.load_sr,
                atoms_frames=self.geom.unit_frames,
                n_fft=n_fft,
                hop_length=hop48,
                feature_names=self.structure_feature_names,
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

    def generate_soundscape(
        self,
        clap_embeddings: List[torch.Tensor],
        nfe: int = 16,
        cfg_scale: float = 3.0,
        output_path: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate audio from a sequence of precomputed CLAP embeddings (one per unit), without an input audio file.
        The past is filled from the generated units (zeros, masked, before the first ones).
        seed: if given, seeds torch before the first solve and is the base seed of stochastic decoders; otherwise the
        global torch RNG is used as it is.

        Returns:
            audio tensor [C, N] at the codec rate
        """
        if seed is not None:
            torch.manual_seed(seed)
        self.decode_seed = 0 if seed is None else int(seed)
        timeline = self.build_base_timeline(
            atoms_129D=[None] * len(clap_embeddings),
            context_embeddings=clap_embeddings,
            structure_embeddings=None,
            default_TF=False,
        )
        timeline = self.generate(timeline, NFE=nfe, cfg_scale=cfg_scale)
        return self.decode_timeline(timeline, output_path=output_path)

    # Unit of the past at step i: the given one when teacher-forced, else the generated one ([1, frame_dim, U])
    def _unit_at(self, timeline, i):
        step = timeline[i]
        return (step["atom_given"] if step["TF"] else step["atom_generated"]).to(self.device)

    # Atoms: the M previous atoms (zeros before the start) -> encoded past [1, M, T', d]; the missing ones are null
    def _atoms_past(self, timeline, t):
        g = self.geom
        M = g.memory_units
        dummy_atom = torch.zeros(1, g.frame_dim, g.unit_frames, device=self.device)
        past_atoms = [dummy_atom if i < 0 else self._unit_at(timeline, i) for i in range(t - M, t)]
        past_buffer = torch.cat(past_atoms, dim=0).unsqueeze(0)

        if self.spectral_mode:
            B_past, N_past, C_past, T_past = past_buffer.shape
            past_lat = past_buffer[:, :, :128, :]
            past_scl = past_buffer[:, :, 128:, :]
            past_lat_flat = past_lat.reshape(-1, 128, T_past).transpose(1, 2)
            past_spec_flat = time_to_spectral(past_lat_flat)
            past_spec = past_spec_flat.reshape(B_past, N_past, self.spectral_frames, 384).transpose(2, 3)
            past_scl_val = past_scl.mean(dim=-1, keepdim=True)
            past_scl_exp = past_scl_val.expand(-1, -1, -1, self.spectral_frames)
            past_buffer = torch.cat([past_spec, past_scl_exp], dim=2)

        encoded_past = self.local_encoder(past_buffer)
        num_nulls = max(0, M - t)
        if num_nulls > 0:
            encoded_past[:, :num_nulls, :, :] = self.model.null_past_embed
        return encoded_past, None

    # AR: one window of the last W frames. When fewer exist (t*B < W) the missing oldest frames are zeros and masked, as
    # in training (generation-start condition). The clean prefix is the last o frames of that window (zeros if missing).
    def _ar_past(self, timeline, t):
        g = self.geom
        W, o = g.memory_frames, g.prefix_frames
        frames = [self._unit_at(timeline, i) for i in range(t)]
        past = torch.cat(frames, dim=-1)[..., -W:] if frames else torch.zeros(1, g.frame_dim, 0, device=self.device)
        missing = W - past.shape[-1]
        if missing > 0:
            past = F.pad(past, (missing, 0))
        encoded_past = self.local_encoder(past.unsqueeze(1))              # [1, 1, W, d_model]
        if missing > 0:
            masked = (torch.arange(W, device=self.device) < missing).view(1, 1, W, 1)
            encoded_past = torch.where(masked, self.model.null_past_embed[:, :, :W], encoded_past)
        prefix = past[..., -o:].transpose(1, 2).contiguous() if o > 0 else None
        return encoded_past, prefix

    @torch.no_grad()
    def generate(self, timeline: List[Dict[str, Any]], NFE: int = 32, cfg_scale: float = 3.0) -> List[Dict[str, Any]]:
        self.model.eval()
        self.local_encoder.eval()

        if not timeline:
            raise ValueError("Timeline is empty!")

        g = self.geom
        total_steps = len(timeline)
        if self.verbose:
            print(f"\n--- Starting Generation over {total_steps} steps (NFE={NFE}, CFG={cfg_scale}) ---")

        for t in tqdm(range(total_steps), desc="Solving ODE", disable=not self.verbose):
            if g.mode == "autoregressive":
                encoded_past, prefix = self._ar_past(timeline, t)
            else:
                encoded_past, prefix = self._atoms_past(timeline, t)

            context = timeline[t]["context_embedding"].to(self.device)
            if context.dim() == 1:
                context = context.unsqueeze(0)

            structure = timeline[t].get("structure_embedding")
            if structure is not None:
                structure = structure.to(self.device)
                if structure.dim() == 1:
                    structure = structure.unsqueeze(0)

            if self.spectral_mode:
                x0 = torch.randn(1, self.spectral_frames, 385, device=self.device)
            else:
                x0 = torch.randn(1, g.unit_frames, g.frame_dim, device=self.device)

            pred = self.model.generate(
                x0=x0,
                encoded_past=encoded_past,
                clap_context=context,
                structure_vector=structure,
                max_nfe=NFE,
                cfg_scale=cfg_scale,
                prefix=prefix,
            )

            if self.spectral_mode:
                latent_spec = pred[:, :, :384]
                latent_time = spectral_to_time(latent_spec, g.unit_frames).to(self.device)
                scale_spec = pred[:, :, 384:]
                scale_val = scale_spec.mean(dim=1, keepdim=True)
                scale_time = scale_val.expand(-1, g.unit_frames, -1)
                pred_time = torch.cat([latent_time, scale_time], dim=-1)
                timeline[t]["atom_generated"] = pred_time.transpose(1, 2)
            else:
                timeline[t]["atom_generated"] = pred.transpose(1, 2)

        if self.verbose:
            print("✅ Generation Complete!")
        return timeline

    # Atoms: overlap-add (with V1's scale smoothing iff the codec has a scale). AR: the streaming decoder, once per block.
    @torch.no_grad()
    def decode_timeline(self, timeline: List[Dict[str, Any]], output_path: str = None, method: str = "ola_smooth"):
        if not timeline:
            raise ValueError("Timeline is empty!")
        if method != "ola_smooth":
            raise ValueError(f"Method '{method}' removed/unsupported. Please use 'ola_smooth'.")
        g = with_overlap_mode(self.geom, self.info, getattr(self, "overlap_mode", "normal"))
        seed = getattr(self, "decode_seed", 0)

        units, generated = [], []
        for step in timeline:
            atom = step.get("atom_generated")
            is_generated = atom is not None
            if atom is None:
                atom = step.get("atom_given")
            if atom is None:
                atom = torch.zeros(1, g.frame_dim, g.unit_frames, device=self.device)
            units.append(atom.to(self.device))
            generated.append(is_generated)

        if g.mode == "autoregressive":
            renderer = StreamRenderer(self.codec)
            for atom in units:
                renderer.push(self._unstandardise(atom[:, :g.latent_dim, :]))
            audio = renderer.finish()
        else:
            if self.verbose:
                print("\n--- Rendering Audio Timeline (Smooth Mode) ---")
            audio = render_atoms(self.codec, units, g, self.mu, self.std, base_seed=seed, smooth=generated)
        final_audio_tensor = audio.cpu()

        if output_path:
            sf.write(str(output_path), final_audio_tensor.transpose(0, 1).numpy(), self.sr)
            if self.verbose:
                print(f"✅ Audio rendered and saved to: {output_path}")
        return final_audio_tensor

    def _unstandardise(self, z):
        if self.mu is None:
            return z
        return z * self.std.view(1, -1, 1).to(z.device) + self.mu.view(1, -1, 1).to(z.device)
