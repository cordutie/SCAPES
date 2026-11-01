"""Generation geometry in seconds, resolved to frames with the codec's rate and quantum (design: Geometry in seconds)."""
import math
import warnings
from dataclasses import dataclass
from typing import Optional
import torch

from SCAPES.codecs import canonical, get_codec_info
from SCAPES.generation import Resolution, resolve_generation
from .config_loader import ConfigError, DataprepConfig, _check_rejected_keys, _flatten_keys, _get_nested


# Seconds -> frames, rounded up to the codec quantum (the 1e-6 absorbs float error: 0.07 * 100 = 7.000000000000001).
def seconds_to_frames(seconds: float, frame_rate: float, quantum: int) -> int:
    return quantum * max(1, math.ceil(seconds * frame_rate / quantum - 1e-6))


# Seconds -> samples, rounded up; 0 stays 0.
def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    return max(0, math.ceil(seconds * sample_rate - 1e-6))


ATOM_TYPES = ("frames", "atoms")   # "frames" is the AR layout; "atoms" is the atoms layout


# Number of atoms of C samples placed H samples apart that fit in num_samples (0 if the audio is shorter than one atom).
def atom_count(num_samples: int, chunk: int, hop: int) -> int:
    if num_samples < chunk:
        return 0
    return 1 + (num_samples - chunk) // hop


@dataclass(frozen=True)
class Geometry:
    mode: str                 # "atoms" | "autoregressive"
    codec: str
    sample_rate: int
    hop_length: int
    frame_rate: float
    latent_dim: int
    frame_quantum: int
    unit_frames: int          # F (atoms) | B (AR)
    unit_hop_frames: int      # B (AR) | 0 (atoms)
    memory_units: int         # M (atoms) | 1 (AR)
    memory_frames: int        # F (atoms) | W (AR)
    stride_frames: int        # S (AR) | 0 (atoms, see stride_ticks)
    crossfade_samples: int    # X (atoms) | 0 (AR)
    prefix_frames: int        # o (AR) | 0 (atoms)
    decode: str               # "ola" (atoms) | "stream" (AR)
    context_seconds: float    # CLAP window
    has_scale: bool           # the codec normalises its input
    frame_dim: int            # latent_dim + has_scale
    encoding: str             # "whole" | f"atoms-f{F}-h{H}" (AR) | f"atoms-c{C}-h{H}" (atoms)
    atom_type: str = "frames"          # "frames" (AR) | "atoms" (atoms)
    atom_samples: int = 0              # C: B*h (AR), the chunk (atoms)
    hop_samples: int = 0               # H: B*h (AR), the hop (atoms)
    hop_atoms: int = 1                 # dataset.hop_atoms
    resolution: Optional[Resolution] = None   # the resolved three keys (generation.py)

    # Samples of one tick: the unit in which atoms are placed (a codec frame for AR, the hop for atoms).
    @property
    def tick_samples(self) -> int:
        return self.hop_length if self.atom_type == "frames" else self.hop_samples

    @property
    def hop_ticks(self) -> int:
        return self.unit_hop_frames if self.atom_type == "frames" else 1

    @property
    def stride_ticks(self) -> int:
        return self.stride_frames if self.atom_type == "frames" else self.hop_atoms

    @property
    def past_ticks(self) -> int:
        return self.past_frames if self.atom_type == "frames" else self.memory_units

    @property
    def stride_samples(self) -> int:
        return self.stride_ticks * self.tick_samples

    # Seconds of audio covered by the past of an atoms model (the first atom plus M - 1 hops).
    @property
    def buffer_span_seconds(self) -> float:
        return (self.atom_samples + (self.memory_units - 1) * self.hop_samples) / self.sample_rate

    @property
    def past_frames(self) -> int:
        """Frames from the start of the past to the start of the target."""
        if self.atom_type != "frames":
            raise ValueError("past_frames is frame-based; use past_ticks")
        return self.memory_units * self.unit_hop_frames if self.mode == "atoms" else self.memory_frames

    # Lines written to inference.gin by training (read back by geometry_from_inference_gin): the three keys as requested,
    # the resolved integers (checked against the codec table when read) and comments with the seconds.
    def to_gin_lines(self) -> list[str]:
        res = self.resolution
        lines = [f'codec = "{self.codec}"', f"conditioning_rate = {res.requested_rate}",
                 f"conditioning_memory_seconds = {res.requested_memory_seconds}",
                 f'resolved.mode = "{res.mode}"', f"resolved.new_latents = {res.n}"]
        if res.mode == "atoms":
            lines += [f"resolved.crossfade_latents = {res.crossfade}", f"resolved.overlapped_latents = {res.overlapped}",
                      f"resolved.atom_latents = {res.atom}", f"resolved.memory_atoms = {res.memory_atoms}"]
        else:
            lines.append(f"resolved.memory_latents = {res.memory_window}")
        lines += [f"resolved.attended_latents = {res.attended}", f"dataset.context_seconds = {self.context_seconds}",
                  f"dataset.hop_atoms = {self.hop_atoms}",
                  f"# step {res.hop_seconds:.4f} s ({res.rate:.2f} Hz), {res.n} new latents"]
        if res.mode == "atoms":
            lines.append(f"# atom {res.atom_seconds:.4f} s, crossfade {res.crossfade_seconds:.4f} s, "
                         f"memory {res.memory_atoms} atoms")
        else:
            lines.append(f"# memory window {res.memory_window_seconds:.4f} s")
        return lines


# AR geometry: sizes in codec frames. Atoms: sizes in samples (see _make_atoms_geometry).
def _make_geometry(info, res, unit, hop, memory_units, memory_frames, stride, crossfade, prefix, context_seconds,
                   hop_atoms=1):
    mode = res.mode
    has_scale = bool(info.normalizes_input)
    encoding = f"atoms-f{unit}-h{hop}" if has_scale else "whole"
    return Geometry(
        mode=mode, codec=res.codec, sample_rate=info.sample_rate, hop_length=info.hop_length,
        frame_rate=info.frame_rate, latent_dim=info.latent_dim, frame_quantum=info.frame_quantum,
        unit_frames=unit, unit_hop_frames=hop, memory_units=memory_units, memory_frames=memory_frames,
        stride_frames=stride, crossfade_samples=crossfade, prefix_frames=prefix,
        decode="ola" if mode == "atoms" else "stream", context_seconds=context_seconds, has_scale=has_scale,
        frame_dim=info.latent_dim + int(has_scale), encoding=encoding,
        atom_type="frames", atom_samples=unit * info.hop_length, hop_samples=hop * info.hop_length, hop_atoms=hop_atoms,
        resolution=res,
    )


# Atoms: unit_hop_frames and stride_frames are 0 (sentinels: the hop is not a whole number of frames).
def _make_atoms_geometry(info, res, F, H, X, M, hop_atoms, context_seconds):
    C = F * info.hop_length
    return Geometry(
        mode="atoms", codec=res.codec, sample_rate=info.sample_rate, hop_length=info.hop_length,
        frame_rate=info.frame_rate, latent_dim=info.latent_dim, frame_quantum=info.frame_quantum,
        unit_frames=F, unit_hop_frames=0, memory_units=M, memory_frames=F, stride_frames=0, crossfade_samples=X,
        prefix_frames=0, decode="ola", context_seconds=context_seconds, has_scale=bool(info.normalizes_input),
        frame_dim=info.latent_dim + int(bool(info.normalizes_input)), encoding=f"atoms-c{C}-h{H}",
        atom_type="atoms", atom_samples=C, hop_samples=H, hop_atoms=hop_atoms, resolution=res,
    )


# The one crossfade window of SCAPES (equal power), see crossfade_window.
def crossfade_window(geom) -> torch.Tensor:
    """The one crossfade window of SCAPES, used by every renderer and tool. To try another window, change only this function.

    Layout of one atom [C]: zeros(C - H - X), rise (X), ones(H - X), fall (X), with C = geom.atom_samples,
    H = geom.hop_samples, X = geom.crossfade_samples. Equal power: rise^2 + fall^2 = 1 (the square root of the periodic
    Hann halves), right for two noisy decodes of the same latents, whose powers add. float32 [C], cpu."""
    C, H, X = geom.atom_samples, geom.hop_samples, geom.crossfade_samples
    angle = math.pi * torch.arange(X, dtype=torch.float64) / (2 * X) if X > 0 else torch.zeros(0, dtype=torch.float64)
    rise, fall = torch.sin(angle).float(), torch.cos(angle).float()
    return torch.cat([torch.zeros(C - H - X), rise, torch.ones(H - X), fall])


def resolve_geometry(cfg: DataprepConfig, info=None) -> Geometry:
    if info is None:
        info = get_codec_info(cfg.codec_name)
    res = resolve_generation(cfg.codec_name, cfg.conditioning_rate, cfg.conditioning_memory_seconds)
    return _geometry_from_resolution(info, res, int(cfg.hop_atoms), cfg.semantic_context_seconds)


# The geometry of a resolution: atoms in samples (hop n latents, crossfade x latents), AR blocks in frames.
def _geometry_from_resolution(info, res, hop_atoms, context_seconds) -> Geometry:
    h = info.hop_length
    if res.mode == "atoms":
        if info.causal:
            warnings.warn(f"atoms mode with the causal codec {info.name}: allowed for comparisons only", UserWarning)
        return _make_atoms_geometry(info, res, res.atom, res.n * h, res.crossfade * h, res.memory_atoms, hop_atoms,
                                  context_seconds)
    return _make_geometry(info, res, res.n, res.n, 1, res.memory_window, res.n, 0, 0, context_seconds)


# From a parsed inference.gin (written by training): V3 files only. The three keys are resolved again and must give the
# integers that training wrote.
def geometry_from_inference_gin(raw: dict, info) -> Geometry:
    _check_rejected_keys(_flatten_keys(raw), "inference.gin")
    resolved = raw.get("resolved")
    if not isinstance(resolved, dict) or "codec" not in raw:
        raise ConfigError("inference.gin has no codec and resolved.* lines: it was not written by this version of SCAPES")
    codec = raw["codec"]
    if canonical(info.name) != codec:
        raise ConfigError(f"inference.gin codec {codec!r} does not match the codec {info.name!r} that was loaded")
    res = resolve_generation(codec, raw.get("conditioning_rate"), raw.get("conditioning_memory_seconds"))
    if res.mode == "atoms":
        counts = {"new_latents": res.n, "crossfade_latents": res.crossfade, "overlapped_latents": res.overlapped,
                  "atom_latents": res.atom, "memory_atoms": res.memory_atoms}
    else:
        counts = {"new_latents": res.n, "memory_latents": res.memory_window}
    counts.update(mode=res.mode, attended_latents=res.attended)
    if any(resolved.get(k) != v for k, v in counts.items()):
        raise ConfigError("this checkpoint was trained with another codec table: inference.gin has "
                          f"{ {k: resolved.get(k) for k in counts} } and the codec table gives {counts}")
    ctx = _get_nested(raw, ["dataset", "context_seconds"], 1.0)
    return _geometry_from_resolution(info, res, int(_get_nested(raw, ["dataset", "hop_atoms"], 1)), ctx)


def representation_key(geom: Geometry) -> str:
    return f"{geom.codec}__{geom.encoding}"
