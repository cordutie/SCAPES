"""Generation geometry in seconds, resolved to frames with the codec's rate and quantum (design: Geometry in seconds)."""
import math
import warnings
from dataclasses import dataclass
from typing import Optional

import torch

from .config_loader import ConfigError, DataprepConfig, _get_nested


# Seconds -> frames, rounded up to the codec quantum (the 1e-6 absorbs float error: 0.07 * 100 = 7.000000000000001).
def seconds_to_frames(seconds: float, frame_rate: float, quantum: int) -> int:
    return quantum * max(1, math.ceil(seconds * frame_rate / quantum - 1e-6))


# Seconds -> samples, rounded up; 0 stays 0.
def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    return max(0, math.ceil(seconds * sample_rate - 1e-6))


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
    unit_hop_frames: int      # H (atoms) | B (AR)
    memory_units: int         # M (atoms) | 1 (AR)
    memory_frames: int        # F (atoms) | W (AR)
    stride_frames: int        # hop_atoms * H (atoms) | S (AR)
    crossfade_samples: int    # X (atoms) | 0 (AR)
    prefix_frames: int        # o (AR) | 0 (atoms)
    decode: str               # "ola" (atoms) | "stream" (AR)
    context_seconds: float    # CLAP window
    has_scale: bool           # the codec normalises its input
    frame_dim: int            # latent_dim + has_scale
    encoding: str             # "whole" | f"atoms-f{F}-h{H}"
    overlap_mode: Optional[str] = None     # atoms of stochastic decoders: normal | strong | brute (the hop follows from it)
    crossfade_law: Optional[str] = None    # "hann" | "equal_power"; None = by codec (crossfade_law_for)

    @property
    def past_frames(self) -> int:
        """Frames from the start of the past to the start of the target."""
        return self.memory_units * self.unit_hop_frames if self.mode == "atoms" else self.memory_frames

    # Frame-level lines, as written to inference.gin by training (read back by geometry_from_inference_gin).
    def to_gin_lines(self) -> list[str]:
        lines = [f'codec.name = "{self.codec}"', f'generation.mode = "{self.mode}"']
        if self.mode == "atoms":
            lines += [f"atoms.frames = {self.unit_frames}", f"atoms.hop_frames = {self.unit_hop_frames}",
                      f"atoms.crossfade_samples = {self.crossfade_samples}",
                      f"dataset.memory_buffer_atoms = {self.memory_units}"]
            if self.overlap_mode is not None:
                lines.append(f'atoms.overlap_mode = "{self.overlap_mode}"')
            if self.crossfade_law is not None:
                lines.append(f'atoms.crossfade_law = "{self.crossfade_law}"')
        else:
            lines += [f"ar.block_frames = {self.unit_frames}", f"ar.memory_frames = {self.memory_frames}",
                      f"ar.stride_frames = {self.stride_frames}", f"ar.prefix_frames = {self.prefix_frames}"]
        lines.append(f"dataset.context_seconds = {self.context_seconds}")
        return lines


def _make_geometry(info, mode, unit, hop, memory_units, memory_frames, stride, crossfade, prefix, context_seconds,
                   overlap_mode=None, crossfade_law=None):
    has_scale = bool(info.normalizes_input)
    encoding = f"atoms-f{unit}-h{hop}" if has_scale else "whole"
    return Geometry(
        mode=mode, codec=info.name, sample_rate=info.sample_rate, hop_length=info.hop_length,
        frame_rate=info.frame_rate, latent_dim=info.latent_dim, frame_quantum=info.frame_quantum,
        unit_frames=unit, unit_hop_frames=hop, memory_units=memory_units, memory_frames=memory_frames,
        stride_frames=stride, crossfade_samples=crossfade, prefix_frames=prefix,
        decode="ola" if mode == "atoms" else "stream", context_seconds=context_seconds, has_scale=has_scale,
        frame_dim=info.latent_dim + int(has_scale), encoding=encoding, overlap_mode=overlap_mode,
        crossfade_law=crossfade_law,
    )


def _check_atoms(info, F, H, X):
    h = info.hop_length
    if H > F:
        raise ConfigError(f"atoms hop ({H} frames) is longer than the atom ({F} frames)")
    limit = min(H * h, (F - H) * h)
    if not 0 <= X <= limit:
        raise ConfigError(
            f"crossfade of {X} samples must be in [0, {limit}] (min of hop and overlap, {H=} {F=} frames of {h} samples)"
            + ("; with F == H no crossfade is possible, use a longer atoms.seconds" if F == H and X > 0 else "")
        )


# ─── Overlap modes (codecs with a stochastic decoder: CoDiCodec, Music2Latent) ───
# The decoded atoms of such a codec are not continuous with each other (the same latents decode to different waveforms), so
# the overlap with the previous atom is only there to crossfade, and the hop follows from the mode: the overlap is a fraction
# of the atom, rounded to the codec quantum and never below one quantum. normal: one quantum, short fade (atoms.crossfade_seconds,
# default 0.05 s); strong: 25 % of the atom, crossfaded over the whole overlap; brute: 50 %, crossfaded over the whole overlap.
OVERLAP_FRACTIONS = {"normal": 0.0, "strong": 0.25, "brute": 0.5}
DEFAULT_NORMAL_FADE_SECONDS = 0.05
CROSSFADE_LAWS = ("hann", "equal_power")


def _overlap_geometry(info, cfg):
    mode = cfg.atoms_overlap_mode
    if mode not in OVERLAP_FRACTIONS:
        raise ConfigError(f"atoms.overlap_mode = {mode!r} is invalid, choose from {list(OVERLAP_FRACTIONS)}")
    if not info.stochastic_decoder:
        raise ConfigError(f"atoms.overlap_mode is for codecs with a stochastic decoder (codicodec, music2latent); "
                          f"{info.name} has a deterministic one: set atoms.hop_seconds and atoms.crossfade_seconds")
    if cfg.atoms_hop_seconds is not None:
        raise ConfigError("atoms.hop_seconds is derived from atoms.overlap_mode: remove it")
    q, h = info.frame_quantum, info.hop_length
    F = seconds_to_frames(cfg.atoms_seconds, info.frame_rate, q)
    O = max(q, min(q * round(OVERLAP_FRACTIONS[mode] * F / q), q * (F // (2 * q))))   # the overlap never exceeds the hop
    H = F - O
    if H < q:
        raise ConfigError(f"overlap_mode {mode!r}: an atom of {F} frames leaves no hop after an overlap of {O} frames; "
                          f"use a longer atoms.seconds")
    if mode == "normal":
        fade = DEFAULT_NORMAL_FADE_SECONDS if cfg.atoms_crossfade_seconds is None else cfg.atoms_crossfade_seconds
        X = seconds_to_samples(fade, info.sample_rate)
    else:
        if cfg.atoms_crossfade_seconds not in (None, 0.0):
            raise ConfigError(f"overlap_mode {mode!r} crossfades the whole overlap: remove atoms.crossfade_seconds")
        X = O * h
    return F, H, X


def crossfade_law_for(geom: Geometry, info) -> str:
    """Law of the overlap-add window: the configured one, else equal power for a stochastic decoder (two decodes of the same
    latents are not coherent, so their powers add) and Hann (amplitudes add to 1) for a deterministic one."""
    if geom.crossfade_law is not None:
        return geom.crossfade_law
    return "equal_power" if info.stochastic_decoder else "hann"


def resolve_geometry(cfg: DataprepConfig, info) -> Geometry:
    f, q, r = info.frame_rate, info.frame_quantum, info.sample_rate
    if cfg.generation_mode == "atoms":
        if info.causal:
            warnings.warn(f"atoms mode with the causal codec {info.name}: allowed for comparisons only", UserWarning)
        if cfg.atoms_overlap_mode is not None:
            F, H, X = _overlap_geometry(info, cfg)
        else:
            F = seconds_to_frames(cfg.atoms_seconds, f, q)
            H = seconds_to_frames(cfg.atoms_hop_seconds, f, q)
            X = seconds_to_samples(cfg.atoms_crossfade_seconds or 0.0, r)
        _check_atoms(info, F, H, X)
        law = cfg.atoms_crossfade_law
        if law is not None and law not in CROSSFADE_LAWS:
            raise ConfigError(f"atoms.crossfade_law = {law!r} is invalid, choose from {list(CROSSFADE_LAWS)}")
        M = int(cfg.memory_buffer_atoms)
        if M < 1:
            raise ConfigError("dataset.memory_buffer_atoms must be >= 1")
        return _make_geometry(info, "atoms", F, H, M, F, int(cfg.hop_atoms) * H, X, 0, cfg.semantic_context_seconds,
                              overlap_mode=cfg.atoms_overlap_mode, crossfade_law=law)
    if cfg.generation_mode == "autoregressive":
        if not info.causal:
            raise ConfigError(f"autoregressive mode needs a causal codec; {info.name} is not causal")
        B = seconds_to_frames(cfg.ar_block_seconds, f, q)
        W = seconds_to_frames(cfg.ar_memory_seconds, f, q)
        stride_s = cfg.ar_stride_seconds if cfg.ar_stride_seconds is not None else cfg.ar_block_seconds
        S = seconds_to_frames(stride_s, f, q)
        o = seconds_to_frames(cfg.ar_prefix_seconds, f, q) if cfg.ar_prefix_seconds > 0 else 0
        if W < B:
            raise ConfigError(f"ar.memory_seconds ({W} frames) must cover the block ({B} frames)")
        if o > W:
            raise ConfigError(f"ar.prefix_seconds ({o} frames) must not exceed the memory window ({W} frames)")
        return _make_geometry(info, "autoregressive", B, B, 1, W, S, 0, o, cfg.semantic_context_seconds)
    raise ConfigError(f"generation.mode = {cfg.generation_mode!r} is invalid, choose from ['atoms', 'autoregressive']")


# From a parsed inference.gin (written by training, in frames). V1 files work with codec.name and generation.mode added.
def geometry_from_inference_gin(raw: dict, info) -> Geometry:
    def get(*keys, default=None):
        return _get_nested(raw, list(keys), default)

    mode = get("generation", "mode")
    if mode is None:
        raise ConfigError("inference.gin has no generation.mode (V1 files need the two lines codec.name and generation.mode)")
    ctx = get("dataset", "context_seconds", default=get("semantic", "context_seconds", default=1.0))
    if mode == "atoms":
        F, H = int(get("atoms", "frames")), int(get("atoms", "hop_frames"))
        X = get("atoms", "crossfade_samples")
        if X is None:
            X = int(get("atoms", "crossfade_frames", default=0)) * info.hop_length
        M = int(get("dataset", "memory_buffer_atoms", default=1))
        _check_atoms(info, F, H, int(X))
        return _make_geometry(info, "atoms", F, H, M, F, int(get("dataset", "hop_atoms", default=1)) * H, int(X), 0, ctx,
                              overlap_mode=get("atoms", "overlap_mode"), crossfade_law=get("atoms", "crossfade_law"))
    if mode == "autoregressive":
        B, W = int(get("ar", "block_frames")), int(get("ar", "memory_frames"))
        S = int(get("ar", "stride_frames", default=B))
        o = int(get("ar", "prefix_frames", default=0))
        return _make_geometry(info, "autoregressive", B, B, 1, W, S, 0, o, ctx)
    raise ConfigError(f"generation.mode = {mode!r} is invalid")


# Overlap-add window of one atom: zeros, hann rise (X), ones, hann fall (X), as V1's _build_ola_window.
# law "hann": rise + fall = 1 (amplitudes add up, right for coherent overlapping audio). law "equal_power": the square root of
# those, so rise^2 + fall^2 = 1 (powers add up, right for incoherent audio such as two noisy decodes of the same latents).
def ola_window(geom: Geometry, law: str = "hann") -> torch.Tensor:
    h, F, H, X = geom.hop_length, geom.unit_frames, geom.unit_hop_frames, geom.crossfade_samples
    if law not in CROSSFADE_LAWS:
        raise ConfigError(f"crossfade law {law!r} is invalid, choose from {list(CROSSFADE_LAWS)}")
    if X > 0:
        hann = torch.hann_window(2 * X)
        rise, fall = hann[:X], hann[X:]
        if law == "equal_power":
            rise, fall = rise.sqrt(), fall.sqrt()
    else:
        rise = fall = torch.zeros(0)
    return torch.cat([torch.zeros((F - H) * h - X), rise, torch.ones(H * h - X), fall])


def representation_key(geom: Geometry) -> str:
    return f"{geom.codec}__{geom.encoding}"
