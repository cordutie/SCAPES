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


ATOM_TYPES = ("frames", "consistent", "independent")
WINDOWS = ("hann", "equal_power")   # the only two windows (Esteban, 2026-10-30)


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
    unit_hop_frames: int      # H (frames atoms) | B (AR) | 0 (new types)
    memory_units: int         # M (atoms) | 1 (AR)
    memory_frames: int        # F (atoms) | W (AR)
    stride_frames: int        # hop_atoms * H (frames atoms) | S (AR) | 0 (new types, see stride_ticks)
    crossfade_samples: int    # X (atoms) | 0 (AR)
    prefix_frames: int        # o (AR) | 0 (atoms)
    decode: str               # "ola" (atoms) | "stream" (AR)
    context_seconds: float    # CLAP window
    has_scale: bool           # the codec normalises its input
    frame_dim: int            # latent_dim + has_scale
    encoding: str             # "whole" | f"atoms-f{F}-h{H}" (legacy) | f"atoms-c{C}-h{H}" (new types)
    atom_type: str = "frames"          # "frames" (legacy atoms and AR) | "consistent" | "independent"
    atom_samples: int = 0              # C: F*h (frames atoms), B*h (AR), the chunk (new types)
    hop_samples: int = 0               # H: H_f*h (frames atoms), B*h (AR), the hop (new types)
    window: Optional[str] = None       # "hann" | "equal_power"; legacy: always None (by-codec default via window_for)
    overlap: Optional[str] = None      # independent only: "normal" | "strong" | "brute"
    hop_atoms: int = 1                 # dataset.hop_atoms

    # Samples of one tick: the unit in which atoms are placed (a codec frame for legacy atoms, the hop for new types).
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
    def memory_seconds(self) -> float:
        return (self.atom_samples + (self.memory_units - 1) * self.hop_samples) / self.sample_rate

    @property
    def past_frames(self) -> int:
        """Frames from the start of the past to the start of the target."""
        if self.atom_type != "frames":
            raise ValueError("past_frames is frame-based; use past_ticks")
        return self.memory_units * self.unit_hop_frames if self.mode == "atoms" else self.memory_frames

    # Frame-level lines, as written to inference.gin by training (read back by geometry_from_inference_gin).
    def to_gin_lines(self) -> list[str]:
        lines = [f'codec.name = "{self.codec}"', f'generation.mode = "{self.mode}"']
        if self.mode == "atoms" and self.atom_type != "frames":
            lines += [f'atoms.type = "{self.atom_type}"', f"atoms.frames = {self.unit_frames}",
                      f"atoms.chunk_samples = {self.atom_samples}", f"atoms.hop_samples = {self.hop_samples}",
                      f"atoms.crossfade_samples = {self.crossfade_samples}"]
            if self.window is not None:
                lines.append(f'atoms.window = "{self.window}"')
            if self.overlap is not None:
                lines.append(f'atoms.overlap = "{self.overlap}"')
            lines += [f"dataset.memory_buffer_atoms = {self.memory_units}", f"dataset.hop_atoms = {self.hop_atoms}"]
        elif self.mode == "atoms":
            lines += [f"atoms.frames = {self.unit_frames}", f"atoms.hop_frames = {self.unit_hop_frames}",
                      f"atoms.crossfade_samples = {self.crossfade_samples}",
                      f"dataset.memory_buffer_atoms = {self.memory_units}"]
        else:
            lines += [f"ar.block_frames = {self.unit_frames}", f"ar.memory_frames = {self.memory_frames}",
                      f"ar.stride_frames = {self.stride_frames}", f"ar.prefix_frames = {self.prefix_frames}"]
        lines.append(f"dataset.context_seconds = {self.context_seconds}")
        return lines


# Legacy geometries (frames atoms and AR): sizes in codec frames. New types: sizes in samples (see _make_new_geometry).
def _make_geometry(info, mode, unit, hop, memory_units, memory_frames, stride, crossfade, prefix, context_seconds,
                   hop_atoms=1):
    has_scale = bool(info.normalizes_input)
    encoding = f"atoms-f{unit}-h{hop}" if has_scale else "whole"
    return Geometry(
        mode=mode, codec=info.name, sample_rate=info.sample_rate, hop_length=info.hop_length,
        frame_rate=info.frame_rate, latent_dim=info.latent_dim, frame_quantum=info.frame_quantum,
        unit_frames=unit, unit_hop_frames=hop, memory_units=memory_units, memory_frames=memory_frames,
        stride_frames=stride, crossfade_samples=crossfade, prefix_frames=prefix,
        decode="ola" if mode == "atoms" else "stream", context_seconds=context_seconds, has_scale=has_scale,
        frame_dim=info.latent_dim + int(has_scale), encoding=encoding,
        atom_type="frames", atom_samples=unit * info.hop_length, hop_samples=hop * info.hop_length, hop_atoms=hop_atoms,
    )


# New atom types: unit_hop_frames and stride_frames are 0 (sentinels: the hop is not a whole number of frames).
def _make_new_geometry(info, atom_type, F, H, X, M, window, overlap, hop_atoms, context_seconds):
    C = F * info.hop_length
    return Geometry(
        mode="atoms", codec=info.name, sample_rate=info.sample_rate, hop_length=info.hop_length,
        frame_rate=info.frame_rate, latent_dim=info.latent_dim, frame_quantum=info.frame_quantum,
        unit_frames=F, unit_hop_frames=0, memory_units=M, memory_frames=F, stride_frames=0, crossfade_samples=X,
        prefix_frames=0, decode="ola", context_seconds=context_seconds, has_scale=bool(info.normalizes_input),
        frame_dim=info.latent_dim + int(bool(info.normalizes_input)), encoding=f"atoms-c{C}-h{H}",
        atom_type=atom_type, atom_samples=C, hop_samples=H, window=window, overlap=overlap, hop_atoms=hop_atoms,
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


# Constraints of the new types, in samples: 0 < H < C and 0 < X <= min(H, C - H).
def _check_new_atoms(C, H, X):
    if not 0 < H < C:
        raise ConfigError(f"atoms hop of {H} samples must be in (0, {C}) (the chunk)")
    if not 0 < X <= min(H, C - H):
        raise ConfigError(f"crossfade of {X} samples must be in (0, {min(H, C - H)}] (min of hop and chunk - hop)")


def _check_window(window):
    if window is not None and window not in WINDOWS:
        raise ConfigError(f"atoms.window = {window!r} is invalid, choose from {list(WINDOWS)}")


# Halves of the crossfade of n samples. hann: periodic Hann of 2n split in two (rise + fall = 1, amplitudes add up, right
# for coherent audio). equal_power: their square roots (rise^2 + fall^2 = 1, powers add up, right for incoherent audio such
# as two noisy decodes of the same latents). float32 [n] each, cpu.
def crossfade_halves(n: int, window: str = "hann"):
    if window not in WINDOWS:
        raise ConfigError(f"window {window!r} is invalid, choose from {list(WINDOWS)}")
    if n == 0:
        return torch.zeros(0), torch.zeros(0)
    hann = torch.hann_window(2 * n)
    rise, fall = hann[:n], hann[n:]
    if window == "equal_power":
        rise, fall = rise.sqrt(), fall.sqrt()
    return rise, fall


def window_for(geom: Geometry, info) -> str:
    """Window of the overlap-add: the configured one, else equal power for a stochastic decoder (two decodes of the same
    latents are not coherent, so their powers add) and Hann (amplitudes add to 1) for a deterministic one."""
    if geom.window is not None:
        return geom.window
    return "equal_power" if info.stochastic_decoder else "hann"


def resolve_geometry(cfg: DataprepConfig, info) -> Geometry:
    f, q, r = info.frame_rate, info.frame_quantum, info.sample_rate
    if cfg.generation_mode == "atoms":
        if info.causal:
            warnings.warn(f"atoms mode with the causal codec {info.name}: allowed for comparisons only", UserWarning)
        M = int(cfg.memory_buffer_atoms)
        if M < 1:
            raise ConfigError("dataset.memory_buffer_atoms must be >= 1")
        if cfg.atoms_type is None:
            if cfg.atoms_overlap is not None or cfg.atoms_window is not None:
                raise ConfigError("atoms.overlap and atoms.window are only for atoms.type consistent | independent; "
                                  "legacy atoms (EnCodec 48k) keep their window")
            F = seconds_to_frames(cfg.atoms_seconds, f, q)
            H = seconds_to_frames(cfg.atoms_hop_seconds, f, q)
            X = seconds_to_samples(cfg.atoms_crossfade_seconds or 0.0, r)
            _check_atoms(info, F, H, X)
            return _make_geometry(info, "atoms", F, H, M, F, int(cfg.hop_atoms) * H, X, 0, cfg.semantic_context_seconds,
                                  hop_atoms=int(cfg.hop_atoms))
        if cfg.atoms_type not in ("consistent", "independent"):
            raise ConfigError(f"atoms.type = {cfg.atoms_type!r} is invalid, choose from ['consistent', 'independent']")
        if cfg.atoms_type == "independent":
            raise ConfigError("independent atoms are phase 2 (postponed by Esteban, 2026-10-30)")
        _check_window(cfg.atoms_window)
        if cfg.atoms_overlap is not None:
            raise ConfigError("atoms.overlap is only for atoms.type = \"independent\"")
        if cfg.atoms_crossfade_seconds is None or cfg.atoms_crossfade_seconds <= 0:
            raise ConfigError("consistent atoms need atoms.crossfade_seconds > 0")
        F = seconds_to_frames(cfg.atoms_seconds, f, q)
        C = F * info.hop_length
        H = seconds_to_samples(cfg.atoms_hop_seconds, r)
        X = seconds_to_samples(cfg.atoms_crossfade_seconds, r)
        _check_new_atoms(C, H, X)
        return _make_new_geometry(info, "consistent", F, H, X, M, cfg.atoms_window, None, int(cfg.hop_atoms),
                                  cfg.semantic_context_seconds)
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


# From a parsed inference.gin (written by training). V1 files work with codec.name and generation.mode added.
def geometry_from_inference_gin(raw: dict, info) -> Geometry:
    def get(*keys, default=None):
        return _get_nested(raw, list(keys), default)

    mode = get("generation", "mode")
    if mode is None:
        raise ConfigError("inference.gin has no generation.mode (V1 files need the two lines codec.name and generation.mode)")
    ctx = get("dataset", "context_seconds", default=get("semantic", "context_seconds", default=1.0))
    if mode == "atoms":
        M = int(get("dataset", "memory_buffer_atoms", default=1))
        hop_atoms = int(get("dataset", "hop_atoms", default=1))
        atom_type = get("atoms", "type")
        if atom_type is not None:
            if atom_type not in ("consistent", "independent"):
                raise ConfigError(f"atoms.type = {atom_type!r} is invalid, choose from ['consistent', 'independent']")
            if atom_type == "independent":
                raise ConfigError("independent atoms are phase 2 (postponed by Esteban, 2026-10-30)")
            F, C = int(get("atoms", "frames")), int(get("atoms", "chunk_samples"))
            H, X = int(get("atoms", "hop_samples")), int(get("atoms", "crossfade_samples"))
            if C != F * info.hop_length:
                raise ConfigError(f"atoms.chunk_samples = {C} is not atoms.frames * hop_length = {F * info.hop_length}")
            _check_new_atoms(C, H, X)
            window = get("atoms", "window")
            _check_window(window)
            return _make_new_geometry(info, atom_type, F, H, X, M, window, get("atoms", "overlap"), hop_atoms, ctx)
        # an inference.gin of the legacy frame atoms holds only these atoms keys; any other one is from a removed V2 layout
        legacy_keys = {"frames", "hop_frames", "crossfade_samples", "crossfade_frames"}
        extra = sorted(set(raw.get("atoms", {})) - legacy_keys)
        if extra:
            raise ConfigError(f"inference.gin has atoms keys {extra} that no atoms layout uses (old V2 checkpoints): "
                              "use atoms.type and atoms.window")
        F, H = int(get("atoms", "frames")), int(get("atoms", "hop_frames"))
        X = get("atoms", "crossfade_samples")
        if X is None:
            X = int(get("atoms", "crossfade_frames", default=0)) * info.hop_length
        _check_atoms(info, F, H, int(X))
        return _make_geometry(info, "atoms", F, H, M, F, hop_atoms * H, int(X), 0, ctx, hop_atoms=hop_atoms)
    if mode == "autoregressive":
        B, W = int(get("ar", "block_frames")), int(get("ar", "memory_frames"))
        S = int(get("ar", "stride_frames", default=B))
        o = int(get("ar", "prefix_frames", default=0))
        return _make_geometry(info, "autoregressive", B, B, 1, W, S, 0, o, ctx)
    raise ConfigError(f"generation.mode = {mode!r} is invalid")


# Overlap-add window of one atom [C]: zeros(C - H - X), rise (X), ones (H - X), fall (X), as V1's _build_ola_window.
# EnCodec 48k keeps Hann because its decoder is deterministic, so two decodes of the same moment are coherent and amplitudes
# add; confirmed by Esteban 2026-10-30; to be checked in the window study of the independent-atoms work (A1).
def ola_window(geom: Geometry, window: str = "hann") -> torch.Tensor:
    C, H, X = geom.atom_samples, geom.hop_samples, geom.crossfade_samples
    rise, fall = crossfade_halves(X, window)
    return torch.cat([torch.zeros(C - H - X), rise, torch.ones(H - X), fall])


def representation_key(geom: Geometry) -> str:
    return f"{geom.codec}__{geom.encoding}"
