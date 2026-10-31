"""Per-codec table and the resolver of the three generation keys (codec, conditioning_rate, conditioning_memory_seconds).
Pure Python: no torch, no codec import."""
import math
import sys
from dataclasses import dataclass

from SCAPES.errors import ConfigError

CAP = 300             # attended latents, every codec
DEFAULT_RATE = 10.0   # Hz, every codec


@dataclass(frozen=True)
class CodecSpec:
    name: str
    mode: str                       # "autoregressive" | "atoms"
    wrapper: str                    # audio_codec_wrappers registry name
    sample_rate: int
    hop_length: int                 # samples per latent
    crossfade_default: "int | None"
    min_atom: "int | None"
    n_min: int
    default_memory_seconds: float
    standardize_latents: bool
    past_noise: float
    user: bool = True

    @property
    def latent_rate(self) -> float:
        return self.sample_rate / self.hop_length


_AR, _AT = "autoregressive", "atoms"
CODECS = {
    "encodec24": CodecSpec("encodec24", _AR, "encodec24k", 24000, 320, None, None, 5, 1.0, True, 0.1),
    "spectrostream": CodecSpec("spectrostream", _AR, "spectrostream", 48000, 1920, None, None, 2, 1.0, True, 0.1),
    "encodec48": CodecSpec("encodec48", _AT, "encodec48k", 48000, 320, 15, 30, 8, 0.5, False, 0.0),
    "music2latent": CodecSpec("music2latent", _AT, "music2latent", 44100, 4096, 2, 6, 1, 0.5, True, 0.0),
    "toy_causal": CodecSpec("toy_causal", _AR, "toy_causal", 8000, 8, None, None, 1, 1.0, True, 0.1, user=False),
    "toy_noncausal": CodecSpec("toy_noncausal", _AT, "toy_noncausal", 8000, 8, 4, 12, 1, 0.5, True, 0.0, user=False),
}
WRAPPER_TO_CANONICAL = {spec.wrapper: name for name, spec in CODECS.items()}


def atom_geometry(spec: CodecSpec, n: int):
    """(x, o, A): crossfade, overlapped and total atom latents for n new latents per step."""
    x = min(-(-n // 2), spec.crossfade_default)
    o = max(n, spec.min_atom - x - n)
    return x, o, o + x + n


def n_max(spec: CodecSpec) -> int:
    if spec.mode == _AR:
        return CAP
    n = spec.n_min
    while atom_geometry(spec, n + 1)[2] <= CAP:
        n += 1
    return n


@dataclass(frozen=True)
class Resolution:
    codec: str
    mode: str
    latent_rate: float
    requested_rate: float
    requested_memory_seconds: float
    n: int
    rate: float
    memory_latents: int
    crossfade: int = 0
    overlapped: int = 0
    atom: int = 0
    memory_atoms: int = 0
    k_max: int = 0
    memory_window: int = 0
    attended: int = 0
    notes: tuple = ()

    @property
    def hop_seconds(self) -> float:
        return self.n / self.latent_rate

    @property
    def atom_seconds(self) -> float:
        return self.atom / self.latent_rate

    @property
    def crossfade_seconds(self) -> float:
        return self.crossfade / self.latent_rate

    @property
    def memory_window_seconds(self) -> float:
        return self.memory_window / self.latent_rate

    @property
    def covered_latents(self) -> int:
        if self.mode == _AR:
            return self.memory_window
        return self.crossfade + self.memory_atoms * self.n

    @property
    def buffer_span_latents(self) -> int:
        if self.mode == _AR:
            return self.memory_window
        return self.atom + (self.memory_atoms - 1) * self.n


def resolve_generation(codec, conditioning_rate=None, conditioning_memory_seconds=None) -> Resolution:
    if codec not in CODECS:
        users = [k for k, v in CODECS.items() if v.user]
        raise ConfigError(f"unknown codec {codec!r}, one of {users}")
    spec = CODECS[codec]
    c = DEFAULT_RATE if conditioning_rate is None else conditioning_rate
    m_s = spec.default_memory_seconds if conditioning_memory_seconds is None else conditioning_memory_seconds
    if not c > 0:
        raise ConfigError(f"conditioning_rate must be > 0, got {c}")
    if not m_s > 0:
        raise ConfigError(f"conditioning_memory_seconds must be > 0, got {m_s}")
    r = spec.latent_rate
    hi = n_max(spec)

    # nearest available rate; ties go to the smaller n
    n, best = spec.n_min, abs(r / spec.n_min - c)
    for k in range(spec.n_min + 1, hi + 1):
        d = abs(r / k - c)
        if d < best - 1e-12 * c:
            n, best = k, d
    notes = []
    if c > r / spec.n_min:
        notes.append("clipped_fast")
    elif c < r / hi:
        notes.append("clipped_slow")
    elif best > 1e-9 * c:
        notes.append("snapped")

    m_l = math.ceil(r * m_s - 1e-9)
    common = dict(codec=codec, mode=spec.mode, latent_rate=r, requested_rate=c, requested_memory_seconds=m_s,
                  n=n, rate=r / n, memory_latents=m_l)
    if spec.mode == _AT:
        x, o, A = atom_geometry(spec, n)
        K = max(1, -(-(m_l - x) // n))
        if K * A > CAP:
            K = CAP // A
            notes.append("cap")
        return Resolution(**common, crossfade=x, overlapped=o, atom=A, memory_atoms=K, k_max=CAP // A,
                          attended=K * A, notes=tuple(notes))
    W = min(m_l, CAP)
    if m_l > CAP:
        notes.append("cap")
    if W < n:
        W = n
        notes.append("memory_raised")
    return Resolution(**common, memory_window=W, attended=W, notes=tuple(notes))


def rate_table(codec: str) -> list:
    spec = CODECS[codec]
    rows = []
    for n in range(spec.n_min, n_max(spec) + 1):
        row = {"n": n, "rate_hz": spec.latent_rate / n, "step_seconds": n / spec.latent_rate}
        if spec.mode == _AT:
            x, o, A = atom_geometry(spec, n)
            row.update(crossfade=x, overlapped=o, atom_latents=A, atom_seconds=A / spec.latent_rate,
                       crossfade_ms=1000 * x / spec.latent_rate, k_max=CAP // A)
        rows.append(row)
    return rows


def describe(res: Resolution) -> list:
    """The startup print: a geometry line and one warning line per note."""
    new = f"{res.n} new latent{'s' if res.n != 1 else ''} per step"
    head = f"[SCAPES] {res.codec} ({res.mode}): {res.rate:.2f} Hz, {new} | "
    mem = f"{res.requested_memory_seconds:g} s"
    if res.mode == _AT:
        line = (head + f"atom {res.atom} latents = {res.atom_seconds:.3f} s (overlapped {res.overlapped}, crossfade "
                f"{res.crossfade} = {round(1000 * res.crossfade_seconds)} ms, new {res.n}) | memory {mem} = "
                f"{res.memory_latents} latents -> {res.memory_atoms} atoms | attended {res.attended} latents (cap {CAP})")
    else:
        line = (head + f"block {res.n} latents = {res.hop_seconds:.3f} s | memory {mem} = {res.memory_latents} latents"
                f" | attended {res.attended} latents (cap {CAP})")
    texts = {
        "snapped": f"conditioning_rate {res.requested_rate:g} Hz snapped to {res.rate:.2f} Hz "
                   f"(nearest available rate, n = {res.n})",
        "clipped_fast": f"conditioning_rate {res.requested_rate:g} Hz is above the fastest available rate, "
                        f"clipped to {res.rate:.2f} Hz (n = {res.n})",
        "clipped_slow": f"conditioning_rate {res.requested_rate:g} Hz is below the slowest available rate, "
                        f"clipped to {res.rate:.2f} Hz (n = {res.n})",
        "cap": f"memory {mem} needs more than {CAP} attended latents, reduced to {res.attended} latents",
        "memory_raised": f"memory {mem} is shorter than one block, raised to {res.memory_window} latents",
    }
    return [line] + [f"[SCAPES] warning: {texts[k]}" for k in res.notes]


def print_resolution(res: Resolution, file=None) -> None:
    for line in describe(res):
        print(line, file=file if file is not None else sys.stdout)
