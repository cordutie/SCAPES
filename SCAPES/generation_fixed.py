"""Candidate atom rule 'fixed', kept apart from generation.py (the rule in use) so that the winner of the test can replace it.

Same codec table and same choice of the step n from the conditioning rate as generation.py. Different crossfade and overlap:
    x = min(CROSSFADE, n)          the crossfade (the latents where two atoms are both audible), never longer than the hop
    o = ZEROED                     the leading latents of an atom that the window zeroes (decoded, then thrown away)
    A = o + x + n                  the atom
Measured by the author: music2latent wants at least 6 latents per atom, encodec48 at least 30. With these numbers A >= 6 (music2latent,
n >= 1: A = 4 + x + n) and A >= 31 (encodec48, n >= 8: A = 15 + x + n) for every n, so the minimum is never the limit.
"""
import math
from dataclasses import dataclass

from SCAPES.generation import CAP, CODECS, DEFAULT_RATE

RULE = "fixed"
FIXED = {
    "music2latent": {"crossfade": 2, "zeroed": 4},
    "encodec48": {"crossfade": 15, "zeroed": 15},
}


def atom_geometry_fixed(codec: str, n: int):
    """(x, o, A) for n new latents per step."""
    f = FIXED[codec]
    x = min(f["crossfade"], n)
    o = f["zeroed"]
    return x, o, o + x + n


def n_max_fixed(codec: str) -> int:
    spec = CODECS[codec]
    n = spec.n_min
    while atom_geometry_fixed(codec, n + 1)[2] <= CAP:
        n += 1
    return n


@dataclass(frozen=True)
class FixedResolution:
    codec: str
    requested_rate: float
    requested_memory_seconds: float
    latent_rate: float
    n: int
    rate: float
    crossfade: int
    zeroed: int
    atom: int
    memory_atoms: int
    attended: int

    @property
    def overlap(self) -> int:              # latents shared by consecutive atoms
        return self.atom - self.n

    @property
    def kept(self) -> float:               # share of the decoded audio that reaches the output
        return (self.n + self.crossfade) / self.atom

    @property
    def attended_seconds(self) -> float:
        return self.attended / self.latent_rate


def resolve_fixed(codec: str, conditioning_rate: float = DEFAULT_RATE, memory_seconds: float = 1.0) -> FixedResolution:
    spec = CODECS[codec]
    if spec.mode != "atoms":
        raise ValueError(f"{codec} is not an atoms codec")
    r = spec.latent_rate
    hi = n_max_fixed(codec)
    # the nearest step to the requested rate; ties go to the smaller n, as in generation.py
    n, best = spec.n_min, abs(r / spec.n_min - conditioning_rate)
    for k in range(spec.n_min + 1, hi + 1):
        d = abs(r / k - conditioning_rate)
        if d < best - 1e-12 * conditioning_rate:
            n, best = k, d
    x, o, A = atom_geometry_fixed(codec, n)
    m = math.ceil(r * memory_seconds - 1e-9)               # the memory in latents
    K = max(1, -(-(m - x) // n))
    K = min(K, CAP // A)                                    # never more than CAP attended latents
    return FixedResolution(codec=codec, requested_rate=conditioning_rate, requested_memory_seconds=memory_seconds, latent_rate=r,
                           n=n, rate=r / n, crossfade=x, zeroed=o, atom=A, memory_atoms=K, attended=K * A)
