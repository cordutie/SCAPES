"""Table of the atom rules for a list of conditioning rates: the rule in use (SCAPES/generation.py) and the candidate
(SCAPES/generation_fixed.py). Run from the repository folder (the one with the SCAPES package):

    python SCAPES/auxiliar/compare_atom_rules.py [--memory 1.0] [--rates 10 8 5 4 3 2 1.5 1 0.5]

Columns: n (new latents per step), step in Hz, x (crossfade), o (the zeroed part, the 'c' of the test), A (atom = o + x + n), overlap
with the previous atom (A - n), kept (share of the decoded audio that reaches the output = (n + x) / A), K (memory atoms) and the
attended seconds at the memory given. 'snapped' marks a rate that is not reachable exactly.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from SCAPES.generation import CODECS, resolve_generation  # noqa: E402
from SCAPES.generation_fixed import resolve_fixed  # noqa: E402

CODECS_TO_SHOW = ("music2latent", "encodec48")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--memory", type=float, default=1.0, help="conditioning memory in seconds (the models use 1.0)")
    p.add_argument("--rates", type=float, nargs="+", default=[10, 8, 5, 4, 3, 2, 1.5, 1, 0.5])
    a = p.parse_args()
    for codec in CODECS_TO_SHOW:
        spec = CODECS[codec]
        print(f"\n{codec}: latent rate {spec.latent_rate:.2f} Hz, memory {a.memory} s")
        print(f"  {'rate':>5} {'rule':>6} | {'n':>3} {'step Hz':>7} {'x':>3} {'o':>3} {'A':>4} {'overlap':>7} {'kept':>5} {'K':>2} {'attended':>9}")
        for c in a.rates:
            cur = resolve_generation(codec, c, a.memory, atoms_rule="old")
            new = resolve_fixed(codec, c, a.memory)
            cur_kept = (cur.n + cur.crossfade) / cur.atom
            snapped = "  snapped" if "snapped" in cur.notes else ""
            print(f"  {c:>5} {'in use':>6} | {cur.n:>3} {cur.rate:>7.2f} {cur.crossfade:>3} {cur.overlapped:>3} {cur.atom:>4} "
                  f"{cur.atom - cur.n:>7} {cur_kept:>5.0%} {cur.memory_atoms:>2} {cur.attended / cur.latent_rate:>8.2f}s{snapped}")
            print(f"  {'':>5} {'fixed':>6} | {new.n:>3} {new.rate:>7.2f} {new.crossfade:>3} {new.zeroed:>3} {new.atom:>4} "
                  f"{new.overlap:>7} {new.kept:>5.0%} {new.memory_atoms:>2} {new.attended_seconds:>8.2f}s")


if __name__ == "__main__":
    main()
