"""S1-S4, D2, D5: the per-codec table and the resolver of SCAPES/generation.py (pure Python, design: Resolution).
Run from SCAPES/: ../venv_SCAPES/bin/python -m pytest tests_v3/test_generation_table.py -q
"""
import math
from fractions import Fraction

import numpy as np
import pytest

from v3_helpers import atom_geometry_ref

USER = ["encodec24", "spectrostream", "encodec48", "music2latent"]
ALL = USER + ["toy_causal", "toy_noncausal"]
X_DEFAULT = {"encodec48": 15, "music2latent": 2, "toy_noncausal": 4}
A_MIN = {"encodec48": 30, "music2latent": 6, "toy_noncausal": 12}
N_MIN = {"encodec24": 5, "spectrostream": 2, "encodec48": 8, "music2latent": 1, "toy_causal": 1, "toy_noncausal": 1}
N_MAX = {"encodec24": 300, "spectrostream": 300, "encodec48": 142, "music2latent": 149, "toy_causal": 300, "toy_noncausal": 148}


def gen():
    import SCAPES.generation as g
    return g


def ref(codec, n):
    return atom_geometry_ref(n, X_DEFAULT[codec], A_MIN[codec])


def expected_atoms(codec, n, ms):
    """Brute-force K of design "Resolution": the smallest K with m_l <= x + K n, then the cap."""
    r = gen().CODECS[codec].latent_rate
    x, o, A = ref(codec, n)
    ml = math.ceil(r * ms - 1e-9)
    K = 1
    while ml > x + K * n:
        K += 1
    if K * A > 300:
        K = 300 // A
    return K, A, K * A


def test_table_matches_wrappers():
    import audio_codec_wrappers as acw
    g = gen()
    assert set(g.CODECS) == set(ALL)
    for name in ALL:
        spec = g.CODECS[name]
        info = acw.codec_info(spec.wrapper)
        assert (spec.sample_rate, spec.hop_length) == (info.sample_rate, info.hop_length), name
        assert spec.mode == ("autoregressive" if info.causal else "atoms"), name
        assert spec.crossfade_default == X_DEFAULT.get(name), name
        assert spec.min_atom == A_MIN.get(name), name
        assert spec.n_min == N_MIN[name], name
        assert spec.user == (name in USER), name
        assert math.isclose(spec.latent_rate, info.sample_rate / info.hop_length, rel_tol=1e-12)
        assert spec.default_memory_seconds == (0.5 if spec.mode == "atoms" else 1.0)
    assert g.CODECS["encodec48"].standardize_latents is False
    assert all(g.CODECS[c].standardize_latents is True for c in ("encodec24", "spectrostream", "music2latent"))
    assert g.CODECS["encodec24"].past_noise == 0.1 and g.CODECS["encodec48"].past_noise == 0.0
    assert g.CAP == 300 and g.DEFAULT_RATE == 10.0
    assert g.WRAPPER_TO_CANONICAL["encodec48k"] == "encodec48" and g.WRAPPER_TO_CANONICAL["encodec24k"] == "encodec24"


def test_atom_rule():
    g = gen()
    # Esteban's rows, spelled out (design: Worked examples)
    assert g.atom_geometry(g.CODECS["encodec48"], 8) == (4, 18, 30)
    assert g.atom_geometry(g.CODECS["encodec48"], 15) == (8, 15, 38)
    assert g.atom_geometry(g.CODECS["music2latent"], 1) == (1, 4, 6)
    assert g.atom_geometry(g.CODECS["music2latent"], 2) == (1, 3, 6)
    assert g.atom_geometry(g.CODECS["music2latent"], 3) == (2, 3, 8)
    assert g.atom_geometry(g.CODECS["music2latent"], 4) == (2, 4, 10)
    for codec in X_DEFAULT:
        for n in range(N_MIN[codec], N_MAX[codec] + 1):
            x, o, A = g.atom_geometry(g.CODECS[codec], n)
            assert (x, o, A) == ref(codec, n), (codec, n)
            assert 1 <= x <= n and o >= n, (codec, n)     # the phase-1 overlap-add layout is valid on every row


WORKED = [  # (codec, c, m_s) -> fields (design "Worked examples")
    (("encodec48", 12, 0.5), dict(n=13, crossfade=7, overlapped=13, atom=33, memory_latents=75, memory_atoms=6, attended=198,
                                  k_max=9, notes=("snapped",))),
    (("music2latent", 3, 1), dict(n=4, crossfade=2, overlapped=4, atom=10, memory_latents=11, memory_atoms=3, attended=30,
                                  k_max=30, notes=("snapped",))),
    (("spectrostream", 5, 2), dict(n=5, memory_latents=50, memory_window=50, attended=50, notes=())),
    (("music2latent", 10, 10), dict(n=1, crossfade=1, overlapped=4, atom=6, memory_latents=108, memory_atoms=50, attended=300,
                                    k_max=50, notes=("snapped", "cap"))),
    (("encodec24", 20, None), dict(n=5, memory_latents=75, memory_window=75, attended=75, notes=("clipped_fast",))),
    (("encodec24", 12, None), dict(n=6, memory_latents=75, memory_window=75, attended=75, notes=("snapped",))),
]


@pytest.mark.parametrize("args,want", WORKED, ids=[f"{a[0]}-{a[1]}-{a[2]}" for a, _ in WORKED])
def test_worked_examples(args, want):
    g = gen()
    codec, c, ms = args
    res = g.resolve_generation(codec, c, ms)
    assert math.isclose(res.rate, g.CODECS[codec].latent_rate / want["n"], rel_tol=1e-9)
    for key, value in want.items():
        assert getattr(res, key) == value, (key, getattr(res, key), value)
    assert res.mode == g.CODECS[codec].mode and res.codec == codec


def test_defaults():
    g = gen()
    want = {"encodec24": dict(n=8, memory_window=75, attended=75, requested_memory_seconds=1.0, notes=("snapped",)),
            "spectrostream": dict(n=3, memory_window=25, attended=25, requested_memory_seconds=1.0, notes=("snapped",)),
            "encodec48": dict(n=15, crossfade=8, overlapped=15, atom=38, memory_latents=75, memory_atoms=5, attended=190,
                              k_max=7, covered_latents=83, buffer_span_latents=98, requested_memory_seconds=0.5, notes=()),
            "music2latent": dict(n=1, crossfade=1, overlapped=4, atom=6, memory_latents=6, memory_atoms=5, attended=30,
                                 k_max=50, covered_latents=6, buffer_span_latents=10, requested_memory_seconds=0.5,
                                 notes=("snapped",))}
    for codec, fields in want.items():
        res = g.resolve_generation(codec)
        assert res.requested_rate == 10.0
        for key, value in fields.items():
            assert getattr(res, key) == value, (codec, key, getattr(res, key))
    assert math.isclose(g.resolve_generation("encodec48").rate, 10.0, rel_tol=1e-12)


@pytest.mark.parametrize("codec", ALL)
def test_every_rate_snaps_to_itself(codec):
    g = gen()
    spec = g.CODECS[codec]
    for n in range(N_MIN[codec], N_MAX[codec] + 1):
        res = g.resolve_generation(codec, spec.latent_rate / n, 0.05)
        assert res.n == n, (codec, n, res.n)
        assert not {"snapped", "clipped_fast", "clipped_slow"} & set(res.notes), (codec, n, res.notes)


def test_ties_go_to_smaller_n():
    g = gen()
    for n in (5, 6, 100):                            # encodec24, latent rate 75: exact midpoints
        mid = float((Fraction(75, n) + Fraction(75, n + 1)) / 2)
        assert g.resolve_generation("encodec24", mid, 1.0).n == n, n
    r = g.CODECS["music2latent"].latent_rate
    for n in (1, 4, 50):                             # irrational-ish latent rate: either side of the midpoint
        mid = (r / n + r / (n + 1)) / 2
        assert g.resolve_generation("music2latent", mid * (1 + 1e-9), 0.5).n == n
        assert g.resolve_generation("music2latent", mid * (1 - 1e-9), 0.5).n == n + 1


@pytest.mark.parametrize("codec", ALL)
def test_clipping(codec):
    from SCAPES.data.config_loader import ConfigError
    g = gen()
    ms = 0.001 if codec.startswith("toy") else 0.1     # small enough for no cap, large enough for no memory raise at n_min
    fast = g.resolve_generation(codec, 1e4, ms)
    assert fast.n == N_MIN[codec] and fast.notes == ("clipped_fast",)
    slow = g.resolve_generation(codec, 1e-4, ms)
    assert slow.n == N_MAX[codec] and slow.notes[0] == "clipped_slow" and "snapped" not in slow.notes
    for bad in (0.0, -1.0):
        with pytest.raises(ConfigError):
            g.resolve_generation(codec, bad, 0.5)
        with pytest.raises(ConfigError):
            g.resolve_generation(codec, 10.0, bad)
    for c in np.logspace(-4, 4, 50):                  # no other rate raises; the cap always holds
        res = g.resolve_generation(codec, float(c), 0.5)
        assert N_MIN[codec] <= res.n <= N_MAX[codec] and res.attended <= 300


def test_memory_raised():
    g = gen()
    res = g.resolve_generation("encodec24", 0.5, 1.0)
    assert (res.n, res.memory_latents, res.memory_window, res.attended) == (150, 75, 150, 150)
    assert res.notes == ("memory_raised",)
    assert "memory_raised" not in g.resolve_generation("encodec24").notes


def test_cap():
    g = gen()
    res = g.resolve_generation("music2latent", 10, 10)
    assert (res.memory_atoms, res.attended) == (50, 300) and res.notes[-1] == "cap"
    res = g.resolve_generation("encodec48", 10, 5)
    assert (res.memory_atoms, res.attended, res.notes) == (7, 266, ("cap",))
    res = g.resolve_generation("encodec24", 10, 5)
    assert (res.memory_window, res.attended) == (300, 300) and res.notes[-1] == "cap"
    assert "cap" not in g.resolve_generation("encodec48", 10, 0.5).notes


def test_n_max():
    g = gen()
    for codec in ALL:
        spec = g.CODECS[codec]
        assert g.n_max(spec) == N_MAX[codec], codec
        if spec.mode == "atoms":
            nm = N_MAX[codec]
            assert ref(codec, nm)[2] <= 300 < ref(codec, nm + 1)[2], codec


@pytest.mark.parametrize("codec", ["encodec48", "music2latent"])
def test_k_is_derived(codec):
    g = gen()
    spec = g.CODECS[codec]
    for n in (1, 2, 4, 8, 15, 30):
        if not N_MIN[codec] <= n <= N_MAX[codec]:
            continue
        for ms in (0.1, 0.5, 2.0):
            res = g.resolve_generation(codec, spec.latent_rate / n, ms)
            assert (res.memory_atoms, res.atom, res.attended) == expected_atoms(codec, n, ms), (n, ms)
            assert res.covered_latents == res.crossfade + res.memory_atoms * n
            assert res.buffer_span_latents == res.atom + (res.memory_atoms - 1) * n


def test_describe_format():
    g = gen()
    assert g.describe(g.resolve_generation("music2latent")) == [
        "[SCAPES] music2latent (atoms): 10.77 Hz, 1 new latent per step | atom 6 latents = 0.557 s (overlapped 4, crossfade 1 "
        "= 93 ms, new 1) | memory 0.5 s = 6 latents -> 5 atoms | attended 30 latents (cap 300)",
        "[SCAPES] warning: conditioning_rate 10 Hz snapped to 10.77 Hz (nearest available rate, n = 1)",
    ]
    assert g.describe(g.resolve_generation("encodec24")) == [
        "[SCAPES] encodec24 (autoregressive): 9.38 Hz, 8 new latents per step | block 8 latents = 0.107 s | memory 1 s = 75 "
        "latents | attended 75 latents (cap 300)",
        "[SCAPES] warning: conditioning_rate 10 Hz snapped to 9.38 Hz (nearest available rate, n = 8)",
    ]
    for codec, c, ms in (("music2latent", 10, 10), ("encodec24", 20, 1), ("encodec24", 0.5, 1)):
        res = g.resolve_generation(codec, c, ms)
        lines = g.describe(res)
        assert len(lines) == 1 + len(res.notes)
        assert lines[0].startswith(f"[SCAPES] {codec} (") and all(ln.startswith("[SCAPES] warning: ") for ln in lines[1:])


def test_print_resolution(capsys):
    g = gen()
    res = g.resolve_generation("music2latent")
    g.print_resolution(res)
    assert capsys.readouterr().out.splitlines() == g.describe(res)


def test_rate_table():
    g = gen()
    rows_expected = {"encodec24": 296, "spectrostream": 299, "encodec48": 135, "music2latent": 149}
    for codec, count in rows_expected.items():
        rows = g.rate_table(codec)
        assert len(rows) == count and [r["n"] for r in rows] == list(range(N_MIN[codec], N_MAX[codec] + 1))
        spec = g.CODECS[codec]
        for row in rows:
            res = g.resolve_generation(codec, row["rate_hz"])
            assert res.n == row["n"]
            assert math.isclose(row["step_seconds"], row["n"] / spec.latent_rate, rel_tol=1e-9)
            if spec.mode == "atoms":
                assert (row["crossfade"], row["overlapped"], row["atom_latents"]) == (res.crossfade, res.overlapped, res.atom)
                assert math.isclose(row["atom_seconds"], res.atom / spec.latent_rate, rel_tol=1e-9)
                assert math.isclose(row["crossfade_ms"], 1000 * res.crossfade / spec.latent_rate, rel_tol=1e-9)
                assert row["k_max"] == 300 // res.atom
