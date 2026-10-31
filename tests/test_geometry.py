"""Rounding of seconds to codec frames and samples. The V3 geometry is tested in tests_v3/test_generation_table.py."""


def test_rounding():
    """Ceiling to the codec quantum, robust to float error (design Phi)."""
    from SCAPES.data.geometry import seconds_to_frames, seconds_to_samples
    f_cd = 44100 / 4096
    assert f_cd == 10.7666015625
    assert seconds_to_frames(0.10, 75, 1) == 8        # 7.5 -> 8
    assert seconds_to_frames(0.10, 25, 1) == 3        # 2.5 -> 3
    assert seconds_to_frames(0.10, 150, 1) == 15      # exact, not 16
    assert seconds_to_frames(0.32, 150, 1) == 48
    assert seconds_to_frames(1.6, 75, 1) == 120
    assert seconds_to_frames(1.6, 25, 1) == 40
    assert seconds_to_frames(0.07, 100, 1) == 7       # 7.000000000000001 in float must not become 8
    assert seconds_to_frames(0.071, 100, 1) == 8      # ceiling, not rounding
    assert seconds_to_frames(0.10, f_cd, 8) == 8      # one CoDiCodec chunk
    assert seconds_to_frames(0.32, f_cd, 8) == 8
    assert seconds_to_frames(1.0, f_cd, 8) == 16
    assert seconds_to_frames(1.486, f_cd, 8) == 16
    assert seconds_to_frames(1.5, f_cd, 8) == 24
    assert seconds_to_frames(0.0, 75, 1) == 1
    assert seconds_to_samples(0.02, 48000) == 960
    assert seconds_to_samples(0.05, 44100) == 2205
    assert seconds_to_samples(0.0, 48000) == 0
