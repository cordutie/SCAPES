"""Slice B2: gin schema (design D10, D20; brief "add two lines" migration; Esteban review: seconds only)."""
import subprocess
import sys

import pytest

from conftest import CONFIG, PROJECT, SCRIPTS, SCAPES_REPO, TOY_AR, TOY_COMMON, V1_MODEL, resolve_lines

REAL = ["encodec24k", "spectrostream", "encodec48k", "codicodec"]
AR_LINES = ['generation.mode = "autoregressive"', "ar.block_seconds = 0.10", "ar.memory_seconds = 1.6"]
ATOMS_LINES = ['generation.mode = "atoms"', "atoms.seconds = 0.32", "atoms.hop_seconds = 0.10",
               "atoms.crossfade_seconds = 0.02", "dataset.memory_buffer_atoms = 1"]


def test_required_lines(tmp_path):
    from SCAPES.data.config_loader import ConfigError
    with pytest.raises(ConfigError, match=r"codec\.name"):
        resolve_lines(tmp_path, ATOMS_LINES + TOY_COMMON, "no_codec")
    with pytest.raises(ConfigError, match=r"generation\.mode"):
        resolve_lines(tmp_path, ['codec.name = "encodec48k"'] + ATOMS_LINES[1:] + TOY_COMMON, "no_mode")


def test_invalid_values(tmp_path):
    from SCAPES.data.config_loader import ConfigError
    with pytest.raises((ConfigError, KeyError)) as e:
        resolve_lines(tmp_path, ['codec.name = "dac"'] + ATOMS_LINES + TOY_COMMON, "bad_codec")
    for name in REAL:
        assert name in str(e.value)
    with pytest.raises(ConfigError, match="foo"):
        resolve_lines(tmp_path, ['codec.name = "encodec48k"', 'generation.mode = "foo"'] + ATOMS_LINES[1:] + TOY_COMMON,
                      "bad_mode")


def test_mode_codec_rules(tmp_path):
    from SCAPES.data.config_loader import ConfigError
    for codec in ("encodec48k", "codicodec", "toy_noncausal"):
        with pytest.raises(ConfigError):
            resolve_lines(tmp_path, [f'codec.name = "{codec}"'] + AR_LINES + TOY_COMMON, f"ar_{codec}")
    with pytest.warns(UserWarning):
        resolve_lines(tmp_path, ['codec.name = "encodec24k"'] + ATOMS_LINES + TOY_COMMON, "atoms_e24")


def test_frame_lines_rejected(tmp_path):
    """User gin files are seconds-only: frame lines and the old AR key raise ConfigError with the seconds equivalent."""
    from SCAPES.data.config_loader import ConfigError
    with pytest.raises(ConfigError, match=r"atoms\.seconds = 0\.32"):
        resolve_lines(tmp_path, ['codec.name = "encodec48k"', "atoms.frames = 48"] + ATOMS_LINES + TOY_COMMON, "frames")
    with pytest.raises(ConfigError, match=r"atoms\.hop_seconds = 0\.1"):
        lines = ['codec.name = "encodec48k"', 'generation.mode = "atoms"', "atoms.seconds = 0.32",
                 "atoms.hop_frames = 15", "atoms.crossfade_seconds = 0.02", "dataset.memory_buffer_atoms = 1"]
        resolve_lines(tmp_path, lines + TOY_COMMON, "hop_frames")
    with pytest.raises(ConfigError, match=r"ar\.memory_seconds"):
        resolve_lines(tmp_path, ['codec.name = "encodec24k"', 'generation.mode = "autoregressive"',
                                 "ar.block_seconds = 0.10", "ar.context_seconds = 1.6"] + TOY_COMMON, "old_ar_key")
    with pytest.raises(ConfigError, match=r"memory_buffer_atoms"):
        resolve_lines(tmp_path, ['codec.name = "encodec24k"'] + AR_LINES + ["dataset.memory_buffer_atoms = 3"]
                      + TOY_COMMON, "ar_with_atom_count")


def test_v1_gin_migrates(tmp_path):
    """V1 checkpoints: their machine-written inference.gin (frames) plus the two lines gives the V1 geometry."""
    from SCAPES.codecs import get_codec_info
    from SCAPES.data.config_loader import parse_gin_config
    from SCAPES.data.geometry import geometry_from_inference_gin
    p = tmp_path / "inference.gin"
    p.write_text((V1_MODEL / "checkpoints" / "inference.gin").read_text()
                 + '\ncodec.name = "encodec48k"\ngeneration.mode = "atoms"\n')
    raw = parse_gin_config(p)
    geom = geometry_from_inference_gin(raw, get_codec_info("encodec48k"))
    assert geom.mode == "atoms"
    assert (geom.unit_frames, geom.unit_hop_frames, geom.crossfade_samples, geom.memory_units) == (48, 15, 960, 1)
    assert geom.memory_frames == 48
    assert geom.frame_dim == 129 and geom.has_scale is True


def test_migrate_v1_gin(tmp_path):
    """scripts/migrate_v1_gin.py turns a V1 dataprep.gin (frames) into a seconds-only V2 file with the same geometry."""
    src = PROJECT / "datasets" / "microtex" / "config" / "dataprep.gin"
    out = tmp_path / "migrated" / "dataprep.gin"
    out.parent.mkdir()
    r = subprocess.run([sys.executable, str(SCRIPTS / "migrate_v1_gin.py"), str(src), str(out)], cwd=SCAPES_REPO,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-3000:]
    text = out.read_text()
    assert "atoms.frames" not in text and "hop_frames" not in text and "crossfade_frames" not in text
    _, geom = resolve_lines(tmp_path, text.splitlines(), "resolved")
    assert (geom.codec, geom.mode) == ("encodec48k", "atoms")
    assert (geom.unit_frames, geom.unit_hop_frames, geom.crossfade_samples, geom.memory_units) == (48, 15, 960, 1)


def test_run_config_real_datasets(tmp_path):
    """R2-C1 / D20: the real (V1) dataset gin folders plus each codec snippet give a valid V2 run config, and the dataset
    files are not modified."""
    from SCAPES.data.config_loader import load_training_config
    from SCAPES.pipeline import build_run_config
    expected = {
        "encodec48k_atoms": ("encodec48k", "atoms", 48, 15),
        "codicodec_atoms": ("codicodec", "atoms", 16, 8),
        "encodec24k_ar": ("encodec24k", "autoregressive", 8, 8),
        "spectrostream_ar": ("spectrostream", "autoregressive", 3, 3),
    }
    for ds in ("micro_microtex", "microtex"):
        src = PROJECT / "datasets" / ds / "config"
        before = {p.name: p.read_text() for p in sorted(src.glob("*.gin"))}
        assert "atoms.frames" in before["dataprep.gin"]  # the dataset files really are V1
        for stem, (codec, mode, unit, hop) in expected.items():
            out = build_run_config(src, [CONFIG / "codecs" / f"{stem}.gin"], ["training.max_steps = 10"],
                                   tmp_path / "runs" / ds / stem)
            text = (out / "dataprep.gin").read_text()
            _, g = resolve_lines(tmp_path, text.splitlines(), f"res_{ds}_{stem}")
            assert (g.codec, g.mode, g.unit_frames, g.unit_hop_frames) == (codec, mode, unit, hop), (ds, stem)
            assert load_training_config(out / "training.gin").max_steps == 10
        assert {p.name: p.read_text() for p in sorted(src.glob("*.gin"))} == before


def test_last_key_wins(tmp_path):
    from SCAPES.data.config_loader import parse_gin_config
    p = tmp_path / "x.gin"
    p.write_text('model.size = "large"\ntraining.max_steps = 300\nmodel.size = "small"\ntraining.max_steps = 20\n')
    cfg = parse_gin_config(p)
    assert cfg["model"]["size"] == "small"
    assert cfg["training"]["max_steps"] == 20


def test_templates_resolve(tmp_path):
    from SCAPES.data.config_loader import load_training_config
    snippets = sorted((CONFIG / "codecs").glob("*.gin"))
    assert {p.stem for p in snippets} >= {"encodec48k_atoms", "codicodec_atoms", "encodec24k_ar", "spectrostream_ar",
                                          "encodec24k_atoms"}
    dp_template = (CONFIG / "dataprep.gin").read_text().splitlines()
    tr_template = (CONFIG / "training.gin").read_text().splitlines()
    for snip in snippets:
        lines = snip.read_text().splitlines()
        with _maybe_warns(snip.stem):
            _, geom = resolve_lines(tmp_path, dp_template + lines, snip.stem)
        assert geom.codec == snip.stem.rsplit("_", 1)[0]
        tr = tmp_path / snip.stem / "training.gin"
        tr.write_text("\n".join(tr_template + lines) + "\n")
        load_training_config(tr)


class _maybe_warns:
    """atoms + causal codec (encodec24k_atoms, the S3 comparator) warns; the others must not raise."""

    def __init__(self, stem):
        self.ctx = pytest.warns(UserWarning) if stem == "encodec24k_atoms" else None

    def __enter__(self):
        return self.ctx.__enter__() if self.ctx else None

    def __exit__(self, *exc):
        return self.ctx.__exit__(*exc) if self.ctx else False
