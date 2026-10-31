"""S9 (old code removed, docs current) and S2 (the EnCodec 48k Hann comment). Design D4, D21, D22; ledger R1-C1.

Lives outside tests/ because that folder is locked. Run from SCAPES/: ../venv_SCAPES/bin/python -m pytest tests_new -q
"""
import re
import subprocess
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _atoms_toy import phase2  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "SCAPES" / "config"
SEARCH = ["SCAPES", "scripts", "tests_new", "README.md", "docs"]
THIS = "tests_new/test_old_code_removed.py"

# Code identifiers of the removed grains / shared-latent overlap modes / crossfade laws and the old templates
OLD_IDENTIFIERS = (r"_overlap_geometry|OVERLAP_FRACTIONS|GRAIN_CODECS|grain_overlap_samples|audio_hop_samples|"
                   r"crossfade_law_for|CROSSFADE_LAWS|atoms_overlap_mode|atoms_crossfade_law|music2latent_atoms|"
                   r"codicodec_grains|DEFAULT_NORMAL_FADE_SECONDS")
# The old gin keys may only be named where they are rejected, and in the tests of that rejection
OLD_KEYS = r"overlap_mode|crossfade_law"
OLD_KEY_EXEMPT = {
    "SCAPES/data/config_loader.py": "the ConfigError messages name the removed keys and their replacements",
    "tests_new/test_atom_geometry.py": "test_rejected_old_keys writes the removed keys to check the rejection",
    THIS: "this test lists the patterns",
}
NO_GRAIN = ["README.md", "docs", "SCAPES/config", "SCAPES/data/geometry.py"]


def _grep(pattern, paths, flags=("-nE",)):
    r = subprocess.run(["git", "grep", "--untracked", *flags, pattern, "--", *paths], cwd=REPO, capture_output=True, text=True)
    assert r.returncode in (0, 1), r.stderr
    return [ln for ln in r.stdout.splitlines() if ln]


def test_no_old_identifiers():
    hits = [h for h in _grep(OLD_IDENTIFIERS, SEARCH) if not h.startswith(THIS + ":")]
    assert not hits, "old identifiers left:\n" + "\n".join(hits[:40])
    key_hits = [h for h in _grep(OLD_KEYS, SEARCH) if h.split(":", 1)[0] not in OLD_KEY_EXEMPT]
    assert not key_hits, "old keys outside the rejection code:\n" + "\n".join(key_hits[:40])
    grain = _grep("grain", NO_GRAIN, flags=("-niE",))
    assert not grain, "grain mentions left in docs / config / geometry:\n" + "\n".join(grain[:40])


def _check_templates(tmp_path, stems):
    sys.path.insert(0, str(REPO / "tests"))
    from conftest import resolve_lines
    from SCAPES.data.config_loader import parse_gin_config
    from SCAPES.data.geometry import WINDOWS
    names = {p.stem for p in (CONFIG / "codecs").glob("*.gin")}
    assert not {"music2latent_atoms", "codicodec_grains"} & names
    assert set(stems) <= names
    windows = {}
    for stem in stems:
        lines = (CONFIG / "dataprep.gin").read_text().splitlines() + (CONFIG / "codecs" / f"{stem}.gin").read_text().splitlines()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _, g = resolve_lines(tmp_path, lines, stem)
        assert g.atom_type == ("consistent" if stem.endswith("consistent") else "independent")
        assert g.memory_units == 3
        windows[stem] = parse_gin_config(CONFIG / "codecs" / f"{stem}.gin")["atoms"]["window"]
        assert windows[stem] in WINDOWS
        assert g.window == windows[stem]
    return windows


def test_templates(tmp_path):
    _check_templates(tmp_path, ["music2latent_consistent"])


@phase2
def test_templates_independent(tmp_path):
    w = _check_templates(tmp_path, ["music2latent_consistent", "music2latent_independent", "codicodec_independent"])
    assert w["music2latent_consistent"] == w["music2latent_independent"], "one window A for Music2Latent"


def _doc_figures(doc):
    figs = re.findall(r"\]\(([^)\s]+\.png)\)", doc)
    assert figs, "docs/overlap_add.md links no figure"
    for f in figs:
        assert (REPO / "docs" / f).exists(), f


def test_docs_current():
    readme = (REPO / "README.md").read_text()
    for key in ("atoms.type", "atoms.window"):
        assert key in readme, key
    doc = (REPO / "docs" / "overlap_add.md").read_text()
    for word in ("consistent", "atoms.window", "4410"):
        assert word in doc, word
    _doc_figures(doc)


@phase2
def test_docs_current_independent():
    new_key = re.compile(r"atoms\.overlap(?!_)")              # the new key, not the removed atoms.overlap_mode
    assert new_key.search((REPO / "README.md").read_text())
    doc = (REPO / "docs" / "overlap_add.md").read_text()
    assert new_key.search(doc) and "independent atoms" in doc
    _doc_figures(doc)


def test_encodec_hann_comment():
    src = (REPO / "SCAPES" / "data" / "geometry.py").read_text().splitlines()
    i = next(n for n, ln in enumerate(src) if ln.startswith("def ola_window"))
    above = "\n".join(ln for ln in src[max(0, i - 15):i] if ln.strip().startswith("#"))
    for word in ("EnCodec 48k", "Hann", "deterministic"):
        assert word in above, word
