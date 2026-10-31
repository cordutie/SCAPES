"""S5, S8: CoDiCodec, independent atoms, phase-2 code, the per-codec templates, the old window functions and the
old-design files (plan slice 8) are gone; one gin remains.
The locked folders (tests/, tests_new/, scripts/tests_eval/) are excluded: their inert phase-2 tests stay until an unlock.
Run from SCAPES/: ../venv_SCAPES/bin/python -m pytest tests_v3/test_removal.py -q
"""
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
CONFIG = REPO / "SCAPES" / "config"

REMOVED = (r"codicodec|independent_split|OVERLAPS|SCAPES_PHASE|atoms\.overlap|atoms\.type|atoms\.window|encodec48k_atoms|"
           r"music2latent_consistent|spectrostream_ar|encodec24k_ar|s3_variants|crossfade_halves|ola_window|window_for")
# Allowed: the rejection list of the reader names the old keys. There is no legacy read path (D7).
EXEMPT = {
    "SCAPES/data/config_loader.py": r"atoms\.overlap|atoms\.type|atoms\.window",
}
# Old V2 run folder names on disk, kept in the study scripts that locked tests import.
for _s in ("compare_small_large", "eval_size", "eval_small_all", "semantic_resynthesis"):
    EXEMPT[f"scripts/{_s}.py"] = r"encodec48k_atoms|spectrostream_ar|encodec24k_ar"


def _grep(pattern, paths):
    r = subprocess.run(["git", "grep", "--untracked", "-niE", pattern, "--", *paths, ":!scripts/tests_eval",
                        ":!quickstart/lorea"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode in (0, 1), r.stderr
    return [ln for ln in r.stdout.splitlines() if ln]


def test_no_removed_identifiers():
    paths = ["SCAPES", "scripts", "docs", "quickstart", "README.md"] + sorted(p.name for p in REPO.glob("requirements*.txt"))
    bad = []
    for hit in _grep(REMOVED, paths):
        path, _, text = hit.split(":", 2)
        allowed = EXEMPT.get(path)
        if allowed and re.search(allowed, text, re.I) and not re.search("codicodec", text, re.I):
            continue
        bad.append(hit)
    assert not bad, "removed identifiers left:\n" + "\n".join(bad[:40])


def test_one_gin():
    assert not (CONFIG / "codecs").exists(), "SCAPES/config/codecs/ must be gone"
    text = (CONFIG / "generation.gin").read_text()
    keys = []
    for line in text.splitlines():
        code = line.split("#", 1)[0].strip()
        if "=" in code:
            keys.append(code.split("=", 1)[0].strip())
    assert keys == ["codec", "conditioning_rate"], keys
    comments = "\n".join(line for line in text.splitlines() if "#" in line)
    assert "conditioning_memory_seconds" in comments
    assert "docs/generation_modes.html" in comments


DELETED = ["quickstart/tutorial_full.ipynb", "docs/overlap_add.md", "docs/overlap_add",
           "scripts/make_atoms_configs.py", "scripts/atoms_check.py", "scripts/atoms_eval.py",
           "scripts/make_overlap_diagrams.py", "scripts/migrate_v1_gin.py", "scripts/make_s5_configs.py",
           "scripts/window_study.py", "scripts/window_metrics.py"]


def test_old_design_files_deleted():
    left = [p for p in DELETED if (REPO / p).exists()]
    left += [p for p in ("open_items.md", "hop_table.html") if (PROJECT / p).exists()]
    assert not left, left
    assert (REPO / "quickstart" / "lorea").is_dir() and (REPO / "quickstart" / "tutorial_simplified.ipynb").is_file()
    assert (REPO / "docs" / "generation_modes.html").is_file()       # hop_table.html goes only once the new page exists


def test_submodule_has_no_codicodec():
    import audio_codec_wrappers as acw
    assert "codicodec" not in acw.list_codecs()
