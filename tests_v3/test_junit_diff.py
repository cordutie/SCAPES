"""Slice 0 tool: scripts/junit_diff.py compares the outcomes of two junit xml files.

Contract: node id = classname with "." replaced by "/" + ".py::" + name (pytest's junit classnames for module-level tests);
outcome = passed (no child), failed (<failure>), error (<error>), skipped (<skipped>). The markdown output has one table
row per node id whose outcome differs, including tests only present in one file (outcome "absent").
Run from SCAPES/: ../venv_SCAPES/bin/python -m pytest tests_v3/test_junit_diff.py -q
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "scripts" / "junit_diff.py"

BASE = """<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">
<testcase classname="tests.test_a" name="test_stays"/>
<testcase classname="tests.test_a" name="test_breaks"/>
<testcase classname="tests.test_b" name="test_unskipped"><skipped message="x"/></testcase>
<testcase classname="tests.test_b" name="test_removed"/>
</testsuite></testsuites>"""
NEW = """<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">
<testcase classname="tests.test_a" name="test_stays"/>
<testcase classname="tests.test_a" name="test_breaks"><failure message="boom"/></testcase>
<testcase classname="tests.test_b" name="test_unskipped"/>
<testcase classname="tests.test_c" name="test_new[p1]"/>
</testsuite></testsuites>"""
CHANGED = ["tests/test_a.py::test_breaks", "tests/test_b.py::test_unskipped", "tests/test_b.py::test_removed",
           "tests/test_c.py::test_new[p1]"]


def _run(tmp_path, predicted=None):
    (tmp_path / "base.xml").write_text(BASE)
    (tmp_path / "new.xml").write_text(NEW)
    cmd = [sys.executable, str(TOOL), str(tmp_path / "base.xml"), str(tmp_path / "new.xml")]
    if predicted is not None:
        (tmp_path / "pred.txt").write_text("\n".join(predicted) + "\n")
        cmd += ["--predicted", str(tmp_path / "pred.txt")]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_rows_are_the_changed_tests(tmp_path):
    r = _run(tmp_path)
    assert r.returncode == 0, r.stderr
    rows = [ln for ln in r.stdout.splitlines() if ln.startswith("|") and "::" in ln]
    assert sorted(ln.split("|")[1].strip() for ln in rows) == sorted(CHANGED)
    assert "test_stays" not in r.stdout


def test_predicted_exit_codes(tmp_path):
    assert _run(tmp_path, predicted=CHANGED[:1]).returncode == 1
    assert _run(tmp_path, predicted=CHANGED).returncode == 0
