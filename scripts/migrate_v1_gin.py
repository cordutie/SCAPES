"""Migrate a V1 dataprep.gin (frames) to V2 (seconds, plus the codec and mode lines).

    python scripts/migrate_v1_gin.py IN OUT [--codec encodec48k --mode atoms]

atoms.frames, atoms.hop_frames and atoms.crossfade_frames become seconds at V1's 150 Hz (frames / 150); every other line is kept.
"""
import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from SCAPES.data.config_loader import frames_to_seconds_text  # noqa: E402

NEW_KEYS = {"frames": "seconds", "hop_frames": "hop_seconds", "crossfade_frames": "crossfade_seconds"}
FRAME_LINE = re.compile(r"^(\s*)atoms\.(frames|hop_frames|crossfade_frames)(\s*=\s*)([0-9.]+)(.*)$")


def migrate(text: str, codec: str, mode: str) -> str:
    out = []
    for line in text.splitlines():
        m = FRAME_LINE.match(line)
        if m:
            indent, key, eq, value, rest = m.groups()
            line = f"{indent}atoms.{NEW_KEYS[key]}{eq}{frames_to_seconds_text(value)}{rest}"
        out.append(line)
    out += ["", "# V2: codec and generation mode", f'codec.name = "{codec}"', f'generation.mode = "{mode}"']
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--codec", default="encodec48k")
    ap.add_argument("--mode", default="atoms")
    args = ap.parse_args()
    pathlib.Path(args.dst).write_text(migrate(pathlib.Path(args.src).read_text(), args.codec, args.mode))
    print("written", args.dst)
