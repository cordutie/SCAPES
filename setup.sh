#!/usr/bin/env bash
# Sets up SCAPES from this folder: environment (.venv), packages, codec wrappers and the SpectroStream weights.
#   ./setup.sh                  (PYTHON=python3.12 to choose the Python, VENV=... the folder, CUDA=0 or CUDA=1 to force CPU or CUDA builds)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"
die() { echo "error: $*" >&2; exit 1; }

command -v git >/dev/null 2>&1 || die "git is missing. Install it with: sudo apt install git"
command -v "$PYTHON" >/dev/null 2>&1 || die "$PYTHON is missing. Install Python 3.12 (or pass PYTHON=/path/to/python)"
"$PYTHON" -c "import sys; sys.exit(sys.version_info < (3, 11))" || die "$PYTHON is older than 3.11. Install Python 3.12 (sudo apt install python3.12 python3.12-venv) and pass PYTHON=python3.12"
"$PYTHON" -c "import ensurepip" >/dev/null 2>&1 || die "the venv module is missing for $PYTHON. Install it with: sudo apt install python3-venv (or python3.12-venv)"

git submodule update --init               # external/audio_codec_wrappers (no-op when cloned with --recurse-submodules)

[ -d "$VENV" ] || "$PYTHON" -m venv "$VENV"
. "$VENV/bin/activate"
python -m pip install --upgrade pip

if [ "${CUDA:-auto}" = 1 ] || { [ "${CUDA:-auto}" = auto ] && nvidia-smi -L >/dev/null 2>&1; }; then
    echo "==> CUDA builds (torch cu128, jax cuda12)"
    python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
else
    echo "==> CPU builds (torch cpu, plain jax)"
    REQ="$(mktemp)"
    grep -v -E '^(torch|torchaudio)==' requirements.txt | sed 's/^jax\[cuda12\]==/jax==/' > "$REQ"
    python -m pip install torch==2.10.0+cpu torchaudio==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
    python -m pip install -r "$REQ"
    rm -f "$REQ"
fi
python -m pip install --no-deps msclap==1.3.4 "magenta-rt @ git+https://github.com/magenta/magenta-realtime@694a545e4ba0b88bf1150137b129582166d3e07f"
python -m pip install music2latent==0.1.8
python -m pip install -e ./external/audio_codec_wrappers
python -m pip install --no-deps -e .                       # `import SCAPES` from any folder
python external/audio_codec_wrappers/scripts/fetch_weights.py --codec spectrostream

python -c "import SCAPES.pipeline, audio_codec_wrappers" && echo "Done. Activate it with: source $VENV/bin/activate"
