#!/usr/bin/env bash
# Resumes a V2 training run detached from <run_dir>/model/checkpoints/last_trainer_state.pt (part 2 of a run).
# Usage (from SCAPES/): scripts/launch_resume.sh <dataset_dir> <run_dir>
# Writes <run_dir>/train_part2.log and <run_dir>/pid_part2.
set -euo pipefail
dataset=$(realpath "$1")
run=$(realpath "$2")
repo=$(cd "$(dirname "$0")/.." && pwd)
python="$repo/../venv_SCAPES/bin/python"

cd "$repo"
setsid nohup "$python" -u -c "from SCAPES.pipeline import run_training; run_training('$dataset', '$run/model', config_dir='$run/config', resume='latest')" \
    > "$run/train_part2.log" 2>&1 < /dev/null &
echo $! > "$run/pid_part2"
echo "started pid $(cat "$run/pid_part2"), log $run/train_part2.log"
