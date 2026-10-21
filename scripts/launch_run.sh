#!/usr/bin/env bash
# Starts a V2 training run detached (setsid + nohup: it survives the terminal and SSH closing).
# Usage (from SCAPES/): scripts/launch_run.sh <dataset_dir> <run_dir>
#   <run_dir>/config/ must hold dataprep.gin and training.gin (pipeline.build_run_config) and the dataset must be prepared.
# Writes <run_dir>/model/ (checkpoints, loss history), <run_dir>/train.log, <run_dir>/pid and <run_dir>/command.txt.
set -euo pipefail
dataset=$(realpath "$1")
run=$(realpath "$2")
repo=$(cd "$(dirname "$0")/.." && pwd)
python="$repo/../venv_SCAPES/bin/python"

cmd="cd $repo && setsid nohup $python -u -c \"from SCAPES.pipeline import run_training; run_training('$dataset', '$run/model', config_dir='$run/config')\" > $run/train.log 2>&1 < /dev/null &"
echo "$cmd" > "$run/command.txt"
cd "$repo"
setsid nohup "$python" -u -c "from SCAPES.pipeline import run_training; run_training('$dataset', '$run/model', config_dir='$run/config')" \
    > "$run/train.log" 2>&1 < /dev/null &
echo $! > "$run/pid"
echo "started pid $(cat "$run/pid"), log $run/train.log"
