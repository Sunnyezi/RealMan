#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /opt/conda/etc/profile.d/conda.sh
conda activate realman-mujoco
mode=${1:-check}; shift || true
export OMP_NUM_THREADS=1
case "$mode" in
 live)
  export DISPLAY=${DISPLAY:-:15}
  export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
  export MUJOCO_GL=glfw LIBGL_ALWAYS_SOFTWARE=1
  exec python grid_live.py "$@";;
 demo)
  export MUJOCO_GL=egl
  unset LIBGL_ALWAYS_SOFTWARE
  exec python validate_script.py --live-loop "$@";;
 check|train|eval|grid-eval|benchmark|deploy)
  export MUJOCO_GL=egl
  unset LIBGL_ALWAYS_SOFTWARE
  case "$mode" in check) script=check.py;; train) script=train.py;; eval) script=evaluate.py;; grid-eval) script=grid_play.py;; benchmark) script=benchmark.py;; deploy) script=deployment.py;; esac
  exec python "$script" "$@";;
 *) echo 'Usage: bash grasp.sh {check|live|demo|train|eval|grid-eval|benchmark|deploy} [options]';exit 2;;
esac
