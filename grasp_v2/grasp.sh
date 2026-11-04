#!/usr/bin/env bash
set -euo pipefail
cd /home/szy2025/Desktop/RealMan/grasp_v2
source /opt/conda/etc/profile.d/conda.sh
conda activate realman-mujoco
mode=${1:-check}; shift || true
export OMP_NUM_THREADS=1
case "$mode" in
 live)
  export DISPLAY=${DISPLAY:-:15}
  export XAUTHORITY=/home/szy2025/.Xauthority
  export MUJOCO_GL=glfw LIBGL_ALWAYS_SOFTWARE=1
  exec python live.py "$@";;
 check|train|eval)
  export MUJOCO_GL=egl
  unset LIBGL_ALWAYS_SOFTWARE
  case "$mode" in check) script=check.py;; train) script=train.py;; eval) script=evaluate.py;; esac
  exec python "$script" "$@";;
 *) echo 'Usage: bash grasp.sh {check|live|train|eval} [options]';exit 2;;
esac
