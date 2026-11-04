#!/usr/bin/env bash
set -euo pipefail
cd /home/szy2025/Desktop/RealMan
export DISPLAY=${DISPLAY:-:15}
export XAUTHORITY=/home/szy2025/.Xauthority
export MUJOCO_GL=glfw
export LIBGL_ALWAYS_SOFTWARE=1
source /opt/conda/etc/profile.d/conda.sh
conda activate realman-mujoco
exec python view.py "$@"
