#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export DISPLAY=${DISPLAY:-:15}
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
export MUJOCO_GL=glfw
export LIBGL_ALWAYS_SOFTWARE=1
source /opt/conda/etc/profile.d/conda.sh
conda activate realman-mujoco
exec python view.py "$@"
