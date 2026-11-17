#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT/.."
export MUJOCO_GL=egl OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/vendor/openpi/packages/openpi-client/src:$ROOT/../work/pi05_setup/sim_client${PYTHONPATH:+:$PYTHONPATH}"
exec "${SIM_PYTHON:-$HOME/.conda/envs/realman-mujoco/bin/python}" "$ROOT/pipeline.py" evaluate "$@"
