#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo 'Usage: bash pi05/train_visible.sh REPO_ID EXPERIMENT RAW_DATA [STEPS=100] [BATCH=1]' >&2
  exit 2
fi
REPO=$1 EXPERIMENT=$2 RAW=$(realpath "$3") STEPS=${4:-100} BATCH=${5:-1}
TASK=${PI05_TASK:-hold}
[[ "$TASK" == hold || "$TASK" == place ]] || { echo 'PI05_TASK must be hold or place' >&2; exit 2; }
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
[[ "$EXPERIMENT" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid experiment name' >&2; exit 2; }
RUN="$PWD/runs/$EXPERIMENT"
mkdir -p "$RUN"
SIM_PYTHON=${SIM_PYTHON:-$HOME/.conda/envs/realman-mujoco/bin/python}
OPENPI_PYTHON="$PWD/vendor/openpi/.venv/bin/python"
export OMP_NUM_THREADS=1 WANDB_MODE=disabled PYTHONUNBUFFERED=1
# Share the GPU with existing tasks: 55% of this server's 48 GiB GPU, rather than 90%.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.55}
SCENE=()
[[ "$TASK" == place ]] && SCENE=(--scene "$PWD/place/scene.xml")
bash ../grasp/grasp.sh live --envs 1 --stream "$RUN/live_state.npz" "${SCENE[@]}" > "$RUN/viewer.log" 2>&1 &
VIEWER_PID=$!
MUJOCO_GL=egl "$SIM_PYTHON" view_data.py --raw "$RAW" --stream "$RUN/live_state.npz" --task "$TASK" \
  > "$RUN/replay.log" 2>&1 &
REPLAY_PID=$!
cleanup() { kill "$REPLAY_PID" "$VIEWER_PID" 2>/dev/null || true; }
trap cleanup EXIT
READY=0
for _ in {1..60}; do
  kill -0 "$VIEWER_PID"
  kill -0 "$REPLAY_PID"
  if "$SIM_PYTHON" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); s=json.loads(p.read_text()) if p.exists() else {}; sys.exit(0 if s.get("active_envs",0)>0 and s.get("fresh") else 1)' "$RUN/grid_status.json"; then
    READY=1
    break
  fi
  sleep 0.5
done
[[ "$READY" == 1 ]] || { echo "Viewer did not receive fresh demonstration frames" >&2; exit 1; }
EXTRA=()
if [[ -n "${PI05_INIT_CHECKPOINT:-}" ]]; then
  EXTRA+=(--init-checkpoint "$PI05_INIT_CHECKPOINT" --peak-lr "${PI05_PEAK_LR:-1e-5}")
fi
"$OPENPI_PYTHON" openpi_run.py train --repo-id "$REPO" --experiment "$EXPERIMENT" \
  --steps "$STEPS" --batch-size "$BATCH" --task "$TASK" "${EXTRA[@]}" 2>&1 | tee "$RUN/train.log"
