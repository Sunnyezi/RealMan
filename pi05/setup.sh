#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
[[ "$(uname -s)" == Linux ]] || { echo 'Requires Linux + NVIDIA GPU' >&2; exit 1; }
nvidia-smi
mkdir -p ../work/pi05_setup vendor/openpi vendor/lerobot
SETUP=$(realpath ../work/pi05_setup)
PYTHON=${SIM_PYTHON:-$HOME/.conda/envs/realman-mujoco/bin/python}
if [[ ! -x "$SETUP/tools/bin/uv" ]]; then
  "$PYTHON" -m pip install --target "$SETUP/tools" uv==0.13.8
fi
if [[ ! -d "$SETUP/tools/google_crc32c" ]]; then
  "$PYTHON" -m pip install --no-deps --target "$SETUP/tools" google-crc32c==1.7.1
fi
UV="$SETUP/tools/bin/uv"
fetch_source() {
  local repository=$1 revision=$2 destination=$3 archive=$4
  if [[ ! -f "$destination/pyproject.toml" ]]; then
    curl -fsSL --retry 3 --connect-timeout 20 --max-time 300 \
      "https://api.github.com/repos/$repository/tarball/$revision" -o "$archive"
    tar -xzf "$archive" --strip-components=1 -C "$destination"
    sha256sum "$archive" >> "$SETUP/source_archives.sha256"
  fi
}
fetch_source Physical-Intelligence/openpi 215abfb217dbac7d5f1273282331b9b1866c0479 vendor/openpi "$SETUP/openpi.tar.gz"
fetch_source huggingface/lerobot 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5 vendor/lerobot "$SETUP/lerobot.tar.gz"
cd vendor/openpi
"$UV" venv --python 3.11 .venv --allow-existing
"$UV" export --frozen --no-annotate --no-header --no-emit-workspace --no-dev \
  > "$SETUP/requirements.txt"
"$PYTHON" - <<'PY'
from pathlib import Path
# Paths are resolved from pi05/vendor/openpi.
p = Path('../../../work/pi05_setup/requirements.txt').resolve()
text = p.read_text().splitlines()
source = Path('../lerobot').resolve().as_uri()
p.write_text('\n'.join('lerobot @ ' + source if x.startswith('lerobot @ ') else x for x in text) + '\n')
PY
"$UV" pip install --index-url "${PYPI_INDEX_URL:-https://pypi.org/simple}" --python .venv/bin/python -r ../../../work/pi05_setup/requirements.txt
"$UV" pip install --python .venv/bin/python --no-deps -e . -e packages/openpi-client
"$UV" pip check --python .venv/bin/python
if [[ ! -d "$SETUP/sim_client/msgpack" ]]; then
  "$PYTHON" -m pip install --no-deps --target "$SETUP/sim_client" msgpack==1.1.0
fi
echo 'OpenPI ready. Run fetch_weights.py to prefetch and verify the public pi05_base weights.'
