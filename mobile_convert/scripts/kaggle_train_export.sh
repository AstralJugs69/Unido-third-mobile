#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

# Ensure teacher checkpoint exists for warm-start.
if [ ! -f "${REPO_ROOT}/ultimate_tiled_multitask.pth" ]; then
  echo "[mobile_convert] Teacher checkpoint missing. Downloading..."
  python download_checkpoint.py
fi

# Ensure image dataset exists.
if [ ! -d "${REPO_ROOT}/Data/images/images" ] || [ -z "$(ls -A "${REPO_ROOT}/Data/images/images" 2>/dev/null)" ]; then
  echo "[mobile_convert] Image dataset missing. Downloading/extracting..."
  python download_unido_images.py
fi

python -m mobile_convert.cli run-core \
  --config mobile_convert/mobile_convert/config/default.yaml \
  --preset mobile_convert/mobile_convert/config/kaggle_tpu_v5e8.yaml \
  "$@"
