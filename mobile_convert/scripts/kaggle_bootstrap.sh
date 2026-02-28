#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python -m pip install --upgrade pip
pip install -e "${REPO_ROOT}/mobile_convert"

# Torch/XLA warns about tensorflow package conflicts in notebook runtimes.
# Keep CPU-only tensorflow to avoid TPU/XLA runtime clashes.
pip uninstall -y tensorflow >/dev/null 2>&1 || true
pip install -q tensorflow-cpu >/dev/null 2>&1 || true
