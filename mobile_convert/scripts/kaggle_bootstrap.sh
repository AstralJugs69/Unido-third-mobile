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

# Prefer GPU ONNX Runtime when a CUDA device is available in the notebook.
# Keep CPU ORT on TPU/CPU runtimes.
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[mobile_convert] GPU detected. Installing onnxruntime-gpu..."
  pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
  pip install -q --upgrade onnxruntime-gpu
else
  echo "[mobile_convert] No GPU detected. Installing CPU onnxruntime..."
  pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
  pip install -q --upgrade onnxruntime
fi

python - <<'PY'
import onnxruntime as ort
providers_fn = getattr(ort, "get_available_providers", None)
providers = providers_fn() if callable(providers_fn) else ["CPUExecutionProvider"]
print("[mobile_convert] ONNX Runtime providers:", providers)
print("[mobile_convert] InferenceSession available:", hasattr(ort, "InferenceSession"))
PY
