#!/usr/bin/env bash
set -euo pipefail

XLA_WORLD_SIZE="${XLA_WORLD_SIZE:-8}"

python -m mobile_convert.cli run-core \
  --config mobile_convert/mobile_convert/config/default.yaml \
  --preset mobile_convert/mobile_convert/config/kaggle_tpu_v5e8.yaml \
  --set runtime.xla_world_size="${XLA_WORLD_SIZE}" \
  "$@"
