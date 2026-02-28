#!/usr/bin/env bash
set -euo pipefail

python -m mobile_convert.cli run-core \
  --config mobile_convert/mobile_convert/config/default.yaml \
  --preset mobile_convert/mobile_convert/config/kaggle_tpu_v5e8.yaml \
  "$@"
