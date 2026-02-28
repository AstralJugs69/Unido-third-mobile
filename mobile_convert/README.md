# mobile_convert

CLI-first package for retraining, exporting, and benchmarking the rice-quality model for mobile deployment.

## Install (local)

```bash
pip install -e ./mobile_convert
```

## Two-cell Kaggle kickstart

Use TPU v5e-8 runtime in Kaggle before running these cells.

### Cell 1: sync repo + install
```bash
%cd /kaggle/working
# If repo is already copied into /kaggle/working, skip clone and just cd into it.
# !git clone <YOUR_REPO_URL> Unido_third_mobile
%cd /kaggle/working/Unido_third_mobile
!bash mobile_convert/scripts/kaggle_bootstrap.sh
```

### Cell 2: run full core pipeline
```bash
%cd /kaggle/working/Unido_third_mobile
# Optional tuning: set TPU process count. Usually keep at 8 for v5e-8.
%env XLA_WORLD_SIZE=8
!bash mobile_convert/scripts/kaggle_train_export.sh
```

### Optional: pass extra overrides in cell 2
```bash
!bash mobile_convert/scripts/kaggle_train_export.sh \
  --set training.epochs=20 \
  --set tiling.grid_rows=5 \
  --set tiling.grid_cols=7
```

## CLI commands

```bash
python -m mobile_convert.cli train --config mobile_convert/mobile_convert/config/default.yaml
python -m mobile_convert.cli eval --config mobile_convert/mobile_convert/config/default.yaml --ckpt /path/to/best.ckpt
python -m mobile_convert.cli export-onnx --config mobile_convert/mobile_convert/config/default.yaml --ckpt /path/to/best.ckpt --out /path/to/model_fp32.onnx
python -m mobile_convert.cli convert-fp16 --in /path/to/model_fp32.onnx --out /path/to/model_fp16.onnx
python -m mobile_convert.cli benchmark-ort --model /path/to/model_fp32.onnx --config mobile_convert/mobile_convert/config/default.yaml
python -m mobile_convert.cli run-core --config mobile_convert/mobile_convert/config/default.yaml
```

## Notes

- Core scope includes: warm-start student training, eval gates, ONNX FP32/FP16, ORT benchmark.
- `quantize-int8` is intentionally a stub in this phase.
- XLA spawn is used when `runtime.mode=xla`, with safe fallback to CUDA/CPU when XLA is unavailable.
