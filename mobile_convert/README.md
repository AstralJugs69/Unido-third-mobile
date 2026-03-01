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
!if [ -d Unido-third-mobile ]; then \
  cd Unido-third-mobile && git pull origin master; \
else \
  git clone https://github.com/AstralJugs69/Unido-third-mobile.git; \
fi
%cd /kaggle/working/Unido-third-mobile
!bash mobile_convert/scripts/kaggle_bootstrap.sh
```

### Cell 2: run full core pipeline
```bash
%cd /kaggle/working/Unido-third-mobile
!bash mobile_convert/scripts/kaggle_train_export.sh
```

`kaggle_train_export.sh` will automatically download the teacher checkpoint and image dataset if they are missing.

### Optional: pass extra overrides in cell 2
```bash
!bash mobile_convert/scripts/kaggle_train_export.sh \
  --set training.epochs=20 \
  --set tiling.grid_rows=5 \
  --set tiling.grid_cols=7
```

### Resume a stopped run to a higher epoch target
```bash
%cd /kaggle/working/Unido-third-mobile
!LATEST_CKPT=$(ls -1dt mobile_convert/runs/* 2>/dev/null | head -n 1)/last.ckpt; \
echo "Resuming from: ${LATEST_CKPT}"; \
bash mobile_convert/scripts/kaggle_train_export.sh \
  --set training.checkpoint="${LATEST_CKPT}" \
  --set training.resume_optimizer=false \
  --set training.epochs=30
```

### TPU utilization profile (higher throughput, safe defaults)
```bash
%cd /kaggle/working/Unido-third-mobile
!bash mobile_convert/scripts/kaggle_train_export.sh \
  --set runtime.use_xla_spawn=true \
  --set runtime.xla_spawn_start_method=spawn \
  --set training.batch_size=3 \
  --set training.grad_accum=1 \
  --set training.num_workers=8
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
- Kaggle preset enables XLA spawn with automatic fallback to single-process XLA if spawn initialization fails.
