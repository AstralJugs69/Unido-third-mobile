# Execution Plan for Retraining and Deploying a Mobile Rice-Quality Model

## Executive recommendation

**Executive recommendation (top end-to-end paths)**

### Path that is most likely to win under tight timelines
**Warm-start fine-tuning of the 256/reduced-grid student from the 512/dense-grid teacher weights, with multi-task loss auto-weighting, trained end-to-end on TPU v5e-8 using PyTorch/XLA (BF16 AMP), then export tile-model to ONNX (PyTorch Dynamo ONNX exporter), ship FP32 + FP16, and only attempt INT8 if drift gates pass.** citeturn5search0turn4search34turn10view2turn6view2turn9view4turn9view0

- **Why it helps**
  - Warm-starting from the existing model removes “from scratch” convergence risk and targets the *pipeline shift* (tile size/grid change) instead of relearning rice-domain features. (This is the same underlying principle as resolution-mismatch fixes: a short fine-tune at the *deployment resolution* can recover much of the loss.) citeturn5search0
  - Multi-task loss auto-weighting reduces manual tuning churn when the output heads respond differently to reduced field-of-view/texture detail (counts vs morphology/measurements). citeturn4search34turn4search7
  - TPU BF16 AMP is officially supported in PyTorch/XLA and is the default “low-risk speed lever” for v5e training throughput and memory headroom. citeturn10view2
  - ONNX export via `torch.onnx.export(..., dynamo=True)` is now the *recommended default* in PyTorch docs, and ORT provides a production path to optimize and package models for mobile. citeturn6view2turn6view3

- **Expected speed / accuracy impact**
  - **Training speed:** BF16 AMP typically increases TPU throughput and/or allows higher batch size before OOM (primary lever: reduced activation memory). citeturn10view2turn19search0
  - **Inference latency:** the dominant compute reduction comes from lowering (a) number of tiles and (b) tile pixel area; ConvNet compute scales roughly with pixels processed, so halving tile width/height reduces per-tile area ~4×, and reducing tile count gives an additional linear reduction. (Recommendation: treat this as a *measured* gain with the Android simulation protocol below, not a guess.) citeturn26view0turn23view5
  - **Accuracy:** warm-start + “deployment-resolution fine-tune” is specifically designed to recover accuracy lost from a resolution/pipeline shift. citeturn5search0

- **Implementation complexity:** **Low–Moderate** (you will touch: tiling pipeline, training dataloader shapes, loss weighting, TPU launch wrapper, ONNX export script).
- **Risk level:** **Low** (all core pieces are officially supported: PyTorch/XLA BF16 AMP + debugging, PyTorch ONNX exporter (Dynamo), ORT conversions/optimizations). citeturn10view2turn18view0turn6view2turn6view3

---

### Path that is most likely to preserve accuracy if the pipeline change is harsh
**Add teacher→student regression distillation (teacher = 512/dense-grid model, student = 256/reduced-grid), using output/target distillation as the default, and only add feature distillation if needed; keep the rest identical to the first path (TPU BF16 + ONNX export + FP32/FP16).** citeturn4search0turn25search2turn25search0turn9view0turn6view2

- **Why it helps**
  - Knowledge distillation is explicitly intended to transfer behavior from a larger/costlier model into a cheaper deployable model. citeturn4search0
  - For **regression outputs**, there are KD formalisms designed beyond softmax/KL—e.g., teacher-guided losses and multi-output setups that improve student training in regression settings. citeturn25search2
  - Feature distillation (FitNets-style “hints”) is a known extension when logits/outputs alone are insufficient, but it increases engineering and tuning cost. citeturn25search0

- **Expected speed / accuracy impact**
  - **Accuracy:** typically improves over “student supervised only” in compression settings, especially when the student’s input/pipeline is information-reduced; output KD is usually the best “accuracy per engineering-hour” starting point. citeturn4search0turn25search2
  - **Training speed:** if teacher runs online during training, wall-clock nearly doubles; therefore, the recommended plan is **offline teacher inference** (precompute teacher targets per training sample/tile) or caching teacher outputs. citeturn21view1turn18view0

- **Implementation complexity:** **Moderate** (need teacher inference/caching + KD loss plumbing).
- **Risk level:** **Moderate** (conceptually well-founded, but increased surface area: data alignment between teacher tiles and student tiles; caching format/IO can become the bottleneck on TPU host). citeturn6view1turn21view1turn18view0

**Support classification of the main building blocks (required)**  
- **Officially supported:** PyTorch/XLA BF16 AMP + sync-free optimizers, recompilation guidance and metrics tooling; PyTorch Dynamo-based ONNX export; ORT Float16 tools; ORT quantization APIs; ORT mobile formats and Android EP docs. citeturn10view2turn6view1turn18view0turn6view2turn9view4turn9view0turn6view3turn6view5turn9view3  
- **Community-tested workaround:** regression KD and feature distillation recipes (supported by primary papers, but not “one official recipe”); Kaggle-specific TPU launch notebooks and TPU environment quirks. citeturn4search0turn25search2turn25search0turn16search3turn16search4  
- **Experimental/risky:** exporting full-image tiling inside ONNX graph (dynamic slicing/stacking often causes exporter/shape issues); aggressive INT8 without calibration discipline; any TPU SPMD auto-sharding refactors under competition deadlines. citeturn6view1turn6view2turn9view0turn21view0

## Decision and experiment matrices

**Decision matrix table**

| Approach | Speed gain | Accuracy risk | Engineering effort | TPU fit | Mobile fit |
|---|---:|---:|---:|---:|---:|
| Warm-start student (256/reduced-grid) from teacher weights + supervised fine-tune | High (primary lever: fewer/smaller tiles) citeturn26view0 | Medium (pipeline shift can hurt counts/morphology differently) citeturn5search0 | Low–Moderate | High (static shapes + BF16 AMP are first-class) citeturn10view2turn6view1 | High (clean ONNX tile-model export; ORT mobile tools available) citeturn6view2turn6view3 |
| Warm-start + multi-task loss auto-weighting (uncertainty weighting or GradNorm) | Same as above | Lower (reduces manual loss-reweight risk after resolution/FOV shift) citeturn4search34turn4search7 | Moderate (loss plumbing + logging) | High | High |
| Output/target regression KD: teacher logits → student outputs (offline cached) + supervised loss | Same inference gain | Low–Medium (usually preserves teacher behavior better) citeturn4search0turn25search2 | Moderate | Medium–High (extra IO; must avoid host bottlenecks) citeturn18view0turn21view1 | High |
| Output KD + feature distillation (FitNets hints) | Same inference gain | Lowest (when output KD alone insufficient) but tuning-heavy citeturn25search0 | High | Medium (more forward hooks/feature tensors) | High |
| Ship FP32 ONNX only (skip FP16/INT8) | Moderate (best accuracy baseline, but not fastest on-device) citeturn26view0 | Lowest | Low | N/A | Medium (may be too slow on low-end devices) |
| FP16 conversion for mobile accelerators (NNAPI fp16 relaxation or GPU) | Potentially high on devices where FP16 is accelerated; **not a CPU guarantee** citeturn9view4turn6view5 | Medium (FP16 can reduce accuracy; NNAPI docs warn) citeturn6view5 | Low–Moderate | N/A | High if NNAPI/GPU path is used; otherwise limited citeturn9view4turn6view5 |
| INT8 via ORT static quantization (QDQ, per-channel) | Potentially high on hardware with efficient INT8; can be worse on older devices citeturn17view0 | Highest (needs calibration + drift gates) citeturn9view0turn17view0 | Moderate | N/A | Medium–High (depends on EP support) citeturn8search3turn6view5 |

---

**Experiment matrix (minimal but sufficient)**  
(Use your existing validation metric as the primary score; add the per-target drift checks below as secondary “safety” metrics.)

| Experiment | Change | Why / what it isolates | Stop rule | Success threshold |
|---|---|---|---|---|
| Teacher baseline | Keep 512 tile + 8×6 grid (current) | Establish reference accuracy + per-target error distribution | 1 full eval pass | Baseline reference |
| Student naïve | 256 tile + reduced grid, load teacher weights, same losses | Measures raw degradation from pipeline change without extra tricks | Early-stop if no improvement in primary metric for 1–2 evals | Primary metric ≥ 0.97× teacher; no target > 1.10× teacher MAE/RMSE |
| Student + head warm-up | Same as “Student naïve” but freeze backbone for short warm-up then unfreeze | Tests if stabilizing early training improves accuracy per unit time | Same as above | Beats “Student naïve” by ≥ small but consistent margin on primary metric; reduced variance across seeds |
| Student + uncertainty loss weighting | Add Kendall-style homoscedastic uncertainty weights across tasks | Tests whether automatic loss rebalancing recovers accuracy after FOV/resolution shift | Same as above | Primary metric ≥ “Student + head warm-up”; fewer target regressions citeturn4search34 |
| Student + GradNorm (alt) | Replace loss weighting with GradNorm | Alternative balancing if uncertainty weighting unstable | Same as above | Either matches uncertainty weighting or improves worst-case targets citeturn4search7 |
| Student + output KD | Add regression KD loss on the 15 outputs (teacher targets cached offline) | Tests largest expected accuracy recovery lever without architectural change | Same as above | Primary metric ≥ 0.985× teacher; no target > 1.05× teacher MAE/RMSE citeturn4search0turn25search2 |
| Student + feature KD (only if needed) | Add FitNets-style intermediate feature matching on one ConvNeXt stage | Tests whether representation matching is needed beyond output KD | Same as above | Only keep if it beats output KD with acceptable added complexity citeturn25search0 |

**Ablation discipline (to keep it fast):** only change one axis at a time, and use XLA recompilation metrics as a hard guardrail—if your experiment triggers frequent compiles, it is not a fair model comparison. citeturn6view1turn18view0

## Copy-paste checklists

**TPU optimization checklist (copy-paste ready)**  
(Goal: maximize TPU v5e-8 utilization while avoiding OOM and recompilation/compile stalls.)

### TPU environment and launch
- [ ] Enable TPU v5e-8 in notebook settings and confirm device count at runtime (do not assume). citeturn16search27turn21view0  
- [ ] Confirm Kaggle TPU v5e-8 memory characteristics (v5e-8 uses 8 chips × 16GB HBM = 128GB total). citeturn7search0  
- [ ] Use multi-process replication via `xmp.spawn(...)` (PyTorch/XLA API) and set `nprocs` to maximum device count (or omit to use default max). citeturn23view0turn21view1  
- [ ] Ensure **all XLA device construction** happens inside the spawned function; this is a documented failure mode when done at global scope. citeturn16search4  

### Static shapes to prevent recompilation storms
- [ ] Enforce fixed tile shape (always 256×256) and fixed channel order throughout train/eval. citeturn6view1turn21view1  
- [ ] Make batch shapes static:
  - [ ] Use `drop_last=True` in DataLoader (otherwise the final smaller batch can change shapes and trigger recompiles). citeturn6view1  
  - [ ] Keep the same number of tiles per sample (pad or mask if you must vary). citeturn6view1turn21view1  
- [ ] Avoid Python-side logging that *materializes tensors* mid-step; XLA docs warn that printing/logging/checkpointing can block tracing and cause slowdowns via host-device transfers. citeturn21view1turn18view0  

### BF16 mixed precision and optimizer choices
- [ ] Use PyTorch/XLA AMP BF16 autocast for forward + loss only; gradient scaling is not needed on TPUs (explicitly documented). citeturn10view2  
- [ ] Prefer PyTorch/XLA “sync-free” optimizers when available to reduce host-device synchronization overhead. citeturn10view2  

### Input pipeline to avoid host bottlenecks
- [ ] Wrap your CPU DataLoader with `ParallelLoader`/`MpDeviceLoader` patterns so host-to-device transfer is overlapped and prefetched. (ParallelLoader explicitly supports background upload and configurable prefetch/threads.) citeturn21view2turn23view1  
- [ ] Tune prefetch knobs (`loader_prefetch_size`, `device_prefetch_size`, `host_to_device_transfer_threads`) only after verifying static shapes and stable compilation. citeturn21view2turn18view0  

### OOM prevention playbook
- [ ] First lever: reduce per-core batch until stable, then use BF16 AMP to reclaim headroom. citeturn10view2  
- [ ] Second lever: activation checkpointing (`torch.utils.checkpoint`) trades compute for memory (official PyTorch API). citeturn19search0  
- [ ] Be cautious with gradient accumulation on XLA: it can increase graph size or interact poorly with compilation boundaries; XLA issues discuss OOM in accumulation loops and suggest careful placement of `xm.mark_step()` barriers. citeturn20search2turn20search18turn21view1  

### Measure real TPU utilization (don’t guess)
- [ ] Print / save XLA metrics reports and use PT_XLA debug tooling for compile/execution analysis; docs explicitly recommend metrics reports first. citeturn18view0turn19search23  
- [ ] Enable `PT_XLA_DEBUG_LEVEL=2` during development to surface frequent `CompileTime`, `TransferFromDeviceTime`, and “ops not lowered”. citeturn18view0  
- [ ] If compile latency dominates repeated runs, initialize the persistent compilation cache *before any computations* using `torch_xla.runtime.initialize_cache(...)`. citeturn21view0  

**Fastest-to-first-good-checkpoint recipe (practical)**  
- [ ] Start with the student initialized from teacher weights; run 1–2 short validation cycles to confirm no recompilation storms (compile once, then reuse). citeturn6view1turn21view1  
- [ ] Warm up with backbone frozen (heads only) briefly, then unfreeze with conservative LR; this reduces early catastrophic drift risk while the heads adapt to new tile statistics. (If this underperforms, revert to full fine-tune + uncertainty weighting.) citeturn24search21turn4search34  

---

**ONNX export + validation checklist (copy-paste ready)**

### Model boundary decision (critical)
- [ ] **Recommended export boundary (low risk):** export the **tile model only** (input: N×3×256×256 → output: N×15, or N×(9+6) as two outputs). Keep tiling + aggregation outside the graph (Python/Java/Kotlin), to avoid dynamic slicing/stacking inside ONNX. This minimizes exporter complexity and reduces the operator surface area needed for ORT mobile builds. citeturn6view2turn6view3  
- [ ] Optional (moderate risk): export tile model + simple reductions (Sum/Mean across tiles) if your aggregation is static and ONNX-friendly; verify carefully because shape handling mistakes are common. citeturn6view2turn22search0  

### Export (PyTorch, officially recommended path)
- [ ] Use `torch.onnx.export(..., dynamo=True)` (this is the default and *recommended* ONNX export path in PyTorch docs). citeturn6view2  
- [ ] Set `model.eval()` and use representative dummy inputs with the same shape policy as deployment. citeturn6view2turn6view1  
- [ ] Use static spatial dims (256×256) and **dynamic batch axis only** via `dynamic_axes` or `dynamic_shapes` to avoid dynamic-shape complexity while keeping throughput flexibility. citeturn23view3turn6view2  
- [ ] Enable exporter verification during development where feasible (PyTorch exporter provides verification hooks/flags). citeturn6view2  

### Validate ONNX correctness
- [ ] Run ONNX structural validation with `onnx.checker.check_model(...)` (official ONNX API). citeturn22search0  
- [ ] Run numerical parity checks between PyTorch and ONNX Runtime on a fixed seed batch (same preprocessing, same tiling). citeturn22search4  

### Optimize for deployment
- [ ] Use ONNX Runtime graph optimizations (online or offline). ORT explicitly documents optimization levels and the ability to save optimized artifacts. citeturn22search1turn6view3  
- [ ] Convert to ORT format (`.ort`) for mobile/reduced-size builds using `convert_onnx_models_to_ort`; ORT format is explicitly intended for size-constrained environments like mobile and supports conversion via the provided script. citeturn6view3  
- [ ] If you need minimal-build sizing later, keep the `required_operators.config` outputs from ORT conversion (same script) to drive reduced operator builds. citeturn6view3  

---

**Quantization checklist with go/no-go gates (copy-paste ready)**

### Baselines and variants to produce
- [ ] **FP32 ONNX** (baseline correctness + accuracy reference). citeturn6view2turn22search0  
- [ ] **FP16 ONNX** (size reduction and potential accelerator speedups); use ONNX converter tools (`convert_float_to_float16`). citeturn9view4  
- [ ] **INT8 ONNX** only if accuracy drift gates pass; prefer static PTQ for CNNs as recommended by ORT docs. citeturn9view2turn17view0  

### FP16: when it helps vs hurts (Android-specific reality)
- [ ] ORT float16 conversion can improve performance on **some GPUs** and reduce model size; however ORT docs state the **CPU** version does not support float16 ops (critical for low-end Android CPU-only paths). citeturn9view4  
- [ ] If using NNAPI EP, consider FP16 relaxation (`NNAPI_FLAG_USE_FP16`), but NNAPI docs explicitly warn it may reduce accuracy. citeturn6view5  

### INT8: choose method and format
- [ ] **Do not expect dynamic quantization to be a strong default for CNNs**: ORT recommends dynamic for RNN/transformers and static for CNNs. citeturn9view2turn17view0  
- [ ] Use **static quantization** (`quantize_static`) with a representative calibration dataset; ORT supports MinMax/Entropy/Percentile calibration. citeturn9view1  
- [ ] Prefer **per-channel quantization** if accuracy loss is large; ORT docs explicitly say it can improve accuracy for models with large weight ranges. citeturn17view0  
- [ ] Choose representation: QOperator vs QDQ; ORT documents both formats and how they represent quantized graphs. citeturn9view0  

### Quantization pre-processing discipline (to avoid “mystery drift”)
- [ ] Perform graph optimization in a **separate pre-processing step**, not during quantization—ORT docs explicitly say optimization during quantization is not recommended because it complicates debugging accuracy loss. citeturn9view1  
- [ ] Use ORT’s quantization debugging guidance: compare weights/activations and exclude sensitive tensors/nodes or change calibration method. citeturn9view1turn22search2  

### Go / no-go gates (15-output guardrails)
Define gates against FP32 ONNX baseline on the same validation set and the same tiling pipeline.

- [ ] **Gate A (global):** primary validation metric degradation ≤ 1.5% relative vs FP32.  
- [ ] **Gate B (per-output):** for each of 15 targets, MAE/RMSE degradation ≤ 5% relative; and no catastrophic outlier bucket (e.g., worst 1% samples) worsens by > 10% relative.  
- [ ] **Gate C (stability):** output sign/constraints respected (e.g., non-negative counts) and post-processing does not introduce invalid values. citeturn22search0turn9view0  
- [ ] **Gate D (performance reality check):** measure on-device or in constrained simulation; ORT explicitly notes quantization speedups depend on hardware instructions and quantization overhead can **worsen** performance on older hardware. citeturn17view0turn26view0  

If any gate fails:
- [ ] Retry static quantization with per-channel weights and alternate calibration method. citeturn17view0turn9view1  
- [ ] Exclude the most sensitive nodes (quantization debugging workflow) and re-evaluate. citeturn9view1turn22search2  
- [ ] Escalation (higher effort): consider QAT in the original framework and re-export; ORT notes it can run QAT-produced quantized models but does not provide retraining. citeturn9view0turn9view2  

## Android simulation and risk register

**Android simulation protocol**

### What you *can* approximate reliably in notebooks
- **Relative model comparisons** (FP32 vs FP16 vs INT8; different grid sizes; different batch-tiling policies) using ONNX Runtime CPU sessions with controlled threading and consistent benchmarking methodology. citeturn23view5turn26view0turn22search4  
- **Operator-level hotspots and overheads** using ORT profiling output (JSON trace) via in-code profiling (enable profiling) or `onnxruntime_perf_test` plus `-p`. citeturn26view0turn22search4  
- **Graph optimization effects** (offline optimized ONNX / ORT format) and their impact on CPU latency and load time. citeturn6view3turn22search1  

### What you *cannot* simulate reliably online (must validate on physical devices)
- **NNAPI partitioning and real accelerator behavior** (DSP/NPU/GPU vendor drivers, operator coverage, memory bandwidth, thermal throttling). ORT’s NNAPI EP placement and flags are device-dependent, and NNAPI itself is a hardware abstraction with varying drivers. citeturn6view5turn8search10turn8search20  
- **True ARM CPU throughput** on low/mid-range Android: x86 notebook CPU results are not an absolute predictor; treat them as directional only. ORT’s XNNPACK EP is designed for Arm®-based platforms, which is precisely why x86-only testing is limited. citeturn9view3turn8search0  

### Minimal pre-device benchmark suite (high correlation in practice)
1. **Model artifact checks**
   - FP32 ONNX passes `onnx.checker` and ORT inference parity. citeturn22search0turn22search4  
   - Optimized `.ort` model loads successfully (if using ORT format). citeturn6view3turn22search4  

2. **Reproducible ORT CPU microbenchmark**
   - Fix batch = 1, tiles per image = deployment value, warm-up runs, then timed runs; set ORT thread counts explicitly. citeturn23view5turn26view0  

3. **Profiling snapshot**
   - Enable ORT profiling and archive the JSON traces for each variant (FP32/FP16/INT8) to compare operator hotspots and overheads. citeturn26view0  

4. **Quantization drift report**
   - Per-target drift gates (above) + worst-case slice analysis (high-count images, boundary-heavy images). citeturn9view1turn17view0  

---

**Risk register (top failure modes, detection signals, mitigations)**

1. **Accuracy regression due to reduced field-of-view and/or sampling bias from fewer tiles**  
   - **Signal:** systematic under/over-count on images with uneven spatial distributions; per-target drift spikes for count heads.  
   - **Mitigation:** warm-start + deployment-resolution fine-tune; add multi-task loss balancing; if still failing, add teacher→student output KD (cached). citeturn5search0turn4search34turn4search7turn25search2  
   - **Risk level:** Medium.

2. **Recompilation storms on TPU (compile stalls / low utilization)**  
   - **Signal:** XLA metrics show frequent `CompileTime`, slow steps, many different graph hashes; recompilation docs emphasize shape changes trigger recompiles. citeturn6view1turn18view0turn21view1  
   - **Mitigation:** enforce static shapes (tile size fixed, batch fixed, `drop_last=True`), avoid data-dependent control flow; use XLA debug tooling (`PT_XLA_DEBUG_LEVEL=2`). citeturn6view1turn18view0  
   - **Risk level:** Medium–High until stabilized.

3. **Host input pipeline bottleneck (TPU idle)**  
   - **Signal:** profiling shows TPU gaps; XLA reports `TransferFromServer`/host-side delays; training throughput does not scale with batch. citeturn21view1turn18view0  
   - **Mitigation:** `ParallelLoader`/prefetch, increase DataLoader workers, reduce Python overhead, cache decoded tiles. citeturn21view2turn18view0  
   - **Risk level:** Medium.

4. **OOM on TPU despite smaller tiles (activation/graph growth, accumulation interactions)**  
   - **Signal:** HBM OOM; memory rises across steps; OOM triggered by accumulation loops. citeturn20search2turn19search0  
   - **Mitigation:** BF16 AMP, reduce per-core batch, activation checkpointing, rework gradient accumulation boundaries (`xm.mark_step()` placement) if used. citeturn10view2turn19search0turn21view1turn20search18  
   - **Risk level:** Medium.

5. **ONNX export mismatch or runtime breakage**  
   - **Signal:** `onnx.checker` failure, ORT inference parity mismatch vs PyTorch, unexpected dynamic shape behavior. citeturn22search0turn6view2turn22search4  
   - **Mitigation:** export tile-model boundary (simpler graph); use Dynamo exporter (default/recommended) and validate with ORT; keep shapes mostly static. citeturn6view2turn23view3  
   - **Risk level:** Medium.

6. **FP16/INT8 accuracy drift unacceptable**  
   - **Signal:** drift gates fail; counts become biased; morphology outputs shift; NNAPI fp16 relaxation reduces accuracy (documented risk). citeturn6view5turn9view4turn9view1  
   - **Mitigation:** FP16 mixed precision (block problematic ops), INT8 per-channel + calibration method sweep, quantization debugging exclusions; fall back to FP32 if device performance acceptable. citeturn9view4turn17view0turn9view1  
   - **Risk level:** Medium–High.

7. **Quantization yields no speedup on target devices**  
   - **Signal:** INT8 model slower or same speed; ORT docs warn old hardware may lack efficient int8 instructions and quant/dequant overhead can dominate. citeturn17view0  
   - **Mitigation:** only enable INT8 behind measured device benchmarks; prefer XNNPACK/NNAPI pathways when appropriate; keep FP32/FP16 available. citeturn9view3turn6view5turn17view0  
   - **Risk level:** Medium.

## Source appendix

**Source appendix (URL + one-line evidence summary; tag = Official / Maintainer / Community-tested)**  
(Each URL is provided in code formatting to comply with the “no raw links” constraint.)

### TPU v5e-8 on Kaggle and PyTorch/XLA execution
- **Official:** `https://www.kaggle.com/product-announcements/607202` — Kaggle announcement discusses TPU v5e-8 rollout and memory breakdown (8×16GB = 128GB). citeturn7search0  
- **Official:** `https://www.kaggle.com/docs/tpu` — Kaggle TPU usage limits and basics (session/week limits). citeturn16search27  
- **Official:** `https://docs.pytorch.org/xla/master/perf/amp.html` — PyTorch/XLA AMP on TPU uses BF16; gradient scaling not needed; sync-free optimizers recommended. citeturn10view2  
- **Official:** `https://docs.pytorch.org/xla/release/r2.9/perf/recompilation.html` — XLA recompiles on shape changes; recompilation is expensive; guidance on avoiding it. citeturn6view1  
- **Official:** `https://docs.pytorch.org/xla/release/r2.6/learn/troubleshoot.html` — Metrics reports and PT_XLA_DEBUG_LEVEL tooling for compile/transfer/op-lowering diagnosis. citeturn18view0  
- **Official:** `https://docs.pytorch.org/xla/release/r2.7/learn/api-guide.html` — `torch_xla.runtime.initialize_cache` API for persistent compilation cache (must be called before computations). citeturn21view0  
- **Official:** `https://docs.pytorch.org/xla/release/2.2/index.html` — Explains tracing, cached compilation reuse, barriers (`xm.mark_step`), and why logging can slow execution. citeturn21view1  
- **Official:** `https://docs.pytorch.org/xla/master/_modules/torch_xla/distributed/parallel_loader.html` — ParallelLoader parameters for background host→device upload and prefetch sizing. citeturn21view2  
- **Official:** `https://docs.pytorch.org/xla/release/r2.5/_modules/torch_xla/distributed/xla_multiprocessing.html` — `xmp.spawn` semantics and `MpModelWrapper` description. citeturn23view0  
- **Community-tested:** `https://www.kaggle.com/code/wcromar/pytorch-xla-2-0-on-kaggle` — Kaggle notebook notes Kaggle uses PJRT runtime by default and demonstrates environment setup patterns. citeturn16search3turn15view0  
- **Community-tested:** `https://discuss.pytorch.org/t/enable-multiprocessing-on-pytorch-xla-for-tpu-vm/177673` — Practical warning: XLA device setup must occur inside the spawned function for multiprocessing. citeturn16search4  

### Fast retraining strategies after tile/grid changes
- **Primary (paper):** `https://arxiv.org/abs/1906.06423` — FixRes: shows resolution mismatch matters and a cheap fine-tune at target resolution can recover accuracy. citeturn5search0  
- **Primary (paper):** `https://ora.ox.ac.uk/objects/uuid:3903e961-25b0-40de-b797-1c455a198d5b` — Kendall et al. uncertainty weighting for balancing multi-task losses across regression/classification tasks. citeturn4search34  
- **Primary (paper):** `https://arxiv.org/abs/1711.02257` — GradNorm: adaptive gradient-based loss balancing for multitask networks. citeturn4search7  
- **Primary (paper):** `https://arxiv.org/abs/1503.02531` — Knowledge distillation motivation and classic teacher→student compression framing. citeturn4search0  
- **Primary (paper):** `https://arxiv.org/abs/2002.12597` — Regression-specific KD formalism and teacher-guided losses for regression tasks. citeturn25search2  
- **Primary (paper):** `https://arxiv.org/abs/1412.6550` — FitNets: feature-level “hint” distillation as an extension beyond output distillation. citeturn25search0  

### ONNX export, validation, ORT optimization, mobile packaging
- **Official:** `https://docs.pytorch.org/docs/stable/onnx.html` — `torch.onnx.export` API; `dynamo=True` recommended default; dynamic axes/shapes support. citeturn6view2turn23view3  
- **Official:** `https://onnx.ai/onnx/api/checker.html` — `onnx.checker.check_model` for structural model validation. citeturn22search0  
- **Official:** `https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html` — ORT graph optimization levels and offline/online optimization capability. citeturn22search1  
- **Official:** `https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html` — ORT format purpose (mobile/size constrained) and `convert_onnx_models_to_ort` script outputs. citeturn6view3  

### Quantization and FP16 model variants
- **Official:** `https://onnxruntime.ai/docs/performance/model-optimizations/float16.html` — FP16 conversion steps; notes CPU ORT lacks float16 op support; mixed precision tool requirements. citeturn9view4  
- **Official:** `https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html` — Dynamic vs static quantization, calibration methods, QDQ vs QOperator, per-channel guidance, and quantization debugging workflow. citeturn9view0turn9view1turn17view0  

### Android deployment and performance approximation
- **Official:** `https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html` — NNAPI EP requirements (Android 8.1+; recommended Android 9+) and FP16 relaxation flag warning. citeturn6view5  
- **Official:** `https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html` — XNNPACK EP purpose (optimized for Arm®-based) and explicit registration example for Android. citeturn9view3  
- **Official:** `https://onnxruntime.ai/docs/performance/tune-performance/threading.html` — ORT thread management controls (critical for reproducible CPU benchmarking). citeturn23view5  
- **Official:** `https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html` — ORT profiling (enable profiling, perf test tool, JSON traces, perf_view). citeturn26view0  
- **Official:** `https://developer.android.com/ndk/guides/neuralnetworks` — NNAPI definition as an Android C API for accelerated ML inference (hardware abstraction context). citeturn8search10