from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mobile_convert.utils.io import get_git_sha, load_config, make_run_dir, write_json, write_yaml
from mobile_convert.utils.logging import setup_logging
from mobile_convert.utils.seeding import set_seed


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to base YAML config")
    parser.add_argument("--preset", default=None, help="Optional preset YAML")
    parser.add_argument("--set", dest="set_values", action="append", default=[], help="Override config value, e.g. training.epochs=5")


def _load_and_prepare(args) -> tuple[dict, Path]:
    cfg = load_config(args.config, args.preset, args.set_values)
    set_seed(int(cfg["training"]["seed"]))
    run_dir = make_run_dir(cfg)
    cfg["meta"] = {"git_sha": get_git_sha()}
    write_yaml(run_dir / "config.snapshot.yaml", cfg)
    return cfg, run_dir


def cmd_train(args) -> int:
    from mobile_convert.train.xla_runner import run_training

    cfg, run_dir = _load_and_prepare(args)
    result = run_training(cfg, run_dir)
    write_json(run_dir / "train_result.json", result)
    logging.info("Training done. best_ckpt=%s", result["best_ckpt"])
    return 0


def cmd_eval(args) -> int:
    from mobile_convert.eval.drift_gates import evaluate_gates
    from mobile_convert.train.engine import evaluate_checkpoint
    from mobile_convert.train.xla_runner import resolve_device

    cfg, run_dir = _load_and_prepare(args)
    device = resolve_device(cfg["runtime"].get("mode", "auto"))
    metrics = evaluate_checkpoint(cfg, args.ckpt, device)
    write_json(run_dir / "eval_result.json", metrics)

    if args.baseline_metrics:
        import json

        with open(args.baseline_metrics, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        gate = evaluate_gates(
            candidate=metrics,
            baseline=baseline,
            global_rel_deg_max=float(cfg["gates"]["global_rel_deg_max"]),
            per_target_rel_deg_max=float(cfg["gates"]["per_target_rel_deg_max"]),
        )
        write_json(run_dir / "gate_report.json", gate)
    return 0


def cmd_export_onnx(args) -> int:
    from mobile_convert.export.onnx_export import export_fp32_onnx

    cfg, run_dir = _load_and_prepare(args)
    out = args.out or str(run_dir / "model_fp32.onnx")
    rep = export_fp32_onnx(cfg, args.ckpt, out)
    write_json(run_dir / "export_onnx_result.json", rep)
    return 0


def cmd_convert_fp16(args) -> int:
    from mobile_convert.export.fp16_convert import convert_to_fp16

    report = convert_to_fp16(args.input, args.out)
    logging.info("FP16 conversion done: %s", report["output"])
    return 0


def cmd_convert_fp16_mixed(args) -> int:
    from mobile_convert.export.fp16_convert import build_calibration_feed, convert_to_fp16_mixed
    from mobile_convert.utils.io import load_config

    cfg = load_config(args.config, args.preset, args.set_values)
    feed = build_calibration_feed(
        image_path=args.image,
        tile_size=int(cfg["tiling"]["tile_size"]),
        grid_rows=int(cfg["tiling"]["grid_rows"]),
        grid_cols=int(cfg["tiling"]["grid_cols"]),
        rice_comment=args.comment,
    )
    report = convert_to_fp16_mixed(
        in_path=args.input,
        out_path=args.out,
        feed_dict=feed,
        rtol=float(args.rtol),
        atol=float(args.atol),
        keep_io_types=not bool(args.no_keep_io_types),
    )
    logging.info("Mixed FP16 conversion done: %s", report["output"])
    return 0


def cmd_benchmark(args) -> int:
    from mobile_convert.bench.ort_bench import benchmark_and_save

    cfg, run_dir = _load_and_prepare(args)
    out = args.out or str(run_dir / "benchmark.json")
    result = benchmark_and_save(args.model, out, cfg)
    logging.info("Benchmark mean latency ms: %.4f", result["latency_ms"]["mean"])
    return 0


def cmd_quantize_int8(_args) -> int:
    logging.warning("quantize-int8 is not implemented in core scope.")
    return 2


def cmd_run_core(args) -> int:
    from mobile_convert.bench.ort_bench import benchmark_and_save
    from mobile_convert.export.fp16_convert import convert_to_fp16
    from mobile_convert.export.onnx_export import export_fp32_onnx
    from mobile_convert.train.engine import evaluate_checkpoint
    from mobile_convert.train.xla_runner import resolve_device, run_training

    cfg, run_dir = _load_and_prepare(args)

    train_result = run_training(cfg, run_dir)
    best_ckpt = train_result["best_ckpt"]

    device = resolve_device(cfg["runtime"].get("mode", "auto"))
    eval_result = evaluate_checkpoint(cfg, best_ckpt, device)
    write_json(run_dir / "eval_result.json", eval_result)

    fp32_path = run_dir / "model_fp32.onnx"
    export_result = export_fp32_onnx(cfg, best_ckpt, str(fp32_path))
    write_json(run_dir / "export_onnx_result.json", export_result)

    fp16_path = run_dir / "model_fp16.onnx"
    convert_result = convert_to_fp16(str(fp32_path), str(fp16_path))
    write_json(run_dir / "convert_fp16_result.json", convert_result)

    b1 = benchmark_and_save(str(fp32_path), str(run_dir / "benchmark_fp32.json"), cfg)
    b2 = benchmark_and_save(str(fp16_path), str(run_dir / "benchmark_fp16.json"), cfg)

    final = {
        "run_dir": str(run_dir),
        "best_ckpt": best_ckpt,
        "eval": eval_result,
        "benchmark_fp32": b1,
        "benchmark_fp16": b2,
    }
    write_json(run_dir / "run_core_result.json", final)
    logging.info("run-core complete: %s", run_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="mobile_convert CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    _add_common_args(p_train)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval")
    _add_common_args(p_eval)
    p_eval.add_argument("--ckpt", required=True)
    p_eval.add_argument("--baseline-metrics", default=None)
    p_eval.set_defaults(func=cmd_eval)

    p_export = sub.add_parser("export-onnx")
    _add_common_args(p_export)
    p_export.add_argument("--ckpt", required=True)
    p_export.add_argument("--out", default=None)
    p_export.set_defaults(func=cmd_export_onnx)

    p_fp16 = sub.add_parser("convert-fp16")
    p_fp16.add_argument("--in", dest="input", required=True)
    p_fp16.add_argument("--out", required=True)
    p_fp16.set_defaults(func=cmd_convert_fp16)

    p_fp16m = sub.add_parser("convert-fp16-mixed")
    p_fp16m.add_argument("--in", dest="input", required=True)
    p_fp16m.add_argument("--out", required=True)
    p_fp16m.add_argument("--config", required=True)
    p_fp16m.add_argument("--preset", default=None)
    p_fp16m.add_argument("--set", dest="set_values", action="append", default=[])
    p_fp16m.add_argument("--image", required=True, help="Calibration image path")
    p_fp16m.add_argument("--comment", default="White", help="Rice comment/meta class: Paddy|White|Brown")
    p_fp16m.add_argument("--rtol", type=float, default=0.01)
    p_fp16m.add_argument("--atol", type=float, default=0.001)
    p_fp16m.add_argument("--no-keep-io-types", action="store_true")
    p_fp16m.set_defaults(func=cmd_convert_fp16_mixed)

    p_bench = sub.add_parser("benchmark-ort")
    _add_common_args(p_bench)
    p_bench.add_argument("--model", required=True)
    p_bench.add_argument("--out", default=None)
    p_bench.set_defaults(func=cmd_benchmark)

    p_q = sub.add_parser("quantize-int8")
    p_q.set_defaults(func=cmd_quantize_int8)

    p_core = sub.add_parser("run-core")
    _add_common_args(p_core)
    p_core.set_defaults(func=cmd_run_core)
    return p


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
