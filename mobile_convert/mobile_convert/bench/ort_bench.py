from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from mobile_convert.utils.io import write_json


def benchmark_onnx(model_path: str, n_tiles: int, tile_size: int, warmup_runs: int, timed_runs: int, intra_threads: int, inter_threads: int, batch_size: int = 1):
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(intra_threads)
    so.inter_op_num_threads = int(inter_threads)
    sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])

    stack = np.random.randn(batch_size, n_tiles, 3, tile_size, tile_size).astype(np.float32)
    meta = np.zeros((batch_size, 3), dtype=np.float32)
    meta[:, 1] = 1.0

    for _ in range(warmup_runs):
        sess.run(None, {"stack": stack, "meta": meta})

    times = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        sess.run(None, {"stack": stack, "meta": meta})
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times, dtype=np.float64)
    return {
        "model": str(Path(model_path).resolve()),
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "latency_ms": {
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
        },
    }


def benchmark_and_save(model_path: str, out_path: str, cfg: dict) -> dict:
    n_tiles = int(cfg["tiling"]["grid_rows"]) * int(cfg["tiling"]["grid_cols"])
    result = benchmark_onnx(
        model_path=model_path,
        n_tiles=n_tiles,
        tile_size=int(cfg["tiling"]["tile_size"]),
        warmup_runs=int(cfg["benchmark"]["warmup_runs"]),
        timed_runs=int(cfg["benchmark"]["timed_runs"]),
        intra_threads=int(cfg["benchmark"]["intra_op_threads"]),
        inter_threads=int(cfg["benchmark"]["inter_op_threads"]),
        batch_size=int(cfg["benchmark"]["batch_size"]),
    )
    write_json(out_path, result)
    return result
