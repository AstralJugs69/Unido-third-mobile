from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from mobile_convert.models.architecture import OnnxTileModel, UltimateSpecialist
from mobile_convert.utils.io import write_json


def export_fp32_onnx(cfg: dict, ckpt_path: str, out_path: str) -> dict:
    model = UltimateSpecialist(cfg["model"]["name"], pretrained_backbone=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    tile_size = int(cfg["tiling"]["tile_size"])
    n_tiles = int(cfg["tiling"]["grid_rows"]) * int(cfg["tiling"]["grid_cols"])

    wrapped = OnnxTileModel(model)

    stack = torch.randn(1, n_tiles, 3, tile_size, tile_size, dtype=torch.float32)
    meta = torch.zeros(1, 3, dtype=torch.float32)
    meta[0, 1] = 1.0

    dynamic_axes = {"stack": {0: "batch"}, "meta": {0: "batch"}, "counts": {0: "batch"}, "measures": {0: "batch"}} if cfg["export"].get("dynamic_batch", True) else None

    torch.onnx.export(
        wrapped,
        (stack, meta),
        out_path,
        input_names=["stack", "meta"],
        output_names=["counts", "measures"],
        dynamic_axes=dynamic_axes,
        opset_version=int(cfg["export"]["opset"]),
        dynamo=True,
    )

    model_onnx = onnx.load(out_path)
    onnx.checker.check_model(model_onnx)

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"stack": stack.numpy(), "meta": meta.numpy()})
    with torch.no_grad():
        pt_out = wrapped(stack, meta)

    counts_max_abs = float(np.max(np.abs(pt_out[0].numpy() - ort_out[0])))
    measures_max_abs = float(np.max(np.abs(pt_out[1].numpy() - ort_out[1])))

    report = {
        "onnx_path": str(Path(out_path).resolve()),
        "parity": {"counts_max_abs": counts_max_abs, "measures_max_abs": measures_max_abs},
    }
    write_json(Path(out_path).with_suffix(".parity.json"), report)
    return report
