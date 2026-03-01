from __future__ import annotations

from pathlib import Path

import onnx
from onnxconverter_common import float16
import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from mobile_convert.data.tiling import split_to_grid
from mobile_convert.utils.io import write_json


def convert_to_fp16(in_path: str, out_path: str) -> dict:
    model = onnx.load(in_path)
    # Keep numerically sensitive reduction/normalization ops in FP32 to avoid
    # overflow/Inf drift in aggregated count outputs.
    op_block_list = [
        "Resize",
        "ReduceSum",
        "ReduceMean",
        "GlobalAveragePool",
        "LayerNormalization",
        "Softmax",
        "LogSoftmax",
    ]
    fp16_model = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        op_block_list=op_block_list,
    )
    onnx.save(fp16_model, out_path)
    report = {
        "input": str(Path(in_path).resolve()),
        "output": str(Path(out_path).resolve()),
        "op_block_list": op_block_list,
        "status": "ok",
    }
    write_json(Path(out_path).with_suffix(".convert.json"), report)
    return report


def build_calibration_feed(
    image_path: str,
    tile_size: int,
    grid_rows: int,
    grid_cols: int,
    rice_comment: str = "White",
) -> dict:
    img = np.array(Image.open(image_path).convert("RGB"))
    tiles = split_to_grid(img, int(grid_rows), int(grid_cols))
    tfm = A.Compose([A.Resize(int(tile_size), int(tile_size)), A.Normalize(), ToTensorV2()])
    stack = torch.stack([tfm(image=t)["image"] for t in tiles]).unsqueeze(0).numpy().astype(np.float32)
    meta = np.zeros((1, 3), dtype=np.float32)
    comment = str(rice_comment).strip().lower()
    # Keep mapping aligned with training schema map: Paddy=0, White=1, Brown=2
    if comment == "paddy":
        meta[0, 0] = 1.0
    elif comment == "brown":
        meta[0, 2] = 1.0
    else:
        meta[0, 1] = 1.0
    return {"stack": stack, "meta": meta}


def convert_to_fp16_mixed(
    in_path: str,
    out_path: str,
    feed_dict: dict,
    rtol: float = 0.01,
    atol: float = 0.001,
    keep_io_types: bool = True,
) -> dict:
    model = onnx.load(in_path)
    try:
        from onnxconverter_common import auto_mixed_precision as amp  # type: ignore
    except Exception as exc:
        raise RuntimeError("onnxconverter_common.auto_mixed_precision is unavailable") from exc

    fn = getattr(amp, "auto_convert_mixed_precision", None)
    if fn is None:
        raise RuntimeError("auto_convert_mixed_precision function not found in onnxconverter_common")

    last_exc: Exception | None = None
    attempts = [
        lambda: fn(model, feed_dict, rtol=rtol, atol=atol, keep_io_types=keep_io_types),
        lambda: fn(model, feed_dict, rtol=rtol, atol=atol),
        lambda: fn(model, feed_dict),
    ]
    converted = None
    for attempt in attempts:
        try:
            converted = attempt()
            break
        except Exception as exc:  # pragma: no cover - depends on installed api variant
            last_exc = exc

    if converted is None:
        raise RuntimeError(f"Mixed precision conversion failed: {last_exc}")

    if isinstance(converted, tuple):
        converted = converted[0]

    onnx.save(converted, out_path)
    report = {
        "input": str(Path(in_path).resolve()),
        "output": str(Path(out_path).resolve()),
        "mode": "mixed_fp16",
        "rtol": float(rtol),
        "atol": float(atol),
        "keep_io_types": bool(keep_io_types),
        "status": "ok",
    }
    write_json(Path(out_path).with_suffix(".mixed.convert.json"), report)
    return report
