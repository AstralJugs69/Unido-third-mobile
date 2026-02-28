from __future__ import annotations

from pathlib import Path

import onnx
from onnxconverter_common import float16

from mobile_convert.utils.io import write_json


def convert_to_fp16(in_path: str, out_path: str) -> dict:
    model = onnx.load(in_path)
    fp16_model = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(fp16_model, out_path)
    report = {
        "input": str(Path(in_path).resolve()),
        "output": str(Path(out_path).resolve()),
        "status": "ok",
    }
    write_json(Path(out_path).with_suffix(".convert.json"), report)
    return report
