from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch

from mobile_convert.train.engine import train_main


def resolve_device(mode: str) -> torch.device:
    mode = mode.lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "xla":
        try:
            import torch_xla.core.xla_model as xm  # type: ignore

            return xm.xla_device()
        except Exception:
            logging.warning("XLA requested but unavailable. Falling back to CUDA/CPU.")
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dump_xla_metrics(run_dir: Path) -> None:
    try:
        import torch_xla.debug.metrics as met  # type: ignore

        (run_dir / "xla_metrics.txt").write_text(met.metrics_report(), encoding="utf-8")
    except Exception:
        pass


def _xla_mp_worker(_index: int, cfg_local: dict, run_dir_str: str, result_file_str: str):
    import torch_xla.core.xla_model as xm  # type: ignore

    device = xm.xla_device()
    rank = xm.get_ordinal()
    world_size = xm.xrt_world_size()
    is_master = xm.is_master_ordinal()

    result = train_main(
        cfg_local,
        Path(run_dir_str),
        device,
        rank=rank,
        world_size=world_size,
        is_master=is_master,
        distributed=True,
    )

    xm.rendezvous("train_main_done")
    if is_master:
        Path(result_file_str).write_text(json.dumps(result, indent=2), encoding="utf-8")
        _dump_xla_metrics(Path(run_dir_str))


def _run_xla_spawn(cfg: dict, run_dir: Path):
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
        import torch_xla.distributed.xla_multiprocessing as xmp  # type: ignore
    except Exception:
        logging.warning("torch_xla unavailable. Falling back to CUDA/CPU.")
        device = resolve_device("auto")
        return train_main(cfg, run_dir, device)

    result_file = run_dir / "xla_train_result.json"

    requested_world_size = cfg.get("runtime", {}).get("xla_world_size", None)
    if requested_world_size is not None:
        try:
            ws = int(requested_world_size)
            if ws > 0:
                # PJRT requires nprocs=None and uses env vars to cap device count.
                os.environ["TPU_NUM_DEVICES"] = str(ws)
                logging.info("Set TPU_NUM_DEVICES=%s", ws)
        except Exception:
            logging.warning("Invalid runtime.xla_world_size=%s; using default TPU device count.", requested_world_size)

    xmp.spawn(_xla_mp_worker, args=(cfg, str(run_dir), str(result_file)), nprocs=None, start_method="fork")

    if result_file.exists():
        return json.loads(result_file.read_text(encoding="utf-8"))

    return {"run_dir": str(run_dir), "best_ckpt": str(run_dir / "best.ckpt"), "metrics": None}


def run_training(cfg: dict, run_dir):
    mode = cfg.get("runtime", {}).get("mode", "auto").lower()
    if mode == "xla":
        logging.info("Using XLA spawn mode")
        return _run_xla_spawn(cfg, Path(run_dir))

    device = resolve_device(mode)
    logging.info("Using device: %s", device)
    result = train_main(cfg, Path(run_dir), device)
    if device.type == "xla":
        _dump_xla_metrics(Path(run_dir))
    return result
