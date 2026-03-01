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
    import torch_xla.runtime as xr  # type: ignore

    device = xm.xla_device()
    rank = int(xr.global_ordinal())
    world_size = int(xr.world_size())
    is_master = rank == 0

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


def _run_xla_spawn(cfg: dict, run_dir: Path, start_method: str = "spawn"):
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
        import torch_xla.distributed.xla_multiprocessing as xmp  # type: ignore
    except Exception:
        logging.warning("torch_xla unavailable. Falling back to CUDA/CPU.")
        device = resolve_device("auto")
        return train_main(cfg, run_dir, device)

    result_file = run_dir / "xla_train_result.json"

    # Under PJRT in managed notebook environments (e.g. Kaggle), forcing device
    # count via env vars can break TPU initialization. Let PJRT discover devices.
    xmp.spawn(_xla_mp_worker, args=(cfg, str(run_dir), str(result_file)), nprocs=None, start_method=start_method)

    if result_file.exists():
        return json.loads(result_file.read_text(encoding="utf-8"))

    return {"run_dir": str(run_dir), "best_ckpt": str(run_dir / "best.ckpt"), "metrics": None}


def run_training(cfg: dict, run_dir):
    mode = cfg.get("runtime", {}).get("mode", "auto").lower()
    if mode == "xla":
        # Must be set before torch_xla runtime initialization in this process.
        os.environ["PJRT_DEVICE"] = "TPU"
        # Defensive cleanup for known bad notebook env contamination.
        for key in ("TPU_PROCESS_ADDRESSES", "CLOUD_TPU_TASK_ID"):
            os.environ.pop(key, None)
        if "TPU_WORKER_HOSTNAMES" in os.environ and "WARNING" in os.environ["TPU_WORKER_HOSTNAMES"]:
            os.environ.pop("TPU_WORKER_HOSTNAMES", None)

        use_spawn = bool(cfg.get("runtime", {}).get("use_xla_spawn", True))
        spawn_method = str(cfg.get("runtime", {}).get("xla_spawn_start_method", "spawn"))
        if use_spawn:
            logging.info("Using XLA spawn mode (start_method=%s)", spawn_method)
            try:
                return _run_xla_spawn(cfg, Path(run_dir), start_method=spawn_method)
            except Exception as exc:
                logging.warning("XLA spawn failed (%s). Falling back to single-process XLA.", exc)

        device = resolve_device("xla")
        logging.info("Using single-process XLA mode on device: %s", device)
        result = train_main(cfg, Path(run_dir), device, distributed=False)
        if device.type == "xla":
            _dump_xla_metrics(Path(run_dir))
        return result

    device = resolve_device(mode)
    logging.info("Using device: %s", device)
    result = train_main(cfg, Path(run_dir), device)
    if device.type == "xla":
        _dump_xla_metrics(Path(run_dir))
    return result
