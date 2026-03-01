from __future__ import annotations

import logging
from pathlib import Path

import torch


def _to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    return obj


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_score: float, measure_stats) -> None:
    # Save portable CPU checkpoint so resume works across CPU/CUDA/XLA runtimes.
    model_state = _to_cpu(model.state_dict())
    optimizer_state = _to_cpu(optimizer.state_dict())
    torch.save(
        {
            "model": model_state,
            "optimizer": optimizer_state,
            "epoch": epoch,
            "best_score": best_score,
            "m_stats": measure_stats,
        },
        path,
    )


def load_warmstart(model: torch.nn.Module, checkpoint_path: str) -> bool:
    if not Path(checkpoint_path).exists():
        logging.warning("Warm-start checkpoint not found: %s. Training will continue without warm-start.", checkpoint_path)
        return False

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logging.info("Warm-start loaded from %s", checkpoint_path)
    logging.info("Missing keys: %d | Unexpected keys: %d", len(missing), len(unexpected))
    return True


def load_resume_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: str, device: torch.device) -> dict:
    # Always load via CPU to avoid backend-tag restore errors (e.g. xla:0 tags).
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)

    if "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as exc:
            logging.warning("Could not load optimizer state from resume checkpoint: %s", exc)

    return {
        "epoch": int(ckpt.get("epoch", 0)),
        "best_score": float(ckpt.get("best_score", float("inf"))),
        "m_stats": ckpt.get("m_stats"),
    }
