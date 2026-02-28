from __future__ import annotations

import logging
from pathlib import Path

import torch


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_score: float, measure_stats) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "m_stats": measure_stats,
        },
        path,
    )


def load_warmstart(model: torch.nn.Module, checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logging.info("Warm-start loaded from %s", checkpoint_path)
    logging.info("Missing keys: %d | Unexpected keys: %d", len(missing), len(unexpected))
