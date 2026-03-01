from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from mobile_convert.data.dataset import RiceDataset
from mobile_convert.data.schema import COUNT_COLS, MEASURE_COLS, assert_schema
from mobile_convert.eval.metrics import mae_per_target, total_mae
from mobile_convert.models.architecture import UltimateSpecialist
from mobile_convert.models.losses import MultiTaskUncertaintyLoss, consistency_loss, weighted_count_loss
from mobile_convert.train.checkpointing import load_resume_checkpoint, load_warmstart, save_checkpoint
from mobile_convert.utils.io import write_json


def _freeze_backbone(model: UltimateSpecialist, freeze: bool) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = not freeze


def _measure_stats(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    arr = df[MEASURE_COLS].values.astype(np.float32)
    return arr.mean(axis=0), arr.std(axis=0)


def _build_model(cfg: dict, device: torch.device, pretrained_backbone: bool = True) -> UltimateSpecialist:
    model = UltimateSpecialist(cfg["model"]["name"], pretrained_backbone=pretrained_backbone)
    model.to(device)
    return model


def build_splits_and_loaders(
    cfg: dict,
    require_targets: bool = True,
    rank: int = 0,
    world_size: int = 1,
    distributed: bool = False,
):
    train_csv = cfg["data"]["train_csv"]
    image_dir = cfg["data"]["image_dir"]

    df = pd.read_csv(train_csv)
    assert_schema(df, require_targets=require_targets)
    tr_df, va_df = train_test_split(
        df,
        test_size=float(cfg["data"]["val_split"]),
        random_state=int(cfg["training"]["seed"]),
        stratify=df["Comment"],
    )

    m_stats = _measure_stats(tr_df)
    common = dict(
        image_dir=image_dir,
        tile_size=int(cfg["tiling"]["tile_size"]),
        grid_rows=int(cfg["tiling"]["grid_rows"]),
        grid_cols=int(cfg["tiling"]["grid_cols"]),
        measure_stats=m_stats,
        require_targets=require_targets,
    )

    train_ds = RiceDataset(tr_df, train_mode=True, **common)
    val_ds = RiceDataset(va_df, train_mode=False, **common)

    train_sampler = None
    if distributed and world_size > 1:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=int(cfg["training"]["num_workers"]),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
        drop_last=False,
    )
    return train_loader, val_loader, m_stats, train_sampler


def evaluate(model: UltimateSpecialist, loader: DataLoader, device: torch.device, scale: float, m_stats):
    model.eval()
    count_errs = []
    measure_errs = []
    with torch.no_grad():
        for stack, meta, counts, measures, _, _ in loader:
            stack, meta = _maybe_to_device(stack, device), _maybe_to_device(meta, device)
            p_c, p_m = model(stack, meta)

            p_c = p_c.cpu().numpy() / scale
            counts_np = counts.detach().cpu().numpy()
            count_errs.append(np.abs(p_c - counts_np))

            p_m = p_m.cpu().numpy() * (m_stats[1] + 1e-8) + m_stats[0]
            m_np = measures.detach().cpu().numpy() * (m_stats[1] + 1e-8) + m_stats[0]
            measure_errs.append(np.abs(p_m - m_np))

    c_err = np.concatenate(count_errs)
    m_err = np.concatenate(measure_errs)
    c_mae = mae_per_target(c_err, np.zeros_like(c_err))
    m_mae = mae_per_target(m_err, np.zeros_like(m_err))
    t_mae = total_mae(c_mae, m_mae)

    target_map = {name: float(val) for name, val in zip(COUNT_COLS + MEASURE_COLS, np.concatenate([c_mae, m_mae]))}
    return {"total_mae": t_mae, "count_mae": c_mae.tolist(), "measure_mae": m_mae.tolist(), "target_mae": target_map}


def _amp_context(device: torch.device, use_bf16: bool):
    if not use_bf16:
        return contextlib.nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.type == "xla":
        try:
            from torch_xla.amp import autocast as xla_autocast  # type: ignore

            return xla_autocast(dtype=torch.bfloat16)
        except Exception:
            return contextlib.nullcontext()
    return contextlib.nullcontext()


def _xla_mark_step():
    try:
        import torch_xla.core.xla_model as xm  # type: ignore

        xm.mark_step()
    except Exception:
        pass


def _xla_rendezvous(tag: str):
    try:
        import torch_xla.core.xla_model as xm  # type: ignore

        xm.rendezvous(tag)
    except Exception:
        pass


def _maybe_to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        return t
    if t.device == device:
        return t
    return t.to(device)


def _optimizer_step(device: torch.device, optimizer: torch.optim.Optimizer) -> None:
    if device.type == "xla":
        try:
            import torch_xla.core.xla_model as xm  # type: ignore

            xm.optimizer_step(optimizer, barrier=False)
            return
        except Exception:
            pass
    optimizer.step()


def train_main(
    cfg: dict,
    run_dir: Path,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
    is_master: bool = True,
    distributed: bool = False,
) -> dict:
    train_loader, val_loader, m_stats, train_sampler = build_splits_and_loaders(
        cfg,
        require_targets=True,
        rank=rank,
        world_size=world_size,
        distributed=distributed,
    )

    if device.type == "xla":
        try:
            import torch_xla.distributed.parallel_loader as pl  # type: ignore

            train_loader = pl.MpDeviceLoader(train_loader, device)
            val_loader = pl.MpDeviceLoader(val_loader, device)
        except Exception as exc:
            logging.warning("Could not enable XLA MpDeviceLoader: %s", exc)

    model = _build_model(cfg, device)
    count_weights = torch.tensor([1.0, 1.5, 1.5, 0.5, 1.5, 2.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    count_weights = count_weights / count_weights.mean()

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["optimizer"]["lr"]), weight_decay=float(cfg["optimizer"]["weight_decay"]))

    l1 = nn.L1Loss()
    uncertainty = MultiTaskUncertaintyLoss().to(device)

    use_bf16 = bool(cfg["runtime"].get("use_bf16", False)) and device.type in {"cuda", "xla"}
    grad_accum = int(cfg["training"]["grad_accum"])
    eval_every = max(1, int(cfg["training"].get("eval_every_n_epochs", 1)))
    grad_clip_norm = float(cfg["training"].get("grad_clip_norm", 0.0))
    warmup_epochs = int(cfg["training"].get("warmup_head_epochs", 0))
    scale = float(cfg["model"]["scale"])

    metrics_path = run_dir / "metrics.jsonl"
    best_score = float("inf")
    start_epoch = 1
    total_epochs = int(cfg["training"]["epochs"])
    resume_optimizer = bool(cfg["training"].get("resume_optimizer", False))
    resumed = False

    resume_ckpt = cfg["training"].get("checkpoint")
    if resume_ckpt:
        if Path(resume_ckpt).exists():
            resume_meta = load_resume_checkpoint(model, optimizer, resume_ckpt, device, load_optimizer=resume_optimizer)
            start_epoch = resume_meta["epoch"] + 1
            best_score = resume_meta["best_score"]
            if resume_meta["m_stats"] is not None:
                m_stats = resume_meta["m_stats"]
            resumed = True
            logging.info(
                "Resumed training from %s at epoch %d (best=%.6f, resume_optimizer=%s)",
                resume_ckpt,
                resume_meta["epoch"],
                best_score,
                resume_optimizer,
            )
        else:
            logging.warning("Resume checkpoint not found: %s. Starting from scratch/warm-start.", resume_ckpt)
            resume_ckpt = None

    if not resume_ckpt:
        teacher_ckpt = cfg["training"].get("teacher_checkpoint")
        if teacher_ckpt:
            load_warmstart(model, teacher_ckpt)

    if resumed and not resume_optimizer:
        resume_lr = float(cfg["training"].get("resume_lr", cfg["optimizer"]["lr"]))
        for pg in optimizer.param_groups:
            pg["lr"] = resume_lr
        scheduler_tmax = max(1, total_epochs - start_epoch + 1)
        logging.info("Resume fine-tune LR set to %.8f with cosine T_max=%d", resume_lr, scheduler_tmax)
    else:
        scheduler_tmax = int(cfg["scheduler"]["t_max"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_tmax)

    if resumed and resume_optimizer and start_epoch > 1:
        # Align scheduler state with resumed epoch when optimizer state was restored.
        for _ in range(start_epoch - 1):
            scheduler.step()

    if start_epoch > total_epochs:
        logging.info("Resume epoch (%d) already beyond requested total epochs (%d). Skipping training loop.", start_epoch, total_epochs)

    for epoch in range(start_epoch, total_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        _freeze_backbone(model, freeze=epoch <= warmup_epochs)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        loop = tqdm(train_loader, desc=f"Epoch {epoch} [rank {rank}]", leave=False, disable=not is_master)
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        for step, (stack, meta, counts, measures, _, _) in enumerate(loop, start=1):
            stack = _maybe_to_device(stack, device)
            meta = _maybe_to_device(meta, device)
            counts = _maybe_to_device(counts, device)
            measures = _maybe_to_device(measures, device)

            with _amp_context(device, use_bf16):
                p_c, p_m = model(stack, meta)
                loss_c = weighted_count_loss(p_c, counts * scale, count_weights) / scale
                loss_m = l1(p_m, measures)
                loss_cons = consistency_loss(p_c) / scale

                if bool(cfg["training"].get("use_uncertainty_weighting", True)):
                    loss = uncertainty(loss_c, loss_m, loss_cons)
                else:
                    loss = 1.5 * loss_c + 0.1 * loss_m + 0.5 * loss_cons

            loss.backward()
            epoch_loss_sum += float(loss.detach().cpu().item())
            epoch_loss_count += 1
            if step % grad_accum == 0:
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                _optimizer_step(device, optimizer)
                optimizer.zero_grad(set_to_none=True)

            if device.type == "xla":
                _xla_mark_step()

        scheduler.step()
        if world_size > 1:
            _xla_rendezvous(f"epoch_{epoch}_train_end")

        # Always save last checkpoint for resume safety, even on non-eval epochs.
        if is_master:
            save_checkpoint(run_dir / "last.ckpt", model, optimizer, epoch, best_score, m_stats)

        should_eval = (epoch % eval_every == 0) or (epoch == total_epochs)

        if is_master:
            avg_train_loss = epoch_loss_sum / max(1, epoch_loss_count)
            current_lr = float(optimizer.param_groups[0]["lr"])
            if should_eval:
                eval_metrics = evaluate(model, val_loader, device, scale, m_stats)
                score = float(eval_metrics["total_mae"])
                is_best = score < best_score
                if is_best:
                    best_score = score
                    save_checkpoint(run_dir / "best.ckpt", model, optimizer, epoch, best_score, m_stats)

                record = {"epoch": epoch, "best": is_best, **eval_metrics}
                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

                logging.info("Epoch=%d train_loss=%.6f lr=%.8f total_mae=%.6f best=%.6f", epoch, avg_train_loss, current_lr, score, best_score)
            else:
                logging.info("Epoch=%d train_loss=%.6f lr=%.8f (eval skipped, every %d epochs)", epoch, avg_train_loss, current_lr, eval_every)

        if world_size > 1 and should_eval:
            _xla_rendezvous(f"epoch_{epoch}_eval_end")

    if is_master:
        best_metrics = evaluate(model, val_loader, device, scale, m_stats)
        write_json(run_dir / "eval_metrics.json", best_metrics)
        return {"run_dir": str(run_dir), "best_ckpt": str(run_dir / "best.ckpt"), "metrics": best_metrics}

    return {"run_dir": str(run_dir), "best_ckpt": str(run_dir / "best.ckpt"), "metrics": None}


def evaluate_checkpoint(cfg: dict, ckpt_path: str, device: torch.device) -> dict:
    _, val_loader, _, _ = build_splits_and_loaders(cfg, require_targets=True)
    model = _build_model(cfg, device, pretrained_backbone=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    m_stats = ckpt["m_stats"]
    return evaluate(model, val_loader, device, float(cfg["model"]["scale"]), m_stats)
