"""E3-VLA Drafter Training Script.

Usage:
  uv run python scripts/train_drafter.py \
      model=default \
      data.cache_dir=/path/to/teacher_cache
"""

import os
import sys
import time
import logging
from typing import Optional
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

# Configure verbose logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_drafter")

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from e3vla.data.dataset_builder import CrossTemporalDataset
from e3vla.utils import set_seed, get_device, save_checkpoint, load_checkpoint


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.count = 0
        self.sum = 0.0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val * n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def build_model(cfg: DictConfig) -> nn.Module:
    """Instantiate drafter model from Hydra config."""
    return hydra.utils.instantiate(cfg.model)


def build_dataloaders(cfg: DictConfig) -> tuple:
    """Build train and validation dataloaders."""
    train_ds = CrossTemporalDataset(
        cache_dir=cfg.data.cache_dir,
        max_delta=cfg.data.max_delta,
        chunk_len=cfg.model.chunk_len,
        split="train",
        val_frac=cfg.data.val_frac,
        split_seed=cfg.data.split_seed,
        perturbation=cfg.perturbation.enabled,
    )

    val_ds = CrossTemporalDataset(
        cache_dir=cfg.data.cache_dir,
        max_delta=cfg.data.max_delta,
        chunk_len=cfg.model.chunk_len,
        split="val",
        val_frac=cfg.data.val_frac,
        split_seed=cfg.data.split_seed,
        perturbation=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, len(train_ds), len(val_ds)


def train_epoch(model, loader, optimizer, cfg, device, epoch, n_epochs) -> dict:
    model.train()
    meters = defaultdict(AverageMeter)
    total_steps = len(loader)

    for batch_idx, sample in enumerate(loader):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in sample.items()
        }
        if hasattr(sample, 'target_action_chunk'):
            batch = _training_sample_to_batch(sample, device)

        if cfg.perturbation.enabled:
            batch = _apply_perturbations(batch, cfg, device)

        losses = model.compute_loss(batch)
        loss = losses["l_total"]

        optimizer.zero_grad()
        loss.backward()

        if cfg.training.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)

        optimizer.step()

        for k, v in losses.items():
            meters[k].update(v.item())

        if batch_idx % cfg.logging.log_interval_steps == 0:
            pct = batch_idx / total_steps * 100
            gpu_mem = torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0
            logger.info(
                f"  Epoch {epoch+1}/{n_epochs} [{batch_idx}/{total_steps} {pct:.0f}%] "
                f"loss={loss.item():.4f} | GPU={gpu_mem:.1f}GB"
            )

    return {k: v.avg for k, v in meters.items()}


@torch.no_grad()
def validate(model, loader, cfg, device) -> dict:
    model.eval()
    meters = defaultdict(AverageMeter)

    for sample in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in sample.items()
        }
        if hasattr(sample, 'target_action_chunk'):
            batch = _training_sample_to_batch(sample, device)

        losses = model.compute_loss(batch)

        for k, v in losses.items():
            meters[k].update(v.item())

    return {k: v.avg for k, v in meters.items()}


def _training_sample_to_batch(sample, device):
    """Convert TrainingSample to batched dict."""
    return {
        "cached_ae_low": sample.cached_full_round_features.ae_low.unsqueeze(0).to(device),
        "cached_ae_mid": sample.cached_full_round_features.ae_mid.unsqueeze(0).to(device),
        "cached_ae_high": sample.cached_full_round_features.ae_high.unsqueeze(0).to(device),
        "cached_ae_mixed": sample.cached_full_round_features.ae_mixed.unsqueeze(0).to(device),
        "current_robot_state": sample.current_robot_state.unsqueeze(0).to(device),
        "action_history": sample.action_history.unsqueeze(0).to(device),
        "target_action_chunk": sample.target_action_chunk.unsqueeze(0).to(device),
        "delta_action_index": sample.delta_action_index,
        "gripper_phase": torch.tensor([sample.gripper_phase]),
    }


def _apply_perturbations(batch: dict, cfg, device) -> dict:
    """Apply staleness training perturbations."""
    batch = dict(batch)  # shallow copy

    # Feature dropout
    if cfg.perturbation.feature_dropout > 0:
        p = cfg.perturbation.feature_dropout
        for key in ["cached_ae_low", "cached_ae_mid", "cached_ae_high", "cached_ae_mixed"]:
            if key in batch:
                mask = torch.bernoulli(
                    torch.full_like(batch[key], 1 - p)
                )
                batch[key] = batch[key] * mask

    return batch


@hydra.main(version_base=None, config_path="../configs", config_name="train/default")
def main(cfg: DictConfig):
    logger.info("=" * 60)
    logger.info("E3-VLA Drafter Training")
    logger.info("=" * 60)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    set_seed(cfg.data.split_seed)
    device = get_device()
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_total:.1f} GB)")
    logger.info(f"Device: {device}")

    # Build model
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} trainable / {n_total:,} total parameters")

    # Build data
    train_loader, val_loader, train_size, val_size = build_dataloaders(cfg)
    steps_per_epoch = len(train_loader)
    logger.info(f"Data: {train_size:,} train / {val_size:,} val samples")
    logger.info(f"      {steps_per_epoch} steps/epoch × {cfg.training.epochs} epochs = "
                f"{steps_per_epoch * cfg.training.epochs:,} total steps")
    logger.info(f"      batch_size={cfg.data.batch_size}, num_workers={cfg.data.num_workers}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs
    )
    logger.info(f"Optimizer: AdamW(lr={cfg.training.lr}, wd={cfg.training.weight_decay})")
    logger.info(f"LR schedule: CosineAnnealing, warmup={cfg.training.warmup_steps} steps")

    if cfg.logging.wandb_project:
        wandb.init(
            project=cfg.logging.wandb_project,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        logger.info(f"WandB: project={cfg.logging.wandb_project}")

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    os.makedirs(cfg.checkpoint.save_dir, exist_ok=True)
    n_epochs = cfg.training.epochs

    logger.info("-" * 60)
    logger.info(f"Starting training: {n_epochs} epochs")
    logger.info("-" * 60)

    for epoch in range(n_epochs):
        t0 = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, cfg, device, epoch, n_epochs)
        val_metrics = validate(model, val_loader, cfg, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        train_l = train_metrics.get("l_total", 0)
        val_l = val_metrics.get("l_total", 0)
        is_best = val_l < best_val_loss - cfg.training.early_stop_min_delta

        # Verbose per-epoch summary
        status = "★ BEST" if is_best else ""
        logger.info(
            f"Epoch {epoch+1:3d}/{n_epochs} | "
            f"train_loss={train_l:.4f} | val_loss={val_l:.4f} | "
            f"lr={lr_now:.2e} | time={elapsed:.1f}s {status}"
        )

        # Per-component loss breakdown
        logger.info(
            f"  Breakdown: action={train_metrics.get('l_action', 0):.4f} "
            f"smooth={train_metrics.get('l_smooth', 0):.4f} "
            f"gripper={train_metrics.get('l_gripper', 0):.4f} "
            f"uncert={train_metrics.get('l_uncert', 0):.4f}"
        )

        if torch.cuda.is_available():
            gpu_used = torch.cuda.memory_allocated(device) / 1e9
            gpu_max = torch.cuda.max_memory_allocated(device) / 1e9
            logger.info(f"  GPU: {gpu_used:.1f}GB used, {gpu_max:.1f}GB peak")

        if wandb.run:
            wandb.log({
                "epoch": epoch,
                "lr": lr_now,
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
                "time_per_epoch": elapsed,
                "gpu_memory_gb": torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0,
            })

        # Checkpoint
        if is_best:
            best_val_loss = val_l
            best_epoch = epoch + 1
            patience_counter = 0
            if cfg.checkpoint.save_best:
                save_checkpoint(
                    model, optimizer,
                    os.path.join(cfg.checkpoint.save_dir, "best.pt"),
                    epoch=epoch, val_loss=val_l,
                )
                logger.info(f"  Checkpoint saved: {cfg.checkpoint.save_dir}/best.pt")
        else:
            patience_counter += 1
            logger.info(f"  No improvement for {patience_counter} epochs "
                        f"(patience={cfg.training.early_stop_patience})")

        if epoch % cfg.checkpoint.save_interval_epochs == 0:
            path = os.path.join(cfg.checkpoint.save_dir, f"epoch_{epoch:04d}.pt")
            save_checkpoint(model, optimizer, path, epoch=epoch)
            logger.info(f"  Periodic checkpoint saved: {path}")

        if patience_counter >= cfg.training.early_stop_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break

    logger.info("=" * 60)
    logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f} at epoch {best_epoch}")
    logger.info(f"Best checkpoint: {cfg.checkpoint.save_dir}/best.pt")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
