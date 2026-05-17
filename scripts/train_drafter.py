"""E3-VLA Drafter Training Script.

Usage:
  uv run python scripts/train_drafter.py \
      model=default \
      data.cache_dir=/path/to/teacher_cache
"""

import os
import sys
import time
from typing import Optional
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

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


def train_epoch(model, loader, optimizer, cfg, device) -> dict:
    model.train()
    meters = defaultdict(AverageMeter)

    for batch_idx, sample in enumerate(loader):
        # Move to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in sample.items()
        }
        # Convert TrainingSample to batched dict if needed
        if hasattr(sample, 'target_action_chunk'):
            batch = _training_sample_to_batch(sample, device)

        # Apply staleness perturbations
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
            log_str = f"  Step {batch_idx}: " + " ".join(
                f"{k}={v.avg:.4f}" for k, v in meters.items()
            )
            print(log_str)

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
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.data.split_seed)
    device = get_device()
    print(f"Device: {device}")

    # Build model
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    # Build data
    train_loader, val_loader, train_size, val_size = build_dataloaders(cfg)
    print(f"Train samples: {train_size}, Val samples: {val_size}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs
    )

    # Wandb
    if cfg.logging.wandb_project:
        wandb.init(
            project=cfg.logging.wandb_project,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    os.makedirs(cfg.checkpoint.save_dir, exist_ok=True)

    for epoch in range(cfg.training.epochs):
        t0 = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, cfg, device)
        val_metrics = validate(model, val_loader, cfg, device)
        scheduler.step()

        elapsed = time.time() - t0

        # Logging
        log_dict = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
            "time_per_epoch": elapsed,
        }

        print(f"Epoch {epoch:3d} | "
              f"train_loss={train_metrics.get('l_total', 0):.4f} | "
              f"val_loss={val_metrics.get('l_total', 0):.4f} | "
              f"time={elapsed:.1f}s")

        if wandb.run:
            wandb.log(log_dict)

        # Checkpoint
        val_loss = val_metrics.get("l_total", 0)
        is_best = val_loss < best_val_loss - cfg.training.early_stop_min_delta

        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            if cfg.checkpoint.save_best:
                save_checkpoint(
                    model, optimizer,
                    os.path.join(cfg.checkpoint.save_dir, "best.pt"),
                    epoch=epoch, val_loss=val_loss,
                )
        else:
            patience_counter += 1

        if epoch % cfg.checkpoint.save_interval_epochs == 0:
            save_checkpoint(
                model, optimizer,
                os.path.join(cfg.checkpoint.save_dir, f"epoch_{epoch:04d}.pt"),
                epoch=epoch,
            )

        # Early stopping
        if patience_counter >= cfg.training.early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Training complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {cfg.checkpoint.save_dir}/best.pt")


if __name__ == "__main__":
    main()
