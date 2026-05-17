"""Utility functions: logging, seed, device, checkpoint management."""

import random
import os
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(model: torch.nn.Module, optimizer, path: str, **extra) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
    }
    checkpoint.update(extra)
    torch.save(checkpoint, path)


def load_checkpoint(model: torch.nn.Module, optimizer, path: str) -> dict:
    checkpoint = torch.load(path, map_location=get_device(), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def compute_ee_pose(robot_state: torch.Tensor) -> torch.Tensor:
    """Extract end-effector pose from robot state tensor.

    Assumes robot_state layout: [joint_positions..., ee_pos(3), ee_rot(4/6), gripper(1)]
    Adapt this based on actual robot state format.
    """
    return robot_state[..., -8:-1]  # default: last 7 dims are ee pose
