"""Training data pipeline for CachedAEDrafter.

Builds cross-temporal training pairs from OfflineTeacherCache.
"""

from typing import Iterator, Optional
import os
import json

import torch
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file as load_safetensors

from e3vla.schema import TrainingSample, AEFeatureBundle


class CrossTemporalDataset(Dataset):
    """Dataset that samples cross-temporal pairs from teacher cache.

    Each sample pairs:
      - Cached AE features from timestep t0 (simulating full refresh cache)
      - Target action chunk from timestep t0 + Δ
      - Offset features between t0 and t0 + Δ
      - Δ ∈ [0, max_delta]
    """

    def __init__(
        self,
        cache_dir: str,
        max_delta: int = 50,
        chunk_len: int = 16,
        split: str = "train",
        val_frac: float = 0.1,
        split_seed: int = 42,
        perturbation: bool = True,
    ):
        super().__init__()
        self.cache_dir = cache_dir
        self.max_delta = max_delta
        self.chunk_len = chunk_len
        self.split = split
        self.perturbation = perturbation

        # Load metadata
        metadata_path = os.path.join(cache_dir, "metadata", "records.jsonl")
        self._records = []
        with open(metadata_path, "r") as f:
            for line in f:
                rec = json.loads(line)
                self._records.append(rec)

        # Episode-level train/val split
        self._split_indices = self._episode_split(val_frac, split_seed)

        # Build valid (t0, Δ) pairs
        self._pairs = self._build_pairs()

    def _episode_split(self, val_frac: float, seed: int) -> list:
        """Deterministic episode-level split using splitmix64 hash."""
        import hashlib
        indices = list(range(len(self._records)))
        train_idx, val_idx = [], []

        for i in indices:
            rec = self._records[i]
            ep_id = rec.get("episode_id", str(i))
            # Splitmix64-inspired deterministic hash
            h = int(hashlib.md5(f"{ep_id}_{seed}".encode()).hexdigest()[:16], 16)
            if (h % 100) < val_frac * 100:
                val_idx.append(i)
            else:
                train_idx.append(i)

        return train_idx if self.split == "train" else val_idx

    def _build_pairs(self) -> list:
        """Build all valid (t0_idx, delta) pairs within episodes.

        Group records by episode_id, then for each timestep t0,
        create pairs with all t0 + Δ within the same episode.
        """
        pairs = []
        ep_records = {}
        for i in self._split_indices:
            rec = self._records[i]
            ep_id = rec["episode_id"]
            ep_records.setdefault(ep_id, []).append((i, rec["timestep"]))

        for ep_id, records in ep_records.items():
            records.sort(key=lambda x: x[1])  # sort by timestep
            for src_idx, (src_i, src_t) in enumerate(records):
                for dst_idx in range(src_idx, min(src_idx + self.max_delta + 1, len(records))):
                    dst_i, dst_t = records[dst_idx]
                    delta = dst_t - src_t
                    if 0 <= delta <= self.max_delta:
                        pairs.append((src_i, dst_i, delta))

        return pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> TrainingSample:
        src_i, dst_i, delta = self._pairs[idx]
        src_rec = self._records[src_i]
        dst_rec = self._records[dst_i]

        # Load features from safetensors
        src_feat = self._load_features(src_rec["shard"], src_rec["feature_idx"])
        dst_feat = self._load_features(dst_rec["shard"], dst_rec["feature_idx"])

        # Load target action
        target = self._load_action(dst_rec["shard"], dst_rec["action_idx"])

        # Apply perturbation if training
        if self.perturbation and self.split == "train":
            delta = self._perturb_delta(delta)

        return TrainingSample(
            cached_full_round_features=src_feat,
            full_round_robot_state=torch.tensor(src_rec["robot_state"]),
            full_round_ee_pose=torch.tensor(src_rec["ee_pose"]),
            current_timestep=dst_rec["timestep"],
            current_robot_state=torch.tensor(dst_rec["robot_state"]),
            current_ee_pose=torch.tensor(dst_rec["ee_pose"]),
            action_history=torch.zeros(5, 7),  # placeholder
            delta_t=delta,
            delta_ee_pose=torch.tensor(dst_rec["ee_pose"]) - torch.tensor(src_rec["ee_pose"]),
            delta_robot_state=torch.tensor(dst_rec["robot_state"]) - torch.tensor(src_rec["robot_state"]),
            delta_action_index=0,
            gripper_phase=0.0,
            target_action_chunk=target,
        )

    def _load_features(self, shard: str, idx: int) -> AEFeatureBundle:
        path = os.path.join(self.cache_dir, "features", f"{shard}.safetensors")
        tensors = load_safetensors(path)
        return AEFeatureBundle(
            ae_low=tensors[f"ae_low_{idx}"],
            ae_mid=tensors[f"ae_mid_{idx}"],
            ae_high=tensors[f"ae_high_{idx}"],
            ae_mixed=tensors[f"ae_mixed_{idx}"],
        )

    def _load_action(self, shard: str, idx: int) -> torch.Tensor:
        path = os.path.join(self.cache_dir, "actions", f"{shard}.safetensors")
        tensors = load_safetensors(path)
        return tensors[f"action_{idx}"]

    @staticmethod
    def _perturb_delta(delta: int) -> int:
        """Apply cache_age jitter during training."""
        import random
        return max(0, delta + random.randint(-2, 2))


def build_dataloader(
    cache_dir: str,
    batch_size: int = 32,
    max_delta: int = 50,
    chunk_len: int = 16,
    split: str = "train",
    num_workers: int = 4,
) -> DataLoader:
    """Build a DataLoader from teacher cache."""
    dataset = CrossTemporalDataset(
        cache_dir=cache_dir,
        max_delta=max_delta,
        chunk_len=chunk_len,
        split=split,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
