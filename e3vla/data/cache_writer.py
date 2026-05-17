"""Offline Teacher Cache writer.

Generates teacher cache from full VLA rollout trajectories.
Stores features/actions/states as safetensors shards + JSONL metadata.
"""

import os
import json
import time
from typing import List, Optional
from dataclasses import dataclass

import torch
import numpy as np
from safetensors.torch import save_file as save_safetensors

from e3vla.schema import Observation, AEFeatureBundle
from e3vla.protocols import BaseVLAAdapter


@dataclass
class CacheConfig:
    cache_dir: str
    shard_size: int = 1000         # records per safetensors shard
    chunk_len: int = 16
    max_episodes: Optional[int] = None


class TeacherCacheWriter:
    """Collects full VLA rollout data and writes to offline teacher cache.

    Usage:
        writer = TeacherCacheWriter(config, adapter)
        for episode in rollout_generator:
            writer.start_episode(episode_id, task_id, instruction)
            for obs, action_chunk in timesteps:
                writer.add_timestep(obs, action_chunk)
            writer.end_episode()
        writer.finalize()
    """

    def __init__(self, config: CacheConfig, adapter: BaseVLAAdapter):
        self.config = config
        self.adapter = adapter

        os.makedirs(os.path.join(config.cache_dir, "metadata"), exist_ok=True)
        os.makedirs(os.path.join(config.cache_dir, "features"), exist_ok=True)
        os.makedirs(os.path.join(config.cache_dir, "actions"), exist_ok=True)
        os.makedirs(os.path.join(config.cache_dir, "states"), exist_ok=True)

        self._records: List[dict] = []
        self._feature_buffers: List[dict] = []
        self._action_buffer: List[torch.Tensor] = []
        self._state_buffer: List[torch.Tensor] = []
        self._shard_idx = 0
        self._record_idx = 0
        self._episode_idx = 0
        self._current_episode = None

    def start_episode(self, episode_id: str, task_id: str, instruction: str):
        self._current_episode = {
            "episode_id": episode_id,
            "task_id": task_id,
            "instruction": instruction,
            "timestep": 0,
        }
        self._episode_idx += 1

        if (self.config.max_episodes is not None
                and self._episode_idx > self.config.max_episodes):
            raise StopIteration("Max episodes reached")

    def add_timestep(
        self, obs: Observation, action_chunk: torch.Tensor
    ) -> None:
        """Record a single timestep: run full VLA, extract features, write."""
        # Extract AE features from the full VLA forward
        ae_features = self.adapter.extract_ae_features(obs, action_chunk)

        robot_state = obs.robot_state.detach().cpu()

        # Build metadata record
        record = {
            "record_id": self._record_idx,
            "episode_id": self._current_episode["episode_id"],
            "task_id": self._current_episode["task_id"],
            "timestep": self._current_episode["timestep"],
            "instruction": self._current_episode["instruction"],
            "robot_state": robot_state.tolist(),
            "ee_pose": robot_state[-8:-1].tolist(),
            "shard": f"shard_{self._shard_idx:04d}",
            "feature_idx": len(self._feature_buffers),
            "action_idx": len(self._action_buffer),
            "chunk_len": self.config.chunk_len,
            "timestamp": time.time(),
        }

        self._records.append(record)

        # Buffer features
        feat_dict = {}
        for key in ["ae_low", "ae_mid", "ae_high", "ae_mixed"]:
            val = getattr(ae_features, key)
            feat_dict[f"{key}_{len(self._feature_buffers)}"] = val.detach().cpu()
        self._feature_buffers.append(feat_dict)

        # Buffer actions and states
        self._action_buffer.append(action_chunk.detach().cpu())
        self._state_buffer.append(robot_state)

        self._record_idx += 1
        self._current_episode["timestep"] += 1

        # Flush shard when full
        if len(self._feature_buffers) >= self.config.shard_size:
            self._flush_shard()

    def end_episode(self) -> None:
        self._current_episode = None

    def finalize(self) -> None:
        """Flush remaining data and write metadata."""
        if self._feature_buffers:
            self._flush_shard()
        self._write_metadata()

    def _flush_shard(self) -> None:
        """Write current buffers to safetensors files."""
        if not self._feature_buffers:
            return

        shard_name = f"shard_{self._shard_idx:04d}"

        # Flatten and save features
        feat_flat = {}
        for buf in self._feature_buffers:
            feat_flat.update(buf)
        save_safetensors(
            feat_flat,
            os.path.join(self.config.cache_dir, "features", f"{shard_name}.safetensors"),
        )

        # Save actions
        action_flat = {}
        for i, a in enumerate(self._action_buffer):
            action_flat[f"action_{i}"] = a
        save_safetensors(
            action_flat,
            os.path.join(self.config.cache_dir, "actions", f"{shard_name}.safetensors"),
        )

        # Save states
        state_flat = {}
        for i, s in enumerate(self._state_buffer):
            state_flat[f"state_{i}"] = s
        save_safetensors(
            state_flat,
            os.path.join(self.config.cache_dir, "states", f"{shard_name}.safetensors"),
        )

        self._feature_buffers.clear()
        self._action_buffer.clear()
        self._state_buffer.clear()
        self._shard_idx += 1

    def _write_metadata(self) -> None:
        path = os.path.join(self.config.cache_dir, "metadata", "records.jsonl")
        with open(path, "w") as f:
            for rec in self._records:
                f.write(json.dumps(rec) + "\n")

    @property
    def num_records(self) -> int:
        return self._record_idx
