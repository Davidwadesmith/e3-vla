"""Cached Full Action Reuse baseline.

The simplest speculative baseline: cache full action chunk during refresh,
reuse remaining steps during speculative rounds.
No drafter, no verifier, no learned components.
"""

import torch

from e3vla.schema import Observation, ActionCommand
from e3vla.utils import compute_ee_pose


class CachedFullActionReuseWrapper:
    """Baseline: cache full chunk, reuse remaining steps.

    If this baseline approaches Ours performance,
    the cached AE drafter contribution is questionable.
    """

    def __init__(self, config: dict):
        self.chunk_len = config.get("chunk_len", 16)
        self.full_exec_len = config.get("full_exec_len", 2)
        self.max_reuse_len = config.get("max_reuse_len", 12)
        self.periodic_full_every_n = config.get("periodic_full_every_n", 10)

        self._cached_chunk = None
        self._cached_ee_pose = None
        self._cursor = 0
        self._draft_rounds = 0

        # Lazy import adapter
        self._adapter = None
        self._config = config

    def reset(self, task_info=None) -> None:
        self._cached_chunk = None
        self._cursor = 0
        self._draft_rounds = 0

    def act(self, obs: Observation) -> ActionCommand:
        if self._should_refresh():
            return self._refresh(obs)
        else:
            return self._reuse(obs)

    def _refresh(self, obs: Observation) -> ActionCommand:
        if self._adapter is None:
            from e3vla.adapters.openpi_adapter import OpenPIAdapter
            self._adapter = OpenPIAdapter(
                checkpoint_path=self._config["checkpoint_path"],
                chunk_len=self.chunk_len,
            )

        chunk = self._adapter.full_inference(obs)
        self._cached_chunk = chunk
        self._cached_ee_pose = compute_ee_pose(obs.robot_state)
        self._cursor = 0
        self._draft_rounds = 0

        return ActionCommand(
            actions=chunk[:self.full_exec_len],
            execute_len=self.full_exec_len,
            can_interrupt=True,
            mode="full_refresh",
        )

    def _reuse(self, obs: Observation) -> ActionCommand:
        remaining = self._cached_chunk[self._cursor:]
        reuse_len = min(len(remaining), self.max_reuse_len)
        actions = remaining[:reuse_len]
        self._cursor += reuse_len
        self._draft_rounds += 1

        return ActionCommand(
            actions=actions,
            execute_len=reuse_len,
            can_interrupt=True,
            mode="action_reuse",
        )

    def _should_refresh(self) -> bool:
        return (
            self._cached_chunk is None
            or self._cursor >= self.chunk_len
            or self._draft_rounds >= self.periodic_full_every_n
        )

    @property
    def method_name(self) -> str:
        return "Cached Full Action Reuse"

    @property
    def method_type(self) -> str:
        return "action_reuse"


class CachedActionReuseOffsetWrapper(CachedFullActionReuseWrapper):
    """CachedFullActionReuse + simple ee_pose offset correction.

    Adds proportional position correction based on ee_pose delta.
    No learned components — just a scalar gain.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.correction_gain = config.get("correction_gain", 0.5)

    def _reuse(self, obs: Observation) -> ActionCommand:
        remaining = self._cached_chunk[self._cursor:].clone()
        reuse_len = min(len(remaining), self.max_reuse_len)
        actions = remaining[:reuse_len]

        # Simple position correction
        current_ee = compute_ee_pose(obs.robot_state)
        delta_ee = current_ee - self._cached_ee_pose
        correction = delta_ee[:3] * self.correction_gain  # position only
        actions[:, :3] += correction.unsqueeze(0)

        self._cursor += reuse_len
        self._draft_rounds += 1

        return ActionCommand(
            actions=actions,
            execute_len=reuse_len,
            can_interrupt=True,
            mode="action_reuse",
        )

    @property
    def method_name(self) -> str:
        return "Cached Action Reuse + Offset Correction"
