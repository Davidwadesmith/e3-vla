"""Speculative Cached Policy.

Orchestrates full refresh → cache → speculative draft → verify → execute loop.
"""

from typing import Optional
import torch

from e3vla.schema import (
    Observation,
    ActionCommand,
    RuntimeCacheRecord,
    CurrentCheapFeatures,
)
from e3vla.protocols import BaseVLAAdapter, BaseDrafter
from e3vla.cache.runtime_cache import RuntimeFeatureCache
from e3vla.align.offset_encoder import OffsetEncoder
from e3vla.verifier.action_expert_anchor_verifier import ActionExpertAnchorVerifier
from e3vla.verifier.prefix_acceptor import PrefixAcceptor
from e3vla.utils import compute_ee_pose


class SpeculativeCachedPolicy:
    """Orchestrates the speculative cached inference pipeline.

    State machine:
      INIT → FULL_REFRESH → SPECULATIVE (loop) → FALLBACK → FULL_REFRESH
    """

    def __init__(
        self,
        adapter: BaseVLAAdapter,
        drafter: BaseDrafter,
        verifier: ActionExpertAnchorVerifier,
        prefix_acceptor: PrefixAcceptor,
        chunk_len: int = 16,
        max_cache_age: int = 50,
        periodic_full_every_n: int = 10,
        full_exec_len: int = 2,     # execute only first N steps from full refresh
        gripper_full_window: int = 3,
        action_history_len: int = 5,
    ):
        self.adapter = adapter
        self.drafter = drafter
        self.verifier = verifier
        self.prefix_acceptor = prefix_acceptor

        self.K = chunk_len
        self.max_cache_age = max_cache_age
        self.periodic_full_every_n = periodic_full_every_n
        self.full_exec_len = min(full_exec_len, chunk_len)
        self.gripper_full_window = gripper_full_window

        self._cache = RuntimeFeatureCache(max_cache_age=max_cache_age)
        self._offset_encoder = OffsetEncoder(chunk_len=chunk_len)

        # State
        self._elapsed_since_full = 0
        self._cache_cursor = 0
        self._draft_rounds_since_full = 0
        self._pending_fallback = False
        self._gripper_full_rounds_left = 0
        self._action_history = []  # list of action tensors
        self._action_history_len = action_history_len

        # Metrics
        self._metrics = {
            "total_acts": 0,
            "speculative_count": 0,
            "fallback_count": 0,
            "full_refresh_count": 0,
            "total_accepted_prefix": 0,
        }

    def reset(self, task_info=None) -> None:
        self._cache.reset()
        self._elapsed_since_full = 0
        self._cache_cursor = 0
        self._draft_rounds_since_full = 0
        self._pending_fallback = False
        self._gripper_full_rounds_left = 0
        self._action_history = []

    def act(self, obs: Observation) -> ActionCommand:
        self._metrics["total_acts"] += 1

        if self._should_full_refresh(obs):
            return self._full_refresh_path(obs)
        else:
            return self._speculative_cached_path(obs)

    def _full_refresh_path(self, obs: Observation) -> ActionCommand:
        self._metrics["full_refresh_count"] += 1

        # Full target VLA forward
        chunk = self.adapter.full_inference(obs)

        # Extract and cache AE features
        ae_features = self.adapter.extract_ae_features(obs, chunk)

        self._cache.update(RuntimeCacheRecord(
            full_round_id=self._metrics["full_refresh_count"],
            episode_id="",
            full_round_timestep=self._metrics["total_acts"],
            ae_features=ae_features,
            full_action_chunk=chunk.detach(),
            full_robot_state=obs.robot_state.detach(),
            full_ee_pose=compute_ee_pose(obs.robot_state).detach(),
            cache_age=0,
            valid_until_step=self.max_cache_age,
        ))

        self._elapsed_since_full = 0
        self._cache_cursor = 0
        self._draft_rounds_since_full = 0
        self._gripper_full_rounds_left = max(0, self._gripper_full_rounds_left - 1)

        return ActionCommand(
            actions=chunk[:self.full_exec_len],
            execute_len=self.full_exec_len,
            can_interrupt=True,
            mode="full_refresh",
        )

    def _speculative_cached_path(self, obs: Observation) -> ActionCommand:
        self._metrics["speculative_count"] += 1

        cached = self._cache.get_latest()
        if cached is None:
            return self._full_refresh_path(obs)

        # Compute offset
        offset = self._offset_encoder.compute(
            obs, cached,
            cache_age=self._cache.cache_age,
            elapsed_steps_since_full=self._elapsed_since_full,
            cache_feature_cursor=self._cache_cursor,
        )

        # Build cheap features
        cheap = CurrentCheapFeatures(
            current_robot_state=obs.robot_state,
            action_history=self._build_action_history(),
            last_executed_action_index=self._cache_cursor,
            gripper_state=obs.robot_state[-1].item(),
        )

        # Draft
        draft = self.drafter(cached.ae_features, offset, cheap)

        # Validity check
        if not self._is_valid_draft(draft):
            self._pending_fallback = True
            self._metrics["fallback_count"] += 1
            return self._full_refresh_path(obs)

        # Low-cost verify
        result = self.verifier.verify(
            obs, draft.action_chunk, self.adapter,
            cached_vlm_context=cached.kv_cache_ref,
            verify_spec={},
            gripper_prev=obs.robot_state[-1].item(),
        )

        # Accept
        accepted_len, fallback_needed = self.prefix_acceptor.accept(
            result.errors_per_step,
            draft.uncertainty.squeeze(0) if draft.uncertainty.dim() == 3 else draft.uncertainty,
            draft.action_chunk.squeeze(0) if draft.action_chunk.dim() == 3 else draft.action_chunk,
            offset.gripper_phase,
        )

        if fallback_needed or accepted_len == 0:
            self._schedule_fallback(gripper=("gripper" in result.reason))
            self._metrics["fallback_count"] += 1
            return self._full_refresh_path(obs)

        # Execute accepted prefix
        actions = draft.action_chunk[:accepted_len]
        self._update_action_history(actions)

        self._elapsed_since_full += accepted_len
        self._cache_cursor = (self._cache_cursor + accepted_len) % self.K
        self._cache.increment_age(accepted_len)
        self._draft_rounds_since_full += 1
        self._metrics["total_accepted_prefix"] += accepted_len

        return ActionCommand(
            actions=actions.squeeze(0) if actions.dim() == 3 else actions,
            execute_len=accepted_len,
            can_interrupt=True,
            mode="speculative",
            prefix_length=accepted_len,
            diagnostics={
                "cache_age": self._cache.cache_age,
                "elapsed_since_full": self._elapsed_since_full,
                "cache_feature_cursor": self._cache_cursor,
                "accepted_len": accepted_len,
            },
        )

    def _should_full_refresh(self, obs) -> bool:
        return (
            self._pending_fallback
            or self._gripper_full_rounds_left > 0
            or self._cache.cache_age >= self.max_cache_age
            or (self.periodic_full_every_n > 0
                and self._draft_rounds_since_full >= self.periodic_full_every_n)
            or not self._cache.is_valid
        )

    def _schedule_fallback(self, gripper: bool = False) -> None:
        self._pending_fallback = True
        if gripper:
            self._gripper_full_rounds_left = self.gripper_full_window

    def _is_valid_draft(self, draft) -> bool:
        chunk = draft.action_chunk
        if torch.isnan(chunk).any() or torch.isinf(chunk).any():
            return False
        return True

    def _build_action_history(self) -> Optional[torch.Tensor]:
        if not self._action_history:
            return None
        hist = torch.cat(self._action_history[-self._action_history_len:], dim=0)
        if hist.shape[0] < self._action_history_len * self.K:
            pad = torch.zeros(
                self._action_history_len * self.K - hist.shape[0], hist.shape[-1]
            )
            hist = torch.cat([pad, hist], dim=0)
        return hist

    def _update_action_history(self, actions: torch.Tensor) -> None:
        self._action_history.append(actions.detach().cpu().reshape(-1, actions.shape[-1]))

    def get_metrics(self) -> dict:
        total = max(1, self._metrics["total_acts"])
        spec = max(1, self._metrics["speculative_count"])
        return {
            **self._metrics,
            "avg_accepted_prefix_len": self._metrics["total_accepted_prefix"] / spec,
            "fallback_rate": self._metrics["fallback_count"] / total,
            "full_refresh_rate": self._metrics["full_refresh_count"] / total,
        }
