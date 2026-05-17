"""Offset features encoder.

Computes OffsetFeatures from current observation, cached full round data,
and elapsed state counters.
"""

from typing import Optional
import torch
import torch.nn as nn

from e3vla.schema import OffsetFeatures, RuntimeCacheRecord, Observation
from e3vla.utils import compute_ee_pose


class OffsetEncoder:
    """Computes temporal/pose/gripper offsets between cached full round and current timestep.

    This is a stateless feature computer (no learned parameters here).
    Learned offset processing happens in OffsetAlignAdapter.
    """

    def __init__(self, chunk_len: int = 16):
        self._chunk_len = chunk_len

    def compute(
        self,
        obs: Observation,
        cached: RuntimeCacheRecord,
        cache_age: int,
        elapsed_steps_since_full: int,
        cache_feature_cursor: int,
    ) -> OffsetFeatures:
        # Current ee pose
        current_ee_pose = compute_ee_pose(obs.robot_state)

        # Delta computation
        delta_ee_pose = current_ee_pose - cached.full_ee_pose
        delta_robot_state = obs.robot_state - cached.full_robot_state

        # Gripper phase classification
        gripper_val = obs.robot_state[-1].item()  # last dim is gripper
        gripper_phase = self._classify_gripper_phase(gripper_val)

        return OffsetFeatures(
            cache_age=cache_age,
            elapsed_steps_since_full=elapsed_steps_since_full,
            cache_feature_cursor=cache_feature_cursor,
            draft_step_index=0,  # new chunk starts from current t
            delta_robot_state=delta_robot_state,
            delta_ee_pose=delta_ee_pose,
            gripper_phase=gripper_phase,
        )

    @staticmethod
    def _classify_gripper_phase(gripper_val: float) -> float:
        """Classify gripper into -1 (close), 0 (transition), +1 (open)."""
        threshold = 0.0
        if gripper_val < -0.2:
            return -1.0
        elif gripper_val > 0.2:
            return 1.0
        else:
            return 0.0
