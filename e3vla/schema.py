"""Core data schemas for E3-VLA.

All tensor shapes follow the convention:
  B: batch size
  K: action chunk length
  D: hidden dimension
  D_a: action dimension (typically 7: pos3 + rot3 + grip1)
  D_r: robot state dimension
  D_ee: end-effector pose dimension
  H: action history length
  T_flow: flow matching timesteps
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal
import torch


@dataclass
class Observation:
    image: torch.Tensor                          # [B, C, H_img, W_img] or list of views
    instruction: str
    robot_state: torch.Tensor                    # [B, D_r]
    history: Optional[torch.Tensor] = None       # [B, H_hist, D_a]
    env_info: Optional[Dict[str, Any]] = None


@dataclass
class ActionCommand:
    actions: torch.Tensor              # [execute_len, D_a]
    execute_len: int                   # number of action steps to actually execute
    can_interrupt: bool                # whether env can interrupt before execute_len
    mode: Literal["full_refresh", "speculative", "action_reuse", "fallback"]
    prefix_length: int = 0
    confidence: float = 1.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AEFeatureBundle:
    """Action Expert multi-level intermediate features from a full refresh round."""
    ae_low: torch.Tensor    # [K, D] or [T_flow, K, D]
    ae_mid: torch.Tensor    # [K, D] or [T_flow, K, D]
    ae_high: torch.Tensor   # [K, D] or [T_flow, K, D]
    ae_mixed: torch.Tensor  # [K, D]  weighted or attention-pooled mixture


@dataclass
class RuntimeCacheRecord:
    full_round_id: int
    episode_id: str
    full_round_timestep: int
    ae_features: AEFeatureBundle
    full_action_chunk: torch.Tensor     # [K, D_a]
    full_robot_state: torch.Tensor      # [D_r]
    full_ee_pose: torch.Tensor          # [D_ee]
    cache_age: int = 0
    valid_until_step: int = 50
    kv_cache_ref: Optional[Any] = None


@dataclass
class CurrentCheapFeatures:
    """Features available cheaply during speculative rounds (no full VLA)."""
    current_robot_state: torch.Tensor                       # [D_r]
    optional_current_image_feature: Optional[torch.Tensor] = None  # [D]
    action_history: Optional[torch.Tensor] = None           # [H, D_a]
    last_executed_action_index: int = 0
    gripper_state: float = 0.0


@dataclass
class OffsetFeatures:
    """Offset between cached full round and current timestep."""
    cache_age: int                          # env steps since last full refresh
    elapsed_steps_since_full: int           # real env steps since full refresh
    cache_feature_cursor: int               # position in cached K-token (0..K-1)
    draft_step_index: int                   # start position in new draft chunk (usually 0)
    delta_robot_state: torch.Tensor         # [D_r]
    delta_ee_pose: torch.Tensor             # [D_ee]
    gripper_phase: float                    # -1 (close), 0 (transition), +1 (open)


@dataclass
class DraftOutput:
    action_chunk: torch.Tensor    # [B, K, D_a]
    uncertainty: torch.Tensor     # [B, K, 3]  pos/rot/grip
    hidden_states: torch.Tensor   # [B, K, D]


@dataclass
class VerificationResult:
    accepted_prefix_length: int
    errors_per_step: torch.Tensor       # [K]
    fallback_required: bool
    confidence_per_step: torch.Tensor   # [K]
    verification_latency_ms: float
    reason: str  # "accepted" | "rejected_action_error" | "rejected_gripper_phase" | ...


@dataclass
class TrainingSample:
    """Cross-temporal training pair for drafter training."""
    cached_full_round_features: AEFeatureBundle
    full_round_robot_state: torch.Tensor     # [D_r]
    full_round_ee_pose: torch.Tensor         # [D_ee]
    current_timestep: int
    current_robot_state: torch.Tensor        # [D_r]
    current_ee_pose: torch.Tensor            # [D_ee]
    action_history: torch.Tensor             # [H, D_a]
    delta_t: int
    delta_ee_pose: torch.Tensor              # [D_ee]
    delta_robot_state: torch.Tensor          # [D_r]
    delta_action_index: int
    gripper_phase: float
    target_action_chunk: torch.Tensor        # [K, D_a]


@dataclass
class PolicyMetrics:
    total_acts: int = 0
    speculative_count: int = 0
    fallback_count: int = 0
    full_refresh_count: int = 0
    total_accepted_prefix: int = 0
    avg_accepted_prefix_len: float = 0.0
    fallback_rate: float = 0.0
    full_refresh_rate: float = 0.0
    avg_cache_age_at_accept: float = 0.0
    latency_breakdown: Dict[str, float] = field(default_factory=dict)
