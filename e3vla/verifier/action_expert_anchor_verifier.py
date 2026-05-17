"""Action Expert Anchor Verifier.

Low-cost verification using only the Action Expert's denoising step
at a few flow timesteps. Does NOT run full VLA forward.

Uses stale VLM/KV context from last full refresh + fresh robot_state.
"""

import math
import torch
import torch.nn as nn

from e3vla.schema import Observation, VerificationResult
from e3vla.protocols import BaseVLAAdapter


class ActionExpertAnchorVerifier:
    """Low-cost verification via flow-consistency check.

    Verifier context:
      - VLM context / KV cache: stale (from last full refresh)
      - robot_state: fresh (from current observation)
      - Semi-open-loop: proprioception is current, vision context is stale.
        Relies on fallback / cache_age guard / gripper guard for safety.

    Reference: FLASH spec_pi0_pytorch._compute_radius_prefix_acceptance
    """

    def __init__(
        self,
        t_list: tuple = (0.10, 0.05),
        tau_radius: float = 0.3,
        dist_dims: int = 6,    # exclude gripper dim from distance
        eval_h: int = 12,
        gripper_switch_threshold: float = 0.0,
    ):
        self.t_list = t_list
        self.tau_radius = tau_radius
        self.dist_dims = dist_dims
        self.eval_h = eval_h
        self.gripper_switch_threshold = gripper_switch_threshold

    def verify(
        self,
        obs: Observation,
        draft_chunk: torch.Tensor,           # [K, D_a] or [B, K, D_a]
        adapter: BaseVLAAdapter,
        cached_vlm_context,                   # stale VLM context
        verify_spec: dict,
        gripper_prev: float = 0.0,
    ) -> VerificationResult:
        """Verify draft chunk using action expert anchor points.

        Args:
            cached_vlm_context: stale VLM context from last full refresh
            robot_state: fresh current robot state
        """
        if draft_chunk.dim() == 2:
            draft_chunk = draft_chunk.unsqueeze(0)  # [1, K, D_a]

        B, H, D_a = draft_chunk.shape
        eval_h = min(H, max(1, self.eval_h))
        eval_d = min(D_a, self.dist_dims)
        if D_a >= 7:
            eval_d = min(eval_d, 6)  # exclude gripper dim

        x0_hat_list = []

        for t_k in self.t_list:
            # 1. Construct intermediate state: x_t = t_k * noise + (1 - t_k) * draft
            noise = torch.randn_like(draft_chunk)
            x_t = t_k * noise + (1.0 - t_k) * draft_chunk

            # 2. Single Action Expert denoising step
            v_t = adapter.action_expert_denoise_step(
                x_t, t_k,
                vlm_context=cached_vlm_context,     # stale
                robot_state=obs.robot_state,         # fresh
            )

            # 3. Reconstruct endpoint: x0_hat = x_t - t_k * v_t
            x0_hat_k = x_t - t_k * v_t
            x0_hat_list.append(x0_hat_k)

        x0_hat = torch.stack(x0_hat_list, dim=1)  # [B, K_v, H, D_a]

        # RMS normalized distance
        diff = x0_hat[:, :, :eval_h, :eval_d] - draft_chunk[:, None, :eval_h, :eval_d]
        dist = torch.norm(diff, dim=-1) / math.sqrt(max(eval_d, 1))  # [B, K_v, eval_h]

        # All verify paths must agree
        ok = (dist <= self.tau_radius)  # [B, K_v, eval_h]
        prefix_mask = ok.cumprod(dim=-1)  # sequential acceptance
        accepted_len_k = prefix_mask.sum(dim=-1)  # [B, K_v]
        accepted_len = accepted_len_k.min(dim=1).values  # [B]

        # Gripper switch detection (pre-verify)
        if D_a >= 7 and accepted_len.item() > 0:
            has_switch = self._detect_gripper_switch(
                x0_hat, gripper_prev, accepted_len.item()
            )
            if has_switch:
                accepted_len = torch.zeros_like(accepted_len)

        # Truncate at first gripper switch within accepted prefix
        if D_a >= 7 and accepted_len.item() > 1:
            accepted_len = self._truncate_on_gripper_switch(
                draft_chunk, gripper_prev, accepted_len
            )

        fallback_required = (accepted_len <= 0).any()

        return VerificationResult(
            accepted_prefix_length=int(accepted_len.item()),
            errors_per_step=dist.mean(dim=1).squeeze(0),  # [H]
            fallback_required=fallback_required,
            confidence_per_step=1.0 - dist.mean(dim=1).squeeze(0) / self.tau_radius,
            verification_latency_ms=0.0,  # filled by profiler
            reason="accepted" if not fallback_required else "rejected_action_error",
        )

    def _detect_gripper_switch(
        self,
        x0_hat: torch.Tensor,   # [B, K_v, H, D_a]
        gripper_prev: float,
        eval_h: int,
    ) -> bool:
        """Check if any verify path shows a gripper state switch."""
        B, K_v, H, _ = x0_hat.shape
        h2 = min(H, eval_h)

        prev_vals = torch.cat([
            torch.full((B, K_v, 1), gripper_prev, device=x0_hat.device),
            x0_hat[:, :, :h2-1, 6],
        ], dim=-1)  # [B, K_v, h2]

        curr_vals = x0_hat[:, :, :h2, 6]  # [B, K_v, h2]

        threshold = self.gripper_switch_threshold
        switch = ((prev_vals < threshold) & (curr_vals >= threshold)) | \
                 ((prev_vals >= threshold) & (curr_vals < threshold))

        return switch.any(dim=-1).any(dim=-1).any().item()

    def _truncate_on_gripper_switch(
        self,
        draft_chunk: torch.Tensor,  # [B, H, D_a]
        gripper_prev: float,
        accepted_len: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """Truncate accepted prefix at first gripper switch position."""
        B, H, _ = draft_chunk.shape
        a_len = int(accepted_len.item())

        prev_vals = torch.cat([
            torch.full((B, 1), gripper_prev, device=draft_chunk.device),
            draft_chunk[:, :a_len-1, 6],
        ], dim=-1)

        curr_vals = draft_chunk[:, :a_len, 6]
        threshold = self.gripper_switch_threshold
        switch = ((prev_vals < threshold) & (curr_vals >= threshold)) | \
                 ((prev_vals >= threshold) & (curr_vals < threshold))

        if switch.any():
            first_switch = switch.float().argmax(dim=-1)
            return first_switch.unsqueeze(0).float()

        return accepted_len
