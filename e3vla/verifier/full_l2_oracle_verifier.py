"""Full L2 Oracle Verifier.

Runs complete target VLA forward for oracle upper bound / debug.
NOT used in main inference path — too expensive for speedup.
"""

import math
import torch

from e3vla.schema import Observation, VerificationResult
from e3vla.protocols import BaseVLAAdapter


class FullL2OracleVerifier:
    """Oracle verifier using full VLA inference + L2 comparison.

    Purpose: oracle upper bound, offline analysis, debug.
    Banned from: main inference path, main speedup table.

    This verifier runs FULL target VLA forward, which provides
    no speedup for non-AR action chunk policies.
    """

    def __init__(
        self,
        tau_pos: float = 0.01,
        tau_rot: float = 0.05,
        tau_grip: float = 0.1,
        eval_h: int = 12,
    ):
        self.tau_pos = tau_pos
        self.tau_rot = tau_rot
        self.tau_grip = tau_grip
        self.eval_h = eval_h

    def verify(
        self,
        obs: Observation,
        draft_chunk: torch.Tensor,     # [K, D_a] or [B, K, D_a]
        adapter: BaseVLAAdapter,
        cached_vlm_context=None,        # unused — runs full VLA
        verify_spec: dict = None,
    ) -> VerificationResult:
        """Run full VLA forward and compare with draft.

        WARNING: This runs the complete target VLA.
        Do NOT use as the primary verifier.
        """
        if draft_chunk.dim() == 2:
            draft_chunk = draft_chunk.unsqueeze(0)

        B, H, D_a = draft_chunk.shape
        eval_h = min(H, max(1, self.eval_h))

        # Full VLA inference — EXPENSIVE
        target_chunk = adapter.full_inference(obs)  # [K, D_a]
        target_chunk = target_chunk.unsqueeze(0) if target_chunk.dim() == 2 else target_chunk

        # Per-dim errors
        err_pos = torch.norm(
            draft_chunk[:, :eval_h, :3] - target_chunk[:, :eval_h, :3],
            dim=-1,
        )  # [B, eval_h]

        err_rot = torch.norm(
            draft_chunk[:, :eval_h, 3:6] - target_chunk[:, :eval_h, 3:6],
            dim=-1,
        )

        err_grip = torch.abs(
            draft_chunk[:, :eval_h, 6] - target_chunk[:, :eval_h, 6]
        )

        # Sequential acceptance with per-dim thresholds
        ok_pos = err_pos <= self.tau_pos
        ok_rot = err_rot <= self.tau_rot
        ok_grip = err_grip <= self.tau_grip
        ok = ok_pos & ok_rot & ok_grip  # [B, eval_h]

        prefix_ok = ok.cumprod(dim=-1)
        accepted_len = int(prefix_ok.sum(dim=-1).min().item())

        return VerificationResult(
            accepted_prefix_length=accepted_len,
            errors_per_step=torch.cat([
                err_pos,
                err_rot,
                err_grip.unsqueeze(-1),
            ], dim=-1).squeeze(0),  # simplified
            fallback_required=(accepted_len == 0),
            confidence_per_step=torch.ones(eval_h),
            verification_latency_ms=0.0,
            reason="oracle_l2",
        )
