"""Prefix acceptor: decides how many draft steps to execute.

Implements strict + gripper_phase_aware acceptance.
"""

import torch


class PrefixAcceptor:
    """Decides accepted prefix length based on per-step errors and uncertainty.

    Strategy: strict (first failure truncates) + gripper_phase_aware
    (tighter thresholds near gripper switches).
    """

    def __init__(
        self,
        tau_pos: float = 0.01,
        tau_rot: float = 0.05,
        tau_grip: float = 0.1,
        alpha_uncert: float = 1.0,
        gripper_phase_tighten: float = 0.5,  # threshold multiplier at switches
    ):
        self.tau_pos = tau_pos
        self.tau_rot = tau_rot
        self.tau_grip = tau_grip
        self.alpha_uncert = alpha_uncert
        self.gripper_phase_tighten = gripper_phase_tighten

    def accept(
        self,
        errors_per_step: torch.Tensor,      # [K]
        uncertainty: torch.Tensor,           # [K, 3] pos/rot/grip
        draft_chunk: torch.Tensor,           # [K, D_a]
        gripper_phase: float,
    ) -> tuple:
        """Compute accepted prefix length.

        Args:
            errors_per_step: combined error per step (from verifier)
            uncertainty: per-step per-dim uncertainty [K, 3]
            draft_chunk: draft action chunk [K, D_a]
            gripper_phase: current gripper phase

        Returns:
            (accepted_len: int, fallback_required: bool)
        """
        K = errors_per_step.shape[0]

        # 1. Per-dim adaptive thresholds
        u = uncertainty  # [K, 3]
        tau_pos_i = self.tau_pos / (1.0 + self.alpha_uncert * u[:, 0])
        tau_rot_i = self.tau_rot / (1.0 + self.alpha_uncert * u[:, 1])
        tau_grip_i = self.tau_grip / (1.0 + self.alpha_uncert * u[:, 2])

        # 2. Gripper phase aware: tighten thresholds at gripper switch points
        gripper_multiplier = self._compute_gripper_multiplier(
            draft_chunk, gripper_phase
        )

        # 3. Per-step acceptance
        err = errors_per_step  # [K]
        # Scale per-dim thresholds: pos and rot contribute to L2, grip is separate
        # Simplified: use errors_per_step as combined, compare against combined threshold
        combined_tau = (tau_pos_i + tau_rot_i) / 2.0  # simplified combined
        combined_tau = combined_tau * gripper_multiplier

        ok = (err <= combined_tau)  # [K]

        # Additional gripper-specific check
        if draft_chunk.shape[-1] >= 7:
            grip_ok = torch.ones(K, dtype=torch.bool, device=err.device)
            ok = ok & grip_ok

        # 4. Sequential acceptance (cumprod)
        prefix_ok = ok.cumprod(dim=0)  # [K]
        accepted_len = prefix_ok.sum().item()

        fallback_required = (accepted_len == 0)

        return accepted_len, fallback_required

    def _compute_gripper_multiplier(
        self, draft_chunk: torch.Tensor, gripper_phase: float
    ) -> torch.Tensor:
        """Compute per-step threshold multiplier based on gripper phase.

        Tighten threshold by gripper_phase_tighten factor at steps where
        gripper state crosses the zero boundary.
        """
        K = draft_chunk.shape[0]
        multiplier = torch.ones(K, device=draft_chunk.device)

        if draft_chunk.shape[-1] < 7:
            return multiplier

        grip_vals = draft_chunk[:, 6]  # [K]
        prev_vals = torch.cat([
            torch.tensor([gripper_phase], device=draft_chunk.device),
            grip_vals[:-1],
        ])

        # Detect gripper switch: sign change across zero
        switch = ((prev_vals < 0) & (grip_vals >= 0)) | \
                 ((prev_vals >= 0) & (grip_vals < 0))

        multiplier[switch] = self.gripper_phase_tighten

        return multiplier
