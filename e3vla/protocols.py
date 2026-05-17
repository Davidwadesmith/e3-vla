"""Base protocols for swappable components.

Defines the abstract interfaces for VLA adapters, drafters, verifiers,
and policies. All concrete implementations must satisfy these protocols.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch

from e3vla.schema import (
    Observation,
    ActionCommand,
    DraftOutput,
    VerificationResult,
    AEFeatureBundle,
)


class BaseVLAAdapter(ABC):
    """Unified interface over different VLA model implementations.

    Shields the rest of the system from model-specific details.
    All access to the target VLA goes through this adapter.
    """

    @abstractmethod
    def full_inference(self, obs: Observation) -> torch.Tensor:
        """Run complete target VLA forward pass.

        Returns action chunk [K, D_a].
        """
        ...

    @abstractmethod
    def extract_ae_features(
        self, obs: Observation, action_chunk: torch.Tensor
    ) -> AEFeatureBundle:
        """Extract Action Expert intermediate features during full inference.

        Called only during full refresh rounds.
        Returns multi-level AE features for caching.
        """
        ...

    @abstractmethod
    def action_expert_denoise_step(
        self,
        x_t: torch.Tensor,     # [B, K, D_a]
        t: float,
        vlm_context,            # cached VLM KV cache (stale, from last full refresh)
        robot_state: torch.Tensor,  # [B, D_r] fresh current robot state
    ) -> torch.Tensor:
        """Single denoising step of the Action Expert.

        Used by ActionExpertAnchorVerifier for low-cost verification.
        Does NOT run vision encoder or VLM prefill.

        Returns predicted velocity v_t [B, K, D_a].
        """
        ...


class BaseDrafter(ABC):
    """Lightweight model that predicts draft action chunks."""

    @abstractmethod
    def forward(
        self,
        cached_ae_features: Optional[AEFeatureBundle],
        offset_features,  # OffsetFeatures or None
        cheap_features,   # CurrentCheapFeatures
    ) -> DraftOutput:
        """Predict draft action chunk and uncertainty."""
        ...

    @abstractmethod
    def compute_loss(
        self,
        batch,            # TrainingSample batch
    ) -> dict:
        """Compute training losses. Returns dict of {name: scalar_tensor}."""
        ...


class BaseVerifier(ABC):
    """Checks draft action chunk quality against target model."""

    @abstractmethod
    def verify(
        self,
        obs: Observation,
        draft_chunk: torch.Tensor,        # [K, D_a]
        adapter: BaseVLAAdapter,
        cached_vlm_context,                # from last full refresh
        verify_spec: dict,
    ) -> VerificationResult:
        """Verify draft chunk and return acceptance result."""
        ...


class BasePrefixAcceptor(ABC):
    """Decides how many prefix steps of a draft chunk to accept."""

    @abstractmethod
    def accept(
        self,
        errors_per_step: torch.Tensor,     # [K]
        uncertainty: torch.Tensor,          # [K, 3]
        draft_chunk: torch.Tensor,          # [K, D_a]
        gripper_phase: float,
    ) -> tuple[int, bool]:
        """Returns (accepted_len, fallback_required)."""
        ...
