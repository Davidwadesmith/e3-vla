"""Action Expert feature mixer.

Pools multi-level AE features (ae_low, ae_mid, ae_high) into a single
mixed representation per action step.
"""

import torch
import torch.nn as nn

from e3vla.schema import AEFeatureBundle


class AEFeatureMixer(nn.Module):
    """Cross-level pooling of Action Expert features.

    Supports:
      - Weighted sum: learnable scalar weights per level
      - Attention pooling: learned query attends over levels
      - Simple mean: fixed equal weights (fallback)

    Default: learnable weighted sum for simplicity and interpretability.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        mode: str = "weighted",  # "weighted" | "attention" | "mean"
    ):
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim

        if mode == "weighted":
            # Learnable scalar weight per level
            self.level_weights = nn.Parameter(torch.tensor([0.2, 0.3, 0.5]))
        elif mode == "attention":
            self.level_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        # Projection per level to common dim
        self.ae_low_proj = nn.Linear(hidden_dim, hidden_dim) if mode != "mean" else None
        self.ae_mid_proj = nn.Linear(hidden_dim, hidden_dim) if mode != "mean" else None
        self.ae_high_proj = nn.Linear(hidden_dim, hidden_dim) if mode != "mean" else None

        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, ae_features: AEFeatureBundle) -> torch.Tensor:
        """Pool multi-level AE features into mixed [K, D].

        Handles both 2D [K, D] and 3D [T_flow, K, D] input.
        For 3D input, pools over T_flow dimension first.
        """
        ae_low = self._ensure_2d(ae_features.ae_low)    # [K, D]
        ae_mid = self._ensure_2d(ae_features.ae_mid)
        ae_high = self._ensure_2d(ae_features.ae_high)

        if self.mode == "mean":
            stacked = torch.stack([ae_low, ae_mid, ae_high], dim=0)  # [3, K, D]
            ae_mixed = stacked.mean(dim=0)  # [K, D]

        elif self.mode == "weighted":
            w = torch.softmax(self.level_weights, dim=0)  # [3]
            ae_low_p = self.ae_low_proj(ae_low)
            ae_mid_p = self.ae_mid_proj(ae_mid)
            ae_high_p = self.ae_high_proj(ae_high)
            ae_mixed = w[0] * ae_low_p + w[1] * ae_mid_p + w[2] * ae_high_p

        elif self.mode == "attention":
            stacked = torch.stack([
                self.ae_low_proj(ae_low),
                self.ae_mid_proj(ae_mid),
                self.ae_high_proj(ae_high),
            ], dim=1)  # [K, 3, D]
            query = self.level_query.expand(stacked.shape[0], -1, -1)  # [K, 1, D]
            ae_mixed, _ = self.attn(query, stacked, stacked)  # [K, 1, D]
            ae_mixed = ae_mixed.squeeze(1)

        return self.out_norm(ae_mixed)

    @staticmethod
    def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            # [T_flow, K, D] -> pool over T_flow
            return x.mean(dim=0)
        return x
