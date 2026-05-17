"""NoCachedAE ablation: FLASH-style drafter without cached AE features."""

import torch
import torch.nn as nn

from e3vla.schema import AEFeatureBundle, CurrentCheapFeatures, DraftOutput


class NoCachedAEDrafter(nn.Module):
    """Baseline drafter using only current cheap features (no cached AE).

    This is equivalent to a FLASH-style drafter input.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        action_dim: int = 7,
        chunk_len: int = 16,
        num_layers: int = 2,
        num_heads: int = 4,
        D_r: int = 16,
        action_history_len: int = 5,
    ):
        super().__init__()
        self.chunk_len = chunk_len

        self.robot_state_proj = nn.Linear(D_r, hidden_dim)
        self.action_history_proj = nn.Linear(action_history_len * action_dim, hidden_dim)

        self.context_fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1,
            activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.pos_emb = nn.Parameter(torch.randn(1, chunk_len, hidden_dim) * 0.02)

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.uncert_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, cached_ae_features, offset_features, cheap_features):
        B = cheap_features.current_robot_state.shape[0]
        K = self.chunk_len
        D = self.pos_emb.shape[-1]

        state_feat = self.robot_state_proj(cheap_features.current_robot_state)
        state_feat = state_feat.unsqueeze(1).expand(-1, K, -1)

        hist = cheap_features.action_history
        if hist is not None:
            hist_flat = hist.reshape(B, -1)
            hist_feat = self.action_history_proj(hist_flat).unsqueeze(1).expand(-1, K, -1)
        else:
            hist_feat = torch.zeros(B, K, D)

        fused = self.context_fusion(torch.cat([state_feat, hist_feat], dim=-1))

        queries = self.pos_emb.expand(B, -1, -1)
        hidden = self.transformer(queries + fused, fused)

        actions = self.action_head(hidden)
        uncertainty = torch.nn.functional.softplus(self.uncert_head(hidden))

        return DraftOutput(
            action_chunk=actions,
            uncertainty=uncertainty,
            hidden_states=hidden,
        )

    def compute_loss(self, batch):
        raise NotImplementedError("Use CachedAEDrafter.compute_loss instead")
