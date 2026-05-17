"""CachedAE-NoOffset ablation: cached AE features without offset alignment."""

import torch
import torch.nn as nn

from e3vla.schema import AEFeatureBundle, CurrentCheapFeatures, DraftOutput
from e3vla.drafter.ae_feature_mixer import AEFeatureMixer


class CachedAENoOffsetDrafter(nn.Module):
    """Uses cached AE features but skips offset alignment.

    Ablation purpose: measure the marginal value of offset alignment.
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
        mixer_mode: str = "weighted",
    ):
        super().__init__()
        self.chunk_len = chunk_len

        self.ae_mixer = AEFeatureMixer(hidden_dim=hidden_dim, mode=mixer_mode)
        self.ae_proj = nn.Linear(hidden_dim, hidden_dim)

        self.robot_state_proj = nn.Linear(D_r, hidden_dim)
        self.action_history_proj = nn.Linear(
            action_history_len * action_dim, hidden_dim
        ) if action_history_len > 0 else None

        fusion_input_dim = hidden_dim * (3 if action_history_len > 0 else 2)
        self.context_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
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

    def forward(
        self,
        cached_ae_features: AEFeatureBundle,
        offset_features,
        cheap_features: CurrentCheapFeatures,
    ) -> DraftOutput:
        B = cheap_features.current_robot_state.shape[0]
        K = self.chunk_len
        D = self.pos_emb.shape[-1]

        # Cached AE mixed (no offset alignment)
        ae_mixed = self.ae_mixer(cached_ae_features)  # [K, D]
        ae_mixed = self.ae_proj(ae_mixed)
        ae_feat = ae_mixed.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]

        # Robot state
        state_feat = self.robot_state_proj(cheap_features.current_robot_state)
        state_feat = state_feat.unsqueeze(1).expand(-1, K, -1)

        context_parts = [ae_feat, state_feat]

        if self.action_history_proj is not None and cheap_features.action_history is not None:
            hist_flat = cheap_features.action_history.reshape(B, -1)
            hist_feat = self.action_history_proj(hist_flat).unsqueeze(1).expand(-1, K, -1)
            context_parts.append(hist_feat)

        fused = self.context_fusion(torch.cat(context_parts, dim=-1))

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
        raise NotImplementedError("Use CachedAEDrafter.compute_loss")
