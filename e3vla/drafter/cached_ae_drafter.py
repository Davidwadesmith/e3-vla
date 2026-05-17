"""Cached Action Expert Drafter.

The core drafter that conditions on cached AE latent features
plus temporal/pose offsets to predict draft action chunks.
"""

from typing import Optional
import torch
import torch.nn as nn

from e3vla.schema import AEFeatureBundle, CurrentCheapFeatures, DraftOutput
from e3vla.align.ae_feature_aligner import OffsetAlignAdapter
from e3vla.drafter.ae_feature_mixer import AEFeatureMixer


class CachedAEDrafter(nn.Module):
    """Drafter conditioned on cached AE features + offset alignment.

    Architecture:
      1. AE Feature Mixer: pool ae_low/ae_mid/ae_high → ae_mixed [K, D]
      2. OffsetAlignAdapter: gated alignment + cross-attention re-indexing
      3. Context fusion: concat aligned AE + state + history + optional image
      4. Lightweight transformer over K action queries
      5. Dual output heads: action + uncertainty
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        action_dim: int = 7,
        chunk_len: int = 16,
        num_layers: int = 2,
        num_heads: int = 8,
        D_r: int = 16,
        D_ee: int = 7,
        action_history_len: int = 5,
        use_image_feature: bool = False,
        mixer_mode: str = "weighted",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.use_image_feature = use_image_feature

        # 1. AE feature mixer
        self.ae_mixer = AEFeatureMixer(hidden_dim=hidden_dim, mode=mixer_mode)

        # 2. Offset alignment adapter
        self.aligner = OffsetAlignAdapter(
            hidden_dim=hidden_dim,
            max_cache_age=50,
            max_chunk_len=chunk_len,
            num_heads=num_heads,
            D_ee=D_ee,
            D_r=D_r,
        )

        # 3. Context encoding
        self.robot_state_proj = nn.Linear(D_r, hidden_dim)
        self.action_history_proj = nn.Linear(
            action_history_len * action_dim, hidden_dim
        ) if action_history_len > 0 else None
        self.image_proj = nn.Linear(hidden_dim, hidden_dim) if use_image_feature else None

        # Compute fusion input dim
        fusion_input_dim = hidden_dim  # aligned AE (always)
        fusion_input_dim += hidden_dim  # robot state (always)
        if action_history_len > 0:
            fusion_input_dim += hidden_dim  # action history
        if use_image_feature:
            fusion_input_dim += hidden_dim  # image feature

        self.context_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 4. Lightweight transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Learned positional embeddings for K action queries
        self.pos_emb = nn.Parameter(torch.randn(1, chunk_len, hidden_dim) * 0.02)

        # 5. Output heads
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.uncert_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),  # pos/rot/grip
        )

    def forward(
        self,
        cached_ae_features: Optional[AEFeatureBundle],
        offset_features,   # OffsetFeatures | None
        cheap_features: CurrentCheapFeatures,
    ) -> DraftOutput:
        """Predict draft action chunk.

        Args:
            cached_ae_features: AE features from last full refresh (None for NoCachedAE)
            offset_features: temporal/pose offsets (None for NoOffset ablation)
            cheap_features: current robot state, action history, etc.

        Returns:
            DraftOutput with action_chunk [B, K, D_a] and uncertainty [B, K, 3]
        """
        B = cheap_features.current_robot_state.shape[0]
        K = self.chunk_len
        D = self.hidden_dim

        # Initialize learned action queries
        draft_queries = self.pos_emb.expand(B, -1, -1)  # [B, K, D]

        # 1. Process cached AE features
        if cached_ae_features is not None:
            ae_mixed = self.ae_mixer(cached_ae_features)  # [K, D]
            ae_mixed = ae_mixed.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]

            if offset_features is not None:
                # Full alignment pipeline
                aligned_ae = self.aligner(ae_mixed, offset_features, draft_queries)
            else:
                # NoOffset ablation: use cached AE directly without alignment
                aligned_ae = self.aligner.out_norm(ae_mixed + draft_queries)
        else:
            # NoCachedAE ablation: learned null token fills AE slot
            aligned_ae = self.aligner.learned_null.expand(B, K, -1)

        # 2. Encode current cheap features
        state_feat = self.robot_state_proj(cheap_features.current_robot_state)
        state_feat = state_feat.unsqueeze(1).expand(-1, K, -1)  # [B, K, D]

        context_parts = [aligned_ae, state_feat]

        # Action history
        if self.action_history_proj is not None and cheap_features.action_history is not None:
            hist_flat = cheap_features.action_history.reshape(B, -1)
            hist_feat = self.action_history_proj(hist_flat)
            hist_feat = hist_feat.unsqueeze(1).expand(-1, K, -1)
            context_parts.append(hist_feat)

        # Optional image feature
        if self.use_image_feature and cheap_features.optional_current_image_feature is not None:
            img_feat = self.image_proj(cheap_features.optional_current_image_feature)
            img_feat = img_feat.unsqueeze(1).expand(-1, K, -1)
            context_parts.append(img_feat)

        # 3. Fuse context
        fused = self.context_fusion(torch.cat(context_parts, dim=-1))  # [B, K, D]

        # 4. Lightweight transformer: self-attention over K action steps
        # Use fused as both query and memory (pure self-attention, no cross-attn)
        hidden = self.transformer(fused, fused)  # [B, K, D]

        # 5. Output heads
        actions = self.action_head(hidden)       # [B, K, D_a]
        uncertainty = self.uncert_head(hidden)   # [B, K, 3]
        uncertainty = torch.nn.functional.softplus(uncertainty)  # ensure > 0

        return DraftOutput(
            action_chunk=actions,
            uncertainty=uncertainty,
            hidden_states=hidden,
        )

    def compute_loss(self, batch) -> dict:
        """Compute training losses.

        Args:
            batch: dict containing TrainingSample fields as batched tensors.

        Returns:
            dict with loss components.
        """
        # Build features from batch
        ae_bundle = AEFeatureBundle(
            ae_low=batch["cached_ae_low"],
            ae_mid=batch["cached_ae_mid"],
            ae_high=batch["cached_ae_high"],
            ae_mixed=batch["cached_ae_mixed"],
        )

        cheap = CurrentCheapFeatures(
            current_robot_state=batch["current_robot_state"],
            action_history=batch.get("action_history"),
            last_executed_action_index=batch.get("delta_action_index", 0),
            gripper_state=batch.get("gripper_phase", 0.0).float(),
        )

        draft = self.forward(ae_bundle, None, cheap)
        target = batch["target_action_chunk"]

        # Huber action loss
        l_action = torch.nn.functional.smooth_l1_loss(
            draft.action_chunk, target, beta=1.0, reduction="none"
        )
        l_action = l_action.mean(dim=(0, 2))  # [K]

        # Prefix-weighted sum
        K = l_action.shape[0]
        prefix_weights = torch.tensor(
            [0.9 ** i for i in range(K)], device=l_action.device
        )
        tail_weight = 0.1
        sample_prefix = torch.randint(1, K + 1, (1,)).item()
        weights = torch.ones(K, device=l_action.device) * tail_weight
        weights[:sample_prefix] = prefix_weights[:sample_prefix]
        weights = weights / weights.sum()

        l_action = (l_action * weights).sum()

        # Temporal smoothness
        l_smooth = torch.nn.functional.mse_loss(
            draft.action_chunk[:, 1:], draft.action_chunk[:, :-1]
        )

        # Gripper BCE
        l_gripper = torch.nn.functional.binary_cross_entropy_with_logits(
            draft.action_chunk[..., 6], target[..., 6]
        )

        # Uncertainty calibration: MSE between uncertainty and actual error
        actual_err = torch.abs(draft.action_chunk.detach() - target)
        # Group by pos(0:3), rot(3:6), grip(6:7)
        err_pos = actual_err[..., :3].norm(dim=-1, keepdim=True)  # [B, K, 1]
        err_rot = actual_err[..., 3:6].norm(dim=-1, keepdim=True)
        err_grip = actual_err[..., 6:7]
        target_uncert = torch.cat([err_pos, err_rot, err_grip], dim=-1)  # [B, K, 3]

        l_uncert = torch.nn.functional.mse_loss(draft.uncertainty, target_uncert)

        return {
            "l_action": l_action,
            "l_smooth": l_smooth,
            "l_gripper": l_gripper,
            "l_uncert": l_uncert,
            "l_total": l_action + 0.1 * l_smooth + 0.5 * l_gripper + 0.1 * l_uncert,
        }
