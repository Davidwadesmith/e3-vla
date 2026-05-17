"""Gated alignment + cross-attention adapter for cached AE features.

Upgrades stale cached Action Expert features to current timestep
using temporal/pose offset embeddings, per-token gating, and
cross-attention re-indexing.
"""

import torch
import torch.nn as nn

from e3vla.schema import OffsetFeatures


class OffsetAlignAdapter(nn.Module):
    """Gated alignment + cross-attention re-indexing.

    Core mechanisms:
      1. Per-token gate: each cached AE token gets independent keep/discard weight
         based on its distance to cache_feature_cursor.
      2. Cross-attention: current K draft queries freely attend to gated cached AE tokens.
         This avoids forcing a one-to-one time alignment.
      3. Offset encoding: temporal/pose offsets embedded as attention bias and gate input.

    Args:
        hidden_dim: common feature dimension
        max_cache_age: maximum cache age for embedding table
        max_chunk_len: maximum action chunk length for positional embeddings
        num_heads: number of attention heads for cross-attention
        D_ee: end-effector pose dimension
        D_r: robot state dimension
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        max_cache_age: int = 50,
        max_chunk_len: int = 32,
        num_heads: int = 8,
        D_ee: int = 7,
        D_r: int = 16,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # --- Offset encoders ---
        self.cache_age_emb = nn.Embedding(max_cache_age, hidden_dim)
        self.gripper_phase_emb = nn.Embedding(3, hidden_dim)  # close/trans/open
        self.elapsed_steps_emb = nn.Embedding(max_cache_age, hidden_dim)
        self.delta_ee_proj = nn.Linear(D_ee, hidden_dim)
        self.delta_state_proj = nn.Linear(D_r, hidden_dim)

        # --- Offset fusion ---
        self.offset_fusion = nn.Sequential(
            nn.Linear(5 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # --- Per-token gating ---
        # Input: offset_tiled [B,K,D] concat with dist_to_cursor [B,K,1]
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.learned_null = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # --- Cross-attention: draft queries ← cached AE tokens ---
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.query_pos_emb = nn.Parameter(torch.randn(1, max_chunk_len, hidden_dim) * 0.02)
        self.key_pos_emb = nn.Parameter(torch.randn(1, max_chunk_len, hidden_dim) * 0.02)

        # Output normalization
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        cached_ae: torch.Tensor,      # [B, K, D]
        offset: OffsetFeatures,
        draft_queries: torch.Tensor,  # [B, K, D]
    ) -> torch.Tensor:
        """Align cached AE features to current timestep.

        Args:
            cached_ae: cached AE mixed features [B, K, D]
            offset: temporal/pose offset between cached and current
            draft_queries: learned action queries for current draft [B, K, D]

        Returns:
            aligned_ae: [B, K, D]
        """
        B, K, D = cached_ae.shape

        # 1. Encode offset information
        offset_emb = self._encode_offset(offset)  # [B, D]

        # 2. Per-token gating based on distance to cursor
        cursor = offset.cache_feature_cursor
        dist_to_cursor = torch.abs(
            torch.arange(K, device=cached_ae.device).float() - cursor
        ).unsqueeze(0).unsqueeze(-1)  # [1, K, 1]

        offset_tiled = offset_emb.unsqueeze(1).expand(-1, K, -1)  # [B, K, D]
        gate_input = torch.cat([offset_tiled, dist_to_cursor.expand(B, -1, -1)], dim=-1)
        gate = self.gate_proj(gate_input)  # [B, K, 1]

        gated_ae = gate * cached_ae + (1 - gate) * self.learned_null  # [B, K, D]

        # 3. Cross-attention: draft queries attend to gated cached tokens
        K_pos = min(K, self.query_pos_emb.shape[1])
        q = draft_queries[:, :K_pos, :] + self.query_pos_emb[:, :K_pos, :]
        k = gated_ae[:, :K_pos, :] + self.key_pos_emb[:, :K_pos, :]
        v = gated_ae[:, :K_pos, :]

        # Temporal decay bias: cached tokens far from cursor get lower attention
        attn_bias = self._compute_temporal_bias(
            K_pos, cursor, offset.elapsed_steps_since_full
        ).to(cached_ae.device)

        aligned_ae, _ = self.cross_attn(q, k, v, attn_mask=attn_bias)
        aligned_ae = self.out_norm(aligned_ae + draft_queries[:, :K_pos, :])

        return aligned_ae

    def _encode_offset(self, offset: OffsetFeatures) -> torch.Tensor:
        """Encode all offset fields into a single embedding vector [B, D]."""
        # Clamp values to embedding table range
        cache_age_idx = min(offset.cache_age, self.cache_age_emb.num_embeddings - 1)
        elapsed_idx = min(
            offset.elapsed_steps_since_full, self.elapsed_steps_emb.num_embeddings - 1
        )
        grip_idx = int(offset.gripper_phase + 1)  # -1→0, 0→1, 1→2

        emb_age = self.cache_age_emb(torch.tensor([cache_age_idx]))
        emb_grip = self.gripper_phase_emb(torch.tensor([grip_idx]))
        emb_elapsed = self.elapsed_steps_emb(torch.tensor([elapsed_idx]))

        delta_ee = offset.delta_ee_pose.unsqueeze(0) if offset.delta_ee_pose.dim() == 1 else offset.delta_ee_pose
        delta_state = offset.delta_robot_state.unsqueeze(0) if offset.delta_robot_state.dim() == 1 else offset.delta_robot_state

        emb_ee = self.delta_ee_proj(delta_ee.float())
        emb_state = self.delta_state_proj(delta_state.float())

        return self.offset_fusion(
            torch.cat([emb_age, emb_grip, emb_elapsed, emb_ee, emb_state], dim=-1)
        )

    @staticmethod
    def _compute_temporal_bias(
        K: int, cursor: int, elapsed: int, decay: float = 0.1
    ) -> torch.Tensor:
        """Compute temporal decay attention bias.

        Cached tokens farther from cursor get lower attention weight
        from all draft queries.
        """
        bias = torch.zeros(K, K)
        for j in range(K):
            dist = abs(j - (cursor % K))
            bias[:, j] = -decay * dist
        return bias
