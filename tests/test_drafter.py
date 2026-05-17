"""Tests for CachedAEDrafter shape and forward correctness."""

import torch
import pytest

from e3vla.schema import AEFeatureBundle, CurrentCheapFeatures
from e3vla.drafter.cached_ae_drafter import CachedAEDrafter


def make_dummy_ae_bundle(B=2, K=16, D=512):
    return AEFeatureBundle(
        ae_low=torch.randn(K, D),
        ae_mid=torch.randn(K, D),
        ae_high=torch.randn(K, D),
        ae_mixed=torch.randn(K, D),
    )


def make_dummy_cheap_features(B=2, D_r=16):
    return CurrentCheapFeatures(
        current_robot_state=torch.randn(B, D_r),
        action_history=torch.randn(B, 5, 7),
        last_executed_action_index=0,
        gripper_state=0.0,
    )


class TestCachedAEDrafterShapes:
    def setup_method(self):
        self.drafter = CachedAEDrafter(
            hidden_dim=256,
            action_dim=7,
            chunk_len=16,
            num_layers=2,
            num_heads=4,
            D_r=16,
            D_ee=7,
            action_history_len=5,
        )

    def test_output_shape_with_cached_ae(self):
        ae = make_dummy_ae_bundle(B=2, K=16, D=256)
        cheap = make_dummy_cheap_features(B=2)

        output = self.drafter(ae, None, cheap)

        assert output.action_chunk.shape == (2, 16, 7)
        assert output.uncertainty.shape == (2, 16, 3)
        assert output.hidden_states.shape == (2, 16, 256)

    def test_output_shape_without_cached_ae(self):
        """NoCachedAE ablation: drafter should still work."""
        cheap = make_dummy_cheap_features(B=2)

        output = self.drafter(None, None, cheap)

        assert output.action_chunk.shape == (2, 16, 7)
        assert output.uncertainty.shape == (2, 16, 3)

    def test_output_shape_batch_1(self):
        ae = make_dummy_ae_bundle(B=1, K=16, D=256)
        cheap = make_dummy_cheap_features(B=1)

        output = self.drafter(ae, None, cheap)

        assert output.action_chunk.shape == (1, 16, 7)

    def test_uncertainty_positive(self):
        """Uncertainty must always be positive (softplus output)."""
        ae = make_dummy_ae_bundle(B=2, K=16, D=256)
        cheap = make_dummy_cheap_features(B=2)

        output = self.drafter(ae, None, cheap)

        assert (output.uncertainty > 0).all()

    def test_no_nan_output(self):
        ae = make_dummy_ae_bundle(B=2, K=16, D=256)
        cheap = make_dummy_cheap_features(B=2)

        output = self.drafter(ae, None, cheap)

        assert not torch.isnan(output.action_chunk).any()
        assert not torch.isinf(output.action_chunk).any()

    def test_compute_loss_returns_dict(self):
        """Test loss computation returns expected keys."""
        B, K, D = 2, 16, 256
        # Build minimal batch
        batch = {
            "cached_ae_low": torch.randn(B, K, D),
            "cached_ae_mid": torch.randn(B, K, D),
            "cached_ae_high": torch.randn(B, K, D),
            "cached_ae_mixed": torch.randn(B, K, D),
            "current_robot_state": torch.randn(B, 16),
            "action_history": torch.randn(B, 5, 7),
            "delta_action_index": 0,
            "gripper_phase": torch.tensor([0.0, 0.0]),
            "target_action_chunk": torch.randn(B, K, 7),
        }

        losses = self.drafter.compute_loss(batch)

        assert "l_action" in losses
        assert "l_smooth" in losses
        assert "l_gripper" in losses
        assert "l_uncert" in losses
        assert "l_total" in losses
        assert losses["l_total"].requires_grad
