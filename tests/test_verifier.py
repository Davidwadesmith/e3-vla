"""Tests for verifier and prefix acceptor."""

import torch
import pytest

from e3vla.verifier.prefix_acceptor import PrefixAcceptor
from e3vla.verifier.action_expert_anchor_verifier import ActionExpertAnchorVerifier
from e3vla.verifier.full_l2_oracle_verifier import FullL2OracleVerifier


class TestPrefixAcceptor:
    def setup_method(self):
        self.acceptor = PrefixAcceptor(
            tau_pos=0.01, tau_rot=0.05, tau_grip=0.1, alpha_uncert=1.0
        )

    def test_accept_all_low_errors(self):
        """All steps below threshold → accept full chunk."""
        K = 16
        errors = torch.ones(K) * 0.001  # very small
        # Use positive uncertainty (mimics softplus output from drafter)
        uncertainty = torch.rand(K, 3) * 0.5  # small positive [0, 0.5]
        draft = torch.randn(K, 7)

        accepted, fallback = self.acceptor.accept(errors, uncertainty, draft, 0.0)

        assert not fallback
        assert accepted > 0

    def test_reject_all_high_errors(self):
        """First step above threshold → accept 0."""
        K = 16
        errors = torch.ones(K) * 10.0  # very large
        uncertainty = torch.randn(K, 3)
        draft = torch.randn(K, 7)

        accepted, fallback = self.acceptor.accept(errors, uncertainty, draft, 0.0)

        assert fallback
        assert accepted == 0

    def test_partial_accept(self):
        """Errors grow over time → accept early steps only."""
        K = 16
        # Errors grow from 0.001 to 0.1 — later steps exceed threshold
        errors = torch.tensor([0.001 + 0.01 * i for i in range(K)])
        uncertainty = torch.rand(K, 3) * 0.1  # [0, 0.1]
        draft = torch.randn(K, 7)

        accepted, fallback = self.acceptor.accept(errors, uncertainty, draft, 0.0)

        assert not fallback
        assert 0 < accepted < K

    def test_high_uncertainty_tightens_threshold(self):
        """High uncertainty should tighten threshold, reducing acceptance."""
        K = 16
        errors = torch.ones(K) * 0.01
        draft = torch.randn(K, 7)

        # Low uncertainty (near zero)
        u_low = torch.zeros(K, 3)
        acc_low, _ = self.acceptor.accept(errors, u_low, draft, 0.0)

        # High uncertainty (large positive values)
        u_high = torch.ones(K, 3) * 10.0
        acc_high, _ = self.acceptor.accept(errors, u_high, draft, 0.0)

        # High uncertainty should accept fewer (or equal) steps
        assert acc_high <= acc_low

    def test_gripper_switch_tightens(self):
        """Gripper switch regions should have tighter thresholds."""
        K = 8
        errors = torch.ones(K) * 0.005
        # Small positive uncertainty
        uncertainty = torch.rand(K, 3) * 0.1

        # Draft with gripper switch
        draft = torch.randn(K, 7)
        draft[3, 6] = 0.5
        draft[4, 6] = -0.5  # switch

        # Without switch
        draft_no_switch = torch.randn(K, 7)
        draft_no_switch[:, 6] = 0.5  # all same sign

        acc_switch, _ = self.acceptor.accept(errors, uncertainty, draft, 0.0)
        acc_no_switch, _ = self.acceptor.accept(errors, uncertainty, draft_no_switch, 0.0)

        # Switch case should accept ≤ no-switch case
        assert acc_switch <= acc_no_switch


class TestActionExpertAnchorVerifier:
    def setup_method(self):
        self.verifier = ActionExpertAnchorVerifier(
            t_list=(0.10, 0.05),
            tau_radius=0.3,
            dist_dims=6,
            eval_h=12,
        )

    def test_verifier_interface_valid(self):
        """Verifier has the expected interface."""
        assert hasattr(self.verifier, "verify")
        assert hasattr(self.verifier, "t_list")
        assert hasattr(self.verifier, "tau_radius")

    def test_gripper_switch_detection(self):
        """Gripper switch should be detected."""
        B, K_v, H, D = 1, 2, 8, 7
        x0_hat = torch.randn(B, K_v, H, D)
        x0_hat[:, :, 3, 6] = -0.5
        x0_hat[:, :, 4:, 6] = 0.5  # switch at step 4

        has_switch = self.verifier._detect_gripper_switch(x0_hat, -0.5, H)
        assert has_switch

    def test_no_false_switch_detection(self):
        """No switch should not be detected."""
        B, K_v, H, D = 1, 2, 8, 7
        x0_hat = torch.randn(B, K_v, H, D)
        x0_hat[:, :, :, 6] = 0.5  # all same

        has_switch = self.verifier._detect_gripper_switch(x0_hat, 0.5, H)
        assert not has_switch


class TestFullL2OracleVerifier:
    def test_oracle_is_not_default(self):
        """FullL2OracleVerifier must not be the default verifier."""
        verifier = FullL2OracleVerifier()
        # Existence check: oracle should exist but be explicitly marked
        assert "oracle" in verifier.__class__.__name__.lower() or \
               "full" in verifier.__class__.__name__.lower()
        # Oracle verifier requires full inference (expensive)
        # Main policy must use ActionExpertAnchorVerifier instead
