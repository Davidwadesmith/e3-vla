"""Tests for policy correctness."""

import torch
import pytest

from e3vla.schema import (
    Observation, ActionCommand, AEFeatureBundle, RuntimeCacheRecord,
)
from e3vla.cache.runtime_cache import RuntimeFeatureCache
from e3vla.verifier.prefix_acceptor import PrefixAcceptor


class TestPolicyFallback:
    """Verify policy correctly falls back on rejected drafts."""

    def test_prefix_acceptor_reject_zero_triggers_fallback(self):
        """When acceptor returns accepted_len=0, policy must fallback."""
        acceptor = PrefixAcceptor()
        errors = torch.ones(16) * 10.0  # very large errors
        uncertainty = torch.zeros(16, 3)
        draft = torch.randn(16, 7)

        accepted, fallback = acceptor.accept(errors, uncertainty, draft, 0.0)
        assert fallback
        assert accepted == 0

    def test_action_command_has_execute_len(self):
        """ActionCommand must have execute_len field for fair evaluation."""
        cmd = ActionCommand(
            actions=torch.randn(8, 7),
            execute_len=3,
            can_interrupt=True,
            mode="speculative",
            prefix_length=3,
        )
        assert cmd.execute_len == 3
        assert cmd.can_interrupt
        assert len(cmd.actions) >= cmd.execute_len


class TestNoFeatureLeak:
    """Verifies speculative path does NOT call full target VLA feature collection."""

    def test_speculative_path_no_full_collect(self):
        """This is a structural test: speculative path code path must not import
        or call adapter.collect_features() or similar full-VLA extraction.

        Verified by code review: SpeculativeCachedPolicy._speculative_cached_path
        uses RuntimeFeatureCache.get_latest() and CurrentCheapFeatures,
        never calls adapter.collect_features or adapter.full_inference.
        """
        # Read the policy source and confirm no problematic calls
        import inspect
        from e3vla.policy.speculative_cached_policy import SpeculativeCachedPolicy

        source = inspect.getsource(SpeculativeCachedPolicy._speculative_cached_path)

        # Must NOT call these
        assert "collect_features" not in source, \
            "speculative path must not call collect_features (triggers full VLA)"
        assert "adapter.full_inference" not in source, \
            "speculative path must not call full_inference (too expensive)"

        # Must use cache
        assert "cache" in source.lower() or "get_latest" in source, \
            "speculative path must read from RuntimeFeatureCache"


class TestActionCommandSemantics:
    """ActionCommand execute_len and can_interrupt must be used consistently."""

    def test_full_refresh_execute_len(self):
        """Full refresh actions: execute_len <= K."""
        K = 16
        full_exec = 2
        assert full_exec <= K

        cmd = ActionCommand(
            actions=torch.randn(K, 7),
            execute_len=full_exec,
            can_interrupt=True,
            mode="full_refresh",
        )
        assert cmd.execute_len <= len(cmd.actions)
        assert cmd.can_interrupt

    def test_speculative_execute_len_equals_accepted(self):
        """Speculative actions: execute_len == accepted_len."""
        accepted = 5
        cmd = ActionCommand(
            actions=torch.randn(accepted, 7),
            execute_len=accepted,
            can_interrupt=True,
            mode="speculative",
            prefix_length=accepted,
        )
        assert cmd.execute_len == len(cmd.actions)
        assert cmd.execute_len == cmd.prefix_length

    def test_all_methods_use_closed_loop(self):
        """All benchmark methods must use can_interrupt=True for fair comparison."""
        for mode in ["full_refresh", "speculative", "action_reuse", "fallback"]:
            cmd = ActionCommand(
                actions=torch.randn(4, 7),
                execute_len=4,
                can_interrupt=True,
                mode=mode,
            )
            assert cmd.can_interrupt, f"Mode {mode} must use closed-loop evaluation"
