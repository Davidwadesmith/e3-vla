"""Tests for RuntimeFeatureCache."""

import torch
import pytest

from e3vla.schema import RuntimeCacheRecord, AEFeatureBundle
from e3vla.cache.runtime_cache import RuntimeFeatureCache


def make_dummy_record(cache_age=0):
    K, D = 16, 512
    return RuntimeCacheRecord(
        full_round_id=1,
        episode_id="test_ep",
        full_round_timestep=0,
        ae_features=AEFeatureBundle(
            ae_low=torch.randn(K, D),
            ae_mid=torch.randn(K, D),
            ae_high=torch.randn(K, D),
            ae_mixed=torch.randn(K, D),
        ),
        full_action_chunk=torch.randn(K, 7),
        full_robot_state=torch.randn(16),
        full_ee_pose=torch.randn(7),
        cache_age=cache_age,
        valid_until_step=50,
    )


class TestRuntimeCache:
    def test_update_and_get(self):
        cache = RuntimeFeatureCache(max_cache_age=50)
        record = make_dummy_record()
        cache.update(record)

        assert cache.is_valid
        retrieved = cache.get_latest()
        assert retrieved is not None
        assert retrieved.full_round_id == 1
        assert torch.equal(retrieved.ae_features.ae_mixed, record.ae_features.ae_mixed)

    def test_increment_age(self):
        cache = RuntimeFeatureCache(max_cache_age=50)
        cache.update(make_dummy_record())

        assert cache.cache_age == 0
        cache.increment_age(5)
        assert cache.cache_age == 5
        assert cache.is_valid

    def test_expires_at_max_age(self):
        cache = RuntimeFeatureCache(max_cache_age=10)
        cache.update(make_dummy_record())
        cache.increment_age(10)
        assert not cache.is_valid

    def test_reset(self):
        cache = RuntimeFeatureCache()
        cache.update(make_dummy_record())
        cache.reset()
        assert not cache.is_valid

    def test_cached_chunk_accessor(self):
        cache = RuntimeFeatureCache()
        record = make_dummy_record()
        cache.update(record)
        chunk = cache.cached_chunk
        assert chunk is not None
        assert chunk.shape == (16, 7)
