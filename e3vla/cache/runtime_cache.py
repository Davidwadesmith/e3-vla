"""Runtime feature cache for speculative VLA inference.

Stores Action Expert features from the most recent full refresh round.
Provides O(1) read access during speculative rounds.
"""

from typing import Optional
import torch

from e3vla.schema import RuntimeCacheRecord, AEFeatureBundle


class RuntimeFeatureCache:
    """In-memory cache holding AE features from the last full refresh round.

    Lifecycle:
      - Created at episode start (reset)
      - Updated by FullRefreshPath after each full VLA forward
      - Read by SpeculativeCachedPath during each speculative round
      - Invalidated when cache_age exceeds max_cache_age or episode reset

    Thread-safe for single-episode use. Not designed for concurrent access.
    """

    def __init__(self, max_cache_age: int = 50):
        self._max_cache_age = max_cache_age
        self._record: Optional[RuntimeCacheRecord] = None

    @property
    def is_valid(self) -> bool:
        return (
            self._record is not None
            and self._record.cache_age < self._record.valid_until_step
            and self._record.cache_age < self._max_cache_age
        )

    @property
    def cache_age(self) -> int:
        return self._record.cache_age if self._record else 0

    @property
    def cached_chunk(self) -> Optional[torch.Tensor]:
        return self._record.full_action_chunk if self._record else None

    @property
    def cached_robot_state(self) -> Optional[torch.Tensor]:
        return self._record.full_robot_state if self._record else None

    @property
    def cached_ee_pose(self) -> Optional[torch.Tensor]:
        return self._record.full_ee_pose if self._record else None

    @property
    def kv_cache_ref(self):
        return self._record.kv_cache_ref if self._record else None

    def get_latest(self) -> Optional[RuntimeCacheRecord]:
        if not self.is_valid:
            return None
        return self._record

    def get_ae_features(self) -> Optional[AEFeatureBundle]:
        if not self.is_valid:
            return None
        return self._record.ae_features

    def update(self, record: RuntimeCacheRecord) -> None:
        self._record = record

    def increment_age(self, steps: int = 1) -> None:
        if self._record is not None:
            self._record.cache_age += steps

    def reset(self) -> None:
        self._record = None
