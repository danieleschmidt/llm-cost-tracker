"""Advanced caching system with multiple backends and intelligent eviction."""

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class CacheBackend(Enum):
    """Cache backend types."""

    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    ttl: Optional[float] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)

    def touch(self) -> None:
        """Update access metadata."""
        self.accessed_at = time.time()
        self.access_count += 1


class CacheBackendInterface(ABC):
    """Abstract interface for cache backends."""

    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key."""
        pass

    @abstractmethod
    async def set(self, key: str, entry: CacheEntry) -> None:
        """Set cache entry."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete cache entry."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class MemoryCacheBackend(CacheBackendInterface):
    """In-memory cache backend with LRU and LFU eviction."""

    def __init__(self, max_size: int = 1000, eviction_policy: str = "lru"):
        self.max_size = max_size
        self.eviction_policy = eviction_policy.lower()
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # For LRU
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key."""
        async with self._lock:
            if key not in self.cache:
                self.misses += 1
                return None

            entry = self.cache[key]

            # Check expiration
            if entry.is_expired():
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
                self.misses += 1
                return None

            # Update access metadata
            entry.touch()

            # Update LRU order
            if self.eviction_policy == "lru" and key in self.access_order:
                self.access_order.remove(key)
                self.access_order.append(key)

            self.hits += 1
            return entry

    async def set(self, key: str, entry: CacheEntry) -> None:
        """Set cache entry with eviction if needed."""
        async with self._lock:
            # If key exists, update
            if key in self.cache:
                self.cache[key] = entry
                if self.eviction_policy == "lru" and key in self.access_order:
                    self.access_order.remove(key)
                    self.access_order.append(key)
                return

            # Check if we need to evict
            if len(self.cache) >= self.max_size:
                await self._evict_entry()

            # Add new entry
            self.cache[key] = entry
            if self.eviction_policy == "lru":
                self.access_order.append(key)

    async def _evict_entry(self) -> None:
        """Evict an entry based on eviction policy."""
        if not self.cache:
            return

        if self.eviction_policy == "lru":
            # Remove least recently used
            if self.access_order:
                key_to_evict = self.access_order.pop(0)
                if key_to_evict in self.cache:
                    del self.cache[key_to_evict]
                    self.evictions += 1

        elif self.eviction_policy == "lfu":
            # Remove least frequently used
            min_access_count = float("inf")
            key_to_evict = None

            for key, entry in self.cache.items():
                if entry.access_count < min_access_count:
                    min_access_count = entry.access_count
                    key_to_evict = key

            if key_to_evict:
                del self.cache[key_to_evict]
                if key_to_evict in self.access_order:
                    self.access_order.remove(key_to_evict)
                self.evictions += 1

        elif self.eviction_policy == "fifo":
            # Remove first in, first out
            if self.cache:
                key_to_evict = next(iter(self.cache))
                del self.cache[key_to_evict]
                if key_to_evict in self.access_order:
                    self.access_order.remove(key_to_evict)
                self.evictions += 1

    async def delete(self, key: str) -> bool:
        """Delete cache entry."""
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
                return True
            return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self.cache.clear()
            self.access_order.clear()

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests) if total_requests > 0 else 0

        return {
            "backend": "memory",
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "eviction_policy": self.eviction_policy,
        }


class CacheManager:
    """Advanced cache manager with intelligent caching strategies."""

    def __init__(
        self,
        backend: CacheBackendInterface,
        default_ttl: Optional[float] = 3600,  # 1 hour
        key_prefix: str = "llm_cost_tracker",
    ):
        self.backend = backend
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self.cache_patterns: Dict[str, Dict[str, Any]] = {}

    def _make_key(self, key: str) -> str:
        """Create full cache key with prefix."""
        return f"{self.key_prefix}:{key}"

    def _hash_key(self, data: Any) -> str:
        """Create hash key from data."""
        if isinstance(data, dict):
            # Sort keys for consistent hashing
            sorted_data = json.dumps(data, sort_keys=True)
        else:
            sorted_data = str(data)

        return hashlib.sha256(sorted_data.encode()).hexdigest()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        full_key = self._make_key(key)
        entry = await self.backend.get(full_key)

        if entry is None:
            return None

        return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Set value in cache."""
        full_key = self._make_key(key)
        current_time = time.time()

        entry = CacheEntry(
            key=full_key,
            value=value,
            created_at=current_time,
            accessed_at=current_time,
            access_count=0,
            ttl=ttl or self.default_ttl,
            tags=tags or [],
        )

        await self.backend.set(full_key, entry)

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        full_key = self._make_key(key)
        return await self.backend.delete(full_key)

    async def clear(self) -> None:
        """Clear all cache."""
        await self.backend.clear()

    async def get_or_set(
        self,
        key: str,
        func: Callable,
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> Any:
        """Get from cache or execute function and cache result."""
        # Try to get from cache first
        cached_value = await self.get(key)
        if cached_value is not None:
            return cached_value

        # Execute function and cache result
        if asyncio.iscoroutinefunction(func):
            value = await func()
        else:
            value = func()

        await self.set(key, value, ttl, tags)
        return value

    async def mget(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple keys from cache."""
        results = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                results[key] = value
        return results

    async def mset(self, data: Dict[str, Any], ttl: Optional[float] = None) -> None:
        """Set multiple keys in cache."""
        for key, value in data.items():
            await self.set(key, value, ttl)

    def register_pattern(
        self,
        pattern_name: str,
        ttl: float,
        key_generator: Callable,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Register a caching pattern for specific use cases."""
        self.cache_patterns[pattern_name] = {
            "ttl": ttl,
            "key_generator": key_generator,
            "tags": tags or [],
        }

    async def get_with_pattern(
        self, pattern_name: str, func: Callable, *args, **kwargs
    ) -> Any:
        """Get value using registered caching pattern."""
        if pattern_name not in self.cache_patterns:
            raise ValueError(f"Unknown cache pattern: {pattern_name}")

        pattern = self.cache_patterns[pattern_name]
        key = pattern["key_generator"](*args, **kwargs)

        return await self.get_or_set(
            key,
            lambda: (
                func(*args, **kwargs)
                if not asyncio.iscoroutinefunction(func)
                else func(*args, **kwargs)
            ),
            ttl=pattern["ttl"],
            tags=pattern["tags"],
        )

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        backend_stats = await self.backend.get_stats()

        return {
            **backend_stats,
            "patterns": len(self.cache_patterns),
            "default_ttl": self.default_ttl,
            "key_prefix": self.key_prefix,
        }


# Pre-configured cache patterns for LLM cost tracking
def create_llm_cache_manager(max_size: int = 10000) -> CacheManager:
    """Create cache manager with LLM-specific patterns."""
    backend = MemoryCacheBackend(max_size=max_size, eviction_policy="lru")
    cache = CacheManager(backend, default_ttl=3600)

    # Register common patterns
    cache.register_pattern(
        "metrics_summary",
        ttl=300,  # 5 minutes
        key_generator=lambda hours, user_id=None: f"metrics:summary:{hours}:{user_id or 'all'}",
        tags=["metrics", "summary"],
    )

    cache.register_pattern(
        "cost_analysis",
        ttl=600,  # 10 minutes
        key_generator=lambda start_date, end_date, model=None: f"cost:analysis:{start_date}:{end_date}:{model or 'all'}",
        tags=["cost", "analysis"],
    )

    cache.register_pattern(
        "budget_status",
        ttl=60,  # 1 minute
        key_generator=lambda user_id: f"budget:status:{user_id}",
        tags=["budget"],
    )

    cache.register_pattern(
        "model_stats",
        ttl=1800,  # 30 minutes
        key_generator=lambda model_name, time_range: f"model:stats:{model_name}:{time_range}",
        tags=["model", "stats"],
    )

    return cache


# Global cache instance
llm_cache = create_llm_cache_manager()


def cache_result(
    key_pattern: str, ttl: float = 3600, key_args: Optional[List[str]] = None
):
    """Decorator for caching function results."""

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Generate cache key, excluding self for methods
            if key_args:
                key_data = {arg: kwargs.get(arg) for arg in key_args if arg in kwargs}
            else:
                # Skip first argument if it's 'self' (for methods)
                cache_args = (
                    args[1:] if args and hasattr(args[0], "__class__") else args
                )
                key_data = {"args": cache_args, "kwargs": kwargs}

            cache_key = f"{key_pattern}:{llm_cache._hash_key(key_data)}"

            return await llm_cache.get_or_set(
                cache_key, lambda: func(*args, **kwargs), ttl=ttl
            )

        def sync_wrapper(*args, **kwargs):
            # For sync functions, we'll need to handle differently
            return asyncio.run(async_wrapper(*args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
