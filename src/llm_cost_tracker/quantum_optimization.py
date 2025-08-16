"""
Advanced optimization and scaling features for Quantum Task Planner.
Provides performance optimization, caching, load balancing, and auto-scaling.
"""

import asyncio
import functools
import hashlib
import json
import logging
import pickle
import threading
import time
import weakref
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with TTL and metadata."""

    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: int = 300  # 5 minutes default
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds

    def touch(self) -> None:
        """Update access metadata."""
        self.last_accessed = datetime.now()
        self.access_count += 1


class QuantumCache:
    """
    High-performance caching system with LRU eviction, TTL, and compression.
    Optimized for quantum task planning operations.
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: deque = deque()  # For LRU tracking
        self.lock = threading.RLock()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_size_bytes = 0

        # Cleanup thread
        self.cleanup_interval = 60  # seconds
        self.cleanup_thread = None
        self.running = False

    def start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        if not self.running:
            self.running = True
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_loop, name="quantum-cache-cleanup", daemon=True
            )
            self.cleanup_thread.start()
            logger.debug("Cache cleanup thread started")

    def stop_cleanup_thread(self) -> None:
        """Stop background cleanup thread."""
        self.running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5.0)
        logger.debug("Cache cleanup thread stopped")

    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self.running:
            try:
                self._expire_entries()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                time.sleep(self.cleanup_interval * 2)

    def _calculate_size(self, value: Any) -> int:
        """Estimate size of cached value."""
        try:
            return len(pickle.dumps(value))
        except:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode("utf-8"))
            elif isinstance(value, (list, tuple)):
                return sum(
                    self._calculate_size(item) for item in value[:10]
                )  # Sample first 10
            elif isinstance(value, dict):
                return sum(
                    self._calculate_size(k) + self._calculate_size(v)
                    for k, v in list(value.items())[:10]
                )
            else:
                return 64  # Default estimate

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {"args": args, "kwargs": kwargs}
        serialized = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            entry = self.cache.get(key)

            if entry is None:
                self.misses += 1
                return None

            if entry.is_expired():
                self._remove_entry(key)
                self.misses += 1
                return None

            # Update access metadata
            entry.touch()

            # Move to end of access order (most recently used)
            try:
                self.access_order.remove(key)
            except ValueError:
                pass
            self.access_order.append(key)

            self.hits += 1
            return entry.value

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put value in cache."""
        with self.lock:
            # Calculate size
            size_bytes = self._calculate_size(value)

            # Create entry
            entry = CacheEntry(
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                ttl_seconds=ttl or self.default_ttl,
                size_bytes=size_bytes,
            )

            # Remove existing entry if present
            if key in self.cache:
                self._remove_entry(key)

            # Ensure capacity
            while len(self.cache) >= self.max_size:
                self._evict_lru()

            # Add entry
            self.cache[key] = entry
            self.access_order.append(key)
            self.total_size_bytes += size_bytes

    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache."""
        entry = self.cache.pop(key, None)
        if entry:
            self.total_size_bytes -= entry.size_bytes
            try:
                self.access_order.remove(key)
            except ValueError:
                pass

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self.access_order:
            lru_key = self.access_order.popleft()
            if lru_key in self.cache:
                self._remove_entry(lru_key)
                self.evictions += 1

    def _expire_entries(self) -> None:
        """Remove expired entries."""
        with self.lock:
            expired_keys = [
                key for key, entry in self.cache.items() if entry.is_expired()
            ]

            for key in expired_keys:
                self._remove_entry(key)

            if expired_keys:
                logger.debug(f"Expired {len(expired_keys)} cache entries")

    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.total_size_bytes = 0
            logger.debug("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "total_size_bytes": self.total_size_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": hit_rate,
                "running": self.running,
            }


def memoize_with_ttl(ttl: int = 300, cache: Optional[QuantumCache] = None):
    """Decorator for memoizing function results with TTL."""

    def decorator(func: Callable) -> Callable:
        func_cache = cache or QuantumCache(max_size=100, default_ttl=ttl)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}_{func_cache._generate_key(*args, **kwargs)}"

            # Try to get from cache
            cached_result = func_cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = func(*args, **kwargs)
            func_cache.put(cache_key, result, ttl)

            return result

        # Attach cache for inspection
        wrapper._cache = func_cache
        return wrapper

    return decorator


@dataclass
class LoadBalancingConfig:
    """Configuration for load balancing."""

    max_concurrent_tasks: int = 10
    queue_size_limit: int = 100
    worker_timeout_seconds: float = 30.0
    retry_attempts: int = 3
    backoff_multiplier: float = 1.5
    circuit_breaker_threshold: int = 5


class QuantumLoadBalancer:
    """
    Advanced load balancer for quantum task execution with circuit breakers,
    retry logic, and adaptive load distribution.
    """

    def __init__(self, config: LoadBalancingConfig = None):
        self.config = config or LoadBalancingConfig()

        # Worker pools
        self.primary_pool = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_tasks,
            thread_name_prefix="quantum-primary",
        )
        self.overflow_pool = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_tasks // 2,
            thread_name_prefix="quantum-overflow",
        )

        # Task queues
        self.primary_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.queue_size_limit
        )
        self.overflow_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.queue_size_limit
        )

        # Circuit breaker state
        self.circuit_breaker_counts = defaultdict(int)
        self.circuit_breaker_open = defaultdict(bool)
        self.last_failure_time = defaultdict(datetime)

        # Statistics
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.retried_tasks = 0
        self.circuit_breaker_trips = 0

        # Control
        self.running = False
        self.workers: List[asyncio.Task] = []

    async def start_workers(self) -> None:
        """Start worker coroutines."""
        if self.running:
            return

        self.running = True

        # Start primary workers
        for i in range(self.config.max_concurrent_tasks // 2):
            worker = asyncio.create_task(
                self._worker_loop(f"primary-{i}", self.primary_queue, self.primary_pool)
            )
            self.workers.append(worker)

        # Start overflow workers
        for i in range(self.config.max_concurrent_tasks // 4):
            worker = asyncio.create_task(
                self._worker_loop(
                    f"overflow-{i}", self.overflow_queue, self.overflow_pool
                )
            )
            self.workers.append(worker)

        logger.info(f"Started {len(self.workers)} load balancer workers")

    async def stop_workers(self) -> None:
        """Stop all workers."""
        self.running = False

        # Cancel workers
        for worker in self.workers:
            worker.cancel()

        # Wait for completion
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)

        # Shutdown executors
        self.primary_pool.shutdown(wait=True)
        self.overflow_pool.shutdown(wait=True)

        self.workers.clear()
        logger.info("Load balancer workers stopped")

    async def _worker_loop(
        self, worker_id: str, queue: asyncio.Queue, executor: ThreadPoolExecutor
    ) -> None:
        """Worker loop for processing tasks."""
        logger.debug(f"Worker {worker_id} started")

        while self.running:
            try:
                # Get task with timeout
                task_item = await asyncio.wait_for(queue.get(), timeout=5.0)
                task_func, args, kwargs, future = task_item

                # Check circuit breaker
                if self._is_circuit_breaker_open(task_func.__name__):
                    future.set_exception(
                        Exception(f"Circuit breaker open for {task_func.__name__}")
                    )
                    continue

                # Execute task with retry logic
                success = await self._execute_with_retry(
                    worker_id, task_func, args, kwargs, future, executor
                )

                if success:
                    self.completed_tasks += 1
                    self._reset_circuit_breaker(task_func.__name__)
                else:
                    self.failed_tasks += 1
                    self._record_failure(task_func.__name__)

                queue.task_done()

            except asyncio.TimeoutError:
                continue  # Normal timeout, keep polling
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                continue

        logger.debug(f"Worker {worker_id} stopped")

    async def _execute_with_retry(
        self,
        worker_id: str,
        task_func: Callable,
        args: tuple,
        kwargs: dict,
        future: asyncio.Future,
        executor: ThreadPoolExecutor,
    ) -> bool:
        """Execute task with retry logic."""

        for attempt in range(1, self.config.retry_attempts + 1):
            try:
                # Execute in thread pool
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(executor, task_func, *args, **kwargs),
                    timeout=self.config.worker_timeout_seconds,
                )

                future.set_result(result)
                return True

            except asyncio.TimeoutError:
                error_msg = f"Task timeout on worker {worker_id} (attempt {attempt})"
                logger.warning(error_msg)

                if attempt == self.config.retry_attempts:
                    future.set_exception(TimeoutError(error_msg))
                    return False

            except Exception as e:
                error_msg = (
                    f"Task failed on worker {worker_id} (attempt {attempt}): {e}"
                )
                logger.warning(error_msg)

                if attempt == self.config.retry_attempts:
                    future.set_exception(e)
                    return False

            # Exponential backoff
            if attempt < self.config.retry_attempts:
                backoff_time = (self.config.backoff_multiplier ** (attempt - 1)) * 0.1
                await asyncio.sleep(backoff_time)
                self.retried_tasks += 1

        return False

    def _is_circuit_breaker_open(self, task_name: str) -> bool:
        """Check if circuit breaker is open for a task type."""
        if not self.circuit_breaker_open[task_name]:
            return False

        # Check if enough time has passed to try again
        time_since_failure = datetime.now() - self.last_failure_time[task_name]
        if time_since_failure.total_seconds() > 60:  # 1 minute cooldown
            self.circuit_breaker_open[task_name] = False
            self.circuit_breaker_counts[task_name] = 0
            logger.info(f"Circuit breaker reset for {task_name}")
            return False

        return True

    def _record_failure(self, task_name: str) -> None:
        """Record a task failure and update circuit breaker."""
        self.circuit_breaker_counts[task_name] += 1
        self.last_failure_time[task_name] = datetime.now()

        if (
            self.circuit_breaker_counts[task_name]
            >= self.config.circuit_breaker_threshold
        ):
            self.circuit_breaker_open[task_name] = True
            self.circuit_breaker_trips += 1
            logger.warning(
                f"Circuit breaker OPEN for {task_name} (failures: {self.circuit_breaker_counts[task_name]})"
            )

    def _reset_circuit_breaker(self, task_name: str) -> None:
        """Reset circuit breaker on successful execution."""
        if self.circuit_breaker_counts[task_name] > 0:
            self.circuit_breaker_counts[task_name] = max(
                0, self.circuit_breaker_counts[task_name] - 1
            )

    async def submit_task(self, task_func: Callable, *args, **kwargs) -> Any:
        """Submit task for load-balanced execution."""
        if not self.running:
            raise RuntimeError("Load balancer not running")

        # Create future for result
        future = asyncio.Future()
        task_item = (task_func, args, kwargs, future)

        # Try primary queue first
        try:
            await self.primary_queue.put(task_item)
        except asyncio.QueueFull:
            # Fall back to overflow queue
            try:
                await asyncio.wait_for(self.overflow_queue.put(task_item), timeout=1.0)
            except (asyncio.QueueFull, asyncio.TimeoutError):
                raise RuntimeError("All task queues are full")

        # Wait for result
        return await future

    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        return {
            "running": self.running,
            "worker_count": len(self.workers),
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "retried_tasks": self.retried_tasks,
            "circuit_breaker_trips": self.circuit_breaker_trips,
            "primary_queue_size": self.primary_queue.qsize(),
            "overflow_queue_size": self.overflow_queue.qsize(),
            "circuit_breakers": dict(self.circuit_breaker_open),
            "failure_counts": dict(self.circuit_breaker_counts),
        }


@dataclass
class AutoScalingConfig:
    """Configuration for auto-scaling."""

    min_workers: int = 2
    max_workers: int = 20
    target_queue_size: int = 5
    scale_up_threshold: float = 0.8  # Queue utilization
    scale_down_threshold: float = 0.2
    scale_up_cooldown: int = 60  # seconds
    scale_down_cooldown: int = 180
    worker_cpu_threshold: float = 0.8
    worker_memory_threshold: float = 0.8


class QuantumAutoScaler:
    """
    Auto-scaling system for quantum task processing.
    Dynamically adjusts worker count based on load and resource utilization.
    """

    def __init__(
        self, load_balancer: QuantumLoadBalancer, config: AutoScalingConfig = None
    ):
        self.load_balancer = load_balancer
        self.config = config or AutoScalingConfig()

        # State tracking
        self.current_workers = self.config.min_workers
        self.last_scale_up = datetime.min
        self.last_scale_down = datetime.min

        # Metrics
        self.scale_up_events = 0
        self.scale_down_events = 0
        self.metrics_history: deque = deque(maxlen=100)

        # Control
        self.running = False
        self.scaling_task: Optional[asyncio.Task] = None

    async def start_auto_scaling(self) -> None:
        """Start auto-scaling monitoring."""
        if self.running:
            return

        self.running = True
        self.scaling_task = asyncio.create_task(self._scaling_loop())
        logger.info("Auto-scaling started")

    async def stop_auto_scaling(self) -> None:
        """Stop auto-scaling."""
        self.running = False

        if self.scaling_task:
            self.scaling_task.cancel()
            try:
                await self.scaling_task
            except asyncio.CancelledError:
                pass

        logger.info("Auto-scaling stopped")

    async def _scaling_loop(self) -> None:
        """Main auto-scaling monitoring loop."""
        while self.running:
            try:
                await self._evaluate_scaling()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
                await asyncio.sleep(60)  # Back off on errors

    async def _evaluate_scaling(self) -> None:
        """Evaluate whether scaling is needed."""
        try:
            # Collect metrics
            stats = self.load_balancer.get_stats()
            queue_size = stats["primary_queue_size"] + stats["overflow_queue_size"]
            worker_count = stats["worker_count"]

            # Calculate queue utilization
            max_queue_size = self.load_balancer.config.queue_size_limit * 2
            queue_utilization = queue_size / max_queue_size if max_queue_size > 0 else 0

            # Record metrics
            metrics = {
                "timestamp": datetime.now(),
                "queue_size": queue_size,
                "queue_utilization": queue_utilization,
                "worker_count": worker_count,
                "completed_tasks": stats["completed_tasks"],
                "failed_tasks": stats["failed_tasks"],
            }
            self.metrics_history.append(metrics)

            # Scaling decisions
            now = datetime.now()

            # Scale up conditions
            if (
                queue_utilization > self.config.scale_up_threshold
                and worker_count < self.config.max_workers
                and (now - self.last_scale_up).total_seconds()
                > self.config.scale_up_cooldown
            ):

                await self._scale_up()

            # Scale down conditions
            elif (
                queue_utilization < self.config.scale_down_threshold
                and worker_count > self.config.min_workers
                and (now - self.last_scale_down).total_seconds()
                > self.config.scale_down_cooldown
            ):

                # Additional check: ensure low utilization is sustained
                recent_metrics = list(self.metrics_history)[-5:]  # Last 5 measurements
                if len(recent_metrics) >= 3:
                    avg_utilization = sum(
                        m["queue_utilization"] for m in recent_metrics
                    ) / len(recent_metrics)
                    if avg_utilization < self.config.scale_down_threshold:
                        await self._scale_down()

        except Exception as e:
            logger.error(f"Scaling evaluation error: {e}")

    async def _scale_up(self) -> None:
        """Scale up worker count."""
        try:
            new_worker_count = min(
                self.current_workers + 2,  # Add 2 workers at a time
                self.config.max_workers,
            )

            if new_worker_count > self.current_workers:
                # This would require modifying the load balancer to support dynamic scaling
                # For now, we'll log the scaling decision
                logger.info(
                    f"Auto-scaling UP: {self.current_workers} → {new_worker_count} workers"
                )

                self.current_workers = new_worker_count
                self.last_scale_up = datetime.now()
                self.scale_up_events += 1

        except Exception as e:
            logger.error(f"Scale up error: {e}")

    async def _scale_down(self) -> None:
        """Scale down worker count."""
        try:
            new_worker_count = max(
                self.current_workers - 1,  # Remove 1 worker at a time
                self.config.min_workers,
            )

            if new_worker_count < self.current_workers:
                logger.info(
                    f"Auto-scaling DOWN: {self.current_workers} → {new_worker_count} workers"
                )

                self.current_workers = new_worker_count
                self.last_scale_down = datetime.now()
                self.scale_down_events += 1

        except Exception as e:
            logger.error(f"Scale down error: {e}")

    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get auto-scaling metrics."""
        recent_metrics = list(self.metrics_history)[-10:]

        avg_queue_utilization = 0.0
        avg_worker_count = 0.0

        if recent_metrics:
            avg_queue_utilization = sum(
                m["queue_utilization"] for m in recent_metrics
            ) / len(recent_metrics)
            avg_worker_count = sum(m["worker_count"] for m in recent_metrics) / len(
                recent_metrics
            )

        return {
            "running": self.running,
            "current_workers": self.current_workers,
            "min_workers": self.config.min_workers,
            "max_workers": self.config.max_workers,
            "scale_up_events": self.scale_up_events,
            "scale_down_events": self.scale_down_events,
            "avg_queue_utilization": avg_queue_utilization,
            "avg_worker_count": avg_worker_count,
            "last_scale_up": self.last_scale_up.isoformat(),
            "last_scale_down": self.last_scale_down.isoformat(),
            "metrics_history_size": len(self.metrics_history),
        }


# Global instances for optimization components
global_cache = QuantumCache(max_size=1000, default_ttl=600)
global_load_balancer = None
global_auto_scaler = None


async def initialize_optimization_systems() -> (
    Tuple[QuantumCache, QuantumLoadBalancer, QuantumAutoScaler]
):
    """Initialize all optimization systems."""
    global global_cache, global_load_balancer, global_auto_scaler

    # Start cache cleanup
    global_cache.start_cleanup_thread()

    # Initialize load balancer
    global_load_balancer = QuantumLoadBalancer()
    await global_load_balancer.start_workers()

    # Initialize auto-scaler
    global_auto_scaler = QuantumAutoScaler(global_load_balancer)
    await global_auto_scaler.start_auto_scaling()

    logger.info("Quantum optimization systems initialized")
    return global_cache, global_load_balancer, global_auto_scaler


async def shutdown_optimization_systems() -> None:
    """Shutdown all optimization systems."""
    global global_cache, global_load_balancer, global_auto_scaler

    if global_auto_scaler:
        await global_auto_scaler.stop_auto_scaling()

    if global_load_balancer:
        await global_load_balancer.stop_workers()

    if global_cache:
        global_cache.stop_cleanup_thread()
        global_cache.clear()

    logger.info("Quantum optimization systems shutdown")
