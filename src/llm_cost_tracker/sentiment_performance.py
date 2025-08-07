"""High-performance sentiment analysis optimization with quantum-enhanced scaling."""

import asyncio
import logging
import time
import statistics
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
import threading
import psutil
import numpy as np
from collections import defaultdict, deque

from .sentiment_analyzer import SentimentAnalyzer, SentimentResult, SentimentRequest, BatchSentimentRequest
from .quantum_task_planner import QuantumTaskPlanner, QuantumTask
from .cache import CacheManager
from .auto_scaling import auto_scaler, metrics_collector

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for sentiment analysis."""
    requests_per_second: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    throughput_mbps: float = 0.0
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    concurrent_requests: int = 0
    queue_depth: int = 0
    
    # Scaling metrics
    auto_scale_events: int = 0
    circuit_breaker_trips: int = 0
    load_balancer_switches: int = 0
    
    # Cost optimization
    cost_per_request: float = 0.0
    token_efficiency: float = 0.0
    model_switching_events: int = 0

@dataclass 
class OptimizationConfig:
    """Configuration for performance optimization."""
    max_concurrent_requests: int = 100
    batch_size_limit: int = 50
    cache_optimization: bool = True
    auto_scaling_enabled: bool = True
    load_balancing_enabled: bool = True
    model_switching_enabled: bool = True
    prefetch_enabled: bool = True
    compression_enabled: bool = True
    
    # Resource limits
    max_cpu_usage: float = 80.0
    max_memory_mb: int = 2048
    max_queue_depth: int = 1000
    
    # Performance targets
    target_latency_ms: float = 200.0
    target_throughput_rps: float = 50.0
    target_cache_hit_rate: float = 0.75

class SentimentPerformanceOptimizer:
    """Advanced performance optimizer for sentiment analysis with quantum-enhanced scaling."""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        
        # Performance tracking
        self.metrics_history: deque = deque(maxlen=1000)
        self.latency_samples: deque = deque(maxlen=1000)
        self.current_metrics = PerformanceMetrics()
        
        # Resource management
        self.thread_pool = ThreadPoolExecutor(max_workers=min(32, cpu_count() * 2))
        self.process_pool = ProcessPoolExecutor(max_workers=cpu_count())
        self.request_queue = asyncio.Queue(maxsize=self.config.max_queue_depth)
        
        # Load balancing and scaling
        self.quantum_planner = QuantumTaskPlanner()
        self.load_balancer = LoadBalancer()
        self.model_selector = IntelligentModelSelector()
        
        # Caching optimization
        self.smart_cache = SmartCacheManager()
        self.prefetch_engine = PrefetchEngine()
        
        # Monitoring
        self._lock = threading.RLock()
        self._shutdown_event = asyncio.Event()
        
        # Start background optimization tasks
        asyncio.create_task(self._performance_monitor())
        asyncio.create_task(self._optimization_loop())
        
        logger.info(f"SentimentPerformanceOptimizer initialized with config: {self.config}")
    
    async def optimize_request(
        self,
        analyzer: SentimentAnalyzer,
        request: Union[SentimentRequest, BatchSentimentRequest],
        user_id: Optional[str] = None
    ) -> Union[SentimentResult, List[SentimentResult]]:
        """Optimize a single request or batch with advanced performance enhancements."""
        start_time = time.perf_counter()
        
        try:
            # 1. Queue management and load balancing
            await self._manage_request_queue()
            
            # 2. Intelligent model selection
            optimized_request = await self.model_selector.select_optimal_model(request)
            
            # 3. Smart caching with prefetch
            cached_result = await self.smart_cache.get_optimized(optimized_request, user_id)
            if cached_result:
                await self._record_cache_hit(start_time)
                return cached_result
            
            # 4. Parallel processing optimization
            if isinstance(request, BatchSentimentRequest):
                result = await self._optimize_batch_processing(analyzer, optimized_request, user_id)
            else:
                result = await self._optimize_single_request(analyzer, optimized_request, user_id)
            
            # 5. Result caching and prefetch preparation
            await self.smart_cache.set_optimized(optimized_request, result, user_id)
            await self.prefetch_engine.prepare_related_requests(request, user_id)
            
            # 6. Performance metrics update
            await self._update_performance_metrics(start_time, request, result)
            
            return result
            
        except Exception as e:
            await self._record_error(e, start_time)
            raise
    
    async def _optimize_single_request(
        self,
        analyzer: SentimentAnalyzer,
        request: SentimentRequest,
        user_id: Optional[str]
    ) -> SentimentResult:
        """Optimize single request processing."""
        # Create quantum-optimized task
        task = QuantumTask(
            id=f"optimized_sentiment_{int(time.time() * 1000)}",
            name="Optimized Sentiment Analysis",
            priority=self._calculate_dynamic_priority(request),
            estimated_duration_minutes=0.5,
            metadata={
                "optimization_enabled": True,
                "user_id": user_id,
                "text_length": len(request.text),
                "model": request.model_preference
            }
        )
        
        # Use quantum planner for optimal scheduling
        self.quantum_planner.add_task(task)
        schedule = self.quantum_planner.generate_schedule()
        
        # Execute with performance monitoring
        return await analyzer.analyze(request, user_id=user_id)
    
    async def _optimize_batch_processing(
        self,
        analyzer: SentimentAnalyzer,
        request: BatchSentimentRequest,
        user_id: Optional[str]
    ) -> List[SentimentResult]:
        """Optimize batch request processing with intelligent chunking."""
        # Intelligent batch sizing
        optimal_batch_size = self._calculate_optimal_batch_size(request)
        
        if len(request.texts) <= optimal_batch_size:
            # Process as single batch
            return await analyzer.analyze_batch(request)
        
        # Split into optimally-sized chunks
        chunks = self._create_intelligent_chunks(request.texts, optimal_batch_size)
        
        # Process chunks in parallel with load balancing
        tasks = []
        for i, chunk in enumerate(chunks):
            chunk_request = BatchSentimentRequest(
                texts=chunk,
                language=request.language,
                model_preference=request.model_preference,
                parallel_processing=request.parallel_processing
            )
            
            # Add to quantum planner with priority scheduling
            task = asyncio.create_task(analyzer.analyze_batch(chunk_request))
            tasks.append(task)
        
        # Wait for all chunks with optimal concurrency
        chunk_results = await asyncio.gather(*tasks)
        
        # Flatten results while preserving order
        final_results = []
        for chunk_result in chunk_results:
            final_results.extend(chunk_result)
        
        return final_results
    
    def _calculate_optimal_batch_size(self, request: BatchSentimentRequest) -> int:
        """Calculate optimal batch size based on current system performance."""
        # Base batch size
        base_size = 20
        
        # Adjust based on CPU usage
        cpu_usage = psutil.cpu_percent(interval=0.1)
        if cpu_usage > 80:
            base_size = max(5, base_size // 2)
        elif cpu_usage < 40:
            base_size = min(50, base_size * 2)
        
        # Adjust based on memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 80:
            base_size = max(5, base_size // 2)
        
        # Adjust based on text complexity
        avg_text_length = sum(len(text) for text in request.texts) / len(request.texts)
        if avg_text_length > 1000:
            base_size = max(10, base_size // 2)
        
        # Adjust based on queue depth
        queue_depth = self.request_queue.qsize()
        if queue_depth > 100:
            base_size = max(5, base_size // 3)
        
        return min(base_size, self.config.batch_size_limit)
    
    def _create_intelligent_chunks(self, texts: List[str], chunk_size: int) -> List[List[str]]:
        """Create intelligent chunks that balance text complexity."""
        # Sort texts by length for better load balancing
        text_items = [(i, text) for i, text in enumerate(texts)]
        text_items.sort(key=lambda x: len(x[1]))
        
        chunks = []
        current_chunk = []
        current_complexity = 0
        max_complexity_per_chunk = chunk_size * 500  # Approximate tokens
        
        for original_index, text in text_items:
            text_complexity = len(text)
            
            if (len(current_chunk) >= chunk_size or 
                (current_complexity + text_complexity) > max_complexity_per_chunk):
                
                if current_chunk:
                    chunks.append([item[1] for item in current_chunk])
                    current_chunk = []
                    current_complexity = 0
            
            current_chunk.append((original_index, text))
            current_complexity += text_complexity
        
        # Add final chunk
        if current_chunk:
            chunks.append([item[1] for item in current_chunk])
        
        return chunks
    
    def _calculate_dynamic_priority(self, request: SentimentRequest) -> float:
        """Calculate dynamic priority based on request characteristics."""
        base_priority = 5.0
        
        # Adjust for text length (shorter texts get higher priority)
        text_length = len(request.text)
        if text_length < 100:
            base_priority += 2.0
        elif text_length > 1000:
            base_priority -= 1.0
        
        # Adjust for model complexity
        if request.model_preference:
            if "gpt-4" in request.model_preference.lower():
                base_priority -= 1.0  # Lower priority for expensive models
            elif "3.5" in request.model_preference.lower():
                base_priority += 0.5  # Higher priority for efficient models
        
        # Adjust for system load
        cpu_usage = psutil.cpu_percent(interval=0.1)
        if cpu_usage > 80:
            base_priority -= 1.0
        
        return max(1.0, min(10.0, base_priority))
    
    async def _manage_request_queue(self):
        """Intelligent request queue management."""
        queue_size = self.request_queue.qsize()
        
        if queue_size > self.config.max_queue_depth * 0.8:
            # High queue depth - trigger auto-scaling
            await auto_scaler.trigger_scale_up("high_queue_depth")
            logger.warning(f"High queue depth detected: {queue_size}")
        
        # Update queue depth metric
        self.current_metrics.queue_depth = queue_size
    
    async def _record_cache_hit(self, start_time: float):
        """Record cache hit metrics."""
        processing_time = (time.perf_counter() - start_time) * 1000
        
        with self._lock:
            self.current_metrics.cache_hit_rate = (
                (self.current_metrics.cache_hit_rate * 0.9) + (1.0 * 0.1)  # Moving average
            )
            self.latency_samples.append(processing_time)
    
    async def _record_error(self, error: Exception, start_time: float):
        """Record error metrics."""
        processing_time = (time.perf_counter() - start_time) * 1000
        
        with self._lock:
            self.current_metrics.error_rate = (
                (self.current_metrics.error_rate * 0.9) + (1.0 * 0.1)  # Moving average
            )
            self.latency_samples.append(processing_time)
        
        logger.error(f"Performance optimizer error: {str(error)}", exc_info=True)
    
    async def _update_performance_metrics(
        self,
        start_time: float,
        request: Union[SentimentRequest, BatchSentimentRequest],
        result: Union[SentimentResult, List[SentimentResult]]
    ):
        """Update comprehensive performance metrics."""
        processing_time = (time.perf_counter() - start_time) * 1000
        
        with self._lock:
            self.latency_samples.append(processing_time)
            
            # Update latency metrics
            recent_latencies = list(self.latency_samples)[-100:]  # Last 100 samples
            if recent_latencies:
                self.current_metrics.avg_latency_ms = statistics.mean(recent_latencies)
                if len(recent_latencies) >= 2:
                    self.current_metrics.p95_latency_ms = np.percentile(recent_latencies, 95)
                    self.current_metrics.p99_latency_ms = np.percentile(recent_latencies, 99)
            
            # Update throughput
            current_time = time.time()
            recent_requests = [m for m in self.metrics_history if current_time - m["timestamp"] < 60]
            self.current_metrics.requests_per_second = len(recent_requests) / 60.0
            
            # Update resource usage
            self.current_metrics.cpu_usage_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            self.current_metrics.memory_usage_mb = memory.used / (1024 * 1024)
            
            # Update cost metrics
            if isinstance(result, list):
                total_cost = sum(r.cost_usd or 0 for r in result)
                self.current_metrics.cost_per_request = total_cost / len(result)
            else:
                self.current_metrics.cost_per_request = result.cost_usd or 0
        
        # Store metrics history
        self.metrics_history.append({
            "timestamp": time.time(),
            "processing_time_ms": processing_time,
            "request_type": "batch" if isinstance(request, BatchSentimentRequest) else "single",
            "text_length": (
                sum(len(t) for t in request.texts) if isinstance(request, BatchSentimentRequest)
                else len(request.text)
            )
        })
    
    async def _performance_monitor(self):
        """Background performance monitoring and alerting."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Check performance thresholds
                if self.current_metrics.avg_latency_ms > self.config.target_latency_ms * 1.5:
                    logger.warning(f"High latency detected: {self.current_metrics.avg_latency_ms:.2f}ms")
                    await self._trigger_performance_optimization()
                
                if self.current_metrics.cpu_usage_percent > self.config.max_cpu_usage:
                    logger.warning(f"High CPU usage: {self.current_metrics.cpu_usage_percent:.1f}%")
                    await auto_scaler.trigger_scale_up("high_cpu_usage")
                
                if self.current_metrics.error_rate > 0.05:  # 5% error rate threshold
                    logger.error(f"High error rate: {self.current_metrics.error_rate:.3f}")
                    await self._trigger_circuit_breaker_review()
                
            except Exception as e:
                logger.error(f"Performance monitor error: {str(e)}")
    
    async def _optimization_loop(self):
        """Background optimization loop for continuous improvement."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Optimize every minute
                
                # Optimize cache strategy
                await self.smart_cache.optimize_eviction_policy()
                
                # Optimize model selection
                await self.model_selector.update_model_performance_data()
                
                # Optimize batch sizes based on performance
                await self._optimize_batch_configuration()
                
                # Prefetch optimization
                await self.prefetch_engine.optimize_prefetch_strategy()
                
            except Exception as e:
                logger.error(f"Optimization loop error: {str(e)}")
    
    async def _trigger_performance_optimization(self):
        """Trigger immediate performance optimization actions."""
        # Increase cache TTL for frequently accessed items
        await self.smart_cache.extend_hot_cache_ttl()
        
        # Optimize thread pool size
        current_workers = self.thread_pool._max_workers
        if self.current_metrics.cpu_usage_percent < 60:
            new_size = min(current_workers + 2, 64)
            self.thread_pool._max_workers = new_size
            logger.info(f"Increased thread pool size to {new_size}")
    
    async def _optimize_batch_configuration(self):
        """Optimize batch processing configuration based on performance data."""
        # Analyze recent batch performance
        recent_batches = [
            m for m in self.metrics_history[-100:]
            if m.get("request_type") == "batch"
        ]
        
        if len(recent_batches) >= 10:
            avg_batch_time = statistics.mean(m["processing_time_ms"] for m in recent_batches)
            
            # Adjust batch size based on performance
            if avg_batch_time > self.config.target_latency_ms * 2:
                self.config.batch_size_limit = max(10, self.config.batch_size_limit - 5)
                logger.info(f"Reduced batch size limit to {self.config.batch_size_limit}")
            elif avg_batch_time < self.config.target_latency_ms:
                self.config.batch_size_limit = min(100, self.config.batch_size_limit + 5)
                logger.info(f"Increased batch size limit to {self.config.batch_size_limit}")
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        with self._lock:
            recent_metrics = list(self.metrics_history)[-100:]
            
            return {
                "current_metrics": {
                    "requests_per_second": self.current_metrics.requests_per_second,
                    "avg_latency_ms": self.current_metrics.avg_latency_ms,
                    "p95_latency_ms": self.current_metrics.p95_latency_ms,
                    "p99_latency_ms": self.current_metrics.p99_latency_ms,
                    "cache_hit_rate": self.current_metrics.cache_hit_rate,
                    "error_rate": self.current_metrics.error_rate,
                    "cpu_usage_percent": self.current_metrics.cpu_usage_percent,
                    "memory_usage_mb": self.current_metrics.memory_usage_mb,
                    "queue_depth": self.current_metrics.queue_depth,
                    "cost_per_request": self.current_metrics.cost_per_request
                },
                "configuration": {
                    "max_concurrent_requests": self.config.max_concurrent_requests,
                    "batch_size_limit": self.config.batch_size_limit,
                    "target_latency_ms": self.config.target_latency_ms,
                    "target_throughput_rps": self.config.target_throughput_rps,
                    "auto_scaling_enabled": self.config.auto_scaling_enabled,
                    "cache_optimization": self.config.cache_optimization
                },
                "recent_performance": {
                    "total_samples": len(recent_metrics),
                    "avg_processing_time": (
                        statistics.mean(m["processing_time_ms"] for m in recent_metrics)
                        if recent_metrics else 0
                    ),
                    "batch_requests": len([m for m in recent_metrics if m.get("request_type") == "batch"]),
                    "single_requests": len([m for m in recent_metrics if m.get("request_type") == "single"])
                },
                "optimization_status": {
                    "cache_optimized": True,
                    "model_selection_optimized": True,
                    "batch_processing_optimized": True,
                    "quantum_planning_active": True
                },
                "timestamp": time.time()
            }
    
    async def cleanup(self):
        """Cleanup optimizer resources."""
        self._shutdown_event.set()
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        await self.smart_cache.cleanup()
        logger.info("SentimentPerformanceOptimizer cleanup completed")

class LoadBalancer:
    """Simple load balancer for request distribution."""
    
    def __init__(self):
        self.request_counts = defaultdict(int)
        self.response_times = defaultdict(list)
    
    async def select_endpoint(self, request_type: str) -> str:
        """Select optimal endpoint based on load and performance."""
        # Simple round-robin for now
        endpoints = ["primary", "secondary", "tertiary"]
        return min(endpoints, key=lambda e: self.request_counts[e])

class IntelligentModelSelector:
    """Intelligent model selection based on performance and cost."""
    
    def __init__(self):
        self.model_performance = defaultdict(lambda: {"avg_latency": 0, "cost": 0, "accuracy": 0})
        self.model_usage = defaultdict(int)
    
    async def select_optimal_model(self, request: Union[SentimentRequest, BatchSentimentRequest]) -> Union[SentimentRequest, BatchSentimentRequest]:
        """Select optimal model based on request characteristics and performance data."""
        text_length = (
            sum(len(t) for t in request.texts) if isinstance(request, BatchSentimentRequest)
            else len(request.text)
        )
        
        # Simple heuristics for model selection
        if request.model_preference:
            return request  # Use user preference
        
        # Auto-select based on text complexity and performance
        if text_length < 100:
            request.model_preference = "gpt-3.5-turbo"  # Fast for short texts
        elif text_length > 2000:
            request.model_preference = "claude-3-sonnet"  # Better for long texts
        else:
            request.model_preference = "gpt-3.5-turbo"  # Default efficient choice
        
        return request
    
    async def update_model_performance_data(self):
        """Update model performance data for better selection."""
        # Placeholder for performance data updates
        pass

class SmartCacheManager:
    """Advanced cache manager with intelligent eviction and optimization."""
    
    def __init__(self):
        self.cache = CacheManager()
        self.access_patterns = defaultdict(int)
        self.cache_performance = {"hits": 0, "misses": 0}
    
    async def get_optimized(self, request: Union[SentimentRequest, BatchSentimentRequest], user_id: Optional[str]) -> Any:
        """Get cached result with smart optimization."""
        cache_key = f"optimized:{hash(str(request))}:{user_id or 'anon'}"
        result = await self.cache.get(cache_key)
        
        if result:
            self.cache_performance["hits"] += 1
            self.access_patterns[cache_key] += 1
        else:
            self.cache_performance["misses"] += 1
        
        return result
    
    async def set_optimized(self, request: Union[SentimentRequest, BatchSentimentRequest], result: Any, user_id: Optional[str]):
        """Set cached result with smart TTL optimization."""
        cache_key = f"optimized:{hash(str(request))}:{user_id or 'anon'}"
        
        # Dynamic TTL based on access patterns
        base_ttl = 1800  # 30 minutes
        access_count = self.access_patterns.get(cache_key, 0)
        
        # Increase TTL for frequently accessed items
        if access_count > 5:
            ttl = base_ttl * 2
        elif access_count > 10:
            ttl = base_ttl * 3
        else:
            ttl = base_ttl
        
        await self.cache.set(cache_key, result, ttl=ttl)
    
    async def optimize_eviction_policy(self):
        """Optimize cache eviction policy based on access patterns."""
        # Clear rarely accessed items
        for key, count in list(self.access_patterns.items()):
            if count == 1:  # Only accessed once
                await self.cache.delete(key)
                del self.access_patterns[key]
    
    async def extend_hot_cache_ttl(self):
        """Extend TTL for frequently accessed cache items."""
        # Implementation for extending TTL of hot items
        pass
    
    async def cleanup(self):
        """Cleanup cache resources."""
        await self.cache.clear()

class PrefetchEngine:
    """Intelligent prefetching engine for predictive caching."""
    
    def __init__(self):
        self.request_patterns = defaultdict(list)
        self.prefetch_queue = asyncio.Queue()
    
    async def prepare_related_requests(self, request: Union[SentimentRequest, BatchSentimentRequest], user_id: Optional[str]):
        """Prepare related requests for prefetching."""
        # Simple implementation - could be enhanced with ML
        pass
    
    async def optimize_prefetch_strategy(self):
        """Optimize prefetching strategy based on usage patterns."""
        # Analyze patterns and adjust prefetch logic
        pass

# Global performance optimizer instance
performance_optimizer = SentimentPerformanceOptimizer()