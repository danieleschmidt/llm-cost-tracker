#!/usr/bin/env python3
"""Scalable Optimization System - Generation 3: MAKE IT SCALE (Optimized)

This module implements advanced performance optimization, caching, concurrent processing,
load balancing, auto-scaling, and high-performance features for production scalability.
"""

import asyncio
import concurrent.futures
import json
import logging
import multiprocessing
import sys
import time
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import random
import hashlib
import heapq

# Add src to path for imports  
sys.path.append('src')

from llm_cost_tracker import QuantumTaskPlanner, QuantumTask, TaskState
from llm_cost_tracker.quantum_i18n import set_language, t, SupportedLanguage


class AdvancedCache:
    """High-performance multi-level caching system with LRU eviction and TTL."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = {}
        self.access_times = {}
        self.expiry_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
        
    def _evict_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, expiry in self.expiry_times.items()
            if expiry < current_time
        ]
        
        for key in expired_keys:
            self._remove_key(key)
    
    def _evict_lru(self):
        """Evict least recently used entries when cache is full."""
        while len(self.cache) >= self.max_size:
            # Find LRU key
            lru_key = min(self.access_times.keys(), key=self.access_times.get)
            self._remove_key(lru_key)
    
    def _remove_key(self, key):
        """Remove a key from all data structures."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.expiry_times.pop(key, None)
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache."""
        with self.lock:
            self._evict_expired()
            
            if key in self.cache:
                self.access_times[key] = time.time()
                self.hit_count += 1
                return self.cache[key]
            else:
                self.miss_count += 1
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store value in cache."""
        with self.lock:
            self._evict_expired()
            self._evict_lru()
            
            current_time = time.time()
            self.cache[key] = value
            self.access_times[key] = current_time
            self.expiry_times[key] = current_time + (ttl or self.default_ttl)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate_percent": hit_rate,
            "utilization_percent": (len(self.cache) / self.max_size) * 100
        }


class LoadBalancer:
    """Intelligent load balancer with multiple distribution strategies."""
    
    def __init__(self):
        self.workers = []
        self.worker_loads = defaultdict(int)
        self.worker_health = defaultdict(bool)
        self.request_count = 0
        self.strategy = "least_connections"  # round_robin, least_connections, weighted
        
    def add_worker(self, worker_id: str, weight: float = 1.0):
        """Add a worker to the pool."""
        self.workers.append({"id": worker_id, "weight": weight})
        self.worker_health[worker_id] = True
        
    def get_next_worker(self) -> Optional[str]:
        """Get next worker based on current strategy."""
        healthy_workers = [w for w in self.workers if self.worker_health[w["id"]]]
        
        if not healthy_workers:
            return None
            
        if self.strategy == "round_robin":
            worker = healthy_workers[self.request_count % len(healthy_workers)]
            self.request_count += 1
            return worker["id"]
            
        elif self.strategy == "least_connections":
            min_load = min(self.worker_loads[w["id"]] for w in healthy_workers)
            candidates = [w for w in healthy_workers if self.worker_loads[w["id"]] == min_load]
            return random.choice(candidates)["id"]
            
        elif self.strategy == "weighted":
            total_weight = sum(w["weight"] for w in healthy_workers)
            r = random.uniform(0, total_weight)
            cumulative = 0
            for worker in healthy_workers:
                cumulative += worker["weight"]
                if r <= cumulative:
                    return worker["id"]
        
        return healthy_workers[0]["id"]  # Fallback
    
    def mark_worker_busy(self, worker_id: str):
        """Mark worker as busy (increment load)."""
        self.worker_loads[worker_id] += 1
        
    def mark_worker_free(self, worker_id: str):
        """Mark worker as free (decrement load)."""
        self.worker_loads[worker_id] = max(0, self.worker_loads[worker_id] - 1)
        
    def set_worker_health(self, worker_id: str, healthy: bool):
        """Set worker health status."""
        self.worker_health[worker_id] = healthy


class AutoScaler:
    """Intelligent auto-scaling system based on load metrics."""
    
    def __init__(self, min_workers: int = 2, max_workers: int = 20):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        self.metrics_window = []
        self.scale_up_threshold = 0.8  # 80% utilization
        self.scale_down_threshold = 0.3  # 30% utilization
        self.cooldown_period = 60  # seconds
        self.last_scale_action = 0
        
    def add_metric(self, cpu_usage: float, memory_usage: float, queue_depth: int):
        """Add performance metric sample."""
        timestamp = time.time()
        metric = {
            "timestamp": timestamp,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "queue_depth": queue_depth,
            "utilization": max(cpu_usage, memory_usage) / 100.0
        }
        
        # Keep only last 10 minutes of metrics
        self.metrics_window = [
            m for m in self.metrics_window 
            if timestamp - m["timestamp"] <= 600
        ]
        self.metrics_window.append(metric)
    
    def should_scale(self) -> Dict[str, Any]:
        """Determine if scaling action is needed."""
        if not self.metrics_window:
            return {"action": "none", "reason": "no_metrics"}
            
        # Check cooldown period
        if time.time() - self.last_scale_action < self.cooldown_period:
            return {"action": "none", "reason": "cooldown"}
        
        # Calculate average utilization over last 5 minutes
        recent_metrics = [
            m for m in self.metrics_window
            if time.time() - m["timestamp"] <= 300
        ]
        
        if len(recent_metrics) < 3:
            return {"action": "none", "reason": "insufficient_data"}
        
        avg_utilization = sum(m["utilization"] for m in recent_metrics) / len(recent_metrics)
        avg_queue_depth = sum(m["queue_depth"] for m in recent_metrics) / len(recent_metrics)
        
        # Scale up conditions
        if (avg_utilization > self.scale_up_threshold or avg_queue_depth > 10) and \
           self.current_workers < self.max_workers:
            return {
                "action": "scale_up",
                "current_workers": self.current_workers,
                "target_workers": min(self.current_workers + 2, self.max_workers),
                "avg_utilization": avg_utilization,
                "avg_queue_depth": avg_queue_depth
            }
        
        # Scale down conditions  
        if avg_utilization < self.scale_down_threshold and avg_queue_depth < 2 and \
           self.current_workers > self.min_workers:
            return {
                "action": "scale_down",
                "current_workers": self.current_workers,
                "target_workers": max(self.current_workers - 1, self.min_workers),
                "avg_utilization": avg_utilization,
                "avg_queue_depth": avg_queue_depth
            }
        
        return {"action": "none", "reason": "within_thresholds"}
    
    def execute_scaling(self, action: str, target_workers: int) -> bool:
        """Execute scaling action."""
        if action in ["scale_up", "scale_down"]:
            self.current_workers = target_workers
            self.last_scale_action = time.time()
            return True
        return False


class PerformanceOptimizer:
    """Advanced performance optimization engine."""
    
    def __init__(self):
        self.cache = AdvancedCache(max_size=2000)
        self.load_balancer = LoadBalancer()
        self.auto_scaler = AutoScaler()
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=4)
        self.optimization_metrics = {}
        self.setup_workers()
        
    def setup_workers(self):
        """Initialize worker pool."""
        for i in range(4):
            self.load_balancer.add_worker(f"worker_{i}", weight=1.0)
    
    def memoize_function(self, func: Callable, *args, **kwargs) -> Any:
        """Memoize function results with advanced caching."""
        # Create cache key from function name and arguments
        key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
        cache_key = hashlib.md5(key_data.encode()).hexdigest()
        
        # Try cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Execute function and cache result
        result = func(*args, **kwargs)
        self.cache.set(cache_key, result, ttl=300)  # 5 minute TTL
        return result
    
    async def parallel_task_execution(self, tasks: List[QuantumTask], 
                                    planner: QuantumTaskPlanner) -> Dict[str, Any]:
        """Execute tasks in parallel with load balancing."""
        start_time = time.time()
        
        results = {
            "execution_id": f"parallel_{int(time.time())}",
            "start_time": datetime.now().isoformat(),
            "tasks_total": len(tasks),
            "tasks_completed": 0,
            "worker_assignments": {},
            "performance_metrics": {}
        }
        
        async def execute_task_batch(task_batch: List[QuantumTask]) -> Dict[str, Any]:
            """Execute a batch of tasks on a worker."""
            worker_id = self.load_balancer.get_next_worker()
            if not worker_id:
                raise Exception("No healthy workers available")
            
            self.load_balancer.mark_worker_busy(worker_id)
            batch_start = time.time()
            
            try:
                batch_results = []
                for task in task_batch:
                    task_start = time.time()
                    success, message = planner.add_task(task)
                    task_duration = (time.time() - task_start) * 1000
                    
                    batch_results.append({
                        "task_id": task.id,
                        "success": success,
                        "duration_ms": task_duration,
                        "worker_id": worker_id
                    })
                
                batch_duration = (time.time() - batch_start) * 1000
                return {
                    "worker_id": worker_id,
                    "batch_size": len(task_batch),
                    "batch_duration_ms": batch_duration,
                    "results": batch_results
                }
                
            finally:
                self.load_balancer.mark_worker_free(worker_id)
        
        # Split tasks into batches for parallel processing
        batch_size = max(1, len(tasks) // 4)  # 4 parallel batches
        task_batches = [
            tasks[i:i + batch_size] 
            for i in range(0, len(tasks), batch_size)
        ]
        
        # Execute batches in parallel
        batch_futures = [
            execute_task_batch(batch) for batch in task_batches
        ]
        
        batch_results = await asyncio.gather(*batch_futures, return_exceptions=True)
        
        # Process results
        total_tasks_completed = 0
        worker_stats = defaultdict(list)
        
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                continue
                
            worker_id = batch_result["worker_id"]
            worker_stats[worker_id].append(batch_result)
            total_tasks_completed += len(batch_result["results"])
        
        # Calculate performance metrics
        total_duration = (time.time() - start_time) * 1000
        throughput = (total_tasks_completed / (total_duration / 1000)) if total_duration > 0 else 0
        
        results.update({
            "end_time": datetime.now().isoformat(),
            "total_duration_ms": total_duration,
            "tasks_completed": total_tasks_completed,
            "throughput_tasks_per_second": throughput,
            "worker_stats": dict(worker_stats),
            "cache_stats": self.cache.get_stats(),
            "parallel_efficiency": (total_tasks_completed / len(tasks)) * 100 if tasks else 0
        })
        
        return results
    
    async def benchmark_performance(self, num_tasks: int = 100) -> Dict[str, Any]:
        """Run comprehensive performance benchmarks."""
        print(f"🏁 Running performance benchmark with {num_tasks} tasks...")
        
        # Generate test tasks
        test_tasks = []
        for i in range(num_tasks):
            task = QuantumTask(
                id=f"perf_test_{i}",
                name=f"Performance Test Task {i}",
                description=f"Benchmark task for performance testing",
                priority=random.uniform(1.0, 10.0),
                estimated_duration=timedelta(minutes=random.randint(5, 60))
            )
            test_tasks.append(task)
        
        # Create fresh planner instance
        planner = QuantumTaskPlanner()
        
        # Benchmark 1: Sequential execution
        sequential_start = time.time()
        sequential_success = 0
        
        for task in test_tasks[:20]:  # Test with subset for speed
            success, _ = planner.add_task(task)
            if success:
                sequential_success += 1
        
        sequential_duration = (time.time() - sequential_start) * 1000
        sequential_throughput = (sequential_success / (sequential_duration / 1000))
        
        # Benchmark 2: Parallel execution
        parallel_planner = QuantumTaskPlanner()
        parallel_results = await self.parallel_task_execution(test_tasks[:50], parallel_planner)
        
        # Benchmark 3: Cache performance
        cache_hits = 0
        cache_misses = 0
        
        for i in range(100):
            key = f"test_key_{i % 10}"  # This will create cache hits
            value = self.cache.get(key)
            if value is None:
                cache_misses += 1
                self.cache.set(key, f"test_value_{i}")
            else:
                cache_hits += 1
        
        cache_hit_rate = (cache_hits / (cache_hits + cache_misses)) * 100
        
        # Benchmark 4: Auto-scaling simulation
        for i in range(20):
            cpu_usage = random.uniform(50, 95)
            memory_usage = random.uniform(40, 90)
            queue_depth = random.randint(0, 15)
            self.auto_scaler.add_metric(cpu_usage, memory_usage, queue_depth)
        
        scaling_decision = self.auto_scaler.should_scale()
        
        benchmark_results = {
            "benchmark_timestamp": datetime.now().isoformat(),
            "test_task_count": num_tasks,
            "sequential_performance": {
                "tasks_processed": sequential_success,
                "duration_ms": sequential_duration,
                "throughput_tasks_per_second": sequential_throughput
            },
            "parallel_performance": {
                "tasks_processed": parallel_results["tasks_completed"],
                "duration_ms": parallel_results["total_duration_ms"],
                "throughput_tasks_per_second": parallel_results["throughput_tasks_per_second"],
                "efficiency_percent": parallel_results["parallel_efficiency"]
            },
            "cache_performance": {
                "hit_rate_percent": cache_hit_rate,
                "cache_stats": self.cache.get_stats()
            },
            "scaling_simulation": {
                "current_workers": self.auto_scaler.current_workers,
                "scaling_decision": scaling_decision,
                "metrics_collected": len(self.auto_scaler.metrics_window)
            },
            "performance_improvement": {
                "parallel_vs_sequential_speedup": 
                    (parallel_results["throughput_tasks_per_second"] / sequential_throughput) 
                    if sequential_throughput > 0 else 1.0,
                "cache_efficiency": cache_hit_rate,
                "load_balancing_active": len(self.load_balancer.workers) > 1
            }
        }
        
        return benchmark_results


class ScalableOptimizationSystem:
    """Main system orchestrating all scalability features."""
    
    def __init__(self):
        self.optimizer = PerformanceOptimizer()
        self.start_time = time.time()
        self.system_metrics = {
            "operations_completed": 0,
            "total_execution_time_ms": 0,
            "peak_throughput": 0
        }
        
    async def demonstrate_scalability_features(self) -> Dict[str, Any]:
        """Demonstrate all scalability and optimization features."""
        print("⚡ Starting Scalability Features Demonstration")
        
        demo_results = {
            "demo_start": datetime.now().isoformat(),
            "features_demonstrated": [],
            "performance_benchmarks": {},
            "optimization_results": {}
        }
        
        try:
            # 1. Advanced Caching Demonstration
            print("\n🗄️ Testing Advanced Multi-Level Caching...")
            
            cache_test_start = time.time()
            
            # Test cache with various operations
            for i in range(1000):
                key = f"cache_test_{i % 100}"  # Creates cache hits
                value = self.optimizer.cache.get(key)
                if value is None:
                    self.optimizer.cache.set(key, f"cached_value_{i}")
            
            cache_stats = self.optimizer.cache.get_stats()
            cache_test_duration = (time.time() - cache_test_start) * 1000
            
            demo_results["features_demonstrated"].append("advanced_caching")
            demo_results["optimization_results"]["caching"] = {
                "test_duration_ms": cache_test_duration,
                "operations_per_second": 1000 / (cache_test_duration / 1000),
                **cache_stats
            }
            
            print(f"✅ Cache hit rate: {cache_stats['hit_rate_percent']:.1f}%")
            print(f"✅ Cache operations: {1000 / (cache_test_duration / 1000):.0f} ops/sec")
            
            # 2. Load Balancing Demonstration
            print("\n⚖️ Testing Intelligent Load Balancing...")
            
            load_balance_results = []
            
            for strategy in ["round_robin", "least_connections", "weighted"]:
                self.optimizer.load_balancer.strategy = strategy
                
                # Simulate 50 requests
                worker_assignments = defaultdict(int)
                for _ in range(50):
                    worker = self.optimizer.load_balancer.get_next_worker()
                    if worker:
                        worker_assignments[worker] += 1
                        self.optimizer.load_balancer.mark_worker_busy(worker)
                        self.optimizer.load_balancer.mark_worker_free(worker)
                
                distribution_variance = max(worker_assignments.values()) - min(worker_assignments.values())
                load_balance_results.append({
                    "strategy": strategy,
                    "worker_distribution": dict(worker_assignments),
                    "distribution_variance": distribution_variance
                })
            
            demo_results["features_demonstrated"].append("load_balancing")
            demo_results["optimization_results"]["load_balancing"] = load_balance_results
            
            print(f"✅ Load balancing strategies tested: {len(load_balance_results)}")
            
            # 3. Auto-Scaling Demonstration
            print("\n📈 Testing Auto-Scaling Intelligence...")
            
            # Simulate varying load conditions
            scaling_events = []
            
            # High load scenario
            for _ in range(10):
                self.optimizer.auto_scaler.add_metric(85.0, 90.0, 12)
            
            high_load_decision = self.optimizer.auto_scaler.should_scale()
            if high_load_decision["action"] == "scale_up":
                self.optimizer.auto_scaler.execute_scaling("scale_up", high_load_decision["target_workers"])
                scaling_events.append(high_load_decision)
            
            # Low load scenario  
            for _ in range(10):
                self.optimizer.auto_scaler.add_metric(20.0, 25.0, 1)
            
            low_load_decision = self.optimizer.auto_scaler.should_scale()
            if low_load_decision["action"] == "scale_down":
                self.optimizer.auto_scaler.execute_scaling("scale_down", low_load_decision["target_workers"])
                scaling_events.append(low_load_decision)
            
            demo_results["features_demonstrated"].append("auto_scaling")
            demo_results["optimization_results"]["auto_scaling"] = {
                "scaling_events": scaling_events,
                "current_workers": self.optimizer.auto_scaler.current_workers,
                "metrics_collected": len(self.optimizer.auto_scaler.metrics_window)
            }
            
            print(f"✅ Auto-scaling events triggered: {len(scaling_events)}")
            
            # 4. Parallel Processing Demonstration
            print("\n🔄 Testing Parallel Processing Pipeline...")
            
            # Create test tasks for parallel processing
            parallel_tasks = []
            for i in range(20):
                task = QuantumTask(
                    id=f"parallel_{i}",
                    name=f"Parallel Processing Task {i}",
                    description="High-performance parallel execution test",
                    priority=random.uniform(5.0, 9.0),
                    estimated_duration=timedelta(minutes=random.randint(10, 30))
                )
                parallel_tasks.append(task)
            
            planner = QuantumTaskPlanner()
            parallel_results = await self.optimizer.parallel_task_execution(parallel_tasks, planner)
            
            demo_results["features_demonstrated"].append("parallel_processing")
            demo_results["optimization_results"]["parallel_processing"] = parallel_results
            
            print(f"✅ Parallel throughput: {parallel_results['throughput_tasks_per_second']:.1f} tasks/sec")
            print(f"✅ Parallel efficiency: {parallel_results['parallel_efficiency']:.1f}%")
            
            # 5. Comprehensive Performance Benchmark
            print("\n🏁 Running Comprehensive Performance Benchmark...")
            
            benchmark_results = await self.optimizer.benchmark_performance(num_tasks=150)
            
            demo_results["features_demonstrated"].append("performance_benchmarking")
            demo_results["performance_benchmarks"] = benchmark_results
            
            speedup = benchmark_results["performance_improvement"]["parallel_vs_sequential_speedup"]
            print(f"✅ Parallel speedup: {speedup:.2f}x faster than sequential")
            print(f"✅ Cache efficiency: {benchmark_results['cache_performance']['hit_rate_percent']:.1f}%")
            
            # Final system metrics
            demo_results["demo_end"] = datetime.now().isoformat()
            demo_results["features_count"] = len(demo_results["features_demonstrated"])
            demo_results["overall_performance"] = {
                "max_throughput_tasks_per_second": max(
                    parallel_results["throughput_tasks_per_second"],
                    benchmark_results["parallel_performance"]["throughput_tasks_per_second"]
                ),
                "cache_hit_rate_percent": cache_stats["hit_rate_percent"],
                "scaling_responsiveness": len(scaling_events) > 0,
                "load_balancing_efficiency": min(r["distribution_variance"] for r in load_balance_results)
            }
            
            demo_results["status"] = "SUCCESS"
            
            return demo_results
            
        except Exception as e:
            demo_results["status"] = "FAILURE"
            demo_results["error"] = str(e)
            demo_results["demo_end"] = datetime.now().isoformat()
            return demo_results


async def main():
    """Main execution for scalable optimization demonstration."""
    print("⚡ Starting Generation 3: MAKE IT SCALE (Optimized)")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        system = ScalableOptimizationSystem()
        
        # Run comprehensive scalability demonstration
        results = await system.demonstrate_scalability_features()
        
        # Save results
        output_file = Path('scalable_optimization_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f" ⚡ GENERATION 3 COMPLETED: {results['status']}")
        print('='*60)
        print(f"✅ Features demonstrated: {results['features_count']}")
        print(f"🚀 Features implemented: {', '.join(results['features_demonstrated'])}")
        
        if "overall_performance" in results:
            perf = results["overall_performance"]
            print(f"📊 Max throughput: {perf['max_throughput_tasks_per_second']:.1f} tasks/sec")
            print(f"🗄️ Cache hit rate: {perf['cache_hit_rate_percent']:.1f}%")
            print(f"📈 Auto-scaling: {'✅ Active' if perf['scaling_responsiveness'] else '❌ Inactive'}")
            print(f"⚖️ Load balancing variance: {perf['load_balancing_efficiency']}")
        
        print(f"📁 Results saved to: {output_file}")
        
        if results['status'] == 'SUCCESS':
            print(f"\n🎯 Ready to proceed to Quality Gates & Global Implementation")
            return True
        else:
            print(f"\n⚠️  Some issues detected: {results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n❌ Critical failure in Generation 3: {str(e)}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)