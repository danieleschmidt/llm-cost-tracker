"""Advanced Production Optimization System with Autonomous Enhancement."""

import asyncio
import json
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from collections import defaultdict, deque
import random
import uuid

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Production optimization strategies."""
    PERFORMANCE = "performance"        # Focus on speed and throughput
    RESOURCE_EFFICIENCY = "resource"   # Focus on resource utilization
    COST_OPTIMIZATION = "cost"         # Focus on cost reduction
    RELIABILITY = "reliability"        # Focus on stability and uptime
    ADAPTIVE = "adaptive"              # Dynamic strategy selection


class MetricType(Enum):
    """Types of production metrics."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    COST_PER_REQUEST = "cost_per_request"
    AVAILABILITY = "availability"
    CONCURRENT_USERS = "concurrent_users"


@dataclass
class OptimizationTarget:
    """Target values for optimization metrics."""
    metric_type: MetricType
    target_value: float
    priority: float  # 0.0 to 1.0
    tolerance: float = 0.1  # Acceptable deviation from target
    
    def is_within_tolerance(self, current_value: float) -> bool:
        """Check if current value is within acceptable tolerance."""
        return abs(current_value - self.target_value) <= self.tolerance * self.target_value


@dataclass
class ProductionMetric:
    """Production metric measurement."""
    metric_type: MetricType
    value: float
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_type": self.metric_type.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata
        }


@dataclass
class OptimizationAction:
    """Action to be taken for optimization."""
    action_id: str
    action_type: str
    description: str
    parameters: Dict[str, Any]
    expected_impact: Dict[MetricType, float]
    priority: float
    estimated_cost: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "description": self.description,
            "parameters": self.parameters,
            "expected_impact": {k.value: v for k, v in self.expected_impact.items()},
            "priority": self.priority,
            "estimated_cost": self.estimated_cost
        }


@dataclass
class OptimizationResult:
    """Result of an optimization action."""
    action_id: str
    success: bool
    applied_at: datetime
    metrics_before: Dict[MetricType, float]
    metrics_after: Dict[MetricType, float]
    actual_impact: Dict[MetricType, float]
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_id": self.action_id,
            "success": self.success,
            "applied_at": self.applied_at.isoformat(),
            "metrics_before": {k.value: v for k, v in self.metrics_before.items()},
            "metrics_after": {k.value: v for k, v in self.metrics_after.items()},
            "actual_impact": {k.value: v for k, v in self.actual_impact.items()},
            "error_message": self.error_message
        }


class AdaptiveLoadBalancer:
    """Adaptive load balancing with real-time optimization."""
    
    def __init__(self):
        """Initialize adaptive load balancer."""
        self.servers: List[Dict[str, Any]] = []
        self.routing_weights: Dict[str, float] = {}
        self.health_scores: Dict[str, float] = {}
        self.current_strategy = "round_robin"
        
    def add_server(self, server_id: str, capacity: float, current_load: float = 0.0) -> None:
        """Add a server to the load balancer."""
        self.servers.append({
            "id": server_id,
            "capacity": capacity,
            "current_load": current_load,
            "health_score": 1.0,
            "last_health_check": datetime.now()
        })
        self.routing_weights[server_id] = 1.0
        self.health_scores[server_id] = 1.0
    
    def update_server_metrics(self, server_id: str, metrics: Dict[str, float]) -> None:
        """Update server metrics for optimization."""
        for server in self.servers:
            if server["id"] == server_id:
                server.update(metrics)
                server["last_update"] = datetime.now()
                
                # Calculate health score based on metrics
                health_score = self._calculate_health_score(metrics)
                self.health_scores[server_id] = health_score
                server["health_score"] = health_score
                break
    
    def _calculate_health_score(self, metrics: Dict[str, float]) -> float:
        """Calculate server health score from metrics."""
        cpu_utilization = metrics.get("cpu_utilization", 0.5)
        memory_utilization = metrics.get("memory_utilization", 0.5)
        error_rate = metrics.get("error_rate", 0.0)
        latency = metrics.get("latency", 0.1)
        
        # Health score based on multiple factors
        cpu_score = max(0, 1 - (cpu_utilization - 0.7) / 0.3) if cpu_utilization > 0.7 else 1.0
        memory_score = max(0, 1 - (memory_utilization - 0.8) / 0.2) if memory_utilization > 0.8 else 1.0
        error_score = max(0, 1 - error_rate / 0.1) if error_rate > 0 else 1.0
        latency_score = max(0, 1 - (latency - 0.1) / 0.9) if latency > 0.1 else 1.0
        
        return (cpu_score + memory_score + error_score + latency_score) / 4
    
    def get_optimal_server(self, request_size: float = 1.0) -> Optional[str]:
        """Get optimal server for request routing."""
        if not self.servers:
            return None
        
        if self.current_strategy == "weighted_health":
            # Route based on health scores and capacity
            healthy_servers = [
                server for server in self.servers
                if server["health_score"] > 0.5
            ]
            
            if not healthy_servers:
                healthy_servers = self.servers  # Fallback to all servers
            
            # Calculate routing weights
            total_weight = 0
            weights = {}
            
            for server in healthy_servers:
                available_capacity = server["capacity"] - server.get("current_load", 0)
                weight = server["health_score"] * max(0.1, available_capacity)
                weights[server["id"]] = weight
                total_weight += weight
            
            if total_weight == 0:
                return healthy_servers[0]["id"]  # Fallback
            
            # Weighted random selection
            rand = random.random() * total_weight
            cumulative = 0
            
            for server_id, weight in weights.items():
                cumulative += weight
                if rand <= cumulative:
                    return server_id
            
            return list(weights.keys())[0]  # Fallback
        
        elif self.current_strategy == "least_loaded":
            # Route to server with lowest load
            available_servers = [
                server for server in self.servers
                if server["health_score"] > 0.3
            ]
            
            if not available_servers:
                available_servers = self.servers
            
            return min(available_servers, key=lambda s: s.get("current_load", 0))["id"]
        
        else:  # round_robin fallback
            healthy_servers = [
                server for server in self.servers
                if server["health_score"] > 0.5
            ]
            
            if not healthy_servers:
                healthy_servers = self.servers
            
            # Simple round-robin among healthy servers
            current_time_ms = int(time.time() * 1000)
            return healthy_servers[current_time_ms % len(healthy_servers)]["id"]
    
    def optimize_routing_strategy(self, recent_metrics: List[ProductionMetric]) -> str:
        """Optimize routing strategy based on recent performance."""
        if not recent_metrics:
            return self.current_strategy
        
        # Analyze performance patterns
        latency_metrics = [m for m in recent_metrics if m.metric_type == MetricType.LATENCY]
        error_metrics = [m for m in recent_metrics if m.metric_type == MetricType.ERROR_RATE]
        
        if latency_metrics:
            avg_latency = statistics.mean([m.value for m in latency_metrics])
            latency_variance = statistics.variance([m.value for m in latency_metrics]) if len(latency_metrics) > 1 else 0
            
            # High latency variance suggests uneven load distribution
            if latency_variance > avg_latency * 0.5:
                self.current_strategy = "least_loaded"
            elif avg_latency > 0.5:  # High average latency
                self.current_strategy = "weighted_health"
            else:
                self.current_strategy = "round_robin"
        
        return self.current_strategy


class AutoScalingEngine:
    """Intelligent auto-scaling engine with predictive capabilities."""
    
    def __init__(self):
        """Initialize auto-scaling engine."""
        self.scaling_rules: List[Dict[str, Any]] = []
        self.scaling_history: List[Dict[str, Any]] = []
        self.current_instances = 1
        self.min_instances = 1
        self.max_instances = 10
        self.scale_up_threshold = 0.7
        self.scale_down_threshold = 0.3
        self.cooldown_period = timedelta(minutes=5)
        self.last_scaling_action: Optional[datetime] = None
        
    def add_scaling_rule(self,
                        metric_type: MetricType,
                        scale_up_threshold: float,
                        scale_down_threshold: float,
                        weight: float = 1.0) -> None:
        """Add a scaling rule based on a metric."""
        rule = {
            "metric_type": metric_type,
            "scale_up_threshold": scale_up_threshold,
            "scale_down_threshold": scale_down_threshold,
            "weight": weight
        }
        self.scaling_rules.append(rule)
    
    def evaluate_scaling_decision(self, current_metrics: Dict[MetricType, float]) -> Optional[str]:
        """Evaluate if scaling action is needed."""
        current_time = datetime.now()
        
        # Check cooldown period
        if (self.last_scaling_action and 
            current_time - self.last_scaling_action < self.cooldown_period):
            return None
        
        scale_up_votes = 0
        scale_down_votes = 0
        total_weight = 0
        
        # Evaluate each scaling rule
        for rule in self.scaling_rules:
            metric_type = rule["metric_type"]
            weight = rule["weight"]
            total_weight += weight
            
            if metric_type in current_metrics:
                value = current_metrics[metric_type]
                
                if value >= rule["scale_up_threshold"]:
                    scale_up_votes += weight
                elif value <= rule["scale_down_threshold"]:
                    scale_down_votes += weight
        
        if total_weight == 0:
            return None
        
        # Make scaling decision
        scale_up_ratio = scale_up_votes / total_weight
        scale_down_ratio = scale_down_votes / total_weight
        
        if scale_up_ratio > 0.6 and self.current_instances < self.max_instances:
            return "scale_up"
        elif scale_down_ratio > 0.6 and self.current_instances > self.min_instances:
            return "scale_down"
        
        return None
    
    def predict_future_load(self, historical_metrics: List[ProductionMetric]) -> float:
        """Predict future load based on historical patterns."""
        if not historical_metrics:
            return 0.5  # Default prediction
        
        # Simple trend analysis
        recent_metrics = historical_metrics[-20:]  # Last 20 data points
        if len(recent_metrics) < 2:
            return recent_metrics[0].value if recent_metrics else 0.5
        
        # Calculate trend
        values = [m.value for m in recent_metrics]
        time_points = list(range(len(values)))
        
        # Simple linear regression
        n = len(values)
        sum_x = sum(time_points)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(time_points, values))
        sum_x2 = sum(x * x for x in time_points)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0
        intercept = (sum_y - slope * sum_x) / n
        
        # Predict next value
        next_point = len(values)
        predicted_value = slope * next_point + intercept
        
        return max(0, min(1, predicted_value))  # Clamp to [0, 1]
    
    def execute_scaling_action(self, action: str) -> bool:
        """Execute scaling action."""
        try:
            if action == "scale_up" and self.current_instances < self.max_instances:
                new_instances = min(self.max_instances, self.current_instances + 1)
                logger.info(f"Scaling up from {self.current_instances} to {new_instances} instances")
                self.current_instances = new_instances
                
            elif action == "scale_down" and self.current_instances > self.min_instances:
                new_instances = max(self.min_instances, self.current_instances - 1)
                logger.info(f"Scaling down from {self.current_instances} to {new_instances} instances")
                self.current_instances = new_instances
            
            else:
                return False
            
            # Record scaling action
            self.last_scaling_action = datetime.now()
            self.scaling_history.append({
                "timestamp": self.last_scaling_action.isoformat(),
                "action": action,
                "instances_before": self.current_instances - (1 if action == "scale_up" else -1),
                "instances_after": self.current_instances
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute scaling action {action}: {e}")
            return False


class AdvancedProductionOptimizer:
    """Advanced production optimization system with autonomous capabilities."""
    
    def __init__(self, project_root: Path = None):
        """Initialize the advanced production optimizer."""
        self.project_root = project_root or Path("/root/repo")
        
        # Core components
        self.load_balancer = AdaptiveLoadBalancer()
        self.auto_scaler = AutoScalingEngine()
        
        # Optimization state
        self.current_strategy = OptimizationStrategy.ADAPTIVE
        self.optimization_targets: List[OptimizationTarget] = []
        self.metrics_history: deque = deque(maxlen=1000)
        self.optimization_actions: List[OptimizationAction] = []
        self.optimization_results: List[OptimizationResult] = []
        
        # Monitoring
        self.monitoring_active = False
        self.optimization_active = False
        
        self._setup_default_configuration()
    
    def _setup_default_configuration(self) -> None:
        """Setup default optimization configuration."""
        # Default optimization targets
        self.optimization_targets = [
            OptimizationTarget(MetricType.LATENCY, 0.1, 0.9, 0.2),           # <100ms, high priority
            OptimizationTarget(MetricType.ERROR_RATE, 0.01, 0.8, 0.5),      # <1% error rate
            OptimizationTarget(MetricType.CPU_UTILIZATION, 0.7, 0.6, 0.1),  # 70% CPU utilization
            OptimizationTarget(MetricType.MEMORY_UTILIZATION, 0.8, 0.5, 0.1), # 80% memory utilization
            OptimizationTarget(MetricType.THROUGHPUT, 1000.0, 0.7, 0.1),    # 1000 req/s throughput
        ]
        
        # Setup load balancer
        self.load_balancer.add_server("server_1", 100.0, 0.0)
        self.load_balancer.add_server("server_2", 100.0, 0.0)
        self.load_balancer.add_server("server_3", 100.0, 0.0)
        
        # Setup auto-scaling rules
        self.auto_scaler.add_scaling_rule(MetricType.CPU_UTILIZATION, 0.8, 0.3, 1.0)
        self.auto_scaler.add_scaling_rule(MetricType.MEMORY_UTILIZATION, 0.85, 0.25, 0.8)
        self.auto_scaler.add_scaling_rule(MetricType.LATENCY, 0.2, 0.05, 0.9)
        self.auto_scaler.add_scaling_rule(MetricType.ERROR_RATE, 0.05, 0.001, 1.0)
    
    async def start_optimization(self) -> None:
        """Start autonomous optimization system."""
        if self.optimization_active:
            logger.warning("Optimization is already active")
            return
        
        self.optimization_active = True
        self.monitoring_active = True
        
        logger.info("Starting advanced production optimization...")
        
        # Start optimization tasks
        asyncio.create_task(self._optimization_loop())
        asyncio.create_task(self._metrics_collection_loop())
        asyncio.create_task(self._auto_scaling_loop())
        asyncio.create_task(self._load_balancing_optimization_loop())
        
        logger.info("Advanced production optimization started")
    
    async def stop_optimization(self) -> None:
        """Stop optimization system."""
        self.optimization_active = False
        self.monitoring_active = False
        logger.info("Production optimization stopped")
    
    def add_metric(self, metric: ProductionMetric) -> None:
        """Add a production metric."""
        self.metrics_history.append(metric)
        
        # Update load balancer with server metrics
        if metric.source.startswith("server_") and metric.source in [s["id"] for s in self.load_balancer.servers]:
            server_metrics = {metric.metric_type.value: metric.value}
            self.load_balancer.update_server_metrics(metric.source, server_metrics)
    
    def generate_optimization_actions(self) -> List[OptimizationAction]:
        """Generate optimization actions based on current state."""
        if not self.metrics_history:
            return []
        
        actions = []
        current_metrics = self._get_current_metrics()
        
        # Analyze each optimization target
        for target in self.optimization_targets:
            if target.metric_type in current_metrics:
                current_value = current_metrics[target.metric_type]
                
                if not target.is_within_tolerance(current_value):
                    # Generate actions to meet target
                    action = self._generate_action_for_target(target, current_value)
                    if action:
                        actions.append(action)
        
        # Sort actions by priority
        actions.sort(key=lambda a: a.priority, reverse=True)
        
        return actions
    
    def _get_current_metrics(self) -> Dict[MetricType, float]:
        """Get current aggregated metrics."""
        if not self.metrics_history:
            return {}
        
        # Get recent metrics (last 5 minutes)
        recent_time = datetime.now() - timedelta(minutes=5)
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp > recent_time
        ]
        
        if not recent_metrics:
            return {}
        
        # Aggregate by metric type
        aggregated = {}
        metric_groups = defaultdict(list)
        
        for metric in recent_metrics:
            metric_groups[metric.metric_type].append(metric.value)
        
        for metric_type, values in metric_groups.items():
            aggregated[metric_type] = statistics.mean(values)
        
        return aggregated
    
    def _generate_action_for_target(self, target: OptimizationTarget, current_value: float) -> Optional[OptimizationAction]:
        """Generate optimization action for a specific target."""
        action_id = str(uuid.uuid4())
        
        if target.metric_type == MetricType.LATENCY:
            if current_value > target.target_value:
                # High latency - need to optimize
                return OptimizationAction(
                    action_id=action_id,
                    action_type="reduce_latency",
                    description="Optimize application latency through caching and connection pooling",
                    parameters={
                        "enable_caching": True,
                        "cache_ttl": 300,
                        "connection_pool_size": 20,
                        "compression_enabled": True
                    },
                    expected_impact={MetricType.LATENCY: -0.05},  # Reduce by 50ms
                    priority=target.priority
                )
        
        elif target.metric_type == MetricType.ERROR_RATE:
            if current_value > target.target_value:
                # High error rate - need to improve reliability
                return OptimizationAction(
                    action_id=action_id,
                    action_type="improve_reliability",
                    description="Improve system reliability through circuit breakers and retries",
                    parameters={
                        "circuit_breaker_enabled": True,
                        "retry_max_attempts": 3,
                        "timeout_seconds": 10,
                        "health_check_interval": 30
                    },
                    expected_impact={MetricType.ERROR_RATE: -0.005},  # Reduce by 0.5%
                    priority=target.priority
                )
        
        elif target.metric_type == MetricType.CPU_UTILIZATION:
            if current_value > target.target_value:
                # High CPU - optimize or scale
                return OptimizationAction(
                    action_id=action_id,
                    action_type="optimize_cpu",
                    description="Optimize CPU usage through algorithm improvements and resource allocation",
                    parameters={
                        "worker_threads": min(8, self.auto_scaler.current_instances * 2),
                        "gc_optimization": True,
                        "background_processing": True
                    },
                    expected_impact={MetricType.CPU_UTILIZATION: -0.1},  # Reduce by 10%
                    priority=target.priority
                )
        
        elif target.metric_type == MetricType.MEMORY_UTILIZATION:
            if current_value > target.target_value:
                # High memory usage - optimize
                return OptimizationAction(
                    action_id=action_id,
                    action_type="optimize_memory",
                    description="Optimize memory usage through garbage collection and memory pooling",
                    parameters={
                        "memory_pool_enabled": True,
                        "gc_frequency": "aggressive",
                        "object_pooling": True,
                        "memory_limit_mb": 512
                    },
                    expected_impact={MetricType.MEMORY_UTILIZATION: -0.1},  # Reduce by 10%
                    priority=target.priority
                )
        
        elif target.metric_type == MetricType.THROUGHPUT:
            if current_value < target.target_value:
                # Low throughput - need to increase
                return OptimizationAction(
                    action_id=action_id,
                    action_type="increase_throughput",
                    description="Increase system throughput through parallelization and batching",
                    parameters={
                        "batch_size": 100,
                        "parallel_workers": 4,
                        "async_processing": True,
                        "buffer_size": 1000
                    },
                    expected_impact={MetricType.THROUGHPUT: 100.0},  # Increase by 100 req/s
                    priority=target.priority
                )
        
        return None
    
    async def apply_optimization_action(self, action: OptimizationAction) -> OptimizationResult:
        """Apply an optimization action and measure results."""
        logger.info(f"Applying optimization action: {action.action_type}")
        
        # Get metrics before optimization
        metrics_before = self._get_current_metrics()
        
        try:
            # Simulate applying the optimization
            success = await self._execute_optimization_action(action)
            
            if success:
                # Wait for metrics to stabilize
                await asyncio.sleep(30)  # 30 second stabilization period
                
                # Get metrics after optimization
                metrics_after = self._get_current_metrics()
                
                # Calculate actual impact
                actual_impact = {}
                for metric_type in metrics_before:
                    if metric_type in metrics_after:
                        actual_impact[metric_type] = metrics_after[metric_type] - metrics_before[metric_type]
                
                result = OptimizationResult(
                    action_id=action.action_id,
                    success=True,
                    applied_at=datetime.now(),
                    metrics_before=metrics_before,
                    metrics_after=metrics_after,
                    actual_impact=actual_impact
                )
                
                logger.info(f"Optimization action {action.action_type} applied successfully")
                
            else:
                result = OptimizationResult(
                    action_id=action.action_id,
                    success=False,
                    applied_at=datetime.now(),
                    metrics_before=metrics_before,
                    metrics_after={},
                    actual_impact={},
                    error_message="Failed to apply optimization action"
                )
                
                logger.error(f"Failed to apply optimization action: {action.action_type}")
            
            self.optimization_results.append(result)
            return result
            
        except Exception as e:
            result = OptimizationResult(
                action_id=action.action_id,
                success=False,
                applied_at=datetime.now(),
                metrics_before=metrics_before,
                metrics_after={},
                actual_impact={},
                error_message=str(e)
            )
            
            self.optimization_results.append(result)
            logger.error(f"Exception during optimization action {action.action_type}: {e}")
            return result
    
    async def _execute_optimization_action(self, action: OptimizationAction) -> bool:
        """Execute a specific optimization action."""
        try:
            # Simulate optimization actions
            if action.action_type == "reduce_latency":
                # Simulate latency reduction optimizations
                await asyncio.sleep(2)  # Simulation delay
                return True
                
            elif action.action_type == "improve_reliability":
                # Simulate reliability improvements
                await asyncio.sleep(3)  # Simulation delay
                return True
                
            elif action.action_type == "optimize_cpu":
                # Simulate CPU optimizations
                await asyncio.sleep(2)  # Simulation delay
                return True
                
            elif action.action_type == "optimize_memory":
                # Simulate memory optimizations
                await asyncio.sleep(2)  # Simulation delay
                return True
                
            elif action.action_type == "increase_throughput":
                # Simulate throughput improvements
                await asyncio.sleep(3)  # Simulation delay
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to execute optimization action {action.action_type}: {e}")
            return False
    
    async def _optimization_loop(self) -> None:
        """Main optimization loop."""
        while self.optimization_active:
            try:
                # Generate optimization actions
                actions = self.generate_optimization_actions()
                
                if actions:
                    # Apply the highest priority action
                    top_action = actions[0]
                    await self.apply_optimization_action(top_action)
                
                # Wait before next optimization cycle
                await asyncio.sleep(60)  # 1 minute cycle
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(30)  # Retry in 30 seconds
    
    async def _metrics_collection_loop(self) -> None:
        """Simulate metrics collection."""
        while self.monitoring_active:
            try:
                # Simulate collecting metrics from various sources
                current_time = datetime.now()
                
                # Simulate server metrics
                for i, server in enumerate(self.load_balancer.servers):
                    server_id = server["id"]
                    
                    # Generate realistic metrics with some randomness
                    base_cpu = 0.4 + i * 0.1  # Different baseline for each server
                    cpu_utilization = max(0.1, min(0.95, base_cpu + random.gauss(0, 0.1)))
                    
                    base_memory = 0.5 + i * 0.05
                    memory_utilization = max(0.2, min(0.9, base_memory + random.gauss(0, 0.05)))
                    
                    base_latency = 0.08 + i * 0.02
                    latency = max(0.01, base_latency + random.gauss(0, 0.02))
                    
                    error_rate = max(0.0, min(0.1, random.gauss(0.005, 0.002)))
                    
                    # Add metrics
                    metrics = [
                        ProductionMetric(MetricType.CPU_UTILIZATION, cpu_utilization, current_time, server_id),
                        ProductionMetric(MetricType.MEMORY_UTILIZATION, memory_utilization, current_time, server_id),
                        ProductionMetric(MetricType.LATENCY, latency, current_time, server_id),
                        ProductionMetric(MetricType.ERROR_RATE, error_rate, current_time, server_id),
                    ]
                    
                    for metric in metrics:
                        self.add_metric(metric)
                
                # Simulate system-wide metrics
                throughput = max(100, 800 + random.gauss(0, 100))
                self.add_metric(ProductionMetric(MetricType.THROUGHPUT, throughput, current_time, "system"))
                
                await asyncio.sleep(30)  # Collect metrics every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(30)
    
    async def _auto_scaling_loop(self) -> None:
        """Auto-scaling decision loop."""
        while self.optimization_active:
            try:
                current_metrics = self._get_current_metrics()
                
                if current_metrics:
                    # Evaluate scaling decision
                    scaling_decision = self.auto_scaler.evaluate_scaling_decision(current_metrics)
                    
                    if scaling_decision:
                        success = self.auto_scaler.execute_scaling_action(scaling_decision)
                        if success:
                            logger.info(f"Auto-scaling action executed: {scaling_decision}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in auto-scaling loop: {e}")
                await asyncio.sleep(60)
    
    async def _load_balancing_optimization_loop(self) -> None:
        """Load balancing optimization loop."""
        while self.optimization_active:
            try:
                # Get recent metrics for load balancing optimization
                recent_metrics = list(self.metrics_history)[-20:] if self.metrics_history else []
                
                # Optimize routing strategy
                new_strategy = self.load_balancer.optimize_routing_strategy(recent_metrics)
                
                if new_strategy != self.load_balancer.current_strategy:
                    logger.info(f"Load balancing strategy changed to: {new_strategy}")
                
                await asyncio.sleep(120)  # Optimize every 2 minutes
                
            except Exception as e:
                logger.error(f"Error in load balancing optimization: {e}")
                await asyncio.sleep(120)
    
    def get_optimization_dashboard_data(self) -> Dict[str, Any]:
        """Get data for optimization dashboard."""
        current_metrics = self._get_current_metrics()
        
        # Target achievement status
        target_status = []
        for target in self.optimization_targets:
            if target.metric_type in current_metrics:
                current_value = current_metrics[target.metric_type]
                within_tolerance = target.is_within_tolerance(current_value)
                
                target_status.append({
                    "metric_type": target.metric_type.value,
                    "target_value": target.target_value,
                    "current_value": current_value,
                    "within_tolerance": within_tolerance,
                    "priority": target.priority
                })
        
        # Recent optimization results
        recent_results = self.optimization_results[-10:] if self.optimization_results else []
        
        # System status
        system_status = {
            "current_instances": self.auto_scaler.current_instances,
            "load_balancing_strategy": self.load_balancer.current_strategy,
            "optimization_active": self.optimization_active,
            "monitoring_active": self.monitoring_active
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "current_metrics": {k.value: v for k, v in current_metrics.items()},
            "target_status": target_status,
            "recent_optimization_results": [r.to_dict() for r in recent_results],
            "system_status": system_status,
            "server_health": {
                server["id"]: server["health_score"]
                for server in self.load_balancer.servers
            },
            "scaling_history": self.auto_scaler.scaling_history[-5:],  # Last 5 scaling actions
            "optimization_strategy": self.current_strategy.value
        }
    
    async def save_optimization_data(self) -> None:
        """Save optimization data to persistent storage."""
        try:
            optimization_data = {
                "timestamp": datetime.now().isoformat(),
                "metrics_history": [m.to_dict() for m in list(self.metrics_history)[-100:]],  # Last 100 metrics
                "optimization_results": [r.to_dict() for r in self.optimization_results],
                "scaling_history": self.auto_scaler.scaling_history,
                "system_configuration": {
                    "optimization_targets": [
                        {
                            "metric_type": t.metric_type.value,
                            "target_value": t.target_value,
                            "priority": t.priority,
                            "tolerance": t.tolerance
                        }
                        for t in self.optimization_targets
                    ],
                    "current_strategy": self.current_strategy.value,
                    "current_instances": self.auto_scaler.current_instances
                }
            }
            
            optimization_file = self.project_root / "optimization_data.json"
            with open(optimization_file, 'w') as f:
                json.dump(optimization_data, f, indent=2)
            
            logger.info(f"Optimization data saved to {optimization_file}")
            
        except Exception as e:
            logger.error(f"Failed to save optimization data: {e}")


async def main():
    """Main function for advanced production optimizer."""
    logger.info("Starting Advanced Production Optimizer...")
    
    optimizer = AdvancedProductionOptimizer()
    
    # Start optimization
    await optimizer.start_optimization()
    
    print("\n" + "="*80)
    print("🚀 ADVANCED PRODUCTION OPTIMIZER")
    print("="*80)
    
    # Let it run for a demo period
    demo_duration = 30  # 30 seconds demo
    
    print(f"Running optimization for {demo_duration} seconds...")
    await asyncio.sleep(demo_duration)
    
    # Get dashboard data
    dashboard_data = optimizer.get_optimization_dashboard_data()
    
    print(f"\n📊 Optimization Dashboard Summary:")
    print(f"   • Current Instances: {dashboard_data['system_status']['current_instances']}")
    print(f"   • Load Balancing Strategy: {dashboard_data['system_status']['load_balancing_strategy']}")
    print(f"   • Optimization Active: {dashboard_data['system_status']['optimization_active']}")
    
    if dashboard_data['current_metrics']:
        print(f"   • Current Metrics:")
        for metric, value in dashboard_data['current_metrics'].items():
            print(f"     - {metric}: {value:.3f}")
    
    if dashboard_data['target_status']:
        targets_met = sum(1 for t in dashboard_data['target_status'] if t['within_tolerance'])
        total_targets = len(dashboard_data['target_status'])
        print(f"   • Targets Met: {targets_met}/{total_targets}")
    
    if dashboard_data['recent_optimization_results']:
        successful_optimizations = sum(1 for r in dashboard_data['recent_optimization_results'] if r['success'])
        total_optimizations = len(dashboard_data['recent_optimization_results'])
        print(f"   • Successful Optimizations: {successful_optimizations}/{total_optimizations}")
    
    # Save optimization data
    await optimizer.save_optimization_data()
    print(f"\n💾 Optimization data saved")
    
    print("\n" + "="*80)
    
    # Stop optimization
    await optimizer.stop_optimization()
    
    return dashboard_data


if __name__ == "__main__":
    asyncio.run(main())