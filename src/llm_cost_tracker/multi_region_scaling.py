"""
Multi-Region Auto-Scaling System

Advanced horizontal and vertical scaling with:
- Multi-region deployment orchestration
- Intelligent load balancing with geo-routing
- Auto-scaling based on quantum metrics and predictions
- Service mesh integration with health monitoring
- Cost-optimized resource allocation across regions
- Disaster recovery and failover capabilities
"""

import asyncio
import json
import logging
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict, deque
import aiohttp
import psutil

logger = logging.getLogger(__name__)


class RegionStatus(Enum):
    """Status of deployment regions."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


class ScalingDirection(Enum):
    """Scaling direction options."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


@dataclass
class RegionMetrics:
    """Metrics for a specific region."""
    region_id: str
    cpu_usage: float
    memory_usage: float
    request_count: int
    response_time_p95: float
    error_rate: float
    active_connections: int
    quantum_tasks_pending: int
    quantum_tasks_executing: int
    cost_per_hour: float
    latency_to_regions: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def load_score(self) -> float:
        """Calculate overall load score for the region."""
        # Weighted combination of various metrics
        cpu_weight = self.cpu_usage * 0.3
        memory_weight = self.memory_usage * 0.3
        response_time_weight = min(100, self.response_time_p95 / 10) * 0.2  # Normalize to 0-100
        error_weight = self.error_rate * 0.1
        connection_weight = min(100, self.active_connections / 100) * 0.1
        
        return cpu_weight + memory_weight + response_time_weight + error_weight + connection_weight
    
    @property
    def health_score(self) -> float:
        """Calculate health score (0-100, higher is better)."""
        # Invert load score and normalize
        base_health = max(0, 100 - self.load_score)
        
        # Apply penalties for high error rates or extreme resource usage
        if self.error_rate > 5.0:
            base_health *= 0.5  # 50% penalty for high errors
        if self.cpu_usage > 90 or self.memory_usage > 90:
            base_health *= 0.7  # 30% penalty for resource exhaustion
        
        return max(0, min(100, base_health))


@dataclass
class ScalingDecision:
    """Auto-scaling decision and rationale."""
    region_id: str
    direction: ScalingDirection
    current_instances: int
    target_instances: int
    confidence: float
    reasoning: List[str]
    cost_impact: float
    quantum_workload_factor: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def instance_delta(self) -> int:
        """Calculate the change in instance count."""
        return self.target_instances - self.current_instances


class QuantumWorkloadPredictor:
    """Predictive model for quantum task workloads."""
    
    def __init__(self, history_window: int = 1000):
        self.history_window = history_window
        self.workload_history: deque = deque(maxlen=history_window)
        self.pattern_cache: Dict[str, List[float]] = {}
        self.last_prediction: Optional[Dict[str, Any]] = None
        
    def record_workload(self, quantum_metrics: Dict[str, Any]):
        """Record quantum workload data."""
        workload_point = {
            'timestamp': datetime.now(),
            'pending_tasks': quantum_metrics.get('pending_tasks', 0),
            'executing_tasks': quantum_metrics.get('executing_tasks', 0),
            'completed_tasks': quantum_metrics.get('completed_tasks', 0),
            'avg_task_complexity': quantum_metrics.get('avg_task_complexity', 1.0),
            'pareto_front_size': quantum_metrics.get('pareto_front_size', 1),
            'convergence_rate': quantum_metrics.get('convergence_rate', 0.5),
            'resource_utilization': quantum_metrics.get('resource_utilization', 0.5)
        }
        
        self.workload_history.append(workload_point)
    
    def predict_workload(self, forecast_minutes: int = 30) -> Dict[str, Any]:
        """Predict future quantum workload."""
        if len(self.workload_history) < 10:
            # Not enough data for prediction
            return {
                'predicted_load': 1.0,
                'confidence': 0.1,
                'reasoning': ['insufficient_historical_data'],
                'recommended_scaling_factor': 1.0
            }
        
        recent_data = list(self.workload_history)[-100:]  # Last 100 points
        
        # Analyze trends
        pending_tasks = [point['pending_tasks'] for point in recent_data]
        executing_tasks = [point['executing_tasks'] for point in recent_data]
        complexity_scores = [point['avg_task_complexity'] for point in recent_data]
        
        # Calculate trends
        pending_trend = self._calculate_trend(pending_tasks)
        execution_trend = self._calculate_trend(executing_tasks)
        complexity_trend = self._calculate_trend(complexity_scores)
        
        # Detect patterns (daily, weekly cycles)
        time_patterns = self._detect_time_patterns(recent_data)
        
        # Calculate predicted load
        base_load = statistics.mean(pending_tasks[-10:]) if pending_tasks else 1.0
        
        # Apply trend adjustments
        trend_adjustment = 1.0
        if pending_trend > 0.1:  # 10% increase trend
            trend_adjustment = 1.3
        elif pending_trend < -0.1:  # 10% decrease trend
            trend_adjustment = 0.8
        
        # Apply complexity adjustment
        complexity_factor = statistics.mean(complexity_scores[-5:]) if complexity_scores else 1.0
        
        # Apply time pattern adjustment
        pattern_adjustment = time_patterns.get('current_factor', 1.0)
        
        predicted_load = base_load * trend_adjustment * complexity_factor * pattern_adjustment
        
        # Calculate confidence based on data consistency and amount
        data_consistency = 1.0 - (statistics.stdev(pending_tasks[-20:]) / statistics.mean(pending_tasks[-20:]) 
                                 if len(pending_tasks) >= 20 and statistics.mean(pending_tasks[-20:]) > 0 else 1.0)
        data_amount_factor = min(1.0, len(self.workload_history) / 100)
        confidence = (data_consistency * 0.6 + data_amount_factor * 0.4)
        
        # Recommend scaling factor
        if predicted_load > statistics.mean(pending_tasks[-10:]) * 1.5:
            recommended_scaling = 1.5
        elif predicted_load < statistics.mean(pending_tasks[-10:]) * 0.7:
            recommended_scaling = 0.8
        else:
            recommended_scaling = 1.0
        
        reasoning = []
        if pending_trend > 0.1:
            reasoning.append(f"increasing_workload_trend_{pending_trend:.2f}")
        if complexity_factor > 1.3:
            reasoning.append("high_task_complexity")
        if pattern_adjustment > 1.2:
            reasoning.append("peak_time_pattern")
        
        prediction = {
            'predicted_load': predicted_load,
            'confidence': confidence,
            'reasoning': reasoning or ['stable_workload'],
            'recommended_scaling_factor': recommended_scaling,
            'trend_analysis': {
                'pending_trend': pending_trend,
                'execution_trend': execution_trend,
                'complexity_trend': complexity_trend
            },
            'pattern_analysis': time_patterns
        }
        
        self.last_prediction = prediction
        return prediction
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values (positive = increasing)."""
        if len(values) < 5:
            return 0.0
        
        # Simple linear regression slope
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        if n * x2_sum - x_sum * x_sum == 0:
            return 0.0
            
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        
        # Normalize by mean to get relative trend
        mean_value = statistics.mean(values) if values else 1.0
        if mean_value == 0:
            return 0.0
            
        return slope / mean_value
    
    def _detect_time_patterns(self, recent_data: List[Dict]) -> Dict[str, Any]:
        """Detect time-based patterns in workload."""
        if len(recent_data) < 24:  # Need at least 24 data points
            return {'current_factor': 1.0}
        
        current_time = datetime.now()
        current_hour = current_time.hour
        current_weekday = current_time.weekday()
        
        # Group data by hour of day
        hourly_loads = defaultdict(list)
        weekday_loads = defaultdict(list)
        
        for point in recent_data:
            timestamp = point['timestamp']
            load = point['pending_tasks'] + point['executing_tasks'] * 0.5
            
            hour = timestamp.hour
            weekday = timestamp.weekday()
            
            hourly_loads[hour].append(load)
            weekday_loads[weekday].append(load)
        
        # Calculate hour-based factor
        hour_averages = {hour: statistics.mean(loads) for hour, loads in hourly_loads.items() if loads}
        overall_average = statistics.mean([avg for avg in hour_averages.values()]) if hour_averages else 1.0
        
        current_hour_factor = 1.0
        if current_hour in hour_averages and overall_average > 0:
            current_hour_factor = hour_averages[current_hour] / overall_average
        
        # Calculate weekday-based factor
        weekday_averages = {day: statistics.mean(loads) for day, loads in weekday_loads.items() if loads}
        weekday_overall_average = statistics.mean([avg for avg in weekday_averages.values()]) if weekday_averages else 1.0
        
        current_weekday_factor = 1.0
        if current_weekday in weekday_averages and weekday_overall_average > 0:
            current_weekday_factor = weekday_averages[current_weekday] / weekday_overall_average
        
        # Combine factors (hour has more weight)
        combined_factor = current_hour_factor * 0.7 + current_weekday_factor * 0.3
        
        return {
            'current_factor': combined_factor,
            'hour_factor': current_hour_factor,
            'weekday_factor': current_weekday_factor,
            'hour_averages': dict(hour_averages),
            'weekday_averages': dict(weekday_averages)
        }


class MultiRegionAutoScaler:
    """
    Advanced multi-region auto-scaling system with quantum workload awareness.
    """
    
    def __init__(self):
        self.regions: Dict[str, RegionMetrics] = {}
        self.scaling_history: List[ScalingDecision] = []
        self.workload_predictor = QuantumWorkloadPredictor()
        
        # Scaling configuration
        self.scaling_config = {
            'min_instances_per_region': 1,
            'max_instances_per_region': 20,
            'target_cpu_utilization': 70.0,
            'target_memory_utilization': 75.0,
            'target_response_time_ms': 500.0,
            'scale_up_threshold': 80.0,
            'scale_down_threshold': 50.0,
            'cooldown_period_minutes': 5,
            'prediction_weight': 0.3,  # Weight given to predictive scaling
            'cost_optimization_weight': 0.2
        }
        
        # Region definitions with costs and capabilities
        self.region_definitions = {
            'us-east-1': {
                'name': 'US East (N. Virginia)',
                'cost_per_instance_hour': 0.096,
                'quantum_processing_capability': 1.0,
                'max_instances': 50,
                'preferred_for_workloads': ['general', 'quantum']
            },
            'us-west-2': {
                'name': 'US West (Oregon)', 
                'cost_per_instance_hour': 0.096,
                'quantum_processing_capability': 1.0,
                'max_instances': 30,
                'preferred_for_workloads': ['general', 'quantum']
            },
            'eu-west-1': {
                'name': 'Europe (Ireland)',
                'cost_per_instance_hour': 0.108,
                'quantum_processing_capability': 0.9,
                'max_instances': 25,
                'preferred_for_workloads': ['general']
            },
            'ap-southeast-1': {
                'name': 'Asia Pacific (Singapore)',
                'cost_per_instance_hour': 0.116,
                'quantum_processing_capability': 0.8,
                'max_instances': 20,
                'preferred_for_workloads': ['general']
            }
        }
        
        # Load balancing weights
        self.load_balancing_weights: Dict[str, float] = {}
        self._update_load_balancing_weights()
        
        # Background tasks
        self._scaling_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_running = False
        
    async def start_auto_scaling(self):
        """Start the auto-scaling system."""
        if self._is_running:
            return
            
        self._is_running = True
        self._scaling_task = asyncio.create_task(self._scaling_loop())
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Multi-region auto-scaling system started")
    
    async def stop_auto_scaling(self):
        """Stop the auto-scaling system."""
        self._is_running = False
        
        tasks = [self._scaling_task, self._monitoring_task]
        for task in tasks:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Multi-region auto-scaling system stopped")
    
    def update_region_metrics(self, region_id: str, metrics: Dict[str, Any]):
        """Update metrics for a specific region."""
        region_metrics = RegionMetrics(
            region_id=region_id,
            cpu_usage=metrics.get('cpu_usage', 0.0),
            memory_usage=metrics.get('memory_usage', 0.0),
            request_count=metrics.get('request_count', 0),
            response_time_p95=metrics.get('response_time_p95', 0.0),
            error_rate=metrics.get('error_rate', 0.0),
            active_connections=metrics.get('active_connections', 0),
            quantum_tasks_pending=metrics.get('quantum_tasks_pending', 0),
            quantum_tasks_executing=metrics.get('quantum_tasks_executing', 0),
            cost_per_hour=self.region_definitions.get(region_id, {}).get('cost_per_instance_hour', 0.1)
        )
        
        # Update inter-region latency if provided
        if 'latency_to_regions' in metrics:
            region_metrics.latency_to_regions = metrics['latency_to_regions']
        
        self.regions[region_id] = region_metrics
        
        # Record quantum workload data
        quantum_metrics = {
            'pending_tasks': metrics.get('quantum_tasks_pending', 0),
            'executing_tasks': metrics.get('quantum_tasks_executing', 0),
            'avg_task_complexity': metrics.get('avg_task_complexity', 1.0),
            'pareto_front_size': metrics.get('pareto_front_size', 1),
            'resource_utilization': (metrics.get('cpu_usage', 0) + metrics.get('memory_usage', 0)) / 200.0
        }
        self.workload_predictor.record_workload(quantum_metrics)
    
    async def make_scaling_decisions(self) -> List[ScalingDecision]:
        """Make intelligent scaling decisions across all regions."""
        decisions = []
        
        if not self.regions:
            return decisions
        
        # Get workload prediction
        workload_prediction = self.workload_predictor.predict_workload()
        
        for region_id, metrics in self.regions.items():
            decision = await self._analyze_region_scaling(region_id, metrics, workload_prediction)
            if decision.direction != ScalingDirection.STABLE:
                decisions.append(decision)
        
        # Apply global optimizations
        decisions = self._optimize_global_scaling(decisions, workload_prediction)
        
        return decisions
    
    async def _analyze_region_scaling(self, 
                                    region_id: str, 
                                    metrics: RegionMetrics,
                                    workload_prediction: Dict[str, Any]) -> ScalingDecision:
        """Analyze scaling needs for a specific region."""
        current_instances = await self._get_current_instance_count(region_id)
        
        reasoning = []
        confidence = 0.5
        target_instances = current_instances
        
        # Analyze current load
        load_score = metrics.load_score
        health_score = metrics.health_score
        
        # Check for scaling triggers
        scale_up_needed = False
        scale_down_needed = False
        
        # CPU/Memory based scaling
        if metrics.cpu_usage > self.scaling_config['scale_up_threshold']:
            scale_up_needed = True
            reasoning.append(f"cpu_high_{metrics.cpu_usage:.1f}%")
            confidence += 0.2
        elif metrics.cpu_usage < self.scaling_config['scale_down_threshold']:
            scale_down_needed = True
            reasoning.append(f"cpu_low_{metrics.cpu_usage:.1f}%")
            confidence += 0.1
        
        if metrics.memory_usage > self.scaling_config['scale_up_threshold']:
            scale_up_needed = True
            reasoning.append(f"memory_high_{metrics.memory_usage:.1f}%")
            confidence += 0.2
        elif metrics.memory_usage < self.scaling_config['scale_down_threshold']:
            scale_down_needed = True
            reasoning.append(f"memory_low_{metrics.memory_usage:.1f}%")
            confidence += 0.1
        
        # Response time based scaling
        if metrics.response_time_p95 > self.scaling_config['target_response_time_ms']:
            scale_up_needed = True
            reasoning.append(f"response_time_high_{metrics.response_time_p95:.0f}ms")
            confidence += 0.3
        
        # Error rate based scaling
        if metrics.error_rate > 5.0:
            scale_up_needed = True
            reasoning.append(f"error_rate_high_{metrics.error_rate:.1f}%")
            confidence += 0.2
        
        # Quantum workload based scaling
        quantum_load_factor = (metrics.quantum_tasks_pending + metrics.quantum_tasks_executing) / 10.0
        if quantum_load_factor > 1.5:
            scale_up_needed = True
            reasoning.append(f"quantum_load_high_{quantum_load_factor:.1f}")
            confidence += 0.2
        elif quantum_load_factor < 0.3 and current_instances > self.scaling_config['min_instances_per_region']:
            scale_down_needed = True
            reasoning.append(f"quantum_load_low_{quantum_load_factor:.1f}")
            confidence += 0.1
        
        # Predictive scaling based on workload prediction
        if workload_prediction['confidence'] > 0.6:
            predicted_scaling_factor = workload_prediction['recommended_scaling_factor']
            prediction_weight = self.scaling_config['prediction_weight']
            
            if predicted_scaling_factor > 1.2:
                scale_up_needed = True
                reasoning.append(f"predicted_increase_{predicted_scaling_factor:.2f}")
                confidence += prediction_weight
            elif predicted_scaling_factor < 0.8:
                scale_down_needed = True
                reasoning.append(f"predicted_decrease_{predicted_scaling_factor:.2f}")
                confidence += prediction_weight * 0.5  # Less confidence in scale down predictions
        
        # Calculate target instances
        if scale_up_needed and not scale_down_needed:
            # Scale up logic
            scale_factor = 1.5  # Default 50% increase
            
            if metrics.cpu_usage > 90 or metrics.memory_usage > 90:
                scale_factor = 2.0  # Double for critical situations
            elif metrics.response_time_p95 > self.scaling_config['target_response_time_ms'] * 2:
                scale_factor = 1.8  # Aggressive scaling for poor response times
            
            target_instances = min(
                self.scaling_config['max_instances_per_region'],
                self.region_definitions.get(region_id, {}).get('max_instances', 20),
                max(current_instances + 1, int(current_instances * scale_factor))
            )
            
        elif scale_down_needed and not scale_up_needed:
            # Scale down logic (more conservative)
            scale_factor = 0.8  # 20% decrease
            
            if health_score > 80 and load_score < 30:
                scale_factor = 0.7  # More aggressive scale down if very healthy
            
            target_instances = max(
                self.scaling_config['min_instances_per_region'],
                int(current_instances * scale_factor)
            )
        
        # Determine scaling direction
        if target_instances > current_instances:
            direction = ScalingDirection.UP
        elif target_instances < current_instances:
            direction = ScalingDirection.DOWN
        else:
            direction = ScalingDirection.STABLE
            reasoning.append("load_balanced")
        
        # Check cooldown period
        if await self._is_in_cooldown_period(region_id):
            direction = ScalingDirection.STABLE
            reasoning.append("cooldown_period")
            confidence *= 0.3
        
        # Calculate cost impact
        region_def = self.region_definitions.get(region_id, {})
        cost_per_hour = region_def.get('cost_per_instance_hour', 0.1)
        cost_impact = (target_instances - current_instances) * cost_per_hour
        
        return ScalingDecision(
            region_id=region_id,
            direction=direction,
            current_instances=current_instances,
            target_instances=target_instances,
            confidence=min(1.0, confidence),
            reasoning=reasoning,
            cost_impact=cost_impact,
            quantum_workload_factor=quantum_load_factor
        )
    
    def _optimize_global_scaling(self, 
                               decisions: List[ScalingDecision],
                               workload_prediction: Dict[str, Any]) -> List[ScalingDecision]:
        """Optimize scaling decisions globally across regions."""
        if not decisions:
            return decisions
        
        optimized_decisions = []
        
        # Calculate total cost impact
        total_cost_impact = sum(decision.cost_impact for decision in decisions)
        
        # If total cost impact is too high, prioritize by confidence and necessity
        cost_threshold = 10.0  # $10/hour threshold
        if total_cost_impact > cost_threshold:
            # Sort by priority: critical scale-ups first, then scale-downs
            scale_ups = [d for d in decisions if d.direction == ScalingDirection.UP]
            scale_downs = [d for d in decisions if d.direction == ScalingDirection.DOWN]
            
            # Prioritize scale-ups by confidence
            scale_ups.sort(key=lambda d: d.confidence, reverse=True)
            
            # Limit scale-ups to most critical
            current_cost = 0
            for decision in scale_ups:
                if current_cost + decision.cost_impact <= cost_threshold * 0.8:  # 80% of budget for scale-ups
                    optimized_decisions.append(decision)
                    current_cost += decision.cost_impact
                else:
                    # Reduce scale-up to fit budget
                    remaining_budget = cost_threshold * 0.8 - current_cost
                    region_def = self.region_definitions.get(decision.region_id, {})
                    cost_per_instance = region_def.get('cost_per_instance_hour', 0.1)
                    
                    if remaining_budget > cost_per_instance:
                        affordable_instances = int(remaining_budget / cost_per_instance)
                        decision.target_instances = decision.current_instances + affordable_instances
                        decision.cost_impact = affordable_instances * cost_per_instance
                        decision.reasoning.append("budget_constrained")
                        optimized_decisions.append(decision)
                        break
            
            # Add scale-downs to recover budget
            for decision in scale_downs:
                optimized_decisions.append(decision)
        
        else:
            # Budget allows all decisions
            optimized_decisions = decisions
        
        # Apply region balancing
        optimized_decisions = self._balance_regions(optimized_decisions)
        
        return optimized_decisions
    
    def _balance_regions(self, decisions: List[ScalingDecision]) -> List[ScalingDecision]:
        """Balance load across regions for optimal performance."""
        if len(decisions) < 2:
            return decisions
        
        # Group decisions by direction
        scale_ups = {d.region_id: d for d in decisions if d.direction == ScalingDirection.UP}
        scale_downs = {d.region_id: d for d in decisions if d.direction == ScalingDirection.DOWN}
        
        # If we have both scale-ups and scale-downs, consider rebalancing
        if scale_ups and scale_downs:
            # Find opportunities to redirect load instead of scaling
            for up_region, up_decision in scale_ups.items():
                for down_region, down_decision in scale_downs.items():
                    # Check if regions can handle each other's load
                    up_metrics = self.regions.get(up_region)
                    down_metrics = self.regions.get(down_region)
                    
                    if up_metrics and down_metrics:
                        # Check latency between regions
                        latency = up_metrics.latency_to_regions.get(down_region, float('inf'))
                        
                        # If latency is acceptable (< 100ms), consider load balancing
                        if latency < 100:
                            # Reduce scale-up and scale-down
                            reduction_factor = 0.5
                            up_decision.target_instances = int(
                                up_decision.current_instances + 
                                (up_decision.target_instances - up_decision.current_instances) * reduction_factor
                            )
                            down_decision.target_instances = max(
                                self.scaling_config['min_instances_per_region'],
                                int(down_decision.target_instances + 
                                   (down_decision.current_instances - down_decision.target_instances) * reduction_factor)
                            )
                            
                            up_decision.reasoning.append(f"load_balance_with_{down_region}")
                            down_decision.reasoning.append(f"load_balance_with_{up_region}")
        
        return decisions
    
    async def execute_scaling_decisions(self, decisions: List[ScalingDecision]) -> Dict[str, Any]:
        """Execute scaling decisions across regions."""
        execution_results = {
            'successful_scalings': 0,
            'failed_scalings': 0,
            'total_cost_impact': 0.0,
            'execution_details': []
        }
        
        for decision in decisions:
            try:
                # Execute the scaling operation
                success = await self._execute_region_scaling(decision)
                
                if success:
                    execution_results['successful_scalings'] += 1
                    execution_results['total_cost_impact'] += decision.cost_impact
                    
                    # Record in scaling history
                    self.scaling_history.append(decision)
                    
                    # Update load balancing weights
                    self._update_load_balancing_weights()
                    
                    logger.info(f"Scaling executed: {decision.region_id} "
                              f"{decision.direction.value} to {decision.target_instances} instances")
                else:
                    execution_results['failed_scalings'] += 1
                    logger.error(f"Scaling failed: {decision.region_id}")
                
                execution_results['execution_details'].append({
                    'region_id': decision.region_id,
                    'direction': decision.direction.value,
                    'target_instances': decision.target_instances,
                    'success': success,
                    'cost_impact': decision.cost_impact
                })
                
            except Exception as e:
                execution_results['failed_scalings'] += 1
                logger.error(f"Scaling execution error for {decision.region_id}: {e}")
        
        # Clean up old scaling history
        cutoff_time = datetime.now() - timedelta(days=7)
        self.scaling_history = [
            decision for decision in self.scaling_history
            if decision.timestamp > cutoff_time
        ]
        
        return execution_results
    
    async def _execute_region_scaling(self, decision: ScalingDecision) -> bool:
        """Execute scaling for a specific region (mock implementation)."""
        # In a real implementation, this would interact with cloud APIs
        # For now, we'll simulate the scaling operation
        
        region_id = decision.region_id
        target_instances = decision.target_instances
        
        # Simulate API call delay
        await asyncio.sleep(random.uniform(1.0, 3.0))
        
        # Simulate 95% success rate
        if random.random() < 0.95:
            logger.info(f"Simulated scaling: {region_id} to {target_instances} instances")
            return True
        else:
            logger.error(f"Simulated scaling failure: {region_id}")
            return False
    
    async def _get_current_instance_count(self, region_id: str) -> int:
        """Get current instance count for a region (mock implementation)."""
        # In real implementation, this would query cloud APIs
        # For simulation, use a deterministic count based on region load
        metrics = self.regions.get(region_id)
        if not metrics:
            return self.scaling_config['min_instances_per_region']
        
        # Simulate instance count based on load score
        base_instances = self.scaling_config['min_instances_per_region']
        load_based_instances = int(metrics.load_score / 20)  # Rough approximation
        
        return max(base_instances, min(self.scaling_config['max_instances_per_region'], 
                                     base_instances + load_based_instances))
    
    async def _is_in_cooldown_period(self, region_id: str) -> bool:
        """Check if region is in cooldown period."""
        cooldown_duration = timedelta(minutes=self.scaling_config['cooldown_period_minutes'])
        cutoff_time = datetime.now() - cooldown_duration
        
        # Check recent scaling history for this region
        recent_scalings = [
            decision for decision in self.scaling_history
            if decision.region_id == region_id and decision.timestamp > cutoff_time
        ]
        
        return len(recent_scalings) > 0
    
    def _update_load_balancing_weights(self):
        """Update load balancing weights based on region health."""
        if not self.regions:
            return
        
        total_health = sum(metrics.health_score for metrics in self.regions.values())
        
        if total_health == 0:
            # Equal weights if all regions unhealthy
            equal_weight = 1.0 / len(self.regions)
            self.load_balancing_weights = {region_id: equal_weight for region_id in self.regions.keys()}
        else:
            # Weight by health score
            for region_id, metrics in self.regions.items():
                self.load_balancing_weights[region_id] = metrics.health_score / total_health
        
        logger.debug(f"Updated load balancing weights: {self.load_balancing_weights}")
    
    async def _scaling_loop(self):
        """Main auto-scaling loop."""
        while self._is_running:
            try:
                # Make scaling decisions
                decisions = await self.make_scaling_decisions()
                
                if decisions:
                    logger.info(f"Auto-scaler found {len(decisions)} scaling opportunities")
                    
                    # Execute scaling decisions
                    results = await self.execute_scaling_decisions(decisions)
                    
                    logger.info(f"Scaling execution results: {results['successful_scalings']} successful, "
                              f"{results['failed_scalings']} failed, "
                              f"${results['total_cost_impact']:.2f}/hour cost impact")
                
                # Wait before next scaling cycle
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _monitoring_loop(self):
        """Monitoring loop for health checks and metrics collection."""
        while self._is_running:
            try:
                # Collect system metrics for all regions (simulated)
                for region_id in self.region_definitions.keys():
                    # Simulate collecting metrics
                    simulated_metrics = await self._simulate_region_metrics(region_id)
                    self.update_region_metrics(region_id, simulated_metrics)
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Wait before next monitoring cycle
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _simulate_region_metrics(self, region_id: str) -> Dict[str, Any]:
        """Simulate region metrics for demonstration."""
        # Generate realistic but random metrics
        base_load = random.uniform(30, 80)
        quantum_tasks = random.randint(0, 20)
        
        return {
            'cpu_usage': base_load + random.uniform(-10, 10),
            'memory_usage': base_load + random.uniform(-15, 15),
            'request_count': random.randint(100, 1000),
            'response_time_p95': random.uniform(200, 800),
            'error_rate': random.uniform(0, 3),
            'active_connections': random.randint(50, 500),
            'quantum_tasks_pending': quantum_tasks,
            'quantum_tasks_executing': random.randint(0, 10),
            'avg_task_complexity': random.uniform(0.5, 2.0),
            'pareto_front_size': random.randint(1, 10)
        }
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data."""
        # Remove old region metrics (keep last update only)
        # Remove old scaling history (older than 7 days)
        cutoff_time = datetime.now() - timedelta(days=7)
        self.scaling_history = [
            decision for decision in self.scaling_history
            if decision.timestamp > cutoff_time
        ]
    
    def get_scaling_summary(self) -> Dict[str, Any]:
        """Get comprehensive scaling system summary."""
        total_instances = sum(
            self._get_current_instance_count_sync(region_id) 
            for region_id in self.regions.keys()
        )
        
        total_cost_per_hour = sum(
            self._get_current_instance_count_sync(region_id) * 
            self.region_definitions.get(region_id, {}).get('cost_per_instance_hour', 0.1)
            for region_id in self.regions.keys()
        )
        
        recent_scalings = [
            decision for decision in self.scaling_history
            if decision.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        return {
            'total_regions': len(self.regions),
            'total_instances': total_instances,
            'total_cost_per_hour': total_cost_per_hour,
            'recent_scalings_24h': len(recent_scalings),
            'load_balancing_weights': self.load_balancing_weights.copy(),
            'region_health_scores': {
                region_id: metrics.health_score 
                for region_id, metrics in self.regions.items()
            },
            'workload_prediction': self.workload_predictor.last_prediction,
            'is_running': self._is_running,
            'last_updated': datetime.now().isoformat()
        }
    
    def _get_current_instance_count_sync(self, region_id: str) -> int:
        """Synchronous version of get_current_instance_count for summary."""
        metrics = self.regions.get(region_id)
        if not metrics:
            return self.scaling_config['min_instances_per_region']
        
        base_instances = self.scaling_config['min_instances_per_region']
        load_based_instances = int(metrics.load_score / 20)
        
        return max(base_instances, min(self.scaling_config['max_instances_per_region'], 
                                     base_instances + load_based_instances))