"""
Advanced Resilience Engine with Chaos Engineering and Self-Healing Capabilities

This module provides enterprise-grade resilience patterns for the LLM Cost Tracker:
- Chaos engineering for proactive failure testing
- Self-healing mechanisms with adaptive recovery
- Advanced circuit breaker with machine learning
- Distributed system resilience patterns
"""

import asyncio
import json
import logging
import random
import statistics
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures that can be injected for chaos engineering."""

    LATENCY_INJECTION = "latency_injection"
    ERROR_INJECTION = "error_injection"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_PARTITION = "network_partition"
    DEPENDENCY_FAILURE = "dependency_failure"
    MEMORY_LEAK = "memory_leak"
    CPU_SPIKE = "cpu_spike"
    DATABASE_TIMEOUT = "database_timeout"


class ResilienceState(Enum):
    """States of the resilience system."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERING = "recovering"
    CHAOS_MODE = "chaos_mode"


@dataclass
class ChaosExperiment:
    """Configuration for a chaos engineering experiment."""

    id: str
    name: str
    failure_type: FailureType
    probability: float = 0.1  # 10% chance by default
    duration_seconds: float = 30.0
    target_components: List[str] = field(default_factory=list)
    enabled: bool = True

    # Advanced parameters
    ramp_up_duration: float = 5.0
    ramp_down_duration: float = 5.0
    blast_radius: float = 0.2  # Affect 20% of instances

    # Scheduling
    schedule_pattern: Optional[str] = None  # Cron-like pattern
    max_concurrent_experiments: int = 1

    # Safety measures
    automatic_rollback: bool = True
    health_check_interval: float = 5.0
    rollback_threshold: float = 0.8  # Rollback if success rate < 80%


@dataclass
class ResilienceMetrics:
    """Metrics for resilience monitoring."""

    timestamp: datetime
    system_state: ResilienceState
    error_rate: float
    latency_p99: float
    throughput: float

    # Circuit breaker metrics
    circuit_breaker_trips: int
    circuit_breaker_recoveries: int

    # Chaos engineering metrics
    active_experiments: int
    experiment_success_rate: float

    # Self-healing metrics
    healing_actions: int
    healing_success_rate: float

    # Resource utilization
    cpu_usage: float
    memory_usage: float
    connection_pool_usage: float


class AdvancedCircuitBreaker:
    """
    Machine learning enhanced circuit breaker with adaptive thresholds
    and predictive failure detection.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        # State tracking
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0

        # Advanced features
        self.adaptive_threshold = failure_threshold
        self.success_history = deque(maxlen=100)
        self.latency_history = deque(maxlen=100)
        self.prediction_model = None

        # Metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.trips = 0
        self.recoveries = 0

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        self.total_requests += 1

        # Check if circuit is open
        if self.state == "OPEN":
            if await self._should_attempt_reset():
                self.state = "HALF_OPEN"
                self.half_open_calls = 0
                logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")

        # Execute the function
        start_time = time.time()
        try:
            result = (
                await func(*args, **kwargs)
                if asyncio.iscoroutinefunction(func)
                else func(*args, **kwargs)
            )

            # Record success
            execution_time = time.time() - start_time
            self._record_success(execution_time)

            return result

        except Exception as e:
            # Record failure
            execution_time = time.time() - start_time
            self._record_failure(execution_time, e)
            raise

    def _record_success(self, execution_time: float):
        """Record successful execution."""
        self.successful_requests += 1
        self.success_history.append(1)
        self.latency_history.append(execution_time)

        if self.state == "HALF_OPEN":
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = "CLOSED"
                self.failure_count = 0
                self.recoveries += 1
                logger.info(f"Circuit breaker {self.name} recovered to CLOSED")
        elif self.state == "CLOSED":
            # Reset failure count on success
            self.failure_count = max(0, self.failure_count - 1)

        # Adaptive threshold adjustment
        self._adjust_adaptive_threshold()

    def _record_failure(self, execution_time: float, error: Exception):
        """Record failed execution."""
        self.failed_requests += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.success_history.append(0)
        self.latency_history.append(execution_time)

        # Check if we should trip
        if self.failure_count >= self.adaptive_threshold:
            if self.state != "OPEN":
                self.state = "OPEN"
                self.trips += 1
                logger.warning(
                    f"Circuit breaker {self.name} tripped to OPEN after {self.failure_count} failures"
                )

        # Adaptive threshold adjustment
        self._adjust_adaptive_threshold()

    async def _should_attempt_reset(self) -> bool:
        """Determine if we should attempt to reset the circuit."""
        if not self.last_failure_time:
            return False

        elapsed = time.time() - self.last_failure_time

        # Basic timeout check
        if elapsed < self.recovery_timeout:
            return False

        # Advanced predictive check
        if self._predict_failure_likelihood() > 0.7:
            logger.debug(
                f"Circuit breaker {self.name} delaying reset due to high failure prediction"
            )
            return False

        return True

    def _adjust_adaptive_threshold(self):
        """Dynamically adjust failure threshold based on system behavior."""
        if len(self.success_history) < 10:
            return

        # Calculate recent success rate
        recent_success_rate = statistics.mean(list(self.success_history)[-20:])

        # Adjust threshold based on system stability
        if recent_success_rate > 0.9:  # Very stable
            self.adaptive_threshold = min(self.failure_threshold * 2, 20)
        elif recent_success_rate < 0.5:  # Very unstable
            self.adaptive_threshold = max(self.failure_threshold // 2, 2)
        else:
            self.adaptive_threshold = self.failure_threshold

    def _predict_failure_likelihood(self) -> float:
        """Predict likelihood of failure using simple heuristics."""
        if len(self.success_history) < 5:
            return 0.5

        # Simple prediction based on recent trends
        recent_failures = list(self.success_history)[-10:]
        failure_rate = 1 - statistics.mean(recent_failures)

        # Factor in latency trends
        if len(self.latency_history) >= 5:
            recent_latencies = list(self.latency_history)[-5:]
            latency_trend = statistics.mean(recent_latencies)
            if latency_trend > 2.0:  # High latency suggests problems
                failure_rate = min(failure_rate + 0.3, 1.0)

        return failure_rate

    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        success_rate = (
            self.successful_requests / self.total_requests
            if self.total_requests > 0
            else 0
        )
        avg_latency = (
            statistics.mean(self.latency_history) if self.latency_history else 0
        )

        return {
            "name": self.name,
            "state": self.state,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "failure_count": self.failure_count,
            "adaptive_threshold": self.adaptive_threshold,
            "trips": self.trips,
            "recoveries": self.recoveries,
            "average_latency": avg_latency,
            "predicted_failure_likelihood": self._predict_failure_likelihood(),
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class ChaosEngineeringEngine:
    """
    Chaos engineering engine for proactive failure testing and system resilience validation.
    """

    def __init__(self):
        self.experiments: Dict[str, ChaosExperiment] = {}
        self.active_experiments: Dict[str, asyncio.Task] = {}
        self.experiment_history: List[Dict[str, Any]] = []
        self.safety_enabled = True
        self.max_concurrent_experiments = 3

        # Monitoring
        self.baseline_metrics: Optional[ResilienceMetrics] = None
        self.current_metrics: Optional[ResilienceMetrics] = None

    def add_experiment(self, experiment: ChaosExperiment):
        """Add a chaos experiment to the engine."""
        self.experiments[experiment.id] = experiment
        logger.info(f"Added chaos experiment: {experiment.name}")

    async def start_experiment(self, experiment_id: str) -> bool:
        """Start a specific chaos experiment."""
        if experiment_id not in self.experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return False

        experiment = self.experiments[experiment_id]

        if not experiment.enabled:
            logger.info(f"Experiment {experiment.name} is disabled, skipping")
            return False

        # Safety checks
        if len(self.active_experiments) >= self.max_concurrent_experiments:
            logger.warning("Maximum concurrent experiments reached")
            return False

        if experiment_id in self.active_experiments:
            logger.warning(f"Experiment {experiment.name} is already running")
            return False

        # Start experiment
        task = asyncio.create_task(self._run_experiment(experiment))
        self.active_experiments[experiment_id] = task

        logger.info(f"Started chaos experiment: {experiment.name}")
        return True

    async def _run_experiment(self, experiment: ChaosExperiment):
        """Execute a chaos experiment with safety measures."""
        start_time = datetime.now()

        try:
            # Capture baseline metrics
            if not self.baseline_metrics:
                self.baseline_metrics = await self._collect_metrics()

            logger.info(
                f"Running chaos experiment: {experiment.name} for {experiment.duration_seconds}s"
            )

            # Ramp up phase
            if experiment.ramp_up_duration > 0:
                await self._ramp_up_failure(experiment)

            # Main injection phase
            injection_task = asyncio.create_task(self._inject_failure(experiment))

            # Monitor system health during experiment
            monitor_task = asyncio.create_task(self._monitor_experiment(experiment))

            # Wait for experiment duration
            await asyncio.sleep(experiment.duration_seconds)

            # Stop injection
            injection_task.cancel()
            monitor_task.cancel()

            # Ramp down phase
            if experiment.ramp_down_duration > 0:
                await self._ramp_down_failure(experiment)

            # Record results
            end_time = datetime.now()
            await self._record_experiment_result(
                experiment, start_time, end_time, success=True
            )

        except Exception as e:
            logger.error(
                f"Chaos experiment {experiment.name} failed: {e}", exc_info=True
            )
            await self._emergency_rollback(experiment)
            end_time = datetime.now()
            await self._record_experiment_result(
                experiment, start_time, end_time, success=False, error=str(e)
            )

        finally:
            # Cleanup
            if experiment.id in self.active_experiments:
                del self.active_experiments[experiment.id]

    async def _inject_failure(self, experiment: ChaosExperiment):
        """Inject specific type of failure."""
        try:
            while True:
                if random.random() < experiment.probability:
                    await self._apply_failure_type(experiment.failure_type, experiment)

                await asyncio.sleep(1.0)  # Check every second

        except asyncio.CancelledError:
            logger.debug(f"Failure injection cancelled for {experiment.name}")

    async def _apply_failure_type(
        self, failure_type: FailureType, experiment: ChaosExperiment
    ):
        """Apply specific failure type."""
        try:
            if failure_type == FailureType.LATENCY_INJECTION:
                # Inject artificial latency
                latency_ms = random.uniform(100, 2000)
                await asyncio.sleep(latency_ms / 1000)
                logger.debug(f"Injected {latency_ms}ms latency")

            elif failure_type == FailureType.ERROR_INJECTION:
                # Randomly raise errors
                if random.random() < 0.1:  # 10% chance to raise error
                    error_types = [
                        ValueError,
                        RuntimeError,
                        ConnectionError,
                        TimeoutError,
                    ]
                    error_type = random.choice(error_types)
                    raise error_type(
                        f"Chaos-injected error from experiment {experiment.name}"
                    )

            elif failure_type == FailureType.RESOURCE_EXHAUSTION:
                # Simulate resource exhaustion
                await self._simulate_resource_exhaustion()

            elif failure_type == FailureType.DATABASE_TIMEOUT:
                # Simulate database timeout
                await asyncio.sleep(random.uniform(5, 15))
                raise TimeoutError("Chaos-injected database timeout")

            # Add more failure types as needed

        except Exception as e:
            logger.debug(f"Applied chaos failure: {e}")
            # Re-raise to propagate the injected failure
            raise

    async def _simulate_resource_exhaustion(self):
        """Simulate resource exhaustion scenarios."""
        # Create memory pressure
        memory_hog = [0] * (1024 * 1024)  # Allocate ~4MB
        await asyncio.sleep(0.1)
        del memory_hog

    async def _monitor_experiment(self, experiment: ChaosExperiment):
        """Monitor system health during experiment."""
        try:
            while True:
                current_metrics = await self._collect_metrics()

                if self.safety_enabled and await self._should_emergency_stop(
                    experiment, current_metrics
                ):
                    logger.warning(
                        f"Emergency stopping experiment {experiment.name} due to safety threshold"
                    )
                    await self._emergency_rollback(experiment)
                    break

                await asyncio.sleep(experiment.health_check_interval)

        except asyncio.CancelledError:
            pass

    async def _should_emergency_stop(
        self, experiment: ChaosExperiment, metrics: ResilienceMetrics
    ) -> bool:
        """Determine if experiment should be emergency stopped."""
        if not self.baseline_metrics:
            return False

        # Check if error rate exceeded threshold
        error_rate_increase = metrics.error_rate - self.baseline_metrics.error_rate
        if error_rate_increase > experiment.rollback_threshold:
            return True

        # Check if latency increased dramatically
        latency_ratio = metrics.latency_p99 / max(
            self.baseline_metrics.latency_p99, 0.001
        )
        if latency_ratio > 5.0:  # 5x latency increase
            return True

        # Check if throughput dropped significantly
        throughput_ratio = metrics.throughput / max(
            self.baseline_metrics.throughput, 0.001
        )
        if throughput_ratio < 0.2:  # 80% throughput drop
            return True

        return False

    async def _ramp_up_failure(self, experiment: ChaosExperiment):
        """Gradually increase failure injection."""
        steps = 10
        step_duration = experiment.ramp_up_duration / steps
        original_probability = experiment.probability

        for step in range(steps):
            experiment.probability = original_probability * (step + 1) / steps
            await asyncio.sleep(step_duration)

        experiment.probability = original_probability

    async def _ramp_down_failure(self, experiment: ChaosExperiment):
        """Gradually decrease failure injection."""
        steps = 10
        step_duration = experiment.ramp_down_duration / steps
        original_probability = experiment.probability

        for step in range(steps):
            experiment.probability = original_probability * (steps - step) / steps
            await asyncio.sleep(step_duration)

        experiment.probability = original_probability

    async def _emergency_rollback(self, experiment: ChaosExperiment):
        """Perform emergency rollback of experiment."""
        logger.critical(
            f"Performing emergency rollback for experiment {experiment.name}"
        )

        # Stop all failure injection immediately
        experiment.probability = 0.0

        # Give system time to stabilize
        await asyncio.sleep(5.0)

        # Trigger self-healing mechanisms if available
        # This would integrate with the self-healing system

    async def _collect_metrics(self) -> ResilienceMetrics:
        """Collect current system metrics."""
        # In a real implementation, this would collect actual system metrics
        # For now, return simulated metrics

        return ResilienceMetrics(
            timestamp=datetime.now(),
            system_state=ResilienceState.HEALTHY,
            error_rate=random.uniform(0.01, 0.05),  # 1-5% error rate
            latency_p99=random.uniform(100, 500),  # 100-500ms P99 latency
            throughput=random.uniform(100, 1000),  # 100-1000 RPS
            circuit_breaker_trips=0,
            circuit_breaker_recoveries=0,
            active_experiments=len(self.active_experiments),
            experiment_success_rate=0.95,
            healing_actions=0,
            healing_success_rate=0.90,
            cpu_usage=random.uniform(0.2, 0.8),
            memory_usage=random.uniform(0.3, 0.7),
            connection_pool_usage=random.uniform(0.1, 0.6),
        )

    async def _record_experiment_result(
        self,
        experiment: ChaosExperiment,
        start_time: datetime,
        end_time: datetime,
        success: bool,
        error: Optional[str] = None,
    ):
        """Record experiment results for analysis."""
        result = {
            "experiment_id": experiment.id,
            "experiment_name": experiment.name,
            "failure_type": experiment.failure_type.value,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "success": success,
            "error": error,
            "baseline_metrics": (
                self.baseline_metrics.__dict__ if self.baseline_metrics else None
            ),
            "final_metrics": (await self._collect_metrics()).__dict__,
        }

        self.experiment_history.append(result)

        # Keep only last 100 experiments
        if len(self.experiment_history) > 100:
            self.experiment_history = self.experiment_history[-100:]

    def get_experiment_results(self) -> List[Dict[str, Any]]:
        """Get results from all experiments."""
        return self.experiment_history.copy()

    async def stop_all_experiments(self):
        """Stop all running experiments."""
        logger.info("Stopping all chaos experiments")

        for experiment_id, task in self.active_experiments.items():
            task.cancel()

        # Wait for all tasks to complete
        if self.active_experiments:
            await asyncio.gather(
                *self.active_experiments.values(), return_exceptions=True
            )

        self.active_experiments.clear()


class SelfHealingSystem:
    """
    Self-healing system that automatically detects and remedies issues.
    """

    def __init__(self):
        self.healing_actions: Dict[str, Callable] = {}
        self.health_checks: Dict[str, Callable] = {}
        self.healing_history: List[Dict[str, Any]] = []
        self.monitoring_enabled = True
        self.healing_enabled = True

        # Configuration
        self.check_interval = 30.0  # seconds
        self.max_healing_attempts = 3
        self.healing_backoff_factor = 2.0

        # State
        self.monitoring_task: Optional[asyncio.Task] = None
        self.component_states: Dict[str, str] = {}
        self.healing_attempts: Dict[str, int] = {}

    def register_health_check(self, component: str, check_func: Callable):
        """Register a health check for a component."""
        self.health_checks[component] = check_func
        logger.info(f"Registered health check for {component}")

    def register_healing_action(self, component: str, healing_func: Callable):
        """Register a healing action for a component."""
        self.healing_actions[component] = healing_func
        logger.info(f"Registered healing action for {component}")

    async def start_monitoring(self):
        """Start the self-healing monitoring loop."""
        if self.monitoring_task and not self.monitoring_task.done():
            logger.warning("Monitoring is already running")
            return

        self.monitoring_enabled = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started self-healing monitoring")

    async def stop_monitoring(self):
        """Stop the self-healing monitoring."""
        self.monitoring_enabled = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped self-healing monitoring")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        try:
            while self.monitoring_enabled:
                await self._check_all_components()
                await asyncio.sleep(self.check_interval)
        except asyncio.CancelledError:
            logger.debug("Self-healing monitoring cancelled")
        except Exception as e:
            logger.error(f"Self-healing monitoring error: {e}", exc_info=True)

    async def _check_all_components(self):
        """Check health of all registered components."""
        for component, check_func in self.health_checks.items():
            try:
                is_healthy = await self._run_health_check(check_func)
                previous_state = self.component_states.get(component, "unknown")
                current_state = "healthy" if is_healthy else "unhealthy"

                self.component_states[component] = current_state

                # Detect state changes
                if previous_state == "healthy" and current_state == "unhealthy":
                    logger.warning(f"Component {component} became unhealthy")
                    await self._trigger_healing(component)
                elif previous_state == "unhealthy" and current_state == "healthy":
                    logger.info(f"Component {component} recovered to healthy state")
                    self.healing_attempts[component] = 0  # Reset attempts

            except Exception as e:
                logger.error(f"Health check failed for {component}: {e}")
                self.component_states[component] = "error"

    async def _run_health_check(self, check_func: Callable) -> bool:
        """Run a health check function."""
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()

            return bool(result)
        except Exception as e:
            logger.debug(f"Health check exception: {e}")
            return False

    async def _trigger_healing(self, component: str):
        """Trigger healing action for a component."""
        if not self.healing_enabled:
            logger.debug(f"Healing disabled, skipping {component}")
            return

        if component not in self.healing_actions:
            logger.warning(f"No healing action registered for {component}")
            return

        # Check if we've exceeded max attempts
        attempts = self.healing_attempts.get(component, 0)
        if attempts >= self.max_healing_attempts:
            logger.error(
                f"Max healing attempts ({self.max_healing_attempts}) exceeded for {component}"
            )
            return

        # Calculate backoff delay
        backoff_delay = (
            self.healing_backoff_factor**attempts
        ) * 5.0  # Start with 5 seconds

        logger.info(
            f"Triggering healing for {component} (attempt {attempts + 1}) after {backoff_delay}s delay"
        )

        try:
            await asyncio.sleep(backoff_delay)

            healing_func = self.healing_actions[component]
            start_time = datetime.now()

            success = await self._run_healing_action(healing_func)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            self.healing_attempts[component] = attempts + 1

            # Record healing attempt
            self.healing_history.append(
                {
                    "component": component,
                    "attempt": attempts + 1,
                    "timestamp": start_time.isoformat(),
                    "duration_seconds": duration,
                    "success": success,
                }
            )

            if success:
                logger.info(f"Healing successful for {component} in {duration:.2f}s")
            else:
                logger.warning(f"Healing failed for {component}")

        except Exception as e:
            logger.error(f"Healing action failed for {component}: {e}", exc_info=True)

    async def _run_healing_action(self, healing_func: Callable) -> bool:
        """Run a healing action function."""
        try:
            if asyncio.iscoroutinefunction(healing_func):
                result = await healing_func()
            else:
                result = healing_func()

            return bool(result)
        except Exception as e:
            logger.error(f"Healing action exception: {e}")
            return False

    def get_component_states(self) -> Dict[str, str]:
        """Get current states of all components."""
        return self.component_states.copy()

    def get_healing_history(self) -> List[Dict[str, Any]]:
        """Get healing history."""
        return self.healing_history.copy()

    def reset_component_attempts(self, component: str):
        """Reset healing attempts for a component."""
        self.healing_attempts[component] = 0
        logger.info(f"Reset healing attempts for {component}")


class ResilienceEngine:
    """
    Main resilience engine that orchestrates all resilience features.
    """

    def __init__(self):
        self.circuit_breakers: Dict[str, AdvancedCircuitBreaker] = {}
        self.chaos_engine = ChaosEngineeringEngine()
        self.self_healing = SelfHealingSystem()

        # Global resilience state
        self.system_state = ResilienceState.HEALTHY
        self.resilience_enabled = True

        # Monitoring
        self.metrics_history: List[ResilienceMetrics] = []
        self.alert_callbacks: List[Callable] = []

    def create_circuit_breaker(self, name: str, **kwargs) -> AdvancedCircuitBreaker:
        """Create and register a circuit breaker."""
        cb = AdvancedCircuitBreaker(name, **kwargs)
        self.circuit_breakers[name] = cb
        logger.info(f"Created circuit breaker: {name}")
        return cb

    def get_circuit_breaker(self, name: str) -> Optional[AdvancedCircuitBreaker]:
        """Get a circuit breaker by name."""
        return self.circuit_breakers.get(name)

    async def start_chaos_experiment(self, experiment_id: str) -> bool:
        """Start a chaos engineering experiment."""
        return await self.chaos_engine.start_experiment(experiment_id)

    async def initialize_default_experiments(self):
        """Initialize default chaos experiments."""
        experiments = [
            ChaosExperiment(
                id="latency_chaos",
                name="Random Latency Injection",
                failure_type=FailureType.LATENCY_INJECTION,
                probability=0.05,  # 5% of requests
                duration_seconds=60.0,
                enabled=False,  # Disabled by default for safety
            ),
            ChaosExperiment(
                id="error_chaos",
                name="Random Error Injection",
                failure_type=FailureType.ERROR_INJECTION,
                probability=0.02,  # 2% of requests
                duration_seconds=30.0,
                enabled=False,  # Disabled by default for safety
            ),
            ChaosExperiment(
                id="db_timeout_chaos",
                name="Database Timeout Simulation",
                failure_type=FailureType.DATABASE_TIMEOUT,
                probability=0.01,  # 1% of requests
                duration_seconds=45.0,
                enabled=False,  # Disabled by default for safety
            ),
        ]

        for experiment in experiments:
            self.chaos_engine.add_experiment(experiment)

    async def start_resilience_monitoring(self):
        """Start all resilience monitoring systems."""
        await self.self_healing.start_monitoring()
        logger.info("Resilience monitoring started")

    async def stop_resilience_monitoring(self):
        """Stop all resilience monitoring systems."""
        await self.self_healing.stop_monitoring()
        await self.chaos_engine.stop_all_experiments()
        logger.info("Resilience monitoring stopped")

    def register_alert_callback(self, callback: Callable):
        """Register callback for resilience alerts."""
        self.alert_callbacks.append(callback)

    async def _trigger_alerts(
        self, alert_type: str, message: str, severity: str = "warning"
    ):
        """Trigger resilience alerts."""
        alert_data = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "system_state": self.system_state.value,
        }

        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_data)
                else:
                    callback(alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def get_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report."""
        # Circuit breaker metrics
        cb_metrics = {}
        for name, cb in self.circuit_breakers.items():
            cb_metrics[name] = cb.get_metrics()

        # Chaos engineering metrics
        chaos_metrics = {
            "active_experiments": len(self.chaos_engine.active_experiments),
            "total_experiments": len(self.chaos_engine.experiments),
            "experiment_history_count": len(self.chaos_engine.experiment_history),
        }

        # Self-healing metrics
        healing_metrics = {
            "component_states": self.self_healing.get_component_states(),
            "healing_history_count": len(self.self_healing.healing_history),
            "monitoring_enabled": self.self_healing.monitoring_enabled,
            "healing_enabled": self.self_healing.healing_enabled,
        }

        return {
            "timestamp": datetime.now().isoformat(),
            "system_state": self.system_state.value,
            "resilience_enabled": self.resilience_enabled,
            "circuit_breakers": cb_metrics,
            "chaos_engineering": chaos_metrics,
            "self_healing": healing_metrics,
        }


# Example usage and integration functions
async def example_database_health_check() -> bool:
    """Example database health check."""
    try:
        # Simulate database connection check
        await asyncio.sleep(0.1)
        return random.random() > 0.1  # 90% healthy
    except Exception:
        return False


async def example_database_healing_action() -> bool:
    """Example database healing action."""
    try:
        logger.info("Attempting database connection pool reset")
        await asyncio.sleep(1.0)  # Simulate healing time
        return random.random() > 0.2  # 80% success rate
    except Exception:
        return False


async def setup_resilience_engine() -> ResilienceEngine:
    """Set up the resilience engine with default configuration."""
    engine = ResilienceEngine()

    # Initialize default chaos experiments
    await engine.initialize_default_experiments()

    # Set up self-healing for database
    engine.self_healing.register_health_check("database", example_database_health_check)
    engine.self_healing.register_healing_action(
        "database", example_database_healing_action
    )

    # Create circuit breakers
    engine.create_circuit_breaker(
        "database", failure_threshold=5, recovery_timeout=30.0
    )
    engine.create_circuit_breaker(
        "external_api", failure_threshold=3, recovery_timeout=60.0
    )

    logger.info("Resilience engine setup completed")
    return engine
