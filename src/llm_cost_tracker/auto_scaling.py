"""Auto-scaling system with intelligent resource management."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction options."""

    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ResourceType(Enum):
    """Types of resources that can be scaled."""

    DATABASE_CONNECTIONS = "database_connections"
    WORKER_THREADS = "worker_threads"
    CACHE_SIZE = "cache_size"
    BATCH_SIZE = "batch_size"
    RATE_LIMITS = "rate_limits"


@dataclass
class ResourceMetrics:
    """Current resource utilization metrics."""

    cpu_percent: float
    memory_percent: float
    disk_io_percent: float
    network_io_percent: float
    active_connections: int
    queue_depth: int
    response_time_ms: float
    error_rate: float
    timestamp: float


@dataclass
class ScalingRule:
    """Rule for auto-scaling decisions."""

    resource_type: ResourceType
    metric_name: str
    threshold_up: float
    threshold_down: float
    min_value: int
    max_value: int
    cooldown_seconds: int
    scale_factor: float
    enabled: bool = True


@dataclass
class ScalingEvent:
    """Record of a scaling event."""

    timestamp: float
    resource_type: ResourceType
    direction: ScalingDirection
    old_value: int
    new_value: int
    trigger_metric: str
    trigger_value: float
    success: bool
    reason: str


class MetricsCollector:
    """Collect system and application metrics for scaling decisions."""

    def __init__(self, collection_interval: float = 10.0):
        self.collection_interval = collection_interval
        self.metrics_history: List[ResourceMetrics] = []
        self.max_history_size = 100
        self.custom_metrics: Dict[str, Callable] = {}
        self.is_collecting = False

    def register_metric(self, name: str, collector_func: Callable) -> None:
        """Register a custom metric collector."""
        self.custom_metrics[name] = collector_func
        logger.info(f"Registered custom metric: {name}")

    async def start_collection(self) -> None:
        """Start continuous metrics collection."""
        if self.is_collecting:
            return

        self.is_collecting = True
        logger.info("Started metrics collection")

        while self.is_collecting:
            try:
                metrics = await self.collect_metrics()
                self.metrics_history.append(metrics)

                # Keep history size manageable
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history.pop(0)

                await asyncio.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(self.collection_interval)

    async def collect_metrics(self) -> ResourceMetrics:
        """Collect current system metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()

        # Calculate IO percentages (simplified)
        disk_io_percent = min(
            100, (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024 * 100)
        )  # MB/s
        network_io_percent = min(
            100, (network_io.bytes_sent + network_io.bytes_recv) / (1024 * 1024 * 10)
        )  # MB/s

        # Application metrics (to be populated by custom collectors)
        active_connections = 0
        queue_depth = 0
        response_time_ms = 0.0
        error_rate = 0.0

        # Collect custom metrics
        for name, collector in self.custom_metrics.items():
            try:
                if asyncio.iscoroutinefunction(collector):
                    value = await collector()
                else:
                    value = collector()

                # Map custom metrics to standard fields
                if name == "active_connections":
                    active_connections = value
                elif name == "queue_depth":
                    queue_depth = value
                elif name == "response_time_ms":
                    response_time_ms = value
                elif name == "error_rate":
                    error_rate = value

            except Exception as e:
                logger.error(f"Error collecting custom metric {name}: {e}")

        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_io_percent=disk_io_percent,
            network_io_percent=network_io_percent,
            active_connections=active_connections,
            queue_depth=queue_depth,
            response_time_ms=response_time_ms,
            error_rate=error_rate,
            timestamp=time.time(),
        )

    def stop_collection(self) -> None:
        """Stop metrics collection."""
        self.is_collecting = False
        logger.info("Stopped metrics collection")

    def get_recent_metrics(self, seconds: int = 60) -> List[ResourceMetrics]:
        """Get metrics from the last N seconds."""
        cutoff_time = time.time() - seconds
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]

    def get_average_metrics(self, seconds: int = 60) -> Optional[ResourceMetrics]:
        """Get average metrics over the last N seconds."""
        recent = self.get_recent_metrics(seconds)
        if not recent:
            return None

        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in recent) / len(recent)
        avg_memory = sum(m.memory_percent for m in recent) / len(recent)
        avg_disk_io = sum(m.disk_io_percent for m in recent) / len(recent)
        avg_network_io = sum(m.network_io_percent for m in recent) / len(recent)
        avg_connections = sum(m.active_connections for m in recent) / len(recent)
        avg_queue_depth = sum(m.queue_depth for m in recent) / len(recent)
        avg_response_time = sum(m.response_time_ms for m in recent) / len(recent)
        avg_error_rate = sum(m.error_rate for m in recent) / len(recent)

        return ResourceMetrics(
            cpu_percent=avg_cpu,
            memory_percent=avg_memory,
            disk_io_percent=avg_disk_io,
            network_io_percent=avg_network_io,
            active_connections=int(avg_connections),
            queue_depth=int(avg_queue_depth),
            response_time_ms=avg_response_time,
            error_rate=avg_error_rate,
            timestamp=time.time(),
        )


class AutoScaler:
    """Intelligent auto-scaling system."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.scaling_rules: Dict[ResourceType, List[ScalingRule]] = {}
        self.current_values: Dict[ResourceType, int] = {}
        self.last_scaling_events: Dict[ResourceType, float] = {}
        self.scaling_history: List[ScalingEvent] = []
        self.resource_handlers: Dict[ResourceType, Callable] = {}
        self.is_enabled = True

        # Initialize default rules
        self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        """Setup default scaling rules."""
        # Database connections scaling
        self.add_scaling_rule(
            ScalingRule(
                resource_type=ResourceType.DATABASE_CONNECTIONS,
                metric_name="active_connections",
                threshold_up=80,  # Scale up when 80% of connections are active
                threshold_down=30,  # Scale down when less than 30% are active
                min_value=2,
                max_value=50,
                cooldown_seconds=300,  # 5 minutes
                scale_factor=1.5,
            )
        )

        # Worker threads scaling
        self.add_scaling_rule(
            ScalingRule(
                resource_type=ResourceType.WORKER_THREADS,
                metric_name="cpu_percent",
                threshold_up=75,
                threshold_down=40,
                min_value=2,
                max_value=20,
                cooldown_seconds=180,  # 3 minutes
                scale_factor=1.3,
            )
        )

        # Cache size scaling
        self.add_scaling_rule(
            ScalingRule(
                resource_type=ResourceType.CACHE_SIZE,
                metric_name="memory_percent",
                threshold_up=70,
                threshold_down=40,
                min_value=1000,
                max_value=50000,
                cooldown_seconds=600,  # 10 minutes
                scale_factor=1.5,
            )
        )

        # Rate limits scaling
        self.add_scaling_rule(
            ScalingRule(
                resource_type=ResourceType.RATE_LIMITS,
                metric_name="error_rate",
                threshold_up=5,  # Increase rate limits if error rate is low
                threshold_down=15,  # Decrease rate limits if error rate is high
                min_value=10,
                max_value=1000,
                cooldown_seconds=120,  # 2 minutes
                scale_factor=1.2,
            )
        )

    def add_scaling_rule(self, rule: ScalingRule) -> None:
        """Add a new scaling rule."""
        if rule.resource_type not in self.scaling_rules:
            self.scaling_rules[rule.resource_type] = []

        self.scaling_rules[rule.resource_type].append(rule)

        # Initialize current value if not set
        if rule.resource_type not in self.current_values:
            self.current_values[rule.resource_type] = rule.min_value

        logger.info(f"Added scaling rule for {rule.resource_type.value}")

    def register_resource_handler(
        self, resource_type: ResourceType, handler: Callable
    ) -> None:
        """Register a handler for scaling a specific resource type."""
        self.resource_handlers[resource_type] = handler
        logger.info(f"Registered handler for {resource_type.value}")

    async def evaluate_scaling(self) -> List[ScalingEvent]:
        """Evaluate all scaling rules and make scaling decisions."""
        if not self.is_enabled:
            return []

        events = []
        avg_metrics = self.metrics_collector.get_average_metrics(
            seconds=120
        )  # 2 minute average

        if not avg_metrics:
            logger.warning("No metrics available for scaling evaluation")
            return []

        for resource_type, rules in self.scaling_rules.items():
            for rule in rules:
                if not rule.enabled:
                    continue

                # Check cooldown
                last_scaling = self.last_scaling_events.get(resource_type, 0)
                if time.time() - last_scaling < rule.cooldown_seconds:
                    continue

                # Get metric value
                metric_value = self._get_metric_value(avg_metrics, rule.metric_name)
                if metric_value is None:
                    continue

                # Evaluate scaling decision
                scaling_decision = self._evaluate_rule(rule, metric_value)

                if scaling_decision != ScalingDirection.STABLE:
                    event = await self._execute_scaling(
                        rule, scaling_decision, metric_value
                    )
                    if event:
                        events.append(event)

        return events

    def _get_metric_value(
        self, metrics: ResourceMetrics, metric_name: str
    ) -> Optional[float]:
        """Extract metric value from metrics object."""
        metric_map = {
            "cpu_percent": metrics.cpu_percent,
            "memory_percent": metrics.memory_percent,
            "disk_io_percent": metrics.disk_io_percent,
            "network_io_percent": metrics.network_io_percent,
            "active_connections": metrics.active_connections,
            "queue_depth": metrics.queue_depth,
            "response_time_ms": metrics.response_time_ms,
            "error_rate": metrics.error_rate,
        }

        return metric_map.get(metric_name)

    def _evaluate_rule(
        self, rule: ScalingRule, metric_value: float
    ) -> ScalingDirection:
        """Evaluate a single scaling rule."""
        current_value = self.current_values.get(rule.resource_type, rule.min_value)

        # Check for scale up conditions
        if metric_value > rule.threshold_up and current_value < rule.max_value:
            return ScalingDirection.UP

        # Check for scale down conditions
        if metric_value < rule.threshold_down and current_value > rule.min_value:
            return ScalingDirection.DOWN

        return ScalingDirection.STABLE

    async def _execute_scaling(
        self, rule: ScalingRule, direction: ScalingDirection, metric_value: float
    ) -> Optional[ScalingEvent]:
        """Execute scaling action."""
        current_value = self.current_values.get(rule.resource_type, rule.min_value)

        # Calculate new value
        if direction == ScalingDirection.UP:
            new_value = min(rule.max_value, int(current_value * rule.scale_factor))
        else:  # ScalingDirection.DOWN
            new_value = max(rule.min_value, int(current_value / rule.scale_factor))

        # No change needed
        if new_value == current_value:
            return None

        success = False
        reason = ""

        try:
            # Execute scaling through registered handler
            if rule.resource_type in self.resource_handlers:
                handler = self.resource_handlers[rule.resource_type]
                if asyncio.iscoroutinefunction(handler):
                    await handler(new_value)
                else:
                    handler(new_value)

                success = True
                reason = f"Scaled {direction.value} due to {rule.metric_name}={metric_value:.2f}"

                # Update current value and last scaling time
                self.current_values[rule.resource_type] = new_value
                self.last_scaling_events[rule.resource_type] = time.time()

                logger.info(
                    f"Scaled {rule.resource_type.value} from {current_value} to {new_value}"
                )
            else:
                reason = f"No handler registered for {rule.resource_type.value}"
                logger.warning(reason)

        except Exception as e:
            reason = f"Scaling failed: {str(e)}"
            logger.error(reason)

        # Create scaling event
        event = ScalingEvent(
            timestamp=time.time(),
            resource_type=rule.resource_type,
            direction=direction,
            old_value=current_value,
            new_value=new_value if success else current_value,
            trigger_metric=rule.metric_name,
            trigger_value=metric_value,
            success=success,
            reason=reason,
        )

        self.scaling_history.append(event)

        # Keep history manageable
        if len(self.scaling_history) > 1000:
            self.scaling_history = self.scaling_history[-500:]

        return event

    def enable_scaling(self, resource_type: Optional[ResourceType] = None) -> None:
        """Enable auto-scaling globally or for specific resource type."""
        if resource_type:
            rules = self.scaling_rules.get(resource_type, [])
            for rule in rules:
                rule.enabled = True
            logger.info(f"Enabled scaling for {resource_type.value}")
        else:
            self.is_enabled = True
            logger.info("Enabled auto-scaling globally")

    def disable_scaling(self, resource_type: Optional[ResourceType] = None) -> None:
        """Disable auto-scaling globally or for specific resource type."""
        if resource_type:
            rules = self.scaling_rules.get(resource_type, [])
            for rule in rules:
                rule.enabled = False
            logger.info(f"Disabled scaling for {resource_type.value}")
        else:
            self.is_enabled = False
            logger.info("Disabled auto-scaling globally")

    def get_current_configuration(self) -> Dict[str, Any]:
        """Get current scaling configuration."""
        return {
            "enabled": self.is_enabled,
            "current_values": {k.value: v for k, v in self.current_values.items()},
            "rules": {
                k.value: [
                    {
                        "metric_name": rule.metric_name,
                        "threshold_up": rule.threshold_up,
                        "threshold_down": rule.threshold_down,
                        "min_value": rule.min_value,
                        "max_value": rule.max_value,
                        "scale_factor": rule.scale_factor,
                        "enabled": rule.enabled,
                    }
                    for rule in rules
                ]
                for k, rules in self.scaling_rules.items()
            },
        }

    def get_scaling_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent scaling history."""
        recent_events = self.scaling_history[-limit:] if limit else self.scaling_history

        return [
            {
                "timestamp": event.timestamp,
                "resource_type": event.resource_type.value,
                "direction": event.direction.value,
                "old_value": event.old_value,
                "new_value": event.new_value,
                "trigger_metric": event.trigger_metric,
                "trigger_value": event.trigger_value,
                "success": event.success,
                "reason": event.reason,
            }
            for event in recent_events
        ]


# Global auto-scaling system
metrics_collector = MetricsCollector(collection_interval=15.0)
auto_scaler = AutoScaler(metrics_collector)


async def start_auto_scaling():
    """Start the auto-scaling system."""
    # Start metrics collection
    asyncio.create_task(metrics_collector.start_collection())

    # Start scaling evaluation loop
    async def scaling_loop():
        while True:
            try:
                events = await auto_scaler.evaluate_scaling()
                if events:
                    logger.info(f"Executed {len(events)} scaling events")

                await asyncio.sleep(60)  # Evaluate every minute
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                await asyncio.sleep(60)

    asyncio.create_task(scaling_loop())
    logger.info("Auto-scaling system started")
