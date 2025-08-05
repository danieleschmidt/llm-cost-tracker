"""
Advanced monitoring and observability for Quantum Task Planner.
Provides comprehensive metrics, health checks, and performance monitoring.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Deque
import logging
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
    import warnings
    warnings.warn("psutil not available, system metrics will be simulated", ImportWarning)
import threading
from concurrent.futures import ThreadPoolExecutor

from .quantum_task_planner import QuantumTaskPlanner, TaskState

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for quantum task execution."""
    
    total_tasks_executed: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    avg_execution_time_ms: float = 0.0
    min_execution_time_ms: float = float('inf')
    max_execution_time_ms: float = 0.0
    total_execution_time_ms: float = 0.0
    
    # Resource utilization metrics
    peak_cpu_usage: float = 0.0
    peak_memory_usage: float = 0.0
    avg_cpu_usage: float = 0.0
    avg_memory_usage: float = 0.0
    
    # Quantum-specific metrics
    avg_execution_probability: float = 0.0
    entanglement_violations: int = 0
    superposition_collapses: int = 0
    interference_effects_applied: int = 0
    
    # Scheduling metrics
    schedule_optimizations: int = 0
    avg_schedule_quality: float = 0.0
    dependency_violations: int = 0
    resource_conflicts: int = 0
    
    def calculate_success_rate(self) -> float:
        """Calculate task execution success rate."""
        if self.total_tasks_executed == 0:
            return 0.0
        return self.successful_executions / self.total_tasks_executed
    
    def update_execution_time(self, execution_time_ms: float) -> None:
        """Update execution time statistics."""
        self.total_execution_time_ms += execution_time_ms
        self.min_execution_time_ms = min(self.min_execution_time_ms, execution_time_ms)
        self.max_execution_time_ms = max(self.max_execution_time_ms, execution_time_ms)
        self.avg_execution_time_ms = self.total_execution_time_ms / max(self.total_tasks_executed, 1)


@dataclass
class SystemHealthMetrics:
    """System health metrics for quantum planner."""
    
    is_healthy: bool = True
    last_health_check: datetime = field(default_factory=datetime.now)
    health_check_duration_ms: float = 0.0
    
    # Component health
    task_planner_healthy: bool = True
    resource_pool_healthy: bool = True
    scheduler_healthy: bool = True
    monitoring_healthy: bool = True
    
    # System resources
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    network_io_bytes: Dict[str, int] = field(default_factory=dict)
    
    # Error tracking
    recent_errors: List[str] = field(default_factory=list)
    error_count_last_hour: int = 0
    critical_error_count: int = 0
    
    def is_system_healthy(self) -> bool:
        """Determine overall system health."""
        return (
            self.task_planner_healthy and
            self.resource_pool_healthy and
            self.scheduler_healthy and
            self.monitoring_healthy and
            self.cpu_usage_percent < 90.0 and
            self.memory_usage_percent < 90.0 and
            self.critical_error_count == 0
        )


class QuantumTaskMonitor:
    """
    Advanced monitoring system for quantum task planning.
    Provides real-time metrics, health checks, and performance analysis.
    """
    
    def __init__(self, planner: QuantumTaskPlanner, monitoring_interval: float = 5.0):
        self.planner = planner
        self.monitoring_interval = monitoring_interval
        
        # Metrics storage
        self.performance_metrics = PerformanceMetrics()
        self.health_metrics = SystemHealthMetrics()
        
        # Time-series data (last 1000 data points)
        self.execution_times: Deque[float] = deque(maxlen=1000)
        self.cpu_usage_history: Deque[float] = deque(maxlen=1000)
        self.memory_usage_history: Deque[float] = deque(maxlen=1000)
        self.task_completion_times: Deque[datetime] = deque(maxlen=1000)
        
        # Error tracking
        self.error_history: Deque[Dict[str, Any]] = deque(maxlen=500)
        self.alert_thresholds = {
            'cpu_usage': 85.0,
            'memory_usage': 85.0,
            'error_rate': 0.1,
            'execution_failure_rate': 0.15
        }
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="quantum-monitor")
        
        # Event handlers
        self.alert_handlers: List[callable] = []
        
    def add_alert_handler(self, handler: callable) -> None:
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
        logger.info(f"Added alert handler: {handler.__name__}")
    
    async def start_monitoring(self) -> None:
        """Start the monitoring system."""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="quantum-task-monitor",
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("Quantum task monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop the monitoring system."""
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10.0)
        
        self.executor.shutdown(wait=True)
        logger.info("Quantum task monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in separate thread."""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                self._update_health_status()
                self._check_alert_conditions()
                
                time.sleep(self.monitoring_interval)
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self._record_error("monitoring_loop", str(e), "critical")
                time.sleep(self.monitoring_interval * 2)  # Back off on errors
    
    def _collect_system_metrics(self) -> None:
        """Collect system resource metrics."""
        try:
            if PSUTIL_AVAILABLE:
                # CPU and memory usage
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Update health metrics
                self.health_metrics.cpu_usage_percent = cpu_percent
                self.health_metrics.memory_usage_percent = memory.percent
                self.health_metrics.disk_usage_percent = disk.percent
            else:
                # Simulate metrics when psutil is not available
                import random
                self.health_metrics.cpu_usage_percent = random.uniform(10, 30)
                self.health_metrics.memory_usage_percent = random.uniform(20, 40)
                self.health_metrics.disk_usage_percent = random.uniform(30, 50)
            
            # Store in time series
            cpu_percent = self.health_metrics.cpu_usage_percent
            memory_percent = self.health_metrics.memory_usage_percent
            self.cpu_usage_history.append(cpu_percent)
            self.memory_usage_history.append(memory_percent)
            
            # Update peak usage
            self.performance_metrics.peak_cpu_usage = max(
                self.performance_metrics.peak_cpu_usage, cpu_percent
            )
            self.performance_metrics.peak_memory_usage = max(
                self.performance_metrics.peak_memory_usage, memory_percent
            )
            
            # Calculate averages
            if len(self.cpu_usage_history) > 0:
                self.performance_metrics.avg_cpu_usage = sum(self.cpu_usage_history) / len(self.cpu_usage_history)
            if len(self.memory_usage_history) > 0:
                self.performance_metrics.avg_memory_usage = sum(self.memory_usage_history) / len(self.memory_usage_history)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            self._record_error("metrics_collection", str(e), "warning")
    
    def _update_health_status(self) -> None:
        """Update overall system health status."""
        start_time = time.time()
        
        try:
            # Check planner health
            self.health_metrics.task_planner_healthy = len(self.planner.tasks) >= 0
            
            # Check resource pool health
            pool = self.planner.resource_pool
            self.health_metrics.resource_pool_healthy = (
                pool.allocated_cpu <= pool.cpu_cores and
                pool.allocated_memory <= pool.memory_gb and
                pool.allocated_storage <= pool.storage_gb and
                pool.allocated_bandwidth <= pool.network_bandwidth
            )
            
            # Check scheduler health (can generate schedules)
            try:
                if len(self.planner.tasks) > 0:
                    test_schedule = self.planner.quantum_anneal_schedule(max_iterations=5)
                    self.health_metrics.scheduler_healthy = len(test_schedule) > 0
            except Exception:
                self.health_metrics.scheduler_healthy = False
            
            # Update overall health
            self.health_metrics.is_healthy = self.health_metrics.is_system_healthy()
            self.health_metrics.last_health_check = datetime.now()
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.health_metrics.is_healthy = False
            self._record_error("health_check", str(e), "critical")
        
        finally:
            # Record health check duration
            self.health_metrics.health_check_duration_ms = (time.time() - start_time) * 1000
    
    def _check_alert_conditions(self) -> None:
        """Check for alert conditions and trigger handlers."""
        alerts = []
        
        # CPU usage alert
        if self.health_metrics.cpu_usage_percent > self.alert_thresholds['cpu_usage']:
            alerts.append({
                'type': 'high_cpu_usage',
                'severity': 'warning',
                'message': f"CPU usage is {self.health_metrics.cpu_usage_percent:.1f}%",
                'threshold': self.alert_thresholds['cpu_usage']
            })
        
        # Memory usage alert
        if self.health_metrics.memory_usage_percent > self.alert_thresholds['memory_usage']:
            alerts.append({
                'type': 'high_memory_usage',
                'severity': 'warning',
                'message': f"Memory usage is {self.health_metrics.memory_usage_percent:.1f}%",
                'threshold': self.alert_thresholds['memory_usage']
            })
        
        # Error rate alert
        if len(self.execution_times) > 10:
            recent_failures = sum(1 for _ in self.error_history if 
                                (datetime.now() - _['timestamp']).total_seconds() < 3600)
            error_rate = recent_failures / len(self.execution_times)
            
            if error_rate > self.alert_thresholds['error_rate']:
                alerts.append({
                    'type': 'high_error_rate',
                    'severity': 'critical',
                    'message': f"Error rate is {error_rate:.2%}",
                    'threshold': self.alert_thresholds['error_rate']
                })
        
        # Task execution failure alert
        success_rate = self.performance_metrics.calculate_success_rate()
        if success_rate < (1.0 - self.alert_thresholds['execution_failure_rate']):
            alerts.append({
                'type': 'high_execution_failure_rate',
                'severity': 'critical',
                'message': f"Task execution failure rate is {1.0 - success_rate:.2%}",
                'threshold': self.alert_thresholds['execution_failure_rate']
            })
        
        # Trigger alert handlers
        for alert in alerts:
            self._trigger_alert(alert)
    
    def _trigger_alert(self, alert: Dict[str, Any]) -> None:
        """Trigger alert handlers."""
        alert['timestamp'] = datetime.now()
        
        logger.warning(f"ALERT [{alert['type']}]: {alert['message']}")
        
        for handler in self.alert_handlers:
            try:
                self.executor.submit(handler, alert)
            except Exception as e:
                logger.error(f"Alert handler {handler.__name__} failed: {e}")
    
    def _record_error(self, component: str, error_message: str, severity: str = "error") -> None:
        """Record an error in the error history."""
        error_record = {
            'timestamp': datetime.now(),
            'component': component,
            'message': error_message,
            'severity': severity
        }
        
        self.error_history.append(error_record)
        
        if severity == "critical":
            self.health_metrics.critical_error_count += 1
        
        # Update recent errors (last 10)
        self.health_metrics.recent_errors = [
            f"{record['component']}: {record['message']}" 
            for record in list(self.error_history)[-10:]
        ]
        
        # Count errors in last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        self.health_metrics.error_count_last_hour = sum(
            1 for record in self.error_history 
            if record['timestamp'] > one_hour_ago
        )
    
    def record_task_execution(self, task_id: str, execution_time_ms: float, 
                            success: bool, execution_probability: float) -> None:
        """Record metrics from task execution."""
        # Update performance metrics
        self.performance_metrics.total_tasks_executed += 1
        if success:
            self.performance_metrics.successful_executions += 1
        else:
            self.performance_metrics.failed_executions += 1
        
        self.performance_metrics.update_execution_time(execution_time_ms)
        
        # Update quantum-specific metrics
        self.performance_metrics.avg_execution_probability = (
            (self.performance_metrics.avg_execution_probability * 
             (self.performance_metrics.total_tasks_executed - 1) + 
             execution_probability) / self.performance_metrics.total_tasks_executed
        )
        
        # Store in time series
        self.execution_times.append(execution_time_ms)
        self.task_completion_times.append(datetime.now())
        
        logger.debug(f"Recorded execution metrics for task {task_id}: "
                    f"{execution_time_ms:.2f}ms, success={success}")
    
    def record_quantum_event(self, event_type: str, task_id: str, details: Dict[str, Any] = None) -> None:
        """Record quantum-specific events."""
        if event_type == "superposition_collapse":
            self.performance_metrics.superposition_collapses += 1
        elif event_type == "entanglement_violation":
            self.performance_metrics.entanglement_violations += 1
        elif event_type == "interference_applied":
            self.performance_metrics.interference_effects_applied += 1
        
        logger.debug(f"Recorded quantum event: {event_type} for task {task_id}")
    
    def record_scheduling_event(self, schedule_quality: float, violations: int, conflicts: int) -> None:
        """Record scheduling optimization metrics."""
        self.performance_metrics.schedule_optimizations += 1
        
        # Update average schedule quality
        current_avg = self.performance_metrics.avg_schedule_quality
        current_count = self.performance_metrics.schedule_optimizations
        
        self.performance_metrics.avg_schedule_quality = (
            (current_avg * (current_count - 1) + schedule_quality) / current_count
        )
        
        self.performance_metrics.dependency_violations += violations
        self.performance_metrics.resource_conflicts += conflicts
        
        logger.debug(f"Recorded scheduling metrics: quality={schedule_quality:.2f}, "
                    f"violations={violations}, conflicts={conflicts}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            "execution_metrics": {
                "total_tasks": self.performance_metrics.total_tasks_executed,
                "success_rate": self.performance_metrics.calculate_success_rate(),
                "avg_execution_time_ms": self.performance_metrics.avg_execution_time_ms,
                "min_execution_time_ms": self.performance_metrics.min_execution_time_ms,
                "max_execution_time_ms": self.performance_metrics.max_execution_time_ms
            },
            "resource_metrics": {
                "peak_cpu_usage": self.performance_metrics.peak_cpu_usage,
                "peak_memory_usage": self.performance_metrics.peak_memory_usage,
                "avg_cpu_usage": self.performance_metrics.avg_cpu_usage,
                "avg_memory_usage": self.performance_metrics.avg_memory_usage
            },
            "quantum_metrics": {
                "avg_execution_probability": self.performance_metrics.avg_execution_probability,
                "superposition_collapses": self.performance_metrics.superposition_collapses,
                "entanglement_violations": self.performance_metrics.entanglement_violations,
                "interference_effects": self.performance_metrics.interference_effects_applied
            },
            "scheduling_metrics": {
                "optimizations": self.performance_metrics.schedule_optimizations,
                "avg_quality": self.performance_metrics.avg_schedule_quality,
                "dependency_violations": self.performance_metrics.dependency_violations,
                "resource_conflicts": self.performance_metrics.resource_conflicts
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        return {
            "overall_health": self.health_metrics.is_healthy,
            "last_check": self.health_metrics.last_health_check.isoformat(),
            "check_duration_ms": self.health_metrics.health_check_duration_ms,
            "component_health": {
                "task_planner": self.health_metrics.task_planner_healthy,
                "resource_pool": self.health_metrics.resource_pool_healthy,
                "scheduler": self.health_metrics.scheduler_healthy,
                "monitoring": self.health_metrics.monitoring_healthy
            },
            "system_resources": {
                "cpu_usage_percent": self.health_metrics.cpu_usage_percent,
                "memory_usage_percent": self.health_metrics.memory_usage_percent,
                "disk_usage_percent": self.health_metrics.disk_usage_percent
            },
            "error_tracking": {
                "errors_last_hour": self.health_metrics.error_count_last_hour,
                "critical_errors": self.health_metrics.critical_error_count,
                "recent_errors": self.health_metrics.recent_errors
            }
        }
    
    def get_time_series_data(self, metric: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get time series data for a specific metric."""
        now = datetime.now()
        
        if metric == "cpu_usage":
            data = list(self.cpu_usage_history)[-limit:]
            return [
                {
                    "timestamp": (now - timedelta(seconds=i * self.monitoring_interval)).isoformat(),
                    "value": value
                }
                for i, value in enumerate(reversed(data))
            ]
        
        elif metric == "memory_usage":
            data = list(self.memory_usage_history)[-limit:]
            return [
                {
                    "timestamp": (now - timedelta(seconds=i * self.monitoring_interval)).isoformat(),
                    "value": value
                }
                for i, value in enumerate(reversed(data))
            ]
        
        elif metric == "execution_times":
            data = list(self.execution_times)[-limit:]
            completion_times = list(self.task_completion_times)[-len(data):]
            return [
                {
                    "timestamp": completion_times[i].isoformat() if i < len(completion_times) else now.isoformat(),
                    "value": value
                }
                for i, value in enumerate(data)
            ]
        
        else:
            return []
    
    def reset_metrics(self) -> None:
        """Reset all metrics and counters."""
        self.performance_metrics = PerformanceMetrics()
        self.health_metrics.critical_error_count = 0
        self.health_metrics.error_count_last_hour = 0
        self.health_metrics.recent_errors.clear()
        
        self.execution_times.clear()
        self.cpu_usage_history.clear()
        self.memory_usage_history.clear()
        self.task_completion_times.clear()
        self.error_history.clear()
        
        logger.info("Quantum task monitoring metrics reset")


# Default alert handlers
def console_alert_handler(alert: Dict[str, Any]) -> None:
    """Default console alert handler."""
    timestamp = alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] QUANTUM ALERT [{alert['severity'].upper()}] {alert['type']}: {alert['message']}")


def log_alert_handler(alert: Dict[str, Any]) -> None:
    """Default logging alert handler."""
    logger.warning(f"Quantum Alert [{alert['type']}]: {alert['message']}")


async def create_monitored_planner(monitoring_interval: float = 5.0, 
                                 enable_console_alerts: bool = True) -> tuple[QuantumTaskPlanner, QuantumTaskMonitor]:
    """Create a quantum task planner with monitoring enabled."""
    planner = QuantumTaskPlanner()
    monitor = QuantumTaskMonitor(planner, monitoring_interval)
    
    # Add default alert handlers
    if enable_console_alerts:
        monitor.add_alert_handler(console_alert_handler)
    monitor.add_alert_handler(log_alert_handler)
    
    # Start monitoring
    await monitor.start_monitoring()
    
    logger.info("Created monitored quantum task planner")
    return planner, monitor