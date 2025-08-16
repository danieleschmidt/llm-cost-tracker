"""
Enhanced Monitoring and Alerting System

Comprehensive monitoring framework with:
- Real-time performance metrics
- Security event tracking
- Predictive anomaly detection
- Multi-channel alerting (Slack, email, webhooks)
- SLA monitoring and reporting
- Cost optimization insights
"""

import asyncio
import json
import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import aiohttp

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics being tracked."""

    PERFORMANCE = "performance"
    SECURITY = "security"
    BUSINESS = "business"
    SYSTEM = "system"
    COST = "cost"


@dataclass
class MetricPoint:
    """Individual metric measurement."""

    timestamp: datetime
    value: float
    metric_name: str
    metric_type: MetricType
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert definition and state."""

    id: str
    name: str
    description: str
    severity: AlertSeverity
    condition: str  # Human readable condition
    threshold_value: float
    current_value: float
    metric_name: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledgment_required: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)
    runbook_url: Optional[str] = None

    @property
    def is_active(self) -> bool:
        return self.resolved_at is None

    @property
    def duration(self) -> timedelta:
        end_time = self.resolved_at or datetime.now()
        return end_time - self.triggered_at


class AnomalyDetector:
    """Predictive anomaly detection using statistical methods."""

    def __init__(self, window_size: int = 100, sensitivity: float = 2.0):
        self.window_size = window_size
        self.sensitivity = sensitivity  # Standard deviations for anomaly threshold
        self.metric_history: Dict[str, deque] = {}
        self.baselines: Dict[str, Dict[str, float]] = {}
        self.last_baseline_update: Dict[str, datetime] = {}

    def add_metric(self, metric: MetricPoint):
        """Add metric point and update baselines."""
        metric_key = f"{metric.metric_name}_{metric.metric_type.value}"

        if metric_key not in self.metric_history:
            self.metric_history[metric_key] = deque(maxlen=self.window_size)
            self.baselines[metric_key] = {}
            self.last_baseline_update[metric_key] = datetime.now()

        self.metric_history[metric_key].append(metric)

        # Update baseline every 10 minutes or when we have enough data
        now = datetime.now()
        should_update = len(self.metric_history[metric_key]) >= 20 and (
            metric_key not in self.last_baseline_update
            or now - self.last_baseline_update[metric_key] > timedelta(minutes=10)
        )

        if should_update:
            self._update_baseline(metric_key)

    def _update_baseline(self, metric_key: str):
        """Update statistical baseline for a metric."""
        if metric_key not in self.metric_history:
            return

        values = [point.value for point in self.metric_history[metric_key]]
        if len(values) < 10:
            return

        # Calculate statistical measures
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        median = statistics.median(values)

        # Calculate recent trend (last 20% of values)
        recent_size = max(5, len(values) // 5)
        recent_values = values[-recent_size:]
        recent_mean = statistics.mean(recent_values)

        trend = (recent_mean - mean) / mean if mean != 0 else 0.0

        self.baselines[metric_key] = {
            "mean": mean,
            "stdev": stdev,
            "median": median,
            "upper_threshold": mean + (self.sensitivity * stdev),
            "lower_threshold": max(0, mean - (self.sensitivity * stdev)),
            "trend": trend,
            "sample_count": len(values),
            "last_updated": datetime.now().isoformat(),
        }

        self.last_baseline_update[metric_key] = datetime.now()

        logger.debug(
            f"Updated baseline for {metric_key}: mean={mean:.2f}, stdev={stdev:.2f}"
        )

    def detect_anomaly(self, metric: MetricPoint) -> Optional[Dict[str, Any]]:
        """Detect if a metric point is anomalous."""
        metric_key = f"{metric.metric_name}_{metric.metric_type.value}"

        if metric_key not in self.baselines or not self.baselines[metric_key]:
            return None  # No baseline yet

        baseline = self.baselines[metric_key]
        value = metric.value

        # Check for statistical anomalies
        anomaly_info = {
            "metric_key": metric_key,
            "value": value,
            "baseline_mean": baseline["mean"],
            "baseline_stdev": baseline["stdev"],
            "anomaly_types": [],
            "severity_score": 0.0,
            "confidence": 0.0,
        }

        # Statistical outlier detection
        if value > baseline["upper_threshold"]:
            anomaly_info["anomaly_types"].append("high_outlier")
            deviation = (
                (value - baseline["mean"]) / baseline["stdev"]
                if baseline["stdev"] > 0
                else 0
            )
            anomaly_info["severity_score"] += min(10.0, deviation)
        elif value < baseline["lower_threshold"] and baseline["lower_threshold"] > 0:
            anomaly_info["anomaly_types"].append("low_outlier")
            deviation = (
                (baseline["mean"] - value) / baseline["stdev"]
                if baseline["stdev"] > 0
                else 0
            )
            anomaly_info["severity_score"] += min(10.0, deviation)

        # Trend-based detection
        if baseline["trend"] > 0.2:  # 20% upward trend
            if value > baseline["mean"] * 1.5:
                anomaly_info["anomaly_types"].append("trend_acceleration")
                anomaly_info["severity_score"] += 3.0
        elif baseline["trend"] < -0.2:  # 20% downward trend
            if value < baseline["mean"] * 0.5:
                anomaly_info["anomaly_types"].append("trend_deceleration")
                anomaly_info["severity_score"] += 3.0

        # Calculate confidence based on sample size and stability
        confidence = min(
            1.0, baseline["sample_count"] / 50.0
        )  # Higher confidence with more samples
        if baseline["stdev"] > 0:
            stability_factor = min(
                1.0, baseline["mean"] / baseline["stdev"]
            )  # More stable = higher confidence
            confidence *= stability_factor

        anomaly_info["confidence"] = confidence

        # Return anomaly if detected with sufficient confidence
        if anomaly_info["anomaly_types"] and anomaly_info["confidence"] > 0.3:
            return anomaly_info

        return None


class EnhancedMonitoringSystem:
    """
    Comprehensive monitoring system with real-time analytics and alerting.
    """

    def __init__(self):
        self.metrics_buffer: List[MetricPoint] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.anomaly_detector = AnomalyDetector()

        # Alert rules and thresholds
        self.alert_rules: Dict[str, Dict[str, Any]] = {
            "high_response_time": {
                "metric_name": "response_time_ms",
                "threshold": 1000.0,
                "operator": "greater_than",
                "severity": AlertSeverity.WARNING,
                "duration_threshold": 60,  # seconds
                "description": "API response time is elevated",
            },
            "critical_response_time": {
                "metric_name": "response_time_ms",
                "threshold": 5000.0,
                "operator": "greater_than",
                "severity": AlertSeverity.CRITICAL,
                "duration_threshold": 30,
                "description": "API response time is critically high",
            },
            "high_error_rate": {
                "metric_name": "error_rate_percent",
                "threshold": 5.0,
                "operator": "greater_than",
                "severity": AlertSeverity.ERROR,
                "duration_threshold": 120,
                "description": "Error rate is above acceptable threshold",
            },
            "memory_usage_high": {
                "metric_name": "memory_usage_percent",
                "threshold": 85.0,
                "operator": "greater_than",
                "severity": AlertSeverity.WARNING,
                "duration_threshold": 300,
                "description": "Memory usage is high",
            },
            "memory_usage_critical": {
                "metric_name": "memory_usage_percent",
                "threshold": 95.0,
                "operator": "greater_than",
                "severity": AlertSeverity.CRITICAL,
                "duration_threshold": 60,
                "description": "Memory usage is critically high",
            },
            "security_threat_detected": {
                "metric_name": "security_violations",
                "threshold": 10.0,
                "operator": "greater_than",
                "severity": AlertSeverity.ERROR,
                "duration_threshold": 0,  # Immediate
                "description": "Security threats detected",
            },
            "cost_budget_exceeded": {
                "metric_name": "cost_usage_percent",
                "threshold": 90.0,
                "operator": "greater_than",
                "severity": AlertSeverity.WARNING,
                "duration_threshold": 0,
                "description": "Cost budget nearly exceeded",
            },
        }

        # Notification channels
        self.notification_channels: Dict[str, Dict[str, Any]] = {}

        # SLA tracking
        self.sla_metrics: Dict[str, Dict[str, Any]] = {
            "availability": {
                "target": 99.9,  # 99.9% uptime
                "current": 100.0,
                "violations": [],
            },
            "response_time": {
                "target": 500.0,  # 500ms average
                "current": 0.0,
                "violations": [],
            },
            "error_rate": {
                "target": 1.0,  # 1% error rate
                "current": 0.0,
                "violations": [],
            },
        }

        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_running = False

    async def start_monitoring(self):
        """Start the monitoring system."""
        if self._is_running:
            return

        self._is_running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Enhanced monitoring system started")

    async def stop_monitoring(self):
        """Stop the monitoring system."""
        self._is_running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Enhanced monitoring system stopped")

    def record_metric(
        self,
        metric_name: str,
        value: float,
        metric_type: MetricType,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a metric measurement."""
        metric = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            metric_name=metric_name,
            metric_type=metric_type,
            tags=tags or {},
            metadata=metadata or {},
        )

        self.metrics_buffer.append(metric)

        # Keep buffer size manageable
        if len(self.metrics_buffer) > 10000:
            self.metrics_buffer = self.metrics_buffer[-5000:]  # Keep last 5000

        # Add to anomaly detector
        self.anomaly_detector.add_metric(metric)

        # Check for anomalies
        anomaly = self.anomaly_detector.detect_anomaly(metric)
        if anomaly:
            self._handle_anomaly_detection(metric, anomaly)

        # Check alert rules
        self._evaluate_alert_rules(metric)

    def _handle_anomaly_detection(self, metric: MetricPoint, anomaly: Dict[str, Any]):
        """Handle detected anomaly."""
        alert_id = f"anomaly_{metric.metric_name}_{int(time.time())}"

        # Determine severity based on anomaly score
        if anomaly["severity_score"] >= 8.0:
            severity = AlertSeverity.CRITICAL
        elif anomaly["severity_score"] >= 5.0:
            severity = AlertSeverity.ERROR
        elif anomaly["severity_score"] >= 3.0:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO

        alert = Alert(
            id=alert_id,
            name=f"Anomaly Detected: {metric.metric_name}",
            description=f"Statistical anomaly detected in {metric.metric_name}. "
            f"Types: {', '.join(anomaly['anomaly_types'])}. "
            f"Value: {metric.value:.2f}, Baseline: {anomaly['baseline_mean']:.2f}",
            severity=severity,
            condition=f"{metric.metric_name} anomaly score > 3.0",
            threshold_value=3.0,
            current_value=anomaly["severity_score"],
            metric_name=metric.metric_name,
            triggered_at=datetime.now(),
            tags={"anomaly_confidence": str(round(anomaly["confidence"], 2))},
        )

        self._trigger_alert(alert)

    def _evaluate_alert_rules(self, metric: MetricPoint):
        """Evaluate alert rules against the metric."""
        for rule_name, rule in self.alert_rules.items():
            if rule["metric_name"] != metric.metric_name:
                continue

            # Check if threshold is breached
            threshold_breached = False
            if rule["operator"] == "greater_than":
                threshold_breached = metric.value > rule["threshold"]
            elif rule["operator"] == "less_than":
                threshold_breached = metric.value < rule["threshold"]
            elif rule["operator"] == "equals":
                threshold_breached = abs(metric.value - rule["threshold"]) < 0.001

            if threshold_breached:
                # Check if alert already exists and is active
                existing_alert_id = f"{rule_name}_{metric.metric_name}"
                if existing_alert_id not in self.active_alerts:
                    # Create new alert
                    alert = Alert(
                        id=existing_alert_id,
                        name=rule_name.replace("_", " ").title(),
                        description=rule["description"],
                        severity=rule["severity"],
                        condition=f"{metric.metric_name} {rule['operator']} {rule['threshold']}",
                        threshold_value=rule["threshold"],
                        current_value=metric.value,
                        metric_name=metric.metric_name,
                        triggered_at=datetime.now(),
                        tags={"rule": rule_name},
                    )

                    self._trigger_alert(alert)
                else:
                    # Update existing alert with current value
                    self.active_alerts[existing_alert_id].current_value = metric.value
            else:
                # Check if we should resolve an existing alert
                existing_alert_id = f"{rule_name}_{metric.metric_name}"
                if existing_alert_id in self.active_alerts:
                    self._resolve_alert(existing_alert_id)

    def _trigger_alert(self, alert: Alert):
        """Trigger a new alert."""
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)

        logger.warning(f"Alert triggered: {alert.name} - {alert.description}")

        # Send notifications asynchronously
        asyncio.create_task(self._send_alert_notifications(alert))

        # Update SLA metrics if applicable
        self._update_sla_metrics(alert)

    def _resolve_alert(self, alert_id: str):
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved_at = datetime.now()
            del self.active_alerts[alert_id]

            logger.info(f"Alert resolved: {alert.name} (duration: {alert.duration})")

            # Send resolution notification
            asyncio.create_task(self._send_resolution_notification(alert))

    async def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications through configured channels."""
        for channel_name, channel_config in self.notification_channels.items():
            try:
                if channel_config["type"] == "slack":
                    await self._send_slack_notification(alert, channel_config)
                elif channel_config["type"] == "webhook":
                    await self._send_webhook_notification(alert, channel_config)
                elif channel_config["type"] == "email":
                    await self._send_email_notification(alert, channel_config)
            except Exception as e:
                logger.error(f"Failed to send notification via {channel_name}: {e}")

    async def _send_resolution_notification(self, alert: Alert):
        """Send alert resolution notifications."""
        resolution_message = f"🟢 Alert Resolved: {alert.name}\nDuration: {alert.duration}\nResolved at: {alert.resolved_at}"

        for channel_name, channel_config in self.notification_channels.items():
            try:
                if channel_config["type"] == "slack":
                    await self._send_slack_message(resolution_message, channel_config)
            except Exception as e:
                logger.error(
                    f"Failed to send resolution notification via {channel_name}: {e}"
                )

    async def _send_slack_notification(self, alert: Alert, channel_config: Dict):
        """Send Slack notification for alert."""
        severity_emojis = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨",
        }

        emoji = severity_emojis.get(alert.severity, "📊")

        message = f"{emoji} *{alert.severity.value.upper()}* Alert: {alert.name}\n"
        message += f"📝 {alert.description}\n"
        message += f"📈 Current Value: {alert.current_value:.2f}\n"
        message += f"🎯 Threshold: {alert.threshold_value:.2f}\n"
        message += f"⏰ Triggered: {alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S')}\n"

        if alert.runbook_url:
            message += f"📖 Runbook: {alert.runbook_url}\n"

        await self._send_slack_message(message, channel_config)

    async def _send_slack_message(self, message: str, channel_config: Dict):
        """Send message to Slack."""
        webhook_url = channel_config.get("webhook_url")
        if not webhook_url:
            return

        payload = {
            "text": message,
            "username": "LLM Cost Tracker",
            "icon_emoji": ":robot_face:",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Slack notification failed: {response.status}")

    async def _send_webhook_notification(self, alert: Alert, channel_config: Dict):
        """Send webhook notification for alert."""
        webhook_url = channel_config.get("url")
        if not webhook_url:
            return

        payload = {
            "alert_id": alert.id,
            "name": alert.name,
            "description": alert.description,
            "severity": alert.severity.value,
            "current_value": alert.current_value,
            "threshold_value": alert.threshold_value,
            "metric_name": alert.metric_name,
            "triggered_at": alert.triggered_at.isoformat(),
            "tags": alert.tags,
        }

        headers = channel_config.get("headers", {})

        async with aiohttp.ClientSession() as session:
            async with session.post(
                webhook_url, json=payload, headers=headers
            ) as response:
                if response.status not in [200, 201, 202]:
                    logger.error(f"Webhook notification failed: {response.status}")

    async def _send_email_notification(self, alert: Alert, channel_config: Dict):
        """Send email notification for alert."""
        # Email implementation would go here
        # For now, just log the notification
        logger.info(
            f"Email notification (simulated): {alert.name} to {channel_config.get('recipients')}"
        )

    def _update_sla_metrics(self, alert: Alert):
        """Update SLA metrics based on alert."""
        if alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            if alert.metric_name in ["response_time_ms", "error_rate_percent"]:
                sla_key = (
                    "response_time"
                    if "response_time" in alert.metric_name
                    else "error_rate"
                )
                if sla_key in self.sla_metrics:
                    self.sla_metrics[sla_key]["violations"].append(
                        {
                            "timestamp": alert.triggered_at.isoformat(),
                            "value": alert.current_value,
                            "severity": alert.severity.value,
                        }
                    )

    async def _monitoring_loop(self):
        """Main monitoring loop for background processing."""
        while self._is_running:
            try:
                # Process metrics buffer
                await self._process_metrics_buffer()

                # Update SLA calculations
                self._calculate_sla_metrics()

                # Clean up old data
                self._cleanup_old_data()

                # Wait before next iteration
                await asyncio.sleep(30)  # Run every 30 seconds

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _process_metrics_buffer(self):
        """Process buffered metrics for aggregation and analysis."""
        if not self.metrics_buffer:
            return

        # Group metrics by name and type for aggregation
        metric_groups: Dict[str, List[MetricPoint]] = {}

        for metric in self.metrics_buffer[-1000:]:  # Process last 1000 metrics
            key = f"{metric.metric_name}_{metric.metric_type.value}"
            if key not in metric_groups:
                metric_groups[key] = []
            metric_groups[key].append(metric)

        # Calculate aggregated metrics
        for key, metrics in metric_groups.items():
            if len(metrics) > 5:  # Only aggregate if we have enough data
                values = [m.value for m in metrics]

                # Record aggregated metrics
                self.record_metric(
                    f"{key}_avg",
                    statistics.mean(values),
                    MetricType.SYSTEM,
                    tags={"aggregated": "true", "sample_count": str(len(values))},
                )

                if len(values) > 1:
                    self.record_metric(
                        f"{key}_p95",
                        sorted(values)[int(len(values) * 0.95)],
                        MetricType.SYSTEM,
                        tags={"aggregated": "true", "percentile": "95"},
                    )

    def _calculate_sla_metrics(self):
        """Calculate current SLA metrics."""
        # This would integrate with actual system metrics
        # For now, we'll use mock calculations based on recent metrics
        current_time = datetime.now()

        # Calculate availability (simplified)
        recent_errors = [
            m
            for m in self.metrics_buffer[-100:]
            if m.metric_name == "error_rate_percent" and m.value > 0
        ]

        if recent_errors:
            error_rate = statistics.mean([m.value for m in recent_errors])
            availability = max(0, 100 - error_rate)
            self.sla_metrics["availability"]["current"] = availability

        # Calculate average response time
        recent_response_times = [
            m for m in self.metrics_buffer[-100:] if m.metric_name == "response_time_ms"
        ]

        if recent_response_times:
            avg_response_time = statistics.mean(
                [m.value for m in recent_response_times]
            )
            self.sla_metrics["response_time"]["current"] = avg_response_time

    def _cleanup_old_data(self):
        """Clean up old monitoring data."""
        cutoff_time = datetime.now() - timedelta(days=7)

        # Clean old alert history
        self.alert_history = [
            alert for alert in self.alert_history if alert.triggered_at > cutoff_time
        ]

        # Clean old metrics buffer
        self.metrics_buffer = [
            metric for metric in self.metrics_buffer if metric.timestamp > cutoff_time
        ]

        # Clean old SLA violations
        for sla_name, sla_data in self.sla_metrics.items():
            sla_data["violations"] = [
                violation
                for violation in sla_data["violations"]
                if datetime.fromisoformat(violation["timestamp"]) > cutoff_time
            ]

    def add_notification_channel(
        self, name: str, channel_type: str, config: Dict[str, Any]
    ):
        """Add a notification channel."""
        self.notification_channels[name] = {"type": channel_type, **config}
        logger.info(f"Added notification channel: {name} ({channel_type})")

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        return {
            "active_alerts": len(self.active_alerts),
            "total_alerts_today": len(
                [
                    a
                    for a in self.alert_history
                    if a.triggered_at > datetime.now() - timedelta(days=1)
                ]
            ),
            "metrics_buffer_size": len(self.metrics_buffer),
            "sla_metrics": self.sla_metrics,
            "notification_channels": len(self.notification_channels),
            "anomaly_detector_baselines": len(self.anomaly_detector.baselines),
            "is_running": self._is_running,
            "last_updated": datetime.now().isoformat(),
        }

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now()
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True
        return False
