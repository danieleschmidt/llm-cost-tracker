"""Alert webhook handlers for Prometheus Alertmanager integration."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


class AlertAnnotations(BaseModel):
    """Alert annotations model."""

    summary: str
    description: str
    action: Optional[str] = None
    runbook_url: Optional[str] = None


class AlertLabels(BaseModel):
    """Alert labels model."""

    alertname: str
    severity: str
    team: Optional[str] = None
    application: Optional[str] = None
    model: Optional[str] = None


class Alert(BaseModel):
    """Individual alert model."""

    status: str  # "firing" or "resolved"
    labels: AlertLabels
    annotations: AlertAnnotations
    starts_at: datetime = Field(alias="startsAt")
    ends_at: Optional[datetime] = Field(None, alias="endsAt")
    generator_url: str = Field(alias="generatorURL")
    fingerprint: str


class AlertWebhook(BaseModel):
    """Alertmanager webhook payload."""

    receiver: str
    status: str
    alerts: List[Alert]
    group_labels: Dict[str, str] = Field(alias="groupLabels")
    common_labels: Dict[str, str] = Field(alias="commonLabels")
    common_annotations: Dict[str, str] = Field(alias="commonAnnotations")
    external_url: str = Field(alias="externalURL")
    version: str
    group_key: str = Field(alias="groupKey")
    truncated_alerts: int = Field(0, alias="truncatedAlerts")


class AlertHandler:
    """Handles incoming alert webhooks and processes them."""

    def __init__(self):
        self.alert_history: List[Dict[str, Any]] = []

    async def process_webhook(self, webhook_data: AlertWebhook) -> Dict[str, Any]:
        """Process incoming alert webhook."""
        try:
            response = {
                "status": "processed",
                "timestamp": datetime.now().isoformat(),
                "alerts_processed": len(webhook_data.alerts),
                "group_key": webhook_data.group_key,
            }

            for alert in webhook_data.alerts:
                await self.handle_alert(alert, webhook_data)

            # Store in history (keep last 100 alerts)
            self.alert_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "webhook_data": webhook_data.model_dump(),
                    "processed": True,
                }
            )

            if len(self.alert_history) > 100:
                self.alert_history = self.alert_history[-100:]

            logger.info(
                f"Processed {len(webhook_data.alerts)} alerts from group {webhook_data.group_key}"
            )
            return response

        except Exception as e:
            logger.error(f"Failed to process alert webhook: {e}")
            raise HTTPException(status_code=500, detail=f"Alert processing failed: {e}")

    async def handle_alert(self, alert: Alert, webhook: AlertWebhook) -> None:
        """Handle individual alert based on type and severity."""
        try:
            alert_type = alert.labels.alertname
            severity = alert.labels.severity
            application = alert.labels.application or "unknown"

            logger.info(
                f"Processing {severity} alert: {alert_type} for application {application}"
            )

            # Log alert details
            if alert.status == "firing":
                logger.warning(f"🚨 FIRING: {alert.annotations.summary}")
                logger.warning(f"   Description: {alert.annotations.description}")
                if alert.annotations.action:
                    logger.warning(f"   Action: {alert.annotations.action}")
            else:
                logger.info(f"✅ RESOLVED: {alert.annotations.summary}")

            # Handle specific alert types
            if "Cost" in alert_type or "Budget" in alert_type:
                await self.handle_cost_alert(alert, webhook)
            elif "Latency" in alert_type or "Error" in alert_type:
                await self.handle_performance_alert(alert, webhook)
            elif "Traffic" in alert_type:
                await self.handle_traffic_alert(alert, webhook)

        except Exception as e:
            logger.error(f"Failed to handle alert {alert.labels.alertname}: {e}")

    async def handle_cost_alert(self, alert: Alert, webhook: AlertWebhook) -> None:
        """Handle cost-related alerts with specific logic."""
        if alert.status == "firing":
            if alert.labels.severity == "critical":
                logger.critical(f"💰 CRITICAL COST ALERT: {alert.annotations.summary}")
                # In production, this could trigger automatic cost controls
                # e.g., enable model switching, implement rate limiting, etc.
                await self.log_cost_alert_action(alert, "cost_control_triggered")
            else:
                logger.warning(f"💰 Cost warning: {alert.annotations.summary}")
                await self.log_cost_alert_action(alert, "cost_monitoring")

    async def handle_performance_alert(
        self, alert: Alert, webhook: AlertWebhook
    ) -> None:
        """Handle performance-related alerts."""
        if alert.status == "firing":
            logger.warning(f"⚡ Performance alert: {alert.annotations.summary}")
            # Could trigger automatic scaling or load balancing adjustments
            await self.log_performance_alert_action(alert, "performance_monitoring")

    async def handle_traffic_alert(self, alert: Alert, webhook: AlertWebhook) -> None:
        """Handle traffic anomaly alerts."""
        if alert.status == "firing":
            logger.warning(f"📈 Traffic alert: {alert.annotations.summary}")
            # Could trigger rate limiting or traffic analysis
            await self.log_traffic_alert_action(alert, "traffic_analysis")

    async def log_cost_alert_action(self, alert: Alert, action: str) -> None:
        """Log cost alert actions for audit trail."""
        logger.info(f"Cost alert action logged: {action} for {alert.labels.alertname}")
        # In production, this would write to audit log or metrics

    async def log_performance_alert_action(self, alert: Alert, action: str) -> None:
        """Log performance alert actions."""
        logger.info(
            f"Performance alert action logged: {action} for {alert.labels.alertname}"
        )

    async def log_traffic_alert_action(self, alert: Alert, action: str) -> None:
        """Log traffic alert actions."""
        logger.info(
            f"Traffic alert action logged: {action} for {alert.labels.alertname}"
        )

    def get_alert_history(self) -> List[Dict[str, Any]]:
        """Get recent alert history."""
        return self.alert_history[-20:]  # Last 20 alerts

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of recent alerts."""
        recent_alerts = self.alert_history[-50:]  # Last 50 alerts

        if not recent_alerts:
            return {"total": 0, "by_severity": {}, "by_type": {}}

        by_severity = {}
        by_type = {}

        for entry in recent_alerts:
            webhook_data = entry["webhook_data"]
            for alert in webhook_data.get("alerts", []):
                severity = alert.get("labels", {}).get("severity", "unknown")
                alert_type = alert.get("labels", {}).get("alertname", "unknown")

                by_severity[severity] = by_severity.get(severity, 0) + 1
                by_type[alert_type] = by_type.get(alert_type, 0) + 1

        return {
            "total": len(recent_alerts),
            "by_severity": by_severity,
            "by_type": by_type,
            "last_updated": recent_alerts[-1]["timestamp"] if recent_alerts else None,
        }


# Global alert handler instance
alert_handler = AlertHandler()


@router.post("/webhooks/alerts")
async def receive_alert_webhook(request: Request) -> Dict[str, Any]:
    """Receive and process Alertmanager webhooks."""
    try:
        payload = await request.json()
        webhook_data = AlertWebhook(**payload)

        result = await alert_handler.process_webhook(webhook_data)
        return result

    except Exception as e:
        logger.error(f"Alert webhook processing failed: {e}")
        # Return 200 to prevent Alertmanager retries on client errors
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@router.get("/webhooks/alerts/health")
async def alert_webhook_health() -> Dict[str, Any]:
    """Health check for alert webhook service."""
    return {
        "status": "healthy",
        "service": "alert-webhooks",
        "alerts_processed": len(alert_handler.alert_history),
    }


@router.get("/webhooks/alerts/history")
async def get_alert_history() -> Dict[str, Any]:
    """Get recent alert history."""
    return {
        "alerts": alert_handler.get_alert_history(),
        "summary": alert_handler.get_alert_summary(),
    }


@router.post("/webhooks/alerts/test")
async def test_alert_webhook() -> Dict[str, Any]:
    """Test endpoint for validating alert webhook functionality."""
    test_alert = AlertWebhook(
        receiver="test",
        status="firing",
        alerts=[
            Alert(
                status="firing",
                labels=AlertLabels(
                    alertname="TestAlert",
                    severity="warning",
                    team="ml-ops",
                    application="test-app",
                ),
                annotations=AlertAnnotations(
                    summary="Test alert for webhook validation",
                    description="This is a test alert to validate webhook functionality",
                    action="No action required - this is a test",
                ),
                starts_at=datetime.now(),
                generator_url="http://prometheus:9090/graph",
                fingerprint="test123",
            )
        ],
        group_labels={"alertname": "TestAlert"},
        common_labels={"severity": "warning"},
        common_annotations={"summary": "Test alert"},
        external_url="http://alertmanager:9093",
        version="4",
        group_key="test-group",
        truncated_alerts=0,
    )

    result = await alert_handler.process_webhook(test_alert)
    result["test"] = True
    return result
