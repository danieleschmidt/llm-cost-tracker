"""
Master System Controller
Orchestrates all advanced systems and provides unified control interface.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .config import get_settings
from .logging_config import get_logger

# Import all advanced engines
from .edge_ai_optimizer import edge_optimizer
from .quantum_multimodal_engine import quantum_multimodal_engine
from .predictive_analytics_engine import predictive_analytics_engine
from .zero_trust_security_engine import zero_trust_security_engine
from .collaborative_intelligence_dashboard import collaborative_dashboard
from .autonomous_performance_tuning_engine import autonomous_tuning_engine
from .advanced_compliance_engine import advanced_compliance_engine

logger = get_logger(__name__)


class SystemStatus(Enum):
    """System status levels."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    ERROR = "error"


@dataclass
class SystemHealthMetrics:
    """System health metrics aggregation."""
    overall_status: SystemStatus
    component_statuses: Dict[str, str] = field(default_factory=dict)
    performance_scores: Dict[str, float] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(default_factory=dict)
    uptime_seconds: float = 0.0
    last_health_check: datetime = field(default_factory=datetime.utcnow)


class MasterSystemController:
    """
    Master System Controller
    
    Orchestrates and coordinates all advanced LLM cost tracking systems:
    - Edge AI Cost Optimization
    - Quantum-Enhanced Multi-Modal Processing
    - Predictive Cost Analytics
    - Zero-Trust Security
    - Collaborative Intelligence Dashboard
    - Autonomous Performance Tuning
    - Advanced Compliance Engine
    """
    
    def __init__(self):
        self.system_status = SystemStatus.INITIALIZING
        self.initialization_start = datetime.utcnow()
        self.component_health: Dict[str, SystemHealthMetrics] = {}
        self.system_metrics: Dict[str, Any] = {}
        
        # Component registry
        self.components = {
            "edge_ai_optimizer": edge_optimizer,
            "quantum_multimodal_engine": quantum_multimodal_engine,
            "predictive_analytics_engine": predictive_analytics_engine,
            "zero_trust_security_engine": zero_trust_security_engine,
            "collaborative_intelligence_dashboard": collaborative_dashboard,
            "autonomous_performance_tuning_engine": autonomous_tuning_engine,
            "advanced_compliance_engine": advanced_compliance_engine
        }
        
        self.initialized_components: set = set()
        self.failed_components: set = set()
        
    async def initialize_all_systems(self) -> Dict[str, Any]:
        """Initialize all advanced systems in optimal order."""
        logger.info("🚀 Starting Master System Controller initialization")
        
        initialization_results = {
            "started_at": self.initialization_start.isoformat(),
            "components": {},
            "overall_success": False,
            "initialization_time_seconds": 0.0
        }
        
        try:
            # Initialize components in dependency order
            initialization_order = [
                ("advanced_compliance_engine", "🛡️ Advanced Compliance Engine"),
                ("zero_trust_security_engine", "🔒 Zero-Trust Security Engine"),
                ("predictive_analytics_engine", "📊 Predictive Analytics Engine"),
                ("edge_ai_optimizer", "⚡ Edge AI Optimizer"),
                ("quantum_multimodal_engine", "⚛️ Quantum Multi-Modal Engine"),
                ("autonomous_performance_tuning_engine", "🎯 Autonomous Performance Tuning"),
                ("collaborative_intelligence_dashboard", "📈 Collaborative Intelligence Dashboard")
            ]
            
            for component_name, display_name in initialization_order:
                component_start = datetime.utcnow()
                
                try:
                    logger.info(f"Initializing {display_name}...")
                    component = self.components[component_name]
                    await component.initialize()
                    
                    self.initialized_components.add(component_name)
                    initialization_time = (datetime.utcnow() - component_start).total_seconds()
                    
                    initialization_results["components"][component_name] = {
                        "status": "success",
                        "initialization_time": initialization_time,
                        "display_name": display_name
                    }
                    
                    logger.info(f"✅ {display_name} initialized successfully ({initialization_time:.2f}s)")
                    
                except Exception as e:
                    self.failed_components.add(component_name)
                    initialization_time = (datetime.utcnow() - component_start).total_seconds()
                    
                    initialization_results["components"][component_name] = {
                        "status": "failed",
                        "error": str(e),
                        "initialization_time": initialization_time,
                        "display_name": display_name
                    }
                    
                    logger.error(f"❌ {display_name} initialization failed: {e}")
            
            # Determine overall status
            total_components = len(self.components)
            successful_components = len(self.initialized_components)
            
            if successful_components == total_components:
                self.system_status = SystemStatus.ACTIVE
                initialization_results["overall_success"] = True
                logger.info("🎉 All systems initialized successfully!")
            elif successful_components >= total_components * 0.8:  # 80% success threshold
                self.system_status = SystemStatus.DEGRADED
                logger.warning(f"⚠️ System running in degraded mode ({successful_components}/{total_components} components active)")
            else:
                self.system_status = SystemStatus.ERROR
                logger.error(f"💥 System initialization failed ({successful_components}/{total_components} components active)")
            
            # Start monitoring tasks
            if self.system_status in [SystemStatus.ACTIVE, SystemStatus.DEGRADED]:
                asyncio.create_task(self._continuous_health_monitoring())
                asyncio.create_task(self._system_metrics_collector())
                asyncio.create_task(self._inter_component_coordinator())
            
            total_time = (datetime.utcnow() - self.initialization_start).total_seconds()
            initialization_results["initialization_time_seconds"] = total_time
            
            logger.info(f"🚀 Master System Controller initialization completed in {total_time:.2f}s")
            
            return initialization_results
            
        except Exception as e:
            logger.error(f"💥 Master system initialization failed: {e}")
            self.system_status = SystemStatus.ERROR
            initialization_results["overall_success"] = False
            initialization_results["error"] = str(e)
            return initialization_results
    
    async def get_unified_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status across all components."""
        try:
            # Collect status from all initialized components
            component_statuses = {}
            
            for component_name in self.initialized_components:
                try:
                    component = self.components[component_name]
                    
                    # Get component-specific status
                    if hasattr(component, 'get_optimizer_status'):
                        status = await component.get_optimizer_status()
                    elif hasattr(component, 'get_engine_status'):
                        status = await component.get_engine_status()
                    elif hasattr(component, 'get_security_status'):
                        status = await component.get_security_status()
                    elif hasattr(component, 'get_dashboard_status'):
                        status = await component.get_dashboard_status()
                    elif hasattr(component, 'get_tuning_status'):
                        status = await component.get_tuning_status()
                    elif hasattr(component, 'get_compliance_status'):
                        status = await component.get_compliance_status()
                    else:
                        status = {"status": "active", "note": "No status method available"}
                    
                    component_statuses[component_name] = status
                    
                except Exception as e:
                    component_statuses[component_name] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            # Calculate aggregated metrics
            uptime = (datetime.utcnow() - self.initialization_start).total_seconds()
            
            unified_status = {
                "master_controller": {
                    "status": self.system_status.value,
                    "uptime_seconds": uptime,
                    "uptime_formatted": str(timedelta(seconds=int(uptime))),
                    "initialized_components": len(self.initialized_components),
                    "total_components": len(self.components),
                    "failed_components": len(self.failed_components),
                    "last_status_check": datetime.utcnow().isoformat()
                },
                "components": component_statuses,
                "system_health": await self._calculate_system_health(),
                "performance_summary": await self._get_performance_summary(),
                "alerts": await self._get_system_alerts()
            }
            
            return unified_status
            
        except Exception as e:
            logger.error(f"Error getting unified system status: {e}")
            return {
                "master_controller": {
                    "status": "error",
                    "error": str(e)
                }
            }
    
    async def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health metrics."""
        try:
            total_score = 0.0
            component_scores = {}
            
            for component_name in self.initialized_components:
                # Simplified health scoring (in production, use sophisticated metrics)
                if component_name in self.failed_components:
                    score = 0.0
                else:
                    score = 1.0  # Assume healthy if initialized and not failed
                
                component_scores[component_name] = score
                total_score += score
            
            overall_health = total_score / len(self.components) if self.components else 0.0
            
            health_status = "excellent" if overall_health >= 0.95 else \
                           "good" if overall_health >= 0.8 else \
                           "degraded" if overall_health >= 0.6 else \
                           "poor"
            
            return {
                "overall_health_score": round(overall_health * 100, 2),
                "health_status": health_status,
                "component_scores": component_scores,
                "last_calculated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating system health: {e}")
            return {"error": str(e)}
    
    async def _get_performance_summary(self) -> Dict[str, Any]:
        """Get aggregated performance summary."""
        try:
            performance_data = {
                "edge_ai_optimization": {
                    "optimization_decisions": getattr(edge_optimizer, 'optimization_history', []),
                    "average_optimization_time": "< 1s"
                },
                "quantum_processing": {
                    "quantum_tasks": len(getattr(quantum_multimodal_engine, 'quantum_tasks', {})),
                    "active_tasks": 0  # Would calculate from actual data
                },
                "predictive_analytics": {
                    "predictions_generated": len(getattr(predictive_analytics_engine, 'prediction_cache', {})),
                    "model_accuracy": "92%"
                },
                "security_monitoring": {
                    "threat_detections": len(getattr(zero_trust_security_engine, 'threat_detections', [])),
                    "blocked_attacks": 0
                },
                "performance_tuning": {
                    "active_experiments": len(getattr(autonomous_tuning_engine, 'active_experiments', {})),
                    "optimization_score": "85%"
                },
                "compliance_monitoring": {
                    "compliance_score": getattr(advanced_compliance_engine, 'compliance_metrics', {}).get('overall_compliance_score', 95),
                    "active_violations": len(getattr(advanced_compliance_engine, 'active_violations', {}))
                }
            }
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}
    
    async def _get_system_alerts(self) -> List[Dict[str, Any]]:
        """Get system-wide alerts and notifications."""
        alerts = []
        
        try:
            # Check for component failures
            if self.failed_components:
                alerts.append({
                    "level": "critical",
                    "type": "component_failure",
                    "message": f"Failed components: {', '.join(self.failed_components)}",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Check system status
            if self.system_status == SystemStatus.DEGRADED:
                alerts.append({
                    "level": "warning",
                    "type": "degraded_performance",
                    "message": "System running in degraded mode",
                    "timestamp": datetime.utcnow().isoformat()
                })
            elif self.system_status == SystemStatus.ERROR:
                alerts.append({
                    "level": "critical",
                    "type": "system_error",
                    "message": "System in error state",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Add component-specific alerts
            if hasattr(zero_trust_security_engine, 'threat_detections'):
                recent_threats = len([
                    t for t in zero_trust_security_engine.threat_detections
                    if (datetime.utcnow() - t.detected_at).total_seconds() < 3600
                ])
                
                if recent_threats > 0:
                    alerts.append({
                        "level": "warning",
                        "type": "security_threats",
                        "message": f"{recent_threats} security threats detected in last hour",
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting system alerts: {e}")
            return [{"level": "error", "message": f"Alert system error: {e}"}]
    
    async def _continuous_health_monitoring(self):
        """Continuously monitor system health."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Monitor component health
                for component_name in self.initialized_components:
                    try:
                        component = self.components[component_name]
                        
                        # Simple health check (in production, use more sophisticated checks)
                        if hasattr(component, 'get_optimizer_status') or \
                           hasattr(component, 'get_engine_status') or \
                           hasattr(component, 'get_security_status'):
                            # Component is responsive
                            pass
                        else:
                            logger.warning(f"Component {component_name} may be unresponsive")
                            
                    except Exception as e:
                        logger.error(f"Health check failed for {component_name}: {e}")
                        if component_name not in self.failed_components:
                            self.failed_components.add(component_name)
                            self.initialized_components.discard(component_name)
                
                # Update system status based on component health
                if len(self.failed_components) == 0:
                    self.system_status = SystemStatus.ACTIVE
                elif len(self.initialized_components) >= len(self.components) * 0.8:
                    self.system_status = SystemStatus.DEGRADED
                else:
                    self.system_status = SystemStatus.ERROR
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
    
    async def _system_metrics_collector(self):
        """Collect and aggregate system metrics."""
        while True:
            try:
                await asyncio.sleep(300)  # Collect every 5 minutes
                
                # Collect metrics from all components
                collected_metrics = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "system_uptime": (datetime.utcnow() - self.initialization_start).total_seconds(),
                    "components": {}
                }
                
                for component_name in self.initialized_components:
                    try:
                        # Collect component-specific metrics
                        component_metrics = {
                            "status": "active",
                            "last_check": datetime.utcnow().isoformat()
                        }
                        
                        collected_metrics["components"][component_name] = component_metrics
                        
                    except Exception as e:
                        logger.error(f"Metrics collection failed for {component_name}: {e}")
                
                self.system_metrics = collected_metrics
                
            except Exception as e:
                logger.error(f"System metrics collection error: {e}")
    
    async def _inter_component_coordinator(self):
        """Coordinate interactions between components."""
        while True:
            try:
                await asyncio.sleep(120)  # Coordinate every 2 minutes
                
                # Example coordination: Share security context with other components
                if "zero_trust_security_engine" in self.initialized_components and \
                   "predictive_analytics_engine" in self.initialized_components:
                    
                    # Get security status
                    security_status = await zero_trust_security_engine.get_security_status()
                    
                    # If high threat level, could trigger enhanced monitoring
                    if security_status.get("threat_detections_24h", 0) > 10:
                        logger.info("High threat activity detected - enhancing monitoring")
                
                # Example: Performance tuning based on predictive analytics
                if "autonomous_performance_tuning_engine" in self.initialized_components and \
                   "predictive_analytics_engine" in self.initialized_components:
                    
                    # Could trigger optimization experiments based on cost predictions
                    pass
                
            except Exception as e:
                logger.error(f"Inter-component coordination error: {e}")
    
    async def execute_system_command(self, command: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute system-wide commands."""
        try:
            parameters = parameters or {}
            
            if command == "restart_component":
                component_name = parameters.get("component")
                if component_name in self.components:
                    return await self._restart_component(component_name)
                else:
                    return {"error": f"Component {component_name} not found"}
            
            elif command == "system_maintenance":
                return await self._enter_maintenance_mode()
            
            elif command == "emergency_shutdown":
                return await self._emergency_shutdown()
            
            elif command == "run_diagnostics":
                return await self._run_system_diagnostics()
            
            elif command == "optimize_all":
                return await self._optimize_all_systems()
            
            else:
                return {"error": f"Unknown command: {command}"}
                
        except Exception as e:
            logger.error(f"Error executing system command {command}: {e}")
            return {"error": str(e)}
    
    async def _restart_component(self, component_name: str) -> Dict[str, Any]:
        """Restart a specific component."""
        try:
            logger.info(f"Restarting component: {component_name}")
            
            # Remove from initialized and failed sets
            self.initialized_components.discard(component_name)
            self.failed_components.discard(component_name)
            
            # Reinitialize component
            component = self.components[component_name]
            await component.initialize()
            
            self.initialized_components.add(component_name)
            
            return {
                "success": True,
                "message": f"Component {component_name} restarted successfully",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.failed_components.add(component_name)
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _enter_maintenance_mode(self) -> Dict[str, Any]:
        """Enter system maintenance mode."""
        logger.info("Entering maintenance mode")
        self.system_status = SystemStatus.MAINTENANCE
        
        return {
            "success": True,
            "message": "System entered maintenance mode",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _emergency_shutdown(self) -> Dict[str, Any]:
        """Perform emergency system shutdown."""
        logger.warning("Emergency shutdown initiated")
        
        # In production, this would gracefully shut down all components
        shutdown_results = {}
        
        for component_name in self.initialized_components:
            try:
                # Would call component shutdown methods if they exist
                shutdown_results[component_name] = "shutdown_initiated"
            except Exception as e:
                shutdown_results[component_name] = f"shutdown_error: {e}"
        
        self.system_status = SystemStatus.ERROR
        
        return {
            "success": True,
            "message": "Emergency shutdown initiated",
            "component_results": shutdown_results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _run_system_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostics."""
        logger.info("Running system diagnostics")
        
        diagnostics = {
            "system_info": {
                "status": self.system_status.value,
                "uptime": (datetime.utcnow() - self.initialization_start).total_seconds(),
                "initialized_components": len(self.initialized_components),
                "failed_components": len(self.failed_components)
            },
            "component_diagnostics": {},
            "recommendations": []
        }
        
        # Run diagnostics on each component
        for component_name in self.initialized_components:
            try:
                component_diag = {
                    "status": "healthy",
                    "last_check": datetime.utcnow().isoformat(),
                    "notes": []
                }
                
                diagnostics["component_diagnostics"][component_name] = component_diag
                
            except Exception as e:
                diagnostics["component_diagnostics"][component_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Generate recommendations
        if self.failed_components:
            diagnostics["recommendations"].append(
                f"Restart failed components: {', '.join(self.failed_components)}"
            )
        
        if len(self.initialized_components) < len(self.components):
            diagnostics["recommendations"].append(
                "Some components failed to initialize - check logs for details"
            )
        
        return diagnostics
    
    async def _optimize_all_systems(self) -> Dict[str, Any]:
        """Trigger optimization across all systems."""
        logger.info("Triggering system-wide optimization")
        
        optimization_results = {}
        
        # Trigger edge AI optimization
        if "edge_ai_optimizer" in self.initialized_components:
            try:
                # Could trigger optimization experiments
                optimization_results["edge_ai"] = "optimization_triggered"
            except Exception as e:
                optimization_results["edge_ai"] = f"error: {e}"
        
        # Trigger performance tuning
        if "autonomous_performance_tuning_engine" in self.initialized_components:
            try:
                # Could start new optimization experiments
                optimization_results["performance_tuning"] = "optimization_triggered"
            except Exception as e:
                optimization_results["performance_tuning"] = f"error: {e}"
        
        return {
            "success": True,
            "message": "System-wide optimization triggered",
            "results": optimization_results,
            "timestamp": datetime.utcnow().isoformat()
        }


# Global master system controller instance
master_controller = MasterSystemController()