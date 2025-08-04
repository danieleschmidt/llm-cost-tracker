"""Health check system for monitoring service health."""

import asyncio
import time
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status values."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['status'] = self.status.value
        result['timestamp'] = self.timestamp.isoformat()
        return result


class HealthCheck:
    """Individual health check implementation."""
    
    def __init__(
        self,
        name: str,
        check_function: Callable,
        timeout: float = 5.0,
        critical: bool = True,
        interval: float = 30.0
    ):
        """
        Initialize health check.
        
        Args:
            name: Name of the health check
            check_function: Async function that performs the check
            timeout: Timeout for the check in seconds
            critical: Whether this check is critical for overall health
            interval: Interval between checks in seconds
        """
        self.name = name
        self.check_function = check_function
        self.timeout = timeout
        self.critical = critical
        self.interval = interval
        self.last_result: Optional[HealthCheckResult] = None
        self.last_check_time = 0.0
    
    async def run(self) -> HealthCheckResult:
        """Run the health check."""
        start_time = time.time()
        
        try:
            # Run check with timeout
            await asyncio.wait_for(
                self.check_function(),
                timeout=self.timeout
            )
            
            duration_ms = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Check passed",
                duration_ms=duration_ms,
                timestamp=datetime.utcnow()
            )
            
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check timed out after {self.timeout}s",
                duration_ms=duration_ms,
                timestamp=datetime.utcnow()
            )
            logger.error(f"Health check '{self.name}' timed out")
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
                details={"exception": str(e), "type": type(e).__name__}
            )
            logger.error(f"Health check '{self.name}' failed: {e}")
        
        self.last_result = result
        self.last_check_time = time.time()
        return result
    
    def should_run(self) -> bool:
        """Check if this health check should run based on interval."""
        return time.time() - self.last_check_time >= self.interval


class HealthChecker:
    """Main health checker that manages multiple health checks."""
    
    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.overall_status = HealthStatus.UNKNOWN
        self.last_overall_check = 0.0
        
    def register_check(self, health_check: HealthCheck) -> None:
        """Register a health check."""
        self.checks[health_check.name] = health_check
        logger.info(f"Registered health check: {health_check.name}")
    
    def unregister_check(self, name: str) -> bool:
        """Unregister a health check."""
        if name in self.checks:
            del self.checks[name]
            logger.info(f"Unregistered health check: {name}")
            return True
        return False
    
    async def run_check(self, name: str) -> Optional[HealthCheckResult]:
        """Run a specific health check."""
        if check := self.checks.get(name):
            return await check.run()
        return None
    
    async def run_all_checks(self, force: bool = False) -> Dict[str, HealthCheckResult]:
        """Run all health checks."""
        results = {}
        
        # Run checks that need to be run
        tasks = []
        for name, check in self.checks.items():
            if force or check.should_run():
                tasks.append(self._run_single_check(name, check))
        
        if tasks:
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in completed_results:
                if isinstance(result, HealthCheckResult):
                    results[result.name] = result
                elif isinstance(result, Exception):
                    logger.error(f"Health check failed with exception: {result}")
        
        # Include cached results for checks that didn't run
        for name, check in self.checks.items():
            if name not in results and check.last_result:
                results[name] = check.last_result
        
        return results
    
    async def _run_single_check(self, name: str, check: HealthCheck) -> HealthCheckResult:
        """Run a single health check with error handling."""
        try:
            return await check.run()
        except Exception as e:
            logger.error(f"Error running health check '{name}': {e}")
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check execution failed: {str(e)}",
                duration_ms=0.0,
                timestamp=datetime.utcnow()
            )
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        results = await self.run_all_checks()
        
        # Determine overall status
        overall_status = HealthStatus.HEALTHY
        critical_failures = 0
        total_failures = 0
        
        for result in results.values():
            check = self.checks[result.name]
            
            if result.status == HealthStatus.UNHEALTHY:
                total_failures += 1
                if check.critical:
                    critical_failures += 1
            elif result.status == HealthStatus.DEGRADED:
                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
        
        # Set overall status based on failures
        if critical_failures > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif total_failures > 0 and overall_status == HealthStatus.HEALTHY:
            overall_status = HealthStatus.DEGRADED
        
        self.overall_status = overall_status
        self.last_overall_check = time.time()
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {name: result.to_dict() for name, result in results.items()},
            "summary": {
                "total_checks": len(results),
                "healthy_checks": sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY),
                "unhealthy_checks": sum(1 for r in results.values() if r.status == HealthStatus.UNHEALTHY),
                "degraded_checks": sum(1 for r in results.values() if r.status == HealthStatus.DEGRADED),
                "critical_failures": critical_failures,
                "total_failures": total_failures
            }
        }


# Pre-defined health checks
async def database_health_check():
    """Check database connectivity."""
    from .database import get_db_manager
    db_manager = get_db_manager()
    
    if not db_manager.pool:
        raise Exception("Database pool not initialized")
    
    # Test a simple query
    async with db_manager.pool.acquire() as conn:
        await conn.execute("SELECT 1")


async def memory_health_check():
    """Check memory usage."""
    import psutil
    
    # Get memory usage
    memory = psutil.virtual_memory()
    
    if memory.percent > 90:
        raise Exception(f"High memory usage: {memory.percent}%")
    elif memory.percent > 80:
        # This would return degraded status in a more sophisticated implementation
        logger.warning(f"Memory usage getting high: {memory.percent}%")


async def disk_health_check():
    """Check disk space."""
    import psutil
    
    # Check disk usage for root partition
    disk = psutil.disk_usage('/')
    
    if disk.percent > 95:
        raise Exception(f"Disk space critically low: {disk.percent}%")
    elif disk.percent > 85:
        logger.warning(f"Disk space getting low: {disk.percent}%")


# Global health checker instance
health_checker = HealthChecker()

# Register default health checks
health_checker.register_check(HealthCheck(
    name="database",
    check_function=database_health_check,
    timeout=5.0,
    critical=True,
    interval=30.0
))

health_checker.register_check(HealthCheck(
    name="memory",
    check_function=memory_health_check,
    timeout=2.0,
    critical=False,
    interval=60.0
))

health_checker.register_check(HealthCheck(
    name="disk",
    check_function=disk_health_check,
    timeout=2.0,
    critical=False,
    interval=300.0  # Check every 5 minutes
))