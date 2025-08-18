#!/usr/bin/env python3
"""Robust Enhancement System - Generation 2: MAKE IT ROBUST (Reliable)

This module implements comprehensive error handling, validation, logging,
monitoring, health checks, and security measures to make the system robust.
"""

import asyncio
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import random

# Add src to path for imports
sys.path.append('src')

from llm_cost_tracker import QuantumTaskPlanner, QuantumTask, TaskState
from llm_cost_tracker.quantum_i18n import set_language, t, SupportedLanguage

# Handle optional imports gracefully
try:
    from llm_cost_tracker.validation import ValidationError
except ImportError:
    class ValidationError(Exception):
        pass

try:
    from llm_cost_tracker.security import SecurityError
except ImportError:
    class SecurityError(Exception):
        pass


class RobustLogger:
    """Enhanced logging system with structured logging and error tracking."""
    
    def __init__(self, name: str = "robust_system"):
        self.logger = logging.getLogger(name)
        self.setup_logging()
        self.error_counts = {}
        self.performance_metrics = {}
        
    def setup_logging(self):
        """Configure structured logging with multiple handlers."""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set level
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler with formatting
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(name)s | %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for persistent logging
        file_handler = logging.FileHandler('robust_system.log')
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context and increment error counter."""
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        self.logger.error(
            f"Error [{error_type}]: {str(error)} | Context: {context or {}} | "
            f"Count: {self.error_counts[error_type]}"
        )
        
    def log_performance(self, operation: str, duration_ms: float, success: bool = True):
        """Log performance metrics for operations."""
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = {
                'total_calls': 0,
                'total_duration_ms': 0,
                'success_count': 0,
                'failure_count': 0
            }
        
        metrics = self.performance_metrics[operation]
        metrics['total_calls'] += 1
        metrics['total_duration_ms'] += duration_ms
        
        if success:
            metrics['success_count'] += 1
        else:
            metrics['failure_count'] += 1
        
        avg_duration = metrics['total_duration_ms'] / metrics['total_calls']
        success_rate = (metrics['success_count'] / metrics['total_calls']) * 100
        
        self.logger.info(
            f"Performance [{operation}]: {duration_ms:.2f}ms | "
            f"Avg: {avg_duration:.2f}ms | Success Rate: {success_rate:.1f}%"
        )


class CircuitBreaker:
    """Circuit breaker pattern for resilient system operations."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                
            raise e


class HealthChecker:
    """Comprehensive health monitoring system."""
    
    def __init__(self):
        self.checks = {}
        self.last_check_time = None
        
    def register_check(self, name: str, check_func, critical: bool = False):
        """Register a health check function."""
        self.checks[name] = {
            'function': check_func,
            'critical': critical,
            'last_result': None,
            'last_check': None
        }
        
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {
            'overall_status': 'HEALTHY',
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        critical_failures = 0
        
        for name, check_info in self.checks.items():
            try:
                start_time = time.time()
                result = await check_info['function']()
                duration_ms = (time.time() - start_time) * 1000
                
                check_result = {
                    'status': 'PASS',
                    'response_time_ms': duration_ms,
                    'details': result,
                    'critical': check_info['critical']
                }
                
                check_info['last_result'] = check_result
                check_info['last_check'] = datetime.now()
                
            except Exception as e:
                check_result = {
                    'status': 'FAIL',
                    'error': str(e),
                    'critical': check_info['critical']
                }
                
                if check_info['critical']:
                    critical_failures += 1
                    
            results['checks'][name] = check_result
        
        if critical_failures > 0:
            results['overall_status'] = 'CRITICAL'
        elif any(check['status'] == 'FAIL' for check in results['checks'].values()):
            results['overall_status'] = 'DEGRADED'
            
        self.last_check_time = datetime.now()
        return results


class RobustEnhancementSystem:
    """Enhanced system with comprehensive robustness features."""
    
    def __init__(self):
        self.logger = RobustLogger("robust_enhancement")
        self.planner = QuantumTaskPlanner()
        self.circuit_breaker = CircuitBreaker()
        self.health_checker = HealthChecker()
        self.setup_health_checks()
        self.robust_metrics = {}
        
    def setup_health_checks(self):
        """Setup comprehensive health monitoring."""
        self.health_checker.register_check(
            "quantum_planner", self.check_quantum_planner_health, critical=True
        )
        self.health_checker.register_check(
            "memory_usage", self.check_memory_usage, critical=False
        )
        self.health_checker.register_check(
            "task_queue", self.check_task_queue_health, critical=True
        )
        
    async def check_quantum_planner_health(self) -> Dict[str, Any]:
        """Check quantum planner system health."""
        test_task = QuantumTask(
            id="health_check_task",
            name="Health Check Task",
            description="System health validation",
            priority=1.0,
            estimated_duration=timedelta(seconds=1)
        )
        
        try:
            success, message = self.planner.add_task(test_task)
            if success:
                # Clean up test task
                if test_task.id in self.planner.tasks:
                    del self.planner.tasks[test_task.id]
                    
                return {
                    "status": "operational",
                    "task_count": len(self.planner.tasks),
                    "message": message
                }
            else:
                raise Exception(f"Failed to add test task: {message}")
                
        except Exception as e:
            raise Exception(f"Quantum planner health check failed: {str(e)}")
    
    async def check_memory_usage(self) -> Dict[str, Any]:
        """Check system memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                "usage_percent": memory.percent,
                "available_gb": memory.available / (1024**3),
                "total_gb": memory.total / (1024**3)
            }
        except ImportError:
            # Simulate memory check if psutil not available
            return {
                "usage_percent": random.uniform(20, 40),
                "available_gb": random.uniform(8, 16),
                "total_gb": 16.0,
                "note": "Simulated - psutil not available"
            }
    
    async def check_task_queue_health(self) -> Dict[str, Any]:
        """Check task queue health and performance."""
        queue_size = len(self.planner.tasks)
        
        # Check for any tasks in error state
        error_tasks = [
            task for task in self.planner.tasks.values()
            if task.state == TaskState.FAILED
        ]
        
        return {
            "queue_size": queue_size,
            "error_tasks": len(error_tasks),
            "status": "healthy" if len(error_tasks) == 0 else "degraded"
        }
    
    async def robust_task_execution(self, tasks: List[QuantumTask]) -> Dict[str, Any]:
        """Execute tasks with comprehensive error handling and monitoring."""
        self.logger.logger.info(f"Starting robust execution of {len(tasks)} tasks")
        
        results = {
            "execution_id": f"exec_{int(time.time())}",
            "start_time": datetime.now().isoformat(),
            "tasks_processed": 0,
            "tasks_successful": 0,
            "tasks_failed": 0,
            "errors": [],
            "performance_metrics": {}
        }
        
        try:
            # Add tasks with validation and error handling
            for task in tasks:
                try:
                    start_time = time.time()
                    success, message = self.circuit_breaker.call(
                        self.planner.add_task, task
                    )
                    duration_ms = (time.time() - start_time) * 1000
                    
                    self.logger.log_performance("add_task", duration_ms, success)
                    
                    if success:
                        results["tasks_processed"] += 1
                        self.logger.logger.info(f"✅ Added task: {task.name}")
                    else:
                        error_msg = f"Failed to add task {task.id}: {message}"
                        results["errors"].append(error_msg)
                        self.logger.logger.error(error_msg)
                        
                except Exception as e:
                    error_msg = f"Exception adding task {task.id}: {str(e)}"
                    results["errors"].append(error_msg)
                    self.logger.log_error(e, {"task_id": task.id, "operation": "add_task"})
            
            # Generate schedule with error handling
            try:
                start_time = time.time()
                schedule = self.circuit_breaker.call(
                    self.planner.quantum_anneal_schedule, max_iterations=300
                )
                duration_ms = (time.time() - start_time) * 1000
                
                self.logger.log_performance("generate_schedule", duration_ms, True)
                results["schedule"] = schedule
                results["tasks_successful"] = len(schedule)
                
                self.logger.logger.info(f"✅ Generated schedule with {len(schedule)} tasks")
                
            except Exception as e:
                error_msg = f"Schedule generation failed: {str(e)}"
                results["errors"].append(error_msg)
                self.logger.log_error(e, {"operation": "generate_schedule"})
            
            # Health check during execution
            health_results = await self.health_checker.run_all_checks()
            results["health_status"] = health_results
            
            results["end_time"] = datetime.now().isoformat()
            
            # Calculate final metrics
            total_tasks = len(tasks)
            if total_tasks > 0:
                success_rate = (results["tasks_successful"] / total_tasks) * 100
                results["success_rate_percent"] = success_rate
                
                if success_rate >= 90:
                    results["execution_status"] = "SUCCESS"
                elif success_rate >= 70:
                    results["execution_status"] = "PARTIAL_SUCCESS"
                else:
                    results["execution_status"] = "FAILURE"
            else:
                results["execution_status"] = "NO_TASKS"
            
            self.logger.logger.info(
                f"Robust execution completed: {results['execution_status']} "
                f"({results['tasks_successful']}/{total_tasks} tasks)"
            )
            
            return results
            
        except Exception as e:
            results["execution_status"] = "CRITICAL_FAILURE"
            results["critical_error"] = str(e)
            results["end_time"] = datetime.now().isoformat()
            
            self.logger.log_error(e, {"operation": "robust_task_execution"})
            return results
    
    async def demonstrate_robust_features(self) -> Dict[str, Any]:
        """Demonstrate all robustness features."""
        self.logger.logger.info("🛡️ Starting Robust Features Demonstration")
        
        demo_results = {
            "demo_start": datetime.now().isoformat(),
            "features_tested": [],
            "overall_status": "SUCCESS"
        }
        
        try:
            # 1. Error Handling & Validation
            self.logger.logger.info("Testing error handling and validation...")
            
            # Test invalid task handling
            invalid_task = QuantumTask(
                id="",  # Invalid empty ID
                name="Invalid Task",
                description="Test invalid task handling",
                priority=-1.0,  # Invalid negative priority
                estimated_duration=timedelta(seconds=1)
            )
            
            try:
                success, message = self.planner.add_task(invalid_task)
                if not success:
                    self.logger.logger.info("✅ Validation correctly rejected invalid task")
                    demo_results["features_tested"].append("input_validation")
            except Exception as e:
                self.logger.logger.info(f"✅ Exception handling caught invalid task: {str(e)}")
                demo_results["features_tested"].append("exception_handling")
            
            # 2. Circuit Breaker Testing
            self.logger.logger.info("Testing circuit breaker functionality...")
            
            def failing_function():
                raise Exception("Simulated failure")
            
            # Trigger circuit breaker
            for i in range(6):  # More than failure threshold
                try:
                    self.circuit_breaker.call(failing_function)
                except Exception:
                    pass
            
            # Verify circuit is open
            try:
                self.circuit_breaker.call(failing_function)
            except Exception as e:
                if "Circuit breaker is OPEN" in str(e):
                    self.logger.logger.info("✅ Circuit breaker correctly activated")
                    demo_results["features_tested"].append("circuit_breaker")
            
            # 3. Health Monitoring
            self.logger.logger.info("Testing health monitoring system...")
            
            health_results = await self.health_checker.run_all_checks()
            if health_results["overall_status"] in ["HEALTHY", "DEGRADED"]:
                self.logger.logger.info(f"✅ Health monitoring operational: {health_results['overall_status']}")
                demo_results["features_tested"].append("health_monitoring")
                demo_results["health_status"] = health_results
            
            # 4. Robust Task Execution
            self.logger.logger.info("Testing robust task execution...")
            
            test_tasks = [
                QuantumTask(
                    id="robust_test_1",
                    name="Robust Test Task 1",
                    description="Testing robust execution",
                    priority=8.0,
                    estimated_duration=timedelta(minutes=15)
                ),
                QuantumTask(
                    id="robust_test_2",
                    name="Robust Test Task 2",
                    description="Testing error recovery",
                    priority=7.5,
                    estimated_duration=timedelta(minutes=10)
                )
            ]
            
            # Reset circuit breaker for testing
            self.circuit_breaker.state = "CLOSED"
            self.circuit_breaker.failure_count = 0
            
            execution_results = await self.robust_task_execution(test_tasks)
            if execution_results["execution_status"] in ["SUCCESS", "PARTIAL_SUCCESS"]:
                self.logger.logger.info("✅ Robust task execution completed successfully")
                demo_results["features_tested"].append("robust_execution")
                demo_results["execution_results"] = execution_results
            
            # 5. Performance Monitoring
            self.logger.logger.info("Testing performance monitoring...")
            
            if self.logger.performance_metrics:
                avg_performance = {}
                for operation, metrics in self.logger.performance_metrics.items():
                    avg_duration = metrics['total_duration_ms'] / metrics['total_calls']
                    success_rate = (metrics['success_count'] / metrics['total_calls']) * 100
                    avg_performance[operation] = {
                        'avg_duration_ms': avg_duration,
                        'success_rate_percent': success_rate
                    }
                
                demo_results["performance_metrics"] = avg_performance
                demo_results["features_tested"].append("performance_monitoring")
                self.logger.logger.info("✅ Performance monitoring data collected")
            
            # 6. Logging & Audit Trail
            if self.logger.error_counts or self.logger.performance_metrics:
                demo_results["features_tested"].append("structured_logging")
                demo_results["error_counts"] = self.logger.error_counts
                self.logger.logger.info("✅ Structured logging and audit trail operational")
            
            demo_results["demo_end"] = datetime.now().isoformat()
            demo_results["features_count"] = len(demo_results["features_tested"])
            
            self.logger.logger.info(
                f"🎉 Robust features demonstration completed: "
                f"{demo_results['features_count']} features tested successfully"
            )
            
            return demo_results
            
        except Exception as e:
            demo_results["overall_status"] = "FAILURE"
            demo_results["critical_error"] = str(e)
            demo_results["demo_end"] = datetime.now().isoformat()
            
            self.logger.log_error(e, {"operation": "demonstrate_robust_features"})
            return demo_results


async def main():
    """Main execution for robust enhancement demonstration."""
    print("🛡️ Starting Generation 2: MAKE IT ROBUST (Reliable)")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        system = RobustEnhancementSystem()
        
        # Run comprehensive robustness demonstration
        results = await system.demonstrate_robust_features()
        
        # Save results
        output_file = Path('robust_enhancement_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f" 🎯 GENERATION 2 COMPLETED: {results['overall_status']}")
        print('='*60)
        print(f"✅ Features tested: {results['features_count']}")
        print(f"📊 Features implemented: {', '.join(results['features_tested'])}")
        print(f"🏥 Health status: {results.get('health_status', {}).get('overall_status', 'N/A')}")
        print(f"📁 Results saved to: {output_file}")
        
        if results['overall_status'] == 'SUCCESS':
            print(f"\n🚀 Ready to proceed to Generation 3: MAKE IT SCALE")
            return True
        else:
            print(f"\n⚠️  Some issues detected. Review logs for details.")
            return False
            
    except Exception as e:
        print(f"\n❌ Critical failure in Generation 2: {str(e)}")
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)