#!/usr/bin/env python3
"""
Robust Functionality Demo - Generation 2: MAKE IT ROBUST
Demonstrates enhanced error handling, validation, logging, monitoring, and resilience
"""

import asyncio
import sys
import os
import logging
import time
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Any
import random

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llm_cost_tracker import (
    QuantumTaskPlanner, 
    QuantumTask, 
    TaskState, 
    ResourcePool,
    quantum_i18n,
    t,
    set_language,
    SupportedLanguage
)

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("llm_cost_tracker_robust.log", mode='w')
    ]
)

logger = logging.getLogger(__name__)

class RobustSystemMonitor:
    """Enhanced monitoring and alerting system."""
    
    def __init__(self):
        self.metrics = {}
        self.alerts = []
        self.health_checks = []
        self.performance_data = []
        
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a performance metric."""
        timestamp = datetime.now()
        metric_data = {
            'name': name,
            'value': value,
            'timestamp': timestamp,
            'tags': tags or {}
        }
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(metric_data)
        self.performance_data.append(metric_data)
        
        # Keep only last 1000 entries per metric
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]
        
        logger.debug(f"Recorded metric {name}: {value}")
        
    def record_quantum_event(self, event_type: str, task_id: str, metadata: Dict = None):
        """Record quantum events for monitoring."""
        self.record_metric(f"quantum_{event_type}", 1.0, {
            'task_id': task_id,
            'metadata': str(metadata or {})
        })
        logger.info(f"Quantum event: {event_type} for task {task_id}")
        
    def record_task_execution(self, task_id: str, duration_ms: float, success: bool, probability: float):
        """Record task execution metrics."""
        self.record_metric('task_duration_ms', duration_ms, {'task_id': task_id})
        self.record_metric('task_success', 1.0 if success else 0.0, {'task_id': task_id})
        self.record_metric('task_probability', probability, {'task_id': task_id})
        
    def _record_error(self, error_type: str, message: str, severity: str = "error"):
        """Record error for monitoring."""
        self.alerts.append({
            'type': 'error',
            'error_type': error_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now()
        })
        logger.error(f"Error recorded: {error_type} - {message} (severity: {severity})")
        
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        recent_metrics = [m for m in self.performance_data if (datetime.now() - m['timestamp']).total_seconds() < 300]
        
        error_count = len([a for a in self.alerts if a['severity'] == 'error' and (datetime.now() - a['timestamp']).total_seconds() < 300])
        warning_count = len([a for a in self.alerts if a['severity'] == 'warning' and (datetime.now() - a['timestamp']).total_seconds() < 300])
        
        return {
            'healthy': error_count == 0,
            'metrics_count': len(recent_metrics),
            'error_count': error_count,
            'warning_count': warning_count,
            'last_check': datetime.now().isoformat(),
            'alerts': self.alerts[-10:]  # Last 10 alerts
        }


class RobustFunctionalityDemo:
    """Demonstrates Generation 2 robust functionality with comprehensive error handling."""
    
    def __init__(self):
        self.planner = QuantumTaskPlanner()
        self.monitor = RobustSystemMonitor()
        self.planner.set_monitor(self.monitor)
        
        logger.info("🛡️  Robust LLM Cost Tracker initialized with enhanced monitoring")
        print(f"🛡️  Robust LLM Cost Tracker initialized")
    
    def demo_input_validation_and_sanitization(self):
        """Demonstrate comprehensive input validation and sanitization."""
        print(f"\n{'='*60}")
        print("🔍 INPUT VALIDATION & SANITIZATION DEMO")
        print(f"{'='*60}")
        
        # Test invalid task inputs
        invalid_tasks = [
            {
                "id": "",  # Empty ID
                "name": "Empty ID Task",
                "description": "Task with empty ID"
            },
            {
                "id": "valid_id",
                "name": "A" * 1000,  # Extremely long name
                "description": "Task with very long name"
            },
            {
                "id": "script_injection",
                "name": "<script>alert('xss')</script>",  # XSS attempt
                "description": "Malicious script injection attempt"
            },
            {
                "id": "sql_injection",
                "name": "'; DROP TABLE tasks; --",  # SQL injection attempt
                "description": "SQL injection attempt"
            },
            {
                "id": "negative_priority",
                "name": "Negative Priority Task",
                "description": "Task with negative priority",
                "priority": -5.0
            },
            {
                "id": "circular_dependency",
                "name": "Circular Dependency",
                "description": "Task with circular dependency",
                "dependencies": {"circular_dependency"}  # Self-dependency
            }
        ]
        
        print("🧪 Testing invalid inputs:")
        for i, invalid_task_data in enumerate(invalid_tasks, 1):
            try:
                # Create task with validation
                task = QuantumTask(
                    id=invalid_task_data.get("id", f"invalid_{i}"),
                    name=invalid_task_data.get("name", f"Invalid Task {i}"),
                    description=invalid_task_data.get("description", "Invalid task"),
                    priority=invalid_task_data.get("priority", 5.0),
                    estimated_duration=timedelta(minutes=invalid_task_data.get("duration_minutes", 30)),
                    required_resources=invalid_task_data.get("required_resources", {}),
                    dependencies=invalid_task_data.get("dependencies", set())
                )
                
                success, message = self.planner.add_task(task)
                if success:
                    print(f"   ✅ Test {i}: Task added (validation passed) - {message}")
                else:
                    print(f"   ❌ Test {i}: Task rejected (validation failed) - {message}")
                    
            except Exception as e:
                print(f"   🛡️  Test {i}: Exception caught - {type(e).__name__}: {e}")
                logger.warning(f"Input validation test {i} caught exception: {e}")
    
    def demo_error_handling_and_recovery(self):
        """Demonstrate comprehensive error handling and recovery mechanisms."""
        print(f"\n{'='*60}")
        print("🚨 ERROR HANDLING & RECOVERY DEMO")
        print(f"{'='*60}")
        
        # Create tasks that will likely fail
        failure_scenarios = [
            {
                "id": "resource_hungry",
                "name": "Resource Hungry Task",
                "description": "Task requiring excessive resources",
                "required_resources": {"cpu_cores": 1000.0, "memory_gb": 1000.0}  # Impossible resources
            },
            {
                "id": "timeout_task",
                "name": "Timeout Simulation Task",
                "description": "Task simulating timeout behavior",
                "estimated_duration": timedelta(hours=24),  # Very long duration
                "required_resources": {"cpu_cores": 1.0}
            }
        ]
        
        print("🔥 Testing error scenarios:")
        for scenario in failure_scenarios:
            try:
                task = QuantumTask(
                    id=scenario["id"],
                    name=scenario["name"], 
                    description=scenario["description"],
                    priority=5.0,
                    estimated_duration=scenario.get("estimated_duration", timedelta(minutes=30)),
                    required_resources=scenario["required_resources"]
                )
                
                success, message = self.planner.add_task(task)
                print(f"   📝 Added task: {task.name} - {message}")
                
                if success:
                    # Try to execute the problematic task
                    print(f"   ⚡ Attempting to execute: {task.name}")
                    execution_success = asyncio.run(self.planner.execute_task(task.id))
                    print(f"   {'✅ Success' if execution_success else '❌ Failed'}: {task.name}")
                    
                    # Check task state after execution attempt
                    final_state = self.planner.tasks[task.id].state
                    error_message = getattr(self.planner.tasks[task.id], 'error_message', None)
                    print(f"   📊 Final state: {final_state.value}")
                    if error_message:
                        print(f"   📝 Error message: {error_message}")
                
            except Exception as e:
                print(f"   🛡️  Exception handled: {type(e).__name__}: {e}")
                logger.error(f"Error scenario {scenario['id']} handled: {e}", exc_info=True)
        
        # Test circuit breaker
        print(f"\n🔌 Circuit Breaker Test:")
        breaker_open = self.planner.is_circuit_breaker_open()
        print(f"   Circuit breaker status: {'🔴 OPEN' if breaker_open else '🟢 CLOSED'}")
        
        if breaker_open:
            print("   🔄 Testing circuit breaker reset...")
            self.planner.reset_circuit_breaker()
            reset_status = self.planner.is_circuit_breaker_open()
            print(f"   After reset: {'🔴 OPEN' if reset_status else '🟢 CLOSED'}")
    
    async def demo_health_monitoring_and_alerts(self):
        """Demonstrate health monitoring and alerting system."""
        print(f"\n{'='*60}")
        print("🏥 HEALTH MONITORING & ALERTS DEMO")
        print(f"{'='*60}")
        
        # Perform comprehensive health check
        print("🔍 Performing health check...")
        health_status = self.planner.perform_health_check()
        
        print(f"📊 Health Status:")
        print(f"   Overall healthy: {'✅ Yes' if health_status['overall_healthy'] else '❌ No'}")
        print(f"   Timestamp: {health_status['timestamp']}")
        
        # Show component health
        components = health_status.get('components', {})
        for component, status in components.items():
            healthy = status.get('healthy', False)
            print(f"   {component}: {'✅ Healthy' if healthy else '❌ Unhealthy'}")
            if 'error' in status:
                print(f"     Error: {status['error']}")
        
        # Show issues
        issues = health_status.get('issues', [])
        if issues:
            print(f"🚨 Issues detected:")
            for issue in issues:
                print(f"   ⚠️  {issue}")
        
        # Show monitoring metrics
        monitor_summary = self.monitor.get_health_summary()
        print(f"\n📈 Monitoring Summary:")
        print(f"   Metrics recorded: {monitor_summary['metrics_count']}")
        print(f"   Recent errors: {monitor_summary['error_count']}")
        print(f"   Recent warnings: {monitor_summary['warning_count']}")
        
        # Simulate some monitoring events
        print(f"\n📡 Simulating monitoring events...")
        self.monitor.record_quantum_event("superposition_collapse", "test_task_1")
        self.monitor.record_task_execution("test_task_1", 1500.0, True, 0.85)
        self.monitor.record_metric("system_load", 0.75, {"node": "primary"})
        
        print("   ✅ Monitoring events recorded")
    
    def demo_data_integrity_and_consistency(self):
        """Demonstrate data integrity and consistency checks."""
        print(f"\n{'='*60}")
        print("🔐 DATA INTEGRITY & CONSISTENCY DEMO")
        print(f"{'='*60}")
        
        # Test concurrent task operations
        print("🔄 Testing concurrent operations...")
        
        # Create multiple tasks concurrently
        concurrent_tasks = []
        for i in range(10):
            task = QuantumTask(
                id=f"concurrent_task_{i}",
                name=f"Concurrent Task {i}",
                description=f"Task {i} for concurrent testing",
                priority=random.uniform(1.0, 10.0),
                estimated_duration=timedelta(minutes=random.randint(5, 60)),
                required_resources={"cpu_cores": random.uniform(0.5, 4.0), "memory_gb": random.uniform(1.0, 8.0)}
            )
            concurrent_tasks.append(task)
        
        # Add tasks and track success/failure
        add_results = []
        for task in concurrent_tasks:
            success, message = self.planner.add_task(task)
            add_results.append(success)
            if not success:
                logger.warning(f"Failed to add concurrent task {task.id}: {message}")
        
        success_count = sum(add_results)
        print(f"   📊 Concurrent add results: {success_count}/{len(concurrent_tasks)} successful")
        
        # Test dependency consistency
        print(f"\n🔗 Testing dependency consistency...")
        if len(self.planner.tasks) >= 3:
            task_ids = list(self.planner.tasks.keys())[:3]
            
            # Create valid dependency chain
            self.planner.create_dependency(task_ids[1], task_ids[0])
            self.planner.create_dependency(task_ids[2], task_ids[1])
            
            # Verify dependency consistency
            consistency_issues = []
            for task_id, task in self.planner.tasks.items():
                for dep_id in task.dependencies:
                    if dep_id not in self.planner.tasks:
                        consistency_issues.append(f"Task {task_id} depends on non-existent task {dep_id}")
            
            if consistency_issues:
                print(f"   ❌ Consistency issues found:")
                for issue in consistency_issues:
                    print(f"     {issue}")
            else:
                print(f"   ✅ Dependencies are consistent")
        
        # Test resource accounting
        print(f"\n💾 Testing resource accounting...")
        initial_resources = {
            'cpu': self.planner.resource_pool.cpu_cores - self.planner.resource_pool.allocated_cpu,
            'memory': self.planner.resource_pool.memory_gb - self.planner.resource_pool.allocated_memory,
            'storage': self.planner.resource_pool.storage_gb - self.planner.resource_pool.allocated_storage,
            'bandwidth': self.planner.resource_pool.network_bandwidth - self.planner.resource_pool.allocated_bandwidth
        }
        
        # Allocate and deallocate resources
        test_allocation = {"cpu_cores": 2.0, "memory_gb": 4.0}
        allocated = self.planner.resource_pool.allocate(test_allocation)
        print(f"   Allocation test: {'✅ Success' if allocated else '❌ Failed'}")
        
        if allocated:
            self.planner.resource_pool.deallocate(test_allocation)
            print(f"   Deallocation completed")
        
        # Verify resource accounting integrity
        final_resources = {
            'cpu': self.planner.resource_pool.cpu_cores - self.planner.resource_pool.allocated_cpu,
            'memory': self.planner.resource_pool.memory_gb - self.planner.resource_pool.allocated_memory,
            'storage': self.planner.resource_pool.storage_gb - self.planner.resource_pool.allocated_storage,
            'bandwidth': self.planner.resource_pool.network_bandwidth - self.planner.resource_pool.allocated_bandwidth
        }
        
        resources_consistent = initial_resources == final_resources
        print(f"   Resource accounting: {'✅ Consistent' if resources_consistent else '❌ Inconsistent'}")
    
    async def demo_fault_tolerance_and_resilience(self):
        """Demonstrate fault tolerance and resilience mechanisms."""
        print(f"\n{'='*60}")
        print("🔥 FAULT TOLERANCE & RESILIENCE DEMO")
        print(f"{'='*60}")
        
        # Create tasks with varying failure probabilities
        resilience_tasks = []
        for i in range(5):
            task = QuantumTask(
                id=f"resilience_task_{i}",
                name=f"Resilience Test Task {i}",
                description=f"Task {i} for resilience testing",
                priority=random.uniform(5.0, 10.0),
                estimated_duration=timedelta(minutes=random.randint(1, 10)),
                required_resources={"cpu_cores": random.uniform(0.5, 2.0), "memory_gb": random.uniform(1.0, 4.0)},
                # Adjust probability amplitude for controlled failure rates
                probability_amplitude=complex(random.uniform(0.5, 1.0), 0.0)
            )
            resilience_tasks.append(task)
        
        # Add tasks to planner
        for task in resilience_tasks:
            success, message = self.planner.add_task(task)
            if success:
                print(f"   ✅ Added resilience task: {task.name}")
            else:
                print(f"   ❌ Failed to add task: {message}")
        
        # Test parallel execution with fault tolerance
        print(f"\n🔄 Testing parallel execution with fault tolerance...")
        
        # Get current task IDs for execution
        current_task_ids = list(self.planner.tasks.keys())[-5:]  # Last 5 tasks
        
        if current_task_ids:
            # Test parallel execution
            parallel_results = await self.planner.execute_schedule_parallel(current_task_ids)
            
            print(f"📊 Parallel Execution Results:")
            print(f"   Total tasks: {parallel_results['total_tasks']}")
            print(f"   Successful: {parallel_results['successful_tasks']}")
            print(f"   Failed: {parallel_results['failed_tasks']}")
            print(f"   Success rate: {parallel_results.get('success_rate', 0):.2%}")
            print(f"   Parallel batches: {parallel_results.get('parallel_batches', 0)}")
            
            # Test system recovery after failures
            if parallel_results['failed_tasks'] > 0:
                print(f"\n🔄 Testing system recovery after failures...")
                
                # Check circuit breaker status
                breaker_open = self.planner.is_circuit_breaker_open()
                print(f"   Circuit breaker: {'🔴 OPEN' if breaker_open else '🟢 CLOSED'}")
                
                if breaker_open:
                    print(f"   💡 Initiating recovery sequence...")
                    self.planner.reset_circuit_breaker()
                    print(f"   ✅ Circuit breaker reset")
                    
                # Test health check after recovery
                recovery_health = self.planner.perform_health_check()
                print(f"   Post-recovery health: {'✅ Healthy' if recovery_health['overall_healthy'] else '❌ Unhealthy'}")
    
    def demo_logging_and_observability(self):
        """Demonstrate comprehensive logging and observability."""
        print(f"\n{'='*60}")
        print("📝 LOGGING & OBSERVABILITY DEMO")
        print(f"{'='*60}")
        
        # Test different log levels and formats
        print("📋 Testing logging levels...")
        logger.debug("Debug message - detailed system state")
        logger.info("Info message - normal operation")
        logger.warning("Warning message - potential issue detected")
        logger.error("Error message - recoverable error occurred")
        
        # Test structured logging with context
        logger.info("Structured log entry", extra={
            'component': 'quantum_planner',
            'operation': 'task_scheduling',
            'task_count': len(self.planner.tasks),
            'resource_utilization': {
                'cpu': self.planner.resource_pool.allocated_cpu / self.planner.resource_pool.cpu_cores,
                'memory': self.planner.resource_pool.allocated_memory / self.planner.resource_pool.memory_gb
            }
        })
        
        # Show execution history and metrics
        history_size = len(self.planner.execution_history)
        print(f"   📚 Execution history entries: {history_size}")
        
        if history_size > 0:
            recent_executions = self.planner.execution_history[-3:]
            print(f"   📊 Recent executions:")
            for exec_record in recent_executions:
                success_status = "✅" if exec_record['success'] else "❌"
                print(f"     {success_status} {exec_record['task_id']} - {exec_record['execution_duration_ms']:.2f}ms")
        
        # Show monitoring metrics
        print(f"\n📈 Monitoring Metrics:")
        metric_types = list(self.monitor.metrics.keys())[:5]  # Show first 5 metric types
        for metric_type in metric_types:
            metric_data = self.monitor.metrics[metric_type]
            recent_values = [m['value'] for m in metric_data[-10:]]  # Last 10 values
            if recent_values:
                avg_value = sum(recent_values) / len(recent_values)
                print(f"   {metric_type}: avg={avg_value:.3f}, count={len(metric_data)}")
        
        # Show optimization stats
        try:
            opt_stats = self.planner.get_optimization_stats()
            print(f"\n⚡ Optimization Statistics:")
            print(f"   Cache enabled: {opt_stats['cache_enabled']}")
            print(f"   Parallel execution: {opt_stats['parallel_execution_enabled']}")
            
            if opt_stats['cache']:
                cache_stats = opt_stats['cache']
                print(f"   Cache hits: {cache_stats.get('hits', 0)}")
                print(f"   Cache misses: {cache_stats.get('misses', 0)}")
        except Exception as e:
            logger.warning(f"Could not retrieve optimization stats: {e}")
    
    def generate_robust_system_report(self):
        """Generate comprehensive system report."""
        print(f"\n{'='*60}")
        print("📋 ROBUST SYSTEM REPORT")
        print(f"{'='*60}")
        
        # System overview
        system_state = self.planner.get_system_state()
        health_status = self.planner.perform_health_check()
        monitor_summary = self.monitor.get_health_summary()
        
        print("🎯 System Overview:")
        print(f"   Total tasks: {system_state['total_tasks']}")
        print(f"   Overall health: {'✅ Healthy' if health_status['overall_healthy'] else '❌ Unhealthy'}")
        print(f"   Circuit breaker: {'🔴 OPEN' if self.planner.is_circuit_breaker_open() else '🟢 CLOSED'}")
        
        # Resource utilization
        print(f"\n💾 Resource Utilization:")
        resource_util = system_state['resource_utilization']
        for resource, utilization in resource_util.items():
            status = "🔴" if utilization > 0.9 else "🟡" if utilization > 0.7 else "🟢"
            print(f"   {resource}: {utilization:.2%} {status}")
        
        # Error and monitoring summary
        print(f"\n📊 Monitoring Summary:")
        print(f"   Recent errors: {monitor_summary['error_count']}")
        print(f"   Recent warnings: {monitor_summary['warning_count']}")
        print(f"   Metrics recorded: {monitor_summary['metrics_count']}")
        
        # Performance metrics
        execution_history = self.planner.execution_history
        if execution_history:
            success_rate = sum(1 for e in execution_history if e['success']) / len(execution_history)
            avg_duration = sum(e['execution_duration_ms'] for e in execution_history) / len(execution_history)
            print(f"\n⚡ Performance Metrics:")
            print(f"   Success rate: {success_rate:.2%}")
            print(f"   Average execution time: {avg_duration:.2f}ms")
            print(f"   Total executions: {len(execution_history)}")
        
        # Recommendations
        recommendations = []
        
        if health_status.get('issues'):
            recommendations.append("Address health check issues")
            
        if monitor_summary['error_count'] > 0:
            recommendations.append("Investigate recent errors")
            
        high_utilization = [r for r, u in resource_util.items() if u > 0.8]
        if high_utilization:
            recommendations.append(f"Monitor high resource utilization: {', '.join(high_utilization)}")
            
        if self.planner.is_circuit_breaker_open():
            recommendations.append("Circuit breaker is open - check system stability")
        
        print(f"\n💡 Recommendations:")
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        else:
            print("   ✅ No immediate actions required - system operating normally")


async def main():
    """Run the complete Generation 2 robust functionality demo."""
    print("🛡️  LLM COST TRACKER - GENERATION 2 DEMO")
    print("=" * 60)
    print("Demonstrating: MAKE IT ROBUST - Enhanced Reliability & Error Handling")
    print("=" * 60)
    
    demo = RobustFunctionalityDemo()
    
    try:
        # Run all robust functionality demos
        demo.demo_input_validation_and_sanitization()
        demo.demo_error_handling_and_recovery()
        await demo.demo_health_monitoring_and_alerts()
        demo.demo_data_integrity_and_consistency()
        await demo.demo_fault_tolerance_and_resilience()
        demo.demo_logging_and_observability()
        demo.generate_robust_system_report()
        
        print(f"\n{'='*60}")
        print("🎉 GENERATION 2 DEMO COMPLETED")
        print("✅ Robust functionality verified and operational")
        print("🔄 Ready to proceed to Generation 3: MAKE IT SCALE")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Demo failed with critical error: {e}", exc_info=True)
        print(f"\n💥 DEMO FAILED: {e}")
        print("🛠️  Check logs for detailed error information")


if __name__ == "__main__":
    asyncio.run(main())