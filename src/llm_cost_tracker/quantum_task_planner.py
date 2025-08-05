"""
Quantum-Inspired Task Planner

This module implements a quantum-inspired approach to task planning and execution,
using concepts from quantum computing to optimize task scheduling and resource allocation.

Key quantum-inspired concepts implemented:
- Superposition: Tasks exist in multiple states until measured/executed
- Entanglement: Task dependencies create correlated execution paths
- Quantum annealing: Optimization through simulated annealing for scheduling
- Interference patterns: Task conflicts and synergies affect execution probability
"""

import asyncio
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import logging
from concurrent.futures import ThreadPoolExecutor
import traceback

from .quantum_validation import validate_task_input, sanitize_task_input, ValidationSeverity
from .quantum_optimization import QuantumCache, memoize_with_ttl, QuantumLoadBalancer

logger = logging.getLogger(__name__)


class TaskState(Enum):
    """Quantum-inspired task states."""
    SUPERPOSITION = "superposition"  # Task exists in multiple potential states
    COLLAPSED = "collapsed"          # Task state has been measured/determined
    ENTANGLED = "entangled"         # Task is dependent on other tasks
    EXECUTING = "executing"         # Task is currently running
    COMPLETED = "completed"         # Task finished successfully
    FAILED = "failed"               # Task execution failed


@dataclass
class QuantumTask:
    """A task with quantum-inspired properties."""
    
    id: str
    name: str
    description: str
    priority: float = 1.0
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(hours=1))
    required_resources: Dict[str, float] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    
    # Quantum-inspired properties
    state: TaskState = TaskState.SUPERPOSITION
    probability_amplitude: complex = complex(1.0, 0.0)
    entangled_tasks: Set[str] = field(default_factory=set)
    interference_pattern: Dict[str, float] = field(default_factory=dict)
    
    # Execution metadata
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def get_execution_probability(self) -> float:
        """Calculate the probability of successful execution."""
        # Base probability from amplitude
        base_prob = abs(self.probability_amplitude) ** 2
        
        # Apply interference effects
        interference_factor = 1.0
        for task_id, effect in self.interference_pattern.items():
            interference_factor *= (1.0 + effect)
        
        # Ensure probability stays within [0, 1]
        return max(0.0, min(1.0, base_prob * interference_factor))
    
    def collapse_state(self, new_state: TaskState) -> None:
        """Collapse the task from superposition to a definite state."""
        if self.state == TaskState.SUPERPOSITION:
            self.state = new_state
            logger.debug(f"Task {self.id} collapsed to state: {new_state}")
    
    def entangle_with(self, other_task_id: str) -> None:
        """Create entanglement with another task."""
        self.entangled_tasks.add(other_task_id)
        self.state = TaskState.ENTANGLED
        logger.debug(f"Task {self.id} entangled with {other_task_id}")


@dataclass 
class ResourcePool:
    """Manages available computational resources."""
    
    cpu_cores: float = 4.0
    memory_gb: float = 8.0
    storage_gb: float = 100.0
    network_bandwidth: float = 100.0  # Mbps
    
    # Current allocations
    allocated_cpu: float = 0.0
    allocated_memory: float = 0.0
    allocated_storage: float = 0.0
    allocated_bandwidth: float = 0.0
    
    def can_allocate(self, requirements: Dict[str, float]) -> bool:
        """Check if resources can be allocated."""
        cpu_req = requirements.get('cpu_cores', 0.0)
        mem_req = requirements.get('memory_gb', 0.0)
        storage_req = requirements.get('storage_gb', 0.0)
        bandwidth_req = requirements.get('network_bandwidth', 0.0)
        
        return (
            self.allocated_cpu + cpu_req <= self.cpu_cores and
            self.allocated_memory + mem_req <= self.memory_gb and
            self.allocated_storage + storage_req <= self.storage_gb and
            self.allocated_bandwidth + bandwidth_req <= self.network_bandwidth
        )
    
    def allocate(self, requirements: Dict[str, float]) -> bool:
        """Allocate resources if available."""
        if self.can_allocate(requirements):
            self.allocated_cpu += requirements.get('cpu_cores', 0.0)
            self.allocated_memory += requirements.get('memory_gb', 0.0)
            self.allocated_storage += requirements.get('storage_gb', 0.0)
            self.allocated_bandwidth += requirements.get('network_bandwidth', 0.0)
            return True
        return False
    
    def deallocate(self, requirements: Dict[str, float]) -> None:
        """Deallocate resources."""
        self.allocated_cpu -= requirements.get('cpu_cores', 0.0)
        self.allocated_memory -= requirements.get('memory_gb', 0.0)
        self.allocated_storage -= requirements.get('storage_gb', 0.0)
        self.allocated_bandwidth -= requirements.get('network_bandwidth', 0.0)
        
        # Ensure we don't go negative
        self.allocated_cpu = max(0.0, self.allocated_cpu)
        self.allocated_memory = max(0.0, self.allocated_memory)
        self.allocated_storage = max(0.0, self.allocated_storage)
        self.allocated_bandwidth = max(0.0, self.allocated_bandwidth)


class QuantumTaskPlanner:
    """
    Quantum-inspired task planner using superposition, entanglement, and 
    quantum annealing principles for optimal task scheduling.
    """
    
    def __init__(self, resource_pool: Optional[ResourcePool] = None):
        self.tasks: Dict[str, QuantumTask] = {}
        self.resource_pool = resource_pool or ResourcePool()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.execution_history: List[Dict[str, Any]] = []
        
        # Quantum annealing parameters
        self.temperature = 100.0
        self.cooling_rate = 0.95
        self.min_temperature = 0.1
        
        # Monitoring integration
        self.monitor = None
        self.health_check_enabled = True
        self.last_health_check = datetime.now()
        self.error_count = 0
        self.max_error_threshold = 10
        
        # Optimization features
        self.cache = QuantumCache(max_size=500, default_ttl=300)
        self.load_balancer: Optional[QuantumLoadBalancer] = None
        self.cache_enabled = True
        self.parallel_execution_enabled = True
        
    def add_task(self, task: QuantumTask) -> Tuple[bool, str]:
        """
        Add a task to the planning system with validation and error handling.
        
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            # Validate task data
            task_data = {
                'id': task.id,
                'name': task.name,
                'description': task.description,
                'priority': task.priority,
                'estimated_duration': task.estimated_duration,
                'required_resources': task.required_resources,
                'dependencies': task.dependencies,
                'probability_amplitude': task.probability_amplitude,
                'interference_pattern': task.interference_pattern,
                'entangled_tasks': task.entangled_tasks
            }
            
            validation_result = validate_task_input(task_data, set(self.tasks.keys()))
            
            # Handle validation failures
            if not validation_result.is_valid:
                error_messages = []
                for error in validation_result.critical_errors + validation_result.errors:
                    error_messages.append(f"{error.field}: {error.message}")
                
                error_msg = f"Task validation failed: {'; '.join(error_messages)}"
                logger.error(f"Failed to add task {task.id}: {error_msg}")
                return False, error_msg
            
            # Log warnings
            for warning in validation_result.warnings:
                logger.warning(f"Task {task.id} validation warning - {warning.field}: {warning.message}")
            
            # Check for duplicate task ID
            if task.id in self.tasks:
                error_msg = f"Task with ID '{task.id}' already exists"
                logger.error(error_msg)
                return False, error_msg
            
            # Add task to planner
            self.tasks[task.id] = task
            logger.info(f"Added task {task.id}: {task.name}")
            
            # Calculate initial interference patterns
            try:
                self._calculate_interference_patterns(task)
            except Exception as e:
                logger.warning(f"Failed to calculate interference patterns for task {task.id}: {e}")
                # Continue anyway as this is not critical
            
            # Record quantum event if monitoring is available
            if hasattr(self, 'monitor') and self.monitor:
                self.monitor.record_quantum_event("task_added", task.id, {
                    'priority': task.priority,
                    'resource_count': len(task.required_resources),
                    'dependency_count': len(task.dependencies)
                })
            
            return True, f"Task {task.id} added successfully"
            
        except Exception as e:
            error_msg = f"Unexpected error adding task: {str(e)}"
            logger.error(f"Failed to add task {task.id}: {error_msg}", exc_info=True)
            return False, error_msg
    
    def _calculate_interference_patterns(self, task: QuantumTask) -> None:
        """Calculate interference patterns with existing tasks."""
        for existing_id, existing_task in self.tasks.items():
            if existing_id == task.id:
                continue
                
            # Calculate interference based on resource overlap
            resource_overlap = self._calculate_resource_overlap(
                task.required_resources, 
                existing_task.required_resources
            )
            
            # Negative interference for resource conflicts
            interference_effect = -0.1 * resource_overlap
            
            # Positive interference for complementary tasks
            if self._are_complementary(task, existing_task):
                interference_effect += 0.15
            
            # Store interference pattern
            task.interference_pattern[existing_id] = interference_effect
            existing_task.interference_pattern[task.id] = interference_effect
    
    def _calculate_resource_overlap(self, resources1: Dict[str, float], 
                                  resources2: Dict[str, float]) -> float:
        """Calculate overlap in resource requirements."""
        overlap = 0.0
        all_resources = set(resources1.keys()) | set(resources2.keys())
        
        for resource in all_resources:
            req1 = resources1.get(resource, 0.0)
            req2 = resources2.get(resource, 0.0)
            overlap += min(req1, req2) / max(req1 + req2, 1.0)
        
        return overlap / max(len(all_resources), 1)
    
    def _are_complementary(self, task1: QuantumTask, task2: QuantumTask) -> bool:
        """Determine if two tasks are complementary."""
        # Tasks are complementary if they share dependencies or have similar priorities
        shared_deps = len(task1.dependencies & task2.dependencies)
        priority_similarity = 1.0 - abs(task1.priority - task2.priority)
        
        return shared_deps > 0 and priority_similarity > 0.8
    
    def create_dependency(self, task_id: str, depends_on: str) -> None:
        """Create a dependency relationship between tasks."""
        if task_id in self.tasks and depends_on in self.tasks:
            self.tasks[task_id].dependencies.add(depends_on)
            
            # Create entanglement for strongly coupled tasks
            if self._should_entangle(task_id, depends_on):
                self.tasks[task_id].entangle_with(depends_on)
                self.tasks[depends_on].entangle_with(task_id)
    
    def _should_entangle(self, task1_id: str, task2_id: str) -> bool:
        """Determine if tasks should be entangled."""
        task1 = self.tasks[task1_id]
        task2 = self.tasks[task2_id]
        
        # High priority tasks with shared resources should be entangled
        high_priority = task1.priority > 8.0 and task2.priority > 8.0
        resource_conflict = self._calculate_resource_overlap(
            task1.required_resources, task2.required_resources
        ) > 0.5
        
        return high_priority and resource_conflict
    
    def quantum_anneal_schedule(self, max_iterations: int = 1000) -> List[str]:
        """
        Use quantum annealing to find optimal task execution order with caching.
        """
        # Generate cache key based on current tasks and their properties
        if self.cache_enabled:
            cache_key = self._generate_schedule_cache_key(max_iterations)
            cached_schedule = self.cache.get(cache_key)
            if cached_schedule is not None:
                logger.debug("Using cached schedule")
                return cached_schedule
        
        # Generate initial schedule
        current_schedule = list(self.tasks.keys())
        random.shuffle(current_schedule)
        
        current_cost = self._calculate_schedule_cost(current_schedule)
        best_schedule = current_schedule.copy()
        best_cost = current_cost
        
        temperature = self.temperature
        
        for iteration in range(max_iterations):
            # Generate neighbor schedule by swapping two tasks
            new_schedule = current_schedule.copy()
            if len(new_schedule) < 2:
                break  # Cannot swap if less than 2 tasks
            i, j = random.sample(range(len(new_schedule)), 2)
            new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]
            
            new_cost = self._calculate_schedule_cost(new_schedule)
            cost_delta = new_cost - current_cost
            
            # Accept or reject the new schedule
            if cost_delta < 0 or random.random() < math.exp(-cost_delta / temperature):
                current_schedule = new_schedule
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_schedule = current_schedule.copy()
                    best_cost = current_cost
            
            # Cool down
            temperature *= self.cooling_rate
            if temperature < self.min_temperature:
                temperature = self.min_temperature
        
        logger.info(f"Quantum annealing complete. Best cost: {best_cost}")
        
        # Cache the result
        if self.cache_enabled:
            cache_key = self._generate_schedule_cache_key(max_iterations)
            self.cache.put(cache_key, best_schedule, ttl=300)  # Cache for 5 minutes
        
        return best_schedule
    
    def set_monitor(self, monitor) -> None:
        """Set monitoring system for the planner."""
        self.monitor = monitor
        logger.info("Monitoring system attached to quantum task planner")
    
    def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of the planner."""
        try:
            self.last_health_check = datetime.now()
            
            health_status = {
                'timestamp': self.last_health_check.isoformat(),
                'overall_healthy': True,
                'components': {},
                'metrics': {},
                'issues': []
            }
            
            # Check task system health
            try:
                task_count = len(self.tasks)
                executing_tasks = sum(1 for t in self.tasks.values() if t.state == TaskState.EXECUTING)
                completed_tasks = sum(1 for t in self.tasks.values() if t.state == TaskState.COMPLETED)
                failed_tasks = sum(1 for t in self.tasks.values() if t.state == TaskState.FAILED)
                
                health_status['components']['task_system'] = {
                    'healthy': True,
                    'total_tasks': task_count,
                    'executing': executing_tasks,
                    'completed': completed_tasks,
                    'failed': failed_tasks
                }
                
                if failed_tasks > completed_tasks and task_count > 5:
                    health_status['components']['task_system']['healthy'] = False
                    health_status['issues'].append("High task failure rate detected")
                
            except Exception as e:
                health_status['components']['task_system'] = {'healthy': False, 'error': str(e)}
                health_status['issues'].append(f"Task system check failed: {e}")
            
            # Check resource system health  
            try:
                pool = self.resource_pool
                cpu_utilization = pool.allocated_cpu / pool.cpu_cores if pool.cpu_cores > 0 else 0
                memory_utilization = pool.allocated_memory / pool.memory_gb if pool.memory_gb > 0 else 0
                
                health_status['components']['resource_system'] = {
                    'healthy': True,
                    'cpu_utilization': cpu_utilization,
                    'memory_utilization': memory_utilization,
                    'overallocated': any([
                        pool.allocated_cpu > pool.cpu_cores,
                        pool.allocated_memory > pool.memory_gb,
                        pool.allocated_storage > pool.storage_gb,
                        pool.allocated_bandwidth > pool.network_bandwidth
                    ])
                }
                
                if cpu_utilization > 0.95 or memory_utilization > 0.95:
                    health_status['components']['resource_system']['healthy'] = False
                    health_status['issues'].append("Resource utilization critically high")
                
            except Exception as e:
                health_status['components']['resource_system'] = {'healthy': False, 'error': str(e)}
                health_status['issues'].append(f"Resource system check failed: {e}")
            
            # Check scheduler health
            try:
                if len(self.tasks) > 1:
                    test_schedule = self.quantum_anneal_schedule(max_iterations=5)
                    scheduler_healthy = len(test_schedule) == len(self.tasks)
                else:
                    scheduler_healthy = True
                
                health_status['components']['scheduler'] = {
                    'healthy': scheduler_healthy,
                    'can_generate_schedule': scheduler_healthy
                }
                
                if not scheduler_healthy:
                    health_status['issues'].append("Scheduler unable to generate valid schedules")
                    
            except Exception as e:
                health_status['components']['scheduler'] = {'healthy': False, 'error': str(e)}
                health_status['issues'].append(f"Scheduler check failed: {e}")
            
            # Check error rate
            try:
                recent_executions = [h for h in self.execution_history 
                                   if (datetime.now() - datetime.fromisoformat(h['started_at'])).total_seconds() < 3600]
                
                if len(recent_executions) > 0:
                    error_rate = sum(1 for h in recent_executions if not h['success']) / len(recent_executions)
                    health_status['metrics']['error_rate'] = error_rate
                    
                    if error_rate > 0.5:  # More than 50% failure rate
                        health_status['issues'].append(f"High error rate: {error_rate:.1%}")
                        
            except Exception as e:
                health_status['issues'].append(f"Error rate check failed: {e}")
            
            # Overall health determination
            component_health = [comp.get('healthy', False) for comp in health_status['components'].values()]
            health_status['overall_healthy'] = all(component_health) and len(health_status['issues']) == 0
            
            # Record health check completion
            if self.monitor:
                if not health_status['overall_healthy']:
                    self.monitor._record_error("health_check", f"Health issues detected: {'; '.join(health_status['issues'])}", "warning")
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed with exception: {e}", exc_info=True)
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_healthy': False,
                'error': str(e),
                'components': {},
                'metrics': {},
                'issues': [f"Health check system failure: {e}"]
            }
    
    def is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker should prevent new task execution."""
        if not self.health_check_enabled:
            return False
            
        # Check error count threshold
        if self.error_count >= self.max_error_threshold:
            logger.warning(f"Circuit breaker OPEN: Error count ({self.error_count}) exceeds threshold ({self.max_error_threshold})")
            return True
        
        # Check recent failure rate
        recent_history = self.execution_history[-20:] if len(self.execution_history) >= 20 else self.execution_history
        if len(recent_history) >= 5:
            failure_rate = sum(1 for h in recent_history if not h['success']) / len(recent_history)
            if failure_rate > 0.8:  # 80% failure rate
                logger.warning(f"Circuit breaker OPEN: High failure rate ({failure_rate:.1%})")
                return True
        
        return False
    
    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker state."""
        self.error_count = 0
        logger.info("Circuit breaker RESET")
    
    def increment_error_count(self) -> None:
        """Increment error count for circuit breaker."""
        self.error_count += 1
        if self.error_count >= self.max_error_threshold:
            logger.warning(f"Circuit breaker threshold reached: {self.error_count} errors")
    
    def _generate_schedule_cache_key(self, max_iterations: int) -> str:
        """Generate cache key for schedule optimization."""
        try:
            # Create key based on tasks and their critical properties
            task_fingerprint = []
            for task_id in sorted(self.tasks.keys()):
                task = self.tasks[task_id]
                fingerprint = f"{task_id}:{task.priority}:{len(task.dependencies)}:{len(task.required_resources)}"
                task_fingerprint.append(fingerprint)
            
            # Include annealing parameters
            key_data = f"schedule:{'|'.join(task_fingerprint)}:iter_{max_iterations}:temp_{self.temperature}"
            
            # Generate hash
            import hashlib
            return hashlib.sha256(key_data.encode()).hexdigest()[:16]
            
        except Exception as e:
            logger.warning(f"Failed to generate cache key: {e}")
            return f"schedule_fallback_{len(self.tasks)}_{max_iterations}"
    
    async def enable_load_balancer(self) -> bool:
        """Enable load balancing for task execution."""
        try:
            if self.load_balancer is None:
                self.load_balancer = QuantumLoadBalancer()
                await self.load_balancer.start_workers()
                logger.info("Load balancer enabled for quantum task planner")
                return True
            return True
        except Exception as e:
            logger.error(f"Failed to enable load balancer: {e}")
            return False
    
    async def disable_load_balancer(self) -> None:
        """Disable load balancing."""
        if self.load_balancer:
            await self.load_balancer.stop_workers()
            self.load_balancer = None
            logger.info("Load balancer disabled")
    
    async def execute_schedule_parallel(self, schedule: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute tasks in parallel where possible, respecting dependencies."""
        if not schedule:
            schedule = self.quantum_anneal_schedule()
        
        # Check circuit breaker
        if self.is_circuit_breaker_open():
            return {
                'error': 'Circuit breaker is open, execution blocked',
                'total_tasks': len(schedule),
                'successful_tasks': 0,
                'failed_tasks': 0,
                'start_time': datetime.now().isoformat(),
                'end_time': datetime.now().isoformat(),
                'task_results': {},
                'success_rate': 0.0
            }
        
        start_time = datetime.now()
        results = {
            'total_tasks': len(schedule),
            'successful_tasks': 0,
            'failed_tasks': 0,
            'start_time': start_time.isoformat(),
            'task_results': {},
            'parallel_batches': 0
        }
        
        logger.info(f"Starting parallel execution of {len(schedule)} tasks")
        
        try:
            # Build dependency graph
            remaining_tasks = set(schedule)
            completed_tasks = set()
            
            while remaining_tasks:
                # Find tasks that can run in parallel (dependencies satisfied)
                ready_tasks = []
                for task_id in remaining_tasks:
                    task = self.tasks[task_id]
                    if task.dependencies.issubset(completed_tasks):
                        ready_tasks.append(task_id)
                
                if not ready_tasks:
                    # Dependency deadlock - break with first remaining task
                    logger.warning("Dependency deadlock detected, executing remaining tasks sequentially")
                    ready_tasks = [next(iter(remaining_tasks))]
                
                results['parallel_batches'] += 1
                logger.debug(f"Executing batch {results['parallel_batches']} with {len(ready_tasks)} parallel tasks")
                
                # Execute ready tasks in parallel
                if self.load_balancer and len(ready_tasks) > 1:
                    # Use load balancer for parallel execution
                    batch_tasks = []
                    for task_id in ready_tasks:
                        batch_tasks.append(self._execute_task_with_load_balancer(task_id))
                    
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    for i, task_id in enumerate(ready_tasks):
                        result = batch_results[i]
                        if isinstance(result, Exception):
                            success = False
                            logger.error(f"Task {task_id} failed with exception: {result}")
                        else:
                            success = result
                        
                        results['task_results'][task_id] = success
                        if success:
                            results['successful_tasks'] += 1
                            completed_tasks.add(task_id)
                        else:
                            results['failed_tasks'] += 1
                            self.increment_error_count()
                        
                        remaining_tasks.discard(task_id)
                else:
                    # Execute tasks sequentially in this batch
                    for task_id in ready_tasks:
                        success = await self.execute_task(task_id)
                        results['task_results'][task_id] = success
                        
                        if success:
                            results['successful_tasks'] += 1
                            completed_tasks.add(task_id)
                        else:
                            results['failed_tasks'] += 1
                            self.increment_error_count()
                        
                        remaining_tasks.discard(task_id)
                
                # Break if circuit breaker opens during execution
                if self.is_circuit_breaker_open():
                    logger.warning("Circuit breaker opened during execution, stopping")
                    break
        
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}", exc_info=True)
            results['error'] = str(e)
        
        finally:
            results['end_time'] = datetime.now().isoformat()
            results['success_rate'] = results['successful_tasks'] / results['total_tasks'] if results['total_tasks'] > 0 else 0.0
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Parallel execution complete in {duration:.2f}s. "
                       f"Success rate: {results['success_rate']:.2%} "
                       f"({results['successful_tasks']}/{results['total_tasks']})")
        
        return results
    
    async def _execute_task_with_load_balancer(self, task_id: str) -> bool:
        """Execute task using load balancer."""
        if not self.load_balancer:
            return await self.execute_task(task_id)
        
        try:
            # Submit task to load balancer
            result = await self.load_balancer.submit_task(
                self._execute_task_sync, task_id
            )
            return result
        except Exception as e:
            logger.error(f"Load balanced execution failed for task {task_id}: {e}")
            return False
    
    def _execute_task_sync(self, task_id: str) -> bool:
        """Synchronous wrapper for task execution (for thread pool)."""
        # This would need to be implemented as a sync version of execute_task
        # For now, we'll simulate the execution
        try:
            task = self.tasks[task_id]
            
            # Simulate execution time
            import time
            time.sleep(min(task.estimated_duration.total_seconds(), 2.0))
            
            # Simple success probability
            success_probability = task.get_execution_probability()
            success = random.random() < success_probability
            
            if success:
                task.state = TaskState.COMPLETED
                task.completed_at = datetime.now()
            else:
                task.state = TaskState.FAILED
                task.error_message = "Load balanced execution failed"
            
            return success
            
        except Exception as e:
            logger.error(f"Sync task execution failed: {e}")
            return False
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization system statistics."""
        stats = {
            'cache': self.cache.get_stats() if self.cache else None,
            'load_balancer': self.load_balancer.get_stats() if self.load_balancer else None,
            'cache_enabled': self.cache_enabled,
            'parallel_execution_enabled': self.parallel_execution_enabled,
            'optimization_features': {
                'caching': self.cache_enabled and bool(self.cache),
                'load_balancing': bool(self.load_balancer),
                'parallel_execution': self.parallel_execution_enabled
            }
        }
        
        return stats
    
    def _calculate_schedule_cost(self, schedule: List[str]) -> float:
        """Calculate the cost of a given task schedule."""
        cost = 0.0
        current_time = datetime.now()
        resource_usage = {
            'cpu_cores': 0.0,
            'memory_gb': 0.0,
            'storage_gb': 0.0,
            'network_bandwidth': 0.0
        }
        
        for task_id in schedule:
            task = self.tasks[task_id]
            
            # Dependency violation penalty
            for dep_id in task.dependencies:
                if dep_id in schedule and schedule.index(dep_id) > schedule.index(task_id):
                    cost += 1000.0  # Heavy penalty for dependency violations
            
            # Resource contention cost
            for resource, requirement in task.required_resources.items():
                if resource_usage.get(resource, 0.0) + requirement > getattr(self.resource_pool, resource.replace('_', ''), 0.0):
                    cost += 100.0 * requirement  # Penalty for resource conflicts
                else:
                    resource_usage[resource] = resource_usage.get(resource, 0.0) + requirement
            
            # Priority-based timing cost
            priority_factor = (11.0 - task.priority) / 10.0  # Higher priority = lower cost
            time_cost = schedule.index(task_id) * priority_factor
            cost += time_cost
            
            # Interference effects
            for interfering_task_id, effect in task.interference_pattern.items():
                if interfering_task_id in schedule:
                    position_delta = abs(schedule.index(task_id) - schedule.index(interfering_task_id))
                    interference_cost = -effect * math.exp(-position_delta / 10.0)
                    cost += interference_cost
        
        return cost
    
    async def execute_task(self, task_id: str) -> bool:
        """Execute a single task with comprehensive error handling and monitoring."""
        execution_start_time = time.time()
        
        try:
            # Validate task exists
            if task_id not in self.tasks:
                error_msg = f"Task {task_id} not found"
                logger.error(error_msg)
                if hasattr(self, 'monitor') and self.monitor:
                    self.monitor._record_error("task_execution", error_msg, "error")
                return False
            
            task = self.tasks[task_id]
            
            # Check if task is already executing or completed
            if task.state in [TaskState.EXECUTING, TaskState.COMPLETED]:
                warning_msg = f"Task {task_id} is already {task.state.value}"
                logger.warning(warning_msg)
                return task.state == TaskState.COMPLETED
            
            # Check dependencies with detailed validation
            unmet_dependencies = []
            for dep_id in task.dependencies:
                dep_task = self.tasks.get(dep_id)
                if not dep_task:
                    unmet_dependencies.append(f"{dep_id} (not found)")
                elif dep_task.state != TaskState.COMPLETED:
                    unmet_dependencies.append(f"{dep_id} ({dep_task.state.value})")
            
            if unmet_dependencies:
                error_msg = f"Task {task_id} dependencies not satisfied: {', '.join(unmet_dependencies)}"
                logger.warning(error_msg)
                if hasattr(self, 'monitor') and self.monitor:
                    self.monitor._record_error("dependency_check", error_msg, "warning")
                return False
            
            # Check resource availability with detailed reporting
            if not self.resource_pool.can_allocate(task.required_resources):
                resource_status = []
                for resource, required in task.required_resources.items():
                    available = getattr(self.resource_pool, resource, 0) - getattr(self.resource_pool, f'allocated_{resource.split("_")[0]}', 0)
                    resource_status.append(f"{resource}: need {required}, have {available}")
                
                error_msg = f"Insufficient resources for task {task_id}: {'; '.join(resource_status)}"
                logger.warning(error_msg)
                if hasattr(self, 'monitor') and self.monitor:
                    self.monitor._record_error("resource_allocation", error_msg, "warning")
                return False
            
            # Begin task execution
            try:
                # Collapse superposition and allocate resources
                task.collapse_state(TaskState.EXECUTING)
                task.started_at = datetime.now()
                
                # Record quantum event
                if hasattr(self, 'monitor') and self.monitor:
                    self.monitor.record_quantum_event("superposition_collapse", task_id)
                
                if not self.resource_pool.allocate(task.required_resources):
                    raise RuntimeError("Resource allocation failed after availability check")
                
                logger.info(f"Executing task {task_id}: {task.name}")
                
                # Calculate execution parameters
                execution_time = task.estimated_duration.total_seconds()
                success_probability = task.get_execution_probability()
                
                # Apply interference effects
                if hasattr(self, 'monitor') and self.monitor and task.interference_pattern:
                    self.monitor.record_quantum_event("interference_applied", task_id, {
                        'interference_count': len(task.interference_pattern),
                        'total_effect': sum(task.interference_pattern.values())
                    })
                
                # Simulate task execution with timeout protection
                simulation_time = min(execution_time, 10.0)  # Cap simulation time to 10 seconds
                await asyncio.wait_for(
                    asyncio.sleep(simulation_time), 
                    timeout=15.0  # Safety timeout
                )
                
                # Determine success based on quantum probability with enhanced logic
                random_value = random.random()
                base_success = random_value < success_probability
                
                # Apply additional success factors
                priority_boost = min(task.priority / 10.0 * 0.1, 0.1)  # Up to 10% boost for high priority
                execution_penalty = max(0, (execution_time - 300) / 3600 * 0.1)  # Penalty for long tasks
                
                adjusted_probability = success_probability + priority_boost - execution_penalty
                final_success = random_value < adjusted_probability
                
                execution_end_time = time.time()
                execution_duration_ms = (execution_end_time - execution_start_time) * 1000
                
                if final_success:
                    task.state = TaskState.COMPLETED
                    task.completed_at = datetime.now()
                    success = True
                    logger.info(f"Task {task_id} completed successfully in {execution_duration_ms:.2f}ms")
                else:
                    task.state = TaskState.FAILED
                    task.error_message = f"Quantum execution probability threshold not met (p={success_probability:.3f}, rolled={random_value:.3f})"
                    success = False
                    logger.warning(f"Task {task_id} failed during execution: {task.error_message}")
                
                # Record comprehensive execution metrics
                execution_record = {
                    'task_id': task_id,
                    'started_at': task.started_at.isoformat(),
                    'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                    'success': success,
                    'execution_probability': success_probability,
                    'adjusted_probability': adjusted_probability,
                    'random_value': random_value,
                    'execution_duration_ms': execution_duration_ms,
                    'resource_requirements': task.required_resources.copy(),
                    'priority': task.priority,
                    'interference_effects': len(task.interference_pattern),
                    'dependencies_count': len(task.dependencies)
                }
                
                self.execution_history.append(execution_record)
                
                # Record monitoring metrics
                if hasattr(self, 'monitor') and self.monitor:
                    self.monitor.record_task_execution(
                        task_id, execution_duration_ms, success, success_probability
                    )
                
                return success
                
            except asyncio.TimeoutError:
                error_msg = f"Task {task_id} execution timed out"
                task.state = TaskState.FAILED
                task.error_message = error_msg
                logger.error(error_msg)
                if hasattr(self, 'monitor') and self.monitor:
                    self.monitor._record_error("task_execution", error_msg, "error")
                return False
                
            except Exception as execution_error:
                error_msg = f"Task {task_id} execution failed: {str(execution_error)}"
                task.state = TaskState.FAILED
                task.error_message = str(execution_error)
                logger.error(error_msg, exc_info=True)
                if hasattr(self, 'monitor') and self.monitor:
                    self.monitor._record_error("task_execution", error_msg, "error")
                return False
                
        except Exception as outer_error:
            error_msg = f"Critical error executing task {task_id}: {str(outer_error)}"
            logger.error(error_msg, exc_info=True)
            if hasattr(self, 'monitor') and self.monitor:
                self.monitor._record_error("task_execution", error_msg, "critical")
            
            # Attempt to set task state
            try:
                if task_id in self.tasks:
                    self.tasks[task_id].state = TaskState.FAILED
                    self.tasks[task_id].error_message = str(outer_error)
            except:
                pass  # Don't let cleanup errors mask the original error
            
            return False
            
        finally:
            # Always attempt to deallocate resources
            try:
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    if task.required_resources:
                        self.resource_pool.deallocate(task.required_resources)
                        logger.debug(f"Deallocated resources for task {task_id}")
            except Exception as cleanup_error:
                logger.error(f"Failed to deallocate resources for task {task_id}: {cleanup_error}")
                if hasattr(self, 'monitor') and self.monitor:
                    self.monitor._record_error("resource_cleanup", str(cleanup_error), "warning")
    
    async def execute_schedule(self, schedule: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute tasks according to the optimal schedule."""
        if not schedule:
            schedule = self.quantum_anneal_schedule()
        
        results = {
            'total_tasks': len(schedule),
            'successful_tasks': 0,
            'failed_tasks': 0,
            'start_time': datetime.now().isoformat(),
            'task_results': {}
        }
        
        logger.info(f"Starting execution of {len(schedule)} tasks")
        
        for task_id in schedule:
            success = await self.execute_task(task_id)
            results['task_results'][task_id] = success
            
            if success:
                results['successful_tasks'] += 1
            else:
                results['failed_tasks'] += 1
        
        results['end_time'] = datetime.now().isoformat()
        results['success_rate'] = results['successful_tasks'] / results['total_tasks']
        
        logger.info(f"Execution complete. Success rate: {results['success_rate']:.2%}")
        return results
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get current state of the quantum task planning system."""
        task_states = {}
        for task_id, task in self.tasks.items():
            task_states[task_id] = {
                'state': task.state.value,
                'probability': task.get_execution_probability(),
                'entangled_with': list(task.entangled_tasks),
                'dependencies': list(task.dependencies)
            }
        
        return {
            'total_tasks': len(self.tasks),
            'resource_utilization': {
                'cpu': self.resource_pool.allocated_cpu / self.resource_pool.cpu_cores,
                'memory': self.resource_pool.allocated_memory / self.resource_pool.memory_gb,
                'storage': self.resource_pool.allocated_storage / self.resource_pool.storage_gb,
                'bandwidth': self.resource_pool.allocated_bandwidth / self.resource_pool.network_bandwidth
            },
            'task_states': task_states,
            'execution_history_size': len(self.execution_history)
        }


async def demo_quantum_planning():
    """Demonstrate quantum-inspired task planning."""
    planner = QuantumTaskPlanner()
    
    # Create sample tasks
    tasks = [
        QuantumTask(
            id="llm_inference",
            name="Run LLM Inference",
            description="Execute large language model inference",
            priority=9.0,
            estimated_duration=timedelta(minutes=5),
            required_resources={'cpu_cores': 2.0, 'memory_gb': 4.0}
        ),
        QuantumTask(
            id="cost_analysis",
            name="Analyze Costs",
            description="Analyze LLM usage costs",
            priority=7.0,
            estimated_duration=timedelta(minutes=2),
            required_resources={'cpu_cores': 1.0, 'memory_gb': 2.0}
        ),
        QuantumTask(
            id="generate_report",
            name="Generate Report",
            description="Generate cost analysis report",
            priority=6.0,
            estimated_duration=timedelta(minutes=3),
            required_resources={'cpu_cores': 0.5, 'memory_gb': 1.0}
        ),
        QuantumTask(
            id="database_backup",
            name="Database Backup",
            description="Backup LLM usage database",
            priority=8.0,
            estimated_duration=timedelta(minutes=10),
            required_resources={'storage_gb': 20.0, 'network_bandwidth': 50.0}
        )
    ]
    
    # Add tasks to planner
    for task in tasks:
        planner.add_task(task)
    
    # Create dependencies
    planner.create_dependency("cost_analysis", "llm_inference")
    planner.create_dependency("generate_report", "cost_analysis")
    
    # Get optimal schedule using quantum annealing
    optimal_schedule = planner.quantum_anneal_schedule()
    
    print(f"Optimal execution order: {optimal_schedule}")
    print(f"System state: {planner.get_system_state()}")
    
    # Execute the schedule
    results = await planner.execute_schedule(optimal_schedule)
    print(f"Execution results: {results}")
    
    return planner, results


if __name__ == "__main__":
    asyncio.run(demo_quantum_planning())