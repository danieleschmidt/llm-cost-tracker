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
import logging
import math
import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .quantum_optimization import QuantumCache, QuantumLoadBalancer, memoize_with_ttl
from .quantum_validation import (
    ValidationSeverity,
    sanitize_task_input,
    validate_task_input,
)

logger = logging.getLogger(__name__)


class TaskState(Enum):
    """Quantum-inspired task states."""

    SUPERPOSITION = "superposition"  # Task exists in multiple potential states
    COLLAPSED = "collapsed"  # Task state has been measured/determined
    ENTANGLED = "entangled"  # Task is dependent on other tasks
    EXECUTING = "executing"  # Task is currently running
    COMPLETED = "completed"  # Task finished successfully
    FAILED = "failed"  # Task execution failed


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
            interference_factor *= 1.0 + effect

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
        cpu_req = requirements.get("cpu_cores", 0.0)
        mem_req = requirements.get("memory_gb", 0.0)
        storage_req = requirements.get("storage_gb", 0.0)
        bandwidth_req = requirements.get("network_bandwidth", 0.0)

        return (
            self.allocated_cpu + cpu_req <= self.cpu_cores
            and self.allocated_memory + mem_req <= self.memory_gb
            and self.allocated_storage + storage_req <= self.storage_gb
            and self.allocated_bandwidth + bandwidth_req <= self.network_bandwidth
        )

    def allocate(self, requirements: Dict[str, float]) -> bool:
        """Allocate resources if available."""
        if self.can_allocate(requirements):
            self.allocated_cpu += requirements.get("cpu_cores", 0.0)
            self.allocated_memory += requirements.get("memory_gb", 0.0)
            self.allocated_storage += requirements.get("storage_gb", 0.0)
            self.allocated_bandwidth += requirements.get("network_bandwidth", 0.0)
            return True
        return False

    def deallocate(self, requirements: Dict[str, float]) -> None:
        """Deallocate resources."""
        self.allocated_cpu -= requirements.get("cpu_cores", 0.0)
        self.allocated_memory -= requirements.get("memory_gb", 0.0)
        self.allocated_storage -= requirements.get("storage_gb", 0.0)
        self.allocated_bandwidth -= requirements.get("network_bandwidth", 0.0)

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
                "id": task.id,
                "name": task.name,
                "description": task.description,
                "priority": task.priority,
                "estimated_duration": task.estimated_duration,
                "required_resources": task.required_resources,
                "dependencies": task.dependencies,
                "probability_amplitude": task.probability_amplitude,
                "interference_pattern": task.interference_pattern,
                "entangled_tasks": task.entangled_tasks,
            }

            validation_result = validate_task_input(task_data, set(self.tasks.keys()))

            # Handle validation failures
            if not validation_result.is_valid:
                error_messages = []
                for error in (
                    validation_result.critical_errors + validation_result.errors
                ):
                    error_messages.append(f"{error.field}: {error.message}")

                error_msg = f"Task validation failed: {'; '.join(error_messages)}"
                logger.error(f"Failed to add task {task.id}: {error_msg}")
                return False, error_msg

            # Log warnings
            for warning in validation_result.warnings:
                logger.warning(
                    f"Task {task.id} validation warning - {warning.field}: {warning.message}"
                )

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
                logger.warning(
                    f"Failed to calculate interference patterns for task {task.id}: {e}"
                )
                # Continue anyway as this is not critical

            # Record quantum event if monitoring is available
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.record_quantum_event(
                    "task_added",
                    task.id,
                    {
                        "priority": task.priority,
                        "resource_count": len(task.required_resources),
                        "dependency_count": len(task.dependencies),
                    },
                )

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
                task.required_resources, existing_task.required_resources
            )

            # Negative interference for resource conflicts
            interference_effect = -0.1 * resource_overlap

            # Positive interference for complementary tasks
            if self._are_complementary(task, existing_task):
                interference_effect += 0.15

            # Store interference pattern
            task.interference_pattern[existing_id] = interference_effect
            existing_task.interference_pattern[task.id] = interference_effect

    def _calculate_resource_overlap(
        self, resources1: Dict[str, float], resources2: Dict[str, float]
    ) -> float:
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
        resource_conflict = (
            self._calculate_resource_overlap(
                task1.required_resources, task2.required_resources
            )
            > 0.5
        )

        return high_priority and resource_conflict

    def generate_schedule(self, max_iterations: int = 100) -> List[str]:
        """Generate optimal schedule using quantum annealing."""
        return self.quantum_anneal_schedule(max_iterations)
    
    def quantum_anneal_schedule(self, max_iterations: int = 1000) -> List[str]:
        """
        Advanced quantum annealing with multiple enhancement techniques:
        - Adaptive temperature scheduling with quantum tunneling
        - Multi-objective optimization with Pareto fronts
        - Entanglement-aware neighborhood generation
        - Quantum interference pattern optimization
        """
        # Generate cache key based on current tasks and their properties
        if self.cache_enabled:
            cache_key = self._generate_schedule_cache_key(max_iterations)
            cached_schedule = self.cache.get(cache_key)
            if cached_schedule is not None:
                logger.debug("Using cached schedule from quantum cache")
                return cached_schedule

        if not self.tasks:
            return []

        # Initialize population for quantum-inspired genetic algorithm
        population_size = min(20, max(4, len(self.tasks) // 2))
        population = self._initialize_quantum_population(population_size)

        # Multi-objective tracking
        pareto_front = []
        convergence_history = []

        # Enhanced quantum annealing parameters
        initial_temperature = self.temperature
        quantum_tunneling_probability = 0.1
        entanglement_factor = 0.2

        best_schedule = population[0].copy()
        best_cost = float("inf")

        for iteration in range(max_iterations):
            temperature = self._adaptive_temperature_schedule(
                iteration, max_iterations, initial_temperature
            )

            # Process each solution in population
            new_population = []

            for schedule in population:
                # Generate quantum-inspired neighbors using multiple operators
                neighbors = self._generate_quantum_neighbors(
                    schedule, entanglement_factor, temperature
                )

                # Evaluate all neighbors with multi-objective criteria
                for neighbor in neighbors:
                    cost_metrics = self._calculate_multi_objective_cost(neighbor)
                    primary_cost = cost_metrics["total_cost"]

                    # Quantum tunneling: occasionally accept worse solutions for exploration
                    tunneling_probability = quantum_tunneling_probability * math.exp(
                        -iteration / (max_iterations * 0.3)
                    )

                    if (
                        primary_cost < best_cost
                        or random.random() < tunneling_probability
                        or self._quantum_acceptance_probability(
                            cost_metrics, temperature
                        )
                        > random.random()
                    ):

                        new_population.append(neighbor)

                        if primary_cost < best_cost:
                            best_schedule = neighbor.copy()
                            best_cost = primary_cost
                            logger.debug(
                                f"New best schedule found at iteration {iteration}, cost: {best_cost:.4f}"
                            )

                        # Update Pareto front for multi-objective optimization
                        self._update_pareto_front(pareto_front, neighbor, cost_metrics)

            # Selection and mutation for next generation
            if new_population:
                # Select best candidates with diversity preservation
                population = self._quantum_selection(
                    new_population + population, population_size
                )

            # Apply quantum interference patterns
            if iteration % 50 == 0:
                population = self._apply_quantum_interference(population)

            # Convergence tracking
            convergence_history.append(
                {
                    "iteration": iteration,
                    "best_cost": best_cost,
                    "temperature": temperature,
                    "population_diversity": self._calculate_population_diversity(
                        population
                    ),
                    "pareto_front_size": len(pareto_front),
                }
            )

            # Early termination conditions
            if self._check_convergence(
                convergence_history, min_iterations=max_iterations // 4
            ):
                logger.info(
                    f"Quantum annealing converged early at iteration {iteration}"
                )
                break

        # Final optimization using best solution from Pareto front
        if pareto_front:
            pareto_best = min(pareto_front, key=lambda x: x[1]["total_cost"])
            if pareto_best[1]["total_cost"] < best_cost:
                best_schedule = pareto_best[0]
                best_cost = pareto_best[1]["total_cost"]

        logger.info(
            f"Advanced quantum annealing complete. Best cost: {best_cost:.4f}, "
            f"Pareto solutions: {len(pareto_front)}, Iterations: {iteration + 1}"
        )

        # Cache the result with extended metadata
        if self.cache_enabled:
            cache_key = self._generate_schedule_cache_key(max_iterations)
            cache_metadata = {
                "schedule": best_schedule,
                "cost": best_cost,
                "pareto_front_size": len(pareto_front),
                "convergence_iterations": iteration + 1,
                "algorithm": "advanced_quantum_annealing_v2",
            }
            self.cache.put(cache_key, cache_metadata, ttl=600)  # Cache for 10 minutes

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
                "timestamp": self.last_health_check.isoformat(),
                "overall_healthy": True,
                "components": {},
                "metrics": {},
                "issues": [],
            }

            # Check task system health
            try:
                task_count = len(self.tasks)
                executing_tasks = sum(
                    1 for t in self.tasks.values() if t.state == TaskState.EXECUTING
                )
                completed_tasks = sum(
                    1 for t in self.tasks.values() if t.state == TaskState.COMPLETED
                )
                failed_tasks = sum(
                    1 for t in self.tasks.values() if t.state == TaskState.FAILED
                )

                health_status["components"]["task_system"] = {
                    "healthy": True,
                    "total_tasks": task_count,
                    "executing": executing_tasks,
                    "completed": completed_tasks,
                    "failed": failed_tasks,
                }

                if failed_tasks > completed_tasks and task_count > 5:
                    health_status["components"]["task_system"]["healthy"] = False
                    health_status["issues"].append("High task failure rate detected")

            except Exception as e:
                health_status["components"]["task_system"] = {
                    "healthy": False,
                    "error": str(e),
                }
                health_status["issues"].append(f"Task system check failed: {e}")

            # Check resource system health
            try:
                pool = self.resource_pool
                cpu_utilization = (
                    pool.allocated_cpu / pool.cpu_cores if pool.cpu_cores > 0 else 0
                )
                memory_utilization = (
                    pool.allocated_memory / pool.memory_gb if pool.memory_gb > 0 else 0
                )

                health_status["components"]["resource_system"] = {
                    "healthy": True,
                    "cpu_utilization": cpu_utilization,
                    "memory_utilization": memory_utilization,
                    "overallocated": any(
                        [
                            pool.allocated_cpu > pool.cpu_cores,
                            pool.allocated_memory > pool.memory_gb,
                            pool.allocated_storage > pool.storage_gb,
                            pool.allocated_bandwidth > pool.network_bandwidth,
                        ]
                    ),
                }

                if cpu_utilization > 0.95 or memory_utilization > 0.95:
                    health_status["components"]["resource_system"]["healthy"] = False
                    health_status["issues"].append(
                        "Resource utilization critically high"
                    )

            except Exception as e:
                health_status["components"]["resource_system"] = {
                    "healthy": False,
                    "error": str(e),
                }
                health_status["issues"].append(f"Resource system check failed: {e}")

            # Check scheduler health
            try:
                if len(self.tasks) > 1:
                    test_schedule = self.quantum_anneal_schedule(max_iterations=5)
                    scheduler_healthy = len(test_schedule) == len(self.tasks)
                else:
                    scheduler_healthy = True

                health_status["components"]["scheduler"] = {
                    "healthy": scheduler_healthy,
                    "can_generate_schedule": scheduler_healthy,
                }

                if not scheduler_healthy:
                    health_status["issues"].append(
                        "Scheduler unable to generate valid schedules"
                    )

            except Exception as e:
                health_status["components"]["scheduler"] = {
                    "healthy": False,
                    "error": str(e),
                }
                health_status["issues"].append(f"Scheduler check failed: {e}")

            # Check error rate
            try:
                recent_executions = [
                    h
                    for h in self.execution_history
                    if (
                        datetime.now() - datetime.fromisoformat(h["started_at"])
                    ).total_seconds()
                    < 3600
                ]

                if len(recent_executions) > 0:
                    error_rate = sum(
                        1 for h in recent_executions if not h["success"]
                    ) / len(recent_executions)
                    health_status["metrics"]["error_rate"] = error_rate

                    if error_rate > 0.5:  # More than 50% failure rate
                        health_status["issues"].append(
                            f"High error rate: {error_rate:.1%}"
                        )

            except Exception as e:
                health_status["issues"].append(f"Error rate check failed: {e}")

            # Overall health determination
            component_health = [
                comp.get("healthy", False)
                for comp in health_status["components"].values()
            ]
            health_status["overall_healthy"] = (
                all(component_health) and len(health_status["issues"]) == 0
            )

            # Record health check completion
            if self.monitor:
                if not health_status["overall_healthy"]:
                    self.monitor._record_error(
                        "health_check",
                        f"Health issues detected: {'; '.join(health_status['issues'])}",
                        "warning",
                    )

            return health_status

        except Exception as e:
            logger.error(f"Health check failed with exception: {e}", exc_info=True)
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_healthy": False,
                "error": str(e),
                "components": {},
                "metrics": {},
                "issues": [f"Health check system failure: {e}"],
            }

    def is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker should prevent new task execution."""
        if not self.health_check_enabled:
            return False

        # Check error count threshold
        if self.error_count >= self.max_error_threshold:
            logger.warning(
                f"Circuit breaker OPEN: Error count ({self.error_count}) exceeds threshold ({self.max_error_threshold})"
            )
            return True

        # Check recent failure rate
        recent_history = (
            self.execution_history[-20:]
            if len(self.execution_history) >= 20
            else self.execution_history
        )
        if len(recent_history) >= 5:
            failure_rate = sum(1 for h in recent_history if not h["success"]) / len(
                recent_history
            )
            if failure_rate > 0.8:  # 80% failure rate
                logger.warning(
                    f"Circuit breaker OPEN: High failure rate ({failure_rate:.1%})"
                )
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
            logger.warning(
                f"Circuit breaker threshold reached: {self.error_count} errors"
            )

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

    async def execute_schedule_parallel(
        self, schedule: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute tasks in parallel where possible, respecting dependencies."""
        if not schedule:
            schedule = self.quantum_anneal_schedule()

        # Check circuit breaker
        if self.is_circuit_breaker_open():
            return {
                "error": "Circuit breaker is open, execution blocked",
                "total_tasks": len(schedule),
                "successful_tasks": 0,
                "failed_tasks": 0,
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "task_results": {},
                "success_rate": 0.0,
            }

        start_time = datetime.now()
        results = {
            "total_tasks": len(schedule),
            "successful_tasks": 0,
            "failed_tasks": 0,
            "start_time": start_time.isoformat(),
            "task_results": {},
            "parallel_batches": 0,
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
                    logger.warning(
                        "Dependency deadlock detected, executing remaining tasks sequentially"
                    )
                    ready_tasks = [next(iter(remaining_tasks))]

                results["parallel_batches"] += 1
                logger.debug(
                    f"Executing batch {results['parallel_batches']} with {len(ready_tasks)} parallel tasks"
                )

                # Execute ready tasks in parallel
                if self.load_balancer and len(ready_tasks) > 1:
                    # Use load balancer for parallel execution
                    batch_tasks = []
                    for task_id in ready_tasks:
                        batch_tasks.append(
                            self._execute_task_with_load_balancer(task_id)
                        )

                    batch_results = await asyncio.gather(
                        *batch_tasks, return_exceptions=True
                    )

                    for i, task_id in enumerate(ready_tasks):
                        result = batch_results[i]
                        if isinstance(result, Exception):
                            success = False
                            logger.error(
                                f"Task {task_id} failed with exception: {result}"
                            )
                        else:
                            success = result

                        results["task_results"][task_id] = success
                        if success:
                            results["successful_tasks"] += 1
                            completed_tasks.add(task_id)
                        else:
                            results["failed_tasks"] += 1
                            self.increment_error_count()

                        remaining_tasks.discard(task_id)
                else:
                    # Execute tasks sequentially in this batch
                    for task_id in ready_tasks:
                        success = await self.execute_task(task_id)
                        results["task_results"][task_id] = success

                        if success:
                            results["successful_tasks"] += 1
                            completed_tasks.add(task_id)
                        else:
                            results["failed_tasks"] += 1
                            self.increment_error_count()

                        remaining_tasks.discard(task_id)

                # Break if circuit breaker opens during execution
                if self.is_circuit_breaker_open():
                    logger.warning("Circuit breaker opened during execution, stopping")
                    break

        except Exception as e:
            logger.error(f"Parallel execution failed: {e}", exc_info=True)
            results["error"] = str(e)

        finally:
            results["end_time"] = datetime.now().isoformat()
            results["success_rate"] = (
                results["successful_tasks"] / results["total_tasks"]
                if results["total_tasks"] > 0
                else 0.0
            )

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Parallel execution complete in {duration:.2f}s. "
                f"Success rate: {results['success_rate']:.2%} "
                f"({results['successful_tasks']}/{results['total_tasks']})"
            )

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
            "cache": self.cache.get_stats() if self.cache else None,
            "load_balancer": (
                self.load_balancer.get_stats() if self.load_balancer else None
            ),
            "cache_enabled": self.cache_enabled,
            "parallel_execution_enabled": self.parallel_execution_enabled,
            "optimization_features": {
                "caching": self.cache_enabled and bool(self.cache),
                "load_balancing": bool(self.load_balancer),
                "parallel_execution": self.parallel_execution_enabled,
            },
        }

        return stats

    def _initialize_quantum_population(self, population_size: int) -> List[List[str]]:
        """Initialize population of schedules using quantum-inspired heuristics."""
        population = []
        task_ids = list(self.tasks.keys())

        # Create diverse initial schedules
        for i in range(population_size):
            if i == 0:
                # Priority-based greedy schedule
                schedule = sorted(
                    task_ids, key=lambda tid: self.tasks[tid].priority, reverse=True
                )
            elif i == 1:
                # Dependency-respecting topological sort
                schedule = self._topological_sort_schedule(task_ids)
            elif i == 2:
                # Resource-optimized schedule
                schedule = self._resource_optimized_schedule(task_ids)
            else:
                # Random with quantum-inspired bias
                schedule = task_ids.copy()
                # Apply quantum superposition bias - prefer higher probability tasks first
                for j in range(len(schedule)):
                    if random.random() < 0.3:  # 30% quantum bias
                        priorities = [
                            self.tasks[tid].get_execution_probability()
                            for tid in schedule[j:]
                        ]
                        if priorities:
                            best_idx = j + priorities.index(max(priorities))
                            schedule[j], schedule[best_idx] = (
                                schedule[best_idx],
                                schedule[j],
                            )
                random.shuffle(schedule)

            population.append(schedule)

        return population

    def _topological_sort_schedule(self, task_ids: List[str]) -> List[str]:
        """Create a schedule respecting all dependencies using topological sorting."""
        from collections import defaultdict, deque

        # Build dependency graph
        graph = defaultdict(list)
        in_degree = defaultdict(int)

        for task_id in task_ids:
            in_degree[task_id] = 0

        for task_id in task_ids:
            task = self.tasks[task_id]
            for dep_id in task.dependencies:
                if dep_id in task_ids:
                    graph[dep_id].append(task_id)
                    in_degree[task_id] += 1

        # Topological sort with priority queue
        queue = deque()
        for task_id in task_ids:
            if in_degree[task_id] == 0:
                queue.append(task_id)

        result = []
        while queue:
            # Sort by priority among ready tasks
            current_batch = list(queue)
            queue.clear()
            current_batch.sort(key=lambda tid: self.tasks[tid].priority, reverse=True)

            for task_id in current_batch:
                result.append(task_id)
                for neighbor in graph[task_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        return result

    def _resource_optimized_schedule(self, task_ids: List[str]) -> List[str]:
        """Create schedule optimizing for resource utilization."""
        schedule = []
        remaining = set(task_ids)
        current_resources = {
            "cpu_cores": 0.0,
            "memory_gb": 0.0,
            "storage_gb": 0.0,
            "network_bandwidth": 0.0,
        }

        while remaining:
            # Find best task that fits current resources
            best_task = None
            best_score = -1

            for task_id in remaining:
                task = self.tasks[task_id]

                # Check if dependencies are satisfied
                if not task.dependencies.issubset(set(schedule)):
                    continue

                # Check resource requirements
                resource_score = 0
                fits = True
                for resource, needed in task.required_resources.items():
                    available = getattr(
                        self.resource_pool, resource, 0
                    ) - current_resources.get(resource, 0)
                    if needed > available:
                        fits = False
                        break
                    resource_score += (
                        min(1.0, needed / available) if available > 0 else 0
                    )

                if fits:
                    # Combined score: priority + resource efficiency
                    combined_score = task.priority + resource_score * 0.5
                    if combined_score > best_score:
                        best_score = combined_score
                        best_task = task_id

            if best_task:
                schedule.append(best_task)
                remaining.remove(best_task)
                # Update resource usage
                for resource, needed in self.tasks[
                    best_task
                ].required_resources.items():
                    current_resources[resource] = (
                        current_resources.get(resource, 0) + needed
                    )
            else:
                # No task fits, add next by priority
                next_task = max(remaining, key=lambda tid: self.tasks[tid].priority)
                schedule.append(next_task)
                remaining.remove(next_task)
                current_resources = {
                    resource: 0.0 for resource in current_resources
                }  # Reset resources

        return schedule

    def _adaptive_temperature_schedule(
        self, iteration: int, max_iterations: int, initial_temp: float
    ) -> float:
        """Advanced adaptive temperature scheduling with quantum tunneling phases."""
        progress = iteration / max_iterations

        # Multi-phase cooling with quantum tunneling
        if progress < 0.3:
            # High temperature exploration phase
            return initial_temp * (1.0 - progress * 0.5)
        elif progress < 0.7:
            # Quantum tunneling phase - periodic temperature increases
            base_temp = initial_temp * (1.0 - progress * 0.8)
            tunneling_effect = 0.3 * math.sin(progress * math.pi * 4) * initial_temp
            return max(base_temp + tunneling_effect, self.min_temperature)
        else:
            # Final convergence phase - rapid cooling
            return max(
                initial_temp * math.exp(-10 * (progress - 0.7)), self.min_temperature
            )

    def _generate_quantum_neighbors(
        self, schedule: List[str], entanglement_factor: float, temperature: float
    ) -> List[List[str]]:
        """Generate neighbors using quantum-inspired operators."""
        neighbors = []

        if len(schedule) < 2:
            return [schedule]

        # 1. Quantum swap based on entanglement
        for _ in range(max(2, int(len(schedule) * entanglement_factor))):
            neighbor = schedule.copy()
            # Prefer swapping entangled tasks
            entangled_pairs = []
            for i, task_id in enumerate(schedule):
                task = self.tasks[task_id]
                for j, other_id in enumerate(schedule):
                    if other_id in task.entangled_tasks:
                        entangled_pairs.append((i, j))

            if entangled_pairs and random.random() < 0.6:
                i, j = random.choice(entangled_pairs)
            else:
                i, j = random.sample(range(len(schedule)), 2)

            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)

        # 2. Quantum interference-based reordering
        if temperature > self.min_temperature * 5:  # Only at higher temperatures
            for _ in range(2):
                neighbor = schedule.copy()
                # Find tasks with strong interference patterns
                interference_tasks = []
                for i, task_id in enumerate(schedule):
                    task = self.tasks[task_id]
                    if task.interference_pattern:
                        total_interference = sum(
                            abs(effect) for effect in task.interference_pattern.values()
                        )
                        if total_interference > 0.5:
                            interference_tasks.append(i)

                if len(interference_tasks) >= 2:
                    # Reorder interference tasks
                    indices = random.sample(
                        interference_tasks, min(3, len(interference_tasks))
                    )
                    tasks_to_reorder = [neighbor[i] for i in indices]
                    random.shuffle(tasks_to_reorder)
                    for i, task_id in enumerate(tasks_to_reorder):
                        neighbor[indices[i]] = task_id

                neighbors.append(neighbor)

        # 3. Priority-based local optimization
        neighbor = schedule.copy()
        for i in range(len(schedule) - 1):
            task1 = self.tasks[schedule[i]]
            task2 = self.tasks[schedule[i + 1]]

            # Swap if priority order is wrong and no dependency conflict
            if (
                task2.priority > task1.priority
                and schedule[i] not in task2.dependencies
                and schedule[i + 1] not in task1.dependencies
            ):
                neighbor[i], neighbor[i + 1] = neighbor[i + 1], neighbor[i]
                break

        neighbors.append(neighbor)

        return neighbors

    def _calculate_multi_objective_cost(self, schedule: List[str]) -> Dict[str, float]:
        """Calculate multiple objectives for Pareto optimization."""
        metrics = {
            "total_cost": 0.0,
            "resource_efficiency": 0.0,
            "dependency_violations": 0.0,
            "priority_score": 0.0,
            "execution_time": 0.0,
            "quantum_coherence": 0.0,
        }

        current_time = 0.0
        resource_usage = {
            "cpu_cores": 0.0,
            "memory_gb": 0.0,
            "storage_gb": 0.0,
            "network_bandwidth": 0.0,
        }

        for pos, task_id in enumerate(schedule):
            task = self.tasks[task_id]

            # Dependency violation cost
            for dep_id in task.dependencies:
                if dep_id in schedule and schedule.index(dep_id) > pos:
                    metrics["dependency_violations"] += 100.0

            # Resource efficiency
            total_required = sum(task.required_resources.values())
            total_available = sum(
                getattr(self.resource_pool, attr, 0)
                for attr in [
                    "cpu_cores",
                    "memory_gb",
                    "storage_gb",
                    "network_bandwidth",
                ]
            )
            if total_available > 0:
                metrics["resource_efficiency"] += total_required / total_available

            # Priority score (higher is better, so negate)
            metrics["priority_score"] -= task.priority * (len(schedule) - pos)

            # Execution time simulation
            execution_duration = task.estimated_duration.total_seconds()
            metrics["execution_time"] += execution_duration
            current_time += execution_duration

            # Quantum coherence (measure of quantum properties utilization)
            coherence_score = 0.0
            if task.entangled_tasks:
                entangled_in_schedule = len(
                    task.entangled_tasks.intersection(set(schedule))
                )
                coherence_score += entangled_in_schedule * 0.1

            if task.interference_pattern:
                total_interference = sum(
                    abs(effect) for effect in task.interference_pattern.values()
                )
                coherence_score += total_interference * 0.05

            coherence_score += abs(task.probability_amplitude) ** 2 * 0.1
            metrics["quantum_coherence"] += coherence_score

        # Combine metrics into total cost (minimize)
        metrics["total_cost"] = (
            metrics["dependency_violations"] * 2.0
            + metrics["resource_efficiency"] * 0.5
            + abs(metrics["priority_score"]) * 0.3
            + metrics["execution_time"] * 0.001
            - metrics["quantum_coherence"]
            * 0.2  # Subtract because higher coherence is better
        )

        return metrics

    def _quantum_acceptance_probability(
        self, cost_metrics: Dict[str, float], temperature: float
    ) -> float:
        """Calculate quantum-inspired acceptance probability."""
        if temperature <= 0:
            return 0.0

        # Multi-objective acceptance with quantum interference
        base_prob = math.exp(-cost_metrics["total_cost"] / temperature)

        # Quantum coherence bonus
        coherence_bonus = min(0.2, cost_metrics["quantum_coherence"] * 0.1)

        # Resource efficiency factor
        efficiency_factor = 1.0 / (1.0 + cost_metrics["resource_efficiency"])

        return min(1.0, base_prob + coherence_bonus) * efficiency_factor

    def _update_pareto_front(
        self,
        pareto_front: List[Tuple[List[str], Dict[str, float]]],
        schedule: List[str],
        cost_metrics: Dict[str, float],
    ) -> None:
        """Update Pareto front with new solution."""
        # Check if solution is dominated by existing solutions
        dominated = False
        for existing_schedule, existing_metrics in pareto_front:
            if self._dominates(existing_metrics, cost_metrics):
                dominated = True
                break

        if not dominated:
            # Remove dominated solutions
            pareto_front[:] = [
                (sched, metrics)
                for sched, metrics in pareto_front
                if not self._dominates(cost_metrics, metrics)
            ]

            # Add new solution
            pareto_front.append((schedule.copy(), cost_metrics.copy()))

            # Limit Pareto front size
            if len(pareto_front) > 50:
                # Keep most diverse solutions
                pareto_front.sort(key=lambda x: x[1]["total_cost"])
                pareto_front = pareto_front[:50]

    def _dominates(
        self, metrics1: Dict[str, float], metrics2: Dict[str, float]
    ) -> bool:
        """Check if metrics1 dominates metrics2 (Pareto dominance)."""
        # For minimization objectives
        minimize_objectives = ["total_cost", "dependency_violations", "execution_time"]
        # For maximization objectives
        maximize_objectives = ["quantum_coherence"]

        better_in_any = False

        for obj in minimize_objectives:
            if metrics1[obj] > metrics2[obj]:
                return False
            elif metrics1[obj] < metrics2[obj]:
                better_in_any = True

        for obj in maximize_objectives:
            if metrics1[obj] < metrics2[obj]:
                return False
            elif metrics1[obj] > metrics2[obj]:
                better_in_any = True

        return better_in_any

    def _quantum_selection(
        self, population: List[List[str]], target_size: int
    ) -> List[List[str]]:
        """Select best solutions while maintaining diversity."""
        if len(population) <= target_size:
            return population

        # Calculate fitness for each solution
        fitness_scores = []
        for schedule in population:
            cost_metrics = self._calculate_multi_objective_cost(schedule)
            fitness = 1.0 / (1.0 + cost_metrics["total_cost"])
            fitness_scores.append((fitness, schedule))

        # Sort by fitness
        fitness_scores.sort(key=lambda x: x[0], reverse=True)

        # Select top performers and diverse solutions
        selected = []
        selected_schedules = set()

        # Always select best solution
        if fitness_scores:
            best_fitness, best_schedule = fitness_scores[0]
            selected.append(best_schedule)
            selected_schedules.add(tuple(best_schedule))

        # Select remaining based on diversity
        for fitness, schedule in fitness_scores[1:]:
            if len(selected) >= target_size:
                break

            schedule_tuple = tuple(schedule)
            if schedule_tuple not in selected_schedules:
                # Check diversity
                diverse_enough = True
                for existing_schedule in selected:
                    similarity = self._calculate_schedule_similarity(
                        schedule, existing_schedule
                    )
                    if similarity > 0.8:  # Too similar
                        diverse_enough = False
                        break

                if diverse_enough or len(selected) < target_size // 2:
                    selected.append(schedule)
                    selected_schedules.add(schedule_tuple)

        # Fill remaining slots with random diverse solutions
        while len(selected) < target_size and len(selected) < len(population):
            remaining_schedules = [
                s for f, s in fitness_scores if tuple(s) not in selected_schedules
            ]
            if remaining_schedules:
                random_schedule = random.choice(remaining_schedules)
                selected.append(random_schedule)
                selected_schedules.add(tuple(random_schedule))
            else:
                break

        return selected

    def _calculate_schedule_similarity(
        self, schedule1: List[str], schedule2: List[str]
    ) -> float:
        """Calculate similarity between two schedules."""
        if len(schedule1) != len(schedule2):
            return 0.0

        # Position-weighted similarity
        same_positions = 0
        for i, (task1, task2) in enumerate(zip(schedule1, schedule2)):
            if task1 == task2:
                same_positions += 1

        return same_positions / len(schedule1)

    def _apply_quantum_interference(
        self, population: List[List[str]]
    ) -> List[List[str]]:
        """Apply quantum interference patterns to population."""
        interfered_population = []

        for schedule in population:
            interfered_schedule = schedule.copy()

            # Apply constructive/destructive interference
            for i, task_id in enumerate(schedule):
                task = self.tasks[task_id]

                if task.interference_pattern:
                    total_interference = 0.0
                    interference_count = 0

                    for other_task_id, effect in task.interference_pattern.items():
                        if other_task_id in schedule:
                            other_pos = schedule.index(other_task_id)
                            distance = abs(i - other_pos)
                            # Interference decreases with distance
                            actual_effect = effect * math.exp(-distance / 5.0)
                            total_interference += actual_effect
                            interference_count += 1

                    # Strong interference triggers position adjustment
                    if interference_count > 0 and abs(total_interference) > 0.3:
                        # Find better position with less interference
                        best_pos = i
                        best_interference = abs(total_interference)

                        for new_pos in range(len(schedule)):
                            if new_pos == i:
                                continue

                            test_interference = 0.0
                            for (
                                other_task_id,
                                effect,
                            ) in task.interference_pattern.items():
                                if other_task_id in schedule:
                                    other_pos = schedule.index(other_task_id)
                                    distance = abs(new_pos - other_pos)
                                    actual_effect = effect * math.exp(-distance / 5.0)
                                    test_interference += abs(actual_effect)

                            if test_interference < best_interference:
                                best_interference = test_interference
                                best_pos = new_pos

                        # Apply interference-based position change
                        if best_pos != i:
                            interfered_schedule.pop(i)
                            interfered_schedule.insert(best_pos, task_id)

            interfered_population.append(interfered_schedule)

        return interfered_population

    def _calculate_population_diversity(self, population: List[List[str]]) -> float:
        """Calculate diversity metric for the population."""
        if len(population) <= 1:
            return 0.0

        total_similarity = 0.0
        comparisons = 0

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                similarity = self._calculate_schedule_similarity(
                    population[i], population[j]
                )
                total_similarity += similarity
                comparisons += 1

        avg_similarity = total_similarity / comparisons if comparisons > 0 else 1.0
        return 1.0 - avg_similarity  # Diversity is inverse of similarity

    def _check_convergence(
        self, convergence_history: List[Dict], min_iterations: int
    ) -> bool:
        """Check if algorithm has converged."""
        if len(convergence_history) < min_iterations:
            return False

        # Check if best cost hasn't improved in recent iterations
        recent_history = convergence_history[-20:]
        if len(recent_history) < 10:
            return False

        best_costs = [entry["best_cost"] for entry in recent_history]
        improvement = best_costs[0] - best_costs[-1]

        # Check cost improvement
        if improvement < 0.001:  # Minimal improvement threshold
            # Also check population diversity
            recent_diversities = [
                entry["population_diversity"] for entry in recent_history
            ]
            avg_diversity = sum(recent_diversities) / len(recent_diversities)

            # Converged if low improvement and low diversity
            return avg_diversity < 0.1

        return False

    def _calculate_schedule_cost(self, schedule: List[str]) -> float:
        """Calculate the cost of a given task schedule."""
        cost = 0.0
        current_time = datetime.now()
        resource_usage = {
            "cpu_cores": 0.0,
            "memory_gb": 0.0,
            "storage_gb": 0.0,
            "network_bandwidth": 0.0,
        }

        for task_id in schedule:
            task = self.tasks[task_id]

            # Dependency violation penalty
            for dep_id in task.dependencies:
                if dep_id in schedule and schedule.index(dep_id) > schedule.index(
                    task_id
                ):
                    cost += 1000.0  # Heavy penalty for dependency violations

            # Resource contention cost
            for resource, requirement in task.required_resources.items():
                if resource_usage.get(resource, 0.0) + requirement > getattr(
                    self.resource_pool, resource.replace("_", ""), 0.0
                ):
                    cost += 100.0 * requirement  # Penalty for resource conflicts
                else:
                    resource_usage[resource] = (
                        resource_usage.get(resource, 0.0) + requirement
                    )

            # Priority-based timing cost
            priority_factor = (
                11.0 - task.priority
            ) / 10.0  # Higher priority = lower cost
            time_cost = schedule.index(task_id) * priority_factor
            cost += time_cost

            # Interference effects
            for interfering_task_id, effect in task.interference_pattern.items():
                if interfering_task_id in schedule:
                    position_delta = abs(
                        schedule.index(task_id) - schedule.index(interfering_task_id)
                    )
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
                if hasattr(self, "monitor") and self.monitor:
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
                if hasattr(self, "monitor") and self.monitor:
                    self.monitor._record_error("dependency_check", error_msg, "warning")
                return False

            # Check resource availability with detailed reporting
            if not self.resource_pool.can_allocate(task.required_resources):
                resource_status = []
                for resource, required in task.required_resources.items():
                    available = getattr(self.resource_pool, resource, 0) - getattr(
                        self.resource_pool, f'allocated_{resource.split("_")[0]}', 0
                    )
                    resource_status.append(
                        f"{resource}: need {required}, have {available}"
                    )

                error_msg = f"Insufficient resources for task {task_id}: {'; '.join(resource_status)}"
                logger.warning(error_msg)
                if hasattr(self, "monitor") and self.monitor:
                    self.monitor._record_error(
                        "resource_allocation", error_msg, "warning"
                    )
                return False

            # Begin task execution
            try:
                # Collapse superposition and allocate resources
                task.collapse_state(TaskState.EXECUTING)
                task.started_at = datetime.now()

                # Record quantum event
                if hasattr(self, "monitor") and self.monitor:
                    self.monitor.record_quantum_event("superposition_collapse", task_id)

                if not self.resource_pool.allocate(task.required_resources):
                    raise RuntimeError(
                        "Resource allocation failed after availability check"
                    )

                logger.info(f"Executing task {task_id}: {task.name}")

                # Calculate execution parameters
                execution_time = task.estimated_duration.total_seconds()
                success_probability = task.get_execution_probability()

                # Apply interference effects
                if (
                    hasattr(self, "monitor")
                    and self.monitor
                    and task.interference_pattern
                ):
                    self.monitor.record_quantum_event(
                        "interference_applied",
                        task_id,
                        {
                            "interference_count": len(task.interference_pattern),
                            "total_effect": sum(task.interference_pattern.values()),
                        },
                    )

                # Simulate task execution with timeout protection
                simulation_time = min(
                    execution_time, 10.0
                )  # Cap simulation time to 10 seconds
                await asyncio.wait_for(
                    asyncio.sleep(simulation_time), timeout=15.0  # Safety timeout
                )

                # Determine success based on quantum probability with enhanced logic
                random_value = random.random()
                base_success = random_value < success_probability

                # Apply additional success factors
                priority_boost = min(
                    task.priority / 10.0 * 0.1, 0.1
                )  # Up to 10% boost for high priority
                execution_penalty = max(
                    0, (execution_time - 300) / 3600 * 0.1
                )  # Penalty for long tasks

                adjusted_probability = (
                    success_probability + priority_boost - execution_penalty
                )
                final_success = random_value < adjusted_probability

                execution_end_time = time.time()
                execution_duration_ms = (
                    execution_end_time - execution_start_time
                ) * 1000

                if final_success:
                    task.state = TaskState.COMPLETED
                    task.completed_at = datetime.now()
                    success = True
                    logger.info(
                        f"Task {task_id} completed successfully in {execution_duration_ms:.2f}ms"
                    )
                else:
                    task.state = TaskState.FAILED
                    task.error_message = f"Quantum execution probability threshold not met (p={success_probability:.3f}, rolled={random_value:.3f})"
                    success = False
                    logger.warning(
                        f"Task {task_id} failed during execution: {task.error_message}"
                    )

                # Record comprehensive execution metrics
                execution_record = {
                    "task_id": task_id,
                    "started_at": task.started_at.isoformat(),
                    "completed_at": (
                        task.completed_at.isoformat() if task.completed_at else None
                    ),
                    "success": success,
                    "execution_probability": success_probability,
                    "adjusted_probability": adjusted_probability,
                    "random_value": random_value,
                    "execution_duration_ms": execution_duration_ms,
                    "resource_requirements": task.required_resources.copy(),
                    "priority": task.priority,
                    "interference_effects": len(task.interference_pattern),
                    "dependencies_count": len(task.dependencies),
                }

                self.execution_history.append(execution_record)

                # Record monitoring metrics
                if hasattr(self, "monitor") and self.monitor:
                    self.monitor.record_task_execution(
                        task_id, execution_duration_ms, success, success_probability
                    )

                return success

            except asyncio.TimeoutError:
                error_msg = f"Task {task_id} execution timed out"
                task.state = TaskState.FAILED
                task.error_message = error_msg
                logger.error(error_msg)
                if hasattr(self, "monitor") and self.monitor:
                    self.monitor._record_error("task_execution", error_msg, "error")
                return False

            except Exception as execution_error:
                error_msg = f"Task {task_id} execution failed: {str(execution_error)}"
                task.state = TaskState.FAILED
                task.error_message = str(execution_error)
                logger.error(error_msg, exc_info=True)
                if hasattr(self, "monitor") and self.monitor:
                    self.monitor._record_error("task_execution", error_msg, "error")
                return False

        except Exception as outer_error:
            error_msg = f"Critical error executing task {task_id}: {str(outer_error)}"
            logger.error(error_msg, exc_info=True)
            if hasattr(self, "monitor") and self.monitor:
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
                logger.error(
                    f"Failed to deallocate resources for task {task_id}: {cleanup_error}"
                )
                if hasattr(self, "monitor") and self.monitor:
                    self.monitor._record_error(
                        "resource_cleanup", str(cleanup_error), "warning"
                    )

    async def execute_schedule(
        self, schedule: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute tasks according to the optimal schedule."""
        if not schedule:
            schedule = self.quantum_anneal_schedule()

        results = {
            "total_tasks": len(schedule),
            "successful_tasks": 0,
            "failed_tasks": 0,
            "start_time": datetime.now().isoformat(),
            "task_results": {},
        }

        logger.info(f"Starting execution of {len(schedule)} tasks")

        for task_id in schedule:
            success = await self.execute_task(task_id)
            results["task_results"][task_id] = success

            if success:
                results["successful_tasks"] += 1
            else:
                results["failed_tasks"] += 1

        results["end_time"] = datetime.now().isoformat()
        results["success_rate"] = results["successful_tasks"] / results["total_tasks"]

        logger.info(f"Execution complete. Success rate: {results['success_rate']:.2%}")
        return results

    def get_system_state(self) -> Dict[str, Any]:
        """Get current state of the quantum task planning system."""
        task_states = {}
        for task_id, task in self.tasks.items():
            task_states[task_id] = {
                "state": task.state.value,
                "probability": task.get_execution_probability(),
                "entangled_with": list(task.entangled_tasks),
                "dependencies": list(task.dependencies),
            }

        return {
            "total_tasks": len(self.tasks),
            "resource_utilization": {
                "cpu": self.resource_pool.allocated_cpu / self.resource_pool.cpu_cores,
                "memory": self.resource_pool.allocated_memory
                / self.resource_pool.memory_gb,
                "storage": self.resource_pool.allocated_storage
                / self.resource_pool.storage_gb,
                "bandwidth": self.resource_pool.allocated_bandwidth
                / self.resource_pool.network_bandwidth,
            },
            "task_states": task_states,
            "execution_history_size": len(self.execution_history),
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
            required_resources={"cpu_cores": 2.0, "memory_gb": 4.0},
        ),
        QuantumTask(
            id="cost_analysis",
            name="Analyze Costs",
            description="Analyze LLM usage costs",
            priority=7.0,
            estimated_duration=timedelta(minutes=2),
            required_resources={"cpu_cores": 1.0, "memory_gb": 2.0},
        ),
        QuantumTask(
            id="generate_report",
            name="Generate Report",
            description="Generate cost analysis report",
            priority=6.0,
            estimated_duration=timedelta(minutes=3),
            required_resources={"cpu_cores": 0.5, "memory_gb": 1.0},
        ),
        QuantumTask(
            id="database_backup",
            name="Database Backup",
            description="Backup LLM usage database",
            priority=8.0,
            estimated_duration=timedelta(minutes=10),
            required_resources={"storage_gb": 20.0, "network_bandwidth": 50.0},
        ),
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
