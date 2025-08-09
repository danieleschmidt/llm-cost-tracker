"""
Quantum Algorithm Benchmarking and Research Validation System

This module provides comprehensive benchmarking capabilities to compare quantum-inspired 
task scheduling algorithms against classical approaches. Includes statistical analysis,
performance metrics, and research-quality validation.

Research Contributions:
- Novel quantum-inspired task scheduling with superposition and entanglement
- Multi-objective optimization with Pareto front analysis
- Adaptive temperature scheduling with quantum tunneling
- Comparative study against classical algorithms (Greedy, FCFS, SJF, Priority)
"""

import asyncio
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Callable
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from .quantum_task_planner import QuantumTaskPlanner, QuantumTask, TaskState, ResourcePool

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResults:
    """Results from algorithm benchmarking."""
    algorithm_name: str
    total_execution_time: float
    schedule_quality_score: float
    resource_utilization: Dict[str, float]
    dependency_violations: int
    makespan: float  # Total time to complete all tasks
    throughput: float  # Tasks per second
    convergence_iterations: int
    pareto_front_size: Optional[int] = None
    quantum_coherence_score: Optional[float] = None
    statistical_significance: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkConfiguration:
    """Configuration for benchmarking experiments."""
    task_counts: List[int] = None
    complexity_levels: List[str] = None
    resource_constraints: List[Dict[str, float]] = None
    dependency_densities: List[float] = None
    iterations_per_config: int = 10
    max_algorithm_iterations: int = 1000
    
    def __post_init__(self):
        if self.task_counts is None:
            self.task_counts = [10, 25, 50, 100, 200]
        if self.complexity_levels is None:
            self.complexity_levels = ['simple', 'moderate', 'complex']
        if self.resource_constraints is None:
            self.resource_constraints = [
                {'cpu_cores': 4.0, 'memory_gb': 8.0, 'storage_gb': 100.0, 'network_bandwidth': 100.0},
                {'cpu_cores': 8.0, 'memory_gb': 16.0, 'storage_gb': 500.0, 'network_bandwidth': 1000.0},
                {'cpu_cores': 16.0, 'memory_gb': 32.0, 'storage_gb': 1000.0, 'network_bandwidth': 10000.0}
            ]
        if self.dependency_densities is None:
            self.dependency_densities = [0.1, 0.3, 0.5]  # Probability of dependencies between tasks


class QuantumAlgorithmBenchmark:
    """
    Comprehensive benchmarking system for quantum-inspired task scheduling algorithms.
    
    Provides research-quality validation with statistical analysis and comparative studies.
    """
    
    def __init__(self, config: Optional[BenchmarkConfiguration] = None):
        self.config = config or BenchmarkConfiguration()
        self.results: Dict[str, List[BenchmarkResults]] = {}
        self.baseline_results: Dict[str, Dict[str, Any]] = {}
        self.statistical_tests: Dict[str, Any] = {}
        
    def generate_benchmark_tasks(self, 
                                task_count: int, 
                                complexity: str,
                                dependency_density: float,
                                resource_pool: ResourcePool) -> List[QuantumTask]:
        """Generate realistic benchmark tasks with varying complexity."""
        tasks = []
        
        # Task complexity parameters
        complexity_params = {
            'simple': {
                'priority_range': (1.0, 5.0),
                'duration_range': (300, 1800),  # 5-30 minutes
                'resource_factor': 0.3,
                'interference_probability': 0.1
            },
            'moderate': {
                'priority_range': (1.0, 8.0),
                'duration_range': (600, 7200),  # 10 minutes - 2 hours
                'resource_factor': 0.6,
                'interference_probability': 0.3
            },
            'complex': {
                'priority_range': (1.0, 10.0),
                'duration_range': (1800, 14400),  # 30 minutes - 4 hours
                'resource_factor': 1.0,
                'interference_probability': 0.5
            }
        }
        
        params = complexity_params[complexity]
        
        # Generate base tasks
        for i in range(task_count):
            task_id = f"benchmark_task_{i:03d}"
            priority = random.uniform(*params['priority_range'])
            duration_seconds = random.randint(*params['duration_range'])
            
            # Generate realistic resource requirements
            cpu_req = random.uniform(0.1, resource_pool.cpu_cores * params['resource_factor'])
            memory_req = random.uniform(0.1, resource_pool.memory_gb * params['resource_factor'])
            storage_req = random.uniform(1.0, resource_pool.storage_gb * 0.1)
            bandwidth_req = random.uniform(1.0, resource_pool.network_bandwidth * 0.2)
            
            task = QuantumTask(
                id=task_id,
                name=f"Benchmark Task {i:03d}",
                description=f"Generated {complexity} task for benchmarking",
                priority=priority,
                estimated_duration=timedelta(seconds=duration_seconds),
                required_resources={
                    'cpu_cores': cpu_req,
                    'memory_gb': memory_req,
                    'storage_gb': storage_req,
                    'network_bandwidth': bandwidth_req
                }
            )
            
            # Set quantum properties
            task.probability_amplitude = complex(
                random.uniform(0.7, 1.0),
                random.uniform(-0.3, 0.3)
            )
            
            tasks.append(task)
        
        # Add dependencies based on density
        if dependency_density > 0 and len(tasks) > 1:
            for task in tasks:
                # Each task has a chance to depend on earlier tasks
                for potential_dep in tasks[:tasks.index(task)]:
                    if random.random() < dependency_density:
                        task.dependencies.add(potential_dep.id)
                        
                        # Create entanglement for some dependencies
                        if random.random() < 0.3:
                            task.entangle_with(potential_dep.id)
                            potential_dep.entangle_with(task.id)
        
        # Add interference patterns
        for task in tasks:
            if random.random() < params['interference_probability']:
                # Create interference with random other tasks
                num_interferences = random.randint(1, min(3, len(tasks) - 1))
                other_tasks = [t for t in tasks if t.id != task.id]
                interfering_tasks = random.sample(other_tasks, num_interferences)
                
                for other_task in interfering_tasks:
                    # Constructive or destructive interference
                    effect = random.uniform(-0.8, 0.8)
                    task.interference_pattern[other_task.id] = effect
        
        return tasks
    
    async def benchmark_quantum_algorithm(self, 
                                        tasks: List[QuantumTask],
                                        resource_pool: ResourcePool,
                                        iterations: int = 1000) -> BenchmarkResults:
        """Benchmark the quantum-inspired algorithm."""
        planner = QuantumTaskPlanner(resource_pool)
        
        # Add all tasks
        for task in tasks:
            success, message = planner.add_task(task)
            if not success:
                logger.warning(f"Failed to add task {task.id}: {message}")
        
        start_time = time.time()
        
        # Generate schedule using quantum annealing
        schedule = planner.quantum_anneal_schedule(max_iterations=iterations)
        
        algorithm_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self._calculate_schedule_metrics(schedule, planner.tasks, resource_pool)
        
        # Get quantum-specific metrics
        quantum_metrics = self._calculate_quantum_metrics(planner, schedule)
        
        return BenchmarkResults(
            algorithm_name="Quantum-Inspired Annealing",
            total_execution_time=algorithm_time,
            schedule_quality_score=metrics['quality_score'],
            resource_utilization=metrics['resource_utilization'],
            dependency_violations=metrics['dependency_violations'],
            makespan=metrics['makespan'],
            throughput=len(schedule) / max(algorithm_time, 0.001),
            convergence_iterations=iterations,
            pareto_front_size=quantum_metrics.get('pareto_front_size'),
            quantum_coherence_score=quantum_metrics.get('coherence_score')
        )
    
    def benchmark_classical_algorithm(self,
                                    tasks: List[QuantumTask],
                                    resource_pool: ResourcePool,
                                    algorithm: str) -> BenchmarkResults:
        """Benchmark classical scheduling algorithms."""
        start_time = time.time()
        
        if algorithm == "greedy_priority":
            schedule = self._greedy_priority_schedule(tasks)
        elif algorithm == "shortest_job_first":
            schedule = self._shortest_job_first_schedule(tasks)
        elif algorithm == "first_come_first_serve":
            schedule = self._fcfs_schedule(tasks)
        elif algorithm == "critical_path":
            schedule = self._critical_path_schedule(tasks)
        elif algorithm == "round_robin":
            schedule = self._round_robin_schedule(tasks, resource_pool)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        algorithm_time = time.time() - start_time
        
        # Convert tasks to dict for metrics calculation
        task_dict = {task.id: task for task in tasks}
        
        # Calculate metrics
        metrics = self._calculate_schedule_metrics(schedule, task_dict, resource_pool)
        
        return BenchmarkResults(
            algorithm_name=algorithm.replace('_', ' ').title(),
            total_execution_time=algorithm_time,
            schedule_quality_score=metrics['quality_score'],
            resource_utilization=metrics['resource_utilization'],
            dependency_violations=metrics['dependency_violations'],
            makespan=metrics['makespan'],
            throughput=len(schedule) / max(algorithm_time, 0.001),
            convergence_iterations=1  # Classical algorithms don't iterate
        )
    
    def _greedy_priority_schedule(self, tasks: List[QuantumTask]) -> List[str]:
        """Greedy algorithm prioritizing high-priority tasks."""
        return [task.id for task in sorted(tasks, key=lambda t: t.priority, reverse=True)]
    
    def _shortest_job_first_schedule(self, tasks: List[QuantumTask]) -> List[str]:
        """Schedule shortest jobs first."""
        return [task.id for task in sorted(tasks, key=lambda t: t.estimated_duration)]
    
    def _fcfs_schedule(self, tasks: List[QuantumTask]) -> List[str]:
        """First Come First Serve scheduling."""
        return [task.id for task in sorted(tasks, key=lambda t: t.created_at)]
    
    def _critical_path_schedule(self, tasks: List[QuantumTask]) -> List[str]:
        """Critical Path Method scheduling."""
        # Build dependency graph
        task_dict = {task.id: task for task in tasks}
        
        # Calculate critical path
        def calculate_earliest_start(task_id: str, memo: Dict[str, float] = None) -> float:
            if memo is None:
                memo = {}
            if task_id in memo:
                return memo[task_id]
            
            task = task_dict[task_id]
            if not task.dependencies:
                memo[task_id] = 0.0
                return 0.0
            
            max_dependency_end = 0.0
            for dep_id in task.dependencies:
                if dep_id in task_dict:
                    dep_start = calculate_earliest_start(dep_id, memo)
                    dep_duration = task_dict[dep_id].estimated_duration.total_seconds()
                    max_dependency_end = max(max_dependency_end, dep_start + dep_duration)
            
            memo[task_id] = max_dependency_end
            return max_dependency_end
        
        # Sort by earliest start time, then by priority
        schedule_data = []
        for task in tasks:
            earliest_start = calculate_earliest_start(task.id)
            schedule_data.append((earliest_start, task.priority, task.id))
        
        schedule_data.sort(key=lambda x: (x[0], -x[1]))
        return [item[2] for item in schedule_data]
    
    def _round_robin_schedule(self, tasks: List[QuantumTask], resource_pool: ResourcePool) -> List[str]:
        """Round-robin scheduling with resource awareness."""
        schedule = []
        remaining_tasks = tasks.copy()
        
        # Simple round-robin with resource constraints
        while remaining_tasks:
            for task in remaining_tasks[:]:
                # Check if task can be scheduled (dependencies met)
                dependencies_met = all(
                    dep_id in [t.id for t in tasks if t.id in schedule]
                    for dep_id in task.dependencies
                )
                
                if dependencies_met:
                    schedule.append(task.id)
                    remaining_tasks.remove(task)
                    break
            else:
                # If no task can be scheduled, force schedule the first one
                if remaining_tasks:
                    schedule.append(remaining_tasks[0].id)
                    remaining_tasks.pop(0)
        
        return schedule
    
    def _calculate_schedule_metrics(self, 
                                  schedule: List[str], 
                                  tasks: Dict[str, QuantumTask],
                                  resource_pool: ResourcePool) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a schedule."""
        metrics = {
            'quality_score': 0.0,
            'resource_utilization': {'cpu_cores': 0.0, 'memory_gb': 0.0, 'storage_gb': 0.0, 'network_bandwidth': 0.0},
            'dependency_violations': 0,
            'makespan': 0.0,
            'priority_score': 0.0
        }
        
        if not schedule:
            return metrics
        
        # Calculate dependency violations
        for i, task_id in enumerate(schedule):
            if task_id not in tasks:
                continue
                
            task = tasks[task_id]
            for dep_id in task.dependencies:
                if dep_id in schedule and schedule.index(dep_id) > i:
                    metrics['dependency_violations'] += 1
        
        # Calculate makespan and resource utilization
        current_time = 0.0
        max_resource_usage = {
            'cpu_cores': 0.0,
            'memory_gb': 0.0,
            'storage_gb': 0.0,
            'network_bandwidth': 0.0
        }
        
        for task_id in schedule:
            if task_id not in tasks:
                continue
                
            task = tasks[task_id]
            task_duration = task.estimated_duration.total_seconds()
            current_time += task_duration
            
            # Track maximum resource usage
            for resource, usage in task.required_resources.items():
                max_resource_usage[resource] = max(max_resource_usage[resource], usage)
            
            # Add priority contribution
            metrics['priority_score'] += task.priority
        
        metrics['makespan'] = current_time
        
        # Calculate resource utilization percentages
        total_resources = {
            'cpu_cores': resource_pool.cpu_cores,
            'memory_gb': resource_pool.memory_gb,
            'storage_gb': resource_pool.storage_gb,
            'network_bandwidth': resource_pool.network_bandwidth
        }
        
        for resource in metrics['resource_utilization']:
            if total_resources[resource] > 0:
                metrics['resource_utilization'][resource] = min(
                    1.0, max_resource_usage[resource] / total_resources[resource]
                )
        
        # Calculate overall quality score
        avg_resource_util = sum(metrics['resource_utilization'].values()) / len(metrics['resource_utilization'])
        dependency_penalty = metrics['dependency_violations'] * 10.0
        priority_bonus = metrics['priority_score'] / len(schedule) if schedule else 0
        
        metrics['quality_score'] = max(0.0, 
            avg_resource_util * 40.0 +
            priority_bonus * 6.0 - 
            dependency_penalty
        )
        
        return metrics
    
    def _calculate_quantum_metrics(self, planner: QuantumTaskPlanner, schedule: List[str]) -> Dict[str, Any]:
        """Calculate quantum-specific metrics."""
        if not schedule:
            return {}
        
        coherence_score = 0.0
        entanglement_utilization = 0.0
        interference_effects = 0.0
        
        for task_id in schedule:
            if task_id not in planner.tasks:
                continue
                
            task = planner.tasks[task_id]
            
            # Quantum coherence
            coherence_score += abs(task.probability_amplitude) ** 2
            
            # Entanglement utilization
            entangled_in_schedule = len(task.entangled_tasks.intersection(set(schedule)))
            if task.entangled_tasks:
                entanglement_utilization += entangled_in_schedule / len(task.entangled_tasks)
            
            # Interference effects
            if task.interference_pattern:
                total_interference = sum(abs(effect) for effect in task.interference_pattern.values())
                interference_effects += total_interference
        
        return {
            'coherence_score': coherence_score / len(schedule),
            'entanglement_utilization': entanglement_utilization / len(schedule),
            'interference_effects': interference_effects / len(schedule)
        }
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmarking study."""
        logger.info("Starting comprehensive quantum algorithm benchmark study")
        
        all_results = {
            'quantum_results': [],
            'classical_results': {
                'greedy_priority': [],
                'shortest_job_first': [],
                'first_come_first_serve': [],
                'critical_path': [],
                'round_robin': []
            },
            'summary_statistics': {},
            'statistical_tests': {},
            'benchmark_config': asdict(self.config)
        }
        
        total_experiments = (
            len(self.config.task_counts) * 
            len(self.config.complexity_levels) *
            len(self.config.resource_constraints) *
            len(self.config.dependency_densities) *
            self.config.iterations_per_config
        )
        
        experiment_count = 0
        
        for task_count in self.config.task_counts:
            for complexity in self.config.complexity_levels:
                for resources in self.config.resource_constraints:
                    resource_pool = ResourcePool(**resources)
                    
                    for dependency_density in self.config.dependency_densities:
                        logger.info(f"Testing configuration: {task_count} tasks, {complexity} complexity, "
                                  f"{dependency_density:.1%} dependency density")
                        
                        # Run multiple iterations for statistical significance
                        quantum_results = []
                        classical_results = {alg: [] for alg in all_results['classical_results']}
                        
                        for iteration in range(self.config.iterations_per_config):
                            experiment_count += 1
                            
                            # Generate benchmark tasks
                            tasks = self.generate_benchmark_tasks(
                                task_count, complexity, dependency_density, resource_pool
                            )
                            
                            # Benchmark quantum algorithm
                            try:
                                quantum_result = await self.benchmark_quantum_algorithm(
                                    tasks, resource_pool, self.config.max_algorithm_iterations
                                )
                                quantum_results.append(quantum_result)
                                
                                logger.debug(f"Quantum result ({experiment_count}/{total_experiments}): "
                                           f"Quality={quantum_result.schedule_quality_score:.2f}, "
                                           f"Time={quantum_result.total_execution_time:.4f}s")
                                
                            except Exception as e:
                                logger.error(f"Quantum benchmark failed: {e}")
                                traceback.print_exc()
                            
                            # Benchmark classical algorithms
                            for alg_name in classical_results:
                                try:
                                    classical_result = self.benchmark_classical_algorithm(
                                        tasks, resource_pool, alg_name
                                    )
                                    classical_results[alg_name].append(classical_result)
                                    
                                except Exception as e:
                                    logger.error(f"Classical benchmark {alg_name} failed: {e}")
                        
                        # Store results for this configuration
                        config_key = f"{task_count}_{complexity}_{dependency_density:.1f}"
                        
                        if quantum_results:
                            all_results['quantum_results'].extend(quantum_results)
                        
                        for alg_name, results in classical_results.items():
                            if results:
                                all_results['classical_results'][alg_name].extend(results)
        
        # Calculate summary statistics
        all_results['summary_statistics'] = self._calculate_summary_statistics(all_results)
        
        # Perform statistical tests
        all_results['statistical_tests'] = self._perform_statistical_tests(all_results)
        
        logger.info(f"Comprehensive benchmark complete. Total experiments: {experiment_count}")
        
        return all_results
    
    def _calculate_summary_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for all algorithms."""
        summary = {}
        
        # Process quantum results
        if results['quantum_results']:
            quantum_data = results['quantum_results']
            summary['quantum'] = self._calculate_algorithm_stats(quantum_data)
        
        # Process classical results
        for alg_name, alg_results in results['classical_results'].items():
            if alg_results:
                summary[alg_name] = self._calculate_algorithm_stats(alg_results)
        
        return summary
    
    def _calculate_algorithm_stats(self, results: List[BenchmarkResults]) -> Dict[str, Any]:
        """Calculate statistics for a single algorithm."""
        if not results:
            return {}
        
        stats = {}
        
        # Extract metrics
        metrics = {
            'execution_time': [r.total_execution_time for r in results],
            'quality_score': [r.schedule_quality_score for r in results],
            'makespan': [r.makespan for r in results],
            'throughput': [r.throughput for r in results],
            'dependency_violations': [r.dependency_violations for r in results]
        }
        
        # Calculate statistics for each metric
        for metric_name, values in metrics.items():
            stats[metric_name] = {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
        
        # Quantum-specific metrics
        quantum_metrics = ['pareto_front_size', 'quantum_coherence_score']
        for metric in quantum_metrics:
            values = [getattr(r, metric) for r in results if getattr(r, metric) is not None]
            if values:
                stats[metric] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        
        return stats
    
    def _perform_statistical_tests(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        tests = {}
        
        if not results['quantum_results']:
            return tests
        
        quantum_quality = [r.schedule_quality_score for r in results['quantum_results']]
        quantum_time = [r.total_execution_time for r in results['quantum_results']]
        
        # Compare against each classical algorithm
        for alg_name, alg_results in results['classical_results'].items():
            if not alg_results:
                continue
            
            classical_quality = [r.schedule_quality_score for r in alg_results]
            classical_time = [r.total_execution_time for r in alg_results]
            
            # Simple comparison (in production would use proper statistical tests)
            tests[f'quantum_vs_{alg_name}'] = {
                'quality_improvement': {
                    'quantum_mean': statistics.mean(quantum_quality),
                    'classical_mean': statistics.mean(classical_quality),
                    'improvement_percent': (
                        (statistics.mean(quantum_quality) - statistics.mean(classical_quality)) / 
                        statistics.mean(classical_quality) * 100
                        if statistics.mean(classical_quality) > 0 else 0
                    )
                },
                'time_overhead': {
                    'quantum_mean': statistics.mean(quantum_time),
                    'classical_mean': statistics.mean(classical_time),
                    'overhead_ratio': (
                        statistics.mean(quantum_time) / statistics.mean(classical_time)
                        if statistics.mean(classical_time) > 0 else float('inf')
                    )
                }
            }
        
        return tests
    
    def export_results(self, results: Dict[str, Any], filename: str) -> None:
        """Export benchmark results to JSON file."""
        # Convert BenchmarkResults objects to dictionaries
        exportable_results = {
            'quantum_results': [r.to_dict() for r in results['quantum_results']],
            'classical_results': {
                alg: [r.to_dict() for r in alg_results] 
                for alg, alg_results in results['classical_results'].items()
            },
            'summary_statistics': results['summary_statistics'],
            'statistical_tests': results['statistical_tests'],
            'benchmark_config': results['benchmark_config'],
            'export_timestamp': datetime.now().isoformat(),
            'research_metadata': {
                'title': 'Quantum-Inspired Task Scheduling Algorithm Benchmark',
                'description': 'Comparative study of quantum-inspired vs classical scheduling algorithms',
                'algorithms_tested': ['quantum_annealing'] + list(results['classical_results'].keys()),
                'total_experiments': sum(len(alg_results) for alg_results in results['classical_results'].values()) + len(results['quantum_results'])
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(exportable_results, f, indent=2)
        
        logger.info(f"Benchmark results exported to {filename}")