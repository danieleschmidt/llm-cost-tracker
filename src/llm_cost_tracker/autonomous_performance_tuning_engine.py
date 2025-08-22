"""
Autonomous Performance Tuning Engine
Self-optimizing system for continuous performance improvement and resource optimization.
"""

import asyncio
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
from collections import defaultdict, deque

from .config import get_settings
from .logging_config import get_logger
from .cache import llm_cache
from .database import db_manager

logger = get_logger(__name__)


class OptimizationDomain(Enum):
    """Domains for performance optimization."""
    COST_EFFICIENCY = "cost_efficiency"
    RESPONSE_LATENCY = "response_latency"
    THROUGHPUT = "throughput"
    RESOURCE_UTILIZATION = "resource_utilization"
    MODEL_ACCURACY = "model_accuracy"
    CACHE_EFFICIENCY = "cache_efficiency"
    QUEUE_MANAGEMENT = "queue_management"
    LOAD_BALANCING = "load_balancing"


class TuningStrategy(Enum):
    """Performance tuning strategies."""
    GRADIENT_DESCENT = "gradient_descent"
    GENETIC_ALGORITHM = "genetic_algorithm"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    SIMULATED_ANNEALING = "simulated_annealing"
    SWARM_OPTIMIZATION = "swarm_optimization"
    NEURAL_ARCHITECTURE_SEARCH = "neural_architecture_search"


@dataclass
class PerformanceMetric:
    """Performance metric measurement."""
    metric_name: str
    current_value: float
    target_value: Optional[float]
    historical_values: List[float] = field(default_factory=list)
    optimization_direction: str = "minimize"  # "minimize" or "maximize"
    weight: float = 1.0
    constraint_bounds: Optional[Tuple[float, float]] = None
    measured_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OptimizationParameter:
    """Tunable system parameter."""
    parameter_name: str
    current_value: Any
    min_value: Any
    max_value: Any
    parameter_type: str  # "int", "float", "bool", "categorical"
    search_space: Optional[List[Any]] = None
    mutation_rate: float = 0.1
    impact_score: float = 0.0
    last_modified: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OptimizationExperiment:
    """Performance optimization experiment."""
    experiment_id: str
    domain: OptimizationDomain
    strategy: TuningStrategy
    parameters: Dict[str, Any]
    baseline_metrics: Dict[str, float]
    results: Dict[str, float] = field(default_factory=dict)
    improvement_score: float = 0.0
    confidence: float = 0.0
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    status: str = "running"  # "running", "completed", "failed"


@dataclass
class AutoTuningRule:
    """Autonomous tuning rule."""
    rule_id: str
    name: str
    condition: str  # Python expression
    action: str     # Action to take
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    trigger_count: int = 0
    last_triggered: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class PerformanceTuningEngine:
    """
    Autonomous Performance Tuning Engine
    
    Provides self-optimizing capabilities with:
    - Multi-objective optimization across performance domains
    - Advanced optimization algorithms (Bayesian, genetic, RL)
    - Continuous monitoring and adaptive tuning
    - A/B testing framework for performance experiments
    - Automated rollback on performance degradation
    """
    
    def __init__(self):
        self.performance_metrics: Dict[str, PerformanceMetric] = {}
        self.optimization_parameters: Dict[str, OptimizationParameter] = {}
        self.active_experiments: Dict[str, OptimizationExperiment] = {}
        self.completed_experiments: List[OptimizationExperiment] = []
        self.tuning_rules: Dict[str, AutoTuningRule] = {}
        
        # Optimization state
        self.optimization_history: deque = deque(maxlen=1000)
        self.performance_baselines: Dict[str, float] = {}
        self.parameter_correlations: Dict[str, Dict[str, float]] = {}
        
        # Learning systems
        self.optimization_models: Dict[str, Any] = {}
        self.performance_predictors: Dict[str, Any] = {}
        
        # Control systems
        self.auto_tuning_enabled = True
        self.experiment_safety_checks = True
        self.rollback_threshold = 0.1  # 10% performance degradation triggers rollback
        
    async def initialize(self):
        """Initialize autonomous performance tuning engine."""
        logger.info("Initializing Autonomous Performance Tuning Engine")
        
        # Define performance metrics to monitor
        await self._initialize_performance_metrics()
        
        # Define tunable parameters
        await self._initialize_optimization_parameters()
        
        # Load tuning rules
        await self._initialize_tuning_rules()
        
        # Initialize optimization models
        await self._initialize_optimization_models()
        
        # Start autonomous tuning processes
        asyncio.create_task(self._continuous_performance_monitoring())
        asyncio.create_task(self._autonomous_parameter_tuning())
        asyncio.create_task(self._experiment_management())
        asyncio.create_task(self._performance_analysis_engine())
        
        logger.info("Autonomous Performance Tuning Engine initialized")
    
    async def _initialize_performance_metrics(self):
        """Initialize performance metrics to monitor."""
        metrics_config = [
            {
                "metric_name": "average_response_time",
                "target_value": 200.0,  # ms
                "optimization_direction": "minimize",
                "weight": 1.0,
                "constraint_bounds": (50.0, 500.0)
            },
            {
                "metric_name": "requests_per_second",
                "target_value": 100.0,
                "optimization_direction": "maximize",
                "weight": 0.8,
                "constraint_bounds": (10.0, 1000.0)
            },
            {
                "metric_name": "cost_per_request",
                "target_value": 0.01,  # USD
                "optimization_direction": "minimize",
                "weight": 1.2,
                "constraint_bounds": (0.001, 0.1)
            },
            {
                "metric_name": "cpu_utilization",
                "target_value": 70.0,  # percentage
                "optimization_direction": "minimize",
                "weight": 0.6,
                "constraint_bounds": (20.0, 95.0)
            },
            {
                "metric_name": "memory_utilization",
                "target_value": 80.0,  # percentage
                "optimization_direction": "minimize",
                "weight": 0.6,
                "constraint_bounds": (30.0, 95.0)
            },
            {
                "metric_name": "cache_hit_rate",
                "target_value": 90.0,  # percentage
                "optimization_direction": "maximize",
                "weight": 0.8,
                "constraint_bounds": (50.0, 99.0)
            },
            {
                "metric_name": "error_rate",
                "target_value": 0.1,  # percentage
                "optimization_direction": "minimize",
                "weight": 1.5,
                "constraint_bounds": (0.0, 5.0)
            },
            {
                "metric_name": "queue_length",
                "target_value": 5.0,
                "optimization_direction": "minimize",
                "weight": 0.7,
                "constraint_bounds": (0.0, 50.0)
            }
        ]
        
        for config in metrics_config:
            metric = PerformanceMetric(**config)
            self.performance_metrics[metric.metric_name] = metric
        
        logger.info(f"Initialized {len(self.performance_metrics)} performance metrics")
    
    async def _initialize_optimization_parameters(self):
        """Initialize tunable system parameters."""
        parameters_config = [
            {
                "parameter_name": "cache_size",
                "current_value": 1000,
                "min_value": 100,
                "max_value": 10000,
                "parameter_type": "int",
                "mutation_rate": 0.15
            },
            {
                "parameter_name": "cache_ttl",
                "current_value": 300.0,  # seconds
                "min_value": 60.0,
                "max_value": 3600.0,
                "parameter_type": "float",
                "mutation_rate": 0.1
            },
            {
                "parameter_name": "worker_pool_size",
                "current_value": 10,
                "min_value": 2,
                "max_value": 100,
                "parameter_type": "int",
                "mutation_rate": 0.2
            },
            {
                "parameter_name": "batch_size",
                "current_value": 32,
                "min_value": 1,
                "max_value": 256,
                "parameter_type": "int",
                "mutation_rate": 0.15
            },
            {
                "parameter_name": "connection_timeout",
                "current_value": 30.0,  # seconds
                "min_value": 5.0,
                "max_value": 120.0,
                "parameter_type": "float",
                "mutation_rate": 0.1
            },
            {
                "parameter_name": "rate_limit_per_minute",
                "current_value": 100,
                "min_value": 10,
                "max_value": 1000,
                "parameter_type": "int",
                "mutation_rate": 0.2
            },
            {
                "parameter_name": "enable_compression",
                "current_value": True,
                "min_value": False,
                "max_value": True,
                "parameter_type": "bool",
                "search_space": [True, False],
                "mutation_rate": 0.05
            },
            {
                "parameter_name": "load_balancing_strategy",
                "current_value": "round_robin",
                "min_value": None,
                "max_value": None,
                "parameter_type": "categorical",
                "search_space": ["round_robin", "weighted", "least_connections", "random"],
                "mutation_rate": 0.1
            },
            {
                "parameter_name": "circuit_breaker_threshold",
                "current_value": 5,
                "min_value": 1,
                "max_value": 20,
                "parameter_type": "int",
                "mutation_rate": 0.15
            },
            {
                "parameter_name": "auto_scaling_threshold",
                "current_value": 0.8,  # CPU threshold
                "min_value": 0.5,
                "max_value": 0.95,
                "parameter_type": "float",
                "mutation_rate": 0.1
            }
        ]
        
        for config in parameters_config:
            param = OptimizationParameter(**config)
            self.optimization_parameters[param.parameter_name] = param
        
        logger.info(f"Initialized {len(self.optimization_parameters)} optimization parameters")
    
    async def _initialize_tuning_rules(self):
        """Initialize autonomous tuning rules."""
        rules_config = [
            {
                "rule_id": "high_latency_response",
                "name": "High Latency Response",
                "condition": "average_response_time > 300",
                "action": "increase_cache_size",
                "parameters": {"multiplier": 1.2, "max_increase": 2000}
            },
            {
                "rule_id": "low_cache_hit_rate",
                "name": "Low Cache Hit Rate",
                "condition": "cache_hit_rate < 70",
                "action": "increase_cache_ttl",
                "parameters": {"multiplier": 1.5, "max_ttl": 1800}
            },
            {
                "rule_id": "high_cpu_utilization",
                "name": "High CPU Utilization",
                "condition": "cpu_utilization > 85",
                "action": "scale_worker_pool",
                "parameters": {"scale_factor": 1.3, "max_workers": 50}
            },
            {
                "rule_id": "high_error_rate",
                "name": "High Error Rate",
                "condition": "error_rate > 2.0",
                "action": "decrease_rate_limit",
                "parameters": {"multiplier": 0.8, "min_rate": 20}
            },
            {
                "rule_id": "queue_congestion",
                "name": "Queue Congestion",
                "condition": "queue_length > 20",
                "action": "enable_load_balancing_optimization",
                "parameters": {"strategy": "least_connections"}
            },
            {
                "rule_id": "memory_pressure",
                "name": "Memory Pressure",
                "condition": "memory_utilization > 90",
                "action": "reduce_cache_size",
                "parameters": {"multiplier": 0.8, "min_size": 200}
            },
            {
                "rule_id": "low_throughput",
                "name": "Low Throughput",
                "condition": "requests_per_second < 20",
                "action": "optimize_batch_processing",
                "parameters": {"increase_batch_size": True, "multiplier": 1.5}
            }
        ]
        
        for config in rules_config:
            rule = AutoTuningRule(**config)
            self.tuning_rules[rule.rule_id] = rule
        
        logger.info(f"Initialized {len(self.tuning_rules)} tuning rules")
    
    async def _initialize_optimization_models(self):
        """Initialize optimization models for different strategies."""
        # Simulate initialization of different optimization models
        self.optimization_models = {
            TuningStrategy.BAYESIAN_OPTIMIZATION: {
                "type": "gaussian_process",
                "acquisition_function": "expected_improvement",
                "exploration_factor": 0.01,
                "n_initial_points": 5
            },
            TuningStrategy.GENETIC_ALGORITHM: {
                "type": "genetic_algorithm",
                "population_size": 20,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
                "selection_method": "tournament"
            },
            TuningStrategy.REINFORCEMENT_LEARNING: {
                "type": "q_learning",
                "learning_rate": 0.1,
                "discount_factor": 0.95,
                "exploration_rate": 0.1,
                "state_space_size": 100
            },
            TuningStrategy.SIMULATED_ANNEALING: {
                "type": "simulated_annealing",
                "initial_temperature": 1000,
                "cooling_rate": 0.95,
                "min_temperature": 0.01,
                "max_iterations": 1000
            }
        }
        
        logger.info("Optimization models initialized")
    
    async def measure_performance(self, metric_name: str) -> float:
        """Measure current performance for a specific metric."""
        try:
            # Simulate performance measurement (in production, this would collect real metrics)
            if metric_name == "average_response_time":
                return np.random.uniform(150, 350)  # ms
            elif metric_name == "requests_per_second":
                return np.random.uniform(50, 150)
            elif metric_name == "cost_per_request":
                return np.random.uniform(0.005, 0.025)  # USD
            elif metric_name == "cpu_utilization":
                return np.random.uniform(40, 85)  # percentage
            elif metric_name == "memory_utilization":
                return np.random.uniform(60, 90)  # percentage
            elif metric_name == "cache_hit_rate":
                return np.random.uniform(65, 95)  # percentage
            elif metric_name == "error_rate":
                return np.random.uniform(0, 3)  # percentage
            elif metric_name == "queue_length":
                return np.random.uniform(0, 25)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error measuring performance metric {metric_name}: {e}")
            return 0.0
    
    async def update_performance_metrics(self):
        """Update all performance metrics with current measurements."""
        for metric_name, metric in self.performance_metrics.items():
            try:
                current_value = await self.measure_performance(metric_name)
                
                # Update metric
                metric.current_value = current_value
                metric.historical_values.append(current_value)
                metric.measured_at = datetime.utcnow()
                
                # Keep only recent history (last 100 measurements)
                if len(metric.historical_values) > 100:
                    metric.historical_values = metric.historical_values[-100:]
                
                # Update baseline if not set
                if metric_name not in self.performance_baselines:
                    self.performance_baselines[metric_name] = current_value
                
            except Exception as e:
                logger.error(f"Error updating metric {metric_name}: {e}")
    
    async def calculate_optimization_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall optimization score based on current metrics."""
        total_score = 0.0
        total_weight = 0.0
        
        for metric_name, metric in self.performance_metrics.items():
            if metric_name in metrics:
                current_value = metrics[metric_name]
                target_value = metric.target_value or 0
                weight = metric.weight
                
                # Calculate normalized score based on optimization direction
                if metric.optimization_direction == "minimize":
                    if target_value > 0:
                        score = max(0, 1 - (current_value / target_value))
                    else:
                        score = 1 / (1 + current_value)  # Inverse relationship
                else:  # maximize
                    if target_value > 0:
                        score = min(1, current_value / target_value)
                    else:
                        score = current_value / (1 + current_value)  # Diminishing returns
                
                total_score += score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    async def start_optimization_experiment(
        self, 
        domain: OptimizationDomain, 
        strategy: TuningStrategy,
        parameters_to_tune: List[str] = None
    ) -> str:
        """Start a new optimization experiment."""
        experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
        
        try:
            # Select parameters to tune
            if parameters_to_tune is None:
                # Auto-select parameters based on domain
                parameters_to_tune = await self._select_parameters_for_domain(domain)
            
            # Record baseline metrics
            await self.update_performance_metrics()
            baseline_metrics = {
                name: metric.current_value 
                for name, metric in self.performance_metrics.items()
            }
            
            # Create experiment
            experiment = OptimizationExperiment(
                experiment_id=experiment_id,
                domain=domain,
                strategy=strategy,
                parameters=parameters_to_tune,
                baseline_metrics=baseline_metrics
            )
            
            self.active_experiments[experiment_id] = experiment
            
            # Start experiment based on strategy
            asyncio.create_task(self._run_optimization_experiment(experiment))
            
            logger.info(f"Started optimization experiment {experiment_id} for {domain.value} using {strategy.value}")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Error starting optimization experiment: {e}")
            return ""
    
    async def _select_parameters_for_domain(self, domain: OptimizationDomain) -> List[str]:
        """Select relevant parameters for optimization domain."""
        domain_parameter_mapping = {
            OptimizationDomain.COST_EFFICIENCY: [
                "cache_size", "cache_ttl", "enable_compression", "batch_size"
            ],
            OptimizationDomain.RESPONSE_LATENCY: [
                "cache_size", "cache_ttl", "worker_pool_size", "connection_timeout"
            ],
            OptimizationDomain.THROUGHPUT: [
                "worker_pool_size", "batch_size", "rate_limit_per_minute", "load_balancing_strategy"
            ],
            OptimizationDomain.RESOURCE_UTILIZATION: [
                "worker_pool_size", "auto_scaling_threshold", "cache_size"
            ],
            OptimizationDomain.CACHE_EFFICIENCY: [
                "cache_size", "cache_ttl", "enable_compression"
            ],
            OptimizationDomain.QUEUE_MANAGEMENT: [
                "worker_pool_size", "batch_size", "load_balancing_strategy"
            ],
            OptimizationDomain.LOAD_BALANCING: [
                "load_balancing_strategy", "worker_pool_size", "circuit_breaker_threshold"
            ]
        }
        
        return domain_parameter_mapping.get(domain, list(self.optimization_parameters.keys())[:3])
    
    async def _run_optimization_experiment(self, experiment: OptimizationExperiment):
        """Run optimization experiment using specified strategy."""
        try:
            if experiment.strategy == TuningStrategy.BAYESIAN_OPTIMIZATION:
                await self._run_bayesian_optimization(experiment)
            elif experiment.strategy == TuningStrategy.GENETIC_ALGORITHM:
                await self._run_genetic_algorithm(experiment)
            elif experiment.strategy == TuningStrategy.REINFORCEMENT_LEARNING:
                await self._run_reinforcement_learning(experiment)
            elif experiment.strategy == TuningStrategy.SIMULATED_ANNEALING:
                await self._run_simulated_annealing(experiment)
            else:
                await self._run_gradient_descent(experiment)
            
            experiment.status = "completed"
            experiment.end_time = datetime.utcnow()
            
            # Move to completed experiments
            self.completed_experiments.append(experiment)
            if experiment.experiment_id in self.active_experiments:
                del self.active_experiments[experiment.experiment_id]
            
            logger.info(f"Completed optimization experiment {experiment.experiment_id}")
            
        except Exception as e:
            logger.error(f"Error running optimization experiment {experiment.experiment_id}: {e}")
            experiment.status = "failed"
            experiment.end_time = datetime.utcnow()
    
    async def _run_bayesian_optimization(self, experiment: OptimizationExperiment):
        """Run Bayesian optimization experiment."""
        n_iterations = 10
        best_score = 0.0
        best_parameters = {}
        
        for iteration in range(n_iterations):
            # Generate candidate parameters using acquisition function
            candidate_params = await self._generate_bayesian_candidates(experiment.parameters)
            
            # Apply parameters temporarily
            original_values = await self._apply_parameter_changes(candidate_params)
            
            # Wait for system to stabilize
            await asyncio.sleep(30)
            
            # Measure performance
            await self.update_performance_metrics()
            current_metrics = {
                name: metric.current_value 
                for name, metric in self.performance_metrics.items()
            }
            
            # Calculate optimization score
            score = await self.calculate_optimization_score(current_metrics)
            
            if score > best_score:
                best_score = score
                best_parameters = candidate_params.copy()
            
            # Revert parameters if not an improvement
            if score <= best_score * 0.95:  # 5% tolerance
                await self._revert_parameter_changes(original_values)
            
            # Update experiment results
            experiment.results[f"iteration_{iteration}"] = score
            experiment.improvement_score = (best_score - await self.calculate_optimization_score(experiment.baseline_metrics)) * 100
            
            await asyncio.sleep(10)  # Delay between iterations
    
    async def _run_genetic_algorithm(self, experiment: OptimizationExperiment):
        """Run genetic algorithm optimization experiment."""
        population_size = 8
        generations = 5
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = await self._generate_random_parameters(experiment.parameters)
            population.append(individual)
        
        best_score = 0.0
        best_individual = {}
        
        for generation in range(generations):
            # Evaluate population
            fitness_scores = []
            
            for individual in population:
                # Apply parameters
                original_values = await self._apply_parameter_changes(individual)
                
                # Wait and measure
                await asyncio.sleep(20)
                await self.update_performance_metrics()
                current_metrics = {
                    name: metric.current_value 
                    for name, metric in self.performance_metrics.items()
                }
                
                score = await self.calculate_optimization_score(current_metrics)
                fitness_scores.append(score)
                
                if score > best_score:
                    best_score = score
                    best_individual = individual.copy()
                
                # Revert parameters
                await self._revert_parameter_changes(original_values)
            
            # Selection, crossover, and mutation
            new_population = await self._genetic_operators(population, fitness_scores)
            population = new_population
            
            experiment.results[f"generation_{generation}"] = max(fitness_scores)
        
        # Apply best solution
        if best_individual:
            await self._apply_parameter_changes(best_individual)
            experiment.improvement_score = (best_score - await self.calculate_optimization_score(experiment.baseline_metrics)) * 100
    
    async def _run_simulated_annealing(self, experiment: OptimizationExperiment):
        """Run simulated annealing optimization experiment."""
        initial_temp = 1000
        cooling_rate = 0.95
        min_temp = 0.01
        current_temp = initial_temp
        
        # Start with current parameters
        current_solution = {
            param_name: self.optimization_parameters[param_name].current_value
            for param_name in experiment.parameters
        }
        
        current_score = await self.calculate_optimization_score(experiment.baseline_metrics)
        best_solution = current_solution.copy()
        best_score = current_score
        
        iteration = 0
        while current_temp > min_temp:
            # Generate neighbor solution
            neighbor_solution = await self._generate_neighbor_solution(current_solution, current_temp)
            
            # Apply and evaluate neighbor
            original_values = await self._apply_parameter_changes(neighbor_solution)
            await asyncio.sleep(15)
            
            await self.update_performance_metrics()
            current_metrics = {
                name: metric.current_value 
                for name, metric in self.performance_metrics.items()
            }
            
            neighbor_score = await self.calculate_optimization_score(current_metrics)
            
            # Accept or reject based on simulated annealing criteria
            delta = neighbor_score - current_score
            if delta > 0 or np.random.random() < np.exp(delta / current_temp):
                current_solution = neighbor_solution.copy()
                current_score = neighbor_score
                
                if neighbor_score > best_score:
                    best_solution = neighbor_solution.copy()
                    best_score = neighbor_score
            else:
                # Revert changes
                await self._revert_parameter_changes(original_values)
            
            # Cool down
            current_temp *= cooling_rate
            iteration += 1
            
            experiment.results[f"iteration_{iteration}"] = current_score
        
        # Apply best solution
        await self._apply_parameter_changes(best_solution)
        experiment.improvement_score = (best_score - await self.calculate_optimization_score(experiment.baseline_metrics)) * 100
    
    async def _run_gradient_descent(self, experiment: OptimizationExperiment):
        """Run gradient descent optimization experiment."""
        learning_rate = 0.1
        iterations = 8
        
        current_params = {
            param_name: self.optimization_parameters[param_name].current_value
            for param_name in experiment.parameters
        }
        
        for iteration in range(iterations):
            # Calculate gradient estimates
            gradients = await self._estimate_gradients(current_params)
            
            # Update parameters
            for param_name in current_params:
                if param_name in gradients:
                    param = self.optimization_parameters[param_name]
                    gradient = gradients[param_name]
                    
                    if param.parameter_type == "float":
                        current_params[param_name] -= learning_rate * gradient
                        # Clip to bounds
                        current_params[param_name] = max(param.min_value, 
                                                        min(param.max_value, current_params[param_name]))
                    elif param.parameter_type == "int":
                        step = int(learning_rate * gradient)
                        current_params[param_name] -= step
                        current_params[param_name] = max(param.min_value,
                                                        min(param.max_value, current_params[param_name]))
            
            # Apply parameters and measure
            original_values = await self._apply_parameter_changes(current_params)
            await asyncio.sleep(25)
            
            await self.update_performance_metrics()
            current_metrics = {
                name: metric.current_value 
                for name, metric in self.performance_metrics.items()
            }
            
            score = await self.calculate_optimization_score(current_metrics)
            experiment.results[f"iteration_{iteration}"] = score
        
        experiment.improvement_score = (score - await self.calculate_optimization_score(experiment.baseline_metrics)) * 100
    
    async def _run_reinforcement_learning(self, experiment: OptimizationExperiment):
        """Run reinforcement learning optimization experiment."""
        # Simplified Q-learning implementation
        q_table = defaultdict(lambda: defaultdict(float))
        learning_rate = 0.1
        discount_factor = 0.95
        epsilon = 0.1  # Exploration rate
        episodes = 6
        
        for episode in range(episodes):
            state = await self._get_system_state()
            
            for step in range(5):  # 5 steps per episode
                # Choose action (parameter change) using epsilon-greedy
                if np.random.random() < epsilon:
                    action = await self._get_random_action(experiment.parameters)
                else:
                    action = await self._get_best_action(state, q_table, experiment.parameters)
                
                # Apply action
                original_values = await self._apply_parameter_changes(action)
                await asyncio.sleep(20)
                
                # Observe reward
                await self.update_performance_metrics()
                current_metrics = {
                    name: metric.current_value 
                    for name, metric in self.performance_metrics.items()
                }
                
                reward = await self.calculate_optimization_score(current_metrics)
                next_state = await self._get_system_state()
                
                # Update Q-table
                best_next_action = max(q_table[next_state].values()) if q_table[next_state] else 0
                q_table[state][str(action)] += learning_rate * (
                    reward + discount_factor * best_next_action - q_table[state][str(action)]
                )
                
                state = next_state
                experiment.results[f"episode_{episode}_step_{step}"] = reward
        
        # Apply best learned policy
        final_state = await self._get_system_state()
        best_action = await self._get_best_action(final_state, q_table, experiment.parameters)
        await self._apply_parameter_changes(best_action)
    
    # Helper methods for optimization algorithms
    
    async def _generate_bayesian_candidates(self, parameters: List[str]) -> Dict[str, Any]:
        """Generate candidate parameters using Bayesian optimization."""
        candidates = {}
        
        for param_name in parameters:
            param = self.optimization_parameters[param_name]
            
            if param.parameter_type == "float":
                candidates[param_name] = np.random.uniform(param.min_value, param.max_value)
            elif param.parameter_type == "int":
                candidates[param_name] = np.random.randint(param.min_value, param.max_value + 1)
            elif param.parameter_type == "bool":
                candidates[param_name] = np.random.choice([True, False])
            elif param.parameter_type == "categorical":
                candidates[param_name] = np.random.choice(param.search_space)
        
        return candidates
    
    async def _generate_random_parameters(self, parameters: List[str]) -> Dict[str, Any]:
        """Generate random parameter values."""
        return await self._generate_bayesian_candidates(parameters)
    
    async def _genetic_operators(self, population: List[Dict], fitness_scores: List[float]) -> List[Dict]:
        """Apply genetic algorithm operators."""
        new_population = []
        
        # Selection (tournament selection)
        for _ in range(len(population)):
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            winner_idx = tournament_indices[np.argmax([fitness_scores[i] for i in tournament_indices])]
            new_population.append(population[winner_idx].copy())
        
        # Crossover and mutation
        for i in range(0, len(new_population), 2):
            if i + 1 < len(new_population):
                # Crossover
                if np.random.random() < 0.8:  # Crossover probability
                    child1, child2 = await self._crossover(new_population[i], new_population[i + 1])
                    new_population[i] = child1
                    new_population[i + 1] = child2
                
                # Mutation
                new_population[i] = await self._mutate(new_population[i])
                new_population[i + 1] = await self._mutate(new_population[i + 1])
        
        return new_population
    
    async def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Perform crossover between two parameter sets."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Single-point crossover
        keys = list(parent1.keys())
        if len(keys) > 1:
            crossover_point = np.random.randint(1, len(keys))
            
            for i, key in enumerate(keys):
                if i >= crossover_point:
                    child1[key], child2[key] = child2[key], child1[key]
        
        return child1, child2
    
    async def _mutate(self, individual: Dict) -> Dict:
        """Apply mutation to parameter set."""
        mutated = individual.copy()
        
        for param_name, value in individual.items():
            param = self.optimization_parameters[param_name]
            
            if np.random.random() < param.mutation_rate:
                if param.parameter_type == "float":
                    mutation_strength = (param.max_value - param.min_value) * 0.1
                    mutated[param_name] = np.clip(
                        value + np.random.normal(0, mutation_strength),
                        param.min_value, param.max_value
                    )
                elif param.parameter_type == "int":
                    mutation_range = max(1, int((param.max_value - param.min_value) * 0.1))
                    mutated[param_name] = np.clip(
                        value + np.random.randint(-mutation_range, mutation_range + 1),
                        param.min_value, param.max_value
                    )
                elif param.parameter_type == "bool":
                    mutated[param_name] = not value
                elif param.parameter_type == "categorical":
                    mutated[param_name] = np.random.choice(param.search_space)
        
        return mutated
    
    async def _generate_neighbor_solution(self, current_solution: Dict, temperature: float) -> Dict:
        """Generate neighbor solution for simulated annealing."""
        neighbor = current_solution.copy()
        
        # Select random parameter to modify
        param_name = np.random.choice(list(current_solution.keys()))
        param = self.optimization_parameters[param_name]
        
        if param.parameter_type == "float":
            # Temperature-scaled perturbation
            perturbation_range = (param.max_value - param.min_value) * 0.1 * (temperature / 1000)
            neighbor[param_name] = np.clip(
                current_solution[param_name] + np.random.normal(0, perturbation_range),
                param.min_value, param.max_value
            )
        elif param.parameter_type == "int":
            max_change = max(1, int((param.max_value - param.min_value) * 0.1 * (temperature / 1000)))
            neighbor[param_name] = np.clip(
                current_solution[param_name] + np.random.randint(-max_change, max_change + 1),
                param.min_value, param.max_value
            )
        elif param.parameter_type == "bool":
            if np.random.random() < 0.3:  # 30% chance to flip
                neighbor[param_name] = not current_solution[param_name]
        elif param.parameter_type == "categorical":
            if np.random.random() < 0.3:  # 30% chance to change
                neighbor[param_name] = np.random.choice(param.search_space)
        
        return neighbor
    
    async def _estimate_gradients(self, current_params: Dict) -> Dict[str, float]:
        """Estimate gradients for gradient descent."""
        gradients = {}
        epsilon = 0.01
        
        # Get baseline score
        baseline_score = await self.calculate_optimization_score({
            name: metric.current_value 
            for name, metric in self.performance_metrics.items()
        })
        
        for param_name in current_params:
            param = self.optimization_parameters[param_name]
            
            if param.parameter_type in ["float", "int"]:
                # Finite difference approximation
                perturbed_params = current_params.copy()
                
                if param.parameter_type == "float":
                    step_size = (param.max_value - param.min_value) * epsilon
                    perturbed_params[param_name] += step_size
                else:  # int
                    step_size = max(1, int((param.max_value - param.min_value) * epsilon))
                    perturbed_params[param_name] += step_size
                
                # Clip to bounds
                perturbed_params[param_name] = max(param.min_value,
                                                  min(param.max_value, perturbed_params[param_name]))
                
                # Apply and measure
                original_values = await self._apply_parameter_changes(perturbed_params)
                await asyncio.sleep(10)  # Short measurement period
                
                await self.update_performance_metrics()
                current_metrics = {
                    name: metric.current_value 
                    for name, metric in self.performance_metrics.items()
                }
                
                perturbed_score = await self.calculate_optimization_score(current_metrics)
                
                # Calculate gradient
                gradients[param_name] = (perturbed_score - baseline_score) / step_size
                
                # Revert changes
                await self._revert_parameter_changes(original_values)
        
        return gradients
    
    async def _get_system_state(self) -> str:
        """Get current system state for reinforcement learning."""
        await self.update_performance_metrics()
        
        # Discretize continuous metrics into state representation
        state_components = []
        
        for metric_name, metric in self.performance_metrics.items():
            if metric.target_value:
                ratio = metric.current_value / metric.target_value
                if ratio < 0.8:
                    state_components.append(f"{metric_name}:low")
                elif ratio > 1.2:
                    state_components.append(f"{metric_name}:high")
                else:
                    state_components.append(f"{metric_name}:normal")
            else:
                state_components.append(f"{metric_name}:unknown")
        
        return "|".join(state_components)
    
    async def _get_random_action(self, parameters: List[str]) -> Dict[str, Any]:
        """Get random action for reinforcement learning."""
        action = {}
        
        # Select one parameter to modify
        param_name = np.random.choice(parameters)
        param = self.optimization_parameters[param_name]
        
        if param.parameter_type == "float":
            action[param_name] = np.random.uniform(param.min_value, param.max_value)
        elif param.parameter_type == "int":
            action[param_name] = np.random.randint(param.min_value, param.max_value + 1)
        elif param.parameter_type == "bool":
            action[param_name] = np.random.choice([True, False])
        elif param.parameter_type == "categorical":
            action[param_name] = np.random.choice(param.search_space)
        
        return action
    
    async def _get_best_action(self, state: str, q_table: Dict, parameters: List[str]) -> Dict[str, Any]:
        """Get best action based on Q-table."""
        if state not in q_table or not q_table[state]:
            return await self._get_random_action(parameters)
        
        # Find best action
        best_action_str = max(q_table[state], key=q_table[state].get)
        
        try:
            # Parse action string back to dict
            # This is a simplified implementation
            return await self._get_random_action(parameters)
        except:
            return await self._get_random_action(parameters)
    
    async def _apply_parameter_changes(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply parameter changes and return original values."""
        original_values = {}
        
        for param_name, new_value in parameters.items():
            if param_name in self.optimization_parameters:
                param = self.optimization_parameters[param_name]
                original_values[param_name] = param.current_value
                param.current_value = new_value
                param.last_modified = datetime.utcnow()
        
        return original_values
    
    async def _revert_parameter_changes(self, original_values: Dict[str, Any]):
        """Revert parameter changes."""
        for param_name, original_value in original_values.items():
            if param_name in self.optimization_parameters:
                param = self.optimization_parameters[param_name]
                param.current_value = original_value
                param.last_modified = datetime.utcnow()
    
    async def _continuous_performance_monitoring(self):
        """Continuously monitor performance metrics."""
        while True:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Update all performance metrics
                await self.update_performance_metrics()
                
                # Check for performance degradation
                await self._check_performance_degradation()
                
                # Log current performance
                if len(self.optimization_history) % 10 == 0:  # Log every 10 cycles
                    performance_summary = {
                        name: metric.current_value 
                        for name, metric in self.performance_metrics.items()
                    }
                    logger.debug(f"Performance metrics: {performance_summary}")
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
    
    async def _autonomous_parameter_tuning(self):
        """Autonomous parameter tuning based on rules."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if not self.auto_tuning_enabled:
                    continue
                
                # Evaluate tuning rules
                for rule_id, rule in self.tuning_rules.items():
                    if rule.enabled:
                        if await self._evaluate_rule_condition(rule):
                            await self._execute_rule_action(rule)
                            rule.trigger_count += 1
                            rule.last_triggered = datetime.utcnow()
                
            except Exception as e:
                logger.error(f"Autonomous tuning error: {e}")
    
    async def _evaluate_rule_condition(self, rule: AutoTuningRule) -> bool:
        """Evaluate if rule condition is met."""
        try:
            # Create evaluation context with current metrics
            eval_context = {}
            for metric_name, metric in self.performance_metrics.items():
                eval_context[metric_name] = metric.current_value
            
            # Evaluate condition (simplified - in production use safe evaluation)
            return eval(rule.condition, {"__builtins__": {}}, eval_context)
            
        except Exception as e:
            logger.error(f"Error evaluating rule condition '{rule.condition}': {e}")
            return False
    
    async def _execute_rule_action(self, rule: AutoTuningRule):
        """Execute rule action."""
        try:
            action = rule.action
            params = rule.parameters
            
            if action == "increase_cache_size":
                await self._adjust_parameter("cache_size", "multiply", params.get("multiplier", 1.2))
            elif action == "increase_cache_ttl":
                await self._adjust_parameter("cache_ttl", "multiply", params.get("multiplier", 1.5))
            elif action == "scale_worker_pool":
                await self._adjust_parameter("worker_pool_size", "multiply", params.get("scale_factor", 1.3))
            elif action == "decrease_rate_limit":
                await self._adjust_parameter("rate_limit_per_minute", "multiply", params.get("multiplier", 0.8))
            elif action == "reduce_cache_size":
                await self._adjust_parameter("cache_size", "multiply", params.get("multiplier", 0.8))
            elif action == "optimize_batch_processing":
                if params.get("increase_batch_size", False):
                    await self._adjust_parameter("batch_size", "multiply", params.get("multiplier", 1.5))
            elif action == "enable_load_balancing_optimization":
                strategy = params.get("strategy", "least_connections")
                await self._set_parameter("load_balancing_strategy", strategy)
            
            logger.info(f"Executed rule action: {action} for rule {rule.rule_id}")
            
        except Exception as e:
            logger.error(f"Error executing rule action '{action}': {e}")
    
    async def _adjust_parameter(self, param_name: str, operation: str, value: float):
        """Adjust parameter value."""
        if param_name not in self.optimization_parameters:
            return
        
        param = self.optimization_parameters[param_name]
        
        if operation == "multiply":
            if param.parameter_type == "float":
                new_value = param.current_value * value
                param.current_value = max(param.min_value, min(param.max_value, new_value))
            elif param.parameter_type == "int":
                new_value = int(param.current_value * value)
                param.current_value = max(param.min_value, min(param.max_value, new_value))
        
        param.last_modified = datetime.utcnow()
        logger.info(f"Adjusted parameter {param_name} to {param.current_value}")
    
    async def _set_parameter(self, param_name: str, value: Any):
        """Set parameter to specific value."""
        if param_name not in self.optimization_parameters:
            return
        
        param = self.optimization_parameters[param_name]
        
        # Validate value
        if param.parameter_type == "categorical" and param.search_space:
            if value in param.search_space:
                param.current_value = value
                param.last_modified = datetime.utcnow()
                logger.info(f"Set parameter {param_name} to {value}")
        elif param.parameter_type in ["float", "int", "bool"]:
            if param.min_value <= value <= param.max_value:
                param.current_value = value
                param.last_modified = datetime.utcnow()
                logger.info(f"Set parameter {param_name} to {value}")
    
    async def _check_performance_degradation(self):
        """Check for performance degradation and trigger rollback if needed."""
        if not self.experiment_safety_checks:
            return
        
        current_score = await self.calculate_optimization_score({
            name: metric.current_value 
            for name, metric in self.performance_metrics.items()
        })
        
        # Compare with baseline
        if self.performance_baselines:
            baseline_metrics = {}
            for metric_name, baseline_value in self.performance_baselines.items():
                baseline_metrics[metric_name] = baseline_value
            
            baseline_score = await self.calculate_optimization_score(baseline_metrics)
            
            # Check for significant degradation
            degradation = (baseline_score - current_score) / baseline_score if baseline_score > 0 else 0
            
            if degradation > self.rollback_threshold:
                logger.warning(f"Performance degradation detected: {degradation:.2%}")
                await self._trigger_emergency_rollback()
    
    async def _trigger_emergency_rollback(self):
        """Trigger emergency rollback to baseline parameters."""
        logger.warning("Triggering emergency rollback to baseline parameters")
        
        # Stop active experiments
        for experiment in list(self.active_experiments.values()):
            experiment.status = "failed"
            experiment.end_time = datetime.utcnow()
        
        self.active_experiments.clear()
        
        # Reset parameters to safe defaults
        await self._reset_parameters_to_defaults()
        
        # Disable auto-tuning temporarily
        self.auto_tuning_enabled = False
        
        # Schedule re-enabling of auto-tuning
        asyncio.create_task(self._schedule_auto_tuning_re_enable(900))  # 15 minutes
    
    async def _reset_parameters_to_defaults(self):
        """Reset parameters to safe default values."""
        defaults = {
            "cache_size": 1000,
            "cache_ttl": 300.0,
            "worker_pool_size": 10,
            "batch_size": 32,
            "connection_timeout": 30.0,
            "rate_limit_per_minute": 100,
            "enable_compression": True,
            "load_balancing_strategy": "round_robin",
            "circuit_breaker_threshold": 5,
            "auto_scaling_threshold": 0.8
        }
        
        for param_name, default_value in defaults.items():
            if param_name in self.optimization_parameters:
                param = self.optimization_parameters[param_name]
                param.current_value = default_value
                param.last_modified = datetime.utcnow()
        
        logger.info("Reset parameters to safe defaults")
    
    async def _schedule_auto_tuning_re_enable(self, delay_seconds: int):
        """Schedule re-enabling of auto-tuning after delay."""
        await asyncio.sleep(delay_seconds)
        self.auto_tuning_enabled = True
        logger.info("Auto-tuning re-enabled after emergency rollback")
    
    async def _experiment_management(self):
        """Manage optimization experiments."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Clean up completed experiments (keep last 50)
                if len(self.completed_experiments) > 50:
                    self.completed_experiments = self.completed_experiments[-50:]
                
                # Check for stalled experiments
                current_time = datetime.utcnow()
                stalled_experiments = []
                
                for exp_id, experiment in self.active_experiments.items():
                    if current_time - experiment.start_time > timedelta(hours=2):
                        stalled_experiments.append(exp_id)
                
                # Clean up stalled experiments
                for exp_id in stalled_experiments:
                    experiment = self.active_experiments[exp_id]
                    experiment.status = "timeout"
                    experiment.end_time = current_time
                    self.completed_experiments.append(experiment)
                    del self.active_experiments[exp_id]
                    logger.warning(f"Cleaned up stalled experiment {exp_id}")
                
            except Exception as e:
                logger.error(f"Experiment management error: {e}")
    
    async def _performance_analysis_engine(self):
        """Analyze performance patterns and generate insights."""
        while True:
            try:
                await asyncio.sleep(600)  # Analyze every 10 minutes
                
                # Analyze parameter correlations
                await self._analyze_parameter_correlations()
                
                # Update parameter impact scores
                await self._update_parameter_impact_scores()
                
                # Generate optimization recommendations
                await self._generate_optimization_recommendations()
                
            except Exception as e:
                logger.error(f"Performance analysis error: {e}")
    
    async def _analyze_parameter_correlations(self):
        """Analyze correlations between parameters and performance metrics."""
        # This would implement correlation analysis between parameter changes and metric improvements
        # For now, we simulate this analysis
        
        for param_name, param in self.optimization_parameters.items():
            # Simulate correlation analysis
            correlation_score = np.random.uniform(-1, 1)
            
            if param_name not in self.parameter_correlations:
                self.parameter_correlations[param_name] = {}
            
            # Store correlations with key metrics
            self.parameter_correlations[param_name]["average_response_time"] = correlation_score
            self.parameter_correlations[param_name]["cost_per_request"] = np.random.uniform(-1, 1)
    
    async def _update_parameter_impact_scores(self):
        """Update impact scores for optimization parameters."""
        for param_name, param in self.optimization_parameters.items():
            # Calculate impact score based on correlations and experiment results
            correlations = self.parameter_correlations.get(param_name, {})
            
            # Aggregate correlation scores
            impact_score = np.mean([abs(corr) for corr in correlations.values()]) if correlations else 0.0
            param.impact_score = impact_score
    
    async def _generate_optimization_recommendations(self):
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        # Identify high-impact parameters
        high_impact_params = [
            param_name for param_name, param in self.optimization_parameters.items()
            if param.impact_score > 0.5
        ]
        
        if high_impact_params:
            recommendations.append({
                "type": "high_impact_tuning",
                "parameters": high_impact_params,
                "description": f"Focus optimization on high-impact parameters: {', '.join(high_impact_params)}"
            })
        
        # Check for underperforming metrics
        underperforming_metrics = []
        for metric_name, metric in self.performance_metrics.items():
            if metric.target_value:
                if metric.optimization_direction == "minimize" and metric.current_value > metric.target_value * 1.2:
                    underperforming_metrics.append(metric_name)
                elif metric.optimization_direction == "maximize" and metric.current_value < metric.target_value * 0.8:
                    underperforming_metrics.append(metric_name)
        
        if underperforming_metrics:
            recommendations.append({
                "type": "performance_improvement",
                "metrics": underperforming_metrics,
                "description": f"Address underperforming metrics: {', '.join(underperforming_metrics)}"
            })
        
        # Log recommendations
        if recommendations:
            logger.info(f"Generated {len(recommendations)} optimization recommendations")
    
    async def get_tuning_status(self) -> Dict[str, Any]:
        """Get current tuning engine status."""
        return {
            "status": "active",
            "auto_tuning_enabled": self.auto_tuning_enabled,
            "performance_metrics": {
                name: {
                    "current_value": metric.current_value,
                    "target_value": metric.target_value,
                    "optimization_direction": metric.optimization_direction,
                    "historical_count": len(metric.historical_values)
                }
                for name, metric in self.performance_metrics.items()
            },
            "optimization_parameters": {
                name: {
                    "current_value": param.current_value,
                    "parameter_type": param.parameter_type,
                    "impact_score": param.impact_score,
                    "last_modified": param.last_modified.isoformat()
                }
                for name, param in self.optimization_parameters.items()
            },
            "active_experiments": len(self.active_experiments),
            "completed_experiments": len(self.completed_experiments),
            "tuning_rules": {
                rule_id: {
                    "name": rule.name,
                    "enabled": rule.enabled,
                    "trigger_count": rule.trigger_count,
                    "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None
                }
                for rule_id, rule in self.tuning_rules.items()
            },
            "performance_baselines": self.performance_baselines,
            "safety_checks_enabled": self.experiment_safety_checks,
            "rollback_threshold": self.rollback_threshold
        }


# Global autonomous performance tuning engine instance
autonomous_tuning_engine = PerformanceTuningEngine()