"""
Test suite for Quantum-Variational Hybrid Optimization Engine

This module provides comprehensive unit and integration tests for the novel
quantum-variational hybrid optimization algorithm, ensuring correctness,
robustness, and performance characteristics.
"""

import asyncio
import numpy as np
import pytest
import math
import statistics
from unittest.mock import Mock, patch
from typing import List, Dict, Any

# Import the modules to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_cost_tracker.quantum_variational_hybrid import (
    QuantumState,
    OptimizationProblem,
    QuantumVariationalOptimizer,
    QuantumOptimizationBenchmark
)


class TestQuantumState:
    """Test suite for QuantumState class."""
    
    def test_quantum_state_initialization(self):
        """Test quantum state initialization and normalization."""
        amplitudes = np.array([0.6, 0.8])
        phases = np.array([0.0, np.pi/2])
        
        state = QuantumState(amplitudes=amplitudes, phases=phases)
        
        # Check normalization
        assert abs(np.linalg.norm(state.amplitudes) - 1.0) < 1e-10
        assert len(state.amplitudes) == len(state.phases)
        assert state.measurement_count == 0
    
    def test_quantum_state_measurement(self):
        """Test quantum measurement with Born rule."""
        # Create state with known probabilities
        amplitudes = np.array([0.6, 0.8])  # Will be normalized
        phases = np.array([0.0, 0.0])
        
        state = QuantumState(amplitudes=amplitudes, phases=phases)
        
        # Perform multiple measurements
        measurements = [state.measure() for _ in range(1000)]
        
        # Check measurement outcomes are valid
        assert all(0 <= m < len(amplitudes) for m in measurements)
        assert state.measurement_count == 1000
        
        # Check probability distribution (should be approximately correct)
        prob_0 = measurements.count(0) / 1000
        expected_prob_0 = (0.6 / math.sqrt(0.6**2 + 0.8**2))**2
        
        # Allow for statistical fluctuation
        assert abs(prob_0 - expected_prob_0) < 0.1
    
    def test_quantum_rotation(self):
        """Test quantum rotation operations."""
        amplitudes = np.array([1.0, 0.0])
        phases = np.array([0.0, 0.0])
        
        state = QuantumState(amplitudes=amplitudes, phases=phases)
        original_amp_0 = state.amplitudes[0]
        
        # Apply rotation
        state.apply_rotation(qubit_idx=0, theta=np.pi/4, phi=0.0)
        
        # Check that state has changed
        assert state.amplitudes[0] != original_amp_0
        
        # Check normalization is preserved
        assert abs(np.linalg.norm(state.amplitudes) - 1.0) < 1e-10
    
    def test_expectation_value(self):
        """Test expectation value calculation."""
        amplitudes = np.array([1.0, 0.0])
        phases = np.array([0.0, 0.0])
        
        state = QuantumState(amplitudes=amplitudes, phases=phases)
        
        # Simple diagonal observable (Pauli-Z like)
        observable = np.array([[1.0, 0.0], [0.0, -1.0]])
        
        expectation = state.get_expectation_value(observable)
        
        # For |0⟩ state, should get +1
        assert abs(expectation - 1.0) < 1e-10
    
    def test_quantum_state_validation(self):
        """Test quantum state validation and error handling."""
        # Mismatched amplitudes and phases
        with pytest.raises(ValueError):
            QuantumState(
                amplitudes=np.array([1.0, 0.0]),
                phases=np.array([0.0])
            )


class TestOptimizationProblem:
    """Test suite for OptimizationProblem class."""
    
    def test_optimization_problem_creation(self):
        """Test optimization problem creation and validation."""
        def simple_function(x):
            return sum(xi**2 for xi in x)
        
        problem = OptimizationProblem(
            cost_function=simple_function,
            dimension=3,
            bounds=[(-5.0, 5.0)] * 3,
            target_precision=1e-6
        )
        
        assert problem.dimension == 3
        assert len(problem.bounds) == 3
        assert problem.target_precision == 1e-6
        assert callable(problem.cost_function)
    
    def test_optimization_problem_validation(self):
        """Test optimization problem validation."""
        def simple_function(x):
            return sum(xi**2 for xi in x)
        
        # Mismatched dimensions and bounds
        with pytest.raises(ValueError):
            OptimizationProblem(
                cost_function=simple_function,
                dimension=3,
                bounds=[(-5.0, 5.0)] * 2  # Wrong number of bounds
            )
    
    def test_cost_function_evaluation(self):
        """Test cost function evaluation."""
        def quadratic_function(x):
            return sum(xi**2 for xi in x)
        
        problem = OptimizationProblem(
            cost_function=quadratic_function,
            dimension=2,
            bounds=[(-10.0, 10.0)] * 2
        )
        
        # Test evaluation
        test_point = [3.0, 4.0]
        result = problem.cost_function(test_point)
        expected = 3.0**2 + 4.0**2
        
        assert abs(result - expected) < 1e-10


class TestQuantumVariationalOptimizer:
    """Test suite for QuantumVariationalOptimizer class."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = QuantumVariationalOptimizer(
            num_qubits=4,
            max_iterations=100,
            learning_rate=0.1,
            momentum=0.9,
            temperature_schedule='exponential'
        )
        
        assert optimizer.num_qubits == 4
        assert optimizer.max_iterations == 100
        assert optimizer.learning_rate == 0.1
        assert optimizer.momentum == 0.9
        assert optimizer.temperature_schedule == 'exponential'
        assert optimizer.quantum_state is not None
        assert len(optimizer.optimization_history) == 0
    
    def test_temperature_computation(self):
        """Test temperature schedule computations."""
        optimizer = QuantumVariationalOptimizer(
            num_qubits=4,
            temperature_schedule='linear'
        )
        
        # Test linear schedule
        temp_start = optimizer._compute_temperature(0)
        temp_mid = optimizer._compute_temperature(optimizer.max_iterations // 2)
        temp_end = optimizer._compute_temperature(optimizer.max_iterations - 1)
        
        assert temp_start > temp_mid > temp_end >= 0
        
        # Test exponential schedule
        optimizer.temperature_schedule = 'exponential'
        temp_exp_start = optimizer._compute_temperature(0)
        temp_exp_end = optimizer._compute_temperature(optimizer.max_iterations - 1)
        
        assert temp_exp_start > temp_exp_end > 0
    
    def test_quantum_gradient_computation(self):
        """Test quantum gradient computation using parameter shift rule."""
        optimizer = QuantumVariationalOptimizer(num_qubits=4)
        
        def simple_cost_function(params):
            return sum(p**2 for p in params)
        
        problem = OptimizationProblem(
            cost_function=simple_cost_function,
            dimension=3,
            bounds=[(-5.0, 5.0)] * 3
        )
        
        params = np.array([1.0, 2.0, 3.0])
        gradient = optimizer._compute_quantum_gradient(problem, params)
        
        # Check gradient shape and basic properties
        assert len(gradient) == len(params)
        assert all(isinstance(g, (int, float, np.number)) for g in gradient)
        
        # For quadratic function, gradient should be proportional to params
        # (allowing for some numerical error from parameter shift rule)
        expected_gradient = 2 * params
        relative_error = np.abs((gradient - expected_gradient) / expected_gradient)
        assert np.all(relative_error < 0.5)  # Allow some error due to shift rule
    
    def test_evolutionary_mutation(self):
        """Test evolutionary mutation mechanism."""
        optimizer = QuantumVariationalOptimizer(
            num_qubits=4,
            use_adaptive_mutation=True
        )
        
        def simple_cost_function(x):
            return sum(xi**2 for xi in x)
        
        problem = OptimizationProblem(
            cost_function=simple_cost_function,
            dimension=3,
            bounds=[(-5.0, 5.0)] * 3
        )
        
        original_params = np.array([1.0, 2.0, 3.0])
        
        # Test mutation (may or may not change parameters depending on randomness)
        mutated_params = optimizer._evolutionary_mutation(original_params, problem)
        
        # Check bounds are respected
        for i, (lower, upper) in enumerate(problem.bounds):
            assert lower <= mutated_params[i] <= upper
        
        # Check dimensions are preserved
        assert len(mutated_params) == len(original_params)
    
    @pytest.mark.asyncio
    async def test_simple_optimization(self):
        """Test optimization on simple quadratic function."""
        optimizer = QuantumVariationalOptimizer(
            num_qubits=4,
            max_iterations=50,  # Keep small for tests
            learning_rate=0.1,
            convergence_threshold=1e-3
        )
        
        def quadratic_function(x):
            return sum(xi**2 for xi in x)
        
        problem = OptimizationProblem(
            cost_function=quadratic_function,
            dimension=2,
            bounds=[(-5.0, 5.0)] * 2,
            target_precision=1e-3
        )
        
        result = await optimizer.optimize(problem)
        
        # Check result structure
        assert 'success' in result
        assert 'best_solution' in result
        assert 'best_cost' in result
        assert 'optimization_time_seconds' in result
        
        if result['success']:
            # Check solution quality (should be close to [0, 0])
            solution = result['best_solution']
            assert len(solution) == 2
            assert result['best_cost'] < 1.0  # Should find reasonably good solution
        
        # Check optimization history is recorded
        assert len(optimizer.optimization_history) > 0
    
    @pytest.mark.asyncio
    async def test_optimization_with_constraints(self):
        """Test optimization with constraint handling."""
        optimizer = QuantumVariationalOptimizer(num_qubits=4, max_iterations=30)
        
        def constrained_function(x):
            # Simple quadratic with penalty for constraint violation
            base_cost = sum(xi**2 for xi in x)
            
            # Add penalty if sum of variables > 1
            constraint_violation = max(0, sum(x) - 1.0)
            penalty = 1000 * constraint_violation**2
            
            return base_cost + penalty
        
        problem = OptimizationProblem(
            cost_function=constrained_function,
            dimension=2,
            bounds=[(-2.0, 2.0)] * 2
        )
        
        result = await optimizer.optimize(problem)
        
        if result['success']:
            # Check that constraint is approximately satisfied
            solution_sum = sum(result['best_solution'])
            assert solution_sum <= 1.1  # Allow some tolerance
    
    def test_performance_statistics(self):
        """Test performance statistics tracking."""
        optimizer = QuantumVariationalOptimizer(num_qubits=4)
        
        # Simulate some optimization runs
        optimizer.convergence_statistics['final_costs'] = [0.1, 0.2, 0.05, 0.15]
        optimizer.convergence_statistics['quantum_advantage_ratio'] = [0.1, 0.2, 0.3, 0.05]
        optimizer.convergence_statistics['iterations_to_convergence'] = [50, 75, 30, 60]
        
        summary = optimizer.get_performance_summary()
        
        assert 'total_optimization_runs' in summary
        assert 'best_cost_achieved' in summary
        assert 'mean_cost' in summary
        assert 'mean_quantum_advantage' in summary
        
        assert summary['total_optimization_runs'] == 4
        assert summary['best_cost_achieved'] == 0.05
        assert abs(summary['mean_cost'] - 0.125) < 1e-10
        assert abs(summary['mean_quantum_advantage'] - 0.1625) < 1e-10


class TestQuantumOptimizationBenchmark:
    """Test suite for QuantumOptimizationBenchmark class."""
    
    def test_benchmark_initialization(self):
        """Test benchmark suite initialization."""
        benchmark = QuantumOptimizationBenchmark()
        
        assert len(benchmark.benchmark_problems) > 0
        assert isinstance(benchmark.results_database, list)
        
        # Check that problems are properly constructed
        for problem in benchmark.benchmark_problems:
            assert isinstance(problem, OptimizationProblem)
            assert problem.dimension > 0
            assert len(problem.bounds) == problem.dimension
    
    def test_benchmark_problems(self):
        """Test individual benchmark problems."""
        benchmark = QuantumOptimizationBenchmark()
        
        for problem in benchmark.benchmark_problems:
            # Test that cost function is callable and returns numeric value
            test_point = [0.5 * (lower + upper) for lower, upper in problem.bounds]
            cost = problem.cost_function(test_point)
            
            assert isinstance(cost, (int, float, np.number))
            assert not math.isnan(cost)
            assert not math.isinf(cost)
    
    @pytest.mark.asyncio
    async def test_small_benchmark_run(self):
        """Test benchmark execution with minimal configuration."""
        benchmark = QuantumOptimizationBenchmark()
        
        # Use only first problem for quick test
        test_problems = benchmark.benchmark_problems[:1]
        
        # Simple optimizer config for testing
        optimizer_configs = [
            {'learning_rate': 0.1, 'momentum': 0.9, 'temperature_schedule': 'linear'}
        ]
        
        # Run with minimal repetitions
        result = await benchmark.run_comprehensive_benchmark(
            num_runs_per_problem=2,  # Very small for testing
            optimizer_configs=optimizer_configs
        )
        
        # Check result structure
        assert 'benchmark_metadata' in result
        assert 'problem_results' in result
        assert 'statistical_summary' in result
        
        # Check metadata
        metadata = result['benchmark_metadata']
        assert metadata['num_problems'] == 1
        assert metadata['num_runs_per_problem'] == 2
        assert 'start_time' in metadata
        assert 'total_time_seconds' in metadata


class TestIntegrationScenarios:
    """Integration tests for complete optimization workflows."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_optimization_workflow(self):
        """Test complete optimization workflow from problem setup to results."""
        
        # Define a realistic optimization problem
        def portfolio_optimization(weights):
            """Simplified portfolio optimization example."""
            # Expected returns (simulated)
            returns = np.array([0.1, 0.15, 0.12, 0.08])
            
            # Risk covariance matrix (simulated)
            cov_matrix = np.array([
                [0.1, 0.02, 0.01, 0.005],
                [0.02, 0.15, 0.03, 0.01],
                [0.01, 0.03, 0.12, 0.02],
                [0.005, 0.01, 0.02, 0.08]
            ])
            
            weights = np.array(weights)
            
            # Ensure weights are valid (sum to 1, non-negative)
            weights = np.clip(weights, 0, 1)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(len(weights)) / len(weights)
            
            # Portfolio return
            portfolio_return = np.dot(weights, returns)
            
            # Portfolio risk
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            # Risk-adjusted return (maximize Sharpe ratio, minimize negative)
            risk_free_rate = 0.02
            if portfolio_risk > 1e-6:
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
                return -sharpe_ratio  # Minimize negative Sharpe ratio
            else:
                return 1000  # High penalty for zero risk (invalid)
        
        # Create optimization problem
        problem = OptimizationProblem(
            cost_function=portfolio_optimization,
            dimension=4,
            bounds=[(0.0, 1.0)] * 4,  # Portfolio weights between 0 and 1
            target_precision=1e-4
        )
        
        # Create and configure optimizer
        optimizer = QuantumVariationalOptimizer(
            num_qubits=6,
            max_iterations=100,
            learning_rate=0.05,
            momentum=0.9,
            temperature_schedule='adaptive',
            use_adaptive_mutation=True
        )
        
        # Run optimization
        result = await optimizer.optimize(problem)
        
        # Validate results
        assert result['success'] == True
        assert 'best_solution' in result
        assert 'best_cost' in result
        assert 'quantum_advantage_ratio' in result
        
        # Check solution validity
        solution = result['best_solution']
        assert len(solution) == 4
        assert all(0 <= w <= 1 for w in solution)  # Valid weight bounds
        
        # Check that we found a reasonable solution
        assert result['best_cost'] < 0  # Should be negative (good Sharpe ratio)
        
        # Verify portfolio weights approximately sum to 1 after normalization
        normalized_weights = np.array(solution) / np.sum(solution) if np.sum(solution) > 0 else solution
        assert abs(np.sum(normalized_weights) - 1.0) < 1e-2
    
    @pytest.mark.asyncio  
    async def test_algorithm_comparison_workflow(self):
        """Test comparison between different algorithm configurations."""
        
        # Simple test function
        def rosenbrock_2d(x):
            return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
        
        problem = OptimizationProblem(
            cost_function=rosenbrock_2d,
            dimension=2,
            bounds=[(-2.0, 2.0)] * 2,
            target_precision=1e-4
        )
        
        # Test two different configurations
        config1 = {
            'num_qubits': 4,
            'max_iterations': 50,
            'learning_rate': 0.1,
            'temperature_schedule': 'exponential'
        }
        
        config2 = {
            'num_qubits': 4,
            'max_iterations': 50,
            'learning_rate': 0.05,
            'temperature_schedule': 'adaptive'
        }
        
        # Run both configurations
        optimizer1 = QuantumVariationalOptimizer(**config1)
        optimizer2 = QuantumVariationalOptimizer(**config2)
        
        result1 = await optimizer1.optimize(problem)
        result2 = await optimizer2.optimize(problem)
        
        # Both should complete successfully
        assert result1['success'] == True
        assert result2['success'] == True
        
        # Compare results
        print(f"Config 1 best cost: {result1['best_cost']:.6f}")
        print(f"Config 2 best cost: {result2['best_cost']:.6f}")
        
        # Both should find reasonable solutions (Rosenbrock optimum is at [1,1] with cost 0)
        assert result1['best_cost'] < 10.0
        assert result2['best_cost'] < 10.0
    
    def test_error_handling_and_robustness(self):
        """Test error handling and robustness scenarios."""
        
        # Test with invalid cost function
        def failing_cost_function(x):
            if x[0] > 0:
                raise ValueError("Simulated failure")
            return sum(xi**2 for xi in x)
        
        problem = OptimizationProblem(
            cost_function=failing_cost_function,
            dimension=2,
            bounds=[(-1.0, 1.0)] * 2
        )
        
        optimizer = QuantumVariationalOptimizer(
            num_qubits=4,
            max_iterations=10
        )
        
        # This should handle the error gracefully
        # (In a real implementation, you'd want robust error handling)
        with pytest.raises((ValueError, Exception)):
            # Note: The actual implementation should handle this more gracefully
            result = asyncio.run(optimizer.optimize(problem))
    
    @pytest.mark.performance
    def test_performance_benchmarks(self):
        """Performance benchmarks for optimization algorithms."""
        
        # This test would typically be marked to run separately
        # and would benchmark actual performance characteristics
        
        def sphere_function(x):
            return sum(xi**2 for xi in x)
        
        problem = OptimizationProblem(
            cost_function=sphere_function,
            dimension=10,
            bounds=[(-10.0, 10.0)] * 10
        )
        
        optimizer = QuantumVariationalOptimizer(
            num_qubits=8,
            max_iterations=500
        )
        
        import time
        start_time = time.time()
        
        # Run optimization (synchronously for timing)
        result = asyncio.run(optimizer.optimize(problem))
        
        end_time = time.time()
        optimization_time = end_time - start_time
        
        print(f"Optimization completed in {optimization_time:.2f} seconds")
        print(f"Best cost achieved: {result['best_cost']:.8f}")
        
        # Performance assertions
        assert optimization_time < 30.0  # Should complete within 30 seconds
        if result['success']:
            assert result['best_cost'] < 1.0  # Should find good solution


# Test fixtures and utilities

@pytest.fixture
def simple_quadratic_problem():
    """Fixture for simple quadratic optimization problem."""
    def cost_function(x):
        return sum(xi**2 for xi in x)
    
    return OptimizationProblem(
        cost_function=cost_function,
        dimension=3,
        bounds=[(-5.0, 5.0)] * 3,
        target_precision=1e-6
    )


@pytest.fixture
def quantum_optimizer():
    """Fixture for quantum variational optimizer."""
    return QuantumVariationalOptimizer(
        num_qubits=6,
        max_iterations=100,
        learning_rate=0.1,
        momentum=0.9,
        temperature_schedule='adaptive'
    )


# Performance markers for pytest
pytestmark = pytest.mark.asyncio