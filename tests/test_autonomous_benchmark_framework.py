"""
Test suite for Autonomous Benchmark Framework

This module provides comprehensive tests for the autonomous benchmarking
framework with statistical validation capabilities.
"""

import asyncio
import numpy as np
import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Import the modules to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_cost_tracker.autonomous_benchmark_framework import (
    StatisticalTest,
    BenchmarkExperiment,
    AlgorithmVariant,
    StatisticalAnalyzer,
    AutonomousBenchmarkFramework
)
from llm_cost_tracker.quantum_variational_hybrid import OptimizationProblem


class TestStatisticalTest:
    """Test suite for StatisticalTest class."""
    
    def test_statistical_test_creation(self):
        """Test statistical test object creation."""
        test = StatisticalTest(
            test_name="Mann-Whitney U",
            test_statistic=123.45,
            p_value=0.023,
            effect_size=0.67,
            confidence_interval=(0.1, 0.9),
            interpretation="Significant difference found"
        )
        
        assert test.test_name == "Mann-Whitney U"
        assert test.test_statistic == 123.45
        assert test.p_value == 0.023
        assert test.effect_size == 0.67
        assert test.confidence_interval == (0.1, 0.9)
        assert test.interpretation == "Significant difference found"
    
    def test_significance_check(self):
        """Test statistical significance checking."""
        # Significant test
        significant_test = StatisticalTest(
            test_name="Test", test_statistic=10, p_value=0.01
        )
        assert significant_test.is_significant(alpha=0.05) == True
        
        # Non-significant test
        non_significant_test = StatisticalTest(
            test_name="Test", test_statistic=5, p_value=0.08
        )
        assert non_significant_test.is_significant(alpha=0.05) == False


class TestStatisticalAnalyzer:
    """Test suite for StatisticalAnalyzer class."""
    
    def setup_method(self):
        """Setup test data."""
        self.analyzer = StatisticalAnalyzer(significance_level=0.05)
        
        # Sample data for testing
        self.group1 = [1.2, 1.5, 1.3, 1.8, 1.1, 1.6, 1.4, 1.7, 1.2, 1.5]
        self.group2 = [2.1, 2.3, 2.0, 2.5, 1.9, 2.4, 2.2, 2.6, 2.0, 2.3]
        self.group3 = [3.1, 3.2, 3.0, 3.4, 2.9, 3.3, 3.1, 3.5, 3.0, 3.2]
    
    def test_mann_whitney_u_test(self):
        """Test Mann-Whitney U test implementation."""
        result = self.analyzer.mann_whitney_u_test(self.group1, self.group2)
        
        assert isinstance(result, StatisticalTest)
        assert result.test_name == "Mann-Whitney U"
        assert isinstance(result.test_statistic, (int, float))
        assert 0 <= result.p_value <= 1
        assert result.effect_size is not None
        assert result.confidence_interval is not None
        assert len(result.interpretation) > 0
    
    def test_mann_whitney_insufficient_data(self):
        """Test Mann-Whitney U test with insufficient sample size."""
        small_group1 = [1.0, 1.1]
        small_group2 = [2.0]
        
        result = self.analyzer.mann_whitney_u_test(small_group1, small_group2)
        
        assert result.p_value == 1.0
        assert "Insufficient sample size" in result.interpretation
    
    def test_friedman_test(self):
        """Test Friedman test for multiple groups."""
        groups = [self.group1, self.group2, self.group3]
        result = self.analyzer.friedman_test(groups)
        
        assert isinstance(result, StatisticalTest)
        assert result.test_name == "Friedman"
        assert isinstance(result.test_statistic, (int, float))
        assert 0 <= result.p_value <= 1
        assert result.effect_size is not None
        assert len(result.interpretation) > 0
    
    def test_friedman_insufficient_groups(self):
        """Test Friedman test with insufficient groups."""
        groups = [self.group1, self.group2]  # Only 2 groups
        result = self.analyzer.friedman_test(groups)
        
        assert result.p_value == 1.0
        assert "Need at least 3 groups" in result.interpretation
    
    def test_cohens_d_calculation(self):
        """Test Cohen's d effect size calculation."""
        cohens_d = self.analyzer.compute_effect_size(
            self.group1, self.group2, method='cohens_d'
        )
        
        assert isinstance(cohens_d, (int, float))
        
        # Should be negative since group2 > group1
        assert cohens_d < 0
        
        # Should be a large effect size
        assert abs(cohens_d) > 0.8
    
    def test_hedges_g_calculation(self):
        """Test Hedge's g effect size calculation."""
        hedges_g = self.analyzer.compute_effect_size(
            self.group1, self.group2, method='hedges_g'
        )
        
        assert isinstance(hedges_g, (int, float))
        
        # Should be similar to Cohen's d but slightly smaller (bias corrected)
        cohens_d = self.analyzer.compute_effect_size(
            self.group1, self.group2, method='cohens_d'
        )
        
        assert abs(hedges_g) <= abs(cohens_d)
    
    def test_cliff_delta_calculation(self):
        """Test Cliff's delta effect size calculation."""
        cliff_delta = self.analyzer.compute_effect_size(
            self.group1, self.group2, method='cliff_delta'
        )
        
        assert isinstance(cliff_delta, (int, float))
        assert -1 <= cliff_delta <= 1
        
        # Should be negative since group2 > group1 (assuming smaller is better)
        assert cliff_delta < 0
    
    def test_multiple_comparison_corrections(self):
        """Test multiple comparison correction methods."""
        p_values = [0.01, 0.03, 0.05, 0.08, 0.12]
        
        # Test Bonferroni correction
        bonferroni = self.analyzer.correct_multiple_comparisons(p_values, 'bonferroni')
        assert len(bonferroni) == len(p_values)
        assert all(corrected >= original for corrected, original in zip(bonferroni, p_values))
        
        # Test Holm correction
        holm = self.analyzer.correct_multiple_comparisons(p_values, 'holm')
        assert len(holm) == len(p_values)
        
        # Test FDR correction
        fdr = self.analyzer.correct_multiple_comparisons(p_values, 'fdr_bh')
        assert len(fdr) == len(p_values)
        
        # FDR should generally be less conservative than Bonferroni
        assert all(fdr_p <= bon_p for fdr_p, bon_p in zip(fdr, bonferroni))
    
    def test_unknown_method_errors(self):
        """Test error handling for unknown methods."""
        with pytest.raises(ValueError):
            self.analyzer.compute_effect_size(self.group1, self.group2, method='unknown')
        
        with pytest.raises(ValueError):
            self.analyzer.correct_multiple_comparisons([0.01, 0.05], method='unknown')


class TestAlgorithmVariant:
    """Test suite for AlgorithmVariant class."""
    
    def test_algorithm_variant_creation(self):
        """Test algorithm variant creation."""
        variant = AlgorithmVariant(
            name="test_variant",
            config={'learning_rate': 0.1, 'momentum': 0.9},
            description="Test algorithm variant",
            expected_advantage=0.15
        )
        
        assert variant.name == "test_variant"
        assert variant.config['learning_rate'] == 0.1
        assert variant.config['momentum'] == 0.9
        assert variant.description == "Test algorithm variant"
        assert variant.expected_advantage == 0.15
    
    def test_optimizer_creation(self):
        """Test optimizer instance creation."""
        variant = AlgorithmVariant(
            name="test_variant",
            config={
                'num_qubits': 4,
                'max_iterations': 100,
                'learning_rate': 0.1
            },
            description="Test variant"
        )
        
        optimizer = variant.create_optimizer()
        
        # Check that optimizer has correct configuration
        assert optimizer.num_qubits == 4
        assert optimizer.max_iterations == 100
        assert optimizer.learning_rate == 0.1


class TestBenchmarkExperiment:
    """Test suite for BenchmarkExperiment class."""
    
    def test_experiment_creation(self):
        """Test benchmark experiment creation."""
        def simple_function(x):
            return sum(xi**2 for xi in x)
        
        problem = OptimizationProblem(
            cost_function=simple_function,
            dimension=2,
            bounds=[(-1.0, 1.0)] * 2
        )
        
        experiment = BenchmarkExperiment(
            experiment_id="test_exp_001",
            name="Test Experiment",
            description="A test experiment",
            algorithms=["alg1", "alg2"],
            problems=[problem],
            num_repetitions=10
        )
        
        assert experiment.experiment_id == "test_exp_001"
        assert experiment.name == "Test Experiment"
        assert experiment.description == "A test experiment"
        assert experiment.algorithms == ["alg1", "alg2"]
        assert len(experiment.problems) == 1
        assert experiment.num_repetitions == 10
        assert experiment.status == "initialized"


class TestAutonomousBenchmarkFramework:
    """Test suite for AutonomousBenchmarkFramework class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.framework = AutonomousBenchmarkFramework(
            results_directory=self.temp_dir,
            significance_level=0.05,
            min_effect_size=0.3,
            target_power=0.8
        )
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_framework_initialization(self):
        """Test framework initialization."""
        assert self.framework.results_dir.exists()
        assert self.framework.significance_level == 0.05
        assert self.framework.min_effect_size == 0.3
        assert self.framework.target_power == 0.8
        assert len(self.framework.algorithm_registry) > 0  # Should have default algorithms
        assert isinstance(self.framework.statistical_analyzer, StatisticalAnalyzer)
    
    def test_algorithm_registration(self):
        """Test algorithm variant registration."""
        variant = AlgorithmVariant(
            name="custom_algorithm",
            config={'learning_rate': 0.05, 'momentum': 0.95},
            description="Custom test algorithm"
        )
        
        self.framework.register_algorithm(variant)
        
        assert "custom_algorithm" in self.framework.algorithm_registry
        assert self.framework.algorithm_registry["custom_algorithm"] == variant
    
    def test_experiment_creation(self):
        """Test experiment creation with sample size calculation."""
        def simple_function(x):
            return sum(xi**2 for xi in x)
        
        problem = OptimizationProblem(
            cost_function=simple_function,
            dimension=2,
            bounds=[(-1.0, 1.0)] * 2
        )
        
        # Use default algorithms that should exist
        algorithm_names = list(self.framework.algorithm_registry.keys())[:2]
        
        exp_id = self.framework.create_experiment(
            name="Test Experiment",
            description="Testing experiment creation",
            algorithm_names=algorithm_names,
            problems=[problem]
        )
        
        assert exp_id in self.framework.experiments
        experiment = self.framework.experiments[exp_id]
        
        assert experiment.name == "Test Experiment"
        assert experiment.algorithms == algorithm_names
        assert len(experiment.problems) == 1
        assert experiment.num_repetitions >= 15  # Should calculate reasonable sample size
    
    def test_experiment_creation_unknown_algorithm(self):
        """Test experiment creation with unknown algorithm."""
        def simple_function(x):
            return sum(xi**2 for xi in x)
        
        problem = OptimizationProblem(
            cost_function=simple_function,
            dimension=2,
            bounds=[(-1.0, 1.0)] * 2
        )
        
        with pytest.raises(ValueError, match="Unknown algorithm"):
            self.framework.create_experiment(
                name="Invalid Experiment",
                description="Testing error handling",
                algorithm_names=["nonexistent_algorithm"],
                problems=[problem]
            )
    
    def test_sample_size_calculation(self):
        """Test sample size calculation for different scenarios."""
        # Test with different parameters
        n1 = self.framework._calculate_required_sample_size(
            num_groups=2, effect_size=0.8, power=0.8
        )
        
        n2 = self.framework._calculate_required_sample_size(
            num_groups=2, effect_size=0.3, power=0.8
        )
        
        n3 = self.framework._calculate_required_sample_size(
            num_groups=5, effect_size=0.5, power=0.9
        )
        
        # Larger effect size should require smaller sample
        assert n2 > n1
        
        # More groups should require larger sample
        assert n3 > n1
        
        # Should have minimum sample size
        assert all(n >= 20 for n in [n1, n2, n3])
    
    @pytest.mark.asyncio
    async def test_small_experiment_run(self):
        """Test running a small experiment."""
        # Create simple test problem
        def quadratic_function(x):
            return sum(xi**2 for xi in x)
        
        problem = OptimizationProblem(
            cost_function=quadratic_function,
            dimension=2,
            bounds=[(-1.0, 1.0)] * 2,
            target_precision=1e-3
        )
        
        # Get available algorithms
        algorithm_names = list(self.framework.algorithm_registry.keys())[:2]
        
        # Create experiment with minimal repetitions
        exp_id = self.framework.create_experiment(
            name="Small Test Experiment",
            description="Minimal experiment for testing",
            algorithm_names=algorithm_names,
            problems=[problem],
            num_repetitions=3  # Very small for testing
        )
        
        # Mock the optimization process to avoid long runtimes
        with patch.object(self.framework.algorithm_registry[algorithm_names[0]], 'create_optimizer') as mock_optimizer1, \
             patch.object(self.framework.algorithm_registry[algorithm_names[1]], 'create_optimizer') as mock_optimizer2:
            
            # Create mock optimizer instances
            mock_opt1 = Mock()
            mock_opt2 = Mock()
            mock_optimizer1.return_value = mock_opt1
            mock_optimizer2.return_value = mock_opt2
            
            # Mock optimization results
            async def mock_optimize(problem):
                return {
                    'success': True,
                    'best_solution': [0.1, 0.2],
                    'best_cost': 0.05,
                    'optimization_time_seconds': 1.0,
                    'iterations': 50,
                    'convergence_achieved': True,
                    'quantum_advantage_ratio': 0.1,
                    'classical_baseline_cost': 0.055
                }
            
            mock_opt1.optimize = mock_optimize
            mock_opt2.optimize = mock_optimize
            
            # Run experiment
            result = await self.framework.run_experiment(exp_id)
            
            # Verify result structure
            assert 'experiment_metadata' in result
            assert 'raw_results' in result
            assert 'statistical_analysis' in result
            assert 'recommendations' in result
            assert 'publication_summary' in result
            
            # Check metadata
            metadata = result['experiment_metadata']
            assert metadata['experiment_id'] == exp_id
            assert metadata['algorithms'] == algorithm_names
            assert metadata['num_repetitions'] == 3
            assert metadata['total_runs'] == 6  # 2 algorithms × 1 problem × 3 reps
    
    def test_results_saving(self):
        """Test saving experiment results to disk."""
        experiment_id = "test_exp_save"
        test_results = {
            'experiment_metadata': {'test': True},
            'raw_results': {},
            'statistical_analysis': {}
        }
        
        self.framework._save_experiment_results(experiment_id, test_results)
        
        # Check that file was created
        results_file = self.framework.results_dir / f"{experiment_id}_results.json"
        assert results_file.exists()
        
        # Check that content is correct
        with open(results_file, 'r') as f:
            loaded_results = json.load(f)
        
        assert loaded_results['experiment_metadata']['test'] == True
    
    def test_regression_detection(self):
        """Test performance regression detection."""
        # Setup baseline results
        for i in range(10):
            self.framework.continuous_results.append({
                'timestamp': f"2023-01-{i+1:02d}T12:00:00",
                'results': {'mean_cost': 1.0 + 0.1 * np.random.randn()},
                'regression_detected': False
            })
        
        # Test normal performance (should not trigger regression)
        normal_result = {'mean_cost': 1.05, 'success_rate': 0.95}
        is_regression = self.framework._check_for_regressions(normal_result)
        assert not is_regression
        
        # Test performance regression (should trigger)
        bad_result = {'mean_cost': 2.0, 'success_rate': 0.95}
        is_regression = self.framework._check_for_regressions(bad_result)
        assert is_regression
    
    def test_experiment_summary_retrieval(self):
        """Test experiment summary retrieval."""
        # Create dummy experiment
        experiment_id = "test_summary"
        test_results = {
            'experiment_metadata': {'name': 'Test Summary Experiment'},
            'statistical_analysis': {'performance_summary': {'best_algorithm': 'test'}}
        }
        
        # Save results
        self.framework._save_experiment_results(experiment_id, test_results)
        
        # Retrieve summary
        summary = self.framework.get_experiment_summary(experiment_id)
        
        assert summary['experiment_metadata']['name'] == 'Test Summary Experiment'
        assert summary['statistical_analysis']['performance_summary']['best_algorithm'] == 'test'
    
    def test_experiment_summary_not_found(self):
        """Test experiment summary retrieval for non-existent experiment."""
        summary = self.framework.get_experiment_summary("nonexistent")
        
        assert 'error' in summary
        assert summary['error'] == 'Results not found'
    
    def test_statistical_analysis_helpers(self):
        """Test statistical analysis helper methods."""
        # Mock statistical results for testing
        mock_statistical_results = {
            'pairwise_comparisons': {
                'alg1_vs_alg2': {
                    'mann_whitney_test': {'significant': True, 'p_value': 0.01},
                    'effect_sizes': {'cohens_d': 0.9}
                }
            },
            'performance_summary': {
                'best_algorithm': 'alg1',
                'algorithm_statistics': {
                    'alg1': {'mean_cost': 0.1, 'std_cost': 0.02, 'success_rate': 0.95}
                }
            }
        }
        
        # Test recommendation generation
        recommendations = self.framework._generate_recommendations(mock_statistical_results)
        
        assert len(recommendations) > 0
        assert any("significant" in rec.lower() for rec in recommendations)
        assert any("alg1" in rec for rec in recommendations)
        
        # Test key findings extraction
        experiment = BenchmarkExperiment(
            experiment_id="test",
            name="Test",
            description="Test",
            algorithms=["alg1", "alg2"],
            problems=[],
            num_repetitions=30
        )
        
        findings = self.framework._extract_key_findings(mock_statistical_results)
        
        assert len(findings) > 0
        assert any("significant" in finding.lower() for finding in findings)


class TestBenchmarkIntegration:
    """Integration tests for the benchmark framework."""
    
    def setup_method(self):
        """Setup integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.framework = AutonomousBenchmarkFramework(
            results_directory=self.temp_dir,
            significance_level=0.05
        )
        
        # Add a simple custom algorithm for testing
        simple_variant = AlgorithmVariant(
            name="simple_test",
            config={
                'num_qubits': 4,
                'max_iterations': 10,  # Very fast for testing
                'learning_rate': 0.1,
                'momentum': 0.9,
                'temperature_schedule': 'linear'
            },
            description="Simple test algorithm"
        )
        self.framework.register_algorithm(simple_variant)
    
    def teardown_method(self):
        """Cleanup integration test environment."""
        shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_end_to_end_benchmark_workflow(self):
        """Test complete benchmark workflow from creation to analysis."""
        
        # Create simple test problems
        def sphere_function(x):
            return sum(xi**2 for xi in x)
        
        def shifted_sphere_function(x):
            return sum((xi - 0.5)**2 for xi in x)
        
        problems = [
            OptimizationProblem(
                cost_function=sphere_function,
                dimension=2,
                bounds=[(-1.0, 1.0)] * 2,
                target_precision=1e-2
            ),
            OptimizationProblem(
                cost_function=shifted_sphere_function,
                dimension=2,
                bounds=[(-1.0, 1.0)] * 2,
                target_precision=1e-2
            )
        ]
        
        # Use available algorithms (should include our custom one)
        available_algorithms = list(self.framework.algorithm_registry.keys())
        algorithm_names = available_algorithms[:2]  # Use first 2 algorithms
        
        # Create experiment
        exp_id = self.framework.create_experiment(
            name="Integration Test Experiment",
            description="End-to-end integration test",
            algorithm_names=algorithm_names,
            problems=problems,
            num_repetitions=2  # Small for testing
        )
        
        # Verify experiment was created
        assert exp_id in self.framework.experiments
        experiment = self.framework.experiments[exp_id]
        assert experiment.status == "initialized"
        
        # Mock the actual optimization to avoid long runtimes
        with patch('llm_cost_tracker.quantum_variational_hybrid.QuantumVariationalOptimizer') as mock_optimizer_class:
            
            # Setup mock optimizer
            mock_optimizer = Mock()
            mock_optimizer_class.return_value = mock_optimizer
            
            # Mock optimization results with some variation
            call_count = 0
            
            async def mock_optimize(problem):
                nonlocal call_count
                call_count += 1
                
                # Vary results slightly to simulate different algorithms/problems
                base_cost = 0.1 + 0.05 * (call_count % 3)
                
                return {
                    'success': True,
                    'best_solution': [0.1, 0.2],
                    'best_cost': base_cost,
                    'optimization_time_seconds': 0.5,
                    'iterations': 25,
                    'convergence_achieved': True,
                    'quantum_advantage_ratio': 0.05 + 0.1 * (call_count % 2),
                    'classical_baseline_cost': base_cost + 0.01
                }
            
            mock_optimizer.optimize = mock_optimize
            
            # Run the experiment
            result = await self.framework.run_experiment(exp_id)
            
            # Verify the experiment completed
            assert experiment.status == "completed"
            
            # Verify result structure
            required_keys = [
                'experiment_metadata', 'raw_results', 'statistical_analysis', 
                'recommendations', 'publication_summary'
            ]
            for key in required_keys:
                assert key in result, f"Missing key: {key}"
            
            # Check experiment metadata
            metadata = result['experiment_metadata']
            assert metadata['experiment_id'] == exp_id
            assert metadata['algorithms'] == algorithm_names
            assert metadata['num_problems'] == 2
            assert metadata['num_repetitions'] == 2
            assert metadata['total_runs'] == 8  # 2 algs × 2 problems × 2 reps
            
            # Check statistical analysis exists
            stats = result['statistical_analysis']
            assert 'performance_summary' in stats
            
            # Check that results were saved to disk
            results_file = self.framework.results_dir / f"{exp_id}_results.json"
            assert results_file.exists()
            
            # Verify saved results can be loaded
            with open(results_file, 'r') as f:
                saved_results = json.load(f)
            assert saved_results['experiment_metadata']['experiment_id'] == exp_id


# Performance and stress tests

@pytest.mark.performance  
class TestBenchmarkPerformance:
    """Performance tests for benchmark framework."""
    
    def setup_method(self):
        """Setup performance test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.framework = AutonomousBenchmarkFramework(
            results_directory=self.temp_dir
        )
    
    def teardown_method(self):
        """Cleanup performance test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_statistical_analyzer_performance(self):
        """Test performance of statistical analysis on large datasets."""
        analyzer = StatisticalAnalyzer()
        
        # Generate large datasets
        np.random.seed(42)
        large_group1 = np.random.normal(1.0, 0.5, 1000).tolist()
        large_group2 = np.random.normal(1.2, 0.5, 1000).tolist()
        
        import time
        
        # Time Mann-Whitney U test
        start_time = time.time()
        mw_result = analyzer.mann_whitney_u_test(large_group1, large_group2)
        mw_time = time.time() - start_time
        
        assert mw_time < 5.0  # Should complete within 5 seconds
        assert mw_result.test_name == "Mann-Whitney U"
        assert 0 <= mw_result.p_value <= 1
        
        # Time effect size calculation
        start_time = time.time()
        cohens_d = analyzer.compute_effect_size(large_group1, large_group2, 'cohens_d')
        effect_time = time.time() - start_time
        
        assert effect_time < 1.0  # Should be very fast
        assert isinstance(cohens_d, (int, float))
    
    def test_multiple_comparison_correction_performance(self):
        """Test performance of multiple comparison corrections."""
        analyzer = StatisticalAnalyzer()
        
        # Generate many p-values
        p_values = np.random.uniform(0.001, 0.999, 1000).tolist()
        
        import time
        
        methods = ['bonferroni', 'holm', 'fdr_bh']
        
        for method in methods:
            start_time = time.time()
            corrected = analyzer.correct_multiple_comparisons(p_values, method)
            correction_time = time.time() - start_time
            
            assert correction_time < 2.0, f"{method} correction took too long: {correction_time}s"
            assert len(corrected) == len(p_values)


# Fixtures for complex test scenarios

@pytest.fixture
def complex_benchmark_problems():
    """Fixture providing complex benchmark problems."""
    
    def rosenbrock_nd(x):
        """N-dimensional Rosenbrock function."""
        return sum(100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 
                  for i in range(len(x)-1))
    
    def rastrigin_nd(x):
        """N-dimensional Rastrigin function."""
        A = 10
        n = len(x)
        return A * n + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)
    
    def ackley_nd(x):
        """N-dimensional Ackley function."""
        x = np.array(x)
        n = len(x)
        term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
        term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
        return term1 + term2 + 20 + np.e
    
    problems = []
    
    for dim in [2, 3, 5]:
        problems.extend([
            OptimizationProblem(
                cost_function=rosenbrock_nd,
                dimension=dim,
                bounds=[(-2.0, 2.0)] * dim,
                target_precision=1e-3
            ),
            OptimizationProblem(
                cost_function=rastrigin_nd,
                dimension=dim,
                bounds=[(-5.12, 5.12)] * dim,
                target_precision=1e-2
            ),
            OptimizationProblem(
                cost_function=ackley_nd,
                dimension=dim,
                bounds=[(-32.768, 32.768)] * dim,
                target_precision=1e-2
            )
        ])
    
    return problems


@pytest.fixture 
def mock_algorithm_variants():
    """Fixture providing mock algorithm variants for testing."""
    variants = []
    
    configs = [
        {'learning_rate': 0.01, 'momentum': 0.9, 'temperature_schedule': 'linear'},
        {'learning_rate': 0.1, 'momentum': 0.8, 'temperature_schedule': 'exponential'}, 
        {'learning_rate': 0.05, 'momentum': 0.95, 'temperature_schedule': 'adaptive'}
    ]
    
    for i, config in enumerate(configs):
        variant = AlgorithmVariant(
            name=f"mock_algorithm_{i}",
            config=config,
            description=f"Mock algorithm variant {i}",
            expected_advantage=0.1 * i
        )
        variants.append(variant)
    
    return variants


# Test markers for different test categories
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.statistical
]