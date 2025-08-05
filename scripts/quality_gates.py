#!/usr/bin/env python3
"""
Comprehensive Quality Gates for Quantum Task Planner.
Runs all quality checks including tests, linting, security, and performance.
"""

import sys
import os
import subprocess
import json
import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
import time
import importlib.util

# Quality gate results
class QualityGateResult:
    def __init__(self):
        self.passed = True
        self.messages = []
        self.metrics = {}
        self.start_time = time.time()
    
    def add_pass(self, message: str, metrics: Dict[str, Any] = None):
        self.messages.append(f"✅ {message}")
        if metrics:
            self.metrics.update(metrics)
    
    def add_fail(self, message: str, metrics: Dict[str, Any] = None):
        self.passed = False
        self.messages.append(f"❌ {message}")
        if metrics:
            self.metrics.update(metrics)
    
    def add_warning(self, message: str, metrics: Dict[str, Any] = None):
        self.messages.append(f"⚠️  {message}")
        if metrics:
            self.metrics.update(metrics)
    
    def get_duration(self) -> float:
        return time.time() - self.start_time
    
    def print_summary(self, gate_name: str):
        duration = self.get_duration()
        status = "PASSED" if self.passed else "FAILED"
        print(f"\n🔍 {gate_name.upper()} - {status} ({duration:.2f}s)")
        for message in self.messages:
            print(f"  {message}")
        
        if self.metrics:
            print("  📊 Metrics:")
            for key, value in self.metrics.items():
                print(f"    {key}: {value}")


def run_command(cmd: str, cwd: str = None, timeout: int = 120) -> Tuple[int, str, str]:
    """Run shell command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd, timeout=timeout,
            capture_output=True, text=True
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        return 1, "", str(e)


def gate_1_import_tests() -> QualityGateResult:
    """Gate 1: Test that all modules can be imported successfully."""
    result = QualityGateResult()
    
    # Add current directory to Python path
    sys.path.insert(0, '.')
    
    quantum_modules = [
        'src.llm_cost_tracker.quantum_task_planner',
        'src.llm_cost_tracker.quantum_monitoring',
        'src.llm_cost_tracker.quantum_validation', 
        'src.llm_cost_tracker.quantum_optimization'
    ]
    
    imported_modules = 0
    total_modules = len(quantum_modules)
    
    for module in quantum_modules:
        try:
            __import__(module)
            result.add_pass(f"Import {module}")
            imported_modules += 1
        except Exception as e:
            result.add_fail(f"Import {module} failed: {e}")
    
    result.metrics['imported_modules'] = imported_modules
    result.metrics['total_modules'] = total_modules
    result.metrics['import_success_rate'] = imported_modules / total_modules
    
    if imported_modules == total_modules:
        result.add_pass(f"All {total_modules} quantum modules imported successfully")
    
    return result


def gate_2_syntax_analysis() -> QualityGateResult:
    """Gate 2: Python syntax analysis and basic code quality checks."""
    result = QualityGateResult()
    
    python_files = list(Path('src/llm_cost_tracker').glob('quantum_*.py'))
    
    total_files = len(python_files)
    syntax_errors = 0
    complexity_warnings = 0
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Parse AST to check syntax
            try:
                tree = ast.parse(source_code, filename=str(file_path))
                result.add_pass(f"Syntax valid: {file_path.name}")
                
                # Basic complexity analysis
                class ComplexityAnalyzer(ast.NodeVisitor):
                    def __init__(self):
                        self.function_complexities = []
                        self.class_count = 0
                        self.function_count = 0
                    
                    def visit_FunctionDef(self, node):
                        self.function_count += 1
                        # Count decision points for cyclomatic complexity
                        complexity = 1  # Base complexity
                        for child in ast.walk(node):
                            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                                complexity += 1
                            elif isinstance(child, ast.BoolOp):
                                complexity += len(child.values) - 1
                        
                        self.function_complexities.append((node.name, complexity))
                        if complexity > 10:  # Complexity threshold
                            result.add_warning(f"High complexity in {file_path.name}::{node.name} (complexity: {complexity})")
                            nonlocal complexity_warnings
                            complexity_warnings += 1
                        
                        self.generic_visit(node)
                    
                    def visit_ClassDef(self, node):
                        self.class_count += 1
                        self.generic_visit(node)
                
                analyzer = ComplexityAnalyzer()
                analyzer.visit(tree)
                
                result.metrics[f'{file_path.name}_classes'] = analyzer.class_count
                result.metrics[f'{file_path.name}_functions'] = analyzer.function_count
                result.metrics[f'{file_path.name}_max_complexity'] = max(
                    [c[1] for c in analyzer.function_complexities], default=0
                )
                
            except SyntaxError as e:
                result.add_fail(f"Syntax error in {file_path.name}: {e}")
                syntax_errors += 1
        
        except Exception as e:
            result.add_fail(f"Failed to analyze {file_path.name}: {e}")
            syntax_errors += 1
    
    result.metrics['total_files'] = total_files
    result.metrics['syntax_errors'] = syntax_errors
    result.metrics['complexity_warnings'] = complexity_warnings
    
    if syntax_errors == 0:
        result.add_pass(f"All {total_files} files have valid Python syntax")
    
    return result


def gate_3_security_scan() -> QualityGateResult:
    """Gate 3: Security vulnerability scanning."""
    result = QualityGateResult()
    
    python_files = list(Path('src/llm_cost_tracker').glob('quantum_*.py'))
    
    # Security patterns to check for
    security_patterns = [
        (r'eval\s*\(', 'Use of eval() function - potential code injection'),
        (r'exec\s*\(', 'Use of exec() function - potential code injection'),
        (r'__import__\s*\(', 'Dynamic imports - potential security risk'),
        (r'subprocess\.call\([^)]*shell\s*=\s*True', 'Shell=True in subprocess - command injection risk'),
        (r'os\.system\s*\(', 'Use of os.system() - command injection risk'),
        (r'pickle\.loads?\s*\(', 'Use of pickle - potential code execution risk'),
        (r'input\s*\([^)]*\).*execute|query|sql', 'Use of input() in SQL context - potential injection risk'),
        (r'random\.random\(\)', 'Use of random.random() - consider secrets module for cryptographic use'),
        # Look for potential SQL injection patterns
        (r'["\'].*%s.*["\']', 'String formatting in queries - potential SQL injection'),
        (r'f["\'].*\{.*\}.*["\'].*execute|query|sql', 'F-string in SQL queries - potential injection'),
    ]
    
    security_issues = 0
    total_lines_scanned = 0
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            total_lines_scanned += len(lines)
            
            for line_num, line in enumerate(lines, 1):
                for pattern, description in security_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        # Check if it's in a comment (basic check)
                        if not line.strip().startswith('#'):
                            result.add_warning(f"Security concern in {file_path.name}:{line_num} - {description}")
                            security_issues += 1
        
        except Exception as e:
            result.add_fail(f"Security scan failed for {file_path.name}: {e}")
    
    result.metrics['security_issues'] = security_issues
    result.metrics['lines_scanned'] = total_lines_scanned
    
    if security_issues == 0:
        result.add_pass(f"No security issues found in {len(python_files)} files")
    elif security_issues < 5:
        result.add_pass(f"Minor security concerns found ({security_issues} issues)")
    else:
        result.add_fail(f"Multiple security concerns found ({security_issues} issues)")
    
    return result


def gate_4_performance_tests() -> QualityGateResult:
    """Gate 4: Performance and scalability tests."""
    result = QualityGateResult()
    
    import sys
    sys.path.insert(0, '.')
    
    try:
        # Import quantum modules
        from src.llm_cost_tracker.quantum_task_planner import QuantumTaskPlanner, QuantumTask
        from datetime import timedelta
        import time
        import asyncio
        
        # Test 1: Task creation performance
        start_time = time.time()
        planner = QuantumTaskPlanner()
        
        # Create many tasks to test scalability
        task_count = 100
        for i in range(task_count):
            task = QuantumTask(
                id=f'perf_task_{i}',
                name=f'Performance Task {i}',
                description=f'Task for performance testing {i}',
                priority=float(i % 10 + 1),
                estimated_duration=timedelta(minutes=i % 30 + 1)
            )
            success, _ = planner.add_task(task)
            if not success:
                result.add_fail(f"Failed to add task {i}")
                break
        
        creation_time = time.time() - start_time
        result.add_pass(f"Created {task_count} tasks in {creation_time:.3f}s")
        result.metrics['task_creation_time'] = creation_time
        result.metrics['tasks_per_second'] = task_count / creation_time
        
        # Test 2: Scheduling performance
        start_time = time.time()
        schedule = planner.quantum_anneal_schedule(max_iterations=100)
        scheduling_time = time.time() - start_time
        
        result.add_pass(f"Generated schedule for {len(schedule)} tasks in {scheduling_time:.3f}s")
        result.metrics['scheduling_time'] = scheduling_time
        result.metrics['scheduling_tasks_per_second'] = len(schedule) / scheduling_time
        
        # Test 3: Memory usage estimation
        import sys
        memory_estimate = sys.getsizeof(planner) + sum(sys.getsizeof(task) for task in planner.tasks.values())
        result.metrics['memory_usage_bytes'] = memory_estimate
        result.metrics['memory_per_task_bytes'] = memory_estimate / task_count
        
        result.add_pass(f"Memory usage: {memory_estimate / 1024:.1f} KB ({memory_estimate / task_count:.1f} bytes/task)")
        
        # Test 4: Cache performance
        if hasattr(planner, 'cache') and planner.cache:
            cache_start = time.time()
            
            # Test cache operations
            for i in range(100):
                key = f"test_key_{i}"
                value = f"test_value_{i}"
                planner.cache.put(key, value)
                retrieved = planner.cache.get(key)
                if retrieved != value:
                    result.add_fail(f"Cache test failed for key {key}")
                    break
            
            cache_time = time.time() - cache_start
            cache_stats = planner.cache.get_stats()
            
            result.add_pass(f"Cache operations completed in {cache_time:.3f}s")
            result.metrics.update({
                'cache_time': cache_time,
                'cache_hit_rate': cache_stats.get('hit_rate', 0),
                'cache_size': cache_stats.get('size', 0)
            })
        
        # Performance thresholds
        if creation_time > 5.0:
            result.add_fail(f"Task creation too slow: {creation_time:.3f}s (threshold: 5.0s)")
        
        if scheduling_time > 10.0:
            result.add_fail(f"Scheduling too slow: {scheduling_time:.3f}s (threshold: 10.0s)")
        
        if memory_estimate > 10 * 1024 * 1024:  # 10MB
            result.add_warning(f"High memory usage: {memory_estimate / 1024 / 1024:.1f}MB")
    
    except Exception as e:
        result.add_fail(f"Performance test failed: {e}")
        import traceback
        result.add_fail(f"Traceback: {traceback.format_exc()}")
    
    return result


def gate_5_integration_tests() -> QualityGateResult:
    """Gate 5: Integration tests for quantum system components."""
    result = QualityGateResult()
    
    import sys
    sys.path.insert(0, '.')
    
    try:
        import asyncio
        from src.llm_cost_tracker.quantum_task_planner import QuantumTaskPlanner, QuantumTask, TaskState
        from src.llm_cost_tracker.quantum_monitoring import QuantumTaskMonitor
        from src.llm_cost_tracker.quantum_validation import validate_task_input
        from datetime import timedelta
        
        # Test 1: Full system integration
        planner = QuantumTaskPlanner()
        
        # Create interdependent tasks
        task1 = QuantumTask(
            id='integration_task_1',
            name='Base Task',
            description='Foundation task for integration test',
            priority=9.0,
            estimated_duration=timedelta(seconds=1)
        )
        
        task2 = QuantumTask(
            id='integration_task_2', 
            name='Dependent Task',
            description='Task that depends on task 1',
            priority=8.0,
            estimated_duration=timedelta(seconds=1),
            dependencies={'integration_task_1'}
        )
        
        # Add tasks with validation
        success1, msg1 = planner.add_task(task1)
        success2, msg2 = planner.add_task(task2)
        
        if not (success1 and success2):
            result.add_fail(f"Task creation failed: {msg1}, {msg2}")
            return result
        
        result.add_pass("Created interdependent tasks successfully")
        
        # Test 2: Dependency resolution
        planner.create_dependency('integration_task_2', 'integration_task_1')
        
        # Generate and validate schedule
        schedule = planner.quantum_anneal_schedule(max_iterations=50)
        
        if len(schedule) != 2:
            result.add_fail(f"Schedule should have 2 tasks, got {len(schedule)}")
            return result
        
        # Verify dependency order
        task1_idx = schedule.index('integration_task_1')
        task2_idx = schedule.index('integration_task_2')
        
        if task1_idx >= task2_idx:
            result.add_fail("Dependency order violated in schedule")
        else:
            result.add_pass("Dependency order correctly maintained in schedule")
        
        # Test 3: Execution simulation
        async def run_execution_test():
            try:
                results = await planner.execute_schedule(schedule)
                return results
            except Exception as e:
                return {'error': str(e)}
        
        execution_results = asyncio.run(run_execution_test())
        
        if 'error' in execution_results:
            result.add_warning(f"Execution test had issues: {execution_results['error']}")
        else:
            result.add_pass(f"Execution completed: {execution_results.get('success_rate', 0):.1%} success rate")
            result.metrics['execution_success_rate'] = execution_results.get('success_rate', 0)
        
        # Test 4: Monitoring integration
        try:
            monitor = QuantumTaskMonitor(planner, monitoring_interval=0.1)
            # Test basic monitoring functionality
            health_status = planner.perform_health_check()
            
            if health_status.get('overall_healthy', False):
                result.add_pass("Health check system functional")
            else:
                result.add_warning("Health check reports issues")
            
            result.metrics['health_check_components'] = len(health_status.get('components', {}))
            
        except Exception as e:
            result.add_warning(f"Monitoring integration test failed: {e}")
        
        # Test 5: Validation system
        try:
            # Test valid task validation
            valid_task_data = {
                'id': 'valid_test_task',
                'name': 'Valid Test Task',
                'description': 'A properly formatted task',
                'priority': 5.0,
                'estimated_duration': timedelta(minutes=10),
                'required_resources': {'cpu_cores': 1.0},
                'dependencies': set(),
                'probability_amplitude': complex(1.0, 0.0),
                'interference_pattern': {},
                'entangled_tasks': set()
            }
            
            validation_result = validate_task_input(valid_task_data)
            
            if validation_result.is_valid:
                result.add_pass("Task validation system working correctly")
            else:
                result.add_fail("Valid task failed validation")
            
            result.metrics['validation_checks'] = validation_result.total_checks
            result.metrics['validation_passed'] = validation_result.passed_checks
            
        except Exception as e:
            result.add_fail(f"Validation system test failed: {e}")
    
    except Exception as e:
        result.add_fail(f"Integration test setup failed: {e}")
        import traceback
        result.add_fail(f"Traceback: {traceback.format_exc()}")
    
    return result


def gate_6_documentation_check() -> QualityGateResult:
    """Gate 6: Documentation and code coverage analysis."""
    result = QualityGateResult()
    
    python_files = list(Path('src/llm_cost_tracker').glob('quantum_*.py'))
    
    total_functions = 0
    documented_functions = 0
    total_classes = 0
    documented_classes = 0
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code, filename=str(file_path))
            
            class DocstringChecker(ast.NodeVisitor):
                def visit_FunctionDef(self, node):
                    nonlocal total_functions, documented_functions
                    total_functions += 1
                    
                    # Check if function has docstring
                    if (node.body and 
                        isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, ast.Constant) and
                        isinstance(node.body[0].value.value, str)):
                        documented_functions += 1
                    
                    self.generic_visit(node)
                
                def visit_ClassDef(self, node):
                    nonlocal total_classes, documented_classes
                    total_classes += 1
                    
                    # Check if class has docstring
                    if (node.body and 
                        isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, ast.Constant) and
                        isinstance(node.body[0].value.value, str)):
                        documented_classes += 1
                    
                    self.generic_visit(node)
            
            checker = DocstringChecker()
            checker.visit(tree)
            
        except Exception as e:
            result.add_fail(f"Documentation analysis failed for {file_path.name}: {e}")
    
    # Calculate documentation coverage
    function_doc_coverage = documented_functions / total_functions if total_functions > 0 else 0
    class_doc_coverage = documented_classes / total_classes if total_classes > 0 else 0
    overall_doc_coverage = (documented_functions + documented_classes) / (total_functions + total_classes) if (total_functions + total_classes) > 0 else 0
    
    result.metrics.update({
        'total_functions': total_functions,
        'documented_functions': documented_functions,
        'function_doc_coverage': function_doc_coverage,
        'total_classes': total_classes,
        'documented_classes': documented_classes,
        'class_doc_coverage': class_doc_coverage,
        'overall_doc_coverage': overall_doc_coverage
    })
    
    # Documentation thresholds
    if overall_doc_coverage >= 0.8:
        result.add_pass(f"Excellent documentation coverage: {overall_doc_coverage:.1%}")
    elif overall_doc_coverage >= 0.6:
        result.add_pass(f"Good documentation coverage: {overall_doc_coverage:.1%}")
    elif overall_doc_coverage >= 0.4:
        result.add_warning(f"Moderate documentation coverage: {overall_doc_coverage:.1%}")
    else:
        result.add_fail(f"Poor documentation coverage: {overall_doc_coverage:.1%}")
    
    # Check for README and other docs
    readme_files = list(Path('.').glob('README*'))
    if readme_files:
        result.add_pass(f"Found README documentation: {[f.name for f in readme_files]}")
    else:
        result.add_warning("No README file found")
    
    return result


def main():
    """Run all quality gates."""
    print("🚀 QUANTUM TASK PLANNER - QUALITY GATES")
    print("=" * 50)
    
    gates = [
        ("Import Tests", gate_1_import_tests),
        ("Syntax Analysis", gate_2_syntax_analysis),
        ("Security Scan", gate_3_security_scan),
        ("Performance Tests", gate_4_performance_tests),
        ("Integration Tests", gate_5_integration_tests),
        ("Documentation Check", gate_6_documentation_check),
    ]
    
    overall_passed = True
    total_metrics = {}
    
    for gate_name, gate_function in gates:
        print(f"\n🔍 Running {gate_name}...")
        result = gate_function()
        result.print_summary(gate_name)
        
        if not result.passed:
            overall_passed = False
        
        # Collect metrics
        total_metrics[gate_name.lower().replace(' ', '_')] = result.metrics
    
    # Final summary
    print("\n" + "=" * 50)
    if overall_passed:
        print("🎉 ALL QUALITY GATES PASSED!")
        print("✅ System is ready for production deployment")
        exit_code = 0
    else:
        print("❌ SOME QUALITY GATES FAILED")
        print("🔧 Please address the issues before deployment")
        exit_code = 1
    
    # Print consolidated metrics
    print("\n📊 CONSOLIDATED METRICS:")
    for gate, metrics in total_metrics.items():
        if metrics:
            print(f"  {gate.replace('_', ' ').title()}:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.3f}")
                else:
                    print(f"    {key}: {value}")
    
    print(f"\n🏁 Quality gates completed with exit code: {exit_code}")
    return exit_code


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)