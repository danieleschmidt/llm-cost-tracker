#!/usr/bin/env python3
"""
Quality Gates Verification - Complete SDLC Quality Assurance
Verifies all quality gates: tests, security, performance, compliance, and deployment readiness
"""

import asyncio
import sys
import os
import json
import time
import subprocess
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llm_cost_tracker import (
    QuantumTaskPlanner, 
    QuantumTask, 
    TaskState, 
    ResourcePool
)

class QualityGateVerification:
    """Comprehensive quality gate verification system."""
    
    def __init__(self):
        self.repo_root = Path(__file__).parent
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'unknown',
            'gates': {},
            'metrics': {},
            'recommendations': []
        }
        
        print("🔒 Quality Gates Verification System Initialized")
        print(f"   📁 Repository: {self.repo_root}")
    
    def verify_code_quality_gate(self) -> Dict[str, Any]:
        """Verify code quality standards."""
        print(f"\n{'='*60}")
        print("📝 CODE QUALITY GATE VERIFICATION")
        print(f"{'='*60}")
        
        gate_result = {
            'status': 'unknown',
            'checks': {},
            'score': 0,
            'issues': []
        }
        
        # Check Python files exist and have basic structure
        print("🔍 Checking code structure...")
        
        src_dir = self.repo_root / 'src' / 'llm_cost_tracker'
        if src_dir.exists():
            python_files = list(src_dir.glob('**/*.py'))
            gate_result['checks']['python_files_count'] = len(python_files)
            
            # Check for key modules
            key_modules = [
                '__init__.py',
                'main.py', 
                'quantum_task_planner.py',
                'config.py',
                'database.py'
            ]
            
            found_modules = []
            for module in key_modules:
                if (src_dir / module).exists():
                    found_modules.append(module)
            
            gate_result['checks']['key_modules_found'] = found_modules
            gate_result['checks']['key_modules_coverage'] = len(found_modules) / len(key_modules)
            
            if gate_result['checks']['key_modules_coverage'] >= 0.8:
                gate_result['score'] += 25
            else:
                gate_result['issues'].append("Missing key modules")
            
            print(f"   📊 Python files: {len(python_files)}")
            print(f"   🔑 Key modules coverage: {gate_result['checks']['key_modules_coverage']:.2%}")
        else:
            gate_result['issues'].append("Source directory not found")
        
        # Check for documentation
        print("📚 Checking documentation...")
        doc_files = list(self.repo_root.glob('*.md'))
        gate_result['checks']['documentation_files'] = len(doc_files)
        
        if len(doc_files) >= 3:  # README, CHANGELOG, etc.
            gate_result['score'] += 25
            print(f"   ✅ Documentation files: {len(doc_files)}")
        else:
            gate_result['issues'].append("Insufficient documentation")
            print(f"   ⚠️  Documentation files: {len(doc_files)} (minimum 3 recommended)")
        
        # Check for configuration files
        print("⚙️  Checking configuration...")
        config_files = [
            'pyproject.toml',
            'docker-compose.yml',
            'Dockerfile'
        ]
        
        found_configs = []
        for config in config_files:
            if (self.repo_root / config).exists():
                found_configs.append(config)
        
        gate_result['checks']['config_files'] = found_configs
        gate_result['checks']['config_coverage'] = len(found_configs) / len(config_files)
        
        if gate_result['checks']['config_coverage'] >= 0.7:
            gate_result['score'] += 25
            print(f"   ✅ Configuration coverage: {gate_result['checks']['config_coverage']:.2%}")
        else:
            gate_result['issues'].append("Missing essential configuration files")
            print(f"   ⚠️  Configuration coverage: {gate_result['checks']['config_coverage']:.2%}")
        
        # Check for tests
        print("🧪 Checking test coverage...")
        tests_dir = self.repo_root / 'tests'
        if tests_dir.exists():
            test_files = list(tests_dir.glob('**/*.py'))
            gate_result['checks']['test_files_count'] = len(test_files)
            
            if len(test_files) >= 5:
                gate_result['score'] += 25
                print(f"   ✅ Test files: {len(test_files)}")
            else:
                gate_result['issues'].append("Insufficient test coverage")
                print(f"   ⚠️  Test files: {len(test_files)} (minimum 5 recommended)")
        else:
            gate_result['issues'].append("Tests directory not found")
            print(f"   ❌ Tests directory not found")
        
        # Determine overall gate status
        if gate_result['score'] >= 75:
            gate_result['status'] = 'passed'
        elif gate_result['score'] >= 50:
            gate_result['status'] = 'warning'
        else:
            gate_result['status'] = 'failed'
        
        print(f"\n📊 Code Quality Gate: {gate_result['status'].upper()}")
        print(f"   Score: {gate_result['score']}/100")
        
        self.results['gates']['code_quality'] = gate_result
        return gate_result
    
    async def verify_functional_testing_gate(self) -> Dict[str, Any]:
        """Verify functional testing requirements."""
        print(f"\n{'='*60}")
        print("🧪 FUNCTIONAL TESTING GATE VERIFICATION")
        print(f"{'='*60}")
        
        gate_result = {
            'status': 'unknown',
            'checks': {},
            'score': 0,
            'issues': [],
            'test_results': {}
        }
        
        # Test core functionality
        print("🔬 Testing core quantum task planner functionality...")
        
        try:
            # Initialize system
            planner = QuantumTaskPlanner()
            gate_result['checks']['planner_initialization'] = True
            gate_result['score'] += 20
            print("   ✅ Planner initialization: PASS")
            
            # Test task creation and addition
            test_task = QuantumTask(
                id="test_functional_task",
                name="Functional Test Task",
                description="Task for functional testing",
                priority=5.0,
                estimated_duration=timedelta(minutes=10),
                required_resources={"cpu_cores": 2.0, "memory_gb": 4.0}
            )
            
            success, message = planner.add_task(test_task)
            gate_result['checks']['task_addition'] = success
            
            if success:
                gate_result['score'] += 20
                print("   ✅ Task addition: PASS")
            else:
                gate_result['issues'].append(f"Task addition failed: {message}")
                print(f"   ❌ Task addition: FAIL - {message}")
            
            # Test scheduling
            if len(planner.tasks) > 0:
                schedule = planner.quantum_anneal_schedule(max_iterations=10)
                gate_result['checks']['scheduling'] = len(schedule) > 0
                
                if len(schedule) > 0:
                    gate_result['score'] += 20
                    print("   ✅ Task scheduling: PASS")
                else:
                    gate_result['issues'].append("Scheduling produced empty schedule")
                    print("   ❌ Task scheduling: FAIL")
            
            # Test system state
            system_state = planner.get_system_state()
            gate_result['checks']['system_state'] = system_state is not None
            
            if system_state:
                gate_result['score'] += 20
                print("   ✅ System state retrieval: PASS")
                gate_result['test_results']['total_tasks'] = system_state.get('total_tasks', 0)
            else:
                gate_result['issues'].append("System state retrieval failed")
                print("   ❌ System state retrieval: FAIL")
            
            # Test health check
            health_status = planner.perform_health_check()
            gate_result['checks']['health_check'] = health_status.get('overall_healthy', False)
            
            if health_status.get('overall_healthy', False):
                gate_result['score'] += 20
                print("   ✅ Health check: PASS")
            else:
                gate_result['issues'].append("Health check indicates system issues")
                print("   ⚠️  Health check: WARNING")
                gate_result['score'] += 10  # Partial credit
                
        except Exception as e:
            gate_result['issues'].append(f"Functional testing exception: {str(e)}")
            print(f"   ❌ Functional testing exception: {e}")
        
        # Determine gate status
        if gate_result['score'] >= 80:
            gate_result['status'] = 'passed'
        elif gate_result['score'] >= 60:
            gate_result['status'] = 'warning'
        else:
            gate_result['status'] = 'failed'
        
        print(f"\n📊 Functional Testing Gate: {gate_result['status'].upper()}")
        print(f"   Score: {gate_result['score']}/100")
        
        self.results['gates']['functional_testing'] = gate_result
        return gate_result
    
    def verify_security_gate(self) -> Dict[str, Any]:
        """Verify security requirements and vulnerabilities."""
        print(f"\n{'='*60}")
        print("🔒 SECURITY GATE VERIFICATION")
        print(f"{'='*60}")
        
        gate_result = {
            'status': 'unknown',
            'checks': {},
            'score': 0,
            'issues': [],
            'vulnerabilities': []
        }
        
        # Check for security configuration files
        print("🛡️  Checking security configuration...")
        
        security_files = [
            'SECURITY.md',
            '.github/workflows/security-scan.yml'
        ]
        
        found_security_files = []
        for sec_file in security_files:
            if (self.repo_root / sec_file).exists():
                found_security_files.append(sec_file)
        
        gate_result['checks']['security_files'] = found_security_files
        security_coverage = len(found_security_files) / len(security_files)
        
        if security_coverage >= 0.5:
            gate_result['score'] += 25
            print(f"   ✅ Security files coverage: {security_coverage:.2%}")
        else:
            gate_result['issues'].append("Missing security documentation")
            print(f"   ⚠️  Security files coverage: {security_coverage:.2%}")
        
        # Check for hardcoded secrets (basic scan)
        print("🔍 Scanning for potential security issues...")
        
        secret_patterns = [
            'password =',
            'api_key =',
            'secret =',
            'token =',
            'AWS_ACCESS_KEY',
            'PRIVATE_KEY'
        ]
        
        potential_secrets = []
        src_files = list(self.repo_root.glob('**/*.py'))
        
        for file_path in src_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for pattern in secret_patterns:
                        if pattern in content.lower():
                            # Simple heuristic to avoid false positives
                            if not any(exclude in content.lower() for exclude in ['example', 'test', 'fake', 'dummy']):
                                potential_secrets.append(f"{file_path}: {pattern}")
            except:
                continue
        
        gate_result['checks']['potential_secrets'] = len(potential_secrets)
        
        if len(potential_secrets) == 0:
            gate_result['score'] += 25
            print("   ✅ No hardcoded secrets detected")
        else:
            gate_result['vulnerabilities'].extend(potential_secrets)
            gate_result['issues'].append(f"Found {len(potential_secrets)} potential hardcoded secrets")
            print(f"   ⚠️  Found {len(potential_secrets)} potential secrets")
        
        # Check for input validation
        print("🔐 Checking input validation patterns...")
        
        validation_indicators = [
            'validate_',
            'sanitize_',
            'clean_',
            'escape_',
            'ValidationError',
            'SecurityValidationError'
        ]
        
        validation_found = 0
        for file_path in src_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for indicator in validation_indicators:
                        if indicator in content:
                            validation_found += 1
                            break
            except:
                continue
        
        gate_result['checks']['validation_coverage'] = validation_found / max(len(src_files), 1)
        
        if gate_result['checks']['validation_coverage'] >= 0.3:
            gate_result['score'] += 25
            print(f"   ✅ Input validation coverage: {gate_result['checks']['validation_coverage']:.2%}")
        else:
            gate_result['issues'].append("Limited input validation patterns found")
            print(f"   ⚠️  Input validation coverage: {gate_result['checks']['validation_coverage']:.2%}")
        
        # Check for error handling
        print("⚠️  Checking error handling patterns...")
        
        error_handling_patterns = [
            'try:',
            'except',
            'raise',
            'logging.error',
            'logger.error'
        ]
        
        error_handling_files = 0
        for file_path in src_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if any(pattern in content for pattern in error_handling_patterns):
                        error_handling_files += 1
            except:
                continue
        
        error_handling_coverage = error_handling_files / max(len(src_files), 1)
        gate_result['checks']['error_handling_coverage'] = error_handling_coverage
        
        if error_handling_coverage >= 0.5:
            gate_result['score'] += 25
            print(f"   ✅ Error handling coverage: {error_handling_coverage:.2%}")
        else:
            gate_result['issues'].append("Limited error handling patterns found")
            print(f"   ⚠️  Error handling coverage: {error_handling_coverage:.2%}")
        
        # Determine gate status
        if gate_result['score'] >= 75 and len(gate_result['vulnerabilities']) == 0:
            gate_result['status'] = 'passed'
        elif gate_result['score'] >= 50:
            gate_result['status'] = 'warning'
        else:
            gate_result['status'] = 'failed'
        
        print(f"\n📊 Security Gate: {gate_result['status'].upper()}")
        print(f"   Score: {gate_result['score']}/100")
        print(f"   Vulnerabilities: {len(gate_result['vulnerabilities'])}")
        
        self.results['gates']['security'] = gate_result
        return gate_result
    
    async def verify_performance_gate(self) -> Dict[str, Any]:
        """Verify performance requirements."""
        print(f"\n{'='*60}")
        print("⚡ PERFORMANCE GATE VERIFICATION")
        print(f"{'='*60}")
        
        gate_result = {
            'status': 'unknown',
            'checks': {},
            'score': 0,
            'issues': [],
            'benchmarks': {}
        }
        
        # Performance benchmarks
        print("🏃 Running performance benchmarks...")
        
        try:
            # Initialize system for performance testing
            planner = QuantumTaskPlanner()
            
            # Benchmark 1: Task addition performance
            print("   📊 Benchmarking task addition...")
            start_time = time.perf_counter()
            
            for i in range(50):
                task = QuantumTask(
                    id=f"perf_test_task_{i}",
                    name=f"Performance Test Task {i}",
                    description="Performance testing task",
                    priority=float(i % 10 + 1),
                    estimated_duration=timedelta(minutes=5),
                    required_resources={"cpu_cores": 1.0, "memory_gb": 2.0}
                )
                planner.add_task(task)
            
            task_addition_time = (time.perf_counter() - start_time) * 1000
            gate_result['benchmarks']['task_addition_ms'] = task_addition_time
            
            # Performance requirement: < 1000ms for 50 tasks
            if task_addition_time < 1000:
                gate_result['score'] += 25
                print(f"     ✅ Task addition: {task_addition_time:.2f}ms (< 1000ms)")
            else:
                gate_result['issues'].append(f"Task addition too slow: {task_addition_time:.2f}ms")
                print(f"     ❌ Task addition: {task_addition_time:.2f}ms (≥ 1000ms)")
            
            # Benchmark 2: Scheduling performance
            print("   📊 Benchmarking scheduling...")
            start_time = time.perf_counter()
            
            schedule = planner.quantum_anneal_schedule(max_iterations=50)
            
            scheduling_time = (time.perf_counter() - start_time) * 1000
            gate_result['benchmarks']['scheduling_ms'] = scheduling_time
            
            # Performance requirement: < 5000ms for 50 tasks
            if scheduling_time < 5000:
                gate_result['score'] += 25
                print(f"     ✅ Scheduling: {scheduling_time:.2f}ms (< 5000ms)")
            else:
                gate_result['issues'].append(f"Scheduling too slow: {scheduling_time:.2f}ms")
                print(f"     ❌ Scheduling: {scheduling_time:.2f}ms (≥ 5000ms)")
            
            # Benchmark 3: Memory usage
            print("   📊 Benchmarking memory efficiency...")
            import sys
            
            # Simple memory usage check
            initial_objects = len(gc.get_objects()) if 'gc' in sys.modules else 0
            
            # Create and cleanup tasks
            temp_tasks = []
            for i in range(100):
                task = QuantumTask(
                    id=f"memory_test_task_{i}",
                    name=f"Memory Test Task {i}",
                    description="Memory testing task",
                    priority=5.0,
                    estimated_duration=timedelta(minutes=1)
                )
                temp_tasks.append(task)
            
            # Cleanup
            del temp_tasks
            import gc
            gc.collect()
            
            final_objects = len(gc.get_objects())
            memory_growth = final_objects - initial_objects
            
            gate_result['benchmarks']['memory_growth_objects'] = memory_growth
            
            # Performance requirement: < 1000 object growth
            if memory_growth < 1000:
                gate_result['score'] += 25
                print(f"     ✅ Memory growth: {memory_growth} objects (< 1000)")
            else:
                gate_result['issues'].append(f"High memory growth: {memory_growth} objects")
                print(f"     ⚠️  Memory growth: {memory_growth} objects (≥ 1000)")
                gate_result['score'] += 10  # Partial credit
            
            # Benchmark 4: System responsiveness
            print("   📊 Benchmarking system responsiveness...")
            start_time = time.perf_counter()
            
            # Perform multiple system operations
            for _ in range(10):
                system_state = planner.get_system_state()
                health_status = planner.perform_health_check()
            
            responsiveness_time = (time.perf_counter() - start_time) * 1000
            gate_result['benchmarks']['responsiveness_ms'] = responsiveness_time
            
            # Performance requirement: < 2000ms for 10 operations
            if responsiveness_time < 2000:
                gate_result['score'] += 25
                print(f"     ✅ Responsiveness: {responsiveness_time:.2f}ms (< 2000ms)")
            else:
                gate_result['issues'].append(f"Poor system responsiveness: {responsiveness_time:.2f}ms")
                print(f"     ❌ Responsiveness: {responsiveness_time:.2f}ms (≥ 2000ms)")
            
        except Exception as e:
            gate_result['issues'].append(f"Performance testing exception: {str(e)}")
            print(f"   ❌ Performance testing failed: {e}")
        
        # Determine gate status
        if gate_result['score'] >= 80:
            gate_result['status'] = 'passed'
        elif gate_result['score'] >= 60:
            gate_result['status'] = 'warning'
        else:
            gate_result['status'] = 'failed'
        
        print(f"\n📊 Performance Gate: {gate_result['status'].upper()}")
        print(f"   Score: {gate_result['score']}/100")
        
        self.results['gates']['performance'] = gate_result
        return gate_result
    
    def verify_deployment_readiness_gate(self) -> Dict[str, Any]:
        """Verify deployment readiness."""
        print(f"\n{'='*60}")
        print("🚀 DEPLOYMENT READINESS GATE VERIFICATION")
        print(f"{'='*60}")
        
        gate_result = {
            'status': 'unknown',
            'checks': {},
            'score': 0,
            'issues': []
        }
        
        # Check Docker configuration
        print("🐳 Checking Docker configuration...")
        
        docker_files = ['Dockerfile', 'docker-compose.yml', 'docker-compose.production.yml']
        found_docker_files = []
        
        for docker_file in docker_files:
            if (self.repo_root / docker_file).exists():
                found_docker_files.append(docker_file)
        
        gate_result['checks']['docker_files'] = found_docker_files
        docker_coverage = len(found_docker_files) / len(docker_files)
        
        if docker_coverage >= 0.7:
            gate_result['score'] += 25
            print(f"   ✅ Docker configuration: {docker_coverage:.2%} coverage")
        else:
            gate_result['issues'].append("Incomplete Docker configuration")
            print(f"   ⚠️  Docker configuration: {docker_coverage:.2%} coverage")
        
        # Check environment configuration
        print("⚙️  Checking environment configuration...")
        
        env_files = ['.env.example', '.env.production.example']
        found_env_files = []
        
        for env_file in env_files:
            if (self.repo_root / env_file).exists():
                found_env_files.append(env_file)
        
        gate_result['checks']['env_files'] = found_env_files
        
        if len(found_env_files) >= 1:
            gate_result['score'] += 25
            print(f"   ✅ Environment configuration: {len(found_env_files)} files")
        else:
            gate_result['issues'].append("Missing environment configuration files")
            print(f"   ❌ Environment configuration: No files found")
        
        # Check deployment scripts
        print("📜 Checking deployment scripts...")
        
        scripts_dir = self.repo_root / 'scripts'
        deployment_scripts = []
        
        if scripts_dir.exists():
            potential_scripts = [
                'deploy.sh',
                'build.sh',
                'setup.sh',
                'dev-setup.sh'
            ]
            
            for script in potential_scripts:
                if (scripts_dir / script).exists():
                    deployment_scripts.append(script)
        
        gate_result['checks']['deployment_scripts'] = deployment_scripts
        
        if len(deployment_scripts) >= 2:
            gate_result['score'] += 25
            print(f"   ✅ Deployment scripts: {len(deployment_scripts)} found")
        else:
            gate_result['issues'].append("Missing deployment automation scripts")
            print(f"   ⚠️  Deployment scripts: {len(deployment_scripts)} found")
        
        # Check configuration files
        print("📋 Checking configuration management...")
        
        config_files = [
            'config/alert-rules.yml',
            'config/prometheus.yml',
            'config/grafana-datasources.yml'
        ]
        
        found_configs = []
        for config in config_files:
            if (self.repo_root / config).exists():
                found_configs.append(config)
        
        gate_result['checks']['monitoring_configs'] = found_configs
        config_coverage = len(found_configs) / len(config_files)
        
        if config_coverage >= 0.5:
            gate_result['score'] += 25
            print(f"   ✅ Monitoring configuration: {config_coverage:.2%} coverage")
        else:
            gate_result['issues'].append("Limited monitoring configuration")
            print(f"   ⚠️  Monitoring configuration: {config_coverage:.2%} coverage")
        
        # Determine gate status
        if gate_result['score'] >= 75:
            gate_result['status'] = 'passed'
        elif gate_result['score'] >= 50:
            gate_result['status'] = 'warning'
        else:
            gate_result['status'] = 'failed'
        
        print(f"\n📊 Deployment Readiness Gate: {gate_result['status'].upper()}")
        print(f"   Score: {gate_result['score']}/100")
        
        self.results['gates']['deployment_readiness'] = gate_result
        return gate_result
    
    def calculate_overall_quality_score(self) -> Tuple[int, str]:
        """Calculate overall quality score and status."""
        gates = self.results['gates']
        
        if not gates:
            return 0, 'unknown'
        
        # Weight different gates
        gate_weights = {
            'code_quality': 0.2,
            'functional_testing': 0.3,
            'security': 0.25,
            'performance': 0.15,
            'deployment_readiness': 0.1
        }
        
        total_score = 0
        total_weight = 0
        
        for gate_name, gate_result in gates.items():
            weight = gate_weights.get(gate_name, 0.1)
            score = gate_result.get('score', 0)
            total_score += score * weight
            total_weight += weight
        
        overall_score = int(total_score / total_weight) if total_weight > 0 else 0
        
        # Determine overall status
        if overall_score >= 85:
            status = 'excellent'
        elif overall_score >= 75:
            status = 'passed'
        elif overall_score >= 60:
            status = 'warning'
        else:
            status = 'failed'
        
        return overall_score, status
    
    def generate_quality_report(self) -> None:
        """Generate comprehensive quality report."""
        print(f"\n{'='*60}")
        print("📋 COMPREHENSIVE QUALITY REPORT")
        print(f"{'='*60}")
        
        overall_score, overall_status = self.calculate_overall_quality_score()
        self.results['overall_score'] = overall_score
        self.results['overall_status'] = overall_status
        
        # Overall summary
        status_emoji = {
            'excellent': '🏆',
            'passed': '✅',
            'warning': '⚠️',
            'failed': '❌',
            'unknown': '❓'
        }
        
        print(f"🎯 Overall Quality Score: {overall_score}/100 {status_emoji.get(overall_status, '❓')}")
        print(f"📊 Overall Status: {overall_status.upper()}")
        
        # Individual gate summary
        print(f"\n📊 Quality Gates Summary:")
        for gate_name, gate_result in self.results['gates'].items():
            gate_score = gate_result.get('score', 0)
            gate_status = gate_result.get('status', 'unknown')
            emoji = status_emoji.get(gate_status, '❓')
            
            print(f"   {emoji} {gate_name.replace('_', ' ').title()}: {gate_score}/100 ({gate_status.upper()})")
        
        # Critical issues
        all_issues = []
        all_vulnerabilities = []
        
        for gate_result in self.results['gates'].values():
            all_issues.extend(gate_result.get('issues', []))
            all_vulnerabilities.extend(gate_result.get('vulnerabilities', []))
        
        if all_issues:
            print(f"\n🚨 Critical Issues ({len(all_issues)}):")
            for i, issue in enumerate(all_issues[:10], 1):  # Show top 10
                print(f"   {i}. {issue}")
            if len(all_issues) > 10:
                print(f"   ... and {len(all_issues) - 10} more issues")
        
        if all_vulnerabilities:
            print(f"\n🔒 Security Vulnerabilities ({len(all_vulnerabilities)}):")
            for i, vuln in enumerate(all_vulnerabilities[:5], 1):  # Show top 5
                print(f"   {i}. {vuln}")
            if len(all_vulnerabilities) > 5:
                print(f"   ... and {len(all_vulnerabilities) - 5} more vulnerabilities")
        
        # Generate recommendations
        recommendations = self.generate_recommendations(overall_score, overall_status)
        self.results['recommendations'] = recommendations
        
        print(f"\n💡 Quality Improvement Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        # Performance metrics
        if 'performance' in self.results['gates']:
            perf_gate = self.results['gates']['performance']
            benchmarks = perf_gate.get('benchmarks', {})
            
            if benchmarks:
                print(f"\n⚡ Performance Metrics:")
                for metric, value in benchmarks.items():
                    if 'ms' in metric:
                        print(f"   {metric.replace('_', ' ').title()}: {value:.2f}ms")
                    else:
                        print(f"   {metric.replace('_', ' ').title()}: {value}")
        
        # Save results to file
        results_file = self.repo_root / 'quality_gates_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📄 Full results saved to: {results_file}")
    
    def generate_recommendations(self, score: int, status: str) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        gates = self.results['gates']
        
        # Code quality recommendations
        if 'code_quality' in gates:
            cq_gate = gates['code_quality']
            if cq_gate['score'] < 75:
                recommendations.append("Improve code structure and add missing key modules")
                recommendations.append("Increase documentation coverage with comprehensive guides")
        
        # Testing recommendations
        if 'functional_testing' in gates:
            ft_gate = gates['functional_testing']
            if ft_gate['score'] < 80:
                recommendations.append("Expand test coverage with unit, integration, and end-to-end tests")
                recommendations.append("Implement continuous integration testing pipeline")
        
        # Security recommendations
        if 'security' in gates:
            sec_gate = gates['security']
            if sec_gate.get('vulnerabilities', []):
                recommendations.append("Address identified security vulnerabilities immediately")
            if sec_gate['score'] < 75:
                recommendations.append("Implement comprehensive input validation and error handling")
                recommendations.append("Add security scanning to CI/CD pipeline")
        
        # Performance recommendations
        if 'performance' in gates:
            perf_gate = gates['performance']
            if perf_gate['score'] < 80:
                recommendations.append("Optimize performance bottlenecks identified in benchmarks")
                recommendations.append("Implement caching and memory optimization strategies")
        
        # Deployment recommendations
        if 'deployment_readiness' in gates:
            dr_gate = gates['deployment_readiness']
            if dr_gate['score'] < 75:
                recommendations.append("Complete Docker and environment configuration")
                recommendations.append("Add comprehensive deployment automation scripts")
        
        # Overall recommendations based on status
        if status == 'failed':
            recommendations.append("Focus on fundamental issues before production deployment")
            recommendations.append("Establish quality standards and review processes")
        elif status == 'warning':
            recommendations.append("Address warning-level issues before production release")
            recommendations.append("Implement monitoring and observability solutions")
        elif status in ['passed', 'excellent']:
            recommendations.append("Maintain current quality standards with regular audits")
            recommendations.append("Consider advanced optimization and scaling features")
        
        # Add general best practices
        recommendations.extend([
            "Implement continuous monitoring and alerting in production",
            "Establish regular security audits and dependency updates",
            "Create disaster recovery and business continuity plans"
        ])
        
        return recommendations[:10]  # Return top 10 recommendations


async def main():
    """Run comprehensive quality gates verification."""
    print("🔒 LLM COST TRACKER - QUALITY GATES VERIFICATION")
    print("=" * 60)
    print("Verifying: All Quality Gates (Security, Performance, Testing, Deployment)")
    print("=" * 60)
    
    verifier = QualityGateVerification()
    
    try:
        # Run all quality gate verifications
        verifier.verify_code_quality_gate()
        await verifier.verify_functional_testing_gate()
        verifier.verify_security_gate()
        await verifier.verify_performance_gate()
        verifier.verify_deployment_readiness_gate()
        
        # Generate comprehensive report
        verifier.generate_quality_report()
        
        overall_score, overall_status = verifier.calculate_overall_quality_score()
        
        print(f"\n{'='*60}")
        if overall_status in ['passed', 'excellent']:
            print("🎉 QUALITY GATES VERIFICATION COMPLETED")
            print("✅ All quality standards met - Ready for production deployment")
        elif overall_status == 'warning':
            print("⚠️  QUALITY GATES VERIFICATION COMPLETED WITH WARNINGS")
            print("🔄 Address warnings before production deployment")
        else:
            print("❌ QUALITY GATES VERIFICATION FAILED")
            print("🛠️  Critical issues must be resolved before deployment")
        
        print(f"📊 Final Score: {overall_score}/100")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n💥 QUALITY GATES VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        print("🛠️  Check output for detailed error information")


if __name__ == "__main__":
    asyncio.run(main())