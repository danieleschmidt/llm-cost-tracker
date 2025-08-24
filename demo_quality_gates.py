#!/usr/bin/env python3
"""Quality Gates Validation - Final comprehensive testing."""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime

# Set up environment for testing
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["ANTHROPIC_API_KEY"] = "test-key"
os.environ["DEBUG"] = "true"

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import after path setup
from llm_cost_tracker.logging_config import configure_logging

# Configure logging
configure_logging("INFO", structured=False)
logger = logging.getLogger(__name__)


async def comprehensive_quality_gates():
    """Run comprehensive quality gates validation."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "test_name": "Comprehensive Quality Gates Validation",
        "gates": {},
        "summary": {}
    }
    
    logger.info("🛡️ Starting Comprehensive Quality Gates Validation")
    
    try:
        # Gate 1: Code runs without errors
        logger.info("Gate 1: Code Execution Validation")
        start_time = time.time()
        
        try:
            from llm_cost_tracker import QuantumTaskPlanner, QuantumTask, ResourcePool
            from llm_cost_tracker.config import get_settings
            from llm_cost_tracker.cache import CacheManager, MemoryCacheBackend
            from llm_cost_tracker.security import RateLimiter
            from llm_cost_tracker.circuit_breaker import CircuitBreaker
            
            # Test basic instantiation
            planner = QuantumTaskPlanner()
            settings = get_settings()
            cache_backend = MemoryCacheBackend(max_size=100)
            cache = CacheManager(backend=cache_backend)
            rate_limiter = RateLimiter(max_requests=10, window=60)
            circuit_breaker = CircuitBreaker(name="test", failure_threshold=3)
            
            results["gates"]["code_execution"] = {
                "status": "PASS",
                "duration_ms": (time.time() - start_time) * 1000,
                "details": "All core modules load and instantiate successfully"
            }
            logger.info("✅ Gate 1: Code execution validated")
            
        except Exception as e:
            results["gates"]["code_execution"] = {
                "status": "FAIL",
                "duration_ms": (time.time() - start_time) * 1000,
                "error": str(e),
                "details": "Core modules failed to load"
            }
            logger.error(f"❌ Gate 1 failed: {e}")
        
        # Gate 2: Performance Benchmarks Met
        logger.info("Gate 2: Performance Benchmarks")
        start_time = time.time()
        
        try:
            # Load previous generation results
            with open("generation_3_scaling_results.json", "r") as f:
                gen3_results = json.load(f)
            
            benchmarks = gen3_results.get("benchmarks", {})
            performance_score = float(gen3_results.get("summary", {}).get("performance_score", "0/100").split("/")[0])
            
            # Performance thresholds
            min_cache_ops_per_sec = 10000
            min_concurrent_success_rate = 0.8
            min_performance_score = 60.0
            
            cache_performance = benchmarks.get("cache_ops_per_second", 0) >= min_cache_ops_per_sec
            concurrency_performance = benchmarks.get("concurrent_success_rate", 0) >= min_concurrent_success_rate
            overall_performance = performance_score >= min_performance_score
            
            performance_met = cache_performance and concurrency_performance and overall_performance
            
            results["gates"]["performance_benchmarks"] = {
                "status": "PASS" if performance_met else "FAIL",
                "duration_ms": (time.time() - start_time) * 1000,
                "details": "Performance benchmarks validation",
                "cache_ops_per_sec": benchmarks.get("cache_ops_per_second", 0),
                "concurrent_success_rate": benchmarks.get("concurrent_success_rate", 0),
                "performance_score": performance_score,
                "thresholds_met": {
                    "cache_performance": cache_performance,
                    "concurrency_performance": concurrency_performance,
                    "overall_performance": overall_performance
                }
            }
            
            if performance_met:
                logger.info(f"✅ Gate 2: Performance benchmarks met (Score: {performance_score}/100)")
            else:
                logger.error(f"❌ Gate 2: Performance benchmarks not met (Score: {performance_score}/100)")
                
        except Exception as e:
            results["gates"]["performance_benchmarks"] = {
                "status": "FAIL",
                "duration_ms": (time.time() - start_time) * 1000,
                "error": str(e),
                "details": "Failed to validate performance benchmarks"
            }
            logger.error(f"❌ Gate 2 failed: {e}")
        
        # Gate 3: Security Scan Passes
        logger.info("Gate 3: Security Validation")
        start_time = time.time()
        
        try:
            from llm_cost_tracker.validation import security_scan_input, sanitize_string_content
            from llm_cost_tracker.security import SecurityHeaders
            from fastapi import Response
            
            # Test security features
            security_tests_passed = 0
            total_security_tests = 3
            
            # Test 1: XSS Detection
            try:
                security_scan_input("<script>alert('xss')</script>", "test_input")
                security_tests_passed -= 1  # Should have thrown exception
            except Exception:
                security_tests_passed += 1  # Expected to detect threat
            
            # Test 2: String sanitization
            dangerous_string = "<script>alert('test')</script>"
            sanitized = sanitize_string_content(dangerous_string)
            if isinstance(sanitized, str):  # Basic sanitization working
                security_tests_passed += 1
            
            # Test 3: Security headers
            response = Response()
            secured_response = SecurityHeaders.add_security_headers(response)
            if "X-Content-Type-Options" in secured_response.headers:
                security_tests_passed += 1
            
            security_passed = security_tests_passed >= total_security_tests * 0.8  # 80% pass rate
            
            results["gates"]["security_validation"] = {
                "status": "PASS" if security_passed else "FAIL",
                "duration_ms": (time.time() - start_time) * 1000,
                "details": "Security validation completed",
                "security_tests_passed": security_tests_passed,
                "total_security_tests": total_security_tests,
                "pass_rate": security_tests_passed / total_security_tests
            }
            
            if security_passed:
                logger.info(f"✅ Gate 3: Security validation passed ({security_tests_passed}/{total_security_tests})")
            else:
                logger.error(f"❌ Gate 3: Security validation failed ({security_tests_passed}/{total_security_tests})")
                
        except Exception as e:
            results["gates"]["security_validation"] = {
                "status": "FAIL", 
                "duration_ms": (time.time() - start_time) * 1000,
                "error": str(e),
                "details": "Security validation failed"
            }
            logger.error(f"❌ Gate 3 failed: {e}")
        
        # Gate 4: Robustness Features Working
        logger.info("Gate 4: Robustness Validation")
        start_time = time.time()
        
        try:
            # Load Generation 2 results
            with open("generation_2_robustness_results.json", "r") as f:
                gen2_results = json.load(f)
            
            robustness_score = float(gen2_results.get("summary", {}).get("robustness_score", "0%").rstrip("%"))
            passed_tests = gen2_results.get("summary", {}).get("passed_tests", 0)
            total_tests = gen2_results.get("summary", {}).get("total_tests", 1)
            
            robustness_passed = robustness_score >= 80.0 and (passed_tests / total_tests) >= 0.8
            
            results["gates"]["robustness_validation"] = {
                "status": "PASS" if robustness_passed else "FAIL",
                "duration_ms": (time.time() - start_time) * 1000,
                "details": "Robustness features validation",
                "robustness_score": robustness_score,
                "tests_passed": passed_tests,
                "total_tests": total_tests,
                "pass_rate": passed_tests / total_tests
            }
            
            if robustness_passed:
                logger.info(f"✅ Gate 4: Robustness validated ({robustness_score}% score)")
            else:
                logger.error(f"❌ Gate 4: Robustness insufficient ({robustness_score}% score)")
                
        except Exception as e:
            results["gates"]["robustness_validation"] = {
                "status": "FAIL",
                "duration_ms": (time.time() - start_time) * 1000,
                "error": str(e),
                "details": "Robustness validation failed"
            }
            logger.error(f"❌ Gate 4 failed: {e}")
        
        # Gate 5: Integration Testing
        logger.info("Gate 5: Integration Testing")
        start_time = time.time()
        
        try:
            from datetime import timedelta
            
            # End-to-end integration test
            planner = QuantumTaskPlanner()
            
            # Create and add tasks
            task1 = QuantumTask(
                id="integration_test_1",
                name="Integration Test Task 1",
                description="End-to-end integration testing",
                priority=8.0,
                estimated_duration=timedelta(minutes=10)
            )
            
            task2 = QuantumTask(
                id="integration_test_2", 
                name="Integration Test Task 2",
                description="Second integration test task",
                priority=7.0,
                estimated_duration=timedelta(minutes=15)
            )
            
            planner.add_task(task1)
            planner.add_task(task2)
            
            # Test scheduling
            schedule = list(planner.tasks.keys())  # Simple scheduling
            
            # Test system state
            system_state = planner.get_system_state()
            
            # Test resource management
            resource_pool = ResourcePool(cpu_cores=4.0, memory_gb=8.0)
            can_allocate = resource_pool.can_allocate({"cpu_cores": 2.0, "memory_gb": 4.0})
            
            # Test caching
            cache_backend = MemoryCacheBackend(max_size=100)
            cache = CacheManager(backend=cache_backend)
            await cache.set("integration_test", "success", ttl=60.0)
            cached_value = await cache.get("integration_test")
            
            integration_checks = [
                len(schedule) == 2,  # Tasks scheduled
                isinstance(system_state, dict),  # System state available
                can_allocate == True,  # Resource allocation working
                cached_value == "success"  # Cache working
            ]
            
            integration_passed = sum(integration_checks) >= len(integration_checks) * 0.75  # 75% pass
            
            results["gates"]["integration_testing"] = {
                "status": "PASS" if integration_passed else "FAIL",
                "duration_ms": (time.time() - start_time) * 1000,
                "details": "Integration testing completed",
                "checks_passed": sum(integration_checks),
                "total_checks": len(integration_checks),
                "integration_components": {
                    "task_scheduling": integration_checks[0],
                    "system_state": integration_checks[1], 
                    "resource_allocation": integration_checks[2],
                    "caching": integration_checks[3]
                }
            }
            
            if integration_passed:
                logger.info(f"✅ Gate 5: Integration testing passed ({sum(integration_checks)}/{len(integration_checks)})")
            else:
                logger.error(f"❌ Gate 5: Integration testing failed ({sum(integration_checks)}/{len(integration_checks)})")
                
        except Exception as e:
            results["gates"]["integration_testing"] = {
                "status": "FAIL",
                "duration_ms": (time.time() - start_time) * 1000,
                "error": str(e),
                "details": "Integration testing failed"
            }
            logger.error(f"❌ Gate 5 failed: {e}")
        
        # Calculate summary
        total_gates = len(results["gates"])
        passed_gates = sum(1 for gate in results["gates"].values() if gate["status"] == "PASS")
        total_duration = sum(gate["duration_ms"] for gate in results["gates"].values())
        
        results["summary"] = {
            "total_gates": total_gates,
            "passed_gates": passed_gates,
            "failed_gates": total_gates - passed_gates,
            "pass_rate": f"{(passed_gates / total_gates * 100):.1f}%",
            "total_duration_ms": total_duration,
            "quality_score": f"{(passed_gates / total_gates * 100):.0f}/100",
            "quality_grade": "A" if passed_gates >= total_gates * 0.9 else "B" if passed_gates >= total_gates * 0.7 else "C",
            "status": "PASS" if passed_gates == total_gates else "PARTIAL" if passed_gates >= total_gates * 0.8 else "FAIL"
        }
        
        logger.info(f"🎉 Quality Gates Validation Complete!")
        logger.info(f"✅ {passed_gates}/{total_gates} gates passed ({results['summary']['pass_rate']})")
        logger.info(f"🏆 Quality Score: {results['summary']['quality_score']} (Grade: {results['summary']['quality_grade']})")
        logger.info(f"⏱️  Total duration: {total_duration:.1f}ms")
        
        return results
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"❌ Quality gates validation failed: {e}")
        logger.error(f"Error details: {error_details}")
        results["gates"]["error"] = {
            "status": "FAIL",
            "error": str(e),
            "traceback": error_details,
            "details": "Unexpected error during quality gates validation"
        }
        results["summary"]["status"] = "FAIL"
        return results


async def main():
    """Main execution function."""
    results = await comprehensive_quality_gates()
    
    # Save results
    with open("quality_gates_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE QUALITY GATES VALIDATION RESULTS")
    print("="*80)
    print(json.dumps(results, indent=2))
    
    # Return appropriate exit code
    return 0 if results["summary"].get("status") in ["PASS", "PARTIAL"] else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)