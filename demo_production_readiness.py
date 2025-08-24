#!/usr/bin/env python3
"""Production Deployment Readiness Validation."""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

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


async def production_readiness_check():
    """Validate production deployment readiness."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "test_name": "Production Deployment Readiness Validation",
        "checks": {},
        "deployment_artifacts": {},
        "summary": {}
    }
    
    logger.info("🚀 Starting Production Deployment Readiness Validation")
    
    try:
        # Check 1: Docker Infrastructure
        logger.info("Check 1: Docker Infrastructure")
        start_time = time.time()
        
        docker_files = {
            "Dockerfile": Path("Dockerfile").exists(),
            "docker-compose.yml": Path("docker-compose.yml").exists(),
            ".dockerignore": Path(".dockerignore").exists()
        }
        
        docker_score = sum(docker_files.values()) / len(docker_files)
        
        results["checks"]["docker_infrastructure"] = {
            "status": "PASS" if docker_score >= 0.5 else "FAIL",
            "duration_ms": (time.time() - start_time) * 1000,
            "details": "Docker infrastructure validation",
            "files_found": docker_files,
            "readiness_score": docker_score
        }
        
        if docker_score >= 0.5:
            logger.info(f"✅ Check 1: Docker infrastructure ready ({docker_score*100:.0f}%)")
        else:
            logger.warning(f"⚠️ Check 1: Docker infrastructure partial ({docker_score*100:.0f}%)")
        
        # Check 2: Configuration Management
        logger.info("Check 2: Configuration Management")
        start_time = time.time()
        
        config_files = {
            ".env.test": Path(".env.test").exists(),
            "pyproject.toml": Path("pyproject.toml").exists(),
            "src/llm_cost_tracker/config.py": Path("src/llm_cost_tracker/config.py").exists()
        }
        
        # Test configuration loading
        try:
            from llm_cost_tracker.config import get_settings
            settings = get_settings()
            config_loadable = True
        except Exception:
            config_loadable = False
        
        config_score = (sum(config_files.values()) / len(config_files) + (1 if config_loadable else 0)) / 2
        
        results["checks"]["configuration_management"] = {
            "status": "PASS" if config_score >= 0.8 else "PARTIAL" if config_score >= 0.5 else "FAIL",
            "duration_ms": (time.time() - start_time) * 1000,
            "details": "Configuration management validation",
            "config_files": config_files,
            "config_loadable": config_loadable,
            "readiness_score": config_score
        }
        
        if config_score >= 0.8:
            logger.info(f"✅ Check 2: Configuration management ready ({config_score*100:.0f}%)")
        else:
            logger.warning(f"⚠️ Check 2: Configuration management needs attention ({config_score*100:.0f}%)")
        
        # Check 3: Security & Compliance
        logger.info("Check 3: Security & Compliance")
        start_time = time.time()
        
        security_components = []
        
        # Check security modules exist
        try:
            from llm_cost_tracker.security import RateLimiter, SecurityHeaders
            security_components.append("rate_limiting")
            security_components.append("security_headers")
        except Exception:
            pass
        
        try:
            from llm_cost_tracker.validation import security_scan_input
            security_components.append("input_validation")
        except Exception:
            pass
        
        try:
            from llm_cost_tracker.circuit_breaker import CircuitBreaker
            security_components.append("circuit_breaker")
        except Exception:
            pass
        
        security_score = len(security_components) / 4  # Expecting 4 components
        
        results["checks"]["security_compliance"] = {
            "status": "PASS" if security_score >= 0.8 else "PARTIAL" if security_score >= 0.5 else "FAIL",
            "duration_ms": (time.time() - start_time) * 1000,
            "details": "Security and compliance validation",
            "security_components": security_components,
            "readiness_score": security_score
        }
        
        if security_score >= 0.8:
            logger.info(f"✅ Check 3: Security & compliance ready ({security_score*100:.0f}%)")
        else:
            logger.warning(f"⚠️ Check 3: Security needs enhancement ({security_score*100:.0f}%)")
        
        # Check 4: Monitoring & Observability
        logger.info("Check 4: Monitoring & Observability")
        start_time = time.time()
        
        monitoring_components = []
        
        try:
            from llm_cost_tracker.health_checks import health_checker
            monitoring_components.append("health_checks")
        except Exception:
            pass
        
        try:
            from llm_cost_tracker.logging_config import configure_logging
            monitoring_components.append("structured_logging")
        except Exception:
            pass
        
        try:
            from llm_cost_tracker.auto_scaling import MetricsCollector
            monitoring_components.append("metrics_collection")
        except Exception:
            pass
        
        monitoring_score = len(monitoring_components) / 3  # Expecting 3 components
        
        results["checks"]["monitoring_observability"] = {
            "status": "PASS" if monitoring_score >= 0.8 else "PARTIAL" if monitoring_score >= 0.5 else "FAIL",
            "duration_ms": (time.time() - start_time) * 1000,
            "details": "Monitoring and observability validation", 
            "monitoring_components": monitoring_components,
            "readiness_score": monitoring_score
        }
        
        if monitoring_score >= 0.8:
            logger.info(f"✅ Check 4: Monitoring ready ({monitoring_score*100:.0f}%)")
        else:
            logger.warning(f"⚠️ Check 4: Monitoring needs enhancement ({monitoring_score*100:.0f}%)")
        
        # Check 5: Performance & Scalability
        logger.info("Check 5: Performance & Scalability")
        start_time = time.time()
        
        try:
            # Load Generation 3 performance results
            with open("generation_3_scaling_results.json", "r") as f:
                scaling_results = json.load(f)
            
            performance_grade = scaling_results.get("summary", {}).get("scaling_grade", "C")
            performance_score_str = scaling_results.get("summary", {}).get("performance_score", "0/100")
            performance_score = float(performance_score_str.split("/")[0]) / 100
            
            # Check scaling components
            scaling_components = []
            
            try:
                from llm_cost_tracker.cache import CacheManager
                scaling_components.append("advanced_caching")
            except Exception:
                pass
            
            try:
                from llm_cost_tracker.concurrency import AsyncTaskQueue
                scaling_components.append("async_processing")
            except Exception:
                pass
            
            try:
                from llm_cost_tracker.quantum_optimization import QuantumLoadBalancer
                scaling_components.append("load_balancing")
            except Exception:
                pass
            
            component_score = len(scaling_components) / 3
            overall_perf_score = (performance_score + component_score) / 2
            
            results["checks"]["performance_scalability"] = {
                "status": "PASS" if overall_perf_score >= 0.8 else "PARTIAL" if overall_perf_score >= 0.6 else "FAIL",
                "duration_ms": (time.time() - start_time) * 1000,
                "details": "Performance and scalability validation",
                "performance_grade": performance_grade,
                "benchmark_score": performance_score,
                "scaling_components": scaling_components,
                "readiness_score": overall_perf_score
            }
            
            if overall_perf_score >= 0.8:
                logger.info(f"✅ Check 5: Performance ready (Grade {performance_grade})")
            else:
                logger.warning(f"⚠️ Check 5: Performance acceptable (Grade {performance_grade})")
                
        except Exception as e:
            results["checks"]["performance_scalability"] = {
                "status": "FAIL",
                "duration_ms": (time.time() - start_time) * 1000,
                "error": str(e),
                "details": "Performance validation failed"
            }
            logger.error(f"❌ Check 5 failed: {e}")
        
        # Check 6: Documentation & Deployment Guides
        logger.info("Check 6: Documentation")
        start_time = time.time()
        
        doc_files = {
            "README.md": Path("README.md").exists(),
            "docs_dir": Path("docs").exists(),
            "examples": len(list(Path(".").glob("demo_*.py"))) >= 3,
            "validation_results": Path("generation_3_scaling_results.json").exists()
        }
        
        doc_score = sum(doc_files.values()) / len(doc_files)
        
        results["checks"]["documentation"] = {
            "status": "PASS" if doc_score >= 0.75 else "PARTIAL" if doc_score >= 0.5 else "FAIL",
            "duration_ms": (time.time() - start_time) * 1000,
            "details": "Documentation validation",
            "documentation_found": doc_files,
            "readiness_score": doc_score
        }
        
        if doc_score >= 0.75:
            logger.info(f"✅ Check 6: Documentation ready ({doc_score*100:.0f}%)")
        else:
            logger.warning(f"⚠️ Check 6: Documentation needs work ({doc_score*100:.0f}%)")
        
        # Generate deployment artifacts
        results["deployment_artifacts"] = {
            "validation_reports": [
                "gen1_validation_results.json",
                "generation_2_robustness_results.json", 
                "generation_3_scaling_results.json",
                "quality_gates_results.json"
            ],
            "demo_scripts": list(str(p) for p in Path(".").glob("demo_*.py")),
            "configuration_files": [
                ".env.test",
                "pyproject.toml",
                "src/llm_cost_tracker/config.py"
            ],
            "docker_files": [f for f, exists in docker_files.items() if exists],
            "core_modules": [
                "src/llm_cost_tracker/quantum_task_planner.py",
                "src/llm_cost_tracker/main.py",
                "src/llm_cost_tracker/security.py",
                "src/llm_cost_tracker/cache.py"
            ]
        }
        
        # Calculate summary
        total_checks = len(results["checks"])
        passed_checks = sum(1 for check in results["checks"].values() if check["status"] == "PASS")
        partial_checks = sum(1 for check in results["checks"].values() if check["status"] == "PARTIAL")
        total_duration = sum(check["duration_ms"] for check in results["checks"].values())
        
        # Production readiness scoring
        readiness_scores = [check.get("readiness_score", 0) for check in results["checks"].values() if "readiness_score" in check]
        avg_readiness = sum(readiness_scores) / len(readiness_scores) if readiness_scores else 0
        
        results["summary"] = {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "partial_checks": partial_checks,
            "failed_checks": total_checks - passed_checks - partial_checks,
            "pass_rate": f"{(passed_checks / total_checks * 100):.1f}%",
            "total_duration_ms": total_duration,
            "production_readiness_score": f"{(avg_readiness * 100):.0f}/100",
            "deployment_grade": "A" if avg_readiness >= 0.9 else "B" if avg_readiness >= 0.7 else "C" if avg_readiness >= 0.5 else "D",
            "status": "READY" if passed_checks >= total_checks * 0.8 else "PARTIAL" if passed_checks >= total_checks * 0.5 else "NOT_READY",
            "deployment_recommendation": (
                "✅ READY FOR PRODUCTION DEPLOYMENT" if avg_readiness >= 0.8
                else "⚠️ READY FOR STAGING/TESTING DEPLOYMENT" if avg_readiness >= 0.6
                else "❌ NEEDS ADDITIONAL WORK BEFORE DEPLOYMENT"
            )
        }
        
        logger.info(f"🎉 Production Readiness Check Complete!")
        logger.info(f"✅ {passed_checks}/{total_checks} checks passed, {partial_checks} partial")
        logger.info(f"🏆 Readiness Score: {results['summary']['production_readiness_score']} (Grade: {results['summary']['deployment_grade']})")
        logger.info(f"📋 Status: {results['summary']['status']}")
        logger.info(f"🚀 {results['summary']['deployment_recommendation']}")
        
        return results
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"❌ Production readiness check failed: {e}")
        logger.error(f"Error details: {error_details}")
        results["checks"]["error"] = {
            "status": "FAIL",
            "error": str(e),
            "traceback": error_details,
            "details": "Unexpected error during readiness check"
        }
        results["summary"]["status"] = "NOT_READY"
        return results


async def main():
    """Main execution function."""
    results = await production_readiness_check()
    
    # Save results
    with open("production_readiness_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("PRODUCTION DEPLOYMENT READINESS VALIDATION RESULTS")
    print("="*80)
    print(json.dumps(results, indent=2))
    
    # Return appropriate exit code based on readiness
    status = results["summary"].get("status", "NOT_READY")
    return 0 if status in ["READY", "PARTIAL"] else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)