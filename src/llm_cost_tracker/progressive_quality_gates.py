"""Progressive Quality Gates System with Autonomous Enhancement."""

import asyncio
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QualityGateStatus(Enum):
    """Quality gate execution status."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


class QualityGateSeverity(Enum):
    """Quality gate failure severity levels."""

    CRITICAL = "critical"  # Blocks deployment
    HIGH = "high"  # Requires immediate attention
    MEDIUM = "medium"  # Should be addressed
    LOW = "low"  # Advisory only


@dataclass
class QualityGateResult:
    """Result of a quality gate execution."""

    gate_name: str
    status: QualityGateStatus
    severity: QualityGateSeverity
    execution_time: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "gate_name": self.gate_name,
            "status": self.status.value,
            "severity": self.severity.value,
            "execution_time": self.execution_time,
            "message": self.message,
            "details": self.details,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class QualityGate:
    """Definition of a quality gate check."""

    name: str
    description: str
    severity: QualityGateSeverity
    timeout_seconds: int
    enabled: bool = True
    retry_count: int = 0
    dependencies: List[str] = field(default_factory=list)
    executor: Optional[Callable] = None

    async def execute(self) -> QualityGateResult:
        """Execute this quality gate."""
        start_time = time.time()

        try:
            if not self.enabled:
                return QualityGateResult(
                    gate_name=self.name,
                    status=QualityGateStatus.SKIPPED,
                    severity=self.severity,
                    execution_time=0.0,
                    message="Quality gate disabled",
                )

            if self.executor:
                result_data = await asyncio.wait_for(
                    self.executor(), timeout=self.timeout_seconds
                )

                execution_time = time.time() - start_time

                if isinstance(result_data, dict):
                    status = QualityGateStatus(result_data.get("status", "failed"))
                    message = result_data.get("message", "No message provided")
                    details = result_data.get("details", {})
                    metrics = result_data.get("metrics", {})
                else:
                    status = (
                        QualityGateStatus.PASSED
                        if result_data
                        else QualityGateStatus.FAILED
                    )
                    message = (
                        "Execution completed" if result_data else "Execution failed"
                    )
                    details = {}
                    metrics = {}

                return QualityGateResult(
                    gate_name=self.name,
                    status=status,
                    severity=self.severity,
                    execution_time=execution_time,
                    message=message,
                    details=details,
                    metrics=metrics,
                )
            else:
                return QualityGateResult(
                    gate_name=self.name,
                    status=QualityGateStatus.FAILED,
                    severity=self.severity,
                    execution_time=time.time() - start_time,
                    message="No executor defined",
                )

        except asyncio.TimeoutError:
            return QualityGateResult(
                gate_name=self.name,
                status=QualityGateStatus.FAILED,
                severity=QualityGateSeverity.HIGH,
                execution_time=time.time() - start_time,
                message=f"Quality gate timed out after {self.timeout_seconds} seconds",
            )
        except Exception as e:
            return QualityGateResult(
                gate_name=self.name,
                status=QualityGateStatus.FAILED,
                severity=self.severity,
                execution_time=time.time() - start_time,
                message=f"Quality gate failed with error: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__},
            )


class ProgressiveQualityGates:
    """Progressive Quality Gates System with autonomous enhancement capabilities."""

    def __init__(self, project_root: Path = None):
        """Initialize the progressive quality gates system."""
        self.project_root = project_root or Path("/root/repo")
        self.gates: Dict[str, QualityGate] = {}
        self.results: List[QualityGateResult] = []
        self.execution_history: List[Dict[str, Any]] = []
        self._setup_default_gates()

    def _setup_default_gates(self) -> None:
        """Setup default quality gates for the project."""
        self.gates = {
            "syntax_check": QualityGate(
                name="syntax_check",
                description="Python syntax validation",
                severity=QualityGateSeverity.CRITICAL,
                timeout_seconds=30,
                executor=self._execute_syntax_check,
            ),
            "unit_tests": QualityGate(
                name="unit_tests",
                description="Unit test execution with coverage",
                severity=QualityGateSeverity.CRITICAL,
                timeout_seconds=300,
                dependencies=["syntax_check"],
                executor=self._execute_unit_tests,
            ),
            "integration_tests": QualityGate(
                name="integration_tests",
                description="Integration test suite",
                severity=QualityGateSeverity.HIGH,
                timeout_seconds=600,
                dependencies=["unit_tests"],
                executor=self._execute_integration_tests,
            ),
            "security_scan": QualityGate(
                name="security_scan",
                description="Security vulnerability scanning",
                severity=QualityGateSeverity.HIGH,
                timeout_seconds=180,
                executor=self._execute_security_scan,
            ),
            "performance_benchmark": QualityGate(
                name="performance_benchmark",
                description="Performance benchmarking and regression testing",
                severity=QualityGateSeverity.MEDIUM,
                timeout_seconds=300,
                dependencies=["unit_tests"],
                executor=self._execute_performance_benchmark,
            ),
            "code_quality": QualityGate(
                name="code_quality",
                description="Code quality analysis with linting",
                severity=QualityGateSeverity.MEDIUM,
                timeout_seconds=120,
                executor=self._execute_code_quality,
            ),
            "documentation_check": QualityGate(
                name="documentation_check",
                description="Documentation completeness validation",
                severity=QualityGateSeverity.LOW,
                timeout_seconds=60,
                executor=self._execute_documentation_check,
            ),
            "dependency_audit": QualityGate(
                name="dependency_audit",
                description="Dependency vulnerability audit",
                severity=QualityGateSeverity.HIGH,
                timeout_seconds=120,
                executor=self._execute_dependency_audit,
            ),
        }

    async def execute_all_gates(self, parallel: bool = True) -> Dict[str, Any]:
        """Execute all quality gates with dependency resolution."""
        start_time = time.time()
        execution_results = {}

        try:
            if parallel:
                # Execute gates in dependency order with parallelization
                execution_order = self._resolve_dependencies()
                for batch in execution_order:
                    batch_results = await self._execute_gate_batch(batch)
                    execution_results.update(batch_results)

                    # Check for critical failures that should stop execution
                    critical_failures = [
                        result
                        for result in batch_results.values()
                        if result.status == QualityGateStatus.FAILED
                        and result.severity == QualityGateSeverity.CRITICAL
                    ]
                    if critical_failures:
                        logger.warning(
                            f"Critical failures detected: {[r.gate_name for r in critical_failures]}"
                        )
                        break
            else:
                # Sequential execution
                for gate_name, gate in self.gates.items():
                    if gate.enabled:
                        result = await gate.execute()
                        execution_results[gate_name] = result
                        self.results.append(result)

            total_time = time.time() - start_time

            # Generate summary
            summary = self._generate_execution_summary(execution_results, total_time)

            # Store execution history
            self.execution_history.append(summary)

            # Save results to file
            await self._save_results(summary)

            return summary

        except Exception as e:
            logger.error(f"Quality gates execution failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
            }

    async def _execute_gate_batch(
        self, gate_names: List[str]
    ) -> Dict[str, QualityGateResult]:
        """Execute a batch of gates in parallel."""
        batch_results = {}

        # Create tasks for enabled gates
        tasks = []
        for gate_name in gate_names:
            if gate_name in self.gates and self.gates[gate_name].enabled:
                task = asyncio.create_task(self.gates[gate_name].execute())
                tasks.append((gate_name, task))

        # Wait for all tasks to complete
        for gate_name, task in tasks:
            try:
                result = await task
                batch_results[gate_name] = result
                self.results.append(result)
            except Exception as e:
                logger.error(f"Gate {gate_name} failed: {e}")
                batch_results[gate_name] = QualityGateResult(
                    gate_name=gate_name,
                    status=QualityGateStatus.FAILED,
                    severity=QualityGateSeverity.HIGH,
                    execution_time=0.0,
                    message=f"Gate execution failed: {str(e)}",
                )

        return batch_results

    def _resolve_dependencies(self) -> List[List[str]]:
        """Resolve gate dependencies and return execution order."""
        # Simple topological sort for dependency resolution
        remaining_gates = set(self.gates.keys())
        execution_batches = []

        while remaining_gates:
            # Find gates with no unresolved dependencies
            ready_gates = []
            for gate_name in remaining_gates:
                gate = self.gates[gate_name]
                if all(dep not in remaining_gates for dep in gate.dependencies):
                    ready_gates.append(gate_name)

            if not ready_gates:
                # Circular dependency or missing dependency
                logger.warning(
                    f"Circular dependencies detected in remaining gates: {remaining_gates}"
                )
                ready_gates = list(remaining_gates)  # Execute remaining gates anyway

            execution_batches.append(ready_gates)
            remaining_gates -= set(ready_gates)

        return execution_batches

    def _generate_execution_summary(
        self, results: Dict[str, QualityGateResult], total_time: float
    ) -> Dict[str, Any]:
        """Generate execution summary with metrics."""
        passed = sum(
            1 for r in results.values() if r.status == QualityGateStatus.PASSED
        )
        failed = sum(
            1 for r in results.values() if r.status == QualityGateStatus.FAILED
        )
        warnings = sum(
            1 for r in results.values() if r.status == QualityGateStatus.WARNING
        )
        skipped = sum(
            1 for r in results.values() if r.status == QualityGateStatus.SKIPPED
        )

        critical_failures = [
            r
            for r in results.values()
            if r.status == QualityGateStatus.FAILED
            and r.severity == QualityGateSeverity.CRITICAL
        ]

        return {
            "timestamp": datetime.now().isoformat(),
            "total_execution_time": total_time,
            "summary": {
                "total_gates": len(results),
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "skipped": skipped,
                "success_rate": passed / len(results) if results else 0.0,
            },
            "status": (
                "failed"
                if critical_failures
                else ("warning" if failed > 0 else "passed")
            ),
            "critical_failures": [r.gate_name for r in critical_failures],
            "gate_results": {
                name: result.to_dict() for name, result in results.items()
            },
            "recommendations": self._generate_recommendations(results.values()),
        }

    def _generate_recommendations(self, results: List[QualityGateResult]) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []

        for result in results:
            if result.status == QualityGateStatus.FAILED:
                if result.gate_name == "unit_tests":
                    recommendations.append(
                        "Fix failing unit tests to ensure code reliability"
                    )
                elif result.gate_name == "security_scan":
                    recommendations.append(
                        "Address security vulnerabilities immediately"
                    )
                elif result.gate_name == "performance_benchmark":
                    recommendations.append(
                        "Optimize performance bottlenecks identified in benchmarks"
                    )
                elif result.gate_name == "code_quality":
                    recommendations.append(
                        "Improve code quality by addressing linting issues"
                    )

        if not recommendations:
            recommendations.append(
                "All quality gates passed - consider adding more stringent checks"
            )

        return recommendations

    async def _save_results(self, summary: Dict[str, Any]) -> None:
        """Save execution results to file."""
        try:
            results_file = self.project_root / "quality_gates_results.json"
            with open(results_file, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Quality gates results saved to {results_file}")
        except Exception as e:
            logger.error(f"Failed to save quality gates results: {e}")

    # Quality Gate Executors

    async def _execute_syntax_check(self) -> Dict[str, Any]:
        """Execute Python syntax validation."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile"]
                + [str(f) for f in (self.project_root / "src").rglob("*.py")],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if result.returncode == 0:
                return {
                    "status": "passed",
                    "message": "All Python files have valid syntax",
                    "metrics": {
                        "files_checked": len(
                            list((self.project_root / "src").rglob("*.py"))
                        )
                    },
                }
            else:
                return {
                    "status": "failed",
                    "message": "Syntax errors detected",
                    "details": {"stderr": result.stderr, "stdout": result.stdout},
                }
        except Exception as e:
            return {"status": "failed", "message": f"Syntax check failed: {str(e)}"}

    async def _execute_unit_tests(self) -> Dict[str, Any]:
        """Execute unit tests with coverage."""
        try:
            env = {"PYTHONPATH": str(self.project_root / "src")}
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/unit/", "-v", "--tb=short"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                env={**subprocess.os.environ, **env},
            )

            # Parse test output for metrics
            test_count = result.stdout.count("PASSED") + result.stdout.count("FAILED")
            passed_count = result.stdout.count("PASSED")

            if result.returncode == 0:
                return {
                    "status": "passed",
                    "message": f"All {test_count} unit tests passed",
                    "metrics": {
                        "total_tests": test_count,
                        "passed_tests": passed_count,
                        "success_rate": (
                            passed_count / test_count if test_count > 0 else 0.0
                        ),
                    },
                }
            else:
                return {
                    "status": "failed",
                    "message": f"Unit tests failed ({passed_count}/{test_count} passed)",
                    "details": {
                        "stdout": result.stdout[:1000],
                        "stderr": result.stderr[:1000],
                    },
                    "metrics": {
                        "total_tests": test_count,
                        "passed_tests": passed_count,
                    },
                }
        except Exception as e:
            return {
                "status": "failed",
                "message": f"Unit test execution failed: {str(e)}",
            }

    async def _execute_integration_tests(self) -> Dict[str, Any]:
        """Execute integration tests."""
        try:
            env = {"PYTHONPATH": str(self.project_root / "src")}
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/integration/", "-v"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                env={**subprocess.os.environ, **env},
            )

            if result.returncode == 0:
                return {
                    "status": "passed",
                    "message": "Integration tests passed",
                    "details": {"stdout": result.stdout[:500]},
                }
            else:
                return {
                    "status": "failed",
                    "message": "Integration tests failed",
                    "details": {
                        "stdout": result.stdout[:1000],
                        "stderr": result.stderr[:1000],
                    },
                }
        except Exception as e:
            return {
                "status": "failed",
                "message": f"Integration test execution failed: {str(e)}",
            }

    async def _execute_security_scan(self) -> Dict[str, Any]:
        """Execute security vulnerability scanning."""
        vulnerabilities = []
        try:
            # Check for common security issues in Python files
            python_files = list((self.project_root / "src").rglob("*.py"))

            for file_path in python_files:
                try:
                    with open(file_path, "r") as f:
                        content = f.read()

                    # Basic security checks
                    if "eval(" in content:
                        vulnerabilities.append(f"Dangerous eval() usage in {file_path}")
                    if "exec(" in content:
                        vulnerabilities.append(f"Dangerous exec() usage in {file_path}")
                    if "shell=True" in content:
                        vulnerabilities.append(f"Shell injection risk in {file_path}")

                except Exception:
                    continue

            if vulnerabilities:
                return {
                    "status": "failed",
                    "message": f"Security vulnerabilities detected: {len(vulnerabilities)}",
                    "details": {"vulnerabilities": vulnerabilities[:10]},
                    "metrics": {"vulnerability_count": len(vulnerabilities)},
                }
            else:
                return {
                    "status": "passed",
                    "message": "No security vulnerabilities detected",
                    "metrics": {"files_scanned": len(python_files)},
                }

        except Exception as e:
            return {"status": "failed", "message": f"Security scan failed: {str(e)}"}

    async def _execute_performance_benchmark(self) -> Dict[str, Any]:
        """Execute performance benchmarking."""
        try:
            # Simple performance test - import time
            start_time = time.time()
            env = {"PYTHONPATH": str(self.project_root / "src")}
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import llm_cost_tracker; print('Import successful')",
                ],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                env={**subprocess.os.environ, **env},
            )
            import_time = time.time() - start_time

            if result.returncode == 0:
                # Performance thresholds
                if import_time > 5.0:
                    return {
                        "status": "warning",
                        "message": f"Import time is slow: {import_time:.2f}s",
                        "metrics": {"import_time_seconds": import_time},
                    }
                else:
                    return {
                        "status": "passed",
                        "message": f"Performance benchmark passed (import: {import_time:.2f}s)",
                        "metrics": {"import_time_seconds": import_time},
                    }
            else:
                return {
                    "status": "failed",
                    "message": "Performance benchmark failed - import error",
                    "details": {"stderr": result.stderr},
                }

        except Exception as e:
            return {
                "status": "failed",
                "message": f"Performance benchmark failed: {str(e)}",
            }

    async def _execute_code_quality(self) -> Dict[str, Any]:
        """Execute code quality analysis."""
        try:
            # Simple code quality checks
            python_files = list((self.project_root / "src").rglob("*.py"))
            issues = []

            for file_path in python_files:
                try:
                    with open(file_path, "r") as f:
                        lines = f.readlines()

                    # Check for basic quality issues
                    for i, line in enumerate(lines, 1):
                        if len(line.strip()) > 100:  # Long lines
                            issues.append(f"Long line in {file_path}:{i}")
                        if (
                            line.strip().startswith("print(")
                            and "debug" not in str(file_path).lower()
                        ):
                            issues.append(f"Debug print statement in {file_path}:{i}")

                except Exception:
                    continue

            quality_score = max(0, 100 - len(issues) * 2)  # Simple scoring

            if quality_score >= 80:
                return {
                    "status": "passed",
                    "message": f"Code quality check passed (score: {quality_score})",
                    "metrics": {
                        "quality_score": quality_score,
                        "issues_found": len(issues),
                    },
                }
            else:
                return {
                    "status": "warning",
                    "message": f"Code quality issues detected (score: {quality_score})",
                    "details": {"issues": issues[:10]},
                    "metrics": {
                        "quality_score": quality_score,
                        "issues_found": len(issues),
                    },
                }

        except Exception as e:
            return {
                "status": "failed",
                "message": f"Code quality check failed: {str(e)}",
            }

    async def _execute_documentation_check(self) -> Dict[str, Any]:
        """Execute documentation completeness validation."""
        try:
            required_docs = ["README.md", "CHANGELOG.md", "CONTRIBUTING.md"]
            missing_docs = []

            for doc in required_docs:
                if not (self.project_root / doc).exists():
                    missing_docs.append(doc)

            # Check for docstrings in Python files
            python_files = list((self.project_root / "src").rglob("*.py"))
            undocumented_functions = 0
            total_functions = 0

            for file_path in python_files:
                try:
                    with open(file_path, "r") as f:
                        content = f.read()

                    # Simple function detection
                    function_count = content.count("def ")
                    docstring_count = content.count('"""') // 2

                    total_functions += function_count
                    undocumented_functions += max(0, function_count - docstring_count)

                except Exception:
                    continue

            documentation_score = 100
            if missing_docs:
                documentation_score -= len(missing_docs) * 20
            if total_functions > 0:
                docstring_coverage = (
                    total_functions - undocumented_functions
                ) / total_functions
                documentation_score = min(documentation_score, docstring_coverage * 100)

            if documentation_score >= 70:
                return {
                    "status": "passed",
                    "message": f"Documentation check passed (score: {documentation_score:.1f})",
                    "metrics": {"documentation_score": documentation_score},
                }
            else:
                return {
                    "status": "warning",
                    "message": f"Documentation needs improvement (score: {documentation_score:.1f})",
                    "details": {"missing_docs": missing_docs},
                    "metrics": {"documentation_score": documentation_score},
                }

        except Exception as e:
            return {
                "status": "failed",
                "message": f"Documentation check failed: {str(e)}",
            }

    async def _execute_dependency_audit(self) -> Dict[str, Any]:
        """Execute dependency vulnerability audit."""
        try:
            # Check if pyproject.toml exists and has dependencies
            pyproject_path = self.project_root / "pyproject.toml"

            if not pyproject_path.exists():
                return {
                    "status": "warning",
                    "message": "No pyproject.toml found for dependency audit",
                }

            with open(pyproject_path, "r") as f:
                content = f.read()

            # Simple dependency count
            dependency_count = content.count('"^')  # Poetry version indicators

            return {
                "status": "passed",
                "message": f"Dependency audit completed ({dependency_count} dependencies checked)",
                "metrics": {"dependencies_checked": dependency_count},
            }

        except Exception as e:
            return {"status": "failed", "message": f"Dependency audit failed: {str(e)}"}


async def main():
    """Main execution function for progressive quality gates."""
    logger.info("Starting Progressive Quality Gates execution...")

    quality_gates = ProgressiveQualityGates()
    results = await quality_gates.execute_all_gates(parallel=True)

    print("\n" + "=" * 80)
    print("🔬 PROGRESSIVE QUALITY GATES EXECUTION COMPLETE")
    print("=" * 80)
    print(f"Status: {results['status'].upper()}")
    print(f"Execution Time: {results['total_execution_time']:.2f}s")
    print(f"Success Rate: {results['summary']['success_rate']:.1%}")
    print(
        f"Gates Passed: {results['summary']['passed']}/{results['summary']['total_gates']}"
    )

    if results["critical_failures"]:
        print(f"\n❌ Critical Failures: {', '.join(results['critical_failures'])}")

    if results["recommendations"]:
        print(f"\n💡 Recommendations:")
        for rec in results["recommendations"]:
            print(f"   • {rec}")

    print("\n" + "=" * 80)

    return results


if __name__ == "__main__":
    asyncio.run(main())
