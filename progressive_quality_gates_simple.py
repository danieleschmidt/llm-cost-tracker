#!/usr/bin/env python3
"""
Progressive Quality Gates - Simple Implementation v1.0
Basic quality validation without external dependencies
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class QualityGateStatus(Enum):
    """Quality gate execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class QualityResult:
    """Simple quality check result"""
    name: str
    status: QualityGateStatus
    score: float
    max_score: float
    details: Dict[str, Any]
    execution_time: float
    
    @property
    def percentage(self) -> float:
        return (self.score / self.max_score * 100) if self.max_score > 0 else 0.0


class SimpleQualityGates:
    """Simple quality gates system for Generation 1 implementation"""
    
    def __init__(self, project_root: str = "/root/repo"):
        self.project_root = Path(project_root)
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup basic logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    async def run_quality_checks(self) -> Dict[str, Any]:
        """Run basic quality checks"""
        start_time = time.time()
        self.logger.info("🚀 Starting Generation 1 Quality Gates")
        
        results = []
        
        # Check 1: Python syntax validation
        results.append(await self._check_syntax())
        
        # Check 2: Basic file structure
        results.append(await self._check_file_structure())
        
        # Check 3: Configuration validation
        results.append(await self._check_configuration())
        
        # Check 4: Import validation
        results.append(await self._check_imports())
        
        # Check 5: Basic documentation
        results.append(await self._check_basic_docs())
        
        # Calculate overall results
        total_score = sum(r.score for r in results)
        max_score = sum(r.max_score for r in results)
        overall_percentage = (total_score / max_score * 100) if max_score > 0 else 0
        
        passed_gates = len([r for r in results if r.status == QualityGateStatus.PASSED])
        total_gates = len(results)
        
        execution_time = time.time() - start_time
        
        report = {
            "execution_id": f"gen1_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "overall_score": overall_percentage,
            "passed_gates": passed_gates,
            "total_gates": total_gates,
            "success_rate": (passed_gates / total_gates * 100) if total_gates > 0 else 0,
            "execution_time": execution_time,
            "results": [asdict(r) for r in results],
            "status": "PASSED" if passed_gates == total_gates else "PARTIAL" if passed_gates > 0 else "FAILED"
        }
        
        # Save report
        report_path = self.project_root / f"quality_report_gen1.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"✅ Generation 1 Quality Gates Complete")
        self.logger.info(f"📊 Overall Score: {overall_percentage:.1f}%")
        self.logger.info(f"🎯 Success Rate: {passed_gates}/{total_gates} gates passed")
        self.logger.info(f"⏱️ Execution Time: {execution_time:.2f}s")
        
        return report
    
    async def _check_syntax(self) -> QualityResult:
        """Check Python syntax validity"""
        start_time = time.time()
        
        python_files = list(self.project_root.rglob("*.py"))
        valid_files = 0
        total_files = len(python_files)
        errors = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                compile(source, str(py_file), 'exec')
                valid_files += 1
            except SyntaxError as e:
                errors.append(f"{py_file}: {e}")
            except UnicodeDecodeError as e:
                errors.append(f"{py_file}: Encoding error")
        
        score = (valid_files / total_files * 10.0) if total_files > 0 else 10.0
        status = QualityGateStatus.PASSED if len(errors) == 0 else QualityGateStatus.FAILED
        
        return QualityResult(
            name="Python Syntax Check",
            status=status,
            score=score,
            max_score=10.0,
            details={
                "total_files": total_files,
                "valid_files": valid_files,
                "errors": errors
            },
            execution_time=time.time() - start_time
        )
    
    async def _check_file_structure(self) -> QualityResult:
        """Check basic project file structure"""
        start_time = time.time()
        
        required_files = [
            "README.md",
            "pyproject.toml", 
            "src/llm_cost_tracker/__init__.py",
            "src/llm_cost_tracker/main.py"
        ]
        
        existing_files = []
        missing_files = []
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                existing_files.append(file_path)
            else:
                missing_files.append(file_path)
        
        score = (len(existing_files) / len(required_files) * 8.0)
        status = QualityGateStatus.PASSED if len(missing_files) == 0 else \
                 QualityGateStatus.WARNING if len(existing_files) > len(required_files) / 2 else \
                 QualityGateStatus.FAILED
        
        return QualityResult(
            name="File Structure Check",
            status=status,
            score=score,
            max_score=8.0,
            details={
                "existing_files": existing_files,
                "missing_files": missing_files,
                "structure_score": len(existing_files) / len(required_files) * 100
            },
            execution_time=time.time() - start_time
        )
    
    async def _check_configuration(self) -> QualityResult:
        """Check configuration files validity"""
        start_time = time.time()
        
        config_files = {
            "pyproject.toml": self.project_root / "pyproject.toml",
            "docker-compose.yml": self.project_root / "docker-compose.yml",
            "Dockerfile": self.project_root / "Dockerfile"
        }
        
        valid_configs = 0
        config_details = {}
        
        for name, path in config_files.items():
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        content = f.read()
                    
                    # Basic validation
                    if name == "pyproject.toml" and "[tool.poetry]" in content:
                        valid_configs += 1
                        config_details[name] = "Valid Poetry configuration"
                    elif name == "docker-compose.yml" and "services:" in content:
                        valid_configs += 1
                        config_details[name] = "Valid Docker Compose"
                    elif name == "Dockerfile" and "FROM" in content:
                        valid_configs += 1
                        config_details[name] = "Valid Dockerfile"
                    else:
                        config_details[name] = "Present but format unclear"
                except Exception as e:
                    config_details[name] = f"Error reading: {e}"
            else:
                config_details[name] = "Missing"
        
        score = (valid_configs / len(config_files) * 7.0)
        status = QualityGateStatus.PASSED if valid_configs >= 2 else \
                 QualityGateStatus.WARNING if valid_configs >= 1 else \
                 QualityGateStatus.FAILED
        
        return QualityResult(
            name="Configuration Check",
            status=status,
            score=score,
            max_score=7.0,
            details=config_details,
            execution_time=time.time() - start_time
        )
    
    async def _check_imports(self) -> QualityResult:
        """Check if main modules can be imported"""
        start_time = time.time()
        
        import_tests = [
            ("src.llm_cost_tracker", "Main package"),
            ("src.llm_cost_tracker.main", "Main application"),
            ("src.llm_cost_tracker.quantum_task_planner", "Quantum planner")
        ]
        
        successful_imports = 0
        import_details = {}
        
        # Add src to Python path temporarily
        original_path = sys.path.copy()
        sys.path.insert(0, str(self.project_root))
        
        try:
            for module_name, description in import_tests:
                try:
                    __import__(module_name.replace('src.', ''))
                    successful_imports += 1
                    import_details[module_name] = "✅ Import successful"
                except ImportError as e:
                    import_details[module_name] = f"❌ Import failed: {e}"
                except Exception as e:
                    import_details[module_name] = f"⚠️ Other error: {e}"
        finally:
            sys.path = original_path
        
        score = (successful_imports / len(import_tests) * 12.0)
        status = QualityGateStatus.PASSED if successful_imports == len(import_tests) else \
                 QualityGateStatus.WARNING if successful_imports > 0 else \
                 QualityGateStatus.FAILED
        
        return QualityResult(
            name="Import Validation",
            status=status,
            score=score,
            max_score=12.0,
            details=import_details,
            execution_time=time.time() - start_time
        )
    
    async def _check_basic_docs(self) -> QualityResult:
        """Check basic documentation presence"""
        start_time = time.time()
        
        doc_files = [
            "README.md",
            "docs/API_REFERENCE.md", 
            "CHANGELOG.md",
            "CONTRIBUTING.md"
        ]
        
        existing_docs = []
        doc_details = {}
        
        for doc_file in doc_files:
            path = self.project_root / doc_file
            if path.exists():
                existing_docs.append(doc_file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    doc_details[doc_file] = f"✅ Present ({len(content)} chars)"
                except:
                    doc_details[doc_file] = "✅ Present (unreadable)"
            else:
                doc_details[doc_file] = "❌ Missing"
        
        score = (len(existing_docs) / len(doc_files) * 5.0)
        status = QualityGateStatus.PASSED if len(existing_docs) >= 3 else \
                 QualityGateStatus.WARNING if len(existing_docs) >= 2 else \
                 QualityGateStatus.FAILED
        
        return QualityResult(
            name="Documentation Check", 
            status=status,
            score=score,
            max_score=5.0,
            details=doc_details,
            execution_time=time.time() - start_time
        )


async def main():
    """Execute Generation 1 Quality Gates"""
    quality_system = SimpleQualityGates()
    report = await quality_system.run_quality_checks()
    
    # Print summary
    print(f"\n🎯 GENERATION 1 QUALITY GATES COMPLETE")
    print(f"📊 Overall Score: {report['overall_score']:.1f}%")
    print(f"✅ Gates Passed: {report['passed_gates']}/{report['total_gates']}")
    print(f"📈 Success Rate: {report['success_rate']:.1f}%") 
    print(f"⏱️ Execution Time: {report['execution_time']:.2f}s")
    print(f"📋 Status: {report['status']}")
    
    return report


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result["status"] == "PASSED" else 1)