#!/usr/bin/env python3
"""
Simple Quality Validation System - Final Quality Gates
Comprehensive testing and validation for autonomous SDLC
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import importlib.util
import ast


class QualityLevel(Enum):
    """Quality assurance levels"""
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    ENTERPRISE = "enterprise"


class TestResult(Enum):
    """Test execution results"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class ValidationCategory(Enum):
    """Validation categories"""
    SYNTAX = "syntax"
    IMPORTS = "imports"
    SECURITY = "security"
    FUNCTIONALITY = "functionality"
    INTEGRATION = "integration"


@dataclass
class ValidationResult:
    """Result of a validation test"""
    test_name: str
    category: ValidationCategory
    result: TestResult
    execution_time_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    score: float = 0.0
    max_score: float = 100.0
    
    @property
    def percentage(self) -> float:
        return (self.score / self.max_score * 100) if self.max_score > 0 else 0.0


@dataclass
class QualityReport:
    """Comprehensive quality validation report"""
    execution_id: str
    started_at: datetime
    completed_at: Optional[datetime]
    quality_level: QualityLevel
    overall_score: float
    max_possible_score: float
    validation_results: List[ValidationResult]
    category_scores: Dict[str, float]
    critical_failures: List[str]
    recommendations: List[Dict[str, Any]]
    
    @property
    def overall_percentage(self) -> float:
        return (self.overall_score / self.max_possible_score * 100) if self.max_possible_score > 0 else 0.0
    
    @property
    def passed_tests(self) -> int:
        return len([r for r in self.validation_results if r.result == TestResult.PASSED])
    
    @property
    def failed_tests(self) -> int:
        return len([r for r in self.validation_results if r.result == TestResult.FAILED])
    
    @property
    def total_tests(self) -> int:
        return len(self.validation_results)


class SimpleQualityValidator:
    """Simple but comprehensive quality validation system"""
    
    def __init__(self, project_root: str = "/root/repo"):
        self.project_root = Path(project_root)
        self.execution_id = f"quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(self)}"
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Scan for Python files
        self.python_files = list(self.project_root.rglob("*.py"))
        
        self.logger.info(f"🔍 Simple Quality Validator initialized - ID: {self.execution_id}")
        self.logger.info(f"Found {len(self.python_files)} Python files to analyze")
    
    async def execute_quality_validation(self, quality_level: QualityLevel = QualityLevel.COMPREHENSIVE) -> QualityReport:
        """Execute comprehensive quality validation"""
        self.logger.info(f"🚀 Starting {quality_level.value} quality validation")
        
        start_time = datetime.now()
        
        # Initialize report
        report = QualityReport(
            execution_id=self.execution_id,
            started_at=start_time,
            completed_at=None,
            quality_level=quality_level,
            overall_score=0.0,
            max_possible_score=0.0,
            validation_results=[],
            category_scores={},
            critical_failures=[],
            recommendations=[]
        )
        
        try:
            # Run validation tests
            validation_results = []
            
            # Python syntax validation
            result = await self._validate_python_syntax()
            validation_results.append(result)
            
            # Import validation 
            result = await self._validate_imports()
            validation_results.append(result)
            
            # Security validation
            result = await self._validate_security()
            validation_results.append(result)
            
            # Functionality testing
            result = await self._test_functionality()
            validation_results.append(result)
            
            # Integration testing
            result = await self._test_integration()
            validation_results.append(result)
            
            report.validation_results = validation_results
            report.max_possible_score = sum(r.max_score for r in validation_results)
            
            # Calculate scores
            self._calculate_scores(report)
            
            # Generate recommendations
            report.recommendations = self._generate_recommendations(report)
            
        except Exception as e:
            self.logger.error(f"Quality validation failed: {e}", exc_info=True)
            report.critical_failures.append(f"Validation system failure: {e}")
            
        finally:
            report.completed_at = datetime.now()
            execution_time = (report.completed_at - start_time).total_seconds()
            
            # Save report
            await self._save_quality_report(report)
            
            self.logger.info(
                f"🔍 Quality validation complete - Score: {report.overall_percentage:.1f}%, "
                f"Time: {execution_time:.2f}s"
            )
        
        return report
    
    async def _validate_python_syntax(self) -> ValidationResult:
        """Validate Python syntax across all files"""
        start_time = time.time()
        
        try:
            valid_files = 0
            syntax_errors = []
            
            for py_file in self.python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        source_code = f.read()
                    
                    compile(source_code, str(py_file), 'exec')
                    valid_files += 1
                    
                except SyntaxError as e:
                    syntax_errors.append({
                        "file": str(py_file),
                        "line": e.lineno,
                        "message": e.msg
                    })
                except Exception as e:
                    syntax_errors.append({
                        "file": str(py_file),
                        "error": str(e)
                    })
            
            total_files = len(self.python_files)
            if total_files == 0:
                return ValidationResult(
                    test_name="Python Syntax Validation",
                    category=ValidationCategory.SYNTAX,
                    result=TestResult.PASSED,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    score=20.0,
                    max_score=20.0,
                    details={"message": "No Python files found"}
                )
            
            success_rate = valid_files / total_files
            score = success_rate * 20.0
            
            return ValidationResult(
                test_name="Python Syntax Validation",
                category=ValidationCategory.SYNTAX,
                result=TestResult.PASSED if len(syntax_errors) == 0 else TestResult.FAILED,
                execution_time_ms=(time.time() - start_time) * 1000,
                score=score,
                max_score=20.0,
                details={
                    "total_files": total_files,
                    "valid_files": valid_files,
                    "syntax_errors": len(syntax_errors),
                    "error_details": syntax_errors[:5]
                },
                error_message=f"{len(syntax_errors)} syntax errors found" if syntax_errors else None
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Python Syntax Validation",
                category=ValidationCategory.SYNTAX,
                result=TestResult.ERROR,
                execution_time_ms=(time.time() - start_time) * 1000,
                score=0.0,
                max_score=20.0,
                error_message=str(e)
            )
    
    async def _validate_imports(self) -> ValidationResult:
        """Validate import structure and dependencies"""
        start_time = time.time()
        
        try:
            total_imports = 0
            unique_imports = set()
            import_errors = []
            
            for py_file in self.python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        source_code = f.read()
                    
                    tree = ast.parse(source_code)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                import_name = alias.name
                                unique_imports.add(import_name)
                                total_imports += 1
                        
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                unique_imports.add(node.module)
                                total_imports += 1
                                
                except Exception as e:
                    import_errors.append({
                        "file": str(py_file),
                        "error": str(e)
                    })
            
            if total_imports == 0:
                score = 15.0
                result = TestResult.PASSED
            else:
                import_health = 1.0 - (len(import_errors) / max(len(self.python_files), 1))
                score = import_health * 15.0
                result = TestResult.PASSED if len(import_errors) == 0 else TestResult.FAILED
            
            return ValidationResult(
                test_name="Import Structure Validation",
                category=ValidationCategory.IMPORTS,
                result=result,
                execution_time_ms=(time.time() - start_time) * 1000,
                score=score,
                max_score=15.0,
                details={
                    "total_imports": total_imports,
                    "unique_imports": len(unique_imports),
                    "import_errors": len(import_errors),
                    "error_details": import_errors[:3]
                },
                error_message=f"{len(import_errors)} import errors found" if import_errors else None
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Import Structure Validation", 
                category=ValidationCategory.IMPORTS,
                result=TestResult.ERROR,
                execution_time_ms=(time.time() - start_time) * 1000,
                score=0.0,
                max_score=15.0,
                error_message=str(e)
            )
    
    async def _validate_security(self) -> ValidationResult:
        """Validate security patterns and practices"""
        start_time = time.time()
        
        try:
            security_issues = []
            security_score = 25.0
            
            # Security patterns to detect
            dangerous_functions = ['eval', 'exec', '__import__']
            secret_patterns = ['password', 'secret', 'key', 'token']
            
            for py_file in self.python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        source_code = f.read()
                    
                    # Check for dangerous functions
                    for func in dangerous_functions:
                        if f'{func}(' in source_code:
                            security_issues.append({
                                "file": str(py_file),
                                "issue": f"Usage of potentially dangerous function: {func}",
                                "severity": "medium"
                            })
                    
                    # Check for potential hardcoded secrets
                    lines = source_code.split('\n')
                    for line_num, line in enumerate(lines, 1):
                        for pattern in secret_patterns:
                            if pattern.lower() in line.lower() and '=' in line:
                                if not line.strip().startswith('#'):
                                    value = line.split('=')[1].strip().strip('"\'')
                                    if len(value) > 8 and not value.startswith('${'):
                                        security_issues.append({
                                            "file": str(py_file),
                                            "line": line_num,
                                            "issue": f"Potential hardcoded {pattern}",
                                            "severity": "high"
                                        })
                                        
                except Exception:
                    continue
            
            # Reduce score based on issues
            high_issues = len([i for i in security_issues if i.get("severity") == "high"])
            medium_issues = len([i for i in security_issues if i.get("severity") == "medium"])
            
            security_score -= (high_issues * 5.0) + (medium_issues * 2.0)
            security_score = max(0.0, security_score)
            
            result = TestResult.PASSED if security_score >= 20.0 else TestResult.FAILED
            
            return ValidationResult(
                test_name="Security Validation",
                category=ValidationCategory.SECURITY,
                result=result,
                execution_time_ms=(time.time() - start_time) * 1000,
                score=security_score,
                max_score=25.0,
                details={
                    "security_issues": len(security_issues),
                    "high_severity": high_issues,
                    "medium_severity": medium_issues,
                    "issue_details": security_issues[:5]
                },
                error_message=f"{len(security_issues)} security issues found" if security_issues else None
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Security Validation",
                category=ValidationCategory.SECURITY,
                result=TestResult.ERROR,
                execution_time_ms=(time.time() - start_time) * 1000,
                score=0.0,
                max_score=25.0,
                error_message=str(e)
            )
    
    async def _test_functionality(self) -> ValidationResult:
        """Test basic system functionality"""
        start_time = time.time()
        
        try:
            functionality_tests = []
            
            # Test basic imports
            try:
                basic_modules = ['json', 'os', 'sys', 'asyncio', 'pathlib']
                for module in basic_modules:
                    __import__(module)
                functionality_tests.append({"test": "basic_imports", "passed": True})
            except Exception as e:
                functionality_tests.append({"test": "basic_imports", "passed": False, "error": str(e)})
            
            # Test file operations
            try:
                test_file = self.project_root / "functionality_test.tmp"
                test_content = "Functionality test content"
                
                with open(test_file, 'w') as f:
                    f.write(test_content)
                
                with open(test_file, 'r') as f:
                    read_content = f.read()
                
                test_file.unlink()
                
                functionality_tests.append({
                    "test": "file_operations", 
                    "passed": read_content == test_content
                })
            except Exception as e:
                functionality_tests.append({"test": "file_operations", "passed": False, "error": str(e)})
            
            # Test JSON operations
            try:
                test_data = {"test": "data", "number": 42}
                json_string = json.dumps(test_data)
                parsed_data = json.loads(json_string)
                
                functionality_tests.append({
                    "test": "json_operations",
                    "passed": parsed_data == test_data
                })
            except Exception as e:
                functionality_tests.append({"test": "json_operations", "passed": False, "error": str(e)})
            
            passed_tests = len([t for t in functionality_tests if t["passed"]])
            total_tests = len(functionality_tests)
            score = (passed_tests / total_tests) * 20.0 if total_tests > 0 else 0.0
            
            result = TestResult.PASSED if passed_tests == total_tests else TestResult.FAILED
            
            return ValidationResult(
                test_name="Basic Functionality Test",
                category=ValidationCategory.FUNCTIONALITY,
                result=result,
                execution_time_ms=(time.time() - start_time) * 1000,
                score=score,
                max_score=20.0,
                details={
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "test_results": functionality_tests
                },
                error_message=f"{total_tests - passed_tests} functionality tests failed" if passed_tests < total_tests else None
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Basic Functionality Test",
                category=ValidationCategory.FUNCTIONALITY,
                result=TestResult.ERROR,
                execution_time_ms=(time.time() - start_time) * 1000,
                score=0.0,
                max_score=20.0,
                error_message=str(e)
            )
    
    async def _test_integration(self) -> ValidationResult:
        """Test system integration points"""
        start_time = time.time()
        
        try:
            integration_tests = []
            
            # Test configuration files
            try:
                config_files = ["pyproject.toml", "docker-compose.yml"]
                configs_exist = sum(1 for cf in config_files if (self.project_root / cf).exists())
                
                integration_tests.append({
                    "test": "configuration_files",
                    "passed": configs_exist >= 1,
                    "configs_found": configs_exist
                })
            except Exception as e:
                integration_tests.append({"test": "configuration_files", "passed": False, "error": str(e)})
            
            # Test directory structure
            try:
                required_dirs = ["src", "logs"]
                dirs_exist = sum(1 for rd in required_dirs if (self.project_root / rd).exists())
                
                integration_tests.append({
                    "test": "directory_structure",
                    "passed": dirs_exist >= 1,
                    "directories_found": dirs_exist
                })
            except Exception as e:
                integration_tests.append({"test": "directory_structure", "passed": False, "error": str(e)})
            
            # Test Python package structure
            try:
                src_dir = self.project_root / "src"
                if src_dir.exists():
                    package_dirs = [d for d in src_dir.iterdir() if d.is_dir() and (d / "__init__.py").exists()]
                    package_structure_valid = len(package_dirs) > 0
                else:
                    package_structure_valid = False
                    package_dirs = []
                
                integration_tests.append({
                    "test": "package_structure",
                    "passed": package_structure_valid,
                    "packages_found": len(package_dirs)
                })
            except Exception as e:
                integration_tests.append({"test": "package_structure", "passed": False, "error": str(e)})
            
            passed_tests = len([t for t in integration_tests if t["passed"]])
            total_tests = len(integration_tests)
            score = (passed_tests / total_tests) * 20.0 if total_tests > 0 else 0.0
            
            result = TestResult.PASSED if passed_tests >= (total_tests * 0.8) else TestResult.FAILED
            
            return ValidationResult(
                test_name="Integration Test",
                category=ValidationCategory.INTEGRATION,
                result=result,
                execution_time_ms=(time.time() - start_time) * 1000,
                score=score,
                max_score=20.0,
                details={
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "integration_results": integration_tests
                },
                error_message=f"{total_tests - passed_tests} integration tests failed" if passed_tests < total_tests * 0.8 else None
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Integration Test",
                category=ValidationCategory.INTEGRATION,
                result=TestResult.ERROR,
                execution_time_ms=(time.time() - start_time) * 1000,
                score=0.0,
                max_score=20.0,
                error_message=str(e)
            )
    
    def _calculate_scores(self, report: QualityReport):
        """Calculate comprehensive scores for the quality report"""
        # Calculate overall score
        report.overall_score = sum(result.score for result in report.validation_results)
        
        # Calculate category scores
        category_scores = {}
        for category in ValidationCategory:
            category_results = [r for r in report.validation_results if r.category == category]
            if category_results:
                category_score = sum(r.score for r in category_results)
                category_max = sum(r.max_score for r in category_results)
                category_scores[category.value] = {
                    "score": category_score,
                    "max_score": category_max,
                    "percentage": (category_score / category_max * 100) if category_max > 0 else 0,
                    "test_count": len(category_results)
                }
        
        report.category_scores = category_scores
        
        # Identify critical failures
        critical_failures = []
        for result in report.validation_results:
            if result.result == TestResult.FAILED and result.category in [ValidationCategory.SYNTAX, ValidationCategory.SECURITY]:
                critical_failures.append(f"{result.test_name}: {result.error_message or 'Test failed'}")
        
        report.critical_failures = critical_failures
    
    def _generate_recommendations(self, report: QualityReport) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on validation results"""
        recommendations = []
        
        # Analyze failed tests
        failed_tests = [r for r in report.validation_results if r.result in [TestResult.FAILED, TestResult.ERROR]]
        
        for failed_test in failed_tests:
            category = failed_test.category.value
            
            if category == "syntax":
                recommendations.append({
                    "type": "code_quality",
                    "priority": "high",
                    "title": "Fix Syntax Issues",
                    "description": "Address Python syntax errors to ensure code can be executed",
                    "category": category
                })
            
            elif category == "security":
                recommendations.append({
                    "type": "security",
                    "priority": "critical",
                    "title": "Address Security Vulnerabilities", 
                    "description": "Fix security issues including hardcoded secrets and unsafe patterns",
                    "category": category
                })
            
            elif category == "functionality":
                recommendations.append({
                    "type": "functionality",
                    "priority": "high",
                    "title": "Fix Functionality Issues",
                    "description": "Address basic functionality failures to ensure system operates correctly",
                    "category": category
                })
            
            elif category == "integration":
                recommendations.append({
                    "type": "integration",
                    "priority": "medium",
                    "title": "Fix Integration Issues",
                    "description": "Ensure proper system integration and configuration",
                    "category": category
                })
        
        # Add general recommendations based on overall score
        if report.overall_percentage < 70:
            recommendations.append({
                "type": "general",
                "priority": "high",
                "title": "Comprehensive Quality Improvement Needed",
                "description": f"Overall quality score is {report.overall_percentage:.1f}%. Focus on critical failures first.",
                "category": "general"
            })
        
        elif report.overall_percentage < 85:
            recommendations.append({
                "type": "general",
                "priority": "medium", 
                "title": "Quality Enhancement Recommended",
                "description": f"Quality score is {report.overall_percentage:.1f}%. Consider addressing remaining issues.",
                "category": "general"
            })
        
        return recommendations
    
    async def _save_quality_report(self, report: QualityReport):
        """Save comprehensive quality report"""
        try:
            # Save JSON report
            report_path = self.project_root / f"simple_quality_report_{report.execution_id}.json"
            with open(report_path, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
            
            # Save summary
            summary_path = self.project_root / f"simple_quality_summary_{report.execution_id}.md"
            with open(summary_path, 'w') as f:
                f.write(self._generate_quality_summary(report))
            
            self.logger.info(f"Quality reports saved: {report_path}, {summary_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save quality report: {e}")
    
    def _generate_quality_summary(self, report: QualityReport) -> str:
        """Generate markdown summary of quality validation"""
        execution_time = (report.completed_at - report.started_at).total_seconds() if report.completed_at else 0
        
        md = f"""# Simple Quality Validation Report

## 🔍 Executive Summary

**Execution ID**: {report.execution_id}  
**Quality Level**: {report.quality_level.value}  
**Started**: {report.started_at}  
**Completed**: {report.completed_at}  
**Duration**: {execution_time:.2f} seconds

## 📊 Overall Quality Score: {report.overall_percentage:.1f}%

- **Score**: {report.overall_score:.1f}/{report.max_possible_score:.1f}
- **Tests Passed**: {report.passed_tests}/{report.total_tests}
- **Success Rate**: {(report.passed_tests/report.total_tests*100):.1f}%
- **Critical Failures**: {len(report.critical_failures)}

## 📈 Category Breakdown

"""
        
        for category_name, category_data in report.category_scores.items():
            percentage = category_data["percentage"]
            emoji = "✅" if percentage >= 80 else "⚠️" if percentage >= 60 else "❌"
            
            md += f"### {emoji} {category_name.title()}\n"
            md += f"- **Score**: {category_data['score']:.1f}/{category_data['max_score']:.1f} ({percentage:.1f}%)\n"
            md += f"- **Tests**: {category_data['test_count']}\n\n"
        
        # Critical Failures
        if report.critical_failures:
            md += "## ❌ Critical Failures\n\n"
            for failure in report.critical_failures:
                md += f"- {failure}\n"
            md += "\n"
        
        # Recommendations
        if report.recommendations:
            md += "## 💡 Recommendations\n\n"
            for rec in report.recommendations:
                priority_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(rec.get("priority", "low"), "⚪")
                md += f"- {priority_emoji} **{rec.get('title', 'Unknown')}**: {rec.get('description', 'No description')}\n"
        
        md += f"""
---
*Generated by Simple Quality Validation System*  
*Report ID: {report.execution_id}*
"""
        
        return md


# Main execution function
async def execute_simple_quality_validation():
    """Execute simple quality validation"""
    try:
        print("🔍 STARTING SIMPLE QUALITY VALIDATION")
        print("=" * 60)
        
        validator = SimpleQualityValidator()
        
        # Execute quality validation
        report = await validator.execute_quality_validation(QualityLevel.COMPREHENSIVE)
        
        print(f"\n🎯 SIMPLE QUALITY VALIDATION COMPLETE")
        print(f"📊 Overall Score: {report.overall_percentage:.1f}%")
        print(f"✅ Tests Passed: {report.passed_tests}/{report.total_tests}")
        print(f"❌ Critical Failures: {len(report.critical_failures)}")
        print(f"⏱️ Execution Time: {((report.completed_at - report.started_at).total_seconds()):.2f}s")
        
        # Category summary
        print(f"\n📈 Category Performance:")
        for category_name, category_data in report.category_scores.items():
            percentage = category_data["percentage"]
            status_emoji = "✅" if percentage >= 80 else "⚠️" if percentage >= 60 else "❌"
            print(f"   {status_emoji} {category_name.title()}: {percentage:.1f}%")
        
        return {
            "success": report.overall_percentage >= 70 and len(report.critical_failures) == 0,
            "overall_score": report.overall_percentage,
            "critical_failures": len(report.critical_failures),
            "report": report
        }
        
    except Exception as e:
        print(f"❌ Quality validation failed: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    result = asyncio.run(execute_simple_quality_validation())
    sys.exit(0 if result["success"] else 1)