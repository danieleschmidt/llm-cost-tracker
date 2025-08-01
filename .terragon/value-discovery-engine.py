#!/usr/bin/env python3
"""
Terragon Autonomous Value Discovery Engine
Advanced continuous value discovery and execution system for mature repositories
"""

import asyncio
import json
import logging
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ItemType(Enum):
    SECURITY_FIX = "security_fix"
    TECHNICAL_DEBT = "technical_debt"
    PERFORMANCE = "performance"
    FEATURE_ENHANCEMENT = "feature_enhancement"
    DEPENDENCY_UPDATE = "dependency_update"
    TEST_IMPROVEMENT = "test_improvement"
    DOCUMENTATION = "documentation"
    COMPLIANCE = "compliance"

class Priority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class ValueItem:
    """Represents a discovered work item with comprehensive scoring."""
    id: str
    title: str
    description: str
    item_type: ItemType
    priority: Priority
    
    # WSJF Components (Weighted Shortest Job First)
    user_business_value: int  # 1-10
    time_criticality: int     # 1-10
    risk_reduction: int       # 1-10
    opportunity_enablement: int # 1-10
    job_size: int            # Story points (1, 2, 3, 5, 8, 13)
    
    # ICE Components (Impact, Confidence, Ease)
    impact: int              # 1-10
    confidence: int          # 1-10
    ease: int               # 1-10
    
    # Technical Debt Scoring
    debt_impact: int         # Maintenance hours saved
    debt_interest: int       # Future cost if not addressed
    hotspot_multiplier: float # 1-5x based on file activity
    
    # Metadata
    discovered_at: datetime
    source: str              # Where this item was discovered
    files_affected: List[str]
    estimated_effort_hours: float
    
    # Calculated scores
    wsjf_score: float = 0.0
    ice_score: float = 0.0
    technical_debt_score: float = 0.0
    composite_score: float = 0.0
    
    def calculate_scores(self, weights: Dict[str, float]) -> None:
        """Calculate all scoring components."""
        # WSJF Score
        cost_of_delay = (
            self.user_business_value + 
            self.time_criticality + 
            self.risk_reduction + 
            self.opportunity_enablement
        )
        self.wsjf_score = cost_of_delay / max(self.job_size, 1)
        
        # ICE Score
        self.ice_score = self.impact * self.confidence * self.ease
        
        # Technical Debt Score
        self.technical_debt_score = (
            (self.debt_impact + self.debt_interest) * self.hotspot_multiplier
        )
        
        # Composite Score with adaptive weights
        self.composite_score = (
            weights.get('wsjf', 0.5) * self._normalize_score(self.wsjf_score, 40) +
            weights.get('ice', 0.1) * self._normalize_score(self.ice_score, 1000) +
            weights.get('technicalDebt', 0.3) * self._normalize_score(self.technical_debt_score, 500) +
            weights.get('security', 0.1) * (2.0 if self.item_type == ItemType.SECURITY_FIX else 1.0)
        )
    
    def _normalize_score(self, score: float, max_value: float) -> float:
        """Normalize score to 0-100 range."""
        return min(100, (score / max_value) * 100)

class ValueDiscoveryEngine:
    """Advanced value discovery engine for continuous optimization."""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.config_path = self.repo_path / ".terragon" / "config.yaml"
        self.metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        self.backlog_path = self.repo_path / ".terragon" / "value-backlog.json"
        
        # Ensure directories exist
        self.repo_path.joinpath(".terragon").mkdir(exist_ok=True)
        
        self.config = self._load_config()
        self.discovered_items: List[ValueItem] = []
        
    def _load_config(self) -> Dict[str, Any]:
        """Load Terragon configuration."""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("Config file not found, using defaults")
            return {
                'scoring': {
                    'weights': {
                        'advanced': {
                            'wsjf': 0.5,
                            'ice': 0.1,
                            'technicalDebt': 0.3,
                            'security': 0.1
                        }
                    }
                }
            }
    
    async def discover_security_vulnerabilities(self) -> List[ValueItem]:
        """Discover security vulnerabilities using multiple tools."""
        items = []
        
        try:
            # Run safety check for Python dependencies
            result = await self._run_command(['safety', 'check', '--json'])
            if result and result.returncode == 0:
                try:
                    safety_data = json.loads(result.stdout)
                    for vuln in safety_data:
                        items.append(ValueItem(
                            id=f"SEC-{vuln.get('id', 'unknown')}",
                            title=f"Fix {vuln.get('package', 'package')} vulnerability",
                            description=f"Security vulnerability in {vuln.get('package')}: {vuln.get('advisory', 'No details')}",
                            item_type=ItemType.SECURITY_FIX,
                            priority=Priority.CRITICAL,
                            user_business_value=9,
                            time_criticality=9,
                            risk_reduction=10,
                            opportunity_enablement=6,
                            job_size=3,
                            impact=9,
                            confidence=9,
                            ease=7,
                            debt_impact=20,
                            debt_interest=50,
                            hotspot_multiplier=2.0,
                            discovered_at=datetime.now(),
                            source="safety_check",
                            files_affected=[],
                            estimated_effort_hours=2.0
                        ))
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            logger.warning(f"Failed to run security scan: {e}")
        
        return items
    
    async def discover_technical_debt(self) -> List[ValueItem]:
        """Discover technical debt through code analysis."""
        items = []
        
        try:
            # Search for TODO/FIXME/HACK comments
            result = await self._run_command([
                'grep', '-r', '-n', '-E', 'TODO|FIXME|XXX|HACK|DEPRECATED',
                str(self.repo_path), '--exclude-dir=.git', '--exclude-dir=.terragon'
            ])
            
            if result and result.stdout:
                debt_comments = result.stdout.strip().split('\n')
                for i, comment in enumerate(debt_comments[:20]):  # Limit to 20 items
                    if ':' in comment:
                        file_path, line_info = comment.split(':', 1)
                        relative_path = Path(file_path).relative_to(self.repo_path)
                        
                        # Calculate hotspot multiplier based on file extension
                        multiplier = 2.0 if relative_path.suffix == '.py' else 1.5
                        
                        items.append(ValueItem(
                            id=f"DEBT-{hash(comment) % 10000:04d}",
                            title=f"Resolve technical debt in {relative_path.name}",
                            description=f"Technical debt comment: {line_info.strip()}",
                            item_type=ItemType.TECHNICAL_DEBT,
                            priority=Priority.MEDIUM,
                            user_business_value=4,
                            time_criticality=3,
                            risk_reduction=5,
                            opportunity_enablement=6,
                            job_size=2,
                            impact=5,
                            confidence=8,
                            ease=6,
                            debt_impact=5,
                            debt_interest=10,
                            hotspot_multiplier=multiplier,
                            discovered_at=datetime.now(),
                            source="code_analysis",
                            files_affected=[str(relative_path)],
                            estimated_effort_hours=1.5
                        ))
        except Exception as e:
            logger.warning(f"Failed to analyze technical debt: {e}")
        
        return items
    
    async def discover_performance_opportunities(self) -> List[ValueItem]:
        """Discover performance optimization opportunities."""
        items = []
        
        try:
            # Look for potential performance issues in Python code
            perf_patterns = [
                r'\.join\(',  # String concatenation opportunities
                r'for.*in.*range\(len\(',  # Inefficient loops
                r'time\.sleep\(',  # Blocking sleep calls
            ]
            
            for pattern in perf_patterns:
                result = await self._run_command([
                    'grep', '-r', '-n', '-E', pattern,
                    str(self.repo_path / 'src'), '--include=*.py'
                ])
                
                if result and result.stdout:
                    matches = result.stdout.strip().split('\n')
                    for match in matches[:5]:  # Limit findings
                        if ':' in match:
                            file_path, line_info = match.split(':', 1)
                            relative_path = Path(file_path).relative_to(self.repo_path)
                            
                            items.append(ValueItem(
                                id=f"PERF-{hash(match) % 10000:04d}",
                                title=f"Optimize performance in {relative_path.name}",
                                description=f"Potential performance improvement: {line_info.strip()}",
                                item_type=ItemType.PERFORMANCE,
                                priority=Priority.MEDIUM,
                                user_business_value=6,
                                time_criticality=4,
                                risk_reduction=3,
                                opportunity_enablement=7,
                                job_size=5,
                                impact=7,
                                confidence=6,
                                ease=5,
                                debt_impact=15,
                                debt_interest=25,
                                hotspot_multiplier=1.5,
                                discovered_at=datetime.now(),
                                source="performance_analysis",
                                files_affected=[str(relative_path)],
                                estimated_effort_hours=4.0
                            ))
        except Exception as e:
            logger.warning(f"Failed to analyze performance: {e}")
        
        return items
    
    async def discover_test_improvements(self) -> List[ValueItem]:
        """Discover test coverage and quality improvements."""
        items = []
        
        try:
            # Check for test coverage gaps
            result = await self._run_command([
                'python', '-m', 'pytest', '--cov=src', '--cov-report=json', '--tb=no', '-q'
            ], cwd=self.repo_path)
            
            # Look for files without tests
            src_files = list(self.repo_path.glob('src/**/*.py'))
            test_files = set(self.repo_path.glob('tests/**/test_*.py'))
            
            for src_file in src_files[:10]:  # Limit to 10 files
                if src_file.name == '__init__.py':
                    continue
                    
                # Check if corresponding test exists
                expected_test = f"test_{src_file.stem}.py"
                has_test = any(expected_test in str(test_file) for test_file in test_files)
                
                if not has_test:
                    items.append(ValueItem(
                        id=f"TEST-{hash(str(src_file)) % 10000:04d}",
                        title=f"Add tests for {src_file.name}",
                        description=f"Missing test coverage for {src_file.relative_to(self.repo_path)}",
                        item_type=ItemType.TEST_IMPROVEMENT,
                        priority=Priority.MEDIUM,
                        user_business_value=5,
                        time_criticality=4,
                        risk_reduction=8,
                        opportunity_enablement=5,
                        job_size=8,
                        impact=6,
                        confidence=7,
                        ease=4,
                        debt_impact=10,
                        debt_interest=30,
                        hotspot_multiplier=1.8,
                        discovered_at=datetime.now(),
                        source="test_analysis",
                        files_affected=[str(src_file.relative_to(self.repo_path))],
                        estimated_effort_hours=6.0
                    ))
        except Exception as e:
            logger.warning(f"Failed to analyze test coverage: {e}")
        
        return items
    
    async def discover_dependency_updates(self) -> List[ValueItem]:
        """Discover outdated dependencies."""
        items = []
        
        try:
            # Check for outdated Python packages
            result = await self._run_command([
                'pip', 'list', '--outdated', '--format=json'
            ])
            
            if result and result.returncode == 0:
                try:
                    outdated = json.loads(result.stdout)
                    for pkg in outdated[:10]:  # Limit to 10 packages
                        items.append(ValueItem(
                            id=f"DEP-{hash(pkg['name']) % 10000:04d}",
                            title=f"Update {pkg['name']} dependency",
                            description=f"Update {pkg['name']} from {pkg['version']} to {pkg['latest_version']}",
                            item_type=ItemType.DEPENDENCY_UPDATE,
                            priority=Priority.LOW,
                            user_business_value=3,
                            time_criticality=2,
                            risk_reduction=4,
                            opportunity_enablement=5,
                            job_size=2,
                            impact=4,
                            confidence=8,
                            ease=8,
                            debt_impact=5,
                            debt_interest=15,
                            hotspot_multiplier=1.2,
                            discovered_at=datetime.now(),
                            source="dependency_analysis",
                            files_affected=["pyproject.toml"],
                            estimated_effort_hours=1.0
                        ))
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            logger.warning(f"Failed to check dependencies: {e}")
        
        return items
    
    async def run_comprehensive_discovery(self) -> List[ValueItem]:
        """Run all discovery methods and consolidate results."""
        logger.info("Starting comprehensive value discovery...")
        
        discovery_tasks = [
            self.discover_security_vulnerabilities(),
            self.discover_technical_debt(),
            self.discover_performance_opportunities(),
            self.discover_test_improvements(),
            self.discover_dependency_updates(),
        ]
        
        # Run all discovery tasks concurrently
        results = await asyncio.gather(*discovery_tasks, return_exceptions=True)
        
        # Consolidate results
        all_items = []
        for result in results:
            if isinstance(result, list):
                all_items.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Discovery task failed: {result}")
        
        # Calculate scores for all items
        weights = self.config.get('scoring', {}).get('weights', {}).get('advanced', {})
        for item in all_items:
            item.calculate_scores(weights)
        
        # Sort by composite score
        all_items.sort(key=lambda x: x.composite_score, reverse=True)
        
        self.discovered_items = all_items
        logger.info(f"Discovery complete. Found {len(all_items)} value opportunities.")
        
        return all_items
    
    async def get_next_best_value_item(self) -> Optional[ValueItem]:
        """Get the highest-value actionable item."""
        if not self.discovered_items:
            await self.run_comprehensive_discovery()
        
        # Filter for actionable items (could add more sophisticated filtering)
        actionable_items = [
            item for item in self.discovered_items
            if item.composite_score > self.config.get('scoring', {}).get('thresholds', {}).get('minScore', 10)
        ]
        
        return actionable_items[0] if actionable_items else None
    
    async def save_value_backlog(self) -> None:
        """Save discovered value items to backlog file."""
        backlog_data = {
            "timestamp": datetime.now().isoformat(),
            "total_items": len(self.discovered_items),
            "items": [asdict(item) for item in self.discovered_items]
        }
        
        with open(self.backlog_path, 'w') as f:
            json.dump(backlog_data, f, indent=2, default=str)
    
    async def generate_value_report(self) -> str:
        """Generate a comprehensive value discovery report."""
        if not self.discovered_items:
            await self.run_comprehensive_discovery()
        
        report = f"""# 📊 Autonomous Value Discovery Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Repository**: {self.repo_path.name}
**Total Opportunities**: {len(self.discovered_items)}

## 🎯 Next Best Value Item
"""
        
        next_item = await self.get_next_best_value_item()
        if next_item:
            report += f"""
**[{next_item.id}] {next_item.title}**
- **Composite Score**: {next_item.composite_score:.1f}
- **WSJF**: {next_item.wsjf_score:.1f} | **ICE**: {next_item.ice_score} | **Tech Debt**: {next_item.technical_debt_score:.1f}
- **Estimated Effort**: {next_item.estimated_effort_hours} hours
- **Type**: {next_item.item_type.value}
- **Priority**: {next_item.priority.value}
"""
        
        report += "\n## 📋 Top 10 Value Opportunities\n\n"
        report += "| Rank | ID | Title | Score | Type | Est. Hours |\n"
        report += "|------|-----|--------|---------|----------|------------|\n"
        
        for i, item in enumerate(self.discovered_items[:10], 1):
            report += f"| {i} | {item.id} | {item.title[:50]} | {item.composite_score:.1f} | {item.item_type.value} | {item.estimated_effort_hours} |\n"
        
        # Add category breakdown
        type_counts = {}
        for item in self.discovered_items:
            type_counts[item.item_type.value] = type_counts.get(item.item_type.value, 0) + 1
        
        report += "\n## 📈 Opportunity Categories\n\n"
        for item_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            report += f"- **{item_type.replace('_', ' ').title()}**: {count} items\n"
        
        report += f"\n## 🔄 Discovery Statistics\n"
        report += f"- **High Priority Items**: {len([i for i in self.discovered_items if i.priority == Priority.HIGH])}\n"
        report += f"- **Security Items**: {len([i for i in self.discovered_items if i.item_type == ItemType.SECURITY_FIX])}\n"
        report += f"- **Technical Debt Items**: {len([i for i in self.discovered_items if i.item_type == ItemType.TECHNICAL_DEBT])}\n"
        report += f"- **Average Composite Score**: {sum(i.composite_score for i in self.discovered_items) / len(self.discovered_items):.1f}\n"
        
        return report
    
    async def _run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> Optional[subprocess.CompletedProcess]:
        """Run a command asynchronously."""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd or self.repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=process.returncode,
                stdout=stdout.decode() if stdout else '',
                stderr=stderr.decode() if stderr else ''
            )
        except Exception as e:
            logger.error(f"Command failed {' '.join(cmd)}: {e}")
            return None

async def main():
    """Main entry point for value discovery engine."""
    engine = ValueDiscoveryEngine()
    
    # Run comprehensive discovery
    items = await engine.run_comprehensive_discovery()
    
    # Save results
    await engine.save_value_backlog()
    
    # Generate and display report
    report = await engine.generate_value_report()
    print(report)
    
    # Get next best item for execution
    next_item = await engine.get_next_best_value_item()
    if next_item:
        logger.info(f"Next recommended action: {next_item.title} (Score: {next_item.composite_score:.1f})")
    else:
        logger.info("No actionable items found.")

if __name__ == "__main__":
    asyncio.run(main())