"""Autonomous backlog management system with WSJF prioritization."""

import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BacklogItem(BaseModel):
    """Backlog item with WSJF scoring."""
    
    id: str
    title: str
    type: str
    description: str
    acceptance_criteria: List[str]
    effort: int = Field(ge=1, le=13)  # Fibonacci scale
    value: int = Field(ge=1, le=13)
    time_criticality: int = Field(ge=1, le=13)
    risk_reduction: int = Field(ge=1, le=13)
    status: str = Field(default="NEW")
    risk_tier: str = Field(default="low")
    created_at: str
    completed_at: Optional[str] = None
    wsjf_score: float = 0.0
    aging_multiplier: float = Field(default=1.0, le=2.0)
    links: List[str] = Field(default_factory=list)

    def calculate_wsjf(self) -> float:
        """Calculate WSJF score with aging multiplier."""
        if self.effort == 0:
            return 0.0
        base_score = (self.value + self.time_criticality + self.risk_reduction) / self.effort
        return round(base_score * self.aging_multiplier, 2)

    def apply_aging(self, reference_date: datetime) -> None:
        """Apply aging multiplier for stale items."""
        created = datetime.fromisoformat(self.created_at)
        days_old = (reference_date - created).days
        
        if days_old > 30:  # Items older than 30 days get aging bonus
            age_factor = min(1 + (days_old - 30) / 100, 2.0)  # Max 2x multiplier
            self.aging_multiplier = age_factor
        
        self.wsjf_score = self.calculate_wsjf()


class BacklogMetrics(BaseModel):
    """Backlog metrics and KPIs."""
    
    total_items: int
    by_status: Dict[str, int]
    by_risk_tier: Dict[str, int]
    avg_wsjf_score: float
    last_updated: str
    cycle_times: Dict[str, float] = Field(default_factory=dict)
    dora_metrics: Dict[str, float] = Field(default_factory=dict)


class AutonomousBacklogManager:
    """Manages backlog with WSJF prioritization and autonomous execution."""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.backlog_file = self.repo_path / "backlog.yml"
        self.metrics_dir = self.repo_path / "docs" / "status"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
    async def load_backlog(self) -> Tuple[List[BacklogItem], BacklogMetrics]:
        """Load backlog from YAML file."""
        try:
            with open(self.backlog_file, 'r') as f:
                data = yaml.safe_load(f)
            
            items = [BacklogItem(**item) for item in data.get('backlog', [])]
            metrics = BacklogMetrics(**data.get('metrics', {}))
            
            return items, metrics
        except Exception as e:
            logger.error(f"Failed to load backlog: {e}")
            return [], BacklogMetrics(
                total_items=0,
                by_status={},
                by_risk_tier={},
                avg_wsjf_score=0.0,
                last_updated=datetime.now().isoformat()
            )

    async def discover_technical_debt(self) -> List[BacklogItem]:
        """Scan for TODO/FIXME comments and create backlog items."""
        new_items = []
        
        # Search for TODO/FIXME comments
        try:
            result = subprocess.run([
                'grep', '-r', '-n', '-E', 'TODO|FIXME|XXX|HACK',
                str(self.repo_path), '--exclude-dir=.git'
            ], capture_output=True, text=True)
            
            for line in result.stdout.split('\n'):
                if line.strip() and not line.startswith('Binary'):
                    parts = line.split(':', 3)
                    if len(parts) >= 3:
                        file_path, line_num, comment = parts[0], parts[1], parts[2]
                        
                        item_id = f"DEBT-{hash(line) % 10000:04d}"
                        new_items.append(BacklogItem(
                            id=item_id,
                            title=f"Address technical debt in {Path(file_path).name}:{line_num}",
                            type="technical_debt",
                            description=f"Technical debt found: {comment.strip()}",
                            acceptance_criteria=[
                                f"Resolve TODO/FIXME comment in {file_path}:{line_num}",
                                "Add appropriate tests if needed",
                                "Update documentation if applicable"
                            ],
                            effort=3,  # Default effort
                            value=2,   # Default value for tech debt
                            time_criticality=1,
                            risk_reduction=3,
                            created_at=datetime.now().isoformat(),
                            links=[f"file://{file_path}#{line_num}"]
                        ))
        except Exception as e:
            logger.warning(f"Failed to scan for technical debt: {e}")
        
        return new_items

    async def discover_failing_tests(self) -> List[BacklogItem]:
        """Check for failing tests and create backlog items."""
        new_items = []
        
        try:
            # Run tests to detect failures
            result = subprocess.run([
                'python', '-m', 'pytest', '--tb=no', '-q'
            ], cwd=self.repo_path, capture_output=True, text=True)
            
            if result.returncode != 0:
                new_items.append(BacklogItem(
                    id=f"TEST-FAIL-{datetime.now().strftime('%Y%m%d')}",
                    title="Fix failing tests",
                    type="bug",
                    description="Test suite is failing",
                    acceptance_criteria=[
                        "All tests pass",
                        "No test regressions introduced",
                        "Coverage maintained or improved"
                    ],
                    effort=5,
                    value=8,
                    time_criticality=8,
                    risk_reduction=8,
                    created_at=datetime.now().isoformat()
                ))
        except Exception as e:
            logger.warning(f"Failed to check test status: {e}")
        
        return new_items

    async def score_and_sort_backlog(self, items: List[BacklogItem]) -> List[BacklogItem]:
        """Apply WSJF scoring and sort backlog by priority."""
        reference_date = datetime.now()
        
        for item in items:
            if item.status not in ['DONE', 'BLOCKED']:
                item.apply_aging(reference_date)
        
        # Sort by WSJF score descending, then by creation date
        return sorted(
            items,
            key=lambda x: (-x.wsjf_score, x.created_at),
            reverse=False
        )

    async def get_next_actionable_item(self, items: List[BacklogItem]) -> Optional[BacklogItem]:
        """Get the next actionable item from backlog."""
        for item in items:
            if item.status in ['NEW', 'READY'] and item.risk_tier != 'blocked':
                return item
        return None

    async def update_metrics(self, items: List[BacklogItem]) -> BacklogMetrics:
        """Calculate and update backlog metrics."""
        status_counts = {}
        risk_counts = {}
        
        for item in items:
            status_counts[item.status] = status_counts.get(item.status, 0) + 1
            risk_counts[item.risk_tier] = risk_counts.get(item.risk_tier, 0) + 1
        
        # Calculate average WSJF score for active items
        active_items = [i for i in items if i.status not in ['DONE', 'BLOCKED']]
        avg_wsjf = sum(i.wsjf_score for i in active_items) / len(active_items) if active_items else 0.0
        
        return BacklogMetrics(
            total_items=len(items),
            by_status=status_counts,
            by_risk_tier=risk_counts,
            avg_wsjf_score=round(avg_wsjf, 2),
            last_updated=datetime.now().isoformat()
        )

    async def save_backlog(self, items: List[BacklogItem], metrics: BacklogMetrics) -> None:
        """Save backlog to YAML file."""
        data = {
            'backlog': [item.model_dump() for item in items],
            'metrics': metrics.model_dump()
        }
        
        with open(self.backlog_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    async def generate_status_report(self, items: List[BacklogItem], metrics: BacklogMetrics) -> None:
        """Generate daily status report."""
        today = datetime.now().strftime('%Y-%m-%d')
        
        # JSON report
        json_report = {
            "timestamp": datetime.now().isoformat(),
            "completed_ids": [i.id for i in items if i.status == 'DONE'],
            "backlog_size_by_status": metrics.by_status,
            "avg_wsjf_score": metrics.avg_wsjf_score,
            "top_priorities": [
                {"id": i.id, "title": i.title, "wsjf_score": i.wsjf_score}
                for i in items[:5] if i.status not in ['DONE', 'BLOCKED']
            ]
        }
        
        json_file = self.metrics_dir / f"{today}.json"
        with open(json_file, 'w') as f:
            json.dump(json_report, f, indent=2)
        
        # Markdown report
        md_content = f"""# Backlog Status Report - {today}

## Summary
- **Total Items**: {metrics.total_items}
- **Average WSJF Score**: {metrics.avg_wsjf_score}
- **Last Updated**: {metrics.last_updated}

## Status Breakdown
"""
        for status, count in metrics.by_status.items():
            md_content += f"- **{status}**: {count}\n"
        
        md_content += "\n## Top Priorities\n"
        for i, item in enumerate(items[:5], 1):
            if item.status not in ['DONE', 'BLOCKED']:
                md_content += f"{i}. **{item.title}** (WSJF: {item.wsjf_score})\n"
        
        md_file = self.metrics_dir / f"{today}.md"
        with open(md_file, 'w') as f:
            f.write(md_content)

    async def run_discovery_cycle(self) -> None:
        """Run one complete discovery and prioritization cycle."""
        logger.info("Starting backlog discovery cycle")
        
        # Load existing backlog
        items, metrics = await self.load_backlog()
        
        # Discover new items
        tech_debt_items = await self.discover_technical_debt()
        test_items = await self.discover_failing_tests()
        
        # Merge new items (deduplicate by ID)
        existing_ids = {item.id for item in items}
        new_items = [item for item in tech_debt_items + test_items 
                     if item.id not in existing_ids]
        
        all_items = items + new_items
        
        # Score and sort
        sorted_items = await self.score_and_sort_backlog(all_items)
        
        # Update metrics
        updated_metrics = await self.update_metrics(sorted_items)
        
        # Save results
        await self.save_backlog(sorted_items, updated_metrics)
        await self.generate_status_report(sorted_items, updated_metrics)
        
        logger.info(f"Discovery complete. Found {len(new_items)} new items. "
                   f"Total backlog: {len(sorted_items)} items")
        
        return sorted_items, updated_metrics

    async def execute_macro_loop(self, max_iterations: int = 10) -> None:
        """Execute the main autonomous loop."""
        logger.info("Starting autonomous backlog execution")
        
        for iteration in range(max_iterations):
            # Discovery cycle
            items, metrics = await self.run_discovery_cycle()
            
            # Get next actionable item
            next_item = await self.get_next_actionable_item(items)
            
            if not next_item:
                logger.info("No actionable items found. Backlog execution complete.")
                break
            
            logger.info(f"Next item to execute: {next_item.title} (WSJF: {next_item.wsjf_score})")
            
            # In a real implementation, this would trigger the micro-cycle execution
            # For now, we just log the intention
            logger.info(f"Would execute item {next_item.id}: {next_item.title}")
            
            # Simulate processing delay
            await asyncio.sleep(1)
        
        logger.info("Autonomous execution cycle complete")


async def main():
    """Main entry point for autonomous backlog management."""
    manager = AutonomousBacklogManager()
    
    # Run discovery cycle
    await manager.run_discovery_cycle()
    
    # Optionally run the full autonomous loop
    # await manager.execute_macro_loop()


if __name__ == "__main__":
    asyncio.run(main())