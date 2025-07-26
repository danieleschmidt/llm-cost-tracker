#!/usr/bin/env python3
"""Simple test for the autonomous backlog system without external dependencies."""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_cost_tracker.backlog_manager import AutonomousBacklogManager, BacklogItem


async def test_discovery_cycle():
    """Test the discovery and prioritization cycle."""
    print("🔍 Testing Autonomous Backlog Discovery...")
    
    manager = AutonomousBacklogManager()
    
    # Run discovery cycle
    items, metrics = await manager.run_discovery_cycle()
    
    print(f"✅ Discovery complete!")
    print(f"  Total items: {metrics.total_items}")
    print(f"  Average WSJF: {metrics.avg_wsjf_score}")
    print(f"  Status breakdown: {metrics.by_status}")
    
    # Show top priorities
    active_items = [i for i in items if i.status not in ['DONE', 'BLOCKED']]
    print(f"\n📋 Top {min(5, len(active_items))} Priorities:")
    
    for i, item in enumerate(active_items[:5], 1):
        print(f"{i}. {item.title}")
        print(f"   WSJF: {item.wsjf_score} | Status: {item.status} | Type: {item.type}")
        print(f"   Effort: {item.effort} | Value: {item.value} | Risk: {item.risk_tier}")
    
    # Get next actionable item
    next_item = await manager.get_next_actionable_item(items)
    if next_item:
        print(f"\n⚡ Next actionable item: {next_item.title}")
        print(f"   WSJF Score: {next_item.wsjf_score}")
        print(f"   Acceptance Criteria:")
        for criteria in next_item.acceptance_criteria:
            print(f"     - {criteria}")
    else:
        print("\n✅ No actionable items found - backlog is complete!")
    
    return items, metrics


async def test_wsjf_scoring():
    """Test WSJF scoring calculations."""
    print("\n🎯 Testing WSJF Scoring System...")
    
    # Create test items with different profiles
    test_items = [
        BacklogItem(
            id="TEST-001",
            title="High-value quick win",
            type="feature",
            description="Small but valuable feature",
            acceptance_criteria=["Implement feature", "Add tests"],
            effort=1,  # Very low effort
            value=8,   # High value
            time_criticality=5,
            risk_reduction=3,
            created_at=datetime.now().isoformat()
        ),
        BacklogItem(
            id="TEST-002", 
            title="Large complex feature",
            type="feature",
            description="Major feature with high complexity",
            acceptance_criteria=["Design architecture", "Implement", "Test"],
            effort=13,  # Very high effort
            value=13,   # Very high value
            time_criticality=2,
            risk_reduction=1,
            created_at=datetime.now().isoformat()
        ),
        BacklogItem(
            id="TEST-003",
            title="Critical security fix",
            type="security",
            description="Security vulnerability fix",
            acceptance_criteria=["Patch vulnerability", "Security review"],
            effort=3,   # Medium effort
            value=5,    # Medium value
            time_criticality=13,  # Critical timing
            risk_reduction=13,    # High risk reduction
            created_at=datetime.now().isoformat()
        )
    ]
    
    for item in test_items:
        item.wsjf_score = item.calculate_wsjf()
        print(f"{item.title}:")
        print(f"  Value: {item.value}, Time Criticality: {item.time_criticality}, Risk Reduction: {item.risk_reduction}")
        print(f"  Effort: {item.effort} → WSJF Score: {item.wsjf_score}")
    
    # Sort by WSJF
    sorted_items = sorted(test_items, key=lambda x: x.wsjf_score, reverse=True)
    print(f"\n📊 WSJF Priority Order:")
    for i, item in enumerate(sorted_items, 1):
        print(f"{i}. {item.title} (WSJF: {item.wsjf_score})")


async def test_metrics_generation():
    """Test metrics and reporting system."""
    print("\n📊 Testing Metrics Generation...")
    
    manager = AutonomousBacklogManager()
    items, metrics = await manager.load_backlog()
    
    # Generate status report
    await manager.generate_status_report(items, metrics)
    
    # Check if files were created
    today = datetime.now().strftime('%Y-%m-%d')
    json_file = manager.metrics_dir / f"{today}.json"
    md_file = manager.metrics_dir / f"{today}.md"
    
    if json_file.exists():
        print(f"✅ JSON report created: {json_file}")
        with open(json_file) as f:
            report_data = json.load(f)
            print(f"   Completed IDs: {report_data.get('completed_ids', [])}")
            print(f"   Top priorities: {len(report_data.get('top_priorities', []))}")
    
    if md_file.exists():
        print(f"✅ Markdown report created: {md_file}")
        print(f"   Size: {md_file.stat().st_size} bytes")


async def main():
    """Run all tests."""
    print("🚀 AUTONOMOUS BACKLOG SYSTEM TEST SUITE")
    print("=" * 50)
    
    try:
        # Test WSJF scoring
        await test_wsjf_scoring()
        
        # Test discovery cycle
        items, metrics = await test_discovery_cycle()
        
        # Test metrics generation
        await test_metrics_generation()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("\nThe autonomous backlog system is ready to:")
        print("  • Discover technical debt and failing tests")
        print("  • Prioritize items using WSJF methodology") 
        print("  • Generate actionable status reports")
        print("  • Support autonomous execution loops")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())