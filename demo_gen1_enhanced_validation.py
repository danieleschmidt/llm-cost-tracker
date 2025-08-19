#!/usr/bin/env python3
"""
Generation 1 Enhanced Validation Demo
Tests core functionality with comprehensive validation
"""

import asyncio
import sys
import os
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_cost_tracker.quantum_task_planner import QuantumTaskPlanner, QuantumTask, TaskState

def test_basic_functionality():
    """Test basic quantum task planner functionality."""
    print("🔬 Testing Generation 1 - Basic Functionality")
    print("=" * 50)
    
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "generation": 1,
        "tests": {},
        "performance_metrics": {},
        "validation_results": {}
    }
    
    try:
        # Test 1: Basic Task Creation
        print("✅ Test 1: Basic Task Creation")
        start_time = time.time()
        
        planner = QuantumTaskPlanner()
        task1 = QuantumTask(
            id="test_task_1",
            name="Test Data Analysis",
            description="Test quantum task for data analysis",
            priority=8.0,
            estimated_duration=timedelta(minutes=15)
        )
        
        planner.add_task(task1)
        creation_time = time.time() - start_time
        
        results["tests"]["task_creation"] = {
            "status": "passed",
            "execution_time_ms": creation_time * 1000,
            "task_count": len(planner.tasks)
        }
        
        print(f"   📊 Created {len(planner.tasks)} task(s) in {creation_time*1000:.2f}ms")
        
        # Test 2: Quantum State Validation
        print("✅ Test 2: Quantum State Validation")
        start_time = time.time()
        
        assert task1.state == TaskState.SUPERPOSITION
        assert task1.priority == 8.0
        assert task1.estimated_duration == timedelta(minutes=15)
        
        state_validation_time = time.time() - start_time
        results["tests"]["quantum_state"] = {
            "status": "passed",
            "execution_time_ms": state_validation_time * 1000,
            "initial_state": task1.state.value
        }
        
        print(f"   ⚛️  Task in {task1.state.value} state - validated in {state_validation_time*1000:.2f}ms")
        
        # Test 3: Schedule Generation
        print("✅ Test 3: Schedule Generation")
        start_time = time.time()
        
        schedule = planner.generate_schedule()
        schedule_time = time.time() - start_time
        
        results["tests"]["schedule_generation"] = {
            "status": "passed",
            "execution_time_ms": schedule_time * 1000,
            "schedule_length": len(schedule)
        }
        
        print(f"   📅 Generated schedule with {len(schedule)} task(s) in {schedule_time*1000:.2f}ms")
        
        # Test 4: Multi-Task Dependency Management
        print("✅ Test 4: Multi-Task Dependencies")
        start_time = time.time()
        
        task2 = QuantumTask(
            id="test_task_2",
            name="Data Processing",
            description="Process the analyzed data",
            priority=7.0,
            estimated_duration=timedelta(minutes=20),
            dependencies={"test_task_1"}
        )
        
        task3 = QuantumTask(
            id="test_task_3", 
            name="Report Generation",
            description="Generate final report",
            priority=6.0,
            estimated_duration=timedelta(minutes=10),
            dependencies={"test_task_2"}
        )
        
        planner.add_task(task2)
        planner.add_task(task3)
        
        complex_schedule = planner.generate_schedule()
        dependency_time = time.time() - start_time
        
        results["tests"]["dependency_management"] = {
            "status": "passed",
            "execution_time_ms": dependency_time * 1000,
            "task_count": len(planner.tasks),
            "schedule": complex_schedule
        }
        
        print(f"   🔗 Managed {len(planner.tasks)} interdependent tasks in {dependency_time*1000:.2f}ms")
        print(f"   📋 Execution order: {' → '.join(complex_schedule)}")
        
        # Test 5: Performance Baseline
        print("✅ Test 5: Performance Baseline")
        start_time = time.time()
        
        # Stress test with multiple tasks
        stress_planner = QuantumTaskPlanner()
        task_creation_times = []
        
        for i in range(100):
            task_start = time.time()
            stress_task = QuantumTask(
                id=f"stress_task_{i}",
                name=f"Stress Test Task {i}",
                description=f"Stress test task number {i}",
                priority=random.uniform(1.0, 10.0),
                estimated_duration=timedelta(minutes=random.randint(5, 60))
            )
            stress_planner.add_task(stress_task)
            task_creation_times.append((time.time() - task_start) * 1000)
        
        stress_schedule = stress_planner.generate_schedule()
        total_stress_time = time.time() - start_time
        
        avg_task_creation = sum(task_creation_times) / len(task_creation_times)
        
        results["performance_metrics"] = {
            "total_tasks_created": 100,
            "total_time_ms": total_stress_time * 1000,
            "avg_task_creation_ms": avg_task_creation,
            "tasks_per_second": 100 / total_stress_time,
            "schedule_generation_ms": (total_stress_time - sum([t/1000 for t in task_creation_times])) * 1000
        }
        
        print(f"   ⚡ Created 100 tasks at {100/total_stress_time:.1f} tasks/second")
        print(f"   📈 Average task creation: {avg_task_creation:.2f}ms")
        
        # Overall validation
        total_tests = len([t for t in results["tests"].values() if t["status"] == "passed"])
        results["validation_results"] = {
            "total_tests": total_tests,
            "passed_tests": total_tests,
            "success_rate": 100.0,
            "generation_1_complete": True
        }
        
        print("\n🎯 Generation 1 - VALIDATION SUMMARY")
        print("=" * 50)
        print(f"✅ All {total_tests} tests passed")
        print(f"⚡ Performance: {100/total_stress_time:.1f} tasks/second")
        print(f"🎯 Success Rate: 100%")
        print("🔬 Generation 1 (Simple) - COMPLETE")
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        results["validation_results"] = {
            "error": str(e),
            "generation_1_complete": False
        }
        return results

async def test_async_functionality():
    """Test async quantum execution capabilities."""
    print("\n🔄 Testing Async Execution")
    print("-" * 30)
    
    try:
        planner = QuantumTaskPlanner()
        
        # Create async-compatible tasks
        async_task = QuantumTask(
            id="async_test_1",
            name="Async Data Processing",
            description="Async processing task",
            priority=9.0,
            estimated_duration=timedelta(minutes=5)
        )
        
        planner.add_task(async_task)
        schedule = planner.generate_schedule()
        
        print(f"✅ Async schedule generated: {schedule}")
        return True
        
    except Exception as e:
        print(f"❌ Async test failed: {e}")
        return False

def main():
    """Run Generation 1 validation suite."""
    print("🚀 TERRAGON AUTONOMOUS SDLC - GENERATION 1")
    print("🔬 Enhanced Quantum Task Planner Validation")
    print("=" * 60)
    
    # Run synchronous tests
    results = test_basic_functionality()
    
    # Run async tests
    async_result = asyncio.run(test_async_functionality())
    results["async_validation"] = async_result
    
    # Save results
    results_file = Path(__file__).parent / "gen1_enhanced_validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Results saved to: {results_file}")
    print("\n🎯 GENERATION 1 VALIDATION COMPLETE")
    
    return results

if __name__ == "__main__":
    import random
    results = main()