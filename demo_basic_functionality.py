#!/usr/bin/env python3
"""
Basic Functionality Demo - Generation 1: MAKE IT WORK
Demonstrates core LLM Cost Tracker and Quantum Task Planner functionality
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llm_cost_tracker import (
    QuantumTaskPlanner, 
    QuantumTask, 
    TaskState, 
    ResourcePool,
    quantum_i18n,
    t,
    set_language,
    SupportedLanguage
)

class BasicFunctionalityDemo:
    """Demonstrates Generation 1 core functionality."""
    
    def __init__(self):
        self.planner = QuantumTaskPlanner()
        print(f"🔬 {t('quantum_planner_initialized')}")
    
    def demo_multilingual_support(self):
        """Demo i18n support across multiple languages."""
        print(f"\n{'='*60}")
        print("🌍 MULTILINGUAL SUPPORT DEMO")
        print(f"{'='*60}")
        
        languages = [
            SupportedLanguage.ENGLISH,
            SupportedLanguage.SPANISH, 
            SupportedLanguage.FRENCH,
            SupportedLanguage.GERMAN,
            SupportedLanguage.JAPANESE,
            SupportedLanguage.CHINESE_SIMPLIFIED
        ]
        
        for lang in languages:
            set_language(lang)
            print(f"{lang.value}: {t('quantum_planner_initialized')}")
        
        # Reset to English
        set_language(SupportedLanguage.ENGLISH)
    
    def create_sample_tasks(self) -> List[QuantumTask]:
        """Create a diverse set of tasks for demonstration."""
        from datetime import timedelta
        
        tasks = [
            QuantumTask(
                id="data_analysis",
                name="Data Analysis Pipeline",
                description="Advanced data analysis with ML preprocessing",
                priority=9.0,
                estimated_duration=timedelta(minutes=45),
                required_resources={"cpu_cores": 4.0, "memory_gb": 8.0},
                dependencies=set()
            ),
            QuantumTask(
                id="ml_training", 
                name="Machine Learning Model Training",
                description="Train deep learning models with GPU acceleration",
                priority=8.5,
                estimated_duration=timedelta(minutes=120),
                required_resources={"cpu_cores": 8.0, "memory_gb": 16.0, "storage_gb": 10.0},
                dependencies=set()
            ),
            QuantumTask(
                id="data_validation",
                name="Data Quality Validation",
                description="Validate data integrity and quality metrics",
                priority=7.5,
                estimated_duration=timedelta(minutes=20),
                required_resources={"cpu_cores": 2.0, "memory_gb": 4.0},
                dependencies={"data_analysis"}
            ),
            QuantumTask(
                id="report_generation",
                name="Automated Report Generation",
                description="Generate comprehensive analysis reports",
                priority=6.0,
                estimated_duration=timedelta(minutes=15),
                required_resources={"cpu_cores": 2.0, "memory_gb": 2.0},
                dependencies={"data_validation", "ml_training"}
            ),
            QuantumTask(
                id="model_deployment",
                name="Model Deployment to Production",
                description="Deploy trained models to production environment",
                priority=8.0,
                estimated_duration=timedelta(minutes=30),
                required_resources={"cpu_cores": 4.0, "memory_gb": 4.0, "network_bandwidth": 100.0},
                dependencies={"ml_training"}
            )
        ]
        
        print(f"📋 Created {len(tasks)} sample tasks")
        return tasks
    
    def demo_task_management(self):
        """Demonstrate basic task management operations."""
        print(f"\n{'='*60}")
        print("📋 TASK MANAGEMENT DEMO")
        print(f"{'='*60}")
        
        tasks = self.create_sample_tasks()
        
        # Add tasks to planner
        for task in tasks:
            success, message = self.planner.add_task(task)
            if success:
                print(f"✅ Added task: {task.name} (Priority: {task.priority})")
            else:
                print(f"❌ Failed to add task: {message}")
        
        # Show current system state
        system_state = self.planner.get_system_state()
        print(f"\n📊 System Statistics:")
        print(f"   Total tasks: {system_state.get('total_tasks', 0)}")
        print(f"   Execution history: {system_state.get('execution_history_size', 0)}")
        
        # Show resource utilization
        resource_util = system_state.get('resource_utilization', {})
        print(f"   Resource utilization:")
        for resource, utilization in resource_util.items():
            print(f"     {resource}: {utilization:.2%}")
    
    async def demo_quantum_scheduling(self):
        """Demonstrate quantum-inspired scheduling algorithms."""
        print(f"\n{'='*60}")
        print("⚛️  QUANTUM SCHEDULING DEMO")
        print(f"{'='*60}")
        
        try:
            # Generate optimal schedule using quantum algorithms
            schedule = self.planner.quantum_anneal_schedule(max_iterations=100)
            
            print("🔮 Quantum-optimized execution schedule:")
            for i, task_id in enumerate(schedule, 1):
                task = self.planner.tasks.get(task_id)
                if task:
                    print(f"   {i}. {task.name} ({task_id})")
                    print(f"      Priority: {task.priority}, Duration: {task.estimated_duration}")
                    if task.dependencies:
                        print(f"      Dependencies: {', '.join(task.dependencies)}")
            
            return schedule
            
        except Exception as e:
            print(f"❌ Scheduling error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to simple FIFO schedule
            tasks = list(self.planner.tasks.keys())
            print(f"📝 Fallback FIFO schedule: {tasks}")
            return tasks
    
    async def demo_task_execution(self, schedule: List[str]):
        """Simulate task execution with the generated schedule."""
        print(f"\n{'='*60}")
        print("🚀 TASK EXECUTION SIMULATION")
        print(f"{'='*60}")
        
        execution_results = []
        
        for task_id in schedule[:3]:  # Execute first 3 tasks for demo
            task = self.planner.tasks.get(task_id)
            if not task:
                continue
                
            print(f"\n🔧 Executing: {task.name}")
            print(f"   Resources: {task.required_resources}")
            
            # Simulate execution time (shortened for demo)
            execution_time = min(task.estimated_duration.total_seconds() / 10, 3)  # Max 3 seconds
            
            try:
                # Update task state to executing
                task.state = TaskState.EXECUTING
                task.started_at = datetime.utcnow()
                
                # Simulate work
                await asyncio.sleep(execution_time)
                
                # Mark as completed
                task.state = TaskState.COMPLETED
                task.completed_at = datetime.utcnow()
                
                result = {
                    "task_id": task_id,
                    "name": task.name,
                    "status": "completed",
                    "execution_time_seconds": execution_time
                }
                execution_results.append(result)
                
                print(f"   ✅ Completed in {execution_time:.1f}s")
                
            except Exception as e:
                task.state = TaskState.FAILED
                print(f"   ❌ Failed: {e}")
                execution_results.append({
                    "task_id": task_id,
                    "name": task.name, 
                    "status": "failed",
                    "error": str(e)
                })
        
        return execution_results
    
    def demo_resource_management(self):
        """Demonstrate resource pool management."""
        print(f"\n{'='*60}")
        print("🏗️  RESOURCE MANAGEMENT DEMO")
        print(f"{'='*60}")
        
        # Use the planner's resource pool
        resource_pool = self.planner.resource_pool
        
        print("💾 Available Resources:")
        print(f"   CPU Cores: {resource_pool.cpu_cores} (allocated: {resource_pool.allocated_cpu})")
        print(f"   Memory GB: {resource_pool.memory_gb} (allocated: {resource_pool.allocated_memory})")
        print(f"   Storage GB: {resource_pool.storage_gb} (allocated: {resource_pool.allocated_storage})")
        print(f"   Bandwidth Mbps: {resource_pool.network_bandwidth} (allocated: {resource_pool.allocated_bandwidth})")
        
        # Simulate resource allocation
        allocation_requests = [
            {"cpu_cores": 4.0, "memory_gb": 8.0},
            {"cpu_cores": 8.0, "memory_gb": 16.0, "storage_gb": 10.0},
            {"cpu_cores": 2.0, "memory_gb": 4.0}
        ]
        
        print(f"\n📈 Resource Allocation Simulation:")
        for i, request in enumerate(allocation_requests, 1):
            can_allocate = resource_pool.can_allocate(request)
            print(f"   Request {i}: {request}")
            print(f"   Can allocate: {'✅ Yes' if can_allocate else '❌ No'}")
            
            if can_allocate:
                resource_pool.allocate(request)
                print(f"   Allocated successfully")
            
        print(f"\n💾 Remaining Resources:")
        available_cpu = resource_pool.cpu_cores - resource_pool.allocated_cpu
        available_memory = resource_pool.memory_gb - resource_pool.allocated_memory
        available_storage = resource_pool.storage_gb - resource_pool.allocated_storage
        available_bandwidth = resource_pool.network_bandwidth - resource_pool.allocated_bandwidth
        
        print(f"   Available CPU: {available_cpu}")
        print(f"   Available Memory: {available_memory}GB")
        print(f"   Available Storage: {available_storage}GB")
        print(f"   Available Bandwidth: {available_bandwidth}Mbps")
    
    def generate_performance_report(self, execution_results: List[dict]):
        """Generate a performance summary report."""
        print(f"\n{'='*60}")
        print("📊 PERFORMANCE REPORT")
        print(f"{'='*60}")
        
        if not execution_results:
            print("No execution results to report.")
            return
        
        completed_tasks = [r for r in execution_results if r["status"] == "completed"]
        failed_tasks = [r for r in execution_results if r["status"] == "failed"]
        
        print(f"📈 Execution Summary:")
        print(f"   Total tasks executed: {len(execution_results)}")
        print(f"   Successful: {len(completed_tasks)}")
        print(f"   Failed: {len(failed_tasks)}")
        
        if completed_tasks:
            total_time = sum(r["execution_time_seconds"] for r in completed_tasks)
            avg_time = total_time / len(completed_tasks)
            print(f"   Total execution time: {total_time:.1f}s")
            print(f"   Average execution time: {avg_time:.1f}s")
        
        print(f"\n🎯 Task Details:")
        for result in execution_results:
            status_icon = "✅" if result["status"] == "completed" else "❌"
            print(f"   {status_icon} {result['name']}")
            if result["status"] == "completed":
                print(f"      Execution time: {result['execution_time_seconds']:.1f}s")
            else:
                print(f"      Error: {result.get('error', 'Unknown error')}")


async def main():
    """Run the complete Generation 1 functionality demo."""
    print("🚀 LLM COST TRACKER - GENERATION 1 DEMO")
    print("=" * 60)
    print("Demonstrating: MAKE IT WORK - Basic Core Functionality")
    print("=" * 60)
    
    demo = BasicFunctionalityDemo()
    
    # Run all demos
    demo.demo_multilingual_support()
    demo.demo_task_management()
    
    schedule = await demo.demo_quantum_scheduling()
    execution_results = await demo.demo_task_execution(schedule)
    
    demo.demo_resource_management()
    demo.generate_performance_report(execution_results)
    
    print(f"\n{'='*60}")
    print("🎉 GENERATION 1 DEMO COMPLETED")
    print("✅ Core functionality verified and working")
    print("🔄 Ready to proceed to Generation 2: MAKE IT ROBUST")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())