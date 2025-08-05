"""Tests for Quantum Task Planner."""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock

from src.llm_cost_tracker.quantum_task_planner import (
    QuantumTaskPlanner,
    QuantumTask,
    TaskState,
    ResourcePool
)


class TestResourcePool:
    """Test ResourcePool functionality."""
    
    def test_resource_pool_creation(self):
        """Test resource pool initialization."""
        pool = ResourcePool()
        assert pool.cpu_cores == 4.0
        assert pool.memory_gb == 8.0
        assert pool.allocated_cpu == 0.0
        assert pool.allocated_memory == 0.0
    
    def test_can_allocate_resources(self):
        """Test resource allocation checking."""
        pool = ResourcePool(cpu_cores=4.0, memory_gb=8.0)
        
        # Should be able to allocate within limits
        assert pool.can_allocate({'cpu_cores': 2.0, 'memory_gb': 4.0})
        
        # Should not be able to allocate beyond limits
        assert not pool.can_allocate({'cpu_cores': 6.0, 'memory_gb': 4.0})
        assert not pool.can_allocate({'cpu_cores': 2.0, 'memory_gb': 10.0})
    
    def test_allocate_and_deallocate_resources(self):
        """Test resource allocation and deallocation."""
        pool = ResourcePool(cpu_cores=4.0, memory_gb=8.0)
        requirements = {'cpu_cores': 2.0, 'memory_gb': 4.0}
        
        # Allocate resources
        assert pool.allocate(requirements)
        assert pool.allocated_cpu == 2.0
        assert pool.allocated_memory == 4.0
        
        # Try to allocate more than available
        assert not pool.allocate({'cpu_cores': 3.0, 'memory_gb': 2.0})
        
        # Deallocate resources
        pool.deallocate(requirements)
        assert pool.allocated_cpu == 0.0
        assert pool.allocated_memory == 0.0


class TestQuantumTask:
    """Test QuantumTask functionality."""
    
    def test_quantum_task_creation(self):
        """Test quantum task initialization."""
        task = QuantumTask(
            id="test_task",
            name="Test Task",
            description="A test task",
            priority=8.0
        )
        
        assert task.id == "test_task"
        assert task.name == "Test Task"
        assert task.state == TaskState.SUPERPOSITION
        assert task.priority == 8.0
        assert task.get_execution_probability() == 1.0  # Default amplitude
    
    def test_execution_probability_calculation(self):
        """Test execution probability calculation with interference."""
        task = QuantumTask(
            id="test_task",
            name="Test Task",
            description="A test task"
        )
        
        # Add interference effects
        task.interference_pattern["other_task"] = 0.1
        task.interference_pattern["conflicting_task"] = -0.2
        
        probability = task.get_execution_probability()
        assert 0.0 <= probability <= 1.0
    
    def test_state_collapse(self):
        """Test quantum state collapse."""
        task = QuantumTask(
            id="test_task",
            name="Test Task", 
            description="A test task"
        )
        
        assert task.state == TaskState.SUPERPOSITION
        task.collapse_state(TaskState.EXECUTING)
        assert task.state == TaskState.EXECUTING
    
    def test_task_entanglement(self):
        """Test task entanglement."""
        task1 = QuantumTask(id="task1", name="Task 1", description="First task")
        task2 = QuantumTask(id="task2", name="Task 2", description="Second task")
        
        task1.entangle_with("task2")
        
        assert "task2" in task1.entangled_tasks
        assert task1.state == TaskState.ENTANGLED


class TestQuantumTaskPlanner:
    """Test QuantumTaskPlanner functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.planner = QuantumTaskPlanner()
        
        # Create test tasks
        self.task1 = QuantumTask(
            id="task1",
            name="High Priority Task",
            description="Important task",
            priority=9.0,
            estimated_duration=timedelta(minutes=5),
            required_resources={'cpu_cores': 1.0, 'memory_gb': 2.0}
        )
        
        self.task2 = QuantumTask(
            id="task2", 
            name="Medium Priority Task",
            description="Regular task",
            priority=5.0,
            estimated_duration=timedelta(minutes=10),
            required_resources={'cpu_cores': 2.0, 'memory_gb': 1.0}
        )
        
        self.task3 = QuantumTask(
            id="task3",
            name="Low Priority Task", 
            description="Background task",
            priority=2.0,
            estimated_duration=timedelta(minutes=15),
            required_resources={'memory_gb': 1.0}
        )
    
    def test_add_task(self):
        """Test adding tasks to planner."""
        self.planner.add_task(self.task1)
        
        assert "task1" in self.planner.tasks
        assert self.planner.tasks["task1"] == self.task1
    
    def test_create_dependency(self):
        """Test creating task dependencies."""
        self.planner.add_task(self.task1)
        self.planner.add_task(self.task2)
        
        self.planner.create_dependency("task2", "task1")
        
        assert "task1" in self.task2.dependencies
    
    def test_resource_overlap_calculation(self):
        """Test resource overlap calculation."""
        self.planner.add_task(self.task1)
        self.planner.add_task(self.task2)
        
        overlap = self.planner._calculate_resource_overlap(
            self.task1.required_resources,
            self.task2.required_resources
        )
        
        assert 0.0 <= overlap <= 1.0
    
    def test_schedule_cost_calculation(self):
        """Test schedule cost calculation."""
        self.planner.add_task(self.task1)
        self.planner.add_task(self.task2)
        
        schedule = ["task1", "task2"]
        cost = self.planner._calculate_schedule_cost(schedule)
        
        assert isinstance(cost, float)
        assert cost >= 0.0
    
    def test_quantum_annealing_schedule(self):
        """Test quantum annealing schedule generation."""
        self.planner.add_task(self.task1)
        self.planner.add_task(self.task2)
        self.planner.add_task(self.task3)
        
        schedule = self.planner.quantum_anneal_schedule(max_iterations=100)
        
        assert len(schedule) == 3
        assert set(schedule) == {"task1", "task2", "task3"}
    
    @pytest.mark.asyncio
    async def test_execute_task_success(self):
        """Test successful task execution."""
        self.planner.add_task(self.task1)
        
        # Mock successful execution
        with patch('asyncio.sleep', new_callable=AsyncMock):
            with patch('random.random', return_value=0.1):  # Ensure success
                result = await self.planner.execute_task("task1")
        
        assert result is True
        assert self.task1.state == TaskState.COMPLETED
        assert self.task1.started_at is not None
        assert self.task1.completed_at is not None
    
    @pytest.mark.asyncio
    async def test_execute_task_failure(self):
        """Test task execution failure."""
        self.planner.add_task(self.task1)
        
        # Mock failed execution
        with patch('asyncio.sleep', new_callable=AsyncMock):
            with patch('random.random', return_value=0.9):  # Ensure failure
                result = await self.planner.execute_task("task1")
        
        assert result is False
        assert self.task1.state == TaskState.FAILED
        assert self.task1.error_message is not None
    
    @pytest.mark.asyncio
    async def test_execute_task_dependency_check(self):
        """Test task execution with unsatisfied dependencies."""
        self.planner.add_task(self.task1)
        self.planner.add_task(self.task2)
        self.planner.create_dependency("task2", "task1")
        
        # Try to execute task2 without completing task1
        result = await self.planner.execute_task("task2")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_execute_task_resource_check(self):
        """Test task execution with insufficient resources."""
        # Create task requiring more resources than available
        big_task = QuantumTask(
            id="big_task",
            name="Resource Heavy Task",
            description="Needs lots of resources",
            required_resources={'cpu_cores': 10.0, 'memory_gb': 20.0}
        )
        
        self.planner.add_task(big_task)
        result = await self.planner.execute_task("big_task")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_execute_schedule(self):
        """Test executing a complete schedule."""
        self.planner.add_task(self.task1)
        self.planner.add_task(self.task2)
        
        schedule = ["task1", "task2"]
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            with patch('random.random', return_value=0.1):  # Ensure success
                results = await self.planner.execute_schedule(schedule)
        
        assert results['total_tasks'] == 2
        assert results['successful_tasks'] >= 0
        assert results['failed_tasks'] >= 0
        assert 'start_time' in results
        assert 'end_time' in results
    
    def test_get_system_state(self):
        """Test system state retrieval."""
        self.planner.add_task(self.task1)
        self.planner.add_task(self.task2)
        
        state = self.planner.get_system_state()
        
        assert 'total_tasks' in state
        assert 'resource_utilization' in state
        assert 'task_states' in state
        assert 'execution_history_size' in state
        assert state['total_tasks'] == 2
    
    def test_interference_pattern_calculation(self):
        """Test interference pattern calculation between tasks."""
        self.planner.add_task(self.task1)
        
        # Add second task to trigger interference calculation
        self.planner.add_task(self.task2)
        
        # Check that interference patterns were calculated
        assert len(self.task1.interference_pattern) > 0
        assert len(self.task2.interference_pattern) > 0
        assert "task2" in self.task1.interference_pattern
        assert "task1" in self.task2.interference_pattern
    
    def test_complementary_tasks_detection(self):
        """Test detection of complementary tasks."""
        # Create tasks with shared dependencies
        shared_dep_task = QuantumTask(
            id="shared_dep",
            name="Shared Dependency",
            description="Common dependency"
        )
        
        task_a = QuantumTask(
            id="task_a",
            name="Task A",
            description="First task",
            priority=9.0,
            dependencies={"shared_dep"}
        )
        
        task_b = QuantumTask(
            id="task_b", 
            name="Task B",
            description="Second task",
            priority=9.5,
            dependencies={"shared_dep"}
        )
        
        self.planner.add_task(shared_dep_task)
        self.planner.add_task(task_a)
        self.planner.add_task(task_b)
        
        # Check if tasks are detected as complementary
        are_complementary = self.planner._are_complementary(task_a, task_b)
        assert are_complementary is True
    
    def test_entanglement_decision(self):
        """Test when tasks should be entangled."""
        # Create high priority tasks with overlapping resources
        high_pri_task1 = QuantumTask(
            id="high1",
            name="High Priority 1",
            description="First high priority task",
            priority=9.0,
            required_resources={'cpu_cores': 2.0, 'memory_gb': 4.0}
        )
        
        high_pri_task2 = QuantumTask(
            id="high2",
            name="High Priority 2", 
            description="Second high priority task",
            priority=9.5,
            required_resources={'cpu_cores': 2.5, 'memory_gb': 3.0}
        )
        
        self.planner.add_task(high_pri_task1)
        self.planner.add_task(high_pri_task2)
        
        should_entangle = self.planner._should_entangle("high1", "high2")
        assert isinstance(should_entangle, bool)


@pytest.mark.asyncio
async def test_demo_quantum_planning():
    """Test the demo quantum planning function."""
    from src.llm_cost_tracker.quantum_task_planner import demo_quantum_planning
    
    with patch('asyncio.sleep', new_callable=AsyncMock):
        with patch('random.random', return_value=0.1):  # Ensure success
            planner, results = await demo_quantum_planning()
    
    assert isinstance(planner, QuantumTaskPlanner)
    assert isinstance(results, dict)
    assert 'total_tasks' in results
    assert 'successful_tasks' in results
    assert 'failed_tasks' in results


class TestQuantumTaskIntegration:
    """Integration tests for quantum task planning."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete quantum planning workflow."""
        planner = QuantumTaskPlanner()
        
        # Create a complex set of tasks with dependencies
        tasks = [
            QuantumTask(
                id="data_prep",
                name="Data Preparation",
                description="Prepare training data",
                priority=8.0,
                estimated_duration=timedelta(minutes=10),
                required_resources={'cpu_cores': 1.0, 'memory_gb': 2.0}
            ),
            QuantumTask(
                id="model_training",
                name="Model Training", 
                description="Train ML model",
                priority=9.0,
                estimated_duration=timedelta(minutes=30),
                required_resources={'cpu_cores': 2.0, 'memory_gb': 4.0},
                dependencies={"data_prep"}
            ),
            QuantumTask(
                id="model_eval",
                name="Model Evaluation",
                description="Evaluate trained model",
                priority=7.0,
                estimated_duration=timedelta(minutes=5),
                required_resources={'cpu_cores': 1.0, 'memory_gb': 1.0},
                dependencies={"model_training"}
            ),
            QuantumTask(
                id="report_gen",
                name="Report Generation",
                description="Generate evaluation report",
                priority=6.0,
                estimated_duration=timedelta(minutes=3),
                required_resources={'cpu_cores': 0.5, 'memory_gb': 1.0},
                dependencies={"model_eval"}
            )
        ]
        
        # Add all tasks
        for task in tasks:
            planner.add_task(task)
        
        # Create dependencies
        planner.create_dependency("model_training", "data_prep")
        planner.create_dependency("model_eval", "model_training")
        planner.create_dependency("report_gen", "model_eval")
        
        # Generate optimal schedule
        schedule = planner.quantum_anneal_schedule(max_iterations=500)
        
        # Verify schedule respects dependencies
        data_prep_idx = schedule.index("data_prep")
        model_training_idx = schedule.index("model_training")
        model_eval_idx = schedule.index("model_eval")
        report_gen_idx = schedule.index("report_gen")
        
        assert data_prep_idx < model_training_idx
        assert model_training_idx < model_eval_idx
        assert model_eval_idx < report_gen_idx
        
        # Execute the schedule
        with patch('asyncio.sleep', new_callable=AsyncMock):
            with patch('random.random', return_value=0.1):  # Ensure success
                results = await planner.execute_schedule(schedule)
        
        assert results['total_tasks'] == 4
        assert results['success_rate'] > 0.0
        
        # Verify all tasks completed
        for task in tasks:
            final_task = planner.tasks[task.id]
            assert final_task.state in [TaskState.COMPLETED, TaskState.FAILED]