"""FastAPI controller for Quantum Task Planner."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from ..logging_config import get_logger
from ..quantum_task_planner import (
    QuantumTask,
    QuantumTaskPlanner,
    ResourcePool,
    TaskState,
)

logger = get_logger(__name__)

# Global planner instance
quantum_planner = QuantumTaskPlanner()

router = APIRouter(prefix="/api/v1/quantum", tags=["quantum-planning"])


class TaskRequest(BaseModel):
    """Request model for creating tasks."""

    id: str
    name: str
    description: str
    priority: float = Field(ge=1.0, le=10.0, default=5.0)
    estimated_duration_minutes: int = Field(ge=1, le=1440, default=60)
    required_resources: Dict[str, float] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)


class TaskResponse(BaseModel):
    """Response model for task operations."""

    id: str
    name: str
    description: str
    state: str
    priority: float
    execution_probability: float
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    dependencies: List[str]
    entangled_tasks: List[str]
    error_message: Optional[str] = None


class SystemStateResponse(BaseModel):
    """Response model for system state."""

    total_tasks: int
    resource_utilization: Dict[str, float]
    task_states: Dict[str, Dict[str, Any]]
    execution_history_size: int


class ExecutionResultsResponse(BaseModel):
    """Response model for execution results."""

    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    success_rate: float
    start_time: str
    end_time: str
    task_results: Dict[str, bool]


class ScheduleResponse(BaseModel):
    """Response model for optimal schedule."""

    schedule: List[str]
    estimated_total_duration_minutes: float
    resource_conflicts: int
    dependency_violations: int


def _task_to_response(task: QuantumTask) -> TaskResponse:
    """Convert QuantumTask to TaskResponse."""
    return TaskResponse(
        id=task.id,
        name=task.name,
        description=task.description,
        state=task.state.value,
        priority=task.priority,
        execution_probability=task.get_execution_probability(),
        created_at=task.created_at.isoformat(),
        started_at=task.started_at.isoformat() if task.started_at else None,
        completed_at=task.completed_at.isoformat() if task.completed_at else None,
        dependencies=list(task.dependencies),
        entangled_tasks=list(task.entangled_tasks),
        error_message=task.error_message,
    )


@router.post("/tasks", response_model=TaskResponse)
async def create_task(task_request: TaskRequest) -> TaskResponse:
    """Create a new quantum task."""
    try:
        if task_request.id in quantum_planner.tasks:
            raise HTTPException(
                status_code=400, detail=f"Task {task_request.id} already exists"
            )

        task = QuantumTask(
            id=task_request.id,
            name=task_request.name,
            description=task_request.description,
            priority=task_request.priority,
            estimated_duration=timedelta(
                minutes=task_request.estimated_duration_minutes
            ),
            required_resources=task_request.required_resources,
            dependencies=set(task_request.dependencies),
        )

        quantum_planner.add_task(task)

        # Create dependencies
        for dep_id in task_request.dependencies:
            if dep_id not in quantum_planner.tasks:
                raise HTTPException(
                    status_code=400, detail=f"Dependency task {dep_id} does not exist"
                )
            quantum_planner.create_dependency(task_request.id, dep_id)

        logger.info(f"Created quantum task: {task_request.id}")
        return _task_to_response(task)

    except Exception as e:
        logger.error(f"Failed to create task {task_request.id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks", response_model=List[TaskResponse])
async def list_tasks() -> List[TaskResponse]:
    """List all quantum tasks."""
    try:
        return [_task_to_response(task) for task in quantum_planner.tasks.values()]
    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str) -> TaskResponse:
    """Get a specific quantum task."""
    try:
        if task_id not in quantum_planner.tasks:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        return _task_to_response(quantum_planner.tasks[task_id])
    except Exception as e:
        logger.error(f"Failed to get task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str) -> Dict[str, str]:
    """Delete a quantum task."""
    try:
        if task_id not in quantum_planner.tasks:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        task = quantum_planner.tasks[task_id]
        if task.state == TaskState.EXECUTING:
            raise HTTPException(
                status_code=400, detail=f"Cannot delete executing task {task_id}"
            )

        del quantum_planner.tasks[task_id]

        # Remove from other tasks' dependencies and entanglements
        for other_task in quantum_planner.tasks.values():
            other_task.dependencies.discard(task_id)
            other_task.entangled_tasks.discard(task_id)
            other_task.interference_pattern.pop(task_id, None)

        logger.info(f"Deleted quantum task: {task_id}")
        return {"message": f"Task {task_id} deleted successfully"}

    except Exception as e:
        logger.error(f"Failed to delete task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schedule", response_model=ScheduleResponse)
async def get_optimal_schedule() -> ScheduleResponse:
    """Get the optimal task execution schedule using quantum annealing."""
    try:
        schedule = quantum_planner.quantum_anneal_schedule()

        # Calculate metrics
        total_duration = 0.0
        resource_conflicts = 0
        dependency_violations = 0

        for i, task_id in enumerate(schedule):
            task = quantum_planner.tasks[task_id]
            total_duration += (
                task.estimated_duration.total_seconds() / 60.0
            )  # Convert to minutes

            # Check for dependency violations
            for dep_id in task.dependencies:
                if dep_id in schedule and schedule.index(dep_id) > i:
                    dependency_violations += 1

        logger.info(f"Generated optimal schedule with {len(schedule)} tasks")
        return ScheduleResponse(
            schedule=schedule,
            estimated_total_duration_minutes=total_duration,
            resource_conflicts=resource_conflicts,
            dependency_violations=dependency_violations,
        )

    except Exception as e:
        logger.error(f"Failed to generate schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute", response_model=ExecutionResultsResponse)
async def execute_tasks(
    background_tasks: BackgroundTasks, schedule: Optional[List[str]] = None
) -> ExecutionResultsResponse:
    """Execute tasks according to optimal or provided schedule."""
    try:
        if not quantum_planner.tasks:
            raise HTTPException(status_code=400, detail="No tasks to execute")

        # Execute in background
        logger.info(
            f"Starting quantum task execution with {len(quantum_planner.tasks)} tasks"
        )
        results = await quantum_planner.execute_schedule(schedule)

        return ExecutionResultsResponse(**results)

    except Exception as e:
        logger.error(f"Failed to execute tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/state", response_model=SystemStateResponse)
async def get_system_state() -> SystemStateResponse:
    """Get current quantum planning system state."""
    try:
        state = quantum_planner.get_system_state()
        return SystemStateResponse(**state)
    except Exception as e:
        logger.error(f"Failed to get system state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/resources")
async def get_resource_state() -> Dict[str, Any]:
    """Get current resource pool state."""
    try:
        pool = quantum_planner.resource_pool
        return {
            "total_resources": {
                "cpu_cores": pool.cpu_cores,
                "memory_gb": pool.memory_gb,
                "storage_gb": pool.storage_gb,
                "network_bandwidth": pool.network_bandwidth,
            },
            "allocated_resources": {
                "cpu_cores": pool.allocated_cpu,
                "memory_gb": pool.allocated_memory,
                "storage_gb": pool.allocated_storage,
                "network_bandwidth": pool.allocated_bandwidth,
            },
            "available_resources": {
                "cpu_cores": pool.cpu_cores - pool.allocated_cpu,
                "memory_gb": pool.memory_gb - pool.allocated_memory,
                "storage_gb": pool.storage_gb - pool.allocated_storage,
                "network_bandwidth": pool.network_bandwidth - pool.allocated_bandwidth,
            },
            "utilization_percentage": {
                "cpu": (pool.allocated_cpu / pool.cpu_cores) * 100,
                "memory": (pool.allocated_memory / pool.memory_gb) * 100,
                "storage": (pool.allocated_storage / pool.storage_gb) * 100,
                "bandwidth": (pool.allocated_bandwidth / pool.network_bandwidth) * 100,
            },
        }
    except Exception as e:
        logger.error(f"Failed to get resource state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/reset")
async def reset_system() -> Dict[str, str]:
    """Reset the quantum planning system."""
    try:
        global quantum_planner
        quantum_planner = QuantumTaskPlanner()
        logger.info("Quantum planning system reset")
        return {"message": "System reset successfully"}
    except Exception as e:
        logger.error(f"Failed to reset system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/demo")
async def run_demo() -> Dict[str, Any]:
    """Run a quantum planning demonstration."""
    try:
        from ..quantum_task_planner import demo_quantum_planning

        planner, results = await demo_quantum_planning()

        return {
            "message": "Demo completed successfully",
            "execution_results": results,
            "system_state": planner.get_system_state(),
        }
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
