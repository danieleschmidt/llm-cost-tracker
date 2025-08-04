"""Advanced concurrency management with resource pooling and load balancing."""

import asyncio
import time
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import functools
import threading
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Task definition with metadata."""
    id: str
    func: Callable
    args: tuple
    kwargs: dict
    priority: TaskPriority
    created_at: float
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        """Compare tasks by priority and creation time."""
        if self.priority.value == other.priority.value:
            return self.created_at < other.created_at
        return self.priority.value > other.priority.value


class ResourcePool:
    """Generic resource pool for managing expensive resources."""
    
    def __init__(
        self,
        create_resource: Callable,
        destroy_resource: Callable,
        max_size: int = 10,
        min_size: int = 2,
        max_idle_time: float = 300,  # 5 minutes
        health_check: Optional[Callable] = None
    ):
        self.create_resource = create_resource
        self.destroy_resource = destroy_resource
        self.max_size = max_size
        self.min_size = min_size
        self.max_idle_time = max_idle_time
        self.health_check = health_check
        
        self.pool: Queue = Queue(maxsize=max_size)
        self.active_resources = 0
        self.total_created = 0
        self.total_destroyed = 0
        self.lock = threading.Lock()
        
        # Pre-populate with minimum resources
        for _ in range(min_size):
            self._create_and_add_resource()
    
    def _create_and_add_resource(self) -> None:
        """Create a new resource and add it to the pool."""
        try:
            resource = self.create_resource()
            self.pool.put((resource, time.time()))
            self.total_created += 1
            logger.debug(f"Created new resource, total: {self.total_created}")
        except Exception as e:
            logger.error(f"Failed to create resource: {e}")
    
    def acquire(self, timeout: float = 30) -> Any:
        """Acquire a resource from the pool."""
        try:
            # Try to get from pool first
            resource, created_at = self.pool.get(timeout=timeout)
            
            # Check if resource is too old or unhealthy
            if (time.time() - created_at > self.max_idle_time or
                (self.health_check and not self.health_check(resource))):
                
                # Destroy old/unhealthy resource
                try:
                    self.destroy_resource(resource)
                    self.total_destroyed += 1
                except Exception as e:
                    logger.error(f"Failed to destroy resource: {e}")
                
                # Create new resource
                resource = self.create_resource()
                self.total_created += 1
            
            with self.lock:
                self.active_resources += 1
            
            return resource
            
        except Empty:
            # Pool is empty, try to create new resource if under max
            with self.lock:
                if self.active_resources < self.max_size:
                    resource = self.create_resource()
                    self.total_created += 1
                    self.active_resources += 1
                    return resource
            
            raise TimeoutError("No resources available and pool is at max capacity")
    
    def release(self, resource: Any) -> None:
        """Release a resource back to the pool."""
        with self.lock:
            self.active_resources -= 1
        
        try:
            # Health check before returning to pool
            if self.health_check and not self.health_check(resource):
                self.destroy_resource(resource)
                self.total_destroyed += 1
                # Create replacement if below minimum
                if self.pool.qsize() + self.active_resources < self.min_size:
                    self._create_and_add_resource()
                return
            
            # Return to pool with current timestamp
            self.pool.put((resource, time.time()), timeout=1)
            
        except Exception as e:
            logger.error(f"Failed to release resource: {e}")
            try:
                self.destroy_resource(resource)
                self.total_destroyed += 1
            except Exception as destroy_error:
                logger.error(f"Failed to destroy resource during cleanup: {destroy_error}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "pool_size": self.pool.qsize(),
            "active_resources": self.active_resources,
            "max_size": self.max_size,
            "min_size": self.min_size,
            "total_created": self.total_created,
            "total_destroyed": self.total_destroyed,
            "utilization": self.active_resources / self.max_size if self.max_size > 0 else 0
        }
    
    def cleanup(self) -> None:
        """Cleanup all resources in the pool."""
        while not self.pool.empty():
            try:
                resource, _ = self.pool.get_nowait()
                self.destroy_resource(resource)
                self.total_destroyed += 1
            except Empty:
                break
            except Exception as e:
                logger.error(f"Error during pool cleanup: {e}")


class AsyncTaskQueue:
    """Advanced async task queue with priority and concurrency control."""
    
    def __init__(
        self,
        max_concurrent_tasks: int = 10,
        max_queue_size: int = 1000,
        worker_pool_size: int = 4
    ):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_queue_size = max_queue_size
        self.worker_pool_size = worker_pool_size
        
        self.task_queue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self.active_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_execution_time = 0
        
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.executor = ThreadPoolExecutor(max_workers=worker_pool_size)
        self.workers_started = False
        self.shutdown_event = asyncio.Event()
    
    async def start_workers(self) -> None:
        """Start background worker tasks."""
        if self.workers_started:
            return
        
        # Start worker tasks
        for i in range(self.max_concurrent_tasks):
            asyncio.create_task(self._worker(f"worker-{i}"))
        
        self.workers_started = True
        logger.info(f"Started {self.max_concurrent_tasks} async workers")
    
    async def _worker(self, worker_name: str) -> None:
        """Background worker that processes tasks from queue."""
        while not self.shutdown_event.is_set():
            try:
                # Get task from queue with timeout
                task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )
                
                await self._execute_task(task, worker_name)
                self.task_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
    
    async def _execute_task(self, task: Task, worker_name: str) -> None:
        """Execute a single task with error handling and retries."""
        async with self.semaphore:
            start_time = time.time()
            
            try:
                # Set timeout if specified
                if task.timeout:
                    result = await asyncio.wait_for(
                        self._run_task_function(task),
                        timeout=task.timeout
                    )
                else:
                    result = await self._run_task_function(task)
                
                execution_time = time.time() - start_time
                self.total_execution_time += execution_time
                self.completed_tasks += 1
                
                logger.debug(f"Task {task.id} completed by {worker_name} in {execution_time:.2f}s")
                
            except Exception as e:
                execution_time = time.time() - start_time
                self.total_execution_time += execution_time
                
                # Retry logic
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    logger.warning(f"Task {task.id} failed, retrying ({task.retry_count}/{task.max_retries}): {e}")
                    await self.submit_task(task)
                else:
                    self.failed_tasks += 1
                    logger.error(f"Task {task.id} failed permanently after {task.max_retries} retries: {e}")
    
    async def _run_task_function(self, task: Task) -> Any:
        """Run task function, handling both sync and async functions."""
        if asyncio.iscoroutinefunction(task.func):
            return await task.func(*task.args, **task.kwargs)
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                functools.partial(task.func, *task.args, **task.kwargs)
            )
    
    async def submit_task(
        self,
        task: Union[Task, Callable],
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
        max_retries: int = 3,
        **kwargs
    ) -> None:
        """Submit a task to the queue."""
        # Ensure workers are started
        if not self.workers_started:
            await self.start_workers()
        
        # Create Task object if callable was passed
        if not isinstance(task, Task):
            task_obj = Task(
                id=f"task-{int(time.time() * 1000000)}",
                func=task,
                args=args,
                kwargs=kwargs,
                priority=priority,
                created_at=time.time(),
                timeout=timeout,
                max_retries=max_retries
            )
        else:
            task_obj = task
        
        try:
            await self.task_queue.put(task_obj)
            logger.debug(f"Submitted task {task_obj.id} with priority {priority.name}")
        except asyncio.QueueFull:
            logger.error(f"Task queue full, dropping task {task_obj.id}")
            raise
    
    async def wait_for_completion(self) -> None:
        """Wait for all queued tasks to complete."""
        await self.task_queue.join()
    
    async def shutdown(self) -> None:
        """Shutdown the task queue and cleanup resources."""
        logger.info("Shutting down task queue...")
        
        # Signal shutdown to workers
        self.shutdown_event.set()
        
        # Wait for current tasks to complete
        await self.wait_for_completion()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        logger.info("Task queue shutdown complete")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task queue statistics."""
        total_tasks = self.completed_tasks + self.failed_tasks
        avg_execution_time = (
            self.total_execution_time / total_tasks
            if total_tasks > 0 else 0
        )
        
        return {
            "queue_size": self.task_queue.qsize(),
            "active_tasks": self.active_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": total_tasks,
            "success_rate": self.completed_tasks / total_tasks if total_tasks > 0 else 0,
            "avg_execution_time": avg_execution_time,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "workers_started": self.workers_started
        }


class BatchProcessor:
    """Batch processing with adaptive batch sizing."""
    
    def __init__(
        self,
        batch_function: Callable,
        min_batch_size: int = 1,
        max_batch_size: int = 100,
        batch_timeout: float = 5.0,
        adaptive_sizing: bool = True
    ):
        self.batch_function = batch_function
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.adaptive_sizing = adaptive_sizing
        
        self.current_batch = []
        self.batch_lock = asyncio.Lock()
        self.last_batch_time = time.time()
        
        # Adaptive batch sizing metrics
        self.avg_processing_time = 1.0
        self.throughput_history = []
        self.optimal_batch_size = min_batch_size
    
    async def add_item(self, item: Any) -> None:
        """Add item to current batch."""
        async with self.batch_lock:
            self.current_batch.append(item)
            
            # Check if we should process batch
            should_process = (
                len(self.current_batch) >= self.optimal_batch_size or
                len(self.current_batch) >= self.max_batch_size or
                (len(self.current_batch) >= self.min_batch_size and
                 time.time() - self.last_batch_time > self.batch_timeout)
            )
            
            if should_process:
                await self._process_current_batch()
    
    async def _process_current_batch(self) -> None:
        """Process the current batch."""
        if not self.current_batch:
            return
        
        batch_to_process = self.current_batch.copy()
        self.current_batch.clear()
        self.last_batch_time = time.time()
        
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(self.batch_function):
                await self.batch_function(batch_to_process)
            else:
                self.batch_function(batch_to_process)
            
            processing_time = time.time() - start_time
            
            # Update metrics for adaptive sizing
            if self.adaptive_sizing:
                self._update_adaptive_metrics(len(batch_to_process), processing_time)
            
            logger.debug(f"Processed batch of {len(batch_to_process)} items in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
    
    def _update_adaptive_metrics(self, batch_size: int, processing_time: float) -> None:
        """Update metrics for adaptive batch sizing."""
        # Update average processing time
        self.avg_processing_time = (self.avg_processing_time * 0.9 + processing_time * 0.1)
        
        # Calculate throughput (items per second)
        throughput = batch_size / processing_time if processing_time > 0 else 0
        self.throughput_history.append(throughput)
        
        # Keep only recent history
        if len(self.throughput_history) > 10:
            self.throughput_history.pop(0)
        
        # Adjust optimal batch size based on throughput
        if len(self.throughput_history) >= 3:
            recent_avg_throughput = sum(self.throughput_history[-3:]) / 3
            
            # If throughput is decreasing, reduce batch size
            if (len(self.throughput_history) >= 6 and
                recent_avg_throughput < sum(self.throughput_history[-6:-3]) / 3):
                self.optimal_batch_size = max(
                    self.min_batch_size,
                    int(self.optimal_batch_size * 0.8)
                )
            # If throughput is stable/increasing, gradually increase batch size
            elif processing_time < self.batch_timeout * 0.5:
                self.optimal_batch_size = min(
                    self.max_batch_size,
                    int(self.optimal_batch_size * 1.1)
                )
    
    async def flush(self) -> None:
        """Process any remaining items in the batch."""
        async with self.batch_lock:
            if self.current_batch:
                await self._process_current_batch()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processor statistics."""
        return {
            "current_batch_size": len(self.current_batch),
            "optimal_batch_size": self.optimal_batch_size,
            "min_batch_size": self.min_batch_size,
            "max_batch_size": self.max_batch_size,
            "avg_processing_time": self.avg_processing_time,
            "avg_throughput": sum(self.throughput_history) / len(self.throughput_history) if self.throughput_history else 0,
            "adaptive_sizing": self.adaptive_sizing
        }


# Global instances
task_queue = AsyncTaskQueue(max_concurrent_tasks=20, max_queue_size=10000)