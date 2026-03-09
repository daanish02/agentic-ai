# Background Processes

## Table of Contents

- [Introduction](#introduction)
- [Long-Running Tasks](#long-running-tasks)
- [Async Execution Models](#async-execution-models)
- [Process Management](#process-management)
- [Status Tracking](#status-tracking)
- [Background Workers](#background-workers)
- [Job Queues](#job-queues)
- [Task Scheduling](#task-scheduling)
- [Foreground-Background Coordination](#foreground-background-coordination)
- [Progress Monitoring](#progress-monitoring)
- [Cancellation and Timeout](#cancellation-and-timeout)
- [Result Handling](#result-handling)
- [Error Recovery](#error-recovery)
- [Real-World Patterns](#real-world-patterns)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Background processes enable agents to handle long-running tasks without blocking interactive operations. Instead of making users wait for expensive computations or lengthy I/O operations, background execution allows:

- **Non-blocking operations**: Start tasks and continue working
- **Long-duration processing**: Handle tasks that take minutes or hours
- **Periodic execution**: Run tasks on schedules
- **Resource-intensive work**: Isolate heavy computations
- **Asynchronous workflows**: Decouple request from completion

> "Background processes turn synchronous agents into responsive, scalable systems."

The challenge is coordinating foreground and background work, tracking status, handling failures, and managing resources effectively.

### Background vs Foreground

```
Foreground (Synchronous):
User Request → Process → Return Result
         |________________|
          User waits here

Background (Asynchronous):
User Request → Start Process → Return Job ID
                    |
                    ↓
            Process continues...
                    |
                    ↓
            Store Result
                    |
                    ↓
User Polls/Notified ← Result Ready
```

## Long-Running Tasks

Handling tasks that take significant time to complete.

### Background Task Manager

```python
import threading
import queue
import time
import uuid
from typing import Callable, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class BackgroundTask:
    """Background task metadata"""
    task_id: str
    func: Callable
    args: tuple
    kwargs: dict
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    
    def duration(self) -> Optional[float]:
        """Get task duration in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.now() - self.started_at).total_seconds()
        return None

class BackgroundTaskManager:
    """Manage background task execution"""
    
    def __init__(self, num_workers: int = 4):
        self.tasks: dict[str, BackgroundTask] = {}
        self.task_queue = queue.Queue()
        self.num_workers = num_workers
        self.workers = []
        self.running = False
        self.lock = threading.Lock()
    
    def start(self):
        """Start background workers"""
        self.running = True
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        print(f"Started {self.num_workers} background workers")
    
    def submit(
        self,
        func: Callable,
        *args,
        task_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Submit task for background execution"""
        if task_id is None:
            task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        task = BackgroundTask(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs
        )
        
        with self.lock:
            self.tasks[task_id] = task
        
        self.task_queue.put(task_id)
        print(f"Submitted task: {task_id}")
        
        return task_id
    
    def _worker(self, worker_id: int):
        """Background worker thread"""
        while self.running:
            try:
                # Get task with timeout so we can check running flag
                task_id = self.task_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            with self.lock:
                task = self.tasks.get(task_id)
            
            if not task:
                continue
            
            # Execute task
            print(f"Worker {worker_id} executing {task_id}")
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            try:
                result = task.func(*task.args, **task.kwargs)
                task.result = result
                task.status = TaskStatus.COMPLETED
                print(f"Task {task_id} completed")
                
            except Exception as e:
                task.error = str(e)
                task.status = TaskStatus.FAILED
                print(f"Task {task_id} failed: {e}")
            
            finally:
                task.completed_at = datetime.now()
                self.task_queue.task_done()
    
    def get_status(self, task_id: str) -> Optional[BackgroundTask]:
        """Get task status"""
        with self.lock:
            return self.tasks.get(task_id)
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for task completion and get result"""
        start_time = time.time()
        
        while True:
            task = self.get_status(task_id)
            
            if not task:
                raise ValueError(f"Task not found: {task_id}")
            
            if task.status == TaskStatus.COMPLETED:
                return task.result
            
            if task.status == TaskStatus.FAILED:
                raise RuntimeError(f"Task failed: {task.error}")
            
            if task.status == TaskStatus.CANCELLED:
                raise RuntimeError("Task was cancelled")
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")
            
            time.sleep(0.1)
    
    def cancel(self, task_id: str) -> bool:
        """Cancel pending task"""
        with self.lock:
            task = self.tasks.get(task_id)
            if task and task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                return True
        return False
    
    def list_tasks(
        self,
        status: Optional[TaskStatus] = None
    ) -> list[BackgroundTask]:
        """List tasks, optionally filtered by status"""
        with self.lock:
            tasks = list(self.tasks.values())
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        return tasks
    
    def stop(self, wait: bool = True):
        """Stop background workers"""
        self.running = False
        
        if wait:
            for worker in self.workers:
                worker.join()
        
        print("Stopped background workers")

# Example usage
def long_running_task(task_name: str, duration: int):
    """Simulate long-running task"""
    print(f"Starting {task_name}...")
    for i in range(duration):
        time.sleep(1)
        print(f"{task_name}: {i+1}/{duration}")
    return f"{task_name} completed!"

# Start manager
manager = BackgroundTaskManager(num_workers=2)
manager.start()

# Submit tasks
task_id_1 = manager.submit(long_running_task, "Task A", 5)
task_id_2 = manager.submit(long_running_task, "Task B", 3)

# Check status
time.sleep(2)
status = manager.get_status(task_id_1)
print(f"Task {task_id_1} status: {status.status.value}")

# Wait for completion
result = manager.get_result(task_id_2)
print(f"Result: {result}")

# Clean up
manager.stop()
```

### Progress Tracking

```python
from typing import Callable

class ProgressTracker:
    """Track task progress"""
    
    def __init__(self):
        self.progress = {}
        self.lock = threading.Lock()
    
    def update(self, task_id: str, progress: float, message: str = ""):
        """Update task progress"""
        with self.lock:
            self.progress[task_id] = {
                "progress": progress,
                "message": message,
                "updated_at": datetime.now()
            }
    
    def get(self, task_id: str) -> Optional[dict]:
        """Get task progress"""
        with self.lock:
            return self.progress.get(task_id)

# Global tracker
progress_tracker = ProgressTracker()

def task_with_progress(task_id: str, num_steps: int):
    """Task that reports progress"""
    for i in range(num_steps):
        # Do work
        time.sleep(0.5)
        
        # Report progress
        progress = (i + 1) / num_steps
        progress_tracker.update(
            task_id,
            progress,
            f"Step {i+1}/{num_steps}"
        )
    
    return "Completed"

# Submit task
task_id = manager.submit(task_with_progress, "task_001", 10)

# Monitor progress
while True:
    progress_info = progress_tracker.get(task_id)
    if progress_info:
        print(f"Progress: {progress_info['progress']:.0%} - {progress_info['message']}")
        
        if progress_info['progress'] >= 1.0:
            break
    
    time.sleep(1)
```

## Async Execution Models

Different models for asynchronous execution.

### Future Pattern

```python
from concurrent.futures import Future
from typing import Callable, Any

class FutureTask:
    """Task that returns a Future"""
    
    def __init__(self):
        self.executor = BackgroundTaskManager(num_workers=4)
        self.executor.start()
    
    def submit(self, func: Callable, *args, **kwargs) -> Future:
        """Submit task and get Future"""
        future = Future()
        
        def wrapper():
            try:
                result = func(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
        
        task_id = self.executor.submit(wrapper)
        
        # Store task_id in future for tracking
        future.task_id = task_id
        
        return future
    
    def shutdown(self):
        """Shutdown executor"""
        self.executor.stop()

# Usage
executor = FutureTask()

# Submit task
future = executor.submit(long_running_task, "Async Task", 3)

# Do other work
print("Doing other work while task runs...")

# Wait for result
result = future.result(timeout=10)
print(f"Got result: {result}")

executor.shutdown()
```

### Promise Pattern

```python
from typing import Callable, Any, Optional
import threading

class Promise:
    """Promise-style async execution"""
    
    def __init__(self, executor: Callable[[], Any]):
        self.executor = executor
        self.result = None
        self.error = None
        self.completed = False
        self.lock = threading.Lock()
        self.then_handlers = []
        self.catch_handlers = []
        
        # Start execution
        thread = threading.Thread(target=self._execute, daemon=True)
        thread.start()
    
    def _execute(self):
        """Execute and handle callbacks"""
        try:
            result = self.executor()
            
            with self.lock:
                self.result = result
                self.completed = True
            
            # Call then handlers
            for handler in self.then_handlers:
                handler(result)
                
        except Exception as e:
            with self.lock:
                self.error = e
                self.completed = True
            
            # Call catch handlers
            for handler in self.catch_handlers:
                handler(e)
    
    def then(self, handler: Callable[[Any], None]) -> "Promise":
        """Register success handler"""
        with self.lock:
            if self.completed and self.error is None:
                # Already completed successfully
                handler(self.result)
            else:
                self.then_handlers.append(handler)
        return self
    
    def catch(self, handler: Callable[[Exception], None]) -> "Promise":
        """Register error handler"""
        with self.lock:
            if self.completed and self.error is not None:
                # Already failed
                handler(self.error)
            else:
                self.catch_handlers.append(handler)
        return self
    
    def wait(self, timeout: Optional[float] = None) -> Any:
        """Wait for completion"""
        start_time = time.time()
        
        while not self.completed:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError("Promise did not complete in time")
            time.sleep(0.01)
        
        if self.error:
            raise self.error
        
        return self.result

# Usage
def async_operation():
    """Async operation"""
    time.sleep(2)
    return "Success!"

promise = Promise(async_operation)

# Register callbacks
promise.then(lambda result: print(f"Got result: {result}"))
promise.catch(lambda error: print(f"Got error: {error}"))

# Or wait for result
result = promise.wait(timeout=5)
```

## Process Management

Managing background processes.

### Process Pool Manager

```python
from multiprocessing import Process, Queue, Manager
from typing import Callable, Any
import os

class ProcessManager:
    """Manage background processes"""
    
    def __init__(self):
        self.processes = {}
        self.manager = Manager()
        self.results = self.manager.dict()
        self.status = self.manager.dict()
    
    def start_process(
        self,
        process_id: str,
        func: Callable,
        *args,
        **kwargs
    ) -> str:
        """Start background process"""
        
        def wrapper(process_id, results, status, func, args, kwargs):
            """Wrapper to capture result"""
            try:
                status[process_id] = "running"
                result = func(*args, **kwargs)
                results[process_id] = {"success": True, "result": result}
                status[process_id] = "completed"
            except Exception as e:
                results[process_id] = {"success": False, "error": str(e)}
                status[process_id] = "failed"
        
        process = Process(
            target=wrapper,
            args=(process_id, self.results, self.status, func, args, kwargs)
        )
        
        process.start()
        self.processes[process_id] = process
        self.status[process_id] = "started"
        
        print(f"Started process {process_id} (PID: {process.pid})")
        return process_id
    
    def get_status(self, process_id: str) -> str:
        """Get process status"""
        return self.status.get(process_id, "unknown")
    
    def get_result(self, process_id: str, timeout: Optional[float] = None) -> Any:
        """Get process result"""
        process = self.processes.get(process_id)
        
        if not process:
            raise ValueError(f"Process not found: {process_id}")
        
        # Wait for process
        process.join(timeout=timeout)
        
        if process.is_alive():
            raise TimeoutError(f"Process {process_id} did not complete in time")
        
        result = self.results.get(process_id)
        
        if not result:
            raise RuntimeError("Process result not available")
        
        if not result["success"]:
            raise RuntimeError(f"Process failed: {result['error']}")
        
        return result["result"]
    
    def terminate(self, process_id: str):
        """Terminate process"""
        process = self.processes.get(process_id)
        if process and process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
            self.status[process_id] = "terminated"
    
    def list_processes(self) -> dict:
        """List all processes"""
        return {
            pid: {
                "status": self.status.get(pid, "unknown"),
                "alive": p.is_alive()
            }
            for pid, p in self.processes.items()
        }
    
    def cleanup(self):
        """Clean up completed processes"""
        for process_id, process in list(self.processes.items()):
            if not process.is_alive():
                process.join()
                del self.processes[process_id]

# Example: CPU-intensive background task
def cpu_intensive(n: int):
    """CPU-intensive computation"""
    result = 0
    for i in range(n):
        result += i ** 2
    return result

manager = ProcessManager()

# Start process
pid = manager.start_process("compute_1", cpu_intensive, 1000000)

# Check status
time.sleep(1)
print(f"Status: {manager.get_status(pid)}")

# Get result
result = manager.get_result(pid, timeout=10)
print(f"Result: {result}")

manager.cleanup()
```

## Background Workers

Dedicated worker processes/threads for background tasks.

### Worker Pool

```python
import threading
import queue
from typing import Callable

class WorkerPool:
    """Pool of background workers"""
    
    def __init__(self, num_workers: int = 4, worker_type: str = "thread"):
        self.num_workers = num_workers
        self.worker_type = worker_type
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers = []
        self.running = False
    
    def start(self):
        """Start worker pool"""
        self.running = True
        
        for i in range(self.num_workers):
            if self.worker_type == "thread":
                worker = threading.Thread(
                    target=self._worker,
                    args=(i,),
                    daemon=True
                )
            else:  # process
                worker = Process(target=self._worker, args=(i,))
            
            worker.start()
            self.workers.append(worker)
        
        print(f"Started {self.num_workers} {self.worker_type} workers")
    
    def _worker(self, worker_id: int):
        """Worker function"""
        print(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Get task
                task = self.task_queue.get(timeout=1)
                
                if task is None:  # Poison pill
                    break
                
                task_id, func, args, kwargs = task
                
                # Execute task
                try:
                    result = func(*args, **kwargs)
                    self.result_queue.put((task_id, {"success": True, "result": result}))
                except Exception as e:
                    self.result_queue.put((task_id, {"success": False, "error": str(e)}))
                
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
        
        print(f"Worker {worker_id} stopped")
    
    def submit(self, task_id: str, func: Callable, *args, **kwargs):
        """Submit task to pool"""
        self.task_queue.put((task_id, func, args, kwargs))
    
    def get_result(self, timeout: Optional[float] = None) -> tuple:
        """Get next result"""
        return self.result_queue.get(timeout=timeout)
    
    def stop(self):
        """Stop worker pool"""
        self.running = False
        
        # Send poison pills
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        
        # Wait for workers
        for worker in self.workers:
            if isinstance(worker, threading.Thread):
                worker.join()
            else:
                worker.join()

# Usage
pool = WorkerPool(num_workers=4)
pool.start()

# Submit tasks
for i in range(10):
    pool.submit(f"task_{i}", time.sleep, 1)

# Get results
for i in range(10):
    task_id, result = pool.get_result(timeout=20)
    print(f"{task_id}: {result}")

pool.stop()
```

## Job Queues

Persistent job queues for reliable background processing.

### Persistent Job Queue

```python
import sqlite3
import json
from datetime import datetime
from typing import Optional, List

class JobQueue:
    """Persistent job queue using SQLite"""
    
    def __init__(self, db_path: str = "jobs.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                func_name TEXT NOT NULL,
                args TEXT NOT NULL,
                kwargs TEXT NOT NULL,
                status TEXT NOT NULL,
                priority INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                result TEXT,
                error TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def enqueue(
        self,
        job_id: str,
        func_name: str,
        args: tuple = (),
        kwargs: dict = None,
        priority: int = 0
    ):
        """Add job to queue"""
        kwargs = kwargs or {}
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO jobs (job_id, func_name, args, kwargs, status, priority)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            job_id,
            func_name,
            json.dumps(args),
            json.dumps(kwargs),
            "pending",
            priority
        ))
        conn.commit()
        conn.close()
        
        print(f"Enqueued job: {job_id}")
    
    def dequeue(self) -> Optional[dict]:
        """Get next job from queue"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        # Get highest priority pending job
        cursor = conn.execute("""
            SELECT * FROM jobs
            WHERE status = 'pending'
            ORDER BY priority DESC, created_at ASC
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        
        if row:
            job = dict(row)
            
            # Mark as running
            conn.execute("""
                UPDATE jobs
                SET status = 'running', started_at = CURRENT_TIMESTAMP
                WHERE job_id = ?
            """, (job["job_id"],))
            conn.commit()
        else:
            job = None
        
        conn.close()
        return job
    
    def complete(self, job_id: str, result: Any):
        """Mark job as completed"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE jobs
            SET status = 'completed',
                completed_at = CURRENT_TIMESTAMP,
                result = ?
            WHERE job_id = ?
        """, (json.dumps(result), job_id))
        conn.commit()
        conn.close()
    
    def fail(self, job_id: str, error: str):
        """Mark job as failed"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE jobs
            SET status = 'failed',
                completed_at = CURRENT_TIMESTAMP,
                error = ?
            WHERE job_id = ?
        """, (error, job_id))
        conn.commit()
        conn.close()
    
    def get_status(self, job_id: str) -> Optional[dict]:
        """Get job status"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?",
            (job_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def list_jobs(self, status: Optional[str] = None) -> List[dict]:
        """List jobs"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        if status:
            cursor = conn.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC",
                (status,)
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC"
            )
        
        jobs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return jobs

# Worker that processes jobs from queue
class QueueWorker:
    """Worker that processes jobs from queue"""
    
    def __init__(self, queue: JobQueue, registry: dict):
        self.queue = queue
        self.registry = registry  # Map of function names to functions
        self.running = False
    
    def start(self):
        """Start processing jobs"""
        self.running = True
        
        while self.running:
            job = self.queue.dequeue()
            
            if not job:
                time.sleep(1)
                continue
            
            # Get function
            func = self.registry.get(job["func_name"])
            
            if not func:
                self.queue.fail(job["job_id"], f"Function not found: {job['func_name']}")
                continue
            
            # Execute
            try:
                args = json.loads(job["args"])
                kwargs = json.loads(job["kwargs"])
                
                result = func(*args, **kwargs)
                self.queue.complete(job["job_id"], result)
                
            except Exception as e:
                self.queue.fail(job["job_id"], str(e))
    
    def stop(self):
        """Stop worker"""
        self.running = False

# Usage
queue = JobQueue()

# Register functions
registry = {
    "process_data": lambda x: x * 2,
    "send_email": lambda to, subject: f"Email sent to {to}: {subject}"
}

# Enqueue jobs
queue.enqueue("job_1", "process_data", args=(42,), priority=1)
queue.enqueue("job_2", "send_email", kwargs={"to": "user@example.com", "subject": "Hello"})

# Start worker
worker = QueueWorker(queue, registry)
threading.Thread(target=worker.start, daemon=True).start()

# Check status
time.sleep(2)
status = queue.get_status("job_1")
print(f"Job 1 status: {status['status']}")
```

## Task Scheduling

Scheduling tasks to run at specific times or intervals.

### Task Scheduler

```python
from datetime import datetime, timedelta
import sched
import time
from typing import Callable

class TaskScheduler:
    """Schedule tasks for future execution"""
    
    def __init__(self):
        self.scheduler = sched.scheduler(time.time, time.sleep)
        self.running = False
        self.thread = None
    
    def schedule_once(
        self,
        delay_seconds: float,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Schedule task to run once after delay"""
        return self.scheduler.enter(
            delay_seconds,
            1,  # Priority
            func,
            args,
            kwargs
        )
    
    def schedule_at(
        self,
        run_time: datetime,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Schedule task to run at specific time"""
        now = datetime.now()
        delay = (run_time - now).total_seconds()
        
        if delay < 0:
            raise ValueError("Cannot schedule in the past")
        
        return self.schedule_once(delay, func, *args, **kwargs)
    
    def schedule_recurring(
        self,
        interval_seconds: float,
        func: Callable,
        *args,
        **kwargs
    ):
        """Schedule task to run repeatedly"""
        
        def recurring_wrapper():
            """Wrapper that reschedules itself"""
            func(*args, **kwargs)
            # Reschedule
            if self.running:
                self.schedule_once(interval_seconds, recurring_wrapper)
        
        self.schedule_once(interval_seconds, recurring_wrapper)
    
    def start(self):
        """Start scheduler in background thread"""
        self.running = True
        
        def run_scheduler():
            while self.running:
                # Run pending events
                if self.scheduler.queue:
                    self.scheduler.run(blocking=False)
                time.sleep(0.1)
        
        self.thread = threading.Thread(target=run_scheduler, daemon=True)
        self.thread.start()
        print("Scheduler started")
    
    def stop(self):
        """Stop scheduler"""
        self.running = False
        if self.thread:
            self.thread.join()
        print("Scheduler stopped")

# Usage
scheduler = TaskScheduler()
scheduler.start()

# Schedule one-time task
scheduler.schedule_once(2, print, "Task executed after 2 seconds")

# Schedule at specific time
future_time = datetime.now() + timedelta(seconds=5)
scheduler.schedule_at(future_time, print, "Task executed at scheduled time")

# Schedule recurring task
def recurring_task():
    print(f"Recurring task executed at {datetime.now()}")

scheduler.schedule_recurring(3, recurring_task)

# Let it run
time.sleep(15)

scheduler.stop()
```

## Foreground-Background Coordination

Coordinating between foreground operations and background tasks.

### Callback Pattern

```python
from typing import Callable, Any

class AsyncTaskWithCallback:
    """Async task with completion callback"""
    
    def __init__(self, manager: BackgroundTaskManager):
        self.manager = manager
        self.callbacks = {}
    
    def execute(
        self,
        func: Callable,
        callback: Callable[[Any], None],
        error_callback: Optional[Callable[[Exception], None]] = None,
        *args,
        **kwargs
    ) -> str:
        """Execute task with callback"""
        
        def wrapper():
            """Wrapper that calls callback on completion"""
            try:
                result = func(*args, **kwargs)
                callback(result)
                return result
            except Exception as e:
                if error_callback:
                    error_callback(e)
                raise
        
        task_id = self.manager.submit(wrapper)
        return task_id

# Usage
manager = BackgroundTaskManager(num_workers=2)
manager.start()

async_task = AsyncTaskWithCallback(manager)

def on_success(result):
    """Called when task completes"""
    print(f"Task completed successfully: {result}")

def on_error(error):
    """Called when task fails"""
    print(f"Task failed: {error}")

# Execute with callbacks
async_task.execute(
    long_running_task,
    on_success,
    on_error,
    "Callback Task",
    3
)

time.sleep(5)
manager.stop()
```

### Event-Based Coordination

```python
import threading
from typing import Callable, Any

class EventCoordinator:
    """Coordinate using events"""
    
    def __init__(self):
        self.events = {}
        self.listeners = {}
    
    def create_event(self, event_name: str) -> threading.Event:
        """Create event"""
        event = threading.Event()
        self.events[event_name] = event
        return event
    
    def wait_for_event(self, event_name: str, timeout: Optional[float] = None) -> bool:
        """Wait for event"""
        event = self.events.get(event_name)
        if not event:
            raise ValueError(f"Event not found: {event_name}")
        return event.wait(timeout=timeout)
    
    def trigger_event(self, event_name: str):
        """Trigger event"""
        event = self.events.get(event_name)
        if event:
            event.set()
            # Call listeners
            for listener in self.listeners.get(event_name, []):
                listener()
    
    def on_event(self, event_name: str, listener: Callable):
        """Register event listener"""
        if event_name not in self.listeners:
            self.listeners[event_name] = []
        self.listeners[event_name].append(listener)

# Usage
coordinator = EventCoordinator()

# Create events
coordinator.create_event("data_ready")
coordinator.create_event("processing_complete")

# Register listeners
coordinator.on_event("data_ready", lambda: print("Data is ready!"))
coordinator.on_event("processing_complete", lambda: print("Processing complete!"))

# Background task that triggers events
def background_task():
    """Task that triggers events"""
    time.sleep(1)
    coordinator.trigger_event("data_ready")
    
    time.sleep(2)
    coordinator.trigger_event("processing_complete")

# Start task
threading.Thread(target=background_task, daemon=True).start()

# Foreground waits for events
print("Waiting for data...")
coordinator.wait_for_event("data_ready")
print("Data received, continuing...")

coordinator.wait_for_event("processing_complete")
print("All done!")
```

## Cancellation and Timeout

Canceling tasks and handling timeouts.

### Cancellation Token

```python
import threading

class CancellationToken:
    """Token for canceling operations"""
    
    def __init__(self):
        self.cancelled = threading.Event()
    
    def cancel(self):
        """Cancel operation"""
        self.cancelled.set()
    
    def is_cancelled(self) -> bool:
        """Check if cancelled"""
        return self.cancelled.is_set()
    
    def throw_if_cancelled(self):
        """Raise exception if cancelled"""
        if self.is_cancelled():
            raise RuntimeError("Operation was cancelled")

def cancellable_task(token: CancellationToken, num_steps: int):
    """Task that can be cancelled"""
    for i in range(num_steps):
        # Check cancellation
        token.throw_if_cancelled()
        
        # Do work
        print(f"Step {i+1}/{num_steps}")
        time.sleep(1)
    
    return "Completed"

# Usage
token = CancellationToken()

# Start task
def run_task():
    try:
        result = cancellable_task(token, 10)
        print(f"Result: {result}")
    except RuntimeError as e:
        print(f"Cancelled: {e}")

thread = threading.Thread(target=run_task)
thread.start()

# Cancel after 3 seconds
time.sleep(3)
print("Cancelling task...")
token.cancel()

thread.join()
```

### Timeout Handler

```python
import signal
from contextlib import contextmanager

class TimeoutError(Exception):
    """Timeout exception"""
    pass

@contextmanager
def timeout(seconds: int):
    """Context manager for timeout"""
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds}s")
    
    # Set alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# Usage (Unix-like systems only)
try:
    with timeout(5):
        # Long operation
        time.sleep(10)
except TimeoutError as e:
    print(f"Timeout: {e}")
```

## Result Handling

Handling results from background tasks.

### Result Store

```python
from typing import Any, Optional
import threading

class ResultStore:
    """Store and retrieve task results"""
    
    def __init__(self):
        self.results = {}
        self.lock = threading.Lock()
    
    def store(self, task_id: str, result: Any):
        """Store result"""
        with self.lock:
            self.results[task_id] = {
                "result": result,
                "stored_at": datetime.now()
            }
    
    def retrieve(self, task_id: str) -> Optional[Any]:
        """Retrieve result"""
        with self.lock:
            entry = self.results.get(task_id)
            return entry["result"] if entry else None
    
    def delete(self, task_id: str):
        """Delete result"""
        with self.lock:
            self.results.pop(task_id, None)
    
    def clear_old(self, max_age_seconds: int = 3600):
        """Clear old results"""
        now = datetime.now()
        with self.lock:
            to_delete = [
                task_id
                for task_id, entry in self.results.items()
                if (now - entry["stored_at"]).total_seconds() > max_age_seconds
            ]
            
            for task_id in to_delete:
                del self.results[task_id]
            
            return len(to_delete)

# Usage
store = ResultStore()

# Store results
store.store("task_1", {"data": [1, 2, 3]})
store.store("task_2", {"data": [4, 5, 6]})

# Retrieve
result = store.retrieve("task_1")
print(f"Result: {result}")

# Clear old results
time.sleep(2)
cleared = store.clear_old(max_age_seconds=1)
print(f"Cleared {cleared} old results")
```

## Error Recovery

Recovering from background task failures.

### Retry Mechanism

```python
from typing import Callable
import time

class RetryableBackgroundTask:
    """Background task with retry logic"""
    
    def __init__(
        self,
        func: Callable,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0
    ):
        self.func = func
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute with retries"""
        delay = self.retry_delay
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                print(f"Attempt {attempt + 1}/{self.max_retries}")
                result = self.func(*args, **kwargs)
                print(f"Success on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_error = e
                print(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    print(f"Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= self.backoff_factor
        
        raise RuntimeError(f"Failed after {self.max_retries} attempts: {last_error}")

# Usage
def unreliable_operation():
    """Operation that might fail"""
    import random
    if random.random() < 0.7:  # 70% failure rate
        raise ValueError("Random failure")
    return "Success"

task = RetryableBackgroundTask(
    unreliable_operation,
    max_retries=5,
    retry_delay=1.0,
    backoff_factor=2.0
)

try:
    result = task.execute()
    print(f"Final result: {result}")
except RuntimeError as e:
    print(f"Task failed: {e}")
```

## Real-World Patterns

### Pattern 1: Document Processing Pipeline

```python
class DocumentProcessingPipeline:
    """Background document processing"""
    
    def __init__(self):
        self.manager = BackgroundTaskManager(num_workers=4)
        self.manager.start()
        self.results = ResultStore()
    
    def process_document(self, doc_id: str, document: dict) -> str:
        """Process document in background"""
        
        def process():
            """Processing pipeline"""
            # Step 1: Extract text
            text = extract_text(document)
            
            # Step 2: Analyze
            analysis = analyze_text(text)
            
            # Step 3: Generate summary
            summary = generate_summary(text)
            
            # Step 4: Extract entities
            entities = extract_entities(text)
            
            result = {
                "doc_id": doc_id,
                "analysis": analysis,
                "summary": summary,
                "entities": entities
            }
            
            # Store result
            self.results.store(doc_id, result)
            
            return result
        
        task_id = self.manager.submit(process)
        return task_id
    
    def get_result(self, doc_id: str, wait: bool = False, timeout: float = 60) -> Optional[dict]:
        """Get processing result"""
        if wait:
            # Wait for result
            start = time.time()
            while time.time() - start < timeout:
                result = self.results.retrieve(doc_id)
                if result:
                    return result
                time.sleep(0.5)
            return None
        else:
            return self.results.retrieve(doc_id)
    
    def shutdown(self):
        """Shutdown pipeline"""
        self.manager.stop()

# Usage
pipeline = DocumentProcessingPipeline()

# Process document
doc_id = "doc_123"
task_id = pipeline.process_document(doc_id, {"content": "Document text..."})
print(f"Processing started: {task_id}")

# Wait for result
result = pipeline.get_result(doc_id, wait=True, timeout=30)
if result:
    print(f"Processing complete: {result}")
else:
    print("Processing timed out")

pipeline.shutdown()
```

## Summary

Background processes enable responsive agent systems:

**Key Concepts**:
- **Long-Running Tasks**: Execute without blocking
- **Async Execution**: Non-blocking operations
- **Process Management**: Control background processes
- **Job Queues**: Reliable task queueing
- **Task Scheduling**: Time-based execution

**Patterns**:
- **Future/Promise**: Async result handling
- **Worker Pool**: Dedicated background workers
- **Job Queue**: Persistent task management
- **Callback**: Event-driven coordination
- **Retry**: Fault-tolerant execution

**Coordination**:
- Status tracking and monitoring
- Progress reporting
- Cancellation and timeout
- Result storage and retrieval
- Error recovery

**Best Practices**:
1. Use background processes for long operations
2. Implement status tracking
3. Handle cancellation gracefully
4. Store results persistently
5. Implement retry logic for failures
6. Monitor resource usage
7. Clean up completed tasks

Background processes transform synchronous agents into scalable, responsive systems capable of handling complex, long-running workflows.

## Next Steps

Explore related orchestration topics:

- **[Parallelization](parallelization.md)**: Concurrent execution patterns
- **[Workflow Patterns](workflow-patterns.md)**: Orchestration structures
- **[State Management](state-management.md)**: Managing long-running state
- **[Event-Driven Patterns](event-driven.md)**: Reactive orchestration

Related topics:
- **[Tool Use](../tool-use/function-calling.md)**: Async tool execution
- **[Memory Systems](../memory-systems/memory-architecture.md)**: Persistent state
