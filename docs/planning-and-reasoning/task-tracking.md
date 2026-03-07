# Task Tracking and Todo Lists

## Table of Contents

- [Introduction](#introduction)
- [Why Task Tracking Matters](#why-task-tracking-matters)
- [Todo List Patterns](#todo-list-patterns)
- [Task Status and States](#task-status-and-states)
- [Progress Monitoring](#progress-monitoring)
- [Priority Management](#priority-management)
- [Dependency Tracking](#dependency-tracking)
- [Implementing Todo Lists](#implementing-todo-lists)
- [External Memory for Agents](#external-memory-for-agents)
- [Completion Checking](#completion-checking)
- [Visualization and Reporting](#visualization-and-reporting)
- [Persistent Task Lists](#persistent-task-lists)
- [Multi-Agent Coordination](#multi-agent-coordination)
- [Best Practices](#best-practices)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Task tracking through todo lists provides external memory for agents tackling complex, multi-step problems. While planning determines **what** to do, task tracking maintains **where we are** in execution--tracking status, managing priorities, and monitoring progress.

> "Getting Things Done begins with getting things out of your head." - David Allen

For agents, todo lists serve as:

- **External memory**: Offload tracking from limited context
- **Progress indicator**: Measurable completion metrics
- **Coordination mechanism**: Share status across components
- **Recovery tool**: Resume after interruptions

### The Todo List Paradigm

```
Without Todo List:                With Todo List:
Agent: "What was I doing?"       ✓ Setup database
Agent: "What's next?"            ✓ Create API endpoints
Agent: "Am I making progress?"   □ Build frontend
Agent: "Did I do X yet?"         □ Write tests
                                 □ Deploy application

                                 Progress: 40% (2/5 complete)
```

### When Agents Need Todo Lists

**Essential for**:

- Long-running tasks spanning multiple sessions
- Complex projects with many subtasks
- Coordination across multiple agents
- Recovery from interruptions or failures
- Progress tracking and reporting

**Less critical for**:

- Simple, atomic actions
- Stateless request-response patterns
- Real-time reactive behaviors

## Why Task Tracking Matters

Understanding the cognitive and practical benefits of structured task tracking.

### External Memory

Language models have limited context windows. Todo lists externalize state:

```python
# Without external memory - must hold everything in prompt
prompt = """
You've completed: setup, database creation, API scaffolding
You're currently working on: authentication
Still need to do: frontend, testing, deployment
Remember: authentication must use JWT tokens
Previous error: forgot to hash passwords
"""

# With external memory - refer to todo list
prompt = """
Check the todo list for current status.
Continue with the next incomplete task.
"""

# Todo list stored externally
todo_list = TodoList([
    Task("Setup", status="complete"),
    Task("Database", status="complete"),
    Task("API scaffolding", status="complete"),
    Task("Authentication", status="in_progress",
         notes="Use JWT tokens, hash passwords"),
    Task("Frontend", status="pending"),
    Task("Testing", status="pending"),
    Task("Deployment", status="pending")
])
```

### Progress Visibility

Quantitative progress metrics enable monitoring:

```python
class ProgressTracker:
    """Track and report progress on task lists."""

    def __init__(self, todo_list):
        self.todo_list = todo_list

    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if not self.todo_list.tasks:
            return 0.0

        completed = sum(1 for task in self.todo_list.tasks
                       if task.status == "complete")
        return (completed / len(self.todo_list.tasks)) * 100

    def estimated_remaining_time(self) -> float:
        """Estimate time remaining based on task durations."""
        incomplete = [task for task in self.todo_list.tasks
                     if task.status != "complete"]
        return sum(task.estimated_duration for task in incomplete)

    def progress_report(self) -> str:
        """Generate human-readable progress report."""
        completed = sum(1 for t in self.todo_list.tasks if t.status == "complete")
        in_progress = sum(1 for t in self.todo_list.tasks if t.status == "in_progress")
        pending = sum(1 for t in self.todo_list.tasks if t.status == "pending")

        report = f"""
        Progress Report
        ===============
        Completed: {completed} tasks ({self.completion_percentage():.1f}%)
        In Progress: {in_progress} tasks
        Pending: {pending} tasks

        Estimated time remaining: {self.estimated_remaining_time():.1f} hours
        """

        return report


# Example usage
todo = TodoList([
    Task("Design", status="complete", estimated_duration=2),
    Task("Implementation", status="in_progress", estimated_duration=8),
    Task("Testing", status="pending", estimated_duration=4),
    Task("Documentation", status="pending", estimated_duration=3)
])

tracker = ProgressTracker(todo)
print(tracker.progress_report())
```

### Coordination and Communication

Todo lists enable multiple agents (or humans) to coordinate:

```python
class SharedTodoList:
    """Todo list shared across multiple agents."""

    def __init__(self):
        self.tasks: List[Task] = []
        self.lock = threading.Lock()

    def claim_task(self, agent_id: str) -> Optional[Task]:
        """
        Claim next available task for an agent.

        Thread-safe task assignment for parallel execution.
        """
        with self.lock:
            available = [t for t in self.tasks
                        if t.status == "pending" and t.can_start()]

            if available:
                task = available[0]
                task.status = "in_progress"
                task.assigned_to = agent_id
                task.started_at = datetime.now()
                return task

            return None

    def complete_task(self, task_id: str, agent_id: str):
        """Mark task as complete."""
        with self.lock:
            task = self.find_task(task_id)
            if task and task.assigned_to == agent_id:
                task.status = "complete"
                task.completed_at = datetime.now()

    def get_status(self) -> Dict[str, int]:
        """Get current status summary."""
        with self.lock:
            return {
                "total": len(self.tasks),
                "pending": sum(1 for t in self.tasks if t.status == "pending"),
                "in_progress": sum(1 for t in self.tasks if t.status == "in_progress"),
                "complete": sum(1 for t in self.tasks if t.status == "complete")
            }
```

## Todo List Patterns

Common patterns for organizing and managing todo lists.

### Pattern 1: Simple Sequential List

Basic ordered list of tasks:

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from enum import Enum

class TaskStatus(Enum):
    """Task status states."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a single task."""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0  # Higher = more important
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration: float = 0.0  # hours
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    def start(self):
        """Mark task as started."""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now()

    def complete(self):
        """Mark task as complete."""
        self.status = TaskStatus.COMPLETE
        self.completed_at = datetime.now()

    def block(self, reason: str):
        """Mark task as blocked."""
        self.status = TaskStatus.BLOCKED
        self.notes += f"\nBlocked: {reason}"

    def cancel(self, reason: str):
        """Cancel task."""
        self.status = TaskStatus.CANCELLED
        self.notes += f"\nCancelled: {reason}"


class SimpleTodoList:
    """Simple sequential todo list."""

    def __init__(self):
        self.tasks: List[Task] = []

    def add_task(self, description: str, priority: int = 0, **kwargs) -> Task:
        """Add new task to list."""
        task_id = f"task_{len(self.tasks) + 1}"
        task = Task(id=task_id, description=description, priority=priority, **kwargs)
        self.tasks.append(task)
        return task

    def get_next_task(self) -> Optional[Task]:
        """Get next pending task (highest priority)."""
        pending = [t for t in self.tasks if t.status == TaskStatus.PENDING]
        if pending:
            return max(pending, key=lambda t: t.priority)
        return None

    def mark_complete(self, task_id: str):
        """Mark task as complete."""
        task = self.find_task(task_id)
        if task:
            task.complete()

    def find_task(self, task_id: str) -> Optional[Task]:
        """Find task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """Get all tasks with specific status."""
        return [t for t in self.tasks if t.status == status]

    def __str__(self) -> str:
        """String representation."""
        lines = ["Todo List:", ""]
        for task in self.tasks:
            symbol = {
                TaskStatus.COMPLETE: "✓",
                TaskStatus.IN_PROGRESS: "→",
                TaskStatus.PENDING: "○",
                TaskStatus.BLOCKED: "✗",
                TaskStatus.CANCELLED: "⊘"
            }[task.status]

            priority_str = f"[P{task.priority}]" if task.priority > 0 else ""
            lines.append(f"{symbol} {task.description} {priority_str}")

        return "\n".join(lines)


# Example usage
todo = SimpleTodoList()
todo.add_task("Design database schema", priority=3)
todo.add_task("Implement authentication", priority=2)
todo.add_task("Create API endpoints", priority=2)
todo.add_task("Write documentation", priority=1)

print(todo)

# Work on tasks
next_task = todo.get_next_task()
print(f"\nNext task: {next_task.description}")
next_task.start()
# ... do work ...
next_task.complete()

print("\n" + str(todo))
```

### Pattern 2: Hierarchical Todo List

Nested tasks with subtasks:

```python
@dataclass
class HierarchicalTask(Task):
    """Task that can have subtasks."""
    subtasks: List['HierarchicalTask'] = field(default_factory=list)
    parent: Optional['HierarchicalTask'] = None

    def add_subtask(self, description: str, **kwargs) -> 'HierarchicalTask':
        """Add a subtask."""
        task_id = f"{self.id}.{len(self.subtasks) + 1}"
        subtask = HierarchicalTask(
            id=task_id,
            description=description,
            parent=self,
            **kwargs
        )
        self.subtasks.append(subtask)
        return subtask

    def is_leaf(self) -> bool:
        """Check if this is a leaf (no subtasks)."""
        return len(self.subtasks) == 0

    def progress(self) -> float:
        """Calculate completion percentage."""
        if self.is_leaf():
            return 1.0 if self.status == TaskStatus.COMPLETE else 0.0

        if not self.subtasks:
            return 0.0

        return sum(st.progress() for st in self.subtasks) / len(self.subtasks)

    def visualize(self, indent: int = 0) -> str:
        """Create tree visualization."""
        symbol = {
            TaskStatus.COMPLETE: "✓",
            TaskStatus.IN_PROGRESS: "→",
            TaskStatus.PENDING: "○",
            TaskStatus.BLOCKED: "✗"
        }.get(self.status, "?")

        progress_pct = self.progress() * 100
        lines = [f"{'  ' * indent}{symbol} {self.description} ({progress_pct:.0f}%)"]

        for subtask in self.subtasks:
            lines.append(subtask.visualize(indent + 1))

        return '\n'.join(lines)


# Example
project = HierarchicalTask(id="project", description="Build Web Application")

# Level 1
backend = project.add_subtask("Backend Development")
frontend = project.add_subtask("Frontend Development")
testing = project.add_subtask("Testing")

# Level 2
backend.add_subtask("Design API")
backend.add_subtask("Implement endpoints")
backend.add_subtask("Database integration")

frontend.add_subtask("Setup React")
frontend.add_subtask("Build components")
frontend.add_subtask("Styling")

testing.add_subtask("Unit tests")
testing.add_subtask("Integration tests")
testing.add_subtask("E2E tests")

# Mark some complete
project.subtasks[0].subtasks[0].complete()  # Design API
project.subtasks[0].subtasks[1].complete()  # Implement endpoints

print(project.visualize())
print(f"\nOverall progress: {project.progress():.1%}")
```

### Pattern 3: Dependency-Based Todo List

Tasks with explicit dependencies:

```python
class DependencyTodoList:
    """Todo list with dependency tracking."""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.dependencies: Dict[str, Set[str]] = {}  # task_id -> {dependency_ids}

    def add_task(self, task_id: str, description: str,
                 depends_on: List[str] = None, **kwargs) -> Task:
        """Add task with dependencies."""
        task = Task(id=task_id, description=description, **kwargs)
        self.tasks[task_id] = task
        self.dependencies[task_id] = set(depends_on or [])
        return task

    def can_start(self, task_id: str) -> bool:
        """Check if task's dependencies are satisfied."""
        if task_id not in self.dependencies:
            return False

        deps = self.dependencies[task_id]
        return all(self.tasks[dep].status == TaskStatus.COMPLETE
                  for dep in deps if dep in self.tasks)

    def get_ready_tasks(self) -> List[Task]:
        """Get all tasks that can start now."""
        ready = []
        for task_id, task in self.tasks.items():
            if (task.status == TaskStatus.PENDING and
                self.can_start(task_id)):
                ready.append(task)
        return ready

    def get_blocked_tasks(self) -> List[tuple[Task, List[str]]]:
        """Get blocked tasks with their blockers."""
        blocked = []
        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.PENDING and not self.can_start(task_id):
                incomplete_deps = [dep for dep in self.dependencies[task_id]
                                  if self.tasks[dep].status != TaskStatus.COMPLETE]
                blocked.append((task, incomplete_deps))
        return blocked

    def visualize_dependencies(self) -> str:
        """Visualize task dependencies."""
        lines = ["Task Dependencies:", ""]

        for task_id, task in self.tasks.items():
            status_symbol = "✓" if task.status == TaskStatus.COMPLETE else "○"
            lines.append(f"{status_symbol} {task.description}")

            if self.dependencies[task_id]:
                lines.append("  Dependencies:")
                for dep_id in self.dependencies[task_id]:
                    dep = self.tasks[dep_id]
                    dep_symbol = "✓" if dep.status == TaskStatus.COMPLETE else "○"
                    lines.append(f"    {dep_symbol} {dep.description}")
            lines.append("")

        return '\n'.join(lines)


# Example
todo = DependencyTodoList()

# Add tasks with dependencies
todo.add_task("install_deps", "Install dependencies")
todo.add_task("setup_db", "Setup database", depends_on=["install_deps"])
todo.add_task("create_models", "Create data models", depends_on=["setup_db"])
todo.add_task("build_api", "Build API", depends_on=["create_models"])
todo.add_task("setup_frontend", "Setup frontend", depends_on=["install_deps"])
todo.add_task("integrate", "Integrate frontend with API",
             depends_on=["build_api", "setup_frontend"])

print(todo.visualize_dependencies())

# Get ready tasks
ready = todo.get_ready_tasks()
print(f"\nReady to start ({len(ready)}):")
for task in ready:
    print(f"  - {task.description}")

# Complete a task and check what becomes available
todo.tasks["install_deps"].complete()
ready = todo.get_ready_tasks()
print(f"\nAfter completing 'Install dependencies' ({len(ready)}):")
for task in ready:
    print(f"  - {task.description}")
```

### Pattern 4: Kanban-Style Board

Organize tasks by workflow stage:

```python
from collections import defaultdict

class KanbanBoard:
    """Kanban-style task board with columns."""

    def __init__(self, columns: List[str]):
        self.columns = columns
        self.tasks: Dict[str, List[Task]] = {col: [] for col in columns}
        self.task_index: Dict[str, str] = {}  # task_id -> column

    def add_task(self, task: Task, column: str = None):
        """Add task to board."""
        if column is None:
            column = self.columns[0]  # First column by default

        if column not in self.columns:
            raise ValueError(f"Invalid column: {column}")

        self.tasks[column].append(task)
        self.task_index[task.id] = column

    def move_task(self, task_id: str, to_column: str):
        """Move task to different column."""
        if to_column not in self.columns:
            raise ValueError(f"Invalid column: {to_column}")

        # Find current column
        from_column = self.task_index.get(task_id)
        if not from_column:
            raise ValueError(f"Task not found: {task_id}")

        # Move task
        task = None
        for t in self.tasks[from_column]:
            if t.id == task_id:
                task = t
                break

        if task:
            self.tasks[from_column].remove(task)
            self.tasks[to_column].append(task)
            self.task_index[task_id] = to_column

    def visualize(self) -> str:
        """Create board visualization."""
        lines = []

        # Header
        header = " | ".join(f"{col:^20}" for col in self.columns)
        lines.append(header)
        lines.append("-" * len(header))

        # Find max tasks in any column
        max_tasks = max(len(tasks) for tasks in self.tasks.values())

        # Print rows
        for i in range(max_tasks):
            row_parts = []
            for col in self.columns:
                if i < len(self.tasks[col]):
                    task = self.tasks[col][i]
                    text = task.description[:18]
                    row_parts.append(f"{text:^20}")
                else:
                    row_parts.append(" " * 20)
            lines.append(" | ".join(row_parts))

        # Summary
        lines.append("-" * len(header))
        summary_parts = []
        for col in self.columns:
            count = len(self.tasks[col])
            summary_parts.append(f"{count:^20}")
        lines.append(" | ".join(summary_parts))

        return '\n'.join(lines)

    def get_wip_limit_status(self, limits: Dict[str, int]) -> Dict[str, dict]:
        """Check Work-In-Progress limits."""
        status = {}
        for col, limit in limits.items():
            current = len(self.tasks[col])
            status[col] = {
                "current": current,
                "limit": limit,
                "exceeded": current > limit
            }
        return status


# Example
board = KanbanBoard(["Backlog", "In Progress", "Review", "Done"])

# Add tasks
tasks = [
    Task(id="1", description="Design database"),
    Task(id="2", description="Build API"),
    Task(id="3", description="Create frontend"),
    Task(id="4", description="Write tests"),
    Task(id="5", description="Documentation"),
]

for task in tasks[:3]:
    board.add_task(task, "Backlog")

board.add_task(tasks[3], "In Progress")
board.add_task(tasks[4], "Done")

print(board.visualize())

# Move task
board.move_task("1", "In Progress")
print("\nAfter moving task 1 to In Progress:")
print(board.visualize())

# Check WIP limits
wip_limits = {"In Progress": 2, "Review": 3}
status = board.get_wip_limit_status(wip_limits)
print("\nWIP Limit Status:")
for col, info in status.items():
    exceeded = "⚠️ EXCEEDED" if info["exceeded"] else "✓"
    print(f"  {col}: {info['current']}/{info['limit']} {exceeded}")
```

## Task Status and States

Managing task lifecycle through state transitions.

### State Machine

```python
from enum import Enum
from typing import Set

class TaskState(Enum):
    """Task states."""
    PENDING = "pending"
    READY = "ready"  # Dependencies satisfied
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    REVIEW = "review"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStateMachine:
    """Manages valid task state transitions."""

    # Define valid transitions
    VALID_TRANSITIONS = {
        TaskState.PENDING: {TaskState.READY, TaskState.BLOCKED, TaskState.CANCELLED},
        TaskState.READY: {TaskState.IN_PROGRESS, TaskState.BLOCKED, TaskState.CANCELLED},
        TaskState.IN_PROGRESS: {TaskState.REVIEW, TaskState.COMPLETE, TaskState.FAILED, TaskState.BLOCKED},
        TaskState.BLOCKED: {TaskState.READY, TaskState.CANCELLED},
        TaskState.REVIEW: {TaskState.COMPLETE, TaskState.IN_PROGRESS, TaskState.FAILED},
        TaskState.COMPLETE: set(),  # Terminal state
        TaskState.FAILED: {TaskState.PENDING, TaskState.CANCELLED},  # Can retry
        TaskState.CANCELLED: set()  # Terminal state
    }

    @classmethod
    def can_transition(cls, from_state: TaskState, to_state: TaskState) -> bool:
        """Check if transition is valid."""
        return to_state in cls.VALID_TRANSITIONS.get(from_state, set())

    @classmethod
    def is_terminal(cls, state: TaskState) -> bool:
        """Check if state is terminal (no transitions out)."""
        return len(cls.VALID_TRANSITIONS.get(state, set())) == 0


@dataclass
class StatefulTask(Task):
    """Task with state machine enforcement."""
    state: TaskState = TaskState.PENDING
    state_history: List[tuple[TaskState, datetime]] = field(default_factory=list)

    def transition_to(self, new_state: TaskState, reason: str = ""):
        """Transition to new state if valid."""
        if not TaskStateMachine.can_transition(self.state, new_state):
            raise ValueError(
                f"Invalid transition from {self.state.value} to {new_state.value}"
            )

        self.state_history.append((self.state, datetime.now()))
        self.state = new_state

        if reason:
            self.notes += f"\n[{new_state.value}] {reason}"

    def get_state_duration(self, state: TaskState) -> float:
        """Get total time spent in a particular state (hours)."""
        total_seconds = 0.0

        for i, (hist_state, timestamp) in enumerate(self.state_history):
            if hist_state == state:
                # Find when we left this state
                if i + 1 < len(self.state_history):
                    next_timestamp = self.state_history[i + 1][1]
                elif self.state == state:
                    # Still in this state
                    next_timestamp = datetime.now()
                else:
                    continue

                duration = (next_timestamp - timestamp).total_seconds()
                total_seconds += duration

        return total_seconds / 3600  # Convert to hours


# Example usage
task = StatefulTask(id="1", description="Implement feature X")

# Valid transitions
task.transition_to(TaskState.READY, "Dependencies completed")
task.transition_to(TaskState.IN_PROGRESS, "Started work")
task.transition_to(TaskState.REVIEW, "Submitted for review")
task.transition_to(TaskState.COMPLETE, "Approved")

print(f"Task state: {task.state.value}")
print(f"\nState history:")
for state, timestamp in task.state_history:
    print(f"  {state.value} at {timestamp.strftime('%H:%M:%S')}")

# Try invalid transition
try:
    task.transition_to(TaskState.IN_PROGRESS)  # Can't go from COMPLETE to IN_PROGRESS
except ValueError as e:
    print(f"\nError: {e}")
```

## Progress Monitoring

Tracking and reporting progress across tasks.

### Progress Metrics

```python
class ProgressMetrics:
    """Calculate various progress metrics for todo lists."""

    @staticmethod
    def completion_rate(tasks: List[Task]) -> float:
        """Calculate completion rate (0.0 to 1.0)."""
        if not tasks:
            return 0.0
        completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETE)
        return completed / len(tasks)

    @staticmethod
    def velocity(tasks: List[Task], time_period_hours: float) -> float:
        """
        Calculate velocity (tasks completed per hour).

        Args:
            tasks: List of tasks
            time_period_hours: Time period to measure over

        Returns:
            Tasks completed per hour
        """
        completed = [t for t in tasks if t.status == TaskStatus.COMPLETE
                    and t.completed_at]

        if not completed:
            return 0.0

        # Count tasks completed within time period
        cutoff = datetime.now() - timedelta(hours=time_period_hours)
        recent_completions = sum(1 for t in completed
                                if t.completed_at > cutoff)

        return recent_completions / time_period_hours

    @staticmethod
    def estimated_completion_time(tasks: List[Task],
                                  current_velocity: float) -> float:
        """
        Estimate hours until all tasks complete.

        Args:
            tasks: List of tasks
            current_velocity: Current velocity (tasks/hour)

        Returns:
            Estimated hours to completion
        """
        incomplete = sum(1 for t in tasks if t.status != TaskStatus.COMPLETE)

        if current_velocity <= 0:
            return float('inf')

        return incomplete / current_velocity

    @staticmethod
    def burndown_data(tasks: List[Task]) -> List[tuple[datetime, int]]:
        """
        Generate burndown chart data.

        Returns:
            List of (timestamp, remaining_tasks) tuples
        """
        # Get all completion timestamps
        completed_tasks = [t for t in tasks if t.completed_at]
        completed_tasks.sort(key=lambda t: t.completed_at)

        # Build burndown data
        burndown = []
        remaining = len(tasks)

        for task in completed_tasks:
            burndown.append((task.completed_at, remaining))
            remaining -= 1

        if remaining > 0:
            burndown.append((datetime.now(), remaining))

        return burndown

    @staticmethod
    def bottleneck_analysis(tasks: List[Task]) -> Dict[str, List[Task]]:
        """
        Identify bottlenecks (tasks stuck in certain states).

        Returns:
            Dictionary of state -> list of tasks stuck there
        """
        bottlenecks = defaultdict(list)
        threshold_hours = 24  # Tasks stuck > 24 hours

        for task in tasks:
            if task.status in [TaskStatus.COMPLETE, TaskStatus.CANCELLED]:
                continue

            # Check how long in current state
            if task.started_at:
                hours_in_state = (datetime.now() - task.started_at).total_seconds() / 3600
                if hours_in_state > threshold_hours:
                    bottlenecks[task.status.value].append(task)

        return dict(bottlenecks)


# Example
tasks = [
    Task(id="1", description="Task 1", status=TaskStatus.COMPLETE,
         completed_at=datetime.now() - timedelta(hours=5)),
    Task(id="2", description="Task 2", status=TaskStatus.COMPLETE,
         completed_at=datetime.now() - timedelta(hours=3)),
    Task(id="3", description="Task 3", status=TaskStatus.IN_PROGRESS,
         started_at=datetime.now() - timedelta(hours=30)),
    Task(id="4", description="Task 4", status=TaskStatus.PENDING),
    Task(id="5", description="Task 5", status=TaskStatus.PENDING),
]

metrics = ProgressMetrics()

print(f"Completion rate: {metrics.completion_rate(tasks):.1%}")
print(f"Velocity (last 24h): {metrics.velocity(tasks, 24):.2f} tasks/hour")

velocity = metrics.velocity(tasks, 24)
if velocity > 0:
    etc = metrics.estimated_completion_time(tasks, velocity)
    print(f"Estimated time to completion: {etc:.1f} hours")

print("\nBottlenecks:")
bottlenecks = metrics.bottleneck_analysis(tasks)
for state, stuck_tasks in bottlenecks.items():
    print(f"  {state}: {len(stuck_tasks)} tasks")
    for task in stuck_tasks:
        print(f"    - {task.description}")
```

## External Memory for Agents

Using todo lists as external memory for complex tasks.

### Memory-Augmented Agent

```python
class MemoryAugmentedAgent:
    """
    Agent that uses todo list as external memory.

    Benefits:
    - Offload task tracking from limited context
    - Persist state across sessions
    - Enable resumption after interruptions
    """

    def __init__(self, model):
        self.model = model
        self.todo_list = SimpleTodoList()
        self.context_memory: List[str] = []  # Recent actions

    def execute_complex_task(self, main_task: str) -> dict:
        """
        Execute complex task using todo list as external memory.

        Process:
        1. Decompose main task into subtasks → add to todo list
        2. Execute subtasks one by one
        3. Track progress in todo list
        4. Resume from todo list after interruptions
        """
        # Step 1: Decompose and populate todo list
        subtasks = self._decompose_task(main_task)
        for subtask_desc in subtasks:
            self.todo_list.add_task(subtask_desc)

        print(f"Created {len(subtasks)} subtasks\n")
        print(self.todo_list)

        # Step 2: Execute subtasks
        while True:
            next_task = self.todo_list.get_next_task()

            if not next_task:
                # All tasks complete
                break

            print(f"\n{'='*50}")
            print(f"Executing: {next_task.description}")
            print(f"{'='*50}")

            # Execute task with context from todo list
            result = self._execute_subtask(next_task)

            if result["success"]:
                self.todo_list.mark_complete(next_task.id)
                print(f"✓ Completed: {next_task.description}")
            else:
                print(f"✗ Failed: {next_task.description}")
                next_task.block(result.get("error", "Unknown error"))
                break

        # Return results
        completed = self.todo_list.get_tasks_by_status(TaskStatus.COMPLETE)
        return {
            "main_task": main_task,
            "subtasks_completed": len(completed),
            "subtasks_total": len(self.todo_list.tasks),
            "success": len(completed) == len(self.todo_list.tasks)
        }

    def _decompose_task(self, task: str) -> List[str]:
        """Decompose main task into subtasks."""
        prompt = f"""
Task: {task}

Break this task into 3-5 concrete subtasks.

Subtasks:
1.
"""

        response = self.model.generate(prompt, temperature=0.3)

        # Parse subtasks
        subtasks = []
        for line in response.split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                subtask = line.split('.', 1)[1].strip()
                subtasks.append(subtask)

        return subtasks

    def _execute_subtask(self, task: Task) -> dict:
        """
        Execute a single subtask using context from todo list.

        Key insight: Agent can reference todo list instead of
        keeping everything in its limited context window.
        """
        # Build context from todo list state
        completed_tasks = self.todo_list.get_tasks_by_status(TaskStatus.COMPLETE)
        completed_desc = [t.description for t in completed_tasks]

        prompt = f"""
Current task: {task.description}

Already completed:
{chr(10).join(f'- {desc}' for desc in completed_desc)}

Execute this task. Describe what you did.

Action:
"""

        response = self.model.generate(prompt, temperature=0.3)

        # Simulate execution
        # In real system, would parse action and execute it
        task.notes = response

        return {"success": True, "output": response}

    def save_state(self, filepath: str):
        """Save todo list state to file."""
        import json

        state = {
            "tasks": [
                {
                    "id": t.id,
                    "description": t.description,
                    "status": t.status.value,
                    "notes": t.notes
                }
                for t in self.todo_list.tasks
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str):
        """Load todo list state from file."""
        import json

        with open(filepath, 'r') as f:
            state = json.load(f)

        self.todo_list = SimpleTodoList()
        for task_data in state["tasks"]:
            task = Task(
                id=task_data["id"],
                description=task_data["description"],
                status=TaskStatus(task_data["status"]),
                notes=task_data.get("notes", "")
            )
            self.todo_list.tasks.append(task)


# Example usage
agent = MemoryAugmentedAgent(model)

result = agent.execute_complex_task("Build a blog website")

print(f"\n{'='*50}")
print("Final Results:")
print(f"  Completed: {result['subtasks_completed']}/{result['subtasks_total']}")
print(f"  Success: {result['success']}")

# Save state for resumption
agent.save_state("agent_state.json")

# Later, resume from saved state
agent2 = MemoryAugmentedAgent(model)
agent2.load_state("agent_state.json")
print("\nLoaded state:")
print(agent2.todo_list)
```

## Completion Checking

Verifying that tasks are truly complete.

### Completion Criteria

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class CompletionCriterion:
    """Criterion for task completion."""
    name: str
    check: Callable[[], bool]  # Function that returns True if criterion met
    description: str = ""


class VerifiableTask(Task):
    """Task with explicit completion criteria."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.completion_criteria: List[CompletionCriterion] = []
        self.verification_results: Dict[str, bool] = {}

    def add_criterion(self, name: str, check: Callable[[], bool],
                     description: str = ""):
        """Add completion criterion."""
        criterion = CompletionCriterion(name, check, description)
        self.completion_criteria.append(criterion)

    def verify_completion(self) -> dict:
        """
        Verify all completion criteria are met.

        Returns:
            Dictionary with verification results
        """
        results = {}
        all_met = True

        for criterion in self.completion_criteria:
            try:
                met = criterion.check()
                results[criterion.name] = met
                self.verification_results[criterion.name] = met

                if not met:
                    all_met = False
            except Exception as e:
                results[criterion.name] = False
                self.verification_results[criterion.name] = False
                all_met = False
                print(f"Error checking {criterion.name}: {e}")

        return {
            "all_criteria_met": all_met,
            "results": results,
            "criteria_count": len(self.completion_criteria)
        }

    def can_mark_complete(self) -> bool:
        """Check if task can be marked complete."""
        verification = self.verify_completion()
        return verification["all_criteria_met"]


# Example
def check_file_exists(filepath: str) -> Callable[[], bool]:
    """Create file existence check."""
    def check():
        import os
        return os.path.exists(filepath)
    return check

def check_tests_pass() -> bool:
    """Check if tests pass."""
    # Simulate running tests
    # In reality, would run actual test suite
    return True

def check_code_reviewed() -> bool:
    """Check if code has been reviewed."""
    # Check review status in system
    return True


# Create task with criteria
task = VerifiableTask(id="1", description="Implement authentication feature")

task.add_criterion(
    "code_written",
    check_file_exists("auth.py"),
    "Authentication code file must exist"
)

task.add_criterion(
    "tests_pass",
    check_tests_pass,
    "All tests must pass"
)

task.add_criterion(
    "code_reviewed",
    check_code_reviewed,
    "Code must be reviewed and approved"
)

# Try to complete task
verification = task.verify_completion()

print("Verification Results:")
print(f"  All criteria met: {verification['all_criteria_met']}")
print(f"\nIndividual criteria:")
for criterion_name, met in verification['results'].items():
    status = "✓" if met else "✗"
    print(f"  {status} {criterion_name}")

if task.can_mark_complete():
    task.complete()
    print("\n✓ Task marked as complete")
else:
    print("\n✗ Task cannot be completed yet - criteria not met")
```

## Summary

Task tracking through todo lists provides essential external memory and progress monitoring for agents handling complex, multi-step problems. From simple sequential lists to hierarchical structures with dependencies, effective task tracking enables coordination, recovery, and measurable progress.

### Key Takeaways

1. **External memory is essential**: Todo lists offload tracking from limited context windows

2. **Multiple patterns exist**: Sequential, hierarchical, dependency-based, and Kanban patterns serve different needs

3. **State management matters**: Explicit state machines prevent invalid transitions

4. **Progress is measurable**: Completion rates, velocity, and burndown metrics quantify progress

5. **Verification prevents errors**: Explicit completion criteria ensure tasks are truly done

6. **Persistence enables recovery**: Saved todo lists allow resumption after interruptions

7. **Coordination requires shared state**: Multiple agents can coordinate through shared todo lists

### Best Practices

1. **Choose appropriate pattern**: Match todo list structure to problem complexity
2. **Make dependencies explicit**: Track prerequisite relationships between tasks
3. **Define completion criteria**: Specify what "done" means for each task
4. **Monitor progress quantitatively**: Use metrics like velocity and completion rate
5. **Enable state persistence**: Save todo lists for recovery and resumption
6. **Enforce state transitions**: Use state machines to prevent invalid transitions
7. **Visualize status**: Create clear visualizations of progress and bottlenecks

## Next Steps

Explore related planning and execution topics:

- **[Task Decomposition](task-decomposition.md)**: Break complex goals into manageable subtasks
- **[Planning Strategies](planning-strategies.md)**: Determine execution order and coordination
- **[Implementation Plans](implementation-plans.md)**: Create detailed execution specifications
- **[Chain-of-Thought](chain-of-thought.md)**: Reason about task status and next steps
- **[ReAct Pattern](../agent-architectures/react.md)**: Integrate reasoning with action execution

Master task tracking to build agents that maintain progress through complex, long-running tasks! ✓
