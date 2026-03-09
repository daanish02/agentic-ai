# Subagent Patterns

## Table of Contents

- [Introduction](#introduction)
- [When to Use Subagents](#when-to-use-subagents)
- [Hierarchical Execution](#hierarchical-execution)
- [Task Delegation](#task-delegation)
- [Subagent Lifecycle](#subagent-lifecycle)
- [Parent-Child Coordination](#parent-child-coordination)
- [Specialized Subagents](#specialized-subagents)
- [Resource Management](#resource-management)
- [Communication Patterns](#communication-patterns)
- [Result Aggregation](#result-aggregation)
- [Error Handling](#error-handling)
- [Scaling Patterns](#scaling-patterns)
- [Real-World Applications](#real-world-applications)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Subagents enable agents to delegate specialized tasks to child agents, creating hierarchical execution structures. Like a manager delegating to team members, parent agents can:

- **Decompose complex tasks**: Break work into specialized subtasks
- **Leverage specialization**: Use agents optimized for specific domains
- **Scale horizontally**: Spawn multiple subagents in parallel
- **Isolate failures**: Contain errors within subagent boundaries
- **Manage resources**: Control execution limits per subagent

> "Subagents turn monolithic agents into collaborative teams."

```
Subagent Hierarchy:

Parent Agent
    ├─> Subagent 1 (Data Collection)
    ├─> Subagent 2 (Analysis)
    │   ├─> Sub-subagent 2a (Statistics)
    │   └─> Sub-subagent 2b (Visualization)
    └─> Subagent 3 (Reporting)

Each level delegates to the next, creating a tree of execution.
```

## When to Use Subagents

Identifying scenarios where subagents add value.

### Decision Framework

```python
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"        # Single agent sufficient
    MODERATE = "moderate"    # May benefit from delegation
    COMPLEX = "complex"      # Requires delegation

@dataclass
class TaskAnalysis:
    """Analysis of whether to use subagents"""
    task_description: str
    complexity: TaskComplexity
    requires_specialization: bool
    parallelizable: bool
    resource_intensive: bool
    estimated_subtasks: int

    def should_delegate(self) -> bool:
        """Determine if delegation is appropriate"""
        # Simple tasks: no delegation
        if self.complexity == TaskComplexity.SIMPLE:
            return False

        # Complex tasks: definitely delegate
        if self.complexity == TaskComplexity.COMPLEX:
            return True

        # Moderate tasks: delegate if multiple factors apply
        factors = [
            self.requires_specialization,
            self.parallelizable,
            self.resource_intensive,
            self.estimated_subtasks > 3
        ]

        return sum(factors) >= 2

    def get_recommendation(self) -> str:
        """Get delegation recommendation"""
        if not self.should_delegate():
            return "Execute directly without subagents"

        if self.parallelizable:
            return "Use parallel subagents for concurrent execution"

        if self.requires_specialization:
            return "Use specialized subagents for different domains"

        return "Use sequential subagents for pipeline execution"

# Example: Analyzing tasks
def analyze_task(task_description: str) -> TaskAnalysis:
    """Analyze task to determine delegation strategy"""
    # Simple heuristics (in practice, use ML or rules engine)
    words = task_description.split()

    complexity = TaskComplexity.SIMPLE
    if len(words) > 20:
        complexity = TaskComplexity.MODERATE
    if len(words) > 40:
        complexity = TaskComplexity.COMPLEX

    requires_specialization = any(
        term in task_description.lower()
        for term in ["analyze", "visualize", "translate", "summarize"]
    )

    parallelizable = "and" in task_description.lower()
    resource_intensive = any(
        term in task_description.lower()
        for term in ["large", "big", "extensive", "comprehensive"]
    )

    estimated_subtasks = task_description.count("and") + task_description.count(",") + 1

    return TaskAnalysis(
        task_description=task_description,
        complexity=complexity,
        requires_specialization=requires_specialization,
        parallelizable=parallelizable,
        resource_intensive=resource_intensive,
        estimated_subtasks=estimated_subtasks
    )

# Usage
task = "Analyze sales data, generate visualizations, and create a comprehensive report"
analysis = analyze_task(task)

print(f"Should delegate: {analysis.should_delegate()}")
print(f"Recommendation: {analysis.get_recommendation()}")
```

## Hierarchical Execution

Creating parent-child agent hierarchies.

### Agent Hierarchy

```python
from typing import Dict, Any, Optional, List
import uuid
from datetime import datetime

class Agent:
    """Base agent class"""

    def __init__(self, agent_id: str, role: str, parent: Optional["Agent"] = None):
        self.agent_id = agent_id
        self.role = role
        self.parent = parent
        self.children: List[Agent] = []
        self.state = {}

    def execute(self, task: Dict[str, Any]) -> Any:
        """Execute task"""
        raise NotImplementedError

    def spawn_subagent(self, role: str) -> "Agent":
        """Spawn child agent"""
        subagent_id = f"{self.agent_id}_sub_{len(self.children)}"
        subagent = Agent(subagent_id, role, parent=self)
        self.children.append(subagent)

        print(f"Agent {self.agent_id} spawned subagent {subagent_id} ({role})")
        return subagent

    def delegate(self, subagent: "Agent", task: Dict[str, Any]) -> Any:
        """Delegate task to subagent"""
        print(f"Agent {self.agent_id} delegating to {subagent.agent_id}: {task.get('description', 'task')}")
        result = subagent.execute(task)
        return result

    def get_hierarchy(self, level: int = 0) -> str:
        """Get hierarchy representation"""
        indent = "  " * level
        lines = [f"{indent}Agent {self.agent_id} ({self.role})"]

        for child in self.children:
            lines.append(child.get_hierarchy(level + 1))

        return "\n".join(lines)

class OrchestratorAgent(Agent):
    """Orchestrator that manages subagents"""

    def execute(self, task: Dict[str, Any]) -> Any:
        """Execute by decomposing and delegating"""
        print(f"\nOrchestrator {self.agent_id} received task: {task.get('description')}")

        # Decompose task
        subtasks = self.decompose_task(task)

        # Spawn specialized subagents
        results = []
        for subtask in subtasks:
            # Create specialized subagent
            subagent = self.spawn_subagent(role=subtask["role"])

            # Delegate
            result = self.delegate(subagent, subtask)
            results.append(result)

        # Aggregate results
        final_result = self.aggregate_results(results)

        print(f"Orchestrator {self.agent_id} completed task")
        return final_result

    def decompose_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose task into subtasks"""
        # Example decomposition
        return [
            {"description": "Collect data", "role": "data_collector"},
            {"description": "Analyze data", "role": "analyzer"},
            {"description": "Generate report", "role": "reporter"}
        ]

    def aggregate_results(self, results: List[Any]) -> Any:
        """Aggregate subagent results"""
        return {"status": "success", "results": results}

class SpecializedAgent(Agent):
    """Specialized agent for specific tasks"""

    def execute(self, task: Dict[str, Any]) -> Any:
        """Execute specialized task"""
        print(f"  Specialized agent {self.agent_id} ({self.role}) executing: {task.get('description')}")

        # Simulate specialized work
        result = {
            "agent_id": self.agent_id,
            "role": self.role,
            "task": task.get("description"),
            "output": f"Completed {self.role} work"
        }

        return result

# Usage
orchestrator = OrchestratorAgent("agent_root", "orchestrator")

task = {
    "description": "Process customer data and generate insights",
    "data_source": "database"
}

result = orchestrator.execute(task)

print("\n" + "="*50)
print("Agent Hierarchy:")
print(orchestrator.get_hierarchy())
print("\nFinal Result:", result)
```

## Task Delegation

Patterns for delegating work to subagents.

### Delegation Manager

```python
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable

class DelegationManager:
    """Manage task delegation to subagents"""

    def __init__(self, max_subagents: int = 5):
        self.max_subagents = max_subagents
        self.active_subagents = {}
        self.executor = ThreadPoolExecutor(max_workers=max_subagents)

    def delegate_sequential(
        self,
        tasks: List[Dict[str, Any]],
        agent_factory: Callable[[str], Agent]
    ) -> List[Any]:
        """Delegate tasks sequentially"""
        results = []

        for i, task in enumerate(tasks):
            # Create subagent
            subagent = agent_factory(f"subagent_{i}")
            self.active_subagents[subagent.agent_id] = subagent

            # Execute
            result = subagent.execute(task)
            results.append(result)

            # Clean up
            del self.active_subagents[subagent.agent_id]

        return results

    def delegate_parallel(
        self,
        tasks: List[Dict[str, Any]],
        agent_factory: Callable[[str], Agent]
    ) -> List[Any]:
        """Delegate tasks in parallel"""
        futures = []

        for i, task in enumerate(tasks):
            # Create subagent
            subagent = agent_factory(f"subagent_{i}")
            self.active_subagents[subagent.agent_id] = subagent

            # Submit for parallel execution
            future = self.executor.submit(subagent.execute, task)
            futures.append((subagent.agent_id, future))

        # Collect results
        results = []
        for agent_id, future in futures:
            result = future.result()
            results.append(result)

            # Clean up
            del self.active_subagents[agent_id]

        return results

    def delegate_with_dependencies(
        self,
        tasks: List[Dict[str, Any]],
        dependencies: Dict[int, List[int]],
        agent_factory: Callable[[str], Agent]
    ) -> List[Any]:
        """Delegate tasks respecting dependencies"""
        results = [None] * len(tasks)
        completed = set()

        while len(completed) < len(tasks):
            # Find tasks ready to execute
            ready = []
            for i, task in enumerate(tasks):
                if i not in completed:
                    deps = dependencies.get(i, [])
                    if all(d in completed for d in deps):
                        ready.append(i)

            if not ready:
                raise ValueError("Circular dependency detected")

            # Execute ready tasks in parallel
            futures = []
            for task_idx in ready:
                subagent = agent_factory(f"subagent_{task_idx}")

                # Pass results from dependencies
                task_with_deps = {
                    **tasks[task_idx],
                    "dependency_results": [
                        results[dep_idx]
                        for dep_idx in dependencies.get(task_idx, [])
                    ]
                }

                future = self.executor.submit(subagent.execute, task_with_deps)
                futures.append((task_idx, future))

            # Collect results
            for task_idx, future in futures:
                results[task_idx] = future.result()
                completed.add(task_idx)

        return results

    def shutdown(self):
        """Shutdown executor"""
        self.executor.shutdown(wait=True)

# Usage
manager = DelegationManager(max_subagents=3)

# Sequential delegation
tasks = [
    {"description": "Task 1"},
    {"description": "Task 2"},
    {"description": "Task 3"}
]

def create_agent(agent_id: str) -> Agent:
    return SpecializedAgent(agent_id, "worker")

print("Sequential delegation:")
results = manager.delegate_sequential(tasks, create_agent)

print("\nParallel delegation:")
results = manager.delegate_parallel(tasks, create_agent)

print("\nDelegation with dependencies:")
# Task 2 depends on Task 0, Task 3 depends on Task 1 and 2
dependencies = {
    1: [0],      # Task 1 depends on Task 0
    2: [0, 1]    # Task 2 depends on Tasks 0 and 1
}
results = manager.delegate_with_dependencies(tasks, dependencies, create_agent)

manager.shutdown()
```

## Subagent Lifecycle

Managing subagent creation, execution, and cleanup.

### Lifecycle Manager

```python
from enum import Enum
from datetime import datetime, timedelta

class SubagentState(Enum):
    """Subagent lifecycle states"""
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"

@dataclass
class SubagentInfo:
    """Subagent information"""
    agent_id: str
    role: str
    state: SubagentState
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None

class SubagentLifecycleManager:
    """Manage subagent lifecycle"""

    def __init__(self):
        self.subagents: Dict[str, SubagentInfo] = {}

    def create(self, role: str) -> str:
        """Create new subagent"""
        agent_id = f"subagent_{uuid.uuid4().hex[:8]}"

        info = SubagentInfo(
            agent_id=agent_id,
            role=role,
            state=SubagentState.CREATED,
            created_at=datetime.now()
        )

        self.subagents[agent_id] = info
        print(f"Created subagent {agent_id} ({role})")

        return agent_id

    def initialize(self, agent_id: str):
        """Initialize subagent"""
        info = self.subagents[agent_id]
        info.state = SubagentState.INITIALIZING

        # Perform initialization
        # ...

        info.state = SubagentState.READY
        print(f"Subagent {agent_id} ready")

    def start_execution(self, agent_id: str):
        """Start executing task"""
        info = self.subagents[agent_id]
        info.state = SubagentState.EXECUTING
        info.started_at = datetime.now()
        print(f"Subagent {agent_id} started execution")

    def complete_execution(self, agent_id: str, result: Any):
        """Complete execution successfully"""
        info = self.subagents[agent_id]
        info.state = SubagentState.COMPLETED
        info.completed_at = datetime.now()
        info.result = result
        print(f"Subagent {agent_id} completed")

    def fail_execution(self, agent_id: str, error: str):
        """Mark execution as failed"""
        info = self.subagents[agent_id]
        info.state = SubagentState.FAILED
        info.completed_at = datetime.now()
        info.error = error
        print(f"Subagent {agent_id} failed: {error}")

    def terminate(self, agent_id: str):
        """Terminate subagent"""
        info = self.subagents[agent_id]
        info.state = SubagentState.TERMINATED
        print(f"Subagent {agent_id} terminated")

    def cleanup_completed(self, max_age: timedelta = timedelta(hours=1)):
        """Clean up old completed subagents"""
        now = datetime.now()
        to_remove = []

        for agent_id, info in self.subagents.items():
            if info.state in [SubagentState.COMPLETED, SubagentState.FAILED]:
                if info.completed_at and (now - info.completed_at) > max_age:
                    to_remove.append(agent_id)

        for agent_id in to_remove:
            del self.subagents[agent_id]

        if to_remove:
            print(f"Cleaned up {len(to_remove)} old subagents")

    def get_status(self, agent_id: str) -> Optional[SubagentInfo]:
        """Get subagent status"""
        return self.subagents.get(agent_id)

    def list_active(self) -> List[SubagentInfo]:
        """List active subagents"""
        return [
            info for info in self.subagents.values()
            if info.state in [SubagentState.READY, SubagentState.EXECUTING]
        ]

# Usage
lifecycle = SubagentLifecycleManager()

# Create and manage subagent
agent_id = lifecycle.create("data_processor")
lifecycle.initialize(agent_id)
lifecycle.start_execution(agent_id)

# Simulate work
import time
time.sleep(1)

lifecycle.complete_execution(agent_id, {"processed": 100})

# Check status
status = lifecycle.get_status(agent_id)
print(f"Status: {status.state.value}")

# Clean up old subagents
lifecycle.cleanup_completed(max_age=timedelta(seconds=0))
```

## Parent-Child Coordination

Coordinating between parent and child agents.

### Coordination Protocol

```python
from queue import Queue
from typing import Dict

class Message:
    """Message between parent and child"""

    def __init__(self, sender: str, recipient: str, msg_type: str, content: Any):
        self.sender = sender
        self.recipient = recipient
        self.msg_type = msg_type
        self.content = content
        self.timestamp = datetime.now()

class CoordinatedAgent:
    """Agent with parent-child coordination"""

    def __init__(self, agent_id: str, role: str):
        self.agent_id = agent_id
        self.role = role
        self.parent: Optional[CoordinatedAgent] = None
        self.children: Dict[str, CoordinatedAgent] = {}
        self.inbox = Queue()

    def spawn_child(self, role: str) -> "CoordinatedAgent":
        """Spawn child agent"""
        child_id = f"{self.agent_id}_child_{len(self.children)}"
        child = CoordinatedAgent(child_id, role)
        child.parent = self
        self.children[child_id] = child

        # Send initialization message
        self.send_to_child(child_id, "init", {"role": role})

        return child

    def send_to_parent(self, msg_type: str, content: Any):
        """Send message to parent"""
        if not self.parent:
            raise ValueError("No parent agent")

        message = Message(
            sender=self.agent_id,
            recipient=self.parent.agent_id,
            msg_type=msg_type,
            content=content
        )

        self.parent.inbox.put(message)
        print(f"{self.agent_id} -> {self.parent.agent_id}: {msg_type}")

    def send_to_child(self, child_id: str, msg_type: str, content: Any):
        """Send message to child"""
        if child_id not in self.children:
            raise ValueError(f"Unknown child: {child_id}")

        child = self.children[child_id]
        message = Message(
            sender=self.agent_id,
            recipient=child_id,
            msg_type=msg_type,
            content=content
        )

        child.inbox.put(message)
        print(f"{self.agent_id} -> {child_id}: {msg_type}")

    def receive_message(self, timeout: float = 1.0) -> Optional[Message]:
        """Receive message"""
        try:
            return self.inbox.get(timeout=timeout)
        except:
            return None

    def process_messages(self):
        """Process all pending messages"""
        while not self.inbox.empty():
            message = self.receive_message()
            if message:
                self.handle_message(message)

    def handle_message(self, message: Message):
        """Handle received message"""
        print(f"{self.agent_id} received {message.msg_type} from {message.sender}")

        if message.msg_type == "init":
            # Handle initialization
            pass
        elif message.msg_type == "task":
            # Handle task assignment
            self.execute_task(message.content)
        elif message.msg_type == "result":
            # Handle result from child
            self.handle_child_result(message.sender, message.content)
        elif message.msg_type == "status":
            # Handle status update
            pass

    def execute_task(self, task: Dict[str, Any]):
        """Execute task and report result"""
        print(f"{self.agent_id} executing task")

        # Simulate work
        result = {"status": "success", "agent": self.agent_id}

        # Report to parent
        if self.parent:
            self.send_to_parent("result", result)

    def handle_child_result(self, child_id: str, result: Any):
        """Handle result from child"""
        print(f"{self.agent_id} received result from {child_id}")

# Usage
parent = CoordinatedAgent("parent", "orchestrator")

# Spawn children
child1 = parent.spawn_child("worker")
child2 = parent.spawn_child("worker")

# Assign tasks
parent.send_to_child(child1.agent_id, "task", {"work": "process_data"})
parent.send_to_child(child2.agent_id, "task", {"work": "analyze_results"})

# Children process messages and execute
child1.process_messages()
child2.process_messages()

# Parent receives results
parent.process_messages()
```

## Specialized Subagents

Creating domain-specific subagents.

### Specialist Registry

```python
class SpecialistAgent(Agent):
    """Domain-specialized agent"""

    def __init__(self, agent_id: str, specialty: str, capabilities: List[str]):
        super().__init__(agent_id, specialty)
        self.specialty = specialty
        self.capabilities = capabilities

    def can_handle(self, task: Dict[str, Any]) -> bool:
        """Check if agent can handle task"""
        task_type = task.get("type")
        return task_type in self.capabilities

    def execute(self, task: Dict[str, Any]) -> Any:
        """Execute specialized task"""
        if not self.can_handle(task):
            raise ValueError(f"Cannot handle task type: {task.get('type')}")

        print(f"Specialist {self.agent_id} ({self.specialty}) handling {task.get('type')}")

        # Execute based on specialty
        method = getattr(self, f"_handle_{task.get('type')}", None)
        if method:
            return method(task)

        return {"status": "completed", "agent": self.agent_id}

class DataSpecialist(SpecialistAgent):
    """Data processing specialist"""

    def __init__(self, agent_id: str):
        super().__init__(
            agent_id,
            specialty="data_processing",
            capabilities=["load_data", "clean_data", "transform_data"]
        )

    def _handle_load_data(self, task: Dict[str, Any]) -> Any:
        """Load data"""
        source = task.get("source")
        print(f"Loading data from {source}")
        return {"data": "loaded_data", "rows": 1000}

    def _handle_clean_data(self, task: Dict[str, Any]) -> Any:
        """Clean data"""
        print(f"Cleaning data")
        return {"data": "cleaned_data", "rows": 950}

class AnalysisSpecialist(SpecialistAgent):
    """Analysis specialist"""

    def __init__(self, agent_id: str):
        super().__init__(
            agent_id,
            specialty="analysis",
            capabilities=["statistical_analysis", "trend_analysis", "anomaly_detection"]
        )

    def _handle_statistical_analysis(self, task: Dict[str, Any]) -> Any:
        """Perform statistical analysis"""
        print(f"Performing statistical analysis")
        return {"mean": 42.5, "std": 10.2, "median": 40.0}

class VisualizationSpecialist(SpecialistAgent):
    """Visualization specialist"""

    def __init__(self, agent_id: str):
        super().__init__(
            agent_id,
            specialty="visualization",
            capabilities=["create_chart", "create_dashboard", "export_visual"]
        )

    def _handle_create_chart(self, task: Dict[str, Any]) -> Any:
        """Create chart"""
        chart_type = task.get("chart_type", "bar")
        print(f"Creating {chart_type} chart")
        return {"chart": f"{chart_type}_chart.png"}

class SpecialistRegistry:
    """Registry of specialist agents"""

    def __init__(self):
        self.specialists: Dict[str, type] = {
            "data_processing": DataSpecialist,
            "analysis": AnalysisSpecialist,
            "visualization": VisualizationSpecialist
        }

    def create_specialist(self, specialty: str, agent_id: str) -> SpecialistAgent:
        """Create specialist agent"""
        specialist_class = self.specialists.get(specialty)
        if not specialist_class:
            raise ValueError(f"Unknown specialty: {specialty}")

        return specialist_class(agent_id)

    def find_specialist(self, task: Dict[str, Any]) -> Optional[str]:
        """Find appropriate specialist for task"""
        task_type = task.get("type")

        for specialty, specialist_class in self.specialists.items():
            # Create temporary instance to check capabilities
            temp = specialist_class("temp")
            if temp.can_handle(task):
                return specialty

        return None

# Usage
registry = SpecialistRegistry()

# Create specialized subagents
data_agent = registry.create_specialist("data_processing", "data_001")
analysis_agent = registry.create_specialist("analysis", "analysis_001")
viz_agent = registry.create_specialist("visualization", "viz_001")

# Execute specialized tasks
tasks = [
    {"type": "load_data", "source": "database"},
    {"type": "statistical_analysis", "data": "loaded_data"},
    {"type": "create_chart", "chart_type": "line"}
]

for task in tasks:
    specialty = registry.find_specialist(task)
    if specialty:
        agent = registry.create_specialist(specialty, f"agent_{uuid.uuid4().hex[:4]}")
        result = agent.execute(task)
        print(f"Result: {result}\n")
```

## Result Aggregation

Combining results from multiple subagents.

### Aggregation Strategies

```python
from typing import List, Any, Callable

class ResultAggregator:
    """Aggregate results from subagents"""

    def aggregate_list(self, results: List[Any]) -> List[Any]:
        """Simple list aggregation"""
        return results

    def aggregate_dict(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge dictionaries"""
        aggregated = {}
        for result in results:
            aggregated.update(result)
        return aggregated

    def aggregate_sum(self, results: List[Any], key: str = None) -> float:
        """Sum numeric results"""
        if key:
            return sum(r.get(key, 0) for r in results)
        return sum(results)

    def aggregate_vote(self, results: List[Any]) -> Any:
        """Majority vote"""
        from collections import Counter
        counter = Counter(results)
        return counter.most_common(1)[0][0]

    def aggregate_consensus(
        self,
        results: List[Any],
        threshold: float = 0.7
    ) -> Optional[Any]:
        """Consensus if threshold met"""
        from collections import Counter
        counter = Counter(results)
        most_common, count = counter.most_common(1)[0]

        if count / len(results) >= threshold:
            return most_common

        return None

    def aggregate_custom(
        self,
        results: List[Any],
        aggregation_func: Callable[[List[Any]], Any]
    ) -> Any:
        """Custom aggregation function"""
        return aggregation_func(results)

# Example: Complex aggregation
class MultiAgentSystem:
    """System that aggregates subagent results"""

    def __init__(self):
        self.aggregator = ResultAggregator()

    def parallel_analysis(self, data: Any) -> Dict[str, Any]:
        """Run parallel analysis with multiple subagents"""
        # Spawn analysis subagents
        subagents = [
            AnalysisSpecialist(f"analyst_{i}")
            for i in range(3)
        ]

        # Each analyzes the data
        results = []
        for subagent in subagents:
            result = subagent.execute({
                "type": "statistical_analysis",
                "data": data
            })
            results.append(result)

        # Aggregate results
        aggregated = {
            "mean": sum(r["mean"] for r in results) / len(results),
            "std": sum(r["std"] for r in results) / len(results),
            "median": sum(r["median"] for r in results) / len(results),
            "num_analysts": len(results)
        }

        return aggregated

# Usage
system = MultiAgentSystem()
result = system.parallel_analysis("sample_data")
print(f"Aggregated analysis: {result}")
```

## Real-World Applications

### Pattern: Research Team

```python
class ResearchTeam:
    """Simulated research team with specialized subagents"""

    def __init__(self):
        self.coordinator = CoordinatedAgent("coordinator", "research_lead")
        self.specialists = {}

    def conduct_research(self, topic: str) -> Dict[str, Any]:
        """Conduct research using specialized subagents"""
        print(f"\n{'='*60}")
        print(f"Research Team investigating: {topic}")
        print(f"{'='*60}\n")

        # Spawn specialists
        data_collector = self.coordinator.spawn_child("data_collector")
        analyst = self.coordinator.spawn_child("analyst")
        writer = self.coordinator.spawn_child("writer")

        self.specialists = {
            "data": data_collector,
            "analysis": analyst,
            "writing": writer
        }

        # Phase 1: Data collection
        print("Phase 1: Data Collection")
        self.coordinator.send_to_child(
            data_collector.agent_id,
            "task",
            {"phase": "collect", "topic": topic}
        )
        data_collector.process_messages()
        data_collector.execute_task({"topic": topic})

        # Phase 2: Analysis
        print("\nPhase 2: Analysis")
        self.coordinator.send_to_child(
            analyst.agent_id,
            "task",
            {"phase": "analyze", "topic": topic}
        )
        analyst.process_messages()
        analyst.execute_task({"topic": topic})

        # Phase 3: Writing
        print("\nPhase 3: Report Writing")
        self.coordinator.send_to_child(
            writer.agent_id,
            "task",
            {"phase": "write", "topic": topic}
        )
        writer.process_messages()
        writer.execute_task({"topic": topic})

        # Coordinator aggregates results
        print("\nPhase 4: Aggregation")
        self.coordinator.process_messages()

        final_report = {
            "topic": topic,
            "data_collected": True,
            "analysis_complete": True,
            "report_generated": True,
            "team_size": len(self.specialists)
        }

        print(f"\n{'='*60}")
        print("Research Complete!")
        print(f"{'='*60}\n")

        return final_report

# Usage
team = ResearchTeam()
report = team.conduct_research("Impact of AI on Software Development")
print(f"Final Report: {report}")
```

## Summary

Subagents enable hierarchical, specialized agent systems:

**Key Concepts**:

- **Task Delegation**: Breaking complex work into subtasks
- **Specialization**: Domain-specific subagents
- **Hierarchy**: Parent-child relationships
- **Lifecycle**: Creation, execution, cleanup
- **Coordination**: Inter-agent communication

**Patterns**:

- **Orchestrator**: Parent coordinates subagents
- **Specialist**: Domain-specific agents
- **Pipeline**: Sequential delegation
- **Parallel**: Concurrent subagents
- **Hierarchical**: Multi-level delegation

**Benefits**:

- Decompose complex tasks
- Leverage specialization
- Scale horizontally
- Isolate failures
- Manage resources

**Best Practices**:

1. Use subagents for complex, decomposable tasks
2. Create specialized subagents for domains
3. Implement proper lifecycle management
4. Coordinate with clear protocols
5. Aggregate results effectively
6. Handle subagent failures gracefully
7. Clean up completed subagents

Subagents transform monolithic agents into collaborative, scalable teams.

## Next Steps

Explore related orchestration topics:

- **[Workflow Patterns](workflow-patterns.md)**: Orchestration structures
- **[Parallelization](parallelization.md)**: Concurrent execution
- **[Human-in-the-Loop](human-in-the-loop.md)**: Human oversight of subagents

Related areas:

- **[Multi-Agent Systems](../multi-agent-systems/coordination.md)**: Agent collaboration
- **[Planning](../planning-and-reasoning/hierarchical-planning.md)**: Task decomposition
