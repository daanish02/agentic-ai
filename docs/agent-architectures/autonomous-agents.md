# Autonomous Agent Patterns

## Table of Contents

- [Introduction](#introduction)
- [What Makes an Agent Autonomous](#what-makes-an-agent-autonomous)
- [Long-Running Agent Architecture](#long-running-agent-architecture)
- [Goal-Directed Autonomy](#goal-directed-autonomy)
- [Self-Directed Task Decomposition](#self-directed-task-decomposition)
- [Continuous Operation](#continuous-operation)
- [Proactive Behavior](#proactive-behavior)
- [State Persistence](#state-persistence)
- [Resource Management](#resource-management)
- [Monitoring and Observability](#monitoring-and-observability)
- [Failure Recovery](#failure-recovery)
- [Learning and Adaptation](#learning-and-adaptation)
- [Ethical Constraints](#ethical-constraints)
- [Common Autonomous Patterns](#common-autonomous-patterns)
- [Implementation Examples](#implementation-examples)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Autonomous agents are designed for **open-ended goal pursuit** and **continuous operation**. Unlike reactive agents that respond to requests, autonomous agents proactively work toward long-term goals with minimal human intervention.

Think of the difference between:

- **Reactive**: "Answer this question" → Answer → Stop
- **Autonomous**: "Monitor this system and fix issues" → Runs continuously, identifies and resolves problems

> "True autonomy is the ability to pursue goals without constant supervision."

Autonomous agents are the most sophisticated agent architecture, requiring careful design around resource management, failure recovery, and ethical constraints.

### Characteristics of Autonomous Agents

1. **Self-directed**: Determines own actions without prompting
2. **Long-running**: Operates continuously over extended periods
3. **Goal-oriented**: Works toward high-level objectives
4. **Adaptive**: Adjusts behavior based on outcomes
5. **Proactive**: Anticipates needs rather than just reacting
6. **Resilient**: Recovers from failures autonomously

### Example Use Cases

- **Personal assistants**: Continuously manage calendar, emails, tasks
- **System monitors**: Watch infrastructure, fix issues automatically
- **Research agents**: Continuously discover and synthesize information
- **Trading bots**: Monitor markets, execute strategies autonomously
- **DevOps agents**: Maintain systems, deploy updates, handle incidents

## What Makes an Agent Autonomous

### Degrees of Autonomy

```
Level 0: Manual - Human does everything
Level 1: Assisted - Agent suggests, human approves
Level 2: Semi-Autonomous - Agent acts, human supervises
Level 3: Conditional Autonomy - Agent handles routine, human handles edge cases
Level 4: High Autonomy - Agent handles almost everything
Level 5: Full Autonomy - Agent operates independently
```

### Key Capabilities

**1. Self-Direction**

```python
class SelfDirectedAgent:
    """Agent that determines own actions."""

    def run(self):
        while self.should_continue():
            # Agent decides what to work on
            task = self.identify_next_task()

            # Execute without asking
            result = self.execute_task(task)

            # Learn and adapt
            self.update_strategy(result)
```

**2. Goal Maintenance**

```python
def maintain_goal(self, goal):
    """Keep working toward goal over time."""
    while not self.goal_achieved(goal):
        # Check current state
        state = self.assess_situation()

        # Adapt approach if needed
        if self.strategy_not_working(state):
            self.replan(goal, state)

        # Take next action
        action = self.decide_action(state, goal)
        self.execute(action)

        # Periodic check-in (but keep going)
        if self.should_report():
            self.report_progress(goal, state)
```

**3. Persistence**

```python
def persistent_operation(self):
    """Continue operation despite interruptions."""
    try:
        while True:
            self.work_toward_goals()

    except Interruption as e:
        # Save state
        self.save_state()

        # Wait for resume
        self.wait_for_resume()

        # Continue where left off
        self.restore_state()
        self.persistent_operation()
```

## Long-Running Agent Architecture

Designing agents for extended operation.

### Architecture Components

```
┌─────────────────────────────────────┐
│         Goal Manager                │
│   (High-level objectives)           │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│       Task Scheduler                │
│  (What to work on when)             │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│       Execution Engine              │
│   (Carries out tasks)               │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│       State Manager                 │
│  (Persist progress)                 │
└─────────────────────────────────────┘
```

### Implementation

```python
from dataclasses import dataclass
from typing import List, Dict, Any
import time
import logging


@dataclass
class Goal:
    """High-level objective."""
    id: str
    description: str
    priority: int
    deadline: float = None
    status: str = "active"  # active, paused, completed


@dataclass
class Task:
    """Specific task to execute."""
    id: str
    goal_id: str
    action: str
    params: Dict[str, Any]
    priority: int
    created_at: float


class AutonomousAgent:
    """
    Long-running autonomous agent.
    """

    def __init__(self, tools, llm, state_file="agent_state.json"):
        self.tools = tools
        self.llm = llm
        self.state_file = state_file

        # Load or initialize state
        self.goals: List[Goal] = []
        self.tasks: List[Task] = []
        self.history: List[Dict] = []

        self.running = False

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start autonomous operation."""
        self.running = True
        self.logger.info("Agent starting autonomous operation")

        try:
            while self.running:
                # Main autonomy loop
                self.autonomy_cycle()

                # Periodic save
                if self.should_save_state():
                    self.save_state()

                # Brief pause
                time.sleep(1)

        except KeyboardInterrupt:
            self.logger.info("Agent stopping gracefully")
            self.save_state()

        except Exception as e:
            self.logger.error(f"Agent error: {e}")
            self.save_state()
            raise

    def autonomy_cycle(self):
        """Single cycle of autonomous operation."""
        # 1. Review goals
        self.review_goals()

        # 2. Identify next task
        task = self.identify_next_task()

        if not task:
            # No tasks - proactively find work
            task = self.discover_new_task()

        if task:
            # 3. Execute task
            self.execute_task(task)

        else:
            # 4. Wait for new goals or work
            self.logger.info("No tasks to execute, waiting...")
            time.sleep(10)

    def review_goals(self):
        """Review and update goal status."""
        for goal in self.goals:
            if goal.status != "active":
                continue

            # Check if goal achieved
            if self.is_goal_achieved(goal):
                goal.status = "completed"
                self.logger.info(f"Goal completed: {goal.description}")
                continue

            # Check if goal expired
            if goal.deadline and time.time() > goal.deadline:
                goal.status = "expired"
                self.logger.warning(f"Goal expired: {goal.description}")
                continue

            # Check if tasks needed
            if not self.has_pending_tasks(goal):
                self.generate_tasks_for_goal(goal)

    def identify_next_task(self) -> Task:
        """Select next task to execute."""
        # Filter pending tasks
        pending = [t for t in self.tasks if self.is_task_pending(t)]

        if not pending:
            return None

        # Sort by priority and deadline
        pending.sort(key=lambda t: (
            -self.get_task_priority(t),
            self.get_task_deadline(t)
        ))

        return pending[0]

    def discover_new_task(self) -> Task:
        """Proactively find new work."""
        # Review active goals for opportunities
        for goal in self.goals:
            if goal.status == "active":
                # Ask LLM to suggest next task
                suggestion = self.llm(f"""
Goal: {goal.description}

Current progress: {self.summarize_progress(goal)}

What task should be done next to advance this goal?
Suggest a specific, actionable task.

Task:""")

                # Create task from suggestion
                task = self.create_task_from_suggestion(goal, suggestion)
                if task:
                    return task

        return None

    def execute_task(self, task: Task):
        """Execute a task."""
        self.logger.info(f"Executing task: {task.action}")

        try:
            # Execute action
            if task.action in self.tools:
                result = self.tools[task.action](**task.params)
            else:
                result = self.execute_with_llm(task)

            # Record success
            self.record_task_completion(task, result, success=True)

            # Update goals based on result
            self.update_goals_from_result(task, result)

        except Exception as e:
            self.logger.error(f"Task failed: {e}")
            self.record_task_completion(task, str(e), success=False)

            # Adaptive response to failure
            self.handle_task_failure(task, e)

    def add_goal(self, description: str, priority: int = 5):
        """Add new goal."""
        goal = Goal(
            id=self.generate_id(),
            description=description,
            priority=priority
        )

        self.goals.append(goal)
        self.logger.info(f"New goal added: {description}")

        # Generate initial tasks
        self.generate_tasks_for_goal(goal)

    def generate_tasks_for_goal(self, goal: Goal):
        """Generate tasks for a goal."""
        prompt = f"""
Goal: {goal.description}

Current progress: {self.summarize_progress(goal)}

Generate 3-5 specific tasks to advance this goal.
Each task should be:
- Concrete and actionable
- Achievable in one step
- Contributing to the goal

Tasks:"""

        response = self.llm(prompt)
        tasks = self.parse_tasks(response, goal)

        self.tasks.extend(tasks)
        self.logger.info(f"Generated {len(tasks)} tasks for goal {goal.id}")

    def save_state(self):
        """Persist agent state."""
        import json

        state = {
            "goals": [vars(g) for g in self.goals],
            "tasks": [vars(t) for t in self.tasks],
            "history": self.history[-100:]  # Keep recent history
        }

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

        self.logger.info("State saved")

    def stop(self):
        """Stop agent gracefully."""
        self.running = False
        self.save_state()
```

## Goal-Directed Autonomy

Managing high-level goals over time.

### Goal Hierarchy

```python
@dataclass
class GoalHierarchy:
    """Hierarchical goal structure."""
    root_goal: Goal
    subgoals: List['GoalHierarchy']

    def all_goals(self) -> List[Goal]:
        """Get all goals in hierarchy."""
        goals = [self.root_goal]
        for sub in self.subgoals:
            goals.extend(sub.all_goals())
        return goals

    def is_achieved(self) -> bool:
        """Check if goal and all subgoals achieved."""
        if self.root_goal.status != "completed":
            return False

        return all(sub.is_achieved() for sub in self.subgoals)
```

### Goal Decomposition

```python
def decompose_goal(goal: Goal, llm) -> List[Goal]:
    """Break goal into subgoals."""
    prompt = f"""
Break this goal into 3-5 concrete subgoals:

Goal: {goal.description}

Each subgoal should be:
- Specific and measurable
- Achievable independently
- Contributing to main goal

Subgoals:"""

    response = llm(prompt)
    subgoals = parse_subgoals(response, parent_goal=goal)

    return subgoals
```

## Self-Directed Task Decomposition

Agents deciding how to break down work.

### Dynamic Task Generation

```python
class TaskGenerator:
    """Generate tasks based on current state."""

    def __init__(self, llm):
        self.llm = llm

    def generate_next_tasks(self, goal, current_state):
        """Generate tasks for current situation."""
        prompt = f"""
Goal: {goal.description}

Current state:
{format_state(current_state)}

What tasks should be done next?
Consider:
- What's been completed
- What remains
- What's most important
- What's feasible now

Generate 3-5 next tasks:"""

        response = self.llm(prompt)
        return self.parse_tasks(response)

    def adapt_task_plan(self, goal, completed, failed):
        """Adapt plan based on results."""
        prompt = f"""
Goal: {goal.description}

Completed tasks:
{format_tasks(completed)}

Failed tasks:
{format_tasks(failed)}

Adapt the plan:
1. What went well? Continue that approach.
2. What failed? Try different approach.
3. What new tasks are needed?

Updated task plan:"""

        response = self.llm(prompt)
        return self.parse_tasks(response)
```

## Continuous Operation

Patterns for long-running agents.

### Event-Driven Operation

```python
class EventDrivenAgent:
    """Agent responding to events while pursuing goals."""

    def __init__(self):
        self.event_queue = []
        self.goals = []

    def run(self):
        """Main event loop."""
        while True:
            # Check for events
            if self.event_queue:
                event = self.event_queue.pop(0)
                self.handle_event(event)

            # Work on goals
            self.work_on_goals()

            # Check for new events
            self.poll_events()

            time.sleep(0.1)

    def handle_event(self, event):
        """Handle incoming event."""
        if event.type == "new_goal":
            self.add_goal(event.data)

        elif event.type == "urgent_task":
            self.handle_urgent_task(event.data)

        elif event.type == "resource_available":
            self.utilize_resource(event.data)

    def work_on_goals(self):
        """Progress on current goals."""
        for goal in self.goals:
            if goal.status == "active":
                task = self.select_task(goal)
                if task:
                    self.execute_task(task)
                    break  # One task per cycle
```

### Time-Based Operation

```python
class ScheduledAgent:
    """Agent with scheduled activities."""

    def __init__(self):
        self.schedule = []
        self.goals = []

    def run(self):
        """Run with scheduled activities."""
        while True:
            current_time = time.time()

            # Check schedule
            for item in self.schedule:
                if self.should_execute(item, current_time):
                    self.execute_scheduled_item(item)

            # Work on goals (when not scheduled)
            if self.has_free_time(current_time):
                self.work_on_goals()

            time.sleep(60)  # Check every minute

    def add_scheduled_task(self, task, schedule):
        """Add task to schedule."""
        self.schedule.append({
            "task": task,
            "schedule": schedule,  # e.g., "every 1 hour"
            "last_run": None
        })
```

## Proactive Behavior

Agents that anticipate and act without prompting.

### Proactive Monitoring

```python
class ProactiveMonitor:
    """Monitor system and act proactively."""

    def __init__(self, monitors, thresholds):
        self.monitors = monitors
        self.thresholds = thresholds

    def run(self):
        """Continuous monitoring."""
        while True:
            for name, monitor in self.monitors.items():
                # Check metric
                value = monitor.get_value()

                # Compare to threshold
                if self.exceeds_threshold(name, value):
                    self.take_proactive_action(name, value)

            time.sleep(10)

    def take_proactive_action(self, metric, value):
        """Act before problem becomes critical."""
        if metric == "disk_space" and value < 0.1:
            self.logger.warning("Low disk space, cleaning up...")
            self.cleanup_disk()

        elif metric == "error_rate" and value > 0.05:
            self.logger.warning("High error rate, investigating...")
            self.investigate_errors()
```

### Anticipatory Behavior

```python
def anticipate_needs(state, goals):
    """Predict future needs and prepare."""
    predictions = []

    # Analyze patterns
    for goal in goals:
        # What will this goal likely need?
        resources_needed = predict_resources(goal)

        # Prepare in advance
        for resource in resources_needed:
            if not is_available(resource):
                predictions.append({
                    "action": "acquire_resource",
                    "resource": resource,
                    "reason": f"Will be needed for {goal.description}"
                })

    return predictions
```

## State Persistence

Maintaining state across sessions.

### State Management

```python
class StatefulAgent:
    """Agent with persistent state."""

    def __init__(self, state_dir="agent_state"):
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)

    def save_state(self):
        """Save complete state."""
        state = {
            "goals": self.serialize_goals(),
            "tasks": self.serialize_tasks(),
            "memory": self.serialize_memory(),
            "metadata": {
                "saved_at": time.time(),
                "version": self.version
            }
        }

        with open(f"{self.state_dir}/state.json", 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def load_state(self):
        """Load state from disk."""
        try:
            with open(f"{self.state_dir}/state.json", 'r') as f:
                state = json.load(f)

            self.deserialize_goals(state["goals"])
            self.deserialize_tasks(state["tasks"])
            self.deserialize_memory(state["memory"])

            self.logger.info("State loaded successfully")

        except FileNotFoundError:
            self.logger.info("No saved state, starting fresh")

    def checkpoint(self):
        """Create checkpoint for recovery."""
        checkpoint_file = f"{self.state_dir}/checkpoint_{int(time.time())}.json"

        with open(checkpoint_file, 'w') as f:
            json.dump(self.get_full_state(), f, default=str)
```

## Resource Management

Managing limited resources over time.

### Resource Tracking

```python
@dataclass
class ResourceBudget:
    """Track resource usage."""
    tokens_total: int
    tokens_used: int = 0
    api_calls_total: int
    api_calls_used: int = 0
    time_limit: float = None
    start_time: float = None


class ResourceAwareAgent:
    """Agent that manages resources."""

    def __init__(self, budget: ResourceBudget):
        self.budget = budget
        self.budget.start_time = time.time()

    def can_afford(self, action) -> bool:
        """Check if action is within budget."""
        cost = self.estimate_cost(action)

        if self.budget.tokens_used + cost["tokens"] > self.budget.tokens_total:
            return False

        if self.budget.api_calls_used + cost["api_calls"] > self.budget.api_calls_total:
            return False

        if self.budget.time_limit:
            elapsed = time.time() - self.budget.start_time
            if elapsed > self.budget.time_limit:
                return False

        return True

    def execute_with_budget(self, action):
        """Execute action and track resources."""
        if not self.can_afford(action):
            raise ResourceExhausted("Insufficient resources")

        # Execute
        result = self.execute(action)

        # Track usage
        cost = self.measure_cost(action, result)
        self.budget.tokens_used += cost["tokens"]
        self.budget.api_calls_used += cost["api_calls"]

        return result
```

## Monitoring and Observability

Tracking autonomous agent behavior.

### Logging and Metrics

```python
class ObservableAgent:
    """Agent with comprehensive logging."""

    def __init__(self):
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "goals_achieved": 0,
            "uptime": 0,
            "errors": []
        }

        self.logger = logging.getLogger(__name__)

    def execute_task_observable(self, task):
        """Execute task with full observability."""
        start_time = time.time()

        self.logger.info(f"Starting task: {task.id}")

        try:
            result = self.execute_task(task)

            duration = time.time() - start_time

            self.metrics["tasks_completed"] += 1
            self.logger.info(f"Task completed: {task.id} ({duration:.2f}s)")

            return result

        except Exception as e:
            duration = time.time() - start_time

            self.metrics["tasks_failed"] += 1
            self.metrics["errors"].append({
                "task": task.id,
                "error": str(e),
                "timestamp": time.time()
            })

            self.logger.error(f"Task failed: {task.id} ({duration:.2f}s) - {e}")

            raise

    def get_metrics(self) -> Dict:
        """Get current metrics."""
        return {
            **self.metrics,
            "success_rate": self.calculate_success_rate(),
            "uptime": time.time() - self.start_time
        }
```

## Failure Recovery

Handling failures in autonomous operation.

### Automatic Recovery

```python
class ResilientAgent:
    """Agent with failure recovery."""

    def execute_with_recovery(self, task, max_retries=3):
        """Execute with automatic retry."""
        for attempt in range(max_retries):
            try:
                return self.execute_task(task)

            except RecoverableError as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:
                    # Wait before retry (exponential backoff)
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)

                    # Adapt approach
                    task = self.adapt_task_after_failure(task, e)

            except CriticalError as e:
                self.logger.error(f"Critical error, cannot recover: {e}")
                self.handle_critical_failure(task, e)
                raise

        # All retries failed
        self.handle_persistent_failure(task)
        raise MaxRetriesExceeded(f"Task {task.id} failed after {max_retries} attempts")
```

## Learning and Adaptation

Agents that improve over time.

### Experience-Based Learning

```python
class LearningAgent:
    """Agent that learns from experience."""

    def __init__(self):
        self.experience = []
        self.strategies = {}

    def learn_from_execution(self, task, result, success):
        """Learn from task execution."""
        self.experience.append({
            "task": task,
            "result": result,
            "success": success,
            "timestamp": time.time()
        })

        # Update strategies
        if success:
            self.reinforce_strategy(task, result)
        else:
            self.adjust_strategy(task, result)

    def select_strategy(self, task):
        """Choose strategy based on experience."""
        similar_experiences = self.find_similar_experiences(task)

        if similar_experiences:
            # Use strategy that worked in similar situations
            successful = [e for e in similar_experiences if e["success"]]
            if successful:
                return successful[-1]["strategy"]

        # Default strategy
        return self.default_strategy
```

## Ethical Constraints

Ensuring autonomous agents behave ethically.

### Constraint System

```python
class EthicalConstraints:
    """Enforce ethical behavior."""

    def __init__(self):
        self.rules = []
        self.violations = []

    def add_constraint(self, constraint):
        """Add ethical constraint."""
        self.rules.append(constraint)

    def check_action(self, action) -> bool:
        """Check if action violates constraints."""
        for rule in self.rules:
            if not rule.allows(action):
                self.violations.append({
                    "action": action,
                    "rule": rule,
                    "timestamp": time.time()
                })
                return False

        return True


class SafeAutonomousAgent:
    """Autonomous agent with ethical constraints."""

    def __init__(self, constraints: EthicalConstraints):
        self.constraints = constraints

    def execute_safely(self, action):
        """Execute action only if ethical."""
        # Check constraints
        if not self.constraints.check_action(action):
            self.logger.warning(f"Action blocked by constraints: {action}")
            return None

        # Check reversibility
        if not self.is_reversible(action):
            # Require explicit approval for irreversible actions
            if not self.get_approval(action):
                return None

        # Execute
        return self.execute(action)
```

## Common Autonomous Patterns

### Background Worker Pattern

```python
def background_worker_pattern():
    """Agent working in background."""
    while True:
        # Check for new work
        work = check_for_work()

        if work:
            execute(work)
        else:
            # Idle - look for proactive opportunities
            opportunity = find_opportunity()
            if opportunity:
                execute(opportunity)
            else:
                sleep(60)
```

### Maintenance Agent Pattern

```python
def maintenance_agent_pattern():
    """Agent maintaining system health."""
    while True:
        # Monitor system
        metrics = collect_metrics()

        # Identify issues
        issues = identify_issues(metrics)

        # Fix problems
        for issue in issues:
            fix_issue(issue)

        # Optimize
        if has_spare_capacity():
            optimize_system()

        sleep(300)  # Check every 5 minutes
```

## Implementation Examples

### Complete Autonomous Research Agent

```python
class AutonomousResearchAgent:
    """Agent that continuously researches topics."""

    def __init__(self, research_topics):
        self.topics = research_topics
        self.knowledge_base = {}
        self.running = False

    def start(self):
        """Start autonomous research."""
        self.running = True

        while self.running:
            # Select topic to research
            topic = self.select_next_topic()

            # Research
            findings = self.research_topic(topic)

            # Integrate findings
            self.integrate_knowledge(topic, findings)

            # Identify new topics
            new_topics = self.discover_related_topics(findings)
            self.topics.extend(new_topics)

            # Report progress
            self.report_progress()

            time.sleep(3600)  # Research every hour

    def research_topic(self, topic):
        """Research a single topic."""
        # Search for information
        papers = self.search_papers(topic)

        # Read and extract
        findings = []
        for paper in papers[:5]:
            content = self.read_paper(paper)
            insights = self.extract_insights(content)
            findings.append(insights)

        return findings
```

## Summary

Autonomous agents operate independently pursuing long-term goals:

1. **Autonomy**: Self-directed action without constant supervision
2. **Long-Running**: Designed for continuous operation over extended periods
3. **Goal-Directed**: Work toward high-level objectives
4. **Task Decomposition**: Self-directed breaking down of work
5. **Continuous Operation**: Event-driven and time-based patterns
6. **Proactive**: Anticipate needs and act before prompted
7. **State Persistence**: Maintain progress across sessions
8. **Resource Management**: Track and manage limited resources
9. **Observability**: Comprehensive logging and metrics
10. **Failure Recovery**: Automatic retry and adaptation
11. **Learning**: Improve strategies based on experience
12. **Ethical Constraints**: Ensure safe, responsible behavior

Autonomous agents are the most sophisticated architecture, requiring careful design around safety, resource management, and monitoring.

## Next Steps

- Continue to [Advanced Patterns](advanced-patterns.md) for sophisticated architectures like subagent systems
- Review [Design Principles](design-principles.md) for building robust agent systems
- Explore [Planning Architectures](planning-architectures.md) for structured long-term planning
- Study [Reflection](reflection.md) for self-improvement capabilities
