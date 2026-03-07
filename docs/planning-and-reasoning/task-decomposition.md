# Task Decomposition

## Table of Contents

- [Introduction](#introduction)
- [Why Decompose Tasks?](#why-decompose-tasks)
- [Decomposition Principles](#decomposition-principles)
- [Hierarchical Decomposition](#hierarchical-decomposition)
- [Decomposition Strategies](#decomposition-strategies)
- [Identifying Subtasks](#identifying-subtasks)
- [Dependency Analysis](#dependency-analysis)
- [Sequencing Subtasks](#sequencing-subtasks)
- [Parallel vs Sequential Execution](#parallel-vs-sequential-execution)
- [Implementing Decomposition](#implementing-decomposition)
- [Bottom-Up vs Top-Down](#bottom-up-vs-top-down)
- [Granularity and Scope](#granularity-and-scope)
- [Validation and Verification](#validation-and-verification)
- [Common Decomposition Patterns](#common-decomposition-patterns)
- [Adaptive Decomposition](#adaptive-decomposition)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Task decomposition is the art and science of breaking complex goals into manageable, actionable subtasks. It's one of the most fundamental cognitive abilities for problem-solving--and essential for agent systems that tackle real-world challenges.

> "A complex system that works is invariably found to have evolved from a simple system that worked." - John Gall

The key insight: **humans naturally decompose complex tasks**, and agents must do the same to handle anything beyond simple, atomic actions.

### The Decomposition Mindset

**Without decomposition**:

```
Task: Build a web application
Agent: *attempts to build entire app in one step*
Agent: *fails due to overwhelming complexity*
```

**With decomposition**:

```
Task: Build a web application
  ↓
Subtask 1: Design database schema
Subtask 2: Create API endpoints
Subtask 3: Build frontend components
Subtask 4: Integrate frontend with API
Subtask 5: Test and deploy
  ↓
Each subtask is manageable and achievable
```

### Why Decomposition Matters

**Benefits**:

1. **Reduces cognitive load**: Each subtask is simpler than the whole
2. **Enables progress tracking**: Can measure completion of subtasks
3. **Allows parallelization**: Independent subtasks can run simultaneously
4. **Improves error recovery**: Failures are localized to specific subtasks
5. **Facilitates reuse**: Subtasks may apply to other problems
6. **Makes plans interpretable**: Clear structure shows reasoning

**Real-world analogy**:

```
Cooking a complex meal:
❌ "Make Thanksgiving dinner" - overwhelming
✅ "Brine turkey" → "Prepare sides" → "Bake desserts" → "Coordinate timing"
   Each step is concrete and achievable
```

## Why Decompose Tasks?

Understanding the deep reasons for decomposition helps build better decomposition strategies.

### Cognitive Limitations

**Human (and AI) working memory is limited**:

```python
# Cognitive load comparison

# No decomposition - high cognitive load
def build_app():
    """
    Single function trying to do everything.
    Must keep track of:
    - Database design
    - API implementation
    - Frontend logic
    - State management
    - Error handling
    - Testing
    - Deployment

    Cognitive load: OVERWHELMING
    """
    # 1000+ lines of tangled code
    pass


# With decomposition - manageable load
def build_app():
    """
    Decomposed into focused subtasks.
    Each function has clear purpose.

    Cognitive load per function: MANAGEABLE
    """
    schema = design_database_schema()
    api = create_api_endpoints(schema)
    frontend = build_frontend()
    integrate_frontend_api(frontend, api)
    test_application()
    deploy()
```

### Complexity Management

**Complex tasks have exponential state spaces**:

```
Task with N steps, K options per step:
- State space: K^N possibilities
- Without decomposition: Must consider all K^N states
- With decomposition: Consider K options at a time

Example:
- 10 steps, 3 options each
- Without decomposition: 3^10 = 59,049 possibilities
- With decomposition: 3 options × 10 steps = 30 decisions
```

### Progress Monitoring

**Decomposition enables concrete progress metrics**:

```python
class Task:
    def __init__(self, name: str, subtasks: list['Task'] = None):
        self.name = name
        self.subtasks = subtasks or []
        self.completed = False

    def progress(self) -> float:
        """
        Calculate progress as fraction of completed subtasks.

        Without subtasks: 0% or 100%
        With subtasks: Granular progress tracking
        """
        if not self.subtasks:
            return 1.0 if self.completed else 0.0

        # Recursive progress calculation
        subtask_progress = [st.progress() for st in self.subtasks]
        return sum(subtask_progress) / len(self.subtasks)


# Example
project = Task("Build Web App", [
    Task("Design Database", completed=True),     # 100%
    Task("Build API", completed=True),            # 100%
    Task("Create Frontend", [                     # Partially complete
        Task("Setup React", completed=True),
        Task("Build Components", completed=False),
        Task("Style UI", completed=False)
    ]),
    Task("Testing", completed=False)              # 0%
])

print(f"Project progress: {project.progress():.1%}")
# Output: Project progress: 54.2%
```

### Error Isolation

**Decomposition contains failures**:

```
Monolithic task:
├─ Step 1 ✓
├─ Step 2 ✓
├─ Step 3 ✗ (fails)
└─ Entire task fails, must restart from beginning

Decomposed task:
├─ Subtask A ✓ (complete)
├─ Subtask B ✓ (complete)
├─ Subtask C ✗ (fails)
│   └─ Only need to retry Subtask C
└─ Subtask D (waiting)
```

## Decomposition Principles

Effective decomposition follows key principles.

### MECE Principle

**Mutually Exclusive, Collectively Exhaustive**:

```
Good decomposition (MECE):
Build Website
├─ Frontend Development (exclusive)
├─ Backend Development (exclusive)
├─ Database Design (exclusive)
└─ Deployment (exclusive)
All aspects covered, no overlap

Bad decomposition (not MECE):
Build Website
├─ Write Code (overlaps with others)
├─ Frontend Development (overlaps with "Write Code")
├─ Testing (not exhaustive - missing deployment)
└─ Add Features (vague, overlaps)
```

### Single Responsibility

**Each subtask should have one clear purpose**:

```python
# Good: Single responsibility
def authenticate_user(username: str, password: str) -> bool:
    """Only handles authentication."""
    return check_credentials(username, password)

def log_user_action(user_id: str, action: str):
    """Only handles logging."""
    write_to_log(user_id, action)

def process_user_login(username: str, password: str):
    """Coordinates login process."""
    if authenticate_user(username, password):
        log_user_action(username, "login")
        return True
    return False


# Bad: Multiple responsibilities
def handle_user_login(username: str, password: str):
    """Does too many things at once."""
    # Authentication
    if not check_credentials(username, password):
        return False

    # Logging
    write_to_log(username, "login")

    # Session management
    create_session(username)

    # Email notification
    send_welcome_email(username)

    # Analytics
    track_event("login", username)

    return True
```

### Appropriate Granularity

**Subtasks should be neither too large nor too small**:

```
Too coarse:
├─ Build entire frontend (still too complex)
└─ Build entire backend (still too complex)

Too fine:
├─ Type the letter 'i'
├─ Type the letter 'm'
├─ Type the letter 'p'
└─ ... (absurdly detailed)

Just right:
├─ Design component architecture
├─ Implement user authentication
├─ Create data models
└─ Build API endpoints
```

### Clear Dependencies

**Make dependencies explicit**:

```python
from dataclasses import dataclass
from typing import List

@dataclass
class Subtask:
    name: str
    dependencies: List[str]
    duration: int

    def can_start(self, completed: set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed for dep in self.dependencies)


# Example task graph
subtasks = [
    Subtask("Install dependencies", dependencies=[], duration=5),
    Subtask("Setup database", dependencies=["Install dependencies"], duration=10),
    Subtask("Create models", dependencies=["Setup database"], duration=15),
    Subtask("Build API", dependencies=["Create models"], duration=20),
    Subtask("Write tests", dependencies=["Build API"], duration=15),
]

# Execute in dependency order
completed = set()
while len(completed) < len(subtasks):
    for task in subtasks:
        if task.name not in completed and task.can_start(completed):
            print(f"Executing: {task.name}")
            # execute_task(task)
            completed.add(task.name)
```

## Hierarchical Decomposition

Complex tasks often require multiple levels of decomposition.

### Multi-Level Hierarchy

```
Level 0: Main Goal
  └─ Build E-commerce Platform

Level 1: Major Components
  ├─ User Management System
  ├─ Product Catalog
  ├─ Shopping Cart
  ├─ Payment Processing
  └─ Order Management

Level 2: Component Features
  ├─ User Management System
  │   ├─ User Registration
  │   ├─ Authentication
  │   ├─ Profile Management
  │   └─ Password Reset
  │
  └─ Product Catalog
      ├─ Product Database
      ├─ Search Functionality
      ├─ Category Management
      └─ Product Pages

Level 3: Feature Implementation
  └─ User Registration
      ├─ Design registration form
      ├─ Validate input data
      ├─ Create database entry
      ├─ Send confirmation email
      └─ Redirect to welcome page
```

### Tree Representation

```python
class TaskNode:
    """Node in hierarchical task tree."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.subtasks: List[TaskNode] = []
        self.completed = False
        self.parent: Optional[TaskNode] = None

    def add_subtask(self, subtask: 'TaskNode'):
        """Add a subtask to this node."""
        subtask.parent = self
        self.subtasks.append(subtask)

    def is_leaf(self) -> bool:
        """Check if this is a leaf (atomic) task."""
        return len(self.subtasks) == 0

    def is_completable(self) -> bool:
        """Check if task can be marked complete."""
        if self.is_leaf():
            return True  # Leaf tasks can be completed directly
        # Non-leaf tasks are complete when all subtasks are complete
        return all(st.completed for st in self.subtasks)

    def depth(self) -> int:
        """Calculate depth of this node in tree."""
        if not self.subtasks:
            return 0
        return 1 + max(st.depth() for st in self.subtasks)

    def visualize(self, indent: int = 0) -> str:
        """Create visual tree representation."""
        status = "✓" if self.completed else "○"
        lines = [f"{'  ' * indent}{status} {self.name}"]

        for subtask in self.subtasks:
            lines.append(subtask.visualize(indent + 1))

        return '\n'.join(lines)

    def get_next_task(self) -> Optional['TaskNode']:
        """
        Get next incomplete atomic task in depth-first order.

        Returns:
            Next task to work on, or None if all complete
        """
        if self.completed:
            return None

        if self.is_leaf():
            return self

        # Check subtasks
        for subtask in self.subtasks:
            next_task = subtask.get_next_task()
            if next_task:
                return next_task

        return None


# Example usage
def build_task_hierarchy() -> TaskNode:
    """Build example task hierarchy."""

    root = TaskNode("Build Blog Platform")

    # Level 1
    design = TaskNode("Design System")
    backend = TaskNode("Backend Development")
    frontend = TaskNode("Frontend Development")
    testing = TaskNode("Testing & QA")

    root.add_subtask(design)
    root.add_subtask(backend)
    root.add_subtask(frontend)
    root.add_subtask(testing)

    # Level 2 - Design
    design.add_subtask(TaskNode("Create wireframes"))
    design.add_subtask(TaskNode("Design database schema"))
    design.add_subtask(TaskNode("Define API contracts"))

    # Level 2 - Backend
    backend.add_subtask(TaskNode("Setup project structure"))
    backend.add_subtask(TaskNode("Implement authentication"))
    backend.add_subtask(TaskNode("Create blog post CRUD"))
    backend.add_subtask(TaskNode("Add comment system"))

    # Level 2 - Frontend
    frontend.add_subtask(TaskNode("Setup React app"))
    frontend.add_subtask(TaskNode("Build post editor"))
    frontend.add_subtask(TaskNode("Create post list view"))
    frontend.add_subtask(TaskNode("Add comment UI"))

    return root


# Visualize hierarchy
task_tree = build_task_hierarchy()
print(task_tree.visualize())
print(f"\nTree depth: {task_tree.depth()}")
print(f"Next task: {task_tree.get_next_task().name}")
```

### Recursive Decomposition

```python
def decompose_recursively(task: str, model, max_depth: int = 3,
                         current_depth: int = 0) -> TaskNode:
    """
    Recursively decompose task using language model.

    Args:
        task: Task description
        model: Language model for decomposition
        max_depth: Maximum recursion depth
        current_depth: Current recursion level

    Returns:
        TaskNode tree with hierarchical decomposition
    """
    node = TaskNode(task)

    # Base case: max depth or task is atomic
    if current_depth >= max_depth or is_atomic(task, model):
        return node

    # Decompose current task
    subtasks = generate_subtasks(task, model)

    # Recursively decompose each subtask
    for subtask_name in subtasks:
        subtask_node = decompose_recursively(
            subtask_name,
            model,
            max_depth,
            current_depth + 1
        )
        node.add_subtask(subtask_node)

    return node


def is_atomic(task: str, model) -> bool:
    """
    Check if task is atomic (cannot be further decomposed).

    Args:
        task: Task description
        model: Language model

    Returns:
        True if task is atomic
    """
    prompt = f"""
Task: {task}

Is this task atomic (a single, concrete action that cannot be meaningfully broken down)?

Answer yes or no:
"""

    response = model.generate(prompt, temperature=0.0)
    return "yes" in response.lower()


def generate_subtasks(task: str, model) -> List[str]:
    """
    Generate subtasks for given task.

    Args:
        task: Task description
        model: Language model

    Returns:
        List of subtask descriptions
    """
    prompt = f"""
Task: {task}

Break this task down into 3-5 subtasks. Each subtask should:
- Be a clear, concrete step
- Contribute to completing the main task
- Be at a similar level of abstraction

Subtasks:
1.
"""

    response = model.generate(prompt, temperature=0.3)

    # Parse numbered list
    subtasks = []
    for line in response.split('\n'):
        line = line.strip()
        if line and line[0].isdigit():
            # Remove number prefix
            subtask = line.split('.', 1)[1].strip() if '.' in line else line
            subtasks.append(subtask)

    return subtasks


# Example usage
task = "Create a RESTful API for a blog"
task_hierarchy = decompose_recursively(task, model, max_depth=3)
print(task_hierarchy.visualize())
```

## Decomposition Strategies

Different strategies work for different types of problems.

### Strategy 1: Process-Based

Break down by sequential process steps:

```python
def process_based_decomposition(task: str) -> List[str]:
    """
    Decompose based on process flow.

    Example: "Deploy application"
    → Build → Test → Package → Upload → Activate
    """
    prompt = f"""
Task: {task}

Decompose this into a sequential process with clear stages.
List the stages in order:

Stages:
1.
"""

    response = model.generate(prompt)
    return parse_numbered_list(response)


# Example
stages = process_based_decomposition("Deploy web application to production")
print(stages)
# Output:
# 1. Run test suite
# 2. Build production artifacts
# 3. Upload to server
# 4. Update configuration
# 5. Restart services
# 6. Verify deployment
```

### Strategy 2: Component-Based

Break down by system components:

```python
def component_based_decomposition(task: str) -> dict[str, List[str]]:
    """
    Decompose based on system components.

    Example: "Build web app"
    → Frontend | Backend | Database | Infrastructure
    """
    prompt = f"""
Task: {task}

Identify the major components or subsystems, then list tasks for each.

Components and tasks:
"""

    response = model.generate(prompt)

    # Parse into components
    components = {}
    current_component = None

    for line in response.split('\n'):
        line = line.strip()
        if line.endswith(':'):
            current_component = line[:-1]
            components[current_component] = []
        elif line and current_component:
            if line.startswith('-') or line.startswith('•'):
                components[current_component].append(line.lstrip('-•').strip())

    return components


# Example
components = component_based_decomposition("Build social media platform")
for component, tasks in components.items():
    print(f"\n{component}:")
    for task in tasks:
        print(f"  - {task}")
```

### Strategy 3: Goal-Based

Break down by goals and subgoals:

```python
def goal_based_decomposition(main_goal: str) -> dict:
    """
    Decompose based on goals hierarchy.

    Example: "Increase website traffic"
    → Improve SEO | Create content | Build backlinks | Social media
    """
    prompt = f"""
Main goal: {main_goal}

Identify 3-5 subgoals that, if achieved, would accomplish the main goal.
For each subgoal, list 2-3 concrete actions.

Subgoals:
"""

    response = model.generate(prompt)

    # Parse goals and actions
    goals = {}
    current_goal = None

    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue

        if line[0].isdigit() and '.' in line:
            # Main subgoal
            current_goal = line.split('.', 1)[1].strip()
            goals[current_goal] = []
        elif line.startswith(('-', '•', '*')) and current_goal:
            # Action for current goal
            action = line.lstrip('-•*').strip()
            goals[current_goal].append(action)

    return goals


# Example
goals = goal_based_decomposition("Launch successful product")
for goal, actions in goals.items():
    print(f"\n{goal}:")
    for action in actions:
        print(f"  → {action}")
```

### Strategy 4: Resource-Based

Break down by required resources or skills:

```python
def resource_based_decomposition(task: str) -> dict[str, List[str]]:
    """
    Decompose based on resources or skills needed.

    Example: "Organize conference"
    → Venue | Speakers | Marketing | Technical setup
    """
    prompt = f"""
Task: {task}

Identify different resources or skill areas needed.
For each, list specific subtasks:

Resource areas:
"""

    response = model.generate(prompt)
    return parse_resource_areas(response)
```

### Strategy 5: Time-Based

Break down by time phases:

```python
def time_based_decomposition(task: str) -> dict[str, List[str]]:
    """
    Decompose based on time phases.

    Example: "Product launch"
    → Pre-launch | Launch day | Post-launch
    """
    prompt = f"""
Task: {task}

Break this into time-based phases (e.g., preparation, execution, follow-up).
List tasks for each phase:

Phases:
"""

    response = model.generate(prompt)

    phases = {}
    current_phase = None

    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue

        if line.endswith(':') or line.startswith('Phase'):
            current_phase = line.rstrip(':')
            phases[current_phase] = []
        elif line and current_phase:
            if line[0].isdigit() or line.startswith(('-', '•')):
                phases[current_phase].append(line.lstrip('0123456789.-•').strip())

    return phases
```

## Identifying Subtasks

How do we identify good subtasks? Several techniques help.

### Question-Based Identification

```python
def identify_subtasks_via_questions(task: str) -> List[str]:
    """
    Identify subtasks by asking decomposition questions.

    Key questions:
    - What needs to happen first?
    - What are the major steps?
    - What resources are needed?
    - What could go wrong?
    - What are the milestones?
    """

    questions = [
        "What needs to happen first before we can proceed?",
        "What are the main steps to complete this task?",
        "What resources or preparations are needed?",
        "What are the key milestones or checkpoints?",
        "What could go wrong and need special handling?"
    ]

    subtasks = []

    for question in questions:
        prompt = f"""
Task: {task}

Question: {question}

Answer:
"""
        response = model.generate(prompt, temperature=0.3)

        # Extract actionable items from response
        items = extract_actionable_items(response)
        subtasks.extend(items)

    # Deduplicate and organize
    unique_subtasks = list(dict.fromkeys(subtasks))
    return unique_subtasks


def extract_actionable_items(text: str) -> List[str]:
    """Extract actionable items from text."""
    items = []

    for line in text.split('\n'):
        line = line.strip()
        # Look for items starting with action verbs
        if line and (line[0].isdigit() or line.startswith(('-', '•', '*'))):
            item = line.lstrip('0123456789.-•*').strip()
            if item:
                items.append(item)

    return items
```

### Dependency-Driven Identification

```python
def identify_subtasks_via_dependencies(task: str) -> List[dict]:
    """
    Identify subtasks by analyzing dependencies.

    Start with goal, work backwards to identify what's needed.
    """
    prompt = f"""
Task: {task}

Work backwards from the goal:
1. What is the final step or deliverable?
2. What must be complete before that final step?
3. What must be complete before those prerequisites?

List subtasks with their dependencies:

Subtasks:
"""

    response = model.generate(prompt)

    # Parse subtasks and dependencies
    subtasks = []

    for line in response.split('\n'):
        line = line.strip()
        if not line or not line[0].isdigit():
            continue

        # Parse format: "1. Task name (depends on: task2, task3)"
        if '(depends on:' in line.lower():
            task_part, dep_part = line.split('(depends on:', 1)
            task_name = task_part.split('.', 1)[1].strip()
            deps = [d.strip().rstrip(')') for d in dep_part.split(',')]
        else:
            task_name = line.split('.', 1)[1].strip()
            deps = []

        subtasks.append({
            'name': task_name,
            'dependencies': deps
        })

    return subtasks
```

### Template-Based Identification

```python
# Common task templates
TASK_TEMPLATES = {
    "software_development": [
        "Requirements gathering",
        "Design architecture",
        "Implement core functionality",
        "Write tests",
        "Documentation",
        "Code review",
        "Deployment"
    ],

    "research": [
        "Define research question",
        "Literature review",
        "Design methodology",
        "Collect data",
        "Analyze results",
        "Draw conclusions",
        "Write paper"
    ],

    "event_planning": [
        "Define objectives",
        "Set budget",
        "Choose venue",
        "Invite participants",
        "Arrange logistics",
        "Execute event",
        "Follow-up"
    ],

    "content_creation": [
        "Brainstorm topics",
        "Research subject",
        "Create outline",
        "Write draft",
        "Edit and revise",
        "Design visuals",
        "Publish"
    ]
}


def identify_subtasks_from_template(task: str) -> List[str]:
    """
    Use templates to identify subtasks.

    Args:
        task: Task description

    Returns:
        List of subtasks based on matching template
    """
    # Identify task type
    prompt = f"""
Task: {task}

What category does this task fall into?
Options: software_development, research, event_planning, content_creation, other

Category:
"""

    response = model.generate(prompt, temperature=0.0)
    category = response.strip().lower()

    if category in TASK_TEMPLATES:
        # Start with template
        template_subtasks = TASK_TEMPLATES[category].copy()

        # Customize template for specific task
        prompt = f"""
Task: {task}

Here's a template breakdown:
{chr(10).join(f'{i+1}. {st}' for i, st in enumerate(template_subtasks))}

Customize these subtasks for the specific task. Modify, add, or remove as needed:

Customized subtasks:
"""

        response = model.generate(prompt, temperature=0.3)
        customized = parse_numbered_list(response)

        return customized if customized else template_subtasks
    else:
        # No template, decompose from scratch
        return generate_subtasks(task, model)


def parse_numbered_list(text: str) -> List[str]:
    """Parse numbered list from text."""
    items = []
    for line in text.split('\n'):
        line = line.strip()
        if line and line[0].isdigit() and '.' in line:
            item = line.split('.', 1)[1].strip()
            items.append(item)
    return items
```

## Dependency Analysis

Understanding dependencies between subtasks is crucial for effective execution.

### Dependency Types

```python
from enum import Enum
from dataclasses import dataclass
from typing import Set

class DependencyType(Enum):
    """Types of dependencies between tasks."""
    PREREQUISITE = "prerequisite"      # Must complete before
    RESOURCE = "resource"               # Shared resource conflict
    DATA = "data"                       # Outputs needed as inputs
    TEMPORAL = "temporal"               # Time-based ordering
    OPTIONAL = "optional"               # Helpful but not required


@dataclass
class TaskDependency:
    """Represents a dependency between tasks."""
    from_task: str
    to_task: str
    dependency_type: DependencyType
    description: str = ""

    def is_satisfied(self, completed_tasks: Set[str]) -> bool:
        """Check if dependency is satisfied."""
        if self.dependency_type == DependencyType.OPTIONAL:
            return True  # Optional dependencies don't block
        return self.from_task in completed_tasks


class DependencyGraph:
    """Graph of task dependencies."""

    def __init__(self):
        self.tasks: Set[str] = set()
        self.dependencies: List[TaskDependency] = []

    def add_task(self, task: str):
        """Add a task to the graph."""
        self.tasks.add(task)

    def add_dependency(self, from_task: str, to_task: str,
                       dep_type: DependencyType = DependencyType.PREREQUISITE,
                       description: str = ""):
        """Add a dependency between tasks."""
        self.tasks.add(from_task)
        self.tasks.add(to_task)

        dep = TaskDependency(from_task, to_task, dep_type, description)
        self.dependencies.append(dep)

    def get_dependencies(self, task: str) -> List[TaskDependency]:
        """Get all dependencies for a task."""
        return [d for d in self.dependencies if d.to_task == task]

    def can_start(self, task: str, completed: Set[str]) -> bool:
        """Check if task can start given completed tasks."""
        deps = self.get_dependencies(task)
        return all(d.is_satisfied(completed) for d in deps)

    def get_ready_tasks(self, completed: Set[str]) -> Set[str]:
        """Get all tasks that can start now."""
        ready = set()
        for task in self.tasks:
            if task not in completed and self.can_start(task, completed):
                ready.add(task)
        return ready

    def topological_sort(self) -> List[str]:
        """
        Return tasks in valid execution order.

        Uses Kahn's algorithm for topological sorting.
        """
        # Calculate in-degree for each task
        in_degree = {task: 0 for task in self.tasks}
        for dep in self.dependencies:
            if dep.dependency_type != DependencyType.OPTIONAL:
                in_degree[dep.to_task] += 1

        # Queue of tasks with no dependencies
        queue = [task for task, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            # Sort to ensure deterministic order
            queue.sort()
            task = queue.pop(0)
            result.append(task)

            # Reduce in-degree of dependent tasks
            for dep in self.dependencies:
                if dep.from_task == task and dep.dependency_type != DependencyType.OPTIONAL:
                    in_degree[dep.to_task] -= 1
                    if in_degree[dep.to_task] == 0:
                        queue.append(dep.to_task)

        if len(result) != len(self.tasks):
            raise ValueError("Circular dependency detected!")

        return result

    def visualize(self) -> str:
        """Create visual representation of dependency graph."""
        lines = ["Task Dependencies:", ""]

        for task in sorted(self.tasks):
            deps = self.get_dependencies(task)
            if deps:
                lines.append(f"{task}:")
                for dep in deps:
                    symbol = "→" if dep.dependency_type == DependencyType.PREREQUISITE else "⇢"
                    lines.append(f"  {symbol} depends on: {dep.from_task}")
                    if dep.description:
                        lines.append(f"     ({dep.description})")
            else:
                lines.append(f"{task}: (no dependencies)")
            lines.append("")

        return '\n'.join(lines)


# Example usage
graph = DependencyGraph()

# Add tasks and dependencies for building a web app
graph.add_dependency("Install dependencies", "Setup database",
                     DependencyType.PREREQUISITE)
graph.add_dependency("Setup database", "Create models",
                     DependencyType.PREREQUISITE)
graph.add_dependency("Create models", "Build API",
                     DependencyType.DATA,
                     description="Models define API data structures")
graph.add_dependency("Build API", "Write API tests",
                     DependencyType.PREREQUISITE)
graph.add_dependency("Create models", "Setup frontend",
                     DependencyType.OPTIONAL,
                     description="Helpful to see data structures")
graph.add_dependency("Build API", "Setup frontend",
                     DependencyType.PREREQUISITE)
graph.add_dependency("Setup frontend", "Integrate API",
                     DependencyType.PREREQUISITE)

print(graph.visualize())
print("\nExecution order:")
for i, task in enumerate(graph.topological_sort(), 1):
    print(f"{i}. {task}")
```

### Automated Dependency Detection

```python
def detect_dependencies(subtasks: List[str], model) -> List[TaskDependency]:
    """
    Automatically detect dependencies between subtasks.

    Args:
        subtasks: List of subtask descriptions
        model: Language model

    Returns:
        List of detected dependencies
    """
    dependencies = []

    # Build prompt with all subtasks
    task_list = '\n'.join(f"{i+1}. {task}" for i, task in enumerate(subtasks))

    prompt = f"""
Subtasks:
{task_list}

For each subtask, identify which other subtasks (if any) must be completed first.

Dependencies (format: "Task X depends on Task Y"):
"""

    response = model.generate(prompt, temperature=0.0)

    # Parse dependencies
    for line in response.split('\n'):
        line = line.lower().strip()
        if 'depends on' in line:
            # Parse "task X depends on task Y"
            parts = line.split('depends on')
            if len(parts) == 2:
                to_task_part = parts[0].strip()
                from_task_part = parts[1].strip()

                # Extract task numbers or names
                to_task = extract_task_reference(to_task_part, subtasks)
                from_task = extract_task_reference(from_task_part, subtasks)

                if to_task and from_task:
                    dependencies.append(
                        TaskDependency(from_task, to_task,
                                     DependencyType.PREREQUISITE)
                    )

    return dependencies


def extract_task_reference(text: str, subtasks: List[str]) -> Optional[str]:
    """Extract task reference from text."""
    import re

    # Look for task number
    match = re.search(r'(\d+)', text)
    if match:
        task_num = int(match.group(1)) - 1
        if 0 <= task_num < len(subtasks):
            return subtasks[task_num]

    # Look for task name
    for task in subtasks:
        if task.lower() in text.lower():
            return task

    return None
```

## Sequencing Subtasks

Once we have subtasks and dependencies, we need to determine execution order.

### Critical Path Analysis

```python
from dataclasses import dataclass
from typing import Dict

@dataclass
class TaskInfo:
    """Information about a task for scheduling."""
    name: str
    duration: int
    dependencies: List[str]
    earliest_start: int = 0
    latest_start: int = 0
    slack: int = 0

    @property
    def earliest_finish(self) -> int:
        return self.earliest_start + self.duration

    @property
    def is_critical(self) -> bool:
        """Task is on critical path if it has no slack."""
        return self.slack == 0


def calculate_critical_path(tasks: List[TaskInfo]) -> Dict[str, TaskInfo]:
    """
    Calculate critical path using Critical Path Method (CPM).

    Returns:
        Dictionary mapping task names to TaskInfo with scheduling info
    """
    task_map = {t.name: t for t in tasks}

    # Forward pass: Calculate earliest start times
    completed = set()
    while len(completed) < len(tasks):
        for task in tasks:
            if task.name in completed:
                continue

            # Check if all dependencies are processed
            if all(dep in completed for dep in task.dependencies):
                # Earliest start is max of dependency finish times
                if task.dependencies:
                    dep_finishes = [task_map[dep].earliest_finish
                                   for dep in task.dependencies]
                    task.earliest_start = max(dep_finishes)
                else:
                    task.earliest_start = 0

                completed.add(task.name)

    # Project completion time
    project_duration = max(t.earliest_finish for t in tasks)

    # Backward pass: Calculate latest start times
    # Tasks with no dependents finish at project end
    dependents = {t.name: [] for t in tasks}
    for task in tasks:
        for dep in task.dependencies:
            dependents[dep].append(task.name)

    for task in reversed(tasks):
        if not dependents[task.name]:
            # No dependents - must finish by project end
            task.latest_start = project_duration - task.duration
        else:
            # Latest start to not delay any dependent
            dep_latest_starts = [task_map[dep_name].latest_start
                                for dep_name in dependents[task.name]]
            task.latest_start = min(dep_latest_starts) - task.duration

        # Calculate slack
        task.slack = task.latest_start - task.earliest_start

    return task_map


# Example
tasks = [
    TaskInfo("Design", 5, []),
    TaskInfo("Backend", 10, ["Design"]),
    TaskInfo("Frontend", 8, ["Design"]),
    TaskInfo("Database", 6, ["Design"]),
    TaskInfo("Testing", 4, ["Backend", "Frontend", "Database"]),
    TaskInfo("Deploy", 2, ["Testing"])
]

task_map = calculate_critical_path(tasks)

print("Critical Path Analysis:\n")
for task in tasks:
    critical = "⚠️ CRITICAL" if task.is_critical else ""
    print(f"{task.name}:")
    print(f"  Duration: {task.duration} days")
    print(f"  Earliest: {task.earliest_start}-{task.earliest_finish}")
    print(f"  Latest: {task.latest_start}-{task.latest_start + task.duration}")
    print(f"  Slack: {task.slack} days {critical}")
    print()

critical_path = [t.name for t in tasks if t.is_critical]
print(f"Critical Path: {' → '.join(critical_path)}")
```

## Parallel vs Sequential Execution

Determining which tasks can run in parallel improves efficiency.

### Parallelization Analysis

```python
def analyze_parallelization(graph: DependencyGraph) -> Dict[int, List[str]]:
    """
    Analyze which tasks can be executed in parallel.

    Returns:
        Dictionary mapping stage number to list of tasks that can run in parallel
    """
    stages = {}
    completed = set()
    stage = 0

    while len(completed) < len(graph.tasks):
        # Get all tasks that can start now
        ready = graph.get_ready_tasks(completed)

        if not ready:
            raise ValueError("Circular dependency or error in graph")

        # All ready tasks can run in parallel
        stages[stage] = sorted(list(ready))
        completed.update(ready)
        stage += 1

    return stages


def visualize_parallel_stages(stages: Dict[int, List[str]]) -> str:
    """Create visual representation of parallel execution stages."""
    lines = ["Parallel Execution Plan:", ""]

    for stage_num, tasks in sorted(stages.items()):
        lines.append(f"Stage {stage_num}:")
        if len(tasks) == 1:
            lines.append(f"  └─ {tasks[0]}")
        else:
            lines.append(f"  ┌─ {tasks[0]}")
            for task in tasks[1:-1]:
                lines.append(f"  ├─ {task}")
            lines.append(f"  └─ {tasks[-1]}")
        lines.append(f"  ({len(tasks)} task(s) in parallel)")
        lines.append("")

    return '\n'.join(lines)


# Example
graph = DependencyGraph()
graph.add_dependency("Install deps", "Setup DB")
graph.add_dependency("Install deps", "Setup API")
graph.add_dependency("Install deps", "Setup Frontend")
graph.add_dependency("Setup DB", "Create models")
graph.add_dependency("Setup API", "Build endpoints")
graph.add_dependency("Setup Frontend", "Create components")
graph.add_dependency("Create models", "Integration")
graph.add_dependency("Build endpoints", "Integration")
graph.add_dependency("Create components", "Integration")
graph.add_dependency("Integration", "Testing")

stages = analyze_parallelization(graph)
print(visualize_parallel_stages(stages))
```

## Implementing Decomposition

Practical implementation of task decomposition in agent systems.

### Decomposition Agent

```python
class TaskDecompositionAgent:
    """Agent that decomposes and executes complex tasks."""

    def __init__(self, model, tools):
        self.model = model
        self.tools = tools
        self.task_tree: Optional[TaskNode] = None
        self.dependency_graph: Optional[DependencyGraph] = None

    def execute(self, task: str) -> dict:
        """
        Execute complex task through decomposition.

        Process:
        1. Decompose task into subtasks
        2. Analyze dependencies
        3. Create execution plan
        4. Execute subtasks in order
        5. Track progress
        """
        print(f"Executing: {task}\n")

        # Step 1: Decompose
        print("Step 1: Decomposing task...")
        self.task_tree = self._decompose_task(task)
        print(self.task_tree.visualize())
        print()

        # Step 2: Analyze dependencies
        print("Step 2: Analyzing dependencies...")
        subtasks = self._extract_leaf_tasks(self.task_tree)
        self.dependency_graph = self._build_dependency_graph(subtasks)
        print(self.dependency_graph.visualize())

        # Step 3: Create execution plan
        print("Step 3: Creating execution plan...")
        execution_order = self.dependency_graph.topological_sort()
        print("Execution order:")
        for i, task in enumerate(execution_order, 1):
            print(f"  {i}. {task}")
        print()

        # Step 4: Execute subtasks
        print("Step 4: Executing subtasks...")
        results = self._execute_subtasks(execution_order)

        # Step 5: Report results
        return {
            "task": task,
            "decomposition": self.task_tree,
            "execution_order": execution_order,
            "results": results,
            "success": all(r.get("success", False) for r in results.values())
        }

    def _decompose_task(self, task: str, max_depth: int = 2) -> TaskNode:
        """Decompose task into hierarchy."""
        return decompose_recursively(task, self.model, max_depth=max_depth)

    def _extract_leaf_tasks(self, node: TaskNode) -> List[str]:
        """Extract all leaf (atomic) tasks from tree."""
        if node.is_leaf():
            return [node.name]

        leaves = []
        for child in node.subtasks:
            leaves.extend(self._extract_leaf_tasks(child))
        return leaves

    def _build_dependency_graph(self, subtasks: List[str]) -> DependencyGraph:
        """Build dependency graph from subtasks."""
        graph = DependencyGraph()

        # Add all tasks
        for task in subtasks:
            graph.add_task(task)

        # Detect dependencies
        dependencies = detect_dependencies(subtasks, self.model)

        for dep in dependencies:
            graph.add_dependency(dep.from_task, dep.to_task, dep.dependency_type)

        return graph

    def _execute_subtasks(self, execution_order: List[str]) -> Dict[str, dict]:
        """Execute subtasks in dependency order."""
        results = {}
        completed = set()

        for task in execution_order:
            print(f"\n  Executing: {task}")

            # Execute task
            result = self._execute_single_task(task)
            results[task] = result

            if result.get("success"):
                completed.add(task)
                print(f"  ✓ Completed: {task}")
            else:
                print(f"  ✗ Failed: {task}")
                print(f"    Error: {result.get('error', 'Unknown error')}")

        print()
        return results

    def _execute_single_task(self, task: str) -> dict:
        """Execute a single atomic task."""
        # Determine action needed
        prompt = f"""
Task: {task}

What action should be taken to complete this task?
Available tools: {', '.join(t.name for t in self.tools)}

Action:
"""

        response = self.model.generate(prompt, temperature=0.3)

        # Parse and execute action
        # (Simplified - would use proper tool calling in production)
        try:
            # Simulate task execution
            return {"success": True, "output": f"Completed: {task}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Example usage
agent = TaskDecompositionAgent(model, tools=[])

result = agent.execute("Create a REST API for a todo list application")

print(f"\nTask: {result['task']}")
print(f"Success: {result['success']}")
print(f"Subtasks completed: {len(result['results'])}")
```

## Common Decomposition Patterns

Recognizable patterns that appear across many domains.

### Pattern 1: ETL (Extract-Transform-Load)

```
1. Extract data from source
2. Transform/clean data
3. Load into destination
4. Verify integrity
```

### Pattern 2: Request-Process-Respond

```
1. Receive request
2. Validate input
3. Process request
4. Format response
5. Send response
```

### Pattern 3: Plan-Do-Check-Act

```
1. Plan approach
2. Execute plan
3. Check results
4. Act on findings
```

### Pattern 4: Divide-and-Conquer

```
1. Divide problem into subproblems
2. Solve each subproblem
3. Combine solutions
```

### Pattern Recognition

```python
COMMON_PATTERNS = {
    "ETL": ["extract", "transform", "load", "verify"],
    "Request-Response": ["receive", "validate", "process", "format", "send"],
    "PDCA": ["plan", "do", "check", "act"],
    "Divide-Conquer": ["divide", "solve", "combine"],
    "Build-Test-Deploy": ["build", "test", "deploy", "monitor"]
}


def detect_pattern(subtasks: List[str]) -> Optional[str]:
    """Detect if subtasks follow a common pattern."""
    subtasks_lower = [t.lower() for t in subtasks]

    for pattern_name, pattern_keywords in COMMON_PATTERNS.items():
        matches = sum(1 for keyword in pattern_keywords
                     if any(keyword in task for task in subtasks_lower))

        if matches >= len(pattern_keywords) * 0.6:  # 60% match
            return pattern_name

    return None
```

## Summary

Task decomposition is essential for agents tackling complex problems. By breaking down tasks hierarchically, analyzing dependencies, and creating execution plans, agents can handle challenges that would be overwhelming as monolithic tasks.

### Key Takeaways

1. **Decomposition reduces complexity**: Breaking tasks into subtasks makes problems manageable

2. **Multiple strategies exist**: Process-based, component-based, goal-based, time-based approaches

3. **Dependencies matter**: Understanding task dependencies enables correct sequencing

4. **Hierarchy provides structure**: Multi-level decomposition handles complex tasks

5. **Parallelization improves efficiency**: Independent subtasks can execute simultaneously

6. **Patterns recur**: Recognizing common patterns speeds decomposition

7. **Appropriate granularity is key**: Subtasks should be neither too large nor too small

### Best Practices

1. **Use MECE principle**: Subtasks should be mutually exclusive and collectively exhaustive
2. **Make dependencies explicit**: Clear dependencies enable proper sequencing
3. **Choose appropriate strategy**: Match decomposition strategy to problem type
4. **Validate completeness**: Ensure subtasks fully cover the main task
5. **Monitor granularity**: Subtasks should be concrete and actionable
6. **Consider parallelization**: Identify independent subtasks that can run simultaneously
7. **Iterate as needed**: Refine decomposition based on execution feedback

## Next Steps

Explore related planning and execution topics:

- **[Planning Strategies](planning-strategies.md)**: Learn forward/backward planning and hierarchical approaches
- **[Task Tracking](task-tracking.md)**: Maintain todo lists and track progress
- **[Implementation Plans](implementation-plans.md)**: Create detailed execution specifications
- **[Chain-of-Thought](chain-of-thought.md)**: Apply reasoning to decomposition decisions
- **[ReAct Pattern](../agent-architectures/react.md)**: Combine reasoning with action execution

Master task decomposition to build agents that handle real-world complexity! 🎯
