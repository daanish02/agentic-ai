# Planning Strategies

## Table of Contents

- [Introduction](#introduction)
- [What is Planning?](#what-is-planning)
- [Forward Planning](#forward-planning)
- [Backward Planning](#backward-planning)
- [Hierarchical Planning](#hierarchical-planning)
- [Reactive vs Deliberative Planning](#reactive-vs-deliberative-planning)
- [When to Plan Upfront](#when-to-plan-upfront)
- [Dynamic Replanning](#dynamic-replanning)
- [Plan Representation](#plan-representation)
- [Plan Execution and Monitoring](#plan-execution-and-monitoring)
- [Handling Plan Failures](#handling-plan-failures)
- [Partial Planning](#partial-planning)
- [Multi-Agent Planning](#multi-agent-planning)
- [Planning Under Constraints](#planning-under-constraints)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Planning is the process of determining a sequence of actions to achieve a goal. While task decomposition breaks problems into subtasks, planning determines **how** and **when** to execute those subtasks, anticipating obstacles and resource constraints.

> "Plans are worthless, but planning is everything." - Dwight D. Eisenhower

The paradox of planning: detailed plans often fail, yet the act of planning prepares you to adapt. Good agents balance planning with flexibility.

### Planning vs Reactive Behavior

```
Reactive Agent:
  Observe → Act → Observe → Act → ...
  No planning, responds to immediate situation

Planning Agent:
  Observe → Plan → Execute Plan → Monitor → Replan if needed
  Considers future consequences, coordinates actions
```

**When you need planning**:
- Multi-step tasks with dependencies
- Resource constraints
- Coordination requirements
- Non-reversible actions
- High cost of failure

**When reactive is sufficient**:
- Simple stimulus-response tasks
- Rapidly changing environments
- Low-stakes decisions
- Real-time constraints

## What is Planning?

Planning is the deliberate process of creating a sequence of actions to transform the current state into a goal state.

### Core Components

**1. Initial State**: Where we are now
```python
initial_state = {
    "location": "home",
    "has_groceries": False,
    "has_ingredients": False,
    "meal_prepared": False
}
```

**2. Goal State**: Where we want to be
```python
goal_state = {
    "meal_prepared": True
}
```

**3. Actions**: Available operations
```python
actions = [
    {"name": "go_to_store", "preconditions": ["at_home"], "effects": ["at_store"]},
    {"name": "buy_groceries", "preconditions": ["at_store"], "effects": ["has_groceries"]},
    {"name": "go_home", "preconditions": ["at_store"], "effects": ["at_home"]},
    {"name": "cook_meal", "preconditions": ["at_home", "has_groceries"], "effects": ["meal_prepared"]}
]
```

**4. Plan**: Sequence of actions
```python
plan = [
    "go_to_store",
    "buy_groceries",
    "go_home",
    "cook_meal"
]
```

### Planning as Search

Planning can be viewed as searching through the space of possible action sequences:

```
                    Initial State
                         │
        ┌────────────────┼────────────────┐
        │                │                │
     Action 1         Action 2        Action 3
        │                │                │
    State 1.1        State 1.2       State 1.3
        │                │                │
      [Continue searching until goal state reached]
```

### Planning Representations

```python
from dataclasses import dataclass
from typing import List, Set, Dict, Optional

@dataclass
class State:
    """Represents a world state."""
    facts: Set[str]  # True propositions in this state
    
    def satisfies(self, conditions: Set[str]) -> bool:
        """Check if this state satisfies all conditions."""
        return conditions.issubset(self.facts)
    
    def apply_effects(self, effects: Dict[str, bool]) -> 'State':
        """Apply effects to create new state."""
        new_facts = self.facts.copy()
        for fact, value in effects.items():
            if value:
                new_facts.add(fact)
            else:
                new_facts.discard(fact)
        return State(new_facts)


@dataclass
class Action:
    """Represents an action that can be performed."""
    name: str
    preconditions: Set[str]
    effects: Dict[str, bool]  # fact -> True (add) or False (remove)
    cost: float = 1.0
    
    def is_applicable(self, state: State) -> bool:
        """Check if action can be performed in this state."""
        return state.satisfies(self.preconditions)
    
    def apply(self, state: State) -> State:
        """Apply action to state, returning new state."""
        if not self.is_applicable(state):
            raise ValueError(f"Action {self.name} not applicable in current state")
        return state.apply_effects(self.effects)


@dataclass
class Plan:
    """Represents a plan (sequence of actions)."""
    actions: List[Action]
    initial_state: State
    goal: Set[str]
    
    def is_valid(self) -> bool:
        """Check if plan achieves goal from initial state."""
        current_state = self.initial_state
        
        for action in self.actions:
            if not action.is_applicable(current_state):
                return False
            current_state = action.apply(current_state)
        
        return current_state.satisfies(self.goal)
    
    def total_cost(self) -> float:
        """Calculate total cost of plan."""
        return sum(action.cost for action in self.actions)
    
    def __len__(self) -> int:
        return len(self.actions)
    
    def __str__(self) -> str:
        steps = [f"{i+1}. {action.name}" for i, action in enumerate(self.actions)]
        return "\n".join(steps)
```

## Forward Planning

Forward planning (progression) starts from the initial state and searches forward toward the goal.

### Forward Search Algorithm

```python
from collections import deque
from typing import List, Optional

def forward_planning(initial_state: State, 
                    goal: Set[str],
                    actions: List[Action]) -> Optional[Plan]:
    """
    Find plan using forward search (BFS).
    
    Args:
        initial_state: Starting state
        goal: Goal conditions to achieve
        actions: Available actions
        
    Returns:
        Plan if found, None otherwise
    """
    # Queue of (state, path) tuples
    queue = deque([(initial_state, [])])
    visited = {frozenset(initial_state.facts)}
    
    while queue:
        current_state, path = queue.popleft()
        
        # Check if goal reached
        if current_state.satisfies(goal):
            return Plan(path, initial_state, goal)
        
        # Try each applicable action
        for action in actions:
            if action.is_applicable(current_state):
                new_state = action.apply(current_state)
                state_key = frozenset(new_state.facts)
                
                if state_key not in visited:
                    visited.add(state_key)
                    new_path = path + [action]
                    queue.append((new_state, new_path))
    
    return None  # No plan found


# Example usage
initial = State(facts={"at_home", "hungry", "tired"})
goal = {"meal_ready", "not_hungry"}

actions = [
    Action(
        name="go_to_kitchen",
        preconditions={"at_home"},
        effects={"at_kitchen": True, "at_home": False}
    ),
    Action(
        name="cook_meal",
        preconditions={"at_kitchen"},
        effects={"meal_ready": True},
        cost=10
    ),
    Action(
        name="eat_meal",
        preconditions={"meal_ready"},
        effects={"not_hungry": True, "hungry": False}
    ),
]

plan = forward_planning(initial, goal, actions)
if plan:
    print("Plan found:")
    print(plan)
    print(f"Total cost: {plan.total_cost()}")
else:
    print("No plan found")
```

### Forward Planning with Heuristics

```python
import heapq

def forward_planning_astar(initial_state: State,
                           goal: Set[str],
                           actions: List[Action],
                           heuristic) -> Optional[Plan]:
    """
    Forward planning using A* search with heuristic.
    
    Args:
        initial_state: Starting state
        goal: Goal conditions
        actions: Available actions
        heuristic: Function estimating cost to goal
        
    Returns:
        Optimal plan if found
    """
    # Priority queue: (f_score, g_score, state, path)
    # f_score = g_score + heuristic(state)
    h_initial = heuristic(initial_state, goal)
    pq = [(h_initial, 0, initial_state, [])]
    visited = {}
    
    while pq:
        f_score, g_score, current_state, path = heapq.heappop(pq)
        
        state_key = frozenset(current_state.facts)
        
        # Skip if we've seen this state with lower cost
        if state_key in visited and visited[state_key] <= g_score:
            continue
        
        visited[state_key] = g_score
        
        # Check goal
        if current_state.satisfies(goal):
            return Plan(path, initial_state, goal)
        
        # Expand neighbors
        for action in actions:
            if action.is_applicable(current_state):
                new_state = action.apply(current_state)
                new_g_score = g_score + action.cost
                new_h_score = heuristic(new_state, goal)
                new_f_score = new_g_score + new_h_score
                new_path = path + [action]
                
                heapq.heappush(pq, (new_f_score, new_g_score, new_state, new_path))
    
    return None


def simple_heuristic(state: State, goal: Set[str]) -> float:
    """
    Simple heuristic: count unsatisfied goal conditions.
    
    Admissible (never overestimates) if each action achieves at most 1 goal.
    """
    unsatisfied = goal - state.facts
    return len(unsatisfied)


# Example with heuristic search
plan = forward_planning_astar(initial, goal, actions, simple_heuristic)
```

### Forward Planning Advantages

**Pros**:
- Natural and intuitive
- Easy to implement
- Good for exploring possibilities
- Works well with complete action models

**Cons**:
- Can explore irrelevant states
- Branching factor can be large
- May take long time to reach distant goals

## Backward Planning

Backward planning (regression) starts from the goal and works backward to the initial state.

### Backward Search Algorithm

```python
def backward_planning(initial_state: State,
                     goal: Set[str],
                     actions: List[Action]) -> Optional[Plan]:
    """
    Find plan using backward search from goal to initial state.
    
    Args:
        initial_state: Starting state  
        goal: Goal conditions
        actions: Available actions
        
    Returns:
        Plan if found
    """
    # Queue of (subgoal, path) tuples
    # Path is built in reverse
    queue = deque([(goal, [])])
    visited = {frozenset(goal)}
    
    while queue:
        current_goal, reverse_path = queue.popleft()
        
        # Check if current subgoal satisfied by initial state
        if initial_state.satisfies(current_goal):
            # Reverse the path to get forward plan
            return Plan(list(reversed(reverse_path)), initial_state, goal)
        
        # Try regressing through each action
        for action in actions:
            # Can we use this action to achieve current goal?
            if action.effects and any(effect in current_goal 
                                     for effect, value in action.effects.items() 
                                     if value):
                # Compute new subgoal (regression)
                new_subgoal = regress(current_goal, action)
                subgoal_key = frozenset(new_subgoal)
                
                if subgoal_key not in visited:
                    visited.add(subgoal_key)
                    new_path = reverse_path + [action]
                    queue.append((new_subgoal, new_path))
    
    return None


def regress(goal: Set[str], action: Action) -> Set[str]:
    """
    Compute regression: subgoal needed before action to achieve goal after.
    
    Formula: (Goal - PositiveEffects) ∪ Preconditions
    """
    # Remove facts added by action
    new_goal = goal.copy()
    for fact, value in action.effects.items():
        if value and fact in new_goal:
            new_goal.remove(fact)
    
    # Add preconditions
    new_goal.update(action.preconditions)
    
    return new_goal


# Example
plan = backward_planning(initial, goal, actions)
```

### Backward Planning Advantages

**Pros**:
- More focused on goal
- Smaller search space for some problems
- Natural for goal-directed reasoning

**Cons**:
- Requires action reversibility
- Can be conceptually complex
- Not suitable for all domains

### Forward vs Backward Comparison

```python
def compare_search_strategies(initial: State, goal: Set[str], 
                             actions: List[Action]):
    """Compare forward and backward planning."""
    
    import time
    
    # Forward planning
    start = time.time()
    forward_plan = forward_planning(initial, goal, actions)
    forward_time = time.time() - start
    
    # Backward planning  
    start = time.time()
    backward_plan = backward_planning(initial, goal, actions)
    backward_time = time.time() - start
    
    print("Forward Planning:")
    print(f"  Time: {forward_time:.4f}s")
    print(f"  Plan length: {len(forward_plan) if forward_plan else 'None'}")
    
    print("\nBackward Planning:")
    print(f"  Time: {backward_time:.4f}s")
    print(f"  Plan length: {len(backward_plan) if backward_plan else 'None'}")
    
    if forward_plan and backward_plan:
        print(f"\nBoth found plans of length {len(forward_plan)} and {len(backward_plan)}")
```

## Hierarchical Planning

Hierarchical planning operates at multiple levels of abstraction, refining high-level plans into detailed actions.

### HTN (Hierarchical Task Network) Planning

```python
from dataclasses import dataclass
from typing import Union

@dataclass
class PrimitiveTask:
    """A primitive (directly executable) task."""
    name: str
    preconditions: Set[str]
    effects: Dict[str, bool]
    
    def is_primitive(self) -> bool:
        return True


@dataclass  
class CompoundTask:
    """A compound task that decomposes into subtasks."""
    name: str
    methods: List['Method']  # Ways to decompose this task
    
    def is_primitive(self) -> bool:
        return False


@dataclass
class Method:
    """A way to decompose a compound task."""
    name: str
    preconditions: Set[str]
    subtasks: List[Union[PrimitiveTask, CompoundTask]]


class HTNPlanner:
    """Hierarchical Task Network planner."""
    
    def __init__(self, primitive_tasks: Dict[str, PrimitiveTask],
                 compound_tasks: Dict[str, CompoundTask]):
        self.primitive_tasks = primitive_tasks
        self.compound_tasks = compound_tasks
    
    def plan(self, task_list: List[str], initial_state: State) -> Optional[List[PrimitiveTask]]:
        """
        Create plan for task list.
        
        Args:
            task_list: List of task names to accomplish
            initial_state: Starting state
            
        Returns:
            List of primitive actions, or None if no plan
        """
        return self._htn_recursive(task_list, initial_state)
    
    def _htn_recursive(self, task_list: List[str], 
                       state: State) -> Optional[List[PrimitiveTask]]:
        """Recursive HTN planning."""
        if not task_list:
            return []  # No tasks, empty plan
        
        # Get first task
        task_name = task_list[0]
        remaining = task_list[1:]
        
        # Check if primitive
        if task_name in self.primitive_tasks:
            task = self.primitive_tasks[task_name]
            
            # Check if applicable
            if state.satisfies(task.preconditions):
                # Apply task
                new_state = state.apply_effects(task.effects)
                
                # Plan for remaining tasks
                rest_plan = self._htn_recursive(remaining, new_state)
                
                if rest_plan is not None:
                    return [task] + rest_plan
            
            return None  # Task not applicable
        
        # Compound task - try methods
        elif task_name in self.compound_tasks:
            compound = self.compound_tasks[task_name]
            
            # Try each method
            for method in compound.methods:
                # Check method preconditions
                if not state.satisfies(method.preconditions):
                    continue
                
                # Decompose: replace compound task with subtasks
                new_task_list = [st.name for st in method.subtasks] + remaining
                
                # Recursively plan
                plan = self._htn_recursive(new_task_list, state)
                
                if plan is not None:
                    return plan
            
            return None  # No method succeeded
        
        else:
            raise ValueError(f"Unknown task: {task_name}")


# Example: Planning a trip
primitive_tasks = {
    "buy_ticket": PrimitiveTask(
        name="buy_ticket",
        preconditions={"has_money"},
        effects={"has_ticket": True, "has_money": False}
    ),
    "pack_bags": PrimitiveTask(
        name="pack_bags",
        preconditions={"at_home"},
        effects={"bags_packed": True}
    ),
    "drive_to_airport": PrimitiveTask(
        name="drive_to_airport",
        preconditions={"at_home", "bags_packed"},
        effects={"at_airport": True, "at_home": False}
    ),
    "check_in": PrimitiveTask(
        name="check_in",
        preconditions={"at_airport", "has_ticket"},
        effects={"checked_in": True}
    ),
}

# Method: travel by plane
travel_by_plane = Method(
    name="travel_by_plane",
    preconditions={"has_money"},
    subtasks=[
        compound_tasks["get_to_airport"],  # Compound
        primitive_tasks["check_in"],        # Primitive
    ]
)

# Method: get to airport
get_to_airport_method = Method(
    name="drive_method",
    preconditions={"at_home"},
    subtasks=[
        primitive_tasks["pack_bags"],
        primitive_tasks["drive_to_airport"],
    ]
)

compound_tasks = {
    "travel": CompoundTask(
        name="travel",
        methods=[travel_by_plane]
    ),
    "get_to_airport": CompoundTask(
        name="get_to_airport",
        methods=[get_to_airport_method]
    ),
}

# Plan
planner = HTNPlanner(primitive_tasks, compound_tasks)
initial = State(facts={"at_home", "has_money"})
plan = planner.plan(["travel"], initial)

if plan:
    print("Hierarchical Plan:")
    for i, action in enumerate(plan, 1):
        print(f"{i}. {action.name}")
```

### Benefits of Hierarchical Planning

**Advantages**:
1. **Captures domain knowledge**: Methods encode expert strategies
2. **Reduces search space**: High-level reasoning before details
3. **More natural**: Matches human planning process
4. **Efficient**: Avoids low-level search
5. **Reusable**: Methods can be reused across problems

**When to use**:
- Complex domains with expert knowledge
- Multiple levels of abstraction
- Recurring subproblems
- Large-scale planning problems

## Reactive vs Deliberative Planning

Balancing planning time with execution flexibility.

### The Spectrum

```
Pure Reactive                              Pure Deliberative
    │                                              │
    │                                              │
No planning ←──────────────────────────────→ Complete planning
Fast response                              Optimal solutions
Adapts instantly                           Slow to adapt
No foresight                               Maximum foresight
```

### Hybrid Approaches

```python
class HybridPlanner:
    """
    Combines reactive and deliberative planning.
    
    Strategy: 
    - Plan high-level strategy (deliberative)
    - Execute with reactive adjustments
    """
    
    def __init__(self, model, replanning_threshold: float = 0.3):
        self.model = model
        self.replanning_threshold = replanning_threshold
        self.current_plan: Optional[List[Action]] = None
        self.plan_step = 0
    
    def execute_with_adaptation(self, initial_state: State, 
                                goal: Set[str],
                                actions: List[Action]) -> List[dict]:
        """
        Execute with adaptive replanning.
        
        Returns:
            Execution trace with states and actions taken
        """
        # Initial planning (deliberative)
        self.current_plan = self._create_plan(initial_state, goal, actions)
        
        if not self.current_plan:
            return [{"error": "No initial plan found"}]
        
        # Execute with monitoring (reactive)
        trace = []
        current_state = initial_state
        self.plan_step = 0
        
        while self.plan_step < len(self.current_plan):
            action = self.current_plan[self.plan_step]
            
            # Check if action still applicable (environment may have changed)
            if not action.is_applicable(current_state):
                # Reactive response: replan
                trace.append({
                    "step": self.plan_step,
                    "event": "plan_invalid",
                    "action_attempted": action.name,
                    "response": "replanning"
                })
                
                self.current_plan = self._create_plan(current_state, goal, actions)
                if not self.current_plan:
                    trace.append({"error": "Replanning failed"})
                    break
                
                self.plan_step = 0
                continue
            
            # Execute action
            new_state = action.apply(current_state)
            
            trace.append({
                "step": self.plan_step,
                "action": action.name,
                "state_before": current_state.facts,
                "state_after": new_state.facts
            })
            
            current_state = new_state
            self.plan_step += 1
            
            # Check if replanning would be beneficial
            if self._should_replan(current_state, goal):
                trace.append({
                    "step": self.plan_step,
                    "event": "opportunistic_replan"
                })
                
                new_plan = self._create_plan(current_state, goal, actions)
                if new_plan and len(new_plan) < len(self.current_plan) - self.plan_step:
                    # Found better plan
                    self.current_plan = new_plan
                    self.plan_step = 0
        
        return trace
    
    def _create_plan(self, state: State, goal: Set[str], 
                    actions: List[Action]) -> Optional[List[Action]]:
        """Create plan using forward search."""
        return forward_planning(state, goal, actions)
    
    def _should_replan(self, state: State, goal: Set[str]) -> bool:
        """Heuristic: should we replan now?"""
        # Simplified: replan with some probability
        import random
        return random.random() < self.replanning_threshold
```

## When to Plan Upfront

Deciding how much to plan before acting.

### Decision Factors

```python
def decide_planning_depth(task_complexity: int,
                         environment_stability: float,
                         action_reversibility: bool,
                         time_available: float,
                         cost_of_failure: float) -> str:
    """
    Decide how much upfront planning to do.
    
    Args:
        task_complexity: 1-10 (simple to complex)
        environment_stability: 0.0-1.0 (chaotic to stable)
        action_reversibility: Can actions be undone?
        time_available: Time budget for planning (seconds)
        cost_of_failure: How bad is failure? (0.0-1.0)
        
    Returns:
        Planning strategy recommendation
    """
    score = 0
    
    # Complex tasks benefit from planning
    score += task_complexity * 2
    
    # Stable environments allow longer-term plans
    score += environment_stability * 10
    
    # Irreversible actions require careful planning
    if not action_reversibility:
        score += 15
    
    # High cost of failure demands more planning
    score += cost_of_failure * 20
    
    # But limited time constrains planning
    if time_available < 5:
        score *= 0.5
    
    # Classify
    if score < 15:
        return "reactive"  # Little/no upfront planning
    elif score < 35:
        return "partial"   # Plan next few steps
    elif score < 55:
        return "moderate"  # Plan to intermediate checkpoints
    else:
        return "extensive" # Plan completely upfront


# Examples
print(decide_planning_depth(
    task_complexity=8,
    environment_stability=0.9,
    action_reversibility=False,
    time_available=60,
    cost_of_failure=0.8
))  # → "extensive"

print(decide_planning_depth(
    task_complexity=3,
    environment_stability=0.3,
    action_reversibility=True,
    time_available=2,
    cost_of_failure=0.2
))  # → "reactive"
```

### Partial Planning Strategy

```python
class PartialPlanner:
    """
    Plans only a few steps ahead, replanning frequently.
    
    Good for dynamic environments where long-term plans become invalid quickly.
    """
    
    def __init__(self, planning_horizon: int = 3):
        self.planning_horizon = planning_horizon
    
    def execute(self, initial_state: State, goal: Set[str],
               actions: List[Action]) -> List[dict]:
        """
        Execute with partial planning (rolling horizon).
        
        Plans only N steps ahead, then replans.
        """
        trace = []
        current_state = initial_state
        total_steps = 0
        
        while not current_state.satisfies(goal):
            # Plan next few steps
            partial_plan = self._plan_horizon(current_state, goal, actions)
            
            if not partial_plan:
                trace.append({"error": "No plan found"})
                break
            
            # Execute planned steps (or until goal reached)
            steps_executed = 0
            for action in partial_plan:
                if current_state.satisfies(goal):
                    break
                
                if action.is_applicable(current_state):
                    current_state = action.apply(current_state)
                    trace.append({
                        "step": total_steps,
                        "action": action.name,
                        "state": current_state.facts
                    })
                    steps_executed += 1
                    total_steps += 1
                else:
                    # Environment changed, break and replan
                    trace.append({
                        "step": total_steps,
                        "event": "action_inapplicable",
                        "action": action.name
                    })
                    break
            
            if steps_executed == 0:
                # Made no progress
                break
        
        return trace
    
    def _plan_horizon(self, state: State, goal: Set[str],
                     actions: List[Action]) -> List[Action]:
        """
        Plan for limited horizon.
        
        Returns up to N actions toward goal.
        """
        plan = forward_planning(state, goal, actions)
        
        if plan:
            # Limit to horizon
            return plan.actions[:self.planning_horizon]
        
        return []
```

## Dynamic Replanning

Adapting plans when circumstances change.

### Triggers for Replanning

```python
from enum import Enum

class ReplanTrigger(Enum):
    """Reasons to replan."""
    ACTION_FAILED = "action_failed"
    PRECONDITION_VIOLATED = "precondition_violated"
    BETTER_OPPORTUNITY = "better_opportunity"
    GOAL_CHANGED = "goal_changed"
    NEW_INFORMATION = "new_information"
    TIME_ELAPSED = "time_elapsed"


class AdaptivePlanner:
    """Planner that adapts to changing circumstances."""
    
    def __init__(self, model):
        self.model = model
        self.current_plan: Optional[List[Action]] = None
        self.execution_history: List[dict] = []
    
    def execute_adaptive(self, initial_state: State, goal: Set[str],
                        actions: List[Action]) -> dict:
        """Execute with continuous monitoring and replanning."""
        
        # Initial plan
        self.current_plan = self._create_initial_plan(initial_state, goal, actions)
        
        current_state = initial_state
        step = 0
        replan_count = 0
        
        while step < len(self.current_plan):
            action = self.current_plan[step]
            
            # Monitor for replan triggers
            trigger = self._check_replan_triggers(current_state, action, goal)
            
            if trigger:
                replan_count += 1
                self.execution_history.append({
                    "step": step,
                    "event": "replan_triggered",
                    "trigger": trigger.value
                })
                
                # Replan from current state
                self.current_plan = self._create_initial_plan(
                    current_state, goal, actions
                )
                
                if not self.current_plan:
                    return {
                        "success": False,
                        "reason": "replanning_failed",
                        "history": self.execution_history
                    }
                
                step = 0  # Restart from new plan
                continue
            
            # Execute action
            if action.is_applicable(current_state):
                new_state = action.apply(current_state)
                
                self.execution_history.append({
                    "step": step,
                    "action": action.name,
                    "state": new_state.facts
                })
                
                current_state = new_state
                step += 1
            else:
                # Action not applicable - must replan
                self.execution_history.append({
                    "step": step,
                    "event": "action_failed",
                    "action": action.name
                })
                
                replan_count += 1
                self.current_plan = self._create_initial_plan(
                    current_state, goal, actions
                )
                step = 0
        
        return {
            "success": current_state.satisfies(goal),
            "steps_taken": len(self.execution_history),
            "replans": replan_count,
            "history": self.execution_history
        }
    
    def _check_replan_triggers(self, state: State, next_action: Action,
                               goal: Set[str]) -> Optional[ReplanTrigger]:
        """Check if we should replan."""
        
        # Check if next action still applicable
        if not next_action.is_applicable(state):
            return ReplanTrigger.PRECONDITION_VIOLATED
        
        # Check if better opportunity exists
        # (simplified - would use more sophisticated analysis)
        
        return None
    
    def _create_initial_plan(self, state: State, goal: Set[str],
                            actions: List[Action]) -> Optional[List[Action]]:
        """Create plan."""
        plan = forward_planning(state, goal, actions)
        return plan.actions if plan else None
```

### Repair-Based Replanning

```python
def repair_plan(original_plan: List[Action],
               failed_step: int,
               current_state: State,
               goal: Set[str],
               actions: List[Action]) -> Optional[List[Action]]:
    """
    Repair plan instead of replanning from scratch.
    
    Tries to keep as much of original plan as possible.
    
    Args:
        original_plan: The plan that failed
        failed_step: Which step failed
        current_state: Current world state
        goal: Goal to achieve
        actions: Available actions
        
    Returns:
        Repaired plan, or None if repair not possible
    """
    # Strategy: Find alternate path to get back to original plan
    
    # Find the state we would have been in
    if failed_step < len(original_plan) - 1:
        target_step = failed_step + 1
        target_action = original_plan[target_step]
        
        # Can we reach the preconditions of the next action?
        target_state = State(facts=target_action.preconditions)
        
        # Plan to reach target state
        bridge = forward_planning(current_state, target_state.facts, actions)
        
        if bridge:
            # Combine: bridge + rest of original plan
            repaired = bridge.actions + original_plan[target_step:]
            return repaired
    
    # Couldn't repair - full replan needed
    return None
```

## Plan Representation

How plans are represented affects reasoning about them.

### Representations

**1. Sequential (List of Actions)**
```python
plan_sequential = ["action1", "action2", "action3"]
```

**2. Partial Order Plan**
```python
from dataclasses import dataclass

@dataclass
class PartialOrderPlan:
    """
    Plan with partial ordering constraints.
    
    Some actions can happen in any order (parallel execution).
    """
    actions: Set[str]
    ordering: List[tuple[str, str]]  # (before, after) pairs
    
    def can_execute_after(self, action: str, completed: Set[str]) -> bool:
        """Check if action's prerequisites are met."""
        for before, after in self.ordering:
            if after == action and before not in completed:
                return False
        return True


# Example: partial order allows parallelization
po_plan = PartialOrderPlan(
    actions={"setup_db", "setup_api", "setup_frontend", "integration"},
    ordering=[
        ("setup_db", "integration"),
        ("setup_api", "integration"),
        ("setup_frontend", "integration")
    ]
)
# setup_db, setup_api, setup_frontend can run in parallel!
```

**3. Hierarchical (Task Network)**
```python
hierarchical_plan = {
    "main_task": "build_app",
    "subtasks": {
        "build_app": ["design", "implement", "test"],
        "implement": ["backend", "frontend"],
        "backend": ["api", "database"],
        "frontend": ["components", "styling"]
    }
}
```

**4. Conditional (Branching Plans)**
```python
@dataclass
class ConditionalPlan:
    """Plan with conditional branches."""
    steps: List[Union[Action, 'ConditionalBranch']]


@dataclass
class ConditionalBranch:
    """Conditional branch in plan."""
    condition: str
    if_true: List[Action]
    if_false: List[Action]


# Example
conditional_plan = ConditionalPlan(steps=[
    Action("check_weather", {}, {"weather_known": True}),
    ConditionalBranch(
        condition="is_raining",
        if_true=[Action("bring_umbrella", {}, {})],
        if_false=[Action("bring_sunglasses", {}, {})]
    ),
    Action("go_outside", {}, {})
])
```

## Plan Execution and Monitoring

Executing plans while monitoring for deviations.

### Execution Monitor

```python
class PlanMonitor:
    """Monitors plan execution for deviations and failures."""
    
    def __init__(self):
        self.execution_log: List[dict] = []
        self.metrics = {
            "actions_succeeded": 0,
            "actions_failed": 0,
            "replans_triggered": 0,
            "total_time": 0.0
        }
    
    def execute_with_monitoring(self, plan: Plan, 
                                get_current_state) -> dict:
        """
        Execute plan with monitoring.
        
        Args:
            plan: Plan to execute
            get_current_state: Function that returns current world state
            
        Returns:
            Execution results with metrics
        """
        import time
        
        start_time = time.time()
        current_state = plan.initial_state
        
        for i, action in enumerate(plan.actions):
            step_start = time.time()
            
            # Get actual state (might differ from expected)
            actual_state = get_current_state()
            
            # Detect deviation
            deviation = self._detect_deviation(current_state, actual_state)
            
            if deviation:
                self.execution_log.append({
                    "step": i,
                    "event": "state_deviation",
                    "expected": current_state.facts,
                    "actual": actual_state.facts,
                    "deviation": deviation
                })
            
            # Try to execute action
            try:
                if action.is_applicable(actual_state):
                    # Execute (in real system, this would interact with environment)
                    new_state = action.apply(actual_state)
                    
                    self.metrics["actions_succeeded"] += 1
                    self.execution_log.append({
                        "step": i,
                        "action": action.name,
                        "duration": time.time() - step_start,
                        "success": True
                    })
                    
                    current_state = new_state
                else:
                    # Action not applicable
                    self.metrics["actions_failed"] += 1
                    self.execution_log.append({
                        "step": i,
                        "action": action.name,
                        "success": False,
                        "reason": "preconditions_not_met"
                    })
                    
                    break  # Stop execution
            
            except Exception as e:
                self.metrics["actions_failed"] += 1
                self.execution_log.append({
                    "step": i,
                    "action": action.name,
                    "success": False,
                    "error": str(e)
                })
                break
        
        self.metrics["total_time"] = time.time() - start_time
        
        return {
            "completed": current_state.satisfies(plan.goal),
            "metrics": self.metrics,
            "log": self.execution_log
        }
    
    def _detect_deviation(self, expected: State, actual: State) -> Optional[dict]:
        """Detect deviation between expected and actual state."""
        expected_facts = expected.facts
        actual_facts = actual.facts
        
        missing = expected_facts - actual_facts
        unexpected = actual_facts - expected_facts
        
        if missing or unexpected:
            return {
                "missing_facts": list(missing),
                "unexpected_facts": list(unexpected)
            }
        
        return None
```

## Handling Plan Failures

Robust agents must handle plan failures gracefully.

### Failure Recovery Strategies

```python
class RobustPlanner:
    """Planner with failure recovery capabilities."""
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
    
    def execute_robust(self, initial_state: State, goal: Set[str],
                      actions: List[Action]) -> dict:
        """
        Execute with automatic failure recovery.
        
        Strategies:
        1. Retry failed action
        2. Try alternative action
        3. Replan from current state
        4. Relax goal if necessary
        """
        attempt = 0
        current_state = initial_state
        
        while attempt < self.max_retries:
            plan = forward_planning(current_state, goal, actions)
            
            if not plan:
                # Try relaxing goal
                relaxed_goal = self._relax_goal(goal)
                plan = forward_planning(current_state, relaxed_goal, actions)
                
                if not plan:
                    return {
                        "success": False,
                        "reason": "no_plan_found",
                        "attempts": attempt + 1
                    }
            
            # Try executing plan
            result = self._try_execute(plan, current_state)
            
            if result["success"]:
                return {
                    "success": True,
                    "attempts": attempt + 1,
                    "final_state": result["final_state"]
                }
            else:
                # Execution failed
                failure_type = result["failure_type"]
                failed_state = result["state_at_failure"]
                
                # Choose recovery strategy
                if failure_type == "action_failed":
                    # Try alternative action
                    current_state = self._try_alternative(
                        failed_state, goal, actions
                    )
                elif failure_type == "state_deviation":
                    # Replan from actual state
                    current_state = failed_state
                else:
                    # Unknown failure
                    break
                
                attempt += 1
        
        return {
            "success": False,
            "reason": "max_retries_exceeded",
            "attempts": attempt
        }
    
    def _relax_goal(self, goal: Set[str]) -> Set[str]:
        """Relax goal by removing least important conditions."""
        # Simplified: remove one random condition
        if len(goal) > 1:
            relaxed = goal.copy()
            relaxed.pop()
            return relaxed
        return goal
    
    def _try_execute(self, plan: Plan, initial_state: State) -> dict:
        """Try to execute plan."""
        # Simplified execution simulation
        current_state = initial_state
        
        for action in plan.actions:
            if not action.is_applicable(current_state):
                return {
                    "success": False,
                    "failure_type": "action_failed",
                    "state_at_failure": current_state
                }
            
            current_state = action.apply(current_state)
        
        return {
            "success": True,
            "final_state": current_state
        }
    
    def _try_alternative(self, state: State, goal: Set[str],
                        actions: List[Action]) -> State:
        """Find alternative action and execute."""
        # Find applicable actions
        applicable = [a for a in actions if a.is_applicable(state)]
        
        if applicable:
            # Try first alternative
            return applicable[0].apply(state)
        
        return state
```

## Summary

Planning strategies enable agents to coordinate complex action sequences to achieve goals. From forward and backward search to hierarchical decomposition and adaptive replanning, effective planning balances deliberation with reactivity.

### Key Takeaways

1. **Planning is search**: Finding action sequences that transform initial state to goal state

2. **Multiple strategies exist**: Forward, backward, hierarchical, reactive vs deliberative

3. **Upfront vs dynamic**: Balance initial planning with adaptation during execution

4. **Hierarchical planning scales**: Operating at multiple abstraction levels handles complexity

5. **Monitoring is essential**: Detect deviations and trigger replanning when needed

6. **Robustness through recovery**: Handle failures with retry, alternatives, and goal relaxation

7. **Representation matters**: How you represent plans affects reasoning and execution

### Best Practices

1. **Choose appropriate strategy**: Match planning approach to problem characteristics
2. **Plan at right level**: Don't over-plan stable environments or under-plan critical tasks
3. **Monitor execution**: Track progress and detect deviations early
4. **Replan when needed**: Have clear triggers for when to abandon and replan
5. **Build in flexibility**: Partial plans and conditional branches handle uncertainty
6. **Consider parallelization**: Identify independent actions that can execute simultaneously
7. **Learn from failures**: Analyze failures to improve future plans

## Next Steps

Explore related planning and execution topics:

- **[Task Decomposition](task-decomposition.md)**: Break complex goals into subtasks
- **[Task Tracking](task-tracking.md)**: Maintain todo lists and track progress
- **[Implementation Plans](implementation-plans.md)**: Create detailed execution specifications
- **[Chain-of-Thought](chain-of-thought.md)**: Explicit reasoning about plans
- **[Tree-of-Thought](tree-of-thought.md)**: Explore multiple planning paths
- **[Reasoning Under Uncertainty](uncertainty.md)**: Plan with incomplete information

Master planning strategies to build agents that reason about complex action sequences! 📋
