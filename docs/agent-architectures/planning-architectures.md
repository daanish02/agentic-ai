# Planning Architectures

## Table of Contents

- [Introduction](#introduction)
- [Why Planning Matters](#why-planning-matters)
- [Plan-and-Execute Pattern](#plan-and-execute-pattern)
- [ReWOO: Reasoning Without Observation](#rewoo-reasoning-without-observation)
- [Hierarchical Planning](#hierarchical-planning)
- [Dynamic Planning](#dynamic-planning)
- [Planning vs Reactive Execution](#planning-vs-reactive-execution)
- [Planning Strategies](#planning-strategies)
- [Plan Representation](#plan-representation)
- [Plan Execution](#plan-execution)
- [Plan Monitoring and Replanning](#plan-monitoring-and-replanning)
- [Resource-Aware Planning](#resource-aware-planning)
- [Collaborative Planning](#collaborative-planning)
- [Common Planning Patterns](#common-planning-patterns)
- [When to Plan vs React](#when-to-plan-vs-react)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Planning architectures separate **thinking** from **doing**. Instead of deciding what to do at each step, planning agents think through the entire approach upfront, then execute the plan systematically.

This mirrors how humans tackle complex tasks: before writing code, we design the architecture; before cooking, we read the recipe; before traveling, we plan the route.

> "Think globally, act locally."

Planning architectures are particularly valuable when:
- The task requires coordination across multiple steps
- Actions have dependencies
- Resources must be allocated efficiently
- The cost of mistakes is high

### The Planning Paradigm

**Reactive approach** (no planning):
```
See → Decide → Act → See → Decide → Act → ...
```

**Planning approach**:
```
See → Plan (think through entire approach) → Execute plan systematically
```

The key difference: **upfront cognitive investment** in exchange for more efficient execution.

## Why Planning Matters

### Benefits of Planning

**1. Efficiency**: Avoid redundant actions

```python
# Without planning (reactive)
search("transformers") → Found 1000 papers → Too many!
search("transformers attention") → Found 500 papers → Still too many!
search("transformers attention 2024") → Found 50 papers → Better!

# With planning
Plan: "I need recent papers on specific topic. 
       Search with: transformers + attention + year filter"
search("transformers attention 2024") → Found 50 papers → Done!
```

**2. Coherence**: Actions work together toward goal

```python
# Without planning
action_1 = "search papers"
action_2 = "analyze trends"  # But haven't read papers yet!
action_3 = "read papers"  # Should have been before analysis

# With planning
plan = [
    "search papers",
    "read papers",      # Logical ordering
    "analyze trends"    # Based on what was read
]
```

**3. Resource optimization**: Minimize costs

```python
# Without planning - expensive!
for paper in papers:  # 100 papers
    read_full_paper(paper)  # 100 API calls
    if is_relevant(paper):
        analyze(paper)

# With planning - efficient!
plan = [
    "filter papers by abstract",     # Cheap
    "read only top 10",              # 10 API calls
    "analyze relevant ones"          # Small subset
]
```

**4. Error prevention**: Catch issues before execution

```python
# Check plan validity before starting
plan = ["read_paper", "search_paper"]  # Wrong order!

validate_plan(plan)
# Error: Cannot read paper before searching for it
# Fix: Reorder to ["search_paper", "read_paper"]
```

### Cost of Planning

Planning isn't free:

1. **Upfront time**: Thinking through approach takes time
2. **Complexity**: Planning systems are more complex
3. **Rigidity**: Plans may not adapt well to unexpected situations
4. **Over-planning**: Can plan in more detail than necessary

The key is knowing **when planning pays off**.

## Plan-and-Execute Pattern

The foundational planning architecture: plan first, then execute.

### Basic Structure

```
┌─────────────────┐
│  Observe Goal   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Create Plan   │ ← Planning phase
│  (think ahead)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Execute Plan   │ ← Execution phase
│ (follow steps)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Return Result  │
└─────────────────┘
```

### Implementation

```python
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class PlanStep:
    """A single step in a plan."""
    action: str
    params: Dict[str, Any]
    description: str
    depends_on: List[int] = None  # Indices of prerequisite steps
    
    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []


class PlanAndExecuteAgent:
    """
    Agent that creates complete plan before execution.
    """
    
    def __init__(self, tools, llm):
        self.tools = tools
        self.llm = llm
    
    def run(self, goal: str) -> str:
        """
        Plan and execute to achieve goal.
        """
        # Phase 1: Planning
        print("=== Planning Phase ===")
        plan = self.create_plan(goal)
        
        print(f"\nCreated plan with {len(plan)} steps:")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step.description}")
        
        # Phase 2: Execution
        print("\n=== Execution Phase ===")
        results = self.execute_plan(plan)
        
        # Phase 3: Synthesis
        print("\n=== Synthesis ===")
        final_result = self.synthesize_results(goal, results)
        
        return final_result
    
    def create_plan(self, goal: str) -> List[PlanStep]:
        """
        Create a complete plan to achieve the goal.
        """
        prompt = f"""Create a detailed plan to achieve this goal:

Goal: {goal}

Available tools:
{self._format_tools()}

Create a plan with these steps:
1. Identify what needs to be done
2. Break into specific actions
3. Order actions logically
4. Specify tool calls with parameters

Format each step as:
Step N: [description]
Action: tool_name(param1=value1, param2=value2)
Depends on: [step numbers this depends on, or "none"]

Create the plan:"""
        
        response = self.llm(prompt)
        plan = self._parse_plan(response)
        
        return plan
    
    def _format_tools(self) -> str:
        """Format available tools."""
        lines = []
        for name, func in self.tools.items():
            doc = func.__doc__ or "No description"
            lines.append(f"- {name}: {doc}")
        return "\n".join(lines)
    
    def _parse_plan(self, response: str) -> List[PlanStep]:
        """
        Parse LLM response into structured plan.
        """
        steps = []
        lines = response.strip().split("\n")
        
        current_step = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("Step"):
                # New step
                parts = line.split(":", 1)
                if len(parts) == 2:
                    description = parts[1].strip()
                    current_step = {"description": description}
            
            elif line.startswith("Action:") and current_step:
                action_str = line[7:].strip()
                action, params = self._parse_action(action_str)
                current_step["action"] = action
                current_step["params"] = params
            
            elif line.startswith("Depends on:") and current_step:
                deps_str = line[11:].strip()
                if deps_str.lower() != "none":
                    # Parse dependency step numbers
                    deps = [int(d.strip()) - 1 for d in deps_str.split(",") if d.strip().isdigit()]
                    current_step["depends_on"] = deps
                else:
                    current_step["depends_on"] = []
                
                # Step complete, add to plan
                steps.append(PlanStep(
                    action=current_step["action"],
                    params=current_step["params"],
                    description=current_step["description"],
                    depends_on=current_step["depends_on"]
                ))
                current_step = None
        
        return steps
    
    def _parse_action(self, action_str: str) -> tuple:
        """Parse action string into tool name and params."""
        import re
        
        match = re.match(r"(\w+)\((.*)\)", action_str)
        if not match:
            return action_str, {}
        
        tool_name = match.group(1)
        params_str = match.group(2)
        
        # Simple parameter parsing
        params = {}
        if params_str:
            for param in params_str.split(","):
                if "=" in param:
                    key, value = param.split("=", 1)
                    params[key.strip()] = value.strip().strip("'\"")
        
        return tool_name, params
    
    def execute_plan(self, plan: List[PlanStep]) -> List[Dict]:
        """
        Execute plan steps in order, respecting dependencies.
        """
        results = []
        
        for i, step in enumerate(plan):
            print(f"\nExecuting step {i + 1}: {step.description}")
            
            # Check dependencies
            if not self._dependencies_met(step, results):
                print(f"  ⚠️  Dependencies not met, skipping")
                results.append({
                    "step": i,
                    "success": False,
                    "error": "Dependencies not met"
                })
                continue
            
            # Execute step
            try:
                result = self._execute_step(step)
                print(f"  ✅ Success: {result}")
                results.append({
                    "step": i,
                    "success": True,
                    "result": result
                })
            
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results.append({
                    "step": i,
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    def _dependencies_met(self, step: PlanStep, results: List[Dict]) -> bool:
        """Check if all dependencies for a step are satisfied."""
        for dep_idx in step.depends_on:
            if dep_idx >= len(results):
                return False  # Dependency not executed yet
            
            if not results[dep_idx]["success"]:
                return False  # Dependency failed
        
        return True
    
    def _execute_step(self, step: PlanStep) -> Any:
        """Execute a single plan step."""
        if step.action not in self.tools:
            raise ValueError(f"Unknown tool: {step.action}")
        
        tool = self.tools[step.action]
        result = tool(**step.params)
        
        return result
    
    def synthesize_results(self, goal: str, results: List[Dict]) -> str:
        """
        Synthesize final answer from execution results.
        """
        prompt = f"""Based on the following plan execution results, 
provide a final answer to the goal.

Goal: {goal}

Execution results:
{self._format_results(results)}

Final answer:"""
        
        return self.llm(prompt)
    
    def _format_results(self, results: List[Dict]) -> str:
        """Format results for prompt."""
        lines = []
        for r in results:
            status = "✅" if r["success"] else "❌"
            content = r.get("result", r.get("error", "N/A"))
            lines.append(f"Step {r['step'] + 1}: {status} {content}")
        return "\n".join(lines)


# Example usage
def example_plan_and_execute():
    """Example of plan-and-execute agent."""
    
    # Define tools
    def search_papers(query: str, max_results: int = 10):
        """Search for academic papers."""
        return [f"Paper on {query}" for i in range(int(max_results))]
    
    def read_paper(paper_id: str):
        """Read a paper."""
        return f"Content of {paper_id}"
    
    def summarize(text: str):
        """Summarize text."""
        return f"Summary of: {text[:50]}..."
    
    tools = {
        "search_papers": search_papers,
        "read_paper": read_paper,
        "summarize": summarize
    }
    
    # Mock LLM
    def mock_llm(prompt: str) -> str:
        if "Create a detailed plan" in prompt:
            return """
Step 1: Search for papers on RAG
Action: search_papers(query='RAG', max_results=5)
Depends on: none

Step 2: Read the first paper
Action: read_paper(paper_id='Paper on RAG 0')
Depends on: 1

Step 3: Summarize the findings
Action: summarize(text='Content of Paper on RAG 0')
Depends on: 2
"""
        else:
            return "Final summary based on results"
    
    # Create and run agent
    agent = PlanAndExecuteAgent(tools=tools, llm=mock_llm)
    result = agent.run("Find and summarize papers on RAG")
    
    print(f"\n{'='*50}")
    print(f"Final Result:\n{result}")
```

### Advantages of Plan-and-Execute

1. **Clear structure**: Explicit phases separate concerns
2. **Predictable**: Know what will happen before it happens
3. **Optimizable**: Can optimize plan before execution
4. **Debuggable**: Can inspect and modify plan
5. **Parallelizable**: Independent steps can run concurrently

### Limitations

1. **Inflexible**: Hard to adapt to unexpected situations
2. **Upfront cost**: Planning takes time even for simple tasks
3. **Validity**: Plans may become invalid during execution
4. **Over-specification**: May plan in unnecessary detail

## ReWOO: Reasoning Without Observation

ReWOO (Reasoning WithOut Observation) is a planning variant that creates the entire plan upfront **without** observing intermediate results.

### The ReWOO Insight

**Traditional planning**:
```
Plan step 1 → Execute → Observe → Plan step 2 → Execute → Observe → ...
```

**ReWOO**:
```
Plan all steps upfront → Execute all → Observe final result
```

### Why ReWOO?

**Efficiency**: Fewer LLM calls

```python
# Traditional: N planning calls for N steps
for i in range(N):
    plan_step = llm.plan_next_step(observations)  # LLM call
    result = execute(plan_step)
    observations.append(result)

# ReWOO: 1 planning call for N steps
full_plan = llm.plan_all_steps(goal)  # 1 LLM call
results = execute_all(full_plan)
```

### ReWOO Implementation

```python
class ReWOOAgent:
    """
    ReWOO: Plan everything upfront, execute without replanning.
    """
    
    def __init__(self, tools, llm):
        self.tools = tools
        self.llm = llm
    
    def run(self, goal: str) -> str:
        """
        Execute ReWOO: plan all, execute all.
        """
        # Single planning phase
        plan = self.create_complete_plan(goal)
        
        # Execute entire plan without replanning
        results = self.execute_without_observation(plan)
        
        # Generate final answer
        answer = self.solve(goal, results)
        
        return answer
    
    def create_complete_plan(self, goal: str) -> List[Dict]:
        """
        Create complete plan with variable references.
        """
        prompt = f"""Create a complete plan to solve this goal.
        
Goal: {goal}

Available tools: {list(self.tools.keys())}

Create a plan where:
1. Each step is numbered #1, #2, etc.
2. Steps can reference outputs of previous steps using #N notation
3. All steps are planned upfront without execution

Example:
#1 = search_papers(query="transformers")
#2 = read_paper(paper_id=#1[0])  # Use first result from #1
#3 = summarize(text=#2)

Create the plan:"""
        
        response = self.llm(prompt)
        plan = self._parse_rewoo_plan(response)
        
        return plan
    
    def _parse_rewoo_plan(self, response: str) -> List[Dict]:
        """Parse ReWOO plan with variable references."""
        plan = []
        
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line or not line.startswith("#"):
                continue
            
            # Parse: #N = action(params)
            parts = line.split("=", 1)
            if len(parts) != 2:
                continue
            
            step_num = parts[0].strip()
            action_str = parts[1].strip()
            
            plan.append({
                "step": step_num,
                "action": action_str
            })
        
        return plan
    
    def execute_without_observation(self, plan: List[Dict]) -> Dict:
        """
        Execute all steps, resolving variable references.
        """
        results = {}
        
        for step in plan:
            step_num = step["step"]
            action_str = step["action"]
            
            # Resolve variable references
            resolved_action = self._resolve_variables(action_str, results)
            
            # Execute
            try:
                result = self._execute_action(resolved_action)
                results[step_num] = result
            except Exception as e:
                results[step_num] = {"error": str(e)}
        
        return results
    
    def _resolve_variables(self, action_str: str, results: Dict) -> str:
        """Resolve #N references to actual values."""
        import re
        
        # Find all #N references
        pattern = r'#(\d+)(?:\[(\d+)\])?'
        
        def replace_var(match):
            step_num = f"#{match.group(1)}"
            index = match.group(2)
            
            if step_num not in results:
                return match.group(0)  # Can't resolve yet
            
            value = results[step_num]
            
            # Handle indexing if present
            if index is not None:
                try:
                    value = value[int(index)]
                except (IndexError, KeyError, TypeError):
                    pass
            
            return str(value)
        
        return re.sub(pattern, replace_var, action_str)
    
    def _execute_action(self, action_str: str):
        """Execute an action string."""
        # Parse and execute (simplified)
        import re
        match = re.match(r"(\w+)\((.*)\)", action_str)
        
        if not match:
            raise ValueError(f"Invalid action: {action_str}")
        
        tool_name = match.group(1)
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        # Execute tool (simplified parameter handling)
        return self.tools[tool_name]()
    
    def solve(self, goal: str, results: Dict) -> str:
        """Generate final answer from results."""
        prompt = f"""Based on these execution results, answer the goal.

Goal: {goal}

Results:
{results}

Answer:"""
        
        return self.llm(prompt)
```

### ReWOO Advantages

1. **Efficient**: Fewer LLM calls
2. **Parallelizable**: All steps can be prepared for parallel execution
3. **Inspectable**: Can see entire plan before execution
4. **Cost-effective**: Less token usage

### ReWOO Limitations

1. **No adaptation**: Cannot adjust based on intermediate results
2. **Error propagation**: Early errors affect later steps
3. **Less robust**: Assumptions may be violated during execution

## Hierarchical Planning

Planning at multiple levels of abstraction.

### The Hierarchy

```
High-level plan:
  1. Gather information
  2. Analyze
  3. Synthesize

Mid-level plan for "Gather information":
  1.1 Search papers
  1.2 Read papers
  1.3 Extract key points

Low-level plan for "Search papers":
  1.1.1 Formulate query
  1.1.2 Execute search
  1.1.3 Filter results
```

### Implementation

```python
class HierarchicalPlanner:
    """
    Planner that operates at multiple levels of abstraction.
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def plan(self, goal: str, depth: int = 0, max_depth: int = 3) -> Dict:
        """
        Create hierarchical plan recursively.
        """
        if depth >= max_depth:
            # Base case: atomic action
            return {
                "type": "action",
                "goal": goal,
                "action": self._goal_to_action(goal)
            }
        
        # Recursive case: decompose into subgoals
        subgoals = self._decompose_goal(goal)
        
        # Plan for each subgoal
        subplans = []
        for subgoal in subgoals:
            subplan = self.plan(subgoal, depth + 1, max_depth)
            subplans.append(subplan)
        
        return {
            "type": "plan",
            "goal": goal,
            "subgoals": subgoals,
            "subplans": subplans
        }
    
    def _decompose_goal(self, goal: str) -> List[str]:
        """Decompose goal into subgoals."""
        prompt = f"""Break down this goal into 3-5 subgoals:

Goal: {goal}

Subgoals:"""
        
        response = self.llm(prompt)
        subgoals = [line.strip() for line in response.strip().split("\n") if line.strip()]
        return subgoals
    
    def _goal_to_action(self, goal: str) -> str:
        """Convert atomic goal to executable action."""
        prompt = f"""Convert this goal to a single executable action:

Goal: {goal}

Action:"""
        
        return self.llm(prompt).strip()
    
    def execute_hierarchical_plan(self, plan: Dict, tools) -> Any:
        """Execute hierarchical plan."""
        if plan["type"] == "action":
            # Execute atomic action
            return self._execute_action(plan["action"], tools)
        
        else:  # plan["type"] == "plan"
            # Execute subplans in order
            results = []
            for subplan in plan["subplans"]:
                result = self.execute_hierarchical_plan(subplan, tools)
                results.append(result)
            
            return results
    
    def _execute_action(self, action: str, tools):
        """Execute a single action."""
        # Simplified execution
        for tool_name, tool_func in tools.items():
            if tool_name in action:
                return tool_func()
        
        return f"Executed: {action}"
```

### Benefits of Hierarchical Planning

1. **Manageable complexity**: Break big problems into smaller ones
2. **Reusable subplans**: Common subgoals can be cached
3. **Flexible abstraction**: Can reason at appropriate level
4. **Progressive refinement**: Refine plan as needed

## Dynamic Planning

Planning that adapts during execution.

### Online Planning

```python
class DynamicPlanner:
    """
    Planner that adapts plan during execution.
    """
    
    def __init__(self, tools, llm):
        self.tools = tools
        self.llm = llm
    
    def run(self, goal: str) -> str:
        """
        Execute with dynamic planning.
        """
        # Initial plan
        plan = self.create_plan(goal, context={})
        
        step_idx = 0
        results = []
        
        while step_idx < len(plan):
            step = plan[step_idx]
            
            # Execute step
            result = self._execute_step(step)
            results.append(result)
            
            # Check if replanning needed
            if self._should_replan(result, plan, step_idx):
                print(f"Replanning from step {step_idx + 1}")
                
                # Create new plan from current state
                context = {
                    "completed_steps": plan[:step_idx + 1],
                    "results": results,
                    "last_result": result
                }
                plan = self.create_plan(goal, context)
                step_idx = 0  # Restart with new plan
            
            else:
                step_idx += 1
        
        return self._synthesize(goal, results)
    
    def _should_replan(self, result, plan, step_idx) -> bool:
        """Determine if replanning is needed."""
        # Replan if:
        # 1. Step failed
        if not result.get("success"):
            return True
        
        # 2. Unexpected result
        if self._result_unexpected(result, plan[step_idx]):
            return True
        
        # 3. Better opportunities emerged
        if self._found_better_path(result, plan, step_idx):
            return True
        
        return False
    
    def _result_unexpected(self, result, step) -> bool:
        """Check if result differs from expectation."""
        # Simplified check
        return False
    
    def _found_better_path(self, result, plan, step_idx) -> bool:
        """Check if better approach is now available."""
        # Simplified check
        return False
```

### Contingency Planning

Planning for potential failures:

```python
class ContingencyPlanner:
    """
    Planner that creates backup plans.
    """
    
    def create_plan_with_contingencies(self, goal: str) -> Dict:
        """
        Create plan with contingency branches.
        """
        primary_plan = self.create_primary_plan(goal)
        
        # Add contingencies for each step
        for step in primary_plan:
            step["on_failure"] = self.create_fallback_plan(step)
            step["alternatives"] = self.create_alternative_approaches(step)
        
        return primary_plan
    
    def execute_with_contingencies(self, plan: List[Dict]) -> List:
        """Execute plan using contingencies as needed."""
        results = []
        
        for step in plan:
            # Try primary approach
            result = self._execute_step(step)
            
            if result["success"]:
                results.append(result)
            
            else:
                # Try fallback
                print(f"Primary approach failed, trying fallback")
                fallback_result = self._execute_step(step["on_failure"])
                
                if fallback_result["success"]:
                    results.append(fallback_result)
                
                else:
                    # Try alternatives
                    for alt in step["alternatives"]:
                        alt_result = self._execute_step(alt)
                        if alt_result["success"]:
                            results.append(alt_result)
                            break
        
        return results
```

## Planning vs Reactive Execution

When to plan vs react dynamically.

### Comparison Matrix

| Dimension | Planning | Reactive |
|-----------|----------|----------|
| **Upfront cost** | High | Low |
| **Execution cost** | Low | High |
| **Adaptability** | Low | High |
| **Efficiency** | High | Low |
| **Predictability** | High | Low |
| **Error handling** | Planned | Adaptive |

### Decision Framework

```python
def choose_approach(task_properties) -> str:
    """
    Decide whether to plan or react based on task properties.
    """
    # Factors favoring planning
    planning_score = 0
    
    if task_properties["complexity"] > 7:  # 0-10 scale
        planning_score += 2
    
    if task_properties["dependencies"]:
        planning_score += 2
    
    if task_properties["resources_constrained"]:
        planning_score += 1
    
    if not task_properties["dynamic_environment"]:
        planning_score += 1
    
    # Factors favoring reactive
    reactive_score = 0
    
    if task_properties["uncertainty_high"]:
        reactive_score += 2
    
    if task_properties["dynamic_environment"]:
        reactive_score += 2
    
    if task_properties["time_critical"]:
        reactive_score += 1
    
    if task_properties["learning_required"]:
        reactive_score += 1
    
    # Decide
    if planning_score > reactive_score + 2:
        return "planning"
    elif reactive_score > planning_score + 2:
        return "reactive"
    else:
        return "hybrid"  # Use both
```

### Hybrid Approach

Combine planning and reactive execution:

```python
class HybridAgent:
    """
    Agent that plans but adapts reactively.
    """
    
    def run(self, goal: str) -> str:
        """
        Execute with hybrid approach.
        """
        # Create high-level plan
        high_level_plan = self.create_high_level_plan(goal)
        
        results = []
        
        # Execute each high-level step
        for step in high_level_plan:
            # Use reactive execution for this step
            step_result = self.reactive_execute(step)
            results.append(step_result)
            
            # Adapt high-level plan if needed
            if self._significant_deviation(step_result, step):
                high_level_plan = self.adapt_plan(
                    high_level_plan,
                    results
                )
        
        return self.synthesize(results)
    
    def create_high_level_plan(self, goal: str) -> List[str]:
        """Create plan at high level of abstraction."""
        # Plan major phases, not detailed steps
        return [
            "information_gathering",
            "analysis",
            "synthesis"
        ]
    
    def reactive_execute(self, phase: str):
        """Execute phase reactively."""
        # Use reactive agent loop for detailed execution
        state = initialize_state(phase)
        
        while not phase_complete(state):
            observation = observe(state)
            decision = decide(observation)
            result = act(decision)
            update_state(state, result)
        
        return state["result"]
```

## Planning Strategies

Different approaches to creating plans.

### Forward Planning

Plan from current state toward goal:

```python
def forward_plan(current_state, goal):
    """
    Plan forward from current state.
    """
    plan = []
    state = current_state
    
    while not satisfies(state, goal):
        # Find action that progresses toward goal
        action = find_progressive_action(state, goal)
        plan.append(action)
        
        # Simulate action effect
        state = simulate(state, action)
    
    return plan
```

### Backward Planning

Plan backward from goal to current state:

```python
def backward_plan(current_state, goal):
    """
    Plan backward from goal.
    """
    plan = []
    state = goal
    
    while not satisfies(current_state, state):
        # Find action that achieves current state
        action = find_achieving_action(state)
        plan.insert(0, action)  # Prepend
        
        # Find precondition state
        state = precondition(state, action)
    
    return plan
```

### Partial-Order Planning

Create plan with partial ordering:

```python
class PartialOrderPlan:
    """
    Plan where some steps can execute in any order.
    """
    
    def __init__(self):
        self.steps = []
        self.ordering_constraints = []  # (before, after) pairs
    
    def add_step(self, step):
        """Add step to plan."""
        self.steps.append(step)
    
    def add_ordering(self, before_idx, after_idx):
        """Specify that one step must come before another."""
        self.ordering_constraints.append((before_idx, after_idx))
    
    def get_valid_orderings(self) -> List[List]:
        """
        Get all valid total orderings of steps.
        """
        # Topological sort considering constraints
        return self._topological_sorts(self.steps, self.ordering_constraints)
    
    def execute_any_valid_order(self, tools):
        """Execute steps in any valid order."""
        valid_order = self.get_valid_orderings()[0]
        
        results = []
        for step_idx in valid_order:
            result = execute_step(self.steps[step_idx], tools)
            results.append(result)
        
        return results
```

## Plan Representation

How to represent plans internally.

### Sequential Plan

```python
@dataclass
class SequentialPlan:
    """Plan as ordered sequence of actions."""
    steps: List[Dict]
    
    def execute(self, tools):
        results = []
        for step in self.steps:
            result = execute(step, tools)
            results.append(result)
        return results
```

### Graph Plan

```python
class GraphPlan:
    """Plan as directed acyclic graph."""
    
    def __init__(self):
        self.nodes = {}  # id -> step
        self.edges = []  # (from_id, to_id) dependencies
    
    def add_step(self, step_id, step):
        """Add step to plan graph."""
        self.nodes[step_id] = step
    
    def add_dependency(self, from_id, to_id):
        """Add dependency edge."""
        self.edges.append((from_id, to_id))
    
    def execute_parallel(self, tools):
        """
        Execute steps in parallel where possible.
        """
        import asyncio
        
        # Find steps with no dependencies (can start immediately)
        ready = self._find_ready_steps()
        completed = set()
        results = {}
        
        while ready or len(completed) < len(self.nodes):
            # Execute all ready steps in parallel
            tasks = [
                self._async_execute(step_id, tools)
                for step_id in ready
            ]
            
            step_results = asyncio.run(asyncio.gather(*tasks))
            
            # Record results
            for step_id, result in zip(ready, step_results):
                results[step_id] = result
                completed.add(step_id)
            
            # Find newly ready steps
            ready = self._find_ready_steps(completed)
        
        return results
    
    def _find_ready_steps(self, completed=None):
        """Find steps whose dependencies are satisfied."""
        if completed is None:
            completed = set()
        
        ready = []
        for step_id in self.nodes:
            if step_id in completed:
                continue
            
            # Check if all dependencies completed
            deps = [from_id for from_id, to_id in self.edges if to_id == step_id]
            if all(dep in completed for dep in deps):
                ready.append(step_id)
        
        return ready
```

### Hierarchical Plan Representation

```python
@dataclass
class HierarchicalPlan:
    """Plan with multiple levels of abstraction."""
    level: int
    goal: str
    steps: List[Union[Dict, 'HierarchicalPlan']]
    
    def flatten(self) -> List[Dict]:
        """Flatten to sequential plan."""
        flat = []
        for step in self.steps:
            if isinstance(step, HierarchicalPlan):
                flat.extend(step.flatten())
            else:
                flat.append(step)
        return flat
    
    def execute_hierarchical(self, tools):
        """Execute maintaining hierarchy."""
        results = []
        for step in self.steps:
            if isinstance(step, HierarchicalPlan):
                result = step.execute_hierarchical(tools)
            else:
                result = execute(step, tools)
            results.append(result)
        return results
```

## Plan Execution

Strategies for executing plans.

### Sequential Execution

```python
def execute_sequentially(plan, tools):
    """Execute steps one at a time."""
    results = []
    
    for i, step in enumerate(plan):
        print(f"Executing step {i + 1}/{len(plan)}")
        result = execute_step(step, tools)
        results.append(result)
        
        if not result["success"]:
            print(f"Step {i + 1} failed, stopping execution")
            break
    
    return results
```

### Parallel Execution

```python
import asyncio

async def execute_parallel(plan, tools):
    """Execute independent steps in parallel."""
    # Group steps by dependencies
    dependency_graph = build_dependency_graph(plan)
    
    # Execute in waves
    results = {}
    completed = set()
    
    while len(completed) < len(plan):
        # Find steps ready to execute
        ready = find_ready_steps(dependency_graph, completed)
        
        # Execute in parallel
        tasks = [
            async_execute_step(step, tools)
            for step in ready
        ]
        
        wave_results = await asyncio.gather(*tasks)
        
        # Record results
        for step, result in zip(ready, wave_results):
            results[step["id"]] = result
            completed.add(step["id"])
    
    return results
```

### Speculative Execution

```python
def execute_speculative(plan, tools):
    """
    Start likely steps before their prerequisites complete.
    """
    results = {}
    futures = {}
    
    for step in plan:
        # Check if prerequisites will likely succeed
        prereqs = get_prerequisites(step, plan)
        likely_success = all(
            estimate_success_probability(p) > 0.8
            for p in prereqs
        )
        
        if likely_success:
            # Start speculatively
            future = start_async(step, tools)
            futures[step["id"]] = future
        else:
            # Wait for prerequisites
            wait_for_prerequisites(prereqs, results)
            result = execute_step(step, tools)
            results[step["id"]] = result
    
    # Collect speculative results
    for step_id, future in futures.items():
        results[step_id] = future.result()
    
    return results
```

## Plan Monitoring and Replanning

Monitoring execution and adapting plans.

### Monitoring Execution

```python
class PlanMonitor:
    """Monitor plan execution for issues."""
    
    def __init__(self, plan):
        self.plan = plan
        self.execution_state = {
            "current_step": 0,
            "completed": [],
            "failed": [],
            "skipped": []
        }
    
    def monitor_step(self, step, result):
        """Monitor a single step execution."""
        if result["success"]:
            self.execution_state["completed"].append(step)
        else:
            self.execution_state["failed"].append((step, result["error"]))
        
        # Check for issues
        issues = self.detect_issues()
        
        return issues
    
    def detect_issues(self) -> List[str]:
        """Detect execution issues."""
        issues = []
        
        # Too many failures
        if len(self.execution_state["failed"]) > 3:
            issues.append("excessive_failures")
        
        # Taking too long
        if self.execution_state["current_step"] > len(self.plan) * 2:
            issues.append("excessive_iterations")
        
        # Critical step failed
        for step, error in self.execution_state["failed"]:
            if step.get("critical"):
                issues.append(f"critical_failure: {step['id']}")
        
        return issues
```

### Replanning

```python
class Replanner:
    """Handle replanning when execution deviates from plan."""
    
    def should_replan(self, plan, execution_state, result) -> bool:
        """Determine if replanning is needed."""
        # Replan if step failed
        if not result["success"]:
            return True
        
        # Replan if result invalidates future steps
        if self._invalidates_future_steps(result, plan, execution_state):
            return True
        
        # Replan if better path found
        if self._better_path_available(result, plan, execution_state):
            return True
        
        return False
    
    def replan(self, goal, current_state, original_plan) -> List[Dict]:
        """Create new plan from current state."""
        # Analyze what's been accomplished
        accomplished = summarize_progress(current_state)
        
        # Identify what remains
        remaining = identify_remaining_work(goal, accomplished)
        
        # Create new plan for remaining work
        new_plan = create_plan(remaining, current_state)
        
        return new_plan
    
    def _invalidates_future_steps(self, result, plan, execution_state) -> bool:
        """Check if result makes future steps invalid."""
        current = execution_state["current_step"]
        future_steps = plan[current + 1:]
        
        for step in future_steps:
            # Check if step's preconditions still valid
            if not check_preconditions(step, result):
                return True
        
        return False
    
    def _better_path_available(self, result, plan, execution_state) -> bool:
        """Check if result enables better approach."""
        # If result provides shortcut
        if enables_shortcut(result, plan):
            return True
        
        # If result suggests more efficient path
        alternative_cost = estimate_alternative_cost(result)
        remaining_cost = estimate_remaining_cost(plan, execution_state)
        
        if alternative_cost < remaining_cost * 0.7:  # 30% better
            return True
        
        return False
```

## Resource-Aware Planning

Planning that considers resource constraints.

### Resource Tracking

```python
@dataclass
class Resources:
    """Track available resources."""
    tokens: int
    api_calls: int
    time_seconds: float
    money_dollars: float


class ResourceAwarePlanner:
    """Planner that considers resource constraints."""
    
    def __init__(self, budget: Resources):
        self.budget = budget
        self.used = Resources(0, 0, 0.0, 0.0)
    
    def create_plan(self, goal: str, tools) -> List[Dict]:
        """Create plan within resource budget."""
        # Estimate costs for different approaches
        approaches = generate_candidate_approaches(goal)
        
        viable_approaches = []
        for approach in approaches:
            cost = estimate_cost(approach, tools)
            
            if self.within_budget(cost):
                viable_approaches.append((approach, cost))
        
        if not viable_approaches:
            raise ValueError("No approach within budget")
        
        # Choose most effective approach within budget
        best_approach = max(
            viable_approaches,
            key=lambda x: estimate_effectiveness(x[0]) / x[1].tokens
        )
        
        return create_detailed_plan(best_approach[0])
    
    def within_budget(self, cost: Resources) -> bool:
        """Check if cost is within remaining budget."""
        remaining = Resources(
            tokens=self.budget.tokens - self.used.tokens,
            api_calls=self.budget.api_calls - self.used.api_calls,
            time_seconds=self.budget.time_seconds - self.used.time_seconds,
            money_dollars=self.budget.money_dollars - self.used.money_dollars
        )
        
        return (
            cost.tokens <= remaining.tokens and
            cost.api_calls <= remaining.api_calls and
            cost.time_seconds <= remaining.time_seconds and
            cost.money_dollars <= remaining.money_dollars
        )
    
    def execute_with_budget(self, plan, tools):
        """Execute plan while tracking resources."""
        results = []
        
        for step in plan:
            # Estimate step cost
            cost = estimate_step_cost(step, tools)
            
            # Check if we can afford it
            if not self.within_budget(cost):
                print(f"Budget exhausted at step {len(results) + 1}")
                break
            
            # Execute
            result = execute_step(step, tools)
            results.append(result)
            
            # Update used resources
            actual_cost = measure_actual_cost(step, result)
            self.used.tokens += actual_cost.tokens
            self.used.api_calls += actual_cost.api_calls
            self.used.time_seconds += actual_cost.time_seconds
            self.used.money_dollars += actual_cost.money_dollars
        
        return results
```

## Collaborative Planning

Multiple agents planning together.

### Multi-Agent Planning

```python
class CollaborativePlanner:
    """Coordinate planning across multiple agents."""
    
    def __init__(self, agents):
        self.agents = agents
    
    def create_collaborative_plan(self, goal: str) -> Dict:
        """Create plan with task allocation."""
        # Decompose goal into subgoals
        subgoals = decompose_goal(goal)
        
        # Assign subgoals to agents
        assignments = self.assign_subgoals(subgoals)
        
        # Each agent plans their part
        agent_plans = {}
        for agent_id, subgoals in assignments.items():
            agent = self.agents[agent_id]
            agent_plans[agent_id] = agent.create_plan(subgoals)
        
        # Coordinate plans
        coordinated_plan = self.coordinate_plans(agent_plans)
        
        return coordinated_plan
    
    def assign_subgoals(self, subgoals) -> Dict[str, List]:
        """Assign subgoals to agents based on capabilities."""
        assignments = {agent_id: [] for agent_id in self.agents}
        
        for subgoal in subgoals:
            # Find best agent for this subgoal
            best_agent = max(
                self.agents.items(),
                key=lambda x: x[1].capability_score(subgoal)
            )[0]
            
            assignments[best_agent].append(subgoal)
        
        return assignments
    
    def coordinate_plans(self, agent_plans: Dict) -> Dict:
        """Coordinate plans to avoid conflicts."""
        # Find dependencies between agent plans
        dependencies = self.find_inter_agent_dependencies(agent_plans)
        
        # Order agent executions
        execution_order = self.topological_sort(agent_plans.keys(), dependencies)
        
        # Create coordinated plan
        coordinated = {
            "execution_order": execution_order,
            "agent_plans": agent_plans,
            "synchronization_points": self.identify_sync_points(agent_plans)
        }
        
        return coordinated
```

## Common Planning Patterns

Frequently-used planning patterns.

### Divide-and-Conquer

```python
def divide_and_conquer_plan(goal):
    """
    Divide problem into independent subproblems.
    """
    plan = {
        "pattern": "divide_and_conquer",
        "phases": [
            {
                "phase": "divide",
                "steps": ["split_problem", "create_subproblems"]
            },
            {
                "phase": "conquer",
                "steps": ["solve_subproblem_1", "solve_subproblem_2", "solve_subproblem_3"],
                "parallel": True  # Can execute in parallel
            },
            {
                "phase": "combine",
                "steps": ["merge_solutions", "verify_combined"]
            }
        ]
    }
    return plan
```

### Pipeline Pattern

```python
def pipeline_plan(goal):
    """
    Chain of transformations.
    """
    plan = {
        "pattern": "pipeline",
        "stages": [
            {"stage": "input", "action": "gather_data"},
            {"stage": "transform_1", "action": "clean_data"},
            {"stage": "transform_2", "action": "analyze_data"},
            {"stage": "transform_3", "action": "visualize_data"},
            {"stage": "output", "action": "present_results"}
        ]
    }
    return plan
```

### Iterative Refinement Pattern

```python
def iterative_refinement_plan(goal, max_iterations=3):
    """
    Repeatedly improve solution.
    """
    plan = {
        "pattern": "iterative_refinement",
        "iterations": []
    }
    
    for i in range(max_iterations):
        plan["iterations"].append({
            "iteration": i + 1,
            "steps": [
                {"action": "generate_solution"},
                {"action": "evaluate_solution"},
                {"action": "identify_improvements"},
                {"action": "refine_solution"}
            ],
            "stop_condition": "quality_threshold_met"
        })
    
    return plan
```

## When to Plan vs React

Decision framework for choosing planning approaches.

### Planning is Better When:

1. **High coordination needed**
   - Many interdependent actions
   - Resource allocation critical
   - Temporal constraints

2. **Efficiency matters**
   - Cost of redundant actions high
   - Limited resources
   - Many steps involved

3. **Predictable environment**
   - Actions have predictable effects
   - Environment stable
   - Few surprises expected

4. **Errors expensive**
   - Wrong actions costly
   - Hard to recover from mistakes
   - Need to verify approach first

### Reactive is Better When:

1. **High uncertainty**
   - Unpredictable outcomes
   - Dynamic environment
   - Learning required

2. **Time-critical**
   - Need immediate action
   - Can't afford planning time
   - Rapid adaptation needed

3. **Simple tasks**
   - Few steps
   - Obvious approach
   - Planning overhead not worth it

## Summary

Planning architectures separate thinking from doing, creating plans upfront before execution:

1. **Plan-and-Execute**: Create complete plan, then execute systematically
2. **ReWOO**: Plan all steps upfront without intermediate observations
3. **Hierarchical Planning**: Plan at multiple levels of abstraction
4. **Dynamic Planning**: Adapt plan during execution based on results
5. **Planning vs Reactive**: Choose based on task properties - coordination, efficiency, predictability
6. **Planning Strategies**: Forward, backward, partial-order approaches
7. **Plan Representation**: Sequential, graph, hierarchical structures
8. **Execution**: Sequential, parallel, speculative strategies
9. **Monitoring**: Track execution and detect issues
10. **Replanning**: Adapt when execution deviates from plan
11. **Resource-Aware**: Consider budgets and constraints
12. **Collaborative**: Multiple agents planning together

Planning pays off for complex, coordinated tasks with predictable dynamics. Reactive execution better for uncertain, dynamic environments.

## Next Steps

- Continue to [Reflection and Self-Critique](reflection.md) to learn how agents evaluate and improve their own performance
- Explore [Autonomous Agent Patterns](autonomous-agents.md) for long-running, open-ended agents
- Study [Advanced Patterns](advanced-patterns.md) for sophisticated architectures
- Review [ReAct](react.md) for combining reasoning with reactive execution
