# The Agent Loop

## Table of Contents

- [Introduction](#introduction)
- [The Fundamental Cycle](#the-fundamental-cycle)
- [Observe: Perception and State](#observe-perception-and-state)
- [Reason: Decision Making](#reason-decision-making)
- [Act: Execution and Effects](#act-execution-and-effects)
- [Loop Iteration and Control](#loop-iteration-and-control)
- [Implementing the Basic Loop](#implementing-the-basic-loop)
- [Loop Variations](#loop-variations)
- [State Management Across Iterations](#state-management-across-iterations)
- [Termination Strategies](#termination-strategies)
- [Performance and Optimization](#performance-and-optimization)
- [Debugging Agent Loops](#debugging-agent-loops)
- [Common Pitfalls](#common-pitfalls)
- [Advanced Loop Patterns](#advanced-loop-patterns)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

The agent loop is the most fundamental architectural pattern in agentic AI. It's the engine that transforms a passive language model into an active, goal-directed agent. Every agent architecture - from simple reactive systems to sophisticated planning agents - is built on this foundation.

At its core, the agent loop embodies a simple idea: **continuous adaptation**. Rather than processing a single input and producing a single output, an agent observes its environment, reasons about what to do, takes action, and repeats this cycle until its goal is achieved.

> "Agency emerges from iteration."

The agent loop is not just a programming pattern - it's a cognitive architecture. It mirrors how intelligent systems (including humans) operate: perceive the world, think about it, act on it, observe the results, and continue.

### Why the Loop Matters

**Without a loop**:

- One-shot interactions only
- No adaptation to results
- Cannot handle multi-step tasks
- No error recovery or retries
- Limited to pre-planned actions

**With a loop**:

- Continuous goal pursuit
- Adapts based on feedback
- Handles complex, multi-step tasks
- Can recover from failures
- Discovers solutions dynamically

### Loop vs Non-Loop Example

**Non-loop (one-shot)**:

```
User: "Find information about transformers and create a summary"
LLM: "Transformers are a type of neural network architecture..."
[END]
```

The model guesses what "find information" means and produces a response from its training data. No actual searching happens.

**With loop**:

```
User: "Find information about transformers and create a summary"
[Loop Iteration 1]
  Observe: User wants information on transformers
  Reason: I should search for recent papers
  Act: search_papers("transformers")

[Loop Iteration 2]
  Observe: Found 50 papers
  Reason: I should read the top 3 most cited
  Act: read_paper(paper_id_1)

[Loop Iteration 3]
  Observe: Read paper 1
  Reason: Still need papers 2 and 3
  Act: read_paper(paper_id_2)

[continues until summary is created]
```

The agent actively searches, reads, and synthesizes information through multiple iterations.

## The Fundamental Cycle

The agent loop consists of three fundamental phases that repeat continuously:

```
     ┌──────────────┐
     │   OBSERVE    │
     │  (Perceive)  │
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │   REASON     │
     │  (Decide)    │
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │     ACT      │
     │  (Execute)   │
     └──────┬───────┘
            │
            └─────► [Repeat until goal achieved]
```

### The Three Phases

**1. Observe (Perception)**

The agent gathers information about its current situation:

- User input or goal
- Results from previous actions
- Current state and context
- Available tools and capabilities
- Environmental constraints

**2. Reason (Decision Making)**

The agent thinks about what to do next:

- Evaluates current state relative to goal
- Considers available actions
- Plans next step
- Predicts outcomes
- Selects optimal action

**3. Act (Execution)**

The agent performs the chosen action:

- Calls a tool or function
- Generates a response
- Updates internal state
- Produces observable effects
- Returns results for next observation

### The Loop Property

What makes this a **loop** rather than a sequence:

1. **Iteration**: The cycle repeats automatically
2. **Feedback**: Each act produces observations for the next cycle
3. **Adaptation**: Reasoning changes based on new observations
4. **Persistence**: Continues until explicit termination condition
5. **Autonomy**: Self-directed iteration without external prompting

## Observe: Perception and State

Observation is how the agent understands its current situation. Good observation is critical - an agent can only be as effective as its perception allows.

### What to Observe

**1. External State** (the world):

```python
observation = {
    "user_goal": "Find papers on RAG and summarize key findings",
    "available_tools": ["search", "read_paper", "summarize"],
    "external_state": {
        "database_status": "online",
        "papers_found": 150,
        "papers_read": 0
    }
}
```

**2. Internal State** (the agent):

```python
internal_state = {
    "iteration_count": 3,
    "actions_taken": ["search", "read_paper", "read_paper"],
    "memory": {
        "papers_to_read": [paper_1, paper_2, paper_3],
        "papers_completed": [paper_1, paper_2]
    },
    "goal_progress": 0.4  # 40% complete
}
```

**3. Action Results**:

```python
last_action_result = {
    "action": "read_paper",
    "success": True,
    "output": {
        "title": "Retrieval Augmented Generation for...",
        "key_findings": [...],
        "citations": 234
    },
    "timestamp": "2026-01-22T10:30:00"
}
```

### Observation Implementation

```python
def observe(state, last_action_result):
    """
    Gather all relevant observations for next reasoning step.
    """
    observation = {
        # Core context
        "goal": state["goal"],
        "progress": assess_progress(state),

        # Action history
        "actions_taken": state["history"],
        "last_action": last_action_result,

        # Current situation
        "available_tools": get_available_tools(),
        "constraints": state["constraints"],

        # Memory/context
        "relevant_memory": retrieve_relevant_context(state),
        "working_memory": state["working_memory"],

        # Meta information
        "iteration": state["iteration_count"],
        "time_elapsed": calculate_time_elapsed(state),
        "tokens_used": state["token_count"]
    }

    return observation
```

### Observation Strategies

**Full observation** (everything):

```python
# Comprehensive but expensive
observation = {
    "entire_conversation_history": [...],  # All previous messages
    "all_tool_results": [...],  # Every action result
    "complete_state": state  # Full state
}
```

**Selective observation** (relevant only):

```python
# Efficient and focused
observation = {
    "current_goal": goal,
    "last_3_actions": recent_actions[-3:],  # Recent context
    "relevant_results": filter_relevant(results),  # Only pertinent info
    "key_state": extract_essential_state(state)  # Just what matters
}
```

**Hierarchical observation** (summarized):

```python
# Compressed representation
observation = {
    "goal": goal,
    "progress_summary": "Completed search, read 2/5 papers",
    "last_result": last_action_result,
    "next_candidates": ["read_paper_3", "synthesize_findings"]
}
```

### Observation Quality

Good observations are:

1. **Complete**: Include all information needed for decision-making
2. **Relevant**: Filter out noise that doesn't aid reasoning
3. **Structured**: Organized for easy consumption by the reasoning module
4. **Timely**: Reflect current state accurately
5. **Efficient**: Don't waste context on redundant information

## Reason: Decision Making

Reasoning is the agent's decision-making process - how it chooses what to do next based on observations.

### Reasoning Inputs

```python
reasoning_input = {
    "observation": current_observation,
    "goal": "Find and summarize papers on RAG",
    "constraints": ["max_papers: 5", "time_limit: 300s"],
    "available_actions": ["search", "read", "summarize", "respond"],
    "reasoning_strategy": "step_by_step"
}
```

### Basic Reasoning Pattern

```python
def reason(observation, goal, available_actions):
    """
    Decide what to do next.
    """
    # 1. Assess current situation
    progress = evaluate_progress(observation, goal)

    # 2. Determine what's needed
    if progress >= 1.0:
        return {"action": "respond", "message": "Task complete"}

    remaining_work = identify_remaining_work(observation, goal)

    # 3. Consider available actions
    candidate_actions = []
    for action in available_actions:
        if can_help_with(action, remaining_work):
            score = estimate_utility(action, remaining_work)
            candidate_actions.append((action, score))

    # 4. Select best action
    best_action = max(candidate_actions, key=lambda x: x[1])

    # 5. Plan parameters
    params = determine_parameters(best_action, observation)

    return {
        "action": best_action[0],
        "params": params,
        "reasoning": f"Need to {remaining_work}, {best_action[0]} will help by..."
    }
```

### Reasoning with Language Models

The most common approach uses LLMs for reasoning:

```python
def reason_with_llm(observation, goal, available_actions):
    """
    Use language model to reason about next action.
    """
    prompt = f"""
You are working toward this goal:
{goal}

Current situation:
{format_observation(observation)}

Available actions:
{format_actions(available_actions)}

Progress so far:
{format_progress(observation)}

What should you do next? Think step by step:
1. What have you accomplished?
2. What remains to be done?
3. Which action will best advance toward the goal?
4. What parameters should you use?

Respond with your reasoning and chosen action.
"""

    response = llm.generate(prompt)
    decision = parse_decision(response)

    return decision
```

### Reasoning Strategies

**1. Greedy (immediate best)**:

```python
# Choose action with highest immediate utility
def greedy_reason(observation, goal):
    actions = get_available_actions()
    scores = [score_action(a, observation) for a in actions]
    return actions[argmax(scores)]
```

**2. Step-by-step decomposition**:

```python
# Break goal into steps, execute sequentially
def step_by_step_reason(observation, goal):
    if not has_plan(observation):
        plan = decompose_goal(goal)
        observation["plan"] = plan

    next_step = plan[observation["current_step"]]
    action = step_to_action(next_step)
    return action
```

**3. Cost-benefit analysis**:

```python
# Weigh costs and benefits
def cost_benefit_reason(observation, goal):
    actions = get_available_actions()

    best_action = None
    best_score = -float('inf')

    for action in actions:
        benefit = estimate_progress_gain(action, goal)
        cost = estimate_cost(action)  # tokens, time, money
        score = benefit - cost

        if score > best_score:
            best_score = score
            best_action = action

    return best_action
```

**4. Reasoning with memory**:

```python
# Learn from past experiences
def memory_based_reason(observation, goal, memory):
    # Find similar past situations
    similar = memory.find_similar(observation)

    # Check what worked before
    successful_actions = [
        s["action"] for s in similar
        if s["outcome"] == "success"
    ]

    # Prefer actions that worked in similar contexts
    if successful_actions:
        return most_common(successful_actions)

    # Fall back to default reasoning
    return default_reason(observation, goal)
```

### Decision Output Format

Structured decision format:

```python
decision = {
    "action": "read_paper",
    "params": {
        "paper_id": "arxiv:2401.12345",
        "sections": ["introduction", "results"]
    },
    "reasoning": "This paper is most cited and relevant to goal",
    "expected_outcome": "Will gain key insights about RAG architectures",
    "confidence": 0.8,
    "alternatives": [
        {"action": "read_paper", "paper_id": "arxiv:2401.54321"}
    ]
}
```

## Act: Execution and Effects

The action phase executes the decision made during reasoning.

### Action Types

**1. Tool calls** (external actions):

```python
# Execute a function/tool
result = tools.search_papers(
    query="Retrieval Augmented Generation",
    year_range=(2024, 2026),
    max_results=20
)
```

**2. State updates** (internal actions):

```python
# Modify internal state
state["memory"]["search_results"] = results
state["working_memory"]["current_papers"] = results[:5]
```

**3. Responses** (communication actions):

```python
# Respond to user
return {
    "type": "response",
    "content": "I found 20 relevant papers. Reading the top 3...",
    "continue": True
}
```

**4. Composite actions** (multiple operations):

```python
# Execute sequence of operations
def composite_action():
    papers = search_papers(query)
    filtered = filter_by_citations(papers, min_citations=50)
    summaries = [summarize_paper(p) for p in filtered[:3]]
    return combine_summaries(summaries)
```

### Action Execution Framework

```python
def act(decision, state, tools):
    """
    Execute the decided action and return results.
    """
    action_name = decision["action"]
    params = decision["params"]

    # 1. Validate action
    if not is_valid_action(action_name, tools):
        return {
            "success": False,
            "error": f"Unknown action: {action_name}"
        }

    # 2. Prepare execution
    action_func = tools[action_name]

    # 3. Execute with error handling
    try:
        result = action_func(**params)

        return {
            "success": True,
            "action": action_name,
            "params": params,
            "result": result,
            "timestamp": time.time()
        }

    except Exception as e:
        return {
            "success": False,
            "action": action_name,
            "params": params,
            "error": str(e),
            "timestamp": time.time()
        }
```

### Action Side Effects

Track what actions change:

```python
def execute_action_with_effects(action, state):
    """
    Execute action and track all side effects.
    """
    # Record pre-action state
    pre_state = snapshot_state(state)

    # Execute
    result = execute(action)

    # Record post-action state
    post_state = snapshot_state(state)

    # Calculate effects
    effects = {
        "state_changes": diff(pre_state, post_state),
        "resources_used": {
            "tokens": count_tokens(action),
            "time": time_elapsed,
            "api_calls": count_api_calls(action)
        },
        "outputs_produced": result,
        "new_information": extract_new_info(result)
    }

    # Update state with effects
    state["history"].append({
        "action": action,
        "result": result,
        "effects": effects
    })

    return result, effects
```

### Retry Logic

Handle failures gracefully:

```python
def act_with_retry(decision, state, max_retries=3):
    """
    Execute action with retry logic.
    """
    for attempt in range(max_retries):
        try:
            result = execute_action(decision, state)

            if result["success"]:
                return result

            # Transient error, retry
            if is_transient_error(result["error"]):
                time.sleep(2 ** attempt)  # Exponential backoff
                continue

            # Permanent error, don't retry
            return result

        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    "success": False,
                    "error": f"Failed after {max_retries} attempts: {e}"
                }

    return {"success": False, "error": "Max retries exceeded"}
```

## Loop Iteration and Control

How the loop repeats and when it continues or stops.

### Basic Loop Structure

```python
def agent_loop(goal, tools, max_iterations=10):
    """
    Basic agent loop implementation.
    """
    # Initialize
    state = initialize_state(goal)
    iteration = 0

    # Loop until done
    while iteration < max_iterations:
        # 1. Observe
        observation = observe(state, last_result if iteration > 0 else None)

        # 2. Reason
        decision = reason(observation, goal, tools)

        # 3. Check for completion
        if decision["action"] == "complete":
            return decision["response"]

        # 4. Act
        result = act(decision, state, tools)

        # 5. Update state
        state = update_state(state, decision, result)
        last_result = result

        # 6. Increment iteration
        iteration += 1

    # Max iterations reached
    return "Task incomplete: maximum iterations exceeded"
```

### Iteration Control Patterns

**1. Count-based**:

```python
for iteration in range(MAX_ITERATIONS):
    observe()
    reason()
    act()
```

**2. Condition-based**:

```python
while not goal_achieved(state):
    observe()
    reason()
    act()
```

**3. Time-based**:

```python
start_time = time.time()
while time.time() - start_time < MAX_TIME:
    observe()
    reason()
    act()
```

**4. Resource-based**:

```python
while tokens_used < TOKEN_BUDGET:
    observe()
    reason()
    act()
    tokens_used += count_tokens(last_action)
```

### Continue vs Stop Decisions

The agent must decide whether to continue iterating:

```python
def should_continue(state, decision, result):
    """
    Determine if loop should continue.
    """
    # Stop if explicit completion
    if decision["action"] == "complete":
        return False

    # Stop if goal achieved
    if assess_progress(state) >= 1.0:
        return False

    # Stop if hit limits
    if state["iteration"] >= MAX_ITERATIONS:
        return False

    if state["tokens_used"] >= TOKEN_BUDGET:
        return False

    # Stop if stuck
    if detect_stuck_loop(state):
        return False

    # Stop if critical error
    if result.get("error") and is_critical_error(result["error"]):
        return False

    # Continue otherwise
    return True
```

## Implementing the Basic Loop

Let's implement a complete, working agent loop from scratch.

### Complete Implementation

```python
import time
from typing import Dict, List, Any, Callable
from dataclasses import dataclass


@dataclass
class ActionResult:
    """Result of an action execution."""
    success: bool
    action: str
    output: Any
    error: str = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class AgentLoop:
    """
    A complete agent loop implementation.
    """

    def __init__(
        self,
        tools: Dict[str, Callable],
        llm: Callable,
        max_iterations: int = 10,
        max_tokens: int = 100000
    ):
        self.tools = tools
        self.llm = llm
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens

    def run(self, goal: str) -> str:
        """
        Execute the agent loop to achieve the goal.
        """
        # Initialize state
        state = self._initialize_state(goal)

        # Main loop
        for iteration in range(self.max_iterations):
            # Observe
            observation = self._observe(state)

            # Reason
            decision = self._reason(observation)

            # Check for completion
            if decision.get("complete"):
                return decision["response"]

            # Act
            result = self._act(decision)

            # Update state
            self._update_state(state, decision, result)

            # Check stopping conditions
            if self._should_stop(state, result):
                break

        # Return final response
        return self._generate_final_response(state)

    def _initialize_state(self, goal: str) -> Dict:
        """Initialize agent state."""
        return {
            "goal": goal,
            "iteration": 0,
            "history": [],
            "working_memory": {},
            "tokens_used": 0,
            "start_time": time.time()
        }

    def _observe(self, state: Dict) -> Dict:
        """Gather observations."""
        observation = {
            "goal": state["goal"],
            "iteration": state["iteration"],
            "history": state["history"][-3:],  # Last 3 actions
            "working_memory": state["working_memory"],
            "available_tools": list(self.tools.keys())
        }
        return observation

    def _reason(self, observation: Dict) -> Dict:
        """Reason about next action using LLM."""
        prompt = self._build_reasoning_prompt(observation)

        response = self.llm(prompt)
        decision = self._parse_decision(response)

        return decision

    def _build_reasoning_prompt(self, observation: Dict) -> str:
        """Build prompt for reasoning step."""
        prompt = f"""
You are working toward this goal:
{observation['goal']}

Iteration: {observation['iteration']}

Available tools:
{self._format_tools()}

Recent actions:
{self._format_history(observation['history'])}

Current working memory:
{observation['working_memory']}

Decide what to do next. You can:
1. Call a tool (specify tool name and parameters)
2. Mark task as complete (if goal is achieved)

Think step by step:
1. What progress have you made?
2. What remains to be done?
3. What's the best next action?

Respond in JSON format:
{{
    "reasoning": "your step-by-step thinking",
    "action": "tool_name or 'complete'",
    "params": {{}},  // parameters for tool
    "complete": false,  // true if task is done
    "response": ""  // final response if complete
}}
"""
        return prompt

    def _format_tools(self) -> str:
        """Format available tools for prompt."""
        tools_desc = []
        for name, func in self.tools.items():
            tools_desc.append(f"- {name}: {func.__doc__ or 'No description'}")
        return "\n".join(tools_desc)

    def _format_history(self, history: List) -> str:
        """Format action history."""
        if not history:
            return "No actions taken yet"

        formatted = []
        for item in history:
            formatted.append(
                f"- {item['action']}: "
                f"{'Success' if item['result'].success else 'Failed'}"
            )
        return "\n".join(formatted)

    def _parse_decision(self, response: str) -> Dict:
        """Parse LLM response into decision."""
        # In real implementation, use robust JSON parsing
        import json
        try:
            decision = json.loads(response)
            return decision
        except:
            # Fallback parsing
            return {
                "reasoning": response,
                "action": "complete",
                "complete": True,
                "response": response
            }

    def _act(self, decision: Dict) -> ActionResult:
        """Execute the decided action."""
        action = decision.get("action")
        params = decision.get("params", {})

        # Handle completion
        if decision.get("complete"):
            return ActionResult(
                success=True,
                action="complete",
                output=decision["response"]
            )

        # Execute tool
        if action not in self.tools:
            return ActionResult(
                success=False,
                action=action,
                output=None,
                error=f"Unknown tool: {action}"
            )

        try:
            tool_func = self.tools[action]
            output = tool_func(**params)

            return ActionResult(
                success=True,
                action=action,
                output=output
            )

        except Exception as e:
            return ActionResult(
                success=False,
                action=action,
                output=None,
                error=str(e)
            )

    def _update_state(
        self,
        state: Dict,
        decision: Dict,
        result: ActionResult
    ):
        """Update state after action."""
        state["iteration"] += 1
        state["history"].append({
            "decision": decision,
            "result": result
        })

        # Update working memory if action produced output
        if result.success and result.output:
            state["working_memory"][result.action] = result.output

    def _should_stop(self, state: Dict, result: ActionResult) -> bool:
        """Determine if loop should stop."""
        # Stop if max iterations
        if state["iteration"] >= self.max_iterations:
            return True

        # Stop if completed
        if result.action == "complete":
            return True

        # Stop if stuck (same action failing repeatedly)
        if self._detect_stuck_loop(state):
            return True

        return False

    def _detect_stuck_loop(self, state: Dict) -> bool:
        """Detect if agent is stuck in a loop."""
        recent = state["history"][-3:]

        if len(recent) < 3:
            return False

        # Check if same action failed 3 times
        actions = [h["result"].action for h in recent]
        failures = [not h["result"].success for h in recent]

        if len(set(actions)) == 1 and all(failures):
            return True  # Same action failing repeatedly

        return False

    def _generate_final_response(self, state: Dict) -> str:
        """Generate final response based on state."""
        if state["iteration"] >= self.max_iterations:
            return "Task incomplete: maximum iterations reached"

        # Get last successful output
        for item in reversed(state["history"]):
            if item["result"].success:
                return str(item["result"].output)

        return "No successful actions completed"


# Example usage
def example_usage():
    """Example of using the agent loop."""

    # Define tools
    def search_papers(query: str, max_results: int = 10):
        """Search for academic papers."""
        # Simulated search
        return [
            {"title": f"Paper on {query}", "id": f"paper_{i}"}
            for i in range(max_results)
        ]

    def read_paper(paper_id: str):
        """Read a paper and extract key information."""
        # Simulated reading
        return {
            "id": paper_id,
            "summary": "This paper discusses...",
            "key_findings": ["Finding 1", "Finding 2"]
        }

    tools = {
        "search_papers": search_papers,
        "read_paper": read_paper
    }

    # Mock LLM (in real implementation, use actual LLM)
    def mock_llm(prompt: str) -> str:
        # Simulated LLM response
        import json
        return json.dumps({
            "reasoning": "I should search for papers first",
            "action": "search_papers",
            "params": {"query": "transformers", "max_results": 5},
            "complete": False
        })

    # Create and run agent
    agent = AgentLoop(tools=tools, llm=mock_llm)
    result = agent.run("Find papers on transformers and summarize them")

    print(result)
```

## Loop Variations

Different loop structures for different needs.

### 1. Linear Loop (Sequential)

```python
def linear_loop(goal, steps):
    """Execute predefined steps in order."""
    for step in steps:
        result = execute(step)
        if not result.success:
            return f"Failed at step: {step}"
    return "All steps completed"
```

**When to use**: Tasks with clear, fixed sequences.

### 2. Reactive Loop (Event-Driven)

```python
def reactive_loop(initial_state):
    """React to events as they occur."""
    state = initial_state

    while True:
        event = wait_for_event()
        response = react_to_event(event, state)
        state = update_state(state, response)

        if event.type == "shutdown":
            break
```

**When to use**: Interactive systems, monitoring agents, chatbots.

### 3. Deliberative Loop (Plan-Execute)

```python
def deliberative_loop(goal):
    """Plan first, then execute."""
    # Planning phase
    plan = create_plan(goal)

    # Execution phase
    for step in plan:
        result = execute_step(step)
        if not result.success:
            # Replan if step fails
            plan = replan(plan, step, result)

    return "Goal achieved"
```

**When to use**: Complex tasks requiring upfront planning.

### 4. Hybrid Loop (Sense-Plan-Act)

```python
def hybrid_loop(goal):
    """Combine reactive and deliberative."""
    plan = create_initial_plan(goal)

    while not goal_achieved():
        # Sense environment
        observation = sense()

        # Replan if environment changed significantly
        if significant_change(observation):
            plan = replan(plan, observation)

        # Execute next step
        next_step = plan[0]
        result = execute(next_step)
        plan = plan[1:]  # Remove completed step

    return "Complete"
```

**When to use**: Dynamic environments requiring both planning and reactivity.

### 5. Parallel Loop (Concurrent Actions)

```python
import asyncio

async def parallel_loop(goals):
    """Execute multiple goals concurrently."""
    tasks = [async_agent_loop(goal) for goal in goals]
    results = await asyncio.gather(*tasks)
    return results

async def async_agent_loop(goal):
    """Async version of agent loop."""
    state = initialize_state(goal)

    while not goal_achieved(state):
        observation = await async_observe(state)
        decision = await async_reason(observation)
        result = await async_act(decision)
        update_state(state, result)

    return state["result"]
```

**When to use**: Multiple independent goals, I/O-bound tasks.

## State Management Across Iterations

How to maintain and evolve state as the loop progresses.

### State Components

```python
state = {
    # Persistent goal
    "goal": "Find and summarize papers",

    # Iteration tracking
    "iteration": 5,
    "start_time": 1705932000.0,

    # Action history
    "history": [
        {"action": "search", "result": {...}},
        {"action": "read", "result": {...}},
        # ...
    ],

    # Working memory (current context)
    "working_memory": {
        "papers_found": [...],
        "papers_read": [...],
        "current_paper": {...}
    },

    # Long-term memory
    "long_term_memory": {
        "learned_patterns": [...],
        "successful_strategies": [...]
    },

    # Resources
    "tokens_used": 45000,
    "api_calls_made": 12,

    # Progress tracking
    "progress": 0.6,
    "remaining_work": ["read_paper_3", "synthesize"]
}
```

### State Update Patterns

**Additive updates** (accumulate):

```python
def update_state_additive(state, result):
    """Add new information without removing old."""
    state["history"].append(result)
    state["tokens_used"] += count_tokens(result)
    state["iteration"] += 1
```

**Replacement updates** (overwrite):

```python
def update_state_replacement(state, result):
    """Replace old information with new."""
    state["current_paper"] = result["paper"]
    state["last_result"] = result
```

**Pruning updates** (manage size):

```python
def update_state_pruning(state, result):
    """Keep state size bounded."""
    # Add new
    state["history"].append(result)

    # Prune old if too large
    if len(state["history"]) > 100:
        # Keep recent + important
        important = filter_important(state["history"])
        recent = state["history"][-20:]
        state["history"] = important + recent
```

### State Persistence

Save state for resumption:

```python
import json
import pickle

def save_state(state, path):
    """Persist state to disk."""
    with open(path, 'w') as f:
        json.dump(state, f, indent=2, default=str)

def load_state(path):
    """Load persisted state."""
    with open(path, 'r') as f:
        return json.load(f)

def resume_loop(state_path):
    """Resume agent loop from saved state."""
    state = load_state(state_path)

    # Continue from where we left off
    while not goal_achieved(state):
        observation = observe(state)
        decision = reason(observation)
        result = act(decision)
        update_state(state, result)

        # Periodically save
        if state["iteration"] % 10 == 0:
            save_state(state, state_path)

    return state["result"]
```

## Termination Strategies

When and how to stop the loop.

### Termination Conditions

**1. Goal achieved**:

```python
def goal_achieved(state):
    """Check if goal is complete."""
    # Explicit completion signal
    if state.get("completed"):
        return True

    # Progress threshold
    if state["progress"] >= 1.0:
        return True

    # All required tasks done
    if not state["remaining_work"]:
        return True

    return False
```

**2. Resource limits**:

```python
def resource_limit_reached(state):
    """Check if resources exhausted."""
    if state["iteration"] >= MAX_ITERATIONS:
        return True

    if state["tokens_used"] >= TOKEN_BUDGET:
        return True

    if time.time() - state["start_time"] >= MAX_TIME:
        return True

    return False
```

**3. Failure conditions**:

```python
def should_abort(state):
    """Check if we should abort due to failures."""
    # Critical error
    if state.get("critical_error"):
        return True

    # Stuck in loop
    if detect_stuck_loop(state):
        return True

    # Repeated failures
    recent_failures = [
        not r["success"]
        for r in state["history"][-5:]
    ]
    if len(recent_failures) >= 5 and all(recent_failures):
        return True

    return False
```

### Graceful Termination

```python
def terminate_gracefully(state, reason):
    """Terminate loop with cleanup and reporting."""
    # Log termination
    log_termination(state, reason)

    # Generate final response
    if reason == "goal_achieved":
        response = generate_success_response(state)
    elif reason == "resource_limit":
        response = generate_partial_response(state)
    else:
        response = generate_failure_response(state, reason)

    # Cleanup
    cleanup_resources(state)

    # Save state for later analysis/resumption
    save_final_state(state)

    return response
```

## Performance and Optimization

Making the loop efficient and cost-effective.

### Performance Metrics

```python
class LoopMetrics:
    """Track loop performance metrics."""

    def __init__(self):
        self.iterations = 0
        self.tokens_used = 0
        self.api_calls = 0
        self.time_elapsed = 0
        self.successful_actions = 0
        self.failed_actions = 0

    def record_iteration(self, decision, result, duration):
        """Record metrics for an iteration."""
        self.iterations += 1
        self.time_elapsed += duration
        self.tokens_used += count_tokens(decision)

        if result.success:
            self.successful_actions += 1
        else:
            self.failed_actions += 1

    def report(self):
        """Generate performance report."""
        return {
            "iterations": self.iterations,
            "tokens_used": self.tokens_used,
            "tokens_per_iteration": self.tokens_used / max(self.iterations, 1),
            "time_elapsed": self.time_elapsed,
            "time_per_iteration": self.time_elapsed / max(self.iterations, 1),
            "success_rate": self.successful_actions / max(self.iterations, 1)
        }
```

### Optimization Techniques

**1. Caching**:

```python
class CachedAgentLoop:
    """Agent loop with caching."""

    def __init__(self):
        self.observation_cache = {}
        self.decision_cache = {}

    def observe_cached(self, state):
        """Cache observations."""
        key = hash_state(state)

        if key not in self.observation_cache:
            self.observation_cache[key] = observe(state)

        return self.observation_cache[key]

    def reason_cached(self, observation):
        """Cache decisions for identical observations."""
        key = hash_observation(observation)

        if key not in self.decision_cache:
            self.decision_cache[key] = reason(observation)

        return self.decision_cache[key]
```

**2. Parallel execution**:

```python
async def optimized_loop(goal):
    """Execute independent actions in parallel."""
    state = initialize_state(goal)

    while not goal_achieved(state):
        observation = observe(state)
        decision = reason(observation)

        # If decision involves multiple independent actions
        if decision["action"] == "parallel":
            results = await asyncio.gather(*[
                async_execute(action)
                for action in decision["actions"]
            ])
        else:
            result = await async_execute(decision)

        update_state(state, result)
```

**3. Early termination**:

```python
def optimized_loop_with_early_stop(goal):
    """Stop as soon as goal is satisfied."""
    state = initialize_state(goal)

    while True:
        # Check goal BEFORE expensive reasoning
        if goal_achieved(state):
            break

        observation = observe(state)

        # Quick check: is goal achievable?
        if not is_achievable(observation, goal):
            break

        decision = reason(observation)
        result = act(decision)

        # Check goal AFTER action
        update_state(state, result)
        if goal_achieved(state):
            break

    return state
```

**4. Adaptive iteration limits**:

```python
def adaptive_loop(goal):
    """Adjust iteration limit based on progress."""
    state = initialize_state(goal)
    base_iterations = 10

    iteration = 0
    max_iterations = base_iterations

    while iteration < max_iterations:
        observation = observe(state)
        decision = reason(observation)
        result = act(decision)
        update_state(state, result)

        # If making good progress, allow more iterations
        if state["progress"] > iteration / max_iterations:
            max_iterations = min(max_iterations + 5, 50)

        iteration += 1

    return state
```

## Debugging Agent Loops

Tools and techniques for debugging loops.

### Logging and Tracing

```python
import logging

class TracedAgentLoop:
    """Agent loop with comprehensive tracing."""

    def __init__(self):
        self.logger = logging.getLogger("agent_loop")
        self.trace = []

    def run_with_trace(self, goal):
        """Run loop with detailed tracing."""
        self.logger.info(f"Starting loop with goal: {goal}")
        state = initialize_state(goal)

        for iteration in range(MAX_ITERATIONS):
            self.logger.info(f"--- Iteration {iteration} ---")

            # Trace observation
            observation = observe(state)
            self.trace.append({"step": "observe", "data": observation})
            self.logger.debug(f"Observation: {observation}")

            # Trace reasoning
            decision = reason(observation)
            self.trace.append({"step": "reason", "data": decision})
            self.logger.debug(f"Decision: {decision}")

            # Trace action
            result = act(decision)
            self.trace.append({"step": "act", "data": result})
            self.logger.info(
                f"Action: {decision['action']}, "
                f"Success: {result.success}"
            )

            update_state(state, result)

            if goal_achieved(state):
                self.logger.info("Goal achieved!")
                break

        return state, self.trace
```

### Visualization

```python
def visualize_loop_execution(trace):
    """
    Visualize agent loop execution.
    """
    print("Agent Loop Execution Trace")
    print("=" * 60)

    for i, step in enumerate(trace):
        if step["step"] == "observe":
            print(f"\n[Iteration {i//3}] OBSERVE:")
            print(f"  Goal: {step['data']['goal']}")
            print(f"  Progress: {step['data'].get('progress', 'N/A')}")

        elif step["step"] == "reason":
            print(f"  REASON:")
            print(f"    Reasoning: {step['data'].get('reasoning', 'N/A')}")
            print(f"    Action: {step['data']['action']}")

        elif step["step"] == "act":
            print(f"  ACT:")
            success = "✓" if step['data'].success else "✗"
            print(f"    {success} {step['data'].action}")
            if step['data'].error:
                print(f"    Error: {step['data'].error}")

    print("\n" + "=" * 60)
```

### Breakpoint Debugging

```python
class DebuggableAgentLoop:
    """Agent loop with breakpoint support."""

    def __init__(self):
        self.breakpoints = set()
        self.step_mode = False

    def add_breakpoint(self, condition):
        """Add a breakpoint condition."""
        self.breakpoints.add(condition)

    def run_debuggable(self, goal):
        """Run with debugging support."""
        state = initialize_state(goal)

        while not goal_achieved(state):
            # Check breakpoints
            if self._check_breakpoints(state):
                self._debug_console(state)

            # Step mode: pause after each iteration
            if self.step_mode:
                input("Press Enter to continue...")

            observation = observe(state)
            decision = reason(observation)
            result = act(decision)
            update_state(state, result)

        return state

    def _check_breakpoints(self, state):
        """Check if any breakpoint condition is met."""
        for condition in self.breakpoints:
            if eval(condition, {"state": state}):
                return True
        return False

    def _debug_console(self, state):
        """Interactive debug console."""
        print("\n=== BREAKPOINT ===")
        print(f"Iteration: {state['iteration']}")
        print(f"Progress: {state.get('progress', 'N/A')}")

        while True:
            cmd = input("debug> ").strip()

            if cmd == "continue":
                break
            elif cmd == "step":
                self.step_mode = True
                break
            elif cmd.startswith("print "):
                var = cmd[6:]
                print(eval(var, {"state": state}))
            elif cmd == "state":
                print(state)
            elif cmd == "help":
                print("Commands: continue, step, print <expr>, state, help")
```

## Common Pitfalls

Mistakes to avoid when implementing agent loops.

### 1. Infinite Loops

**Problem**:

```python
# No termination condition!
while True:
    action = decide()
    execute(action)
```

**Solution**:

```python
MAX_ITERATIONS = 100
iteration = 0

while not goal_achieved() and iteration < MAX_ITERATIONS:
    action = decide()
    execute(action)
    iteration += 1
```

### 2. Context Overflow

**Problem**:

```python
# Keeping entire history in context
observation = {
    "history": all_past_actions,  # Grows unbounded!
    "goal": goal
}
```

**Solution**:

```python
# Keep only recent and relevant history
observation = {
    "recent_history": history[-5:],  # Last 5 actions
    "relevant_history": filter_relevant(history, goal),
    "goal": goal
}
```

### 3. Lost Goal

**Problem**:

```python
# Goal not referenced in reasoning
decision = llm.generate("What should I do next?")
```

**Solution**:

```python
# Always include goal in reasoning
decision = llm.generate(f"""
Goal: {goal}
Progress: {progress}
What should I do next to achieve the goal?
""")
```

### 4. No Error Handling

**Problem**:

```python
# Crash on any error
result = execute_tool(decision)
```

**Solution**:

```python
# Handle errors gracefully
try:
    result = execute_tool(decision)
except Exception as e:
    result = ActionResult(
        success=False,
        error=str(e)
    )
    # Attempt recovery or alternative action
```

### 5. Ignoring Previous Failures

**Problem**:

```python
# Retry same failing action
if not result.success:
    result = execute_tool(decision)  # Try again same way
```

**Solution**:

```python
# Learn from failures
if not result.success:
    # Include failure in context
    observation["previous_failure"] = {
        "action": decision,
        "error": result.error
    }
    # Reason about alternative
    decision = reason_with_failure_context(observation)
```

## Advanced Loop Patterns

Sophisticated loop architectures.

### Hierarchical Loops

```python
def hierarchical_loop(high_level_goal):
    """
    Outer loop for high-level planning,
    inner loops for execution.
    """
    # High-level loop: Planning
    plan = create_plan(high_level_goal)

    for subgoal in plan:
        # Low-level loop: Execution
        result = execution_loop(subgoal)

        if not result.success:
            # Replan at high level
            plan = replan(plan, subgoal, result)

    return "High-level goal achieved"

def execution_loop(subgoal):
    """Execute a single subgoal."""
    state = initialize_state(subgoal)

    while not goal_achieved(state):
        observation = observe(state)
        decision = reason(observation)
        result = act(decision)
        update_state(state, result)

    return result
```

### Meta-Loop (Loop about Loops)

```python
def meta_loop(goal):
    """
    Meta-loop that monitors and adjusts
    the execution loop strategy.
    """
    strategy = "reactive"  # Start with reactive

    while not goal_achieved():
        # Run execution loop with current strategy
        result = execution_loop(goal, strategy)

        # Meta-reasoning: How well did this strategy work?
        performance = evaluate_performance(result)

        # Adjust strategy based on performance
        if performance < THRESHOLD:
            strategy = select_better_strategy(strategy, performance)

        # Continue with potentially new strategy

    return "Goal achieved with optimal strategy"

def select_better_strategy(current, performance):
    """Select better loop strategy based on performance."""
    strategies = ["reactive", "deliberative", "hybrid"]

    if performance < 0.3:  # Reactive not working
        return "deliberative"
    elif performance < 0.6:  # Need balance
        return "hybrid"
    else:
        return current
```

### Ensemble Loops

```python
async def ensemble_loop(goal):
    """
    Run multiple loop strategies in parallel,
    use best result.
    """
    # Start multiple loops with different strategies
    loops = [
        reactive_loop(goal),
        deliberative_loop(goal),
        hybrid_loop(goal)
    ]

    # Run in parallel
    results = await asyncio.gather(*loops)

    # Select best result
    best_result = max(results, key=lambda r: r.quality_score)

    return best_result
```

## Summary

The agent loop is the fundamental architectural pattern that enables autonomous agents:

1. **The Cycle**: Observe → Reason → Act → Repeat until goal achieved
2. **Observe**: Gather current state, action results, and context
3. **Reason**: Decide next action based on goal and observations
4. **Act**: Execute chosen action and produce effects
5. **Iteration**: Continue cycling until termination condition met
6. **State Management**: Maintain context across iterations
7. **Termination**: Stop when goal achieved, resources exhausted, or failure detected
8. **Variations**: Linear, reactive, deliberative, hybrid, parallel loops
9. **Optimization**: Caching, parallelization, early termination, adaptive limits
10. **Debugging**: Logging, tracing, visualization, breakpoint support

The agent loop transforms language models from passive responders into active, goal-directed agents. Master this pattern - it's the foundation for all sophisticated agent architectures.

## Next Steps

- Continue to [ReAct: Reasoning and Acting](react.md) to learn how to interleave explicit reasoning with action execution
- Explore [Planning Architectures](planning-architectures.md) for upfront planning strategies
- Study [Design Principles](design-principles.md) for building robust agent systems
- Review [Simple Agent Loops](../fundamentals/simple-agent-loops.md) for implementation basics
