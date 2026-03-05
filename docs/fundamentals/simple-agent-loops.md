# Simple Agent Loops

## Table of Contents

- [Introduction](#introduction)
- [The Agent Loop Concept](#the-agent-loop-concept)
- [Basic Loop Structure](#basic-loop-structure)
- [Implementing Your First Agent Loop](#implementing-your-first-agent-loop)
- [Observation: Gathering Information](#observation-gathering-information)
- [Reasoning: Deciding What to Do](#reasoning-deciding-what-to-do)
- [Action: Executing Tools](#action-executing-tools)
- [Iteration Control](#iteration-control)
- [Stopping Conditions](#stopping-conditions)
- [State Maintenance](#state-maintenance)
- [Common Loop Patterns](#common-loop-patterns)
- [Loop Failures and Edge Cases](#loop-failures-and-edge-cases)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

The agent loop is the heartbeat of autonomous systems. It's the fundamental pattern that transforms a language model from a one-shot responder into a persistent, goal-directed agent. Understanding the agent loop is essential before building complex architectures.

An agent loop is deceptively simple: **observe, reason, act, repeat**. But within this simple pattern lies the key to agency - the ability to pursue goals over multiple steps, adapt to feedback, and operate autonomously.

> "In the loop, we find autonomy."

This guide walks through building simple agent loops from scratch: the mechanics, the patterns, the gotchas, and the foundations for more sophisticated agent systems.

## The Agent Loop Concept

### From One-Shot to Continuous

**One-shot interaction**:

```
User: "What's 2+2?"
LLM: "4"
[END]
```

**Agent loop**:

```
User: "Find and summarize papers on RAG"
Agent: [Observes task]
Agent: [Reasons: I need to search]
Agent: [Acts: search_papers("RAG")]
Agent: [Observes search results]
Agent: [Reasons: Now I need to read top papers]
Agent: [Acts: read_paper(paper_id)]
Agent: [Observes paper content]
Agent: [Reasons: I can now summarize]
Agent: [Acts: Provides summary]
[END]
```

The difference: **iteration**. The agent doesn't stop after one response - it continues until the goal is achieved.

### The Observe-Reason-Act Cycle

```
    ┌─────────────┐
    │   OBSERVE   │  ← Perception
    │ Environment │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   REASON    │  ← Decision
    │  Decide     │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │    ACT      │  ← Action
    │  Execute    │
    └──────┬──────┘
           │
           └──────┐
                  │
           (repeat)
```

Each cycle:

1. **Observe**: Gather current information (task, tool outputs, state)
2. **Reason**: Decide what to do next
3. **Act**: Execute a tool or provide a response
4. **Repeat** until goal achieved

### Why Loops Matter

**Without loops** (one-shot):

- Can only handle single-step tasks
- No adaptation to results
- No error recovery
- Limited to pre-planned actions

**With loops** (iterative):

- Can tackle multi-step tasks
- Adapts based on observations
- Can retry failed actions
- Discovers solutions dynamically

## Basic Loop Structure

### Pseudocode

```
initialize_agent()
goal = get_user_request()
state = initialize_state()

while not goal_achieved(state):
    # Observe
    observation = gather_observations(state)

    # Reason
    decision = agent_reason(observation, goal, state)

    # Act
    if decision.type == "tool_call":
        result = execute_tool(decision.tool, decision.params)
        state = update_state(state, result)
    elif decision.type == "respond":
        return decision.response

    # Safety check
    if exceeded_max_iterations():
        return "Task incomplete: max iterations reached"
```

### Key Components

1. **Initialization**: Set up agent, goal, initial state
2. **Loop condition**: Continue while goal not achieved
3. **Observation gathering**: Collect current information
4. **Reasoning**: Decide next action
5. **Action execution**: Perform chosen action
6. **State update**: Record what happened
7. **Termination check**: Stop if done or max iterations

## Implementing Your First Agent Loop

### Minimal Python Implementation

```python
from typing import Dict, List, Any
import openai

class SimpleAgent:
    def __init__(self, system_prompt: str, tools: List[Dict], max_iterations: int = 10):
        self.system_prompt = system_prompt
        self.tools = tools
        self.max_iterations = max_iterations
        self.messages = []

    def run(self, user_request: str) -> str:
        """Main agent loop"""
        # Initialize
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_request}
        ]

        iteration = 0

        # Agent loop
        while iteration < self.max_iterations:
            iteration += 1

            # Observe + Reason
            response = self._get_llm_response()

            # Act
            if self._is_tool_call(response):
                tool_result = self._execute_tool(response)
                self.messages.append({
                    "role": "assistant",
                    "content": f"Tool call: {response['tool_name']}"
                })
                self.messages.append({
                    "role": "user",
                    "content": f"Tool result: {tool_result}"
                })
            else:
                # Agent wants to respond to user
                return response["content"]

        return "Max iterations reached without completion"

    def _get_llm_response(self) -> Dict[str, Any]:
        """Get LLM decision (observe + reason)"""
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=self.messages,
            functions=self.tools  # Tool schemas
        )
        return response.choices[0].message

    def _is_tool_call(self, response: Dict) -> bool:
        """Check if response is a tool call"""
        return "function_call" in response

    def _execute_tool(self, response: Dict) -> str:
        """Execute the requested tool"""
        tool_name = response["function_call"]["name"]
        tool_params = json.loads(response["function_call"]["arguments"])

        # Execute actual tool (implementation depends on your tools)
        result = self.tool_executor.execute(tool_name, tool_params)
        return str(result)

# Usage
agent = SimpleAgent(
    system_prompt="You are a helpful assistant with access to tools.",
    tools=[
        {
            "name": "search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }
        }
    ],
    max_iterations=10
)

result = agent.run("What's the weather in Tokyo?")
print(result)
```

### Example Run

```
Iteration 1:
  Observe: User wants weather in Tokyo
  Reason: I need to use the search tool
  Act: search("Tokyo weather")

Iteration 2:
  Observe: Search returned "Tokyo weather: 15°C, sunny"
  Reason: I have the answer
  Act: Respond to user

Output: "The weather in Tokyo is currently 15°C and sunny."
```

## Observation: Gathering Information

Observation is how the agent perceives its environment.

### What to Observe

**1. User Input**

```python
observation = {
    "user_request": "Find papers on transformers",
    "timestamp": "2024-01-15 10:30:00"
}
```

**2. Tool Outputs**

```python
observation = {
    "last_tool": "search_papers",
    "last_result": {
        "papers": [
            {"title": "Attention Is All You Need", "id": "1706.03762"},
            # ...
        ],
        "count": 10
    }
}
```

**3. Current State**

```python
observation = {
    "papers_found": 10,
    "papers_read": 3,
    "goal": "summarize key findings",
    "progress": "60%"
}
```

**4. Context History**

```python
observation = {
    "conversation_history": [
        {"role": "user", "content": "Find papers..."},
        {"role": "assistant", "content": "I'll search..."},
        # ...
    ]
}
```

### Structuring Observations

**Simple approach** (for LLM):

```
Current situation:
- User asked: "Find papers on transformers"
- I searched and found 10 papers
- I have read 3 papers so far
- I need to read 2 more before summarizing

Available information:
- Paper titles and abstracts
- Access to read_paper tool

Goal: Provide summary of key findings
```

**Structured approach** (for logging/debugging):

```python
class Observation:
    def __init__(self):
        self.iteration = 0
        self.user_request = ""
        self.last_action = None
        self.last_result = None
        self.state = {}
        self.available_tools = []

    def to_prompt(self) -> str:
        """Convert observation to prompt text"""
        return f"""
        Iteration: {self.iteration}
        Goal: {self.user_request}
        Last action: {self.last_action}
        Last result: {self.last_result}
        Current state: {self.state}
        """
```

## Reasoning: Deciding What to Do

Reasoning is the agent's decision-making process.

### Reasoning Patterns

**Pattern 1: Status Check**

```
First, assess where I am:
- What have I done so far?
- What information do I have?
- What's still missing?
```

**Pattern 2: Plan Next Step**

```
Based on current state:
- What's the logical next action?
- Which tool should I use?
- What parameters do I need?
```

**Pattern 3: Progress Evaluation**

```
Am I making progress toward the goal?
- If yes: continue current approach
- If no: try different approach
- If stuck: ask for help
```

### Reasoning in Prompts

```python
REASONING_TEMPLATE = """
Current situation:
{observation}

Reflect:
1. What have I accomplished so far?
2. What do I still need to do?
3. What's the next logical step?

Decision:
- Next action: [tool_name or respond]
- Reasoning: [why this action]
- Expected outcome: [what I hope to achieve]

Choose your next action.
"""
```

### Decision Types

**Tool Call Decision**:

```json
{
  "decision_type": "tool_call",
  "tool": "search_papers",
  "parameters": {
    "query": "transformer architecture",
    "year_min": 2017
  },
  "reasoning": "Need to find relevant papers to answer user's question"
}
```

**Response Decision**:

```json
{
  "decision_type": "respond",
  "content": "I found 10 relevant papers. The key finding is...",
  "reasoning": "I have enough information to answer the user's question"
}
```

**Clarification Decision**:

```json
{
  "decision_type": "clarify",
  "question": "Do you want papers from the last 5 years or all years?",
  "reasoning": "The time range affects which papers to include"
}
```

## Action: Executing Tools

Action is how the agent affects its environment.

### Action Execution

```python
def execute_action(decision: Dict) -> Any:
    """Execute the decided action"""

    if decision["type"] == "tool_call":
        tool_name = decision["tool"]
        params = decision["parameters"]

        # Execute tool
        try:
            result = TOOLS[tool_name](**params)
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    elif decision["type"] == "respond":
        # Return response to user
        return {
            "success": True,
            "response": decision["content"],
            "complete": True
        }
```

### Tool Execution Patterns

**Pattern 1: Simple Execution**

```python
def search_papers(query: str, limit: int = 10) -> List[Dict]:
    """Search for papers"""
    # Call external API
    results = arxiv_api.search(query, max_results=limit)
    return [{"title": r.title, "id": r.id} for r in results]

# Agent decides to call this
result = search_papers("transformers", limit=5)
```

**Pattern 2: Validated Execution**

```python
def execute_tool(tool_name: str, params: Dict) -> Dict:
    """Execute tool with validation"""

    # Validate tool exists
    if tool_name not in AVAILABLE_TOOLS:
        return {"error": f"Unknown tool: {tool_name}"}

    # Validate parameters
    if not validate_params(tool_name, params):
        return {"error": f"Invalid parameters for {tool_name}"}

    # Execute
    try:
        result = AVAILABLE_TOOLS[tool_name](**params)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

**Pattern 3: Logged Execution**

```python
def execute_with_logging(tool_name: str, params: Dict) -> Dict:
    """Execute tool with logging"""
    log_info(f"Executing {tool_name} with params: {params}")

    start_time = time.time()
    result = execute_tool(tool_name, params)
    duration = time.time() - start_time

    log_info(f"Tool {tool_name} completed in {duration:.2f}s")
    log_info(f"Result: {result}")

    return result
```

## Iteration Control

Controlling loop iterations prevents infinite loops and manages resources.

### Iteration Counter

```python
class AgentLoop:
    def __init__(self, max_iterations: int = 10):
        self.max_iterations = max_iterations
        self.iteration_count = 0

    def run(self, task: str):
        while self.iteration_count < self.max_iterations:
            self.iteration_count += 1

            # Agent loop logic
            decision = self.reason()
            result = self.act(decision)

            if self.is_complete(result):
                return result

        # Max iterations reached
        return self.handle_timeout()
```

### Adaptive Iteration Limits

```python
def calculate_iteration_limit(task_complexity: str) -> int:
    """Adjust iteration limit based on task"""
    complexity_map = {
        "simple": 5,      # "What's 2+2?"
        "moderate": 10,   # "Search and summarize"
        "complex": 20     # "Analyze, compare, and synthesize"
    }
    return complexity_map.get(task_complexity, 10)
```

### Progress Tracking

```python
class ProgressTracker:
    def __init__(self, goal: str):
        self.goal = goal
        self.steps_completed = []
        self.estimated_steps = None

    def update(self, step: str):
        self.steps_completed.append(step)

    def progress_percentage(self) -> float:
        if not self.estimated_steps:
            return 0.0
        return len(self.steps_completed) / self.estimated_steps

    def is_making_progress(self) -> bool:
        """Check if new steps are being completed"""
        # If last 3 iterations had the same step, not progressing
        if len(self.steps_completed) < 3:
            return True
        return len(set(self.steps_completed[-3:])) > 1
```

## Stopping Conditions

When should the agent loop stop?

### Condition 1: Goal Achieved

```python
def goal_achieved(state: Dict, goal: str) -> bool:
    """Check if goal is met"""

    # Explicit completion marker
    if state.get("status") == "complete":
        return True

    # Agent decided to respond
    if state.get("action_type") == "respond":
        return True

    # Task-specific completion check
    if goal == "find_papers":
        return len(state.get("papers", [])) >= state.get("target_count", 5)

    return False
```

### Condition 2: Max Iterations

```python
if iteration_count >= max_iterations:
    logger.warning(f"Max iterations ({max_iterations}) reached")
    return {
        "status": "timeout",
        "message": "Task incomplete",
        "progress": state
    }
```

### Condition 3: Error Threshold

```python
class ErrorTracker:
    def __init__(self, max_errors: int = 3):
        self.max_errors = max_errors
        self.error_count = 0

    def record_error(self, error: Exception):
        self.error_count += 1
        if self.error_count >= self.max_errors:
            raise TooManyErrorsException(
                f"Exceeded {self.max_errors} errors"
            )
```

### Condition 4: No Progress

```python
def detect_stuck_loop(action_history: List[str], window: int = 3) -> bool:
    """Detect if agent is repeating same action"""
    if len(action_history) < window:
        return False

    recent_actions = action_history[-window:]
    # If all recent actions are the same, agent is stuck
    return len(set(recent_actions)) == 1
```

### Graceful Termination

```python
def terminate_gracefully(state: Dict, reason: str) -> Dict:
    """Terminate with useful information"""
    return {
        "status": "incomplete",
        "reason": reason,
        "progress": {
            "completed_steps": state.get("steps_completed", []),
            "current_step": state.get("current_step"),
            "data_gathered": state.get("data", {})
        },
        "suggestion": "Try breaking task into smaller subtasks"
    }
```

## State Maintenance

Maintaining state across iterations is crucial for coherent agent behavior.

### State Structure

```python
class AgentState:
    def __init__(self):
        self.iteration = 0
        self.goal = ""
        self.completed_actions = []
        self.data = {}
        self.context = []

    def update(self, action: str, result: Any):
        """Update state after an action"""
        self.iteration += 1
        self.completed_actions.append({
            "iteration": self.iteration,
            "action": action,
            "result": result,
            "timestamp": time.time()
        })

    def get_context_window(self, size: int = 5) -> List:
        """Get recent context"""
        return self.completed_actions[-size:]
```

### State Persistence

**In-memory** (simple):

```python
# State stored in agent object
self.state = {
    "papers_found": [],
    "papers_read": [],
    "summaries": {}
}
```

**Message-based** (LLM context):

```python
# State implicitly in conversation
messages = [
    {"role": "system", "content": "You are..."},
    {"role": "user", "content": "Find papers..."},
    {"role": "assistant", "content": "I'll search..."},
    {"role": "user", "content": "Results: [...]"},  # State here
    {"role": "assistant", "content": "Now I'll read..."},
]
```

**Structured** (explicit):

```python
class PersistentState:
    def __init__(self, state_file: str):
        self.state_file = state_file
        self.state = self.load()

    def load(self) -> Dict:
        if os.path.exists(self.state_file):
            with open(self.state_file) as f:
                return json.load(f)
        return {}

    def save(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def update(self, key: str, value: Any):
        self.state[key] = value
        self.save()  # Persist immediately
```

## Common Loop Patterns

### Pattern 1: Linear Sequence

```
Goal: Multi-step task with clear sequence

Loop:
  Step 1: Search for information
  Step 2: Analyze results
  Step 3: Generate output
  Done

Best for: Tasks with known sequence
```

```python
def linear_sequence_loop(steps: List[str]):
    for step in steps:
        result = execute_step(step)
        if not result.success:
            return f"Failed at step: {step}"
    return "All steps completed"
```

### Pattern 2: Iterative Refinement

```
Goal: Improve output through iteration

Loop:
  Generate initial output
  While output quality < threshold:
    Evaluate output
    Identify problems
    Refine output
  Done

Best for: Creative or optimization tasks
```

```python
def iterative_refinement_loop(task: str, quality_threshold: float):
    output = generate_initial(task)
    iteration = 0

    while quality(output) < quality_threshold and iteration < max_iterations:
        problems = evaluate(output)
        output = refine(output, problems)
        iteration += 1

    return output
```

### Pattern 3: Explore-Then-Exploit

```
Goal: Find then use information

Loop:
  Phase 1 (Explore): Gather information
    Search
    Read
    Collect data

  Phase 2 (Exploit): Use information
    Analyze
    Synthesize
    Generate output
  Done

Best for: Research or discovery tasks
```

```python
def explore_exploit_loop(query: str):
    # Exploration phase
    information = []
    while not enough_information(information):
        results = search(query)
        information.extend(results)

    # Exploitation phase
    analysis = analyze(information)
    output = synthesize(analysis)
    return output
```

### Pattern 4: Reactive Loop

```
Goal: Respond to external events

Loop:
  While running:
    Wait for event
    Process event
    Take appropriate action
  Done when: Shutdown signal

Best for: Monitoring, assistants, daemons
```

```python
def reactive_loop():
    while not shutdown_flag:
        event = wait_for_event()
        action = decide_action(event)
        execute(action)
```

## Loop Failures and Edge Cases

### Failure 1: Infinite Loop

**Cause**: No progress toward goal

```python
# Bad: Agent repeats same action
while True:
    action = "search"  # Always searches
    execute(action)     # Never progresses
```

**Fix**: Track action history

```python
action_history = []
while not done:
    action = decide_action()

    # Detect repetition
    action_history.append(action)
    if action_history[-3:] == [action, action, action]:
        raise InfiniteLoopError("Repeating same action")

    execute(action)
```

### Failure 2: Context Overflow

**Cause**: Too much information in context

```python
# Bad: Keep adding to context
messages.append({"role": "user", "content": huge_result})
# Context grows unbounded
```

**Fix**: Summarize or truncate

```python
if total_tokens(messages) > context_limit:
    # Summarize older messages
    summary = summarize(messages[1:-3])  # Keep system and recent
    messages = [
        messages[0],  # System prompt
        {"role": "system", "content": f"Summary: {summary}"},
        *messages[-3:]  # Recent messages
    ]
```

### Failure 3: Action Failure Cascade

**Cause**: One failure breaks everything

```python
# Bad: No error handling
result1 = tool1()  # Might fail
result2 = tool2(result1)  # Will fail if result1 failed
result3 = tool3(result2)  # Will fail if result2 failed
```

**Fix**: Graceful degradation

```python
result1 = tool1()
if not result1.success:
    return fallback_approach()

result2 = tool2(result1)
if not result2.success:
    # Try alternative
    result2 = alternative_tool2(result1)

result3 = tool3(result2)
```

### Failure 4: Lost Goal

**Cause**: Agent forgets original goal

```python
# Bad: Goal not maintained
while True:
    # Agent makes decisions without reference to goal
    action = decide()  # What was I trying to do?
    execute(action)
```

**Fix**: Goal-aware reasoning

```python
goal = "Find and summarize papers on transformers"

while not achieved(goal):
    # Always reference goal in reasoning
    prompt = f"""
    Goal: {goal}
    Progress: {current_progress}
    Next step to achieve goal?
    """
    action = decide(prompt)
    execute(action)
```

## Summary

Simple agent loops are the foundation of autonomous agents:

1. **The Loop**: Observe → Reason → Act → Repeat
2. **Observation**: Gather current state, tool results, context
3. **Reasoning**: Decide next action based on goal and observations
4. **Action**: Execute tools or respond to user
5. **Iteration Control**: Limit iterations, track progress
6. **Stopping Conditions**: Goal achieved, max iterations, errors, stuck detection
7. **State Management**: Maintain context and progress across iterations
8. **Common Patterns**: Linear sequence, iterative refinement, explore-exploit, reactive
9. **Failure Modes**: Infinite loops, context overflow, cascading failures, lost goals

The agent loop is simple in concept but requires careful implementation. Master this pattern - it's the foundation for all more complex agent architectures.

## Next Steps

Continue to [Basic Tool Calling](basic-tool-calling.md) to learn how agents extend their capabilities by invoking external functions and integrating with APIs.
