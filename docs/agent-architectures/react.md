# ReAct: Reasoning and Acting

## Table of Contents

- [Introduction](#introduction)
- [The ReAct Pattern](#the-react-pattern)
- [Thought-Action Pairs](#thought-action-pairs)
- [Implementing ReAct](#implementing-react)
- [ReAct vs Basic Loops](#react-vs-basic-loops)
- [Reasoning Strategies](#reasoning-strategies)
- [Multi-Step Reasoning](#multi-step-reasoning)
- [Error Handling in ReAct](#error-handling-in-react)
- [ReAct with Different Tools](#react-with-different-tools)
- [Optimizing ReAct](#optimizing-react)
- [Common Patterns](#common-patterns)
- [Debugging ReAct Agents](#debugging-react-agents)
- [When to Use ReAct](#when-to-use-react)
- [Advanced ReAct Variants](#advanced-react-variants)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

ReAct (Reasoning and Acting) is an agent architecture pattern that explicitly interleaves reasoning traces with action execution. Instead of jumping directly from observation to action, ReAct agents **think out loud** about what to do before doing it.

The key insight: making reasoning explicit improves decision quality, enables better error recovery, and makes agent behavior more interpretable.

> "Think before you act - and show your thinking."

ReAct was introduced in the paper "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2023) and has become one of the most widely-used agent patterns due to its simplicity and effectiveness.

### The Core Idea

**Traditional agent loop**:

```
Observe → Decide → Act
```

**ReAct pattern**:

```
Observe → Think → Act
         (explicit reasoning)
```

The "Think" step makes reasoning **visible and verifiable**. The agent doesn't just decide what to do - it explains why.

### Why ReAct Matters

**Benefits**:

1. **Better decisions**: Explicit reasoning reduces impulsive errors
2. **Interpretability**: We can see why the agent chose an action
3. **Debugging**: Reasoning traces reveal faulty logic
4. **Error recovery**: Agent can recognize and correct mistakes
5. **Learning**: Reasoning patterns can be analyzed and improved

**Example comparison**:

Without ReAct:

```
Action: search("transformers")
```

With ReAct:

```
Thought: I need to find information about transformers.
         First, I should search for academic papers.
Action: search("transformers attention mechanism papers")
```

The second approach is more focused and more likely to succeed.

## The ReAct Pattern

### Basic Structure

ReAct alternates between **Thought** and **Action**:

```
┌─────────────┐
│  Observe    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Thought    │  "I should..."
└──────┬──────┘  "This will help by..."
       │          "Expected outcome..."
       ▼
┌─────────────┐
│  Action     │  execute(tool, params)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Observe    │  (action result)
└──────┬──────┘
       │
       └──────► [Repeat]
```

### Trace Format

A ReAct trace consists of interleaved thoughts and actions:

```
Goal: Find and summarize recent papers on RAG

Thought 1: I need to search for papers on Retrieval Augmented Generation.
           I'll search for papers from 2024-2026 to get recent work.
Action 1: search_papers(query="Retrieval Augmented Generation",
                       year_range=(2024, 2026), max_results=20)
Observation 1: Found 20 papers. Top paper: "Advanced RAG Techniques"
               (2025, 156 citations)

Thought 2: Good, I found relevant papers. Now I should read the most
           cited ones to understand key contributions. I'll start with
           the top 3.
Action 2: read_paper(paper_id="arxiv:2501.12345")
Observation 2: Paper discusses improved retrieval methods using dense
               vectors...

Thought 3: This paper focuses on retrieval. I should also look at
           papers about generation quality to get a complete picture.
Action 3: read_paper(paper_id="arxiv:2501.54321")
Observation 3: This paper covers generation improvements...

Thought 4: I've read papers on retrieval and generation. Now I have
           enough information to create a comprehensive summary.
Action 4: FINISH[Summary of RAG advances: 1) Improved retrieval via
          dense embeddings, 2) Better generation through prompt
          engineering...]
```

### Key Components

**1. Thought**: Reasoning trace explaining the next action

```python
thought = {
    "situation_assessment": "I have searched but not read papers yet",
    "goal_progress": "Need to read papers to create summary",
    "next_action_rationale": "Reading top papers will give key insights",
    "expected_outcome": "Will understand main contributions"
}
```

**2. Action**: Executable operation

```python
action = {
    "tool": "read_paper",
    "params": {"paper_id": "arxiv:2501.12345"},
    "justification": "Most cited, likely most influential"
}
```

**3. Observation**: Result of action

```python
observation = {
    "success": True,
    "output": {...},
    "new_information": "Paper introduces novel retrieval method"
}
```

## Thought-Action Pairs

The fundamental unit of ReAct is the **Thought-Action pair**.

### Anatomy of a Thought

A good thought includes:

1. **Situation assessment**: What's the current state?
2. **Goal reference**: What are we trying to achieve?
3. **Action rationale**: Why this action?
4. **Expected outcome**: What will this accomplish?

```python
def generate_thought(observation, goal, history):
    """Generate a reasoning trace."""
    thought = f"""
Current situation: {assess_situation(observation)}
Goal: {goal}
Progress so far: {summarize_progress(history)}

Reasoning:
- What I know: {summarize_knowledge(observation)}
- What I need: {identify_gaps(observation, goal)}
- Best next action: {propose_action(observation, goal)}
- Why: {justify_action(observation, goal)}
- Expected outcome: {predict_outcome()}

Therefore, I will: {describe_action()}
"""
    return thought
```

### Anatomy of an Action

An action specifies:

1. **Tool/function**: What to execute
2. **Parameters**: How to execute it
3. **Expectation**: What should happen

```python
action = {
    "tool": "search_papers",
    "params": {
        "query": "Retrieval Augmented Generation",
        "year_range": (2024, 2026),
        "max_results": 20
    },
    "expectation": "Will return 10-20 recent papers on RAG"
}
```

### Thought Quality

**Poor thought** (not useful):

```
Thought: I should do something.
Action: search("stuff")
```

**Good thought** (informative):

```
Thought: I need to find papers on RAG. The user wants recent work,
         so I'll limit to 2024-2026. I'll search with the full term
         "Retrieval Augmented Generation" to avoid ambiguity with
         other uses of "RAG".
Action: search("Retrieval Augmented Generation", years=2024-2026)
```

The good thought:

- References the goal
- Explains constraints (recent work)
- Justifies parameter choices
- Shows understanding of context

## Implementing ReAct

### Basic ReAct Loop

```python
def react_loop(goal, tools, max_iterations=10):
    """
    Basic ReAct implementation.
    """
    state = {
        "goal": goal,
        "history": [],
        "iteration": 0
    }

    while state["iteration"] < max_iterations:
        # 1. Observe current situation
        observation = create_observation(state)

        # 2. Generate thought (reasoning)
        thought = generate_thought(observation, goal)

        # 3. Decide action based on thought
        action = extract_action_from_thought(thought)

        # 4. Check for completion
        if action["type"] == "FINISH":
            return action["content"]

        # 5. Execute action
        result = execute_action(action, tools)

        # 6. Record in history
        state["history"].append({
            "thought": thought,
            "action": action,
            "observation": result
        })

        state["iteration"] += 1

    return "Max iterations reached"
```

### Complete Implementation

```python
from typing import Dict, List, Any, Callable
from dataclasses import dataclass


@dataclass
class ReActStep:
    """A single ReAct step: thought + action + observation."""
    thought: str
    action: Dict[str, Any]
    observation: Dict[str, Any]
    iteration: int


class ReActAgent:
    """
    A complete ReAct agent implementation.
    """

    def __init__(
        self,
        tools: Dict[str, Callable],
        llm: Callable,
        max_iterations: int = 15,
        verbose: bool = False
    ):
        self.tools = tools
        self.llm = llm
        self.max_iterations = max_iterations
        self.verbose = verbose

    def run(self, goal: str) -> str:
        """
        Execute ReAct loop to achieve goal.
        """
        history = []

        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n=== Iteration {iteration + 1} ===")

            # Generate thought + action
            thought, action = self._think_and_act(goal, history)

            if self.verbose:
                print(f"Thought: {thought}")
                print(f"Action: {action}")

            # Check for completion
            if action["type"] == "FINISH":
                return action["content"]

            # Execute action
            observation = self._execute(action)

            if self.verbose:
                print(f"Observation: {observation}")

            # Record step
            step = ReActStep(
                thought=thought,
                action=action,
                observation=observation,
                iteration=iteration
            )
            history.append(step)

        # Max iterations reached
        return self._generate_final_answer(history)

    def _think_and_act(
        self,
        goal: str,
        history: List[ReActStep]
    ) -> tuple[str, Dict]:
        """
        Generate thought and action using LLM.
        """
        prompt = self._build_prompt(goal, history)
        response = self.llm(prompt)
        thought, action = self._parse_response(response)
        return thought, action

    def _build_prompt(
        self,
        goal: str,
        history: List[ReActStep]
    ) -> str:
        """
        Build ReAct prompt with goal and history.
        """
        prompt = f"""You are an agent using the ReAct pattern (Reasoning and Acting).

Goal: {goal}

Available tools:
{self._format_tools()}

Instructions:
1. Think step-by-step about what to do next
2. Choose an action that advances toward the goal
3. Explain your reasoning clearly

Format your response as:
Thought: <your reasoning>
Action: <tool_name>(<params>)

Or, if goal is achieved:
Thought: <final reasoning>
Action: FINISH[<final answer>]

"""

        # Add history
        if history:
            prompt += "Previous steps:\n"
            for i, step in enumerate(history[-5:], 1):  # Last 5 steps
                prompt += f"\nStep {step.iteration + 1}:\n"
                prompt += f"Thought: {step.thought}\n"
                prompt += f"Action: {step.action}\n"
                prompt += f"Observation: {step.observation}\n"

        prompt += "\nWhat is your next thought and action?"

        return prompt

    def _format_tools(self) -> str:
        """Format tool descriptions."""
        descriptions = []
        for name, func in self.tools.items():
            doc = func.__doc__ or "No description"
            descriptions.append(f"- {name}: {doc}")
        return "\n".join(descriptions)

    def _parse_response(self, response: str) -> tuple[str, Dict]:
        """
        Parse LLM response into thought and action.
        """
        lines = response.strip().split("\n")

        thought = ""
        action_str = ""

        for line in lines:
            if line.startswith("Thought:"):
                thought = line[8:].strip()
            elif line.startswith("Action:"):
                action_str = line[7:].strip()

        # Parse action
        if action_str.startswith("FINISH"):
            # Extract final answer
            content = action_str[7:-1]  # Remove "FINISH[" and "]"
            action = {"type": "FINISH", "content": content}
        else:
            # Parse tool call
            action = self._parse_tool_call(action_str)

        return thought, action

    def _parse_tool_call(self, action_str: str) -> Dict:
        """
        Parse tool call string into structured action.
        Example: "search_papers(query='RAG', max=10)"
        """
        import re

        # Extract tool name and params
        match = re.match(r"(\w+)\((.*)\)", action_str)
        if not match:
            return {"type": "ERROR", "message": f"Invalid action: {action_str}"}

        tool_name = match.group(1)
        params_str = match.group(2)

        # Simple parameter parsing (in production, use proper parser)
        params = {}
        if params_str:
            for param in params_str.split(","):
                if "=" in param:
                    key, value = param.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    params[key] = value

        return {
            "type": "TOOL_CALL",
            "tool": tool_name,
            "params": params
        }

    def _execute(self, action: Dict) -> Dict:
        """
        Execute action and return observation.
        """
        if action["type"] == "ERROR":
            return {"error": action["message"]}

        if action["type"] != "TOOL_CALL":
            return {"error": "Unknown action type"}

        tool_name = action["tool"]
        params = action["params"]

        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            result = self.tools[tool_name](**params)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _generate_final_answer(self, history: List[ReActStep]) -> str:
        """
        Generate final answer from history.
        """
        prompt = f"""Based on the following ReAct history,
generate a final answer:

{self._format_history(history)}

Final answer:"""

        return self.llm(prompt)

    def _format_history(self, history: List[ReActStep]) -> str:
        """Format history for display."""
        formatted = []
        for step in history:
            formatted.append(f"Thought: {step.thought}")
            formatted.append(f"Action: {step.action}")
            formatted.append(f"Observation: {step.observation}")
            formatted.append("")
        return "\n".join(formatted)


# Example usage
def example():
    """Example of using ReActAgent."""

    # Define tools
    def search_papers(query: str, max_results: int = 10):
        """Search for academic papers."""
        # Simulated search
        return [
            {"title": f"Paper on {query} #{i}", "id": f"paper_{i}"}
            for i in range(int(max_results))
        ]

    def read_paper(paper_id: str):
        """Read a paper."""
        return {
            "id": paper_id,
            "content": "This paper discusses...",
            "key_findings": ["Finding 1", "Finding 2"]
        }

    tools = {
        "search_papers": search_papers,
        "read_paper": read_paper
    }

    # Mock LLM
    def mock_llm(prompt: str) -> str:
        # In production, call real LLM
        return """Thought: I need to search for papers first
Action: search_papers(query='RAG', max_results=5)"""

    # Create and run agent
    agent = ReActAgent(tools=tools, llm=mock_llm, verbose=True)
    result = agent.run("Find and summarize papers on RAG")

    print(f"\nFinal result: {result}")
```

## ReAct vs Basic Loops

### Comparison

**Basic Loop**:

```python
while not done:
    observation = observe()
    action = decide(observation)  # Black box
    result = execute(action)
```

**ReAct Loop**:

```python
while not done:
    observation = observe()
    thought = reason(observation)  # Explicit
    action = extract_action(thought)
    result = execute(action)
```

### When Basic Loop is Sufficient

- Simple, deterministic tasks
- Clear action sequences
- Low stakes (errors not critical)
- Speed is paramount

### When ReAct is Better

- Complex reasoning required
- Multiple viable action paths
- High stakes (errors costly)
- Interpretability important
- Debugging needed
- Learning from traces

### Performance Trade-offs

**Basic Loop**:

- Faster (fewer LLM calls)
- Cheaper (less token usage)
- Less interpretable

**ReAct**:

- Slower (explicit reasoning)
- More expensive (reasoning traces)
- More interpretable
- Better decisions

## Reasoning Strategies

Different approaches to generating thoughts in ReAct.

### 1. Step-by-Step Reasoning

```python
def step_by_step_thought(observation, goal):
    """
    Break down reasoning into clear steps.
    """
    thought = f"""
Let me think step by step:

1. Current situation: {describe_situation(observation)}
2. Goal: {goal}
3. What I've accomplished: {list_accomplishments(observation)}
4. What remains: {identify_remaining_work(goal, observation)}
5. Next logical step: {determine_next_step()}
6. Why this step: {justify_step()}

Therefore, I will: {describe_action()}
"""
    return thought
```

### 2. Hypothesis-Driven Reasoning

```python
def hypothesis_thought(observation, goal):
    """
    Form hypothesis, test with action.
    """
    thought = f"""
Hypothesis: {form_hypothesis(goal, observation)}

Evidence so far: {summarize_evidence(observation)}

To test this hypothesis, I should: {design_test()}

Expected outcome if hypothesis correct: {predict_outcome()}

Action: {propose_action()}
"""
    return thought
```

### 3. Cost-Benefit Reasoning

```python
def cost_benefit_thought(observation, goal):
    """
    Weigh costs and benefits of actions.
    """
    actions = generate_candidate_actions(observation, goal)

    thought = "Considering possible actions:\n"

    for action in actions:
        benefit = estimate_benefit(action, goal)
        cost = estimate_cost(action)
        score = benefit - cost

        thought += f"\n{action}:"
        thought += f"\n  Benefit: {benefit}"
        thought += f"\n  Cost: {cost}"
        thought += f"\n  Score: {score}"

    best_action = max(actions, key=lambda a: estimate_benefit(a, goal) - estimate_cost(a))

    thought += f"\n\nBest action: {best_action}"
    thought += f"\nReason: {justify_choice(best_action, actions)}"

    return thought
```

### 4. Analogical Reasoning

```python
def analogical_thought(observation, goal, memory):
    """
    Reason by analogy to past experiences.
    """
    similar_cases = memory.find_similar(observation)

    thought = f"""
This situation reminds me of: {describe_similar(similar_cases[0])}

In that case, I: {similar_cases[0]['action']}
Result: {similar_cases[0]['outcome']}

Key similarity: {identify_similarity(observation, similar_cases[0])}
Key difference: {identify_difference(observation, similar_cases[0])}

Adapting the approach: {adapt_solution(similar_cases[0], observation)}

Therefore: {propose_adapted_action()}
"""
    return thought
```

### 5. Self-Questioning

```python
def self_questioning_thought(observation, goal):
    """
    Generate and answer questions.
    """
    thought = f"""
Let me ask myself some questions:

Q: What is the goal?
A: {goal}

Q: What do I know so far?
A: {summarize_knowledge(observation)}

Q: What don't I know yet?
A: {identify_unknowns(observation, goal)}

Q: How can I find out?
A: {propose_method()}

Q: What's the risk?
A: {assess_risk()}

Q: Is this the best approach?
A: {evaluate_approach()}

Decision: {make_decision()}
"""
    return thought
```

## Multi-Step Reasoning

Handling tasks that require multiple reasoning steps.

### Sequential Reasoning

```python
def sequential_react(goal, tools):
    """
    ReAct with sequential sub-goals.
    """
    # Decompose into sub-goals
    sub_goals = decompose_goal(goal)

    results = []

    for sub_goal in sub_goals:
        thought = f"""
Main goal: {goal}
Current sub-goal: {sub_goal}
Previous results: {results}

To achieve this sub-goal, I will: {plan_sub_goal(sub_goal)}
"""

        action = extract_action(thought)
        result = execute(action)

        results.append({
            "sub_goal": sub_goal,
            "thought": thought,
            "action": action,
            "result": result
        })

    # Synthesize results
    final_thought = f"""
I've completed all sub-goals:
{format_results(results)}

Final answer: {synthesize(results)}
"""

    return final_thought
```

### Nested Reasoning

```python
def nested_react_thought(observation, goal, depth=0):
    """
    Recursive reasoning for complex problems.
    """
    indent = "  " * depth

    thought = f"{indent}Considering: {goal}\n"

    # Base case: simple enough to act directly
    if is_simple(goal):
        thought += f"{indent}This is straightforward: {propose_simple_action(goal)}\n"
        return thought

    # Recursive case: break down further
    sub_goals = decompose(goal)
    thought += f"{indent}Breaking into sub-goals:\n"

    for sub_goal in sub_goals:
        thought += nested_react_thought(observation, sub_goal, depth + 1)

    thought += f"{indent}After solving sub-goals, I can: {synthesize_solution()}\n"

    return thought
```

### Iterative Refinement

```python
def iterative_react(goal, tools, max_refinements=3):
    """
    Iteratively refine solution.
    """
    solution = None

    for iteration in range(max_refinements):
        thought = f"""
Iteration {iteration + 1}:

Current solution: {solution}

Evaluation: {evaluate_solution(solution, goal)}

Issues identified: {identify_issues(solution)}

Refinement plan: {plan_refinement(solution)}

Next action: {determine_refinement_action()}
"""

        action = extract_action(thought)
        result = execute(action)

        solution = update_solution(solution, result)

        # Check if good enough
        if is_satisfactory(solution, goal):
            break

    return solution
```

## Error Handling in ReAct

ReAct's explicit reasoning enables sophisticated error handling.

### Error Recognition

```python
def error_aware_thought(observation, goal, last_result):
    """
    Recognize and reason about errors.
    """
    if last_result.get("error"):
        thought = f"""
Error encountered: {last_result['error']}

Analysis:
- What went wrong: {analyze_error(last_result['error'])}
- Why it happened: {diagnose_cause(last_result)}
- Is it recoverable: {is_recoverable(last_result['error'])}

Recovery strategy:
{plan_recovery(last_result['error'])}

Adjusted action: {propose_alternative_action()}
"""
    else:
        thought = generate_normal_thought(observation, goal)

    return thought
```

### Retry with Reasoning

```python
def react_with_retry(goal, tools):
    """
    ReAct with intelligent retrying.
    """
    max_retries = 3
    history = []

    for iteration in range(max_iterations):
        observation = create_observation(history)

        # Check for repeated failures
        recent_failures = count_recent_failures(history, window=3)

        if recent_failures > 0:
            thought = f"""
I've failed {recent_failures} times recently.

Previous attempts:
{summarize_failures(history)}

Common pattern: {identify_failure_pattern(history)}

I need to try a different approach: {propose_alternative()}

Specifically: {detail_new_approach()}
"""
        else:
            thought = generate_normal_thought(observation, goal)

        action = extract_action(thought)
        result = execute(action)

        history.append({"thought": thought, "action": action, "result": result})

        if result["success"]:
            if action["type"] == "FINISH":
                return result
        else:
            # Failed - continue to next iteration with error context
            continue
```

### Self-Correction

```python
def self_correcting_react(goal, tools):
    """
    ReAct with self-correction.
    """
    history = []

    while not done:
        observation = create_observation(history)
        thought = generate_thought(observation, goal)
        action = extract_action(thought)

        # Before executing, verify action makes sense
        verification = verify_action(action, thought, goal)

        if not verification["valid"]:
            # Self-correct
            correction_thought = f"""
Wait, I was about to: {action}

But this doesn't make sense because: {verification['reason']}

Let me reconsider: {reconsider(observation, goal, action)}

Better action: {propose_better_action()}
"""
            action = extract_action(correction_thought)
            thought = correction_thought

        result = execute(action)
        history.append({"thought": thought, "action": action, "result": result})
```

## ReAct with Different Tools

How ReAct adapts to different tool types.

### Information Retrieval Tools

```python
def react_with_search(goal, search_tools):
    """
    ReAct optimized for search and retrieval.
    """
    thought = f"""
Goal: {goal}

I need to gather information. My strategy:
1. Start with broad search to understand landscape
2. Refine based on results
3. Deep dive into most relevant sources

First search query: {formulate_query(goal)}
Why this query: {justify_query()}
"""

    action = {"tool": "search", "query": formulate_query(goal)}
    return thought, action
```

### Computational Tools

```python
def react_with_computation(goal, compute_tools):
    """
    ReAct for computational tasks.
    """
    thought = f"""
Goal: {goal}

This requires computation. Let me plan:
1. Identify what to compute: {identify_computation(goal)}
2. Choose appropriate tool: {select_tool(compute_tools)}
3. Prepare inputs: {prepare_inputs()}
4. Expected output: {describe_expected_output()}

Executing: {describe_computation()}
"""

    action = {"tool": "calculate", "expression": prepare_computation(goal)}
    return thought, action
```

### Communication Tools

```python
def react_with_communication(goal, comm_tools):
    """
    ReAct for communication tasks.
    """
    thought = f"""
Goal: {goal}

I need to communicate. Considerations:
- Audience: {identify_audience(goal)}
- Purpose: {identify_purpose(goal)}
- Tone: {determine_tone(goal)}
- Content: {outline_content(goal)}

Crafting message: {describe_message_strategy()}
"""

    action = {"tool": "send_message", "content": craft_message(goal)}
    return thought, action
```

## Optimizing ReAct

Making ReAct more efficient.

### Compact Thoughts

```python
def compact_thought(observation, goal):
    """
    Generate concise but informative thoughts.
    """
    # Full thought (verbose)
    full_thought = """
    Current situation: I have searched for papers
    Goal: Summarize findings
    Progress: Found 20 papers, read 2
    Reasoning: I need to read more papers to get comprehensive view
    Next: Read paper #3 which covers generation quality
    Expected: Will gain insights on generation improvements
    """

    # Compact thought (efficient)
    compact_thought = """
    Status: Read 2/20 papers
    Next: Read paper #3 (generation quality) for comprehensive view
    """

    # Choose based on situation
    if needs_detailed_reasoning(observation):
        return full_thought
    else:
        return compact_thought
```

### Cached Reasoning

```python
class CachedReActAgent:
    """
    ReAct with reasoning caching.
    """

    def __init__(self):
        self.reasoning_cache = {}

    def think(self, observation, goal):
        """
        Generate thought with caching.
        """
        # Create cache key from observation
        cache_key = hash_observation(observation, goal)

        # Check cache
        if cache_key in self.reasoning_cache:
            cached_thought = self.reasoning_cache[cache_key]

            # Add note that this is cached
            return f"[Cached reasoning]\n{cached_thought}"

        # Generate new thought
        thought = generate_thought(observation, goal)

        # Cache for future
        self.reasoning_cache[cache_key] = thought

        return thought
```

### Adaptive Reasoning Depth

```python
def adaptive_react(observation, goal, complexity):
    """
    Adjust reasoning depth based on complexity.
    """
    if complexity == "simple":
        # Minimal reasoning
        thought = f"Simple task: {goal}. Action: {quick_decision(observation)}"

    elif complexity == "medium":
        # Standard reasoning
        thought = f"""
        Goal: {goal}
        Analysis: {brief_analysis(observation)}
        Action: {decide(observation)}
        """

    else:  # complex
        # Deep reasoning
        thought = f"""
        Goal: {goal}
        Situation analysis: {deep_analysis(observation)}
        Options: {enumerate_options(observation)}
        Evaluation: {evaluate_options(observation, goal)}
        Decision: {make_decision()}
        Rationale: {full_justification()}
        """

    return thought
```

## Common Patterns

Frequently-used ReAct patterns.

### Search-Read-Synthesize

```python
def search_read_synthesize_pattern(goal):
    """
    Common pattern for research tasks.
    """
    # Phase 1: Search
    thought_1 = """
    I need to find relevant information.
    Searching with query: {query}
    """
    action_1 = "search(query)"

    # Phase 2: Read
    thought_2 = """
    Found {n} results. Reading top {k} most relevant.
    This will give me {expected_insights}.
    """
    action_2 = "read(top_k_results)"

    # Phase 3: Synthesize
    thought_3 = """
    I've read {k} sources. Key findings:
    - {finding_1}
    - {finding_2}
    Now synthesizing into {output_format}.
    """
    action_3 = "synthesize(findings)"

    return [
        (thought_1, action_1),
        (thought_2, action_2),
        (thought_3, action_3)
    ]
```

### Verify-Then-Act

```python
def verify_then_act_pattern(observation, goal):
    """
    Verify preconditions before acting.
    """
    thought = f"""
Before acting, let me verify:
- Precondition 1: {check_precondition_1(observation)}
- Precondition 2: {check_precondition_2(observation)}

All conditions met: {all_conditions_met()}

Safe to proceed: {is_safe(observation, action)}

Action: {action}
"""
    return thought
```

### Try-Catch-Retry

```python
def try_catch_retry_pattern(observation, goal, attempt):
    """
    Handle errors with explicit reasoning.
    """
    if attempt == 0:
        thought = """
        First attempt at: {action}
        Expected to: {expected_outcome}
        """
    else:
        thought = f"""
        Attempt {attempt} failed with: {{last_error}}

        Analysis: {{analyze_error(last_error)}}

        Alternative approach: {{alternative}}

        Why this might work: {{reasoning}}
        """

    return thought
```

## Debugging ReAct Agents

Tools for debugging ReAct implementations.

### Trace Visualization

```python
def visualize_react_trace(history):
    """
    Visualize ReAct execution trace.
    """
    print("=" * 80)
    print("ReAct Execution Trace")
    print("=" * 80)

    for i, step in enumerate(history, 1):
        print(f"\n{'─' * 80}")
        print(f"Step {i}")
        print(f"{'─' * 80}")

        print(f"\n💭 Thought:")
        print(f"   {step['thought']}")

        print(f"\n⚡ Action:")
        print(f"   {step['action']}")

        print(f"\n👁️  Observation:")
        if step['observation'].get('success'):
            print(f"   ✅ Success: {step['observation']['result']}")
        else:
            print(f"   ❌ Error: {step['observation']['error']}")

    print("\n" + "=" * 80)
```

### Thought Quality Analysis

```python
def analyze_thought_quality(thought, action, outcome):
    """
    Analyze quality of reasoning.
    """
    analysis = {
        "has_goal_reference": "goal" in thought.lower(),
        "has_justification": any(word in thought.lower() for word in ["because", "since", "therefore"]),
        "has_expectation": any(word in thought.lower() for word in ["expect", "will", "should"]),
        "led_to_success": outcome.get("success", False),
        "action_matches_thought": check_alignment(thought, action)
    }

    score = sum(analysis.values()) / len(analysis)

    return {
        "score": score,
        "details": analysis,
        "suggestions": generate_suggestions(analysis)
    }
```

### Reasoning Path Analysis

```python
def analyze_reasoning_path(history):
    """
    Analyze the reasoning path taken.
    """
    # Extract key decisions
    decisions = [step["action"] for step in history]

    # Check for patterns
    loops = detect_loops(decisions)
    backtracking = detect_backtracking(decisions)
    efficiency = calculate_efficiency(decisions, goal)

    # Identify improvements
    optimal_path = find_optimal_path(goal)
    wasted_actions = identify_waste(decisions, optimal_path)

    return {
        "path_taken": decisions,
        "optimal_path": optimal_path,
        "efficiency": efficiency,
        "loops_detected": loops,
        "backtracking": backtracking,
        "wasted_actions": wasted_actions,
        "suggestions": suggest_improvements(decisions, optimal_path)
    }
```

## When to Use ReAct

Guidelines for choosing ReAct.

### Use ReAct When:

1. **Complex reasoning required**
   - Multi-step logic
   - Many decision points
   - Non-obvious solutions

2. **Interpretability matters**
   - Need to explain decisions
   - Debugging difficult issues
   - Building trust with users

3. **Error-prone tasks**
   - High failure risk
   - Expensive mistakes
   - Need recovery strategies

4. **Learning from traces**
   - Want to analyze behavior
   - Improve future performance
   - Build reasoning datasets

### Don't Use ReAct When:

1. **Simple, deterministic tasks**
   - Clear action sequences
   - No decision-making needed

2. **Speed is critical**
   - Real-time responses required
   - Cost constraints tight

3. **Reasoning is trivial**
   - Obvious next steps
   - Single-action tasks

## Advanced ReAct Variants

Sophisticated variations of ReAct.

### ReAct with Self-Consistency

```python
def self_consistent_react(goal, tools, num_samples=3):
    """
    Generate multiple reasoning paths, choose most common answer.
    """
    results = []

    for i in range(num_samples):
        # Generate independent reasoning path
        thought = generate_thought(goal, temperature=0.7)
        action = extract_action(thought)
        result = execute(action)

        results.append({
            "thought": thought,
            "action": action,
            "result": result
        })

    # Find most common result
    final_result = most_common_result(results)

    return final_result
```

### ReAct with Reflection

```python
def reflective_react(goal, tools):
    """
    ReAct with post-action reflection.
    """
    history = []

    while not done:
        # Standard ReAct step
        observation = create_observation(history)
        thought = generate_thought(observation, goal)
        action = extract_action(thought)
        result = execute(action)

        # Reflection step
        reflection = f"""
Reflecting on what just happened:

Action taken: {action}
Result: {result}

Questions:
- Did this advance toward the goal? {assess_progress(result, goal)}
- Was this the best action? {evaluate_choice(action, observation)}
- What did I learn? {extract_learning(result)}
- What would I do differently? {consider_alternatives(action, result)}

Insight: {synthesize_insight(result, goal)}
"""

        history.append({
            "thought": thought,
            "action": action,
            "result": result,
            "reflection": reflection
        })
```

### Hierarchical ReAct

```python
def hierarchical_react(goal, tools):
    """
    Multi-level ReAct: high-level planning + low-level execution.
    """
    # High-level thought
    high_level_thought = f"""
Goal: {goal}

High-level plan:
1. {subgoal_1}
2. {subgoal_2}
3. {subgoal_3}

Starting with: {subgoal_1}
"""

    # Execute each subgoal with low-level ReAct
    for subgoal in subgoals:
        low_level_result = react_loop(subgoal, tools)

        # High-level reflection
        high_level_thought = f"""
Completed: {subgoal}
Result: {low_level_result}

Progress toward main goal: {assess_progress()}

Next subgoal: {next_subgoal}
"""
```

## Summary

ReAct (Reasoning and Acting) is a powerful agent architecture that interleaves explicit reasoning with action execution:

1. **Core Pattern**: Thought → Action → Observation (repeated)
2. **Key Benefit**: Explicit reasoning improves decision quality and interpretability
3. **Thought Components**: Situation assessment, goal reference, action rationale, expected outcome
4. **Implementation**: Generate thought with LLM, extract action, execute, observe result
5. **vs Basic Loops**: More expensive but better decisions and interpretability
6. **Reasoning Strategies**: Step-by-step, hypothesis-driven, cost-benefit, analogical
7. **Error Handling**: Explicit reasoning enables sophisticated error recovery
8. **Optimization**: Compact thoughts, caching, adaptive depth
9. **Common Patterns**: Search-read-synthesize, verify-then-act, try-catch-retry
10. **When to Use**: Complex reasoning, interpretability needed, error-prone tasks

ReAct has become one of the most widely-used agent patterns because it strikes a good balance between simplicity, effectiveness, and interpretability.

## Next Steps

- Continue to [Planning Architectures](planning-architectures.md) to learn about upfront planning strategies
- Explore [Reflection and Self-Critique](reflection.md) for agents that evaluate and improve their own performance
- Study [Design Principles](design-principles.md) for building robust agent systems
- Review [The Agent Loop](agent-loop.md) for foundational concepts
