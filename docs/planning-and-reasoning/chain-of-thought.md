# Chain-of-Thought Reasoning

## Table of Contents

- [Introduction](#introduction)
- [What is Chain-of-Thought?](#what-is-chain-of-thought)
- [The Science Behind CoT](#the-science-behind-cot)
- [Basic CoT Prompting](#basic-cot-prompting)
- [Zero-Shot CoT](#zero-shot-cot)
- [Few-Shot CoT](#few-shot-cot)
- [When to Use CoT](#when-to-use-cot)
- [CoT for Complex Tasks](#cot-for-complex-tasks)
- [Implementing CoT in Agents](#implementing-cot-in-agents)
- [Structuring Reasoning Chains](#structuring-reasoning-chains)
- [CoT Quality and Evaluation](#cot-quality-and-evaluation)
- [Common CoT Patterns](#common-cot-patterns)
- [Advanced CoT Techniques](#advanced-cot-techniques)
- [Self-Consistency with CoT](#self-consistency-with-cot)
- [CoT for Multi-Step Problems](#cot-for-multi-step-problems)
- [Debugging CoT Reasoning](#debugging-cot-reasoning)
- [CoT in Production Systems](#cot-in-production-systems)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Chain-of-Thought (CoT) reasoning is a technique where agents explicitly articulate their thinking process step-by-step before arriving at a final answer. Instead of jumping directly from question to solution, the agent shows its work--breaking down complex problems into intermediate steps.

> "Show your work" - Elementary school wisdom that applies to AI agents

The revolutionary insight from CoT research: **making reasoning explicit dramatically improves performance on complex tasks**. By articulating intermediate thoughts, language models access more of their latent reasoning capabilities.

### The Core Concept

**Without CoT**:

```
Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls each.
   How many tennis balls does he have now?
A: 11
```

**With CoT**:

```
Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls each.
   How many tennis balls does he have now?
A: Let me think through this step by step:
   1. Roger starts with 5 tennis balls
   2. He buys 2 cans, each containing 3 balls
   3. New balls from cans: 2 × 3 = 6 balls
   4. Total balls: 5 + 6 = 11 balls
   Answer: 11 tennis balls
```

The second approach isn't just more transparent--it's **more reliable**. The explicit reasoning reduces errors and catches mistakes.

### Why CoT Matters for Agents

**For simple tasks**: CoT might be unnecessary overhead

**For complex tasks**: CoT is often essential:

- **Multi-step reasoning**: Breaking problems into steps
- **Mathematical calculations**: Showing arithmetic work
- **Logical deduction**: Following chains of implications
- **Decision making**: Weighing options systematically
- **Error recovery**: Catching and correcting mistakes

CoT transforms agent behavior from **reactive** to **deliberative**.

## What is Chain-of-Thought?

Chain-of-Thought is both a **prompting technique** and a **reasoning pattern** that encourages step-by-step problem decomposition.

### Definition

**Chain-of-Thought (CoT)**: A reasoning approach where intermediate steps are explicitly articulated before reaching a conclusion. Each step follows logically from previous steps, forming a "chain" of reasoning.

### Key Characteristics

**1. Sequential**: Steps follow in logical order

```
Step 1 → Step 2 → Step 3 → Conclusion
```

**2. Explicit**: Every thought is written out

```
Not: "The answer is X"
But: "First I'll... then I'll... so therefore X"
```

**3. Intermediate**: Focuses on the path, not just destination

```
Not just: INPUT → OUTPUT
But:     INPUT → STEP1 → STEP2 → ... → OUTPUT
```

**4. Natural Language**: Uses human-readable reasoning

```
"Let me think about this..."
"First, I need to..."
"This means that..."
```

### CoT vs Direct Answer

```
┌────────────────────────────────────────┐
│ Direct Answer Pattern                   │
├────────────────────────────────────────┤
│ Problem → Model → Answer               │
│                                         │
│ Fast but opaque                        │
│ Works for simple problems              │
│ Hard to debug failures                 │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│ Chain-of-Thought Pattern                │
├────────────────────────────────────────┤
│ Problem → Model → Reasoning Steps      │
│                  → Answer               │
│                                         │
│ Slower but transparent                 │
│ Better for complex problems            │
│ Easy to debug reasoning                │
└────────────────────────────────────────┘
```

## The Science Behind CoT

The effectiveness of CoT is backed by extensive research showing significant performance improvements across many task types.

### Empirical Results

**Original CoT Paper (Wei et al., 2022)**:

- 8x improvement on math word problems (GSM8K)
- 3x improvement on symbolic reasoning tasks
- Works best with models ≥100B parameters

**Key Finding**: CoT helps models that **already have latent reasoning ability** express it more effectively.

### Why Does CoT Work?

**Hypothesis 1: Computational Steps**

More tokens = more computation = better answers

```python
# Direct answer: ~10 tokens generated
"The answer is 42"

# CoT answer: ~100 tokens generated
"First we need to... Then we calculate... Finally..."
# More computation time to process the problem
```

**Hypothesis 2: Working Memory**

CoT provides external memory for intermediate results

```
Without CoT:
- Model must hold all intermediate values in attention
- Limited by context window and attention span

With CoT:
- Intermediate values written as tokens
- Can be referenced later in the chain
- Reduces cognitive load on the model
```

**Hypothesis 3: Error Correction**

Explicit steps allow catching and fixing mistakes

```
Step 1: Calculate 2 × 3 = 6
Step 2: Calculate 5 + 6 = 12  ← Error visible!
Step 3: Wait, that should be 11
Step 2 (corrected): 5 + 6 = 11
```

**Hypothesis 4: Better Priors**

Training data contains many examples of step-by-step reasoning

```
Internet text includes:
- Worked examples in textbooks
- Stack Overflow answers with explanations
- Tutorial articles with step-by-step guides
- Mathematical proofs with explicit steps

CoT prompting aligns with these training patterns
```

### When CoT Helps Most

**Task characteristics where CoT shines**:

1. **Multi-step problems**: Require several operations in sequence
2. **Mathematical reasoning**: Need arithmetic/algebra steps
3. **Logical deduction**: Follow chains of implications
4. **Commonsense reasoning**: Need to consider context
5. **Symbolic manipulation**: Transform expressions step-by-step

**Task characteristics where CoT helps less**:

1. **Simple retrieval**: "What is the capital of France?"
2. **Pattern matching**: "Is this email spam?"
3. **Single-step calculation**: "What is 2 + 2?"
4. **Learned associations**: "Sky is to blue as grass is to \_\_\_"

## Basic CoT Prompting

The simplest way to elicit CoT reasoning is through explicit instruction.

### Explicit Instruction

**Template**:

```
{problem}

Think through this step by step.
```

**Example**:

```python
prompt = """
A bakery makes 120 cupcakes. They sell 2/3 in the morning and
1/4 of the remainder in the afternoon. How many cupcakes are left?

Think through this step by step.
"""

# Model generates:
# "Let me work through this carefully:
#  1. Start with 120 cupcakes
#  2. Morning sales: 2/3 of 120 = 80 cupcakes
#  3. After morning: 120 - 80 = 40 cupcakes remain
#  4. Afternoon sales: 1/4 of 40 = 10 cupcakes
#  5. Final amount: 40 - 10 = 30 cupcakes
#  Answer: 30 cupcakes are left"
```

### Common CoT Triggers

Different phrases that encourage CoT reasoning:

**1. "Let's think step by step"**

```python
prompt = f"{problem}\n\nLet's think step by step."
```

**2. "First... then... finally..."**

```python
prompt = f"{problem}\n\nFirst, let's identify what we know. Then..."
```

**3. "Break this down:"**

```python
prompt = f"{problem}\n\nBreak this down into steps:"
```

**4. "Solve this by:"**

```python
prompt = f"{problem}\n\nSolve this by:\n1."
```

**5. "Think through this carefully:"**

```python
prompt = f"{problem}\n\nThink through this carefully:"
```

### Implementation

```python
def chain_of_thought_query(problem: str, model) -> dict:
    """
    Query a model with CoT prompting.

    Args:
        problem: The question or task to solve
        model: Language model instance

    Returns:
        Dictionary with reasoning steps and final answer
    """
    prompt = f"""
{problem}

Let's think through this step by step:
"""

    response = model.generate(prompt)

    # Parse the response
    reasoning = extract_reasoning_steps(response)
    answer = extract_final_answer(response)

    return {
        "problem": problem,
        "reasoning": reasoning,
        "answer": answer,
        "full_response": response
    }


def extract_reasoning_steps(response: str) -> list[str]:
    """Extract individual reasoning steps from response."""
    steps = []

    # Look for numbered steps
    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        # Match "1.", "Step 1:", "First,", etc.
        if (line and
            (line[0].isdigit() or
             line.lower().startswith(('step', 'first', 'next', 'then', 'finally')))):
            steps.append(line)

    return steps


def extract_final_answer(response: str) -> str:
    """Extract the final answer from CoT reasoning."""
    # Look for answer indicators
    answer_markers = [
        "answer:",
        "therefore,",
        "so the answer is",
        "the final answer is",
        "in conclusion,"
    ]

    response_lower = response.lower()
    for marker in answer_markers:
        if marker in response_lower:
            idx = response_lower.find(marker)
            # Extract text after marker
            answer_text = response[idx:].split('\n')[0]
            return answer_text.strip()

    # If no marker found, return last line
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    return lines[-1] if lines else ""


# Example usage
problem = """
A store has 45 apples. They receive a shipment of 8 boxes, each
containing 12 apples. If they sell 60 apples, how many are left?
"""

result = chain_of_thought_query(problem, model)
print(f"Reasoning:\n{chr(10).join(result['reasoning'])}")
print(f"\nFinal Answer: {result['answer']}")
```

### Output Example

```
Reasoning:
1. Store starts with 45 apples
2. Shipment arrives: 8 boxes × 12 apples = 96 apples
3. Total after shipment: 45 + 96 = 141 apples
4. Apples sold: 60
5. Apples remaining: 141 - 60 = 81 apples

Final Answer: 81 apples are left
```

## Zero-Shot CoT

Zero-shot CoT is the remarkable finding that simply adding "Let's think step by step" to any prompt can induce reasoning behavior--**without providing any examples**.

### The Magic Phrase

**Discovery (Kojima et al., 2022)**: A single phrase dramatically improves reasoning:

```
"Let's think step by step."
```

**Why it's remarkable**:

- No examples needed (zero-shot)
- Works across diverse tasks
- Simple to implement
- Significant performance gains

### Implementation

```python
def zero_shot_cot(problem: str, model) -> str:
    """
    Zero-shot Chain-of-Thought prompting.

    Simply append the magic phrase to any problem.
    """
    prompt = f"{problem}\n\nLet's think step by step."
    return model.generate(prompt)


# Comparison
def compare_with_without_cot(problem: str, model):
    """Compare direct answer vs CoT."""

    # Direct answer
    direct = model.generate(problem)

    # CoT answer
    cot = zero_shot_cot(problem, model)

    print("WITHOUT CoT:")
    print(direct)
    print("\nWITH CoT:")
    print(cot)
    print("\n" + "="*50)


# Example problems
problems = [
    "If a train travels 60 mph for 2.5 hours, how far does it go?",
    "What is 15% of 240?",
    "A rectangle has length 8 and width 5. What is its area and perimeter?",
]

for problem in problems:
    compare_with_without_cot(problem, model)
```

### Zero-Shot CoT in Agent Loops

```python
class ZeroShotCoTAgent:
    """Agent that uses zero-shot CoT for all reasoning."""

    def __init__(self, model, tools):
        self.model = model
        self.tools = tools
        self.cot_trigger = "\n\nLet's think step by step."

    def act(self, observation: str, task: str) -> dict:
        """
        Decide on an action using zero-shot CoT.

        Returns reasoning trace and chosen action.
        """
        prompt = f"""
Task: {task}
Current situation: {observation}

Available actions:
{self._format_tools()}

What should I do next?{self.cot_trigger}
"""

        response = self.model.generate(prompt)

        # Parse the response
        reasoning = self._extract_reasoning(response)
        action = self._extract_action(response)

        return {
            "reasoning": reasoning,
            "action": action,
            "full_response": response
        }

    def _format_tools(self) -> str:
        """Format available tools for the prompt."""
        return "\n".join(f"- {tool.name}: {tool.description}"
                        for tool in self.tools)

    def _extract_reasoning(self, response: str) -> str:
        """Extract the reasoning steps."""
        # Everything before the action is reasoning
        if "Action:" in response:
            return response.split("Action:")[0].strip()
        return response

    def _extract_action(self, response: str) -> dict:
        """Extract the chosen action."""
        if "Action:" not in response:
            return None

        action_text = response.split("Action:")[1].strip()
        # Parse action format: tool_name(arg1, arg2, ...)
        # Simplified parsing
        action_name = action_text.split("(")[0].strip()
        return {"tool": action_name, "params": action_text}


# Example usage
agent = ZeroShotCoTAgent(model, tools)

result = agent.act(
    observation="I have searched for 'Python tutorials' and found 10 results",
    task="Find and summarize beginner-friendly Python tutorials"
)

print("Agent's Reasoning:")
print(result['reasoning'])
print("\nChosen Action:")
print(result['action'])
```

### When Zero-Shot CoT Works Best

**Strong performance**:

- Math word problems
- Multi-step reasoning
- Logical puzzles
- Arithmetic calculations
- Commonsense reasoning

**Weaker performance**:

- Very complex problems (few-shot helps more)
- Domain-specific tasks (examples provide context)
- Tasks with specific output formats
- Problems requiring specialized knowledge

## Few-Shot CoT

Few-shot CoT provides examples of reasoning chains, teaching the model **how** to think about similar problems.

### The Power of Examples

**Principle**: Show the model examples of good reasoning, and it will imitate that reasoning pattern.

```python
few_shot_examples = """
Q: A restaurant has 23 tables. Each table seats 4 people.
   If 78 people are dining, how many tables are occupied?

A: Let me solve this step by step:
   1. Each table seats 4 people
   2. We have 78 people total
   3. Tables needed: 78 ÷ 4 = 19.5 tables
   4. Since we can't have half a table, we need 20 tables
   5. We have 23 tables total, and 20 are occupied
   Answer: 20 tables are occupied

---

Q: A bookstore sells hardcovers for $28 and paperbacks for $12.
   If I buy 3 hardcovers and 5 paperbacks, what's my total?

A: Let me work through this:
   1. Hardcover price: $28 each
   2. Hardcovers bought: 3
   3. Hardcover cost: 3 × $28 = $84
   4. Paperback price: $12 each
   5. Paperbacks bought: 5
   6. Paperback cost: 5 × $12 = $60
   7. Total cost: $84 + $60 = $144
   Answer: $144

---

Q: {NEW_PROBLEM}

A: """
```

### Choosing Good Examples

**Characteristics of effective CoT examples**:

1. **Similar structure** to target problems
2. **Clear, explicit steps** that are easy to follow
3. **Correct reasoning** (obviously!)
4. **Appropriate complexity** - not too simple, not too complex
5. **Diverse strategies** - show different reasoning approaches

### Implementation

```python
class FewShotCoTAgent:
    """Agent using few-shot CoT with example bank."""

    def __init__(self, model, examples: list[dict]):
        self.model = model
        self.examples = examples

    def solve(self, problem: str, num_examples: int = 3) -> dict:
        """
        Solve a problem using few-shot CoT.

        Args:
            problem: The problem to solve
            num_examples: Number of examples to include

        Returns:
            Dictionary with reasoning and answer
        """
        # Select most relevant examples
        selected_examples = self._select_examples(problem, num_examples)

        # Build prompt with examples
        prompt = self._build_prompt(selected_examples, problem)

        # Generate response
        response = self.model.generate(prompt)

        return {
            "problem": problem,
            "examples_used": selected_examples,
            "reasoning": response,
            "answer": self._extract_answer(response)
        }

    def _select_examples(self, problem: str, num: int) -> list[dict]:
        """
        Select most relevant examples for the problem.

        Could use:
        - Random selection
        - Similarity-based selection (embeddings)
        - Task-type matching
        """
        # Simple: return first N examples
        # TODO: Implement smarter selection
        return self.examples[:num]

    def _build_prompt(self, examples: list[dict], problem: str) -> str:
        """Construct few-shot prompt with examples."""
        prompt_parts = []

        # Add each example
        for ex in examples:
            prompt_parts.append(f"Q: {ex['problem']}\n")
            prompt_parts.append(f"A: {ex['reasoning']}\n")
            prompt_parts.append("---\n")

        # Add the actual problem
        prompt_parts.append(f"Q: {problem}\n")
        prompt_parts.append("A: Let me solve this step by step:\n")

        return "\n".join(prompt_parts)

    def _extract_answer(self, response: str) -> str:
        """Extract final answer from response."""
        # Look for "Answer:" or similar
        if "Answer:" in response:
            return response.split("Answer:")[-1].strip()
        # Otherwise return last line
        return response.strip().split('\n')[-1]


# Example bank
examples = [
    {
        "problem": "A car travels 240 miles in 4 hours. What is its average speed?",
        "reasoning": """Let me calculate step by step:
1. Distance traveled: 240 miles
2. Time taken: 4 hours
3. Average speed = Distance ÷ Time
4. Average speed = 240 ÷ 4 = 60 mph
Answer: 60 mph"""
    },
    {
        "problem": "If 5 workers can build a wall in 12 days, how long will 3 workers take?",
        "reasoning": """Let me reason through this:
1. 5 workers take 12 days
2. Total work = 5 × 12 = 60 worker-days
3. With 3 workers: Days = 60 ÷ 3 = 20 days
4. More workers take less time (inverse proportion)
Answer: 20 days"""
    },
    {
        "problem": "A box contains 36 chocolates. If 3/4 are dark chocolate, how many are milk chocolate?",
        "reasoning": """Let me work this out:
1. Total chocolates: 36
2. Dark chocolate: 3/4 of 36 = 27
3. Milk chocolate: Total - Dark = 36 - 27 = 9
4. We can verify: 9 is 1/4 of 36 ✓
Answer: 9 milk chocolates"""
    }
]

# Create agent
agent = FewShotCoTAgent(model, examples)

# Solve new problem
result = agent.solve(
    "A garden has 48 plants. If 2/3 are flowers and the rest are herbs, how many herbs are there?"
)

print("Reasoning:")
print(result['reasoning'])
print(f"\nAnswer: {result['answer']}")
```

### Dynamic Example Selection

For better performance, select examples similar to the current problem:

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SmartFewShotCoT:
    """Few-shot CoT with intelligent example selection."""

    def __init__(self, model, examples: list[dict], embedding_model):
        self.model = model
        self.examples = examples
        self.embedding_model = embedding_model

        # Pre-compute example embeddings
        self.example_embeddings = self._embed_examples()

    def _embed_examples(self) -> np.ndarray:
        """Compute embeddings for all examples."""
        texts = [ex['problem'] for ex in self.examples]
        return self.embedding_model.encode(texts)

    def _select_similar_examples(self, problem: str, k: int = 3) -> list[dict]:
        """Select k most similar examples to the problem."""
        # Embed the problem
        problem_embedding = self.embedding_model.encode([problem])

        # Compute similarities
        similarities = cosine_similarity(
            problem_embedding,
            self.example_embeddings
        )[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[-k:][::-1]

        return [self.examples[i] for i in top_indices]

    def solve(self, problem: str, k: int = 3) -> dict:
        """Solve using most relevant examples."""
        examples = self._select_similar_examples(problem, k)

        prompt = self._build_prompt(examples, problem)
        response = self.model.generate(prompt)

        return {
            "problem": problem,
            "selected_examples": [ex['problem'] for ex in examples],
            "reasoning": response,
            "answer": self._extract_answer(response)
        }

    def _build_prompt(self, examples: list[dict], problem: str) -> str:
        """Build prompt with selected examples."""
        parts = ["Here are some examples of problem-solving:\n"]

        for i, ex in enumerate(examples, 1):
            parts.append(f"\nExample {i}:")
            parts.append(f"Q: {ex['problem']}")
            parts.append(f"A: {ex['reasoning']}\n")

        parts.append(f"\nNow solve this problem:")
        parts.append(f"Q: {problem}")
        parts.append("A: Let me solve this step by step:")

        return "\n".join(parts)

    def _extract_answer(self, response: str) -> str:
        """Extract the final answer."""
        if "Answer:" in response:
            return response.split("Answer:")[-1].strip()
        return response.strip().split('\n')[-1]
```

## When to Use CoT

Not every problem benefits from CoT. Understanding when to use it is crucial for efficient agent design.

### Decision Framework

```
┌─────────────────────────────────────────┐
│          Is CoT Appropriate?             │
├─────────────────────────────────────────┤
│                                          │
│  Single-step task? ────────► NO         │
│  (e.g., simple lookup)                   │
│                                          │
│  Multi-step reasoning? ───► YES         │
│  (e.g., math, logic)                     │
│                                          │
│  Performance-critical? ───► MAYBE       │
│  (CoT is slower)                        │
│                                          │
│  Need interpretability? ──► YES         │
│  (CoT shows reasoning)                   │
│                                          │
│  Cost-sensitive? ─────────► MAYBE       │
│  (CoT uses more tokens)                  │
│                                          │
└─────────────────────────────────────────┘
```

### Use Cases: When CoT Helps

**1. Mathematical Reasoning**

```python
# CoT shines here
problem = "If a = 3, b = 2a + 5, and c = b × 2, what is c?"

# Without CoT: might get wrong answer
# With CoT: step-by-step calculation ensures correctness
```

**2. Multi-Step Planning**

```python
# CoT breaks down complex plans
task = "Book a flight to Paris, reserve a hotel, and plan daily activities for 5 days"

# CoT helps structure the multi-step process
```

**3. Logical Deduction**

```python
# CoT tracks logical implications
problem = """
All birds have wings. All eagles are birds.
Tweety is an eagle. Does Tweety have wings?
"""

# CoT explicitly tracks: eagles → birds → wings → Tweety has wings
```

**4. Debugging and Error Analysis**

```python
# CoT makes errors visible
problem = "Why did this code fail?"

# CoT traces through execution step-by-step to find the bug
```

**5. Complex Decision Making**

```python
# CoT weighs multiple factors
decision = "Should we migrate this service to Kubernetes?"

# CoT considers: current state, costs, benefits, risks, timeline
```

### Use Cases: When CoT Doesn't Help

**1. Simple Retrieval**

```python
# CoT is overkill
question = "What is the capital of France?"
# Direct answer is fine: "Paris"
```

**2. Pattern Recognition**

```python
# CoT might hurt performance
task = "Classify this email as spam or not spam"
# Model recognizes patterns directly, CoT is unnecessary
```

**3. Cached Knowledge**

```python
# CoT doesn't add value
question = "What is 2 + 2?"
# Direct answer: "4"
```

**4. Real-Time Systems**

```python
# CoT is too slow
task = "React to incoming sensor data in <10ms"
# No time for step-by-step reasoning
```

**5. Creative Generation**

```python
# CoT might constrain creativity
task = "Write a creative poem about autumn"
# Direct generation allows free-form creativity
```

### Adaptive CoT: Choosing Dynamically

```python
class AdaptiveCoTAgent:
    """Agent that decides when to use CoT based on task characteristics."""

    def __init__(self, model):
        self.model = model

    def should_use_cot(self, task: str) -> bool:
        """
        Decide whether to use CoT for this task.

        Heuristics:
        - Use CoT for multi-step, mathematical, or logical tasks
        - Skip CoT for simple lookups or pattern matching
        """
        task_lower = task.lower()

        # Indicators for CoT
        cot_indicators = [
            'calculate', 'solve', 'reason', 'analyze', 'plan',
            'step', 'how many', 'why', 'explain', 'compare',
            'if', 'then', 'because', 'therefore'
        ]

        # Indicators against CoT
        no_cot_indicators = [
            'what is', 'define', 'list', 'name', 'who is',
            'when was', 'simple', 'quick'
        ]

        cot_score = sum(1 for ind in cot_indicators if ind in task_lower)
        no_cot_score = sum(1 for ind in no_cot_indicators if ind in task_lower)

        # Additional heuristics
        if len(task.split()) > 30:  # Long task likely complex
            cot_score += 2

        if '?' in task and task.count('?') == 1 and len(task.split()) < 10:
            no_cot_score += 1  # Simple question

        return cot_score > no_cot_score

    def solve(self, task: str) -> dict:
        """Solve task with or without CoT based on heuristics."""
        use_cot = self.should_use_cot(task)

        if use_cot:
            prompt = f"{task}\n\nLet's think step by step:"
        else:
            prompt = task

        response = self.model.generate(prompt)

        return {
            "task": task,
            "used_cot": use_cot,
            "response": response
        }


# Example usage
agent = AdaptiveCoTAgent(model)

# Tasks of varying complexity
tasks = [
    "What is the capital of Japan?",  # No CoT needed
    "If a train leaves at 2pm going 60mph and another leaves at 3pm going 80mph, when do they meet?",  # CoT needed
    "List three primary colors",  # No CoT needed
    "Analyze the trade-offs between microservices and monolithic architecture",  # CoT helpful
]

for task in tasks:
    result = agent.solve(task)
    print(f"Task: {task}")
    print(f"Used CoT: {result['used_cot']}")
    print(f"Response: {result['response'][:100]}...")
    print()
```

## CoT for Complex Tasks

For truly complex tasks, basic CoT may not be enough. We need structured approaches.

### Hierarchical CoT

Break down complex problems hierarchically:

```python
class HierarchicalCoT:
    """CoT with hierarchical problem decomposition."""

    def __init__(self, model):
        self.model = model

    def solve_complex(self, problem: str, max_depth: int = 3) -> dict:
        """
        Solve complex problem with hierarchical decomposition.

        Strategy:
        1. Decompose problem into subproblems
        2. Solve each subproblem (recursively if needed)
        3. Synthesize final answer from subproblem solutions
        """
        solution = {
            "problem": problem,
            "depth": 0,
            "subproblems": [],
            "reasoning": [],
            "answer": None
        }

        # Step 1: Decompose
        subproblems = self._decompose(problem)
        solution["subproblems"] = subproblems

        # Step 2: Solve each subproblem
        subproblem_solutions = []
        for subproblem in subproblems:
            if max_depth > 1 and self._is_complex(subproblem):
                # Recursive decomposition
                subsolution = self.solve_complex(subproblem, max_depth - 1)
                subproblem_solutions.append(subsolution)
            else:
                # Solve directly with CoT
                subsolution = self._solve_with_cot(subproblem)
                subproblem_solutions.append(subsolution)

        # Step 3: Synthesize
        solution["answer"] = self._synthesize(problem, subproblem_solutions)
        solution["reasoning"] = self._build_reasoning_trace(
            subproblems,
            subproblem_solutions
        )

        return solution

    def _decompose(self, problem: str) -> list[str]:
        """Decompose problem into subproblems."""
        prompt = f"""
Problem: {problem}

Break this problem down into 2-4 simpler subproblems that, when solved,
would give us everything needed to solve the main problem.

List the subproblems:
"""
        response = self.model.generate(prompt)

        # Parse subproblems
        subproblems = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering/bullets
                subproblem = line.lstrip('0123456789.-) ')
                subproblems.append(subproblem)

        return subproblems

    def _is_complex(self, problem: str) -> bool:
        """Check if problem is complex enough to warrant decomposition."""
        # Simple heuristic: length and keywords
        keywords = ['calculate', 'analyze', 'compare', 'determine']
        return len(problem.split()) > 15 or any(k in problem.lower() for k in keywords)

    def _solve_with_cot(self, problem: str) -> dict:
        """Solve a single problem with CoT."""
        prompt = f"{problem}\n\nLet's solve this step by step:"
        response = self.model.generate(prompt)

        return {
            "problem": problem,
            "reasoning": response,
            "answer": self._extract_answer(response)
        }

    def _synthesize(self, main_problem: str, subproblem_solutions: list[dict]) -> str:
        """Synthesize final answer from subproblem solutions."""
        # Build context from subproblem answers
        context = "Subproblem solutions:\n"
        for i, sol in enumerate(subproblem_solutions, 1):
            answer = sol.get("answer", sol.get("reasoning", ""))
            context += f"{i}. {answer}\n"

        prompt = f"""
Original problem: {main_problem}

{context}

Now, using these subproblem solutions, provide the final answer to the original problem:
"""
        return self.model.generate(prompt)

    def _build_reasoning_trace(self, subproblems: list[str],
                                solutions: list[dict]) -> list[str]:
        """Build complete reasoning trace."""
        trace = []
        for subproblem, solution in zip(subproblems, solutions):
            trace.append(f"Subproblem: {subproblem}")
            reasoning = solution.get("reasoning", "")
            trace.append(f"Solution: {reasoning}")
        return trace

    def _extract_answer(self, response: str) -> str:
        """Extract answer from response."""
        if "Answer:" in response:
            return response.split("Answer:")[-1].strip()
        return response.strip().split('\n')[-1]


# Example usage
agent = HierarchicalCoT(model)

complex_problem = """
A company wants to migrate their monolithic application to microservices.
They have 50,000 daily active users, 100GB database, and $10,000 monthly budget.
Should they proceed, and if so, what should their migration plan be?
"""

solution = agent.solve_complex(complex_problem)

print("Problem:", solution['problem'])
print("\nSubproblems identified:")
for i, sp in enumerate(solution['subproblems'], 1):
    print(f"{i}. {sp}")

print("\nReasoning trace:")
for line in solution['reasoning']:
    print(line)

print("\nFinal answer:")
print(solution['answer'])
```

### Multi-Path CoT

Generate multiple reasoning paths and select the best:

```python
class MultiPathCoT:
    """Generate multiple reasoning paths and aggregate."""

    def __init__(self, model):
        self.model = model

    def solve_multipath(self, problem: str, num_paths: int = 5) -> dict:
        """
        Solve problem using multiple reasoning paths.

        Strategy:
        1. Generate N different reasoning paths
        2. Evaluate each path
        3. Select best answer (most common or highest confidence)
        """
        paths = []

        # Generate multiple paths
        for i in range(num_paths):
            path = self._generate_reasoning_path(problem, temperature=0.7)
            paths.append(path)

        # Aggregate answers
        answers = [p['answer'] for p in paths]
        best_answer = self._select_best_answer(answers, paths)

        return {
            "problem": problem,
            "paths": paths,
            "all_answers": answers,
            "best_answer": best_answer,
            "confidence": self._compute_confidence(answers, best_answer)
        }

    def _generate_reasoning_path(self, problem: str, temperature: float) -> dict:
        """Generate one reasoning path."""
        prompt = f"{problem}\n\nLet's think step by step:"

        response = self.model.generate(prompt, temperature=temperature)

        return {
            "reasoning": response,
            "answer": self._extract_answer(response)
        }

    def _select_best_answer(self, answers: list[str], paths: list[dict]) -> str:
        """
        Select best answer from multiple paths.

        Strategy: Majority voting
        """
        from collections import Counter

        # Normalize answers for comparison
        normalized = [self._normalize_answer(a) for a in answers]

        # Count occurrences
        counts = Counter(normalized)

        # Return most common
        most_common = counts.most_common(1)[0][0]

        # Return original (non-normalized) version
        for i, norm in enumerate(normalized):
            if norm == most_common:
                return answers[i]

        return answers[0]  # Fallback

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        # Remove punctuation, lowercase, strip whitespace
        import re
        answer = answer.lower().strip()
        answer = re.sub(r'[^\w\s]', '', answer)
        return answer

    def _compute_confidence(self, answers: list[str], best_answer: str) -> float:
        """Compute confidence based on agreement."""
        normalized_best = self._normalize_answer(best_answer)
        normalized_all = [self._normalize_answer(a) for a in answers]

        agreement = sum(1 for a in normalized_all if a == normalized_best)
        return agreement / len(answers)

    def _extract_answer(self, response: str) -> str:
        """Extract final answer."""
        if "Answer:" in response:
            return response.split("Answer:")[-1].strip()
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        return lines[-1] if lines else ""


# Example usage
agent = MultiPathCoT(model)

problem = "A farmer has chickens and rabbits. He counts 35 heads and 94 legs. How many chickens and how many rabbits?"

result = agent.solve_multipath(problem, num_paths=5)

print(f"Problem: {problem}\n")
print("Different reasoning paths:")
for i, path in enumerate(result['paths'], 1):
    print(f"\nPath {i}:")
    print(path['reasoning'][:200] + "...")
    print(f"Answer: {path['answer']}")

print(f"\n{'='*50}")
print(f"Best answer: {result['best_answer']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Implementing CoT in Agents

Integrating CoT into agent architectures requires careful design.

### CoT in Agent Loops

```python
class CoTAgent:
    """Agent with integrated CoT reasoning."""

    def __init__(self, model, tools, use_cot: bool = True):
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.use_cot = use_cot
        self.history = []

    def run(self, task: str, max_iterations: int = 10) -> dict:
        """
        Run agent on task with CoT reasoning.

        Returns:
            Dictionary with task result, reasoning trace, and actions taken
        """
        self.history = []
        observation = f"Task: {task}"

        for iteration in range(max_iterations):
            # Agent thinks (with CoT) and acts
            step_result = self._step(observation, task, iteration)

            self.history.append(step_result)

            # Check if done
            if step_result['done']:
                return {
                    "task": task,
                    "result": step_result['result'],
                    "history": self.history,
                    "iterations": iteration + 1
                }

            # Execute action and get observation
            action = step_result['action']
            observation = self._execute_action(action)

        return {
            "task": task,
            "result": "Max iterations reached",
            "history": self.history,
            "iterations": max_iterations
        }

    def _step(self, observation: str, task: str, iteration: int) -> dict:
        """Execute one reasoning + action step."""

        # Build prompt with history
        prompt = self._build_prompt(observation, task, iteration)

        # Generate response (with CoT if enabled)
        if self.use_cot:
            prompt += "\n\nLet me think through this step by step:"

        response = self.model.generate(prompt)

        # Parse response
        reasoning = self._extract_reasoning(response)
        action = self._extract_action(response)
        done = self._check_if_done(response)
        result = self._extract_result(response) if done else None

        return {
            "iteration": iteration,
            "observation": observation,
            "reasoning": reasoning,
            "action": action,
            "done": done,
            "result": result,
            "full_response": response
        }

    def _build_prompt(self, observation: str, task: str, iteration: int) -> str:
        """Build prompt including history and current state."""
        prompt_parts = [
            f"Task: {task}",
            f"\nAvailable tools:",
        ]

        for tool_name, tool in self.tools.items():
            prompt_parts.append(f"- {tool_name}: {tool.description}")

        prompt_parts.append(f"\nCurrent observation: {observation}")

        # Add recent history
        if self.history:
            prompt_parts.append("\nPrevious steps:")
            for step in self.history[-3:]:  # Last 3 steps
                prompt_parts.append(f"Thought: {step['reasoning']}")
                prompt_parts.append(f"Action: {step['action']}")

        prompt_parts.append("\nWhat should I do next?")

        return "\n".join(prompt_parts)

    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from response."""
        # Find the thinking part (before action)
        if "Action:" in response:
            return response.split("Action:")[0].strip()
        return response

    def _extract_action(self, response: str) -> dict:
        """Extract action from response."""
        if "Action:" not in response:
            return None

        action_line = response.split("Action:")[1].split('\n')[0].strip()

        # Parse action format
        if '(' in action_line:
            tool_name = action_line.split('(')[0].strip()
            params_str = action_line.split('(')[1].split(')')[0]

            return {
                "tool": tool_name,
                "params": params_str,
                "raw": action_line
            }

        return {"tool": action_line, "params": "", "raw": action_line}

    def _check_if_done(self, response: str) -> bool:
        """Check if agent thinks it's done."""
        done_indicators = ["FINISH", "DONE", "COMPLETE", "Final answer:"]
        response_upper = response.upper()
        return any(indicator in response_upper for indicator in done_indicators)

    def _extract_result(self, response: str) -> str:
        """Extract final result if done."""
        if "Final answer:" in response:
            return response.split("Final answer:")[-1].strip()
        if "FINISH[" in response:
            # Extract content between FINISH[ and ]
            start = response.find("FINISH[") + 7
            end = response.find("]", start)
            if end > start:
                return response[start:end]
        return response

    def _execute_action(self, action: dict) -> str:
        """Execute an action and return observation."""
        if action is None:
            return "No action specified"

        tool_name = action['tool']

        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"

        tool = self.tools[tool_name]

        try:
            result = tool.execute(action['params'])
            return f"Action result: {result}"
        except Exception as e:
            return f"Action failed: {str(e)}"


# Example tools
class Tool:
    def __init__(self, name: str, description: str, func):
        self.name = name
        self.description = description
        self.func = func

    def execute(self, params: str):
        return self.func(params)


# Define some tools
def search_tool(query):
    return f"Search results for '{query}': [Result 1, Result 2, Result 3]"

def calculate_tool(expression):
    try:
        return str(eval(expression))
    except:
        return "Calculation error"

def finish_tool(answer):
    return answer


tools = [
    Tool("search", "Search for information", search_tool),
    Tool("calculate", "Perform calculations", calculate_tool),
    Tool("finish", "Finish with final answer", finish_tool)
]

# Run agent
agent = CoTAgent(model, tools, use_cot=True)

result = agent.run("How many days are in a year that's divisible by 4 but not by 100?")

print("Task:", result['task'])
print(f"\nCompleted in {result['iterations']} iterations\n")

for step in result['history']:
    print(f"Step {step['iteration'] + 1}:")
    print(f"  Reasoning: {step['reasoning'][:100]}...")
    print(f"  Action: {step['action']}")
    print()

print(f"Final result: {result['result']}")
```

## Structuring Reasoning Chains

Well-structured CoT reasoning is more effective than free-form thinking.

### Reasoning Templates

```python
class StructuredCoT:
    """CoT with structured reasoning templates."""

    TEMPLATES = {
        "problem_solving": """
Problem: {problem}

Let me approach this systematically:

1. UNDERSTAND: What is being asked?
   {understanding}

2. IDENTIFY: What information do I have?
   {information}

3. PLAN: What steps are needed?
   {plan}

4. EXECUTE: Work through the steps
   {execution}

5. VERIFY: Check the answer
   {verification}

Answer: {answer}
""",

        "decision_making": """
Decision: {decision}

Analysis:

1. OPTIONS: What are the choices?
   {options}

2. CRITERIA: What matters most?
   {criteria}

3. EVALUATION: How does each option perform?
   {evaluation}

4. TRADEOFFS: What are the pros/cons?
   {tradeoffs}

5. RECOMMENDATION: What's the best choice?
   {recommendation}

Decision: {final_decision}
""",

        "debugging": """
Issue: {issue}

Diagnosis:

1. SYMPTOMS: What's going wrong?
   {symptoms}

2. HYPOTHESIS: What might cause this?
   {hypothesis}

3. TEST: How can we verify?
   {test}

4. ROOT CAUSE: What's the actual problem?
   {root_cause}

5. SOLUTION: How to fix it?
   {solution}

Fix: {fix}
"""
    }

    def __init__(self, model):
        self.model = model

    def solve_with_template(self, problem: str, template_name: str) -> dict:
        """Solve problem using a structured template."""

        if template_name not in self.TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}")

        template = self.TEMPLATES[template_name]

        # Generate each section
        filled = self._fill_template(template, problem, template_name)

        return {
            "problem": problem,
            "template": template_name,
            "structured_reasoning": filled,
            "answer": self._extract_answer(filled)
        }

    def _fill_template(self, template: str, problem: str, template_name: str) -> str:
        """Fill in template sections using the model."""

        # Extract sections that need filling
        import re
        placeholders = re.findall(r'\{(\w+)\}', template)

        filled_values = {"problem": problem}

        # Fill each section sequentially
        for placeholder in placeholders:
            if placeholder == "problem":
                continue

            # Build context from already-filled sections
            context = template
            for key, value in filled_values.items():
                context = context.replace(f"{{{key}}}", str(value))

            # Generate this section
            prompt = f"""
{context}

Please fill in the {placeholder.upper()} section:
"""
            response = self.model.generate(prompt)
            filled_values[placeholder] = response.strip()

        # Fill template with all values
        result = template
        for key, value in filled_values.items():
            result = result.replace(f"{{{key}}}", str(value))

        return result

    def _extract_answer(self, filled_template: str) -> str:
        """Extract the final answer from filled template."""
        lines = filled_template.split('\n')
        for line in lines:
            if line.startswith(("Answer:", "Decision:", "Fix:")):
                return line.split(":", 1)[1].strip()
        return ""


# Example usage
cot = StructuredCoT(model)

# Problem solving
problem = "A rectangle has a perimeter of 24 and a length that's twice its width. What's the area?"
result = cot.solve_with_template(problem, "problem_solving")
print(result['structured_reasoning'])

# Decision making
decision = "Should we refactor this legacy codebase or rewrite from scratch?"
result = cot.solve_with_template(decision, "decision_making")
print(result['structured_reasoning'])
```

### Verification Steps

Add verification to catch errors:

```python
class VerifiedCoT:
    """CoT with built-in verification steps."""

    def __init__(self, model):
        self.model = model

    def solve_with_verification(self, problem: str) -> dict:
        """
        Solve problem with CoT and verify the answer.

        Process:
        1. Generate initial solution with CoT
        2. Verify the solution
        3. If incorrect, regenerate
        """
        max_attempts = 3

        for attempt in range(max_attempts):
            # Generate solution
            solution = self._generate_solution(problem)

            # Verify solution
            verification = self._verify_solution(problem, solution)

            if verification['correct']:
                return {
                    "problem": problem,
                    "solution": solution,
                    "verification": verification,
                    "attempts": attempt + 1,
                    "verified": True
                }

            # If incorrect, try again with feedback
            problem = self._add_feedback(problem, solution, verification)

        return {
            "problem": problem,
            "solution": solution,
            "verification": verification,
            "attempts": max_attempts,
            "verified": False
        }

    def _generate_solution(self, problem: str) -> dict:
        """Generate solution with CoT."""
        prompt = f"""
{problem}

Let me solve this step by step:
"""
        response = self.model.generate(prompt)

        return {
            "reasoning": response,
            "answer": self._extract_answer(response)
        }

    def _verify_solution(self, problem: str, solution: dict) -> dict:
        """Verify the solution is correct."""
        prompt = f"""
Problem: {problem}

Proposed solution:
{solution['reasoning']}

Answer: {solution['answer']}

Verification: Is this solution correct? Check each step carefully.
If there are errors, identify them.
"""
        verification = self.model.generate(prompt)

        # Check if verification indicates correctness
        correct = self._check_correctness(verification)

        return {
            "correct": correct,
            "verification_text": verification,
            "errors_found": self._extract_errors(verification) if not correct else []
        }

    def _check_correctness(self, verification: str) -> bool:
        """Check if verification indicates solution is correct."""
        positive = ["correct", "right", "accurate", "valid", "yes"]
        negative = ["incorrect", "wrong", "error", "mistake", "no"]

        verification_lower = verification.lower()

        pos_count = sum(1 for word in positive if word in verification_lower)
        neg_count = sum(1 for word in negative if word in verification_lower)

        return pos_count > neg_count

    def _extract_errors(self, verification: str) -> list[str]:
        """Extract identified errors from verification."""
        errors = []
        for line in verification.split('\n'):
            if any(word in line.lower() for word in ['error', 'wrong', 'incorrect', 'mistake']):
                errors.append(line.strip())
        return errors

    def _add_feedback(self, problem: str, solution: dict, verification: dict) -> str:
        """Add verification feedback to problem for retry."""
        feedback = f"""
{problem}

Previous attempt had errors:
{chr(10).join(verification['errors_found'])}

Try again, paying attention to these issues:
"""
        return feedback

    def _extract_answer(self, response: str) -> str:
        """Extract answer from response."""
        if "Answer:" in response:
            return response.split("Answer:")[-1].strip()
        return response.strip().split('\n')[-1]


# Example usage
verifier = VerifiedCoT(model)

problem = """
A store sells apples for $0.50 each and oranges for $0.75 each.
If someone spends exactly $6.00 and buys 10 pieces of fruit,
how many apples and how many oranges did they buy?
"""

result = verifier.solve_with_verification(problem)

print(f"Problem: {problem}\n")
print(f"Solution (attempt {result['attempts']}):")
print(result['solution']['reasoning'])
print(f"\nAnswer: {result['solution']['answer']}")
print(f"\nVerified: {result['verified']}")
print(f"Verification: {result['verification']['verification_text']}")
```

## CoT Quality and Evaluation

Not all CoT reasoning is equally good. Evaluating and improving CoT quality is important.

### Evaluating CoT Quality

```python
class CoTEvaluator:
    """Evaluate the quality of CoT reasoning."""

    def __init__(self, model):
        self.model = model

    def evaluate_reasoning(self, reasoning: str, problem: str, answer: str) -> dict:
        """
        Evaluate CoT reasoning quality across multiple dimensions.

        Dimensions:
        - Correctness: Is the final answer correct?
        - Completeness: Are all steps present?
        - Clarity: Is the reasoning easy to follow?
        - Logical flow: Do steps follow logically?
        - Efficiency: Are there unnecessary steps?
        """
        scores = {}

        scores['correctness'] = self._evaluate_correctness(problem, answer)
        scores['completeness'] = self._evaluate_completeness(reasoning, problem)
        scores['clarity'] = self._evaluate_clarity(reasoning)
        scores['logical_flow'] = self._evaluate_logical_flow(reasoning)
        scores['efficiency'] = self._evaluate_efficiency(reasoning)

        scores['overall'] = sum(scores.values()) / len(scores)

        return {
            "scores": scores,
            "feedback": self._generate_feedback(scores, reasoning),
            "quality_level": self._classify_quality(scores['overall'])
        }

    def _evaluate_correctness(self, problem: str, answer: str) -> float:
        """Evaluate if answer is correct."""
        # In practice, would compare to ground truth
        # Here we use model to judge
        prompt = f"""
Problem: {problem}
Answer: {answer}

Is this answer correct? Rate from 0.0 (completely wrong) to 1.0 (completely correct).
Score:
"""
        response = self.model.generate(prompt)

        # Extract score
        try:
            import re
            score_match = re.search(r'(\d\.?\d*)', response)
            if score_match:
                return float(score_match.group(1))
        except:
            pass

        return 0.5  # Default if parsing fails

    def _evaluate_completeness(self, reasoning: str, problem: str) -> float:
        """Evaluate if all necessary steps are present."""
        # Count reasoning steps
        steps = [line for line in reasoning.split('\n')
                if line.strip() and (line[0].isdigit() or
                    line.lower().startswith(('step', 'first', 'next', 'then', 'finally')))]

        num_steps = len(steps)

        # More steps generally indicates completeness
        # But too many might indicate inefficiency
        if num_steps < 2:
            return 0.3  # Too few steps
        elif num_steps <= 5:
            return 1.0  # Good number of steps
        elif num_steps <= 10:
            return 0.8  # Many steps but okay
        else:
            return 0.6  # Too many steps

    def _evaluate_clarity(self, reasoning: str) -> float:
        """Evaluate clarity of reasoning."""
        # Heuristics for clarity:
        # - Not too long per step
        # - Uses clear language
        # - Well-structured

        lines = [l.strip() for l in reasoning.split('\n') if l.strip()]

        # Check average line length (not too long)
        avg_length = sum(len(l) for l in lines) / max(len(lines), 1)
        length_score = 1.0 if 20 < avg_length < 100 else 0.7

        # Check for clear transition words
        transition_words = ['first', 'then', 'next', 'finally', 'therefore', 'so']
        reasoning_lower = reasoning.lower()
        has_transitions = sum(1 for word in transition_words if word in reasoning_lower)
        transition_score = min(has_transitions / 3, 1.0)

        # Check for explicit numbering
        has_numbering = any(line[0].isdigit() for line in lines if line)
        numbering_score = 1.0 if has_numbering else 0.7

        return (length_score + transition_score + numbering_score) / 3

    def _evaluate_logical_flow(self, reasoning: str) -> float:
        """Evaluate logical consistency of steps."""
        # Use model to judge logical flow
        prompt = f"""
Evaluate the logical flow of this reasoning:

{reasoning}

Are the steps logically connected? Does each step follow from the previous?
Rate from 0.0 (illogical) to 1.0 (perfectly logical).
Score:
"""
        response = self.model.generate(prompt)

        try:
            import re
            score_match = re.search(r'(\d\.?\d*)', response)
            if score_match:
                return float(score_match.group(1))
        except:
            pass

        return 0.7  # Default

    def _evaluate_efficiency(self, reasoning: str) -> float:
        """Evaluate if reasoning is efficient (no redundancy)."""
        lines = [l.strip() for l in reasoning.split('\n') if l.strip()]
        num_lines = len(lines)

        # Penalize very long reasoning chains
        if num_lines < 3:
            return 0.5  # Too short
        elif num_lines <= 7:
            return 1.0  # Efficient
        elif num_lines <= 15:
            return 0.7  # Acceptable
        else:
            return 0.5  # Too verbose

    def _generate_feedback(self, scores: dict, reasoning: str) -> list[str]:
        """Generate specific feedback for improvement."""
        feedback = []

        if scores['correctness'] < 0.7:
            feedback.append("⚠️ Answer may be incorrect - verify calculations")

        if scores['completeness'] < 0.7:
            feedback.append("📝 Reasoning may be missing steps")

        if scores['clarity'] < 0.7:
            feedback.append("💡 Reasoning could be clearer - use numbered steps and transitions")

        if scores['logical_flow'] < 0.7:
            feedback.append("🔗 Steps may not flow logically - check connections")

        if scores['efficiency'] < 0.7:
            feedback.append("⚡ Reasoning could be more concise")

        if not feedback:
            feedback.append("✅ High-quality reasoning!")

        return feedback

    def _classify_quality(self, score: float) -> str:
        """Classify overall quality level."""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.5:
            return "Acceptable"
        else:
            return "Needs Improvement"


# Example usage
evaluator = CoTEvaluator(model)

reasoning = """
Let me solve this step by step:
1. The store starts with 100 items
2. They sell 25 items in the morning
3. So they have 100 - 25 = 75 items left
4. Then they restock with 40 new items
5. Now they have 75 + 40 = 115 items
6. They sell 30 more in the afternoon
7. Final count: 115 - 30 = 85 items
"""

problem = "A store starts with 100 items. They sell 25, restock 40, then sell 30. How many items remain?"
answer = "85 items"

evaluation = evaluator.evaluate_reasoning(reasoning, problem, answer)

print("Quality Scores:")
for dimension, score in evaluation['scores'].items():
    print(f"  {dimension}: {score:.2f}")

print(f"\nOverall Quality: {evaluation['quality_level']}")
print("\nFeedback:")
for item in evaluation['feedback']:
    print(f"  {item}")
```

## CoT in Production Systems

Deploying CoT in production requires addressing practical concerns.

### Latency Optimization

```python
class OptimizedCoTAgent:
    """CoT agent optimized for production use."""

    def __init__(self, model, enable_caching: bool = True):
        self.model = model
        self.enable_caching = enable_caching
        self.reasoning_cache = {}

    def solve(self, problem: str, use_cot: bool = True,
              max_reasoning_length: int = 500) -> dict:
        """
        Solve problem with production optimizations.

        Optimizations:
        - Caching of similar problems
        - Length limits on reasoning
        - Adaptive CoT (skip for simple problems)
        """
        import hashlib
        import time

        start_time = time.time()

        # Check cache
        if self.enable_caching:
            cache_key = hashlib.md5(problem.encode()).hexdigest()
            if cache_key in self.reasoning_cache:
                cached = self.reasoning_cache[cache_key]
                cached['from_cache'] = True
                cached['latency_ms'] = (time.time() - start_time) * 1000
                return cached

        # Decide whether to use CoT
        if use_cot and not self._is_simple_problem(problem):
            prompt = f"{problem}\n\nLet's think step by step (be concise):"
        else:
            prompt = problem

        # Generate with length limit
        response = self.model.generate(
            prompt,
            max_tokens=max_reasoning_length if use_cot else 100
        )

        result = {
            "problem": problem,
            "response": response,
            "used_cot": use_cot,
            "from_cache": False,
            "latency_ms": (time.time() - start_time) * 1000
        }

        # Cache result
        if self.enable_caching:
            self.reasoning_cache[cache_key] = result

        return result

    def _is_simple_problem(self, problem: str) -> bool:
        """Heuristic to detect simple problems that don't need CoT."""
        # Simple questions
        simple_patterns = [
            "what is",
            "define",
            "who is",
            "when did",
            "list"
        ]

        problem_lower = problem.lower()

        # Check for simple patterns
        if any(pattern in problem_lower for pattern in simple_patterns):
            return True

        # Check length (very short questions are likely simple)
        if len(problem.split()) < 8:
            return True

        return False

    def clear_cache(self):
        """Clear reasoning cache."""
        self.reasoning_cache = {}

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "cache_size": len(self.reasoning_cache),
            "cache_enabled": self.enable_caching
        }


# Batch processing for throughput
class BatchCoTProcessor:
    """Process multiple problems efficiently in batches."""

    def __init__(self, model):
        self.model = model

    def process_batch(self, problems: list[str], use_cot: bool = True) -> list[dict]:
        """
        Process multiple problems in a batch.

        Can use batched API calls for better throughput.
        """
        results = []

        # Prepare all prompts
        prompts = []
        for problem in problems:
            if use_cot:
                prompt = f"{problem}\n\nLet's think step by step:"
            else:
                prompt = problem
            prompts.append(prompt)

        # Batch API call (if supported)
        if hasattr(self.model, 'generate_batch'):
            responses = self.model.generate_batch(prompts)
        else:
            # Fall back to sequential
            responses = [self.model.generate(p) for p in prompts]

        # Package results
        for problem, response in zip(problems, responses):
            results.append({
                "problem": problem,
                "response": response,
                "answer": self._extract_answer(response)
            })

        return results

    def _extract_answer(self, response: str) -> str:
        """Extract answer from response."""
        if "Answer:" in response:
            return response.split("Answer:")[-1].strip()
        return response.strip().split('\n')[-1]
```

### Cost Management

```python
class CostAwareCoT:
    """CoT agent with cost tracking and management."""

    def __init__(self, model, cost_per_token: float = 0.00002):
        self.model = model
        self.cost_per_token = cost_per_token
        self.total_cost = 0.0
        self.query_count = 0

    def solve(self, problem: str, max_cost: float = 0.10) -> dict:
        """
        Solve problem with cost constraints.

        Args:
            problem: Problem to solve
            max_cost: Maximum cost allowed for this query (in dollars)

        Returns:
            Solution with cost information
        """
        # Estimate tokens needed
        estimated_tokens = self._estimate_tokens(problem)
        estimated_cost = estimated_tokens * self.cost_per_token

        if estimated_cost > max_cost:
            return {
                "problem": problem,
                "error": f"Estimated cost ${estimated_cost:.4f} exceeds max ${max_cost:.4f}",
                "cost": 0.0
            }

        # Generate solution
        prompt = f"{problem}\n\nLet's solve this concisely:"
        response = self.model.generate(prompt)

        # Calculate actual cost
        actual_tokens = self._count_tokens(prompt + response)
        actual_cost = actual_tokens * self.cost_per_token

        self.total_cost += actual_cost
        self.query_count += 1

        return {
            "problem": problem,
            "response": response,
            "tokens_used": actual_tokens,
            "cost": actual_cost,
            "cumulative_cost": self.total_cost
        }

    def _estimate_tokens(self, problem: str) -> int:
        """Estimate tokens needed for problem + solution."""
        # Rough estimate: 4 chars per token, 5x expansion for reasoning
        problem_tokens = len(problem) // 4
        estimated_solution_tokens = problem_tokens * 5
        return problem_tokens + estimated_solution_tokens

    def _count_tokens(self, text: str) -> int:
        """Count actual tokens in text."""
        # Simplified - use proper tokenizer in production
        return len(text) // 4

    def get_cost_summary(self) -> dict:
        """Get cost statistics."""
        return {
            "total_cost": self.total_cost,
            "query_count": self.query_count,
            "avg_cost_per_query": self.total_cost / max(self.query_count, 1),
            "cost_per_token": self.cost_per_token
        }

    def reset_costs(self):
        """Reset cost tracking."""
        self.total_cost = 0.0
        self.query_count = 0
```

## Summary

Chain-of-Thought reasoning is a powerful technique that makes agent thinking explicit and improves performance on complex tasks.

### Key Takeaways

1. **CoT makes reasoning explicit**: By articulating intermediate steps, agents make better decisions and are easier to debug

2. **Zero-shot CoT is remarkably effective**: Simply adding "Let's think step by step" can dramatically improve performance

3. **Few-shot CoT teaches reasoning patterns**: Providing examples shows the model **how** to reason about similar problems

4. **Not all tasks need CoT**: Simple lookups and pattern recognition work fine with direct answers

5. **Structure improves quality**: Using templates and verification steps leads to better reasoning

6. **Multiple paths increase reliability**: Generating several reasoning paths and aggregating improves accuracy

7. **Production deployment requires optimization**: Caching, batching, and cost management are essential

### When to Use CoT

**Use CoT for**:

- Multi-step problems requiring sequential reasoning
- Mathematical calculations and logical deduction
- Complex decision-making with multiple factors
- Tasks where interpretability is important
- Situations where errors must be minimized

**Skip CoT for**:

- Simple factual lookups
- Pattern recognition tasks
- Real-time systems with strict latency requirements
- Tasks where the model already performs well directly

### Best Practices

1. **Start simple**: Try zero-shot CoT before more complex approaches
2. **Provide examples**: Few-shot CoT works better for domain-specific tasks
3. **Structure reasoning**: Use templates for consistency
4. **Verify answers**: Add verification steps to catch errors
5. **Measure quality**: Evaluate reasoning chains, not just final answers
6. **Optimize for production**: Use caching, batching, and length limits
7. **Track costs**: CoT uses more tokens, so monitor expenses

## Next Steps

Now that you understand Chain-of-Thought reasoning, explore related topics:

- **[Tree-of-Thought](tree-of-thought.md)**: Explore multiple reasoning paths simultaneously through tree search
- **[Task Decomposition](task-decomposition.md)**: Break complex goals into manageable subproblems
- **[Planning Strategies](planning-strategies.md)**: Learn forward planning, backward chaining, and hierarchical planning
- **[ReAct Pattern](../agent-architectures/react.md)**: See how CoT integrates with action-taking agents
- **[Self-Consistency](self-consistency.md)**: Improve reliability through multiple reasoning paths
- **[Reasoning Under Uncertainty](uncertainty.md)**: Handle incomplete information and probabilistic reasoning

Continue your journey through agent reasoning and planning techniques! 🧠
