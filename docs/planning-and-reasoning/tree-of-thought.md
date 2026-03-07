# Tree-of-Thought and Search

## Table of Contents

- [Introduction](#introduction)
- [From Chain to Tree](#from-chain-to-tree)
- [The Tree-of-Thought Framework](#the-tree-of-thought-framework)
- [Thought Generation](#thought-generation)
- [State Evaluation](#state-evaluation)
- [Search Algorithms](#search-algorithms)
- [Breadth-First Search](#breadth-first-search)
- [Depth-First Search](#depth-first-search)
- [Best-First Search](#best-first-search)
- [Beam Search](#beam-search)
- [Monte Carlo Tree Search](#monte-carlo-tree-search)
- [Backtracking and Pruning](#backtracking-and-pruning)
- [Implementing ToT](#implementing-tot)
- [ToT for Different Task Types](#tot-for-different-task-types)
- [Evaluation and Selection](#evaluation-and-selection)
- [Optimizing ToT Performance](#optimizing-tot-performance)
- [When to Use ToT](#when-to-use-tot)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Tree-of-Thought (ToT) extends Chain-of-Thought reasoning by exploring **multiple reasoning paths simultaneously**. Instead of following a single chain of reasoning, ToT builds a tree where each branch represents a different way to approach the problem.

> "Don't put all your eggs in one basket" - Applied to reasoning

The key insight: many complex problems have multiple solution paths, and the first path you explore might not be the best one. ToT allows agents to:

- Explore different approaches in parallel
- Backtrack when a path seems unpromising
- Compare alternative solutions
- Find optimal or near-optimal answers

### The Evolution of Reasoning

```
Chain-of-Thought:          Tree-of-Thought:
Linear reasoning           Multi-path exploration

A → B → C → D             A → B → C → D
                           ↘ E → F
                          ↘ G → H → I
```

**CoT limitation**: If you make a wrong turn early, you're stuck following that path

**ToT advantage**: You can explore multiple paths and choose the best one

### When Trees Beat Chains

**CoT is sufficient when**:

- There's one clear solution path
- Early mistakes are unlikely
- The problem is straightforward

**ToT shines when**:

- Multiple valid approaches exist
- Early decisions significantly impact outcomes
- The solution space is complex
- You need the optimal solution, not just any solution

## From Chain to Tree

Understanding the progression from CoT to ToT clarifies when and why to use tree search.

### Chain-of-Thought Limitations

**Example: Planning a trip**

CoT approach:

```
1. Choose destination: Paris
2. Book flight for specific date
3. Reserve hotel near airport
4. Plan daily activities
...
```

**Problem**: What if the Paris flight is too expensive? You've already committed to that path.

### Tree-of-Thought Solution

ToT approach:

```
                    Start Planning
                    /      |      \
              Paris    Tokyo    London
             /    \    /    \    /    \
         Cheap  Exp  Cheap Exp Cheap  Exp
         Flight     Flight     Flight
           |
      (Evaluate and choose best path)
```

Now we can compare different destinations and flight options before committing.

### Key Differences

| Aspect             | Chain-of-Thought         | Tree-of-Thought              |
| ------------------ | ------------------------ | ---------------------------- |
| Paths explored     | One sequential path      | Multiple parallel paths      |
| Backtracking       | Difficult/impossible     | Natural and expected         |
| Computational cost | Lower (linear)           | Higher (exponential)         |
| Solution quality   | First solution found     | Can find optimal solution    |
| When to use        | Straightforward problems | Complex, multi-path problems |

### Conceptual Model

```python
# Chain-of-Thought (simplified)
def cot_solve(problem):
    thought = generate_next_thought(problem)
    if is_solution(thought):
        return thought
    return cot_solve(thought)  # Continue chain

# Tree-of-Thought (simplified)
def tot_solve(problem):
    candidates = generate_multiple_thoughts(problem)
    best = None
    for candidate in candidates:
        if is_solution(candidate):
            if is_better(candidate, best):
                best = candidate
        else:
            # Recursively explore this branch
            sub_solution = tot_solve(candidate)
            if is_better(sub_solution, best):
                best = sub_solution
    return best
```

## The Tree-of-Thought Framework

ToT consists of three main components: thought generation, state evaluation, and search strategy.

### ToT Components

```
┌─────────────────────────────────────────┐
│           Tree-of-Thought                │
├─────────────────────────────────────────┤
│                                          │
│  1. Thought Generation                   │
│     Generate multiple next steps         │
│                                          │
│  2. State Evaluation                     │
│     Assess quality of each state         │
│                                          │
│  3. Search Algorithm                     │
│     Decide which paths to explore        │
│                                          │
└─────────────────────────────────────────┘
```

### The ToT Process

**Step-by-step**:

1. **Start with initial state**: The problem
2. **Generate thoughts**: Create multiple possible next steps
3. **Evaluate each thought**: Score how promising each path is
4. **Select thoughts to explore**: Based on scores and search strategy
5. **Recursively expand**: Repeat for selected thoughts
6. **Track best solution**: Remember the best answer found
7. **Backtrack if needed**: Abandon unpromising paths
8. **Return final solution**: When search completes

### Visual Example: Game of 24

**Problem**: Use numbers [4, 5, 6, 10] with operations [+, -, ×, ÷] to make 24

```
                    [4, 5, 6, 10] → 24
                    /       |       \
            4+5=9         4×5=20      6-5=1
           [9,6,10]      [20,6,10]   [1,4,10]
           /     \         /    \        |
     9+6=15   9×6=54  20+6=26  20+10=30  ...
    [15,10]  [54,10]  [26,10]  [30,6]
       |        |         |       |
   15+10=25  (dead end) (close) 30-6=24 ✓
```

The tree allows exploring multiple operation sequences to find one that works.

## Thought Generation

The first step in ToT is generating multiple possible next steps.

### Generation Strategies

**1. Model-Based Generation**

Let the language model propose multiple thoughts:

```python
def generate_thoughts(state: str, num_thoughts: int = 3) -> list[str]:
    """
    Generate multiple possible next thoughts for the current state.

    Args:
        state: Current problem state
        num_thoughts: Number of alternative thoughts to generate

    Returns:
        List of possible next thoughts
    """
    prompt = f"""
Current state: {state}

Propose {num_thoughts} different possible next steps to solve this problem.
Make each step distinct and promising.

Next steps:
1.
"""

    response = model.generate(prompt, temperature=0.7, n=num_thoughts)

    # Parse multiple completions
    thoughts = []
    for completion in response:
        thought = extract_thought(completion)
        thoughts.append(thought)

    return thoughts


def extract_thought(completion: str) -> str:
    """Extract a single thought from model completion."""
    lines = [l.strip() for l in completion.split('\n') if l.strip()]
    return lines[0] if lines else ""
```

**2. Programmatic Generation**

For structured problems, generate thoughts algorithmically:

```python
def generate_arithmetic_thoughts(numbers: list[int]) -> list[dict]:
    """
    Generate all possible arithmetic operations on a list of numbers.

    For Game of 24, this generates all possible
    (number1, operation, number2) combinations.
    """
    thoughts = []
    operations = ['+', '-', '*', '/']

    # Try all pairs of numbers
    for i, num1 in enumerate(numbers):
        for j, num2 in enumerate(numbers):
            if i >= j:  # Avoid duplicates and same number
                continue

            for op in operations:
                # Calculate result
                try:
                    if op == '+':
                        result = num1 + num2
                    elif op == '-':
                        result = num1 - num2
                    elif op == '*':
                        result = num1 * num2
                    elif op == '/':
                        if num2 == 0:
                            continue
                        result = num1 / num2

                    # Create new state
                    new_numbers = [n for k, n in enumerate(numbers)
                                  if k != i and k != j]
                    new_numbers.append(result)

                    thoughts.append({
                        'operation': f"{num1} {op} {num2} = {result}",
                        'numbers': new_numbers,
                        'description': f"Apply {num1} {op} {num2}"
                    })
                except:
                    continue

    return thoughts
```

**3. Hybrid Generation**

Combine model creativity with programmatic structure:

```python
def generate_hybrid_thoughts(state: dict, domain: str) -> list[dict]:
    """
    Generate thoughts using both model and programmatic approaches.

    Args:
        state: Current problem state
        domain: Problem domain (e.g., 'math', 'planning', 'reasoning')

    Returns:
        List of candidate thoughts
    """
    thoughts = []

    # Programmatic generation (fast, exhaustive)
    if domain == 'arithmetic':
        programmatic_thoughts = generate_arithmetic_thoughts(state['numbers'])
        thoughts.extend(programmatic_thoughts)

    # Model generation (creative, diverse)
    model_thoughts = generate_thoughts(state['description'], num_thoughts=3)
    for thought_text in model_thoughts:
        thoughts.append({
            'description': thought_text,
            'source': 'model'
        })

    return thoughts
```

### Thought Diversity

Ensuring diverse thoughts improves exploration:

```python
def generate_diverse_thoughts(state: str, num_thoughts: int = 5) -> list[str]:
    """
    Generate diverse thoughts by using different prompting strategies.
    """
    thoughts = []

    # Strategy 1: Direct continuation
    prompt1 = f"{state}\n\nNext logical step:"
    thought1 = model.generate(prompt1, temperature=0.3)
    thoughts.append(thought1)

    # Strategy 2: Alternative approach
    prompt2 = f"{state}\n\nDifferent approach to try:"
    thought2 = model.generate(prompt2, temperature=0.7)
    thoughts.append(thought2)

    # Strategy 3: Creative solution
    prompt3 = f"{state}\n\nCreative solution:"
    thought3 = model.generate(prompt3, temperature=1.0)
    thoughts.append(thought3)

    # Strategy 4: Systematic method
    prompt4 = f"{state}\n\nSystematic method:"
    thought4 = model.generate(prompt4, temperature=0.3)
    thoughts.append(thought4)

    # Remove duplicates
    unique_thoughts = []
    seen = set()
    for thought in thoughts:
        normalized = thought.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            unique_thoughts.append(thought)

    return unique_thoughts[:num_thoughts]
```

## State Evaluation

After generating thoughts, we need to evaluate which ones are most promising.

### Evaluation Approaches

**1. Model-Based Evaluation**

Ask the model to assess thought quality:

```python
def evaluate_thought_quality(thought: str, goal: str) -> float:
    """
    Evaluate how promising a thought is for reaching the goal.

    Returns:
        Score from 0.0 (unpromising) to 1.0 (very promising)
    """
    prompt = f"""
Goal: {goal}

Current thought: {thought}

How promising is this thought for achieving the goal?
Rate from 0.0 (completely wrong direction) to 1.0 (definitely leads to solution).

Consider:
- Does it move toward the goal?
- Is it logically sound?
- Does it avoid dead ends?

Score:
"""

    response = model.generate(prompt, temperature=0.0)

    # Extract numerical score
    import re
    score_match = re.search(r'(\d\.?\d*)', response)
    if score_match:
        score = float(score_match.group(1))
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]

    return 0.5  # Default if parsing fails


def evaluate_thought_batch(thoughts: list[str], goal: str) -> list[float]:
    """Evaluate multiple thoughts efficiently."""
    scores = []

    # Batch evaluation prompt
    prompt_parts = [f"Goal: {goal}", "\nEvaluate each thought (0.0 to 1.0):\n"]

    for i, thought in enumerate(thoughts, 1):
        prompt_parts.append(f"{i}. {thought}")

    prompt_parts.append("\nScores:")
    prompt = "\n".join(prompt_parts)

    response = model.generate(prompt, temperature=0.0)

    # Parse scores
    import re
    score_matches = re.findall(r'(\d\.?\d*)', response)

    for match in score_matches[:len(thoughts)]:
        try:
            score = float(match)
            scores.append(max(0.0, min(1.0, score)))
        except:
            scores.append(0.5)

    # Fill in missing scores
    while len(scores) < len(thoughts):
        scores.append(0.5)

    return scores
```

**2. Heuristic-Based Evaluation**

Use domain-specific heuristics:

```python
def evaluate_arithmetic_state(numbers: list[float], target: float) -> float:
    """
    Evaluate arithmetic state for Game of 24.

    Heuristics:
    - Closer to target is better
    - Fewer numbers remaining is better
    - Avoiding very large/small numbers is better
    """
    if target in numbers:
        return 1.0  # Found solution!

    # How close are we to target?
    closest = min(numbers, key=lambda x: abs(x - target))
    distance = abs(closest - target)
    proximity_score = 1.0 / (1.0 + distance / target)

    # How many numbers remain?
    # Fewer is better (closer to solution)
    numbers_score = 1.0 - (len(numbers) - 1) / 10.0

    # Are numbers reasonable?
    # Very large or very small numbers are harder to work with
    reasonableness_score = 1.0
    for num in numbers:
        if num > 1000 or num < 0.01:
            reasonableness_score *= 0.5

    # Combine scores
    total_score = (
        proximity_score * 0.5 +
        numbers_score * 0.3 +
        reasonableness_score * 0.2
    )

    return total_score


def evaluate_planning_state(state: dict, goal: dict) -> float:
    """
    Evaluate planning state.

    Heuristics:
    - How many goals achieved?
    - How many preconditions satisfied?
    - How many steps taken (prefer fewer)?
    """
    goals_achieved = sum(1 for g in goal.values() if g in state.get('achieved', []))
    total_goals = len(goal)
    goal_score = goals_achieved / max(total_goals, 1)

    # Check preconditions for remaining goals
    preconditions_satisfied = 0
    preconditions_total = 0
    for goal_item in goal.values():
        if goal_item not in state.get('achieved', []):
            prereqs = get_prerequisites(goal_item)
            preconditions_total += len(prereqs)
            preconditions_satisfied += sum(1 for p in prereqs if p in state)

    precond_score = (preconditions_satisfied / max(preconditions_total, 1)
                     if preconditions_total > 0 else 0.5)

    # Prefer fewer steps
    steps_taken = state.get('steps', 0)
    efficiency_score = 1.0 / (1.0 + steps_taken / 10.0)

    total_score = (
        goal_score * 0.5 +
        precond_score * 0.3 +
        efficiency_score * 0.2
    )

    return total_score
```

**3. Learned Evaluation**

Train a value function to assess states:

```python
class LearnedEvaluator:
    """
    Learned value function for state evaluation.

    In practice, this would be a neural network trained on
    past problem-solving episodes.
    """

    def __init__(self):
        self.value_network = self._build_network()
        self.training_data = []

    def evaluate(self, state_features: dict) -> float:
        """
        Evaluate state using learned value function.

        Args:
            state_features: Numerical features representing state

        Returns:
            Value estimate (0.0 to 1.0)
        """
        # Convert features to input tensor
        features = self._featurize(state_features)

        # Forward pass through network
        value = self.value_network.predict(features)

        return float(value)

    def update(self, state: dict, actual_value: float):
        """
        Update value function based on observed outcomes.

        Args:
            state: State that was evaluated
            actual_value: Actual value (1.0 if led to solution, 0.0 if not)
        """
        features = self._featurize(state)
        self.training_data.append((features, actual_value))

        # Periodically retrain
        if len(self.training_data) >= 100:
            self._retrain()

    def _featurize(self, state: dict) -> list[float]:
        """Convert state to feature vector."""
        # Domain-specific feature engineering
        features = []

        # Example features for arithmetic domain
        if 'numbers' in state:
            features.extend([
                len(state['numbers']),
                max(state['numbers']) if state['numbers'] else 0,
                min(state['numbers']) if state['numbers'] else 0,
                sum(state['numbers']) if state['numbers'] else 0,
            ])

        return features

    def _build_network(self):
        """Build neural network for value prediction."""
        # Simplified - use actual ML framework in production
        class SimpleNetwork:
            def predict(self, features):
                # Placeholder
                return 0.5

        return SimpleNetwork()

    def _retrain(self):
        """Retrain value network on accumulated data."""
        # Train network using self.training_data
        pass
```

## Search Algorithms

Different search algorithms explore the tree in different ways.

### Search Strategy Comparison

| Algorithm   | Exploration       | Memory | Completeness     | Optimality         |
| ----------- | ----------------- | ------ | ---------------- | ------------------ |
| BFS         | Level by level    | High   | Yes              | Yes (uniform cost) |
| DFS         | Deep first        | Low    | Yes (finite)     | No                 |
| Best-First  | Most promising    | Medium | No               | No                 |
| Beam Search | Top-k each level  | Fixed  | No               | No                 |
| MCTS        | Guided simulation | Medium | Yes (given time) | Asymptotically     |

## Breadth-First Search

Explore all nodes at depth N before moving to depth N+1.

### BFS Implementation

```python
from collections import deque

class BFSTreeOfThought:
    """Tree-of-Thought with Breadth-First Search."""

    def __init__(self, model, max_depth: int = 5):
        self.model = model
        self.max_depth = max_depth

    def solve(self, problem: str) -> dict:
        """
        Solve problem using BFS exploration of thought tree.

        Process:
        1. Start with initial state
        2. Generate thoughts at current level
        3. Evaluate all thoughts
        4. Move to next level
        5. Repeat until solution found or max depth reached
        """
        # Queue of (state, depth, path) tuples
        queue = deque([(problem, 0, [])])

        best_solution = None
        best_score = 0.0

        nodes_explored = 0

        while queue:
            state, depth, path = queue.popleft()
            nodes_explored += 1

            # Check if we've found a solution
            if self._is_solution(state):
                score = self._evaluate_solution(state)
                if score > best_score:
                    best_score = score
                    best_solution = state
                continue

            # Max depth reached
            if depth >= self.max_depth:
                continue

            # Generate next thoughts
            thoughts = self._generate_thoughts(state)

            # Evaluate thoughts
            scores = self._evaluate_thoughts(thoughts, state)

            # Add all thoughts to queue
            for thought, score in zip(thoughts, scores):
                new_path = path + [thought]
                queue.append((thought, depth + 1, new_path))

        return {
            "problem": problem,
            "solution": best_solution,
            "score": best_score,
            "nodes_explored": nodes_explored,
            "search_type": "BFS"
        }

    def _generate_thoughts(self, state: str) -> list[str]:
        """Generate possible next thoughts."""
        prompt = f"""
Current state: {state}

Generate 3 different next steps:
1.
"""
        responses = self.model.generate(prompt, n=3, temperature=0.7)
        return [r.strip() for r in responses]

    def _evaluate_thoughts(self, thoughts: list[str], state: str) -> list[float]:
        """Evaluate promise of each thought."""
        return [0.5 for _ in thoughts]  # Simplified

    def _is_solution(self, state: str) -> bool:
        """Check if state is a solution."""
        return "solution:" in state.lower() or "answer:" in state.lower()

    def _evaluate_solution(self, state: str) -> float:
        """Evaluate quality of solution."""
        return 1.0  # Simplified
```

### BFS Characteristics

**Advantages**:

- Finds shortest path to solution
- Explores all possibilities at each level
- Guaranteed to find solution if one exists

**Disadvantages**:

- High memory usage (stores all nodes at current level)
- Can be slow if solution is deep in tree
- Explores many irrelevant branches

**When to use**:

- When solution is likely at shallow depth
- When you need the optimal solution
- When you have sufficient memory

## Depth-First Search

Explore one branch fully before backtracking to try others.

### DFS Implementation

```python
class DFSTreeOfThought:
    """Tree-of-Thought with Depth-First Search."""

    def __init__(self, model, max_depth: int = 10):
        self.model = model
        self.max_depth = max_depth
        self.nodes_explored = 0

    def solve(self, problem: str) -> dict:
        """
        Solve problem using DFS exploration.

        Uses recursion to explore each branch fully before backtracking.
        """
        self.nodes_explored = 0
        solution = self._dfs_recursive(problem, depth=0, path=[])

        return {
            "problem": problem,
            "solution": solution,
            "nodes_explored": self.nodes_explored,
            "search_type": "DFS"
        }

    def _dfs_recursive(self, state: str, depth: int, path: list) -> str:
        """
        Recursive DFS exploration.

        Returns best solution found in this subtree.
        """
        self.nodes_explored += 1

        # Check if solution
        if self._is_solution(state):
            return state

        # Max depth reached
        if depth >= self.max_depth:
            return None

        # Generate thoughts
        thoughts = self._generate_thoughts(state)

        # Evaluate thoughts
        scores = self._evaluate_thoughts(thoughts, state)

        # Sort by score (explore most promising first)
        sorted_thoughts = sorted(zip(thoughts, scores),
                                key=lambda x: x[1],
                                reverse=True)

        # Explore each thought (depth-first)
        for thought, score in sorted_thoughts:
            new_path = path + [thought]

            # Recursively explore this branch
            solution = self._dfs_recursive(thought, depth + 1, new_path)

            if solution is not None:
                return solution  # Found solution, stop searching

        return None  # No solution in this subtree

    def _generate_thoughts(self, state: str) -> list[str]:
        """Generate next thoughts."""
        prompt = f"{state}\n\nNext step:"
        responses = self.model.generate(prompt, n=3, temperature=0.7)
        return responses

    def _evaluate_thoughts(self, thoughts: list[str], state: str) -> list[float]:
        """Evaluate thoughts."""
        return [0.5 for _ in thoughts]  # Simplified

    def _is_solution(self, state: str) -> bool:
        """Check if state is solution."""
        return "solution:" in state.lower()
```

### DFS Characteristics

**Advantages**:

- Low memory usage (only stores current path)
- Fast if solution is deep in tree
- Easy to implement with recursion

**Disadvantages**:

- Can get stuck exploring dead-end branches
- No guarantee of finding shortest path
- May explore very deep before backtracking

**When to use**:

- When memory is limited
- When solution is likely deep in tree
- When you need any solution quickly

## Best-First Search

Always expand the most promising node based on evaluation.

### Best-First Implementation

```python
import heapq

class BestFirstTreeOfThought:
    """Tree-of-Thought with Best-First Search."""

    def __init__(self, model, max_nodes: int = 100):
        self.model = model
        self.max_nodes = max_nodes

    def solve(self, problem: str) -> dict:
        """
        Solve problem using best-first search.

        Always expands the most promising node according to
        evaluation function.
        """
        # Priority queue: (negative_score, state, depth, path)
        # Negative because heapq is min-heap
        pq = [(-1.0, problem, 0, [])]

        best_solution = None
        best_score = 0.0
        nodes_explored = 0

        while pq and nodes_explored < self.max_nodes:
            neg_score, state, depth, path = heapq.heappop(pq)
            score = -neg_score
            nodes_explored += 1

            # Check if solution
            if self._is_solution(state):
                if score > best_score:
                    best_score = score
                    best_solution = state
                continue

            # Generate and evaluate thoughts
            thoughts = self._generate_thoughts(state)
            scores = self._evaluate_thoughts(thoughts, state)

            # Add to priority queue
            for thought, thought_score in zip(thoughts, scores):
                new_path = path + [thought]
                heapq.heappush(pq,
                              (-thought_score, thought, depth + 1, new_path))

        return {
            "problem": problem,
            "solution": best_solution,
            "score": best_score,
            "nodes_explored": nodes_explored,
            "search_type": "Best-First"
        }

    def _generate_thoughts(self, state: str) -> list[str]:
        """Generate thoughts."""
        prompt = f"{state}\n\nPossible next steps:\n"
        responses = self.model.generate(prompt, n=3, temperature=0.7)
        return responses

    def _evaluate_thoughts(self, thoughts: list[str], state: str) -> list[float]:
        """Evaluate each thought's promise."""
        scores = []
        for thought in thoughts:
            prompt = f"""
State: {state}
Next thought: {thought}

How promising is this (0.0 to 1.0)?
"""
            response = self.model.generate(prompt, temperature=0.0)

            import re
            match = re.search(r'(\d\.?\d*)', response)
            if match:
                scores.append(float(match.group(1)))
            else:
                scores.append(0.5)

        return scores

    def _is_solution(self, state: str) -> bool:
        """Check if solution."""
        return "answer:" in state.lower()
```

## Beam Search

Keep only the top-k most promising nodes at each level.

### Beam Search Implementation

```python
class BeamSearchToT:
    """Tree-of-Thought with Beam Search."""

    def __init__(self, model, beam_width: int = 3, max_depth: int = 5):
        self.model = model
        self.beam_width = beam_width
        self.max_depth = max_depth

    def solve(self, problem: str) -> dict:
        """
        Solve problem using beam search.

        Maintains top-k candidates at each level, pruning less promising paths.
        """
        # Current beam: list of (state, score, path) tuples
        beam = [(problem, 1.0, [])]

        best_solution = None
        best_score = 0.0

        for depth in range(self.max_depth):
            new_beam = []

            # Expand each candidate in current beam
            for state, score, path in beam:
                # Check if solution
                if self._is_solution(state):
                    if score > best_score:
                        best_score = score
                        best_solution = state
                    continue

                # Generate thoughts
                thoughts = self._generate_thoughts(state)
                thought_scores = self._evaluate_thoughts(thoughts, state)

                # Add to new beam
                for thought, thought_score in zip(thoughts, thought_scores):
                    # Combine parent score with thought score
                    combined_score = score * 0.5 + thought_score * 0.5
                    new_path = path + [(state, thought)]
                    new_beam.append((thought, combined_score, new_path))

            # Keep only top-k candidates
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:self.beam_width]

            # If beam is empty, no solution found
            if not beam:
                break

        # Check final beam for solutions
        for state, score, path in beam:
            if self._is_solution(state):
                if score > best_score:
                    best_score = score
                    best_solution = state

        return {
            "problem": problem,
            "solution": best_solution,
            "score": best_score,
            "beam_width": self.beam_width,
            "search_type": "Beam Search"
        }

    def _generate_thoughts(self, state: str) -> list[str]:
        """Generate next thoughts."""
        prompt = f"{state}\n\nNext steps (generate 3):\n"
        responses = self.model.generate(prompt, n=3, temperature=0.7)
        return responses

    def _evaluate_thoughts(self, thoughts: list[str], state: str) -> list[float]:
        """Evaluate thoughts."""
        # Simplified evaluation
        return [0.5 + i*0.1 for i in range(len(thoughts))]

    def _is_solution(self, state: str) -> bool:
        """Check if solution."""
        return "final answer:" in state.lower()
```

### Beam Search Trade-offs

**Beam Width Impact**:

```
Width = 1: Greedy search (fast but may miss good solutions)
Width = 3: Balanced (explores some alternatives)
Width = 10: More thorough (but more expensive)
Width = ∞: Becomes BFS (exhaustive but costly)
```

**When to use different widths**:

- **Narrow beam (1-3)**: Real-time systems, simple problems
- **Medium beam (3-7)**: General purpose, balanced exploration
- **Wide beam (10+)**: Critical decisions, complex problems

## Monte Carlo Tree Search

MCTS combines tree search with random simulation to balance exploration and exploitation.

### MCTS Components

**Four phases**:

1. **Selection**: Navigate tree using policy (e.g., UCB1)
2. **Expansion**: Add new node to tree
3. **Simulation**: Random playout from new node
4. **Backpropagation**: Update statistics along path

### MCTS Implementation

```python
import math
import random

class MCTSNode:
    """Node in MCTS tree."""

    def __init__(self, state: str, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_thoughts = None

    def ucb1_score(self, exploration_weight: float = 1.41) -> float:
        """
        Compute UCB1 score for this node.

        Balances exploitation (value) and exploration (visits).
        """
        if self.visits == 0:
            return float('inf')

        exploitation = self.value / self.visits
        exploration = exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

        return exploitation + exploration

    def select_child(self) -> 'MCTSNode':
        """Select child with highest UCB1 score."""
        return max(self.children, key=lambda c: c.ucb1_score())

    def add_child(self, thought: str) -> 'MCTSNode':
        """Add a child node."""
        child = MCTSNode(thought, parent=self)
        self.children.append(child)
        return child

    def update(self, value: float):
        """Update node statistics."""
        self.visits += 1
        self.value += value


class MCTSTreeOfThought:
    """Tree-of-Thought with Monte Carlo Tree Search."""

    def __init__(self, model, num_simulations: int = 100):
        self.model = model
        self.num_simulations = num_simulations

    def solve(self, problem: str) -> dict:
        """
        Solve problem using MCTS.

        Args:
            problem: Problem to solve

        Returns:
            Best solution found
        """
        root = MCTSNode(problem)

        for _ in range(self.num_simulations):
            # 1. Selection: Navigate to leaf node
            node = self._select(root)

            # 2. Expansion: Add child node
            if not self._is_terminal(node.state):
                node = self._expand(node)

            # 3. Simulation: Random playout
            value = self._simulate(node.state)

            # 4. Backpropagation: Update path
            self._backpropagate(node, value)

        # Return best solution found
        best_child = max(root.children,
                        key=lambda c: c.visits if c.children else 0,
                        default=None)

        return {
            "problem": problem,
            "solution": best_child.state if best_child else None,
            "simulations": self.num_simulations,
            "search_type": "MCTS"
        }

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase: Navigate tree using UCB1.

        Returns leaf node to expand.
        """
        while node.children and not self._is_terminal(node.state):
            node = node.select_child()
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expansion phase: Add new child to tree.

        Returns newly added child.
        """
        if node.untried_thoughts is None:
            node.untried_thoughts = self._generate_thoughts(node.state)

        if node.untried_thoughts:
            thought = node.untried_thoughts.pop()
            return node.add_child(thought)

        return node

    def _simulate(self, state: str) -> float:
        """
        Simulation phase: Random playout to terminal state.

        Returns value estimate (0.0 to 1.0).
        """
        current_state = state
        depth = 0
        max_depth = 5

        while not self._is_terminal(current_state) and depth < max_depth:
            # Generate thoughts
            thoughts = self._generate_thoughts(current_state)

            # Random selection
            if thoughts:
                current_state = random.choice(thoughts)
            else:
                break

            depth += 1

        # Evaluate terminal state
        return self._evaluate_state(current_state)

    def _backpropagate(self, node: MCTSNode, value: float):
        """
        Backpropagation phase: Update node statistics along path.
        """
        while node is not None:
            node.update(value)
            node = node.parent

    def _generate_thoughts(self, state: str) -> list[str]:
        """Generate possible next thoughts."""
        prompt = f"{state}\n\nNext step:"
        responses = self.model.generate(prompt, n=3, temperature=0.8)
        return responses

    def _is_terminal(self, state: str) -> bool:
        """Check if state is terminal."""
        return "answer:" in state.lower() or "solution:" in state.lower()

    def _evaluate_state(self, state: str) -> float:
        """Evaluate state value."""
        if self._is_terminal(state):
            return 1.0
        return 0.5  # Neutral value for non-terminal states
```

### MCTS Advantages

**Key benefits**:

1. **Balances exploration and exploitation**: UCB1 formula
2. **Anytime algorithm**: Can return best result so far if interrupted
3. **Asymptotically optimal**: Converges to optimal policy
4. **Handles large branching factors**: Doesn't explore all branches
5. **Works without domain knowledge**: Random playouts require no heuristics

**When to use**:

- Complex problems with large search spaces
- When good heuristics are unavailable
- When you have time for many simulations
- For game playing and sequential decision making

## Backtracking and Pruning

Efficiently abandoning unpromising paths is crucial for ToT performance.

### Pruning Strategies

```python
class PrunedToT:
    """ToT with aggressive pruning of unpromising branches."""

    def __init__(self, model, prune_threshold: float = 0.3):
        self.model = model
        self.prune_threshold = prune_threshold
        self.pruned_count = 0

    def solve(self, problem: str, beam_width: int = 3) -> dict:
        """Solve with pruning of low-scoring thoughts."""
        beam = [(problem, 1.0, [])]

        best_solution = None
        best_score = 0.0

        for depth in range(5):
            new_beam = []

            for state, score, path in beam:
                if self._is_solution(state):
                    if score > best_score:
                        best_score = score
                        best_solution = state
                    continue

                # Generate and evaluate
                thoughts = self._generate_thoughts(state)
                scores = self._evaluate_thoughts(thoughts, state)

                # Prune low-scoring thoughts
                for thought, thought_score in zip(thoughts, scores):
                    if thought_score < self.prune_threshold:
                        self.pruned_count += 1
                        continue  # Skip this thought

                    combined_score = score * 0.5 + thought_score * 0.5
                    new_beam.append((thought, combined_score, path + [thought]))

            # Keep top-k
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_width]

        return {
            "solution": best_solution,
            "score": best_score,
            "pruned_nodes": self.pruned_count
        }

    def _generate_thoughts(self, state: str) -> list[str]:
        """Generate thoughts."""
        return ["thought1", "thought2", "thought3"]  # Placeholder

    def _evaluate_thoughts(self, thoughts: list[str], state: str) -> list[float]:
        """Evaluate thoughts."""
        return [0.7, 0.4, 0.2]  # Example scores

    def _is_solution(self, state: str) -> bool:
        """Check if solution."""
        return "answer:" in state.lower()
```

### Early Stopping

```python
def solve_with_early_stopping(problem: str,
                               max_nodes: int = 100,
                               solution_threshold: float = 0.95) -> dict:
    """
    Solve with early stopping when good-enough solution is found.

    Args:
        problem: Problem to solve
        max_nodes: Maximum nodes to explore
        solution_threshold: Stop if solution score exceeds this

    Returns:
        Solution (possibly suboptimal but good enough)
    """
    nodes_explored = 0
    beam = [(problem, 1.0, [])]

    while nodes_explored < max_nodes:
        # ... exploration logic ...

        # Check if we found good enough solution
        if best_score >= solution_threshold:
            print(f"Early stopping: found solution with score {best_score:.2f}")
            break

        nodes_explored += 1

    return {"solution": best_solution, "score": best_score}
```

## Implementing ToT

A complete, practical ToT implementation.

### Full ToT System

```python
class TreeOfThought:
    """
    Complete Tree-of-Thought system with multiple search strategies.
    """

    def __init__(self, model, search_algorithm: str = "beam"):
        self.model = model
        self.search_algorithm = search_algorithm
        self.history = []

    def solve(self,
              problem: str,
              num_thoughts: int = 3,
              max_depth: int = 5,
              beam_width: int = 3) -> dict:
        """
        Solve problem using Tree-of-Thought.

        Args:
            problem: Problem description
            num_thoughts: Number of alternative thoughts per state
            max_depth: Maximum tree depth
            beam_width: Beam width for beam search

        Returns:
            Dictionary with solution and search statistics
        """
        if self.search_algorithm == "bfs":
            return self._solve_bfs(problem, num_thoughts, max_depth)
        elif self.search_algorithm == "dfs":
            return self._solve_dfs(problem, num_thoughts, max_depth)
        elif self.search_algorithm == "beam":
            return self._solve_beam(problem, num_thoughts, max_depth, beam_width)
        elif self.search_algorithm == "mcts":
            return self._solve_mcts(problem, num_simulations=50)
        else:
            raise ValueError(f"Unknown search algorithm: {self.search_algorithm}")

    def _solve_beam(self, problem: str, num_thoughts: int,
                    max_depth: int, beam_width: int) -> dict:
        """Beam search implementation."""
        beam = [(problem, 1.0, [], 0)]  # (state, score, path, depth)

        best_solution = None
        best_score = 0.0
        nodes_explored = 0

        iteration = 0
        while beam and iteration < max_depth:
            new_beam = []

            for state, parent_score, path, depth in beam:
                nodes_explored += 1

                # Check for solution
                is_solution, solution_quality = self._check_solution(state, problem)
                if is_solution:
                    score = parent_score * solution_quality
                    if score > best_score:
                        best_score = score
                        best_solution = {
                            "state": state,
                            "path": path,
                            "depth": depth
                        }
                    continue

                # Generate thoughts
                thoughts = self._generate_thoughts(state, num_thoughts)

                # Evaluate thoughts
                scores = self._evaluate_thoughts(thoughts, state, problem)

                # Add to new beam
                for thought, score in zip(thoughts, scores):
                    combined_score = parent_score * 0.7 + score * 0.3
                    new_path = path + [{"state": state, "thought": thought}]
                    new_beam.append((thought, combined_score, new_path, depth + 1))

            # Sort and keep top-k
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_width]

            iteration += 1

        return {
            "problem": problem,
            "solution": best_solution,
            "best_score": best_score,
            "nodes_explored": nodes_explored,
            "search_algorithm": "beam",
            "beam_width": beam_width
        }

    def _solve_dfs(self, problem: str, num_thoughts: int, max_depth: int) -> dict:
        """DFS implementation."""
        self.nodes_explored = 0
        best_solution = self._dfs_recursive(
            problem, problem, depth=0, path=[], max_depth=max_depth, num_thoughts=num_thoughts
        )

        return {
            "problem": problem,
            "solution": best_solution,
            "nodes_explored": self.nodes_explored,
            "search_algorithm": "dfs"
        }

    def _dfs_recursive(self, initial_problem: str, state: str,
                       depth: int, path: list,
                       max_depth: int, num_thoughts: int) -> dict:
        """Recursive DFS helper."""
        self.nodes_explored += 1

        # Check solution
        is_solution, quality = self._check_solution(state, initial_problem)
        if is_solution:
            return {"state": state, "path": path, "quality": quality}

        # Max depth
        if depth >= max_depth:
            return None

        # Generate and evaluate thoughts
        thoughts = self._generate_thoughts(state, num_thoughts)
        scores = self._evaluate_thoughts(thoughts, state, initial_problem)

        # Sort by score
        sorted_thoughts = sorted(zip(thoughts, scores),
                                key=lambda x: x[1],
                                reverse=True)

        # Try each thought
        for thought, score in sorted_thoughts:
            new_path = path + [{"state": state, "thought": thought}]
            result = self._dfs_recursive(
                initial_problem, thought, depth + 1, new_path, max_depth, num_thoughts
            )
            if result:
                return result

        return None

    def _solve_bfs(self, problem: str, num_thoughts: int, max_depth: int) -> dict:
        """BFS implementation."""
        from collections import deque

        queue = deque([(problem, 0, [])])
        best_solution = None
        best_score = 0.0
        nodes_explored = 0

        while queue:
            state, depth, path = queue.popleft()
            nodes_explored += 1

            # Check solution
            is_solution, quality = self._check_solution(state, problem)
            if is_solution and quality > best_score:
                best_score = quality
                best_solution = {"state": state, "path": path, "quality": quality}
                continue

            # Max depth
            if depth >= max_depth:
                continue

            # Generate thoughts
            thoughts = self._generate_thoughts(state, num_thoughts)

            # Add to queue
            for thought in thoughts:
                new_path = path + [{"state": state, "thought": thought}]
                queue.append((thought, depth + 1, new_path))

        return {
            "problem": problem,
            "solution": best_solution,
            "best_score": best_score,
            "nodes_explored": nodes_explored,
            "search_algorithm": "bfs"
        }

    def _solve_mcts(self, problem: str, num_simulations: int) -> dict:
        """MCTS implementation (simplified)."""
        # Would use full MCTS logic here
        return {
            "problem": problem,
            "solution": None,
            "search_algorithm": "mcts"
        }

    def _generate_thoughts(self, state: str, num_thoughts: int) -> list[str]:
        """Generate multiple alternative next thoughts."""
        prompt = f"""
Current state: {state}

Generate {num_thoughts} different possible next steps.
Each should be a distinct approach.

Next steps:
"""

        responses = []
        for i in range(num_thoughts):
            response = self.model.generate(prompt, temperature=0.7 + i*0.1)
            responses.append(response.strip())

        return responses

    def _evaluate_thoughts(self, thoughts: list[str],
                          state: str, problem: str) -> list[float]:
        """Evaluate how promising each thought is."""
        scores = []

        for thought in thoughts:
            prompt = f"""
Original problem: {problem}
Current state: {state}
Proposed next step: {thought}

Rate how promising this step is (0.0 to 1.0):
- Will it help solve the problem?
- Is it a logical next step?
- Does it avoid dead ends?

Score:
"""

            response = self.model.generate(prompt, temperature=0.0)

            # Extract score
            import re
            match = re.search(r'(\d\.?\d*)', response)
            if match:
                score = float(match.group(1))
                scores.append(max(0.0, min(1.0, score)))
            else:
                scores.append(0.5)

        return scores

    def _check_solution(self, state: str, problem: str) -> tuple[bool, float]:
        """
        Check if state is a solution and evaluate quality.

        Returns:
            (is_solution, quality_score)
        """
        # Simple heuristic
        if any(marker in state.lower()
               for marker in ["answer:", "solution:", "final answer:"]):
            # Verify solution
            prompt = f"""
Problem: {problem}
Proposed solution: {state}

Is this a correct and complete solution?
Rate quality from 0.0 (wrong) to 1.0 (perfect).

Quality:
"""
            response = self.model.generate(prompt, temperature=0.0)

            import re
            match = re.search(r'(\d\.?\d*)', response)
            if match:
                quality = float(match.group(1))
                return (quality > 0.5, quality)

        return (False, 0.0)


# Example usage
tot = TreeOfThought(model, search_algorithm="beam")

problem = """
Plan a 3-day trip to San Francisco with these constraints:
- Budget: $1000
- Must visit Golden Gate Bridge
- Need both cultural and outdoor activities
- Prefer public transportation
"""

result = tot.solve(problem, num_thoughts=3, max_depth=4, beam_width=3)

print(f"Problem: {problem}")
print(f"\nSolution found: {result['solution']}")
print(f"Nodes explored: {result['nodes_explored']}")
print(f"Search algorithm: {result['search_algorithm']}")
```

## ToT for Different Task Types

ToT works differently for different problem domains.

### Mathematical Reasoning

```python
def solve_math_with_tot(problem: str) -> dict:
    """
    Solve math problem using ToT.

    Key insight: Generate different solution approaches.
    """
    tot = TreeOfThought(model, search_algorithm="beam")

    # Customize thought generation for math
    def generate_math_thoughts(state: str) -> list[str]:
        approaches = [
            f"{state}\n\nAlgebraic approach:",
            f"{state}\n\nGeometric approach:",
            f"{state}\n\nNumerical approach:",
        ]

        thoughts = []
        for approach_prompt in approaches:
            thought = model.generate(approach_prompt)
            thoughts.append(thought)

        return thoughts

    result = tot.solve(problem, num_thoughts=3, beam_width=2)
    return result
```

### Creative Writing

```python
def write_story_with_tot(prompt: str) -> dict:
    """
    Generate creative story using ToT to explore plot branches.
    """
    tot = TreeOfThought(model, search_algorithm="beam")

    # Explore different plot directions
    result = tot.solve(
        problem=prompt,
        num_thoughts=4,  # More creative alternatives
        max_depth=3,     # Story develops over few steps
        beam_width=5     # Keep more options for creativity
    )

    return result
```

### Strategic Planning

```python
def plan_strategy_with_tot(goal: str, constraints: dict) -> dict:
    """
    Create strategic plan using ToT to explore options.
    """
    problem = f"""
Goal: {goal}
Constraints: {constraints}

Create a strategic plan.
"""

    tot = TreeOfThought(model, search_algorithm="beam")

    result = tot.solve(
        problem=problem,
        num_thoughts=3,
        max_depth=5,  # Multi-step planning
        beam_width=3
    )

    return result
```

## When to Use ToT

Understanding when ToT provides value over simpler approaches is crucial.

### Decision Framework

```python
def should_use_tot(problem: dict) -> bool:
    """
    Decide whether to use ToT based on problem characteristics.

    Args:
        problem: Dict with keys:
            - complexity: int (1-10)
            - branching_factor: int (how many alternatives exist)
            - correctness_critical: bool
            - time_budget: float (seconds available)
            - solution_quality_matters: bool

    Returns:
        True if ToT is recommended
    """
    # Extract characteristics
    complexity = problem.get('complexity', 5)
    branching = problem.get('branching_factor', 1)
    critical = problem.get('correctness_critical', False)
    time_budget = problem.get('time_budget', 60)
    quality_matters = problem.get('solution_quality_matters', True)

    # Decision logic
    if branching == 1:
        return False  # No alternatives to explore

    if complexity < 3:
        return False  # Simple problem, CoT sufficient

    if time_budget < 5:
        return False  # Not enough time for tree search

    if critical or quality_matters:
        return True  # Need thorough exploration

    if branching >= 3 and complexity >= 5:
        return True  # Complex with many options

    return False


# Example usage
problems = [
    {
        "description": "What is 2+2?",
        "complexity": 1,
        "branching_factor": 1,
        "correctness_critical": False,
        "time_budget": 1
    },
    {
        "description": "Design microservices architecture",
        "complexity": 8,
        "branching_factor": 5,
        "correctness_critical": True,
        "time_budget": 120
    },
    {
        "description": "Find optimal route in complex city",
        "complexity": 7,
        "branching_factor": 10,
        "correctness_critical": False,
        "time_budget": 30
    }
]

for prob in problems:
    use_tot = should_use_tot(prob)
    print(f"{prob['description']}: {'ToT' if use_tot else 'CoT'}")
```

## Summary

Tree-of-Thought extends Chain-of-Thought reasoning by exploring multiple paths simultaneously, enabling more thorough problem-solving.

### Key Takeaways

1. **ToT explores multiple reasoning paths**: Unlike CoT's single chain, ToT builds a tree of possibilities

2. **Different search algorithms serve different needs**:
   - BFS: Optimal solutions, exhaustive search
   - DFS: Memory-efficient, depth-first exploration
   - Beam Search: Balanced, practical for production
   - MCTS: Sophisticated, handles large spaces

3. **State evaluation is critical**: Good evaluation functions guide search to promising paths

4. **Pruning improves efficiency**: Aggressively cutting unpromising branches saves computation

5. **ToT has overhead**: More complex and costly than CoT; use when benefits justify costs

6. **Problem characteristics matter**: ToT shines for problems with multiple solution paths

### When to Use ToT

**Use ToT when**:

- Multiple valid approaches exist
- Solution quality matters significantly
- Early decisions have big impact
- You can afford the computational cost
- Correctness is critical

**Use CoT instead when**:

- Solution path is clear
- Speed matters more than optimality
- Problem is straightforward
- Resources are limited

### Best Practices

1. **Start with narrow beams**: Use beam width 2-3 initially, expand if needed
2. **Implement pruning**: Don't waste time on obviously bad paths
3. **Use heuristic evaluation**: Good evaluation functions dramatically improve efficiency
4. **Set depth limits**: Prevent infinite exploration
5. **Monitor performance**: Track nodes explored and time spent
6. **Adapt search strategy**: Match algorithm to problem type
7. **Verify solutions**: Always check that final answer is correct

## Next Steps

Continue exploring planning and reasoning techniques:

- **[Chain-of-Thought](chain-of-thought.md)**: Review the foundation that ToT builds upon
- **[Task Decomposition](task-decomposition.md)**: Break complex problems into subproblems
- **[Planning Strategies](planning-strategies.md)**: Learn hierarchical and forward/backward planning
- **[Self-Consistency](self-consistency.md)**: Use multiple paths to improve reliability
- **[Reasoning Under Uncertainty](uncertainty.md)**: Handle incomplete information
- **[ReAct](../agent-architectures/react.md)**: Combine reasoning with actions

Master these techniques to build agents that reason deeply about complex problems! 🌲
