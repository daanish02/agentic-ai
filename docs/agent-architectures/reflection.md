# Reflection and Self-Critique

## Table of Contents

- [Introduction](#introduction)
- [The Reflection Pattern](#the-reflection-pattern)
- [Self-Critique Mechanisms](#self-critique-mechanisms)
- [Implementing Reflection](#implementing-reflection)
- [Iterative Refinement](#iterative-refinement)
- [Quality Evaluation](#quality-evaluation)
- [Learning from Mistakes](#learning-from-mistakes)
- [Meta-Cognition](#meta-cognition)
- [Reflection Prompting](#reflection-prompting)
- [Multi-Level Reflection](#multi-level-reflection)
- [Reflection in Multi-Agent Systems](#reflection-in-multi-agent-systems)
- [Common Reflection Patterns](#common-reflection-patterns)
- [Avoiding Reflection Loops](#avoiding-reflection-loops)
- [When to Use Reflection](#when-to-use-reflection)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Reflection is the agent's ability to **examine its own work, identify flaws, and improve**. Rather than producing output and moving on, reflective agents pause to evaluate, critique, and refine.

This mirrors how skilled professionals work: write code, review it, refactor; write essay, revise it, polish; solve problem, check answer, verify logic.

> "The unexamined output is not worth delivering."

Reflection transforms agents from one-shot generators into iterative improvers, dramatically increasing output quality.

### Reflection vs Direct Generation

**Without reflection**:
```
Task → Generate → Output
```

**With reflection**:
```
Task → Generate → Reflect → Refine → Output
                     ↑         ↓
                     └─────────┘ (repeat)
```

### Why Reflection Matters

1. **Quality improvement**: Catch and fix errors
2. **Learning**: Agents improve over time
3. **Robustness**: Reduce mistakes and edge cases
4. **Explainability**: Understanding why choices were made
5. **Adaptability**: Adjust based on feedback

### Example: Code Generation

**Without reflection**:
```python
def calculate_average(numbers):
    return sum(numbers) / len(numbers)
```

**With reflection**:
```
Initial: return sum(numbers) / len(numbers)

Reflection: What if numbers is empty? Division by zero!

Refined:
def calculate_average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

Second reflection: Should empty list return 0 or None?
Edge case: What about non-numeric values?

Final:
def calculate_average(numbers):
    if not numbers:
        return None
    if not all(isinstance(n, (int, float)) for n in numbers):
        raise TypeError("All elements must be numeric")
    return sum(numbers) / len(numbers)
```

## The Reflection Pattern

Core pattern: generate, evaluate, refine.

### Basic Reflection Loop

```python
def reflection_loop(task, max_iterations=3):
    """
    Basic reflection loop implementation.
    """
    output = generate_initial_output(task)
    
    for iteration in range(max_iterations):
        # Reflect on current output
        critique = reflect(output, task)
        
        # Check if good enough
        if critique["quality"] >= QUALITY_THRESHOLD:
            break
        
        # Refine based on critique
        output = refine(output, critique)
    
    return output


def generate_initial_output(task):
    """Generate initial attempt."""
    return llm.generate(f"Solve: {task}")


def reflect(output, task):
    """Evaluate output and identify issues."""
    prompt = f"""
Task: {task}
Output: {output}

Critically evaluate this output:
1. Does it fully address the task?
2. Are there any errors or issues?
3. What could be improved?
4. Rate quality 0-10.

Analysis:"""
    
    critique = llm.generate(prompt)
    return parse_critique(critique)


def refine(output, critique):
    """Improve output based on critique."""
    prompt = f"""
Original output: {output}

Issues identified:
{critique['issues']}

Improve the output addressing these issues:"""
    
    return llm.generate(prompt)
```

### Reflection Architecture

```
┌──────────────┐
│   Generate   │
│    Output    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Reflect    │
│  (Critique)  │
└──────┬───────┘
       │
       ▼
    Good Enough? ──Yes──► Return Output
       │
       No
       │
       ▼
┌──────────────┐
│    Refine    │
│  (Improve)   │
└──────┬───────┘
       │
       └─────► (Repeat)
```

## Self-Critique Mechanisms

Different ways agents can critique their own work.

### Checklist-Based Critique

```python
def checklist_critique(output, task):
    """
    Evaluate output against predefined checklist.
    """
    checklist = {
        "completeness": "Does output address all parts of task?",
        "correctness": "Is the output factually correct?",
        "clarity": "Is the output clear and understandable?",
        "efficiency": "Is the approach efficient?",
        "edge_cases": "Are edge cases handled?",
        "style": "Does it follow best practices?"
    }
    
    critique = {}
    issues = []
    
    for criterion, question in checklist.items():
        evaluation = evaluate_criterion(output, question)
        critique[criterion] = evaluation
        
        if not evaluation["pass"]:
            issues.append({
                "criterion": criterion,
                "issue": evaluation["problem"],
                "suggestion": evaluation["fix"]
            })
    
    return {
        "checklist_results": critique,
        "issues": issues,
        "quality_score": calculate_score(critique)
    }


def evaluate_criterion(output, question):
    """Evaluate single criterion."""
    prompt = f"""
Output to evaluate:
{output}

Question: {question}

Evaluation:
- Pass? (yes/no)
- Problem (if any):
- Suggested fix (if any):
"""
    
    response = llm.generate(prompt)
    return parse_evaluation(response)
```

### Comparative Critique

```python
def comparative_critique(output, task, references):
    """
    Compare output to reference examples.
    """
    prompt = f"""
Task: {task}

Your output:
{output}

Reference examples:
{format_references(references)}

Compare your output to references:
1. What do references do well that yours doesn't?
2. What does yours do better?
3. What should be improved?

Analysis:"""
    
    critique = llm.generate(prompt)
    
    return {
        "strengths": extract_strengths(critique),
        "weaknesses": extract_weaknesses(critique),
        "improvements": extract_improvements(critique)
    }
```

### Self-Consistency Critique

```python
def self_consistency_critique(task, num_samples=3):
    """
    Generate multiple outputs and compare.
    """
    outputs = []
    
    # Generate multiple attempts
    for i in range(num_samples):
        output = generate_output(task, temperature=0.7)
        outputs.append(output)
    
    # Compare for consistency
    prompt = f"""
Task: {task}

Multiple attempts at solving this task:

Attempt 1:
{outputs[0]}

Attempt 2:
{outputs[1]}

Attempt 3:
{outputs[2]}

Analysis:
1. Where do they agree?
2. Where do they disagree?
3. Which approach seems most correct?
4. What's the best answer combining insights?

Critique:"""
    
    critique = llm.generate(prompt)
    
    return {
        "outputs": outputs,
        "critique": critique,
        "best_output": select_best(outputs, critique)
    }
```

### Adversarial Critique

```python
def adversarial_critique(output, task):
    """
    Agent acts as adversary trying to find flaws.
    """
    prompt = f"""
You are an adversarial critic. Your job is to find flaws 
in this output.

Task: {task}
Output: {output}

Be harsh and thorough. Find:
1. Logical errors
2. Missing edge cases
3. Incorrect assumptions
4. Potential failures
5. Security issues
6. Performance problems

Critique:"""
    
    critique = llm.generate(prompt)
    
    return {
        "adversarial_critique": critique,
        "severity": assess_severity(critique),
        "must_fix": extract_critical_issues(critique)
    }
```

## Implementing Reflection

Complete reflection system implementation.

```python
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum


class ReflectionStrategy(Enum):
    """Different reflection strategies."""
    CHECKLIST = "checklist"
    COMPARATIVE = "comparative"
    ADVERSARIAL = "adversarial"
    SELF_CONSISTENCY = "self_consistency"


@dataclass
class ReflectionResult:
    """Result of reflection."""
    quality_score: float  # 0-1
    issues: List[Dict]
    strengths: List[str]
    improvements: List[str]
    should_refine: bool


class ReflectiveAgent:
    """
    Agent with reflection capabilities.
    """
    
    def __init__(
        self,
        llm,
        max_iterations: int = 3,
        quality_threshold: float = 0.8,
        strategy: ReflectionStrategy = ReflectionStrategy.CHECKLIST
    ):
        self.llm = llm
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.strategy = strategy
    
    def generate_with_reflection(self, task: str) -> Dict:
        """
        Generate output with iterative reflection.
        """
        # Initial generation
        output = self.generate_initial(task)
        history = []
        
        for iteration in range(self.max_iterations):
            print(f"\n=== Reflection Iteration {iteration + 1} ===")
            
            # Reflect on current output
            reflection = self.reflect(output, task)
            
            history.append({
                "iteration": iteration,
                "output": output,
                "reflection": reflection
            })
            
            print(f"Quality score: {reflection.quality_score:.2f}")
            print(f"Issues found: {len(reflection.issues)}")
            
            # Check if good enough
            if not reflection.should_refine:
                print("✅ Quality threshold met")
                break
            
            # Refine
            output = self.refine(output, task, reflection)
        
        return {
            "final_output": output,
            "history": history,
            "iterations": len(history)
        }
    
    def generate_initial(self, task: str) -> str:
        """Generate initial output."""
        prompt = f"""Solve this task:

{task}

Provide your solution:"""
        
        return self.llm(prompt)
    
    def reflect(self, output: str, task: str) -> ReflectionResult:
        """
        Reflect on output using configured strategy.
        """
        if self.strategy == ReflectionStrategy.CHECKLIST:
            return self.checklist_reflection(output, task)
        elif self.strategy == ReflectionStrategy.ADVERSARIAL:
            return self.adversarial_reflection(output, task)
        elif self.strategy == ReflectionStrategy.COMPARATIVE:
            return self.comparative_reflection(output, task)
        else:
            return self.default_reflection(output, task)
    
    def default_reflection(self, output: str, task: str) -> ReflectionResult:
        """Default reflection strategy."""
        prompt = f"""Critically evaluate this output.

Task: {task}

Output:
{output}

Evaluation:
1. Quality score (0-10):
2. Strengths:
3. Issues:
4. Suggested improvements:

Analysis:"""
        
        critique = self.llm(prompt)
        
        # Parse critique
        quality_score = self.extract_quality_score(critique) / 10
        issues = self.extract_issues(critique)
        strengths = self.extract_strengths(critique)
        improvements = self.extract_improvements(critique)
        
        return ReflectionResult(
            quality_score=quality_score,
            issues=issues,
            strengths=strengths,
            improvements=improvements,
            should_refine=quality_score < self.quality_threshold
        )
    
    def checklist_reflection(self, output: str, task: str) -> ReflectionResult:
        """Checklist-based reflection."""
        criteria = [
            "Completeness: Addresses all parts of task?",
            "Correctness: Factually accurate?",
            "Clarity: Easy to understand?",
            "Robustness: Handles edge cases?",
            "Efficiency: Optimal approach?"
        ]
        
        prompt = f"""Evaluate this output against criteria.

Task: {task}

Output:
{output}

Criteria:
{chr(10).join(f'{i+1}. {c}' for i, c in enumerate(criteria))}

For each criterion, provide:
- Pass/Fail
- Score (0-10)
- Issues (if any)
- Improvement suggestions

Evaluation:"""
        
        critique = self.llm(prompt)
        
        # Parse and aggregate
        quality_score = self.extract_quality_score(critique) / 10
        issues = self.extract_issues(critique)
        improvements = self.extract_improvements(critique)
        
        return ReflectionResult(
            quality_score=quality_score,
            issues=issues,
            strengths=[],
            improvements=improvements,
            should_refine=quality_score < self.quality_threshold
        )
    
    def adversarial_reflection(self, output: str, task: str) -> ReflectionResult:
        """Adversarial reflection - find all flaws."""
        prompt = f"""Act as a harsh critic finding flaws.

Task: {task}
Output:
{output}

Find all issues:
- Logical errors
- Missing cases
- Incorrect assumptions
- Potential failures
- Any other problems

Be thorough and critical:"""
        
        critique = self.llm(prompt)
        
        issues = self.extract_issues(critique)
        quality_score = max(0, 1.0 - len(issues) * 0.15)  # Penalty per issue
        
        return ReflectionResult(
            quality_score=quality_score,
            issues=issues,
            strengths=[],
            improvements=[f"Fix: {issue}" for issue in issues],
            should_refine=len(issues) > 0
        )
    
    def comparative_reflection(self, output: str, task: str) -> ReflectionResult:
        """Compare to ideal examples."""
        # Generate reference solution
        reference = self.generate_reference(task)
        
        prompt = f"""Compare output to reference.

Task: {task}

Your output:
{output}

Reference output:
{reference}

Comparison:
1. What does reference do better?
2. What does yours do better?
3. Quality score (0-10):
4. Improvements needed:

Analysis:"""
        
        critique = self.llm(prompt)
        
        quality_score = self.extract_quality_score(critique) / 10
        issues = self.extract_issues(critique)
        improvements = self.extract_improvements(critique)
        
        return ReflectionResult(
            quality_score=quality_score,
            issues=issues,
            strengths=[],
            improvements=improvements,
            should_refine=quality_score < self.quality_threshold
        )
    
    def refine(
        self,
        output: str,
        task: str,
        reflection: ReflectionResult
    ) -> str:
        """
        Refine output based on reflection.
        """
        prompt = f"""Improve this output based on critique.

Task: {task}

Current output:
{output}

Issues identified:
{self.format_issues(reflection.issues)}

Improvements needed:
{chr(10).join(f'- {imp}' for imp in reflection.improvements)}

Provide improved version:"""
        
        improved = self.llm(prompt)
        
        return improved
    
    def generate_reference(self, task: str) -> str:
        """Generate reference solution."""
        prompt = f"""Provide an ideal, high-quality solution to:

{task}

Reference solution:"""
        
        return self.llm(prompt)
    
    def extract_quality_score(self, critique: str) -> float:
        """Extract quality score from critique."""
        import re
        
        # Look for patterns like "Quality: 7/10" or "Score: 7"
        patterns = [
            r'quality[:\s]+(\d+)',
            r'score[:\s]+(\d+)',
            r'(\d+)/10'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, critique, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        # Default: moderate quality
        return 5.0
    
    def extract_issues(self, critique: str) -> List[str]:
        """Extract issues from critique."""
        # Simple extraction (in production, use more robust parsing)
        issues = []
        
        lines = critique.split('\n')
        in_issues_section = False
        
        for line in lines:
            line = line.strip()
            
            if 'issue' in line.lower() or 'problem' in line.lower():
                in_issues_section = True
                continue
            
            if in_issues_section and line.startswith('-'):
                issues.append(line[1:].strip())
        
        return issues
    
    def extract_strengths(self, critique: str) -> List[str]:
        """Extract strengths from critique."""
        strengths = []
        
        lines = critique.split('\n')
        in_strengths_section = False
        
        for line in lines:
            line = line.strip()
            
            if 'strength' in line.lower():
                in_strengths_section = True
                continue
            
            if in_strengths_section and line.startswith('-'):
                strengths.append(line[1:].strip())
        
        return strengths
    
    def extract_improvements(self, critique: str) -> List[str]:
        """Extract improvement suggestions from critique."""
        improvements = []
        
        lines = critique.split('\n')
        in_improvements_section = False
        
        for line in lines:
            line = line.strip()
            
            if 'improve' in line.lower() or 'suggest' in line.lower():
                in_improvements_section = True
                continue
            
            if in_improvements_section and line.startswith('-'):
                improvements.append(line[1:].strip())
        
        return improvements
    
    def format_issues(self, issues: List[str]) -> str:
        """Format issues for prompt."""
        if not issues:
            return "No issues found"
        
        return "\n".join(f"- {issue}" for issue in issues)


# Example usage
def example_reflection():
    """Example of reflective agent."""
    
    # Mock LLM
    def mock_llm(prompt: str) -> str:
        if "Solve this task" in prompt:
            return "Initial solution (may have issues)"
        elif "Critically evaluate" in prompt:
            return """
Quality score (0-10): 6
Strengths:
- Good structure
Issues:
- Missing edge case handling
- Could be more efficient
Suggested improvements:
- Add input validation
- Optimize algorithm
"""
        else:
            return "Improved solution addressing issues"
    
    # Create reflective agent
    agent = ReflectiveAgent(
        llm=mock_llm,
        max_iterations=3,
        quality_threshold=0.8
    )
    
    # Generate with reflection
    result = agent.generate_with_reflection(
        "Write a function to calculate factorial"
    )
    
    print(f"\nFinal output:\n{result['final_output']}")
    print(f"\nIterations: {result['iterations']}")
```

## Iterative Refinement

Continuously improving output through multiple iterations.

### Refinement Strategies

**Incremental refinement**:

```python
def incremental_refinement(task, initial_output):
    """
    Refine one aspect at a time.
    """
    output = initial_output
    aspects = ["correctness", "clarity", "efficiency", "robustness"]
    
    for aspect in aspects:
        critique = evaluate_aspect(output, aspect)
        
        if critique["needs_improvement"]:
            output = refine_aspect(output, aspect, critique)
    
    return output
```

**Iterative deepening**:

```python
def iterative_deepening(task, max_depth=3):
    """
    Start simple, add depth iteratively.
    """
    depth = 1
    output = generate_simple_solution(task)
    
    while depth < max_depth:
        # Evaluate current depth
        evaluation = evaluate_depth(output, task)
        
        if evaluation["sufficient"]:
            break
        
        # Add more depth
        output = add_depth(output, evaluation["missing"])
        depth += 1
    
    return output
```

## Quality Evaluation

How to assess output quality.

### Multi-Dimensional Quality

```python
def evaluate_quality(output, task) -> Dict[str, float]:
    """
    Evaluate output on multiple dimensions.
    """
    dimensions = {
        "correctness": evaluate_correctness(output, task),
        "completeness": evaluate_completeness(output, task),
        "clarity": evaluate_clarity(output),
        "efficiency": evaluate_efficiency(output),
        "robustness": evaluate_robustness(output),
        "style": evaluate_style(output)
    }
    
    # Overall score (weighted average)
    weights = {
        "correctness": 0.3,
        "completeness": 0.25,
        "clarity": 0.15,
        "efficiency": 0.15,
        "robustness": 0.10,
        "style": 0.05
    }
    
    overall = sum(
        dimensions[dim] * weights[dim]
        for dim in dimensions
    )
    
    return {
        "dimensions": dimensions,
        "overall": overall,
        "weights": weights
    }
```

### Automated Testing

```python
def test_based_reflection(code, task):
    """
    Reflect using automated tests.
    """
    # Generate tests
    tests = generate_tests(task)
    
    # Run tests
    results = run_tests(code, tests)
    
    # Analyze failures
    if results["failed"]:
        issues = analyze_failures(results["failed"])
        
        # Refine based on failures
        improved_code = fix_issues(code, issues)
        
        return improved_code
    
    return code
```

## Learning from Mistakes

Using reflection for continuous improvement.

### Mistake Database

```python
class MistakeMemory:
    """
    Remember past mistakes to avoid repeating them.
    """
    
    def __init__(self):
        self.mistakes = []
    
    def record_mistake(self, context, mistake, correction):
        """Record a mistake and its correction."""
        self.mistakes.append({
            "context": context,
            "mistake": mistake,
            "correction": correction,
            "timestamp": time.time()
        })
    
    def find_similar_mistakes(self, current_context):
        """Find past mistakes in similar contexts."""
        similar = []
        
        for past in self.mistakes:
            similarity = calculate_similarity(
                current_context,
                past["context"]
            )
            
            if similarity > 0.7:
                similar.append(past)
        
        return similar
    
    def generate_warnings(self, current_context):
        """Generate warnings based on past mistakes."""
        similar = self.find_similar_mistakes(current_context)
        
        warnings = []
        for past in similar:
            warnings.append({
                "warning": f"Watch out for: {past['mistake']}",
                "suggestion": f"Instead: {past['correction']}"
            })
        
        return warnings
```

### Learning Loop

```python
def learning_reflection_loop(task, memory):
    """
    Reflection loop that learns from mistakes.
    """
    # Check for relevant past mistakes
    warnings = memory.generate_warnings(task)
    
    # Generate with warnings in mind
    prompt = f"""Task: {task}

Warnings from past mistakes:
{format_warnings(warnings)}

Solution:"""
    
    output = llm.generate(prompt)
    
    # Reflect
    critique = reflect(output, task)
    
    # If issues found, record as mistakes
    for issue in critique["issues"]:
        memory.record_mistake(
            context=task,
            mistake=issue["problem"],
            correction=issue["fix"]
        )
    
    # Refine if needed
    if critique["quality"] < THRESHOLD:
        output = refine_with_learning(output, critique, memory)
    
    return output
```

## Meta-Cognition

Reflecting on the reflection process itself.

### Meta-Reflection

```python
def meta_reflection(reflection_history):
    """
    Reflect on the reflection process.
    """
    prompt = f"""Analyze this sequence of reflections:

{format_reflection_history(reflection_history)}

Meta-analysis:
1. Is reflection helping? How much improvement?
2. Is the strategy effective?
3. Are we stuck in a loop?
4. Should we change approach?
5. What's working well?

Analysis:"""
    
    meta_critique = llm.generate(prompt)
    
    return {
        "effectiveness": extract_effectiveness(meta_critique),
        "should_continue": should_continue_reflecting(meta_critique),
        "strategy_adjustment": extract_strategy_adjustment(meta_critique)
    }
```

### Adaptive Reflection

```python
class AdaptiveReflector:
    """
    Adjusts reflection strategy based on effectiveness.
    """
    
    def __init__(self):
        self.strategy = "default"
        self.history = []
    
    def reflect_adaptive(self, output, task):
        """Reflect using current strategy."""
        reflection = self.reflect_with_strategy(output, task, self.strategy)
        
        self.history.append({
            "strategy": self.strategy,
            "reflection": reflection,
            "improvement": calculate_improvement(output, reflection)
        })
        
        # Meta-reflect periodically
        if len(self.history) % 5 == 0:
            self.adapt_strategy()
        
        return reflection
    
    def adapt_strategy(self):
        """Adjust strategy based on effectiveness."""
        # Analyze recent history
        recent = self.history[-5:]
        avg_improvement = sum(h["improvement"] for h in recent) / len(recent)
        
        if avg_improvement < 0.1:
            # Current strategy not effective, try different one
            self.strategy = self.select_alternative_strategy()
        
        print(f"Strategy adapted to: {self.strategy}")
```

## Reflection Prompting

Effective prompts for reflection.

### Structured Reflection Prompt

```python
REFLECTION_TEMPLATE = """
Critically evaluate this output.

TASK:
{task}

OUTPUT:
{output}

EVALUATION CRITERIA:
1. Correctness: Is it factually accurate?
2. Completeness: Does it fully address the task?
3. Clarity: Is it easy to understand?
4. Efficiency: Is the approach optimal?
5. Robustness: Does it handle edge cases?

For each criterion, provide:
- Score (1-10)
- Assessment
- Issues (if any)
- Improvements (if needed)

OVERALL ASSESSMENT:
- Overall score (1-10):
- Key strengths:
- Key weaknesses:
- Priority improvements:
- Should refine? (yes/no)

CRITIQUE:
"""
```

### Socratic Questioning

```python
def socratic_reflection(output, task):
    """
    Use Socratic questioning for reflection.
    """
    questions = [
        "What assumptions did you make?",
        "What evidence supports this?",
        "What could go wrong?",
        "What alternatives exist?",
        "What are the implications?",
        "How do you know this is correct?"
    ]
    
    prompt = f"""Task: {task}
Output: {output}

Answer these questions about your output:

{chr(10).join(f'{i+1}. {q}' for i, q in enumerate(questions))}

Reflection:"""
    
    return llm.generate(prompt)
```

## Multi-Level Reflection

Reflection at different levels of abstraction.

### Hierarchical Reflection

```python
def hierarchical_reflection(output, task):
    """
    Reflect at multiple levels.
    """
    # High-level reflection
    high_level = reflect_on_approach(output, task)
    
    # Mid-level reflection
    mid_level = reflect_on_structure(output)
    
    # Low-level reflection
    low_level = reflect_on_details(output)
    
    # Synthesize
    synthesis = synthesize_reflections(high_level, mid_level, low_level)
    
    return {
        "high_level": high_level,
        "mid_level": mid_level,
        "low_level": low_level,
        "synthesis": synthesis
    }


def reflect_on_approach(output, task):
    """High-level: Is this the right approach?"""
    prompt = f"""Task: {task}
Output: {output}

High-level reflection:
- Is this the right general approach?
- Are there better alternatives?
- Does the structure make sense?

Analysis:"""
    
    return llm.generate(prompt)


def reflect_on_structure(output):
    """Mid-level: Is it well-organized?"""
    prompt = f"""Output: {output}

Structure reflection:
- Is it well-organized?
- Are components properly separated?
- Is the flow logical?

Analysis:"""
    
    return llm.generate(prompt)


def reflect_on_details(output):
    """Low-level: Are details correct?"""
    prompt = f"""Output: {output}

Detail reflection:
- Are all details correct?
- Any syntax/logical errors?
- Edge cases handled?

Analysis:"""
    
    return llm.generate(prompt)
```

## Reflection in Multi-Agent Systems

Agents reflecting on each other's work.

### Peer Reflection

```python
def peer_reflection(agent_outputs, task):
    """
    Agents reflect on each other's outputs.
    """
    reflections = {}
    
    for agent_id, output in agent_outputs.items():
        # Other agents reflect on this output
        peer_critiques = []
        
        for peer_id, peer in agents.items():
            if peer_id != agent_id:
                critique = peer.critique(output, task)
                peer_critiques.append({
                    "peer": peer_id,
                    "critique": critique
                })
        
        reflections[agent_id] = {
            "output": output,
            "peer_critiques": peer_critiques,
            "consensus": find_consensus(peer_critiques)
        }
    
    return reflections
```

### Collaborative Refinement

```python
def collaborative_refinement(initial_output, agents, task):
    """
    Multiple agents collaboratively refine output.
    """
    output = initial_output
    
    for iteration in range(MAX_ITERATIONS):
        # Each agent suggests improvements
        suggestions = []
        
        for agent in agents:
            critique = agent.reflect(output, task)
            improvement = agent.suggest_improvement(output, critique)
            suggestions.append(improvement)
        
        # Synthesize suggestions
        output = synthesize_improvements(output, suggestions)
        
        # Check consensus on quality
        if all(agent.is_satisfied(output, task) for agent in agents):
            break
    
    return output
```

## Common Reflection Patterns

Frequently-used reflection patterns.

### Generate-Test-Refine

```python
def generate_test_refine(task):
    """
    Generate solution, test it, refine based on results.
    """
    # Generate
    solution = generate_solution(task)
    
    # Test
    test_results = run_tests(solution, task)
    
    # Refine if needed
    while not all_tests_pass(test_results):
        # Analyze failures
        failures = get_failures(test_results)
        
        # Refine
        solution = refine_based_on_failures(solution, failures)
        
        # Retest
        test_results = run_tests(solution, task)
    
    return solution
```

### Draft-Critique-Revise

```python
def draft_critique_revise(task, num_iterations=3):
    """
    Draft, critique, revise iteratively.
    """
    draft = create_draft(task)
    
    for i in range(num_iterations):
        critique = critique_draft(draft, task)
        
        if critique["quality"] >= THRESHOLD:
            break
        
        draft = revise_draft(draft, critique)
    
    return draft
```

### Ensemble Reflection

```python
def ensemble_reflection(output, task, num_critics=3):
    """
    Multiple independent critiques combined.
    """
    critiques = []
    
    # Generate multiple critiques
    for i in range(num_critics):
        critique = generate_critique(output, task, temperature=0.7)
        critiques.append(critique)
    
    # Find common issues (mentioned by multiple critics)
    common_issues = find_common_issues(critiques)
    
    # Aggregate scores
    avg_score = sum(c["score"] for c in critiques) / len(critiques)
    
    return {
        "critiques": critiques,
        "common_issues": common_issues,
        "average_score": avg_score,
        "should_refine": avg_score < THRESHOLD
    }
```

## Avoiding Reflection Loops

Preventing infinite refinement cycles.

### Loop Detection

```python
def detect_reflection_loop(history):
    """
    Detect if stuck in repetitive refinement.
    """
    if len(history) < 3:
        return False
    
    recent = history[-3:]
    
    # Check for similar critiques
    critiques = [h["reflection"]["issues"] for h in recent]
    
    if all_similar(critiques):
        return True  # Same issues repeatedly
    
    # Check for minimal improvement
    scores = [h["reflection"]["quality_score"] for h in recent]
    improvements = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
    
    if all(imp < 0.05 for imp in improvements):
        return True  # Minimal progress
    
    return False
```

### Termination Conditions

```python
def should_terminate_reflection(history, iteration):
    """
    Decide when to stop reflecting.
    """
    # Max iterations reached
    if iteration >= MAX_ITERATIONS:
        return True
    
    # Quality threshold met
    current_quality = history[-1]["reflection"]["quality_score"]
    if current_quality >= QUALITY_THRESHOLD:
        return True
    
    # Stuck in loop
    if detect_reflection_loop(history):
        return True
    
    # Diminishing returns
    if len(history) >= 2:
        prev_quality = history[-2]["reflection"]["quality_score"]
        improvement = current_quality - prev_quality
        
        if improvement < MINIMAL_IMPROVEMENT_THRESHOLD:
            return True
    
    return False
```

### Diversification Strategy

```python
def diversified_reflection(output, task, iteration):
    """
    Vary reflection approach to avoid loops.
    """
    strategies = [
        "checklist",
        "adversarial",
        "comparative",
        "socratic"
    ]
    
    # Use different strategy each iteration
    strategy = strategies[iteration % len(strategies)]
    
    return reflect_with_strategy(output, task, strategy)
```

## When to Use Reflection

Guidelines for when reflection is beneficial.

### Use Reflection When:

1. **Quality is critical**
   - High-stakes outputs
   - User-facing content
   - Code going to production

2. **Complex tasks**
   - Multiple requirements
   - Subtle constraints
   - Room for interpretation

3. **Learning is valuable**
   - Want to improve over time
   - Building expertise
   - Creating training data

4. **Errors are costly**
   - Expensive to fix later
   - Safety-critical
   - Reputation at stake

### Skip Reflection When:

1. **Simple tasks**
   - Straightforward requirements
   - One clear approach
   - Low error risk

2. **Speed critical**
   - Real-time responses needed
   - Time constraints tight
   - Latency matters

3. **Resources constrained**
   - Token budget limited
   - API costs high
   - Compute expensive

4. **Diminishing returns**
   - First attempt sufficient
   - Improvements marginal
   - Over-optimization

## Summary

Reflection enables agents to evaluate and improve their own work:

1. **Pattern**: Generate → Reflect → Refine (iterate)
2. **Critique Mechanisms**: Checklist, comparative, adversarial, self-consistency
3. **Implementation**: ReflectiveAgent with multiple strategies
4. **Iterative Refinement**: Continuous improvement through multiple iterations
5. **Quality Evaluation**: Multi-dimensional assessment
6. **Learning**: Remember mistakes, avoid repetition
7. **Meta-Cognition**: Reflect on reflection itself
8. **Prompting**: Structured templates, Socratic questioning
9. **Multi-Level**: Reflect at different abstractions
10. **Multi-Agent**: Peer reflection, collaborative refinement
11. **Patterns**: Generate-test-refine, draft-critique-revise, ensemble
12. **Loop Avoidance**: Detection, termination conditions, diversification
13. **When to Use**: Quality-critical, complex tasks; skip for simple, time-critical

Reflection dramatically improves output quality at the cost of additional compute. Use wisely.

## Next Steps

- Continue to [Autonomous Agent Patterns](autonomous-agents.md) for long-running, self-directed agents
- Explore [Advanced Patterns](advanced-patterns.md) for sophisticated architectures
- Study [Planning Architectures](planning-architectures.md) for upfront planning strategies
- Review [ReAct](react.md) for combining reasoning with action
