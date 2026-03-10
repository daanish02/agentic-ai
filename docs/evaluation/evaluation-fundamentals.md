# Evaluation Fundamentals

## Table of Contents

- [Introduction](#introduction)
- [Why Agent Evaluation is Hard](#why-agent-evaluation-is-hard)
- [The Non-Determinism Problem](#the-non-determinism-problem)
- [Open-Ended Tasks](#open-ended-tasks)
- [Multiple Valid Approaches](#multiple-valid-approaches)
- [Defining Correctness](#defining-correctness)
- [Evaluation Philosophy](#evaluation-philosophy)
- [Evaluation Dimensions](#evaluation-dimensions)
- [The Ground Truth Challenge](#the-ground-truth-challenge)
- [Compositional Evaluation](#compositional-evaluation)
- [Evaluation Design Principles](#evaluation-design-principles)
- [Measuring What Matters](#measuring-what-matters)
- [Evaluation Hierarchies](#evaluation-hierarchies)
- [Building Evaluation Pipelines](#building-evaluation-pipelines)
- [Common Pitfalls](#common-pitfalls)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Evaluating agentic systems is fundamentally different from evaluating traditional software. A unit test can verify that `add(2, 3)` returns `5` with complete certainty. But how do you verify that an agent "correctly" planned a trip, wrote good code, or conducted thorough research?

**Traditional software**:
- Deterministic behavior
- Clear correct/incorrect answers
- Testable with assertions
- Reproducible results

**Agentic systems**:
- Non-deterministic behavior (temperature > 0)
- Multiple valid solutions
- Context-dependent correctness
- Stochastic outputs

This fundamental difference requires a **completely different evaluation philosophy**. We can't rely on simple assertions. We need probabilistic thinking, multi-dimensional assessment, and often human judgment.

> "You can't manage what you can't measure, but in agent evaluation, measurement itself is the hardest problem."

This guide covers why agent evaluation is uniquely challenging, what dimensions we should evaluate, and how to think about assessing agent performance in principled ways.

## Why Agent Evaluation is Hard

Agent evaluation introduces challenges rarely seen in traditional software testing.

### Challenge 1: Non-Determinism

Agents use language models with temperature > 0, meaning **every run produces different outputs**:

```python
# Same agent, same input, different outputs
def run_agent_multiple_times():
    results = []
    for _ in range(5):
        result = agent.run("Book a restaurant for 2 at 7pm")
        results.append(result)
    
    # All different!
    # Run 1: Books Italian restaurant at 7:00pm
    # Run 2: Books French restaurant at 7:15pm  
    # Run 3: Asks clarification about cuisine preference
    # Run 4: Books Italian restaurant at 6:45pm
    # Run 5: Books Thai restaurant at 7:00pm
    
    return results
```

**Implication**: Single-run evaluation is insufficient. We need statistical methods.

### Challenge 2: Open-Ended Tasks

Many agent tasks lack single correct answers:

```python
task = "Research and summarize recent developments in quantum computing"

# All potentially valid:
# - 500-word summary focusing on algorithms
# - 2000-word deep-dive covering hardware and software
# - Bullet-point list of 10 key breakthroughs
# - Comparative analysis of different approaches
# - Timeline of major milestones
```

**Implication**: We need multi-dimensional quality metrics, not binary pass/fail.

### Challenge 3: Context Dependence

What's "correct" depends heavily on context:

```python
# User: "Book a flight to Paris"

# Context A: Business trip next week
# Good response: Premium economy, direct flight, morning departure

# Context B: Budget vacation in 6 months
# Good response: Economy, one stop acceptable, flexible dates

# Context C: Emergency
# Good response: Next available flight regardless of cost
```

**Implication**: Evaluation criteria must be context-aware.

### Challenge 4: Emergent Behavior

Agents exhibit emergent properties not easily decomposable:

```
Task: "Plan a week-long trip to Japan"

Requires:
- Flight booking (tool use)
- Hotel research (information retrieval)
- Itinerary planning (reasoning)
- Budget management (calculation)
- Cultural awareness (knowledge)
- User preference modeling (personalization)
- Error recovery (robustness)

Success depends on ALL of these working together.
```

**Implication**: Component-level tests don't guarantee system-level success.

### Challenge 5: Delayed Consequences

Some agent actions have consequences that aren't immediately visible:

```python
# Agent's decision tree
agent.book_flight(date="Jan 15")  # Seems fine
agent.book_hotel(checkin="Jan 15", checkout="Jan 20")  # Seems fine
agent.schedule_meeting(date="Jan 16")  # Seems fine

# But wait... the flight arrives at 11pm on Jan 15
# The meeting is scheduled for 9am on Jan 16
# That's not enough rest time!

# This error only becomes apparent when viewing the full plan
```

**Implication**: Evaluation must consider temporal and causal relationships.

## The Non-Determinism Problem

Non-determinism is perhaps the single biggest challenge in agent evaluation.

### Sources of Non-Determinism

**1. Model Sampling**

```python
# Temperature > 0 means different outputs
completion = llm.complete(
    prompt="Next action:",
    temperature=0.7  # Introduces randomness
)

# Possible outputs:
# "search for information"
# "ask user for clarification"  
# "use calculator tool"
# "analyze the context first"
```

**2. Tool Execution Variability**

```python
# Web search returns different results over time
results = search_web("latest AI news")
# Results change as new articles are published

# APIs may have rate limits, timeouts
api_result = call_external_api()
# Might succeed now, fail later, or have delayed response
```

**3. Timing Dependencies**

```python
# Time-dependent behavior
if datetime.now().hour < 9:
    agent.schedule_meeting("tomorrow morning")
else:
    agent.schedule_meeting("this afternoon")

# Evaluation results depend on when you run the test!
```

### Strategies for Handling Non-Determinism

**Strategy 1: Multiple Runs with Statistical Analysis**

```python
def evaluate_with_multiple_runs(agent, task, n_runs=20):
    """Run agent multiple times and compute statistics."""
    results = []
    
    for i in range(n_runs):
        result = agent.run(task)
        score = score_result(result)
        results.append(score)
    
    return {
        'mean': np.mean(results),
        'std': np.std(results),
        'min': np.min(results),
        'max': np.max(results),
        'success_rate': np.mean([r > threshold for r in results]),
        'confidence_interval': confidence_interval(results, 0.95)
    }

# Example output:
# {
#   'mean': 0.78,
#   'std': 0.12,
#   'min': 0.45,
#   'max': 0.95,
#   'success_rate': 0.85,  # 85% of runs succeeded
#   'confidence_interval': (0.72, 0.84)
# }
```

**Strategy 2: Controlled Randomness**

```python
def evaluate_with_fixed_seed(agent, task, seed=42):
    """Fix random seed for reproducible evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Still non-deterministic across different LLM APIs
    # But provides some reproducibility
    result = agent.run(task)
    return score_result(result)
```

**Strategy 3: Acceptance Regions**

```python
def evaluate_with_acceptance_region(agent, task):
    """Define acceptable output region rather than exact match."""
    result = agent.run(task)
    
    # Instead of: result == expected
    # Use: result in acceptance_region
    
    checks = [
        check_task_completed(result),
        check_no_hallucinations(result),
        check_reasonable_approach(result),
        check_safety_constraints(result)
    ]
    
    # Pass if all critical checks pass
    return all(checks)
```

**Strategy 4: Differential Testing**

```python
def differential_evaluation(agent_v1, agent_v2, tasks):
    """Compare relative performance rather than absolute."""
    
    improvements = 0
    regressions = 0
    
    for task in tasks:
        score_v1 = np.mean([agent_v1.run(task) for _ in range(10)])
        score_v2 = np.mean([agent_v2.run(task) for _ in range(10)])
        
        if score_v2 > score_v1 + threshold:
            improvements += 1
        elif score_v2 < score_v1 - threshold:
            regressions += 1
    
    return {
        'improvements': improvements,
        'regressions': regressions,
        'net_change': improvements - regressions
    }
```

### Quantifying Variance

Understanding the variance in agent behavior is crucial:

```python
class VarianceAnalyzer:
    """Analyze variance in agent outputs."""
    
    def analyze_output_variance(self, agent, task, n_runs=50):
        """Measure how much outputs vary across runs."""
        outputs = [agent.run(task) for _ in range(n_runs)]
        
        return {
            # Lexical variance (how different are the text outputs?)
            'lexical_diversity': self._compute_diversity(outputs),
            
            # Semantic variance (how different are the meanings?)
            'semantic_variance': self._semantic_similarity_variance(outputs),
            
            # Behavioral variance (do they take same actions?)
            'action_variance': self._action_sequence_variance(outputs),
            
            # Outcome variance (do they achieve same results?)
            'outcome_variance': self._outcome_variance(outputs)
        }
    
    def _compute_diversity(self, outputs):
        """Compute lexical diversity using set ratio."""
        unique_outputs = len(set(outputs))
        total_outputs = len(outputs)
        return unique_outputs / total_outputs
    
    def _semantic_similarity_variance(self, outputs):
        """Compute variance in semantic similarity."""
        embeddings = [embed(output) for output in outputs]
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)
        return np.var(similarities)
    
    def _action_sequence_variance(self, outputs):
        """Measure variance in action sequences."""
        action_sequences = [extract_actions(output) for output in outputs]
        
        # Compare using edit distance
        distances = []
        for i in range(len(action_sequences)):
            for j in range(i+1, len(action_sequences)):
                dist = edit_distance(action_sequences[i], action_sequences[j])
                distances.append(dist)
        
        return np.mean(distances)
    
    def _outcome_variance(self, outputs):
        """Measure variance in final outcomes."""
        outcomes = [extract_outcome(output) for output in outputs]
        
        # Binary: same outcome or different?
        same_outcomes = sum(1 for i in range(len(outcomes)-1) 
                          if outcomes[i] == outcomes[i+1])
        
        return 1 - (same_outcomes / (len(outcomes) - 1))
```

## Open-Ended Tasks

Open-ended tasks are those without a single correct answer or clear success criteria.

### Characteristics of Open-Ended Tasks

```python
# Examples across the spectrum

# Closed: Single correct answer
task_closed = "What is 15% of 250?"
# Answer: 37.5 (deterministic, verifiable)

# Semi-open: Constrained but flexible
task_semi_open = "Summarize this article in 100 words"
# Multiple valid summaries, but constrained format

# Open: Highly subjective
task_open = "Write a creative story about AI"
# Unbounded solution space

# Fully open: Multiple dimensions, no clear "best"
task_fully_open = "Research and synthesize information about quantum computing"
# Many valid approaches, formats, depth levels, focuses
```

### Evaluation Strategies for Open-Ended Tasks

**Strategy 1: Rubric-Based Evaluation**

```python
class RubricEvaluator:
    """Evaluate open-ended tasks with scoring rubrics."""
    
    def __init__(self):
        self.rubric = {
            'completeness': {
                'weight': 0.3,
                'levels': {
                    5: 'Addresses all aspects thoroughly',
                    4: 'Addresses all aspects adequately',
                    3: 'Addresses most aspects',
                    2: 'Addresses some aspects',
                    1: 'Addresses few aspects'
                }
            },
            'accuracy': {
                'weight': 0.3,
                'levels': {
                    5: 'All information is accurate',
                    4: 'Mostly accurate, minor errors',
                    3: 'Some inaccuracies',
                    2: 'Several inaccuracies',
                    1: 'Mostly inaccurate'
                }
            },
            'coherence': {
                'weight': 0.2,
                'levels': {
                    5: 'Highly logical and well-structured',
                    4: 'Logical with good structure',
                    3: 'Reasonably coherent',
                    2: 'Somewhat disjointed',
                    1: 'Incoherent'
                }
            },
            'usefulness': {
                'weight': 0.2,
                'levels': {
                    5: 'Highly actionable and valuable',
                    4: 'Useful and actionable',
                    3: 'Moderately useful',
                    2: 'Limited usefulness',
                    1: 'Not useful'
                }
            }
        }
    
    def evaluate(self, output, human_scores=None):
        """Evaluate output using rubric."""
        if human_scores:
            # Use human judgment
            scores = human_scores
        else:
            # Use automated proxy metrics
            scores = self._automated_scoring(output)
        
        # Compute weighted average
        total_score = 0
        for criterion, score in scores.items():
            weight = self.rubric[criterion]['weight']
            total_score += score * weight
        
        return {
            'total_score': total_score,
            'criterion_scores': scores,
            'grade': self._score_to_grade(total_score)
        }
    
    def _automated_scoring(self, output):
        """Approximate scoring using automated metrics."""
        return {
            'completeness': self._score_completeness(output),
            'accuracy': self._score_accuracy(output),
            'coherence': self._score_coherence(output),
            'usefulness': self._score_usefulness(output)
        }
    
    def _score_to_grade(self, score):
        """Convert numeric score to letter grade."""
        if score >= 4.5: return 'A'
        elif score >= 3.5: return 'B'
        elif score >= 2.5: return 'C'
        elif score >= 1.5: return 'D'
        else: return 'F'
```

**Strategy 2: Comparative Evaluation**

```python
def comparative_evaluation(agent_outputs, reference_outputs):
    """
    Compare agent outputs to reference outputs from other agents
    or humans, rather than to a single ground truth.
    """
    
    scores = []
    
    for agent_out, ref_out in zip(agent_outputs, reference_outputs):
        # Compare along multiple dimensions
        score = {
            'better_than_baseline': rate_comparison(agent_out, ref_out),
            'similar_quality': semantic_similarity(agent_out, ref_out),
            'different_approach': approach_diversity(agent_out, ref_out)
        }
        scores.append(score)
    
    return aggregate_scores(scores)
```

**Strategy 3: Model-Based Evaluation**

```python
class ModelBasedEvaluator:
    """Use LLM as judge for open-ended tasks."""
    
    def __init__(self, judge_model):
        self.judge = judge_model
    
    def evaluate(self, task, output):
        """Have LLM judge evaluate the output."""
        
        prompt = f"""
        Evaluate the following output for the given task.
        
        Task: {task}
        
        Output: {output}
        
        Evaluate on a scale of 1-5 for each criterion:
        1. Completeness: Does it fully address the task?
        2. Accuracy: Is the information correct?
        3. Clarity: Is it well-organized and understandable?
        4. Usefulness: Would this be helpful to the user?
        
        Provide:
        - Score for each criterion (1-5)
        - Brief justification for each score
        - Overall assessment
        
        Format as JSON.
        """
        
        evaluation = self.judge.complete(prompt)
        return parse_evaluation(evaluation)
    
    def evaluate_with_consistency_check(self, task, output, n_judges=3):
        """Use multiple judge evaluations for consistency."""
        evaluations = [
            self.evaluate(task, output) 
            for _ in range(n_judges)
        ]
        
        # Check inter-rater reliability
        agreement = self._compute_agreement(evaluations)
        
        return {
            'evaluations': evaluations,
            'agreement': agreement,
            'consensus': self._compute_consensus(evaluations)
        }
```

## Multiple Valid Approaches

Many tasks can be solved in fundamentally different but equally valid ways.

### The Approach Space

```python
# Task: "Find and summarize recent AI research papers"

# Approach A: Breadth-first
# - Search broadly
# - Get many papers
# - Provide brief summaries of each

# Approach B: Depth-first  
# - Find highly-cited papers
# - Read thoroughly
# - Provide deep analysis of few papers

# Approach C: Trend-focused
# - Identify research trends
# - Group papers by theme
# - Synthesize common patterns

# Approach D: Chronological
# - Organize by publication date
# - Show evolution of ideas
# - Highlight recent breakthroughs

# All are valid! Which is "best" depends on user needs.
```

### Evaluating Approach Diversity

```python
class ApproachAnalyzer:
    """Analyze and evaluate different solution approaches."""
    
    def identify_approach(self, agent_trace):
        """Identify the approach used by the agent."""
        
        # Extract action patterns
        actions = extract_action_sequence(agent_trace)
        
        # Classify approach based on patterns
        approach_features = {
            'search_breadth': count_unique_queries(actions),
            'search_depth': max_iterations_per_query(actions),
            'tool_diversity': count_unique_tools(actions),
            'backtracking': count_backtrack_events(actions),
            'systematic': is_systematic_exploration(actions),
            'opportunistic': is_opportunistic(actions)
        }
        
        return self.classify_approach(approach_features)
    
    def classify_approach(self, features):
        """Classify approach type based on features."""
        if features['search_breadth'] > 10 and features['search_depth'] < 3:
            return 'breadth_first'
        elif features['search_breadth'] < 5 and features['search_depth'] > 5:
            return 'depth_first'
        elif features['systematic']:
            return 'systematic'
        elif features['opportunistic']:
            return 'opportunistic'
        else:
            return 'mixed'
    
    def compare_approaches(self, approach_a, approach_b):
        """Compare two different approaches."""
        return {
            'efficiency': self._compare_efficiency(approach_a, approach_b),
            'thoroughness': self._compare_thoroughness(approach_a, approach_b),
            'quality': self._compare_quality(approach_a, approach_b),
            'context_fit': self._compare_context_fit(approach_a, approach_b)
        }
```

### Approach-Aware Evaluation

```python
def approach_aware_evaluation(agent, task, evaluation_criteria):
    """
    Evaluate agent while accounting for different valid approaches.
    """
    
    # Run agent
    trace = agent.run_with_trace(task)
    approach = identify_approach(trace)
    
    # Use approach-specific criteria
    if approach == 'breadth_first':
        weights = {
            'coverage': 0.4,
            'efficiency': 0.3,
            'accuracy': 0.3
        }
    elif approach == 'depth_first':
        weights = {
            'depth': 0.4,
            'accuracy': 0.4,
            'efficiency': 0.2
        }
    else:
        weights = evaluation_criteria.default_weights
    
    # Score using appropriate weights
    scores = compute_scores(trace, evaluation_criteria)
    weighted_score = sum(scores[k] * weights[k] for k in weights)
    
    return {
        'approach': approach,
        'weighted_score': weighted_score,
        'component_scores': scores
    }
```

## Defining Correctness

What does "correct" mean for agent behavior?

### Levels of Correctness

```
┌─────────────────────────────────────┐
│ Level 1: Functional Correctness     │
│ Does it achieve the stated goal?    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Level 2: Process Correctness        │
│ Did it use reasonable methods?      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Level 3: Safety Correctness         │
│ Did it avoid harmful actions?       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Level 4: Preference Alignment       │
│ Did it match user preferences?      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Level 5: Optimal Efficiency         │
│ Did it use minimum resources?       │
└─────────────────────────────────────┘
```

### Correctness Specification

```python
class CorrectnessSpec:
    """Specify what constitutes correct behavior."""
    
    def __init__(self, task):
        self.task = task
        self.spec = self._define_spec()
    
    def _define_spec(self):
        """Define multi-level correctness criteria."""
        return {
            # Must satisfy (hard constraints)
            'must': [
                'completes primary objective',
                'produces valid output format',
                'stays within safety boundaries',
                'does not hallucinate facts'
            ],
            
            # Should satisfy (soft constraints)
            'should': [
                'uses efficient approach',
                'provides clear reasoning',
                'handles edge cases',
                'considers user context'
            ],
            
            # Nice to have (preferences)
            'prefer': [
                'minimizes tool calls',
                'provides detailed explanations',
                'anticipates follow-up needs',
                'offers alternatives'
            ]
        }
    
    def check_correctness(self, agent_output):
        """Check if output satisfies correctness spec."""
        
        results = {
            'must': self._check_constraints(agent_output, self.spec['must']),
            'should': self._check_constraints(agent_output, self.spec['should']),
            'prefer': self._check_constraints(agent_output, self.spec['prefer'])
        }
        
        # Overall correctness: must all pass, should most pass, prefer some pass
        is_correct = (
            all(results['must']) and
            sum(results['should']) >= len(self.spec['should']) * 0.7
        )
        
        return {
            'correct': is_correct,
            'details': results,
            'score': self._compute_score(results)
        }
    
    def _compute_score(self, results):
        """Compute overall correctness score."""
        # Must: 0.5 weight (binary: all or nothing)
        # Should: 0.3 weight (proportion that pass)
        # Prefer: 0.2 weight (proportion that pass)
        
        must_score = 1.0 if all(results['must']) else 0.0
        should_score = sum(results['should']) / len(results['should'])
        prefer_score = sum(results['prefer']) / len(results['prefer'])
        
        return 0.5 * must_score + 0.3 * should_score + 0.2 * prefer_score
```

## Evaluation Philosophy

How should we think about agent evaluation philosophically?

### Evaluation as Scientific Measurement

```python
"""
Good evaluation shares characteristics with scientific measurement:

1. VALIDITY: Measures what it claims to measure
2. RELIABILITY: Consistent results across runs
3. SENSITIVITY: Detects real differences in performance
4. SPECIFICITY: Doesn't flag false positives
5. PRACTICALITY: Feasible to conduct regularly
"""

class EvaluationPhilosophy:
    """Principled approach to agent evaluation."""
    
    principles = {
        'no_single_metric': """
            No single metric captures full performance.
            Always use multiple complementary metrics.
        """,
        
        'context_matters': """
            Performance is always relative to context.
            Same behavior can be good or bad depending on situation.
        """,
        
        'tradeoffs_exist': """
            Optimizing one dimension often hurts others.
            Recognize and measure trade-offs explicitly.
        """,
        
        'variance_is_real': """
            Non-determinism is inherent, not a bug.
            Measure distributions, not single points.
        """,
        
        'humans_in_loop': """
            Automated metrics are proxies, not truth.
            Human evaluation is often necessary.
        """,
        
        'continuous_improvement': """
            Evaluation should drive improvement.
            Metrics should be actionable and informative.
        """
    }
```

### The Evaluation Pyramid

```
                    ▲
                   ╱ ╲
                  ╱   ╲
                 ╱ End ╲
                ╱ Users ╲              (Highest signal, lowest throughput)
               ╱─────────╲
              ╱           ╲
             ╱   Expert   ╲
            ╱  Evaluation  ╲           (High signal, low throughput)
           ╱───────────────╲
          ╱                 ╲
         ╱  Model-as-Judge  ╲         (Medium signal, medium throughput)
        ╱───────────────────╲
       ╱                     ╲
      ╱   Automated Metrics  ╲       (Lower signal, high throughput)
     ╱─────────────────────────╲
    ╱                           ╲
   ╱    Unit Tests & Assertions ╲   (Lowest signal, highest throughput)
  ╱─────────────────────────────╲

Use multiple layers:
- Unit tests for basic functionality
- Automated metrics for rapid iteration
- Model judges for nuanced assessment
- Expert evaluation for quality assurance
- End user feedback for final validation
```

## Evaluation Dimensions

Agent performance is multi-dimensional. We must evaluate across all relevant dimensions.

### Core Dimensions

```python
class EvaluationDimensions:
    """Comprehensive set of evaluation dimensions."""
    
    dimensions = {
        'effectiveness': {
            'description': 'Does it achieve the goal?',
            'metrics': [
                'task_completion_rate',
                'goal_achievement_score',
                'output_quality'
            ]
        },
        
        'efficiency': {
            'description': 'How quickly/cheaply does it succeed?',
            'metrics': [
                'steps_to_completion',
                'token_usage',
                'api_calls',
                'latency',
                'cost'
            ]
        },
        
        'correctness': {
            'description': 'Is the output accurate?',
            'metrics': [
                'factual_accuracy',
                'logical_validity',
                'hallucination_rate',
                'error_rate'
            ]
        },
        
        'robustness': {
            'description': 'Does it handle edge cases?',
            'metrics': [
                'success_rate_on_hard_examples',
                'graceful_degradation',
                'error_recovery_rate',
                'adversarial_robustness'
            ]
        },
        
        'safety': {
            'description': 'Does it avoid harm?',
            'metrics': [
                'unsafe_action_rate',
                'privacy_violations',
                'resource_abuse',
                'jailbreak_resistance'
            ]
        },
        
        'usefulness': {
            'description': 'Is it helpful to users?',
            'metrics': [
                'user_satisfaction',
                'perceived_quality',
                'actionability',
                'relevance'
            ]
        },
        
        'interpretability': {
            'description': 'Can we understand its behavior?',
            'metrics': [
                'reasoning_clarity',
                'explanation_quality',
                'trace_comprehensibility',
                'predictability'
            ]
        }
    }
    
    def evaluate_all_dimensions(self, agent, task):
        """Evaluate agent across all dimensions."""
        
        results = {}
        
        for dim_name, dim_spec in self.dimensions.items():
            dim_scores = {}
            
            for metric in dim_spec['metrics']:
                dim_scores[metric] = self.evaluate_metric(
                    agent, task, metric
                )
            
            results[dim_name] = {
                'scores': dim_scores,
                'aggregate': np.mean(list(dim_scores.values()))
            }
        
        return results
```

### Dimension Trade-offs

```python
class TradeoffAnalyzer:
    """Analyze trade-offs between evaluation dimensions."""
    
    def analyze_tradeoffs(self, agent_results):
        """Identify trade-offs in agent performance."""
        
        # Common trade-offs
        tradeoffs = [
            ('efficiency', 'thoroughness'),  # Fast but superficial vs slow but deep
            ('accuracy', 'cost'),            # High accuracy costs more tokens
            ('simplicity', 'capability'),    # Simple prompts vs complex behaviors
            ('safety', 'usefulness'),        # Overly safe can be unhelpful
            ('speed', 'quality'),            # Quick responses vs polished outputs
        ]
        
        analysis = {}
        
        for dim_a, dim_b in tradeoffs:
            correlation = self._compute_correlation(
                agent_results[dim_a],
                agent_results[dim_b]
            )
            
            analysis[f'{dim_a}_vs_{dim_b}'] = {
                'correlation': correlation,
                'interpretation': self._interpret_correlation(correlation)
            }
        
        return analysis
    
    def _interpret_correlation(self, corr):
        """Interpret correlation value."""
        if corr < -0.5:
            return 'strong_tradeoff'  # Improving one hurts the other
        elif corr < -0.2:
            return 'moderate_tradeoff'
        elif corr > 0.5:
            return 'strong_synergy'   # Both improve together
        elif corr > 0.2:
            return 'moderate_synergy'
        else:
            return 'independent'      # No clear relationship
```

## The Ground Truth Challenge

Establishing "ground truth" for agent tasks is often impossible or impractical.

### Types of Ground Truth

```python
class GroundTruthType:
    """Different types of ground truth for evaluation."""
    
    DETERMINISTIC = "deterministic"
    # Example: "What is 2+2?" → 4
    # Clear, unambiguous correct answer
    
    REFERENCE_BASED = "reference_based"
    # Example: "Summarize this article" → Compare to human summary
    # Ground truth is reference output, but not unique
    
    ORACLE_BASED = "oracle_based"
    # Example: "Is this response helpful?" → Ask humans
    # Ground truth comes from human judgment
    
    OUTCOME_BASED = "outcome_based"
    # Example: "Write code that passes tests" → Run tests
    # Ground truth is whether objective is achieved
    
    CONSENSUS_BASED = "consensus_based"
    # Example: "Is this reasoning sound?" → Ask multiple experts
    # Ground truth is majority opinion
    
    NO_GROUND_TRUTH = "no_ground_truth"
    # Example: "Write a creative story"
    # No objective ground truth exists
```

### Dealing with Absent Ground Truth

```python
class NoGroundTruthEvaluator:
    """Evaluation strategies when ground truth is unavailable."""
    
    def evaluate_without_ground_truth(self, agent_output, task):
        """Multiple strategies for evaluation without ground truth."""
        
        strategies = {
            # Compare to other systems
            'comparative': self._comparative_eval(agent_output),
            
            # Use proxy metrics
            'proxy_metrics': self._proxy_metrics_eval(agent_output),
            
            # Human judgment
            'human_preference': self._human_preference_eval(agent_output),
            
            # Consistency checks
            'self_consistency': self._consistency_eval(agent_output),
            
            # Process evaluation
            'process_quality': self._process_eval(agent_output)
        }
        
        return self._aggregate_strategies(strategies)
    
    def _comparative_eval(self, output):
        """Compare to baselines and other systems."""
        baselines = [
            ('random', random_baseline()),
            ('simple', simple_heuristic_baseline()),
            ('previous_version', previous_agent_version())
        ]
        
        better_than = sum(
            1 for name, baseline in baselines
            if self._is_better(output, baseline)
        )
        
        return better_than / len(baselines)
    
    def _proxy_metrics_eval(self, output):
        """Use proxy metrics that correlate with quality."""
        proxies = {
            'length_appropriate': self._check_length(output),
            'coherence': self._check_coherence(output),
            'specificity': self._check_specificity(output),
            'completeness': self._check_completeness(output)
        }
        
        return np.mean(list(proxies.values()))
    
    def _consistency_eval(self, output):
        """Check if output is self-consistent."""
        
        # Run multiple times, check agreement
        outputs = [generate_output() for _ in range(5)]
        
        # High agreement suggests confidence
        agreement = compute_pairwise_agreement(outputs)
        
        return agreement
```

### Creating Synthetic Ground Truth

```python
class SyntheticGroundTruthGenerator:
    """Generate synthetic evaluation data with known ground truth."""
    
    def generate_synthetic_tasks(self, n_tasks=100):
        """Create tasks with verifiable answers."""
        
        tasks = []
        
        # Math problems (deterministic ground truth)
        for _ in range(n_tasks // 4):
            task = self._generate_math_problem()
            tasks.append(task)
        
        # Code generation (test-based ground truth)
        for _ in range(n_tasks // 4):
            task = self._generate_coding_problem()
            tasks.append(task)
        
        # Information retrieval (document-based ground truth)
        for _ in range(n_tasks // 4):
            task = self._generate_qa_problem()
            tasks.append(task)
        
        # Reasoning tasks (logical ground truth)
        for _ in range(n_tasks // 4):
            task = self._generate_reasoning_problem()
            tasks.append(task)
        
        return tasks
    
    def _generate_math_problem(self):
        """Generate math problem with exact answer."""
        a, b = random.randint(10, 100), random.randint(10, 100)
        operation = random.choice(['+', '-', '*', '/'])
        
        problem = f"Calculate {a} {operation} {b}"
        answer = eval(f"{a} {operation} {b}")
        
        return {
            'task': problem,
            'ground_truth': answer,
            'type': 'deterministic'
        }
    
    def _generate_coding_problem(self):
        """Generate coding task with test cases."""
        
        specs = [
            {
                'description': 'Write a function that reverses a string',
                'test_cases': [
                    ('hello', 'olleh'),
                    ('world', 'dlrow'),
                    ('', '')
                ]
            },
            {
                'description': 'Write a function that checks if a number is prime',
                'test_cases': [
                    (2, True),
                    (4, False),
                    (17, True)
                ]
            }
        ]
        
        spec = random.choice(specs)
        
        return {
            'task': spec['description'],
            'ground_truth': spec['test_cases'],
            'type': 'outcome_based'
        }
```

## Compositional Evaluation

Complex agent behaviors should be evaluated compositionally - breaking down into evaluable components.

### Decomposition Strategy

```python
class CompositionalEvaluator:
    """Evaluate complex behaviors by decomposing into components."""
    
    def __init__(self):
        self.components = {
            'perception': self._evaluate_perception,
            'reasoning': self._evaluate_reasoning,
            'planning': self._evaluate_planning,
            'action': self._evaluate_action,
            'learning': self._evaluate_learning
        }
    
    def evaluate_compositionally(self, agent, task):
        """Break down evaluation into component assessments."""
        
        # Run agent and capture full trace
        trace = agent.run_with_full_trace(task)
        
        # Evaluate each component
        component_scores = {}
        for component_name, evaluator in self.components.items():
            score = evaluator(trace)
            component_scores[component_name] = score
        
        # Aggregate scores
        overall_score = self._aggregate_components(component_scores)
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(component_scores)
        
        return {
            'component_scores': component_scores,
            'overall_score': overall_score,
            'bottlenecks': bottlenecks,
            'improvement_priorities': self._prioritize_improvements(
                component_scores, bottlenecks
            )
        }
    
    def _evaluate_perception(self, trace):
        """Evaluate how well agent perceives/understands input."""
        metrics = {
            'input_parsing': self._check_input_parsing(trace),
            'context_extraction': self._check_context_extraction(trace),
            'ambiguity_handling': self._check_ambiguity_handling(trace)
        }
        return np.mean(list(metrics.values()))
    
    def _evaluate_reasoning(self, trace):
        """Evaluate quality of reasoning."""
        metrics = {
            'logical_validity': self._check_logical_validity(trace),
            'coherence': self._check_reasoning_coherence(trace),
            'depth': self._check_reasoning_depth(trace)
        }
        return np.mean(list(metrics.values()))
    
    def _evaluate_planning(self, trace):
        """Evaluate planning quality."""
        metrics = {
            'goal_decomposition': self._check_goal_decomposition(trace),
            'plan_feasibility': self._check_plan_feasibility(trace),
            'adaptability': self._check_plan_adaptability(trace)
        }
        return np.mean(list(metrics.values()))
    
    def _evaluate_action(self, trace):
        """Evaluate action execution."""
        metrics = {
            'tool_selection': self._check_tool_selection(trace),
            'parameter_correctness': self._check_parameters(trace),
            'execution_success': self._check_execution(trace)
        }
        return np.mean(list(metrics.values()))
    
    def _identify_bottlenecks(self, component_scores):
        """Find weakest components."""
        sorted_components = sorted(
            component_scores.items(),
            key=lambda x: x[1]
        )
        
        # Bottom 2 are bottlenecks
        return [comp for comp, score in sorted_components[:2]]
```

### Hierarchical Evaluation

```python
class HierarchicalEvaluator:
    """Evaluate at multiple levels of abstraction."""
    
    def evaluate_hierarchically(self, agent, task):
        """
        Evaluate from low-level actions to high-level goals.
        """
        
        results = {
            'action_level': self._evaluate_actions(agent, task),
            'subtask_level': self._evaluate_subtasks(agent, task),
            'task_level': self._evaluate_task(agent, task),
            'goal_level': self._evaluate_goal_achievement(agent, task)
        }
        
        # Check consistency across levels
        consistency = self._check_level_consistency(results)
        
        return {
            'level_results': results,
            'consistency': consistency,
            'overall_assessment': self._aggregate_levels(results)
        }
    
    def _evaluate_actions(self, agent, task):
        """Micro-level: Individual action correctness."""
        trace = agent.run_with_trace(task)
        actions = extract_actions(trace)
        
        scores = []
        for action in actions:
            score = {
                'valid': is_valid_action(action),
                'appropriate': is_appropriate_context(action),
                'successful': was_successful(action)
            }
            scores.append(np.mean(list(score.values())))
        
        return np.mean(scores)
    
    def _evaluate_subtasks(self, agent, task):
        """Meso-level: Subtask completion."""
        subtasks = decompose_into_subtasks(task)
        
        completion_rates = []
        for subtask in subtasks:
            completed = agent.completed_subtask(subtask)
            completion_rates.append(1.0 if completed else 0.0)
        
        return np.mean(completion_rates)
    
    def _evaluate_task(self, agent, task):
        """Task-level: Overall task success."""
        result = agent.run(task)
        return score_task_completion(result, task)
    
    def _evaluate_goal_achievement(self, agent, task):
        """Macro-level: Did it achieve the user's goal?"""
        # Goal may be implicit or broader than stated task
        goal = infer_user_goal(task)
        result = agent.run(task)
        return score_goal_achievement(result, goal)
```

## Evaluation Design Principles

Principles for designing effective evaluations.

### Principle 1: Start with Clear Objectives

```python
class EvaluationObjective:
    """Define clear evaluation objectives."""
    
    def __init__(self, purpose):
        self.purpose = purpose
        self.objectives = self._define_objectives()
    
    def _define_objectives(self):
        """Map purpose to specific objectives."""
        
        if self.purpose == 'development':
            return {
                'primary': 'rapid feedback on changes',
                'metrics': ['fast_to_run', 'actionable', 'sensitive_to_changes'],
                'acceptable_tradeoffs': ['precision_for_speed']
            }
        
        elif self.purpose == 'benchmarking':
            return {
                'primary': 'compare systems objectively',
                'metrics': ['standardized', 'reproducible', 'comprehensive'],
                'acceptable_tradeoffs': ['speed_for_thoroughness']
            }
        
        elif self.purpose == 'deployment_decision':
            return {
                'primary': 'determine if ready for production',
                'metrics': ['safety', 'robustness', 'user_satisfaction'],
                'acceptable_tradeoffs': ['cost_for_quality']
            }
        
        elif self.purpose == 'research':
            return {
                'primary': 'understand capabilities and limitations',
                'metrics': ['comprehensive', 'interpretable', 'novel'],
                'acceptable_tradeoffs': ['practicality_for_insight']
            }
```

### Principle 2: Use Multiple Metrics

```python
def design_metric_suite(evaluation_goals):
    """Design complementary suite of metrics."""
    
    suite = {
        # Lagging indicators (outcome measures)
        'lagging': [
            'task_success_rate',
            'user_satisfaction',
            'business_impact'
        ],
        
        # Leading indicators (process measures)
        'leading': [
            'reasoning_quality',
            'tool_use_accuracy',
            'error_rate'
        ],
        
        # Diagnostic metrics (root cause analysis)
        'diagnostic': [
            'failure_mode_distribution',
            'bottleneck_identification',
            'error_attribution'
        ]
    }
    
    return suite
```

### Principle 3: Match Evaluation to Stage

```python
class StageBasedEvaluation:
    """Different evaluation strategies for different stages."""
    
    stages = {
        'early_development': {
            'frequency': 'continuous',
            'speed': 'fast',
            'coverage': 'focused',
            'methods': ['unit_tests', 'smoke_tests', 'dev_examples']
        },
        
        'pre_commit': {
            'frequency': 'per_change',
            'speed': 'moderate',
            'coverage': 'broad',
            'methods': ['regression_tests', 'automated_metrics']
        },
        
        'pre_release': {
            'frequency': 'per_release',
            'speed': 'slow',
            'coverage': 'comprehensive',
            'methods': ['full_benchmarks', 'human_evaluation', 'stress_tests']
        },
        
        'production': {
            'frequency': 'continuous',
            'speed': 'real_time',
            'coverage': 'sampled',
            'methods': ['monitoring', 'ab_testing', 'user_feedback']
        }
    }
```

## Building Evaluation Pipelines

Practical implementation of evaluation systems.

### Basic Evaluation Pipeline

```python
class EvaluationPipeline:
    """End-to-end evaluation pipeline."""
    
    def __init__(self, agent, test_suite):
        self.agent = agent
        self.test_suite = test_suite
        self.results = []
    
    def run_evaluation(self):
        """Run full evaluation pipeline."""
        
        print("Starting evaluation...")
        
        # 1. Load test cases
        test_cases = self.test_suite.load()
        print(f"Loaded {len(test_cases)} test cases")
        
        # 2. Run agent on each test case
        for i, test_case in enumerate(test_cases):
            print(f"Running test {i+1}/{len(test_cases)}")
            
            result = self._run_single_test(test_case)
            self.results.append(result)
        
        # 3. Compute aggregate metrics
        metrics = self._compute_metrics()
        
        # 4. Generate report
        report = self._generate_report(metrics)
        
        # 5. Save results
        self._save_results(report)
        
        return report
    
    def _run_single_test(self, test_case):
        """Run agent on single test case."""
        
        try:
            # Run agent
            start_time = time.time()
            output = self.agent.run(test_case['input'])
            end_time = time.time()
            
            # Score output
            scores = self._score_output(output, test_case)
            
            return {
                'test_id': test_case['id'],
                'output': output,
                'scores': scores,
                'latency': end_time - start_time,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'test_id': test_case['id'],
                'error': str(e),
                'status': 'error'
            }
    
    def _score_output(self, output, test_case):
        """Score agent output."""
        
        scores = {}
        
        # Correctness
        if 'ground_truth' in test_case:
            scores['correctness'] = self._score_correctness(
                output, test_case['ground_truth']
            )
        
        # Quality dimensions
        scores['completeness'] = self._score_completeness(output, test_case)
        scores['coherence'] = self._score_coherence(output)
        scores['safety'] = self._score_safety(output)
        
        return scores
    
    def _compute_metrics(self):
        """Compute aggregate metrics across all results."""
        
        successful_results = [r for r in self.results if r['status'] == 'success']
        
        if not successful_results:
            return {'error': 'No successful runs'}
        
        metrics = {
            'success_rate': len(successful_results) / len(self.results),
            'avg_latency': np.mean([r['latency'] for r in successful_results]),
            'avg_correctness': np.mean([
                r['scores']['correctness'] 
                for r in successful_results 
                if 'correctness' in r['scores']
            ])
        }
        
        # Per-dimension statistics
        for dimension in ['completeness', 'coherence', 'safety']:
            values = [r['scores'][dimension] for r in successful_results]
            metrics[f'{dimension}_mean'] = np.mean(values)
            metrics[f'{dimension}_std'] = np.std(values)
        
        return metrics
    
    def _generate_report(self, metrics):
        """Generate evaluation report."""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'agent_version': self.agent.version,
            'test_suite': self.test_suite.name,
            'num_tests': len(self.results),
            'metrics': metrics,
            'detailed_results': self.results
        }
        
        return report
    
    def _save_results(self, report):
        """Save evaluation results."""
        
        filename = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Results saved to {filename}")
```

### Continuous Evaluation System

```python
class ContinuousEvaluator:
    """System for continuous evaluation during development."""
    
    def __init__(self, agent, test_suite):
        self.agent = agent
        self.test_suite = test_suite
        self.history = []
    
    def watch_and_evaluate(self):
        """Watch for changes and automatically evaluate."""
        
        print("Starting continuous evaluation...")
        print("Watching for agent changes...")
        
        last_agent_hash = self._compute_agent_hash()
        
        while True:
            time.sleep(60)  # Check every minute
            
            current_hash = self._compute_agent_hash()
            
            if current_hash != last_agent_hash:
                print("\nAgent changed! Running evaluation...")
                
                results = self._run_quick_eval()
                self.history.append({
                    'timestamp': datetime.now(),
                    'results': results
                })
                
                # Check for regressions
                if self.history[-1]:
                    self._check_regressions()
                
                last_agent_hash = current_hash
    
    def _run_quick_eval(self):
        """Run fast subset of evaluations."""
        # Use smaller test set for speed
        quick_tests = self.test_suite.get_quick_tests()
        
        results = []
        for test in quick_tests:
            result = self.agent.run(test['input'])
            score = self._quick_score(result, test)
            results.append(score)
        
        avg_score = np.mean(results)
        print(f"Quick eval score: {avg_score:.2f}")
        
        return {'avg_score': avg_score, 'individual_scores': results}
    
    def _check_regressions(self):
        """Check if latest version regressed."""
        
        if len(self.history) < 2:
            return
        
        current_score = self.history[-1]['results']['avg_score']
        previous_score = self.history[-2]['results']['avg_score']
        
        if current_score < previous_score - 0.1:  # 10% drop
            print(f"⚠️  REGRESSION DETECTED!")
            print(f"  Previous: {previous_score:.2f}")
            print(f"  Current:  {current_score:.2f}")
            print(f"  Change:   {current_score - previous_score:.2f}")
```

## Common Pitfalls

Mistakes to avoid in agent evaluation.

### Pitfall 1: Over-reliance on Single Metrics

```python
# ❌ BAD: Single metric tells incomplete story
def evaluate_agent_bad(agent, tests):
    accuracy = compute_accuracy(agent, tests)
    return accuracy  # Ignores efficiency, safety, usefulness...

# ✅ GOOD: Multiple complementary metrics
def evaluate_agent_good(agent, tests):
    return {
        'accuracy': compute_accuracy(agent, tests),
        'efficiency': compute_efficiency(agent, tests),
        'robustness': compute_robustness(agent, tests),
        'safety': compute_safety(agent, tests),
        'usefulness': compute_usefulness(agent, tests)
    }
```

### Pitfall 2: Not Accounting for Variance

```python
# ❌ BAD: Single run, unreliable
def compare_agents_bad(agent_a, agent_b, test):
    score_a = agent_a.run(test)
    score_b = agent_b.run(test)
    return score_a > score_b  # Might be due to randomness!

# ✅ GOOD: Multiple runs with statistical testing
def compare_agents_good(agent_a, agent_b, test, n_runs=20):
    scores_a = [agent_a.run(test) for _ in range(n_runs)]
    scores_b = [agent_b.run(test) for _ in range(n_runs)]
    
    # Statistical significance test
    t_stat, p_value = ttest_ind(scores_a, scores_b)
    
    return {
        'mean_a': np.mean(scores_a),
        'mean_b': np.mean(scores_b),
        'statistically_significant': p_value < 0.05,
        'p_value': p_value
    }
```

### Pitfall 3: Overfitting to Benchmarks

```python
# ❌ BAD: Optimizing only for known benchmarks
def optimize_for_benchmarks(agent):
    # Agent becomes good at benchmarks but not real tasks
    # "Teaching to the test"
    while True:
        agent.train_on(STANDARD_BENCHMARKS)
        if agent.score(STANDARD_BENCHMARKS) > 0.9:
            break
    return agent

# ✅ GOOD: Hold-out sets and diverse evaluation
def optimize_properly(agent):
    # Train on training set
    agent.train_on(TRAIN_SET)
    
    # Evaluate on diverse held-out sets
    scores = {
        'benchmark': agent.score(BENCHMARK_HOLDOUT),
        'real_world': agent.score(REAL_WORLD_TASKS),
        'adversarial': agent.score(ADVERSARIAL_TESTS),
        'edge_cases': agent.score(EDGE_CASES)
    }
    
    # Good performance across all sets
    return all(score > threshold for score in scores.values())
```

### Pitfall 4: Ignoring Real-World Distribution

```python
# ❌ BAD: Uniform test distribution
test_suite = [
    {'type': 'math', 'difficulty': 'easy'},
    {'type': 'math', 'difficulty': 'hard'},
    {'type': 'code', 'difficulty': 'easy'},
    {'type': 'code', 'difficulty': 'hard'},
    # Equal weight to all types
]

# ✅ GOOD: Representative distribution
def create_representative_test_suite(production_data):
    # Sample according to real-world distribution
    task_distribution = analyze_production_distribution(production_data)
    
    # Example: 60% simple queries, 30% medium, 10% complex
    test_suite = sample_according_to_distribution(task_distribution)
    
    return test_suite
```

### Pitfall 5: Not Testing Failure Modes

```python
# ❌ BAD: Only test happy paths
happy_path_tests = [
    "valid input A",
    "valid input B",
    "valid input C"
]

# ✅ GOOD: Test edge cases and failure modes
comprehensive_tests = [
    # Happy paths
    "valid input A",
    "valid input B",
    
    # Edge cases
    "empty input",
    "extremely long input",
    "ambiguous input",
    
    # Adversarial
    "jailbreak attempt",
    "prompt injection",
    "resource abuse",
    
    # Error conditions
    "API timeout",
    "tool failure",
    "conflicting constraints"
]
```

## Summary

Agent evaluation is fundamentally challenging due to non-determinism, open-ended tasks, and the difficulty of defining "correctness". Key principles:

**Evaluation Challenges**:
- Non-deterministic behavior requires statistical approaches
- Open-ended tasks need multi-dimensional assessment
- Multiple valid approaches complicate comparison
- Ground truth is often unavailable or subjective

**Evaluation Philosophy**:
- No single metric captures full performance
- Context matters - same behavior can be good or bad
- Measure distributions, not single points
- Use multiple evaluation layers (unit tests → human evaluation)

**Best Practices**:
- Evaluate across multiple dimensions (effectiveness, efficiency, safety, etc.)
- Use compositional evaluation for complex behaviors
- Account for variance with multiple runs
- Design evaluation to match your purpose and stage
- Avoid common pitfalls (single metrics, ignoring variance, overfitting)

**Practical Implementation**:
- Build evaluation pipelines for automated assessment
- Use continuous evaluation during development
- Combine automated metrics with human judgment
- Track metrics over time to detect regressions

Effective evaluation is essential for building reliable agents, measuring progress, and making informed design decisions.

## Next Steps

- **[Success Metrics](success-metrics.md)**: Measuring task completion and achievement
- **[Reasoning Quality](reasoning-quality.md)**: Evaluating agent reasoning processes
- **[Tool Use Evaluation](tool-use-evaluation.md)**: Assessing tool selection and usage
- **[Efficiency Metrics](efficiency-metrics.md)**: Measuring resource consumption
- **[Robustness Testing](robustness.md)**: Testing reliability and error handling
- **[Benchmarks](benchmarks.md)**: Standard evaluation datasets
- **[Human Evaluation](human-evaluation.md)**: Incorporating human judgment
- **[Continuous Evaluation](continuous-evaluation.md)**: Production monitoring
