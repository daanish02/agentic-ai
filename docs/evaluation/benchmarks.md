# Benchmarks and Datasets

## Table of Contents

- [Introduction](#introduction)
- [Why Benchmarks Matter](#why-benchmarks-matter)
- [Popular Agent Benchmarks](#popular-agent-benchmarks)
- [ToolBench](#toolbench)
- [AgentBench](#agentbench)
- [WebShop](#webshop)
- [SWE-bench](#swe-bench)
- [GAIA](#gaia)
- [Task-Specific Benchmarks](#task-specific-benchmarks)
- [Creating Custom Benchmarks](#creating-custom-benchmarks)
- [Benchmark Limitations](#benchmark-limitations)
- [Benchmark Best Practices](#benchmark-best-practices)
- [Evaluation Datasets](#evaluation-datasets)
- [Leaderboards](#leaderboards)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Benchmarks provide **standardized evaluation** of agent capabilities. They enable:

- Comparing different agents objectively
- Tracking progress over time
- Identifying strengths and weaknesses
- Reproducing results
- Community-wide progress measurement

This guide covers major agent benchmarks, how to use them, their limitations, and how to create custom benchmarks.

## Popular Agent Benchmarks

### ToolBench

```python
"""
ToolBench: Real-world API tool usage benchmark

Focus: Tool discovery, planning, and execution
Tasks: 16,000+ real-world API calling scenarios
Tools: 16,000+ API documentation from RapidAPI
Metrics: Success rate, tool selection accuracy, execution correctness

Example task:
"Find Italian restaurants in Seattle with ratings above 4.0"

Requires:
1. Search for restaurants (API)
2. Filter by location and cuisine
3. Filter by rating
4. Format and return results
"""

def evaluate_on_toolbench(agent):
    from toolbench import load_benchmark

    tasks = load_benchmark()
    results = []

    for task in tasks:
        result = agent.run(task['instruction'])
        score = evaluate_result(result, task['expected'])
        results.append(score)

    return {
        'success_rate': np.mean(results),
        'per_category': compute_category_scores(results, tasks)
    }
```

### AgentBench

```python
"""
AgentBench: Multi-dimensional agent evaluation

Focus: Reasoning, tool use, planning across diverse environments
Environments: 8 different (OS, DB, Web, Game, etc.)
Tasks: Varying complexity
Metrics: Success rate, efficiency, reasoning quality

Example environments:
- Operating System: Execute bash commands
- Database: Query and manipulate data
- Knowledge Graph: Navigate and reason
- Card Game: Strategic planning
"""

def evaluate_on_agentbench(agent):
    from agentbench import AgentBenchEvaluator

    evaluator = AgentBenchEvaluator()
    results = evaluator.evaluate(agent)

    return {
        'overall_score': results['overall'],
        'environment_scores': results['per_environment'],
        'rank': results['rank']
    }
```

### WebShop

```python
"""
WebShop: E-commerce interaction benchmark

Focus: Grounded language agent in shopping environment
Tasks: Buy products matching specifications
Environment: Simulated shopping website
Metrics: Purchase success, product match, efficiency

Example task:
"I'm looking for a waterproof hiking backpack with at least 40L capacity
and a laptop compartment, price under $100"

Requires:
1. Search products
2. Filter and compare
3. Read product details
4. Check specifications
5. Add to cart and purchase
"""
```

## Creating Custom Benchmarks

Building task-specific benchmarks.

### Benchmark Design

```python
class CustomBenchmark:
    """Template for creating custom benchmarks."""

    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.tasks = []
        self.metrics = []

    def add_task(self, task_spec):
        """
        Add a task to the benchmark.

        task_spec = {
            'id': unique identifier,
            'input': task input,
            'expected_output': expected result,
            'difficulty': 'easy'|'medium'|'hard',
            'category': task category,
            'metadata': additional info
        }
        """
        self.tasks.append(task_spec)

    def evaluate_agent(self, agent):
        """Evaluate agent on all benchmark tasks."""

        results = []

        for task in self.tasks:
            try:
                output = agent.run(task['input'])
                score = self._score_output(output, task)

                results.append({
                    'task_id': task['id'],
                    'success': score['success'],
                    'score': score['value'],
                    'difficulty': task['difficulty'],
                    'category': task['category']
                })
            except Exception as e:
                results.append({
                    'task_id': task['id'],
                    'success': False,
                    'error': str(e)
                })

        return self._aggregate_results(results)

    def _score_output(self, output, task):
        """Score agent output against expected."""

        if 'expected_output' in task:
            # Exact match or similarity
            if output == task['expected_output']:
                return {'success': True, 'value': 1.0}
            else:
                similarity = self._compute_similarity(output, task['expected_output'])
                return {'success': similarity > 0.8, 'value': similarity}

        # Use custom scorer
        if 'scorer' in task:
            return task['scorer'](output, task)

        # Default: any output is success
        return {'success': True, 'value': 1.0}
```

## Benchmark Limitations

Understanding what benchmarks don't tell you.

### Common Pitfalls

```python
class BenchmarkLimitations:
    """Understanding benchmark limitations."""

    limitations = {
        'overfitting': {
            'description': 'Agents optimize for benchmark, not real tasks',
            'example': 'High benchmark score but poor real-world performance',
            'mitigation': 'Use multiple benchmarks, test on real tasks'
        },

        'coverage': {
            'description': 'Benchmarks don\'t cover all scenarios',
            'example': 'Works on benchmark but fails on edge cases',
            'mitigation': 'Supplement with custom tests'
        },

        'static': {
            'description': 'Benchmarks become outdated',
            'example': 'Benchmark from 2020 may not reflect 2024 needs',
            'mitigation': 'Use recent benchmarks, create custom ones'
        },

        'metric_mismatch': {
            'description': 'Benchmark metrics ≠ real-world success',
            'example': 'High accuracy but unusable outputs',
            'mitigation': 'Define success metrics that match your needs'
        },

        'gaming': {
            'description': 'Benchmark-specific tricks don\'t generalize',
            'example': 'Memorizing benchmark patterns',
            'mitigation': 'Hold-out sets, diverse evaluation'
        }
    }
```

## Summary

Benchmarks provide standardized evaluation but should be used wisely:

**Major Benchmarks**:

- **ToolBench**: Real-world API usage
- **AgentBench**: Multi-environment evaluation
- **WebShop**: E-commerce tasks
- **SWE-bench**: Software engineering
- **GAIA**: General AI assistants

**Best Practices**:

- Use multiple benchmarks
- Understand limitations
- Don't overfit to benchmarks
- Create custom benchmarks for your domain
- Combine with real-world testing

Benchmarks are tools for measurement, not goals in themselves.

## Next Steps

- **[Continuous Evaluation](continuous-evaluation.md)**: Production monitoring
- **[Human Evaluation](human-evaluation.md)**: Human assessment
- **[Success Metrics](success-metrics.md)**: Defining success
