# Efficiency Metrics

## Table of Contents

- [Introduction](#introduction)
- [Step Count Metrics](#step-count-metrics)
- [Token Usage](#token-usage)
- [Cost Tracking](#cost-tracking)
- [Latency Measurement](#latency-measurement)
- [Resource Consumption](#resource-consumption)
- [Quality-Efficiency Trade-offs](#quality-efficiency-trade-offs)
- [Efficiency Optimization](#efficiency-optimization)
- [Batch Efficiency](#batch-efficiency)
- [Caching and Reuse](#caching-and-reuse)
- [Efficiency Profiling](#efficiency-profiling)
- [Cost-Benefit Analysis](#cost-benefit-analysis)
- [Efficiency Benchmarks](#efficiency-benchmarks)
- [Real-Time Efficiency Monitoring](#real-time-efficiency-monitoring)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

An agent that achieves perfect results but costs $1000 and takes 10 minutes per query is not practical. **Efficiency matters**. Real-world agents must balance quality with resource constraints: tokens, time, cost, and compute.

```python
# Scenario comparison
agent_a = {
    'quality': 0.95,
    'tokens': 50000,
    'cost': '$0.50',
    'latency': '45s',
    'steps': 20
}

agent_b = {
    'quality': 0.90,
    'tokens': 5000,
    'cost': '$0.05',
    'latency': '8s',
    'steps': 5
}

# Which is better? Depends on your constraints!
# For production with high volume: agent_b
# For critical high-stakes tasks: agent_a
```

This guide covers efficiency measurement: from token counting to cost tracking, from latency optimization to quality-efficiency trade-offs.

> "The best agent is not the most capable, but the most capable within your resource constraints."

## Step Count Metrics

Number of reasoning/action steps taken.

### Measuring Steps

```python
class StepCounter:
    """Count and analyze agent steps."""

    def count_steps(self, agent_trace):
        """Count different types of steps."""

        steps = {
            'reasoning_steps': 0,
            'tool_calls': 0,
            'observations': 0,
            'total_steps': 0
        }

        for step in agent_trace:
            steps['total_steps'] += 1

            if step['type'] == 'thought':
                steps['reasoning_steps'] += 1
            elif step['type'] == 'action':
                steps['tool_calls'] += 1
            elif step['type'] == 'observation':
                steps['observations'] += 1

        return steps

    def compare_to_optimal(self, agent_trace, task):
        """Compare step count to optimal."""

        actual_steps = self.count_steps(agent_trace)['total_steps']
        optimal_steps = self._compute_optimal_steps(task)

        efficiency = optimal_steps / actual_steps if actual_steps > 0 else 0

        return {
            'actual_steps': actual_steps,
            'optimal_steps': optimal_steps,
            'efficiency_ratio': efficiency,
            'excess_steps': max(0, actual_steps - optimal_steps)
        }
```

## Token Usage

Tracking token consumption.

### Token Tracking

```python
class TokenUsageTracker:
    """Track token usage across agent operations."""

    def __init__(self):
        self.token_counts = []

    def track_call(self, prompt_tokens, completion_tokens, model):
        """Track tokens for a single LLM call."""

        self.token_counts.append({
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
            'model': model,
            'timestamp': datetime.now()
        })

    def get_statistics(self):
        """Compute token usage statistics."""

        if not self.token_counts:
            return {}

        total_prompt = sum(c['prompt_tokens'] for c in self.token_counts)
        total_completion = sum(c['completion_tokens'] for c in self.token_counts)
        total = total_prompt + total_completion

        return {
            'total_tokens': total,
            'prompt_tokens': total_prompt,
            'completion_tokens': total_completion,
            'num_calls': len(self.token_counts),
            'avg_tokens_per_call': total / len(self.token_counts),
            'token_efficiency': completion_tokens / total if total > 0 else 0
        }

    def estimate_cost(self, pricing):
        """Estimate cost based on token usage and pricing."""

        cost = 0

        for call in self.token_counts:
            model_pricing = pricing.get(call['model'], {})

            prompt_cost = (call['prompt_tokens'] / 1000) * model_pricing.get('prompt', 0)
            completion_cost = (call['completion_tokens'] / 1000) * model_pricing.get('completion', 0)

            cost += prompt_cost + completion_cost

        return cost
```

## Cost Tracking

Comprehensive cost monitoring.

### Cost Calculator

```python
class CostCalculator:
    """Calculate costs for agent operations."""

    def __init__(self):
        self.pricing = {
            'gpt-4': {'prompt': 0.03, 'completion': 0.06},
            'gpt-3.5-turbo': {'prompt': 0.001, 'completion': 0.002},
            'claude-3': {'prompt': 0.015, 'completion': 0.075}
        }

    def calculate_llm_cost(self, model, prompt_tokens, completion_tokens):
        """Calculate cost for LLM call."""

        if model not in self.pricing:
            return 0

        prompt_cost = (prompt_tokens / 1000) * self.pricing[model]['prompt']
        completion_cost = (completion_tokens / 1000) * self.pricing[model]['completion']

        return prompt_cost + completion_cost

    def calculate_total_cost(self, agent_trace):
        """Calculate total cost for agent execution."""

        costs = {
            'llm_cost': 0,
            'tool_cost': 0,
            'storage_cost': 0,
            'total_cost': 0
        }

        for step in agent_trace:
            if step['type'] == 'llm_call':
                costs['llm_cost'] += self.calculate_llm_cost(
                    step['model'],
                    step['prompt_tokens'],
                    step['completion_tokens']
                )

            elif step['type'] == 'tool_call':
                costs['tool_cost'] += self._estimate_tool_cost(step)

        costs['total_cost'] = sum(costs[k] for k in costs if k != 'total_cost')

        return costs
```

## Latency Measurement

Tracking response times.

### Latency Profiler

```python
class LatencyProfiler:
    """Profile latency of agent operations."""

    def __init__(self):
        self.timings = []

    def measure_operation(self, operation_name, operation_func, *args, **kwargs):
        """Measure latency of an operation."""

        start_time = time.time()
        result = operation_func(*args, **kwargs)
        end_time = time.time()

        latency = end_time - start_time

        self.timings.append({
            'operation': operation_name,
            'latency': latency,
            'timestamp': start_time
        })

        return result, latency

    def get_latency_breakdown(self):
        """Get breakdown of latencies by operation type."""

        breakdown = {}

        for timing in self.timings:
            op = timing['operation']
            if op not in breakdown:
                breakdown[op] = []
            breakdown[op].append(timing['latency'])

        return {
            op: {
                'count': len(latencies),
                'total': sum(latencies),
                'mean': np.mean(latencies),
                'p50': np.percentile(latencies, 50),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99)
            }
            for op, latencies in breakdown.items()
        }
```

## Quality-Efficiency Trade-offs

Balancing quality and efficiency.

### Trade-off Analysis

```python
class QualityEfficiencyAnalyzer:
    """Analyze quality-efficiency trade-offs."""

    def analyze_tradeoff(self, agent_runs):
        """
        Analyze trade-off between quality and efficiency.

        agent_runs: List of {'quality': score, 'cost': cost, 'latency': latency}
        """

        # Plot Pareto frontier
        pareto_frontier = self._find_pareto_frontier(agent_runs)

        # Compute efficiency scores
        efficiency_scores = [
            self._compute_efficiency_score(run)
            for run in agent_runs
        ]

        # Find optimal operating point
        optimal_point = self._find_optimal_point(agent_runs, efficiency_scores)

        return {
            'pareto_frontier': pareto_frontier,
            'efficiency_scores': efficiency_scores,
            'optimal_point': optimal_point,
            'quality_cost_correlation': self._compute_correlation(
                [r['quality'] for r in agent_runs],
                [r['cost'] for r in agent_runs]
            )
        }

    def _compute_efficiency_score(self, run):
        """Compute overall efficiency score."""

        # Normalize metrics
        quality_norm = run['quality']  # Already 0-1
        cost_norm = 1.0 / (1.0 + run['cost'])  # Lower cost is better
        latency_norm = 1.0 / (1.0 + run['latency'] / 10.0)  # Lower latency is better

        # Weighted combination
        efficiency = (
            0.4 * quality_norm +
            0.3 * cost_norm +
            0.3 * latency_norm
        )

        return efficiency
```

## Resource Consumption

Measuring compute and memory usage.

### Resource Monitor

```python
class ResourceMonitor:
    """Monitor resource consumption."""

    def __init__(self):
        self.measurements = []

    def measure_resources(self, operation_func, *args, **kwargs):
        """Measure resource usage during operation."""

        import psutil
        import tracemalloc

        # Start monitoring
        process = psutil.Process()
        tracemalloc.start()

        cpu_before = process.cpu_percent()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        start_time = time.time()

        # Run operation
        result = operation_func(*args, **kwargs)

        # Measure
        end_time = time.time()
        cpu_after = process.cpu_percent()
        memory_after = process.memory_info().rss / 1024 / 1024  # MB

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        measurement = {
            'duration': end_time - start_time,
            'cpu_usage': cpu_after - cpu_before,
            'memory_delta': memory_after - memory_before,
            'peak_memory': peak / 1024 / 1024,  # MB
            'timestamp': start_time
        }

        self.measurements.append(measurement)

        return result, measurement
```

## Summary

Efficiency metrics track resource consumption and performance:

**Key Metrics**:

- **Step Count**: Number of reasoning/action steps
- **Token Usage**: Input and output tokens consumed
- **Cost**: Monetary cost of operations
- **Latency**: Response time
- **Resource Usage**: CPU, memory, storage

**Trade-offs**:

- Quality vs cost
- Quality vs latency
- Completeness vs efficiency

**Best Practices**:

- Track all resource dimensions
- Understand quality-efficiency trade-offs
- Set resource budgets
- Monitor in production
- Optimize hot paths

Efficiency is critical for practical agent deployment - measure and optimize it continuously.

## Next Steps

- **[Robustness Testing](robustness.md)**: Testing under stress
- **[Success Metrics](success-metrics.md)**: Task completion metrics
- **[Continuous Evaluation](continuous-evaluation.md)**: Production monitoring
