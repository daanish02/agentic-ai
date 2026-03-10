# Continuous Evaluation

## Table of Contents

- [Introduction](#introduction)
- [Production Monitoring](#production-monitoring)
- [Metrics Over Time](#metrics-over-time)
- [Degradation Detection](#degradation-detection)
- [A/B Testing](#ab-testing)
- [Shadow Mode](#shadow-mode)
- [Online vs Offline Evaluation](#online-vs-offline-evaluation)
- [Real-Time Dashboards](#real-time-dashboards)
- [Alerting Systems](#alerting-systems)
- [Continuous Improvement Loop](#continuous-improvement-loop)
- [Sampling Strategies](#sampling-strategies)
- [Cost Management](#cost-management)
- [Quality Assurance](#quality-assurance)
- [Production Debugging](#production-debugging)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Evaluation doesn't stop at deployment. In production, agents face **real user queries, evolving patterns, and changing conditions**. Continuous evaluation ensures agents maintain quality over time.

```python
# The continuous evaluation cycle

┌──────────────────────────────────────────┐
│         Deploy Agent to Production        │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│     Monitor Performance Continuously      │
│   • Track metrics                         │
│   • Detect anomalies                      │
│   • Collect feedback                      │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│     Detect Issues & Opportunities         │
│   • Performance degradation               │
│   • New failure modes                     │
│   • Improvement opportunities             │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│        Investigate & Improve              │
│   • Root cause analysis                   │
│   • Fix issues                            │
│   • A/B test improvements                 │
└──────────────┬───────────────────────────┘
               │
               └───────────► Back to Deploy
```

This guide covers production monitoring, degradation detection, A/B testing, and maintaining agent quality in deployed systems.

> "Deployment is not the end - it's the beginning of continuous evaluation."

## Production Monitoring

Tracking agent performance in real-time.

### Monitoring System

```python
class ProductionMonitor:
    """Monitor agent performance in production."""

    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.metrics_store = MetricsStore()
        self.alert_manager = AlertManager()

    def log_interaction(self, interaction):
        """Log a single agent interaction."""

        metrics = {
            'timestamp': datetime.now(),
            'agent_id': self.agent_id,
            'user_id': interaction['user_id'],
            'task': interaction['task'],
            'success': interaction['success'],
            'latency': interaction['latency'],
            'tokens': interaction['tokens'],
            'cost': interaction['cost'],
            'user_feedback': interaction.get('feedback'),
            'error': interaction.get('error')
        }

        # Store metrics
        self.metrics_store.record(metrics)

        # Check for anomalies
        self._check_anomalies(metrics)

        # Update real-time statistics
        self._update_statistics(metrics)

    def get_current_performance(self, window_hours=24):
        """Get performance over recent time window."""

        recent_interactions = self.metrics_store.query(
            agent_id=self.agent_id,
            since=datetime.now() - timedelta(hours=window_hours)
        )

        if not recent_interactions:
            return None

        return {
            'success_rate': np.mean([i['success'] for i in recent_interactions]),
            'avg_latency': np.mean([i['latency'] for i in recent_interactions]),
            'avg_cost': np.mean([i['cost'] for i in recent_interactions]),
            'total_requests': len(recent_interactions),
            'error_rate': np.mean([i['error'] is not None for i in recent_interactions]),
            'user_satisfaction': np.mean([
                i['user_feedback'] for i in recent_interactions
                if i.get('user_feedback') is not None
            ])
        }

    def _check_anomalies(self, metrics):
        """Check for anomalous behavior."""

        # Get baseline
        baseline = self._get_baseline_metrics()

        # Check for significant deviations
        anomalies = []

        if metrics['latency'] > baseline['latency'] * 2:
            anomalies.append({
                'type': 'high_latency',
                'value': metrics['latency'],
                'baseline': baseline['latency']
            })

        if metrics['cost'] > baseline['cost'] * 1.5:
            anomalies.append({
                'type': 'high_cost',
                'value': metrics['cost'],
                'baseline': baseline['cost']
            })

        if not metrics['success'] and baseline['success_rate'] > 0.8:
            anomalies.append({
                'type': 'unexpected_failure',
                'error': metrics.get('error')
            })

        # Alert if anomalies found
        if anomalies:
            self.alert_manager.send_alert(anomalies)
```

## Degradation Detection

Identifying when performance declines.

### Degradation Detector

```python
class DegradationDetector:
    """Detect performance degradation over time."""

    def __init__(self, baseline_window=7, detection_window=1):
        self.baseline_window_days = baseline_window
        self.detection_window_days = detection_window

    def detect_degradation(self, metrics_history):
        """
        Detect if performance has degraded.

        Compare recent performance to historical baseline.
        """

        # Split into baseline and recent periods
        baseline_end = datetime.now() - timedelta(days=self.detection_window_days)
        baseline_start = baseline_end - timedelta(days=self.baseline_window_days)

        baseline_metrics = [
            m for m in metrics_history
            if baseline_start <= m['timestamp'] <= baseline_end
        ]

        recent_metrics = [
            m for m in metrics_history
            if m['timestamp'] > baseline_end
        ]

        if not baseline_metrics or not recent_metrics:
            return {'degradation_detected': False, 'reason': 'insufficient_data'}

        # Compare key metrics
        comparisons = {}

        for metric_name in ['success_rate', 'latency', 'cost', 'user_satisfaction']:
            baseline_value = np.mean([m[metric_name] for m in baseline_metrics
                                     if metric_name in m])
            recent_value = np.mean([m[metric_name] for m in recent_metrics
                                   if metric_name in m])

            # Statistical test for significance
            baseline_values = [m[metric_name] for m in baseline_metrics if metric_name in m]
            recent_values = [m[metric_name] for m in recent_metrics if metric_name in m]

            from scipy import stats
            t_stat, p_value = stats.ttest_ind(baseline_values, recent_values)

            # Determine if degradation
            if metric_name in ['success_rate', 'user_satisfaction']:
                degraded = recent_value < baseline_value * 0.95  # 5% drop
            else:  # latency, cost
                degraded = recent_value > baseline_value * 1.1  # 10% increase

            comparisons[metric_name] = {
                'baseline': baseline_value,
                'recent': recent_value,
                'change_pct': ((recent_value - baseline_value) / baseline_value * 100) if baseline_value != 0 else 0,
                'degraded': degraded,
                'statistically_significant': p_value < 0.05,
                'p_value': p_value
            }

        # Overall degradation if any key metric degraded significantly
        degradation_detected = any(
            c['degraded'] and c['statistically_significant']
            for c in comparisons.values()
        )

        return {
            'degradation_detected': degradation_detected,
            'comparisons': comparisons,
            'degraded_metrics': [k for k, v in comparisons.items()
                               if v['degraded'] and v['statistically_significant']]
        }
```

## A/B Testing

Testing improvements in production.

### A/B Test Framework

```python
class ABTest:
    """A/B testing framework for agents."""

    def __init__(self, test_name, agent_a, agent_b, split_ratio=0.5):
        self.test_name = test_name
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.split_ratio = split_ratio
        self.results = {'a': [], 'b': []}

    def assign_variant(self, user_id):
        """Assign user to variant A or B."""

        # Consistent hashing for user assignment
        hash_value = hashlib.md5(f"{user_id}_{self.test_name}".encode()).hexdigest()
        hash_int = int(hash_value, 16)

        return 'a' if (hash_int % 100) < (self.split_ratio * 100) else 'b'

    def run_test(self, user_id, task):
        """Run test and record result."""

        variant = self.assign_variant(user_id)
        agent = self.agent_a if variant == 'a' else self.agent_b

        # Run agent
        start_time = time.time()
        result = agent.run(task)
        latency = time.time() - start_time

        # Record result
        self.results[variant].append({
            'success': result.get('success', False),
            'latency': latency,
            'user_satisfaction': result.get('user_feedback'),
            'cost': result.get('cost', 0)
        })

        return result

    def analyze_results(self, min_samples=100):
        """Analyze A/B test results."""

        if len(self.results['a']) < min_samples or len(self.results['b']) < min_samples:
            return {'status': 'insufficient_data'}

        from scipy import stats

        # Compare success rates
        success_a = [r['success'] for r in self.results['a']]
        success_b = [r['success'] for r in self.results['b']]

        # Chi-square test for proportions
        chi2, p_value_success = stats.chisquare([sum(success_a), sum(success_b)])

        # Compare latencies
        latency_a = [r['latency'] for r in self.results['a']]
        latency_b = [r['latency'] for r in self.results['b']]

        t_stat_latency, p_value_latency = stats.ttest_ind(latency_a, latency_b)

        return {
            'status': 'complete',
            'sample_size_a': len(self.results['a']),
            'sample_size_b': len(self.results['b']),
            'success_rate_a': np.mean(success_a),
            'success_rate_b': np.mean(success_b),
            'avg_latency_a': np.mean(latency_a),
            'avg_latency_b': np.mean(latency_b),
            'winner': self._determine_winner(success_a, success_b, latency_a, latency_b),
            'confidence': 1 - min(p_value_success, p_value_latency),
            'statistically_significant': min(p_value_success, p_value_latency) < 0.05
        }

    def _determine_winner(self, success_a, success_b, latency_a, latency_b):
        """Determine which variant is better."""

        success_rate_a = np.mean(success_a)
        success_rate_b = np.mean(success_b)

        avg_latency_a = np.mean(latency_a)
        avg_latency_b = np.mean(latency_b)

        # Simple decision: higher success rate wins, latency as tiebreaker
        if abs(success_rate_a - success_rate_b) > 0.02:  # >2% difference
            return 'a' if success_rate_a > success_rate_b else 'b'
        else:
            # Similar success, choose faster
            return 'a' if avg_latency_a < avg_latency_b else 'b'
```

## Real-Time Dashboards

Visualizing production metrics.

### Dashboard System

```python
class ProductionDashboard:
    """Real-time dashboard for agent monitoring."""

    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.monitor = ProductionMonitor(agent_id)

    def get_dashboard_data(self):
        """Get current dashboard data."""

        # Current performance
        current = self.monitor.get_current_performance(window_hours=1)

        # Trends (last 24 hours)
        trends = self._compute_trends(hours=24)

        # Top errors
        top_errors = self._get_top_errors(limit=5)

        # Usage statistics
        usage = self._get_usage_stats()

        return {
            'current_metrics': current,
            'trends': trends,
            'top_errors': top_errors,
            'usage': usage,
            'alerts': self.monitor.alert_manager.get_active_alerts()
        }

    def _compute_trends(self, hours=24):
        """Compute metric trends over time."""

        # Get hourly data points
        data_points = []
        for h in range(hours):
            hour_start = datetime.now() - timedelta(hours=h+1)
            hour_end = datetime.now() - timedelta(hours=h)

            metrics = self.monitor.metrics_store.query(
                agent_id=self.agent_id,
                since=hour_start,
                until=hour_end
            )

            if metrics:
                data_points.append({
                    'timestamp': hour_end,
                    'success_rate': np.mean([m['success'] for m in metrics]),
                    'avg_latency': np.mean([m['latency'] for m in metrics]),
                    'request_count': len(metrics)
                })

        return data_points
```

## Sampling Strategies

Balancing coverage and cost.

### Smart Sampling

```python
class SamplingStrategy:
    """Intelligent sampling for continuous evaluation."""

    def __init__(self, base_rate=0.1):
        self.base_rate = base_rate  # Sample 10% by default

    def should_evaluate(self, interaction):
        """Decide whether to evaluate this interaction."""

        # Always evaluate errors
        if interaction.get('error'):
            return True

        # Always evaluate low user satisfaction
        if interaction.get('user_feedback') and interaction['user_feedback'] < 3:
            return True

        # Sample high-cost interactions more
        if interaction.get('cost', 0) > 1.0:  # Expensive
            rate = self.base_rate * 2
        else:
            rate = self.base_rate

        # Random sampling
        return random.random() < rate

    def adaptive_sampling(self, recent_performance):
        """Adjust sampling rate based on recent performance."""

        success_rate = recent_performance.get('success_rate', 1.0)

        # Increase sampling when performance drops
        if success_rate < 0.8:
            self.base_rate = min(0.5, self.base_rate * 1.5)
        elif success_rate > 0.95:
            self.base_rate = max(0.05, self.base_rate * 0.8)
```

## Summary

Continuous evaluation maintains agent quality in production:

**Key Components**:

- **Monitoring**: Real-time performance tracking
- **Degradation Detection**: Identifying performance drops
- **A/B Testing**: Testing improvements safely
- **Alerting**: Notifying of issues promptly
- **Dashboards**: Visualizing metrics

**Best Practices**:

- Monitor key metrics continuously
- Use statistical methods for degradation detection
- A/B test before full rollout
- Implement smart sampling to manage costs
- Set up alerts for critical issues
- Track trends over time

**Metrics to Monitor**:

- Success rate
- Latency
- Cost
- Error rate
- User satisfaction

Continuous evaluation ensures agents remain effective as conditions evolve.

## Next Steps

- **[Success Metrics](success-metrics.md)**: Defining success criteria
- **[Efficiency Metrics](efficiency-metrics.md)**: Resource monitoring
- **[Robustness Testing](robustness.md)**: Testing reliability
