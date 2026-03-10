# Success Metrics

## Table of Contents

- [Introduction](#introduction)
- [Defining Success](#defining-success)
- [Binary Success Metrics](#binary-success-metrics)
- [Partial Credit Scoring](#partial-credit-scoring)
- [Quality Scoring](#quality-scoring)
- [Task-Specific Success Criteria](#task-specific-success-criteria)
- [Multi-Objective Success](#multi-objective-success)
- [Success Rates and Statistics](#success-rates-and-statistics)
- [Goal Achievement Metrics](#goal-achievement-metrics)
- [User-Centric Success](#user-centric-success)
- [Automated vs Human Success Assessment](#automated-vs-human-success-assessment)
- [Success Thresholds](#success-thresholds)
- [Measuring Progress Toward Goals](#measuring-progress-toward-goals)
- [Success Attribution](#success-attribution)
- [Real-World Success Examples](#real-world-success-examples)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Measuring success is the most fundamental aspect of agent evaluation. But what does "success" mean for an agent? Unlike traditional software where success is binary (test passes or fails), agent success is **multifaceted, contextual, and often subjective**.

Consider these scenarios:

```python
# Scenario 1: Customer service agent
User: "I need help with my order"
Agent: "I've found your order and initiated a refund"
# Success? Depends on whether refund was appropriate!

# Scenario 2: Research agent
User: "Summarize recent AI papers"
Agent: [Returns 2-page summary of 5 papers]
# Success? Depends on depth needed, time available, user expertise!

# Scenario 3: Code generation agent
User: "Write a function to sort a list"
Agent: [Provides working but inefficient O(n³) sort]
# Success? Works but not optimal!
```

This guide covers how to define, measure, and track success for different types of agent tasks, from simple binary metrics to nuanced quality assessments.

> "Success is not a point but a region. The question is not 'did it succeed?' but 'how well did it succeed?'"

## Defining Success

Before measuring success, we must define what it means for a specific task and context.

### Dimensions of Success

```python
class SuccessDefinition:
    """Framework for defining success comprehensively."""

    def __init__(self, task):
        self.task = task
        self.dimensions = self._define_dimensions()

    def _define_dimensions(self):
        """Define all dimensions of success for this task."""
        return {
            # Primary objective
            'goal_achievement': {
                'description': 'Did it accomplish the stated goal?',
                'weight': 0.4,
                'threshold': 'must_satisfy'
            },

            # Quality of execution
            'execution_quality': {
                'description': 'How well was it done?',
                'weight': 0.3,
                'threshold': 'should_satisfy'
            },

            # Efficiency
            'resource_usage': {
                'description': 'Was it done efficiently?',
                'weight': 0.1,
                'threshold': 'prefer_satisfy'
            },

            # Safety
            'safety_compliance': {
                'description': 'Were safety constraints respected?',
                'weight': 0.2,
                'threshold': 'must_satisfy'
            }
        }

    def is_successful(self, agent_output):
        """Determine if output represents success."""

        scores = {}

        for dimension, spec in self.dimensions.items():
            score = self._evaluate_dimension(agent_output, dimension, spec)
            scores[dimension] = score

        # Check thresholds
        must_satisfy = all(
            scores[d] >= 0.9
            for d, spec in self.dimensions.items()
            if spec['threshold'] == 'must_satisfy'
        )

        should_satisfy = sum(
            scores[d] >= 0.7
            for d, spec in self.dimensions.items()
            if spec['threshold'] == 'should_satisfy'
        ) >= len([s for s in self.dimensions.values()
                  if s['threshold'] == 'should_satisfy']) * 0.8

        # Compute weighted score
        weighted_score = sum(
            scores[d] * spec['weight']
            for d, spec in self.dimensions.items()
        )

        return {
            'successful': must_satisfy and should_satisfy,
            'weighted_score': weighted_score,
            'dimension_scores': scores
        }
```

### Context-Dependent Success

```python
class ContextualSuccessEvaluator:
    """Evaluate success based on context."""

    def evaluate(self, agent_output, task, context):
        """Success depends on context."""

        # Extract context factors
        urgency = context.get('urgency', 'normal')
        user_expertise = context.get('user_expertise', 'intermediate')
        resource_constraints = context.get('resource_constraints', 'moderate')

        # Adjust success criteria based on context
        if urgency == 'high':
            # Prioritize speed over perfection
            criteria = {
                'completeness': 0.7,  # Can be partial
                'speed': 0.9,         # Must be fast
                'quality': 0.6        # Acceptable quality
            }
        elif user_expertise == 'expert':
            # Expert users want depth and technical accuracy
            criteria = {
                'technical_accuracy': 0.95,
                'depth': 0.9,
                'completeness': 0.85
            }
        elif resource_constraints == 'strict':
            # Minimize resource usage
            criteria = {
                'efficiency': 0.9,
                'cost': 0.9,
                'quality': 0.7
            }
        else:
            # Balanced criteria
            criteria = {
                'completeness': 0.8,
                'quality': 0.8,
                'efficiency': 0.7
            }

        # Evaluate against context-specific criteria
        scores = self._evaluate_criteria(agent_output, criteria)

        # Success if all criteria met
        success = all(score >= threshold
                     for score, threshold in zip(scores.values(), criteria.values()))

        return {
            'success': success,
            'context': context,
            'criteria_used': criteria,
            'scores': scores
        }
```

## Binary Success Metrics

The simplest success metric: did the agent succeed or fail?

### Basic Binary Evaluation

```python
class BinarySuccessEvaluator:
    """Evaluate with binary success/failure."""

    def evaluate(self, agent_output, ground_truth=None):
        """Return 1 for success, 0 for failure."""

        # Check multiple success conditions
        conditions = [
            self._check_goal_achieved(agent_output),
            self._check_output_valid(agent_output),
            self._check_no_errors(agent_output),
            self._check_constraints_met(agent_output)
        ]

        # Success requires ALL conditions
        success = all(conditions)

        return {
            'success': 1 if success else 0,
            'conditions_met': sum(conditions),
            'total_conditions': len(conditions),
            'details': {
                'goal_achieved': conditions[0],
                'output_valid': conditions[1],
                'no_errors': conditions[2],
                'constraints_met': conditions[3]
            }
        }

    def _check_goal_achieved(self, output):
        """Check if primary goal was achieved."""
        # Task-specific implementation
        return True

    def _check_output_valid(self, output):
        """Check if output format is valid."""
        return output is not None and len(output) > 0

    def _check_no_errors(self, output):
        """Check for error indicators in output."""
        error_indicators = ['error', 'failed', 'exception', 'timeout']
        return not any(indicator in str(output).lower()
                      for indicator in error_indicators)

    def _check_constraints_met(self, output):
        """Check if all constraints were respected."""
        # Check length limits, format requirements, etc.
        return True
```

### Success Rate Computation

```python
def compute_success_rate(agent, test_cases, n_runs_per_case=1):
    """
    Compute success rate across test cases.

    Args:
        agent: Agent to evaluate
        test_cases: List of test cases
        n_runs_per_case: Number of runs per test (for non-deterministic agents)

    Returns:
        Dict with success rate and statistics
    """

    results = []

    for test_case in test_cases:
        case_results = []

        # Run multiple times to handle non-determinism
        for _ in range(n_runs_per_case):
            output = agent.run(test_case['input'])
            success = evaluate_success(output, test_case)
            case_results.append(success)

        # Case succeeds if it succeeds in majority of runs
        case_success = sum(case_results) / len(case_results)
        results.append(case_success)

    # Compute statistics
    return {
        'overall_success_rate': np.mean(results),
        'strict_success_rate': np.mean([r == 1.0 for r in results]),  # 100% runs succeeded
        'partial_success_rate': np.mean([r > 0.5 for r in results]),  # >50% runs succeeded
        'std': np.std(results),
        'per_case_success': results,
        'total_cases': len(test_cases)
    }
```

### Pass@k Metric

```python
def compute_pass_at_k(agent, test_cases, k_values=[1, 5, 10]):
    """
    Compute pass@k: probability that at least one of k attempts succeeds.

    Common in code generation evaluation.
    """

    results = {}

    for k in k_values:
        successes = 0

        for test_case in test_cases:
            # Generate k attempts
            attempts = [agent.run(test_case['input']) for _ in range(k)]

            # Success if ANY attempt succeeds
            any_success = any(
                evaluate_success(attempt, test_case)
                for attempt in attempts
            )

            if any_success:
                successes += 1

        results[f'pass@{k}'] = successes / len(test_cases)

    return results

# Example usage:
# {
#   'pass@1': 0.65,   # 65% succeed on first try
#   'pass@5': 0.87,   # 87% succeed within 5 tries
#   'pass@10': 0.94   # 94% succeed within 10 tries
# }
```

## Partial Credit Scoring

Not all failures are equal. Partial credit rewards getting close to success.

### Partial Credit Framework

```python
class PartialCreditScorer:
    """Award partial credit for incomplete or imperfect solutions."""

    def score(self, agent_output, task):
        """
        Score from 0.0 (complete failure) to 1.0 (perfect success).
        """

        components = self._identify_components(task)
        component_scores = []

        for component in components:
            score = self._score_component(agent_output, component)
            weight = component['weight']
            component_scores.append(score * weight)

        # Total score is weighted sum
        total_score = sum(component_scores)

        return {
            'total_score': total_score,
            'component_scores': {
                comp['name']: score
                for comp, score in zip(components, component_scores)
            },
            'grade': self._score_to_grade(total_score)
        }

    def _identify_components(self, task):
        """Break task into scoreable components."""

        if task['type'] == 'research':
            return [
                {'name': 'finding_sources', 'weight': 0.3},
                {'name': 'comprehension', 'weight': 0.3},
                {'name': 'synthesis', 'weight': 0.4}
            ]

        elif task['type'] == 'coding':
            return [
                {'name': 'correctness', 'weight': 0.5},
                {'name': 'efficiency', 'weight': 0.2},
                {'name': 'code_quality', 'weight': 0.2},
                {'name': 'documentation', 'weight': 0.1}
            ]

        elif task['type'] == 'planning':
            return [
                {'name': 'goal_decomposition', 'weight': 0.3},
                {'name': 'action_sequencing', 'weight': 0.3},
                {'name': 'resource_allocation', 'weight': 0.2},
                {'name': 'contingency_planning', 'weight': 0.2}
            ]

    def _score_component(self, output, component):
        """Score individual component (0.0 to 1.0)."""
        # Component-specific scoring logic
        return 0.8  # Placeholder

    def _score_to_grade(self, score):
        """Convert numeric score to letter grade."""
        if score >= 0.9: return 'A'
        elif score >= 0.8: return 'B'
        elif score >= 0.7: return 'C'
        elif score >= 0.6: return 'D'
        else: return 'F'
```

### Progressive Achievement Scoring

```python
class ProgressiveScorer:
    """Score based on how far agent progressed toward goal."""

    def score(self, agent_trace, goal):
        """
        Score based on progress through required steps.
        """

        # Define milestone sequence
        milestones = self._define_milestones(goal)

        # Check which milestones were reached
        reached = []
        for i, milestone in enumerate(milestones):
            if self._milestone_reached(agent_trace, milestone):
                reached.append(i)
            else:
                # Stop at first unreached milestone
                break

        # Score is fraction of milestones reached
        progress_score = len(reached) / len(milestones)

        # Bonus for quality of each milestone
        quality_bonus = sum(
            self._milestone_quality(agent_trace, milestones[i])
            for i in reached
        ) / len(milestones)

        total_score = 0.7 * progress_score + 0.3 * quality_bonus

        return {
            'score': total_score,
            'milestones_reached': len(reached),
            'total_milestones': len(milestones),
            'progress_percentage': progress_score * 100,
            'reached_milestones': [milestones[i]['name'] for i in reached]
        }

    def _define_milestones(self, goal):
        """Define ordered milestones for goal."""

        if goal['type'] == 'booking_task':
            return [
                {'name': 'understand_request', 'description': 'Parse user intent'},
                {'name': 'search_options', 'description': 'Find available options'},
                {'name': 'filter_options', 'description': 'Apply constraints'},
                {'name': 'select_best', 'description': 'Choose best option'},
                {'name': 'execute_booking', 'description': 'Complete transaction'},
                {'name': 'confirm_user', 'description': 'Send confirmation'}
            ]

        # Other goal types...
        return []
```

### Rubric-Based Partial Credit

```python
class RubricScorer:
    """Use detailed rubrics for partial credit."""

    def __init__(self):
        self.rubrics = self._load_rubrics()

    def score(self, agent_output, task_type):
        """Score using task-specific rubric."""

        rubric = self.rubrics[task_type]
        scores = {}

        for criterion, levels in rubric.items():
            # Determine which level was achieved
            level_achieved = self._assess_level(agent_output, criterion, levels)
            scores[criterion] = level_achieved

        # Convert to numeric score
        numeric_score = self._rubric_to_score(scores, rubric)

        return {
            'numeric_score': numeric_score,
            'rubric_scores': scores,
            'feedback': self._generate_feedback(scores, rubric)
        }

    def _load_rubrics(self):
        """Load scoring rubrics for different task types."""
        return {
            'summarization': {
                'completeness': {
                    4: 'Covers all main points and key details',
                    3: 'Covers main points, missing some details',
                    2: 'Covers some main points, incomplete',
                    1: 'Missing most main points',
                    0: 'No relevant content'
                },
                'accuracy': {
                    4: 'All information accurate',
                    3: 'Mostly accurate, minor errors',
                    2: 'Several inaccuracies',
                    1: 'Major inaccuracies',
                    0: 'Mostly inaccurate'
                },
                'coherence': {
                    4: 'Highly logical flow',
                    3: 'Good structure',
                    2: 'Somewhat disjointed',
                    1: 'Poor organization',
                    0: 'Incoherent'
                },
                'conciseness': {
                    4: 'Perfectly concise',
                    3: 'Appropriately detailed',
                    2: 'Somewhat verbose',
                    1: 'Very verbose',
                    0: 'Excessively long'
                }
            }
        }

    def _assess_level(self, output, criterion, levels):
        """Assess which rubric level was achieved."""
        # Use heuristics or LLM to determine level
        # For now, placeholder
        return 3

    def _rubric_to_score(self, rubric_scores, rubric):
        """Convert rubric scores to 0-1 numeric score."""
        max_possible = sum(max(levels.keys()) for levels in rubric.values())
        actual = sum(rubric_scores.values())
        return actual / max_possible
```

## Quality Scoring

Beyond binary success, measure the quality of the output.

### Multi-Dimensional Quality

```python
class QualityScorer:
    """Evaluate output quality across multiple dimensions."""

    def score_quality(self, agent_output, reference=None):
        """
        Compute multi-dimensional quality scores.
        """

        quality_dimensions = {
            'accuracy': self._score_accuracy(agent_output, reference),
            'completeness': self._score_completeness(agent_output, reference),
            'relevance': self._score_relevance(agent_output, reference),
            'coherence': self._score_coherence(agent_output),
            'clarity': self._score_clarity(agent_output),
            'usefulness': self._score_usefulness(agent_output)
        }

        # Compute aggregate quality score
        aggregate = np.mean(list(quality_dimensions.values()))

        return {
            'aggregate_quality': aggregate,
            'dimensions': quality_dimensions,
            'quality_grade': self._categorize_quality(aggregate)
        }

    def _score_accuracy(self, output, reference):
        """Score factual accuracy."""
        if reference is None:
            # Use fact-checking tools
            return self._automated_fact_check(output)

        # Compare to reference
        facts_in_output = extract_facts(output)
        facts_in_reference = extract_facts(reference)

        correct_facts = sum(1 for fact in facts_in_output
                          if fact in facts_in_reference)

        precision = correct_facts / len(facts_in_output) if facts_in_output else 0
        recall = correct_facts / len(facts_in_reference) if facts_in_reference else 0

        # F1 score
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    def _score_completeness(self, output, reference):
        """Score how complete the output is."""
        if reference is None:
            # Check for expected components
            expected_components = self._identify_expected_components()
            present_components = self._identify_present_components(output)
            return len(present_components) / len(expected_components)

        # Compare coverage to reference
        output_topics = extract_topics(output)
        reference_topics = extract_topics(reference)

        coverage = len(output_topics & reference_topics) / len(reference_topics)
        return coverage

    def _score_relevance(self, output, reference):
        """Score relevance to task."""
        # Measure semantic similarity to task and reference
        task_similarity = semantic_similarity(output, self.task_description)

        if reference:
            ref_similarity = semantic_similarity(output, reference)
            return (task_similarity + ref_similarity) / 2

        return task_similarity

    def _score_coherence(self, output):
        """Score logical coherence and flow."""
        sentences = split_into_sentences(output)

        if len(sentences) < 2:
            return 1.0

        # Measure sentence-to-sentence coherence
        coherence_scores = []
        for i in range(len(sentences) - 1):
            similarity = semantic_similarity(sentences[i], sentences[i+1])
            coherence_scores.append(similarity)

        return np.mean(coherence_scores)

    def _score_clarity(self, output):
        """Score clarity and readability."""
        metrics = {
            'readability': compute_readability_score(output),
            'sentence_length': self._score_sentence_length(output),
            'vocabulary': self._score_vocabulary_appropriateness(output),
            'structure': self._score_structural_clarity(output)
        }

        return np.mean(list(metrics.values()))

    def _score_usefulness(self, output):
        """Score practical usefulness."""
        # Check for actionability, specificity, examples
        actionability = count_actionable_items(output) > 0
        specificity = measure_specificity(output)
        examples = count_examples(output) > 0

        return np.mean([actionability, specificity, examples])

    def _categorize_quality(self, score):
        """Categorize quality score."""
        if score >= 0.9: return 'excellent'
        elif score >= 0.75: return 'good'
        elif score >= 0.6: return 'acceptable'
        elif score >= 0.4: return 'poor'
        else: return 'unacceptable'
```

### Comparative Quality Scoring

```python
class ComparativeQualityScorer:
    """Score quality relative to baselines."""

    def score_relative_quality(self, agent_output, baselines):
        """
        Score quality relative to baseline outputs.

        Args:
            agent_output: Output to evaluate
            baselines: Dict of baseline outputs (e.g., {'gpt-4': output1, 'human': output2})

        Returns:
            Relative quality scores
        """

        scores = {}

        for baseline_name, baseline_output in baselines.items():
            comparison = self._compare_outputs(agent_output, baseline_output)
            scores[f'vs_{baseline_name}'] = comparison

        return {
            'baseline_comparisons': scores,
            'better_than_baselines': sum(1 for s in scores.values() if s > 0),
            'total_baselines': len(baselines),
            'average_relative_quality': np.mean(list(scores.values()))
        }

    def _compare_outputs(self, output1, output2):
        """
        Compare two outputs.

        Returns:
            > 0 if output1 is better
            < 0 if output2 is better
            ≈ 0 if similar quality
        """

        dimensions = ['accuracy', 'completeness', 'clarity', 'usefulness']

        differences = []
        for dim in dimensions:
            score1 = self._score_dimension(output1, dim)
            score2 = self._score_dimension(output2, dim)
            differences.append(score1 - score2)

        return np.mean(differences)
```

## Task-Specific Success Criteria

Different task types require different success definitions.

### Task Type Taxonomies

```python
class TaskSpecificEvaluator:
    """Evaluate success based on specific task type."""

    def evaluate(self, agent_output, task):
        """Route to appropriate evaluator based on task type."""

        evaluators = {
            'question_answering': self._evaluate_qa,
            'information_retrieval': self._evaluate_retrieval,
            'code_generation': self._evaluate_code,
            'planning': self._evaluate_planning,
            'booking': self._evaluate_booking,
            'writing': self._evaluate_writing,
            'analysis': self._evaluate_analysis
        }

        task_type = task['type']
        evaluator = evaluators.get(task_type, self._evaluate_generic)

        return evaluator(agent_output, task)

    def _evaluate_qa(self, output, task):
        """Evaluate question answering."""
        ground_truth = task.get('ground_truth')

        if ground_truth:
            # Exact match or semantic similarity
            exact_match = output.strip().lower() == ground_truth.strip().lower()
            semantic_match = semantic_similarity(output, ground_truth) > 0.9

            success = exact_match or semantic_match
        else:
            # Check for answer presence and correctness
            has_answer = len(output) > 0
            is_coherent = self._check_coherence(output)
            no_hallucination = self._check_no_hallucination(output)

            success = has_answer and is_coherent and no_hallucination

        return {'success': success, 'type': 'qa'}

    def _evaluate_retrieval(self, output, task):
        """Evaluate information retrieval."""
        relevant_docs = task.get('relevant_docs', [])

        retrieved = extract_retrieved_docs(output)

        # Compute precision and recall
        relevant_retrieved = set(retrieved) & set(relevant_docs)

        precision = len(relevant_retrieved) / len(retrieved) if retrieved else 0
        recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0

        # F1 score
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {
            'success': f1 > 0.7,  # Success threshold
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'type': 'retrieval'
        }

    def _evaluate_code(self, output, task):
        """Evaluate code generation."""
        test_cases = task.get('test_cases', [])

        # Extract code from output
        code = extract_code(output)

        if not code:
            return {'success': False, 'reason': 'no_code_found'}

        # Run test cases
        test_results = []
        for test in test_cases:
            try:
                result = execute_code(code, test['input'])
                passed = result == test['expected_output']
                test_results.append(passed)
            except Exception as e:
                test_results.append(False)

        pass_rate = sum(test_results) / len(test_results) if test_results else 0

        return {
            'success': pass_rate == 1.0,  # All tests must pass
            'pass_rate': pass_rate,
            'tests_passed': sum(test_results),
            'total_tests': len(test_results),
            'type': 'code'
        }

    def _evaluate_planning(self, output, task):
        """Evaluate planning tasks."""
        plan = extract_plan(output)

        criteria = {
            'has_clear_steps': len(plan) > 0,
            'steps_are_actionable': all(is_actionable(step) for step in plan),
            'logical_order': self._check_logical_ordering(plan),
            'achieves_goal': self._check_goal_achievement(plan, task['goal']),
            'handles_constraints': self._check_constraints(plan, task.get('constraints', []))
        }

        success = all(criteria.values())

        return {
            'success': success,
            'criteria_met': criteria,
            'plan_quality': sum(criteria.values()) / len(criteria),
            'type': 'planning'
        }

    def _evaluate_booking(self, output, task):
        """Evaluate booking tasks."""
        booking_details = extract_booking_details(output)

        required_fields = ['date', 'time', 'location', 'confirmation']

        criteria = {
            'all_fields_present': all(field in booking_details for field in required_fields),
            'date_valid': self._validate_date(booking_details.get('date')),
            'time_valid': self._validate_time(booking_details.get('time')),
            'preferences_met': self._check_preferences(booking_details, task.get('preferences', {})),
            'confirmation_received': 'confirmation' in booking_details
        }

        return {
            'success': all(criteria.values()),
            'booking_completed': True if all(criteria.values()) else False,
            'criteria_met': criteria,
            'type': 'booking'
        }

    def _evaluate_writing(self, output, task):
        """Evaluate writing tasks."""
        requirements = task.get('requirements', {})

        criteria = {
            'meets_length': self._check_length_requirement(output, requirements),
            'appropriate_tone': self._check_tone(output, requirements.get('tone')),
            'covers_topics': self._check_topic_coverage(output, requirements.get('topics', [])),
            'grammatical': self._check_grammar(output),
            'coherent': self._check_coherence(output)
        }

        quality_score = sum(criteria.values()) / len(criteria)

        return {
            'success': quality_score >= 0.8,
            'quality_score': quality_score,
            'criteria_met': criteria,
            'type': 'writing'
        }
```

## Multi-Objective Success

Many tasks have multiple objectives that may conflict.

### Pareto Optimality

```python
class MultiObjectiveEvaluator:
    """Evaluate success across multiple potentially conflicting objectives."""

    def evaluate_multi_objective(self, agent_output, objectives):
        """
        Evaluate multiple objectives.

        Args:
            agent_output: Agent's output
            objectives: List of (objective_name, weight, direction)
                       direction: 'maximize' or 'minimize'

        Returns:
            Multi-objective scores and Pareto analysis
        """

        scores = {}

        for obj_name, weight, direction in objectives:
            score = self._evaluate_objective(agent_output, obj_name)

            # Normalize based on direction
            if direction == 'minimize':
                score = 1.0 - score  # Invert for minimization

            scores[obj_name] = {
                'score': score,
                'weight': weight,
                'direction': direction
            }

        # Compute weighted aggregate
        weighted_score = sum(
            obj['score'] * obj['weight']
            for obj in scores.values()
        )

        return {
            'objective_scores': scores,
            'weighted_aggregate': weighted_score,
            'success': weighted_score >= 0.7
        }

    def compare_pareto(self, outputs, objectives):
        """
        Compare outputs using Pareto dominance.

        Output A dominates output B if:
        - A is at least as good as B on all objectives
        - A is strictly better than B on at least one objective
        """

        # Evaluate all outputs
        evaluations = [
            self.evaluate_multi_objective(output, objectives)
            for output in outputs
        ]

        # Find Pareto frontier
        pareto_frontier = []

        for i, eval_i in enumerate(evaluations):
            dominated = False

            for j, eval_j in enumerate(evaluations):
                if i != j and self._dominates(eval_j, eval_i, objectives):
                    dominated = True
                    break

            if not dominated:
                pareto_frontier.append(i)

        return {
            'pareto_frontier': pareto_frontier,
            'pareto_optimal_outputs': [outputs[i] for i in pareto_frontier],
            'evaluations': evaluations
        }

    def _dominates(self, eval_a, eval_b, objectives):
        """Check if evaluation A dominates evaluation B."""

        scores_a = [eval_a['objective_scores'][obj[0]]['score'] for obj in objectives]
        scores_b = [eval_b['objective_scores'][obj[0]]['score'] for obj in objectives]

        # A dominates B if:
        # 1. A >= B on all objectives
        # 2. A > B on at least one objective

        all_greater_or_equal = all(a >= b for a, b in zip(scores_a, scores_b))
        any_strictly_greater = any(a > b for a, b in zip(scores_a, scores_b))

        return all_greater_or_equal and any_strictly_greater
```

### Trade-off Analysis

```python
class TradeoffAnalyzer:
    """Analyze trade-offs in multi-objective success."""

    def analyze_tradeoffs(self, agent_outputs, objectives):
        """
        Analyze how objectives trade off against each other.
        """

        # Evaluate all outputs on all objectives
        evaluations = []
        for output in agent_outputs:
            scores = {
                obj_name: self._score_objective(output, obj_name)
                for obj_name, _, _ in objectives
            }
            evaluations.append(scores)

        # Compute correlations between objectives
        obj_names = [obj[0] for obj in objectives]
        correlations = {}

        for i, obj1 in enumerate(obj_names):
            for j, obj2 in enumerate(obj_names[i+1:], i+1):
                scores1 = [ev[obj1] for ev in evaluations]
                scores2 = [ev[obj2] for ev in evaluations]

                corr = np.corrcoef(scores1, scores2)[0, 1]
                correlations[f'{obj1}_vs_{obj2}'] = corr

        # Identify trade-offs (negative correlations)
        tradeoffs = {
            pair: corr
            for pair, corr in correlations.items()
            if corr < -0.3
        }

        return {
            'correlations': correlations,
            'tradeoffs': tradeoffs,
            'synergies': {
                pair: corr
                for pair, corr in correlations.items()
                if corr > 0.3
            }
        }
```

## Success Rates and Statistics

Computing success rates with statistical rigor.

### Confidence Intervals

```python
def compute_success_rate_with_ci(agent, test_cases, confidence=0.95, n_runs=20):
    """
    Compute success rate with confidence interval.
    """

    successes = []

    for test_case in test_cases:
        # Multiple runs per test case
        case_successes = []
        for _ in range(n_runs):
            output = agent.run(test_case['input'])
            success = evaluate_success(output, test_case)
            case_successes.append(success)

        # Take mean success rate for this case
        successes.append(np.mean(case_successes))

    # Compute statistics
    mean_success = np.mean(successes)
    std_success = np.std(successes)

    # Compute confidence interval
    z_score = 1.96 if confidence == 0.95 else 2.576  # For 95% or 99%
    margin_of_error = z_score * (std_success / np.sqrt(len(successes)))

    ci_lower = max(0, mean_success - margin_of_error)
    ci_upper = min(1, mean_success + margin_of_error)

    return {
        'success_rate': mean_success,
        'confidence_level': confidence,
        'confidence_interval': (ci_lower, ci_upper),
        'std': std_success,
        'n_samples': len(successes)
    }
```

### Statistical Significance Testing

```python
def compare_agents_statistically(agent_a, agent_b, test_cases, alpha=0.05):
    """
    Compare two agents with statistical significance testing.
    """

    from scipy import stats

    # Run both agents on all test cases
    scores_a = []
    scores_b = []

    for test_case in test_cases:
        # Multiple runs to handle non-determinism
        runs_a = [agent_a.run(test_case['input']) for _ in range(10)]
        runs_b = [agent_b.run(test_case['input']) for _ in range(10)]

        scores_a.extend([evaluate_success(r, test_case) for r in runs_a])
        scores_b.extend([evaluate_success(r, test_case) for r in runs_b])

    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(scores_a, scores_b)

    # Compute effect size (Cohen's d)
    mean_a = np.mean(scores_a)
    mean_b = np.mean(scores_b)
    pooled_std = np.sqrt((np.var(scores_a) + np.var(scores_b)) / 2)
    cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0

    return {
        'agent_a_mean': mean_a,
        'agent_b_mean': mean_b,
        'difference': mean_a - mean_b,
        'statistically_significant': p_value < alpha,
        'p_value': p_value,
        't_statistic': t_statistic,
        'cohens_d': cohens_d,
        'effect_size': classify_effect_size(cohens_d),
        'winner': 'agent_a' if mean_a > mean_b else 'agent_b',
        'confidence': 1 - p_value
    }

def classify_effect_size(cohens_d):
    """Classify effect size magnitude."""
    abs_d = abs(cohens_d)
    if abs_d < 0.2: return 'negligible'
    elif abs_d < 0.5: return 'small'
    elif abs_d < 0.8: return 'medium'
    else: return 'large'
```

## Goal Achievement Metrics

Measuring how well agents achieve stated goals.

### Goal Decomposition

```python
class GoalAchievementEvaluator:
    """Evaluate goal achievement through decomposition."""

    def evaluate_goal_achievement(self, agent_output, goal):
        """
        Break goal into sub-goals and evaluate achievement.
        """

        # Decompose goal
        sub_goals = self._decompose_goal(goal)

        # Evaluate each sub-goal
        sub_goal_achievements = {}
        for sub_goal in sub_goals:
            achieved = self._check_sub_goal(agent_output, sub_goal)
            sub_goal_achievements[sub_goal['name']] = achieved

        # Compute overall achievement
        achievement_rate = sum(sub_goal_achievements.values()) / len(sub_goals)

        # Check critical vs optional sub-goals
        critical_goals = [sg for sg in sub_goals if sg['critical']]
        critical_achieved = all(
            sub_goal_achievements[sg['name']]
            for sg in critical_goals
        )

        return {
            'overall_achievement': achievement_rate,
            'critical_achieved': critical_achieved,
            'sub_goal_achievements': sub_goal_achievements,
            'success': critical_achieved and achievement_rate >= 0.8
        }

    def _decompose_goal(self, goal):
        """Decompose high-level goal into sub-goals."""

        # Example: "Plan a trip to Japan"
        if 'trip' in goal.lower():
            return [
                {'name': 'select_destination', 'critical': True},
                {'name': 'book_flights', 'critical': True},
                {'name': 'book_accommodation', 'critical': True},
                {'name': 'plan_activities', 'critical': False},
                {'name': 'check_requirements', 'critical': True},
                {'name': 'organize_itinerary', 'critical': False}
            ]

        # Generic decomposition
        return self._generic_decomposition(goal)
```

### Intent Satisfaction

```python
class IntentSatisfactionScorer:
    """Score how well agent satisfied user's underlying intent."""

    def score_intent_satisfaction(self, agent_output, user_input, context=None):
        """
        Score intent satisfaction (may differ from literal task completion).
        """

        # Extract user intent (may be implicit)
        explicit_intent = extract_explicit_intent(user_input)
        implicit_intent = infer_implicit_intent(user_input, context)

        # Score satisfaction of both
        explicit_score = self._score_explicit_intent(agent_output, explicit_intent)
        implicit_score = self._score_implicit_intent(agent_output, implicit_intent)

        # Combine scores
        total_score = 0.6 * explicit_score + 0.4 * implicit_score

        return {
            'total_intent_satisfaction': total_score,
            'explicit_intent_score': explicit_score,
            'implicit_intent_score': implicit_score,
            'intent_analysis': {
                'explicit': explicit_intent,
                'implicit': implicit_intent
            }
        }

    def _score_explicit_intent(self, output, intent):
        """Score satisfaction of explicitly stated intent."""
        # Check if explicit requirements are met
        return 0.8  # Placeholder

    def _score_implicit_intent(self, output, intent):
        """Score satisfaction of implicit user needs."""
        # Check if implicit needs are addressed
        return 0.7  # Placeholder
```

## User-Centric Success

Success from the user's perspective.

### User Satisfaction Modeling

```python
class UserSatisfactionPredictor:
    """Predict user satisfaction with agent output."""

    def __init__(self, satisfaction_model=None):
        self.model = satisfaction_model or self._default_model()

    def predict_satisfaction(self, agent_output, user_context):
        """
        Predict user satisfaction score (0-1).
        """

        features = self._extract_features(agent_output, user_context)

        # Use learned model or heuristics
        if self.model:
            satisfaction = self.model.predict(features)
        else:
            satisfaction = self._heuristic_satisfaction(features)

        return {
            'predicted_satisfaction': satisfaction,
            'features': features,
            'confidence': self._compute_confidence(features)
        }

    def _extract_features(self, output, context):
        """Extract features for satisfaction prediction."""
        return {
            'output_length': len(output),
            'response_time': context.get('response_time', 0),
            'clarity_score': compute_clarity(output),
            'completeness_score': compute_completeness(output),
            'relevance_score': compute_relevance(output, context),
            'user_expertise': context.get('user_expertise', 'intermediate'),
            'task_complexity': context.get('task_complexity', 'medium')
        }

    def _heuristic_satisfaction(self, features):
        """Heuristic-based satisfaction prediction."""

        # Factors that increase satisfaction
        positive_factors = [
            features['clarity_score'] > 0.7,
            features['completeness_score'] > 0.7,
            features['relevance_score'] > 0.8,
            features['response_time'] < 30  # seconds
        ]

        # Factors that decrease satisfaction
        negative_factors = [
            features['output_length'] < 10,  # Too short
            features['response_time'] > 60,  # Too slow
            features['clarity_score'] < 0.5  # Unclear
        ]

        positive_score = sum(positive_factors) / len(positive_factors)
        negative_score = sum(negative_factors) / len(negative_factors)

        return max(0, positive_score - negative_score)
```

### Value-Based Success

```python
class ValueBasedEvaluator:
    """Evaluate success based on value delivered to user."""

    def evaluate_value(self, agent_output, user_context):
        """
        Measure value delivered beyond task completion.
        """

        value_dimensions = {
            # Time saved
            'time_saved': self._estimate_time_saved(agent_output, user_context),

            # Quality improvement
            'quality_gain': self._estimate_quality_gain(agent_output, user_context),

            # Learning/insight provided
            'learning_value': self._estimate_learning_value(agent_output),

            # Problem prevention
            'error_prevention': self._estimate_errors_prevented(agent_output),

            # Opportunity identification
            'opportunities_found': self._count_opportunities(agent_output)
        }

        # Convert to monetary value if possible
        monetary_value = self._estimate_monetary_value(value_dimensions, user_context)

        return {
            'value_dimensions': value_dimensions,
            'estimated_monetary_value': monetary_value,
            'value_score': self._aggregate_value(value_dimensions)
        }
```

## Summary

Success metrics are the foundation of agent evaluation, but "success" is multifaceted and context-dependent:

**Key Concepts**:

- Success is not binary - use partial credit and quality scoring
- Different tasks require different success criteria
- Multiple objectives may conflict - understand trade-offs
- Statistical rigor is essential for reliable comparisons
- User satisfaction is the ultimate measure

**Metric Types**:

- **Binary**: Pass/fail, success rate, pass@k
- **Partial Credit**: Component scoring, progressive achievement, rubrics
- **Quality**: Multi-dimensional quality assessment
- **Task-Specific**: QA, retrieval, code generation, planning
- **Multi-Objective**: Pareto optimality, trade-off analysis
- **Goal-Based**: Sub-goal achievement, intent satisfaction
- **User-Centric**: Satisfaction prediction, value delivered

**Best Practices**:

- Define success criteria before evaluation
- Use multiple complementary metrics
- Account for non-determinism with multiple runs
- Consider context when defining success
- Measure both task completion and quality
- Include user perspective in evaluation

Success metrics should be actionable, helping you understand not just whether an agent succeeded, but why and how to improve it.

## Next Steps

- **[Reasoning Quality](reasoning-quality.md)**: Evaluate reasoning processes
- **[Tool Use Evaluation](tool-use-evaluation.md)**: Assess tool selection and usage
- **[Efficiency Metrics](efficiency-metrics.md)**: Measure resource consumption
- **[Robustness Testing](robustness.md)**: Test reliability under stress
- **[Benchmarks](benchmarks.md)**: Standard evaluation datasets
- **[Human Evaluation](human-evaluation.md)**: Incorporate human judgment
- **[Continuous Evaluation](continuous-evaluation.md)**: Monitor in production
