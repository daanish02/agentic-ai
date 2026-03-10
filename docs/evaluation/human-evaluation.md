# Human Evaluation

## Table of Contents

- [Introduction](#introduction)
- [When Human Evaluation is Necessary](#when-human-evaluation-is-necessary)
- [Evaluation Dimensions](#evaluation-dimensions)
- [Designing Human Evaluation Studies](#designing-human-evaluation-studies)
- [Evaluation Protocols](#evaluation-protocols)
- [Rating Scales](#rating-scales)
- [Comparative Evaluation](#comparative-evaluation)
- [Inter-Rater Reliability](#inter-rater-reliability)
- [Evaluation Guidelines](#evaluation-guidelines)
- [Crowdsourced Evaluation](#crowdsourced-evaluation)
- [Expert vs Lay Evaluation](#expert-vs-lay-evaluation)
- [Bias in Human Evaluation](#bias-in-human-evaluation)
- [Cost-Benefit Analysis](#cost-benefit-analysis)
- [Hybrid Approaches](#hybrid-approaches)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Some qualities can only be judged by humans: usefulness, trustworthiness, coherence, user satisfaction. While automated metrics provide scale, **human evaluation provides ground truth**.

The challenge: human evaluation is expensive, slow, and subjective. This guide covers when to use human evaluation, how to design effective studies, and how to ensure reliability.

## When Human Evaluation is Necessary

Cases where automated metrics are insufficient.

### Scenarios Requiring Human Judgment

```python
class HumanEvaluationDecider:
    """Decide when human evaluation is necessary."""

    def needs_human_evaluation(self, task_context):
        """Determine if human evaluation is needed."""

        # High stakes decisions
        if task_context.get('safety_critical'):
            return {
                'needed': True,
                'reason': 'Safety-critical decisions require human oversight',
                'urgency': 'high'
            }

        # Subjective quality
        if task_context.get('subjective_quality'):
            return {
                'needed': True,
                'reason': 'Quality is subjective and context-dependent',
                'urgency': 'medium'
            }

        # User-facing outputs
        if task_context.get('user_facing'):
            return {
                'needed': True,
                'reason': 'User satisfaction best assessed by humans',
                'urgency': 'medium'
            }

        # Novel tasks
        if task_context.get('novel_task') and not task_context.get('has_metrics'):
            return {
                'needed': True,
                'reason': 'No established automated metrics for this task',
                'urgency': 'high'
            }

        # Can use automated metrics
        return {
            'needed': False,
            'reason': 'Automated metrics sufficient',
            'suggestion': 'Occasional human validation recommended'
        }
```

## Evaluation Dimensions

What aspects should humans evaluate?

### Key Evaluation Dimensions

```python
class HumanEvaluationDimensions:
    """Dimensions for human evaluation."""

    dimensions = {
        'usefulness': {
            'question': 'How useful is this output to you?',
            'scale': '1 (not useful) to 5 (very useful)',
            'description': 'Practical value and actionability'
        },

        'correctness': {
            'question': 'Is the information correct?',
            'scale': '1 (incorrect) to 5 (fully correct)',
            'description': 'Factual accuracy and validity'
        },

        'coherence': {
            'question': 'How coherent and logical is the output?',
            'scale': '1 (incoherent) to 5 (highly coherent)',
            'description': 'Logical flow and consistency'
        },

        'completeness': {
            'question': 'Is the output complete and thorough?',
            'scale': '1 (incomplete) to 5 (comprehensive)',
            'description': 'Coverage of required aspects'
        },

        'clarity': {
            'question': 'How clear and understandable is the output?',
            'scale': '1 (unclear) to 5 (very clear)',
            'description': 'Ease of understanding'
        },

        'trustworthiness': {
            'question': 'How much do you trust this output?',
            'scale': '1 (don\'t trust) to 5 (fully trust)',
            'description': 'Confidence in reliability'
        },

        'satisfaction': {
            'question': 'How satisfied are you with this output?',
            'scale': '1 (very unsatisfied) to 5 (very satisfied)',
            'description': 'Overall satisfaction'
        }
    }

    def create_evaluation_form(self, dimensions_to_use):
        """Create evaluation form with selected dimensions."""

        form = {
            'instructions': 'Please rate the following output on each dimension.',
            'dimensions': {}
        }

        for dim in dimensions_to_use:
            if dim in self.dimensions:
                form['dimensions'][dim] = self.dimensions[dim]

        return form
```

## Designing Human Evaluation Studies

Creating effective evaluation protocols.

### Study Design

```python
class HumanEvaluationStudy:
    """Design and conduct human evaluation study."""

    def __init__(self, study_name, objectives):
        self.study_name = study_name
        self.objectives = objectives
        self.protocol = self._design_protocol()

    def _design_protocol(self):
        """Design evaluation protocol."""

        return {
            'sample_size': self._determine_sample_size(),
            'evaluator_criteria': self._define_evaluator_criteria(),
            'task_design': self._design_tasks(),
            'rating_scales': self._choose_rating_scales(),
            'guidelines': self._create_guidelines(),
            'quality_control': self._design_quality_control()
        }

    def _determine_sample_size(self):
        """Calculate required sample size for statistical power."""

        # Effect size, confidence level, power
        effect_size = 0.5  # Medium effect
        alpha = 0.05  # 95% confidence
        power = 0.8  # 80% power

        # Sample size calculation (simplified)
        n = (2 * (1.96 + 0.84)**2) / (effect_size**2)

        return {
            'min_samples': int(np.ceil(n)),
            'recommended_samples': int(np.ceil(n * 1.5)),
            'evaluators_per_sample': 3  # For inter-rater reliability
        }

    def conduct_study(self, agent_outputs, evaluators):
        """Conduct the evaluation study."""

        results = []

        for output in agent_outputs:
            # Get evaluations from multiple raters
            evaluations = []
            for evaluator in evaluators:
                evaluation = self._get_evaluation(output, evaluator)
                evaluations.append(evaluation)

            # Aggregate ratings
            aggregated = self._aggregate_evaluations(evaluations)
            results.append(aggregated)

        # Compute inter-rater reliability
        reliability = self._compute_reliability(results)

        return {
            'results': results,
            'reliability': reliability,
            'summary_statistics': self._compute_statistics(results)
        }
```

## Inter-Rater Reliability

Ensuring consistency across evaluators.

### Reliability Metrics

```python
class InterRaterReliability:
    """Compute inter-rater reliability metrics."""

    def compute_cohens_kappa(self, rater1_scores, rater2_scores):
        """
        Cohen's Kappa for agreement between two raters.

        Kappa interpretation:
        < 0: No agreement
        0-0.20: Slight agreement
        0.21-0.40: Fair agreement
        0.41-0.60: Moderate agreement
        0.61-0.80: Substantial agreement
        0.81-1.00: Almost perfect agreement
        """
        from sklearn.metrics import cohen_kappa_score

        kappa = cohen_kappa_score(rater1_scores, rater2_scores)

        return {
            'kappa': kappa,
            'interpretation': self._interpret_kappa(kappa)
        }

    def compute_icc(self, ratings_matrix):
        """
        Intraclass Correlation Coefficient for multiple raters.

        ratings_matrix: n_samples x n_raters
        """
        import pingouin as pg

        icc = pg.intraclass_corr(
            data=ratings_matrix,
            targets='sample',
            raters='rater',
            ratings='rating'
        )

        return {
            'icc': icc['ICC'][2],  # ICC(2,1) - common choice
            'confidence_interval': (icc['CI95%'][2][0], icc['CI95%'][2][1])
        }

    def _interpret_kappa(self, kappa):
        """Interpret kappa value."""
        if kappa < 0: return 'no_agreement'
        elif kappa < 0.20: return 'slight'
        elif kappa < 0.40: return 'fair'
        elif kappa < 0.60: return 'moderate'
        elif kappa < 0.80: return 'substantial'
        else: return 'almost_perfect'
```

## Comparative Evaluation

Comparing multiple systems.

### Pairwise Comparison

```python
class PairwiseComparison:
    """Conduct pairwise comparisons of agent outputs."""

    def conduct_comparison(self, output_a, output_b, evaluators):
        """
        Have evaluators compare two outputs.

        Returns preference and confidence.
        """

        preferences = []

        for evaluator in evaluators:
            # Present outputs in random order to avoid position bias
            if random.random() < 0.5:
                first, second = output_a, output_b
                preference_map = {first: 'a', second: 'b'}
            else:
                first, second = output_b, output_a
                preference_map = {first: 'b', second: 'a'}

            # Get preference
            preference = evaluator.compare(first, second)

            preferences.append({
                'prefers': preference_map[preference['choice']],
                'confidence': preference['confidence'],
                'reasoning': preference['reasoning']
            })

        # Aggregate preferences
        a_wins = sum(1 for p in preferences if p['prefers'] == 'a')
        b_wins = sum(1 for p in preferences if p['prefers'] == 'b')

        return {
            'winner': 'a' if a_wins > b_wins else 'b',
            'win_rate': max(a_wins, b_wins) / len(preferences),
            'preferences': preferences,
            'statistical_significance': self._test_significance(a_wins, b_wins)
        }
```

## Bias in Human Evaluation

Addressing common biases.

### Bias Mitigation

```python
class BiasMitigation:
    """Strategies to mitigate bias in human evaluation."""

    strategies = {
        'position_bias': {
            'description': 'Preferring first or last shown option',
            'mitigation': 'Randomize presentation order'
        },

        'anchoring': {
            'description': 'First rating influences subsequent ratings',
            'mitigation': 'Randomize order, use within-subject design carefully'
        },

        'confirmation_bias': {
            'description': 'Seeking evidence that confirms expectations',
            'mitigation': 'Blind evaluation, diverse evaluators'
        },

        'halo_effect': {
            'description': 'One positive trait influences overall rating',
            'mitigation': 'Evaluate dimensions independently'
        },

        'leniency_bias': {
            'description': 'Rating too generously',
            'mitigation': 'Clear standards, calibration, multiple raters'
        }
    }

    def apply_mitigation_strategies(self, evaluation_design):
        """Apply bias mitigation to evaluation design."""

        mitigated_design = {
            'blinding': 'Evaluators don\'t know which system generated output',
            'randomization': 'Randomize presentation order',
            'calibration': 'Train evaluators with examples',
            'multiple_raters': 'Use at least 3 raters per sample',
            'clear_rubrics': 'Provide detailed evaluation criteria',
            'diverse_evaluators': 'Mix of backgrounds and expertise'
        }

        return mitigated_design
```

## Summary

Human evaluation provides essential qualitative assessment:

**When to Use**:

- Subjective quality assessment
- User satisfaction measurement
- Safety-critical decisions
- Novel tasks without metrics

**Best Practices**:

- Use clear rating scales
- Multiple evaluators per sample
- Measure inter-rater reliability
- Mitigate biases
- Provide detailed guidelines
- Consider cost-benefit trade-offs

**Key Metrics**:

- Inter-rater reliability (Kappa, ICC)
- Preference rates (pairwise comparison)
- Satisfaction scores
- Qualitative feedback

Human evaluation is the gold standard but use it strategically due to cost.

## Next Steps

- **[Continuous Evaluation](continuous-evaluation.md)**: Production monitoring
- **[Success Metrics](success-metrics.md)**: Defining success
- **[Benchmarks](benchmarks.md)**: Standard evaluations
