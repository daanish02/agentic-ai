# Reasoning Quality

## Table of Contents

- [Introduction](#introduction)
- [What is Reasoning Quality?](#what-is-reasoning-quality)
- [Logical Correctness](#logical-correctness)
- [Reasoning Coherence](#reasoning-coherence)
- [Reasoning Depth](#reasoning-depth)
- [Reasoning Thoroughness](#reasoning-thoroughness)
- [Process vs Outcome](#process-vs-outcome)
- [Reasoning Traces](#reasoning-traces)
- [Logical Fallacies](#logical-fallacies)
- [Reasoning Patterns](#reasoning-patterns)
- [Causal Reasoning](#causal-reasoning)
- [Counterfactual Reasoning](#counterfactual-reasoning)
- [Reasoning Under Uncertainty](#reasoning-under-uncertainty)
- [Multi-Step Reasoning](#multi-step-reasoning)
- [Reasoning Evaluation Frameworks](#reasoning-evaluation-frameworks)
- [Automated Reasoning Evaluation](#automated-reasoning-evaluation)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

An agent might arrive at the correct answer through flawed reasoning, or use sound reasoning but make a small error leading to a wrong answer. **Reasoning quality** evaluates the thought process itself, not just the final output.

Consider these scenarios:

```python
# Scenario 1: Right answer, wrong reasoning
Question: "What's 7 × 8?"
Reasoning: "I'll add 7 eight times: 7+7=14, 14+7=21, 21+7=28, 28+7=35, 35+7=42, 42+7=49, 49+7=66... wait that's only 7 times. Add one more 7: 66-10=56"
Answer: 56  # Correct by luck, reasoning is nonsense

# Scenario 2: Sound reasoning, small error
Question: "What's 7 × 8?"
Reasoning: "I'll use the distributive property: 7 × 8 = 7 × (10 - 2) = 70 - 14 = 57"
Answer: 57  # Wrong answer, but reasoning approach is sound (arithmetic error)

# Scenario 3: Both correct
Question: "What's 7 × 8?"
Reasoning: "I'll use the distributive property: 7 × 8 = 7 × (10 - 2) = 70 - 14 = 56"
Answer: 56  # Correct answer AND sound reasoning
```

Evaluating reasoning quality is crucial because:

- **Reliability**: Sound reasoning is more likely to generalize
- **Trustworthiness**: We can trust agents that think correctly
- **Debuggability**: Understanding reasoning helps fix issues
- **Learning**: Good reasoning can be reinforced
- **Safety**: Flawed reasoning can lead to dangerous actions

This guide covers how to evaluate agent reasoning: from logical correctness to coherence, from depth to thoroughness, and from simple checks to sophisticated automated evaluation.

> "The right answer for the wrong reasons is still the wrong answer in the long run."

## What is Reasoning Quality?

Reasoning quality encompasses multiple dimensions beyond correctness.

### Dimensions of Reasoning Quality

```python
class ReasoningQualityFramework:
    """Framework for evaluating reasoning quality."""

    dimensions = {
        'logical_correctness': {
            'description': 'Are the logical steps valid?',
            'aspects': [
                'valid_inferences',
                'no_logical_fallacies',
                'sound_deductions',
                'proper_use_of_logic'
            ]
        },

        'coherence': {
            'description': 'Do the steps flow logically?',
            'aspects': [
                'connected_steps',
                'consistent_reasoning',
                'no_contradictions',
                'unified_narrative'
            ]
        },

        'completeness': {
            'description': 'Are all necessary steps present?',
            'aspects': [
                'no_unjustified_leaps',
                'explicit_assumptions',
                'thorough_analysis',
                'complete_chain'
            ]
        },

        'depth': {
            'description': 'How deeply is the problem analyzed?',
            'aspects': [
                'multi_level_analysis',
                'root_cause_identification',
                'consideration_of_alternatives',
                'second_order_effects'
            ]
        },

        'relevance': {
            'description': 'Is the reasoning focused on the task?',
            'aspects': [
                'stays_on_topic',
                'no_tangents',
                'addresses_key_points',
                'efficient_reasoning'
            ]
        },

        'clarity': {
            'description': 'Is the reasoning easy to follow?',
            'aspects': [
                'well_structured',
                'explicit_transitions',
                'clear_language',
                'understandable_flow'
            ]
        }
    }

    def evaluate_reasoning(self, reasoning_trace):
        """Evaluate reasoning across all dimensions."""

        scores = {}

        for dimension, spec in self.dimensions.items():
            dimension_score = self._evaluate_dimension(
                reasoning_trace,
                dimension,
                spec['aspects']
            )
            scores[dimension] = dimension_score

        # Compute aggregate
        aggregate = np.mean(list(scores.values()))

        return {
            'dimension_scores': scores,
            'aggregate_quality': aggregate,
            'quality_grade': self._score_to_grade(aggregate)
        }

    def _score_to_grade(self, score):
        if score >= 0.9: return 'excellent'
        elif score >= 0.75: return 'good'
        elif score >= 0.6: return 'acceptable'
        elif score >= 0.4: return 'poor'
        else: return 'unacceptable'
```

### The Reasoning Quality Spectrum

```
Low Quality                                           High Quality
├─────────────┼─────────────┼─────────────┼─────────────┤
│             │             │             │             │
Random      Illogical    Superficial   Sound but    Rigorous &
guessing    but arrives   reasoning    incomplete   complete
            at answer                  reasoning    reasoning
```

## Logical Correctness

The foundation of reasoning quality: are the logical inferences valid?

### Evaluating Logical Validity

```python
class LogicalCorrectnessEvaluator:
    """Evaluate logical validity of reasoning steps."""

    def evaluate_logical_correctness(self, reasoning_trace):
        """Check if reasoning follows valid logical rules."""

        # Extract reasoning steps
        steps = self._parse_reasoning_steps(reasoning_trace)

        # Check each inference
        inference_validity = []
        for i in range(len(steps) - 1):
            premise = steps[i]
            conclusion = steps[i + 1]

            validity = self._check_inference(premise, conclusion)
            inference_validity.append(validity)

        # Check for logical fallacies
        fallacies = self._detect_fallacies(reasoning_trace)

        # Compute score
        valid_inferences = sum(inference_validity) / len(inference_validity) if inference_validity else 1.0
        fallacy_penalty = len(fallacies) * 0.1

        score = max(0, valid_inferences - fallacy_penalty)

        return {
            'logical_correctness_score': score,
            'valid_inferences': sum(inference_validity),
            'total_inferences': len(inference_validity),
            'fallacies_detected': fallacies,
            'issues': self._identify_issues(inference_validity, fallacies)
        }

    def _check_inference(self, premise, conclusion):
        """
        Check if conclusion follows from premise.

        Types of valid inferences:
        - Modus ponens: If P then Q, P, therefore Q
        - Modus tollens: If P then Q, not Q, therefore not P
        - Hypothetical syllogism: If P then Q, if Q then R, therefore if P then R
        - Disjunctive syllogism: P or Q, not P, therefore Q
        """

        # Simplified check (in practice, use formal logic tools)
        patterns = [
            self._check_modus_ponens(premise, conclusion),
            self._check_modus_tollens(premise, conclusion),
            self._check_syllogism(premise, conclusion),
            self._check_deductive_valid(premise, conclusion)
        ]

        return any(patterns)

    def _check_modus_ponens(self, premise, conclusion):
        """Check for modus ponens pattern."""
        # If premise is "If A then B" and "A"
        # Then conclusion should be "B"

        # Simplified pattern matching
        if 'if' in premise.lower() and 'then' in premise.lower():
            # Extract A and B
            parts = self._parse_conditional(premise)
            if parts:
                antecedent, consequent = parts
                # Check if conclusion matches consequent
                return self._semantic_match(conclusion, consequent)

        return False

    def _detect_fallacies(self, reasoning):
        """Detect common logical fallacies."""

        fallacies = []

        # Ad hominem
        if self._contains_ad_hominem(reasoning):
            fallacies.append('ad_hominem')

        # Circular reasoning
        if self._contains_circular_reasoning(reasoning):
            fallacies.append('circular_reasoning')

        # False dichotomy
        if self._contains_false_dichotomy(reasoning):
            fallacies.append('false_dichotomy')

        # Hasty generalization
        if self._contains_hasty_generalization(reasoning):
            fallacies.append('hasty_generalization')

        # Correlation implies causation
        if self._contains_correlation_causation_fallacy(reasoning):
            fallacies.append('correlation_causation')

        # Appeal to authority
        if self._contains_appeal_to_authority(reasoning):
            fallacies.append('appeal_to_authority')

        return fallacies
```

### Formal Logic Checking

```python
class FormalLogicChecker:
    """Check reasoning using formal logic."""

    def __init__(self):
        # Could integrate with tools like Prover9, Z3, or Coq
        self.logic_engine = None

    def check_formal_validity(self, reasoning_trace):
        """
        Convert reasoning to formal logic and check validity.
        """

        # Convert natural language to logical formulas
        premises = self._extract_premises(reasoning_trace)
        conclusion = self._extract_conclusion(reasoning_trace)

        # Convert to formal logic
        formal_premises = [self._to_formal_logic(p) for p in premises]
        formal_conclusion = self._to_formal_logic(conclusion)

        # Check if conclusion follows from premises
        is_valid = self._check_entailment(formal_premises, formal_conclusion)

        return {
            'formally_valid': is_valid,
            'premises': formal_premises,
            'conclusion': formal_conclusion,
            'explanation': self._explain_validity(is_valid, formal_premises, formal_conclusion)
        }

    def _to_formal_logic(self, natural_language):
        """Convert natural language to formal logic notation."""

        # Simplified conversion (real implementation would be more sophisticated)
        # Example: "All dogs are animals" -> ∀x(Dog(x) → Animal(x))

        # Use LLM or rule-based parser
        formal = self._parse_to_logic(natural_language)

        return formal

    def _check_entailment(self, premises, conclusion):
        """
        Check if conclusion is entailed by premises.

        In formal logic: Γ ⊨ φ
        where Γ is the set of premises and φ is the conclusion
        """

        # Use automated theorem prover
        # For now, simplified check

        return True  # Placeholder
```

## Reasoning Coherence

Coherence measures how well reasoning steps connect and flow logically.

### Measuring Coherence

```python
class CoherenceEvaluator:
    """Evaluate coherence of reasoning."""

    def evaluate_coherence(self, reasoning_trace):
        """Measure how coherently the reasoning flows."""

        steps = self._parse_steps(reasoning_trace)

        if len(steps) < 2:
            return {'coherence_score': 1.0, 'reason': 'single_step'}

        # Measure step-to-step coherence
        step_coherence = []
        for i in range(len(steps) - 1):
            coherence = self._measure_step_coherence(steps[i], steps[i+1])
            step_coherence.append(coherence)

        # Check for contradictions
        contradictions = self._detect_contradictions(steps)

        # Check for consistency
        consistency = self._check_consistency(steps)

        # Aggregate score
        avg_coherence = np.mean(step_coherence)
        contradiction_penalty = len(contradictions) * 0.15

        score = max(0, avg_coherence * consistency - contradiction_penalty)

        return {
            'coherence_score': score,
            'step_coherence': step_coherence,
            'contradictions': contradictions,
            'consistency_score': consistency,
            'issues': self._identify_coherence_issues(step_coherence, contradictions)
        }

    def _measure_step_coherence(self, step1, step2):
        """
        Measure how well two consecutive steps connect.
        """

        # Semantic similarity
        semantic_sim = semantic_similarity(step1, step2)

        # Check for explicit connectors
        has_connector = self._has_logical_connector(step2)

        # Check if step2 references concepts from step1
        conceptual_continuity = self._check_conceptual_continuity(step1, step2)

        # Combine metrics
        coherence = (
            0.4 * semantic_sim +
            0.3 * (1.0 if has_connector else 0.5) +
            0.3 * conceptual_continuity
        )

        return coherence

    def _detect_contradictions(self, steps):
        """Detect contradictory statements in reasoning."""

        contradictions = []

        for i, step_i in enumerate(steps):
            for j, step_j in enumerate(steps[i+1:], i+1):
                # Check if statements contradict
                if self._are_contradictory(step_i, step_j):
                    contradictions.append({
                        'step_i': i,
                        'step_j': j,
                        'statement_i': step_i,
                        'statement_j': step_j
                    })

        return contradictions

    def _are_contradictory(self, statement1, statement2):
        """Check if two statements contradict."""

        # Extract claims
        claim1 = self._extract_claim(statement1)
        claim2 = self._extract_claim(statement2)

        # Check for direct contradiction
        # "X is true" vs "X is false"

        # Use NLI model to check for contradiction
        nli_result = self._nli_check(claim1, claim2)

        return nli_result == 'contradiction'

    def _check_consistency(self, steps):
        """Check overall consistency of reasoning."""

        # Extract all claims
        claims = [self._extract_claim(step) for step in steps]

        # Check if claims are mutually consistent
        inconsistencies = 0
        total_pairs = 0

        for i in range(len(claims)):
            for j in range(i+1, len(claims)):
                total_pairs += 1
                if self._are_inconsistent(claims[i], claims[j]):
                    inconsistencies += 1

        consistency = 1.0 - (inconsistencies / total_pairs) if total_pairs > 0 else 1.0

        return consistency
```

### Narrative Flow Analysis

```python
class NarrativeFlowAnalyzer:
    """Analyze the narrative flow of reasoning."""

    def analyze_flow(self, reasoning_trace):
        """Analyze how reasoning flows as a narrative."""

        # Check for clear structure
        has_introduction = self._has_introduction(reasoning_trace)
        has_development = self._has_development(reasoning_trace)
        has_conclusion = self._has_conclusion(reasoning_trace)

        # Check for transitions
        transition_quality = self._assess_transitions(reasoning_trace)

        # Check for logical progression
        progression = self._assess_progression(reasoning_trace)

        return {
            'structure_score': sum([has_introduction, has_development, has_conclusion]) / 3,
            'transition_quality': transition_quality,
            'progression_quality': progression,
            'overall_flow': self._compute_overall_flow(
                has_introduction, has_development, has_conclusion,
                transition_quality, progression
            )
        }

    def _assess_transitions(self, reasoning_trace):
        """Assess quality of transitions between reasoning steps."""

        steps = self._parse_steps(reasoning_trace)

        transition_words = [
            'therefore', 'thus', 'hence', 'consequently',
            'because', 'since', 'as', 'given that',
            'furthermore', 'moreover', 'additionally',
            'however', 'nevertheless', 'on the other hand',
            'first', 'second', 'finally'
        ]

        transitions_found = 0
        for step in steps[1:]:  # Skip first step
            if any(word in step.lower() for word in transition_words):
                transitions_found += 1

        # Good transitions in most steps (but not required in all)
        expected_transitions = len(steps) - 1
        transition_rate = transitions_found / expected_transitions if expected_transitions > 0 else 0

        # Optimal is 60-80% of steps having transitions
        if 0.6 <= transition_rate <= 0.8:
            quality = 1.0
        else:
            quality = max(0, 1.0 - abs(transition_rate - 0.7) / 0.7)

        return quality
```

## Reasoning Depth

Depth measures how thoroughly the agent analyzes the problem.

### Measuring Reasoning Depth

```python
class ReasoningDepthEvaluator:
    """Evaluate depth of reasoning."""

    def evaluate_depth(self, reasoning_trace, task):
        """Measure reasoning depth."""

        # Count reasoning levels
        levels = self._identify_reasoning_levels(reasoning_trace)

        # Check for multi-level analysis
        has_surface_level = 'surface' in levels
        has_intermediate_level = 'intermediate' in levels
        has_deep_level = 'deep' in levels

        # Count reasoning steps
        num_steps = self._count_reasoning_steps(reasoning_trace)

        # Check for consideration of alternatives
        alternatives_considered = self._count_alternatives(reasoning_trace)

        # Check for second-order reasoning
        has_second_order = self._has_second_order_reasoning(reasoning_trace)

        # Check for root cause analysis
        has_root_cause = self._has_root_cause_analysis(reasoning_trace)

        # Compute depth score
        depth_score = self._compute_depth_score(
            levels, num_steps, alternatives_considered,
            has_second_order, has_root_cause
        )

        return {
            'depth_score': depth_score,
            'reasoning_levels': levels,
            'num_steps': num_steps,
            'alternatives_considered': alternatives_considered,
            'has_second_order_reasoning': has_second_order,
            'has_root_cause_analysis': has_root_cause,
            'depth_category': self._categorize_depth(depth_score)
        }

    def _identify_reasoning_levels(self, reasoning):
        """
        Identify depth levels in reasoning.

        Levels:
        - Surface: Direct observations, obvious connections
        - Intermediate: One level of abstraction, basic analysis
        - Deep: Multiple levels, root causes, system thinking
        """

        levels = []

        # Surface level indicators
        surface_indicators = ['observe', 'see', 'notice', 'appears', 'seems']
        if any(ind in reasoning.lower() for ind in surface_indicators):
            levels.append('surface')

        # Intermediate level indicators
        intermediate_indicators = ['because', 'therefore', 'implies', 'suggests', 'leads to']
        if any(ind in reasoning.lower() for ind in intermediate_indicators):
            levels.append('intermediate')

        # Deep level indicators
        deep_indicators = [
            'underlying', 'fundamental', 'root cause', 'systemic',
            'second-order', 'downstream', 'upstream', 'cascading'
        ]
        if any(ind in reasoning.lower() for ind in deep_indicators):
            levels.append('deep')

        return levels

    def _count_alternatives(self, reasoning):
        """Count how many alternatives were considered."""

        alternative_indicators = [
            'alternatively', 'another option', 'could also',
            'on the other hand', 'conversely', 'instead'
        ]

        count = sum(1 for ind in alternative_indicators
                   if ind in reasoning.lower())

        return count

    def _has_second_order_reasoning(self, reasoning):
        """
        Check for second-order reasoning (reasoning about reasoning).

        Example: "This approach might not work because I'm assuming X,
                 but what if X is false?"
        """

        second_order_indicators = [
            'assuming', 'if we assume', 'this assumes',
            'might be wrong if', 'depends on whether',
            'question our assumption', 'reconsidering'
        ]

        return any(ind in reasoning.lower() for ind in second_order_indicators)

    def _compute_depth_score(self, levels, num_steps, alternatives,
                           second_order, root_cause):
        """Compute overall depth score."""

        # Base score from levels
        level_score = len(levels) / 3.0  # Max 3 levels

        # Steps contribution (logarithmic - diminishing returns)
        steps_score = min(1.0, np.log(num_steps + 1) / np.log(10))

        # Alternatives contribution
        alternatives_score = min(1.0, alternatives / 3.0)

        # Bonuses for second-order and root cause
        second_order_bonus = 0.2 if second_order else 0
        root_cause_bonus = 0.2 if root_cause else 0

        # Weighted combination
        depth = (
            0.3 * level_score +
            0.3 * steps_score +
            0.2 * alternatives_score +
            second_order_bonus +
            root_cause_bonus
        )

        return min(1.0, depth)

    def _categorize_depth(self, score):
        """Categorize depth score."""
        if score >= 0.8: return 'deep'
        elif score >= 0.5: return 'moderate'
        else: return 'shallow'
```

### Depth Patterns

```python
class DepthPatternAnalyzer:
    """Analyze specific patterns indicating reasoning depth."""

    patterns = {
        'first_principles': {
            'indicators': ['fundamental', 'first principles', 'from scratch', 'ground up'],
            'depth_value': 0.9
        },

        'analogy': {
            'indicators': ['similar to', 'analogous', 'like', 'comparable to'],
            'depth_value': 0.6
        },

        'counterfactual': {
            'indicators': ['what if', 'suppose', 'imagine if', 'hypothetically'],
            'depth_value': 0.8
        },

        'systems_thinking': {
            'indicators': ['interconnected', 'feedback loop', 'emergent', 'system-wide'],
            'depth_value': 0.9
        },

        'multi_perspective': {
            'indicators': ['from another angle', 'different perspective', 'alternatively'],
            'depth_value': 0.7
        }
    }

    def analyze_patterns(self, reasoning):
        """Identify depth-indicating patterns."""

        detected_patterns = []

        for pattern_name, pattern_spec in self.patterns.items():
            if any(ind in reasoning.lower() for ind in pattern_spec['indicators']):
                detected_patterns.append({
                    'pattern': pattern_name,
                    'depth_value': pattern_spec['depth_value']
                })

        return {
            'patterns_detected': detected_patterns,
            'pattern_depth_score': np.mean([p['depth_value'] for p in detected_patterns])
                                  if detected_patterns else 0.0
        }
```

## Reasoning Thoroughness

Thoroughness measures completeness - are all necessary considerations addressed?

### Completeness Checking

```python
class ThoroughnessEvaluator:
    """Evaluate thoroughness of reasoning."""

    def evaluate_thoroughness(self, reasoning_trace, task):
        """Check if reasoning is thorough and complete."""

        # Identify required considerations for this task
        required_considerations = self._identify_required_considerations(task)

        # Check which considerations were addressed
        addressed = []
        for consideration in required_considerations:
            if self._is_addressed(consideration, reasoning_trace):
                addressed.append(consideration)

        # Check for unjustified leaps
        leaps = self._detect_unjustified_leaps(reasoning_trace)

        # Check for explicit assumptions
        assumptions = self._extract_assumptions(reasoning_trace)
        implicit_assumptions = self._detect_implicit_assumptions(reasoning_trace)

        # Compute thoroughness score
        coverage = len(addressed) / len(required_considerations) if required_considerations else 1.0
        leap_penalty = len(leaps) * 0.1
        assumption_bonus = 0.1 if assumptions else 0
        implicit_penalty = len(implicit_assumptions) * 0.05

        score = max(0, coverage - leap_penalty + assumption_bonus - implicit_penalty)

        return {
            'thoroughness_score': score,
            'coverage': coverage,
            'addressed_considerations': addressed,
            'missing_considerations': [c for c in required_considerations if c not in addressed],
            'unjustified_leaps': leaps,
            'explicit_assumptions': assumptions,
            'implicit_assumptions': implicit_assumptions
        }

    def _identify_required_considerations(self, task):
        """Identify what should be considered for this task."""

        task_type = task.get('type', 'generic')

        if task_type == 'decision_making':
            return [
                'identify_options',
                'evaluate_criteria',
                'assess_tradeoffs',
                'consider_constraints',
                'evaluate_risks',
                'predict_outcomes'
            ]

        elif task_type == 'problem_solving':
            return [
                'understand_problem',
                'identify_root_cause',
                'generate_solutions',
                'evaluate_feasibility',
                'consider_side_effects'
            ]

        elif task_type == 'analysis':
            return [
                'gather_information',
                'identify_patterns',
                'evaluate_evidence',
                'consider_alternatives',
                'draw_conclusions'
            ]

        # Generic considerations
        return [
            'understand_context',
            'identify_key_factors',
            'evaluate_evidence',
            'consider_alternatives'
        ]

    def _detect_unjustified_leaps(self, reasoning):
        """Detect logical leaps without justification."""

        steps = self._parse_steps(reasoning)
        leaps = []

        for i in range(len(steps) - 1):
            step1 = steps[i]
            step2 = steps[i + 1]

            # Check if step2 follows logically from step1
            if not self._follows_logically(step1, step2):
                # Check if there's explicit justification
                if not self._has_justification(step1, step2):
                    leaps.append({
                        'from_step': i,
                        'to_step': i + 1,
                        'from': step1,
                        'to': step2
                    })

        return leaps

    def _extract_assumptions(self, reasoning):
        """Extract explicitly stated assumptions."""

        assumption_patterns = [
            r'assuming (?:that )?(.+)',
            r'if we assume (.+)',
            r'given that (.+)',
            r'presupposing (.+)'
        ]

        assumptions = []
        for pattern in assumption_patterns:
            matches = re.findall(pattern, reasoning, re.IGNORECASE)
            assumptions.extend(matches)

        return assumptions

    def _detect_implicit_assumptions(self, reasoning):
        """Detect assumptions that are used but not stated."""

        # This is complex - would require deep semantic analysis
        # Simplified version: check for common implicit assumptions

        implicit = []

        # Check for causal assumptions
        if 'causes' in reasoning or 'leads to' in reasoning:
            if 'correlation' not in reasoning:
                implicit.append('assumes_causation_from_correlation')

        # Check for generalization assumptions
        if 'always' in reasoning or 'never' in reasoning:
            if 'except' not in reasoning and 'unless' not in reasoning:
                implicit.append('assumes_no_exceptions')

        return implicit
```

## Process vs Outcome

Should we evaluate the reasoning process or just the outcome?

### Process-Outcome Trade-off

```python
class ProcessOutcomeEvaluator:
    """Evaluate both process and outcome, understanding trade-offs."""

    def evaluate_both(self, agent_trace):
        """Evaluate both process quality and outcome correctness."""

        # Extract reasoning process
        reasoning_process = extract_reasoning(agent_trace)

        # Extract final answer/outcome
        outcome = extract_outcome(agent_trace)

        # Evaluate process
        process_score = self._evaluate_process(reasoning_process)

        # Evaluate outcome
        outcome_score = self._evaluate_outcome(outcome)

        # Categorize result
        category = self._categorize(process_score, outcome_score)

        return {
            'process_score': process_score,
            'outcome_score': outcome_score,
            'category': category,
            'overall_score': self._compute_overall(process_score, outcome_score),
            'analysis': self._analyze_category(category)
        }

    def _categorize(self, process_score, outcome_score):
        """
        Categorize based on process and outcome quality.

        Matrix:
                        Outcome
                    Good    |    Bad
              ─────────────────────────
        Good  │  Ideal     │  Unlucky   │
      Process │            │            │
        ──────────────────────────────
        Bad   │  Lucky     │  Failed    │
              │            │            │
              ─────────────────────────
        """

        process_good = process_score > 0.7
        outcome_good = outcome_score > 0.7

        if process_good and outcome_good:
            return 'ideal'
        elif process_good and not outcome_good:
            return 'unlucky'  # Good process, bad outcome (might be execution error)
        elif not process_good and outcome_good:
            return 'lucky'    # Bad process, good outcome (not reliable)
        else:
            return 'failed'   # Both bad

    def _analyze_category(self, category):
        """Provide analysis based on category."""

        analyses = {
            'ideal': {
                'assessment': 'Sound reasoning led to correct outcome',
                'reliability': 'high',
                'action': 'Reinforce this reasoning pattern'
            },
            'unlucky': {
                'assessment': 'Sound reasoning but incorrect outcome (possible execution error)',
                'reliability': 'medium-high',
                'action': 'Check for execution errors or edge cases'
            },
            'lucky': {
                'assessment': 'Correct outcome despite flawed reasoning (not reliable)',
                'reliability': 'low',
                'action': 'Fix reasoning process - correct outcome may not generalize'
            },
            'failed': {
                'assessment': 'Both reasoning and outcome are problematic',
                'reliability': 'very low',
                'action': 'Major revision needed'
            }
        }

        return analyses.get(category, {})

    def _compute_overall(self, process_score, outcome_score):
        """
        Compute overall score weighing both process and outcome.

        Process is more important for long-term reliability.
        """

        # Weight process more heavily (70/30)
        return 0.7 * process_score + 0.3 * outcome_score
```

### When to Prioritize Process vs Outcome

```python
def choose_evaluation_priority(task_context):
    """
    Decide whether to prioritize process or outcome evaluation.
    """

    # Outcome priority contexts
    if task_context.get('safety_critical'):
        return {
            'priority': 'outcome',
            'process_weight': 0.3,
            'outcome_weight': 0.7,
            'reason': 'Safety-critical - correct outcome is essential'
        }

    if task_context.get('one_time_task'):
        return {
            'priority': 'outcome',
            'process_weight': 0.4,
            'outcome_weight': 0.6,
            'reason': 'One-time task - outcome matters most'
        }

    # Process priority contexts
    if task_context.get('learning_system'):
        return {
            'priority': 'process',
            'process_weight': 0.8,
            'outcome_weight': 0.2,
            'reason': 'Learning system - good process will improve over time'
        }

    if task_context.get('reusable_reasoning'):
        return {
            'priority': 'process',
            'process_weight': 0.75,
            'outcome_weight': 0.25,
            'reason': 'Reasoning will be reused - process quality is key'
        }

    # Balanced contexts
    return {
        'priority': 'balanced',
        'process_weight': 0.5,
        'outcome_weight': 0.5,
        'reason': 'Both process and outcome matter equally'
    }
```

## Reasoning Traces

Analyzing explicit reasoning traces.

### Trace Analysis

```python
class ReasoningTraceAnalyzer:
    """Analyze explicit reasoning traces from agents."""

    def analyze_trace(self, trace):
        """
        Comprehensive analysis of reasoning trace.

        Trace format:
        [
            {'type': 'thought', 'content': '...'},
            {'type': 'action', 'content': '...'},
            {'type': 'observation', 'content': '...'},
            ...
        ]
        """

        # Extract thoughts (reasoning steps)
        thoughts = [step['content'] for step in trace if step['type'] == 'thought']

        if not thoughts:
            return {'error': 'No reasoning steps found in trace'}

        # Analyze structure
        structure = self._analyze_structure(thoughts)

        # Analyze content
        content = self._analyze_content(thoughts)

        # Analyze flow
        flow = self._analyze_flow(thoughts)

        # Analyze quality
        quality = self._analyze_quality(thoughts)

        return {
            'structure': structure,
            'content': content,
            'flow': flow,
            'quality': quality,
            'overall_score': self._compute_overall_score(structure, content, flow, quality)
        }

    def _analyze_structure(self, thoughts):
        """Analyze structural properties."""

        return {
            'num_steps': len(thoughts),
            'avg_length': np.mean([len(t) for t in thoughts]),
            'has_intro': self._has_intro(thoughts),
            'has_conclusion': self._has_conclusion(thoughts),
            'logical_progression': self._check_progression(thoughts)
        }

    def _analyze_content(self, thoughts):
        """Analyze content quality."""

        return {
            'specificity': self._measure_specificity(thoughts),
            'evidence_based': self._check_evidence_use(thoughts),
            'assumption_explicit': self._check_assumptions(thoughts),
            'alternatives_considered': self._count_alternatives(thoughts)
        }

    def _analyze_flow(self, thoughts):
        """Analyze flow and coherence."""

        return {
            'coherence': self._measure_coherence(thoughts),
            'transitions': self._assess_transitions(thoughts),
            'consistency': self._check_consistency(thoughts),
            'focus': self._measure_focus(thoughts)
        }

    def _analyze_quality(self, thoughts):
        """Analyze overall quality."""

        return {
            'logical_validity': self._check_logical_validity(thoughts),
            'depth': self._measure_depth(thoughts),
            'thoroughness': self._measure_thoroughness(thoughts),
            'clarity': self._measure_clarity(thoughts)
        }
```

### Trace Comparison

```python
def compare_reasoning_traces(trace_a, trace_b):
    """
    Compare two reasoning traces to understand differences.
    """

    analyzer = ReasoningTraceAnalyzer()

    analysis_a = analyzer.analyze_trace(trace_a)
    analysis_b = analyzer.analyze_trace(trace_b)

    # Compare dimensions
    comparisons = {}
    for dimension in ['structure', 'content', 'flow', 'quality']:
        dim_a = analysis_a[dimension]
        dim_b = analysis_b[dimension]

        differences = {}
        for key in dim_a.keys():
            if key in dim_b:
                diff = dim_a[key] - dim_b[key] if isinstance(dim_a[key], (int, float)) else 'different'
                differences[key] = diff

        comparisons[dimension] = differences

    # Overall comparison
    overall_diff = analysis_a['overall_score'] - analysis_b['overall_score']

    return {
        'comparisons': comparisons,
        'overall_difference': overall_diff,
        'better_trace': 'trace_a' if overall_diff > 0 else 'trace_b',
        'significance': 'significant' if abs(overall_diff) > 0.2 else 'marginal'
    }
```

## Logical Fallacies

Detecting common reasoning errors.

### Fallacy Detection

```python
class FallacyDetector:
    """Detect logical fallacies in reasoning."""

    fallacy_patterns = {
        'ad_hominem': {
            'description': 'Attacking the person instead of the argument',
            'indicators': [
                r'(?:he|she|they) (?:is|are) (?:stupid|wrong|biased|incompetent)',
                r'coming from (?:him|her|them)',
                r'(?:can\'t|cannot) trust (?:him|her|them)'
            ]
        },

        'straw_man': {
            'description': 'Misrepresenting an argument to make it easier to attack',
            'indicators': [
                r'so you\'re saying',
                r'what you really mean',
                r'in other words you think'
            ]
        },

        'false_dichotomy': {
            'description': 'Presenting only two options when more exist',
            'indicators': [
                r'either .+ or .+, there\'s no other',
                r'only two (?:options|choices|possibilities)',
                r'must be .+ or .+'
            ]
        },

        'slippery_slope': {
            'description': 'Claiming one thing will inevitably lead to extreme consequences',
            'indicators': [
                r'will (?:inevitably|certainly|definitely) lead to',
                r'before you know it',
                r'next thing you know'
            ]
        },

        'circular_reasoning': {
            'description': 'Assuming the conclusion in the premise',
            'indicators': [
                # This is harder to detect with patterns
                'by_definition',
                'naturally_follows'
            ]
        },

        'hasty_generalization': {
            'description': 'Drawing broad conclusions from limited evidence',
            'indicators': [
                r'(?:all|every|always|never) .+ (?:are|do|have)',
                r'therefore (?:all|every|always)',
                r'one .+ so (?:all|every)'
            ]
        },

        'appeal_to_authority': {
            'description': 'Claiming something is true because an authority says so',
            'indicators': [
                r'(?:expert|professor|doctor) says',
                r'according to (?:expert|authority)',
                r'must be true because .+ said'
            ]
        },

        'appeal_to_emotion': {
            'description': 'Using emotion instead of logic',
            'indicators': [
                r'think of the (?:children|victims)',
                r'how would you feel',
                r'imagine if it were you'
            ]
        },

        'false_cause': {
            'description': 'Assuming correlation implies causation',
            'indicators': [
                r'.+ causes .+ (?:because|since) they (?:correlate|occur together)',
                r'happened (?:after|when) .+ so .+ caused',
                r'correlation .+ therefore .+ causes'
            ]
        }
    }

    def detect_fallacies(self, reasoning):
        """Detect logical fallacies in reasoning."""

        detected = []

        for fallacy_name, fallacy_spec in self.fallacy_patterns.items():
            # Check patterns
            for pattern in fallacy_spec['indicators']:
                matches = re.finditer(pattern, reasoning, re.IGNORECASE)
                for match in matches:
                    detected.append({
                        'fallacy': fallacy_name,
                        'description': fallacy_spec['description'],
                        'location': match.span(),
                        'matched_text': match.group()
                    })

        # Check for circular reasoning (requires semantic analysis)
        if self._has_circular_reasoning(reasoning):
            detected.append({
                'fallacy': 'circular_reasoning',
                'description': 'Premise assumes conclusion',
                'location': None
            })

        return detected

    def _has_circular_reasoning(self, reasoning):
        """Detect circular reasoning through semantic analysis."""

        # Extract premise and conclusion
        premise = self._extract_premise(reasoning)
        conclusion = self._extract_conclusion(reasoning)

        if not premise or not conclusion:
            return False

        # Check if premise and conclusion are semantically similar
        similarity = semantic_similarity(premise, conclusion)

        return similarity > 0.9  # Very high similarity suggests circularity
```

## Multi-Step Reasoning

Evaluating complex multi-step reasoning chains.

### Chain Analysis

```python
class MultiStepReasoningEvaluator:
    """Evaluate multi-step reasoning chains."""

    def evaluate_chain(self, reasoning_chain):
        """
        Evaluate a multi-step reasoning chain.

        Chain format: [step1, step2, step3, ...]
        """

        if len(reasoning_chain) < 2:
            return {'score': 1.0, 'reason': 'single_step'}

        # Evaluate each step
        step_scores = [self._evaluate_step(step) for step in reasoning_chain]

        # Evaluate transitions between steps
        transition_scores = [
            self._evaluate_transition(reasoning_chain[i], reasoning_chain[i+1])
            for i in range(len(reasoning_chain) - 1)
        ]

        # Evaluate overall chain coherence
        chain_coherence = self._evaluate_chain_coherence(reasoning_chain)

        # Check for error propagation
        error_propagation = self._check_error_propagation(reasoning_chain)

        # Aggregate scores
        avg_step_quality = np.mean(step_scores)
        avg_transition_quality = np.mean(transition_scores)

        overall_score = (
            0.4 * avg_step_quality +
            0.3 * avg_transition_quality +
            0.3 * chain_coherence
        ) * (1.0 - error_propagation)  # Penalty for error propagation

        return {
            'overall_score': overall_score,
            'step_scores': step_scores,
            'transition_scores': transition_scores,
            'chain_coherence': chain_coherence,
            'error_propagation': error_propagation,
            'chain_length': len(reasoning_chain)
        }

    def _evaluate_step(self, step):
        """Evaluate individual reasoning step quality."""

        criteria = {
            'clarity': self._step_clarity(step),
            'specificity': self._step_specificity(step),
            'logical': self._step_logical(step)
        }

        return np.mean(list(criteria.values()))

    def _evaluate_transition(self, step1, step2):
        """Evaluate how well step2 follows from step1."""

        # Logical connection
        logical_connection = self._has_logical_connection(step1, step2)

        # Semantic coherence
        semantic_coherence = semantic_similarity(step1, step2)

        # Explicit linking
        has_connector = self._has_connector(step2)

        score = (
            0.5 * logical_connection +
            0.3 * semantic_coherence +
            0.2 * (1.0 if has_connector else 0.5)
        )

        return score

    def _check_error_propagation(self, chain):
        """
        Check if errors in early steps propagate through the chain.
        """

        # Identify potential errors in early steps
        errors = []
        for i, step in enumerate(chain[:len(chain)//2]):  # First half
            if self._has_error(step):
                errors.append(i)

        if not errors:
            return 0.0

        # Check if later steps depend on erroneous steps
        propagation_count = 0
        for error_idx in errors:
            for later_idx in range(error_idx + 1, len(chain)):
                if self._depends_on(chain[later_idx], chain[error_idx]):
                    propagation_count += 1

        # Normalize
        max_propagation = len(errors) * (len(chain) - max(errors) - 1)

        return propagation_count / max_propagation if max_propagation > 0 else 0.0
```

## Automated Reasoning Evaluation

Using automated tools to evaluate reasoning.

### LLM-as-Judge for Reasoning

```python
class LLMReasoningJudge:
    """Use LLM to judge reasoning quality."""

    def __init__(self, judge_model):
        self.judge = judge_model

    def evaluate_reasoning(self, reasoning_trace):
        """Have LLM evaluate reasoning quality."""

        prompt = f"""
        Evaluate the following reasoning on a scale of 1-10 for each criterion:

        Reasoning:
        {reasoning_trace}

        Criteria:
        1. Logical Correctness: Are the logical steps valid?
        2. Coherence: Do the steps flow logically?
        3. Completeness: Are all necessary steps present?
        4. Depth: How thoroughly is the problem analyzed?
        5. Clarity: Is the reasoning easy to follow?

        For each criterion:
        - Provide a score (1-10)
        - Provide a brief justification
        - Identify specific issues if score < 7

        Also provide:
        - Overall assessment
        - Key strengths
        - Key weaknesses
        - Suggestions for improvement

        Format as JSON.
        """

        response = self.judge.complete(prompt)
        evaluation = self._parse_response(response)

        return evaluation

    def evaluate_with_consensus(self, reasoning_trace, n_judges=3):
        """Use multiple LLM judges for consensus."""

        evaluations = [
            self.evaluate_reasoning(reasoning_trace)
            for _ in range(n_judges)
        ]

        # Compute inter-judge agreement
        agreement = self._compute_agreement(evaluations)

        # Compute consensus scores
        consensus = self._compute_consensus(evaluations)

        return {
            'individual_evaluations': evaluations,
            'consensus': consensus,
            'agreement': agreement,
            'confidence': self._confidence_from_agreement(agreement)
        }

    def _compute_agreement(self, evaluations):
        """Compute inter-judge agreement."""

        criteria = ['logical_correctness', 'coherence', 'completeness', 'depth', 'clarity']

        agreements = []
        for criterion in criteria:
            scores = [eval[criterion]['score'] for eval in evaluations]
            # Compute standard deviation as measure of disagreement
            std = np.std(scores)
            # Convert to agreement (lower std = higher agreement)
            agreement = 1.0 / (1.0 + std)
            agreements.append(agreement)

        return np.mean(agreements)
```

## Summary

Reasoning quality evaluation assesses the thought process, not just outcomes:

**Key Dimensions**:

- **Logical Correctness**: Valid inferences, no fallacies
- **Coherence**: Connected steps, consistency, no contradictions
- **Depth**: Multi-level analysis, second-order reasoning
- **Thoroughness**: Complete consideration of relevant factors
- **Clarity**: Understandable, well-structured reasoning

**Evaluation Approaches**:

- Logical validity checking
- Coherence measurement
- Depth analysis
- Completeness assessment
- Fallacy detection
- Trace analysis
- LLM-as-judge evaluation

**Key Insights**:

- Process matters as much as outcome for long-term reliability
- Good reasoning with wrong answer is often better than right answer with bad reasoning
- Multiple evaluation methods provide comprehensive assessment
- Automated tools can augment but not replace human judgment
- Reasoning quality predicts generalization better than outcome correctness

**Best Practices**:

- Evaluate process and outcome separately
- Look for logical fallacies
- Check for completeness and thoroughness
- Analyze reasoning traces explicitly
- Use multiple judges for subjective assessments
- Consider context when evaluating reasoning

Quality reasoning is the foundation of reliable agents - invest in evaluating and improving it.

## Next Steps

- **[Tool Use Evaluation](tool-use-evaluation.md)**: Assessing tool selection and usage
- **[Efficiency Metrics](efficiency-metrics.md)**: Measuring resource consumption
- **[Robustness Testing](robustness.md)**: Testing reliability under stress
- **[Success Metrics](success-metrics.md)**: Measuring task completion
- **[Planning and Reasoning](../planning-and-reasoning/chain-of-thought.md)**: Improving reasoning
