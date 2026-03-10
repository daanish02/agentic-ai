# Tool Use Evaluation

## Table of Contents

- [Introduction](#introduction)
- [Tool Selection Accuracy](#tool-selection-accuracy)
- [Parameter Correctness](#parameter-correctness)
- [Tool Chaining](#tool-chaining)
- [Tool Use Appropriateness](#tool-use-appropriateness)
- [Tool Use Efficiency](#tool-use-efficiency)
- [Error Handling in Tool Use](#error-handling-in-tool-use)
- [Tool Sequence Evaluation](#tool-sequence-evaluation)
- [Parallel Tool Use](#parallel-tool-use)
- [Tool Discovery](#tool-discovery)
- [Tool Misuse Detection](#tool-misuse-detection)
- [Tool Learning](#tool-learning)
- [Benchmarking Tool Use](#benchmarking-tool-use)
- [Tool Use Patterns](#tool-use-patterns)
- [Real-World Tool Use Evaluation](#real-world-tool-use-evaluation)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Tool use is a defining capability of agentic systems. An agent without tools is just a chatbot; an agent with tools can take actions, retrieve information, and interact with the world. But **using tools correctly is surprisingly difficult**.

Consider these tool use challenges:

```python
# Challenge 1: Tool Selection
User: "What's the weather in London?"

# Wrong tool
agent.use(calculator, "weather in London")  # Calculator can't check weather!

# Right tool
agent.use(weather_api, city="London")

# Challenge 2: Parameter Correctness
User: "Search for papers from 2024"

# Wrong parameters
agent.use(search_papers, query="2024")  # Search text, not year filter!

# Right parameters
agent.use(search_papers, query="machine learning", year=2024)

# Challenge 3: Tool Chaining
User: "Find the most cited paper on RAG and summarize it"

# Wrong: Single tool
agent.use(summarize, "RAG paper")  # Can't summarize without retrieving!

# Right: Chain tools
papers = agent.use(search_papers, query="RAG", sort_by="citations")
top_paper = papers[0]
content = agent.use(fetch_paper, paper_id=top_paper.id)
summary = agent.use(summarize, text=content)
```

This guide covers how to evaluate agent tool use: from selection accuracy to parameter correctness, from efficiency to error handling, and from simple metrics to comprehensive frameworks.

> "Give an agent a tool and it will try to use it for everything. Teach an agent when to use tools and it becomes powerful."

## Tool Selection Accuracy

Did the agent choose the right tools for the task?

### Selection Metrics

```python
class ToolSelectionEvaluator:
    """Evaluate tool selection accuracy."""

    def __init__(self, tool_registry):
        self.tool_registry = tool_registry

    def evaluate_selection(self, task, selected_tools, expected_tools=None):
        """
        Evaluate whether agent selected appropriate tools.

        Args:
            task: The task being performed
            selected_tools: List of tools agent actually selected
            expected_tools: List of tools that should be used (if known)

        Returns:
            Selection accuracy metrics
        """

        if expected_tools:
            # Compare to ground truth
            precision = len(set(selected_tools) & set(expected_tools)) / len(selected_tools) if selected_tools else 0
            recall = len(set(selected_tools) & set(expected_tools)) / len(expected_tools) if expected_tools else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            return {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'exact_match': set(selected_tools) == set(expected_tools)
            }
        else:
            # Evaluate appropriateness without ground truth
            appropriateness_scores = [
                self._score_tool_appropriateness(tool, task)
                for tool in selected_tools
            ]

            return {
                'avg_appropriateness': np.mean(appropriateness_scores) if appropriateness_scores else 0,
                'all_appropriate': all(score > 0.7 for score in appropriateness_scores),
                'tool_appropriateness': dict(zip(selected_tools, appropriateness_scores))
            }

    def _score_tool_appropriateness(self, tool, task):
        """Score how appropriate a tool is for the task."""

        tool_capabilities = self.tool_registry.get_capabilities(tool)
        task_requirements = self._extract_requirements(task)

        # Check capability overlap
        overlap = len(set(tool_capabilities) & set(task_requirements))
        relevance = overlap / len(task_requirements) if task_requirements else 0

        # Check for overkill (tool is way more powerful than needed)
        complexity_ratio = self._tool_complexity(tool) / self._task_complexity(task)
        overkill_penalty = max(0, complexity_ratio - 2.0) * 0.1

        score = max(0, relevance - overkill_penalty)

        return score

    def evaluate_selection_process(self, agent_trace):
        """
        Evaluate how the agent selected tools (reasoning quality).
        """

        # Extract tool selection reasoning
        selection_reasoning = self._extract_selection_reasoning(agent_trace)

        criteria = {
            'considered_alternatives': self._checked_alternatives(selection_reasoning),
            'justified_selection': self._has_justification(selection_reasoning),
            'considered_constraints': self._considered_constraints(selection_reasoning),
            'systematic_selection': self._is_systematic(selection_reasoning)
        }

        return {
            'process_score': np.mean(list(criteria.values())),
            'criteria_met': criteria
        }
```

### Common Selection Errors

```python
class ToolSelectionErrorAnalyzer:
    """Analyze common tool selection errors."""

    error_types = {
        'wrong_tool': 'Selected tool that cannot perform required function',
        'missing_tool': 'Failed to select necessary tool',
        'redundant_tool': 'Selected tool redundantly with another tool',
        'overkill': 'Selected overly complex tool for simple task',
        'deprecated': 'Selected outdated tool when better alternative exists',
        'no_tool': 'Failed to use any tool when tool use was required'
    }

    def analyze_errors(self, task, selected_tools, execution_results):
        """Identify and categorize tool selection errors."""

        errors = []

        # Check for wrong tool usage
        for tool in selected_tools:
            if not self._can_tool_help(tool, task):
                errors.append({
                    'type': 'wrong_tool',
                    'tool': tool,
                    'reason': f'{tool} cannot help with {task}'
                })

        # Check for missing tools
        required_tools = self._identify_required_tools(task)
        missing = set(required_tools) - set(selected_tools)
        for tool in missing:
            errors.append({
                'type': 'missing_tool',
                'tool': tool,
                'reason': f'{tool} is needed but was not used'
            })

        # Check for redundancy
        redundant = self._find_redundant_tools(selected_tools)
        for tool_pair in redundant:
            errors.append({
                'type': 'redundant_tool',
                'tools': tool_pair,
                'reason': f'{tool_pair[0]} and {tool_pair[1]} are redundant'
            })

        # Check for no tool use when needed
        if not selected_tools and self._requires_tools(task):
            errors.append({
                'type': 'no_tool',
                'reason': 'Task requires tools but none were used'
            })

        return {
            'errors': errors,
            'error_count': len(errors),
            'error_types': Counter(e['type'] for e in errors)
        }
```

## Parameter Correctness

Did the agent provide correct parameters when calling tools?

### Parameter Validation

```python
class ParameterCorrectnessEvaluator:
    """Evaluate parameter correctness in tool calls."""

    def evaluate_parameters(self, tool_call, expected_params=None):
        """
        Evaluate correctness of parameters in tool call.

        Args:
            tool_call: Dict with 'tool', 'parameters'
            expected_params: Expected parameters (if known)

        Returns:
            Parameter correctness metrics
        """

        tool = tool_call['tool']
        params = tool_call['parameters']

        # Get tool schema
        schema = self._get_tool_schema(tool)

        # Check schema compliance
        schema_compliance = self._check_schema_compliance(params, schema)

        # Check semantic correctness
        semantic_correctness = self._check_semantic_correctness(params, tool, expected_params)

        # Check for common errors
        common_errors = self._detect_common_parameter_errors(params, schema)

        # Aggregate score
        score = (
            0.4 * schema_compliance['score'] +
            0.6 * semantic_correctness['score']
        ) * (1.0 - len(common_errors) * 0.1)

        return {
            'correctness_score': max(0, score),
            'schema_compliance': schema_compliance,
            'semantic_correctness': semantic_correctness,
            'errors': common_errors
        }

    def _check_schema_compliance(self, params, schema):
        """Check if parameters match tool schema."""

        issues = []

        # Check required parameters
        required = schema.get('required', [])
        for param in required:
            if param not in params:
                issues.append(f'Missing required parameter: {param}')

        # Check parameter types
        for param, value in params.items():
            if param in schema['parameters']:
                expected_type = schema['parameters'][param].get('type')
                actual_type = type(value).__name__

                if not self._types_compatible(actual_type, expected_type):
                    issues.append(f'Wrong type for {param}: expected {expected_type}, got {actual_type}')

        # Check for unknown parameters
        valid_params = set(schema['parameters'].keys())
        for param in params:
            if param not in valid_params:
                issues.append(f'Unknown parameter: {param}')

        score = 1.0 - (len(issues) / max(1, len(schema['parameters'])))

        return {
            'score': max(0, score),
            'issues': issues,
            'compliant': len(issues) == 0
        }

    def _check_semantic_correctness(self, params, tool, expected_params):
        """Check if parameters are semantically correct for the task."""

        if expected_params:
            # Compare to ground truth
            matches = sum(1 for k, v in params.items()
                         if k in expected_params and
                         self._values_match(v, expected_params[k]))

            score = matches / len(expected_params) if expected_params else 0
        else:
            # Use heuristics
            score = self._heuristic_semantic_check(params, tool)

        return {
            'score': score,
            'semantically_correct': score > 0.8
        }

    def _detect_common_parameter_errors(self, params, schema):
        """Detect common parameter errors."""

        errors = []

        for param, value in params.items():
            if param not in schema['parameters']:
                continue

            param_spec = schema['parameters'][param]

            # Empty string when shouldn't be
            if isinstance(value, str) and len(value) == 0:
                if not param_spec.get('allow_empty', False):
                    errors.append(f'{param}: empty string')

            # Out of range
            if 'min' in param_spec and value < param_spec['min']:
                errors.append(f'{param}: below minimum ({value} < {param_spec["min"]})')

            if 'max' in param_spec and value > param_spec['max']:
                errors.append(f'{param}: above maximum ({value} > {param_spec["max"]})')

            # Wrong enum value
            if 'enum' in param_spec and value not in param_spec['enum']:
                errors.append(f'{param}: invalid value {value}, must be one of {param_spec["enum"]}')

        return errors
```

### Parameter Extraction Quality

```python
class ParameterExtractionEvaluator:
    """Evaluate how well agent extracts parameters from user input."""

    def evaluate_extraction(self, user_input, extracted_params, tool_schema):
        """
        Evaluate parameter extraction quality.

        Example:
        User: "Search for papers on RAG from 2024"
        Extracted: {'query': 'RAG', 'year': 2024}
        """

        # Check completeness: did we extract all extractable parameters?
        extractable_params = self._identify_extractable_params(user_input, tool_schema)
        extracted_names = set(extracted_params.keys())
        expected_names = set(extractable_params.keys())

        precision = len(extracted_names & expected_names) / len(extracted_names) if extracted_names else 0
        recall = len(extracted_names & expected_names) / len(expected_names) if expected_names else 0

        # Check accuracy: are extracted values correct?
        accuracy_scores = []
        for param, value in extracted_params.items():
            if param in extractable_params:
                correct = self._check_value_correctness(value, extractable_params[param])
                accuracy_scores.append(1.0 if correct else 0.0)

        avg_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0

        return {
            'extraction_precision': precision,
            'extraction_recall': recall,
            'extraction_f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
            'value_accuracy': avg_accuracy,
            'overall_score': 0.5 * recall + 0.5 * avg_accuracy
        }

    def _identify_extractable_params(self, user_input, schema):
        """Identify parameters that can be extracted from user input."""

        extractable = {}

        # Parse user input
        entities = self._extract_entities(user_input)

        for param_name, param_spec in schema['parameters'].items():
            # Check if this parameter type appears in input
            if param_spec['type'] == 'string':
                # Look for relevant text
                if self._contains_relevant_text(user_input, param_spec):
                    extractable[param_name] = self._extract_string_value(user_input, param_spec)

            elif param_spec['type'] == 'integer':
                # Look for numbers
                numbers = re.findall(r'\d+', user_input)
                if numbers:
                    extractable[param_name] = int(numbers[0])

            elif param_spec['type'] == 'date':
                # Look for dates
                dates = self._extract_dates(user_input)
                if dates:
                    extractable[param_name] = dates[0]

        return extractable
```

## Tool Chaining

Evaluating sequences of tool calls.

### Chain Correctness

```python
class ToolChainEvaluator:
    """Evaluate tool chaining correctness and quality."""

    def evaluate_chain(self, tool_chain, task):
        """
        Evaluate a sequence of tool calls.

        Args:
            tool_chain: List of tool calls in order
            task: The task being accomplished

        Returns:
            Chain evaluation metrics
        """

        # Check if chain accomplishes goal
        accomplishes_goal = self._check_goal_accomplishment(tool_chain, task)

        # Check logical ordering
        ordering_correct = self._check_ordering(tool_chain)

        # Check for dependencies
        dependencies_satisfied = self._check_dependencies(tool_chain)

        # Check for redundancy
        redundant_calls = self._find_redundant_calls(tool_chain)

        # Check for missing steps
        missing_steps = self._find_missing_steps(tool_chain, task)

        # Compute scores
        correctness = 1.0 if accomplishes_goal else 0.0
        ordering_score = self._score_ordering(ordering_correct)
        dependency_score = len([d for d in dependencies_satisfied if d]) / len(dependencies_satisfied) if dependencies_satisfied else 1.0
        redundancy_penalty = len(redundant_calls) * 0.1
        missing_penalty = len(missing_steps) * 0.15

        overall_score = max(0,
            0.4 * correctness +
            0.2 * ordering_score +
            0.2 * dependency_score -
            redundancy_penalty -
            missing_penalty
        )

        return {
            'overall_score': overall_score,
            'accomplishes_goal': accomplishes_goal,
            'ordering_correct': ordering_correct,
            'dependencies_satisfied': all(dependencies_satisfied),
            'redundant_calls': redundant_calls,
            'missing_steps': missing_steps,
            'chain_length': len(tool_chain)
        }

    def _check_ordering(self, tool_chain):
        """Check if tools are called in correct order."""

        issues = []

        for i in range(len(tool_chain) - 1):
            current_tool = tool_chain[i]
            next_tool = tool_chain[i + 1]

            # Check if next_tool can use outputs from current_tool
            if self._requires_input(next_tool):
                required_input = self._get_required_input(next_tool)
                provided_output = self._get_output(current_tool)

                if not self._compatible(required_input, provided_output):
                    # Check if any previous tool provides compatible output
                    compatible_found = False
                    for j in range(i):
                        if self._compatible(required_input, self._get_output(tool_chain[j])):
                            compatible_found = True
                            break

                    if not compatible_found:
                        issues.append({
                            'position': i + 1,
                            'tool': next_tool['tool'],
                            'issue': 'Missing required input'
                        })

        return issues

    def _check_dependencies(self, tool_chain):
        """Check if dependencies between tools are satisfied."""

        satisfied = []

        for i, tool_call in enumerate(tool_chain):
            dependencies = self._get_dependencies(tool_call['tool'])

            for dep in dependencies:
                # Check if dependency was called earlier in chain
                dep_satisfied = any(
                    tc['tool'] == dep
                    for tc in tool_chain[:i]
                )
                satisfied.append(dep_satisfied)

        return satisfied

    def _find_redundant_calls(self, tool_chain):
        """Find redundant tool calls in chain."""

        redundant = []

        for i, call1 in enumerate(tool_chain):
            for j, call2 in enumerate(tool_chain[i+1:], i+1):
                # Same tool with same parameters
                if (call1['tool'] == call2['tool'] and
                    self._params_equivalent(call1['parameters'], call2['parameters'])):
                    redundant.append({
                        'positions': (i, j),
                        'tool': call1['tool']
                    })

        return redundant
```

### Optimal Chain Analysis

```python
class OptimalChainAnalyzer:
    """Analyze if tool chain is optimal."""

    def analyze_optimality(self, agent_chain, task):
        """
        Compare agent's chain to optimal chain.
        """

        # Generate optimal chain
        optimal_chain = self._generate_optimal_chain(task)

        # Compare lengths
        length_ratio = len(agent_chain) / len(optimal_chain)

        # Compare tool selections
        agent_tools = {tc['tool'] for tc in agent_chain}
        optimal_tools = {tc['tool'] for tc in optimal_chain}
        tool_overlap = len(agent_tools & optimal_tools) / len(optimal_tools)

        # Compute edit distance
        edit_distance = self._compute_chain_edit_distance(agent_chain, optimal_chain)
        similarity = 1.0 - (edit_distance / max(len(agent_chain), len(optimal_chain)))

        return {
            'is_optimal': similarity > 0.9 and length_ratio <= 1.1,
            'optimality_score': similarity,
            'length_efficiency': min(1.0, 1.0 / length_ratio),
            'tool_selection_accuracy': tool_overlap,
            'edit_distance': edit_distance,
            'optimal_chain': optimal_chain,
            'suggestions': self._generate_suggestions(agent_chain, optimal_chain)
        }

    def _generate_optimal_chain(self, task):
        """Generate theoretically optimal tool chain for task."""

        # Use planning algorithm or heuristics
        # This is simplified

        goal = task['goal']
        available_tools = self.tool_registry.list_tools()

        # Simple greedy approach
        chain = []
        current_state = task['initial_state']

        while not self._goal_achieved(current_state, goal):
            # Find best tool for current state
            best_tool = self._find_best_tool(current_state, goal, available_tools)

            if not best_tool:
                break

            chain.append(best_tool)
            current_state = self._simulate_tool_effect(best_tool, current_state)

        return chain
```

## Tool Use Appropriateness

Beyond correctness, is tool use appropriate and reasonable?

### Appropriateness Scoring

```python
class ToolAppropriatenessEvaluator:
    """Evaluate whether tool use is appropriate."""

    def evaluate_appropriateness(self, tool_call, context):
        """
        Evaluate if tool use is appropriate given context.
        """

        criteria = {
            'necessary': self._is_necessary(tool_call, context),
            'sufficient': self._is_sufficient(tool_call, context),
            'efficient': self._is_efficient(tool_call, context),
            'safe': self._is_safe(tool_call, context),
            'respectful_of_constraints': self._respects_constraints(tool_call, context)
        }

        # Compute weighted score
        weights = {
            'necessary': 0.3,
            'sufficient': 0.2,
            'efficient': 0.2,
            'safe': 0.2,
            'respectful_of_constraints': 0.1
        }

        score = sum(criteria[k] * weights[k] for k in criteria)

        return {
            'appropriateness_score': score,
            'criteria_met': criteria,
            'appropriate': score > 0.7,
            'issues': [k for k, v in criteria.items() if not v]
        }

    def _is_necessary(self, tool_call, context):
        """Is this tool call necessary?"""

        # Could task be accomplished without this tool?
        alternatives = self._find_alternatives(tool_call, context)

        # Tool is necessary if no viable alternatives exist
        necessary = len([alt for alt in alternatives if not alt['requires_tool']]) == 0

        return necessary

    def _is_sufficient(self, tool_call, context):
        """Is this tool call sufficient (doesn't require many more tools)?"""

        # Check if this tool call gets us close to goal
        progress = self._estimate_progress_toward_goal(tool_call, context)

        return progress > 0.3  # Makes significant progress

    def _is_efficient(self, tool_call, context):
        """Is this the most efficient tool for the job?"""

        tool_efficiency = self._compute_tool_efficiency(tool_call)

        # Compare to other tools that could do the job
        alternatives = self._find_tool_alternatives(tool_call)

        if not alternatives:
            return True

        max_alternative_efficiency = max(self._compute_tool_efficiency(alt) for alt in alternatives)

        # Acceptable if within 80% of best alternative
        return tool_efficiency >= 0.8 * max_alternative_efficiency

    def _is_safe(self, tool_call, context):
        """Is this tool call safe?"""

        safety_checks = [
            not self._is_destructive(tool_call),
            not self._violates_privacy(tool_call, context),
            not self._wastes_resources(tool_call),
            not self._has_negative_side_effects(tool_call)
        ]

        return all(safety_checks)
```

## Tool Use Efficiency

Measuring efficiency of tool use.

### Efficiency Metrics

```python
class ToolUseEfficiencyEvaluator:
    """Evaluate efficiency of tool use."""

    def evaluate_efficiency(self, tool_trace, task):
        """
        Evaluate multiple dimensions of tool use efficiency.
        """

        metrics = {
            # Number of tool calls
            'num_calls': len(tool_trace),

            # Ratio to optimal
            'calls_vs_optimal': self._compute_vs_optimal(tool_trace, task),

            # Redundancy
            'redundancy_rate': self._compute_redundancy(tool_trace),

            # Token efficiency
            'tokens_per_call': self._compute_avg_tokens(tool_trace),

            # Time efficiency
            'avg_latency': self._compute_avg_latency(tool_trace),

            # Cost efficiency
            'total_cost': self._compute_total_cost(tool_trace),

            # Success rate
            'success_rate': self._compute_success_rate(tool_trace)
        }

        # Compute aggregate efficiency score
        efficiency_score = self._compute_efficiency_score(metrics)

        return {
            'efficiency_score': efficiency_score,
            'metrics': metrics,
            'efficiency_grade': self._grade_efficiency(efficiency_score)
        }

    def _compute_vs_optimal(self, tool_trace, task):
        """Compare to optimal number of tool calls."""

        optimal_calls = self._compute_optimal_call_count(task)
        actual_calls = len(tool_trace)

        if optimal_calls == 0:
            return 1.0 if actual_calls == 0 else 0.0

        # Ratio (1.0 is perfect, lower is better)
        ratio = actual_calls / optimal_calls

        # Convert to score (1.0 is best)
        if ratio <= 1.0:
            return 1.0  # Used fewer or equal calls
        else:
            return max(0, 1.0 - (ratio - 1.0) / ratio)

    def _compute_redundancy(self, tool_trace):
        """Compute redundancy rate in tool calls."""

        unique_calls = len(set(
            (call['tool'], frozenset(call['parameters'].items()))
            for call in tool_trace
        ))

        total_calls = len(tool_trace)

        redundancy = 1.0 - (unique_calls / total_calls) if total_calls > 0 else 0

        return redundancy

    def _compute_efficiency_score(self, metrics):
        """Aggregate efficiency metrics into single score."""

        # Normalize metrics
        normalized = {
            'call_efficiency': metrics['calls_vs_optimal'],
            'no_redundancy': 1.0 - metrics['redundancy_rate'],
            'success': metrics['success_rate']
        }

        # Weighted average
        weights = {
            'call_efficiency': 0.4,
            'no_redundancy': 0.3,
            'success': 0.3
        }

        score = sum(normalized[k] * weights[k] for k in weights)

        return score
```

## Error Handling in Tool Use

How well does the agent handle tool failures?

### Error Recovery Evaluation

```python
class ToolErrorHandlingEvaluator:
    """Evaluate agent's tool error handling."""

    def evaluate_error_handling(self, agent_trace):
        """
        Evaluate how agent handles tool errors.
        """

        # Find tool failures in trace
        failures = self._extract_failures(agent_trace)

        if not failures:
            return {'score': 1.0, 'reason': 'no_failures_to_handle'}

        recovery_scores = []

        for failure in failures:
            recovery = self._evaluate_recovery(failure, agent_trace)
            recovery_scores.append(recovery)

        avg_recovery = np.mean([r['score'] for r in recovery_scores])

        return {
            'error_handling_score': avg_recovery,
            'num_failures': len(failures),
            'recovery_details': recovery_scores,
            'recovery_rate': sum(1 for r in recovery_scores if r['recovered']) / len(failures)
        }

    def _evaluate_recovery(self, failure, agent_trace):
        """Evaluate recovery from a specific failure."""

        failure_idx = failure['index']

        # Check if agent acknowledged the error
        acknowledged = self._check_acknowledgment(agent_trace, failure_idx)

        # Check if agent took recovery action
        recovery_action = self._find_recovery_action(agent_trace, failure_idx)

        # Check if recovery was successful
        recovered = recovery_action is not None and recovery_action['successful']

        # Check recovery strategy quality
        strategy_quality = self._evaluate_recovery_strategy(recovery_action, failure) if recovery_action else 0.0

        score = (
            0.2 * (1.0 if acknowledged else 0.0) +
            0.4 * (1.0 if recovery_action is not None else 0.0) +
            0.2 * (1.0 if recovered else 0.0) +
            0.2 * strategy_quality
        )

        return {
            'score': score,
            'recovered': recovered,
            'acknowledged': acknowledged,
            'recovery_action': recovery_action,
            'strategy_quality': strategy_quality
        }

    def _evaluate_recovery_strategy(self, recovery_action, failure):
        """Evaluate quality of recovery strategy."""

        strategies = {
            'retry': 0.5,  # Simple but sometimes effective
            'retry_with_different_params': 0.7,  # Better
            'try_alternative_tool': 0.9,  # Smart
            'graceful_degradation': 0.8,  # Good
            'ask_for_help': 0.6,  # Reasonable
            'give_up': 0.0  # Not ideal
        }

        strategy_type = self._classify_strategy(recovery_action, failure)

        return strategies.get(strategy_type, 0.5)
```

## Tool Misuse Detection

Detecting dangerous or inappropriate tool use.

### Misuse Patterns

```python
class ToolMisuseDetector:
    """Detect tool misuse patterns."""

    misuse_patterns = {
        'excessive_calls': {
            'description': 'Calling same tool too many times',
            'threshold': 10,
            'severity': 'medium'
        },
        'parameter_bruteforce': {
            'description': 'Trying many parameter combinations',
            'threshold': 5,
            'severity': 'high'
        },
        'ignored_errors': {
            'description': 'Continuing after repeated failures',
            'threshold': 3,
            'severity': 'high'
        },
        'unauthorized_access': {
            'description': 'Attempting to access restricted resources',
            'threshold': 1,
            'severity': 'critical'
        },
        'resource_abuse': {
            'description': 'Using excessive resources',
            'threshold': 'dynamic',
            'severity': 'high'
        }
    }

    def detect_misuse(self, tool_trace):
        """Detect tool misuse in trace."""

        issues = []

        # Check for excessive calls
        tool_counts = Counter(call['tool'] for call in tool_trace)
        for tool, count in tool_counts.items():
            if count > self.misuse_patterns['excessive_calls']['threshold']:
                issues.append({
                    'pattern': 'excessive_calls',
                    'tool': tool,
                    'count': count,
                    'severity': 'medium'
                })

        # Check for parameter brute-forcing
        param_attempts = self._analyze_parameter_attempts(tool_trace)
        for tool, attempts in param_attempts.items():
            if attempts > self.misuse_patterns['parameter_bruteforce']['threshold']:
                issues.append({
                    'pattern': 'parameter_bruteforce',
                    'tool': tool,
                    'attempts': attempts,
                    'severity': 'high'
                })

        # Check for ignored errors
        error_sequences = self._find_error_sequences(tool_trace)
        for seq in error_sequences:
            if len(seq) >= self.misuse_patterns['ignored_errors']['threshold']:
                issues.append({
                    'pattern': 'ignored_errors',
                    'sequence': seq,
                    'severity': 'high'
                })

        # Check for unauthorized access attempts
        unauthorized = self._detect_unauthorized_attempts(tool_trace)
        for attempt in unauthorized:
            issues.append({
                'pattern': 'unauthorized_access',
                'attempt': attempt,
                'severity': 'critical'
            })

        # Compute overall misuse score
        misuse_score = self._compute_misuse_score(issues)

        return {
            'misuse_detected': len(issues) > 0,
            'misuse_score': misuse_score,
            'issues': issues,
            'severity_counts': Counter(issue['severity'] for issue in issues)
        }
```

## Benchmarking Tool Use

Standard benchmarks for tool use evaluation.

### Tool Use Benchmarks

```python
class ToolUseBenchmark:
    """Benchmark suite for tool use evaluation."""

    def __init__(self):
        self.benchmarks = self._load_benchmarks()

    def _load_benchmarks(self):
        """Load standard tool use benchmarks."""

        return {
            'toolbench': {
                'description': 'Real-world API usage benchmark',
                'tasks': 'load_toolbench_tasks()',
                'metrics': ['success_rate', 'tool_selection', 'parameter_accuracy']
            },

            'api_bank': {
                'description': 'API calling benchmark with diverse tools',
                'tasks': 'load_api_bank_tasks()',
                'metrics': ['correctness', 'efficiency', 'safety']
            },

            'tool_llm_bench': {
                'description': 'Tool use reasoning benchmark',
                'tasks': 'load_tool_llm_tasks()',
                'metrics': ['reasoning_quality', 'tool_selection', 'chaining']
            }
        }

    def run_benchmark(self, agent, benchmark_name):
        """Run agent on standard benchmark."""

        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        benchmark = self.benchmarks[benchmark_name]
        tasks = eval(benchmark['tasks'])

        results = []
        for task in tasks:
            result = agent.run(task)
            score = self._score_result(result, task, benchmark['metrics'])
            results.append(score)

        # Aggregate scores
        aggregate = {
            metric: np.mean([r[metric] for r in results])
            for metric in benchmark['metrics']
        }

        return {
            'benchmark': benchmark_name,
            'aggregate_scores': aggregate,
            'individual_results': results,
            'overall_score': np.mean(list(aggregate.values()))
        }
```

## Summary

Tool use evaluation assesses how well agents select, parameterize, and chain tools:

**Key Dimensions**:

- **Selection Accuracy**: Right tools for the task
- **Parameter Correctness**: Correct parameters for each tool
- **Chaining**: Appropriate sequencing of tools
- **Efficiency**: Minimal calls, no redundancy
- **Error Handling**: Recovery from failures
- **Safety**: No misuse or dangerous patterns

**Evaluation Approaches**:

- Compare to expected tools/parameters
- Analyze tool chains for correctness
- Measure efficiency metrics
- Detect misuse patterns
- Use standard benchmarks
- Evaluate error recovery

**Best Practices**:

- Evaluate both selection and execution
- Check parameter extraction quality
- Analyze tool chains, not just individual calls
- Monitor for misuse patterns
- Measure efficiency, not just correctness
- Test error handling explicitly

Tool use is what makes agents powerful - ensure they use tools correctly, efficiently, and safely.

## Next Steps

- **[Efficiency Metrics](efficiency-metrics.md)**: Detailed resource usage metrics
- **[Robustness Testing](robustness.md)**: Testing under failures
- **[Success Metrics](success-metrics.md)**: Overall task success
- **[Tool Use Fundamentals](../tool-use/function-calling.md)**: How function calling works
