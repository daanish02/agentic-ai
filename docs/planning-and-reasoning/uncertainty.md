# Uncertainty and Probabilistic Reasoning

## Table of Contents

- [Introduction](#introduction)
- [Sources of Uncertainty](#sources-of-uncertainty)
- [Confidence Estimation](#confidence-estimation)
- [Handling Incomplete Information](#handling-incomplete-information)
- [Dealing with Ambiguity](#dealing-with-ambiguity)
- [Asking Clarifying Questions](#asking-clarifying-questions)
- [Hedging and Qualified Statements](#hedging-and-qualified-statements)
- [Probabilistic Reasoning](#probabilistic-reasoning)
- [Decision Making Under Uncertainty](#decision-making-under-uncertainty)
- [Uncertainty Propagation](#uncertainty-propagation)
- [Risk Assessment](#risk-assessment)
- [Information Gathering Strategies](#information-gathering-strategies)
- [Uncertainty-Aware Planning](#uncertainty-aware-planning)
- [Best Practices](#best-practices)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Real-world agents rarely operate with complete, certain information. Effective agents must reason about confidence levels, handle ambiguity, and make decisions despite incomplete knowledge.

> "It is impossible to know everything about everything. The real skill is in knowing what matters." - Venkat Subramaniam

Agents face uncertainty from:
- **Incomplete information**: Missing data or context
- **Ambiguous requirements**: Multiple valid interpretations
- **Noisy observations**: Unreliable or conflicting information
- **Unpredictable environments**: Dynamic, changing conditions
- **Model limitations**: Inherent uncertainty in AI predictions

### The Uncertainty Spectrum

```
Certainty                    Uncertainty                    Total Ignorance
    |───────────────────────────────────────────────────────|
    
    Known                    Estimated                Unknown
    Facts                    Probabilities             Unknowns
    
    "File exists"            "Likely correct"         "No idea"
    "2 + 2 = 4"              "Probably a bug"         "What is X?"
```

### Why Uncertainty Management Matters

```python
# Without uncertainty handling
def solve_task(task):
    # Assumes perfect information
    solution = determine_solution(task)
    return solution  # What if we're wrong?


# With uncertainty handling
def solve_task(task):
    # Assess confidence
    solution, confidence = determine_solution_with_confidence(task)
    
    if confidence > 0.8:
        return solution
    elif confidence > 0.5:
        # Verify before acting
        if verify_solution(solution):
            return solution
        else:
            return ask_for_clarification(task)
    else:
        # Too uncertain - need more information
        return request_additional_info(task)
```

## Sources of Uncertainty

Understanding where uncertainty comes from.

### 1. Incomplete Information

```python
class InformationState:
    """Track what information we have and what's missing."""
    
    def __init__(self):
        self.known: Dict[str, Any] = {}
        self.unknown: Set[str] = set()
        self.assumptions: Dict[str, Any] = {}
    
    def assess_completeness(self, required_info: List[str]) -> dict:
        """
        Assess information completeness.
        
        Returns:
            Dictionary with completeness metrics
        """
        known_items = set(self.known.keys())
        required_items = set(required_info)
        
        missing = required_items - known_items
        extra = known_items - required_items
        
        completeness = len(known_items & required_items) / len(required_items)
        
        return {
            "completeness": completeness,
            "known": list(known_items & required_items),
            "missing": list(missing),
            "assumptions_made": list(self.assumptions.keys())
        }


# Example
info = InformationState()
info.known["user_id"] = 12345
info.known["username"] = "alice"
info.assumptions["email"] = "alice@example.com"  # Assumed, not verified
info.unknown.add("phone_number")

required = ["user_id", "username", "email", "phone_number"]
status = info.assess_completeness(required)

print(f"Completeness: {status['completeness']:.1%}")
print(f"Missing: {status['missing']}")
print(f"Assumptions: {status['assumptions_made']}")
```

### 2. Ambiguous Requirements

```python
class AmbiguityDetector:
    """Detect ambiguity in requirements or instructions."""
    
    AMBIGUOUS_TERMS = [
        "soon", "later", "some", "many", "few", "several",
        "recent", "old", "large", "small", "fast", "slow",
        "good", "better", "best", "optimize", "improve"
    ]
    
    def __init__(self):
        self.ambiguities: List[str] = []
    
    def detect_ambiguity(self, requirement: str) -> List[dict]:
        """
        Identify ambiguous terms in requirement.
        
        Returns:
            List of detected ambiguities with suggested clarifications
        """
        ambiguities = []
        
        # Check for vague quantifiers
        for term in self.AMBIGUOUS_TERMS:
            if term.lower() in requirement.lower():
                ambiguities.append({
                    "term": term,
                    "type": "vague_quantifier",
                    "clarification_needed": self._suggest_clarification(term)
                })
        
        # Check for subjective terms
        if any(word in requirement.lower() for word in ["good", "better", "best", "optimal"]):
            ambiguities.append({
                "type": "subjective_criteria",
                "clarification_needed": "Specify measurable criteria or metrics"
            })
        
        # Check for incomplete conditions
        if "if" in requirement.lower() and "else" not in requirement.lower():
            ambiguities.append({
                "type": "incomplete_conditional",
                "clarification_needed": "Specify behavior for alternative cases"
            })
        
        return ambiguities
    
    def _suggest_clarification(self, term: str) -> str:
        """Suggest how to clarify ambiguous term."""
        clarifications = {
            "soon": "Specify time: hours, days, weeks?",
            "many": "Specify quantity: >10, >100, >1000?",
            "large": "Specify size: >1MB, >100MB, >1GB?",
            "fast": "Specify time: <1s, <100ms, <10ms?",
            "optimize": "Specify metric: latency, throughput, memory?",
            "improve": "Specify baseline and target: from X to Y"
        }
        return clarifications.get(term, f"Clarify what '{term}' means specifically")


# Example
detector = AmbiguityDetector()

requirement = "Process many files quickly and store in large database"
ambiguities = detector.detect_ambiguity(requirement)

print(f"Detected {len(ambiguities)} ambiguities:")
for amb in ambiguities:
    if "term" in amb:
        print(f"  - '{amb['term']}': {amb['clarification_needed']}")
    else:
        print(f"  - {amb['type']}: {amb['clarification_needed']}")
```

### 3. Noisy or Conflicting Information

```python
from collections import Counter

class ConflictResolver:
    """Resolve conflicting information from multiple sources."""
    
    def __init__(self):
        self.sources: Dict[str, float] = {}  # source -> reliability score
    
    def set_source_reliability(self, source: str, reliability: float):
        """Set reliability score for information source."""
        self.sources[source] = reliability
    
    def resolve_conflict(self, claims: List[dict]) -> dict:
        """
        Resolve conflicting claims.
        
        Args:
            claims: List of {source, claim, confidence} dicts
            
        Returns:
            Best estimate with confidence
        """
        if not claims:
            return {"value": None, "confidence": 0.0, "method": "no_data"}
        
        # If all agree, high confidence
        values = [c["claim"] for c in claims]
        if len(set(values)) == 1:
            return {
                "value": values[0],
                "confidence": 0.95,
                "method": "unanimous",
                "supporting_sources": len(claims)
            }
        
        # Weighted voting by source reliability
        weighted_votes = Counter()
        for claim in claims:
            source_reliability = self.sources.get(claim["source"], 0.5)
            claim_confidence = claim.get("confidence", 0.5)
            weight = source_reliability * claim_confidence
            weighted_votes[claim["claim"]] += weight
        
        # Get best estimate
        best_claim, best_weight = weighted_votes.most_common(1)[0]
        total_weight = sum(weighted_votes.values())
        
        confidence = best_weight / total_weight if total_weight > 0 else 0.0
        
        return {
            "value": best_claim,
            "confidence": confidence,
            "method": "weighted_voting",
            "votes": dict(weighted_votes),
            "conflicts": len(set(values)) - 1
        }


# Example
resolver = ConflictResolver()

# Set source reliabilities
resolver.set_source_reliability("official_docs", 0.9)
resolver.set_source_reliability("user_report", 0.6)
resolver.set_source_reliability("blog_post", 0.4)

# Conflicting claims about API rate limit
claims = [
    {"source": "official_docs", "claim": 1000, "confidence": 0.9},
    {"source": "user_report", "claim": 500, "confidence": 0.7},
    {"source": "blog_post", "claim": 500, "confidence": 0.6}
]

result = resolver.resolve_conflict(claims)

print(f"Best estimate: {result['value']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Method: {result['method']}")
print(f"Conflicts: {result['conflicts']}")
```

## Confidence Estimation

Estimating confidence in decisions and predictions.

### Confidence-Aware Agent

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class ConfidentAnswer:
    """Answer with confidence score."""
    answer: Any
    confidence: float  # 0.0 to 1.0
    reasoning: str
    evidence: List[str]


class ConfidenceAwareAgent:
    """Agent that explicitly tracks confidence in its answers."""
    
    def __init__(self, model):
        self.model = model
        self.confidence_threshold = 0.7  # Act only if confident enough
    
    def answer_with_confidence(self, question: str) -> ConfidentAnswer:
        """
        Answer question with explicit confidence estimate.
        
        Returns:
            Answer with confidence score and reasoning
        """
        # Generate answer
        prompt = f"""
Question: {question}

Provide your answer along with:
1. The answer itself
2. Your confidence level (0-100%)
3. Reasoning for your confidence
4. Evidence supporting your answer

Response:
"""
        
        response = self.model.generate(prompt, temperature=0.3)
        
        # Parse response (simplified)
        answer, confidence, reasoning, evidence = self._parse_response(response)
        
        return ConfidentAnswer(
            answer=answer,
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence
        )
    
    def act_with_confidence(self, task: str) -> dict:
        """
        Decide whether to act based on confidence.
        
        Returns:
            Action taken or request for more information
        """
        # Assess confidence
        result = self.answer_with_confidence(f"How should I accomplish: {task}")
        
        if result.confidence >= self.confidence_threshold:
            # Confident enough to act
            return {
                "action": "execute",
                "plan": result.answer,
                "confidence": result.confidence
            }
        elif result.confidence >= 0.4:
            # Moderate confidence - seek verification
            return {
                "action": "verify",
                "tentative_plan": result.answer,
                "confidence": result.confidence,
                "verification_needed": "Confirm approach before executing"
            }
        else:
            # Low confidence - need more information
            return {
                "action": "clarify",
                "confidence": result.confidence,
                "questions": self._generate_clarifying_questions(task, result)
            }
    
    def _parse_response(self, response: str) -> tuple:
        """Parse response into components."""
        # Simplified parsing
        lines = response.strip().split('\n')
        
        answer = lines[0] if lines else "Unknown"
        confidence = 0.5  # Default
        reasoning = "No reasoning provided"
        evidence = []
        
        # Extract confidence if mentioned
        for line in lines:
            if '%' in line or 'confidence' in line.lower():
                # Extract number
                import re
                numbers = re.findall(r'\d+', line)
                if numbers:
                    confidence = int(numbers[0]) / 100
        
        return answer, confidence, reasoning, evidence
    
    def _generate_clarifying_questions(self, task: str, 
                                      result: ConfidentAnswer) -> List[str]:
        """Generate questions to increase confidence."""
        prompt = f"""
Task: {task}
Current understanding: {result.answer}
Confidence: {result.confidence:.0%}

What questions should be asked to increase confidence?

Questions:
"""
        
        response = self.model.generate(prompt, temperature=0.4)
        
        # Parse questions
        questions = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or '?' in line):
                questions.append(line.lstrip('0123456789.-) '))
        
        return questions[:5]  # Top 5 questions


# Example usage
agent = ConfidenceAwareAgent(model)

# High confidence task
result1 = agent.act_with_confidence("Sort a list of numbers")
print(f"Action: {result1['action']}")
print(f"Confidence: {result1['confidence']:.0%}")

# Low confidence task
result2 = agent.act_with_confidence("Optimize the system")
print(f"\nAction: {result2['action']}")
print(f"Confidence: {result2['confidence']:.0%}")
if result2['action'] == 'clarify':
    print("Questions needed:")
    for q in result2['questions']:
        print(f"  - {q}")
```

### Bayesian Confidence Updates

```python
class BayesianAgent:
    """
    Agent that updates confidence based on evidence using Bayes' theorem.
    
    P(H|E) = P(E|H) * P(H) / P(E)
    
    where:
    - P(H|E) = posterior probability (updated belief)
    - P(E|H) = likelihood (probability of evidence given hypothesis)
    - P(H) = prior probability (initial belief)
    - P(E) = marginal probability (probability of evidence)
    """
    
    def __init__(self):
        self.beliefs: Dict[str, float] = {}  # hypothesis -> probability
    
    def set_prior(self, hypothesis: str, probability: float):
        """Set prior belief in hypothesis."""
        self.beliefs[hypothesis] = probability
    
    def update_belief(self, hypothesis: str, evidence: str, 
                     likelihood: float):
        """
        Update belief in hypothesis given new evidence.
        
        Args:
            hypothesis: Hypothesis to update
            evidence: New evidence observed
            likelihood: P(evidence | hypothesis)
        """
        if hypothesis not in self.beliefs:
            self.beliefs[hypothesis] = 0.5  # Neutral prior
        
        prior = self.beliefs[hypothesis]
        
        # Simplified Bayesian update (assumes binary hypothesis)
        # P(H|E) ∝ P(E|H) * P(H)
        posterior = likelihood * prior
        
        # Normalize (simplified - assumes complement hypothesis)
        complement_posterior = (1 - likelihood) * (1 - prior)
        normalizer = posterior + complement_posterior
        
        if normalizer > 0:
            self.beliefs[hypothesis] = posterior / normalizer
        
        return self.beliefs[hypothesis]
    
    def get_confidence(self, hypothesis: str) -> float:
        """Get current confidence in hypothesis."""
        return self.beliefs.get(hypothesis, 0.5)


# Example
agent = BayesianAgent()

# Hypothesis: "This code has a bug"
agent.set_prior("code_has_bug", 0.3)  # 30% prior probability

print(f"Prior belief: {agent.get_confidence('code_has_bug'):.1%}")

# Evidence 1: Tests are failing (strong evidence for bug)
confidence = agent.update_belief("code_has_bug", "tests_failing", likelihood=0.9)
print(f"After seeing test failures: {confidence:.1%}")

# Evidence 2: Code review found issues (more evidence)
confidence = agent.update_belief("code_has_bug", "review_issues", likelihood=0.85)
print(f"After code review: {confidence:.1%}")

# Evidence 3: Static analysis clean (weak counter-evidence)
confidence = agent.update_belief("code_has_bug", "static_analysis_clean", likelihood=0.4)
print(f"After static analysis: {confidence:.1%}")
```

## Handling Incomplete Information

Strategies for operating with missing data.

### Pattern 1: Assumptions with Verification

```python
class AssumptionTracker:
    """Track assumptions made to handle incomplete information."""
    
    def __init__(self):
        self.assumptions: List[dict] = []
    
    def make_assumption(self, key: str, value: Any, 
                       rationale: str, risk: str = "medium") -> Any:
        """
        Make assumption and track it for later verification.
        
        Args:
            key: What we're assuming
            value: Assumed value
            rationale: Why this assumption
            risk: "low" | "medium" | "high"
            
        Returns:
            Assumed value
        """
        assumption = {
            "key": key,
            "value": value,
            "rationale": rationale,
            "risk": risk,
            "timestamp": datetime.now(),
            "verified": False
        }
        
        self.assumptions.append(assumption)
        
        print(f"⚠ Assuming {key} = {value}")
        print(f"  Rationale: {rationale}")
        print(f"  Risk: {risk}")
        
        return value
    
    def verify_assumption(self, key: str, actual_value: Any) -> bool:
        """
        Verify assumption against actual value.
        
        Returns:
            True if assumption was correct
        """
        for assumption in self.assumptions:
            if assumption["key"] == key and not assumption["verified"]:
                assumption["verified"] = True
                assumption["actual_value"] = actual_value
                assumption["correct"] = (assumption["value"] == actual_value)
                
                if assumption["correct"]:
                    print(f"✓ Assumption confirmed: {key} = {actual_value}")
                else:
                    print(f"✗ Assumption wrong: {key}")
                    print(f"  Assumed: {assumption['value']}")
                    print(f"  Actual: {actual_value}")
                    print(f"  Risk: {assumption['risk']}")
                
                return assumption["correct"]
        
        return False
    
    def get_unverified_assumptions(self) -> List[dict]:
        """Get assumptions that haven't been verified yet."""
        return [a for a in self.assumptions if not a["verified"]]
    
    def assess_risk(self) -> dict:
        """Assess overall risk from unverified assumptions."""
        unverified = self.get_unverified_assumptions()
        
        risk_levels = {"low": 1, "medium": 5, "high": 10}
        total_risk = sum(risk_levels.get(a["risk"], 5) for a in unverified)
        
        return {
            "unverified_count": len(unverified),
            "total_risk_score": total_risk,
            "high_risk_assumptions": [a for a in unverified if a["risk"] == "high"]
        }


# Example
tracker = AssumptionTracker()

# Make assumptions when information is missing
file_format = tracker.make_assumption(
    key="file_format",
    value="CSV",
    rationale="Most data files in this system are CSV",
    risk="low"
)

encoding = tracker.make_assumption(
    key="encoding",
    value="UTF-8",
    rationale="UTF-8 is standard, no BOM detected",
    risk="low"
)

max_file_size = tracker.make_assumption(
    key="max_file_size",
    value=100_000_000,  # 100MB
    rationale="Assuming typical file sizes, no spec provided",
    risk="high"  # Could be wrong and cause problems
)

# Later, verify assumptions
tracker.verify_assumption("file_format", "CSV")  # Correct
tracker.verify_assumption("encoding", "UTF-8")   # Correct
tracker.verify_assumption("max_file_size", 1_000_000_000)  # Wrong!

# Assess risk
risk = tracker.assess_risk()
print(f"\n Risk Assessment:")
print(f"  Unverified assumptions: {risk['unverified_count']}")
print(f"  Total risk score: {risk['total_risk_score']}")
```

### Pattern 2: Request Missing Information

```python
class InformationRequester:
    """Request missing information before proceeding."""
    
    def __init__(self):
        self.requests: List[dict] = []
    
    def check_requirements(self, task: str, 
                          required_info: List[str],
                          available_info: Dict[str, Any]) -> dict:
        """
        Check if we have all required information.
        
        Returns:
            Status dictionary with missing information
        """
        missing = []
        
        for req in required_info:
            if req not in available_info or available_info[req] is None:
                missing.append(req)
        
        if missing:
            return {
                "ready": False,
                "missing": missing,
                "action": "request_information"
            }
        else:
            return {
                "ready": True,
                "action": "proceed"
            }
    
    def request_information(self, task: str, missing: List[str]) -> str:
        """
        Generate request for missing information.
        
        Returns:
            Formatted request message
        """
        request = {
            "task": task,
            "missing": missing,
            "timestamp": datetime.now()
        }
        
        self.requests.append(request)
        
        message = f"""
Cannot proceed with task: {task}

Missing required information:
{chr(10).join(f'  - {item}' for item in missing)}

Please provide:
{chr(10).join(f'  1. {item}' for item in missing)}
"""
        
        return message.strip()


# Example
requester = InformationRequester()

# Task requires specific information
task = "Deploy application to production"
required = ["deployment_environment", "version", "approval", "rollback_plan"]
available = {
    "deployment_environment": "production",
    "version": "2.1.0"
    # Missing: approval, rollback_plan
}

status = requester.check_requirements(task, required, available)

if not status["ready"]:
    print(requester.request_information(task, status["missing"]))
else:
    print("All requirements satisfied, proceeding...")
```

## Asking Clarifying Questions

Generating targeted questions to reduce uncertainty.

### Question Generator

```python
class ClarifyingQuestionGenerator:
    """Generate questions to resolve ambiguity and uncertainty."""
    
    def __init__(self, model):
        self.model = model
    
    def generate_questions(self, task: str, 
                          uncertainties: List[str]) -> List[str]:
        """
        Generate clarifying questions for task.
        
        Args:
            task: Task description
            uncertainties: List of unclear aspects
            
        Returns:
            List of clarifying questions
        """
        prompt = f"""
Task: {task}

Unclear aspects:
{chr(10).join(f'- {u}' for u in uncertainties)}

Generate 3-5 clarifying questions that would help resolve these uncertainties.
Questions should be:
- Specific and concrete
- Have measurable/verifiable answers
- Help guide implementation decisions

Questions:
"""
        
        response = self.model.generate(prompt, temperature=0.4)
        
        # Parse questions
        questions = []
        for line in response.split('\n'):
            line = line.strip()
            if line and ('?' in line or line[0].isdigit()):
                # Clean up question
                question = line.lstrip('0123456789.-) ')
                if question:
                    questions.append(question)
        
        return questions
    
    def prioritize_questions(self, questions: List[str], 
                           task: str) -> List[dict]:
        """
        Prioritize questions by impact on task success.
        
        Returns:
            List of questions with priority scores
        """
        prioritized = []
        
        for question in questions:
            prompt = f"""
Task: {task}
Question: {question}

On a scale of 1-10, how critical is answering this question
for successfully completing the task?

Consider:
- Does it affect correctness?
- Does it affect performance/efficiency?
- Does it affect user experience?
- Could we make a reasonable assumption instead?

Criticality score (1-10):
"""
            
            response = self.model.generate(prompt, temperature=0.3)
            
            # Extract score
            import re
            numbers = re.findall(r'\b([1-9]|10)\b', response)
            score = int(numbers[0]) if numbers else 5
            
            prioritized.append({
                "question": question,
                "priority": score,
                "rationale": response.strip()
            })
        
        # Sort by priority
        prioritized.sort(key=lambda x: x["priority"], reverse=True)
        
        return prioritized


# Example
generator = ClarifyingQuestionGenerator(model)

task = "Optimize database queries"
uncertainties = [
    "Which queries need optimization?",
    "What is the target performance?",
    "What is the current performance?",
    "Are there any constraints (e.g., can't change schema)?"
]

questions = generator.generate_questions(task, uncertainties)

print("Clarifying Questions:")
for i, q in enumerate(questions, 1):
    print(f"{i}. {q}")

# Prioritize
prioritized = generator.prioritize_questions(questions, task)

print("\nPrioritized:")
for item in prioritized:
    print(f"\n[Priority {item['priority']}/10] {item['question']}")
```

## Decision Making Under Uncertainty

Making good decisions despite incomplete information.

### Expected Value Framework

```python
from typing import List, Tuple

class DecisionMaker:
    """
    Make decisions under uncertainty using expected value.
    
    Expected Value = Σ (Outcome Value × Probability)
    """
    
    @dataclass
    class Option:
        """Decision option with possible outcomes."""
        name: str
        outcomes: List[Tuple[str, float, float]]  # (description, probability, value)
    
    def evaluate_options(self, options: List[Option]) -> dict:
        """
        Evaluate options by expected value.
        
        Returns:
            Best option with analysis
        """
        evaluations = []
        
        for option in options:
            expected_value = sum(prob * value 
                               for _, prob, value in option.outcomes)
            
            # Calculate variance (risk measure)
            variance = sum(prob * (value - expected_value) ** 2 
                         for _, prob, value in option.outcomes)
            risk = variance ** 0.5  # Standard deviation
            
            evaluations.append({
                "option": option.name,
                "expected_value": expected_value,
                "risk": risk,
                "outcomes": option.outcomes
            })
        
        # Sort by expected value
        evaluations.sort(key=lambda x: x["expected_value"], reverse=True)
        
        return {
            "best_option": evaluations[0],
            "all_options": evaluations
        }
    
    def make_decision(self, options: List[Option], 
                     risk_tolerance: float = 0.5) -> str:
        """
        Make decision considering expected value and risk.
        
        Args:
            options: List of options to consider
            risk_tolerance: 0.0 (risk-averse) to 1.0 (risk-seeking)
            
        Returns:
            Chosen option name
        """
        evaluation = self.evaluate_options(options)
        
        # Risk-adjusted selection
        best_score = -float('inf')
        best_option = None
        
        for option_eval in evaluation["all_options"]:
            # Score = EV - (1 - risk_tolerance) * Risk
            score = (option_eval["expected_value"] - 
                    (1 - risk_tolerance) * option_eval["risk"])
            
            if score > best_score:
                best_score = score
                best_option = option_eval["option"]
        
        return best_option


# Example: Should we refactor or add new feature?
maker = DecisionMaker()

options = [
    DecisionMaker.Option(
        name="Refactor first",
        outcomes=[
            ("Refactor succeeds, feature easy", 0.6, 100),
            ("Refactor succeeds, feature still hard", 0.3, 60),
            ("Refactor fails, time wasted", 0.1, -20)
        ]
    ),
    DecisionMaker.Option(
        name="Add feature directly",
        outcomes=[
            ("Feature works, no issues", 0.4, 70),
            ("Feature works, code messy", 0.4, 50),
            ("Feature doesn't work", 0.2, -10)
        ]
    ),
    DecisionMaker.Option(
        name="Prototype first",
        outcomes=[
            ("Prototype reveals simple path", 0.5, 80),
            ("Prototype reveals complexity", 0.4, 40),
            ("Prototype inconclusive", 0.1, 10)
        ]
    )
]

result = maker.evaluate_options(options)

print("Decision Analysis:")
for opt in result["all_options"]:
    print(f"\n{opt['option']}:")
    print(f"  Expected Value: {opt['expected_value']:.1f}")
    print(f"  Risk (σ): {opt['risk']:.1f}")

print(f"\nBest Option (by EV): {result['best_option']['option']}")

# Consider risk
print("\nRisk-adjusted decisions:")
for risk_tolerance in [0.0, 0.5, 1.0]:
    choice = maker.make_decision(options, risk_tolerance)
    print(f"  Risk tolerance {risk_tolerance:.1f}: {choice}")
```

### Information Value

```python
class InformationValueAnalyzer:
    """
    Calculate value of obtaining additional information.
    
    Value of Information = Expected Value with Info - Expected Value without Info
    """
    
    def calculate_voi(self, decision_without_info: float,
                     decision_with_info_outcomes: List[Tuple[float, float]]) -> float:
        """
        Calculate value of perfect information.
        
        Args:
            decision_without_info: Expected value of best decision without information
            decision_with_info_outcomes: List of (probability, best_value) tuples
                                        for each possible information outcome
            
        Returns:
            Value of obtaining perfect information
        """
        ev_with_info = sum(prob * value 
                          for prob, value in decision_with_info_outcomes)
        
        return ev_with_info - decision_without_info
    
    def should_gather_info(self, voi: float, info_cost: float) -> dict:
        """
        Decide whether to gather information.
        
        Returns:
            Decision with rationale
        """
        net_value = voi - info_cost
        
        if net_value > 0:
            return {
                "decision": "gather_information",
                "rationale": f"Net value ${net_value:.2f} (VOI ${voi:.2f} - Cost ${info_cost:.2f})",
                "net_value": net_value
            }
        else:
            return {
                "decision": "proceed_without_information",
                "rationale": f"Information not worth cost (VOI ${voi:.2f} < Cost ${info_cost:.2f})",
                "net_value": net_value
            }


# Example: Should we run expensive tests before deployment?
analyzer = InformationValueAnalyzer()

# Without testing: deploy and hope for best
ev_without_testing = 0.7 * 100 + 0.3 * (-50)  # 70% success, 30% failure

# With testing: know whether to deploy
ev_with_testing = [
    (0.7, 100),  # 70% chance tests pass → deploy successfully
    (0.3, 0)     # 30% chance tests fail → don't deploy (avoid failure)
]

voi = analyzer.calculate_voi(ev_without_testing, ev_with_testing)
test_cost = 10

decision = analyzer.should_gather_info(voi, test_cost)

print(f"Expected value without testing: ${ev_without_testing:.2f}")
print(f"Value of information: ${voi:.2f}")
print(f"Test cost: ${test_cost:.2f}")
print(f"\nDecision: {decision['decision']}")
print(f"Rationale: {decision['rationale']}")
```

## Uncertainty-Aware Planning

Creating plans that account for uncertainty.

### Contingent Plans

```python
@dataclass
class ContingentPlan:
    """Plan with branches for different scenarios."""
    primary_plan: List[str]
    contingencies: List[dict]  # [{condition, alternative_plan}]
    checkpoints: List[dict]    # [{step, verification}]


class ContingentPlanner:
    """Create plans with explicit handling of uncertainty."""
    
    def create_contingent_plan(self, goal: str, 
                              uncertainties: List[str]) -> ContingentPlan:
        """
        Create plan with contingencies for uncertain aspects.
        
        Returns:
            Plan with primary path and alternatives
        """
        # Primary plan (best case)
        primary_plan = [
            f"Step 1: Attempt {goal}",
            "Step 2: Verify success",
            "Step 3: Finalize"
        ]
        
        # Create contingencies for uncertainties
        contingencies = []
        for uncertainty in uncertainties:
            contingency = {
                "condition": f"If {uncertainty}",
                "alternative_plan": [
                    f"Fallback: Address {uncertainty}",
                    "Re-attempt primary approach",
                    "Proceed to next step"
                ]
            }
            contingencies.append(contingency)
        
        # Checkpoints to detect when contingencies needed
        checkpoints = [
            {
                "after_step": 1,
                "verification": "Check if primary approach working",
                "triggers": ["uncertainty_detected", "error_occurred"]
            }
        ]
        
        return ContingentPlan(
            primary_plan=primary_plan,
            contingencies=contingencies,
            checkpoints=checkpoints
        )
    
    def execute_with_contingencies(self, plan: ContingentPlan) -> dict:
        """Execute plan, switching to contingencies as needed."""
        current_plan = plan.primary_plan
        executed_steps = []
        contingencies_used = []
        
        step_index = 0
        while step_index < len(current_plan):
            step = current_plan[step_index]
            
            print(f"\nExecuting: {step}")
            
            # Check checkpoints
            checkpoint = self._find_checkpoint(plan.checkpoints, step_index)
            if checkpoint:
                verification = self._verify_checkpoint(checkpoint)
                
                if not verification["passed"]:
                    # Switch to contingency
                    print(f"⚠ Checkpoint failed: {verification['reason']}")
                    
                    contingency = self._select_contingency(
                        plan.contingencies, 
                        verification["reason"]
                    )
                    
                    if contingency:
                        print(f"→ Switching to contingency: {contingency['condition']}")
                        contingencies_used.append(contingency)
                        current_plan = contingency["alternative_plan"]
                        step_index = 0
                        continue
            
            # Execute step
            executed_steps.append(step)
            step_index += 1
        
        return {
            "completed": True,
            "steps_executed": len(executed_steps),
            "contingencies_used": len(contingencies_used),
            "path_taken": executed_steps
        }
    
    def _find_checkpoint(self, checkpoints: List[dict], 
                        step_index: int) -> Optional[dict]:
        """Find checkpoint for current step."""
        for cp in checkpoints:
            if cp["after_step"] == step_index:
                return cp
        return None
    
    def _verify_checkpoint(self, checkpoint: dict) -> dict:
        """Verify checkpoint condition."""
        # Simulate verification
        import random
        passed = random.random() > 0.3  # 70% success rate
        
        return {
            "passed": passed,
            "reason": "uncertainty_detected" if not passed else None
        }
    
    def _select_contingency(self, contingencies: List[dict], 
                           reason: str) -> Optional[dict]:
        """Select appropriate contingency for reason."""
        for contingency in contingencies:
            if reason.lower() in contingency["condition"].lower():
                return contingency
        
        return contingencies[0] if contingencies else None


# Example
planner = ContingentPlanner()

plan = planner.create_contingent_plan(
    goal="Deploy new feature",
    uncertainties=[
        "performance degradation",
        "integration issues",
        "data migration problems"
    ]
)

print("Contingent Plan:")
print("\nPrimary Plan:")
for i, step in enumerate(plan.primary_plan, 1):
    print(f"  {i}. {step}")

print("\nContingencies:")
for cont in plan.contingencies:
    print(f"\n  {cont['condition']}:")
    for step in cont["alternative_plan"]:
        print(f"    - {step}")

# Execute
result = planner.execute_with_contingencies(plan)
print(f"\nExecution complete:")
print(f"  Steps: {result['steps_executed']}")
print(f"  Contingencies used: {result['contingencies_used']}")
```

## Best Practices

Guidelines for handling uncertainty effectively.

### 1. Be Explicit About Confidence

```python
# Bad: False certainty
def analyze(data):
    return "The problem is X"  # How sure are we?


# Good: Explicit confidence
def analyze(data):
    if len(data) > 1000 and data.is_representative():
        confidence = 0.9
        analysis = "The problem is likely X"
    elif len(data) > 100:
        confidence = 0.6
        analysis = "The problem may be X, but more data would help"
    else:
        confidence = 0.3
        analysis = "Insufficient data for confident analysis"
    
    return {
        "analysis": analysis,
        "confidence": confidence,
        "data_quality": data.assess_quality()
    }
```

### 2. Track and Verify Assumptions

```python
# Use assumption tracker consistently
tracker = AssumptionTracker()

# Make assumptions explicit
encoding = tracker.make_assumption(
    "file_encoding", 
    "UTF-8",
    "Common default, no BOM detected",
    risk="low"
)

# Verify when possible
actual_encoding = detect_encoding(file)
tracker.verify_assumption("file_encoding", actual_encoding)

# Review unverified assumptions before critical operations
risk = tracker.assess_risk()
if risk["high_risk_assumptions"]:
    print("⚠ High-risk assumptions unverified!")
    for assumption in risk["high_risk_assumptions"]:
        print(f"  - {assumption['key']}: {assumption['rationale']}")
```

### 3. Ask Questions Early

```python
# Bad: Proceed with unclear requirements
def implement_feature(vague_requirement):
    # Guess what they want
    return implementation


# Good: Clarify before implementing
def implement_feature(requirement):
    # Check for ambiguity
    ambiguities = detect_ambiguity(requirement)
    
    if ambiguities:
        questions = generate_clarifying_questions(ambiguities)
        return {
            "status": "needs_clarification",
            "questions": questions
        }
    
    # Proceed with clear requirements
    return implementation
```

### 4. Use Appropriate Confidence Thresholds

```python
class AdaptiveThresholds:
    """Adjust confidence thresholds based on stakes."""
    
    @staticmethod
    def get_threshold(task_criticality: str) -> float:
        """
        Get confidence threshold based on task criticality.
        
        High stakes → require high confidence
        Low stakes → can proceed with lower confidence
        """
        thresholds = {
            "critical": 0.95,  # Life, safety, security
            "high": 0.85,      # Data loss, financial impact
            "medium": 0.70,    # User experience, performance
            "low": 0.50        # Aesthetics, preferences
        }
        
        return thresholds.get(task_criticality, 0.70)


# Example usage
if task_criticality == "critical":
    threshold = 0.95
    if confidence < threshold:
        return "Cannot proceed - confidence too low for critical task"
elif task_criticality == "low":
    threshold = 0.50
    if confidence >= threshold:
        return proceed_with_implementation()
```

## Summary

Effective uncertainty management is essential for robust agents. By explicitly tracking confidence, making assumptions clear, asking clarifying questions, and planning for contingencies, agents can make sound decisions despite incomplete information.

### Key Takeaways

1. **Uncertainty is inevitable**: Real-world agents rarely have complete information

2. **Be explicit about confidence**: Track and communicate confidence levels

3. **Make assumptions visible**: Track assumptions and verify when possible

4. **Ask questions early**: Clarify ambiguity before implementing

5. **Use expected value**: Make decisions based on probabilities and outcomes

6. **Calculate information value**: Gather information when benefits exceed costs

7. **Plan for contingencies**: Create backup plans for uncertain scenarios

8. **Adapt confidence thresholds**: Require higher confidence for higher-stakes decisions

### Uncertainty Management Checklist

- [ ] Identify sources of uncertainty
- [ ] Estimate confidence in decisions
- [ ] Track assumptions explicitly
- [ ] Generate clarifying questions for ambiguities
- [ ] Calculate expected value of options
- [ ] Assess value of gathering more information
- [ ] Create contingent plans for key uncertainties
- [ ] Set appropriate confidence thresholds
- [ ] Verify assumptions when possible
- [ ] Update beliefs as new evidence arrives

## Next Steps

Explore related topics in planning and reasoning:

- **[Chain-of-Thought](chain-of-thought.md)**: Articulate reasoning about uncertain situations
- **[Planning Strategies](planning-strategies.md)**: Create robust plans despite uncertainty
- **[Implementation Plans](implementation-plans.md)**: Document assumptions in plans
- **[Task Tracking](task-tracking.md)**: Track progress through uncertain paths
- **[ReAct Pattern](../agent-architectures/react.md)**: Reason about uncertainty while acting

Master uncertainty management to build agents that make sound decisions despite incomplete information! 🎯
