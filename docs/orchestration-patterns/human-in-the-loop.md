# Human-in-the-Loop

## Table of Contents

- [Introduction](#introduction)
- [When to Involve Humans](#when-to-involve-humans)
- [Approval Workflows](#approval-workflows)
- [Interactive Verification](#interactive-verification)
- [Feedback Integration](#feedback-integration)
- [Oversight Patterns](#oversight-patterns)
- [Escalation Mechanisms](#escalation-mechanisms)
- [Confidence Thresholds](#confidence-thresholds)
- [Human-Agent Collaboration](#human-agent-collaboration)
- [Async Human Input](#async-human-input)
- [Audit and Logging](#audit-and-logging)
- [Progressive Autonomy](#progressive-autonomy)
- [Real-World Applications](#real-world-applications)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Human-in-the-loop (HITL) patterns keep humans involved in agent workflows for oversight, approval, and guidance. Rather than fully autonomous agents, HITL enables:

- **Safety**: Human approval for high-stakes decisions
- **Quality**: Human verification of outputs
- **Learning**: Humans provide feedback for improvement
- **Trust**: Transparency and human control
- **Compliance**: Required human oversight for regulations

> "The most effective agents know when to ask for help."

```
Human-in-the-Loop Flow:

Agent → Decision Point → [Low Risk] → Autonomous Action
                     ↓
               [High Risk]
                     ↓
            Request Human Review
                     ↓
            Human Decision
                     ↓
        [Approved] → Execute
        [Rejected] → Alternative/Stop
        [Modified] → Execute with Changes
```

## When to Involve Humans

Criteria for human involvement.

### Decision Framework

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List

class RiskLevel(Enum):
    """Risk levels for decisions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ConfidenceLevel(Enum):
    """Agent confidence levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class Decision:
    """Agent decision requiring evaluation"""
    action: str
    confidence: ConfidenceLevel
    risk_level: RiskLevel
    cost: float
    reversible: bool
    context: dict

class HITLDecisionFramework:
    """Framework for deciding when to involve humans"""

    def __init__(
        self,
        confidence_threshold: ConfidenceLevel = ConfidenceLevel.MEDIUM,
        cost_threshold: float = 1000.0,
        require_approval_for_high_risk: bool = True
    ):
        self.confidence_threshold = confidence_threshold
        self.cost_threshold = cost_threshold
        self.require_approval_for_high_risk = require_approval_for_high_risk

    def requires_human_approval(self, decision: Decision) -> tuple[bool, str]:
        """Determine if decision requires human approval"""

        # Critical risk always requires approval
        if decision.risk_level == RiskLevel.CRITICAL:
            return True, "Critical risk level"

        # High risk requires approval if configured
        if decision.risk_level == RiskLevel.HIGH and self.require_approval_for_high_risk:
            return True, "High risk level"

        # Low confidence requires approval
        confidence_order = {
            ConfidenceLevel.VERY_LOW: 0,
            ConfidenceLevel.LOW: 1,
            ConfidenceLevel.MEDIUM: 2,
            ConfidenceLevel.HIGH: 3,
            ConfidenceLevel.VERY_HIGH: 4
        }

        if confidence_order[decision.confidence] < confidence_order[self.confidence_threshold]:
            return True, f"Confidence below threshold ({decision.confidence.value})"

        # High cost requires approval
        if decision.cost > self.cost_threshold:
            return True, f"Cost exceeds threshold (${decision.cost})"

        # Irreversible actions require approval
        if not decision.reversible:
            return True, "Action is not reversible"

        return False, "Automatic approval"

    def get_approval_urgency(self, decision: Decision) -> str:
        """Get urgency level for approval request"""
        if decision.risk_level == RiskLevel.CRITICAL:
            return "immediate"
        elif decision.risk_level == RiskLevel.HIGH:
            return "urgent"
        else:
            return "normal"

# Example usage
framework = HITLDecisionFramework(
    confidence_threshold=ConfidenceLevel.MEDIUM,
    cost_threshold=500.0
)

# Test decisions
decisions = [
    Decision(
        action="Send marketing email",
        confidence=ConfidenceLevel.HIGH,
        risk_level=RiskLevel.LOW,
        cost=10.0,
        reversible=True,
        context={}
    ),
    Decision(
        action="Delete customer database",
        confidence=ConfidenceLevel.HIGH,
        risk_level=RiskLevel.CRITICAL,
        cost=0.0,
        reversible=False,
        context={}
    ),
    Decision(
        action="Purchase $2000 of compute credits",
        confidence=ConfidenceLevel.MEDIUM,
        risk_level=RiskLevel.MEDIUM,
        cost=2000.0,
        reversible=False,
        context={}
    )
]

for decision in decisions:
    requires_approval, reason = framework.requires_human_approval(decision)
    urgency = framework.get_approval_urgency(decision)

    print(f"\nAction: {decision.action}")
    print(f"Requires approval: {requires_approval}")
    print(f"Reason: {reason}")
    if requires_approval:
        print(f"Urgency: {urgency}")
```

## Approval Workflows

Implementing human approval processes.

### Approval Manager

```python
from datetime import datetime, timedelta
from typing import Callable, Optional
import time

class ApprovalStatus(Enum):
    """Status of approval request"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    MODIFIED = "modified"

@dataclass
class ApprovalRequest:
    """Request for human approval"""
    request_id: str
    action: str
    description: str
    risk_level: RiskLevel
    proposed_changes: dict
    requested_at: datetime
    requested_by: str
    timeout: timedelta
    status: ApprovalStatus = ApprovalStatus.PENDING
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None
    feedback: Optional[str] = None
    modifications: Optional[dict] = None

class ApprovalManager:
    """Manage human approval workflows"""

    def __init__(self):
        self.pending_requests: dict[str, ApprovalRequest] = {}
        self.completed_requests: dict[str, ApprovalRequest] = {}

    def request_approval(
        self,
        action: str,
        description: str,
        risk_level: RiskLevel,
        proposed_changes: dict,
        requested_by: str = "agent",
        timeout: timedelta = timedelta(hours=24)
    ) -> str:
        """Request human approval"""
        request_id = f"approval_{uuid.uuid4().hex[:8]}"

        request = ApprovalRequest(
            request_id=request_id,
            action=action,
            description=description,
            risk_level=risk_level,
            proposed_changes=proposed_changes,
            requested_at=datetime.now(),
            requested_by=requested_by,
            timeout=timeout
        )

        self.pending_requests[request_id] = request

        print(f"\n{'='*60}")
        print(f"APPROVAL REQUEST: {request_id}")
        print(f"{'='*60}")
        print(f"Action: {action}")
        print(f"Description: {description}")
        print(f"Risk Level: {risk_level.value}")
        print(f"Proposed Changes: {proposed_changes}")
        print(f"Requested By: {requested_by}")
        print(f"Timeout: {timeout}")
        print(f"{'='*60}\n")

        return request_id

    def approve(
        self,
        request_id: str,
        reviewed_by: str,
        feedback: str = ""
    ):
        """Approve request"""
        request = self.pending_requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        request.status = ApprovalStatus.APPROVED
        request.reviewed_at = datetime.now()
        request.reviewed_by = reviewed_by
        request.feedback = feedback

        self._complete_request(request_id)

        print(f"✓ Request {request_id} APPROVED by {reviewed_by}")
        if feedback:
            print(f"  Feedback: {feedback}")

    def reject(
        self,
        request_id: str,
        reviewed_by: str,
        reason: str
    ):
        """Reject request"""
        request = self.pending_requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        request.status = ApprovalStatus.REJECTED
        request.reviewed_at = datetime.now()
        request.reviewed_by = reviewed_by
        request.feedback = reason

        self._complete_request(request_id)

        print(f"✗ Request {request_id} REJECTED by {reviewed_by}")
        print(f"  Reason: {reason}")

    def modify_and_approve(
        self,
        request_id: str,
        reviewed_by: str,
        modifications: dict,
        feedback: str = ""
    ):
        """Approve with modifications"""
        request = self.pending_requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        request.status = ApprovalStatus.MODIFIED
        request.reviewed_at = datetime.now()
        request.reviewed_by = reviewed_by
        request.modifications = modifications
        request.feedback = feedback

        self._complete_request(request_id)

        print(f"✓ Request {request_id} APPROVED WITH MODIFICATIONS by {reviewed_by}")
        print(f"  Modifications: {modifications}")
        if feedback:
            print(f"  Feedback: {feedback}")

    def wait_for_approval(
        self,
        request_id: str,
        poll_interval: float = 1.0
    ) -> ApprovalRequest:
        """Wait for approval decision"""
        request = self.pending_requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        deadline = request.requested_at + request.timeout

        while datetime.now() < deadline:
            # Check if request was completed
            if request_id in self.completed_requests:
                return self.completed_requests[request_id]

            time.sleep(poll_interval)

        # Timeout
        request.status = ApprovalStatus.TIMEOUT
        self._complete_request(request_id)

        return request

    def _complete_request(self, request_id: str):
        """Move request to completed"""
        request = self.pending_requests.pop(request_id)
        self.completed_requests[request_id] = request

    def get_pending_requests(self) -> List[ApprovalRequest]:
        """Get all pending requests"""
        return list(self.pending_requests.values())

    def get_request_status(self, request_id: str) -> Optional[ApprovalStatus]:
        """Get request status"""
        if request_id in self.pending_requests:
            return self.pending_requests[request_id].status
        elif request_id in self.completed_requests:
            return self.completed_requests[request_id].status
        return None

# Example usage
manager = ApprovalManager()

# Agent requests approval for risky action
request_id = manager.request_approval(
    action="delete_old_records",
    description="Delete customer records older than 7 years",
    risk_level=RiskLevel.HIGH,
    proposed_changes={"records_to_delete": 1500},
    timeout=timedelta(seconds=5)
)

# Simulate human review (in practice, this would be async)
import threading

def human_reviewer():
    """Simulate human reviewer"""
    time.sleep(2)  # Review takes 2 seconds

    # Approve with modifications
    manager.modify_and_approve(
        request_id,
        reviewed_by="John Doe",
        modifications={"records_to_delete": 1000},
        feedback="Reduce number of deletions to be safer"
    )

reviewer_thread = threading.Thread(target=human_reviewer)
reviewer_thread.start()

# Agent waits for approval
print("Agent waiting for approval...")
result = manager.wait_for_approval(request_id)

print(f"\nFinal Status: {result.status.value}")
if result.modifications:
    print(f"Using modified parameters: {result.modifications}")
```

## Interactive Verification

Getting human verification during execution.

### Verification Checkpoints

```python
class VerificationCheckpoint:
    """Checkpoint requiring human verification"""

    def __init__(self, name: str, question: str, options: List[str] = None):
        self.name = name
        self.question = question
        self.options = options or ["yes", "no"]
        self.response = None
        self.verified_at = None

    def verify(self, response: str) -> bool:
        """Verify with human response"""
        if self.options and response not in self.options:
            raise ValueError(f"Invalid response. Options: {self.options}")

        self.response = response
        self.verified_at = datetime.now()

        return True

class VerificationWorkflow:
    """Workflow with verification checkpoints"""

    def __init__(self):
        self.checkpoints: List[VerificationCheckpoint] = []
        self.current_checkpoint = 0

    def add_checkpoint(
        self,
        name: str,
        question: str,
        options: List[str] = None
    ):
        """Add verification checkpoint"""
        checkpoint = VerificationCheckpoint(name, question, options)
        self.checkpoints.append(checkpoint)

    def execute_with_verification(self, steps: List[Callable]):
        """Execute workflow with verification checkpoints"""
        checkpoint_idx = 0

        for i, step in enumerate(steps):
            print(f"\nStep {i+1}: Executing...")

            # Execute step
            step()

            # Check if verification needed
            if checkpoint_idx < len(self.checkpoints):
                checkpoint = self.checkpoints[checkpoint_idx]

                print(f"\n{'='*60}")
                print(f"VERIFICATION CHECKPOINT: {checkpoint.name}")
                print(f"{'='*60}")
                print(f"Question: {checkpoint.question}")
                print(f"Options: {checkpoint.options}")
                print(f"{'='*60}")

                # In practice, this would wait for human input
                # For demo, we'll simulate
                response = "yes"  # Simulated human response

                checkpoint.verify(response)
                print(f"✓ Verified: {response}\n")

                checkpoint_idx += 1

                # Stop if verification failed
                if response == "no":
                    print("Workflow stopped by human")
                    return False

        return True

# Example
workflow = VerificationWorkflow()

# Add checkpoints
workflow.add_checkpoint(
    "data_quality",
    "Does the loaded data look correct?",
    ["yes", "no", "unsure"]
)

workflow.add_checkpoint(
    "proceed_analysis",
    "Proceed with analysis?",
    ["yes", "no"]
)

# Define steps
def load_data():
    print("  Loading data...")

def clean_data():
    print("  Cleaning data...")

def analyze_data():
    print("  Analyzing data...")

# Execute with verification
workflow.execute_with_verification([load_data, clean_data, analyze_data])
```

## Feedback Integration

Incorporating human feedback into agent behavior.

### Feedback Loop

```python
@dataclass
class Feedback:
    """Human feedback on agent action"""
    feedback_id: str
    action_id: str
    rating: int  # 1-5 scale
    comments: str
    corrections: Optional[dict]
    timestamp: datetime

class FeedbackManager:
    """Manage and learn from human feedback"""

    def __init__(self):
        self.feedback_history: List[Feedback] = []
        self.action_ratings: dict[str, List[int]] = {}

    def submit_feedback(
        self,
        action_id: str,
        rating: int,
        comments: str = "",
        corrections: dict = None
    ) -> str:
        """Submit feedback"""
        feedback_id = f"feedback_{uuid.uuid4().hex[:8]}"

        feedback = Feedback(
            feedback_id=feedback_id,
            action_id=action_id,
            rating=rating,
            comments=comments,
            corrections=corrections,
            timestamp=datetime.now()
        )

        self.feedback_history.append(feedback)

        # Track ratings by action type
        if action_id not in self.action_ratings:
            self.action_ratings[action_id] = []
        self.action_ratings[action_id].append(rating)

        print(f"Feedback submitted: {rating}/5 - {comments}")

        return feedback_id

    def get_action_quality(self, action_id: str) -> Optional[float]:
        """Get average quality score for action"""
        ratings = self.action_ratings.get(action_id)
        if not ratings:
            return None
        return sum(ratings) / len(ratings)

    def get_improvement_suggestions(self) -> List[dict]:
        """Get suggestions based on feedback"""
        suggestions = []

        for action_id, ratings in self.action_ratings.items():
            avg_rating = sum(ratings) / len(ratings)

            if avg_rating < 3.0:
                # Poor performance
                recent_feedback = [
                    f for f in self.feedback_history
                    if f.action_id == action_id
                ]

                comments = [f.comments for f in recent_feedback if f.comments]

                suggestions.append({
                    "action": action_id,
                    "avg_rating": avg_rating,
                    "issue": "Below acceptable quality",
                    "sample_comments": comments[:3]
                })

        return suggestions

    def apply_corrections(self, action_id: str) -> Optional[dict]:
        """Get corrections from feedback"""
        # Get most recent feedback with corrections
        for feedback in reversed(self.feedback_history):
            if feedback.action_id == action_id and feedback.corrections:
                return feedback.corrections
        return None

# Example
feedback_mgr = FeedbackManager()

# Agent performs actions and gets feedback
action_id = "generate_report_001"

# Simulate feedback
feedback_mgr.submit_feedback(
    action_id=action_id,
    rating=2,
    comments="Report is too technical, needs simpler language",
    corrections={"tone": "simple", "avoid_jargon": True}
)

feedback_mgr.submit_feedback(
    action_id=action_id,
    rating=2,
    comments="Missing key metrics"
)

# Check quality
quality = feedback_mgr.get_action_quality(action_id)
print(f"\nAction quality: {quality:.1f}/5.0")

# Get corrections
corrections = feedback_mgr.apply_corrections(action_id)
print(f"Suggested corrections: {corrections}")

# Get improvement suggestions
suggestions = feedback_mgr.get_improvement_suggestions()
print(f"\nImprovement suggestions:")
for suggestion in suggestions:
    print(f"  - {suggestion['action']}: {suggestion['issue']}")
```

## Oversight Patterns

Monitoring agent actions with human oversight.

### Oversight Dashboard

```python
class AgentAction:
    """Record of agent action"""

    def __init__(
        self,
        action_id: str,
        action_type: str,
        description: str,
        risk_level: RiskLevel,
        auto_approved: bool
    ):
        self.action_id = action_id
        self.action_type = action_type
        self.description = description
        self.risk_level = risk_level
        self.auto_approved = auto_approved
        self.timestamp = datetime.now()
        self.flagged = False
        self.flag_reason = None

class OversightDashboard:
    """Dashboard for monitoring agent actions"""

    def __init__(self):
        self.actions: List[AgentAction] = []
        self.flagged_actions: List[AgentAction] = []

    def log_action(
        self,
        action_type: str,
        description: str,
        risk_level: RiskLevel,
        auto_approved: bool
    ) -> str:
        """Log agent action"""
        action_id = f"action_{uuid.uuid4().hex[:8]}"

        action = AgentAction(
            action_id=action_id,
            action_type=action_type,
            description=description,
            risk_level=risk_level,
            auto_approved=auto_approved
        )

        self.actions.append(action)

        # Auto-flag high-risk actions
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            self.flag_action(action_id, "High-risk action requires review")

        return action_id

    def flag_action(self, action_id: str, reason: str):
        """Flag action for human review"""
        action = next((a for a in self.actions if a.action_id == action_id), None)
        if action:
            action.flagged = True
            action.flag_reason = reason
            self.flagged_actions.append(action)

            print(f"⚠ Action flagged: {action_id}")
            print(f"  Reason: {reason}")

    def get_summary(self) -> dict:
        """Get oversight summary"""
        return {
            "total_actions": len(self.actions),
            "auto_approved": sum(1 for a in self.actions if a.auto_approved),
            "flagged": len(self.flagged_actions),
            "high_risk": sum(
                1 for a in self.actions
                if a.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
            ),
            "by_type": self._count_by_type()
        }

    def _count_by_type(self) -> dict:
        """Count actions by type"""
        counts = {}
        for action in self.actions:
            counts[action.action_type] = counts.get(action.action_type, 0) + 1
        return counts

    def get_flagged_actions(self) -> List[AgentAction]:
        """Get all flagged actions"""
        return self.flagged_actions

    def display_dashboard(self):
        """Display oversight dashboard"""
        summary = self.get_summary()

        print(f"\n{'='*60}")
        print("AGENT OVERSIGHT DASHBOARD")
        print(f"{'='*60}")
        print(f"Total Actions: {summary['total_actions']}")
        print(f"Auto-Approved: {summary['auto_approved']}")
        print(f"Flagged for Review: {summary['flagged']}")
        print(f"High-Risk Actions: {summary['high_risk']}")
        print(f"\nActions by Type:")
        for action_type, count in summary['by_type'].items():
            print(f"  - {action_type}: {count}")

        if self.flagged_actions:
            print(f"\nFlagged Actions:")
            for action in self.flagged_actions:
                print(f"  - {action.action_id}: {action.description}")
                print(f"    Reason: {action.flag_reason}")

        print(f"{'='*60}\n")

# Example
dashboard = OversightDashboard()

# Log various actions
dashboard.log_action(
    "send_email",
    "Send marketing email to 1000 users",
    RiskLevel.LOW,
    auto_approved=True
)

dashboard.log_action(
    "delete_records",
    "Delete 500 old customer records",
    RiskLevel.HIGH,
    auto_approved=False
)

dashboard.log_action(
    "api_call",
    "Call external payment API",
    RiskLevel.MEDIUM,
    auto_approved=True
)

dashboard.log_action(
    "database_migration",
    "Migrate production database schema",
    RiskLevel.CRITICAL,
    auto_approved=False
)

# Display dashboard
dashboard.display_dashboard()
```

## Progressive Autonomy

Gradually increasing agent autonomy based on performance.

### Autonomy Manager

```python
class AutonomyLevel(Enum):
    """Levels of agent autonomy"""
    SUPERVISED = "supervised"          # Always requires approval
    SEMI_AUTONOMOUS = "semi_autonomous"  # Approval for high-risk
    AUTONOMOUS = "autonomous"          # Rarely requires approval
    FULLY_AUTONOMOUS = "fully_autonomous"  # Almost never requires approval

class AutonomyManager:
    """Manage agent autonomy levels based on performance"""

    def __init__(self, initial_level: AutonomyLevel = AutonomyLevel.SUPERVISED):
        self.current_level = initial_level
        self.success_count = 0
        self.failure_count = 0
        self.approval_history: List[bool] = []

    def record_action(self, success: bool, required_approval: bool):
        """Record action outcome"""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        if required_approval:
            self.approval_history.append(success)

    def get_success_rate(self) -> float:
        """Get overall success rate"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total

    def get_approval_success_rate(self) -> float:
        """Get success rate for approved actions"""
        if not self.approval_history:
            return 0.0
        return sum(self.approval_history) / len(self.approval_history)

    def should_increase_autonomy(self) -> bool:
        """Check if autonomy should increase"""
        # Need minimum actions
        if self.success_count + self.failure_count < 10:
            return False

        # High success rate
        if self.get_success_rate() < 0.95:
            return False

        # Good performance on approved actions
        if self.approval_history and self.get_approval_success_rate() < 0.98:
            return False

        return True

    def should_decrease_autonomy(self) -> bool:
        """Check if autonomy should decrease"""
        # Recent failures
        if self.failure_count > 0 and self.get_success_rate() < 0.90:
            return True

        # Poor performance on approved actions
        if self.approval_history and self.get_approval_success_rate() < 0.85:
            return True

        return False

    def update_autonomy(self):
        """Update autonomy level based on performance"""
        if self.should_increase_autonomy():
            self._increase_autonomy()
        elif self.should_decrease_autonomy():
            self._decrease_autonomy()

    def _increase_autonomy(self):
        """Increase autonomy level"""
        levels = [
            AutonomyLevel.SUPERVISED,
            AutonomyLevel.SEMI_AUTONOMOUS,
            AutonomyLevel.AUTONOMOUS,
            AutonomyLevel.FULLY_AUTONOMOUS
        ]

        current_idx = levels.index(self.current_level)
        if current_idx < len(levels) - 1:
            self.current_level = levels[current_idx + 1]
            print(f"✓ Autonomy increased to: {self.current_level.value}")

            # Reset counters
            self.success_count = 0
            self.failure_count = 0
            self.approval_history = []

    def _decrease_autonomy(self):
        """Decrease autonomy level"""
        levels = [
            AutonomyLevel.SUPERVISED,
            AutonomyLevel.SEMI_AUTONOMOUS,
            AutonomyLevel.AUTONOMOUS,
            AutonomyLevel.FULLY_AUTONOMOUS
        ]

        current_idx = levels.index(self.current_level)
        if current_idx > 0:
            self.current_level = levels[current_idx - 1]
            print(f"⚠ Autonomy decreased to: {self.current_level.value}")

            # Reset counters
            self.success_count = 0
            self.failure_count = 0
            self.approval_history = []

    def requires_approval(self, risk_level: RiskLevel) -> bool:
        """Check if approval required for risk level"""
        if self.current_level == AutonomyLevel.SUPERVISED:
            return True
        elif self.current_level == AutonomyLevel.SEMI_AUTONOMOUS:
            return risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        elif self.current_level == AutonomyLevel.AUTONOMOUS:
            return risk_level == RiskLevel.CRITICAL
        else:  # FULLY_AUTONOMOUS
            return False  # Only manual flags

# Example
autonomy = AutonomyManager(initial_level=AutonomyLevel.SUPERVISED)

print(f"Starting at: {autonomy.current_level.value}\n")

# Simulate successful actions
for i in range(15):
    autonomy.record_action(success=True, required_approval=True)

# Check if can increase
autonomy.update_autonomy()

print(f"\nCurrent level: {autonomy.current_level.value}")
print(f"Success rate: {autonomy.get_success_rate():.1%}")
```

## Real-World Applications

### Pattern: Code Review Agent

```python
class CodeReviewAgent:
    """Agent that reviews code with human oversight"""

    def __init__(self):
        self.approval_mgr = ApprovalManager()
        self.feedback_mgr = FeedbackManager()
        self.autonomy_mgr = AutonomyManager(AutonomyLevel.SEMI_AUTONOMOUS)

    def review_code(self, code: str, file_path: str) -> dict:
        """Review code with HITL"""
        print(f"\n{'='*60}")
        print(f"Code Review: {file_path}")
        print(f"{'='*60}\n")

        # Analyze code
        issues = self._analyze_code(code)

        print(f"Found {len(issues)} potential issues")

        # Categorize by severity
        critical_issues = [i for i in issues if i["severity"] == "critical"]
        high_issues = [i for i in issues if i["severity"] == "high"]

        # Determine if human approval needed
        risk_level = RiskLevel.LOW
        if critical_issues:
            risk_level = RiskLevel.CRITICAL
        elif high_issues:
            risk_level = RiskLevel.HIGH

        requires_approval = self.autonomy_mgr.requires_approval(risk_level)

        if requires_approval:
            print(f"\n⚠ Human review required (Risk: {risk_level.value})")

            # Request approval
            request_id = self.approval_mgr.request_approval(
                action=f"approve_code_review_{file_path}",
                description=f"Code review found {len(critical_issues)} critical and {len(high_issues)} high severity issues",
                risk_level=risk_level,
                proposed_changes={"issues": issues},
                timeout=timedelta(seconds=10)
            )

            # Simulate human review
            def simulate_review():
                time.sleep(2)
                self.approval_mgr.approve(
                    request_id,
                    reviewed_by="Senior Developer",
                    feedback="Issues are valid, proceed with fixes"
                )

            threading.Thread(target=simulate_review).start()

            # Wait for approval
            result = self.approval_mgr.wait_for_approval(request_id)

            if result.status != ApprovalStatus.APPROVED:
                print("❌ Review not approved")
                return {"approved": False}

        # Apply fixes
        print("\n✓ Applying fixes...")

        # Record success
        self.autonomy_mgr.record_action(success=True, required_approval=requires_approval)

        # Update autonomy
        self.autonomy_mgr.update_autonomy()

        return {
            "approved": True,
            "issues_found": len(issues),
            "fixes_applied": len(issues),
            "required_human_review": requires_approval
        }

    def _analyze_code(self, code: str) -> List[dict]:
        """Analyze code for issues"""
        # Simplified analysis
        issues = []

        if "eval(" in code:
            issues.append({
                "severity": "critical",
                "message": "Use of eval() is dangerous"
            })

        if "import *" in code:
            issues.append({
                "severity": "high",
                "message": "Wildcard imports should be avoided"
            })

        return issues

# Usage
agent = CodeReviewAgent()

# Review some code
code_sample = """
from os import *
x = eval(user_input)
"""

result = agent.review_code(code_sample, "app.py")
print(f"\nReview Result: {result}")
```

## Summary

Human-in-the-loop patterns ensure safe, trustworthy agent systems:

**Key Concepts**:

- **Approval Workflows**: Human approval for risky actions
- **Verification**: Checkpoints for human validation
- **Feedback**: Learning from human input
- **Oversight**: Monitoring agent behavior
- **Progressive Autonomy**: Earning trust over time

**When to Use HITL**:

- High-risk decisions
- Low confidence actions
- Expensive operations
- Irreversible changes
- Regulatory requirements

**Patterns**:

- **Approval Gate**: Block until human approves
- **Verification Checkpoint**: Confirm at milestones
- **Feedback Loop**: Learn from corrections
- **Oversight Dashboard**: Monitor all actions
- **Progressive Trust**: Increase autonomy gradually

**Best Practices**:

1. Define clear approval criteria
2. Set appropriate timeouts
3. Provide context for decisions
4. Make approval/rejection easy
5. Learn from feedback
6. Track performance metrics
7. Adjust autonomy dynamically

Human-in-the-loop transforms agents from black boxes into collaborative partners.

## Next Steps

Explore related orchestration topics:

- **[Subagent Patterns](subagent-patterns.md)**: Hierarchical oversight
- **[Event-Driven](event-driven.md)**: Reactive human notifications
- **[Workflow Patterns](workflow-patterns.md)**: HITL in workflows

Related areas:

- **[Evaluation](../evaluation/human-evaluation.md)**: Human assessment
- **[Agent Architectures](../agent-architectures/react.md)**: Building interactive agents
