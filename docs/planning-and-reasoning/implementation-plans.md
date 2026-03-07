# Implementation Plans

## Table of Contents

- [Introduction](#introduction)
- [Why Implementation Plans Matter](#why-implementation-plans-matter)
- [Plan Structure](#plan-structure)
- [Level of Detail](#level-of-detail)
- [Creating Implementation Plans](#creating-implementation-plans)
- [Plan Templates](#plan-templates)
- [Technical Specifications](#technical-specifications)
- [Communicating Plans](#communicating-plans)
- [Plan-Driven Development](#plan-driven-development)
- [Coordination and Focus](#coordination-and-focus)
- [Iterative Refinement](#iterative-refinement)
- [Plan Execution](#plan-execution)
- [Common Pitfalls](#common-pitfalls)
- [Best Practices](#best-practices)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Implementation plans are detailed execution specifications created **before** writing code. They serve as blueprints that guide development, coordinate work, and maintain focus on objectives.

> "Give me six hours to chop down a tree and I will spend the first four sharpening the axe." - Abraham Lincoln

Implementation plans:
- **Clarify requirements** before writing code
- **Surface design issues** early when they're cheap to fix
- **Coordinate work** across team members or agents
- **Maintain focus** on the current objective
- **Document decisions** for future reference

### The Planning-Implementation Cycle

```
Traditional Development:        Plan-Driven Development:
─────────────────────          ──────────────────────────

Think → Code → Debug            Plan → Review → Code → Done
   ↓       ↓                           ↓
  ?      More                     Clear
       Debugging                 Direction

Time wasted: ███████             Time wasted: ██
Quality: Medium                  Quality: High
```

### When to Create Implementation Plans

**Always create plans for**:
- Non-trivial features or changes
- Work that affects multiple components
- Code that other agents/developers will review or extend
- Complex algorithms or data structures
- Coordination across multiple tasks

**Skip plans for**:
- Trivial changes (formatting, simple renames)
- Exploratory prototypes
- Immediate bug fixes with obvious solutions

## Why Implementation Plans Matter

Understanding the benefits of detailed planning before implementation.

### Early Problem Detection

Plans expose issues when they're cheap to fix:

```python
# Without plan - discover problem during implementation
def process_data(data):
    # Oops, need to handle missing values
    # Oops, need to normalize data
    # Oops, need to sort before processing
    # Oops, this is slow for large data
    # ...many hours of trial and error...
    pass


# With plan - identify requirements upfront
"""
Implementation Plan: process_data()

Requirements:
- Handle missing values (drop, impute, or raise error?)
- Normalize numerical features (min-max or z-score?)
- Sort data by timestamp before processing
- Efficient for datasets up to 1M rows

Algorithm:
1. Validate input (check types, required columns)
2. Handle missing values using mean imputation
3. Normalize using z-score normalization
4. Sort by 'timestamp' column
5. Process in batches of 10K rows for memory efficiency

Edge cases:
- Empty input → return empty result
- All missing values → raise ValueError
- Unsorted data → auto-sort with warning

Performance: O(n log n) due to sorting
Memory: O(1) using batched processing
"""

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """Process data with validation and normalization."""
    # Implementation now follows clear plan
    pass
```

### Coordination

Plans enable multiple agents/developers to work together:

```python
class CollaborativeAgent:
    """
    Agent that creates and shares implementation plans
    before executing tasks.
    """
    
    def __init__(self, model, shared_workspace):
        self.model = model
        self.workspace = shared_workspace
    
    def execute_task(self, task_description: str) -> dict:
        """
        Execute task with explicit planning phase.
        
        Process:
        1. Create detailed implementation plan
        2. Share plan with workspace for review/coordination
        3. Wait for approval or feedback
        4. Execute according to plan
        """
        # Phase 1: Create plan
        plan = self.create_plan(task_description)
        
        # Phase 2: Share for coordination
        plan_id = self.workspace.share_plan(plan)
        print(f"Plan created: {plan_id}")
        print(f"Plan: {plan}")
        
        # Phase 3: Get approval
        approval = self.workspace.get_approval(plan_id)
        
        if not approval["approved"]:
            print(f"Plan rejected: {approval['feedback']}")
            # Revise and resubmit
            return self.execute_task(task_description)
        
        # Phase 4: Execute
        result = self.execute_plan(plan)
        
        return result
    
    def create_plan(self, task: str) -> dict:
        """Create detailed implementation plan."""
        prompt = f"""
Task: {task}

Create a detailed implementation plan with:

1. **Requirements**: What must the solution satisfy?
2. **Approach**: High-level strategy
3. **Algorithm**: Step-by-step procedure
4. **Data Structures**: What structures will you use?
5. **Edge Cases**: What special cases must be handled?
6. **Testing**: How will you verify correctness?

Implementation Plan:
"""
        
        plan_text = self.model.generate(prompt, temperature=0.3)
        
        return {
            "task": task,
            "plan": plan_text,
            "created_at": datetime.now()
        }
    
    def execute_plan(self, plan: dict) -> dict:
        """Execute implementation according to plan."""
        # Implementation follows the plan
        pass


# Multiple agents can coordinate through shared plans
workspace = SharedWorkspace()

agent1 = CollaborativeAgent(model, workspace)
agent2 = CollaborativeAgent(model, workspace)

# Agent 1 creates plan
agent1.execute_task("Implement user authentication")

# Agent 2 can see plan and coordinate
plans = workspace.get_pending_plans()
print(f"Agent 2 sees {len(plans)} pending plans")
```

### Maintaining Focus

Plans keep agents on track during long implementations:

```python
class FocusedAgent:
    """
    Agent that uses implementation plans to maintain focus
    and avoid scope creep.
    """
    
    def __init__(self, model):
        self.model = model
        self.current_plan: Optional[dict] = None
        self.plan_steps: List[str] = []
        self.completed_steps: Set[int] = set()
    
    def execute_with_plan(self, task: str) -> dict:
        """Execute task while maintaining focus on plan."""
        # Create plan
        self.current_plan = self.create_detailed_plan(task)
        self.plan_steps = self.current_plan["steps"]
        
        print("Implementation Plan:")
        for i, step in enumerate(self.plan_steps, 1):
            print(f"  {i}. {step}")
        print()
        
        # Execute each step
        for i, step in enumerate(self.plan_steps):
            print(f"\n{'='*50}")
            print(f"Step {i+1}/{len(self.plan_steps)}: {step}")
            print(f"{'='*50}")
            
            # Execute step with focus check
            result = self.execute_step_with_focus(i, step)
            
            if result["focused"]:
                self.completed_steps.add(i)
                print(f"✓ Completed: {step}")
            else:
                print(f"⚠ Went off-track: {result['deviation']}")
                print("Refocusing on plan...")
                # Retry step
                i -= 1
        
        return {
            "success": len(self.completed_steps) == len(self.plan_steps),
            "completed": len(self.completed_steps),
            "total": len(self.plan_steps)
        }
    
    def execute_step_with_focus(self, step_index: int, step: str) -> dict:
        """
        Execute step while checking for deviation from plan.
        
        Returns:
            Dictionary with 'focused' flag and 'deviation' if off-track
        """
        prompt = f"""
Current task: {step}

Plan context:
{self._get_plan_context()}

Execute this specific step. Stay focused on the plan.

Action:
"""
        
        response = self.model.generate(prompt, temperature=0.3)
        
        # Check if response is focused on current step
        is_focused = self._check_focus(step, response)
        
        if is_focused:
            return {"focused": True, "output": response}
        else:
            return {
                "focused": False,
                "deviation": "Implementation deviated from planned step",
                "output": response
            }
    
    def _get_plan_context(self) -> str:
        """Get context from plan to maintain focus."""
        completed = [self.plan_steps[i] for i in sorted(self.completed_steps)]
        remaining = [self.plan_steps[i] for i in range(len(self.plan_steps))
                    if i not in self.completed_steps]
        
        context = f"""
Completed steps:
{chr(10).join(f'  ✓ {s}' for s in completed)}

Remaining steps:
{chr(10).join(f'  ○ {s}' for s in remaining)}
"""
        return context
    
    def _check_focus(self, step: str, response: str) -> bool:
        """Check if response is focused on current step."""
        # Simple heuristic: check if key terms from step appear in response
        step_keywords = set(step.lower().split())
        response_keywords = set(response.lower().split())
        
        overlap = step_keywords & response_keywords
        return len(overlap) >= len(step_keywords) * 0.5


# Example usage
agent = FocusedAgent(model)

result = agent.execute_with_plan(
    "Build REST API for user management with CRUD operations"
)

print(f"\n{'='*50}")
print(f"Result: {result['completed']}/{result['total']} steps completed")
print(f"Success: {result['success']}")
```

## Plan Structure

Elements of a well-structured implementation plan.

### Essential Components

```python
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class ImplementationPlan:
    """Structured implementation plan."""
    
    # What are we building?
    task: str
    objective: str
    
    # What must it satisfy?
    requirements: List[str]
    constraints: List[str]
    
    # How will we build it?
    approach: str
    algorithm: List[str]  # Step-by-step procedure
    data_structures: List[str]
    
    # What could go wrong?
    edge_cases: List[str]
    error_handling: List[str]
    
    # How will we verify?
    test_cases: List[str]
    acceptance_criteria: List[str]
    
    # Supporting info
    assumptions: List[str]
    dependencies: List[str]
    estimated_effort: str
    
    # Metadata
    created_by: str
    created_at: datetime
    version: int = 1
    
    def to_markdown(self) -> str:
        """Convert plan to markdown document."""
        md = f"""# Implementation Plan: {self.task}

## Objective

{self.objective}

## Requirements

{self._list_items(self.requirements)}

## Constraints

{self._list_items(self.constraints)}

## Approach

{self.approach}

## Algorithm

{self._numbered_items(self.algorithm)}

## Data Structures

{self._list_items(self.data_structures)}

## Edge Cases

{self._list_items(self.edge_cases)}

## Error Handling

{self._list_items(self.error_handling)}

## Test Cases

{self._list_items(self.test_cases)}

## Acceptance Criteria

{self._list_items(self.acceptance_criteria)}

## Assumptions

{self._list_items(self.assumptions)}

## Dependencies

{self._list_items(self.dependencies)}

## Estimated Effort

{self.estimated_effort}

---

*Created by {self.created_by} on {self.created_at.strftime('%Y-%m-%d %H:%M')}*
*Version: {self.version}*
"""
        return md
    
    def _list_items(self, items: List[str]) -> str:
        """Format list items."""
        if not items:
            return "*(None specified)*"
        return '\n'.join(f'- {item}' for item in items)
    
    def _numbered_items(self, items: List[str]) -> str:
        """Format numbered items."""
        if not items:
            return "*(None specified)*"
        return '\n'.join(f'{i}. {item}' for i, item in enumerate(items, 1))


# Example plan
plan = ImplementationPlan(
    task="Implement binary search tree",
    objective="Create efficient BST with insert, search, and delete operations",
    
    requirements=[
        "Support integer keys",
        "Maintain BST property (left < parent < right)",
        "O(log n) average case for operations",
        "Handle duplicate keys"
    ],
    
    constraints=[
        "No external libraries except standard library",
        "Memory efficient (no parent pointers)",
        "Thread-safe not required"
    ],
    
    approach="""
Implement recursive BST using Node class with left/right pointers.
Insert and search are straightforward recursion. Delete handles
three cases: leaf, one child, two children (use successor).
""",
    
    algorithm=[
        "Create Node class with key, left, right attributes",
        "Implement insert(key) with recursive insertion",
        "Implement search(key) with recursive search",
        "Implement delete(key) with three-case logic",
        "Implement find_min() helper for delete",
        "Add in-order traversal for testing"
    ],
    
    data_structures=[
        "Node class: stores key and child pointers",
        "BST class: stores root pointer and methods"
    ],
    
    edge_cases=[
        "Empty tree",
        "Single node tree",
        "Deleting root",
        "Duplicate keys (reject or handle?)",
        "Searching for non-existent key"
    ],
    
    error_handling=[
        "Raise ValueError for invalid inputs (None, non-integer)",
        "Return None for search misses",
        "Handle deletion of non-existent key gracefully"
    ],
    
    test_cases=[
        "Insert into empty tree",
        "Insert multiple values and verify in-order traversal",
        "Search for existing and non-existing keys",
        "Delete leaf node",
        "Delete node with one child",
        "Delete node with two children",
        "Delete root node"
    ],
    
    acceptance_criteria=[
        "All test cases pass",
        "In-order traversal produces sorted sequence",
        "No memory leaks (verified with simple test)",
        "Operations complete in reasonable time"
    ],
    
    assumptions=[
        "Keys are integers",
        "No balancing required (simple BST, not AVL/Red-Black)",
        "Single-threaded usage"
    ],
    
    dependencies=[],
    
    estimated_effort="2-3 hours",
    
    created_by="Agent",
    created_at=datetime.now()
)

print(plan.to_markdown())
```

### Plan Checklist

```python
class PlanValidator:
    """Validate that implementation plan is complete."""
    
    @staticmethod
    def validate(plan: ImplementationPlan) -> dict:
        """
        Check if plan is complete and well-formed.
        
        Returns:
            Dictionary with validation results
        """
        issues = []
        warnings = []
        
        # Required fields
        if not plan.task:
            issues.append("Missing task description")
        
        if not plan.objective:
            issues.append("Missing objective")
        
        if not plan.requirements:
            warnings.append("No requirements specified")
        
        if not plan.algorithm or len(plan.algorithm) < 2:
            issues.append("Algorithm not detailed enough (need at least 2 steps)")
        
        if not plan.test_cases:
            warnings.append("No test cases specified")
        
        # Completeness checks
        if not plan.edge_cases:
            warnings.append("No edge cases identified - are you sure?")
        
        if not plan.error_handling:
            warnings.append("No error handling strategy specified")
        
        # Quality checks
        if len(plan.algorithm) > 20:
            warnings.append("Algorithm has >20 steps - consider breaking down")
        
        if not plan.acceptance_criteria:
            warnings.append("No acceptance criteria - how will you know it's done?")
        
        is_valid = len(issues) == 0
        
        return {
            "valid": is_valid,
            "issues": issues,
            "warnings": warnings,
            "score": 100 - (len(issues) * 20 + len(warnings) * 5)
        }


# Validate plan
validation = PlanValidator.validate(plan)

print("Plan Validation:")
print(f"  Valid: {validation['valid']}")
print(f"  Score: {validation['score']}/100")

if validation['issues']:
    print("\nIssues:")
    for issue in validation['issues']:
        print(f"  ✗ {issue}")

if validation['warnings']:
    print("\nWarnings:")
    for warning in validation['warnings']:
        print(f"  ⚠ {warning}")
```

## Level of Detail

Determining appropriate plan granularity.

### Too Little Detail

```python
# Bad: Too vague
"""
Implementation Plan: User authentication

1. Setup database
2. Create login
3. Add security
4. Test it

Done!
"""

# Problem: Doesn't specify:
# - What database schema?
# - What authentication method?
# - What security measures?
# - What test cases?
```

### Too Much Detail

```python
# Bad: Too specific
"""
Implementation Plan: User authentication

1. Import hashlib library
2. Create variable named 'salt' of type string
3. Generate salt using os.urandom(16)
4. Convert salt to hex string using .hex() method
5. Create variable named 'password' of type string
6. Get password from user input
7. Concatenate salt + password
8. Hash using hashlib.sha256()
9. Convert hash to hex string
... (50 more micro-steps)
"""

# Problem: Too prescriptive, leaves no room for
# implementation decisions, basically pseudo-code
```

### Just Right

```python
# Good: Right level of abstraction
"""
Implementation Plan: User authentication

Requirements:
- Store user credentials securely
- Prevent brute force attacks
- Session management for logged-in users

Algorithm:
1. Design user database schema
   - user_id (primary key)
   - username (unique)
   - password_hash (hashed + salted)
   - created_at, last_login

2. Implement registration
   - Validate username (alphanumeric, 3-20 chars)
   - Validate password (min 8 chars, complexity)
   - Generate salt, hash password
   - Store in database

3. Implement login
   - Look up user by username
   - Verify password hash
   - Create session token (JWT)
   - Return token to client

4. Implement session validation
   - Extract token from request header
   - Verify token signature and expiration
   - Load user from token claims

5. Add security measures
   - Rate limiting (max 5 attempts per minute)
   - Account lockout (after 10 failed attempts)
   - Secure token storage (httpOnly cookies)

Edge Cases:
- Duplicate username during registration → reject
- Invalid credentials → generic error message
- Expired token → return 401, require re-login
- Missing token → return 401

Testing:
- Register new user successfully
- Reject weak passwords
- Login with correct credentials
- Reject wrong password
- Access protected endpoint with valid token
- Reject expired token
"""

# Sweet spot: Enough detail to guide implementation,
# but flexible on specific technical choices
```

### Adaptive Detail

```python
class AdaptivePlanner:
    """
    Create plans with appropriate detail level based on
    task complexity and agent capability.
    """
    
    def create_plan(self, task: str, complexity: str, 
                   agent_experience: str) -> str:
        """
        Create plan with detail level adapted to context.
        
        Args:
            task: Task description
            complexity: "simple" | "moderate" | "complex"
            agent_experience: "novice" | "intermediate" | "expert"
            
        Returns:
            Implementation plan with appropriate detail
        """
        # Determine detail level
        if complexity == "simple" and agent_experience == "expert":
            detail_level = "high_level"
        elif complexity == "complex" or agent_experience == "novice":
            detail_level = "detailed"
        else:
            detail_level = "moderate"
        
        prompt = f"""
Task: {task}
Complexity: {complexity}
Agent experience: {agent_experience}
Detail level: {detail_level}

Create implementation plan with {detail_level} detail:
- high_level: Major steps only, assume expertise
- moderate: Clear steps with key decisions highlighted
- detailed: Specific steps with rationale and alternatives

Plan:
"""
        
        return self.model.generate(prompt, temperature=0.3)


planner = AdaptivePlanner()

# Expert tackling simple task → high-level plan
plan1 = planner.create_plan(
    task="Add logging to function",
    complexity="simple",
    agent_experience="expert"
)

# Novice tackling complex task → detailed plan
plan2 = planner.create_plan(
    task="Implement distributed cache",
    complexity="complex",
    agent_experience="novice"
)
```

## Creating Implementation Plans

Automated plan generation from requirements.

### Plan Generator

```python
class PlanGenerator:
    """Generate implementation plans from task descriptions."""
    
    def __init__(self, model):
        self.model = model
    
    def generate_plan(self, task: str, context: dict = None) -> ImplementationPlan:
        """
        Generate comprehensive implementation plan.
        
        Args:
            task: Task description
            context: Additional context (requirements, constraints, etc.)
            
        Returns:
            Structured implementation plan
        """
        context = context or {}
        
        prompt = f"""
Task: {task}

{self._format_context(context)}

Create a detailed implementation plan with these sections:

1. **Objective**: One sentence describing what we're building
2. **Requirements**: List of must-have features/properties
3. **Constraints**: Limitations or restrictions
4. **Approach**: High-level strategy (2-3 sentences)
5. **Algorithm**: Step-by-step procedure (numbered list)
6. **Data Structures**: What data structures will be used
7. **Edge Cases**: Special cases that must be handled
8. **Error Handling**: How errors will be handled
9. **Test Cases**: How to verify correctness
10. **Acceptance Criteria**: What defines "done"

Implementation Plan:
"""
        
        response = self.model.generate(prompt, temperature=0.3)
        
        # Parse response into structured plan
        plan = self._parse_plan(task, response)
        
        return plan
    
    def _format_context(self, context: dict) -> str:
        """Format context dictionary for prompt."""
        if not context:
            return ""
        
        lines = ["Context:"]
        for key, value in context.items():
            lines.append(f"- {key}: {value}")
        
        return '\n'.join(lines)
    
    def _parse_plan(self, task: str, response: str) -> ImplementationPlan:
        """Parse LLM response into structured plan."""
        # Simple parsing - in practice, would be more robust
        sections = {}
        current_section = None
        current_content = []
        
        for line in response.split('\n'):
            line = line.strip()
            
            # Check if line is a section header
            if line.startswith('**') and line.endswith('**'):
                # Save previous section
                if current_section:
                    sections[current_section] = current_content
                
                # Start new section
                current_section = line.strip('*').strip(':').lower()
                current_content = []
            elif line and current_section:
                current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = current_content
        
        # Build plan
        return ImplementationPlan(
            task=task,
            objective=' '.join(sections.get('objective', [])),
            requirements=self._extract_list(sections.get('requirements', [])),
            constraints=self._extract_list(sections.get('constraints', [])),
            approach=' '.join(sections.get('approach', [])),
            algorithm=self._extract_list(sections.get('algorithm', [])),
            data_structures=self._extract_list(sections.get('data structures', [])),
            edge_cases=self._extract_list(sections.get('edge cases', [])),
            error_handling=self._extract_list(sections.get('error handling', [])),
            test_cases=self._extract_list(sections.get('test cases', [])),
            acceptance_criteria=self._extract_list(sections.get('acceptance criteria', [])),
            assumptions=[],
            dependencies=[],
            estimated_effort="TBD",
            created_by="PlanGenerator",
            created_at=datetime.now()
        )
    
    def _extract_list(self, lines: List[str]) -> List[str]:
        """Extract list items from lines."""
        items = []
        for line in lines:
            line = line.strip()
            # Remove list markers
            if line.startswith('-') or line.startswith('*'):
                line = line[1:].strip()
            elif line and line[0].isdigit() and line[1] == '.':
                line = line[2:].strip()
            
            if line:
                items.append(line)
        
        return items


# Example usage
generator = PlanGenerator(model)

plan = generator.generate_plan(
    task="Implement LRU cache with O(1) get and put operations",
    context={
        "capacity": "Configurable maximum size",
        "eviction": "Least Recently Used policy",
        "requirements": "Thread-safe not required"
    }
)

print(plan.to_markdown())
```

## Plan Execution

Implementing code according to the plan.

### Plan-Driven Executor

```python
class PlanDrivenExecutor:
    """Execute implementation following a detailed plan."""
    
    def __init__(self, model):
        self.model = model
    
    def execute_plan(self, plan: ImplementationPlan) -> dict:
        """
        Execute implementation plan step by step.
        
        Returns:
            Dictionary with implementation results
        """
        print(f"Executing plan: {plan.task}")
        print(f"Algorithm has {len(plan.algorithm)} steps\n")
        
        implementation = {
            "task": plan.task,
            "code": "",
            "tests": "",
            "documentation": "",
            "steps_completed": 0
        }
        
        # Execute algorithm steps sequentially
        for i, step in enumerate(plan.algorithm, 1):
            print(f"\nStep {i}/{len(plan.algorithm)}: {step}")
            
            step_result = self._execute_step(plan, i, step, implementation)
            
            implementation["code"] += step_result["code"]
            implementation["steps_completed"] = i
            
            print(f"✓ Completed: {step}")
        
        # Generate tests based on test cases
        implementation["tests"] = self._generate_tests(plan)
        
        # Generate documentation
        implementation["documentation"] = self._generate_docs(plan)
        
        return implementation
    
    def _execute_step(self, plan: ImplementationPlan, step_num: int, 
                     step: str, current_impl: dict) -> dict:
        """Execute a single algorithm step."""
        prompt = f"""
Task: {plan.task}
Objective: {plan.objective}

Current step ({step_num}/{len(plan.algorithm)}):
{step}

Context from plan:
- Requirements: {', '.join(plan.requirements[:3])}
- Approach: {plan.approach[:200]}
- Data structures: {', '.join(plan.data_structures)}

Current implementation:
```python
{current_impl['code'][-500:] if current_impl['code'] else '# Empty'}
```

Generate Python code for this step only. Provide clean, documented code.

Code:
```python
"""
        
        code = self.model.generate(prompt, temperature=0.2)
        
        # Extract code from markdown
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        
        return {"code": code + "\n\n"}
    
    def _generate_tests(self, plan: ImplementationPlan) -> str:
        """Generate tests based on test cases in plan."""
        prompt = f"""
Task: {plan.task}

Test cases from plan:
{chr(10).join(f'- {tc}' for tc in plan.test_cases)}

Edge cases to cover:
{chr(10).join(f'- {ec}' for ec in plan.edge_cases)}

Generate Python pytest tests for these cases.

Tests:
```python
"""
        
        tests = self.model.generate(prompt, temperature=0.2)
        
        if "```python" in tests:
            tests = tests.split("```python")[1].split("```")[0]
        
        return tests
    
    def _generate_docs(self, plan: ImplementationPlan) -> str:
        """Generate documentation from plan."""
        return plan.to_markdown()


# Example
executor = PlanDrivenExecutor(model)

result = executor.execute_plan(plan)

print(f"\n{'='*50}")
print("Execution complete!")
print(f"Steps completed: {result['steps_completed']}/{len(plan.algorithm)}")
print(f"Code length: {len(result['code'])} chars")
print(f"Tests generated: {len(result['tests'])} chars")
```

## Best Practices

Guidelines for effective implementation planning.

### 1. Start with Why

Always state the objective clearly:

```python
# Bad
"""
Plan: Build caching system
1. Create cache class
2. Add methods
3. Test
"""

# Good
"""
Plan: Build caching system

Objective: Reduce database load by caching frequently accessed data,
targeting 80% cache hit rate for user profile lookups.

Why: Database queries are slow (200ms avg) and repetitive. Caching
can reduce response time to <10ms for cached data.
...
"""
```

### 2. Make Assumptions Explicit

```python
plan = ImplementationPlan(
    task="Build search feature",
    assumptions=[
        "Dataset fits in memory (~100K records)",
        "Case-insensitive matching is sufficient",
        "No fuzzy matching required",
        "Single-threaded access only"
    ],
    # ...
)

# These assumptions guide design decisions
# If assumptions change, plan needs revision
```

### 3. Specify Acceptance Criteria

```python
plan = ImplementationPlan(
    task="Optimize query performance",
    acceptance_criteria=[
        "Queries complete in <100ms for 99th percentile",
        "No regression in correctness (all existing tests pass)",
        "Memory usage does not exceed 2x current baseline",
        "Implementation verified on production-scale dataset"
    ],
    # ...
)

# Clear criteria prevent scope creep and endless optimization
```

### 4. Plan for Failure

```python
plan = ImplementationPlan(
    task="API integration",
    error_handling=[
        "Network timeout: Retry with exponential backoff (max 3 attempts)",
        "Rate limit exceeded: Queue request and retry after delay",
        "Invalid response: Log error, return cached data if available",
        "Authentication failure: Refresh token and retry once"
    ],
    edge_cases=[
        "Empty response",
        "Malformed JSON",
        "Unexpected status codes (500, 503)",
        "Connection refused"
    ],
    # ...
)
```

### 5. Review and Refine

```python
class PlanReviewer:
    """Review and provide feedback on implementation plans."""
    
    def review_plan(self, plan: ImplementationPlan) -> dict:
        """Review plan and suggest improvements."""
        feedback = {
            "strengths": [],
            "weaknesses": [],
            "suggestions": []
        }
        
        # Check completeness
        if len(plan.requirements) >= 3:
            feedback["strengths"].append("Well-defined requirements")
        else:
            feedback["weaknesses"].append("Requirements seem incomplete")
            feedback["suggestions"].append("Add more specific requirements")
        
        # Check algorithm detail
        if 5 <= len(plan.algorithm) <= 15:
            feedback["strengths"].append("Algorithm well-structured")
        elif len(plan.algorithm) < 5:
            feedback["weaknesses"].append("Algorithm lacks detail")
            feedback["suggestions"].append("Break down steps further")
        else:
            feedback["weaknesses"].append("Algorithm too detailed")
            feedback["suggestions"].append("Consolidate into higher-level steps")
        
        # Check test coverage
        if plan.test_cases and plan.edge_cases:
            feedback["strengths"].append("Good test coverage planned")
        else:
            feedback["weaknesses"].append("Insufficient test planning")
            feedback["suggestions"].append("Add specific test cases")
        
        return feedback


reviewer = PlanReviewer()
feedback = reviewer.review_plan(plan)

print("Plan Review:")
print("\nStrengths:")
for s in feedback["strengths"]:
    print(f"  ✓ {s}")

print("\nWeaknesses:")
for w in feedback["weaknesses"]:
    print(f"  ✗ {w}")

print("\nSuggestions:")
for s in feedback["suggestions"]:
    print(f"  → {s}")
```

## Summary

Implementation plans transform vague tasks into clear execution blueprints. By investing time in planning before coding, we surface issues early, coordinate work effectively, and maintain focus through complex implementations.

### Key Takeaways

1. **Plans save time**: Early problem detection prevents costly rework

2. **Structure matters**: Complete plans include requirements, approach, algorithm, edge cases, testing

3. **Right detail level**: Too vague is useless, too specific is constraining

4. **Plans enable coordination**: Multiple agents can work together through shared plans

5. **Plans maintain focus**: Explicit steps prevent scope creep and distraction

6. **Acceptance criteria define "done"**: Clear criteria prevent endless tweaking

7. **Review before executing**: Plan reviews catch issues before implementation

### When to Create Implementation Plans

**Always plan for**:
- Complex features or algorithms
- Work affecting multiple components  
- Tasks requiring coordination
- Critical or high-risk changes

**Consider skipping for**:
- Trivial changes
- Exploratory prototypes
- Immediate bug fixes with obvious solutions

### Implementation Plan Checklist

- [ ] Clear objective stated
- [ ] Requirements listed (3+ items)
- [ ] Constraints identified
- [ ] High-level approach described
- [ ] Algorithm broken into steps (5-15 steps)
- [ ] Data structures specified
- [ ] Edge cases identified (3+ cases)
- [ ] Error handling strategy defined
- [ ] Test cases listed (5+ cases)
- [ ] Acceptance criteria specified
- [ ] Assumptions made explicit
- [ ] Dependencies noted

## Next Steps

Continue exploring planning and execution:

- **[Planning Strategies](planning-strategies.md)**: Learn multi-step planning approaches
- **[Task Decomposition](task-decomposition.md)**: Break complex tasks into subtasks
- **[Task Tracking](task-tracking.md)**: Track progress through implementation
- **[Chain-of-Thought](chain-of-thought.md)**: Reason about implementation decisions
- **[Uncertainty](uncertainty.md)**: Handle ambiguity in requirements and design

Master implementation planning to build agents that execute complex tasks with clarity and coordination! 📋
