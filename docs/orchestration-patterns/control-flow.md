# Control Flow

## Table of Contents

- [Introduction](#introduction)
- [Conditional Execution](#conditional-execution)
- [Looping Constructs](#looping-constructs)
- [Branching Logic](#branching-logic)
- [Dynamic Routing](#dynamic-routing)
- [Pattern Matching](#pattern-matching)
- [Error-Driven Control Flow](#error-driven-control-flow)
- [Early Exit Patterns](#early-exit-patterns)
- [Recursive Control Flow](#recursive-control-flow)
- [Computed Routing](#computed-routing)
- [Guard Conditions](#guard-conditions)
- [Complex Control Structures](#complex-control-structures)
- [Control Flow Optimization](#control-flow-optimization)
- [Real-World Examples](#real-world-examples)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Control flow determines **how execution moves** through an agent workflow. While traditional programming uses if statements and loops, agent control flow is more dynamic: routing decisions based on LLM outputs, conditional execution based on tool results, and adaptive iteration based on quality checks.

> "Control flow is the nervous system of agent orchestration - routing signals where they need to go."

Effective control flow enables agents to:
- **Make intelligent routing decisions** based on runtime conditions
- **Adapt execution paths** based on intermediate results
- **Handle errors gracefully** with fallbacks and retries
- **Optimize paths** by skipping unnecessary work
- **Iterate intelligently** until quality thresholds are met

### Control Flow Primitives

```
Conditionals:    if/then/else routing
Loops:           while/for iteration
Branches:        parallel execution paths
Switches:        multi-way routing
Guards:          precondition checking
Exceptions:      error-driven flow
```

## Conditional Execution

The foundation of dynamic control flow: execute different code based on conditions.

### Basic Conditionals

```python
from typing import Callable, Any

class ConditionalExecutor:
    """Execute code conditionally"""
    
    def __init__(self, condition: Callable[[Any], bool]):
        self.condition = condition
        self.if_true: Callable = None
        self.if_false: Callable = None
    
    def then(self, handler: Callable) -> "ConditionalExecutor":
        """Set true handler"""
        self.if_true = handler
        return self
    
    def otherwise(self, handler: Callable) -> "ConditionalExecutor":
        """Set false handler"""
        self.if_false = handler
        return self
    
    def execute(self, data: Any) -> Any:
        """Execute conditional logic"""
        if self.condition(data):
            if self.if_true:
                return self.if_true(data)
        else:
            if self.if_false:
                return self.if_false(data)
        
        return data

# Example: Content routing
def is_technical(content: dict) -> bool:
    """Check if content is technical"""
    return "code" in content or "technical" in content.get("tags", [])

def handle_technical(content: dict) -> dict:
    """Handle technical content"""
    return {
        **content,
        "route": "technical_team",
        "reviewers": ["tech_lead", "senior_dev"]
    }

def handle_general(content: dict) -> dict:
    """Handle general content"""
    return {
        **content,
        "route": "general_team",
        "reviewers": ["editor"]
    }

router = ConditionalExecutor(is_technical)
router.then(handle_technical).otherwise(handle_general)

content = {"text": "Here's some code...", "tags": ["technical"]}
result = router.execute(content)
# Routes to technical_team
```

### Chained Conditionals

```python
class ConditionalChain:
    """Chain multiple conditions"""
    
    def __init__(self):
        self.conditions: List[tuple] = []
        self.default_handler: Callable = None
    
    def when(
        self,
        condition: Callable[[Any], bool],
        handler: Callable[[Any], Any]
    ) -> "ConditionalChain":
        """Add condition and handler"""
        self.conditions.append((condition, handler))
        return self
    
    def default(self, handler: Callable[[Any], Any]) -> "ConditionalChain":
        """Set default handler"""
        self.default_handler = handler
        return self
    
    def execute(self, data: Any) -> Any:
        """Execute first matching condition"""
        for condition, handler in self.conditions:
            if condition(data):
                return handler(data)
        
        # No condition matched, use default
        if self.default_handler:
            return self.default_handler(data)
        
        return data

# Example: Priority routing
chain = ConditionalChain()

chain.when(
    lambda task: task.get("priority") == "urgent",
    lambda task: handle_urgent(task)
).when(
    lambda task: task.get("priority") == "high",
    lambda task: handle_high_priority(task)
).when(
    lambda task: task.get("priority") == "normal",
    lambda task: handle_normal(task)
).default(
    lambda task: handle_low_priority(task)
)

task = {"description": "Fix bug", "priority": "urgent"}
result = chain.execute(task)
```

### LLM-Driven Conditionals

```python
from enum import Enum

class Decision(Enum):
    """LLM decision outcomes"""
    APPROVED = "approved"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"

def llm_conditional(
    data: Any,
    decision_prompt: str,
    handlers: dict
) -> Any:
    """Use LLM to make routing decision"""
    
    # Ask LLM to make decision
    prompt = f"""
    {decision_prompt}
    
    Data: {data}
    
    Decide: approved, needs_review, or rejected
    
    Return only the decision word.
    """
    
    llm_response = llm.complete(prompt).strip().lower()
    
    # Parse decision
    try:
        decision = Decision(llm_response)
    except ValueError:
        decision = Decision.NEEDS_REVIEW  # Default to review
    
    # Route to handler
    handler = handlers.get(decision)
    if handler:
        return handler(data)
    
    return data

# Example: Content moderation
result = llm_conditional(
    data={"content": "User-generated content..."},
    decision_prompt="Review this content for appropriateness",
    handlers={
        Decision.APPROVED: auto_publish,
        Decision.NEEDS_REVIEW: queue_for_human,
        Decision.REJECTED: auto_reject
    }
)
```

## Looping Constructs

Repetitive execution with various termination conditions.

### While Loop

```python
class WhileLoop:
    """Execute while condition holds"""
    
    def __init__(
        self,
        condition: Callable[[Any], bool],
        body: Callable[[Any], Any],
        max_iterations: int = 100
    ):
        self.condition = condition
        self.body = body
        self.max_iterations = max_iterations
    
    def execute(self, initial_state: Any) -> Any:
        """Execute loop"""
        state = initial_state
        iteration = 0
        
        while iteration < self.max_iterations:
            if not self.condition(state):
                break
            
            state = self.body(state)
            iteration += 1
        
        return state

# Example: Iterative refinement
def quality_insufficient(result: dict) -> bool:
    """Check if quality needs improvement"""
    return result.get("quality_score", 0) < 0.85

def improve_quality(result: dict) -> dict:
    """Improve result quality"""
    improved_text = refine_text(result["text"])
    new_score = evaluate_quality(improved_text)
    
    return {
        "text": improved_text,
        "quality_score": new_score,
        "iterations": result.get("iterations", 0) + 1
    }

refinement = WhileLoop(
    condition=quality_insufficient,
    body=improve_quality,
    max_iterations=5
)

initial = {"text": "Draft text", "quality_score": 0.6}
final = refinement.execute(initial)
```

### For-Each Loop

```python
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

class ForEachLoop:
    """Process each item in collection"""
    
    def __init__(
        self,
        processor: Callable[[Any], Any],
        parallel: bool = False,
        max_workers: int = 4
    ):
        self.processor = processor
        self.parallel = parallel
        self.max_workers = max_workers
    
    def execute(self, items: List[Any]) -> List[Any]:
        """Process all items"""
        if self.parallel:
            return self._execute_parallel(items)
        else:
            return self._execute_sequential(items)
    
    def _execute_sequential(self, items: List[Any]) -> List[Any]:
        """Sequential processing"""
        results = []
        for i, item in enumerate(items):
            print(f"Processing {i+1}/{len(items)}")
            result = self.processor(item)
            results.append(result)
        return results
    
    def _execute_parallel(self, items: List[Any]) -> List[Any]:
        """Parallel processing"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self.processor, items))
        return results

# Example: Batch processing
def process_document(doc: dict) -> dict:
    """Process single document"""
    return {
        "id": doc["id"],
        "summary": summarize(doc["text"]),
        "keywords": extract_keywords(doc["text"])
    }

documents = [
    {"id": 1, "text": "Document 1..."},
    {"id": 2, "text": "Document 2..."},
    {"id": 3, "text": "Document 3..."}
]

# Sequential
loop_seq = ForEachLoop(process_document, parallel=False)
results_seq = loop_seq.execute(documents)

# Parallel
loop_par = ForEachLoop(process_document, parallel=True)
results_par = loop_par.execute(documents)
```

### Retry Loop

```python
import time
from typing import Optional, Tuple

class RetryLoop:
    """Retry with exponential backoff"""
    
    def __init__(
        self,
        operation: Callable[[], Any],
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0
    ):
        self.operation = operation
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
    
    def execute(self) -> Tuple[bool, Optional[Any], Optional[str]]:
        """Execute with retries"""
        delay = self.initial_delay
        last_error = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = self.operation()
                return True, result, None
                
            except Exception as e:
                last_error = str(e)
                print(f"Attempt {attempt} failed: {e}")
                
                if attempt < self.max_attempts:
                    print(f"Retrying in {delay}s...")
                    time.sleep(delay)
                    delay = min(delay * self.backoff_factor, self.max_delay)
        
        return False, None, last_error

# Example: API call with retry
def call_unreliable_api():
    """API call that might fail"""
    response = requests.get("https://api.example.com/data", timeout=5)
    response.raise_for_status()
    return response.json()

retry = RetryLoop(
    operation=call_unreliable_api,
    max_attempts=5,
    initial_delay=1.0,
    backoff_factor=2.0
)

success, data, error = retry.execute()
if success:
    print(f"Got data: {data}")
else:
    print(f"Failed after retries: {error}")
```

### Until Loop

```python
class UntilLoop:
    """Execute until condition becomes true"""
    
    def __init__(
        self,
        body: Callable[[Any], Any],
        condition: Callable[[Any], bool],
        max_iterations: int = 100
    ):
        self.body = body
        self.condition = condition
        self.max_iterations = max_iterations
    
    def execute(self, initial_state: Any) -> Any:
        """Execute until condition is met"""
        state = initial_state
        iteration = 0
        
        while iteration < self.max_iterations:
            state = self.body(state)
            iteration += 1
            
            if self.condition(state):
                print(f"Condition met after {iteration} iterations")
                break
        
        return state

# Example: Search until found
def search_step(state: dict) -> dict:
    """Perform one search iteration"""
    state["page"] += 1
    results = search_page(state["query"], state["page"])
    state["results"].extend(results)
    return state

def found_enough(state: dict) -> bool:
    """Check if we have enough results"""
    return len(state["results"]) >= state["target_count"]

until = UntilLoop(
    body=search_step,
    condition=found_enough,
    max_iterations=10
)

initial = {
    "query": "AI research",
    "page": 0,
    "results": [],
    "target_count": 50
}

final = until.execute(initial)
print(f"Found {len(final['results'])} results")
```

## Branching Logic

Creating parallel execution paths that diverge and may converge.

### Binary Branch

```python
class BinaryBranch:
    """Split into two execution paths"""
    
    def __init__(
        self,
        condition: Callable[[Any], bool],
        left_path: Callable[[Any], Any],
        right_path: Callable[[Any], Any],
        merge: Optional[Callable[[Any, Any], Any]] = None
    ):
        self.condition = condition
        self.left_path = left_path
        self.right_path = right_path
        self.merge = merge
    
    def execute(self, data: Any) -> Any:
        """Execute branch"""
        if self.condition(data):
            left_result = self.left_path(data)
            if self.merge:
                # Execute right path too and merge
                right_result = self.right_path(data)
                return self.merge(left_result, right_result)
            return left_result
        else:
            return self.right_path(data)

# Example: Dual processing
def needs_detailed_analysis(doc: dict) -> bool:
    """Check if document needs detailed analysis"""
    return doc.get("length", 0) > 1000

def quick_summary(doc: dict) -> dict:
    """Quick summarization"""
    return {
        "type": "quick",
        "summary": summarize_short(doc["text"])
    }

def detailed_analysis(doc: dict) -> dict:
    """Detailed analysis"""
    return {
        "type": "detailed",
        "summary": summarize_long(doc["text"]),
        "entities": extract_entities(doc["text"]),
        "sentiment": analyze_sentiment(doc["text"])
    }

branch = BinaryBranch(
    condition=needs_detailed_analysis,
    left_path=detailed_analysis,
    right_path=quick_summary
)

result = branch.execute(document)
```

### Multi-Way Branch

```python
from typing import Dict

class MultiBranch:
    """Split into multiple paths"""
    
    def __init__(
        self,
        router: Callable[[Any], str],
        paths: Dict[str, Callable[[Any], Any]],
        default: Optional[Callable[[Any], Any]] = None
    ):
        self.router = router
        self.paths = paths
        self.default = default
    
    def execute(self, data: Any) -> Any:
        """Route to appropriate path"""
        route = self.router(data)
        
        handler = self.paths.get(route)
        if handler:
            return handler(data)
        elif self.default:
            return self.default(data)
        else:
            raise ValueError(f"No handler for route: {route}")

# Example: Content type routing
def detect_content_type(content: dict) -> str:
    """Detect content type"""
    text = content.get("text", "")
    if "```" in text:
        return "code"
    elif text.startswith("#"):
        return "markdown"
    elif "<html>" in text:
        return "html"
    else:
        return "plain"

multi = MultiBranch(
    router=detect_content_type,
    paths={
        "code": process_code,
        "markdown": process_markdown,
        "html": process_html,
        "plain": process_plain_text
    }
)

result = multi.execute(content)
```

## Dynamic Routing

Making routing decisions based on runtime information.

### LLM-Based Router

```python
class LLMRouter:
    """Use LLM to make routing decisions"""
    
    def __init__(self, routes: Dict[str, Callable]):
        self.routes = routes
    
    def execute(self, data: Any, context: str = "") -> Any:
        """Route using LLM decision"""
        # Build routing prompt
        route_names = list(self.routes.keys())
        prompt = f"""
        Given the following task and context, choose the most appropriate route.
        
        Task: {data}
        Context: {context}
        
        Available routes:
        {', '.join(route_names)}
        
        Choose one route name. Respond with only the route name.
        """
        
        # Get LLM decision
        decision = llm.complete(prompt).strip()
        
        # Find matching route
        route_handler = self.routes.get(decision)
        if not route_handler:
            # Fuzzy match
            for route_name in route_names:
                if route_name.lower() in decision.lower():
                    route_handler = self.routes[route_name]
                    break
        
        if route_handler:
            print(f"Routing to: {decision}")
            return route_handler(data)
        else:
            raise ValueError(f"Invalid route decision: {decision}")

# Example: Query routing
router = LLMRouter({
    "web_search": handle_web_search,
    "database_query": handle_database,
    "calculation": handle_calculation,
    "code_execution": handle_code
})

result = router.execute(
    "What's the population of Tokyo?",
    context="User asked about demographic information"
)
```

### Score-Based Routing

```python
class ScoreBasedRouter:
    """Route based on scoring functions"""
    
    def __init__(self):
        self.routes: Dict[str, tuple] = {}  # {name: (handler, scorer)}
    
    def add_route(
        self,
        name: str,
        handler: Callable[[Any], Any],
        scorer: Callable[[Any], float]
    ):
        """Add route with scoring function"""
        self.routes[name] = (handler, scorer)
    
    def execute(self, data: Any) -> Any:
        """Route to highest-scoring handler"""
        scores = {}
        
        for name, (handler, scorer) in self.routes.items():
            try:
                score = scorer(data)
                scores[name] = score
            except Exception as e:
                print(f"Scorer error for {name}: {e}")
                scores[name] = 0.0
        
        # Get highest-scoring route
        best_route = max(scores.items(), key=lambda x: x[1])
        route_name, score = best_route
        
        print(f"Best route: {route_name} (score: {score:.2f})")
        
        handler, _ = self.routes[route_name]
        return handler(data)

# Example: Intent-based routing
router = ScoreBasedRouter()

router.add_route(
    "search",
    handle_search,
    lambda q: 1.0 if any(w in q.lower() for w in ["find", "search", "look"]) else 0.0
)

router.add_route(
    "create",
    handle_create,
    lambda q: 1.0 if any(w in q.lower() for w in ["create", "make", "build"]) else 0.0
)

router.add_route(
    "analyze",
    handle_analyze,
    lambda q: 1.0 if any(w in q.lower() for w in ["analyze", "examine", "study"]) else 0.0
)

query = "Find information about Python"
result = router.execute(query)
```

## Pattern Matching

Advanced routing using pattern matching.

### Pattern Matcher

```python
import re
from typing import Pattern

class PatternMatcher:
    """Match patterns to handlers"""
    
    def __init__(self):
        self.patterns: List[tuple] = []  # [(pattern, handler)]
    
    def register(
        self,
        pattern: str,
        handler: Callable[[dict], Any]
    ):
        """Register pattern and handler"""
        compiled = re.compile(pattern, re.IGNORECASE)
        self.patterns.append((compiled, handler))
    
    def execute(self, text: str) -> Any:
        """Match text to pattern and execute handler"""
        for pattern, handler in self.patterns:
            match = pattern.search(text)
            if match:
                # Extract groups as context
                context = {
                    "full_match": match.group(0),
                    "groups": match.groups(),
                    "groupdict": match.groupdict()
                }
                return handler(context)
        
        return None

# Example: Command patterns
matcher = PatternMatcher()

matcher.register(
    r"search for (?P<query>.+)",
    lambda ctx: search(ctx["groupdict"]["query"])
)

matcher.register(
    r"create (?P<item_type>\w+) named (?P<name>.+)",
    lambda ctx: create_item(
        ctx["groupdict"]["item_type"],
        ctx["groupdict"]["name"]
    )
)

matcher.register(
    r"delete (?P<item>.+)",
    lambda ctx: delete_item(ctx["groupdict"]["item"])
)

# Execute
result = matcher.execute("search for Python tutorials")
# Calls: search("Python tutorials")
```

### Structural Pattern Matching

```python
from dataclasses import dataclass
from typing import Union

@dataclass
class SearchQuery:
    query: str
    filters: dict

@dataclass
class CreateRequest:
    item_type: str
    data: dict

@dataclass
class UpdateRequest:
    item_id: str
    updates: dict

def dispatch_request(request: Union[SearchQuery, CreateRequest, UpdateRequest]) -> Any:
    """Dispatch based on request type"""
    match request:
        case SearchQuery(query=q, filters=f):
            return handle_search(q, f)
        
        case CreateRequest(item_type=t, data=d):
            return handle_create(t, d)
        
        case UpdateRequest(item_id=i, updates=u):
            return handle_update(i, u)
        
        case _:
            raise ValueError(f"Unknown request type: {type(request)}")

# Usage
request = SearchQuery(query="Python", filters={"year": 2024})
result = dispatch_request(request)
```

## Error-Driven Control Flow

Using errors to control execution flow.

### Try-Except Routing

```python
class TryExceptRouter:
    """Route based on exception types"""
    
    def __init__(self, operation: Callable):
        self.operation = operation
        self.exception_handlers: Dict[type, Callable] = {}
        self.default_handler: Callable = None
    
    def on_exception(
        self,
        exception_type: type,
        handler: Callable[[Exception, Any], Any]
    ) -> "TryExceptRouter":
        """Register exception handler"""
        self.exception_handlers[exception_type] = handler
        return self
    
    def on_default(self, handler: Callable) -> "TryExceptRouter":
        """Register default handler"""
        self.default_handler = handler
        return self
    
    def execute(self, data: Any) -> Any:
        """Execute with exception routing"""
        try:
            return self.operation(data)
            
        except Exception as e:
            # Find matching handler
            for exc_type, handler in self.exception_handlers.items():
                if isinstance(e, exc_type):
                    return handler(e, data)
            
            # Use default handler
            if self.default_handler:
                return self.default_handler(e, data)
            
            # Re-raise if no handler
            raise

# Example: API error handling
def call_api(params: dict) -> dict:
    """Call API (might fail)"""
    response = requests.get("https://api.example.com/data", params=params)
    response.raise_for_status()
    return response.json()

router = TryExceptRouter(call_api)

router.on_exception(
    requests.Timeout,
    lambda e, data: {"error": "timeout", "fallback": cached_data(data)}
)

router.on_exception(
    requests.HTTPError,
    lambda e, data: {"error": "http_error", "fallback": default_data()}
)

router.on_default(
    lambda e, data: {"error": "unknown", "message": str(e)}
)

result = router.execute({"query": "test"})
```

### Fallback Chain

```python
class FallbackChain:
    """Try operations in sequence until one succeeds"""
    
    def __init__(self):
        self.operations: List[Callable] = []
    
    def try_operation(self, operation: Callable) -> "FallbackChain":
        """Add operation to chain"""
        self.operations.append(operation)
        return self
    
    def execute(self, data: Any) -> Any:
        """Execute operations until one succeeds"""
        errors = []
        
        for i, operation in enumerate(self.operations):
            try:
                print(f"Trying operation {i+1}/{len(self.operations)}")
                result = operation(data)
                print(f"Operation {i+1} succeeded")
                return result
                
            except Exception as e:
                print(f"Operation {i+1} failed: {e}")
                errors.append((operation.__name__, str(e)))
        
        # All failed
        raise RuntimeError(
            f"All {len(self.operations)} operations failed. "
            f"Errors: {errors}"
        )

# Example: Data source fallback
chain = FallbackChain()
chain.try_operation(fetch_from_primary_api)
chain.try_operation(fetch_from_backup_api)
chain.try_operation(fetch_from_cache)
chain.try_operation(use_default_data)

data = chain.execute({"query": "test"})
```

## Early Exit Patterns

Optimizing execution by exiting early when possible.

### Short-Circuit Evaluation

```python
class ShortCircuit:
    """Evaluate until satisfied"""
    
    def __init__(self, satisfied: Callable[[Any], bool]):
        self.operations: List[Callable] = []
        self.satisfied = satisfied
    
    def add_operation(self, operation: Callable) -> "ShortCircuit":
        """Add operation"""
        self.operations.append(operation)
        return self
    
    def execute(self, initial_data: Any) -> Any:
        """Execute operations, stopping when satisfied"""
        data = initial_data
        
        for i, operation in enumerate(self.operations):
            print(f"Executing operation {i+1}")
            data = operation(data)
            
            if self.satisfied(data):
                print(f"Satisfied after operation {i+1}, stopping early")
                return data
        
        return data

# Example: Search with early exit
def has_enough_results(state: dict) -> bool:
    """Check if we have enough results"""
    return len(state.get("results", [])) >= state.get("target", 10)

sc = ShortCircuit(has_enough_results)
sc.add_operation(search_local_cache)
sc.add_operation(search_primary_index)
sc.add_operation(search_backup_index)
sc.add_operation(search_web)

result = sc.execute({"query": "Python", "target": 10, "results": []})
```

### Guard Clauses

```python
class GuardedExecution:
    """Execute with guard conditions"""
    
    def __init__(self):
        self.guards: List[tuple] = []  # [(guard, message)]
        self.operation: Callable = None
    
    def guard(
        self,
        condition: Callable[[Any], bool],
        message: str = "Guard failed"
    ) -> "GuardedExecution":
        """Add guard condition"""
        self.guards.append((condition, message))
        return self
    
    def execute_operation(self, operation: Callable) -> "GuardedExecution":
        """Set operation to execute"""
        self.operation = operation
        return self
    
    def execute(self, data: Any) -> Any:
        """Execute with guards"""
        # Check all guards
        for guard, message in self.guards:
            if not guard(data):
                raise ValueError(f"Guard failed: {message}")
        
        # All guards passed, execute operation
        if self.operation:
            return self.operation(data)
        
        return data

# Example: File processing with guards
execution = GuardedExecution()

execution.guard(
    lambda data: os.path.exists(data["path"]),
    "File does not exist"
).guard(
    lambda data: os.path.getsize(data["path"]) > 0,
    "File is empty"
).guard(
    lambda data: data["path"].endswith(('.json', '.yaml')),
    "Invalid file type"
).execute_operation(
    lambda data: process_config_file(data["path"])
)

result = execution.execute({"path": "config.json"})
```

## Recursive Control Flow

Self-referential control structures.

### Recursive Execution

```python
class RecursiveExecutor:
    """Execute recursively with depth limit"""
    
    def __init__(
        self,
        base_case: Callable[[Any], bool],
        recursive_step: Callable[[Any], Any],
        max_depth: int = 100
    ):
        self.base_case = base_case
        self.recursive_step = recursive_step
        self.max_depth = max_depth
    
    def execute(self, data: Any, depth: int = 0) -> Any:
        """Execute recursively"""
        if depth >= self.max_depth:
            raise RecursionError(f"Max depth ({self.max_depth}) exceeded")
        
        # Check base case
        if self.base_case(data):
            return data
        
        # Recursive step
        new_data = self.recursive_step(data)
        return self.execute(new_data, depth + 1)

# Example: Recursive refinement
def is_acceptable(result: dict) -> bool:
    """Check if result is acceptable"""
    return result.get("quality", 0) >= 0.9

def refine_step(result: dict) -> dict:
    """One refinement step"""
    refined = improve_quality(result["content"])
    new_quality = evaluate_quality(refined)
    
    return {
        "content": refined,
        "quality": new_quality,
        "iterations": result.get("iterations", 0) + 1
    }

recursive = RecursiveExecutor(
    base_case=is_acceptable,
    recursive_step=refine_step,
    max_depth=10
)

initial = {"content": "Draft", "quality": 0.5}
final = recursive.execute(initial)
```

### Tree Traversal

```python
from typing import List

class TreeNode:
    """Tree node"""
    def __init__(self, value: Any, children: List["TreeNode"] = None):
        self.value = value
        self.children = children or []

class TreeTraversal:
    """Traverse tree with control flow"""
    
    def __init__(
        self,
        process_node: Callable[[Any], Any],
        should_visit_children: Callable[[Any], bool] = None
    ):
        self.process_node = process_node
        self.should_visit_children = should_visit_children or (lambda x: True)
    
    def traverse_dfs(self, root: TreeNode) -> List[Any]:
        """Depth-first traversal"""
        results = []
        
        def visit(node: TreeNode):
            # Process current node
            result = self.process_node(node.value)
            results.append(result)
            
            # Visit children if allowed
            if self.should_visit_children(node.value):
                for child in node.children:
                    visit(child)
        
        visit(root)
        return results
    
    def traverse_bfs(self, root: TreeNode) -> List[Any]:
        """Breadth-first traversal"""
        results = []
        queue = [root]
        
        while queue:
            node = queue.pop(0)
            
            # Process node
            result = self.process_node(node.value)
            results.append(result)
            
            # Add children if allowed
            if self.should_visit_children(node.value):
                queue.extend(node.children)
        
        return results

# Example: Directory traversal
def process_file(path: str) -> dict:
    """Process file"""
    return {"path": path, "size": os.path.getsize(path)}

def should_descend(path: str) -> bool:
    """Check if should descend into directory"""
    return not path.startswith(".")  # Skip hidden dirs

traversal = TreeTraversal(
    process_node=process_file,
    should_visit_children=should_descend
)

# Build tree and traverse
file_tree = build_file_tree("/path/to/directory")
results = traversal.traverse_dfs(file_tree)
```

## Computed Routing

Dynamic routing based on computations.

### Routing Table

```python
class RoutingTable:
    """Dynamic routing based on computed keys"""
    
    def __init__(self):
        self.routes: Dict[Any, Callable] = {}
        self.key_computer: Callable[[Any], Any] = None
    
    def set_key_computer(self, computer: Callable[[Any], Any]):
        """Set function to compute routing key"""
        self.key_computer = computer
    
    def register_route(self, key: Any, handler: Callable):
        """Register route for key"""
        self.routes[key] = handler
    
    def execute(self, data: Any) -> Any:
        """Compute key and route"""
        if not self.key_computer:
            raise ValueError("Key computer not set")
        
        # Compute routing key
        key = self.key_computer(data)
        print(f"Computed routing key: {key}")
        
        # Find handler
        handler = self.routes.get(key)
        if not handler:
            raise ValueError(f"No route for key: {key}")
        
        return handler(data)

# Example: Load-based routing
def compute_load_key(task: dict) -> str:
    """Compute load level"""
    size = task.get("size", 0)
    if size < 100:
        return "light"
    elif size < 1000:
        return "medium"
    else:
        return "heavy"

router = RoutingTable()
router.set_key_computer(compute_load_key)

router.register_route("light", handle_light_task)
router.register_route("medium", handle_medium_task)
router.register_route("heavy", handle_heavy_task)

task = {"name": "Process data", "size": 500}
result = router.execute(task)
```

## Complex Control Structures

Combining multiple control flow patterns.

### State Machine Router

```python
from enum import Enum

class State(Enum):
    """Workflow states"""
    INIT = "init"
    PROCESSING = "processing"
    WAITING = "waiting"
    COMPLETE = "complete"
    FAILED = "failed"

class StateMachineRouter:
    """Route based on state machine"""
    
    def __init__(self, initial_state: State):
        self.current_state = initial_state
        self.transitions: Dict[tuple, Callable] = {}  # {(from, to): handler}
    
    def register_transition(
        self,
        from_state: State,
        to_state: State,
        handler: Callable[[Any], Any]
    ):
        """Register state transition handler"""
        self.transitions[(from_state, to_state)] = handler
    
    def transition(self, to_state: State, data: Any) -> Any:
        """Transition to new state"""
        transition_key = (self.current_state, to_state)
        handler = self.transitions.get(transition_key)
        
        if not handler:
            raise ValueError(
                f"No handler for transition: {self.current_state.value} -> {to_state.value}"
            )
        
        print(f"Transitioning: {self.current_state.value} -> {to_state.value}")
        result = handler(data)
        self.current_state = to_state
        
        return result

# Example: Workflow state machine
sm = StateMachineRouter(State.INIT)

sm.register_transition(State.INIT, State.PROCESSING, start_processing)
sm.register_transition(State.PROCESSING, State.WAITING, wait_for_approval)
sm.register_transition(State.WAITING, State.PROCESSING, resume_processing)
sm.register_transition(State.PROCESSING, State.COMPLETE, finalize)
sm.register_transition(State.PROCESSING, State.FAILED, handle_failure)

# Execute workflow
data = {"task_id": "task_001"}
data = sm.transition(State.PROCESSING, data)
data = sm.transition(State.WAITING, data)
# ... wait for approval ...
data = sm.transition(State.PROCESSING, data)
data = sm.transition(State.COMPLETE, data)
```

## Control Flow Optimization

Optimizing control flow for performance.

### Memoized Routing

```python
from functools import lru_cache

class MemoizedRouter:
    """Cache routing decisions"""
    
    def __init__(self, router: Callable[[str], str]):
        self.router = router
        self.handlers: Dict[str, Callable] = {}
        
        # Memoize router
        self._cached_router = lru_cache(maxsize=1000)(router)
    
    def register(self, route: str, handler: Callable):
        """Register route handler"""
        self.handlers[route] = handler
    
    def execute(self, data: Any, routing_key: str) -> Any:
        """Execute with cached routing"""
        # Get cached route decision
        route = self._cached_router(routing_key)
        
        handler = self.handlers.get(route)
        if not handler:
            raise ValueError(f"No handler for route: {route}")
        
        return handler(data)

# Example: Query routing with caching
def compute_route(query: str) -> str:
    """Expensive route computation"""
    # Analyze query (expensive)
    return analyze_intent(query)

router = MemoizedRouter(compute_route)
router.register("search", handle_search)
router.register("create", handle_create)

# First call: computes route
result1 = router.execute(data, "find Python docs")

# Second call with same query: uses cache
result2 = router.execute(data, "find Python docs")  # Fast!
```

## Real-World Examples

### Example 1: Content Moderation Pipeline

```python
class ModerationPipeline:
    """Content moderation with complex control flow"""
    
    def moderate(self, content: dict) -> dict:
        """Moderate content with multi-stage control flow"""
        
        # Stage 1: Quick checks (early exit pattern)
        if self._is_obvious_spam(content):
            return {"status": "rejected", "reason": "spam"}
        
        if self._is_obviously_safe(content):
            return {"status": "approved", "reason": "safe"}
        
        # Stage 2: Detailed analysis (conditional routing)
        toxicity_score = self._analyze_toxicity(content)
        
        if toxicity_score > 0.8:
            return {"status": "rejected", "reason": "toxic"}
        elif toxicity_score > 0.5:
            # High risk: human review
            return {"status": "review", "priority": "high"}
        
        # Stage 3: Contextual analysis (LLM routing)
        context_decision = self._llm_context_analysis(content)
        
        if context_decision == "flag":
            return {"status": "review", "priority": "normal"}
        
        # Stage 4: Final checks
        if self._has_prohibited_content(content):
            return {"status": "rejected", "reason": "prohibited"}
        
        # Approved
        return {"status": "approved", "reason": "passed_checks"}
    
    def _is_obvious_spam(self, content: dict) -> bool:
        """Quick spam check"""
        text = content.get("text", "")
        return len(text) < 10 or text.count("http") > 5
    
    def _is_obviously_safe(self, content: dict) -> bool:
        """Quick safety check"""
        return len(content.get("text", "")) < 50
    
    def _analyze_toxicity(self, content: dict) -> float:
        """Analyze toxicity"""
        return compute_toxicity_score(content["text"])
    
    def _llm_context_analysis(self, content: dict) -> str:
        """LLM-based context analysis"""
        return llm_analyze_context(content["text"])
    
    def _has_prohibited_content(self, content: dict) -> bool:
        """Check for prohibited content"""
        return check_prohibited_patterns(content["text"])

# Usage
pipeline = ModerationPipeline()
result = pipeline.moderate({"text": "User content here..."})
```

### Example 2: Adaptive Query Processing

```python
class AdaptiveQueryProcessor:
    """Process queries with adaptive control flow"""
    
    def process(self, query: str) -> dict:
        """Process query adaptively"""
        
        # Stage 1: Try cache (early exit)
        cached = self._check_cache(query)
        if cached:
            return {"source": "cache", "result": cached}
        
        # Stage 2: Classify query complexity
        complexity = self._assess_complexity(query)
        
        if complexity == "simple":
            # Direct execution
            result = self._simple_execution(query)
        elif complexity == "medium":
            # Iterative refinement
            result = self._iterative_execution(query)
        else:  # complex
            # Multi-stage execution
            result = self._multi_stage_execution(query)
        
        # Cache result
        self._cache_result(query, result)
        
        return {"source": "computed", "result": result}
    
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity"""
        word_count = len(query.split())
        if word_count < 5:
            return "simple"
        elif word_count < 15:
            return "medium"
        else:
            return "complex"
    
    def _simple_execution(self, query: str) -> Any:
        """Simple query execution"""
        return execute_simple_query(query)
    
    def _iterative_execution(self, query: str) -> Any:
        """Iterative query execution"""
        result = execute_query(query)
        
        # Refine until quality threshold
        while evaluate_quality(result) < 0.8:
            result = refine_result(result)
        
        return result
    
    def _multi_stage_execution(self, query: str) -> Any:
        """Multi-stage query execution"""
        # Decompose query
        subqueries = decompose_query(query)
        
        # Execute subqueries
        subresults = [execute_query(sq) for sq in subqueries]
        
        # Synthesize results
        return synthesize_results(subresults)

# Usage
processor = AdaptiveQueryProcessor()
result = processor.process("Complex multi-part query about...")
```

## Summary

Control flow orchestrates agent execution:

**Core Patterns**:
- **Conditionals**: Route based on conditions
- **Loops**: Iterate until conditions are met
- **Branches**: Create parallel execution paths
- **Dynamic Routing**: Make routing decisions at runtime
- **Error Handling**: Control flow based on exceptions
- **Early Exit**: Optimize by exiting early when possible

**Advanced Techniques**:
- **LLM-Based Routing**: Use LLM to make routing decisions
- **Pattern Matching**: Route based on pattern matches
- **State Machines**: Enforce valid state transitions
- **Recursive Flow**: Self-referential control structures
- **Computed Routing**: Dynamic routing based on computations

**Key Principles**:
1. Choose the simplest control structure that works
2. Use early exits to optimize performance
3. Handle errors explicitly with fallbacks
4. Make routing decisions observable
5. Cache routing decisions when possible
6. Design for clarity and maintainability

**Best Practices**:
- Keep control flow explicit and traceable
- Use guard clauses for preconditions
- Implement fallback chains for reliability
- Add timeouts and iteration limits
- Log routing decisions for debugging
- Test edge cases and error paths

Effective control flow enables agents to make intelligent routing decisions, handle errors gracefully, and adapt execution based on runtime conditions.

## Next Steps

Continue exploring orchestration patterns:

- **[Workflow Patterns](workflow-patterns.md)**: Fundamental workflow structures
- **[State Management](state-management.md)**: Managing state across control flow
- **[Parallelization](parallelization.md)**: Concurrent execution patterns
- **[Error Handling](../fundamentals/error-handling.md)**: Comprehensive error handling strategies

Related topics:
- **[ReAct Architecture](../agent-architectures/react.md)**: Reasoning-driven control flow
- **[Planning](../planning-and-reasoning/planning-strategies.md)**: High-level planning and execution
