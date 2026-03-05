# Basic Error Handling

## Table of Contents

- [Introduction](#introduction)
- [Why Error Handling Matters](#why-error-handling-matters)
- [Types of Errors in Agent Systems](#types-of-errors-in-agent-systems)
- [Error Detection](#error-detection)
- [Error Classification](#error-classification)
- [Basic Error Handling Patterns](#basic-error-handling-patterns)
- [Retry Strategies](#retry-strategies)
- [Fallback Mechanisms](#fallback-mechanisms)
- [Error Recovery](#error-recovery)
- [Graceful Degradation](#graceful-degradation)
- [Error Reporting](#error-reporting)
- [Logging and Monitoring](#logging-and-monitoring)
- [User-Facing Error Messages](#user-facing-error-messages)
- [Circuit Breakers](#circuit-breakers)
- [Error Budgets](#error-budgets)
- [Best Practices](#best-practices)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Agents fail. LLMs hallucinate. Tools error out. Networks disconnect. APIs rate limit. Parsing breaks. This is reality. The question isn't whether your agent will encounter errors--it's how it will handle them.

Poor error handling leads to:

- Silent failures
- Lost user trust
- Cascading errors
- Difficult debugging
- System instability

Good error handling means:

- Graceful degradation
- Clear error messages
- Automatic recovery
- Reliable operation
- Easy debugging

> "An agent without error handling is a time bomb."

This guide covers error handling from fundamentals to production-ready patterns: detecting errors, classifying them, implementing retries, building fallbacks, recovering gracefully, and ensuring your agent keeps working even when things go wrong.

## Why Error Handling Matters

### Without Error Handling

```python
def fragile_agent(query: str) -> str:
    """Agent with no error handling."""

    # This will crash on any error
    result = call_llm(query)
    parsed = json.loads(result)
    tool_output = execute_tool(parsed["tool"], parsed["args"])
    final = call_llm(f"Process: {tool_output}")

    return final


# What happens when:
# - LLM returns invalid JSON?
# - Tool doesn't exist?
# - Tool execution fails?
# - Network times out?
# Answer: Everything crashes
```

### With Error Handling

```python
def robust_agent(query: str) -> str:
    """Agent with comprehensive error handling."""

    try:
        result = call_llm(query)
    except TimeoutError:
        return "I'm experiencing delays. Please try again."
    except Exception as e:
        log_error("LLM call failed", e)
        return "I encountered an error. Please try again."

    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        # Try to extract JSON
        parsed = extract_json_with_fallback(result)
        if not parsed:
            return "I had trouble understanding the response."

    try:
        tool_output = execute_tool(parsed["tool"], parsed["args"])
    except KeyError:
        return "I couldn't determine which tool to use."
    except ToolExecutionError as e:
        log_error(f"Tool {parsed['tool']} failed", e)
        return f"The {parsed['tool']} operation failed. Please try a different approach."

    try:
        final = call_llm(f"Process: {tool_output}")
        return final
    except Exception as e:
        # At least return the tool output
        return f"I got results but had trouble formatting them: {tool_output}"


# Now the agent handles:
# - Timeouts gracefully
# - Parsing errors with fallbacks
# - Tool failures with messages
# - Partial results when possible
```

### Real-World Impact

**Scenario**: User asks agent to search and summarize papers.

**Without error handling**:

1. Search API times out
2. Agent crashes
3. User sees stack trace
4. User loses trust

**With error handling**:

1. Search API times out
2. Agent detects timeout
3. Retries with exponential backoff
4. If still fails, uses cached results
5. Tells user: "Search is slow, using recent results..."
6. User gets answer, knows what happened

## Types of Errors in Agent Systems

Understanding error types helps you handle them appropriately.

### 1. LLM Errors

**Timeout errors**:

```python
try:
    response = llm.generate(prompt, timeout=30)
except TimeoutError:
    # LLM took too long
    pass
```

**Rate limit errors**:

```python
try:
    response = llm.generate(prompt)
except RateLimitError:
    # Too many requests
    time.sleep(60)  # Wait and retry
```

**Context length errors**:

```python
try:
    response = llm.generate(prompt)
except ContextLengthError:
    # Prompt too long
    prompt = compress_context(prompt)
    response = llm.generate(prompt)
```

**Invalid responses**:

```python
response = llm.generate(prompt)
if not is_valid_response(response):
    # LLM returned garbage
    pass
```

### 2. Parsing Errors

**JSON parse errors**:

```python
try:
    data = json.loads(response)
except json.JSONDecodeError:
    # Invalid JSON
    pass
```

**Missing fields**:

```python
data = json.loads(response)
if "required_field" not in data:
    # Missing expected field
    pass
```

**Type errors**:

```python
if not isinstance(data["age"], int):
    # Wrong type
    pass
```

### 3. Tool Execution Errors

**Tool not found**:

```python
if tool_name not in tools:
    raise ToolNotFoundError(f"Unknown tool: {tool_name}")
```

**Invalid arguments**:

```python
if "required_arg" not in arguments:
    raise InvalidArgumentError("Missing required argument")
```

**Execution failure**:

```python
try:
    result = tool.execute(**arguments)
except Exception as e:
    raise ToolExecutionError(f"Tool failed: {e}")
```

### 4. System Errors

**Network errors**:

```python
try:
    response = requests.get(url)
except requests.ConnectionError:
    # Network unreachable
    pass
```

**File system errors**:

```python
try:
    with open(file_path) as f:
        data = f.read()
except FileNotFoundError:
    # File doesn't exist
    pass
```

**Resource exhaustion**:

```python
try:
    process_large_data(data)
except MemoryError:
    # Out of memory
    pass
```

### 5. Validation Errors

**Schema validation**:

```python
try:
    validate(data, schema)
except ValidationError as e:
    # Data doesn't match schema
    pass
```

**Business logic validation**:

```python
if temperature < -273.15:  # Below absolute zero
    raise ValidationError("Invalid temperature")
```

## Error Detection

Proactively detect errors before they cause problems.

### Pre-Execution Validation

```python
class PreExecutionValidator:
    """Validate before executing operations."""

    def validate_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        tools: dict
    ) -> tuple[bool, Optional[str]]:
        """Validate tool call before execution."""

        # Check tool exists
        if tool_name not in tools:
            return False, f"Tool '{tool_name}' not found"

        tool = tools[tool_name]

        # Check required parameters
        required = tool.get("required_params", [])
        for param in required:
            if param not in arguments:
                return False, f"Missing required parameter: {param}"

        # Check parameter types
        param_types = tool.get("param_types", {})
        for param, value in arguments.items():
            expected_type = param_types.get(param)
            if expected_type and not isinstance(value, expected_type):
                return False, f"Parameter '{param}' should be {expected_type.__name__}"

        return True, None

    def validate_context_size(
        self,
        prompt: str,
        max_tokens: int
    ) -> tuple[bool, Optional[str]]:
        """Validate context fits in limits."""

        estimated_tokens = len(prompt) // 4

        if estimated_tokens > max_tokens:
            return False, f"Prompt too long: {estimated_tokens} > {max_tokens} tokens"

        return True, None

    def validate_response_schema(
        self,
        response: dict,
        schema: dict
    ) -> tuple[bool, Optional[str]]:
        """Validate response matches schema."""

        try:
            jsonschema.validate(response, schema)
            return True, None
        except jsonschema.ValidationError as e:
            return False, str(e)
```

### Runtime Detection

```python
class RuntimeErrorDetector:
    """Detect errors during execution."""

    def detect_timeout(
        self,
        start_time: float,
        timeout: float
    ) -> bool:
        """Detect if operation has timed out."""
        return time.time() - start_time > timeout

    def detect_infinite_loop(
        self,
        iteration: int,
        max_iterations: int
    ) -> bool:
        """Detect potential infinite loop."""
        return iteration > max_iterations

    def detect_resource_exhaustion(self) -> tuple[bool, str]:
        """Detect resource exhaustion."""
        import psutil

        # Check memory
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            return True, "Memory usage > 90%"

        # Check CPU
        cpu = psutil.cpu_percent(interval=1)
        if cpu > 95:
            return True, "CPU usage > 95%"

        return False, ""

    def detect_invalid_state(
        self,
        state: dict,
        expected_keys: list
    ) -> tuple[bool, Optional[str]]:
        """Detect invalid agent state."""

        for key in expected_keys:
            if key not in state:
                return True, f"Missing state key: {key}"

        return False, None
```

### Post-Execution Validation

```python
class PostExecutionValidator:
    """Validate after execution."""

    def validate_tool_output(
        self,
        output: Any,
        expected_type: type
    ) -> tuple[bool, Optional[str]]:
        """Validate tool output type."""

        if not isinstance(output, expected_type):
            return False, f"Expected {expected_type}, got {type(output)}"

        return True, None

    def validate_response_completeness(
        self,
        response: str,
        required_sections: list
    ) -> tuple[bool, Optional[str]]:
        """Validate response contains required sections."""

        response_lower = response.lower()

        for section in required_sections:
            if section.lower() not in response_lower:
                return False, f"Missing required section: {section}"

        return True, None

    def validate_no_hallucination(
        self,
        response: str,
        source_data: str
    ) -> tuple[bool, Optional[str]]:
        """Basic hallucination detection."""

        # Extract facts from response
        # Check if they exist in source
        # This is simplified - real implementation would be more sophisticated

        # Look for numerical claims
        import re
        numbers_in_response = re.findall(r'\d+', response)
        numbers_in_source = re.findall(r'\d+', source_data)

        for num in numbers_in_response:
            if num not in numbers_in_source:
                return False, f"Number {num} not found in source data"

        return True, None
```

## Error Classification

Classify errors to handle them appropriately.

### Error Taxonomy

```python
from enum import Enum

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = 1          # Minor issue, can continue
    MEDIUM = 2       # Degraded functionality
    HIGH = 3         # Major failure, need recovery
    CRITICAL = 4     # System failure, stop immediately


class ErrorType(Enum):
    """Types of errors."""
    TRANSIENT = "transient"        # Temporary, retry may work
    PERMANENT = "permanent"         # Won't succeed on retry
    VALIDATION = "validation"       # Invalid input/output
    RESOURCE = "resource"           # Resource unavailable
    LOGIC = "logic"                 # Programming error


class AgentError(Exception):
    """Base exception for agent errors."""

    def __init__(
        self,
        message: str,
        error_type: ErrorType,
        severity: ErrorSeverity,
        recoverable: bool = True,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.error_type = error_type
        self.severity = severity
        self.recoverable = recoverable
        self.original_error = original_error
        self.timestamp = time.time()


# Specific error types
class LLMTimeoutError(AgentError):
    """LLM call timed out."""
    def __init__(self, message: str):
        super().__init__(
            message,
            ErrorType.TRANSIENT,
            ErrorSeverity.MEDIUM,
            recoverable=True
        )


class ToolExecutionError(AgentError):
    """Tool execution failed."""
    def __init__(self, message: str, original_error: Exception):
        super().__init__(
            message,
            ErrorType.PERMANENT,
            ErrorSeverity.HIGH,
            recoverable=False,
            original_error=original_error
        )


class ParseError(AgentError):
    """Failed to parse response."""
    def __init__(self, message: str):
        super().__init__(
            message,
            ErrorType.VALIDATION,
            ErrorSeverity.MEDIUM,
            recoverable=True
        )
```

### Error Classifier

```python
class ErrorClassifier:
    """Classify errors for appropriate handling."""

    def classify(self, error: Exception) -> tuple[ErrorType, ErrorSeverity, bool]:
        """
        Classify an error.
        Returns: (error_type, severity, is_recoverable)
        """

        # Network errors - transient
        if isinstance(error, (
            requests.ConnectionError,
            requests.Timeout,
            TimeoutError
        )):
            return ErrorType.TRANSIENT, ErrorSeverity.MEDIUM, True

        # Rate limits - transient
        if "rate limit" in str(error).lower():
            return ErrorType.TRANSIENT, ErrorSeverity.LOW, True

        # Parse errors - validation, often recoverable
        if isinstance(error, (json.JSONDecodeError, ValueError)):
            return ErrorType.VALIDATION, ErrorSeverity.MEDIUM, True

        # File not found - permanent for that file
        if isinstance(error, FileNotFoundError):
            return ErrorType.PERMANENT, ErrorSeverity.HIGH, False

        # Resource errors
        if isinstance(error, (MemoryError, OSError)):
            return ErrorType.RESOURCE, ErrorSeverity.CRITICAL, False

        # Logic errors - programming bugs
        if isinstance(error, (AttributeError, KeyError, TypeError)):
            return ErrorType.LOGIC, ErrorSeverity.HIGH, False

        # Unknown - assume permanent, high severity
        return ErrorType.PERMANENT, ErrorSeverity.HIGH, False

    def should_retry(self, error: Exception, attempt: int, max_attempts: int) -> bool:
        """Determine if should retry based on error type."""

        if attempt >= max_attempts:
            return False

        error_type, severity, recoverable = self.classify(error)

        # Only retry transient errors
        if error_type == ErrorType.TRANSIENT:
            return True

        # Retry validation errors once
        if error_type == ErrorType.VALIDATION and attempt < 2:
            return True

        # Don't retry permanent or logic errors
        return False
```

## Basic Error Handling Patterns

Fundamental patterns for handling errors.

### Try-Catch-Finally

```python
def basic_error_handling(operation: callable) -> Any:
    """Basic try-catch pattern."""

    try:
        result = operation()
        return result

    except SpecificError as e:
        # Handle specific error
        log_error("Specific error occurred", e)
        return default_value

    except Exception as e:
        # Handle any other error
        log_error("Unexpected error", e)
        raise

    finally:
        # Always runs, even if error
        cleanup()
```

### Guard Clauses

```python
def guard_clause_pattern(data: dict) -> str:
    """Use guard clauses to fail fast."""

    # Validate inputs early
    if not data:
        raise ValueError("Data cannot be empty")

    if "required_field" not in data:
        raise ValueError("Missing required field")

    if not isinstance(data["value"], int):
        raise TypeError("Value must be integer")

    # Main logic only runs if all guards pass
    return process(data)
```

### Error Wrapping

```python
def error_wrapping_pattern():
    """Wrap low-level errors in domain-specific errors."""

    try:
        result = external_api_call()
        return result

    except requests.ConnectionError as e:
        # Wrap in domain error
        raise AgentError(
            "Failed to connect to external service",
            ErrorType.TRANSIENT,
            ErrorSeverity.HIGH,
            original_error=e
        )
```

### Default Values

```python
def with_defaults(operation: callable, default: Any) -> Any:
    """Return default value on error."""

    try:
        return operation()
    except Exception as e:
        log_error("Operation failed, using default", e)
        return default


# Usage
user_settings = with_defaults(
    lambda: load_settings(),
    default={"theme": "light"}
)
```

## Retry Strategies

Retry operations that might succeed on subsequent attempts.

### Simple Retry

```python
def simple_retry(
    operation: callable,
    max_attempts: int = 3
) -> Any:
    """Retry operation up to max_attempts."""

    last_error = None

    for attempt in range(max_attempts):
        try:
            return operation()

        except Exception as e:
            last_error = e
            print(f"Attempt {attempt + 1} failed: {e}")

            if attempt < max_attempts - 1:
                time.sleep(1)  # Brief pause

    raise last_error
```

### Exponential Backoff

```python
def exponential_backoff_retry(
    operation: callable,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0
) -> Any:
    """Retry with exponential backoff."""

    for attempt in range(max_attempts):
        try:
            return operation()

        except Exception as e:
            if attempt == max_attempts - 1:
                raise

            # Calculate delay: base_delay * 2^attempt
            delay = min(base_delay * (2 ** attempt), max_delay)

            print(f"Attempt {attempt + 1} failed, retrying in {delay}s")
            time.sleep(delay)
```

### Smart Retry

```python
class SmartRetry:
    """Retry with error classification."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.classifier = ErrorClassifier()

    def execute(self, operation: callable) -> Any:
        """Execute with smart retry."""

        last_error = None

        for attempt in range(self.max_attempts):
            try:
                return operation()

            except Exception as e:
                last_error = e

                # Should we retry?
                if not self.classifier.should_retry(e, attempt, self.max_attempts):
                    raise

                # Calculate delay
                error_type, severity, _ = self.classifier.classify(e)

                if error_type == ErrorType.TRANSIENT:
                    # Exponential backoff for transient errors
                    delay = self.base_delay * (2 ** attempt)
                else:
                    # Fixed delay for other errors
                    delay = self.base_delay

                print(f"Retrying after {delay}s due to {error_type.value} error")
                time.sleep(delay)

        raise last_error
```

### Retry with Jitter

```python
import random

def retry_with_jitter(
    operation: callable,
    max_attempts: int = 3,
    base_delay: float = 1.0
) -> Any:
    """Retry with jitter to avoid thundering herd."""

    for attempt in range(max_attempts):
        try:
            return operation()

        except Exception as e:
            if attempt == max_attempts - 1:
                raise

            # Add random jitter to delay
            delay = base_delay * (2 ** attempt)
            jitter = random.uniform(0, delay * 0.3)  # Up to 30% jitter
            total_delay = delay + jitter

            print(f"Retrying in {total_delay:.2f}s")
            time.sleep(total_delay)
```

## Fallback Mechanisms

Provide alternatives when primary approach fails.

### Primary-Secondary Pattern

```python
def with_fallback(
    primary: callable,
    fallback: callable
) -> Any:
    """Try primary, use fallback on failure."""

    try:
        return primary()
    except Exception as e:
        log_error("Primary failed, using fallback", e)
        return fallback()
```

### Waterfall Fallbacks

```python
def waterfall_fallback(operations: list[callable]) -> Any:
    """Try operations in order until one succeeds."""

    errors = []

    for i, operation in enumerate(operations):
        try:
            return operation()
        except Exception as e:
            errors.append((i, e))
            print(f"Operation {i} failed: {e}")

    # All failed
    raise Exception(f"All operations failed: {errors}")


# Usage
result = waterfall_fallback([
    lambda: call_primary_api(),
    lambda: call_backup_api(),
    lambda: use_cached_data(),
    lambda: return_default_value()
])
```

### Feature-Based Fallback

```python
class FeatureFallback:
    """Fallback with degraded features."""

    def search_and_summarize(self, query: str) -> dict:
        """Search and summarize with fallback."""

        results = {"query": query}

        # Try full search
        try:
            results["search_results"] = self.search(query)
        except Exception as e:
            log_error("Search failed", e)
            # Fallback to cached results
            results["search_results"] = self.get_cached_results(query)
            results["warning"] = "Using cached results"

        # Try summarization
        try:
            results["summary"] = self.summarize(results["search_results"])
        except Exception as e:
            log_error("Summarization failed", e)
            # Fallback to simple extraction
            results["summary"] = self.extract_snippets(results["search_results"])
            results["warning"] = "Using basic extraction"

        return results
```

### Cached Fallback

```python
class CachedFallback:
    """Use cache as fallback."""

    def __init__(self):
        self.cache = {}

    def get_with_cache_fallback(
        self,
        key: str,
        fetch_operation: callable
    ) -> Any:
        """Try fetching, fall back to cache."""

        try:
            # Try to fetch fresh data
            data = fetch_operation()

            # Update cache
            self.cache[key] = {
                "data": data,
                "timestamp": time.time()
            }

            return data

        except Exception as e:
            log_error("Fetch failed, using cache", e)

            # Check cache
            if key in self.cache:
                cached = self.cache[key]
                age = time.time() - cached["timestamp"]

                print(f"Using cached data (age: {age:.0f}s)")
                return cached["data"]

            # No cache available
            raise Exception("Fetch failed and no cache available")
```

## Error Recovery

Recover from errors and continue operation.

### State Restoration

```python
class StateBasedRecovery:
    """Recover by restoring previous state."""

    def __init__(self):
        self.state_history = []
        self.current_state = {}

    def save_state(self):
        """Save current state."""
        self.state_history.append(self.current_state.copy())

    def restore_state(self) -> bool:
        """Restore previous state."""
        if not self.state_history:
            return False

        self.current_state = self.state_history.pop()
        return True

    def execute_with_recovery(self, operation: callable) -> Any:
        """Execute with state recovery on error."""

        # Save state before operation
        self.save_state()

        try:
            result = operation()
            return result

        except Exception as e:
            log_error("Operation failed, restoring state", e)

            # Restore previous state
            self.restore_state()

            raise
```

### Compensating Actions

```python
class CompensatingActions:
    """Undo operations on failure."""

    def __init__(self):
        self.actions = []

    def add_compensating_action(self, action: callable):
        """Add action to undo on failure."""
        self.actions.append(action)

    def execute_with_compensation(
        self,
        operations: list[tuple[callable, callable]]
    ) -> Any:
        """
        Execute operations with compensation.
        operations: list of (action, compensating_action) tuples
        """

        completed = []

        try:
            for action, compensating in operations:
                result = action()
                completed.append((result, compensating))

            return [r for r, _ in completed]

        except Exception as e:
            log_error("Operation failed, compensating", e)

            # Execute compensating actions in reverse
            for result, compensating in reversed(completed):
                try:
                    compensating(result)
                except Exception as comp_error:
                    log_error("Compensation failed", comp_error)

            raise


# Usage
compensator = CompensatingActions()

result = compensator.execute_with_compensation([
    (
        lambda: create_user(name="John"),
        lambda user: delete_user(user.id)
    ),
    (
        lambda: send_email(to="john@example.com"),
        lambda: None  # Can't unsend email
    )
])
```

### Partial Success Handling

```python
def handle_partial_success(
    operations: list[callable]
) -> tuple[list, list]:
    """Execute operations, return successes and failures."""

    successes = []
    failures = []

    for i, operation in enumerate(operations):
        try:
            result = operation()
            successes.append((i, result))
        except Exception as e:
            failures.append((i, e))
            log_error(f"Operation {i} failed", e)

    return successes, failures


# Usage
results, errors = handle_partial_success([
    lambda: search_database("query1"),
    lambda: search_database("query2"),
    lambda: search_database("query3")
])

print(f"Succeeded: {len(results)}, Failed: {len(errors)}")

# Process partial results
if results:
    process_results(results)
```

## Graceful Degradation

Keep system working with reduced functionality.

### Degraded Mode

```python
class DegradedModeAgent:
    """Agent that operates in degraded mode on errors."""

    def __init__(self):
        self.mode = "normal"
        self.features = {
            "search": True,
            "summarize": True,
            "tools": True
        }

    def handle_repeated_failures(self, component: str):
        """Degrade mode after repeated failures."""

        self.features[component] = False

        # Check if should enter degraded mode
        active_features = sum(self.features.values())
        if active_features < len(self.features) / 2:
            self.mode = "degraded"
            print("Entering degraded mode")

    def query(self, user_input: str) -> str:
        """Process query with available features."""

        if self.mode == "degraded":
            print("Warning: Operating in degraded mode")

        # Try full processing
        if self.features["search"]:
            try:
                results = self.search(user_input)
            except Exception as e:
                log_error("Search failed", e)
                self.handle_repeated_failures("search")
                results = []
        else:
            results = []

        if self.features["summarize"] and results:
            try:
                summary = self.summarize(results)
            except Exception as e:
                log_error("Summarize failed", e)
                self.handle_repeated_failures("summarize")
                summary = results[0] if results else "No results"
        else:
            summary = results[0] if results else "No results"

        return summary
```

### Progressive Enhancement

```python
def progressive_enhancement(query: str) -> dict:
    """Start with basic functionality, add features if available."""

    response = {
        "query": query,
        "basic_response": "I understand your question."
    }

    # Try to add search results
    try:
        response["search_results"] = search(query)
    except Exception:
        pass  # Continue without search

    # Try to add analysis
    try:
        if "search_results" in response:
            response["analysis"] = analyze(response["search_results"])
    except Exception:
        pass  # Continue without analysis

    # Try to add visualization
    try:
        if "analysis" in response:
            response["visualization"] = create_chart(response["analysis"])
    except Exception:
        pass  # Continue without viz

    return response
```

## Error Reporting

Report errors effectively for debugging.

### Detailed Error Context

```python
class DetailedErrorReporter:
    """Report errors with comprehensive context."""

    def report_error(
        self,
        error: Exception,
        context: dict
    ) -> dict:
        """Create detailed error report."""

        import traceback

        report = {
            "timestamp": time.time(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context
        }

        # Add system information
        report["system"] = {
            "python_version": sys.version,
            "platform": sys.platform
        }

        # Add classification
        classifier = ErrorClassifier()
        error_type, severity, recoverable = classifier.classify(error)
        report["classification"] = {
            "type": error_type.value,
            "severity": severity.value,
            "recoverable": recoverable
        }

        return report

    def format_for_user(self, report: dict) -> str:
        """Format error report for user."""

        severity = report["classification"]["severity"]

        if severity == ErrorSeverity.LOW.value:
            return "A minor issue occurred. Continuing..."

        elif severity == ErrorSeverity.MEDIUM.value:
            return f"An error occurred: {report['error_message']}. I'll try to continue."

        elif severity == ErrorSeverity.HIGH.value:
            return f"A significant error occurred: {report['error_message']}. Please try again."

        else:  # CRITICAL
            return f"A critical error occurred: {report['error_message']}. Please contact support."

    def format_for_logs(self, report: dict) -> str:
        """Format error report for logs."""

        log_entry = f"""
ERROR REPORT
============
Time: {report['timestamp']}
Type: {report['error_type']}
Message: {report['error_message']}
Classification: {report['classification']}

Context:
{json.dumps(report['context'], indent=2)}

Traceback:
{report['traceback']}
"""
        return log_entry
```

## Logging and Monitoring

Track errors for analysis and debugging.

### Error Logger

```python
import logging
from collections import defaultdict

class ErrorLogger:
    """Comprehensive error logging."""

    def __init__(self):
        self.logger = logging.getLogger("agent_errors")
        self.error_counts = defaultdict(int)
        self.recent_errors = []

    def log_error(
        self,
        error: Exception,
        context: dict,
        severity: str = "ERROR"
    ):
        """Log error with context."""

        error_type = type(error).__name__

        # Count errors
        self.error_counts[error_type] += 1

        # Store recent errors
        self.recent_errors.append({
            "timestamp": time.time(),
            "type": error_type,
            "message": str(error),
            "context": context
        })

        # Keep only recent 100
        if len(self.recent_errors) > 100:
            self.recent_errors = self.recent_errors[-100:]

        # Log to logger
        self.logger.log(
            getattr(logging, severity),
            f"{error_type}: {error}",
            extra={"context": context}
        )

    def get_statistics(self) -> dict:
        """Get error statistics."""

        total_errors = sum(self.error_counts.values())

        return {
            "total_errors": total_errors,
            "by_type": dict(self.error_counts),
            "most_common": self._get_most_common(5),
            "recent_errors": self.recent_errors[-10:]
        }

    def _get_most_common(self, n: int) -> list:
        """Get most common error types."""

        sorted_errors = sorted(
            self.error_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_errors[:n]
```

### Error Monitoring

```python
class ErrorMonitor:
    """Monitor error rates and patterns."""

    def __init__(self, alert_threshold: int = 10):
        self.alert_threshold = alert_threshold
        self.error_window = []  # Recent errors
        self.window_size = 300  # 5 minutes

    def record_error(self, error: Exception):
        """Record an error occurrence."""

        now = time.time()

        # Add to window
        self.error_window.append(now)

        # Remove old errors
        cutoff = now - self.window_size
        self.error_window = [t for t in self.error_window if t > cutoff]

        # Check if should alert
        if len(self.error_window) >= self.alert_threshold:
            self._alert_high_error_rate()

    def _alert_high_error_rate(self):
        """Alert on high error rate."""

        rate = len(self.error_window) / self.window_size * 60

        print(f"ALERT: High error rate: {rate:.1f} errors/minute")

        # In production, send actual alert
        # send_alert(f"Error rate: {rate:.1f}/min")

    def get_error_rate(self) -> float:
        """Get current error rate (errors per minute)."""

        if not self.error_window:
            return 0.0

        window_duration = time.time() - self.error_window[0]
        return len(self.error_window) / window_duration * 60
```

## User-Facing Error Messages

Communicate errors effectively to users.

### Error Message Formatter

```python
class UserErrorMessages:
    """Generate user-friendly error messages."""

    def format_message(
        self,
        error: Exception,
        context: dict
    ) -> str:
        """Format error for user."""

        error_type = type(error).__name__

        # Map internal errors to user messages
        messages = {
            "TimeoutError": "I'm taking longer than usual. Please try again.",
            "ConnectionError": "I'm having trouble connecting. Please check your internet.",
            "RateLimitError": "I'm receiving too many requests. Please wait a moment.",
            "ParseError": "I had trouble understanding the response. Please try rephrasing.",
            "ToolExecutionError": "The operation couldn't be completed. Please try a different approach.",
            "ValidationError": "The input doesn't look right. Please check and try again."
        }

        return messages.get(error_type, "Something went wrong. Please try again.")

    def add_suggestions(self, error: Exception) -> list[str]:
        """Provide suggestions based on error."""

        error_type = type(error).__name__

        suggestions = {
            "TimeoutError": [
                "Try a simpler query",
                "Wait a moment and retry"
            ],
            "ConnectionError": [
                "Check your internet connection",
                "Try again in a moment"
            ],
            "ParseError": [
                "Rephrase your question",
                "Be more specific"
            ]
        }

        return suggestions.get(error_type, ["Try again"])
```

## Circuit Breakers

Prevent cascading failures.

### Circuit Breaker Implementation

```python
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker pattern for error handling."""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    def call(self, operation: callable) -> Any:
        """Execute operation through circuit breaker."""

        if self.state == CircuitState.OPEN:
            # Check if should try again
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = operation()
            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            print(f"Circuit breaker opened after {self.failure_count} failures")

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try again."""
        if self.last_failure_time is None:
            return True

        return time.time() - self.last_failure_time >= self.timeout


# Usage
breaker = CircuitBreaker(failure_threshold=3, timeout=60.0)

try:
    result = breaker.call(lambda: unreliable_api_call())
except Exception as e:
    print(f"Call failed: {e}")
```

## Error Budgets

Manage acceptable error rates.

### Error Budget Tracker

```python
class ErrorBudget:
    """Track error budget for SLOs."""

    def __init__(
        self,
        total_requests: int,
        error_budget_percent: float = 1.0  # 1% error budget
    ):
        self.total_requests = total_requests
        self.error_budget_percent = error_budget_percent
        self.error_count = 0
        self.success_count = 0

    def record_request(self, success: bool):
        """Record a request result."""
        if success:
            self.success_count += 1
        else:
            self.error_count += 1

    def is_budget_exhausted(self) -> bool:
        """Check if error budget is exhausted."""
        total = self.error_count + self.success_count
        if total == 0:
            return False

        error_rate = self.error_count / total * 100
        return error_rate >= self.error_budget_percent

    def remaining_budget(self) -> float:
        """Get remaining error budget."""
        total = self.error_count + self.success_count
        if total == 0:
            return self.error_budget_percent

        current_rate = self.error_count / total * 100
        return max(0, self.error_budget_percent - current_rate)

    def get_stats(self) -> dict:
        """Get budget statistics."""
        total = self.error_count + self.success_count

        return {
            "total_requests": total,
            "errors": self.error_count,
            "successes": self.success_count,
            "error_rate": self.error_count / total * 100 if total > 0 else 0,
            "budget_percent": self.error_budget_percent,
            "remaining_budget": self.remaining_budget(),
            "budget_exhausted": self.is_budget_exhausted()
        }
```

## Best Practices

Guidelines for effective error handling.

### 1. Fail Fast for Logic Errors

```python
def fail_fast(data: dict):
    """Validate early and fail fast."""

    # Don't try to recover from programming errors
    if "required_field" not in data:
        raise ValueError("Missing required field")  # Don't catch

    if not isinstance(data["value"], int):
        raise TypeError("Value must be integer")  # Don't catch

    # Continue only with valid data
    process(data)
```

### 2. Be Specific with Exceptions

```python
# ❌ BAD: Catch all exceptions
try:
    operation()
except Exception:
    pass  # Hides all errors!

# ✅ GOOD: Catch specific exceptions
try:
    operation()
except TimeoutError:
    retry()
except ValueError as e:
    log_error("Invalid value", e)
    raise
```

### 3. Always Log Errors

```python
def with_logging(operation: callable) -> Any:
    """Always log errors."""

    try:
        return operation()
    except Exception as e:
        # Always log before handling
        logger.error(f"Operation failed: {e}", exc_info=True)
        raise
```

### 4. Provide Context in Errors

```python
# ❌ BAD: No context
raise Exception("Failed")

# ✅ GOOD: Rich context
raise ToolExecutionError(
    f"Tool '{tool_name}' failed with arguments {arguments}: {error}",
    original_error=error
)
```

### 5. Don't Swallow Errors Silently

```python
# ❌ BAD: Silent failure
try:
    critical_operation()
except:
    pass  # Silently fails!

# ✅ GOOD: At least log
try:
    critical_operation()
except Exception as e:
    logger.error("Critical operation failed", e)
    # Then decide how to handle
```

### 6. Test Error Paths

```python
def test_error_handling():
    """Test error handling paths."""

    # Test timeout handling
    with pytest.raises(TimeoutError):
        agent.query("test", timeout=0.001)

    # Test invalid input
    with pytest.raises(ValueError):
        agent.query("")

    # Test recovery
    result = agent.query_with_retry("test", max_attempts=3)
    assert result is not None
```

## Summary

Error handling is what separates toy agents from production systems. Key principles:

**Error Types**:

- LLM errors (timeouts, rate limits, context length)
- Parsing errors (invalid JSON, missing fields)
- Tool errors (not found, invalid arguments, execution failures)
- System errors (network, file system, resources)
- Validation errors (schema, business logic)

**Patterns**:

- Try-catch-finally for cleanup
- Guard clauses for validation
- Error wrapping for abstraction
- Default values for graceful degradation
- Retries with exponential backoff
- Fallbacks for alternatives
- Circuit breakers for cascading failures

**Best Practices**:

- Fail fast for logic errors
- Be specific with exceptions
- Always log errors
- Provide context
- Don't swallow errors
- Test error paths
- Monitor error rates
- Communicate clearly to users

Error handling isn't optional--it's the foundation of reliability.

## Next Steps

Now that you understand error handling, explore:

- **[Basic Tool Calling](basic-tool-calling.md)** - Handling tool errors
- **[Context Management](context-management.md)** - Error recovery with context
- **[State Basics](state-basics.md)** - Maintaining state through errors
- **Monitoring and Observability** - Advanced error tracking
- **Resilience Patterns** - Building fault-tolerant systems

**Practice exercises**:

1. Implement a complete error handling system for an agent
2. Build a retry mechanism with exponential backoff and jitter
3. Create a circuit breaker for protecting external services
4. Implement graceful degradation for multi-feature agent
5. Build an error monitoring system with alerting

**Advanced topics**:

- Distributed error handling
- Error correlation across services
- Automated error recovery
- Chaos engineering for agents
- Error budget management

Errors are inevitable. Handle them well, and your agent becomes bulletproof.
