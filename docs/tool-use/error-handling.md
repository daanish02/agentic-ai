# Error Handling and Resilience

## Table of Contents

- [Introduction](#introduction)
- [Types of Tool Failures](#types-of-tool-failures)
- [Retry Strategies](#retry-strategies)
- [Fallback Mechanisms](#fallback-mechanisms)
- [Handling Malformed Parameters](#handling-malformed-parameters)
- [Rate Limiting](#rate-limiting)
- [Timeouts](#timeouts)
- [Circuit Breakers](#circuit-breakers)
- [Error Recovery Patterns](#error-recovery-patterns)
- [Graceful Degradation](#graceful-degradation)
- [Error Context and Logging](#error-context-and-logging)
- [Validation and Prevention](#validation-and-prevention)
- [Idempotency](#idempotency)
- [Compensation and Rollback](#compensation-and-rollback)
- [Error Budgets](#error-budgets)
- [Testing Error Handling](#testing-error-handling)
- [Best Practices](#best-practices)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Tools fail. Networks are unreliable. APIs have limits. Services go down. This is not a question of **if** but **when**.

Robust agent systems don't just handle the happy path - they **anticipate, detect, and recover from failures** gracefully. The difference between a brittle demo and a production-ready agent is error handling.

> "Hope for the best, plan for the worst, and prepare to be surprised."  
> -- Engineering wisdom

This guide covers comprehensive error handling strategies: from simple retries to sophisticated recovery mechanisms, from timeout management to graceful degradation.

## Types of Tool Failures

Understanding different failure modes.

### Network Failures

```python
# Connection errors
ConnectionError: Failed to connect to API endpoint
TimeoutError: Request timed out after 30 seconds
DNSError: Could not resolve hostname

# Transient network issues
- Packet loss
- Temporary unavailability
- Network congestion
```

### API Failures

```python
# HTTP status codes
400: Bad Request - malformed input
401: Unauthorized - missing/invalid credentials
403: Forbidden - insufficient permissions
404: Not Found - endpoint doesn't exist
429: Too Many Requests - rate limit exceeded
500: Internal Server Error - server-side problem
502: Bad Gateway - upstream service down
503: Service Unavailable - temporary outage
504: Gateway Timeout - upstream timeout
```

### Validation Failures

```python
# Parameter validation
ValueError: Invalid parameter type
ValidationError: Parameter doesn't meet constraints
SchemaError: Parameters don't match schema

# Business logic validation
InsufficientCreditsError: User has no API credits
QuotaExceededError: Daily quota exceeded
InvalidStateError: Operation not allowed in current state
```

### Execution Failures

```python
# Runtime errors
MemoryError: Out of memory
OverflowError: Numeric overflow
RecursionError: Maximum recursion depth exceeded

# Tool-specific errors
ParseError: Could not parse response
TransformationError: Data transformation failed
IntegrationError: External service integration failed
```

### Classification System

```python
from enum import Enum

class ErrorType(Enum):
    """Classification of errors."""
    TRANSIENT = "transient"  # Temporary, retry likely to succeed
    PERMANENT = "permanent"  # Won't succeed on retry
    RATE_LIMIT = "rate_limit"  # Exceeded rate limit
    VALIDATION = "validation"  # Invalid input
    TIMEOUT = "timeout"  # Operation timed out
    NETWORK = "network"  # Network connectivity issue
    AUTHENTICATION = "authentication"  # Auth/permission issue
    UNKNOWN = "unknown"  # Unclassified error


class ToolError(Exception):
    """Base class for tool errors."""

    def __init__(
        self,
        message: str,
        error_type: ErrorType,
        tool_name: str,
        retryable: bool,
        details: dict = None
    ):
        self.message = message
        self.error_type = error_type
        self.tool_name = tool_name
        self.retryable = retryable
        self.details = details or {}
        super().__init__(message)


def classify_error(exception: Exception, tool_name: str) -> ToolError:
    """Classify an exception into a ToolError."""

    # Network errors - usually transient
    if isinstance(exception, (ConnectionError, TimeoutError)):
        return ToolError(
            message=str(exception),
            error_type=ErrorType.NETWORK,
            tool_name=tool_name,
            retryable=True
        )

    # HTTP errors
    if hasattr(exception, 'status_code'):
        status = exception.status_code

        if status == 429:
            return ToolError(
                message="Rate limit exceeded",
                error_type=ErrorType.RATE_LIMIT,
                tool_name=tool_name,
                retryable=True,
                details={"retry_after": getattr(exception, 'retry_after', 60)}
            )

        elif status in [500, 502, 503, 504]:
            return ToolError(
                message=f"Server error: {status}",
                error_type=ErrorType.TRANSIENT,
                tool_name=tool_name,
                retryable=True
            )

        elif status in [400, 404, 422]:
            return ToolError(
                message=f"Client error: {status}",
                error_type=ErrorType.VALIDATION,
                tool_name=tool_name,
                retryable=False
            )

        elif status in [401, 403]:
            return ToolError(
                message=f"Authentication error: {status}",
                error_type=ErrorType.AUTHENTICATION,
                tool_name=tool_name,
                retryable=False
            )

    # Validation errors - not retryable
    if isinstance(exception, (ValueError, TypeError)):
        return ToolError(
            message=str(exception),
            error_type=ErrorType.VALIDATION,
            tool_name=tool_name,
            retryable=False
        )

    # Unknown error - assume not retryable
    return ToolError(
        message=str(exception),
        error_type=ErrorType.UNKNOWN,
        tool_name=tool_name,
        retryable=False,
        details={"exception_type": type(exception).__name__}
    )
```

## Retry Strategies

Different approaches to retrying failed operations.

### Simple Retry

```python
import time
from typing import Callable, Any

def simple_retry(
    func: Callable,
    max_attempts: int = 3,
    delay: float = 1.0
) -> Any:
    """
    Simple retry with fixed delay.
    """
    last_exception = None

    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            last_exception = e
            print(f"Attempt {attempt + 1} failed: {e}")

            if attempt < max_attempts - 1:
                time.sleep(delay)
            else:
                raise last_exception


# Usage
result = simple_retry(
    lambda: call_api("endpoint"),
    max_attempts=3,
    delay=2.0
)
```

### Exponential Backoff

```python
import random

def exponential_backoff_retry(
    func: Callable,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
) -> Any:
    """
    Retry with exponential backoff.

    Delay grows exponentially: base_delay * (exponential_base ** attempt)
    """
    last_exception = None

    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            last_exception = e

            if attempt < max_attempts - 1:
                # Calculate delay
                delay = min(
                    base_delay * (exponential_base ** attempt),
                    max_delay
                )

                # Add jitter to avoid thundering herd
                if jitter:
                    delay = delay * (0.5 + random.random())

                print(f"Attempt {attempt + 1} failed. Retrying in {delay:.2f}s...")
                time.sleep(delay)
            else:
                raise last_exception


# Usage
result = exponential_backoff_retry(
    lambda: call_unreliable_api(),
    max_attempts=5,
    base_delay=1.0,
    max_delay=60.0
)
```

### Smart Retry

```python
def smart_retry(
    func: Callable,
    should_retry: Callable[[Exception], bool],
    max_attempts: int = 3,
    delay_calculator: Callable[[int, Exception], float] = None
) -> Any:
    """
    Retry with custom logic for which errors to retry and how long to wait.
    """
    if delay_calculator is None:
        delay_calculator = lambda attempt, error: 2 ** attempt

    last_exception = None

    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            last_exception = e

            # Check if we should retry this error
            if not should_retry(e):
                print(f"Error not retryable: {e}")
                raise

            if attempt < max_attempts - 1:
                delay = delay_calculator(attempt, e)
                print(f"Attempt {attempt + 1} failed. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise last_exception


# Usage
def should_retry_error(error: Exception) -> bool:
    """Determine if error is retryable."""
    # Retry network errors
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True

    # Retry 5xx server errors
    if hasattr(error, 'status_code') and 500 <= error.status_code < 600:
        return True

    # Don't retry client errors
    return False


def calculate_delay(attempt: int, error: Exception) -> float:
    """Calculate delay based on error type."""
    if hasattr(error, 'retry_after'):
        # Respect Retry-After header
        return float(error.retry_after)

    # Exponential backoff
    return min(2 ** attempt, 60)


result = smart_retry(
    func=lambda: call_api(),
    should_retry=should_retry_error,
    max_attempts=5,
    delay_calculator=calculate_delay
)
```

### Async Retry

```python
import asyncio

async def async_retry(
    func: Callable,
    max_attempts: int = 3,
    base_delay: float = 1.0
) -> Any:
    """
    Async retry with exponential backoff.
    """
    last_exception = None

    for attempt in range(max_attempts):
        try:
            return await func()
        except Exception as e:
            last_exception = e

            if attempt < max_attempts - 1:
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
            else:
                raise last_exception


# Usage
# result = await async_retry(
#     lambda: fetch_data_async(),
#     max_attempts=3
# )
```

### Retry with Circuit Breaker

```python
from datetime import datetime, timedelta

class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_duration: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout_duration = timeout_duration
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open

    def call(self, func: Callable) -> Any:
        """Execute function with circuit breaker protection."""

        if self.state == "open":
            # Check if we should try again
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout_duration):
                self.state = "half_open"
                print("Circuit breaker: Trying again (half-open)")
            else:
                raise Exception("Circuit breaker is OPEN - not attempting call")

        try:
            result = func()

            # Success - reset if we were half-open
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
                print("Circuit breaker: Reset to closed")

            return result

        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                print(f"Circuit breaker: OPENED after {self.failure_count} failures")

            raise


# Usage
breaker = CircuitBreaker(failure_threshold=3, timeout_duration=30)

for i in range(10):
    try:
        result = breaker.call(lambda: call_flaky_api())
        print(f"Success: {result}")
    except Exception as e:
        print(f"Failed: {e}")
    time.sleep(5)
```

## Fallback Mechanisms

Alternative approaches when primary method fails.

### Simple Fallback

```python
def with_fallback(
    primary: Callable,
    fallback: Callable,
    fallback_condition: Callable[[Exception], bool] = None
) -> Any:
    """
    Try primary function, fall back to alternative if it fails.
    """
    try:
        return primary()
    except Exception as e:
        # Check if we should use fallback for this error
        if fallback_condition is None or fallback_condition(e):
            print(f"Primary failed ({e}), using fallback")
            return fallback()
        else:
            raise


# Usage
result = with_fallback(
    primary=lambda: fetch_from_primary_api(),
    fallback=lambda: fetch_from_cache(),
    fallback_condition=lambda e: isinstance(e, TimeoutError)
)
```

### Cascade Fallback

```python
def cascade_fallback(
    strategies: List[tuple[str, Callable]]
) -> tuple[Any, str]:
    """
    Try multiple strategies in order until one succeeds.
    """
    last_exception = None

    for strategy_name, strategy_func in strategies:
        try:
            result = strategy_func()
            return result, strategy_name
        except Exception as e:
            print(f"Strategy '{strategy_name}' failed: {e}")
            last_exception = e
            continue

    raise Exception(f"All strategies failed. Last error: {last_exception}")


# Usage
result, used_strategy = cascade_fallback([
    ("primary_api", lambda: call_primary_api()),
    ("secondary_api", lambda: call_secondary_api()),
    ("cache", lambda: get_from_cache()),
    ("default", lambda: get_default_value())
])

print(f"Used strategy: {used_strategy}")
```

### Intelligent Fallback Selection

```python
from typing import List, Tuple, Optional

class FallbackStrategy:
    """Represents a fallback strategy."""

    def __init__(
        self,
        name: str,
        func: Callable,
        cost: float,  # Resource cost (0-1)
        latency: float,  # Expected latency in seconds
        reliability: float  # Reliability (0-1)
    ):
        self.name = name
        self.func = func
        self.cost = cost
        self.latency = latency
        self.reliability = reliability


def intelligent_fallback(
    strategies: List[FallbackStrategy],
    optimize_for: str = "reliability"  # "cost", "latency", or "reliability"
) -> Tuple[Any, str]:
    """
    Select fallback strategy based on optimization goal.
    """
    # Sort strategies by optimization criterion
    if optimize_for == "cost":
        sorted_strategies = sorted(strategies, key=lambda s: s.cost)
    elif optimize_for == "latency":
        sorted_strategies = sorted(strategies, key=lambda s: s.latency)
    else:  # reliability
        sorted_strategies = sorted(strategies, key=lambda s: -s.reliability)

    # Try strategies in order
    for strategy in sorted_strategies:
        try:
            result = strategy.func()
            return result, strategy.name
        except Exception as e:
            print(f"Strategy {strategy.name} failed: {e}")
            continue

    raise Exception("All fallback strategies failed")


# Usage
strategies = [
    FallbackStrategy(
        name="premium_api",
        func=lambda: call_premium_api(),
        cost=0.9,
        latency=0.5,
        reliability=0.99
    ),
    FallbackStrategy(
        name="standard_api",
        func=lambda: call_standard_api(),
        cost=0.5,
        latency=1.0,
        reliability=0.95
    ),
    FallbackStrategy(
        name="cache",
        func=lambda: get_from_cache(),
        cost=0.1,
        latency=0.1,
        reliability=0.8
    )
]

# Optimize for cost
result, used = intelligent_fallback(strategies, optimize_for="cost")
```

### Partial Fallback

```python
def partial_fallback(
    primary: Callable,
    fallback: Callable,
    combine: Callable
) -> Any:
    """
    Use fallback to supplement partial primary results.
    """
    primary_result = None
    fallback_result = None

    # Try primary
    try:
        primary_result = primary()
    except Exception as e:
        print(f"Primary failed: {e}")

    # Try fallback if primary failed or returned partial results
    if primary_result is None or is_partial(primary_result):
        try:
            fallback_result = fallback()
        except Exception as e:
            print(f"Fallback failed: {e}")

    # Combine results
    return combine(primary_result, fallback_result)


def is_partial(result: Any) -> bool:
    """Check if result is partial/incomplete."""
    if isinstance(result, list):
        return len(result) < 5  # Expecting at least 5 items
    return False


def combine_results(primary: Any, fallback: Any) -> Any:
    """Combine primary and fallback results."""
    if primary is None:
        return fallback
    if fallback is None:
        return primary

    # Merge both
    if isinstance(primary, list) and isinstance(fallback, list):
        return primary + fallback

    return primary


# Usage
result = partial_fallback(
    primary=lambda: search_primary_source(),
    fallback=lambda: search_fallback_source(),
    combine=combine_results
)
```

## Handling Malformed Parameters

Dealing with invalid input gracefully.

### Parameter Validation

```python
from typing import Any, Dict, List
from dataclasses import dataclass

@dataclass
class ValidationRule:
    """A parameter validation rule."""
    param_name: str
    validator: Callable[[Any], bool]
    error_message: str
    fixer: Callable[[Any], Any] = None  # Optional auto-fix


class ParameterValidator:
    """Validate and fix tool parameters."""

    def __init__(self):
        self.rules: List[ValidationRule] = []

    def add_rule(
        self,
        param_name: str,
        validator: Callable,
        error_message: str,
        fixer: Callable = None
    ):
        """Add a validation rule."""
        self.rules.append(ValidationRule(
            param_name=param_name,
            validator=validator,
            error_message=error_message,
            fixer=fixer
        ))

    def validate_and_fix(
        self,
        parameters: Dict[str, Any]
    ) -> tuple[Dict[str, Any], List[str]]:
        """
        Validate parameters and attempt to fix issues.
        Returns (fixed_parameters, errors).
        """
        fixed_params = parameters.copy()
        errors = []

        for rule in self.rules:
            param_name = rule.param_name

            # Skip if parameter not provided
            if param_name not in fixed_params:
                continue

            param_value = fixed_params[param_name]

            # Validate
            if not rule.validator(param_value):
                # Try to fix
                if rule.fixer:
                    try:
                        fixed_value = rule.fixer(param_value)
                        fixed_params[param_name] = fixed_value
                        print(f"Auto-fixed {param_name}: {param_value} -> {fixed_value}")
                    except Exception:
                        errors.append(f"{param_name}: {rule.error_message}")
                else:
                    errors.append(f"{param_name}: {rule.error_message}")

        return fixed_params, errors


# Example usage
validator = ParameterValidator()

# Add validation rules with auto-fix
validator.add_rule(
    param_name="email",
    validator=lambda v: isinstance(v, str) and "@" in v,
    error_message="Must be valid email address",
    fixer=lambda v: v.strip().lower() if isinstance(v, str) else None
)

validator.add_rule(
    param_name="age",
    validator=lambda v: isinstance(v, int) and 0 <= v <= 150,
    error_message="Must be integer between 0 and 150",
    fixer=lambda v: int(v) if isinstance(v, (str, float)) else None
)

validator.add_rule(
    param_name="tags",
    validator=lambda v: isinstance(v, list),
    error_message="Must be a list",
    fixer=lambda v: [v] if not isinstance(v, list) else v
)

# Validate and fix parameters
params = {
    "email": " USER@EXAMPLE.COM ",
    "age": "25",
    "tags": "python"
}

fixed_params, errors = validator.validate_and_fix(params)

if errors:
    print(f"Validation errors: {errors}")
else:
    print(f"Fixed parameters: {fixed_params}")
```

### Coercion and Normalization

```python
class ParameterCoercer:
    """Coerce parameters to expected types."""

    @staticmethod
    def coerce_to_type(value: Any, target_type: type) -> Any:
        """Coerce value to target type."""

        # Already correct type
        if isinstance(value, target_type):
            return value

        # String conversions
        if target_type == str:
            return str(value)

        # Integer conversions
        if target_type == int:
            if isinstance(value, str):
                # Try to parse
                return int(value.strip())
            if isinstance(value, float):
                return int(value)
            return int(value)

        # Float conversions
        if target_type == float:
            if isinstance(value, str):
                return float(value.strip())
            return float(value)

        # Boolean conversions
        if target_type == bool:
            if isinstance(value, str):
                return value.lower() in ['true', '1', 'yes', 'on']
            return bool(value)

        # List conversions
        if target_type == list:
            if isinstance(value, str):
                # Split by comma
                return [v.strip() for v in value.split(',')]
            if not isinstance(value, list):
                return [value]
            return value

        # Dict conversions
        if target_type == dict:
            if isinstance(value, str):
                import json
                return json.loads(value)
            return dict(value)

        raise TypeError(f"Cannot coerce {type(value)} to {target_type}")

    @staticmethod
    def normalize_parameters(
        parameters: Dict[str, Any],
        schema: Dict[str, type]
    ) -> Dict[str, Any]:
        """Normalize parameters according to schema."""
        normalized = {}

        for param_name, expected_type in schema.items():
            if param_name not in parameters:
                continue

            value = parameters[param_name]

            try:
                normalized[param_name] = ParameterCoercer.coerce_to_type(
                    value,
                    expected_type
                )
            except Exception as e:
                raise ValueError(
                    f"Cannot coerce parameter '{param_name}' to {expected_type}: {e}"
                )

        return normalized


# Usage
schema = {
    "age": int,
    "name": str,
    "active": bool,
    "tags": list
}

params = {
    "age": "25",
    "name": 123,
    "active": "yes",
    "tags": "python, coding"
}

normalized = ParameterCoercer.normalize_parameters(params, schema)
# Result: {"age": 25, "name": "123", "active": True, "tags": ["python", "coding"]}
```

## Rate Limiting

Handling API rate limits.

### Rate Limiter

```python
from collections import deque
from datetime import datetime, timedelta
import time

class RateLimiter:
    """
    Rate limiter using token bucket algorithm.
    """

    def __init__(self, max_calls: int, time_window: int):
        """
        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()

    def is_allowed(self) -> bool:
        """Check if a call is allowed."""
        now = datetime.now()

        # Remove old calls outside time window
        cutoff = now - timedelta(seconds=self.time_window)
        while self.calls and self.calls[0] < cutoff:
            self.calls.popleft()

        # Check if under limit
        return len(self.calls) < self.max_calls

    def record_call(self):
        """Record a call."""
        self.calls.append(datetime.now())

    def wait_if_needed(self):
        """Wait if rate limit is exceeded."""
        while not self.is_allowed():
            # Calculate wait time
            if self.calls:
                oldest_call = self.calls[0]
                wait_until = oldest_call + timedelta(seconds=self.time_window)
                wait_seconds = (wait_until - datetime.now()).total_seconds()

                if wait_seconds > 0:
                    print(f"Rate limit reached. Waiting {wait_seconds:.2f}s...")
                    time.sleep(wait_seconds + 0.1)  # Small buffer
            else:
                break

    def __call__(self, func: Callable) -> Callable:
        """Decorator to rate limit a function."""
        def wrapper(*args, **kwargs):
            self.wait_if_needed()
            self.record_call()
            return func(*args, **kwargs)
        return wrapper


# Usage
limiter = RateLimiter(max_calls=10, time_window=60)  # 10 calls per minute

@limiter
def call_api():
    """This function is rate limited."""
    return "API result"


# Or manually
for i in range(20):
    limiter.wait_if_needed()
    result = call_api()
    limiter.record_call()
```

### Adaptive Rate Limiting

```python
class AdaptiveRateLimiter:
    """
    Rate limiter that adapts to 429 responses.
    """

    def __init__(self, initial_rate: int = 10, time_window: int = 60):
        self.current_rate = initial_rate
        self.time_window = time_window
        self.calls = deque()
        self.last_429_time = None
        self.backoff_multiplier = 0.5

    def handle_response(self, success: bool, response=None):
        """Adjust rate based on response."""

        if not success:
            # Check if rate limit error
            if response and hasattr(response, 'status_code'):
                if response.status_code == 429:
                    # Reduce rate
                    self.current_rate = max(
                        1,
                        int(self.current_rate * self.backoff_multiplier)
                    )
                    self.last_429_time = datetime.now()
                    print(f"Rate limit hit. Reduced rate to {self.current_rate} calls per {self.time_window}s")

        else:
            # Successful call - gradually increase rate
            if self.last_429_time:
                time_since_429 = (datetime.now() - self.last_429_time).total_seconds()

                # If no 429 for a while, increase rate
                if time_since_429 > self.time_window * 2:
                    self.current_rate = min(
                        self.current_rate + 1,
                        50  # Max rate
                    )
                    print(f"Increased rate to {self.current_rate}")

    def wait_if_needed(self):
        """Wait based on current rate."""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.time_window)

        # Remove old calls
        while self.calls and self.calls[0] < cutoff:
            self.calls.popleft()

        # Wait if at limit
        if len(self.calls) >= self.current_rate:
            oldest = self.calls[0]
            wait_until = oldest + timedelta(seconds=self.time_window)
            wait_seconds = (wait_until - now).total_seconds()

            if wait_seconds > 0:
                time.sleep(wait_seconds)

        self.calls.append(now)


# Usage
adaptive_limiter = AdaptiveRateLimiter(initial_rate=10)

def call_with_adaptive_limit():
    adaptive_limiter.wait_if_needed()

    try:
        response = call_api()
        adaptive_limiter.handle_response(success=True)
        return response
    except Exception as e:
        adaptive_limiter.handle_response(success=False, response=e)
        raise
```

## Timeouts

Preventing operations from hanging indefinitely.

### Simple Timeout

```python
import signal
from contextlib import contextmanager

class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds: int):
    """Context manager for timeout."""

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    # Set the signal handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)  # Disable alarm


# Usage (Unix only)
try:
    with timeout(5):
        result = slow_operation()
except TimeoutError as e:
    print(f"Timeout: {e}")
```

### Async Timeout

```python
import asyncio

async def with_timeout(
    coro,
    timeout_seconds: float
) -> Any:
    """Execute async operation with timeout."""

    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation timed out after {timeout_seconds}s")


# Usage
# try:
#     result = await with_timeout(
#         fetch_data(),
#         timeout_seconds=5.0
#     )
# except TimeoutError:
#     print("Operation timed out")
```

### Progressive Timeout

```python
class ProgressiveTimeout:
    """
    Timeout that increases with retries.
    """

    def __init__(
        self,
        initial_timeout: float,
        max_timeout: float,
        multiplier: float = 1.5
    ):
        self.initial_timeout = initial_timeout
        self.max_timeout = max_timeout
        self.multiplier = multiplier
        self.current_attempt = 0

    def get_timeout(self) -> float:
        """Get timeout for current attempt."""
        timeout = self.initial_timeout * (self.multiplier ** self.current_attempt)
        return min(timeout, self.max_timeout)

    async def execute(self, coro) -> Any:
        """Execute with progressive timeout."""
        timeout_seconds = self.get_timeout()

        try:
            result = await asyncio.wait_for(coro, timeout=timeout_seconds)
            self.current_attempt = 0  # Reset on success
            return result

        except asyncio.TimeoutError:
            self.current_attempt += 1
            raise TimeoutError(
                f"Timeout after {timeout_seconds}s (attempt {self.current_attempt})"
            )


# Usage
progressive = ProgressiveTimeout(
    initial_timeout=5.0,
    max_timeout=30.0,
    multiplier=1.5
)

for attempt in range(3):
    try:
        # result = await progressive.execute(fetch_data())
        break
    except TimeoutError as e:
        print(f"Attempt {attempt + 1} timed out: {e}")
```

## Circuit Breakers

Preventing cascading failures (expanded from earlier).

### Advanced Circuit Breaker

```python
from enum import Enum
from datetime import datetime, timedelta
from typing import Callable, Any

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if recovered


class AdvancedCircuitBreaker:
    """
    Advanced circuit breaker with monitoring.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: int = 60,
        monitored_exceptions: tuple = (Exception,)
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.monitored_exceptions = monitored_exceptions

        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

        # Metrics
        self.total_calls = 0
        self.total_failures = 0
        self.total_rejections = 0

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker."""

        self.total_calls += 1

        # Check if circuit should transition
        self._check_state_transition()

        if self.state == CircuitState.OPEN:
            self.total_rejections += 1
            raise Exception(
                f"Circuit breaker is OPEN. "
                f"Service has failed {self.failure_count} times. "
                f"Will retry after {self.timeout}s"
            )

        try:
            result = func(*args, **kwargs)

            # Success
            self._on_success()

            return result

        except self.monitored_exceptions as e:
            # Failure
            self._on_failure()
            raise

    def _check_state_transition(self):
        """Check if state should transition."""

        if self.state == CircuitState.OPEN:
            if self.last_failure_time:
                time_since_failure = (
                    datetime.now() - self.last_failure_time
                ).total_seconds()

                if time_since_failure >= self.timeout:
                    print("Circuit breaker: Transitioning to HALF_OPEN")
                    self.state = CircuitState.HALF_OPEN
                    self.failure_count = 0
                    self.success_count = 0

    def _on_success(self):
        """Handle successful call."""

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1

            if self.success_count >= self.success_threshold:
                print("Circuit breaker: Transitioning to CLOSED")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0

    def _on_failure(self):
        """Handle failed call."""

        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            # Failed during test - back to open
            print("Circuit breaker: Test failed, back to OPEN")
            self.state = CircuitState.OPEN

        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                print(
                    f"Circuit breaker: OPENING after "
                    f"{self.failure_count} failures"
                )
                self.state = CircuitState.OPEN

    def get_metrics(self) -> dict:
        """Get circuit breaker metrics."""
        return {
            "state": self.state.value,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_rejections": self.total_rejections,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "failure_rate": (
                self.total_failures / self.total_calls
                if self.total_calls > 0
                else 0
            )
        }


# Usage
breaker = AdvancedCircuitBreaker(
    failure_threshold=3,
    success_threshold=2,
    timeout=30
)

for i in range(10):
    try:
        result = breaker.call(call_unreliable_service)
        print(f"Call {i}: Success")
    except Exception as e:
        print(f"Call {i}: Failed - {e}")

    time.sleep(2)

print("Metrics:", breaker.get_metrics())
```

## Error Recovery Patterns

Patterns for recovering from errors.

### Saga Pattern

```python
from typing import List, Callable, Any

class SagaStep:
    """A step in a saga with compensation."""

    def __init__(
        self,
        name: str,
        action: Callable,
        compensation: Callable
    ):
        self.name = name
        self.action = action
        self.compensation = compensation


class Saga:
    """
    Saga pattern for distributed transactions.
    """

    def __init__(self):
        self.steps: List[SagaStep] = []
        self.completed_steps: List[tuple[SagaStep, Any]] = []

    def add_step(
        self,
        name: str,
        action: Callable,
        compensation: Callable
    ):
        """Add a step to the saga."""
        self.steps.append(SagaStep(name, action, compensation))

    def execute(self, initial_data: Any) -> Any:
        """
        Execute saga.
        If any step fails, compensate all completed steps.
        """
        current_data = initial_data

        try:
            for step in self.steps:
                print(f"Executing: {step.name}")
                result = step.action(current_data)
                self.completed_steps.append((step, result))
                current_data = result

            return current_data

        except Exception as e:
            print(f"Saga failed at step: {step.name}")
            print(f"Error: {e}")
            print("Compensating completed steps...")

            # Compensate in reverse order
            for completed_step, step_result in reversed(self.completed_steps):
                try:
                    print(f"Compensating: {completed_step.name}")
                    completed_step.compensation(step_result)
                except Exception as comp_error:
                    print(f"Compensation failed: {comp_error}")

            raise


# Example: Multi-service transaction
saga = Saga()

saga.add_step(
    name="reserve_inventory",
    action=lambda data: reserve_items(data["items"]),
    compensation=lambda result: release_items(result["reservation_id"])
)

saga.add_step(
    name="charge_payment",
    action=lambda data: charge_card(data["card"], data["amount"]),
    compensation=lambda result: refund_charge(result["charge_id"])
)

saga.add_step(
    name="create_shipment",
    action=lambda data: create_shipment(data["order_id"]),
    compensation=lambda result: cancel_shipment(result["shipment_id"])
)

try:
    result = saga.execute({"items": [...], "card": "...", "amount": 100})
except Exception:
    print("Transaction failed and was compensated")
```

## Graceful Degradation

Providing reduced functionality when full functionality isn't available.

### Degraded Mode

```python
from enum import Enum

class ServiceMode(Enum):
    FULL = "full"
    DEGRADED = "degraded"
    MINIMAL = "minimal"


class DegradableService:
    """
    Service that can operate in degraded modes.
    """

    def __init__(self):
        self.mode = ServiceMode.FULL
        self.failure_count = 0

    def execute(self, request: Any) -> Any:
        """Execute request in appropriate mode."""

        if self.mode == ServiceMode.FULL:
            try:
                return self._full_processing(request)
            except Exception as e:
                self._on_failure()
                # Fall back to degraded mode
                return self._degraded_processing(request)

        elif self.mode == ServiceMode.DEGRADED:
            try:
                return self._degraded_processing(request)
            except Exception:
                self._on_failure()
                # Fall back to minimal mode
                return self._minimal_processing(request)

        else:  # MINIMAL
            return self._minimal_processing(request)

    def _full_processing(self, request: Any) -> Any:
        """Full processing with all features."""
        # Call external APIs, do complex processing, etc.
        return {"result": "full", "quality": "high"}

    def _degraded_processing(self, request: Any) -> Any:
        """Degraded processing with essential features only."""
        # Use cache, simpler algorithms
        return {"result": "degraded", "quality": "medium"}

    def _minimal_processing(self, request: Any) -> Any:
        """Minimal processing - always works."""
        # Return basic/default result
        return {"result": "minimal", "quality": "low"}

    def _on_failure(self):
        """Handle failure and potentially degrade mode."""
        self.failure_count += 1

        if self.failure_count > 5:
            if self.mode == ServiceMode.FULL:
                print("Degrading to DEGRADED mode")
                self.mode = ServiceMode.DEGRADED
            elif self.mode == ServiceMode.DEGRADED:
                print("Degrading to MINIMAL mode")
                self.mode = ServiceMode.MINIMAL

    def restore(self):
        """Attempt to restore full functionality."""
        try:
            # Test if full processing works
            self._full_processing({"test": True})
            print("Restored to FULL mode")
            self.mode = ServiceMode.FULL
            self.failure_count = 0
        except Exception:
            print("Cannot restore to FULL mode yet")
```

## Error Context and Logging

Capturing comprehensive error information.

### Error Context

```python
import traceback
from datetime import datetime
from typing import Any, Dict

class ErrorContext:
    """Comprehensive error context."""

    def __init__(
        self,
        error: Exception,
        tool_name: str,
        parameters: Dict[str, Any],
        user_context: Dict[str, Any] = None
    ):
        self.error = error
        self.error_type = type(error).__name__
        self.error_message = str(error)
        self.tool_name = tool_name
        self.parameters = parameters
        self.user_context = user_context or {}
        self.timestamp = datetime.now()
        self.traceback = traceback.format_exc()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "user_context": self.user_context,
            "traceback": self.traceback
        }

    def __str__(self) -> str:
        """Human-readable error summary."""
        return f"""
Error in tool '{self.tool_name}':
  Type: {self.error_type}
  Message: {self.error_message}
  Parameters: {self.parameters}
  Time: {self.timestamp}
"""


def execute_with_context(
    tool_name: str,
    tool_func: Callable,
    parameters: Dict[str, Any],
    user_context: Dict[str, Any] = None
) -> tuple[Any, ErrorContext]:
    """
    Execute tool and capture error context.
    """
    try:
        result = tool_func(**parameters)
        return result, None

    except Exception as e:
        error_context = ErrorContext(
            error=e,
            tool_name=tool_name,
            parameters=parameters,
            user_context=user_context
        )

        # Log error
        import logging
        logging.error(error_context)

        return None, error_context
```

## Validation and Prevention

Preventing errors before they occur.

### Pre-execution Validation

```python
def validate_before_execute(
    tool_name: str,
    parameters: Dict[str, Any],
    schema: Dict[str, Any]
) -> List[str]:
    """
    Validate parameters before execution.
    Returns list of validation errors.
    """
    errors = []

    # Check required parameters
    required = schema.get("required", [])
    for param in required:
        if param not in parameters:
            errors.append(f"Missing required parameter: {param}")

    # Check parameter types and constraints
    properties = schema.get("properties", {})
    for param_name, param_value in parameters.items():
        if param_name not in properties:
            errors.append(f"Unknown parameter: {param_name}")
            continue

        param_schema = properties[param_name]

        # Type check
        expected_type = param_schema.get("type")
        if expected_type == "string" and not isinstance(param_value, str):
            errors.append(f"{param_name}: expected string, got {type(param_value)}")

        # Range check for numbers
        if expected_type in ["integer", "number"]:
            if "minimum" in param_schema and param_value < param_schema["minimum"]:
                errors.append(
                    f"{param_name}: value {param_value} below minimum "
                    f"{param_schema['minimum']}"
                )

    return errors


# Usage before execution
errors = validate_before_execute(tool_name, parameters, tool_schema)
if errors:
    raise ValueError(f"Validation failed: {errors}")

result = execute_tool(tool_name, parameters)
```

## Idempotency

Ensuring operations can be safely retried.

### Idempotent Operations

```python
import uuid
from typing import Any, Dict

class IdempotencyManager:
    """
    Manage idempotent operations using request IDs.
    """

    def __init__(self):
        self.completed_requests: Dict[str, Any] = {}

    def execute_idempotent(
        self,
        operation: Callable,
        request_id: str = None,
        **kwargs
    ) -> Any:
        """
        Execute operation idempotently.
        If same request_id is used, return cached result.
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        # Check if already completed
        if request_id in self.completed_requests:
            print(f"Request {request_id} already completed, returning cached result")
            return self.completed_requests[request_id]

        # Execute operation
        result = operation(**kwargs)

        # Cache result
        self.completed_requests[request_id] = result

        return result


# Usage
manager = IdempotencyManager()

# First call
result1 = manager.execute_idempotent(
    operation=create_user,
    request_id="unique-123",
    username="john"
)

# Retry with same ID - returns cached result, doesn't create duplicate
result2 = manager.execute_idempotent(
    operation=create_user,
    request_id="unique-123",
    username="john"
)

assert result1 == result2
```

## Compensation and Rollback

Undoing actions when errors occur (expanded from Saga).

### Transaction Rollback

```python
class Transaction:
    """
    Transaction with automatic rollback on failure.
    """

    def __init__(self):
        self.actions: List[tuple[Callable, Callable]] = []
        self.completed: List[Any] = []

    def add_action(
        self,
        forward: Callable,
        backward: Callable
    ):
        """
        Add an action with its rollback.
        """
        self.actions.append((forward, backward))

    def execute(self, initial_data: Any) -> Any:
        """
        Execute transaction.
        Automatically rolls back on failure.
        """
        current = initial_data

        try:
            for forward, backward in self.actions:
                result = forward(current)
                self.completed.append((result, backward))
                current = result

            # Success - commit
            return current

        except Exception as e:
            print(f"Transaction failed: {e}")
            self.rollback()
            raise

    def rollback(self):
        """Roll back completed actions."""
        print("Rolling back transaction...")

        for result, backward in reversed(self.completed):
            try:
                backward(result)
            except Exception as e:
                print(f"Rollback error: {e}")

        self.completed.clear()


# Example
transaction = Transaction()

transaction.add_action(
    forward=lambda data: create_order(data),
    backward=lambda result: delete_order(result["order_id"])
)

transaction.add_action(
    forward=lambda data: process_payment(data),
    backward=lambda result: refund_payment(result["payment_id"])
)

try:
    result = transaction.execute(order_data)
except Exception:
    print("Transaction was rolled back")
```

## Error Budgets

Managing acceptable error rates.

### Error Budget Tracker

```python
from datetime import datetime, timedelta
from collections import deque

class ErrorBudget:
    """
    Track error budget over time window.
    """

    def __init__(
        self,
        budget_percentage: float,  # Allowed error rate (0-100)
        time_window_hours: int = 24
    ):
        self.budget_percentage = budget_percentage
        self.time_window = timedelta(hours=time_window_hours)
        self.events = deque()  # (timestamp, success: bool)

    def record_event(self, success: bool):
        """Record an event (success or failure)."""
        now = datetime.now()
        self.events.append((now, success))

        # Remove old events
        cutoff = now - self.time_window
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def current_error_rate(self) -> float:
        """Calculate current error rate (0-100)."""
        if not self.events:
            return 0.0

        failures = sum(1 for _, success in self.events if not success)
        total = len(self.events)

        return (failures / total) * 100

    def budget_remaining(self) -> float:
        """Calculate remaining error budget (0-100)."""
        current_rate = self.current_error_rate()
        return max(0, self.budget_percentage - current_rate)

    def is_budget_exceeded(self) -> bool:
        """Check if error budget is exceeded."""
        return self.current_error_rate() > self.budget_percentage

    def get_status(self) -> Dict[str, Any]:
        """Get error budget status."""
        current_rate = self.current_error_rate()

        return {
            "error_rate": current_rate,
            "budget": self.budget_percentage,
            "remaining": self.budget_remaining(),
            "exceeded": self.is_budget_exceeded(),
            "total_events": len(self.events)
        }


# Usage
budget = ErrorBudget(budget_percentage=1.0, time_window_hours=24)  # 1% error budget

for i in range(100):
    success = call_service()
    budget.record_event(success)

    status = budget.get_status()
    print(f"Error rate: {status['error_rate']:.2f}%")

    if status["exceeded"]:
        print("ERROR BUDGET EXCEEDED - Taking action")
        # Switch to degraded mode, page ops, etc.
```

## Testing Error Handling

Ensuring error handling works correctly.

### Error Injection Testing

```python
import pytest

def test_retry_logic():
    """Test that retry logic works correctly."""

    call_count = 0

    def flaky_function():
        nonlocal call_count
        call_count += 1

        if call_count < 3:
            raise ConnectionError("Temporary failure")

        return "success"

    # Should succeed on 3rd attempt
    result = exponential_backoff_retry(
        flaky_function,
        max_attempts=5,
        base_delay=0.1
    )

    assert result == "success"
    assert call_count == 3


def test_circuit_breaker():
    """Test circuit breaker opens after failures."""

    breaker = CircuitBreaker(failure_threshold=3)

    def failing_function():
        raise Exception("Service unavailable")

    # First 3 calls should fail and open circuit
    for i in range(3):
        with pytest.raises(Exception):
            breaker.call(failing_function)

    # Circuit should now be open
    assert breaker.state == "open"

    # Next call should be rejected
    with pytest.raises(Exception, match="Circuit breaker is OPEN"):
        breaker.call(failing_function)


def test_fallback():
    """Test fallback mechanism."""

    def primary():
        raise Exception("Primary failed")

    def fallback():
        return "fallback result"

    result = with_fallback(primary, fallback)

    assert result == "fallback result"
```

## Best Practices

Guidelines for robust error handling.

**1. Classify Errors**: Know which errors are retryable vs permanent

**2. Fail Fast**: Don't retry validation errors or client errors

**3. Use Exponential Backoff**: Avoid overwhelming failed services

**4. Implement Circuit Breakers**: Prevent cascading failures

**5. Log Everything**: Capture full context for debugging

**6. Graceful Degradation**: Provide reduced functionality when needed

**7. Test Failure Scenarios**: Explicitly test error paths

**8. Set Timeouts**: Never wait indefinitely

**9. Monitor Error Rates**: Track and alert on error budgets

**10. Document Error Behavior**: Make error handling explicit

## Summary

Robust error handling is essential for production agent systems:

**Error Types**:
- Transient: Network issues, temporary unavailability
- Permanent: Validation errors, missing resources
- Rate limits: Too many requests
- Timeouts: Operations taking too long

**Strategies**:
- Retry with exponential backoff for transient errors
- Fallback to alternatives when primary fails
- Circuit breakers to prevent cascading failures
- Timeouts to prevent hanging
- Graceful degradation when full functionality unavailable

**Prevention**:
- Validate parameters before execution
- Normalize and coerce inputs
- Implement idempotency for safe retries
- Use error budgets to monitor health

**Recovery**:
- Saga pattern for distributed transactions
- Compensation actions for rollback
- State management for partial failures

Build resilient agents by expecting and handling failures at every level.

## Next Steps

Continue your journey:

- [Function Calling](function-calling.md) - Invoking tools reliably
- [Tool Chaining](tool-chaining.md) - Error handling in tool chains
- [API Integration](api-integration.md) - Handling real-world API failures
- [ReAct](../agent-architectures/react.md) - Error recovery with reasoning
