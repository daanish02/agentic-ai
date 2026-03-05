# State Basics

## Table of Contents

- [Introduction](#introduction)
- [What is State?](#what-is-state)
- [Why State Matters](#why-state-matters)
- [Types of State](#types-of-state)
- [State Storage](#state-storage)
- [State Management Patterns](#state-management-patterns)
- [In-Memory State](#in-memory-state)
- [Persistent State](#persistent-state)
- [State Transitions](#state-transitions)
- [State Consistency](#state-consistency)
- [Conversation State](#conversation-state)
- [Task State](#task-state)
- [Session State](#session-state)
- [State Serialization](#state-serialization)
- [State Recovery](#state-recovery)
- [State Best Practices](#state-best-practices)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Language models are stateless--each call is independent, with no memory of previous interactions. But agents need memory. They need to remember what they've done, what they've learned, where they are in a task, and what the user has told them. This memory is **state**.

State transforms a stateless model into a stateful agent. It's what allows an agent to:

- Continue conversations across multiple exchanges
- Track progress through multi-step tasks
- Remember user preferences
- Build context over time
- Resume after interruptions

Without state management:

```python
user: "My name is Alice"
agent: "Hello! How can I help?"
user: "What's my name?"
agent: "I don't know your name."  # Forgot!
```

With state management:

```python
user: "My name is Alice"
agent: "Hello Alice! How can I help?"
[state: {"user_name": "Alice"}]
user: "What's my name?"
agent: "Your name is Alice."  # Remembered!
```

> "State is the agent's memory. Without memory, there's no agent."

This guide covers state management from basics to production patterns: what state is, how to store it, how to manage it, how to keep it consistent, and how to build reliable stateful agents.

## What is State?

**State** is data that persists across agent interactions. It's everything the agent needs to remember between calls.

### State Components

```
┌─────────────────────────────────────────┐
│            AGENT STATE                  │
├─────────────────────────────────────────┤
│ CONVERSATION HISTORY                    │
│  - Messages exchanged                   │
│  - Context from previous turns          │
├─────────────────────────────────────────┤
│ TASK STATE                              │
│  - Current task                         │
│  - Progress through task                │
│  - Subtasks completed                   │
├─────────────────────────────────────────┤
│ USER CONTEXT                            │
│  - User preferences                     │
│  - User information                     │
│  - Interaction history                  │
├─────────────────────────────────────────┤
│ TOOL STATE                              │
│  - Tool outputs                         │
│  - Tool call history                    │
│  - Cached results                       │
├─────────────────────────────────────────┤
│ WORKING MEMORY                          │
│  - Temporary variables                  │
│  - Intermediate results                 │
│  - Current reasoning                    │
└─────────────────────────────────────────┘
```

### State Example

```python
agent_state = {
    # Conversation
    "messages": [
        {"role": "user", "content": "Hi, I'm Alice"},
        {"role": "assistant", "content": "Hello Alice!"}
    ],

    # User context
    "user": {
        "name": "Alice",
        "preferences": {
            "language": "English",
            "theme": "dark"
        }
    },

    # Current task
    "current_task": {
        "type": "research",
        "query": "Find papers on RAG",
        "status": "in_progress",
        "steps_completed": ["search"],
        "steps_remaining": ["analyze", "summarize"]
    },

    # Tool state
    "tool_cache": {
        "last_search": {
            "query": "RAG papers",
            "results": [...],
            "timestamp": 1234567890
        }
    },

    # Metadata
    "session_id": "abc123",
    "created_at": 1234567890,
    "updated_at": 1234567899
}
```

## Why State Matters

### Continuity

**Without state**:

```python
turn1:
user: "Search for papers on RAG"
agent: [searches and returns results]

turn2:
user: "Summarize the first one"
agent: "What should I summarize?"  # Lost context!
```

**With state**:

```python
turn1:
user: "Search for papers on RAG"
agent: [searches, stores results in state]

turn2:
user: "Summarize the first one"
agent: [retrieves results from state]
      "The first paper discusses..."  # Has context!
```

### Progress Tracking

```python
# Track progress through multi-step task
state = {
    "task": "analyze_dataset",
    "steps": [
        {"name": "load_data", "status": "completed"},
        {"name": "clean_data", "status": "completed"},
        {"name": "analyze", "status": "in_progress"},
        {"name": "visualize", "status": "pending"},
        {"name": "report", "status": "pending"}
    ]
}

# Agent knows where it is and what's next
```

### Personalization

```python
# Remember user preferences
state = {
    "user_id": "user123",
    "preferences": {
        "temperature_unit": "celsius",
        "date_format": "YYYY-MM-DD",
        "language": "English",
        "detail_level": "concise"
    }
}

# Agent adapts to user preferences
weather = get_weather(city="Tokyo", unit=state["preferences"]["temperature_unit"])
```

### Recovery

```python
# State enables recovery from interruptions
try:
    process_task(state)
except InterruptionError:
    # Save state
    save_state(state)

# Later, resume from saved state
state = load_state()
resume_task(state)
```

## Types of State

Different types of state serve different purposes.

### 1. Ephemeral State

**Characteristics**:

- Short-lived (single session)
- Stored in memory
- Lost when session ends
- Fast access

```python
class EphemeralState:
    """State that exists only for current session."""

    def __init__(self):
        self.data = {}

    def set(self, key: str, value: Any):
        """Set state value."""
        self.data[key] = value

    def get(self, key: str) -> Any:
        """Get state value."""
        return self.data.get(key)

    def clear(self):
        """Clear all state."""
        self.data = {}


# Usage
state = EphemeralState()
state.set("current_query", "weather in Tokyo")
# Lost when program ends
```

### 2. Session State

**Characteristics**:

- Persists during session
- Cleared between sessions
- User-specific
- Medium-term memory

```python
class SessionState:
    """State that persists during session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.data = {}
        self.created_at = time.time()
        self.expires_at = self.created_at + 3600  # 1 hour

    def is_expired(self) -> bool:
        """Check if session is expired."""
        return time.time() > self.expires_at

    def extend(self, duration: int = 3600):
        """Extend session expiration."""
        self.expires_at = time.time() + duration
```

### 3. Persistent State

**Characteristics**:

- Survives sessions
- Stored in database/disk
- Long-term memory
- Slower access

```python
class PersistentState:
    """State that persists across sessions."""

    def __init__(self, user_id: str, storage_path: str):
        self.user_id = user_id
        self.storage_path = storage_path
        self.data = self.load()

    def load(self) -> dict:
        """Load state from disk."""
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save(self):
        """Save state to disk."""
        with open(self.storage_path, 'w') as f:
            json.dump(self.data, f)

    def set(self, key: str, value: Any):
        """Set and save state."""
        self.data[key] = value
        self.save()
```

### 4. Shared State

**Characteristics**:

- Shared across multiple agents/components
- Requires synchronization
- Enables collaboration

```python
class SharedState:
    """State shared between multiple agents."""

    def __init__(self):
        self.data = {}
        self._lock = threading.Lock()

    def set(self, key: str, value: Any):
        """Thread-safe set."""
        with self._lock:
            self.data[key] = value

    def get(self, key: str) -> Any:
        """Thread-safe get."""
        with self._lock:
            return self.data.get(key)
```

## State Storage

Where and how to store state.

### In-Memory Storage

```python
class InMemoryStateStore:
    """Simple in-memory state storage."""

    def __init__(self):
        self.sessions = {}

    def get_session(self, session_id: str) -> dict:
        """Get session state."""
        if session_id not in self.sessions:
            self.sessions[session_id] = {}
        return self.sessions[session_id]

    def update_session(self, session_id: str, data: dict):
        """Update session state."""
        self.sessions[session_id] = data

    def delete_session(self, session_id: str):
        """Delete session state."""
        if session_id in self.sessions:
            del self.sessions[session_id]


# Usage
store = InMemoryStateStore()
state = store.get_session("session123")
state["user_name"] = "Alice"
store.update_session("session123", state)
```

### File-Based Storage

```python
class FileStateStore:
    """Store state in files."""

    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def save(self, session_id: str, state: dict):
        """Save state to file."""
        file_path = os.path.join(self.base_path, f"{session_id}.json")

        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2)

    def load(self, session_id: str) -> dict:
        """Load state from file."""
        file_path = os.path.join(self.base_path, f"{session_id}.json")

        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def delete(self, session_id: str):
        """Delete state file."""
        file_path = os.path.join(self.base_path, f"{session_id}.json")

        if os.path.exists(file_path):
            os.remove(file_path)
```

### Database Storage

```python
import sqlite3

class DatabaseStateStore:
    """Store state in database."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_state (
                session_id TEXT PRIMARY KEY,
                state_data TEXT,
                created_at REAL,
                updated_at REAL
            )
        """)

        conn.commit()
        conn.close()

    def save(self, session_id: str, state: dict):
        """Save state to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = time.time()
        state_json = json.dumps(state)

        cursor.execute("""
            INSERT OR REPLACE INTO agent_state
            (session_id, state_data, created_at, updated_at)
            VALUES (?, ?, COALESCE((SELECT created_at FROM agent_state WHERE session_id = ?), ?), ?)
        """, (session_id, state_json, session_id, now, now))

        conn.commit()
        conn.close()

    def load(self, session_id: str) -> dict:
        """Load state from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT state_data FROM agent_state WHERE session_id = ?",
            (session_id,)
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return json.loads(row[0])
        return {}
```

### Redis Storage (for production)

```python
import redis

class RedisStateStore:
    """Store state in Redis."""

    def __init__(self, host: str = 'localhost', port: int = 6379):
        self.redis = redis.Redis(host=host, port=port, decode_responses=True)

    def save(self, session_id: str, state: dict, ttl: int = 3600):
        """Save state with TTL."""
        key = f"agent_state:{session_id}"
        self.redis.setex(key, ttl, json.dumps(state))

    def load(self, session_id: str) -> dict:
        """Load state."""
        key = f"agent_state:{session_id}"
        data = self.redis.get(key)

        if data:
            return json.loads(data)
        return {}

    def delete(self, session_id: str):
        """Delete state."""
        key = f"agent_state:{session_id}"
        self.redis.delete(key)

    def extend_ttl(self, session_id: str, ttl: int = 3600):
        """Extend state expiration."""
        key = f"agent_state:{session_id}"
        self.redis.expire(key, ttl)
```

## State Management Patterns

Patterns for managing state effectively.

### State Manager

```python
class StateManager:
    """Centralized state management."""

    def __init__(self, storage):
        self.storage = storage
        self.current_session = None

    def start_session(self, session_id: str):
        """Start a new session."""
        self.current_session = session_id
        return self.storage.load(session_id)

    def get_state(self) -> dict:
        """Get current state."""
        if not self.current_session:
            raise ValueError("No active session")

        return self.storage.load(self.current_session)

    def update_state(self, updates: dict):
        """Update state."""
        state = self.get_state()
        state.update(updates)
        self.storage.save(self.current_session, state)

    def set_value(self, key: str, value: Any):
        """Set a single value."""
        state = self.get_state()
        state[key] = value
        self.storage.save(self.current_session, state)

    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a single value."""
        state = self.get_state()
        return state.get(key, default)

    def end_session(self):
        """End current session."""
        if self.current_session:
            self.storage.delete(self.current_session)
            self.current_session = None


# Usage
storage = FileStateStore("./state")
manager = StateManager(storage)

manager.start_session("session123")
manager.set_value("user_name", "Alice")
manager.update_state({"theme": "dark", "language": "en"})

name = manager.get_value("user_name")
print(f"Hello {name}")
```

### State Context Manager

```python
from contextlib import contextmanager

class StateContext:
    """Context manager for state."""

    def __init__(self, storage, session_id: str):
        self.storage = storage
        self.session_id = session_id
        self.state = None

    def __enter__(self):
        """Load state on entry."""
        self.state = self.storage.load(self.session_id)
        return self.state

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Save state on exit."""
        if exc_type is None:  # No exception
            self.storage.save(self.session_id, self.state)
        return False


# Usage
with StateContext(storage, "session123") as state:
    state["user_name"] = "Alice"
    state["count"] = state.get("count", 0) + 1
    # Automatically saved on exit
```

### State Versioning

```python
class VersionedState:
    """State with version history."""

    def __init__(self):
        self.versions = []
        self.current = {}

    def update(self, updates: dict):
        """Update state and save version."""
        # Save current version
        self.versions.append({
            "state": self.current.copy(),
            "timestamp": time.time()
        })

        # Apply updates
        self.current.update(updates)

    def revert(self, steps: int = 1) -> dict:
        """Revert to previous version."""
        if len(self.versions) < steps:
            raise ValueError("Not enough versions to revert")

        for _ in range(steps):
            version = self.versions.pop()

        self.current = version["state"]
        return self.current

    def get_version(self, index: int) -> dict:
        """Get specific version."""
        if index < len(self.versions):
            return self.versions[index]["state"]
        return self.current
```

## In-Memory State

Fast, temporary state for single sessions.

### Simple In-Memory State

```python
class InMemoryState:
    """Simple in-memory state for agent."""

    def __init__(self):
        self.conversation = []
        self.user_context = {}
        self.tool_outputs = []
        self.working_memory = {}

    def add_message(self, role: str, content: str):
        """Add message to conversation."""
        self.conversation.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })

    def add_tool_output(self, tool: str, output: Any):
        """Add tool output."""
        self.tool_outputs.append({
            "tool": tool,
            "output": output,
            "timestamp": time.time()
        })

    def set_user_info(self, key: str, value: Any):
        """Set user information."""
        self.user_context[key] = value

    def get_user_info(self, key: str) -> Any:
        """Get user information."""
        return self.user_context.get(key)

    def set_variable(self, key: str, value: Any):
        """Set working variable."""
        self.working_memory[key] = value

    def get_variable(self, key: str) -> Any:
        """Get working variable."""
        return self.working_memory.get(key)

    def get_full_state(self) -> dict:
        """Get complete state."""
        return {
            "conversation": self.conversation,
            "user_context": self.user_context,
            "tool_outputs": self.tool_outputs,
            "working_memory": self.working_memory
        }


# Usage
state = InMemoryState()
state.add_message("user", "Hi, I'm Alice")
state.set_user_info("name", "Alice")
state.add_message("assistant", "Hello Alice!")
state.add_tool_output("search", ["result1", "result2"])
state.set_variable("search_query", "AI agents")
```

### State with Expiration

```python
class ExpiringState:
    """State with automatic expiration."""

    def __init__(self, default_ttl: int = 3600):
        self.data = {}
        self.default_ttl = default_ttl

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value with expiration."""
        expires_at = time.time() + (ttl or self.default_ttl)
        self.data[key] = {
            "value": value,
            "expires_at": expires_at
        }

    def get(self, key: str) -> Optional[Any]:
        """Get value if not expired."""
        if key not in self.data:
            return None

        entry = self.data[key]

        # Check expiration
        if time.time() > entry["expires_at"]:
            del self.data[key]
            return None

        return entry["value"]

    def cleanup(self):
        """Remove expired entries."""
        now = time.time()
        expired = [
            key for key, entry in self.data.items()
            if now > entry["expires_at"]
        ]

        for key in expired:
            del self.data[key]
```

## Persistent State

State that survives across sessions.

### User Profile State

```python
class UserProfileState:
    """Persistent user profile state."""

    def __init__(self, user_id: str, storage):
        self.user_id = user_id
        self.storage = storage
        self.profile = self._load_profile()

    def _load_profile(self) -> dict:
        """Load user profile."""
        profile = self.storage.load(f"user_{self.user_id}")

        if not profile:
            # Initialize default profile
            profile = {
                "user_id": self.user_id,
                "created_at": time.time(),
                "preferences": {},
                "history": [],
                "metadata": {}
            }
            self._save_profile(profile)

        return profile

    def _save_profile(self, profile: dict):
        """Save user profile."""
        profile["updated_at"] = time.time()
        self.storage.save(f"user_{self.user_id}", profile)

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference."""
        return self.profile["preferences"].get(key, default)

    def set_preference(self, key: str, value: Any):
        """Set user preference."""
        self.profile["preferences"][key] = value
        self._save_profile(self.profile)

    def add_to_history(self, entry: dict):
        """Add to user history."""
        self.profile["history"].append({
            **entry,
            "timestamp": time.time()
        })

        # Keep last 100 entries
        if len(self.profile["history"]) > 100:
            self.profile["history"] = self.profile["history"][-100:]

        self._save_profile(self.profile)

    def get_history(self, limit: int = 10) -> list:
        """Get recent history."""
        return self.profile["history"][-limit:]
```

### Long-Term Memory

```python
class LongTermMemory:
    """Persistent long-term memory."""

    def __init__(self, storage):
        self.storage = storage
        self.memories = {}

    def store_memory(
        self,
        key: str,
        content: str,
        tags: list = None,
        importance: float = 1.0
    ):
        """Store a memory."""
        memory = {
            "content": content,
            "tags": tags or [],
            "importance": importance,
            "created_at": time.time(),
            "access_count": 0,
            "last_accessed": time.time()
        }

        self.memories[key] = memory
        self._persist()

    def retrieve_memory(self, key: str) -> Optional[dict]:
        """Retrieve a memory."""
        if key not in self.memories:
            return None

        memory = self.memories[key]
        memory["access_count"] += 1
        memory["last_accessed"] = time.time()

        self._persist()
        return memory

    def search_memories(
        self,
        query: str,
        limit: int = 5
    ) -> list:
        """Search memories by content/tags."""
        query_lower = query.lower()

        scored = []
        for key, memory in self.memories.items():
            score = 0

            # Content match
            if query_lower in memory["content"].lower():
                score += 2

            # Tag match
            for tag in memory["tags"]:
                if query_lower in tag.lower():
                    score += 1

            # Boost by importance
            score *= memory["importance"]

            # Boost by recency
            age = time.time() - memory["last_accessed"]
            recency_factor = 1.0 / (1.0 + age / 86400)  # Decay over days
            score *= (1.0 + recency_factor)

            if score > 0:
                scored.append((score, key, memory))

        # Sort by score
        scored.sort(reverse=True, key=lambda x: x[0])

        return [
            {"key": key, "memory": memory, "score": score}
            for score, key, memory in scored[:limit]
        ]

    def _persist(self):
        """Save memories to storage."""
        self.storage.save("long_term_memory", self.memories)
```

## State Transitions

Managing changes in state.

### State Machine

```python
from enum import Enum

class TaskState(Enum):
    """States for a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class StateMachine:
    """Simple state machine for tasks."""

    def __init__(self):
        self.current_state = TaskState.PENDING
        self.transitions = {
            TaskState.PENDING: [TaskState.IN_PROGRESS],
            TaskState.IN_PROGRESS: [TaskState.PAUSED, TaskState.COMPLETED, TaskState.FAILED],
            TaskState.PAUSED: [TaskState.IN_PROGRESS, TaskState.FAILED],
            TaskState.COMPLETED: [],
            TaskState.FAILED: []
        }
        self.history = []

    def transition_to(self, new_state: TaskState) -> bool:
        """Transition to new state."""
        # Check if transition is allowed
        if new_state not in self.transitions[self.current_state]:
            print(f"Invalid transition: {self.current_state} -> {new_state}")
            return False

        # Record transition
        self.history.append({
            "from": self.current_state,
            "to": new_state,
            "timestamp": time.time()
        })

        # Update state
        self.current_state = new_state
        return True

    def can_transition_to(self, new_state: TaskState) -> bool:
        """Check if transition is possible."""
        return new_state in self.transitions[self.current_state]


# Usage
sm = StateMachine()
print(sm.current_state)  # PENDING

sm.transition_to(TaskState.IN_PROGRESS)
print(sm.current_state)  # IN_PROGRESS

sm.transition_to(TaskState.COMPLETED)
print(sm.current_state)  # COMPLETED

# This would fail
sm.transition_to(TaskState.IN_PROGRESS)  # Can't go back from COMPLETED
```

### Event-Driven State Updates

```python
class EventDrivenState:
    """State that updates based on events."""

    def __init__(self):
        self.state = {}
        self.handlers = {}

    def register_handler(self, event_type: str, handler: callable):
        """Register event handler."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    def emit_event(self, event_type: str, data: dict):
        """Emit event and trigger handlers."""
        if event_type in self.handlers:
            for handler in self.handlers[event_type]:
                handler(self.state, data)

    def get_state(self) -> dict:
        """Get current state."""
        return self.state.copy()


# Usage
state = EventDrivenState()

# Register handlers
def on_user_message(state, data):
    state["last_message"] = data["content"]
    state["message_count"] = state.get("message_count", 0) + 1

def on_tool_call(state, data):
    state["last_tool"] = data["tool_name"]
    state["tool_count"] = state.get("tool_count", 0) + 1

state.register_handler("user_message", on_user_message)
state.register_handler("tool_call", on_tool_call)

# Emit events
state.emit_event("user_message", {"content": "Hello"})
state.emit_event("tool_call", {"tool_name": "search"})

print(state.get_state())
# {"last_message": "Hello", "message_count": 1, "last_tool": "search", "tool_count": 1}
```

## State Consistency

Ensuring state remains consistent.

### Transactional State Updates

```python
class TransactionalState:
    """State with transactional updates."""

    def __init__(self, storage):
        self.storage = storage
        self.state = {}
        self.transaction_stack = []

    def begin_transaction(self):
        """Start a transaction."""
        # Save current state
        self.transaction_stack.append(self.state.copy())

    def commit(self):
        """Commit transaction."""
        if not self.transaction_stack:
            raise ValueError("No active transaction")

        # Remove saved state (transaction successful)
        self.transaction_stack.pop()

        # Persist to storage
        self.storage.save("state", self.state)

    def rollback(self):
        """Rollback transaction."""
        if not self.transaction_stack:
            raise ValueError("No active transaction")

        # Restore saved state
        self.state = self.transaction_stack.pop()

    def update(self, key: str, value: Any):
        """Update state within transaction."""
        self.state[key] = value


# Usage
state = TransactionalState(storage)

state.begin_transaction()
try:
    state.update("step1", "completed")
    state.update("step2", "completed")
    # Some operation that might fail
    risky_operation()
    state.commit()
except Exception:
    state.rollback()
    print("Transaction rolled back")
```

### State Validation

```python
class ValidatedState:
    """State with validation rules."""

    def __init__(self):
        self.state = {}
        self.validators = {}

    def register_validator(self, key: str, validator: callable):
        """Register validator for key."""
        self.validators[key] = validator

    def set(self, key: str, value: Any):
        """Set value with validation."""
        # Validate if validator exists
        if key in self.validators:
            if not self.validators[key](value):
                raise ValueError(f"Validation failed for {key}")

        self.state[key] = value

    def get(self, key: str) -> Any:
        """Get value."""
        return self.state.get(key)


# Usage
state = ValidatedState()

# Register validators
state.register_validator("age", lambda x: isinstance(x, int) and x >= 0)
state.register_validator("email", lambda x: "@" in str(x))

state.set("age", 30)  # OK
# state.set("age", -5)  # Would raise ValueError
state.set("email", "user@example.com")  # OK
# state.set("email", "invalid")  # Would raise ValueError
```

## Conversation State

Managing conversation-specific state.

### Conversation Manager

```python
class ConversationState:
    """Manage conversation state."""

    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.messages = []
        self.context = {}
        self.metadata = {
            "started_at": time.time(),
            "turn_count": 0
        }

    def add_turn(self, user_message: str, agent_response: str):
        """Add conversation turn."""
        turn = {
            "turn_number": self.metadata["turn_count"],
            "user_message": user_message,
            "agent_response": agent_response,
            "timestamp": time.time()
        }

        self.messages.append(turn)
        self.metadata["turn_count"] += 1

    def get_recent_context(self, n_turns: int = 5) -> str:
        """Get recent conversation context."""
        recent = self.messages[-n_turns:]

        context = []
        for turn in recent:
            context.append(f"User: {turn['user_message']}")
            context.append(f"Assistant: {turn['agent_response']}")

        return "\n".join(context)

    def set_context(self, key: str, value: Any):
        """Set conversation context."""
        self.context[key] = value

    def get_context(self, key: str) -> Any:
        """Get conversation context."""
        return self.context.get(key)

    def get_full_state(self) -> dict:
        """Get complete conversation state."""
        return {
            "conversation_id": self.conversation_id,
            "messages": self.messages,
            "context": self.context,
            "metadata": self.metadata
        }
```

## Task State

Managing task execution state.

### Task State Tracker

```python
class TaskStateTracker:
    """Track state of multi-step tasks."""

    def __init__(self, task_id: str, steps: list):
        self.task_id = task_id
        self.steps = steps
        self.current_step = 0
        self.step_results = {}
        self.status = "pending"
        self.started_at = None
        self.completed_at = None

    def start(self):
        """Start task execution."""
        self.status = "in_progress"
        self.started_at = time.time()

    def complete_step(self, result: Any):
        """Complete current step."""
        step_name = self.steps[self.current_step]
        self.step_results[step_name] = result
        self.current_step += 1

        # Check if all steps completed
        if self.current_step >= len(self.steps):
            self.complete()

    def complete(self):
        """Mark task as completed."""
        self.status = "completed"
        self.completed_at = time.time()

    def fail(self, error: str):
        """Mark task as failed."""
        self.status = "failed"
        self.step_results["error"] = error
        self.completed_at = time.time()

    def get_progress(self) -> dict:
        """Get task progress."""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "current_step": self.current_step,
            "total_steps": len(self.steps),
            "progress": self.current_step / len(self.steps),
            "completed_steps": list(self.step_results.keys()),
            "remaining_steps": self.steps[self.current_step:]
        }

    def get_state(self) -> dict:
        """Get complete task state."""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "steps": self.steps,
            "current_step": self.current_step,
            "step_results": self.step_results,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": (
                (self.completed_at or time.time()) - self.started_at
                if self.started_at else None
            )
        }


# Usage
tracker = TaskStateTracker(
    "task123",
    ["search", "analyze", "summarize", "report"]
)

tracker.start()
tracker.complete_step(["result1", "result2"])  # search
tracker.complete_step({"key_findings": [...}])  # analyze
print(tracker.get_progress())
```

## Session State

Managing session-level state.

### Session Manager

```python
class SessionManager:
    """Manage session state."""

    def __init__(self, storage, session_timeout: int = 3600):
        self.storage = storage
        self.session_timeout = session_timeout
        self.active_sessions = {}

    def create_session(self, user_id: str) -> str:
        """Create new session."""
        session_id = self._generate_session_id()

        session = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": time.time(),
            "last_activity": time.time(),
            "data": {}
        }

        self.active_sessions[session_id] = session
        self.storage.save(session_id, session)

        return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get session if active."""
        # Check in-memory first
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
        else:
            # Load from storage
            session = self.storage.load(session_id)
            if session:
                self.active_sessions[session_id] = session

        if not session:
            return None

        # Check expiration
        if self._is_expired(session):
            self.end_session(session_id)
            return None

        # Update activity
        session["last_activity"] = time.time()

        return session

    def update_session(self, session_id: str, data: dict):
        """Update session data."""
        session = self.get_session(session_id)

        if not session:
            raise ValueError("Session not found or expired")

        session["data"].update(data)
        session["last_activity"] = time.time()

        self.storage.save(session_id, session)

    def end_session(self, session_id: str):
        """End session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

        self.storage.delete(session_id)

    def _is_expired(self, session: dict) -> bool:
        """Check if session is expired."""
        age = time.time() - session["last_activity"]
        return age > self.session_timeout

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        import uuid
        return str(uuid.uuid4())

    def cleanup_expired_sessions(self):
        """Remove expired sessions."""
        expired = [
            sid for sid, session in self.active_sessions.items()
            if self._is_expired(session)
        ]

        for session_id in expired:
            self.end_session(session_id)
```

## State Serialization

Converting state to/from storage formats.

### JSON Serialization

```python
class StateSerializer:
    """Serialize state to JSON."""

    def serialize(self, state: dict) -> str:
        """Convert state to JSON string."""
        # Handle special types
        serializable = self._make_serializable(state)
        return json.dumps(serializable, indent=2)

    def deserialize(self, data: str) -> dict:
        """Convert JSON string to state."""
        return json.loads(data)

    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return str(obj)


# Usage
serializer = StateSerializer()

state = {
    "user": "Alice",
    "count": 5,
    "items": ["a", "b", "c"],
    "metadata": {"created": time.time()}
}

# Serialize
json_str = serializer.serialize(state)

# Deserialize
loaded_state = serializer.deserialize(json_str)
```

### Pickle Serialization

```python
import pickle

class PickleStateSerializer:
    """Serialize state using pickle (supports more types)."""

    def serialize(self, state: Any) -> bytes:
        """Convert state to bytes."""
        return pickle.dumps(state)

    def deserialize(self, data: bytes) -> Any:
        """Convert bytes to state."""
        return pickle.loads(data)

    def save_to_file(self, state: Any, filepath: str):
        """Save state to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_from_file(self, filepath: str) -> Any:
        """Load state from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
```

## State Recovery

Recovering from interruptions or failures.

### Checkpoint System

```python
class StateCheckpoint:
    """Checkpoint system for state recovery."""

    def __init__(self, storage):
        self.storage = storage
        self.checkpoints = []

    def create_checkpoint(self, state: dict, label: str = ""):
        """Create state checkpoint."""
        checkpoint = {
            "label": label,
            "state": state.copy(),
            "timestamp": time.time()
        }

        self.checkpoints.append(checkpoint)

        # Save to storage
        checkpoint_id = f"checkpoint_{len(self.checkpoints)}"
        self.storage.save(checkpoint_id, checkpoint)

    def restore_checkpoint(self, index: int = -1) -> dict:
        """Restore from checkpoint."""
        if not self.checkpoints:
            raise ValueError("No checkpoints available")

        checkpoint = self.checkpoints[index]
        return checkpoint["state"].copy()

    def list_checkpoints(self) -> list:
        """List all checkpoints."""
        return [
            {
                "index": i,
                "label": cp["label"],
                "timestamp": cp["timestamp"]
            }
            for i, cp in enumerate(self.checkpoints)
        ]


# Usage
checkpoint_system = StateCheckpoint(storage)

# Create checkpoint before risky operation
checkpoint_system.create_checkpoint(state, "before_search")

try:
    # Risky operation
    perform_search()
    update_state()
except Exception:
    # Restore from checkpoint
    state = checkpoint_system.restore_checkpoint()
```

### Auto-Save State

```python
class AutoSaveState:
    """State with automatic saving."""

    def __init__(self, storage, auto_save_interval: int = 60):
        self.storage = storage
        self.state = {}
        self.auto_save_interval = auto_save_interval
        self.last_save = time.time()
        self.modified = False

    def set(self, key: str, value: Any):
        """Set value and mark as modified."""
        self.state[key] = value
        self.modified = True
        self._maybe_auto_save()

    def update(self, updates: dict):
        """Update multiple values."""
        self.state.update(updates)
        self.modified = True
        self._maybe_auto_save()

    def _maybe_auto_save(self):
        """Auto-save if interval elapsed."""
        if not self.modified:
            return

        if time.time() - self.last_save > self.auto_save_interval:
            self.save()

    def save(self):
        """Force save."""
        if self.modified:
            self.storage.save("state", self.state)
            self.last_save = time.time()
            self.modified = False
```

## State Best Practices

Guidelines for effective state management.

### 1. Keep State Minimal

```python
# ❌ BAD: Store too much
state = {
    "entire_conversation_history": [...],  # 1000 messages
    "all_tool_outputs": [...],  # Everything ever
    "full_document_cache": {...}  # Huge
}

# ✅ GOOD: Store what's needed
state = {
    "recent_messages": [...],  # Last 10
    "relevant_tool_outputs": [...],  # Current task only
    "document_references": [...]  # IDs, not full docs
}
```

### 2. Separate Concerns

```python
class WellOrganizedState:
    """State with clear separation."""

    def __init__(self):
        # Conversation state
        self.conversation = ConversationState()

        # Task state
        self.task = TaskStateTracker()

        # User state
        self.user = UserProfileState()

        # Working memory (ephemeral)
        self.working = {}
```

### 3. Validate State

```python
def validate_state(state: dict) -> bool:
    """Validate state structure."""
    required_keys = ["session_id", "created_at"]

    for key in required_keys:
        if key not in state:
            return False

    # Type checking
    if not isinstance(state.get("created_at"), (int, float)):
        return False

    return True
```

### 4. Version State Schema

```python
STATE_SCHEMA_VERSION = "1.0"

def create_state() -> dict:
    """Create state with version."""
    return {
        "_schema_version": STATE_SCHEMA_VERSION,
        "data": {}
    }

def migrate_state(state: dict) -> dict:
    """Migrate state to current version."""
    version = state.get("_schema_version", "0.0")

    if version == "0.0":
        # Migrate from v0 to v1
        state = migrate_v0_to_v1(state)

    return state
```

### 5. Clean Up Old State

```python
def cleanup_old_state(storage, max_age: int = 86400):
    """Remove state older than max_age."""
    # Implementation depends on storage backend
    pass
```

### 6. Monitor State Size

```python
def check_state_size(state: dict):
    """Monitor state size."""
    state_json = json.dumps(state)
    size_bytes = len(state_json.encode('utf-8'))

    if size_bytes > 1_000_000:  # 1MB
        print(f"WARNING: State is large: {size_bytes / 1000:.1f}KB")
```

## Summary

State management is how agents remember. Key principles:

**State Types**:

- Ephemeral (session only)
- Session (current session)
- Persistent (across sessions)
- Shared (between components)

**Storage**:

- In-memory (fast, temporary)
- File-based (simple, persistent)
- Database (scalable, queryable)
- Redis (fast, distributed)

**Patterns**:

- State managers for centralization
- Context managers for automatic save
- Versioning for history
- State machines for transitions
- Transactions for consistency
- Checkpoints for recovery

**Best Practices**:

- Keep state minimal
- Separate concerns
- Validate state structure
- Version schemas
- Clean up old state
- Monitor state size
- Auto-save periodically

State is memory. Without memory, agents can't function. Manage it well.

## Next Steps

Now that you understand state management, explore:

- **[Simple Agent Loops](simple-agent-loops.md)** - Maintaining state through loops
- **[Context Management](context-management.md)** - State in context
- **[Basic Error Handling](basic-error-handling.md)** - State recovery on errors
- **Memory Systems** - Advanced long-term memory
- **Multi-Agent State** - Shared state patterns

**Practice exercises**:

1. Implement a complete state management system
2. Build a conversation state manager with history
3. Create a task state tracker for multi-step workflows
4. Implement checkpoint-based recovery
5. Build a session manager with expiration

**Advanced topics**:

- Distributed state management
- State synchronization
- Conflict resolution
- State streaming
- State compression

State is the foundation of agent memory. Build it right.
