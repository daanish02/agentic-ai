# State Management

## Table of Contents

- [Introduction](#introduction)
- [Types of State](#types-of-state)
- [Agent State](#agent-state)
- [Conversation State](#conversation-state)
- [Workflow State](#workflow-state)
- [State Persistence](#state-persistence)
- [State Transitions](#state-transitions)
- [Maintaining Context](#maintaining-context)
- [Stateful Interactions](#stateful-interactions)
- [State Synchronization](#state-synchronization)
- [State Management Patterns](#state-management-patterns)
- [State Recovery](#state-recovery)
- [Performance Optimization](#performance-optimization)
- [Real-World Scenarios](#real-world-scenarios)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

State management is the backbone of sophisticated agent systems. While language models are inherently stateless, agents must maintain state to:

- **Remember** past interactions and decisions
- **Track progress** through multi-step workflows
- **Maintain context** across tool calls and reasoning steps
- **Persist information** between sessions
- **Coordinate** distributed operations

> "State is what transforms a stateless model into a stateful agent."

Effective state management enables agents to be coherent, consistent, and capable of handling complex, long-running tasks that span multiple interactions.

### Why State Matters

**Without state**:

```python
# Each call is isolated
agent.run("Start task A")  # Executes
agent.run("Continue task A")  # ❌ Doesn't know about previous call
```

**With state**:

```python
# State persists across calls
agent.run("Start task A")  # state["current_task"] = "A", status = "started"
agent.run("Continue task A")  # Retrieves state, continues from where it left off
```

### State Categories

```
┌─────────────────────────────────────────┐
│           Agent State                    │
│  - Identity and role                    │
│  - Available tools                       │
│  - Configuration                         │
├─────────────────────────────────────────┤
│        Conversation State                │
│  - Message history                       │
│  - User preferences                      │
│  - Session context                       │
├─────────────────────────────────────────┤
│         Workflow State                   │
│  - Current step                          │
│  - Completed steps                       │
│  - Intermediate results                  │
│  - Progress tracking                     │
└─────────────────────────────────────────┘
```

## Types of State

Different types of state serve different purposes in agent systems.

### Ephemeral State

**Characteristics**:
- Exists only during execution
- In-memory storage
- Lost when process ends
- Fast access

```python
class EphemeralState:
    """Temporary state for single execution"""
    
    def __init__(self):
        self.variables = {}
        self.scratch_space = {}
        self.temp_results = []
    
    def set(self, key: str, value: Any):
        """Set temporary variable"""
        self.variables[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get temporary variable"""
        return self.variables.get(key, default)
    
    def clear(self):
        """Clear all temporary state"""
        self.variables.clear()
        self.scratch_space.clear()
        self.temp_results.clear()

# Usage
state = EphemeralState()
state.set("current_step", 3)
state.set("intermediate_result", {"count": 42})

# State is lost when object is destroyed
```

### Session State

**Characteristics**:
- Persists for duration of session
- Stored in memory or cache
- Cleared when session ends
- Medium-term persistence

```python
from typing import Optional
from datetime import datetime, timedelta
import uuid

class SessionState:
    """State for a user session"""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.data = {}
        self.ttl = timedelta(hours=24)
    
    def set(self, key: str, value: Any):
        """Set session data"""
        self.data[key] = value
        self.last_accessed = datetime.now()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get session data"""
        self.last_accessed = datetime.now()
        return self.data.get(key, default)
    
    def is_expired(self) -> bool:
        """Check if session expired"""
        age = datetime.now() - self.last_accessed
        return age > self.ttl
    
    def to_dict(self) -> dict:
        """Serialize session"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "data": self.data
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SessionState":
        """Deserialize session"""
        session = cls(data["session_id"])
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.last_accessed = datetime.fromisoformat(data["last_accessed"])
        session.data = data["data"]
        return session

# Usage
session = SessionState()
session.set("user_name", "Alice")
session.set("preferences", {"theme": "dark"})

# Later in the session
name = session.get("user_name")  # "Alice"
```

### Persistent State

**Characteristics**:
- Survives process restarts
- Stored in database/files
- Long-term retention
- Slower access but durable

```python
import json
import sqlite3
from pathlib import Path

class PersistentState:
    """Durable state storage"""
    
    def __init__(self, db_path: str = "agent_state.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    
    def set(self, key: str, value: Any):
        """Set persistent value"""
        conn = sqlite3.connect(self.db_path)
        serialized = json.dumps(value)
        conn.execute(
            "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
            (key, serialized)
        )
        conn.commit()
        conn.close()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get persistent value"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT value FROM state WHERE key = ?",
            (key,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return json.loads(row[0])
        return default
    
    def delete(self, key: str):
        """Delete value"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM state WHERE key = ?", (key,))
        conn.commit()
        conn.close()
    
    def clear(self):
        """Clear all state"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM state")
        conn.commit()
        conn.close()

# Usage
state = PersistentState()
state.set("user_profile", {
    "name": "Alice",
    "history": ["task1", "task2"]
})

# Later, even after restart
profile = state.get("user_profile")  # Retrieved from database
```

## Agent State

State specific to the agent itself: identity, capabilities, and configuration.

### Agent Identity

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class AgentIdentity:
    """Agent identity and capabilities"""
    
    agent_id: str
    name: str
    role: str
    capabilities: List[str]
    tools: List[str]
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def can_use_tool(self, tool_name: str) -> bool:
        """Check if agent can use tool"""
        return tool_name in self.tools
    
    def has_capability(self, capability: str) -> bool:
        """Check if agent has capability"""
        return capability in self.capabilities

# Example
agent_identity = AgentIdentity(
    agent_id="agent_001",
    name="ResearchAssistant",
    role="research_and_analysis",
    capabilities=[
        "web_search",
        "document_analysis",
        "data_extraction",
        "summarization"
    ],
    tools=[
        "search_papers",
        "read_pdf",
        "extract_citations",
        "generate_summary"
    ],
    config={
        "max_papers": 20,
        "summary_length": "medium"
    }
)

if agent_identity.can_use_tool("search_papers"):
    # Use tool
    pass
```

### Agent Configuration

```python
class AgentConfig:
    """Mutable agent configuration"""
    
    def __init__(self):
        self.settings = {
            "temperature": 0.7,
            "max_tokens": 2000,
            "timeout": 30,
            "retry_attempts": 3
        }
        self.flags = set()
    
    def update_setting(self, key: str, value: Any):
        """Update configuration setting"""
        self.settings[key] = value
    
    def enable_flag(self, flag: str):
        """Enable feature flag"""
        self.flags.add(flag)
    
    def disable_flag(self, flag: str):
        """Disable feature flag"""
        self.flags.discard(flag)
    
    def is_enabled(self, flag: str) -> bool:
        """Check if flag is enabled"""
        return flag in self.flags
    
    def to_dict(self) -> dict:
        """Serialize config"""
        return {
            "settings": self.settings,
            "flags": list(self.flags)
        }

# Usage
config = AgentConfig()
config.update_setting("temperature", 0.9)
config.enable_flag("verbose_logging")

if config.is_enabled("verbose_logging"):
    print("Detailed logs enabled")
```

### Complete Agent State

```python
from datetime import datetime

class AgentState:
    """Complete agent state management"""
    
    def __init__(self, identity: AgentIdentity):
        self.identity = identity
        self.config = AgentConfig()
        self.status = "initialized"
        self.current_task = None
        self.metrics = {
            "requests_handled": 0,
            "tools_called": 0,
            "errors": 0
        }
        self.created_at = datetime.now()
        self.last_active = datetime.now()
    
    def start_task(self, task_description: str):
        """Begin new task"""
        self.current_task = {
            "description": task_description,
            "started_at": datetime.now(),
            "status": "in_progress"
        }
        self.status = "busy"
        self.last_active = datetime.now()
    
    def complete_task(self):
        """Mark current task complete"""
        if self.current_task:
            self.current_task["status"] = "completed"
            self.current_task["completed_at"] = datetime.now()
        self.status = "idle"
        self.metrics["requests_handled"] += 1
        self.last_active = datetime.now()
    
    def record_tool_call(self, tool_name: str):
        """Record tool usage"""
        self.metrics["tools_called"] += 1
        self.last_active = datetime.now()
    
    def record_error(self, error: Exception):
        """Record error"""
        self.metrics["errors"] += 1
        self.last_active = datetime.now()
    
    def get_status_summary(self) -> dict:
        """Get status summary"""
        return {
            "agent_id": self.identity.agent_id,
            "name": self.identity.name,
            "status": self.status,
            "current_task": self.current_task,
            "metrics": self.metrics,
            "uptime": (datetime.now() - self.created_at).total_seconds()
        }

# Usage
identity = AgentIdentity(
    agent_id="agent_001",
    name="Assistant",
    role="general",
    capabilities=["search", "analyze"],
    tools=["web_search", "calculator"]
)

agent_state = AgentState(identity)
agent_state.start_task("Research AI safety")
agent_state.record_tool_call("web_search")
agent_state.complete_task()

print(agent_state.get_status_summary())
```

## Conversation State

Managing the history and context of conversations.

### Message History

```python
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
from enum import Enum

class Role(Enum):
    """Message roles"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

@dataclass
class Message:
    """Single conversation message"""
    role: Role
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Serialize message"""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

class ConversationHistory:
    """Manage conversation history"""
    
    def __init__(self, max_messages: int = 100):
        self.messages: List[Message] = []
        self.max_messages = max_messages
    
    def add_message(self, role: Role, content: str, **metadata):
        """Add message to history"""
        message = Message(
            role=role,
            content=content,
            metadata=metadata
        )
        self.messages.append(message)
        
        # Trim if over limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_recent(self, n: int = 10) -> List[Message]:
        """Get n most recent messages"""
        return self.messages[-n:]
    
    def get_by_role(self, role: Role) -> List[Message]:
        """Get all messages from role"""
        return [m for m in self.messages if m.role == role]
    
    def search(self, query: str) -> List[Message]:
        """Search message content"""
        query_lower = query.lower()
        return [
            m for m in self.messages 
            if query_lower in m.content.lower()
        ]
    
    def to_openai_format(self) -> List[dict]:
        """Convert to OpenAI API format"""
        return [
            {"role": m.role.value, "content": m.content}
            for m in self.messages
        ]
    
    def summarize(self, max_length: int = 1000) -> str:
        """Create summary of conversation"""
        if not self.messages:
            return "No messages"
        
        summary_parts = []
        for msg in self.messages[-10:]:  # Last 10 messages
            preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            summary_parts.append(f"{msg.role.value}: {preview}")
        
        return "\n".join(summary_parts)

# Usage
history = ConversationHistory()
history.add_message(Role.SYSTEM, "You are a helpful assistant")
history.add_message(Role.USER, "What's the weather?")
history.add_message(Role.ASSISTANT, "I'll check the weather for you")

recent = history.get_recent(5)
user_messages = history.get_by_role(Role.USER)
```

### Context Window Management

```python
class ContextWindowManager:
    """Manage limited context window"""
    
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.history = ConversationHistory()
        self.system_prompt = ""
        self.pinned_messages = []
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation"""
        # Simple approximation: ~4 chars per token
        return len(text) // 4
    
    def set_system_prompt(self, prompt: str):
        """Set system prompt (always included)"""
        self.system_prompt = prompt
    
    def pin_message(self, message: Message):
        """Pin important message (always included)"""
        self.pinned_messages.append(message)
    
    def get_context(self) -> List[Message]:
        """Get messages that fit in context window"""
        # Start with system prompt and pinned messages
        messages = []
        token_count = 0
        
        if self.system_prompt:
            messages.append(Message(Role.SYSTEM, self.system_prompt))
            token_count += self.estimate_tokens(self.system_prompt)
        
        for msg in self.pinned_messages:
            messages.append(msg)
            token_count += self.estimate_tokens(msg.content)
        
        # Add recent messages until we hit limit
        for msg in reversed(self.history.messages):
            msg_tokens = self.estimate_tokens(msg.content)
            if token_count + msg_tokens > self.max_tokens:
                break
            messages.insert(len(messages) - len(self.pinned_messages), msg)
            token_count += msg_tokens
        
        return messages
    
    def add_message(self, role: Role, content: str):
        """Add message with context management"""
        self.history.add_message(role, content)
        
        # Check if we need to summarize old messages
        context = self.get_context()
        if len(context) < len(self.history.messages) // 2:
            self._compress_history()
    
    def _compress_history(self):
        """Compress old messages into summary"""
        # Get messages to compress (old ones not in current context)
        context = self.get_context()
        to_compress = self.history.messages[:-len(context)]
        
        if to_compress:
            # Create summary
            summary_text = f"Previous conversation summary: {len(to_compress)} messages about [topics]"
            summary = Message(Role.SYSTEM, summary_text)
            
            # Replace compressed messages with summary
            self.pinned_messages.insert(0, summary)
            
            # Keep only recent messages
            self.history.messages = self.history.messages[-len(context):]

# Usage
context_mgr = ContextWindowManager(max_tokens=4000)
context_mgr.set_system_prompt("You are a helpful assistant")

context_mgr.add_message(Role.USER, "Tell me about Python")
context_mgr.add_message(Role.ASSISTANT, "Python is a programming language...")

# Get messages that fit in context
current_context = context_mgr.get_context()
```

## Workflow State

State for tracking multi-step workflows and task progress.

### Workflow State Model

```python
from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class StepState:
    """State of a single workflow step"""
    step_id: str
    name: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    input: Optional[Any] = None
    output: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def start(self, input_data: Any):
        """Mark step as started"""
        self.status = WorkflowStatus.RUNNING
        self.input = input_data
        self.started_at = datetime.now()
    
    def complete(self, output_data: Any):
        """Mark step as completed"""
        self.status = WorkflowStatus.COMPLETED
        self.output = output_data
        self.completed_at = datetime.now()
    
    def fail(self, error: str):
        """Mark step as failed"""
        self.status = WorkflowStatus.FAILED
        self.error = error
        self.completed_at = datetime.now()
    
    def duration(self) -> Optional[float]:
        """Get step duration in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

@dataclass
class WorkflowState:
    """Complete workflow state"""
    workflow_id: str
    name: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_step: Optional[str] = None
    steps: Dict[str, StepState] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_step(self, step_id: str, name: str) -> StepState:
        """Add step to workflow"""
        step = StepState(step_id=step_id, name=name)
        self.steps[step_id] = step
        self.updated_at = datetime.now()
        return step
    
    def get_step(self, step_id: str) -> Optional[StepState]:
        """Get step by ID"""
        return self.steps.get(step_id)
    
    def set_variable(self, key: str, value: Any):
        """Set workflow variable"""
        self.variables[key] = value
        self.updated_at = datetime.now()
    
    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get workflow variable"""
        return self.variables.get(key, default)
    
    def get_completed_steps(self) -> List[StepState]:
        """Get all completed steps"""
        return [
            step for step in self.steps.values()
            if step.status == WorkflowStatus.COMPLETED
        ]
    
    def get_progress(self) -> float:
        """Get workflow progress (0.0 to 1.0)"""
        if not self.steps:
            return 0.0
        completed = len(self.get_completed_steps())
        return completed / len(self.steps)
    
    def to_dict(self) -> dict:
        """Serialize workflow state"""
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "status": self.status.value,
            "current_step": self.current_step,
            "steps": {
                sid: {
                    "name": step.name,
                    "status": step.status.value,
                    "duration": step.duration()
                }
                for sid, step in self.steps.items()
            },
            "variables": self.variables,
            "progress": self.get_progress(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

# Usage
workflow = WorkflowState(
    workflow_id="wf_001",
    name="Data Processing Pipeline"
)

# Add steps
extract_step = workflow.add_step("extract", "Extract Data")
transform_step = workflow.add_step("transform", "Transform Data")
load_step = workflow.add_step("load", "Load Data")

# Execute step
extract_step.start({"source": "s3://bucket/data"})
# ... do work ...
extract_step.complete({"rows": 1000, "columns": 50})

# Set workflow variables
workflow.set_variable("processed_rows", 1000)
workflow.set_variable("start_time", datetime.now())

# Check progress
print(f"Progress: {workflow.get_progress() * 100:.1f}%")
print(f"State: {workflow.to_dict()}")
```

### Workflow State Manager

```python
class WorkflowStateManager:
    """Manage workflow state persistence and transitions"""
    
    def __init__(self, storage: PersistentState):
        self.storage = storage
        self.active_workflows: Dict[str, WorkflowState] = {}
    
    def create_workflow(self, name: str) -> WorkflowState:
        """Create new workflow"""
        workflow_id = f"wf_{uuid.uuid4().hex[:8]}"
        workflow = WorkflowState(workflow_id=workflow_id, name=name)
        self.active_workflows[workflow_id] = workflow
        self._save_workflow(workflow)
        return workflow
    
    def get_workflow(self, workflow_id: str) -> Optional[WorkflowState]:
        """Get workflow by ID"""
        # Check active workflows first
        if workflow_id in self.active_workflows:
            return self.active_workflows[workflow_id]
        
        # Load from storage
        data = self.storage.get(f"workflow:{workflow_id}")
        if data:
            workflow = self._deserialize_workflow(data)
            self.active_workflows[workflow_id] = workflow
            return workflow
        
        return None
    
    def update_workflow(self, workflow: WorkflowState):
        """Update workflow state"""
        workflow.updated_at = datetime.now()
        self.active_workflows[workflow.workflow_id] = workflow
        self._save_workflow(workflow)
    
    def delete_workflow(self, workflow_id: str):
        """Delete workflow"""
        self.active_workflows.pop(workflow_id, None)
        self.storage.delete(f"workflow:{workflow_id}")
    
    def list_active_workflows(self) -> List[WorkflowState]:
        """List all active workflows"""
        return list(self.active_workflows.values())
    
    def _save_workflow(self, workflow: WorkflowState):
        """Save workflow to storage"""
        self.storage.set(
            f"workflow:{workflow.workflow_id}",
            workflow.to_dict()
        )
    
    def _deserialize_workflow(self, data: dict) -> WorkflowState:
        """Deserialize workflow from storage"""
        # Simplified deserialization
        workflow = WorkflowState(
            workflow_id=data["workflow_id"],
            name=data["name"]
        )
        workflow.status = WorkflowStatus(data["status"])
        workflow.variables = data["variables"]
        return workflow

# Usage
storage = PersistentState()
manager = WorkflowStateManager(storage)

# Create workflow
workflow = manager.create_workflow("ETL Pipeline")
workflow.add_step("extract", "Extract")
workflow.add_step("transform", "Transform")
workflow.add_step("load", "Load")

# Update workflow
manager.update_workflow(workflow)

# Later, retrieve workflow
retrieved = manager.get_workflow(workflow.workflow_id)
```

## State Persistence

Saving and loading state across sessions and restarts.

### File-Based Persistence

```python
import json
from pathlib import Path
from typing import Any, Dict

class FileStateStore:
    """File-based state persistence"""
    
    def __init__(self, base_dir: str = ".agent_state"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def save(self, key: str, value: Any):
        """Save state to file"""
        file_path = self.base_dir / f"{key}.json"
        with open(file_path, 'w') as f:
            json.dump(value, f, indent=2, default=str)
    
    def load(self, key: str) -> Optional[Any]:
        """Load state from file"""
        file_path = self.base_dir / f"{key}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return None
    
    def delete(self, key: str):
        """Delete state file"""
        file_path = self.base_dir / f"{key}.json"
        if file_path.exists():
            file_path.unlink()
    
    def list_keys(self) -> List[str]:
        """List all stored keys"""
        return [f.stem for f in self.base_dir.glob("*.json")]

# Usage
store = FileStateStore()

# Save state
store.save("agent_config", {
    "temperature": 0.7,
    "model": "gpt-4"
})

# Load state
config = store.load("agent_config")
```

### Redis-Based Persistence

```python
import redis
import json
from typing import Any, Optional

class RedisStateStore:
    """Redis-based state persistence"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
    
    def save(self, key: str, value: Any, ttl: Optional[int] = None):
        """Save state to Redis"""
        serialized = json.dumps(value, default=str)
        if ttl:
            self.client.setex(key, ttl, serialized)
        else:
            self.client.set(key, serialized)
    
    def load(self, key: str) -> Optional[Any]:
        """Load state from Redis"""
        value = self.client.get(key)
        if value:
            return json.loads(value)
        return None
    
    def delete(self, key: str):
        """Delete state"""
        self.client.delete(key)
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        return self.client.exists(key) > 0
    
    def set_ttl(self, key: str, ttl: int):
        """Set time-to-live for key"""
        self.client.expire(key, ttl)
    
    def keys_by_pattern(self, pattern: str) -> List[str]:
        """Find keys matching pattern"""
        return self.client.keys(pattern)

# Usage
store = RedisStateStore()

# Save with TTL (expires in 1 hour)
store.save("session:user123", {
    "user_id": "user123",
    "last_query": "What is AI?"
}, ttl=3600)

# Load
session = store.load("session:user123")
```

### Checkpoint System

```python
from typing import Callable
import pickle

class CheckpointManager:
    """Manage execution checkpoints"""
    
    def __init__(self, checkpoint_dir: str = ".checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(
        self,
        workflow_id: str,
        step_id: str,
        state: Any
    ):
        """Save checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{workflow_id}_{step_id}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                "workflow_id": workflow_id,
                "step_id": step_id,
                "state": state,
                "timestamp": datetime.now()
            }, f)
        print(f"Checkpoint saved: {step_id}")
    
    def load_checkpoint(
        self,
        workflow_id: str,
        step_id: Optional[str] = None
    ) -> Optional[dict]:
        """Load checkpoint"""
        if step_id:
            # Load specific checkpoint
            checkpoint_path = self.checkpoint_dir / f"{workflow_id}_{step_id}.pkl"
            if checkpoint_path.exists():
                with open(checkpoint_path, 'rb') as f:
                    return pickle.load(f)
        else:
            # Load latest checkpoint
            checkpoints = list(self.checkpoint_dir.glob(f"{workflow_id}_*.pkl"))
            if checkpoints:
                latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                with open(latest, 'rb') as f:
                    return pickle.load(f)
        
        return None
    
    def resume_from_checkpoint(
        self,
        workflow_id: str,
        workflow_func: Callable,
        step_id: Optional[str] = None
    ):
        """Resume workflow from checkpoint"""
        checkpoint = self.load_checkpoint(workflow_id, step_id)
        
        if checkpoint:
            print(f"Resuming from checkpoint: {checkpoint['step_id']}")
            state = checkpoint["state"]
            return workflow_func(state, resume_from=checkpoint["step_id"])
        else:
            print("No checkpoint found, starting from beginning")
            return workflow_func(None)

# Usage
checkpoint_mgr = CheckpointManager()

def long_running_workflow(state: Any = None, resume_from: str = None):
    """Workflow with checkpointing"""
    workflow_id = "long_task_001"
    
    if resume_from == "step2":
        print("Resuming from step 2")
        # Skip to step 2
    else:
        # Step 1
        print("Executing step 1...")
        state = {"step1_result": "data"}
        checkpoint_mgr.save_checkpoint(workflow_id, "step1", state)
    
    # Step 2
    print("Executing step 2...")
    state["step2_result"] = "more data"
    checkpoint_mgr.save_checkpoint(workflow_id, "step2", state)
    
    # Step 3
    print("Executing step 3...")
    state["step3_result"] = "final data"
    
    return state

# Run workflow
result = long_running_workflow()

# Or resume from checkpoint
result = checkpoint_mgr.resume_from_checkpoint(
    "long_task_001",
    long_running_workflow
)
```

## State Transitions

Managing state changes and ensuring valid transitions.

### State Machine

```python
from enum import Enum
from typing import Dict, Set, Callable, Optional

class TaskState(Enum):
    """Task states"""
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StateMachine:
    """Finite state machine for task states"""
    
    def __init__(self, initial_state: TaskState):
        self.current_state = initial_state
        self.transitions: Dict[TaskState, Set[TaskState]] = {
            TaskState.CREATED: {TaskState.QUEUED, TaskState.CANCELLED},
            TaskState.QUEUED: {TaskState.RUNNING, TaskState.CANCELLED},
            TaskState.RUNNING: {TaskState.PAUSED, TaskState.COMPLETED, TaskState.FAILED},
            TaskState.PAUSED: {TaskState.RUNNING, TaskState.CANCELLED},
            TaskState.COMPLETED: set(),  # Terminal state
            TaskState.FAILED: set(),      # Terminal state
            TaskState.CANCELLED: set()    # Terminal state
        }
        self.callbacks: Dict[tuple, Callable] = {}
        self.history: List[tuple] = []
    
    def can_transition(self, to_state: TaskState) -> bool:
        """Check if transition is valid"""
        return to_state in self.transitions.get(self.current_state, set())
    
    def transition(self, to_state: TaskState) -> bool:
        """Attempt state transition"""
        if not self.can_transition(to_state):
            raise ValueError(
                f"Invalid transition: {self.current_state.value} -> {to_state.value}"
            )
        
        old_state = self.current_state
        self.current_state = to_state
        self.history.append((old_state, to_state, datetime.now()))
        
        # Execute callback if registered
        callback = self.callbacks.get((old_state, to_state))
        if callback:
            callback(old_state, to_state)
        
        return True
    
    def on_transition(
        self,
        from_state: TaskState,
        to_state: TaskState,
        callback: Callable
    ):
        """Register transition callback"""
        self.callbacks[(from_state, to_state)] = callback
    
    def get_available_transitions(self) -> Set[TaskState]:
        """Get valid next states"""
        return self.transitions.get(self.current_state, set())
    
    def is_terminal(self) -> bool:
        """Check if in terminal state"""
        return len(self.get_available_transitions()) == 0

# Usage
sm = StateMachine(TaskState.CREATED)

# Register callbacks
sm.on_transition(
    TaskState.QUEUED,
    TaskState.RUNNING,
    lambda old, new: print(f"Task started: {old.value} -> {new.value}")
)

# Transition through states
sm.transition(TaskState.QUEUED)     # OK
sm.transition(TaskState.RUNNING)    # OK, triggers callback
sm.transition(TaskState.COMPLETED)  # OK

try:
    sm.transition(TaskState.RUNNING)  # Error: terminal state
except ValueError as e:
    print(e)
```

### Transaction-Style State Updates

```python
from contextlib import contextmanager
from copy import deepcopy

class TransactionalState:
    """State with transaction support"""
    
    def __init__(self, initial_state: dict):
        self.state = initial_state
        self._snapshot_stack = []
    
    @contextmanager
    def transaction(self):
        """Transaction context manager"""
        # Save current state
        snapshot = deepcopy(self.state)
        self._snapshot_stack.append(snapshot)
        
        try:
            yield self
            # Commit: pop snapshot but keep changes
            self._snapshot_stack.pop()
        except Exception:
            # Rollback: restore snapshot
            self.state = self._snapshot_stack.pop()
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value"""
        return self.state.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set value"""
        self.state[key] = value
    
    def update(self, updates: dict):
        """Update multiple values"""
        self.state.update(updates)

# Usage
state = TransactionalState({"counter": 0, "status": "ready"})

try:
    with state.transaction():
        state.set("counter", 1)
        state.set("status", "processing")
        
        # Simulate error
        if state.get("counter") > 0:
            raise ValueError("Something went wrong")
        
        state.set("counter", 2)
except ValueError:
    pass

# State rolled back
print(state.get("counter"))  # 0
print(state.get("status"))   # "ready"

# Successful transaction
with state.transaction():
    state.set("counter", 1)
    state.set("status", "processing")

print(state.get("counter"))  # 1
print(state.get("status"))   # "processing"
```

## Maintaining Context

Strategies for keeping relevant context available.

### Context Accumulation

```python
class ContextAccumulator:
    """Accumulate context across operations"""
    
    def __init__(self, max_context_items: int = 50):
        self.context_items: List[dict] = []
        self.max_items = max_context_items
        self.importance_scores: Dict[int, float] = {}
    
    def add_context(
        self,
        item: dict,
        importance: float = 0.5,
        category: str = "general"
    ):
        """Add context item"""
        context_entry = {
            "id": len(self.context_items),
            "item": item,
            "category": category,
            "timestamp": datetime.now(),
            "access_count": 0
        }
        
        self.context_items.append(context_entry)
        self.importance_scores[context_entry["id"]] = importance
        
        # Prune if needed
        if len(self.context_items) > self.max_items:
            self._prune_context()
    
    def get_context(
        self,
        category: Optional[str] = None,
        min_importance: float = 0.0
    ) -> List[dict]:
        """Get relevant context"""
        relevant = []
        
        for entry in self.context_items:
            entry_id = entry["id"]
            importance = self.importance_scores.get(entry_id, 0.0)
            
            # Filter by category and importance
            if category and entry["category"] != category:
                continue
            if importance < min_importance:
                continue
            
            entry["access_count"] += 1
            relevant.append(entry["item"])
        
        return relevant
    
    def _prune_context(self):
        """Remove least important context"""
        # Calculate composite score (importance + recency + access count)
        now = datetime.now()
        scores = []
        
        for entry in self.context_items:
            entry_id = entry["id"]
            importance = self.importance_scores.get(entry_id, 0.0)
            
            # Recency score (newer is better)
            age_hours = (now - entry["timestamp"]).total_seconds() / 3600
            recency = 1.0 / (1.0 + age_hours)
            
            # Access frequency score
            access = min(entry["access_count"] / 10.0, 1.0)
            
            # Composite score
            score = (importance * 0.5) + (recency * 0.3) + (access * 0.2)
            scores.append((entry_id, score))
        
        # Keep top N items
        scores.sort(key=lambda x: x[1], reverse=True)
        keep_ids = {eid for eid, _ in scores[:self.max_items]}
        
        # Remove low-scoring items
        self.context_items = [
            entry for entry in self.context_items
            if entry["id"] in keep_ids
        ]

# Usage
accumulator = ContextAccumulator(max_context_items=20)

accumulator.add_context(
    {"type": "user_preference", "theme": "dark"},
    importance=0.8,
    category="preferences"
)

accumulator.add_context(
    {"type": "recent_query", "query": "Python tips"},
    importance=0.6,
    category="history"
)

# Get relevant context
prefs = accumulator.get_context(category="preferences")
important = accumulator.get_context(min_importance=0.7)
```

## Stateful Interactions

Handling multi-turn, stateful conversations and tasks.

### Conversation Manager

```python
class ConversationManager:
    """Manage stateful conversation"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.history = ConversationHistory()
        self.context = ContextAccumulator()
        self.user_profile = {}
        self.current_intent = None
        self.pending_clarifications = []
    
    def process_message(self, user_message: str) -> str:
        """Process message with state"""
        # Add to history
        self.history.add_message(Role.USER, user_message)
        
        # Extract intent and entities
        intent = self._extract_intent(user_message)
        entities = self._extract_entities(user_message)
        
        # Update context
        self.context.add_context({
            "type": "user_message",
            "content": user_message,
            "intent": intent,
            "entities": entities
        }, importance=0.7)
        
        # Check for pending clarifications
        if self.pending_clarifications:
            response = self._handle_clarification(user_message)
        else:
            # Handle based on intent
            response = self._handle_intent(intent, entities, user_message)
        
        # Add response to history
        self.history.add_message(Role.ASSISTANT, response)
        
        return response
    
    def _extract_intent(self, message: str) -> str:
        """Extract user intent"""
        # Simplified intent extraction
        message_lower = message.lower()
        if "search" in message_lower or "find" in message_lower:
            return "search"
        elif "help" in message_lower:
            return "help"
        elif "remember" in message_lower or "save" in message_lower:
            return "save_preference"
        else:
            return "general"
    
    def _extract_entities(self, message: str) -> dict:
        """Extract entities from message"""
        # Simplified entity extraction
        return {}
    
    def _handle_intent(
        self,
        intent: str,
        entities: dict,
        message: str
    ) -> str:
        """Handle message based on intent"""
        if intent == "search":
            return self._handle_search(message)
        elif intent == "help":
            return self._handle_help()
        elif intent == "save_preference":
            return self._save_preference(message)
        else:
            return self._handle_general(message)
    
    def _handle_search(self, message: str) -> str:
        """Handle search intent"""
        # Get relevant context
        history_context = self.history.get_recent(3)
        
        # Perform search with context
        results = perform_search(message, context=history_context)
        
        return f"Found {len(results)} results: ..."
    
    def _handle_help(self) -> str:
        """Handle help request"""
        return "I can help you search, answer questions, and remember preferences."
    
    def _save_preference(self, message: str) -> str:
        """Save user preference"""
        # Extract preference
        # Simplified: just save the message
        self.user_profile["last_preference"] = message
        return "I've saved your preference."
    
    def _handle_general(self, message: str) -> str:
        """Handle general message"""
        # Get relevant context
        relevant_context = self.context.get_context(min_importance=0.5)
        
        # Generate response using context
        response = generate_response(message, context=relevant_context)
        
        return response
    
    def _handle_clarification(self, message: str) -> str:
        """Handle pending clarification"""
        clarification = self.pending_clarifications.pop(0)
        # Process clarification
        return f"Thanks for clarifying. {clarification['followup']}"
    
    def request_clarification(self, question: str, followup: str):
        """Request clarification from user"""
        self.pending_clarifications.append({
            "question": question,
            "followup": followup
        })
        return question

# Usage
manager = ConversationManager("agent_001")

response1 = manager.process_message("Find papers on AI safety")
# ... uses context to search

response2 = manager.process_message("Show me the most recent ones")
# ... remembers previous search context

response3 = manager.process_message("Save my preference for recent papers")
# ... saves to user profile
```

## State Synchronization

Coordinating state across distributed components.

### State Sync Protocol

```python
import threading
import queue
from typing import Callable

class StateSyncManager:
    """Synchronize state across components"""
    
    def __init__(self):
        self.state = {}
        self.lock = threading.Lock()
        self.subscribers: Dict[str, List[Callable]] = {}
        self.update_queue = queue.Queue()
        self._start_sync_worker()
    
    def set(self, key: str, value: Any):
        """Set value and notify subscribers"""
        with self.lock:
            old_value = self.state.get(key)
            self.state[key] = value
        
        # Queue notification
        self.update_queue.put({
            "key": key,
            "old_value": old_value,
            "new_value": value
        })
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value thread-safely"""
        with self.lock:
            return self.state.get(key, default)
    
    def subscribe(self, key: str, callback: Callable):
        """Subscribe to key changes"""
        if key not in self.subscribers:
            self.subscribers[key] = []
        self.subscribers[key].append(callback)
    
    def _start_sync_worker(self):
        """Start background sync worker"""
        def worker():
            while True:
                update = self.update_queue.get()
                if update is None:
                    break
                
                key = update["key"]
                if key in self.subscribers:
                    for callback in self.subscribers[key]:
                        try:
                            callback(
                                update["old_value"],
                                update["new_value"]
                            )
                        except Exception as e:
                            print(f"Subscriber error: {e}")
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

# Usage
sync_manager = StateSyncManager()

# Subscribe to changes
def on_status_change(old_value, new_value):
    print(f"Status changed: {old_value} -> {new_value}")

sync_manager.subscribe("task_status", on_status_change)

# Update state (triggers notification)
sync_manager.set("task_status", "running")
```

## State Recovery

Recovering state after failures.

### Automatic Recovery

```python
class RecoverableState:
    """State with automatic recovery"""
    
    def __init__(self, state_id: str, storage: PersistentState):
        self.state_id = state_id
        self.storage = storage
        self.state = self._load_or_initialize()
        self.auto_save = True
        self.save_interval = 10  # seconds
        self._last_save = datetime.now()
    
    def _load_or_initialize(self) -> dict:
        """Load existing state or initialize new"""
        saved_state = self.storage.get(self.state_id)
        if saved_state:
            print(f"Recovered state: {self.state_id}")
            return saved_state
        else:
            print(f"Initialized new state: {self.state_id}")
            return {}
    
    def update(self, updates: dict):
        """Update state with auto-save"""
        self.state.update(updates)
        
        if self.auto_save:
            # Check if save interval elapsed
            if (datetime.now() - self._last_save).total_seconds() >= self.save_interval:
                self.save()
    
    def save(self):
        """Save state"""
        self.storage.set(self.state_id, self.state)
        self._last_save = datetime.now()
        print(f"State saved: {self.state_id}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - always save"""
        self.save()
        return False

# Usage
storage = PersistentState()

with RecoverableState("workflow_state", storage) as state:
    state.update({"step": 1, "data": [1, 2, 3]})
    # ... do work ...
    state.update({"step": 2, "data": [4, 5, 6]})
    # State automatically saved on exit

# After crash/restart
with RecoverableState("workflow_state", storage) as state:
    # State recovered from storage
    print(state.state)  # {"step": 2, "data": [4, 5, 6]}
```

## Performance Optimization

Optimizing state management for performance.

### State Caching

```python
from functools import lru_cache
import time

class CachedStateManager:
    """State manager with caching"""
    
    def __init__(self, storage: PersistentState, cache_ttl: int = 60):
        self.storage = storage
        self.cache: Dict[str, tuple] = {}  # {key: (value, timestamp)}
        self.cache_ttl = cache_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get with caching"""
        # Check cache
        if key in self.cache:
            value, timestamp = self.cache[key]
            age = time.time() - timestamp
            
            if age < self.cache_ttl:
                return value
            else:
                # Expired
                del self.cache[key]
        
        # Load from storage
        value = self.storage.get(key)
        
        if value is not None:
            # Cache it
            self.cache[key] = (value, time.time())
        
        return value
    
    def set(self, key: str, value: Any):
        """Set with cache update"""
        # Update storage
        self.storage.set(key, value)
        
        # Update cache
        self.cache[key] = (value, time.time())
    
    def invalidate(self, key: str):
        """Invalidate cache entry"""
        self.cache.pop(key, None)
    
    def clear_cache(self):
        """Clear all cache"""
        self.cache.clear()

# Usage
storage = PersistentState()
cached_mgr = CachedStateManager(storage, cache_ttl=30)

# First access - loads from storage
value1 = cached_mgr.get("config")

# Subsequent accesses - from cache
value2 = cached_mgr.get("config")  # Fast!
value3 = cached_mgr.get("config")  # Fast!
```

## Real-World Scenarios

### Scenario 1: Multi-Step Research Task

```python
class ResearchTaskState:
    """State for research task"""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.query = ""
        self.papers_found = []
        self.papers_read = []
        self.summaries = []
        self.final_report = None
        self.current_step = "initialized"
        self.metadata = {
            "started_at": datetime.now(),
            "steps_completed": []
        }
    
    def mark_step_complete(self, step: str):
        """Mark step complete"""
        self.current_step = step
        self.metadata["steps_completed"].append({
            "step": step,
            "completed_at": datetime.now()
        })
    
    def to_dict(self) -> dict:
        """Serialize state"""
        return {
            "task_id": self.task_id,
            "query": self.query,
            "papers_found": len(self.papers_found),
            "papers_read": len(self.papers_read),
            "current_step": self.current_step,
            "metadata": self.metadata
        }

def research_with_state(query: str) -> ResearchTaskState:
    """Research task with state management"""
    state = ResearchTaskState(f"research_{uuid.uuid4().hex[:8]}")
    state.query = query
    
    # Step 1: Search
    state.papers_found = search_papers(query)
    state.mark_step_complete("search")
    
    # Step 2: Read papers
    for paper in state.papers_found[:5]:
        content = read_paper(paper)
        state.papers_read.append(content)
    state.mark_step_complete("reading")
    
    # Step 3: Summarize
    for paper in state.papers_read:
        summary = summarize_paper(paper)
        state.summaries.append(summary)
    state.mark_step_complete("summarization")
    
    # Step 4: Generate report
    state.final_report = generate_report(state.summaries)
    state.mark_step_complete("report_generation")
    
    return state

# Execute
result = research_with_state("AI safety research")
print(result.to_dict())
```

## Summary

State management is foundational for agent orchestration:

**Key Concepts**:
- **Agent State**: Identity, capabilities, and configuration
- **Conversation State**: Message history and context
- **Workflow State**: Progress tracking and intermediate results

**State Types**:
- **Ephemeral**: Temporary, in-memory state
- **Session**: Per-session persistence
- **Persistent**: Durable, long-term storage

**Core Patterns**:
- **Checkpointing**: Save progress for recovery
- **Transactions**: Atomic state updates with rollback
- **State Machines**: Enforce valid state transitions
- **Context Accumulation**: Build up relevant context
- **Synchronization**: Coordinate distributed state

**Best Practices**:
1. Choose appropriate persistence level for each state type
2. Implement checkpointing for long-running tasks
3. Use state machines to enforce valid transitions
4. Cache frequently accessed state
5. Sync state across distributed components
6. Design for state recovery after failures

Effective state management enables agents to maintain coherence, track progress, and handle complex multi-step workflows reliably.

## Next Steps

Explore related orchestration topics:

- **[Workflow Patterns](workflow-patterns.md)**: Patterns for orchestrating agent workflows
- **[Control Flow](control-flow.md)**: Conditionals, loops, and branching in workflows
- **[Parallelization](parallelization.md)**: Concurrent execution and coordination
- **[Filesystem Context](filesystem-context.md)**: Using files for state persistence

Related concepts:
- **[Memory Architecture](../memory-systems/memory-architecture.md)**: Long-term memory systems
- **[Tool Use](../tool-use/function-calling.md)**: Stateful tool execution
