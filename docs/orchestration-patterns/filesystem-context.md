# Filesystem Context

## Table of Contents

- [Introduction](#introduction)
- [File-Based State](#file-based-state)
- [Context Files](#context-files)
- [Directory Structures](#directory-structures)
- [File Watching](#file-watching)
- [Configuration Management](#configuration-management)
- [Workspace Patterns](#workspace-patterns)
- [Project State](#project-state)
- [Cross-Session Persistence](#cross-session-persistence)
- [File-Based Communication](#file-based-communication)
- [Workspace Synchronization](#workspace-synchronization)
- [Best Practices](#best-practices)
- [Real-World Applications](#real-world-applications)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

The filesystem provides a natural, durable medium for agent context and state. Unlike in-memory state that vanishes when agents restart, file-based context enables:

- **Persistence**: Context survives agent restarts
- **Visibility**: Humans can inspect and modify context
- **Versioning**: Git and other tools track changes
- **Collaboration**: Multiple agents share filesystem state
- **Simplicity**: No databases or special infrastructure needed

> "The filesystem is the universal persistence layer--use it."

Agents can leverage files for:

- **Configuration**: Settings and parameters
- **State**: Current workflow status
- **Context**: Conversation history and working memory
- **Artifacts**: Generated outputs
- **Coordination**: Inter-agent communication

```
Filesystem Context Architecture:

Agent Memory ←→ Files
     ↓           ↓
  RAM State   Disk State
     ↓           ↓
 (ephemeral) (persistent)

Agent reads/writes:
- .agent/state.json (current state)
- .agent/context/ (context files)
- .agent/artifacts/ (outputs)
- .agent/config/ (settings)
```

## File-Based State

Using files to store agent state.

### State File Manager

```python
import json
from pathlib import Path
from typing import Any, Optional
from datetime import datetime
import threading

class StateFileManager:
    """Manage agent state in files"""

    def __init__(self, state_dir: str = ".agent/state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def save_state(self, state_id: str, state: dict):
        """Save state to file"""
        state_file = self.state_dir / f"{state_id}.json"

        # Add metadata
        state_with_meta = {
            **state,
            "_metadata": {
                "updated_at": datetime.now().isoformat(),
                "state_id": state_id
            }
        }

        with self.lock:
            with open(state_file, 'w') as f:
                json.dump(state_with_meta, f, indent=2)

        print(f"Saved state: {state_id}")

    def load_state(self, state_id: str) -> Optional[dict]:
        """Load state from file"""
        state_file = self.state_dir / f"{state_id}.json"

        if not state_file.exists():
            return None

        with self.lock:
            with open(state_file, 'r') as f:
                state = json.load(f)

        # Remove metadata before returning
        state.pop("_metadata", None)
        return state

    def delete_state(self, state_id: str):
        """Delete state file"""
        state_file = self.state_dir / f"{state_id}.json"

        if state_file.exists():
            with self.lock:
                state_file.unlink()
            print(f"Deleted state: {state_id}")

    def list_states(self) -> list[str]:
        """List all state files"""
        return [f.stem for f in self.state_dir.glob("*.json")]

    def archive_state(self, state_id: str):
        """Archive old state"""
        archive_dir = self.state_dir / "archive"
        archive_dir.mkdir(exist_ok=True)

        state_file = self.state_dir / f"{state_id}.json"
        if state_file.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_file = archive_dir / f"{state_id}_{timestamp}.json"

            with self.lock:
                state_file.rename(archive_file)

            print(f"Archived state: {state_id}")

# Example usage
state_mgr = StateFileManager()

# Save agent state
agent_state = {
    "current_task": "data_analysis",
    "progress": 0.65,
    "variables": {"x": 42, "y": "hello"},
    "history": ["step1", "step2", "step3"]
}

state_mgr.save_state("agent_001", agent_state)

# Load state (e.g., after restart)
loaded_state = state_mgr.load_state("agent_001")
print(f"Loaded state: {loaded_state}")

# List all states
states = state_mgr.list_states()
print(f"Available states: {states}")
```

### Incremental State Updates

```python
class IncrementalStateFile:
    """Append-only state file for audit trail"""

    def __init__(self, state_file: str = ".agent/state.jsonl"):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.current_state = {}
        self._load_all()

    def _load_all(self):
        """Load all state updates"""
        if not self.state_file.exists():
            return

        with open(self.state_file, 'r') as f:
            for line in f:
                update = json.loads(line)
                self._apply_update(update)

    def _apply_update(self, update: dict):
        """Apply state update"""
        action = update.get("action")

        if action == "set":
            self.current_state[update["key"]] = update["value"]
        elif action == "delete":
            self.current_state.pop(update["key"], None)
        elif action == "update":
            if update["key"] in self.current_state:
                self.current_state[update["key"]].update(update["value"])
            else:
                self.current_state[update["key"]] = update["value"]

    def set(self, key: str, value: Any):
        """Set state value"""
        update = {
            "timestamp": datetime.now().isoformat(),
            "action": "set",
            "key": key,
            "value": value
        }

        self._append_update(update)
        self._apply_update(update)

    def get(self, key: str) -> Any:
        """Get state value"""
        return self.current_state.get(key)

    def delete(self, key: str):
        """Delete state key"""
        update = {
            "timestamp": datetime.now().isoformat(),
            "action": "delete",
            "key": key
        }

        self._append_update(update)
        self._apply_update(update)

    def _append_update(self, update: dict):
        """Append update to file"""
        with open(self.state_file, 'a') as f:
            f.write(json.dumps(update) + '\n')

    def get_all(self) -> dict:
        """Get complete current state"""
        return self.current_state.copy()

    def get_history(self) -> list[dict]:
        """Get full update history"""
        history = []

        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                for line in f:
                    history.append(json.loads(line))

        return history

    def compact(self):
        """Compact state file"""
        # Create new file with just current state
        temp_file = self.state_file.with_suffix('.tmp')

        # Write current state as set operations
        with open(temp_file, 'w') as f:
            timestamp = datetime.now().isoformat()
            for key, value in self.current_state.items():
                update = {
                    "timestamp": timestamp,
                    "action": "set",
                    "key": key,
                    "value": value
                }
                f.write(json.dumps(update) + '\n')

        # Replace original
        temp_file.replace(self.state_file)
        print("State file compacted")

# Usage
state = IncrementalStateFile()

# Track state changes
state.set("task", "processing")
state.set("progress", 0.1)
state.set("progress", 0.5)
state.set("progress", 1.0)
state.set("task", "complete")

# Get current state
print(f"Current state: {state.get_all()}")

# View history
history = state.get_history()
print(f"History has {len(history)} updates")

# Compact file
state.compact()
```

## Context Files

Using files to maintain agent context.

### Context Directory

```python
class ContextDirectory:
    """Manage context files in a directory"""

    def __init__(self, context_dir: str = ".agent/context"):
        self.context_dir = Path(context_dir)
        self.context_dir.mkdir(parents=True, exist_ok=True)

    def save_context(self, name: str, content: str, metadata: dict = None):
        """Save context file"""
        context_file = self.context_dir / f"{name}.txt"

        # Write content
        with open(context_file, 'w') as f:
            f.write(content)

        # Save metadata
        if metadata:
            meta_file = context_file.with_suffix('.meta.json')
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)

        print(f"Saved context: {name}")

    def load_context(self, name: str) -> Optional[tuple[str, dict]]:
        """Load context file and metadata"""
        context_file = self.context_dir / f"{name}.txt"

        if not context_file.exists():
            return None

        # Load content
        with open(context_file, 'r') as f:
            content = f.read()

        # Load metadata if exists
        meta_file = context_file.with_suffix('.meta.json')
        metadata = {}
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                metadata = json.load(f)

        return content, metadata

    def list_contexts(self) -> list[str]:
        """List all context files"""
        return [f.stem for f in self.context_dir.glob("*.txt")]

    def search_context(self, query: str) -> list[tuple[str, str]]:
        """Search context files"""
        results = []

        for context_file in self.context_dir.glob("*.txt"):
            with open(context_file, 'r') as f:
                content = f.read()

            if query.lower() in content.lower():
                # Find snippet around match
                idx = content.lower().index(query.lower())
                start = max(0, idx - 50)
                end = min(len(content), idx + 50)
                snippet = content[start:end]

                results.append((context_file.stem, snippet))

        return results

    def delete_context(self, name: str):
        """Delete context file"""
        context_file = self.context_dir / f"{name}.txt"
        meta_file = context_file.with_suffix('.meta.json')

        if context_file.exists():
            context_file.unlink()

        if meta_file.exists():
            meta_file.unlink()

        print(f"Deleted context: {name}")

# Usage
context = ContextDirectory()

# Save conversation context
context.save_context(
    "conversation_001",
    "User: How do I deploy?\nAgent: Use these steps...",
    metadata={"session_id": "abc123", "timestamp": datetime.now().isoformat()}
)

# Save working memory
context.save_context(
    "working_memory",
    "Current task: Analyzing data\nProgress: 50%\nNext: Generate report"
)

# Load context
conv_content, conv_meta = context.load_context("conversation_001")
print(f"Loaded conversation: {conv_content[:50]}...")

# Search contexts
results = context.search_context("deploy")
print(f"Found '{len(results)}' matching contexts")
```

### Markdown Context

```python
class MarkdownContext:
    """Use markdown files for structured context"""

    def __init__(self, context_dir: str = ".agent/markdown"):
        self.context_dir = Path(context_dir)
        self.context_dir.mkdir(parents=True, exist_ok=True)

    def create_context(self, name: str) -> "MarkdownContextFile":
        """Create new markdown context file"""
        return MarkdownContextFile(self.context_dir / f"{name}.md")

class MarkdownContextFile:
    """Single markdown context file"""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.sections = {}
        self._load()

    def _load(self):
        """Load existing file"""
        if not self.file_path.exists():
            return

        with open(self.file_path, 'r') as f:
            content = f.read()

        # Parse sections
        current_section = None
        current_content = []

        for line in content.split('\n'):
            if line.startswith('## '):
                # Save previous section
                if current_section:
                    self.sections[current_section] = '\n'.join(current_content)

                # Start new section
                current_section = line[3:].strip()
                current_content = []
            elif current_section:
                current_content.append(line)

        # Save last section
        if current_section:
            self.sections[current_section] = '\n'.join(current_content)

    def set_section(self, section: str, content: str):
        """Set section content"""
        self.sections[section] = content
        self._save()

    def get_section(self, section: str) -> Optional[str]:
        """Get section content"""
        return self.sections.get(section)

    def append_to_section(self, section: str, content: str):
        """Append to section"""
        current = self.sections.get(section, "")
        self.sections[section] = current + "\n" + content
        self._save()

    def _save(self):
        """Save to file"""
        lines = []

        for section, content in self.sections.items():
            lines.append(f"## {section}")
            lines.append(content)
            lines.append("")

        with open(self.file_path, 'w') as f:
            f.write('\n'.join(lines))

    def list_sections(self) -> list[str]:
        """List all sections"""
        return list(self.sections.keys())

# Usage
md_context = MarkdownContext()

# Create context file
task_context = md_context.create_context("current_task")

# Add sections
task_context.set_section("Objective", "Analyze sales data and generate report")
task_context.set_section("Progress", "- ✓ Data loaded\n- ✓ Analysis complete\n- ⏳ Generating report")
task_context.set_section("Notes", "Customer segment A shows 20% growth")

# Append to section
task_context.append_to_section("Notes", "Need to investigate segment B decline")

# Read section
progress = task_context.get_section("Progress")
print(f"Progress:\n{progress}")
```

## Directory Structures

Organizing filesystem context.

### Agent Workspace

```python
class AgentWorkspace:
    """Structured agent workspace"""

    def __init__(self, workspace_root: str = ".agent"):
        self.root = Path(workspace_root)
        self._init_structure()

    def _init_structure(self):
        """Initialize directory structure"""
        dirs = [
            "state",        # Current state files
            "context",      # Context and memory
            "artifacts",    # Generated outputs
            "config",       # Configuration
            "logs",         # Execution logs
            "tmp",          # Temporary files
            "cache",        # Cached data
        ]

        for dir_name in dirs:
            (self.root / dir_name).mkdir(parents=True, exist_ok=True)

    def get_state_dir(self) -> Path:
        """Get state directory"""
        return self.root / "state"

    def get_context_dir(self) -> Path:
        """Get context directory"""
        return self.root / "context"

    def get_artifacts_dir(self) -> Path:
        """Get artifacts directory"""
        return self.root / "artifacts"

    def get_config_file(self, name: str) -> Path:
        """Get config file path"""
        return self.root / "config" / f"{name}.json"

    def save_config(self, name: str, config: dict):
        """Save configuration"""
        config_file = self.get_config_file(name)
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

    def load_config(self, name: str) -> Optional[dict]:
        """Load configuration"""
        config_file = self.get_config_file(name)
        if not config_file.exists():
            return None

        with open(config_file, 'r') as f:
            return json.load(f)

    def log(self, message: str, level: str = "info"):
        """Write to log file"""
        log_file = self.root / "logs" / f"{datetime.now().strftime('%Y-%m-%d')}.log"

        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [{level.upper()}] {message}\n"

        with open(log_file, 'a') as f:
            f.write(log_entry)

    def clean_tmp(self):
        """Clean temporary files"""
        tmp_dir = self.root / "tmp"
        for file in tmp_dir.glob("*"):
            if file.is_file():
                file.unlink()
        print("Cleaned temporary files")

    def get_info(self) -> dict:
        """Get workspace info"""
        def get_dir_size(path: Path) -> int:
            return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())

        return {
            "root": str(self.root),
            "size_bytes": get_dir_size(self.root),
            "state_files": len(list((self.root / "state").glob("*"))),
            "context_files": len(list((self.root / "context").glob("*"))),
            "artifacts": len(list((self.root / "artifacts").glob("*"))),
        }

# Usage
workspace = AgentWorkspace()

# Save configuration
workspace.save_config("agent", {
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 2000
})

# Load configuration
config = workspace.load_config("agent")
print(f"Config: {config}")

# Log messages
workspace.log("Agent started")
workspace.log("Processing task", level="info")
workspace.log("Task completed", level="info")

# Get workspace info
info = workspace.get_info()
print(f"Workspace info: {info}")
```

## File Watching

Monitoring file changes for reactive behavior.

### File Watcher

```python
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ContextFileWatcher(FileSystemEventHandler):
    """Watch context files for changes"""

    def __init__(self, on_change: callable):
        self.on_change = on_change

    def on_modified(self, event):
        """Handle file modification"""
        if not event.is_directory:
            print(f"File modified: {event.src_path}")
            self.on_change(event.src_path, "modified")

    def on_created(self, event):
        """Handle file creation"""
        if not event.is_directory:
            print(f"File created: {event.src_path}")
            self.on_change(event.src_path, "created")

    def on_deleted(self, event):
        """Handle file deletion"""
        if not event.is_directory:
            print(f"File deleted: {event.src_path}")
            self.on_change(event.src_path, "deleted")

class ReactiveAgent:
    """Agent that reacts to file changes"""

    def __init__(self, watch_dir: str = ".agent/context"):
        self.watch_dir = Path(watch_dir)
        self.observer = None

    def start_watching(self):
        """Start watching for file changes"""
        event_handler = ContextFileWatcher(self.handle_change)
        self.observer = Observer()
        self.observer.schedule(event_handler, str(self.watch_dir), recursive=True)
        self.observer.start()
        print(f"Watching directory: {self.watch_dir}")

    def stop_watching(self):
        """Stop watching"""
        if self.observer:
            self.observer.stop()
            self.observer.join()

    def handle_change(self, file_path: str, event_type: str):
        """Handle file change"""
        print(f"Processing {event_type} event for {file_path}")

        # React to change
        if event_type == "modified":
            self.on_file_modified(file_path)
        elif event_type == "created":
            self.on_file_created(file_path)
        elif event_type == "deleted":
            self.on_file_deleted(file_path)

    def on_file_modified(self, file_path: str):
        """React to file modification"""
        # Read new content
        with open(file_path, 'r') as f:
            content = f.read()

        print(f"File updated with {len(content)} bytes")
        # Agent can process new content here

    def on_file_created(self, file_path: str):
        """React to new file"""
        print(f"New file detected: {file_path}")
        # Agent can process new file here

    def on_file_deleted(self, file_path: str):
        """React to file deletion"""
        print(f"File removed: {file_path}")
        # Agent can clean up references here

# Usage
agent = ReactiveAgent()
agent.start_watching()

# Simulate file changes
context_dir = Path(".agent/context")
context_dir.mkdir(parents=True, exist_ok=True)

# Create file
test_file = context_dir / "test.txt"
test_file.write_text("Initial content")

# Modify file
time.sleep(1)
test_file.write_text("Updated content")

# Wait for events
time.sleep(2)

agent.stop_watching()
```

## Configuration Management

Managing agent configuration via files.

### Configuration Manager

```python
import yaml

class ConfigManager:
    """Manage agent configuration"""

    def __init__(self, config_file: str = ".agent/config.yaml"):
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.config = self._load_or_create()

    def _load_or_create(self) -> dict:
        """Load config or create default"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        else:
            # Create default config
            default = self._default_config()
            self._save(default)
            return default

    def _default_config(self) -> dict:
        """Default configuration"""
        return {
            "agent": {
                "name": "Agent",
                "version": "1.0.0",
                "model": "gpt-4"
            },
            "limits": {
                "max_iterations": 10,
                "timeout_seconds": 300
            },
            "filesystem": {
                "workspace": ".agent",
                "auto_save": True,
                "save_interval": 60
            },
            "features": {
                "file_watching": False,
                "auto_backup": True
            }
        }

    def _save(self, config: dict):
        """Save configuration"""
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def get(self, key_path: str, default=None) -> any:
        """Get configuration value by dot path"""
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default

            if value is None:
                return default

        return value

    def set(self, key_path: str, value: any):
        """Set configuration value"""
        keys = key_path.split('.')
        config = self.config

        # Navigate to parent
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set value
        config[keys[-1]] = value

        # Save
        self._save(self.config)
        print(f"Set {key_path} = {value}")

    def reload(self):
        """Reload configuration from file"""
        self.config = self._load_or_create()
        print("Configuration reloaded")

# Usage
config = ConfigManager()

# Get values
model = config.get("agent.model")
print(f"Model: {model}")

max_iterations = config.get("limits.max_iterations")
print(f"Max iterations: {max_iterations}")

# Set values
config.set("agent.temperature", 0.7)
config.set("features.debug_mode", True)

# Reload
config.reload()
```

## Project State

Managing project-level state across sessions.

### Project State Manager

```python
class ProjectState:
    """Track project state across sessions"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.state_file = self.project_root / ".agent" / "project_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load()

    def _load(self) -> dict:
        """Load project state"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)

        return {
            "project_name": self.project_root.name,
            "created_at": datetime.now().isoformat(),
            "sessions": [],
            "tasks": {},
            "artifacts": []
        }

    def _save(self):
        """Save project state"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def start_session(self) -> str:
        """Start new session"""
        session_id = f"session_{uuid.uuid4().hex[:8]}"

        session = {
            "session_id": session_id,
            "started_at": datetime.now().isoformat(),
            "ended_at": None,
            "tasks_completed": 0
        }

        self.state["sessions"].append(session)
        self._save()

        print(f"Started session: {session_id}")
        return session_id

    def end_session(self, session_id: str):
        """End session"""
        for session in self.state["sessions"]:
            if session["session_id"] == session_id:
                session["ended_at"] = datetime.now().isoformat()
                break

        self._save()
        print(f"Ended session: {session_id}")

    def add_task(self, task_id: str, task_info: dict):
        """Add task to project"""
        self.state["tasks"][task_id] = {
            **task_info,
            "created_at": datetime.now().isoformat(),
            "status": "pending"
        }
        self._save()

    def update_task(self, task_id: str, updates: dict):
        """Update task"""
        if task_id in self.state["tasks"]:
            self.state["tasks"][task_id].update(updates)
            self._save()

    def complete_task(self, task_id: str):
        """Mark task complete"""
        self.update_task(task_id, {
            "status": "completed",
            "completed_at": datetime.now().isoformat()
        })

    def register_artifact(self, artifact_path: str, artifact_type: str):
        """Register generated artifact"""
        artifact = {
            "path": artifact_path,
            "type": artifact_type,
            "created_at": datetime.now().isoformat()
        }

        self.state["artifacts"].append(artifact)
        self._save()

    def get_summary(self) -> dict:
        """Get project summary"""
        return {
            "project_name": self.state["project_name"],
            "total_sessions": len(self.state["sessions"]),
            "total_tasks": len(self.state["tasks"]),
            "completed_tasks": sum(
                1 for t in self.state["tasks"].values()
                if t.get("status") == "completed"
            ),
            "total_artifacts": len(self.state["artifacts"])
        }

# Usage
project = ProjectState()

# Start session
session_id = project.start_session()

# Add tasks
project.add_task("task_001", {
    "description": "Analyze data",
    "priority": "high"
})

project.add_task("task_002", {
    "description": "Generate report",
    "priority": "medium"
})

# Complete task
project.complete_task("task_001")

# Register artifact
project.register_artifact("report.pdf", "document")

# Get summary
summary = project.get_summary()
print(f"Project summary: {summary}")

# End session
project.end_session(session_id)
```

## File-Based Communication

Using files for agent coordination.

### Message Queue

```python
class FileBasedMessageQueue:
    """Message queue using files"""

    def __init__(self, queue_dir: str = ".agent/messages"):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def send(self, recipient: str, message: dict):
        """Send message to recipient"""
        message_id = f"msg_{uuid.uuid4().hex[:8]}"

        message_data = {
            "message_id": message_id,
            "recipient": recipient,
            "timestamp": datetime.now().isoformat(),
            **message
        }

        # Write to recipient's inbox
        inbox_dir = self.queue_dir / recipient
        inbox_dir.mkdir(exist_ok=True)

        message_file = inbox_dir / f"{message_id}.json"
        with open(message_file, 'w') as f:
            json.dump(message_data, f, indent=2)

        print(f"Sent message {message_id} to {recipient}")
        return message_id

    def receive(self, recipient: str) -> Optional[dict]:
        """Receive next message"""
        inbox_dir = self.queue_dir / recipient

        if not inbox_dir.exists():
            return None

        # Get oldest message
        messages = sorted(inbox_dir.glob("*.json"))

        if not messages:
            return None

        message_file = messages[0]

        # Read message
        with open(message_file, 'r') as f:
            message = json.load(f)

        # Delete message file
        message_file.unlink()

        return message

    def peek(self, recipient: str, count: int = 1) -> list[dict]:
        """Peek at messages without removing"""
        inbox_dir = self.queue_dir / recipient

        if not inbox_dir.exists():
            return []

        messages = sorted(inbox_dir.glob("*.json"))[:count]

        result = []
        for message_file in messages:
            with open(message_file, 'r') as f:
                result.append(json.load(f))

        return result

    def count(self, recipient: str) -> int:
        """Count messages"""
        inbox_dir = self.queue_dir / recipient

        if not inbox_dir.exists():
            return 0

        return len(list(inbox_dir.glob("*.json")))

# Usage: Inter-agent communication
queue = FileBasedMessageQueue()

# Agent 1 sends message to Agent 2
queue.send("agent_2", {
    "type": "task_request",
    "task": "analyze_data",
    "data_path": "/data/input.csv"
})

# Agent 2 receives message
message = queue.receive("agent_2")
if message:
    print(f"Received: {message['type']}")

    # Process task...

    # Send response
    queue.send("agent_1", {
        "type": "task_result",
        "task": message["task"],
        "result": "Analysis complete"
    })

# Agent 1 checks for response
response = queue.receive("agent_1")
if response:
    print(f"Got response: {response['result']}")
```

## Real-World Applications

### Pattern: Resumable Agent

```python
class ResumableAgent:
    """Agent that can resume from filesystem state"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.workspace = AgentWorkspace()
        self.state_file = self.workspace.get_state_dir() / f"{agent_id}.json"
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Load saved state"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)

        return {
            "current_task": None,
            "completed_steps": [],
            "working_memory": {},
            "iteration": 0
        }

    def _save_state(self):
        """Save current state"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def execute_task(self, task: dict):
        """Execute task with checkpointing"""
        self.state["current_task"] = task
        self.state["iteration"] = 0
        self._save_state()

        steps = task.get("steps", [])

        # Resume from last completed step
        start_idx = len(self.state["completed_steps"])

        for i, step in enumerate(steps[start_idx:], start=start_idx):
            print(f"Executing step {i+1}/{len(steps)}: {step}")

            # Execute step
            self._execute_step(step)

            # Mark completed
            self.state["completed_steps"].append(step)
            self.state["iteration"] += 1

            # Save after each step
            self._save_state()

        self.state["current_task"] = None
        self._save_state()

        print("Task completed")

    def _execute_step(self, step: str):
        """Execute single step"""
        # Simulate work
        time.sleep(0.5)

        # Update working memory
        self.state["working_memory"][f"step_{self.state['iteration']}"] = {
            "step": step,
            "timestamp": datetime.now().isoformat()
        }

# Usage
agent = ResumableAgent("agent_001")

# Start task
task = {
    "name": "Data Pipeline",
    "steps": [
        "load_data",
        "clean_data",
        "transform_data",
        "analyze_data",
        "generate_report"
    ]
}

# Execute (can be interrupted and resumed)
try:
    agent.execute_task(task)
except KeyboardInterrupt:
    print("\nAgent interrupted - state saved")
    print(f"Completed {len(agent.state['completed_steps'])} steps")

# Resume by creating new agent instance
# agent = ResumableAgent("agent_001")
# agent.execute_task(task)  # Will resume from saved state
```

## Summary

Filesystem context provides durable, visible state for agents:

**Key Concepts**:

- **File-Based State**: Persist state in files
- **Context Files**: Maintain conversation and working memory
- **Directory Structure**: Organized workspace
- **Configuration**: File-based settings
- **Project State**: Track across sessions

**Patterns**:

- **State Files**: JSON/YAML state persistence
- **Context Directory**: Organized context storage
- **Agent Workspace**: Structured directories
- **File Watching**: Reactive agents
- **Message Queue**: File-based coordination

**Benefits**:

- Survives agent restarts
- Human-readable and editable
- Version control friendly
- No database required
- Simple debugging

**Best Practices**:

1. Use structured directories
2. Save state incrementally
3. Include metadata
4. Implement file locking
5. Handle concurrent access
6. Clean up temporary files
7. Version control appropriate files

The filesystem is a universal, durable persistence layer perfectly suited for agent context.

## Next Steps

Explore related topics:

- **[Artifacts](artifacts.md)**: Generated outputs and artifacts
- **[State Management](state-management.md)**: Runtime state patterns
- **[Background Processes](background-processes.md)**: Long-running file operations

Related areas:

- **[Memory Systems](../memory-systems/memory-architecture.md)**: Persistent memory
- **[Tool Use](../tool-use/function-calling.md)**: File-based tools
