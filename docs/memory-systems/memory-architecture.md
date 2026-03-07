# Memory Architecture

## Table of Contents

- [Introduction](#introduction)
- [Memory Systems in Agents](#memory-systems-in-agents)
- [Short-Term Memory](#short-term-memory)
- [Long-Term Memory](#long-term-memory)
- [Episodic Memory](#episodic-memory)
- [Semantic Memory](#semantic-memory)
- [Memory Hierarchies](#memory-hierarchies)
- [Memory Interactions](#memory-interactions)
- [Architectural Patterns](#architectural-patterns)
- [Memory System Design](#memory-system-design)
- [Implementation Considerations](#implementation-considerations)
- [Performance and Scaling](#performance-and-scaling)
- [Memory in Different Agent Types](#memory-in-different-agent-types)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Memory is what transforms stateless language models into stateful agents. While a language model processes each request in isolation, an agent with memory can:

- **Remember past interactions** and maintain conversational continuity
- **Learn from experience** and improve over time
- **Build knowledge** about users, tasks, and domains
- **Maintain context** across long-running tasks
- **Recall relevant information** when needed

> "An agent without memory is like a person with amnesia - capable but perpetually starting from scratch."

Memory architecture determines how information flows through an agent system: what gets stored, how it's organized, when it's retrieved, and how it influences behavior. Understanding memory architecture is crucial for building agents that are coherent, capable, and continuously improving.

### Why Memory Matters

**Without memory**:

```python
# Each interaction is isolated
response_1 = agent.run("My name is Alice")
# Returns: "Nice to meet you!"

response_2 = agent.run("What's my name?")
# Returns: "I don't know your name" ❌
```

**With memory**:

```python
# Agent maintains state across interactions
response_1 = agent.run("My name is Alice")
# Stores: user.name = "Alice"
# Returns: "Nice to meet you, Alice!"

response_2 = agent.run("What's my name?")
# Retrieves: user.name = "Alice"
# Returns: "Your name is Alice!" ✓
```

### Memory System Goals

A good memory architecture should:

1. **Maintain coherence**: Enable consistent behavior across interactions
2. **Scale effectively**: Handle growing information without degrading
3. **Retrieve efficiently**: Find relevant information quickly
4. **Forget appropriately**: Not all information needs permanent storage
5. **Prioritize intelligently**: Surface the most relevant memories
6. **Adapt gracefully**: Learn from experience and improve over time

## Memory Systems in Agents

Agent memory systems are inspired by human cognitive architecture, adapted for computational constraints and capabilities.

### The Human Memory Analogy

Human memory consists of multiple interacting systems:

```
┌─────────────────────────────────────────────────────┐
│ Sensory Memory (milliseconds)                       │
│ Brief sensory impressions                           │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│ Working Memory (seconds to minutes)                 │
│ Active processing, limited capacity (~7 items)      │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│ Long-Term Memory (minutes to lifetime)              │
│                                                      │
│  ┌────────────────────┐  ┌─────────────────────┐   │
│  │ Episodic Memory    │  │ Semantic Memory     │   │
│  │ Personal events    │  │ Facts & concepts    │   │
│  │ "What happened"    │  │ "What I know"       │   │
│  └────────────────────┘  └─────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

### Agent Memory Architecture

Adapted for AI agents:

```
┌─────────────────────────────────────────────────────┐
│ Context Window (immediate)                          │
│ Current conversation, prompt, recent messages       │
│ Capacity: Model-dependent (4K-200K+ tokens)         │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│ Short-Term Memory (session)                         │
│ Working context, temporary variables, task state    │
│ Capacity: Constrained by context window             │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│ Long-Term Memory (persistent)                       │
│                                                      │
│  ┌────────────────────┐  ┌─────────────────────┐   │
│  │ Episodic Memory    │  │ Semantic Memory     │   │
│  │ Interaction logs   │  │ Knowledge base      │   │
│  │ Experience history │  │ Facts & entities    │   │
│  │ Temporal sequences │  │ Learned concepts    │   │
│  └────────────────────┘  └─────────────────────┘   │
│                                                      │
│  Storage: Vector DB, SQL, Documents, Graphs         │
└──────────────────────────────────────────────────────┘
```

## Short-Term Memory

Short-term memory (STM) holds information needed for the current task or conversation. It's the agent's "working memory."

### Characteristics

**Capacity**: Limited by context window (4K-200K tokens typically)

**Duration**: Current session or task

**Access Speed**: Immediate (in-context)

**Volatility**: Lost when conversation ends or context resets

### Components

**1. Context Window**

The model's immediate attention span:

```python
class ShortTermMemory:
    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
        self.messages: list[Message] = []
    
    def add_message(self, message: Message):
        """Add message to short-term memory."""
        self.messages.append(message)
        self._prune_if_needed()
    
    def _prune_if_needed(self):
        """Remove old messages if over capacity."""
        while self._count_tokens() > self.max_tokens:
            if len(self.messages) > 1:  # Keep at least system prompt
                self.messages.pop(1)  # Remove oldest user message
    
    def _count_tokens(self) -> int:
        """Count total tokens in messages."""
        return sum(count_tokens(msg.content) for msg in self.messages)
    
    def get_context(self) -> list[Message]:
        """Get all messages in short-term memory."""
        return self.messages.copy()
```

**2. Working Variables**

Temporary state for current task:

```python
class WorkingMemory:
    def __init__(self):
        self.variables: dict[str, Any] = {}
        self.task_state: dict[str, Any] = {}
    
    def set(self, key: str, value: Any):
        """Store temporary variable."""
        self.variables[key] = value
    
    def get(self, key: str) -> Any:
        """Retrieve temporary variable."""
        return self.variables.get(key)
    
    def update_task_state(self, **kwargs):
        """Update current task state."""
        self.task_state.update(kwargs)
    
    def clear_task(self):
        """Clear task-specific state."""
        self.task_state.clear()
    
    def to_context_string(self) -> str:
        """Convert working memory to context string."""
        context_parts = []
        
        if self.variables:
            context_parts.append("Working Variables:")
            for key, value in self.variables.items():
                context_parts.append(f"  {key}: {value}")
        
        if self.task_state:
            context_parts.append("Task State:")
            for key, value in self.task_state.items():
                context_parts.append(f"  {key}: {value}")
        
        return "\n".join(context_parts)
```

**3. Recent Context**

Information from recent interactions:

```python
from collections import deque
from datetime import datetime

class RecentContext:
    def __init__(self, max_items: int = 10):
        self.items = deque(maxlen=max_items)
    
    def add(self, content: str, metadata: dict = None):
        """Add item to recent context."""
        self.items.append({
            'content': content,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        })
    
    def get_recent(self, n: int = 5) -> list[dict]:
        """Get n most recent items."""
        return list(self.items)[-n:]
    
    def search(self, query: str) -> list[dict]:
        """Simple keyword search in recent context."""
        results = []
        query_lower = query.lower()
        
        for item in self.items:
            if query_lower in item['content'].lower():
                results.append(item)
        
        return results
```

### Short-Term Memory Management

```python
class ShortTermMemoryManager:
    def __init__(self, max_tokens: int = 4096):
        self.context_window = ShortTermMemory(max_tokens)
        self.working_memory = WorkingMemory()
        self.recent_context = RecentContext()
    
    def add_interaction(self, user_input: str, agent_output: str):
        """Add interaction to short-term memory."""
        # Add to context window
        self.context_window.add_message(
            Message(role='user', content=user_input)
        )
        self.context_window.add_message(
            Message(role='assistant', content=agent_output)
        )
        
        # Add to recent context
        self.recent_context.add(
            f"User: {user_input}\nAgent: {agent_output}",
            metadata={'type': 'interaction'}
        )
    
    def get_full_context(self) -> str:
        """Get complete short-term memory context."""
        parts = []
        
        # Working memory
        working_ctx = self.working_memory.to_context_string()
        if working_ctx:
            parts.append(working_ctx)
        
        # Recent context
        recent = self.recent_context.get_recent(3)
        if recent:
            parts.append("Recent Context:")
            for item in recent:
                parts.append(f"  - {item['content'][:100]}...")
        
        return "\n\n".join(parts)
```

## Long-Term Memory

Long-term memory (LTM) stores information persistently across sessions. It's the agent's "knowledge base."

### Characteristics

**Capacity**: Unlimited (practically constrained by storage)

**Duration**: Permanent (until explicitly deleted)

**Access Speed**: Requires retrieval (slower than STM)

**Volatility**: Persists across sessions and restarts

### Storage Backends

**1. Vector Databases**

For semantic search and similarity retrieval:

```python
from typing import Protocol
import numpy as np

class VectorStore(Protocol):
    def add(self, id: str, vector: np.ndarray, metadata: dict):
        """Add vector with metadata."""
        ...
    
    def search(self, query_vector: np.ndarray, top_k: int) -> list:
        """Search for similar vectors."""
        ...

class ChromaDBStore:
    def __init__(self, collection_name: str):
        import chromadb
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )
    
    def add(self, id: str, text: str, embedding: list[float], 
            metadata: dict = None):
        """Add document with embedding."""
        self.collection.add(
            ids=[id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata or {}]
        )
    
    def search(self, query_embedding: list[float], 
               top_k: int = 5) -> list[dict]:
        """Search for similar documents."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return [
            {
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'distance': results['distances'][0][i],
                'metadata': results['metadatas'][0][i]
            }
            for i in range(len(results['ids'][0]))
        ]
    
    def delete(self, id: str):
        """Delete document by id."""
        self.collection.delete(ids=[id])
```

**2. Relational Databases**

For structured data and relationships:

```python
import sqlite3
from datetime import datetime

class SQLMemoryStore:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
    
    def _create_tables(self):
        """Create memory tables."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                importance REAL DEFAULT 0.5,
                metadata TEXT
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                entity_type TEXT,
                attributes TEXT,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_type 
            ON memories(memory_type)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON memories(timestamp DESC)
        """)
        
        self.conn.commit()
    
    def add_memory(self, content: str, memory_type: str,
                   importance: float = 0.5, metadata: dict = None):
        """Add memory to database."""
        import json
        
        self.conn.execute("""
            INSERT INTO memories (content, memory_type, importance, metadata)
            VALUES (?, ?, ?, ?)
        """, (content, memory_type, importance, 
              json.dumps(metadata or {})))
        
        self.conn.commit()
    
    def get_recent_memories(self, memory_type: str = None,
                           limit: int = 10) -> list[dict]:
        """Get recent memories."""
        import json
        
        if memory_type:
            cursor = self.conn.execute("""
                SELECT id, content, memory_type, timestamp, 
                       importance, metadata
                FROM memories
                WHERE memory_type = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (memory_type, limit))
        else:
            cursor = self.conn.execute("""
                SELECT id, content, memory_type, timestamp,
                       importance, metadata
                FROM memories
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
        
        return [
            {
                'id': row[0],
                'content': row[1],
                'memory_type': row[2],
                'timestamp': row[3],
                'importance': row[4],
                'metadata': json.loads(row[5])
            }
            for row in cursor.fetchall()
        ]
    
    def add_entity(self, name: str, entity_type: str,
                   attributes: dict):
        """Add or update entity."""
        import json
        
        self.conn.execute("""
            INSERT OR REPLACE INTO entities 
            (name, entity_type, attributes, last_updated)
            VALUES (?, ?, ?, ?)
        """, (name, entity_type, json.dumps(attributes), 
              datetime.now().isoformat()))
        
        self.conn.commit()
    
    def get_entity(self, name: str) -> dict | None:
        """Get entity by name."""
        import json
        
        cursor = self.conn.execute("""
            SELECT name, entity_type, attributes, last_updated
            FROM entities
            WHERE name = ?
        """, (name,))
        
        row = cursor.fetchone()
        if row:
            return {
                'name': row[0],
                'entity_type': row[1],
                'attributes': json.loads(row[2]),
                'last_updated': row[3]
            }
        return None
```

**3. Document Stores**

For unstructured or semi-structured data:

```python
import json
from pathlib import Path

class DocumentStore:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    def save_document(self, doc_id: str, content: dict,
                     category: str = "general"):
        """Save document to file."""
        category_dir = self.base_path / category
        category_dir.mkdir(exist_ok=True)
        
        file_path = category_dir / f"{doc_id}.json"
        
        with open(file_path, 'w') as f:
            json.dump({
                'id': doc_id,
                'content': content,
                'timestamp': datetime.now().isoformat(),
                'category': category
            }, f, indent=2)
    
    def load_document(self, doc_id: str, 
                     category: str = "general") -> dict | None:
        """Load document from file."""
        file_path = self.base_path / category / f"{doc_id}.json"
        
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return None
    
    def list_documents(self, category: str = None) -> list[str]:
        """List document IDs."""
        if category:
            category_dir = self.base_path / category
            if category_dir.exists():
                return [
                    f.stem for f in category_dir.glob("*.json")
                ]
        else:
            doc_ids = []
            for category_dir in self.base_path.iterdir():
                if category_dir.is_dir():
                    doc_ids.extend([
                        f.stem for f in category_dir.glob("*.json")
                    ])
            return doc_ids
        return []
    
    def search_documents(self, query: str, 
                        category: str = None) -> list[dict]:
        """Simple keyword search across documents."""
        results = []
        query_lower = query.lower()
        
        search_dirs = (
            [self.base_path / category] if category
            else list(self.base_path.iterdir())
        )
        
        for dir_path in search_dirs:
            if not dir_path.is_dir():
                continue
            
            for file_path in dir_path.glob("*.json"):
                with open(file_path, 'r') as f:
                    doc = json.load(f)
                
                content_str = json.dumps(doc['content']).lower()
                if query_lower in content_str:
                    results.append(doc)
        
        return results
```

## Episodic Memory

Episodic memory stores experiences and events - the "what happened when" of agent life.

### Characteristics

**Content**: Temporal sequences, interaction history, experiences

**Organization**: Chronological, event-based

**Use Cases**: Learning from past attempts, maintaining context about what happened

### Structure

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Episode:
    """A single episode in agent memory."""
    id: str
    timestamp: datetime
    event_type: str
    content: dict
    outcome: Optional[str] = None
    importance: float = 0.5
    tags: list[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class EpisodicMemory:
    def __init__(self, storage: SQLMemoryStore):
        self.storage = storage
        self._create_episode_table()
    
    def _create_episode_table(self):
        """Create episodes table."""
        self.storage.conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                event_type TEXT NOT NULL,
                content TEXT NOT NULL,
                outcome TEXT,
                importance REAL DEFAULT 0.5,
                tags TEXT,
                session_id TEXT
            )
        """)
        
        self.storage.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episode_timestamp
            ON episodes(timestamp DESC)
        """)
        
        self.storage.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episode_type
            ON episodes(event_type)
        """)
        
        self.storage.conn.commit()
    
    def record_episode(self, episode: Episode):
        """Record an episode."""
        import json
        
        self.storage.conn.execute("""
            INSERT INTO episodes 
            (id, timestamp, event_type, content, outcome, 
             importance, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            episode.id,
            episode.timestamp.isoformat(),
            episode.event_type,
            json.dumps(episode.content),
            episode.outcome,
            episode.importance,
            json.dumps(episode.tags)
        ))
        
        self.storage.conn.commit()
    
    def get_recent_episodes(self, limit: int = 10,
                           event_type: str = None) -> list[Episode]:
        """Get recent episodes."""
        import json
        
        query = """
            SELECT id, timestamp, event_type, content, outcome,
                   importance, tags
            FROM episodes
        """
        
        if event_type:
            query += " WHERE event_type = ?"
            params = (event_type, limit)
        else:
            params = (limit,)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        
        cursor = self.storage.conn.execute(query, params)
        
        episodes = []
        for row in cursor.fetchall():
            episodes.append(Episode(
                id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                event_type=row[2],
                content=json.loads(row[3]),
                outcome=row[4],
                importance=row[5],
                tags=json.loads(row[6]) if row[6] else []
            ))
        
        return episodes
    
    def find_similar_episodes(self, event_type: str,
                             context: dict,
                             limit: int = 5) -> list[Episode]:
        """Find similar past episodes."""
        import json
        
        # Simple similarity based on event type and content overlap
        cursor = self.storage.conn.execute("""
            SELECT id, timestamp, event_type, content, outcome,
                   importance, tags
            FROM episodes
            WHERE event_type = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (event_type, limit * 3))  # Get more for filtering
        
        episodes = []
        context_str = json.dumps(context).lower()
        
        for row in cursor.fetchall():
            content = json.loads(row[3])
            content_str = json.dumps(content).lower()
            
            # Calculate simple overlap score
            overlap = sum(
                1 for word in context_str.split()
                if word in content_str
            )
            
            if overlap > 0:
                episodes.append((
                    overlap,
                    Episode(
                        id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        event_type=row[2],
                        content=content,
                        outcome=row[4],
                        importance=row[5],
                        tags=json.loads(row[6]) if row[6] else []
                    )
                ))
        
        # Sort by overlap and return top results
        episodes.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in episodes[:limit]]
```

### Using Episodic Memory

```python
class AgentWithEpisodicMemory:
    def __init__(self):
        self.storage = SQLMemoryStore("agent_memory.db")
        self.episodic = EpisodicMemory(self.storage)
        self.session_id = str(uuid.uuid4())
    
    def execute_task(self, task: str):
        """Execute task and record episode."""
        import uuid
        
        episode_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # Record task start
        self.episodic.record_episode(Episode(
            id=f"{episode_id}_start",
            timestamp=start_time,
            event_type="task_start",
            content={
                "task": task,
                "session": self.session_id
            },
            tags=["task", "start"]
        ))
        
        # Check for similar past episodes
        similar = self.episodic.find_similar_episodes(
            event_type="task_complete",
            context={"task": task},
            limit=3
        )
        
        if similar:
            print(f"Found {len(similar)} similar past experiences:")
            for ep in similar:
                print(f"  - {ep.timestamp}: {ep.outcome}")
        
        # Execute task (simplified)
        result = self._do_task(task)
        
        # Record task completion
        self.episodic.record_episode(Episode(
            id=f"{episode_id}_complete",
            timestamp=datetime.now(),
            event_type="task_complete",
            content={
                "task": task,
                "result": result,
                "duration": (datetime.now() - start_time).seconds,
                "session": self.session_id
            },
            outcome="success" if result else "failure",
            importance=0.7,
            tags=["task", "complete"]
        ))
        
        return result
    
    def _do_task(self, task: str) -> dict:
        """Placeholder for actual task execution."""
        return {"status": "completed", "output": "..."}
```

## Semantic Memory

Semantic memory stores factual knowledge and learned concepts - the "what I know" of agent understanding.

### Characteristics

**Content**: Facts, entities, relationships, domain knowledge

**Organization**: Structured (entities, attributes, relations)

**Use Cases**: Answering questions, understanding context, making informed decisions

### Knowledge Representation

```python
from dataclasses import dataclass
from typing import Set, Dict

@dataclass
class Entity:
    """An entity in semantic memory."""
    id: str
    name: str
    entity_type: str
    attributes: Dict[str, any]
    created_at: datetime
    updated_at: datetime

@dataclass
class Relation:
    """A relationship between entities."""
    subject_id: str
    predicate: str
    object_id: str
    confidence: float = 1.0
    source: str = "learned"

class SemanticMemory:
    def __init__(self, storage: SQLMemoryStore):
        self.storage = storage
        self._create_semantic_tables()
    
    def _create_semantic_tables(self):
        """Create tables for semantic memory."""
        # Entities table already created in SQLMemoryStore
        
        self.storage.conn.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_id TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.storage.conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                statement TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                tags TEXT
            )
        """)
        
        self.storage.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_subject
            ON relations(subject_id)
        """)
        
        self.storage.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_object
            ON relations(object_id)
        """)
        
        self.storage.conn.commit()
    
    def add_entity(self, entity: Entity):
        """Add entity to semantic memory."""
        import json
        
        self.storage.conn.execute("""
            INSERT OR REPLACE INTO entities
            (name, entity_type, attributes, last_updated)
            VALUES (?, ?, ?, ?)
        """, (
            entity.name,
            entity.entity_type,
            json.dumps(entity.attributes),
            entity.updated_at.isoformat()
        ))
        
        self.storage.conn.commit()
    
    def add_relation(self, relation: Relation):
        """Add relation between entities."""
        self.storage.conn.execute("""
            INSERT INTO relations
            (subject_id, predicate, object_id, confidence, source)
            VALUES (?, ?, ?, ?, ?)
        """, (
            relation.subject_id,
            relation.predicate,
            relation.object_id,
            relation.confidence,
            relation.source
        ))
        
        self.storage.conn.commit()
    
    def add_fact(self, statement: str, confidence: float = 1.0,
                 source: str = "learned", tags: list[str] = None):
        """Add a fact to semantic memory."""
        import json
        
        self.storage.conn.execute("""
            INSERT INTO facts
            (statement, confidence, source, tags)
            VALUES (?, ?, ?, ?)
        """, (
            statement,
            confidence,
            source,
            json.dumps(tags or [])
        ))
        
        self.storage.conn.commit()
    
    def get_entity_relations(self, entity_id: str) -> list[Relation]:
        """Get all relations involving an entity."""
        cursor = self.storage.conn.execute("""
            SELECT subject_id, predicate, object_id, confidence, source
            FROM relations
            WHERE subject_id = ? OR object_id = ?
        """, (entity_id, entity_id))
        
        return [
            Relation(
                subject_id=row[0],
                predicate=row[1],
                object_id=row[2],
                confidence=row[3],
                source=row[4]
            )
            for row in cursor.fetchall()
        ]
    
    def query_facts(self, query: str, min_confidence: float = 0.5) -> list[dict]:
        """Query facts by keyword."""
        import json
        
        cursor = self.storage.conn.execute("""
            SELECT statement, confidence, source, tags, created_at
            FROM facts
            WHERE statement LIKE ? AND confidence >= ?
            ORDER BY confidence DESC
        """, (f"%{query}%", min_confidence))
        
        return [
            {
                'statement': row[0],
                'confidence': row[1],
                'source': row[2],
                'tags': json.loads(row[3]),
                'created_at': row[4]
            }
            for row in cursor.fetchall()
        ]
    
    def build_knowledge_graph(self, entity_ids: list[str]) -> dict:
        """Build knowledge graph for given entities."""
        graph = {
            'entities': {},
            'relations': []
        }
        
        # Get entities
        for entity_id in entity_ids:
            entity = self.storage.get_entity(entity_id)
            if entity:
                graph['entities'][entity_id] = entity
        
        # Get relations
        for entity_id in entity_ids:
            relations = self.get_entity_relations(entity_id)
            for rel in relations:
                if (rel.subject_id in entity_ids and 
                    rel.object_id in entity_ids):
                    graph['relations'].append({
                        'subject': rel.subject_id,
                        'predicate': rel.predicate,
                        'object': rel.object_id,
                        'confidence': rel.confidence
                    })
        
        return graph
```

### Using Semantic Memory

```python
class AgentWithSemanticMemory:
    def __init__(self):
        self.storage = SQLMemoryStore("agent_memory.db")
        self.semantic = SemanticMemory(self.storage)
    
    def learn_about_user(self, user_id: str, facts: dict):
        """Learn facts about a user."""
        # Create/update user entity
        user_entity = Entity(
            id=user_id,
            name=facts.get('name', user_id),
            entity_type='user',
            attributes=facts,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.semantic.add_entity(user_entity)
        
        # Add specific facts
        if 'occupation' in facts:
            self.semantic.add_fact(
                f"{facts['name']} works as a {facts['occupation']}",
                confidence=0.9,
                source="user_provided",
                tags=['user', 'occupation']
            )
        
        if 'interests' in facts:
            for interest in facts['interests']:
                self.semantic.add_relation(Relation(
                    subject_id=user_id,
                    predicate="interested_in",
                    object_id=interest,
                    confidence=0.8,
                    source="user_provided"
                ))
    
    def answer_question(self, question: str) -> str:
        """Answer question using semantic memory."""
        # Query facts
        relevant_facts = self.semantic.query_facts(question)
        
        if relevant_facts:
            # Use facts to answer
            context = "\n".join([
                f"- {fact['statement']} (confidence: {fact['confidence']})"
                for fact in relevant_facts[:3]
            ])
            
            return f"Based on what I know:\n{context}"
        else:
            return "I don't have information about that."
```

## Memory Hierarchies

Memory systems are organized hierarchically, with different levels optimized for different access patterns.

### Three-Level Hierarchy

```
┌─────────────────────────────────────────────────────┐
│ L1: Context Window (Immediate Access)               │
│                                                      │
│  - Current conversation                             │
│  - Active task state                                │
│  - Recent tool outputs                              │
│                                                      │
│  Access: 0ms (in-context)                           │
│  Capacity: 4K-200K tokens                           │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│ L2: Session Memory (Fast Access)                    │
│                                                      │
│  - Recent interactions (this session)               │
│  - Working variables                                │
│  - Temporary context                                │
│                                                      │
│  Access: <100ms (in-memory)                         │
│  Capacity: Limited by RAM                           │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│ L3: Persistent Memory (Slower Access)               │
│                                                      │
│  - All past interactions                            │
│  - Long-term knowledge                              │
│  - Learned patterns                                 │
│                                                      │
│  Access: 10-1000ms (database/disk)                  │
│  Capacity: Unlimited (disk-bound)                   │
└──────────────────────────────────────────────────────┘
```

### Implementing Memory Hierarchy

```python
class HierarchicalMemory:
    def __init__(self, max_context_tokens: int = 4096):
        # L1: Context window
        self.l1_context = ShortTermMemory(max_context_tokens)
        
        # L2: Session memory (in-memory)
        self.l2_session = {
            'interactions': [],
            'variables': {},
            'facts': []
        }
        
        # L3: Persistent storage
        self.l3_storage = SQLMemoryStore("agent_memory.db")
        self.l3_vector = ChromaDBStore("agent_vectors")
    
    def add_interaction(self, user_input: str, agent_output: str):
        """Add interaction across all levels."""
        message_user = Message(role='user', content=user_input)
        message_agent = Message(role='assistant', content=agent_output)
        
        # L1: Add to context window
        self.l1_context.add_message(message_user)
        self.l1_context.add_message(message_agent)
        
        # L2: Add to session memory
        self.l2_session['interactions'].append({
            'user': user_input,
            'agent': agent_output,
            'timestamp': datetime.now()
        })
        
        # L3: Persist to database
        self.l3_storage.add_memory(
            content=f"User: {user_input}\nAgent: {agent_output}",
            memory_type='interaction',
            importance=self._calculate_importance(user_input, agent_output)
        )
    
    def retrieve_context(self, query: str, max_items: int = 5) -> dict:
        """Retrieve relevant context from all levels."""
        context = {
            'l1_immediate': [],
            'l2_session': [],
            'l3_historical': []
        }
        
        # L1: Current context (always included)
        context['l1_immediate'] = self.l1_context.get_context()
        
        # L2: Search session memory
        for interaction in self.l2_session['interactions'][-10:]:
            if query.lower() in interaction['user'].lower():
                context['l2_session'].append(interaction)
        
        # L3: Semantic search in persistent memory
        # (simplified - would use embeddings in practice)
        recent_memories = self.l3_storage.get_recent_memories(
            memory_type='interaction',
            limit=max_items
        )
        context['l3_historical'] = recent_memories
        
        return context
    
    def _calculate_importance(self, user_input: str, 
                             agent_output: str) -> float:
        """Calculate importance score for memory."""
        # Simple heuristic: longer = more important
        total_length = len(user_input) + len(agent_output)
        
        # Normalize to 0-1
        importance = min(1.0, total_length / 1000)
        
        return importance
```

## Memory Interactions

Different memory types interact and influence each other.

### Memory Flow Patterns

**1. Short-Term to Long-Term (Consolidation)**

```python
class MemoryConsolidation:
    def __init__(self, hierarchical_memory: HierarchicalMemory):
        self.memory = hierarchical_memory
    
    def consolidate_session(self):
        """Move important session memories to long-term storage."""
        # Get all session interactions
        interactions = self.memory.l2_session['interactions']
        
        # Identify important interactions
        important = [
            interaction for interaction in interactions
            if self._is_important(interaction)
        ]
        
        # Persist to long-term memory
        for interaction in important:
            self.memory.l3_storage.add_memory(
                content=f"User: {interaction['user']}\n"
                       f"Agent: {interaction['agent']}",
                memory_type='consolidated_interaction',
                importance=0.8,
                metadata={
                    'timestamp': interaction['timestamp'].isoformat(),
                    'session_consolidated': True
                }
            )
    
    def _is_important(self, interaction: dict) -> bool:
        """Determine if interaction is important enough to consolidate."""
        # Heuristics for importance
        user_text = interaction['user']
        agent_text = interaction['agent']
        
        # Long interactions
        if len(user_text) + len(agent_text) > 500:
            return True
        
        # Contains key phrases
        key_phrases = ['remember', 'important', 'always', 'never']
        if any(phrase in user_text.lower() for phrase in key_phrases):
            return True
        
        # Decision/commitment made
        if 'will' in agent_text.lower() or 'should' in agent_text.lower():
            return True
        
        return False
```

**2. Long-Term to Short-Term (Retrieval)**

```python
class MemoryRetrieval:
    def __init__(self, hierarchical_memory: HierarchicalMemory):
        self.memory = hierarchical_memory
    
    def augment_context(self, current_input: str) -> list[Message]:
        """Retrieve relevant long-term memories and add to context."""
        # Get current context
        current_context = self.memory.l1_context.get_context()
        
        # Search long-term memory
        relevant_memories = self._search_long_term(current_input)
        
        # Create augmented context
        augmented_context = current_context.copy()
        
        if relevant_memories:
            # Add retrieved memories as system message
            memory_text = "Relevant past context:\n" + "\n".join([
                f"- {mem['content'][:200]}..."
                for mem in relevant_memories[:3]
            ])
            
            augmented_context.insert(1, Message(
                role='system',
                content=memory_text
            ))
        
        return augmented_context
    
    def _search_long_term(self, query: str) -> list[dict]:
        """Search long-term memory for relevant information."""
        # This would use vector search in practice
        memories = self.memory.l3_storage.get_recent_memories(limit=20)
        
        # Simple keyword matching
        query_lower = query.lower()
        relevant = [
            mem for mem in memories
            if any(word in mem['content'].lower() 
                   for word in query_lower.split())
        ]
        
        # Sort by importance
        relevant.sort(key=lambda m: m['importance'], reverse=True)
        
        return relevant[:5]
```

**3. Episodic to Semantic (Learning)**

```python
class EpisodicToSemanticLearning:
    def __init__(self, episodic: EpisodicMemory, 
                 semantic: SemanticMemory):
        self.episodic = episodic
        self.semantic = semantic
    
    def extract_patterns(self, event_type: str, min_occurrences: int = 3):
        """Extract patterns from repeated episodes."""
        # Get all episodes of this type
        episodes = self.episodic.get_recent_episodes(
            event_type=event_type,
            limit=100
        )
        
        # Group by outcome
        outcomes = {}
        for episode in episodes:
            outcome = episode.outcome or 'unknown'
            if outcome not in outcomes:
                outcomes[outcome] = []
            outcomes[outcome].append(episode)
        
        # Extract patterns from successful episodes
        if 'success' in outcomes and len(outcomes['success']) >= min_occurrences:
            pattern = self._find_common_pattern(outcomes['success'])
            
            # Store as semantic knowledge
            self.semantic.add_fact(
                statement=f"Pattern for {event_type}: {pattern}",
                confidence=len(outcomes['success']) / len(episodes),
                source="learned_from_experience",
                tags=[event_type, 'pattern', 'learned']
            )
    
    def _find_common_pattern(self, episodes: list[Episode]) -> str:
        """Find common elements in successful episodes."""
        # Simplified pattern detection
        common_keys = set()
        
        for episode in episodes:
            if common_keys:
                common_keys &= set(episode.content.keys())
            else:
                common_keys = set(episode.content.keys())
        
        if common_keys:
            pattern_elements = [
                f"{key} is important"
                for key in common_keys
            ]
            return "; ".join(pattern_elements)
        
        return "No clear pattern identified"
```

## Architectural Patterns

Common patterns for organizing memory systems.

### Pattern 1: Unified Memory Interface

```python
from abc import ABC, abstractmethod

class MemoryInterface(ABC):
    """Abstract interface for memory operations."""
    
    @abstractmethod
    def store(self, key: str, value: any, memory_type: str):
        """Store information in memory."""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, memory_type: str = None) -> list:
        """Retrieve information from memory."""
        pass
    
    @abstractmethod
    def forget(self, key: str, memory_type: str = None):
        """Remove information from memory."""
        pass

class UnifiedMemorySystem(MemoryInterface):
    """Unified interface to all memory types."""
    
    def __init__(self):
        self.short_term = ShortTermMemoryManager()
        self.episodic = EpisodicMemory(SQLMemoryStore("memory.db"))
        self.semantic = SemanticMemory(SQLMemoryStore("memory.db"))
        self.vector_store = ChromaDBStore("vectors")
    
    def store(self, key: str, value: any, memory_type: str):
        """Store in appropriate memory system."""
        if memory_type == "short_term":
            self.short_term.working_memory.set(key, value)
        
        elif memory_type == "episodic":
            self.episodic.record_episode(value)  # value is Episode
        
        elif memory_type == "semantic":
            if isinstance(value, Entity):
                self.semantic.add_entity(value)
            elif isinstance(value, Relation):
                self.semantic.add_relation(value)
            else:
                self.semantic.add_fact(str(value))
        
        elif memory_type == "vector":
            # Assuming value is (text, embedding) tuple
            text, embedding = value
            self.vector_store.add(key, text, embedding)
    
    def retrieve(self, query: str, memory_type: str = None) -> list:
        """Retrieve from specified or all memory types."""
        results = []
        
        if memory_type is None or memory_type == "short_term":
            # Get from short-term memory
            recent = self.short_term.recent_context.search(query)
            results.extend([('short_term', r) for r in recent])
        
        if memory_type is None or memory_type == "episodic":
            # Search episodes
            # (simplified - would use better search)
            episodes = self.episodic.get_recent_episodes(limit=10)
            relevant_episodes = [
                ep for ep in episodes
                if query.lower() in str(ep.content).lower()
            ]
            results.extend([('episodic', ep) for ep in relevant_episodes])
        
        if memory_type is None or memory_type == "semantic":
            # Search facts
            facts = self.semantic.query_facts(query)
            results.extend([('semantic', f) for f in facts])
        
        return results
    
    def forget(self, key: str, memory_type: str = None):
        """Remove from memory."""
        if memory_type == "short_term":
            self.short_term.working_memory.variables.pop(key, None)
        
        elif memory_type == "vector":
            self.vector_store.delete(key)
        
        # Episodic and semantic deletion would require more logic
```

### Pattern 2: Memory Manager

```python
class MemoryManager:
    """Centralized memory management."""
    
    def __init__(self):
        self.unified = UnifiedMemorySystem()
        self.consolidation = MemoryConsolidation(None)
        self.retrieval = MemoryRetrieval(None)
    
    def process_interaction(self, user_input: str, agent_output: str):
        """Process an interaction across all memory systems."""
        # Store in short-term
        self.unified.store(
            key='latest_interaction',
            value={'user': user_input, 'agent': agent_output},
            memory_type='short_term'
        )
        
        # Create episode
        episode = Episode(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            event_type='interaction',
            content={'user': user_input, 'agent': agent_output},
            importance=self._score_importance(user_input, agent_output)
        )
        self.unified.store(
            key=episode.id,
            value=episode,
            memory_type='episodic'
        )
        
        # Extract entities and facts
        entities = self._extract_entities(user_input + " " + agent_output)
        for entity in entities:
            self.unified.store(
                key=entity.id,
                value=entity,
                memory_type='semantic'
            )
    
    def get_relevant_context(self, query: str) -> str:
        """Get relevant context for a query."""
        # Retrieve from all memory types
        results = self.unified.retrieve(query)
        
        # Organize by type and importance
        context_parts = []
        
        # Add short-term context
        short_term = [r for r in results if r[0] == 'short_term']
        if short_term:
            context_parts.append("Recent context:")
            for _, item in short_term[:3]:
                context_parts.append(f"  - {item['content'][:100]}...")
        
        # Add relevant episodes
        episodes = [r for r in results if r[0] == 'episodic']
        if episodes:
            context_parts.append("\nRelevant past experiences:")
            for _, episode in episodes[:2]:
                context_parts.append(
                    f"  - {episode.timestamp}: {episode.outcome}"
                )
        
        # Add semantic knowledge
        facts = [r for r in results if r[0] == 'semantic']
        if facts:
            context_parts.append("\nRelevant knowledge:")
            for _, fact in facts[:3]:
                context_parts.append(f"  - {fact['statement']}")
        
        return "\n".join(context_parts)
    
    def _score_importance(self, user_input: str, agent_output: str) -> float:
        """Score importance of interaction."""
        # Simple heuristic
        score = 0.5
        
        # Longer interactions are more important
        length = len(user_input) + len(agent_output)
        score += min(0.3, length / 2000)
        
        # Key phrases increase importance
        key_phrases = ['remember', 'important', 'always', 'never', 
                      'prefer', 'like', 'dislike']
        for phrase in key_phrases:
            if phrase in user_input.lower():
                score += 0.1
        
        return min(1.0, score)
    
    def _extract_entities(self, text: str) -> list[Entity]:
        """Extract entities from text (simplified)."""
        # In practice, would use NER
        entities = []
        
        # Simple pattern matching for demo
        import re
        
        # Find potential names (capitalized words)
        names = re.findall(r'\b[A-Z][a-z]+\b', text)
        
        for name in set(names):
            entities.append(Entity(
                id=name.lower(),
                name=name,
                entity_type='unknown',
                attributes={'mentioned_in': text[:50]},
                created_at=datetime.now(),
                updated_at=datetime.now()
            ))
        
        return entities
```

## Memory System Design

Key design considerations for agent memory systems.

### Design Principles

**1. Separation of Concerns**

Different memory types serve different purposes:

```python
class MemorySystemDesign:
    """Example of well-separated memory design."""
    
    def __init__(self):
        # Each memory type is independent
        self.context = ShortTermMemory()      # For immediate context
        self.episodes = EpisodicMemory()       # For experiences
        self.knowledge = SemanticMemory()      # For facts
        self.vectors = VectorStore()           # For semantic search
    
    # Each can be modified independently
    def upgrade_vector_store(self, new_store: VectorStore):
        """Swap vector store implementation without affecting others."""
        self.vectors = new_store
```

**2. Clear Data Flow**

Information should flow predictably:

```
Input → Short-Term → [Consolidation] → Long-Term
                ↓                          ↓
              Use Immediately        Retrieve When Needed
```

**3. Graceful Degradation**

System should work even if some components fail:

```python
class ResilientMemorySystem:
    def retrieve(self, query: str) -> list:
        """Retrieve with fallback."""
        try:
            # Try vector search first (best results)
            return self.vector_store.search(query)
        except Exception as e:
            print(f"Vector search failed: {e}")
            
            try:
                # Fall back to SQL search
                return self.sql_store.search(query)
            except Exception as e:
                print(f"SQL search failed: {e}")
                
                # Last resort: return short-term memory
                return self.short_term.get_recent()
```

### Memory Configuration

```python
@dataclass
class MemoryConfig:
    """Configuration for memory system."""
    
    # Short-term memory
    context_window_tokens: int = 4096
    working_memory_size: int = 100
    
    # Long-term memory
    persist_threshold: float = 0.6  # Min importance to persist
    consolidation_interval: int = 10  # Consolidate every N interactions
    
    # Retrieval
    max_retrieval_results: int = 5
    recency_weight: float = 0.3
    relevance_weight: float = 0.5
    importance_weight: float = 0.2
    
    # Storage
    vector_db_path: str = "vectors/"
    sql_db_path: str = "memory.db"
    document_store_path: str = "documents/"

class ConfigurableMemorySystem:
    def __init__(self, config: MemoryConfig):
        self.config = config
        
        # Initialize components based on config
        self.short_term = ShortTermMemory(
            max_tokens=config.context_window_tokens
        )
        
        self.sql_store = SQLMemoryStore(config.sql_db_path)
        self.vector_store = ChromaDBStore(config.vector_db_path)
        self.doc_store = DocumentStore(config.document_store_path)
        
        self.interaction_count = 0
    
    def should_persist(self, importance: float) -> bool:
        """Check if memory should be persisted."""
        return importance >= self.config.persist_threshold
    
    def should_consolidate(self) -> bool:
        """Check if consolidation should run."""
        return (self.interaction_count % 
                self.config.consolidation_interval == 0)
```

## Implementation Considerations

### Token Budget Management

```python
class TokenBudgetManager:
    """Manage token budget across memory components."""
    
    def __init__(self, total_budget: int):
        self.total_budget = total_budget
        self.allocations = {
            'system_prompt': 0.2,      # 20% for system prompt
            'current_input': 0.2,       # 20% for current input
            'conversation_history': 0.3, # 30% for history
            'retrieved_memory': 0.2,    # 20% for retrieved memories
            'output_buffer': 0.1        # 10% for output
        }
    
    def get_allocation(self, component: str) -> int:
        """Get token allocation for component."""
        return int(self.total_budget * self.allocations[component])
    
    def build_context(self, components: dict) -> list[Message]:
        """Build context respecting token budget."""
        context = []
        
        # System prompt (fixed)
        system_tokens = self.get_allocation('system_prompt')
        context.append(Message(
            role='system',
            content=truncate_to_tokens(
                components['system_prompt'],
                system_tokens
            )
        ))
        
        # Retrieved memory
        memory_tokens = self.get_allocation('retrieved_memory')
        if 'retrieved_memory' in components:
            memory_text = self._format_memories(
                components['retrieved_memory'],
                memory_tokens
            )
            if memory_text:
                context.append(Message(
                    role='system',
                    content=f"Relevant context:\n{memory_text}"
                ))
        
        # Conversation history
        history_tokens = self.get_allocation('conversation_history')
        history = components.get('conversation_history', [])
        context.extend(
            self._fit_history(history, history_tokens)
        )
        
        # Current input
        input_tokens = self.get_allocation('current_input')
        context.append(Message(
            role='user',
            content=truncate_to_tokens(
                components['current_input'],
                input_tokens
            )
        ))
        
        return context
    
    def _format_memories(self, memories: list, max_tokens: int) -> str:
        """Format memories to fit token budget."""
        formatted = []
        tokens_used = 0
        
        for memory in memories:
            memory_text = f"- {memory['content']}"
            memory_tokens = count_tokens(memory_text)
            
            if tokens_used + memory_tokens > max_tokens:
                break
            
            formatted.append(memory_text)
            tokens_used += memory_tokens
        
        return "\n".join(formatted)
    
    def _fit_history(self, history: list[Message], 
                    max_tokens: int) -> list[Message]:
        """Fit conversation history in token budget."""
        fitted = []
        tokens_used = 0
        
        # Add messages from most recent backwards
        for message in reversed(history):
            msg_tokens = count_tokens(message.content)
            
            if tokens_used + msg_tokens > max_tokens:
                break
            
            fitted.insert(0, message)
            tokens_used += msg_tokens
        
        return fitted

def count_tokens(text: str) -> int:
    """Count tokens in text (simplified)."""
    # In practice, use actual tokenizer
    return len(text.split())

def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to token limit."""
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens]) + "..."
```

### Embedding Management

```python
class EmbeddingManager:
    """Manage embeddings for vector search."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.cache = {}
    
    def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        # Check cache
        if text in self.cache:
            return self.cache[text]
        
        # Generate embedding
        embedding = self.model.encode(text).tolist()
        
        # Cache for reuse
        self.cache[text] = embedding
        
        return embedding
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        # Filter out cached
        uncached = [t for t in texts if t not in self.cache]
        
        if uncached:
            # Batch encode uncached texts
            embeddings = self.model.encode(uncached)
            
            # Update cache
            for text, embedding in zip(uncached, embeddings):
                self.cache[text] = embedding.tolist()
        
        # Return all embeddings in order
        return [self.cache[t] for t in texts]
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        
        # Cosine similarity
        import numpy as np
        return np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2)
        )
```

## Performance and Scaling

### Caching Strategies

```python
from functools import lru_cache
from typing import Optional
import time

class CachedMemorySystem:
    """Memory system with caching."""
    
    def __init__(self):
        self.storage = SQLMemoryStore("memory.db")
        self.cache_ttl = 300  # 5 minutes
        self.cache = {}
        self.cache_timestamps = {}
    
    def get_memory(self, key: str, memory_type: str) -> Optional[dict]:
        """Get memory with caching."""
        cache_key = f"{memory_type}:{key}"
        
        # Check cache
        if cache_key in self.cache:
            if time.time() - self.cache_timestamps[cache_key] < self.cache_ttl:
                return self.cache[cache_key]
        
        # Fetch from storage
        memory = self._fetch_from_storage(key, memory_type)
        
        # Update cache
        if memory:
            self.cache[cache_key] = memory
            self.cache_timestamps[cache_key] = time.time()
        
        return memory
    
    def _fetch_from_storage(self, key: str, memory_type: str) -> Optional[dict]:
        """Fetch from actual storage."""
        # Implementation depends on memory type
        return None
    
    def invalidate_cache(self, key: str, memory_type: str):
        """Invalidate cache entry."""
        cache_key = f"{memory_type}:{key}"
        self.cache.pop(cache_key, None)
        self.cache_timestamps.pop(cache_key, None)
```

### Batch Operations

```python
class BatchMemoryOperations:
    """Optimize memory operations with batching."""
    
    def __init__(self, storage: SQLMemoryStore):
        self.storage = storage
        self.batch_size = 100
        self.pending_writes = []
    
    def queue_write(self, memory: dict):
        """Queue memory for batch write."""
        self.pending_writes.append(memory)
        
        if len(self.pending_writes) >= self.batch_size:
            self.flush()
    
    def flush(self):
        """Write all pending memories."""
        if not self.pending_writes:
            return
        
        # Batch insert
        self.storage.conn.executemany("""
            INSERT INTO memories (content, memory_type, importance, metadata)
            VALUES (?, ?, ?, ?)
        """, [
            (m['content'], m['memory_type'], 
             m['importance'], json.dumps(m.get('metadata', {})))
            for m in self.pending_writes
        ])
        
        self.storage.conn.commit()
        self.pending_writes.clear()
    
    def __del__(self):
        """Ensure all writes are flushed on cleanup."""
        self.flush()
```

### Indexing and Optimization

```python
class OptimizedStorage:
    """Optimized storage with proper indexing."""
    
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._optimize_database()
    
    def _optimize_database(self):
        """Create indexes and optimize database."""
        # Create indexes for common queries
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance DESC)",
            "CREATE INDEX IF NOT EXISTS idx_type_time ON memories(memory_type, timestamp DESC)",
        ]
        
        for index in indexes:
            self.conn.execute(index)
        
        # Analyze for query optimization
        self.conn.execute("ANALYZE")
        
        self.conn.commit()
    
    def optimize_query(self, query: str, params: tuple) -> list:
        """Execute optimized query."""
        # Use query plan analysis
        explain = self.conn.execute(f"EXPLAIN QUERY PLAN {query}", params)
        
        # Log query plan for debugging
        # print("Query plan:", explain.fetchall())
        
        # Execute actual query
        cursor = self.conn.execute(query, params)
        return cursor.fetchall()
```

## Memory in Different Agent Types

Different agent architectures have different memory needs.

### ReAct Agents

```python
class ReActAgentMemory:
    """Memory system optimized for ReAct agents."""
    
    def __init__(self):
        self.thought_history = []
        self.action_history = []
        self.episodic = EpisodicMemory(SQLMemoryStore("react.db"))
    
    def record_thought(self, thought: str):
        """Record reasoning step."""
        self.thought_history.append({
            'thought': thought,
            'timestamp': datetime.now()
        })
    
    def record_action(self, action: str, result: any):
        """Record action and result."""
        self.action_history.append({
            'action': action,
            'result': result,
            'timestamp': datetime.now()
        })
        
        # Create episode from thought-action pair
        if self.thought_history:
            last_thought = self.thought_history[-1]
            
            episode = Episode(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                event_type='thought_action',
                content={
                    'thought': last_thought['thought'],
                    'action': action,
                    'result': str(result)
                },
                importance=0.7
            )
            
            self.episodic.record_episode(episode)
    
    def get_reasoning_context(self) -> str:
        """Get recent reasoning for context."""
        if not self.thought_history:
            return ""
        
        recent = self.thought_history[-3:]
        
        context = ["Recent reasoning:"]
        for item in recent:
            context.append(f"- {item['thought']}")
        
        return "\n".join(context)
```

### Conversational Agents

```python
class ConversationalAgentMemory:
    """Memory for conversational agents."""
    
    def __init__(self):
        self.conversation_history = []
        self.user_profile = {}
        self.semantic = SemanticMemory(SQLMemoryStore("conversation.db"))
    
    def add_turn(self, user_input: str, agent_response: str):
        """Add conversation turn."""
        self.conversation_history.append({
            'user': user_input,
            'agent': agent_response,
            'timestamp': datetime.now()
        })
        
        # Extract and store user information
        self._update_user_profile(user_input)
    
    def _update_user_profile(self, user_input: str):
        """Extract user information from input."""
        # Simple pattern matching (would use NLU in practice)
        if "my name is" in user_input.lower():
            import re
            match = re.search(r'my name is (\w+)', user_input, re.I)
            if match:
                name = match.group(1)
                self.user_profile['name'] = name
                
                # Store in semantic memory
                self.semantic.add_fact(
                    f"User's name is {name}",
                    confidence=0.9,
                    source="user_provided"
                )
    
    def get_conversation_context(self, max_turns: int = 5) -> str:
        """Get recent conversation context."""
        recent = self.conversation_history[-max_turns:]
        
        context = []
        for turn in recent:
            context.append(f"User: {turn['user']}")
            context.append(f"Agent: {turn['agent']}")
        
        return "\n".join(context)
    
    def get_user_context(self) -> str:
        """Get user profile context."""
        if not self.user_profile:
            return ""
        
        parts = ["User information:"]
        for key, value in self.user_profile.items():
            parts.append(f"- {key}: {value}")
        
        return "\n".join(parts)
```

### Task-Oriented Agents

```python
class TaskOrientedAgentMemory:
    """Memory for task-oriented agents."""
    
    def __init__(self):
        self.current_task = None
        self.task_history = []
        self.episodic = EpisodicMemory(SQLMemoryStore("tasks.db"))
    
    def start_task(self, task: dict):
        """Start a new task."""
        self.current_task = {
            'id': str(uuid.uuid4()),
            'description': task,
            'start_time': datetime.now(),
            'steps': [],
            'status': 'in_progress'
        }
    
    def record_step(self, step: dict):
        """Record a task step."""
        if self.current_task:
            self.current_task['steps'].append({
                'step': step,
                'timestamp': datetime.now()
            })
    
    def complete_task(self, outcome: dict):
        """Complete current task."""
        if not self.current_task:
            return
        
        self.current_task['end_time'] = datetime.now()
        self.current_task['outcome'] = outcome
        self.current_task['status'] = 'completed'
        
        # Store as episode
        episode = Episode(
            id=self.current_task['id'],
            timestamp=self.current_task['end_time'],
            event_type='task_completion',
            content=self.current_task,
            outcome='success' if outcome.get('success') else 'failure',
            importance=0.8
        )
        
        self.episodic.record_episode(episode)
        
        # Move to history
        self.task_history.append(self.current_task)
        self.current_task = None
    
    def find_similar_tasks(self, task_description: str) -> list[Episode]:
        """Find similar past tasks."""
        return self.episodic.find_similar_episodes(
            event_type='task_completion',
            context={'description': task_description},
            limit=3
        )
    
    def get_task_context(self) -> str:
        """Get current task context."""
        if not self.current_task:
            return "No active task"
        
        context = [
            f"Current task: {self.current_task['description']}",
            f"Started: {self.current_task['start_time']}",
            f"Steps completed: {len(self.current_task['steps'])}"
        ]
        
        if self.current_task['steps']:
            context.append("\nRecent steps:")
            for step in self.current_task['steps'][-3:]:
                context.append(f"- {step['step']}")
        
        return "\n".join(context)
```

## Summary

Memory architecture is fundamental to agent capability:

**Key Concepts**:

- **Short-term memory**: Immediate context and working memory
- **Long-term memory**: Persistent storage across sessions
- **Episodic memory**: Experiences and temporal sequences
- **Semantic memory**: Facts, entities, and knowledge
- **Memory hierarchies**: Multi-level organization for efficiency
- **Memory interactions**: How different memory types work together

**Implementation Strategies**:

- **Unified interfaces**: Abstract memory operations
- **Multiple backends**: Vector, SQL, document stores
- **Token management**: Fit relevant memory in context windows
- **Caching and optimization**: Performance at scale
- **Graceful degradation**: Resilient to component failures

**Design Principles**:

- Separate concerns (different memory types for different purposes)
- Clear data flow (predictable information movement)
- Appropriate persistence (not everything needs long-term storage)
- Efficient retrieval (balance relevance, recency, importance)
- Continuous learning (improve from experience)

## Next Steps

- **[Short-Term Memory](short-term-memory.md)**: Deep dive into working memory and context management
- **[Long-Term Memory](long-term-memory.md)**: Persistent storage systems and strategies
- **[Memory Retrieval](memory-retrieval.md)**: Finding relevant information efficiently
- **[Memory Consolidation](memory-consolidation.md)**: Moving information from short-term to long-term storage
