# Long-Term Memory

## Table of Contents

- [Introduction](#introduction)
- [Persistent Storage Systems](#persistent-storage-systems)
- [Vector Databases](#vector-databases)
- [Relational Databases](#relational-databases)
- [Document Stores](#document-stores)
- [Graph Databases](#graph-databases)
- [Hybrid Approaches](#hybrid-approaches)
- [When to Persist](#when-to-persist)
- [Retrieval Strategies](#retrieval-strategies)
- [Indexing and Optimization](#indexing-and-optimization)
- [Scaling Long-Term Memory](#scaling-long-term-memory)
- [Data Lifecycle Management](#data-lifecycle-management)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Long-term memory (LTM) is where agents store information that persists beyond individual conversations and sessions. While short-term memory is limited and volatile, long-term memory is essentially unlimited and permanent.

> "Long-term memory is what separates a capable agent from a continuously improving one."

Long-term memory enables agents to:

- **Remember past interactions** across sessions
- **Learn from experience** and improve over time
- **Build knowledge** about users, domains, and tasks
- **Maintain consistency** in behavior and responses
- **Scale beyond** context window limitations

### Key Characteristics

**Persistence**: Survives agent restarts and session boundaries

**Capacity**: Essentially unlimited (constrained only by storage)

**Access Speed**: Slower than short-term memory (requires retrieval)

**Organization**: Must be structured for efficient retrieval

**Maintenance**: Requires indexing, cleanup, and optimization

### Storage Requirements

Different types of information require different storage solutions:

```
┌─────────────────────────────────────────────────────┐
│ Information Type    │ Best Storage     │ Access     │
├─────────────────────┼──────────────────┼────────────┤
│ Semantic search     │ Vector DB        │ Similarity │
│ Structured data     │ SQL Database     │ Queries    │
│ Documents/logs      │ Document Store   │ Key/search │
│ Relationships       │ Graph Database   │ Traversal  │
│ Simple key-value    │ Redis/Dict       │ Direct     │
└─────────────────────────────────────────────────────┘
```

## Persistent Storage Systems

Overview of storage systems for long-term memory.

### Storage System Comparison

```python
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
from datetime import datetime

class PersistentStore(ABC):
    """Abstract interface for persistent storage."""
    
    @abstractmethod
    def save(self, key: str, value: Any, metadata: Dict = None):
        """Save data to persistent storage."""
        pass
    
    @abstractmethod
    def load(self, key: str) -> Optional[Any]:
        """Load data from persistent storage."""
        pass
    
    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[Any]:
        """Search for data."""
        pass
    
    @abstractmethod
    def delete(self, key: str):
        """Delete data."""
        pass
    
    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys with optional prefix."""
        pass
```

### Simple File-Based Storage

For small-scale or prototyping:

```python
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

class FileStore(PersistentStore):
    """Simple file-based persistent storage."""
    
    def __init__(self, base_path: str = "./memory"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save(self, key: str, value: Any, metadata: Dict = None):
        """Save to JSON file."""
        safe_key = self._sanitize_key(key)
        file_path = self.base_path / f"{safe_key}.json"
        
        data = {
            'value': value,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'key': key
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load(self, key: str) -> Optional[Any]:
        """Load from JSON file."""
        safe_key = self._sanitize_key(key)
        file_path = self.base_path / f"{safe_key}.json"
        
        if not file_path.exists():
            return None
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return data['value']
    
    def search(self, query: str, limit: int = 10) -> List[Any]:
        """Simple keyword search across files."""
        results = []
        query_lower = query.lower()
        
        for file_path in self.base_path.glob("*.json"):
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Search in value and metadata
            data_str = json.dumps(data).lower()
            
            if query_lower in data_str:
                results.append(data)
            
            if len(results) >= limit:
                break
        
        return results
    
    def delete(self, key: str):
        """Delete file."""
        safe_key = self._sanitize_key(key)
        file_path = self.base_path / f"{safe_key}.json"
        
        if file_path.exists():
            file_path.unlink()
    
    def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys."""
        keys = []
        
        for file_path in self.base_path.glob("*.json"):
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            key = data['key']
            if key.startswith(prefix):
                keys.append(key)
        
        return keys
    
    def _sanitize_key(self, key: str) -> str:
        """Convert key to safe filename."""
        # Replace unsafe characters
        safe = key.replace('/', '_').replace('\\', '_')
        safe = ''.join(c for c in safe if c.isalnum() or c in '_-.')
        return safe[:100]  # Limit length


# Usage
store = FileStore("./agent_memory")

store.save(
    "user:alice:preferences",
    {"theme": "dark", "language": "en"},
    metadata={"category": "user_profile"}
)

prefs = store.load("user:alice:preferences")
print(prefs)  # {'theme': 'dark', 'language': 'en'}
```

## Vector Databases

Vector databases enable semantic search over stored information.

### Vector Store Implementation

```python
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class VectorEntry:
    """An entry in the vector store."""
    id: str
    vector: np.ndarray
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime

class InMemoryVectorStore:
    """Simple in-memory vector store for demonstration."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.entries: Dict[str, VectorEntry] = {}
    
    def add(self, id: str, vector: List[float], content: str,
            metadata: Dict = None):
        """Add vector to store."""
        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector dimension {len(vector)} != {self.dimension}"
            )
        
        entry = VectorEntry(
            id=id,
            vector=np.array(vector),
            content=content,
            metadata=metadata or {},
            timestamp=datetime.now()
        )
        
        self.entries[id] = entry
    
    def search(self, query_vector: List[float], top_k: int = 5,
               filter_metadata: Dict = None) -> List[Tuple[str, float, str]]:
        """
        Search for similar vectors.
        
        Returns list of (id, similarity_score, content) tuples.
        """
        if len(query_vector) != self.dimension:
            raise ValueError("Query vector dimension mismatch")
        
        query_vec = np.array(query_vector)
        
        # Calculate similarities
        results = []
        
        for entry in self.entries.values():
            # Apply metadata filter if provided
            if filter_metadata:
                if not all(
                    entry.metadata.get(k) == v
                    for k, v in filter_metadata.items()
                ):
                    continue
            
            # Cosine similarity
            similarity = self._cosine_similarity(query_vec, entry.vector)
            results.append((entry.id, similarity, entry.content, entry.metadata))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def delete(self, id: str):
        """Delete entry."""
        self.entries.pop(id, None)
    
    def get(self, id: str) -> Optional[VectorEntry]:
        """Get entry by ID."""
        return self.entries.get(id)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity."""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norm_product == 0:
            return 0.0
        
        return dot_product / norm_product


# Usage
vector_store = InMemoryVectorStore(dimension=384)

# Add some memories
vector_store.add(
    "mem_001",
    [0.1] * 384,  # Would be actual embedding
    "User prefers dark mode",
    metadata={"type": "preference", "user": "alice"}
)

vector_store.add(
    "mem_002",
    [0.2] * 384,
    "User is interested in machine learning",
    metadata={"type": "interest", "user": "alice"}
)

# Search
query_vector = [0.15] * 384
results = vector_store.search(query_vector, top_k=2)

for id, score, content, metadata in results:
    print(f"{id} (score: {score:.3f}): {content}")
```

### ChromaDB Integration

Production-ready vector database:

```python
import chromadb
from chromadb.config import Settings

class ChromaMemoryStore:
    """Long-term memory using ChromaDB."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="agent_memory",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_memory(self, id: str, content: str, embedding: List[float],
                   metadata: Dict = None):
        """Add memory to ChromaDB."""
        self.collection.add(
            ids=[id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata or {}]
        )
    
    def add_batch(self, ids: List[str], contents: List[str],
                  embeddings: List[List[float]], metadatas: List[Dict] = None):
        """Add multiple memories efficiently."""
        self.collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas or [{} for _ in ids]
        )
    
    def search(self, query_embedding: List[float], n_results: int = 5,
               where: Dict = None) -> List[Dict]:
        """
        Search for similar memories.
        
        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            where: Metadata filter (e.g., {"type": "preference"})
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        # Format results
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'distance': results['distances'][0][i],
                'metadata': results['metadatas'][0][i]
            })
        
        return formatted
    
    def update_memory(self, id: str, content: str = None,
                     embedding: List[float] = None, metadata: Dict = None):
        """Update existing memory."""
        update_kwargs = {'ids': [id]}
        
        if content is not None:
            update_kwargs['documents'] = [content]
        
        if embedding is not None:
            update_kwargs['embeddings'] = [embedding]
        
        if metadata is not None:
            update_kwargs['metadatas'] = [metadata]
        
        self.collection.update(**update_kwargs)
    
    def delete_memory(self, id: str):
        """Delete memory."""
        self.collection.delete(ids=[id])
    
    def get_memory(self, id: str) -> Optional[Dict]:
        """Get memory by ID."""
        results = self.collection.get(ids=[id])
        
        if not results['ids']:
            return None
        
        return {
            'id': results['ids'][0],
            'content': results['documents'][0],
            'metadata': results['metadatas'][0]
        }
    
    def count(self) -> int:
        """Get total number of memories."""
        return self.collection.count()
    
    def persist(self):
        """Persist to disk."""
        self.client.persist()


# Usage with embeddings
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_store = ChromaMemoryStore("./agent_chroma")

# Add memory
content = "User Alice prefers morning meetings"
embedding = model.encode(content).tolist()

chroma_store.add_memory(
    id="alice_pref_001",
    content=content,
    embedding=embedding,
    metadata={"type": "preference", "user": "alice"}
)

# Search
query = "What time does Alice like meetings?"
query_embedding = model.encode(query).tolist()

results = chroma_store.search(
    query_embedding,
    n_results=3,
    where={"user": "alice"}
)

for result in results:
    print(f"{result['id']}: {result['content']}")
    print(f"  Distance: {result['distance']:.3f}")
```

## Relational Databases

SQL databases for structured, relational data.

### Schema Design for Agent Memory

```python
import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

class SQLMemoryStore:
    """SQL-based long-term memory store."""
    
    def __init__(self, db_path: str = "agent_memory.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Access columns by name
        self._create_schema()
    
    def _create_schema(self):
        """Create database schema."""
        
        # Conversations table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                started_at DATETIME NOT NULL,
                ended_at DATETIME,
                summary TEXT,
                metadata TEXT
            )
        """)
        
        # Messages table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                tokens INTEGER,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)
        
        # User facts table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS user_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                fact_type TEXT NOT NULL,
                content TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
        """)
        
        # Entities table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                attributes TEXT,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
        """)
        
        # Relations table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_id TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                created_at DATETIME NOT NULL,
                FOREIGN KEY (subject_id) REFERENCES entities(id),
                FOREIGN KEY (object_id) REFERENCES entities(id)
            )
        """)
        
        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)",
            "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_user_facts_user ON user_facts(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_user_facts_type ON user_facts(fact_type)",
            "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)",
            "CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject_id)",
            "CREATE INDEX IF NOT EXISTS idx_relations_object ON relations(object_id)",
        ]
        
        for index in indexes:
            self.conn.execute(index)
        
        self.conn.commit()
    
    def add_conversation(self, conv_id: str, user_id: str,
                        metadata: Dict = None):
        """Start a new conversation."""
        self.conn.execute("""
            INSERT INTO conversations (id, user_id, started_at, metadata)
            VALUES (?, ?, ?, ?)
        """, (
            conv_id,
            user_id,
            datetime.now(),
            json.dumps(metadata or {})
        ))
        self.conn.commit()
    
    def add_message(self, msg_id: str, conv_id: str, role: str,
                   content: str, tokens: int = None):
        """Add message to conversation."""
        self.conn.execute("""
            INSERT INTO messages (id, conversation_id, role, content, timestamp, tokens)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            msg_id,
            conv_id,
            role,
            content,
            datetime.now(),
            tokens
        ))
        self.conn.commit()
    
    def get_conversation_messages(self, conv_id: str) -> List[Dict]:
        """Get all messages in a conversation."""
        cursor = self.conn.execute("""
            SELECT id, role, content, timestamp, tokens
            FROM messages
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
        """, (conv_id,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def add_user_fact(self, user_id: str, fact_type: str, content: str,
                     confidence: float = 1.0, source: str = "learned"):
        """Add or update a fact about the user."""
        now = datetime.now()
        
        self.conn.execute("""
            INSERT INTO user_facts 
            (user_id, fact_type, content, confidence, source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_id, fact_type, content, confidence, source, now, now))
        
        self.conn.commit()
    
    def get_user_facts(self, user_id: str, fact_type: str = None) -> List[Dict]:
        """Get facts about a user."""
        if fact_type:
            cursor = self.conn.execute("""
                SELECT id, fact_type, content, confidence, source, created_at
                FROM user_facts
                WHERE user_id = ? AND fact_type = ?
                ORDER BY updated_at DESC
            """, (user_id, fact_type))
        else:
            cursor = self.conn.execute("""
                SELECT id, fact_type, content, confidence, source, created_at
                FROM user_facts
                WHERE user_id = ?
                ORDER BY updated_at DESC
            """, (user_id,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def add_entity(self, entity_id: str, name: str, entity_type: str,
                  attributes: Dict = None):
        """Add or update an entity."""
        now = datetime.now()
        
        self.conn.execute("""
            INSERT OR REPLACE INTO entities
            (id, name, entity_type, attributes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            entity_id,
            name,
            entity_type,
            json.dumps(attributes or {}),
            now,
            now
        ))
        
        self.conn.commit()
    
    def add_relation(self, subject_id: str, predicate: str, object_id: str,
                    confidence: float = 1.0):
        """Add a relation between entities."""
        self.conn.execute("""
            INSERT INTO relations
            (subject_id, predicate, object_id, confidence, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (subject_id, predicate, object_id, confidence, datetime.now()))
        
        self.conn.commit()
    
    def get_entity_relations(self, entity_id: str) -> List[Dict]:
        """Get all relations involving an entity."""
        cursor = self.conn.execute("""
            SELECT id, subject_id, predicate, object_id, confidence
            FROM relations
            WHERE subject_id = ? OR object_id = ?
        """, (entity_id, entity_id))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def search_conversations(self, user_id: str, 
                           limit: int = 10) -> List[Dict]:
        """Get recent conversations for a user."""
        cursor = self.conn.execute("""
            SELECT id, started_at, ended_at, summary
            FROM conversations
            WHERE user_id = ?
            ORDER BY started_at DESC
            LIMIT ?
        """, (user_id, limit))
        
        return [dict(row) for row in cursor.fetchall()]


# Usage
sql_store = SQLMemoryStore("agent_memory.db")

# Start conversation
import uuid

conv_id = str(uuid.uuid4())
sql_store.add_conversation(
    conv_id,
    "alice",
    metadata={"source": "web_chat"}
)

# Add messages
sql_store.add_message(
    str(uuid.uuid4()),
    conv_id,
    "user",
    "I prefer morning meetings",
    tokens=5
)

sql_store.add_message(
    str(uuid.uuid4()),
    conv_id,
    "assistant",
    "I'll remember that you prefer morning meetings.",
    tokens=8
)

# Add user fact
sql_store.add_user_fact(
    "alice",
    "preference",
    "prefers morning meetings",
    confidence=0.9,
    source="conversation"
)

# Query user facts
facts = sql_store.get_user_facts("alice", fact_type="preference")
for fact in facts:
    print(f"{fact['content']} (confidence: {fact['confidence']})")
```

## Document Stores

For semi-structured and unstructured data.

### MongoDB-Style Document Store

```python
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
from datetime import datetime

class DocumentStore:
    """Document-oriented storage for agent memory."""
    
    def __init__(self, base_path: str = "./documents"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory index for fast lookups
        self.index = self._build_index()
    
    def _build_index(self) -> Dict[str, Path]:
        """Build index of all documents."""
        index = {}
        
        for collection_dir in self.base_path.iterdir():
            if not collection_dir.is_dir():
                continue
            
            for doc_file in collection_dir.glob("*.json"):
                doc_id = doc_file.stem
                index[f"{collection_dir.name}:{doc_id}"] = doc_file
        
        return index
    
    def insert(self, collection: str, document: Dict) -> str:
        """Insert document into collection."""
        # Generate ID if not provided
        doc_id = document.get('_id', str(uuid.uuid4()))
        
        # Add metadata
        document['_id'] = doc_id
        document['_created_at'] = datetime.now().isoformat()
        document['_updated_at'] = datetime.now().isoformat()
        
        # Save to file
        collection_dir = self.base_path / collection
        collection_dir.mkdir(exist_ok=True)
        
        doc_path = collection_dir / f"{doc_id}.json"
        
        with open(doc_path, 'w') as f:
            json.dump(document, f, indent=2, default=str)
        
        # Update index
        self.index[f"{collection}:{doc_id}"] = doc_path
        
        return doc_id
    
    def find_one(self, collection: str, doc_id: str) -> Optional[Dict]:
        """Find document by ID."""
        key = f"{collection}:{doc_id}"
        
        if key not in self.index:
            return None
        
        doc_path = self.index[key]
        
        with open(doc_path, 'r') as f:
            return json.load(f)
    
    def find(self, collection: str, query: Dict = None,
            limit: int = 100) -> List[Dict]:
        """
        Find documents matching query.
        
        Simplified query format:
        {"field": "value"} - exact match
        {"field": {"$gt": value}} - greater than
        {"field": {"$regex": "pattern"}} - regex match
        """
        collection_dir = self.base_path / collection
        
        if not collection_dir.exists():
            return []
        
        results = []
        
        for doc_file in collection_dir.glob("*.json"):
            with open(doc_file, 'r') as f:
                doc = json.load(f)
            
            if self._matches_query(doc, query or {}):
                results.append(doc)
            
            if len(results) >= limit:
                break
        
        return results
    
    def update(self, collection: str, doc_id: str, update: Dict):
        """Update document."""
        doc = self.find_one(collection, doc_id)
        
        if not doc:
            raise ValueError(f"Document {doc_id} not found in {collection}")
        
        # Apply updates
        doc.update(update)
        doc['_updated_at'] = datetime.now().isoformat()
        
        # Save
        doc_path = self.index[f"{collection}:{doc_id}"]
        
        with open(doc_path, 'w') as f:
            json.dump(doc, f, indent=2, default=str)
    
    def delete(self, collection: str, doc_id: str):
        """Delete document."""
        key = f"{collection}:{doc_id}"
        
        if key in self.index:
            doc_path = self.index[key]
            doc_path.unlink()
            del self.index[key]
    
    def _matches_query(self, doc: Dict, query: Dict) -> bool:
        """Check if document matches query."""
        for field, condition in query.items():
            if field not in doc:
                return False
            
            if isinstance(condition, dict):
                # Handle operators
                for op, value in condition.items():
                    if op == "$gt":
                        if not doc[field] > value:
                            return False
                    elif op == "$lt":
                        if not doc[field] < value:
                            return False
                    elif op == "$regex":
                        import re
                        if not re.search(value, str(doc[field])):
                            return False
            else:
                # Exact match
                if doc[field] != condition:
                    return False
        
        return True


# Usage
doc_store = DocumentStore("./agent_docs")

# Insert user profile
user_id = doc_store.insert("users", {
    "name": "Alice",
    "email": "alice@example.com",
    "preferences": {
        "theme": "dark",
        "notifications": True
    },
    "joined": datetime.now()
})

# Insert conversation summary
doc_store.insert("conversation_summaries", {
    "user_id": user_id,
    "date": datetime.now().date().isoformat(),
    "topics": ["machine learning", "data science"],
    "key_points": [
        "User is learning about neural networks",
        "Interested in practical applications"
    ]
})

# Find conversations about machine learning
ml_conversations = doc_store.find(
    "conversation_summaries",
    {"topics": "machine learning"}
)

print(f"Found {len(ml_conversations)} ML conversations")
```

## Graph Databases

For relationship-heavy data.

### Simple Graph Store

```python
from typing import Set, Dict, List, Tuple, Optional
from dataclasses import dataclass, field

@dataclass
class GraphNode:
    """A node in the knowledge graph."""
    id: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GraphEdge:
    """An edge in the knowledge graph."""
    source: str
    target: str
    relation: str
    properties: Dict[str, Any] = field(default_factory=dict)

class KnowledgeGraph:
    """Simple knowledge graph for agent memory."""
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        
        # Index for fast lookups
        self.outgoing_edges: Dict[str, List[GraphEdge]] = {}
        self.incoming_edges: Dict[str, List[GraphEdge]] = {}
    
    def add_node(self, node_id: str, label: str, properties: Dict = None):
        """Add node to graph."""
        node = GraphNode(
            id=node_id,
            label=label,
            properties=properties or {}
        )
        
        self.nodes[node_id] = node
        
        # Initialize edge indexes
        if node_id not in self.outgoing_edges:
            self.outgoing_edges[node_id] = []
        if node_id not in self.incoming_edges:
            self.incoming_edges[node_id] = []
    
    def add_edge(self, source: str, target: str, relation: str,
                properties: Dict = None):
        """Add edge between nodes."""
        # Ensure nodes exist
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("Both nodes must exist before adding edge")
        
        edge = GraphEdge(
            source=source,
            target=target,
            relation=relation,
            properties=properties or {}
        )
        
        self.edges.append(edge)
        
        # Update indexes
        self.outgoing_edges[source].append(edge)
        self.incoming_edges[target].append(edge)
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)
    
    def get_neighbors(self, node_id: str,
                     relation: str = None) -> List[GraphNode]:
        """Get neighboring nodes."""
        neighbors = []
        
        for edge in self.outgoing_edges.get(node_id, []):
            if relation is None or edge.relation == relation:
                neighbors.append(self.nodes[edge.target])
        
        return neighbors
    
    def find_path(self, start: str, end: str,
                 max_depth: int = 3) -> Optional[List[str]]:
        """Find path between two nodes (BFS)."""
        if start not in self.nodes or end not in self.nodes:
            return None
        
        from collections import deque
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            if current == end:
                return path
            
            for edge in self.outgoing_edges.get(current, []):
                if edge.target not in visited:
                    visited.add(edge.target)
                    queue.append((edge.target, path + [edge.target]))
        
        return None
    
    def query(self, pattern: str) -> List[Dict]:
        """
        Simple pattern matching query.
        
        Pattern format: "(source)-[relation]->(target)"
        Variables start with ?
        """
        # Parse pattern (simplified)
        import re
        match = re.match(r'\((\??\w+)\)-\[(\??\w+)\]->\((\??\w+)\)', pattern)
        
        if not match:
            raise ValueError(f"Invalid pattern: {pattern}")
        
        source_pattern, relation_pattern, target_pattern = match.groups()
        
        results = []
        
        for edge in self.edges:
            # Check if edge matches pattern
            if (self._matches_pattern(edge.source, source_pattern) and
                self._matches_pattern(edge.relation, relation_pattern) and
                self._matches_pattern(edge.target, target_pattern)):
                
                results.append({
                    'source': self.nodes[edge.source],
                    'relation': edge.relation,
                    'target': self.nodes[edge.target],
                    'properties': edge.properties
                })
        
        return results
    
    def _matches_pattern(self, value: str, pattern: str) -> bool:
        """Check if value matches pattern."""
        if pattern.startswith('?'):
            return True  # Variable matches anything
        return value == pattern
    
    def to_dict(self) -> Dict:
        """Export graph as dictionary."""
        return {
            'nodes': [
                {
                    'id': node.id,
                    'label': node.label,
                    'properties': node.properties
                }
                for node in self.nodes.values()
            ],
            'edges': [
                {
                    'source': edge.source,
                    'target': edge.target,
                    'relation': edge.relation,
                    'properties': edge.properties
                }
                for edge in self.edges
            ]
        }


# Usage
kg = KnowledgeGraph()

# Add nodes
kg.add_node("alice", "Person", {"name": "Alice", "role": "user"})
kg.add_node("ml", "Topic", {"name": "Machine Learning"})
kg.add_node("python", "Language", {"name": "Python"})
kg.add_node("project", "Project", {"name": "Chatbot"})

# Add relationships
kg.add_edge("alice", "ml", "INTERESTED_IN")
kg.add_edge("alice", "python", "KNOWS")
kg.add_edge("alice", "project", "WORKING_ON")
kg.add_edge("project", "python", "USES")
kg.add_edge("project", "ml", "APPLIES")

# Query: What is Alice interested in?
interests = kg.get_neighbors("alice", relation="INTERESTED_IN")
print("Alice is interested in:", [n.properties['name'] for n in interests])

# Query: Find path from Alice to ML
path = kg.find_path("alice", "ml")
print("Path from Alice to ML:", path)

# Pattern query: What does Alice know?
results = kg.query("(alice)-[KNOWS]->(?target)")
print("Alice knows:", [r['target'].properties['name'] for r in results])
```

## Hybrid Approaches

Combining multiple storage systems for optimal performance.

### Hybrid Memory System

```python
class HybridLongTermMemory:
    """
    Combines vector, SQL, and document storage.
    
    - Vector DB: Semantic search
    - SQL: Structured queries and relationships
    - Document store: Flexible schemas
    """
    
    def __init__(self):
        self.vector_store = ChromaMemoryStore("./hybrid_vectors")
        self.sql_store = SQLMemoryStore("./hybrid_sql.db")
        self.doc_store = DocumentStore("./hybrid_docs")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def store_memory(self, memory_type: str, content: str,
                    metadata: Dict = None):
        """
        Store memory using appropriate backend(s).
        
        Args:
            memory_type: 'conversation', 'fact', 'document', etc.
            content: The memory content
            metadata: Additional metadata
        """
        memory_id = str(uuid.uuid4())
        metadata = metadata or {}
        metadata['type'] = memory_type
        metadata['timestamp'] = datetime.now().isoformat()
        
        # Always store in vector DB for semantic search
        embedding = self.embedder.encode(content).tolist()
        self.vector_store.add_memory(
            memory_id,
            content,
            embedding,
            metadata
        )
        
        # Store in appropriate structured storage
        if memory_type == 'conversation':
            # SQL for structured conversation data
            self.sql_store.add_message(
                memory_id,
                metadata.get('conversation_id', 'default'),
                metadata.get('role', 'assistant'),
                content
            )
        
        elif memory_type == 'fact':
            # SQL for structured facts
            self.sql_store.add_user_fact(
                metadata.get('user_id', 'default'),
                metadata.get('fact_type', 'general'),
                content,
                confidence=metadata.get('confidence', 1.0)
            )
        
        elif memory_type == 'document':
            # Document store for flexible documents
            self.doc_store.insert('memories', {
                '_id': memory_id,
                'content': content,
                **metadata
            })
        
        return memory_id
    
    def search(self, query: str, memory_type: str = None,
              limit: int = 5) -> List[Dict]:
        """
        Search across all storage systems.
        
        Uses vector search for initial retrieval,
        then enriches with structured data.
        """
        # Semantic search in vector DB
        query_embedding = self.embedder.encode(query).tolist()
        
        where_filter = {"type": memory_type} if memory_type else None
        
        vector_results = self.vector_store.search(
            query_embedding,
            n_results=limit,
            where=where_filter
        )
        
        # Enrich results with structured data
        enriched = []
        
        for result in vector_results:
            memory_id = result['id']
            memory_type = result['metadata'].get('type')
            
            enriched_result = {
                'id': memory_id,
                'content': result['content'],
                'score': 1 - result['distance'],  # Convert distance to similarity
                'metadata': result['metadata'],
                'type': memory_type
            }
            
            # Add structured data if available
            if memory_type == 'conversation':
                # Get conversation context
                conv_id = result['metadata'].get('conversation_id')
                if conv_id:
                    messages = self.sql_store.get_conversation_messages(conv_id)
                    enriched_result['conversation'] = messages
            
            elif memory_type == 'fact':
                # Get related facts
                user_id = result['metadata'].get('user_id')
                if user_id:
                    facts = self.sql_store.get_user_facts(user_id)
                    enriched_result['related_facts'] = facts[:3]
            
            enriched.append(enriched_result)
        
        return enriched
    
    def get_user_context(self, user_id: str) -> Dict:
        """
        Get complete context for a user.
        
        Combines data from all storage systems.
        """
        context = {
            'user_id': user_id,
            'facts': [],
            'recent_conversations': [],
            'documents': []
        }
        
        # Get facts from SQL
        context['facts'] = self.sql_store.get_user_facts(user_id)
        
        # Get recent conversations
        context['recent_conversations'] = self.sql_store.search_conversations(
            user_id,
            limit=5
        )
        
        # Get documents
        context['documents'] = self.doc_store.find(
            'memories',
            {'metadata.user_id': user_id},
            limit=10
        )
        
        return context


# Usage
hybrid_memory = HybridLongTermMemory()

# Store different types of memories
hybrid_memory.store_memory(
    'fact',
    "User prefers Python over JavaScript",
    metadata={'user_id': 'alice', 'fact_type': 'preference', 'confidence': 0.9}
)

hybrid_memory.store_memory(
    'conversation',
    "I think Python is more readable",
    metadata={'conversation_id': 'conv_123', 'role': 'user', 'user_id': 'alice'}
)

hybrid_memory.store_memory(
    'document',
    "Project ideas: 1. Chatbot with memory, 2. Code analyzer",
    metadata={'user_id': 'alice', 'category': 'projects'}
)

# Search across all types
results = hybrid_memory.search("What programming language does Alice prefer?")

for result in results:
    print(f"[{result['type']}] {result['content'][:50]}...")
    print(f"  Score: {result['score']:.3f}")
```

## When to Persist

Not all information needs long-term storage.

### Persistence Policy

```python
from enum import Enum

class PersistencePolicy(Enum):
    """When to persist information."""
    ALWAYS = "always"
    IMPORTANT = "important"
    ON_REQUEST = "on_request"
    NEVER = "never"

class PersistenceDecider:
    """Decide what to persist to long-term memory."""
    
    def __init__(self):
        # Define policies for different information types
        self.policies = {
            'user_fact': PersistencePolicy.ALWAYS,
            'preference': PersistencePolicy.ALWAYS,
            'casual_chat': PersistencePolicy.NEVER,
            'task_result': PersistencePolicy.IMPORTANT,
            'error': PersistencePolicy.IMPORTANT,
            'code': PersistencePolicy.ON_REQUEST,
        }
        
        self.importance_threshold = 0.6
    
    def should_persist(self, content: str, metadata: Dict) -> bool:
        """Decide if content should be persisted."""
        memory_type = metadata.get('type', 'general')
        importance = metadata.get('importance', 0.5)
        
        # Get policy for this type
        policy = self.policies.get(memory_type, PersistencePolicy.IMPORTANT)
        
        if policy == PersistencePolicy.ALWAYS:
            return True
        
        elif policy == PersistencePolicy.NEVER:
            return False
        
        elif policy == PersistencePolicy.IMPORTANT:
            return importance >= self.importance_threshold
        
        elif policy == PersistencePolicy.ON_REQUEST:
            return metadata.get('persist_requested', False)
        
        return False
    
    def calculate_importance(self, content: str, metadata: Dict) -> float:
        """Calculate importance score for content."""
        importance = 0.5
        
        # Length-based importance
        if len(content) > 200:
            importance += 0.1
        
        # Keyword-based importance
        important_keywords = [
            'important', 'remember', 'always', 'never',
            'critical', 'essential', 'must', 'should not'
        ]
        
        content_lower = content.lower()
        keyword_count = sum(
            1 for keyword in important_keywords
            if keyword in content_lower
        )
        
        importance += min(0.3, keyword_count * 0.1)
        
        # Metadata-based importance
        if metadata.get('user_requested'):
            importance += 0.2
        
        if metadata.get('type') in ['preference', 'user_fact']:
            importance += 0.2
        
        return min(1.0, importance)


# Usage
decider = PersistenceDecider()

# Check if should persist
content = "User mentioned they're allergic to peanuts"
metadata = {'type': 'user_fact', 'importance': 0.9}

if decider.should_persist(content, metadata):
    print("Persisting to long-term memory")
    # Store in long-term memory
else:
    print("Keeping only in short-term memory")
```

## Retrieval Strategies

Efficiently retrieving relevant information from long-term memory.

### Multi-Strategy Retriever

```python
class LongTermMemoryRetriever:
    """Retrieve from long-term memory using multiple strategies."""
    
    def __init__(self, hybrid_memory: HybridLongTermMemory):
        self.memory = hybrid_memory
    
    def retrieve(self, query: str, strategy: str = "hybrid",
                limit: int = 5) -> List[Dict]:
        """
        Retrieve relevant memories.
        
        Strategies:
        - semantic: Vector similarity
        - temporal: Most recent
        - importance: Highest importance
        - hybrid: Combination
        """
        if strategy == "semantic":
            return self._semantic_retrieval(query, limit)
        
        elif strategy == "temporal":
            return self._temporal_retrieval(query, limit)
        
        elif strategy == "importance":
            return self._importance_retrieval(query, limit)
        
        elif strategy == "hybrid":
            return self._hybrid_retrieval(query, limit)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _semantic_retrieval(self, query: str, limit: int) -> List[Dict]:
        """Retrieve by semantic similarity."""
        return self.memory.search(query, limit=limit)
    
    def _temporal_retrieval(self, query: str, limit: int) -> List[Dict]:
        """Retrieve most recent memories."""
        # Get all memories and sort by timestamp
        all_memories = self.memory.search(query, limit=limit * 2)
        
        # Sort by timestamp
        all_memories.sort(
            key=lambda x: x['metadata'].get('timestamp', ''),
            reverse=True
        )
        
        return all_memories[:limit]
    
    def _importance_retrieval(self, query: str, limit: int) -> List[Dict]:
        """Retrieve by importance."""
        all_memories = self.memory.search(query, limit=limit * 2)
        
        # Sort by importance
        all_memories.sort(
            key=lambda x: x['metadata'].get('importance', 0),
            reverse=True
        )
        
        return all_memories[:limit]
    
    def _hybrid_retrieval(self, query: str, limit: int) -> List[Dict]:
        """
        Combine semantic, temporal, and importance.
        
        Score = 0.5 * semantic + 0.3 * temporal + 0.2 * importance
        """
        # Get more results for reranking
        results = self.memory.search(query, limit=limit * 3)
        
        # Calculate combined scores
        from datetime import datetime
        
        now = datetime.now()
        
        for result in results:
            # Semantic score (already in result)
            semantic_score = result['score']
            
            # Temporal score (decay based on age)
            timestamp_str = result['metadata'].get('timestamp')
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
                age_days = (now - timestamp).days
                temporal_score = max(0, 1 - (age_days / 365))  # Decay over year
            else:
                temporal_score = 0
            
            # Importance score
            importance_score = result['metadata'].get('importance', 0.5)
            
            # Combined score
            result['combined_score'] = (
                0.5 * semantic_score +
                0.3 * temporal_score +
                0.2 * importance_score
            )
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return results[:limit]
```

## Indexing and Optimization

Optimize long-term memory for fast retrieval.

### Indexing Strategies

```python
class MemoryIndexer:
    """Build indexes for fast memory retrieval."""
    
    def __init__(self, sql_store: SQLMemoryStore):
        self.sql_store = sql_store
    
    def create_indexes(self):
        """Create database indexes."""
        indexes = [
            # Conversations
            "CREATE INDEX IF NOT EXISTS idx_conv_user_time ON conversations(user_id, started_at DESC)",
            
            # Messages
            "CREATE INDEX IF NOT EXISTS idx_msg_conv_time ON messages(conversation_id, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_msg_role ON messages(role)",
            
            # User facts
            "CREATE INDEX IF NOT EXISTS idx_facts_user_type ON user_facts(user_id, fact_type)",
            "CREATE INDEX IF NOT EXISTS idx_facts_confidence ON user_facts(confidence DESC)",
            "CREATE INDEX IF NOT EXISTS idx_facts_updated ON user_facts(updated_at DESC)",
            
            # Entities
            "CREATE INDEX IF NOT EXISTS idx_entities_type_name ON entities(entity_type, name)",
            
            # Relations
            "CREATE INDEX IF NOT EXISTS idx_relations_subj_pred ON relations(subject_id, predicate)",
        ]
        
        for index in indexes:
            self.sql_store.conn.execute(index)
        
        self.sql_store.conn.commit()
    
    def analyze_tables(self):
        """Update query planner statistics."""
        self.sql_store.conn.execute("ANALYZE")
        self.sql_store.conn.commit()
    
    def vacuum_database(self):
        """Reclaim space and optimize."""
        self.sql_store.conn.execute("VACUUM")
```

## Scaling Long-Term Memory

Strategies for scaling as memory grows.

### Partitioning Strategy

```python
class PartitionedMemoryStore:
    """Partition memory by time periods."""
    
    def __init__(self, base_path: str = "./partitioned_memory"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.partitions = {}  # period -> store
    
    def _get_partition(self, timestamp: datetime) -> str:
        """Get partition key for timestamp (YYYY-MM format)."""
        return timestamp.strftime("%Y-%m")
    
    def _get_store(self, partition: str) -> SQLMemoryStore:
        """Get or create store for partition."""
        if partition not in self.partitions:
            db_path = self.base_path / f"{partition}.db"
            self.partitions[partition] = SQLMemoryStore(str(db_path))
        
        return self.partitions[partition]
    
    def add_memory(self, content: str, metadata: Dict = None,
                  timestamp: datetime = None):
        """Add memory to appropriate partition."""
        timestamp = timestamp or datetime.now()
        partition = self._get_partition(timestamp)
        store = self._get_store(partition)
        
        # Store in partition
        # (simplified - would use actual store methods)
        print(f"Storing in partition {partition}")
    
    def search_recent(self, query: str, months: int = 3,
                     limit: int = 10) -> List[Dict]:
        """Search recent partitions only."""
        # Determine which partitions to search
        partitions_to_search = self._get_recent_partitions(months)
        
        results = []
        
        for partition in partitions_to_search:
            if partition in self.partitions:
                store = self.partitions[partition]
                # Search this partition
                # results.extend(store.search(query))
        
        return results[:limit]
    
    def _get_recent_partitions(self, months: int) -> List[str]:
        """Get list of recent partition keys."""
        from dateutil.relativedelta import relativedelta
        
        partitions = []
        current = datetime.now()
        
        for i in range(months):
            date = current - relativedelta(months=i)
            partitions.append(self._get_partition(date))
        
        return partitions
```

## Data Lifecycle Management

Managing memory lifecycle: creation, updates, archival, deletion.

### Memory Lifecycle Manager

```python
class MemoryLifecycleManager:
    """Manage memory lifecycle."""
    
    def __init__(self, memory_store):
        self.store = memory_store
        self.archive_age_days = 365  # Archive after 1 year
        self.delete_age_days = 730   # Delete after 2 years
    
    def archive_old_memories(self):
        """Archive old memories to cold storage."""
        cutoff_date = datetime.now() - timedelta(days=self.archive_age_days)
        
        # Find old memories
        # Archive them to separate storage
        print(f"Archiving memories older than {cutoff_date}")
    
    def delete_expired_memories(self):
        """Delete very old memories."""
        cutoff_date = datetime.now() - timedelta(days=self.delete_age_days)
        
        print(f"Deleting memories older than {cutoff_date}")
    
    def update_memory_importance(self, memory_id: str,
                                new_importance: float):
        """Update importance based on usage."""
        # Would update in actual store
        pass
    
    def decay_importance(self):
        """Gradually decrease importance of unused memories."""
        # Apply time-based decay to all memories
        decay_factor = 0.95  # 5% decay per period
        
        print("Applying importance decay")
```

## Summary

Long-term memory enables agents to persist knowledge across sessions:

**Key Storage Systems**:

- **Vector databases**: Semantic search (ChromaDB, Pinecone, Weaviate)
- **SQL databases**: Structured data and relationships
- **Document stores**: Flexible schemas (MongoDB-style)
- **Graph databases**: Complex relationships

**Key Concepts**:

- **Persistence policies**: Not everything needs long-term storage
- **Retrieval strategies**: Semantic, temporal, importance-based, hybrid
- **Indexing**: Critical for performance at scale
- **Partitioning**: Scale by time or other dimensions
- **Lifecycle management**: Archive and delete old data

**Best Practices**:

- Use hybrid approaches combining multiple storage types
- Index frequently queried fields
- Partition large datasets
- Implement proper lifecycle management
- Cache frequently accessed data
- Monitor storage usage and performance

## Next Steps

- **[Memory Retrieval](memory-retrieval.md)**: Advanced retrieval strategies and ranking
- **[Memory Consolidation](memory-consolidation.md)**: Moving from short-term to long-term memory
- **[Episodic Memory](episodic-memory.md)**: Storing and retrieving experiences
- **[Semantic Memory](semantic-memory.md)**: Knowledge representation and reasoning
