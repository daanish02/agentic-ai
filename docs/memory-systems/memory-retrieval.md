# Memory Retrieval

## Table of Contents

- [Introduction](#introduction)
- [Retrieval Fundamentals](#retrieval-fundamentals)
- [Semantic Search](#semantic-search)
- [Keyword Search](#keyword-search)
- [Recency-Based Retrieval](#recency-based-retrieval)
- [Importance Weighting](#importance-weighting)
- [Hybrid Retrieval](#hybrid-retrieval)
- [Relevance Scoring](#relevance-scoring)
- [Retrieval Strategies](#retrieval-strategies)
- [Query Expansion](#query-expansion)
- [Retrieval Optimization](#retrieval-optimization)
- [Context-Aware Retrieval](#context-aware-retrieval)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Memory retrieval determines which information from memory is surfaced for the agent's current context. Effective retrieval balances multiple factors: relevance, recency, importance, and diversity.

> "Finding the right memories at the right time is as crucial as storing them - retrieval is the bridge between knowledge and action."

Challenges in memory retrieval:

- **Relevance**: Finding semantically related memories
- **Recency**: Prioritizing recent information
- **Importance**: Surfacing significant memories
- **Diversity**: Avoiding redundant results
- **Efficiency**: Fast retrieval at scale

### Retrieval Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Query Processing                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐│
│  │ Semantic │ │ Keyword  │ │ Recency  │ │ Importance ││
│  │ Analysis │ │ Extract  │ │ Filter   │ │ Analysis   ││
│  └─────┬────┘ └─────┬────┘ └─────┬────┘ └──────┬─────┘│
│        │            │             │              │      │
│        └────────────┴─────────────┴──────────────┘      │
│                          │                              │
│                          ▼                              │
│               ┌──────────────────┐                      │
│               │ Candidate Set    │                      │
│               │ Generation       │                      │
│               └────────┬─────────┘                      │
│                        │                                │
│                        ▼                                │
│               ┌──────────────────┐                      │
│               │ Relevance        │                      │
│               │ Scoring          │                      │
│               └────────┬─────────┘                      │
│                        │                                │
│                        ▼                                │
│               ┌──────────────────┐                      │
│               │ Ranking &        │                      │
│               │ Reranking        │                      │
│               └────────┬─────────┘                      │
│                        │                                │
│                        ▼                                │
│               ┌──────────────────┐                      │
│               │ Retrieved        │                      │
│               │ Memories         │                      │
│               └──────────────────┘                      │
└─────────────────────────────────────────────────────────┘
```

## Retrieval Fundamentals

Core concepts in memory retrieval.

### Memory Item

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime
import numpy as np

@dataclass
class MemoryItem:
    """A memory item with retrieval metadata."""
    id: str
    content: str
    memory_type: str  # 'episodic', 'semantic', 'fact', etc.
    embedding: Optional[np.ndarray] = None
    keywords: List[str] = None
    timestamp: datetime = None
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'content': self.content,
            'memory_type': self.memory_type,
            'keywords': self.keywords,
            'timestamp': self.timestamp.isoformat(),
            'importance': self.importance,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'metadata': self.metadata
        }

@dataclass
class RetrievalQuery:
    """A retrieval query with multiple dimensions."""
    text: str
    embedding: Optional[np.ndarray] = None
    keywords: List[str] = None
    filters: Dict[str, Any] = None
    max_results: int = 10
    min_relevance: float = 0.0
    time_window: Optional[tuple] = None  # (start_time, end_time)
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.filters is None:
            self.filters = {}

@dataclass
class RetrievalResult:
    """Result from memory retrieval."""
    memory: MemoryItem
    relevance_score: float
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    recency_score: float = 0.0
    importance_score: float = 0.0
    
    def __lt__(self, other):
        """For sorting by relevance."""
        return self.relevance_score < other.relevance_score
```

### Base Retriever

```python
from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    """Base class for memory retrievers."""
    
    @abstractmethod
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve memories matching query."""
        pass
    
    def update_access_stats(self, memory: MemoryItem):
        """Update access statistics for a memory."""
        memory.access_count += 1
        memory.last_accessed = datetime.now()
```

## Semantic Search

Vector-based semantic similarity search.

### Semantic Retriever

```python
from typing import Callable
import numpy as np

class SemanticRetriever(BaseRetriever):
    """Retrieve memories using semantic similarity."""
    
    def __init__(self, embedding_fn: Callable[[str], np.ndarray]):
        """
        Args:
            embedding_fn: Function to convert text to embeddings
        """
        self.embedding_fn = embedding_fn
        self.memories: List[MemoryItem] = []
    
    def add_memory(self, memory: MemoryItem):
        """Add memory with embedding."""
        if memory.embedding is None:
            memory.embedding = self.embedding_fn(memory.content)
        self.memories.append(memory)
    
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve using semantic similarity."""
        # Get query embedding
        if query.embedding is None:
            query.embedding = self.embedding_fn(query.text)
        
        results = []
        
        for memory in self.memories:
            # Apply filters
            if not self._passes_filters(memory, query.filters):
                continue
            
            # Apply time window
            if query.time_window:
                start, end = query.time_window
                if not (start <= memory.timestamp <= end):
                    continue
            
            # Calculate semantic similarity
            semantic_score = self._cosine_similarity(
                query.embedding,
                memory.embedding
            )
            
            # Filter by minimum relevance
            if semantic_score < query.min_relevance:
                continue
            
            result = RetrievalResult(
                memory=memory,
                relevance_score=semantic_score,
                semantic_score=semantic_score
            )
            results.append(result)
        
        # Sort by relevance
        results.sort(reverse=True)
        
        # Update access stats
        for result in results[:query.max_results]:
            self.update_access_stats(result.memory)
        
        return results[:query.max_results]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity."""
        if vec1 is None or vec2 is None:
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def _passes_filters(self, memory: MemoryItem, filters: Dict) -> bool:
        """Check if memory passes filters."""
        for key, value in filters.items():
            if key == 'memory_type':
                if memory.memory_type != value:
                    return False
            elif key in memory.metadata:
                if memory.metadata[key] != value:
                    return False
        
        return True


# Usage with simple embeddings (would use real embedding model in practice)
def simple_embedding(text: str) -> np.ndarray:
    """Simple embedding function for demo."""
    # In practice, use sentence-transformers or OpenAI embeddings
    # This is just a dummy implementation
    import hashlib
    hash_value = hashlib.md5(text.encode()).digest()
    embedding = np.frombuffer(hash_value, dtype=np.uint8).astype(float)
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding

semantic_retriever = SemanticRetriever(simple_embedding)

# Add memories
import uuid

memory1 = MemoryItem(
    id=str(uuid.uuid4()),
    content="Python is a high-level programming language",
    memory_type="semantic",
    keywords=["python", "programming"]
)
semantic_retriever.add_memory(memory1)

memory2 = MemoryItem(
    id=str(uuid.uuid4()),
    content="User prefers coding in JavaScript",
    memory_type="episodic",
    keywords=["user", "javascript", "preference"]
)
semantic_retriever.add_memory(memory2)

# Query
query = RetrievalQuery(
    text="programming languages",
    max_results=5
)

results = semantic_retriever.retrieve(query)
print(f"Found {len(results)} results:")
for result in results:
    print(f"  {result.memory.content[:50]}... (score: {result.semantic_score:.3f})")
```

### Vector Index for Scale

```python
class VectorIndexRetriever(BaseRetriever):
    """Efficient retrieval using approximate nearest neighbor search."""
    
    def __init__(self, embedding_fn: Callable, dimension: int):
        self.embedding_fn = embedding_fn
        self.dimension = dimension
        self.memories: List[MemoryItem] = []
        self.index_dirty = True
        
        # Would use FAISS, Annoy, or similar in practice
        # This is a simple brute-force implementation
    
    def add_memory(self, memory: MemoryItem):
        """Add memory and mark index as needing rebuild."""
        if memory.embedding is None:
            memory.embedding = self.embedding_fn(memory.content)
        
        self.memories.append(memory)
        self.index_dirty = True
    
    def build_index(self):
        """Build search index."""
        # In practice, would build FAISS index here
        # For demo, we just mark as clean
        self.index_dirty = False
        print(f"Built index with {len(self.memories)} memories")
    
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve using approximate nearest neighbor search."""
        if self.index_dirty:
            self.build_index()
        
        # Get query embedding
        if query.embedding is None:
            query.embedding = self.embedding_fn(query.text)
        
        # Brute force search (would use index in practice)
        similarities = []
        for i, memory in enumerate(self.memories):
            score = self._cosine_similarity(query.embedding, memory.embedding)
            similarities.append((i, score))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for idx, score in similarities[:query.max_results]:
            if score >= query.min_relevance:
                result = RetrievalResult(
                    memory=self.memories[idx],
                    relevance_score=score,
                    semantic_score=score
                )
                results.append(result)
        
        return results
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity."""
        if vec1 is None or vec2 is None:
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))


# Usage
vector_retriever = VectorIndexRetriever(simple_embedding, dimension=16)
vector_retriever.add_memory(memory1)
vector_retriever.add_memory(memory2)

results = vector_retriever.retrieve(query)
```

## Keyword Search

Traditional keyword-based retrieval.

### Keyword Retriever

```python
from collections import Counter
import re

class KeywordRetriever(BaseRetriever):
    """Retrieve memories using keyword matching."""
    
    def __init__(self):
        self.memories: List[MemoryItem] = []
        self.inverted_index: Dict[str, List[int]] = {}
    
    def add_memory(self, memory: MemoryItem):
        """Add memory and update inverted index."""
        memory_idx = len(self.memories)
        self.memories.append(memory)
        
        # Extract keywords from content if not provided
        if not memory.keywords:
            memory.keywords = self._extract_keywords(memory.content)
        
        # Update inverted index
        for keyword in memory.keywords:
            keyword_lower = keyword.lower()
            if keyword_lower not in self.inverted_index:
                self.inverted_index[keyword_lower] = []
            self.inverted_index[keyword_lower].append(memory_idx)
    
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve using keyword matching."""
        # Extract query keywords
        query_keywords = query.keywords or self._extract_keywords(query.text)
        
        # Find candidate memories
        candidate_scores = Counter()
        
        for keyword in query_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in self.inverted_index:
                for memory_idx in self.inverted_index[keyword_lower]:
                    candidate_scores[memory_idx] += 1
        
        # Build results
        results = []
        
        for memory_idx, keyword_count in candidate_scores.most_common():
            memory = self.memories[memory_idx]
            
            # Calculate keyword score
            keyword_score = keyword_count / max(len(query_keywords), 1)
            
            if keyword_score >= query.min_relevance:
                result = RetrievalResult(
                    memory=memory,
                    relevance_score=keyword_score,
                    keyword_score=keyword_score
                )
                results.append(result)
        
        # Update access stats
        for result in results[:query.max_results]:
            self.update_access_stats(result.memory)
        
        return results[:query.max_results]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple word tokenization
        words = re.findall(r'\w+', text.lower())
        
        # Remove common stopwords
        stopwords = {'the', 'is', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at'}
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return keywords


# Usage
keyword_retriever = KeywordRetriever()
keyword_retriever.add_memory(memory1)
keyword_retriever.add_memory(memory2)

query = RetrievalQuery(
    text="python programming language",
    max_results=5
)

results = keyword_retriever.retrieve(query)
print(f"\nKeyword search found {len(results)} results:")
for result in results:
    print(f"  {result.memory.content[:50]}... (score: {result.keyword_score:.3f})")
```

### TF-IDF Retriever

```python
import math

class TFIDFRetriever(BaseRetriever):
    """Retrieve using TF-IDF scoring."""
    
    def __init__(self):
        self.memories: List[MemoryItem] = []
        self.document_frequencies: Dict[str, int] = {}
        self.needs_rebuild = True
    
    def add_memory(self, memory: MemoryItem):
        """Add memory."""
        self.memories.append(memory)
        self.needs_rebuild = True
    
    def build_index(self):
        """Build TF-IDF index."""
        # Count document frequencies
        self.document_frequencies = {}
        
        for memory in self.memories:
            keywords = set(self._extract_keywords(memory.content))
            for keyword in keywords:
                self.document_frequencies[keyword] = \
                    self.document_frequencies.get(keyword, 0) + 1
        
        self.needs_rebuild = False
    
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve using TF-IDF."""
        if self.needs_rebuild:
            self.build_index()
        
        query_keywords = self._extract_keywords(query.text)
        
        # Calculate TF-IDF scores for each memory
        scores = []
        
        for memory in self.memories:
            score = self._calculate_tfidf_score(memory, query_keywords)
            
            if score >= query.min_relevance:
                result = RetrievalResult(
                    memory=memory,
                    relevance_score=score,
                    keyword_score=score
                )
                scores.append(result)
        
        # Sort by score
        scores.sort(reverse=True)
        
        return scores[:query.max_results]
    
    def _calculate_tfidf_score(self, memory: MemoryItem,
                               query_keywords: List[str]) -> float:
        """Calculate TF-IDF score for memory against query."""
        content_keywords = self._extract_keywords(memory.content)
        
        # Count term frequencies in memory
        term_frequencies = Counter(content_keywords)
        
        total_score = 0.0
        
        for query_term in query_keywords:
            if query_term in term_frequencies:
                # Term frequency
                tf = term_frequencies[query_term] / len(content_keywords)
                
                # Inverse document frequency
                df = self.document_frequencies.get(query_term, 1)
                idf = math.log(len(self.memories) / df)
                
                # TF-IDF score
                total_score += tf * idf
        
        return total_score
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        words = re.findall(r'\w+', text.lower())
        stopwords = {'the', 'is', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at'}
        return [w for w in words if w not in stopwords and len(w) > 2]


# Usage
tfidf_retriever = TFIDFRetriever()
tfidf_retriever.add_memory(memory1)
tfidf_retriever.add_memory(memory2)

results = tfidf_retriever.retrieve(query)
print(f"\nTF-IDF search found {len(results)} results:")
for result in results:
    print(f"  {result.memory.content[:50]}... (score: {result.keyword_score:.3f})")
```

## Recency-Based Retrieval

Prioritizing recent memories.

### Recency Retriever

```python
class RecencyRetriever(BaseRetriever):
    """Retrieve memories with recency bias."""
    
    def __init__(self, decay_factor: float = 0.99):
        """
        Args:
            decay_factor: How quickly older memories decay (0-1)
        """
        self.memories: List[MemoryItem] = []
        self.decay_factor = decay_factor
    
    def add_memory(self, memory: MemoryItem):
        """Add memory."""
        self.memories.append(memory)
    
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve with recency bias."""
        current_time = datetime.now()
        
        results = []
        
        for memory in self.memories:
            # Apply time window filter
            if query.time_window:
                start, end = query.time_window
                if not (start <= memory.timestamp <= end):
                    continue
            
            # Calculate recency score
            time_diff = (current_time - memory.timestamp).total_seconds()
            days_ago = time_diff / (24 * 3600)
            
            # Exponential decay
            recency_score = math.pow(self.decay_factor, days_ago)
            
            if recency_score >= query.min_relevance:
                result = RetrievalResult(
                    memory=memory,
                    relevance_score=recency_score,
                    recency_score=recency_score
                )
                results.append(result)
        
        # Sort by recency
        results.sort(reverse=True)
        
        return results[:query.max_results]


# Usage
recency_retriever = RecencyRetriever(decay_factor=0.95)

# Add memories with different timestamps
from datetime import timedelta

old_memory = MemoryItem(
    id=str(uuid.uuid4()),
    content="Old fact about Python 2",
    memory_type="semantic",
    timestamp=datetime.now() - timedelta(days=30)
)

recent_memory = MemoryItem(
    id=str(uuid.uuid4()),
    content="Recent fact about Python 3.12",
    memory_type="semantic",
    timestamp=datetime.now() - timedelta(hours=1)
)

recency_retriever.add_memory(old_memory)
recency_retriever.add_memory(recent_memory)

query = RetrievalQuery(text="Python", max_results=10)
results = recency_retriever.retrieve(query)

print("\nRecency-based retrieval:")
for result in results:
    print(f"  {result.memory.content[:40]}... (recency: {result.recency_score:.3f})")
```

### Time-Decayed Retrieval

```python
class TimeDecayedRetriever(BaseRetriever):
    """Retrieve with customizable time decay functions."""
    
    def __init__(self, decay_fn: Callable[[float], float] = None):
        """
        Args:
            decay_fn: Function that takes days_ago and returns score (0-1)
        """
        self.memories: List[MemoryItem] = []
        self.decay_fn = decay_fn or self._exponential_decay
    
    def add_memory(self, memory: MemoryItem):
        """Add memory."""
        self.memories.append(memory)
    
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve with time decay."""
        current_time = datetime.now()
        
        results = []
        
        for memory in self.memories:
            time_diff = (current_time - memory.timestamp).total_seconds()
            days_ago = time_diff / (24 * 3600)
            
            # Apply decay function
            recency_score = self.decay_fn(days_ago)
            
            result = RetrievalResult(
                memory=memory,
                relevance_score=recency_score,
                recency_score=recency_score
            )
            results.append(result)
        
        results.sort(reverse=True)
        return results[:query.max_results]
    
    def _exponential_decay(self, days_ago: float, half_life: float = 7.0) -> float:
        """Exponential decay with configurable half-life."""
        return math.pow(0.5, days_ago / half_life)
    
    @staticmethod
    def linear_decay(days_ago: float, max_days: float = 30.0) -> float:
        """Linear decay over max_days."""
        return max(0.0, 1.0 - (days_ago / max_days))
    
    @staticmethod
    def step_decay(days_ago: float, thresholds: List[tuple] = None) -> float:
        """Step function decay."""
        if thresholds is None:
            thresholds = [(7, 1.0), (30, 0.5), (90, 0.1)]
        
        for threshold_days, score in thresholds:
            if days_ago <= threshold_days:
                return score
        
        return 0.0


# Usage with different decay functions
exp_retriever = TimeDecayedRetriever()  # Exponential
lin_retriever = TimeDecayedRetriever(TimeDecayedRetriever.linear_decay)
step_retriever = TimeDecayedRetriever(TimeDecayedRetriever.step_decay)
```

## Importance Weighting

Prioritizing important memories.

### Importance Retriever

```python
class ImportanceRetriever(BaseRetriever):
    """Retrieve memories by importance."""
    
    def __init__(self):
        self.memories: List[MemoryItem] = []
    
    def add_memory(self, memory: MemoryItem):
        """Add memory."""
        self.memories.append(memory)
    
    def set_importance(self, memory_id: str, importance: float):
        """Set importance score for a memory."""
        for memory in self.memories:
            if memory.id == memory_id:
                memory.importance = max(0.0, min(1.0, importance))
                break
    
    def auto_calculate_importance(self, memory: MemoryItem) -> float:
        """
        Automatically calculate importance based on heuristics.
        
        Factors:
        - Access frequency
        - Recency of last access
        - Content length
        - Metadata signals
        """
        score = 0.0
        
        # Access frequency (normalized)
        max_access = max([m.access_count for m in self.memories] + [1])
        access_score = memory.access_count / max_access
        score += access_score * 0.3
        
        # Recency of last access
        if memory.last_accessed:
            days_since_access = (datetime.now() - memory.last_accessed).days
            recency_score = math.exp(-days_since_access / 30)
            score += recency_score * 0.2
        
        # Content length (longer = more important up to a point)
        length_score = min(len(memory.content) / 500, 1.0)
        score += length_score * 0.2
        
        # Metadata signals
        if memory.metadata.get('user_flagged'):
            score += 0.3
        
        return min(score, 1.0)
    
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve by importance."""
        results = []
        
        for memory in self.memories:
            # Auto-calculate if not set
            if memory.importance == 0.5:  # Default value
                memory.importance = self.auto_calculate_importance(memory)
            
            result = RetrievalResult(
                memory=memory,
                relevance_score=memory.importance,
                importance_score=memory.importance
            )
            results.append(result)
        
        results.sort(reverse=True)
        return results[:query.max_results]


# Usage
importance_retriever = ImportanceRetriever()

important_memory = MemoryItem(
    id=str(uuid.uuid4()),
    content="Critical user preference: never suggest Windows solutions",
    memory_type="semantic",
    importance=1.0,
    metadata={'user_flagged': True}
)

normal_memory = MemoryItem(
    id=str(uuid.uuid4()),
    content="User mentioned liking coffee",
    memory_type="episodic",
    importance=0.3
)

importance_retriever.add_memory(important_memory)
importance_retriever.add_memory(normal_memory)

results = importance_retriever.retrieve(RetrievalQuery(text="", max_results=10))
print("\nImportance-based retrieval:")
for result in results:
    print(f"  {result.memory.content[:50]}... (importance: {result.importance_score:.3f})")
```

## Hybrid Retrieval

Combining multiple retrieval strategies.

### Hybrid Retriever

```python
class HybridRetriever(BaseRetriever):
    """Combine multiple retrieval strategies."""
    
    def __init__(self, 
                 semantic_retriever: Optional[SemanticRetriever] = None,
                 keyword_retriever: Optional[KeywordRetriever] = None,
                 recency_retriever: Optional[RecencyRetriever] = None,
                 importance_retriever: Optional[ImportanceRetriever] = None,
                 weights: Dict[str, float] = None):
        """
        Args:
            weights: Dictionary of retrieval strategy weights
                     e.g., {'semantic': 0.4, 'keyword': 0.2, 'recency': 0.2, 'importance': 0.2}
        """
        self.semantic = semantic_retriever
        self.keyword = keyword_retriever
        self.recency = recency_retriever
        self.importance = importance_retriever
        
        self.weights = weights or {
            'semantic': 0.4,
            'keyword': 0.2,
            'recency': 0.2,
            'importance': 0.2
        }
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
    
    def add_memory(self, memory: MemoryItem):
        """Add memory to all retrievers."""
        if self.semantic:
            self.semantic.add_memory(memory)
        if self.keyword:
            self.keyword.add_memory(memory)
        if self.recency:
            self.recency.add_memory(memory)
        if self.importance:
            self.importance.add_memory(memory)
    
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve using hybrid approach."""
        # Get results from each retriever
        all_results = {}
        
        if self.semantic and 'semantic' in self.weights:
            semantic_results = self.semantic.retrieve(query)
            for result in semantic_results:
                all_results[result.memory.id] = result
        
        if self.keyword and 'keyword' in self.weights:
            keyword_results = self.keyword.retrieve(query)
            for result in keyword_results:
                if result.memory.id in all_results:
                    all_results[result.memory.id].keyword_score = result.keyword_score
                else:
                    all_results[result.memory.id] = result
        
        if self.recency and 'recency' in self.weights:
            recency_results = self.recency.retrieve(query)
            for result in recency_results:
                if result.memory.id in all_results:
                    all_results[result.memory.id].recency_score = result.recency_score
                else:
                    all_results[result.memory.id] = result
        
        if self.importance and 'importance' in self.weights:
            importance_results = self.importance.retrieve(query)
            for result in importance_results:
                if result.memory.id in all_results:
                    all_results[result.memory.id].importance_score = result.importance_score
                else:
                    all_results[result.memory.id] = result
        
        # Calculate combined scores
        for memory_id, result in all_results.items():
            combined_score = (
                result.semantic_score * self.weights.get('semantic', 0) +
                result.keyword_score * self.weights.get('keyword', 0) +
                result.recency_score * self.weights.get('recency', 0) +
                result.importance_score * self.weights.get('importance', 0)
            )
            result.relevance_score = combined_score
        
        # Sort by combined score
        results = list(all_results.values())
        results.sort(reverse=True)
        
        return results[:query.max_results]


# Usage
hybrid = HybridRetriever(
    semantic_retriever=semantic_retriever,
    keyword_retriever=keyword_retriever,
    recency_retriever=recency_retriever,
    importance_retriever=importance_retriever,
    weights={'semantic': 0.5, 'keyword': 0.2, 'recency': 0.2, 'importance': 0.1}
)

# Add memories
test_memory = MemoryItem(
    id=str(uuid.uuid4()),
    content="Python 3.12 introduces new features for async programming",
    memory_type="semantic",
    keywords=["python", "async", "programming"],
    importance=0.8,
    timestamp=datetime.now() - timedelta(days=2)
)
hybrid.add_memory(test_memory)

query = RetrievalQuery(text="python async", max_results=5)
results = hybrid.retrieve(query)

print("\nHybrid retrieval:")
for result in results:
    print(f"  {result.memory.content[:50]}...")
    print(f"    Combined: {result.relevance_score:.3f} "
          f"(sem: {result.semantic_score:.3f}, "
          f"kw: {result.keyword_score:.3f}, "
          f"rec: {result.recency_score:.3f}, "
          f"imp: {result.importance_score:.3f})")
```

### Adaptive Weight Adjustment

```python
class AdaptiveHybridRetriever(HybridRetriever):
    """Hybrid retriever that adapts weights based on context."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_history = []
    
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve with adaptive weights."""
        # Adjust weights based on query characteristics
        adjusted_weights = self._adjust_weights(query)
        
        # Temporarily update weights
        original_weights = self.weights.copy()
        self.weights = adjusted_weights
        
        # Retrieve
        results = super().retrieve(query)
        
        # Restore weights
        self.weights = original_weights
        
        # Track query
        self.query_history.append({
            'query': query.text,
            'timestamp': datetime.now(),
            'result_count': len(results)
        })
        
        return results
    
    def _adjust_weights(self, query: RetrievalQuery) -> Dict[str, float]:
        """Adjust weights based on query characteristics."""
        weights = self.weights.copy()
        
        # If time window specified, boost recency
        if query.time_window:
            weights['recency'] *= 1.5
        
        # If specific keywords, boost keyword search
        if query.keywords and len(query.keywords) > 2:
            weights['keyword'] *= 1.3
        
        # If short query (likely specific), boost semantic
        if len(query.text.split()) <= 3:
            weights['semantic'] *= 1.2
        
        # Normalize
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}


# Usage
adaptive = AdaptiveHybridRetriever(
    semantic_retriever=semantic_retriever,
    keyword_retriever=keyword_retriever,
    recency_retriever=recency_retriever,
    importance_retriever=importance_retriever
)
```

## Relevance Scoring

Advanced relevance computation.

### Relevance Scorer

```python
class RelevanceScorer:
    """Advanced relevance scoring."""
    
    @staticmethod
    def bm25_score(query_terms: List[str], document_terms: List[str],
                   doc_freq: Dict[str, int], num_docs: int,
                   k1: float = 1.5, b: float = 0.75, avg_doc_len: float = 100) -> float:
        """
        Calculate BM25 relevance score.
        
        Args:
            query_terms: Terms in query
            document_terms: Terms in document
            doc_freq: Document frequency for each term
            num_docs: Total number of documents
            k1: Term frequency saturation parameter
            b: Length normalization parameter
            avg_doc_len: Average document length
        """
        term_freq = Counter(document_terms)
        doc_len = len(document_terms)
        
        score = 0.0
        
        for term in query_terms:
            if term not in term_freq:
                continue
            
            # Term frequency component
            tf = term_freq[term]
            tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
            
            # IDF component
            df = doc_freq.get(term, 1)
            idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1)
            
            score += idf * tf_component
        
        return score
    
    @staticmethod
    def diversify_results(results: List[RetrievalResult],
                         diversity_factor: float = 0.5) -> List[RetrievalResult]:
        """
        Maximal Marginal Relevance (MMR) for diversity.
        
        Args:
            results: Initial results sorted by relevance
            diversity_factor: Weight for diversity (0=pure relevance, 1=pure diversity)
        """
        if len(results) <= 1:
            return results
        
        selected = [results[0]]  # Always take top result
        remaining = results[1:]
        
        while remaining and len(selected) < len(results):
            # For each remaining result, compute MMR score
            mmr_scores = []
            
            for candidate in remaining:
                relevance = candidate.relevance_score
                
                # Max similarity to already selected
                max_sim = 0.0
                for selected_result in selected:
                    sim = RelevanceScorer._similarity(
                        candidate.memory,
                        selected_result.memory
                    )
                    max_sim = max(max_sim, sim)
                
                # MMR score
                mmr = diversity_factor * relevance - (1 - diversity_factor) * max_sim
                mmr_scores.append((candidate, mmr))
            
            # Select highest MMR score
            best = max(mmr_scores, key=lambda x: x[1])
            selected.append(best[0])
            remaining.remove(best[0])
        
        return selected
    
    @staticmethod
    def _similarity(mem1: MemoryItem, mem2: MemoryItem) -> float:
        """Calculate similarity between two memories."""
        if mem1.embedding is not None and mem2.embedding is not None:
            # Use embedding similarity
            dot_product = np.dot(mem1.embedding, mem2.embedding)
            norm1 = np.linalg.norm(mem1.embedding)
            norm2 = np.linalg.norm(mem2.embedding)
            
            if norm1 > 0 and norm2 > 0:
                return float(dot_product / (norm1 * norm2))
        
        # Fallback to keyword overlap
        keywords1 = set(mem1.keywords)
        keywords2 = set(mem2.keywords)
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = keywords1 & keywords2
        union = keywords1 | keywords2
        
        return len(intersection) / len(union)


# Usage
diversified = RelevanceScorer.diversify_results(results, diversity_factor=0.7)
print("\nDiversified results:")
for result in diversified:
    print(f"  {result.memory.content[:50]}...")
```

## Retrieval Strategies

Different strategies for different use cases.

### Top-K Retrieval

```python
class TopKRetriever:
    """Retrieve top-K most relevant memories."""
    
    def __init__(self, base_retriever: BaseRetriever, k: int = 10):
        self.base_retriever = base_retriever
        self.k = k
    
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve top-K results."""
        query.max_results = self.k
        return self.base_retriever.retrieve(query)
```

### Threshold Retrieval

```python
class ThresholdRetriever:
    """Retrieve all memories above relevance threshold."""
    
    def __init__(self, base_retriever: BaseRetriever,
                 threshold: float = 0.5):
        self.base_retriever = base_retriever
        self.threshold = threshold
    
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve above threshold."""
        query.min_relevance = self.threshold
        query.max_results = 1000  # Get many candidates
        
        results = self.base_retriever.retrieve(query)
        
        # Filter by threshold
        return [r for r in results if r.relevance_score >= self.threshold]
```

### Sliding Window Retrieval

```python
class SlidingWindowRetriever:
    """Retrieve memories in a sliding time window."""
    
    def __init__(self, base_retriever: BaseRetriever,
                 window_size: timedelta = timedelta(days=7)):
        self.base_retriever = base_retriever
        self.window_size = window_size
    
    def retrieve(self, query: RetrievalQuery,
                 window_end: datetime = None) -> List[RetrievalResult]:
        """Retrieve within time window."""
        if window_end is None:
            window_end = datetime.now()
        
        window_start = window_end - self.window_size
        
        query.time_window = (window_start, window_end)
        
        return self.base_retriever.retrieve(query)
```

## Query Expansion

Expanding queries for better retrieval.

### Query Expander

```python
class QueryExpander:
    """Expand queries with synonyms and related terms."""
    
    def __init__(self):
        # Simple synonym dictionary (would use WordNet or embeddings in practice)
        self.synonyms = {
            'python': ['python', 'py'],
            'code': ['code', 'program', 'script'],
            'function': ['function', 'method', 'procedure'],
            'error': ['error', 'exception', 'bug', 'issue'],
        }
    
    def expand(self, query: str) -> List[str]:
        """Expand query with synonyms."""
        words = query.lower().split()
        
        expanded_terms = []
        
        for word in words:
            # Add original word
            expanded_terms.append(word)
            
            # Add synonyms
            if word in self.synonyms:
                expanded_terms.extend(self.synonyms[word])
        
        return list(set(expanded_terms))
    
    def expand_query(self, query: RetrievalQuery) -> RetrievalQuery:
        """Create expanded query."""
        expanded_text = " ".join(self.expand(query.text))
        
        return RetrievalQuery(
            text=expanded_text,
            keywords=query.keywords,
            filters=query.filters,
            max_results=query.max_results,
            min_relevance=query.min_relevance,
            time_window=query.time_window
        )


# Usage
expander = QueryExpander()
original_query = RetrievalQuery(text="python error")
expanded_query = expander.expand_query(original_query)

print(f"\nOriginal: {original_query.text}")
print(f"Expanded: {expanded_query.text}")
```

## Retrieval Optimization

Optimizing retrieval performance.

### Caching Retriever

```python
from functools import lru_cache

class CachingRetriever:
    """Cache retrieval results."""
    
    def __init__(self, base_retriever: BaseRetriever,
                 cache_size: int = 100,
                 cache_ttl: timedelta = timedelta(minutes=5)):
        self.base_retriever = base_retriever
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.cache = {}
    
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve with caching."""
        cache_key = self._make_cache_key(query)
        
        # Check cache
        if cache_key in self.cache:
            cached_results, cached_time = self.cache[cache_key]
            
            # Check if still valid
            if datetime.now() - cached_time < self.cache_ttl:
                return cached_results
        
        # Not in cache or expired, retrieve
        results = self.base_retriever.retrieve(query)
        
        # Cache results
        self.cache[cache_key] = (results, datetime.now())
        
        # Evict old entries if cache too large
        if len(self.cache) > self.cache_size:
            oldest_key = min(self.cache.keys(),
                           key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        return results
    
    def _make_cache_key(self, query: RetrievalQuery) -> str:
        """Create cache key from query."""
        return f"{query.text}_{query.max_results}_{query.min_relevance}"
    
    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()


# Usage
cached_retriever = CachingRetriever(semantic_retriever)
```

## Context-Aware Retrieval

Retrieval that considers current context.

### Context-Aware Retriever

```python
class ContextAwareRetriever:
    """Retrieve memories relevant to current context."""
    
    def __init__(self, base_retriever: BaseRetriever):
        self.base_retriever = base_retriever
        self.conversation_history = []
    
    def add_context(self, context: str):
        """Add to conversation context."""
        self.conversation_history.append({
            'content': context,
            'timestamp': datetime.now()
        })
        
        # Keep only recent context (last 10 turns)
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve with conversation context."""
        # Enhance query with recent context
        context_text = " ".join([
            turn['content']
            for turn in self.conversation_history[-3:]
        ])
        
        enhanced_query = RetrievalQuery(
            text=f"{context_text} {query.text}",
            keywords=query.keywords,
            filters=query.filters,
            max_results=query.max_results,
            min_relevance=query.min_relevance,
            time_window=query.time_window
        )
        
        return self.base_retriever.retrieve(enhanced_query)


# Usage
context_retriever = ContextAwareRetriever(semantic_retriever)
context_retriever.add_context("User is learning Python")
context_retriever.add_context("User asked about async programming")

query = RetrievalQuery(text="how do I do it?", max_results=5)
results = context_retriever.retrieve(query)
```

## Summary

Memory retrieval is crucial for agent performance:

**Retrieval Methods**:

- **Semantic search**: Vector similarity for conceptual matching
- **Keyword search**: Traditional text matching with TF-IDF
- **Recency-based**: Time decay functions for fresh information
- **Importance-based**: Prioritizing significant memories
- **Hybrid**: Combining multiple strategies with weighted scores

**Key Techniques**:

- **Relevance scoring**: BM25, TF-IDF, cosine similarity
- **Diversification**: MMR for non-redundant results
- **Query expansion**: Synonyms and related terms
- **Caching**: Performance optimization
- **Context-awareness**: Using conversation history

**Trade-offs**:

- Semantic vs keyword: Understanding vs precision
- Recency vs importance: New vs significant
- Speed vs accuracy: Index size vs quality
- Diversity vs relevance: Variety vs focus

## Next Steps

- **[Memory Consolidation](memory-consolidation.md)**: When to persist retrieved memories
- **[Short-Term Memory](short-term-memory.md)**: Using retrieved memories in context window
- **[Episodic Memory](episodic-memory.md)**: Retrieving past experiences
- **[Semantic Memory](semantic-memory.md)**: Querying factual knowledge
