# Semantic Memory

## Table of Contents

- [Introduction](#introduction)
- [What is Semantic Memory](#what-is-semantic-memory)
- [Knowledge Representation](#knowledge-representation)
- [Entity Management](#entity-management)
- [Relation Modeling](#relation-modeling)
- [Fact Storage](#fact-storage)
- [Knowledge Graphs](#knowledge-graphs)
- [Domain Knowledge](#domain-knowledge)
- [Concept Learning](#concept-learning)
- [Knowledge Updates](#knowledge-updates)
- [Querying Semantic Memory](#querying-semantic-memory)
- [Reasoning with Knowledge](#reasoning-with-knowledge)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Semantic memory stores factual knowledge and learned concepts - the "what I know" of an agent's understanding. Unlike episodic memory that stores experiences, semantic memory stores timeless facts, entities, relationships, and concepts.

> "Semantic memory is the encyclopedia of an agent's mind - facts, concepts, and relationships that persist beyond individual experiences."

Semantic memory enables agents to:

- **Store facts** about users, domains, and the world
- **Build knowledge** systematically over time
- **Understand relationships** between entities and concepts
- **Answer questions** using accumulated knowledge
- **Make inferences** from known facts

### Key Characteristics

**Decontextualized**: Facts separated from specific experiences

**Structured**: Organized as entities, attributes, and relations

**Persistent**: Knowledge that remains valid over time

**Inferential**: Can derive new knowledge from existing facts

## What is Semantic Memory

Semantic memory contrasts with episodic memory in focus and structure.

### Semantic vs Episodic

```
┌────────────────────────────────────────────────────────────┐
│ Aspect       │ Episodic Memory        │ Semantic Memory   │
├──────────────┼────────────────────────┼───────────────────┤
│ Content      │ "What happened"        │ "What I know"     │
│ Example      │ "User asked about      │ "Python is a      │
│              │  Python yesterday"     │  language"        │
│ Structure    │ Temporal sequences     │ Facts & relations │
│ Context      │ Situation-specific     │ Context-free      │
│ Change       │ Immutable (history)    │ Updates with      │
│              │                        │  new information  │
└────────────────────────────────────────────────────────────┘
```

### Types of Semantic Knowledge

```python
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime

class KnowledgeType(Enum):
    """Types of semantic knowledge."""
    ENTITY = "entity"          # People, places, things
    ATTRIBUTE = "attribute"    # Properties of entities
    RELATION = "relation"      # Connections between entities
    FACT = "fact"             # General facts
    CONCEPT = "concept"       # Abstract ideas
    RULE = "rule"             # Conditional knowledge

@dataclass
class Entity:
    """An entity in semantic memory."""
    id: str
    name: str
    entity_type: str  # person, place, organization, concept, etc.
    attributes: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    confidence: float = 1.0
    source: str = "learned"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'entity_type': self.entity_type,
            'attributes': self.attributes,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'confidence': self.confidence,
            'source': self.source
        }

@dataclass
class Relation:
    """A relation between entities."""
    subject_id: str
    predicate: str  # The type of relationship
    object_id: str
    confidence: float = 1.0
    source: str = "learned"
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_triple(self) -> tuple:
        """Convert to RDF-style triple."""
        return (self.subject_id, self.predicate, self.object_id)

@dataclass
class Fact:
    """A standalone fact."""
    id: str
    statement: str
    confidence: float = 1.0
    source: str = "learned"
    tags: List[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now()
```

## Knowledge Representation

Structuring semantic knowledge effectively.

### Semantic Memory Store

```python
import sqlite3
import json
from typing import List, Optional, Dict, Any

class SemanticMemoryStore:
    """Storage for semantic knowledge."""
    
    def __init__(self, db_path: str = "semantic_memory.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()
    
    def _create_schema(self):
        """Create database schema."""
        
        # Entities table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                attributes TEXT,
                confidence REAL DEFAULT 1.0,
                source TEXT,
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
                source TEXT,
                created_at DATETIME NOT NULL,
                FOREIGN KEY (subject_id) REFERENCES entities(id),
                FOREIGN KEY (object_id) REFERENCES entities(id)
            )
        """)
        
        # Facts table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                statement TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT,
                tags TEXT,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
        """)
        
        # Concepts table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                definition TEXT,
                examples TEXT,
                related_concepts TEXT,
                created_at DATETIME NOT NULL
            )
        """)
        
        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)",
            "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)",
            "CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject_id)",
            "CREATE INDEX IF NOT EXISTS idx_relations_object ON relations(object_id)",
            "CREATE INDEX IF NOT EXISTS idx_relations_predicate ON relations(predicate)",
            "CREATE INDEX IF NOT EXISTS idx_facts_tags ON facts(tags)",
        ]
        
        for index in indexes:
            self.conn.execute(index)
        
        self.conn.commit()
    
    def add_entity(self, entity: Entity):
        """Add or update entity."""
        self.conn.execute("""
            INSERT OR REPLACE INTO entities
            (id, name, entity_type, attributes, confidence, source, 
             created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entity.id,
            entity.name,
            entity.entity_type,
            json.dumps(entity.attributes),
            entity.confidence,
            entity.source,
            entity.created_at.isoformat(),
            entity.updated_at.isoformat()
        ))
        self.conn.commit()
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM entities WHERE id = ?",
            (entity_id,)
        )
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return Entity(
            id=row['id'],
            name=row['name'],
            entity_type=row['entity_type'],
            attributes=json.loads(row['attributes']),
            confidence=row['confidence'],
            source=row['source'],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at'])
        )
    
    def search_entities(self, query: str = None, entity_type: str = None,
                       limit: int = 10) -> List[Entity]:
        """Search for entities."""
        conditions = []
        params = []
        
        if query:
            conditions.append("name LIKE ?")
            params.append(f"%{query}%")
        
        if entity_type:
            conditions.append("entity_type = ?")
            params.append(entity_type)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)
        
        cursor = self.conn.execute(f"""
            SELECT * FROM entities
            WHERE {where_clause}
            ORDER BY updated_at DESC
            LIMIT ?
        """, params)
        
        entities = []
        for row in cursor.fetchall():
            entities.append(Entity(
                id=row['id'],
                name=row['name'],
                entity_type=row['entity_type'],
                attributes=json.loads(row['attributes']),
                confidence=row['confidence'],
                source=row['source'],
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at'])
            ))
        
        return entities
    
    def add_relation(self, relation: Relation):
        """Add a relation."""
        self.conn.execute("""
            INSERT INTO relations
            (subject_id, predicate, object_id, confidence, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            relation.subject_id,
            relation.predicate,
            relation.object_id,
            relation.confidence,
            relation.source,
            relation.created_at.isoformat()
        ))
        self.conn.commit()
    
    def get_relations(self, entity_id: str = None, predicate: str = None) -> List[Relation]:
        """Get relations involving an entity or with a specific predicate."""
        conditions = []
        params = []
        
        if entity_id:
            conditions.append("(subject_id = ? OR object_id = ?)")
            params.extend([entity_id, entity_id])
        
        if predicate:
            conditions.append("predicate = ?")
            params.append(predicate)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        cursor = self.conn.execute(f"""
            SELECT * FROM relations
            WHERE {where_clause}
            ORDER BY created_at DESC
        """, params)
        
        relations = []
        for row in cursor.fetchall():
            relations.append(Relation(
                subject_id=row['subject_id'],
                predicate=row['predicate'],
                object_id=row['object_id'],
                confidence=row['confidence'],
                source=row['source'],
                created_at=datetime.fromisoformat(row['created_at'])
            ))
        
        return relations
    
    def add_fact(self, fact: Fact):
        """Add a fact."""
        self.conn.execute("""
            INSERT OR REPLACE INTO facts
            (id, statement, confidence, source, tags, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            fact.id,
            fact.statement,
            fact.confidence,
            fact.source,
            json.dumps(fact.tags),
            fact.created_at.isoformat(),
            datetime.now().isoformat()
        ))
        self.conn.commit()
    
    def search_facts(self, query: str = None, tags: List[str] = None,
                    min_confidence: float = 0.0, limit: int = 10) -> List[Fact]:
        """Search for facts."""
        conditions = ["confidence >= ?"]
        params = [min_confidence]
        
        if query:
            conditions.append("statement LIKE ?")
            params.append(f"%{query}%")
        
        where_clause = " AND ".join(conditions)
        params.append(limit)
        
        cursor = self.conn.execute(f"""
            SELECT * FROM facts
            WHERE {where_clause}
            ORDER BY confidence DESC, updated_at DESC
            LIMIT ?
        """, params)
        
        facts = []
        for row in cursor.fetchall():
            fact = Fact(
                id=row['id'],
                statement=row['statement'],
                confidence=row['confidence'],
                source=row['source'],
                tags=json.loads(row['tags']),
                created_at=datetime.fromisoformat(row['created_at'])
            )
            
            # Filter by tags if specified
            if tags and not any(tag in fact.tags for tag in tags):
                continue
            
            facts.append(fact)
        
        return facts[:limit]


# Usage
import uuid

semantic_store = SemanticMemoryStore()

# Add entity
alice = Entity(
    id="user_alice",
    name="Alice",
    entity_type="person",
    attributes={
        "role": "user",
        "preferences": {"language": "Python", "theme": "dark"}
    },
    created_at=datetime.now(),
    updated_at=datetime.now()
)
semantic_store.add_entity(alice)

# Add relation
python_entity = Entity(
    id="lang_python",
    name="Python",
    entity_type="programming_language",
    attributes={"paradigm": "multi-paradigm"},
    created_at=datetime.now(),
    updated_at=datetime.now()
)
semantic_store.add_entity(python_entity)

knows_relation = Relation(
    subject_id="user_alice",
    predicate="knows",
    object_id="lang_python",
    confidence=0.9
)
semantic_store.add_relation(knows_relation)

# Add fact
fact = Fact(
    id=str(uuid.uuid4()),
    statement="Python is a high-level programming language",
    confidence=1.0,
    source="common_knowledge",
    tags=["python", "programming", "language"]
)
semantic_store.add_fact(fact)
```

## Entity Management

Managing entities in semantic memory.

### Entity Manager

```python
class EntityManager:
    """Manage entities in semantic memory."""
    
    def __init__(self, store: SemanticMemoryStore):
        self.store = store
    
    def create_or_update_entity(self, name: str, entity_type: str,
                               attributes: Dict = None,
                               confidence: float = 1.0) -> str:
        """Create new entity or update existing."""
        # Check if entity exists
        existing = self.store.search_entities(query=name, entity_type=entity_type, limit=1)
        
        if existing:
            # Update existing entity
            entity = existing[0]
            entity.attributes.update(attributes or {})
            entity.updated_at = datetime.now()
            entity.confidence = max(entity.confidence, confidence)
        else:
            # Create new entity
            entity = Entity(
                id=self._generate_entity_id(name, entity_type),
                name=name,
                entity_type=entity_type,
                attributes=attributes or {},
                confidence=confidence,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        
        self.store.add_entity(entity)
        return entity.id
    
    def update_attribute(self, entity_id: str, attribute: str, value: Any):
        """Update a single attribute of an entity."""
        entity = self.store.get_entity(entity_id)
        
        if not entity:
            raise ValueError(f"Entity not found: {entity_id}")
        
        entity.attributes[attribute] = value
        entity.updated_at = datetime.now()
        
        self.store.add_entity(entity)
    
    def merge_entities(self, entity_id1: str, entity_id2: str) -> str:
        """Merge two entities into one."""
        entity1 = self.store.get_entity(entity_id1)
        entity2 = self.store.get_entity(entity_id2)
        
        if not entity1 or not entity2:
            raise ValueError("Both entities must exist")
        
        # Merge attributes
        merged_attributes = {**entity1.attributes, **entity2.attributes}
        
        # Keep higher confidence
        merged_confidence = max(entity1.confidence, entity2.confidence)
        
        # Update first entity
        entity1.attributes = merged_attributes
        entity1.confidence = merged_confidence
        entity1.updated_at = datetime.now()
        
        self.store.add_entity(entity1)
        
        # Update relations pointing to entity2
        relations = self.store.get_relations(entity_id=entity_id2)
        for rel in relations:
            if rel.subject_id == entity_id2:
                new_rel = Relation(
                    subject_id=entity_id1,
                    predicate=rel.predicate,
                    object_id=rel.object_id,
                    confidence=rel.confidence,
                    source=rel.source
                )
            else:
                new_rel = Relation(
                    subject_id=rel.subject_id,
                    predicate=rel.predicate,
                    object_id=entity_id1,
                    confidence=rel.confidence,
                    source=rel.source
                )
            self.store.add_relation(new_rel)
        
        return entity_id1
    
    def get_entity_profile(self, entity_id: str) -> Dict:
        """Get complete profile of an entity."""
        entity = self.store.get_entity(entity_id)
        
        if not entity:
            return None
        
        # Get relations
        relations = self.store.get_relations(entity_id=entity_id)
        
        # Organize relations
        outgoing = [r for r in relations if r.subject_id == entity_id]
        incoming = [r for r in relations if r.object_id == entity_id]
        
        return {
            'entity': entity.to_dict(),
            'outgoing_relations': [
                {
                    'predicate': r.predicate,
                    'object': self.store.get_entity(r.object_id).name if self.store.get_entity(r.object_id) else r.object_id,
                    'confidence': r.confidence
                }
                for r in outgoing
            ],
            'incoming_relations': [
                {
                    'subject': self.store.get_entity(r.subject_id).name if self.store.get_entity(r.subject_id) else r.subject_id,
                    'predicate': r.predicate,
                    'confidence': r.confidence
                }
                for r in incoming
            ]
        }
    
    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """Generate unique entity ID."""
        safe_name = name.lower().replace(" ", "_")
        return f"{entity_type}_{safe_name}_{str(uuid.uuid4())[:8]}"


# Usage
entity_mgr = EntityManager(semantic_store)

# Create user entity
user_id = entity_mgr.create_or_update_entity(
    name="Bob",
    entity_type="person",
    attributes={
        "email": "bob@example.com",
        "role": "developer"
    }
)

# Update attribute
entity_mgr.update_attribute(user_id, "preferred_language", "JavaScript")

# Get profile
profile = entity_mgr.get_entity_profile(user_id)
print("Entity Profile:")
print(json.dumps(profile, indent=2, default=str))
```

## Relation Modeling

Managing relationships between entities.

### Relation Manager

```python
class RelationManager:
    """Manage relations in semantic memory."""
    
    def __init__(self, store: SemanticMemoryStore):
        self.store = store
        
        # Define relation types and their properties
        self.relation_types = {
            'is_a': {'symmetric': False, 'transitive': True},
            'part_of': {'symmetric': False, 'transitive': True},
            'knows': {'symmetric': True, 'transitive': False},
            'likes': {'symmetric': False, 'transitive': False},
            'created_by': {'symmetric': False, 'transitive': False},
            'related_to': {'symmetric': True, 'transitive': False},
        }
    
    def add_relation(self, subject: str, predicate: str, object: str,
                    confidence: float = 1.0, bidirectional: bool = None):
        """Add relation with automatic symmetry handling."""
        # Add primary relation
        relation = Relation(
            subject_id=subject,
            predicate=predicate,
            object_id=object,
            confidence=confidence
        )
        self.store.add_relation(relation)
        
        # Handle symmetric relations
        if bidirectional or self._is_symmetric(predicate):
            reverse_relation = Relation(
                subject_id=object,
                predicate=predicate,
                object_id=subject,
                confidence=confidence
            )
            self.store.add_relation(reverse_relation)
    
    def get_related_entities(self, entity_id: str, predicate: str = None,
                           direction: str = 'outgoing') -> List[Entity]:
        """Get entities related to this entity."""
        relations = self.store.get_relations(entity_id=entity_id, predicate=predicate)
        
        entity_ids = set()
        
        for rel in relations:
            if direction == 'outgoing' and rel.subject_id == entity_id:
                entity_ids.add(rel.object_id)
            elif direction == 'incoming' and rel.object_id == entity_id:
                entity_ids.add(rel.subject_id)
            elif direction == 'both':
                if rel.subject_id == entity_id:
                    entity_ids.add(rel.object_id)
                else:
                    entity_ids.add(rel.subject_id)
        
        entities = []
        for eid in entity_ids:
            entity = self.store.get_entity(eid)
            if entity:
                entities.append(entity)
        
        return entities
    
    def find_path(self, start_id: str, end_id: str,
                 max_depth: int = 3) -> Optional[List[tuple]]:
        """
        Find path between two entities.
        
        Returns list of (entity_id, predicate) tuples.
        """
        from collections import deque
        
        queue = deque([(start_id, [])])
        visited = {start_id}
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) >= max_depth:
                continue
            
            if current == end_id:
                return path
            
            # Get all relations from current
            relations = self.store.get_relations(entity_id=current)
            
            for rel in relations:
                next_id = None
                if rel.subject_id == current:
                    next_id = rel.object_id
                elif rel.object_id == current:
                    next_id = rel.subject_id
                
                if next_id and next_id not in visited:
                    visited.add(next_id)
                    new_path = path + [(current, rel.predicate, next_id)]
                    queue.append((next_id, new_path))
        
        return None
    
    def infer_transitive_relations(self, predicate: str):
        """
        Infer transitive relations.
        
        If A -> B and B -> C, then A -> C for transitive predicates.
        """
        if not self._is_transitive(predicate):
            return
        
        relations = self.store.get_relations(predicate=predicate)
        
        # Build graph
        graph = {}
        for rel in relations:
            if rel.subject_id not in graph:
                graph[rel.subject_id] = []
            graph[rel.subject_id].append(rel.object_id)
        
        # Find transitive closures
        new_relations = []
        
        for start in graph:
            # BFS to find all reachable nodes
            visited = set()
            queue = [start]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                
                for neighbor in graph.get(current, []):
                    if neighbor not in visited:
                        queue.append(neighbor)
                        
                        # Check if relation already exists
                        if neighbor != start:
                            new_relations.append(Relation(
                                subject_id=start,
                                predicate=predicate,
                                object_id=neighbor,
                                confidence=0.8,  # Lower confidence for inferred
                                source="inferred"
                            ))
        
        # Add inferred relations
        for rel in new_relations:
            self.store.add_relation(rel)
    
    def _is_symmetric(self, predicate: str) -> bool:
        """Check if predicate is symmetric."""
        return self.relation_types.get(predicate, {}).get('symmetric', False)
    
    def _is_transitive(self, predicate: str) -> bool:
        """Check if predicate is transitive."""
        return self.relation_types.get(predicate, {}).get('transitive', False)


# Usage
relation_mgr = RelationManager(semantic_store)

# Add relations
relation_mgr.add_relation("user_alice", "knows", "lang_python")
relation_mgr.add_relation("lang_python", "is_a", "programming_language")
relation_mgr.add_relation("programming_language", "is_a", "formal_language")

# Infer transitive relations
relation_mgr.infer_transitive_relations("is_a")

# Find path
path = relation_mgr.find_path("user_alice", "formal_language")
if path:
    print("Path found:")
    for step in path:
        print(f"  {step[0]} --[{step[1]}]--> {step[2]}")
```

## Fact Storage

Managing standalone facts.

### Fact Manager

```python
class FactManager:
    """Manage facts in semantic memory."""
    
    def __init__(self, store: SemanticMemoryStore):
        self.store = store
    
    def add_fact(self, statement: str, confidence: float = 1.0,
                source: str = "learned", tags: List[str] = None) -> str:
        """Add a new fact."""
        fact_id = str(uuid.uuid4())
        
        fact = Fact(
            id=fact_id,
            statement=statement,
            confidence=confidence,
            source=source,
            tags=tags or []
        )
        
        self.store.add_fact(fact)
        return fact_id
    
    def update_confidence(self, fact_id: str, new_confidence: float):
        """Update confidence of a fact."""
        # Would need to implement fact retrieval by ID in store
        facts = self.store.search_facts(limit=1000)
        for fact in facts:
            if fact.id == fact_id:
                fact.confidence = new_confidence
                self.store.add_fact(fact)
                return
    
    def find_facts_about(self, topic: str, min_confidence: float = 0.5) -> List[Fact]:
        """Find facts about a topic."""
        return self.store.search_facts(
            query=topic,
            min_confidence=min_confidence
        )
    
    def get_facts_by_tags(self, tags: List[str], 
                         min_confidence: float = 0.5) -> List[Fact]:
        """Get facts with specific tags."""
        return self.store.search_facts(
            tags=tags,
            min_confidence=min_confidence
        )
    
    def check_contradiction(self, new_statement: str) -> List[Fact]:
        """Check if new statement contradicts existing facts."""
        # Simple contradiction detection using keywords
        # In practice, would use more sophisticated NLU
        
        # Extract key terms
        key_terms = set(new_statement.lower().split())
        
        # Check for negation
        has_negation = any(
            neg in new_statement.lower()
            for neg in ['not', 'no', "isn't", "doesn't", "never"]
        )
        
        # Find facts about same topic
        all_facts = self.store.search_facts(limit=1000)
        
        potential_contradictions = []
        
        for fact in all_facts:
            fact_terms = set(fact.statement.lower().split())
            
            # Check term overlap
            overlap = key_terms & fact_terms
            
            if len(overlap) > 2:  # Significant overlap
                fact_has_negation = any(
                    neg in fact.statement.lower()
                    for neg in ['not', 'no', "isn't", "doesn't", "never"]
                )
                
                # If one has negation and other doesn't, potential contradiction
                if has_negation != fact_has_negation:
                    potential_contradictions.append(fact)
        
        return potential_contradictions
    
    def merge_similar_facts(self, similarity_threshold: float = 0.8):
        """Merge similar facts."""
        # Would use embeddings for real similarity
        # This is a simplified version
        
        facts = self.store.search_facts(limit=1000)
        
        # Group similar facts
        merged = []
        processed = set()
        
        for i, fact1 in enumerate(facts):
            if fact1.id in processed:
                continue
            
            similar_group = [fact1]
            
            for fact2 in facts[i+1:]:
                if fact2.id in processed:
                    continue
                
                # Simple similarity check
                if self._are_similar(fact1.statement, fact2.statement):
                    similar_group.append(fact2)
                    processed.add(fact2.id)
            
            if len(similar_group) > 1:
                # Merge the group
                merged_fact = self._merge_fact_group(similar_group)
                merged.append(merged_fact)
                processed.add(fact1.id)
        
        return merged
    
    def _are_similar(self, statement1: str, statement2: str) -> bool:
        """Check if two statements are similar."""
        words1 = set(statement1.lower().split())
        words2 = set(statement2.lower().split())
        
        if not words1 or not words2:
            return False
        
        # Jaccard similarity
        intersection = words1 & words2
        union = words1 | words2
        
        similarity = len(intersection) / len(union)
        
        return similarity > 0.7
    
    def _merge_fact_group(self, facts: List[Fact]) -> Fact:
        """Merge multiple similar facts."""
        # Use fact with highest confidence as base
        base_fact = max(facts, key=lambda f: f.confidence)
        
        # Average confidence
        avg_confidence = sum(f.confidence for f in facts) / len(facts)
        
        # Combine tags
        all_tags = []
        for fact in facts:
            all_tags.extend(fact.tags)
        unique_tags = list(set(all_tags))
        
        merged = Fact(
            id=base_fact.id,
            statement=base_fact.statement,
            confidence=avg_confidence,
            source="merged",
            tags=unique_tags
        )
        
        return merged


# Usage
fact_mgr = FactManager(semantic_store)

# Add facts
fact_mgr.add_fact(
    "Python was created by Guido van Rossum",
    confidence=1.0,
    source="wikipedia",
    tags=["python", "history", "creator"]
)

fact_mgr.add_fact(
    "Python emphasizes code readability",
    confidence=0.95,
    source="documentation",
    tags=["python", "philosophy"]
)

# Find facts
python_facts = fact_mgr.find_facts_about("Python")
print(f"Found {len(python_facts)} facts about Python:")
for fact in python_facts:
    print(f"  - {fact.statement} (confidence: {fact.confidence})")

# Check contradiction
new_fact = "Python is not a programming language"
contradictions = fact_mgr.check_contradiction(new_fact)
if contradictions:
    print(f"\nPotential contradictions found:")
    for fact in contradictions:
        print(f"  - {fact.statement}")
```

## Knowledge Graphs

Building and querying knowledge graphs.

### Knowledge Graph Builder

```python
class KnowledgeGraphBuilder:
    """Build and manage knowledge graphs."""
    
    def __init__(self, store: SemanticMemoryStore):
        self.store = store
    
    def build_subgraph(self, entity_id: str, depth: int = 2) -> Dict:
        """Build subgraph around an entity."""
        visited = set()
        graph = {
            'nodes': [],
            'edges': []
        }
        
        self._explore_entity(entity_id, depth, visited, graph)
        
        return graph
    
    def _explore_entity(self, entity_id: str, remaining_depth: int,
                       visited: set, graph: Dict):
        """Recursively explore entity and its relations."""
        if entity_id in visited or remaining_depth < 0:
            return
        
        visited.add(entity_id)
        
        # Add entity node
        entity = self.store.get_entity(entity_id)
        if entity:
            graph['nodes'].append({
                'id': entity.id,
                'name': entity.name,
                'type': entity.entity_type,
                'attributes': entity.attributes
            })
        
        if remaining_depth == 0:
            return
        
        # Get relations
        relations = self.store.get_relations(entity_id=entity_id)
        
        for rel in relations:
            # Add edge
            graph['edges'].append({
                'source': rel.subject_id,
                'target': rel.object_id,
                'relation': rel.predicate,
                'confidence': rel.confidence
            })
            
            # Explore related entity
            related_id = (
                rel.object_id if rel.subject_id == entity_id
                else rel.subject_id
            )
            
            self._explore_entity(related_id, remaining_depth - 1, visited, graph)
    
    def visualize_graph(self, graph: Dict) -> str:
        """Create ASCII visualization of graph."""
        lines = ["Knowledge Graph:", "=" * 60]
        
        # Show nodes
        lines.append("\nNodes:")
        for node in graph['nodes']:
            lines.append(f"  [{node['type']}] {node['name']} (id: {node['id']})")
        
        # Show edges
        lines.append("\nEdges:")
        for edge in graph['edges']:
            source_name = next(
                (n['name'] for n in graph['nodes'] if n['id'] == edge['source']),
                edge['source']
            )
            target_name = next(
                (n['name'] for n in graph['nodes'] if n['id'] == edge['target']),
                edge['target']
            )
            lines.append(
                f"  {source_name} --[{edge['relation']}]--> {target_name}"
            )
        
        return "\n".join(lines)
    
    def find_communities(self, graph: Dict) -> List[List[str]]:
        """Find communities (clusters) in the graph."""
        # Simple community detection using connected components
        
        # Build adjacency list
        adj = {}
        for edge in graph['edges']:
            if edge['source'] not in adj:
                adj[edge['source']] = []
            if edge['target'] not in adj:
                adj[edge['target']] = []
            
            adj[edge['source']].append(edge['target'])
            adj[edge['target']].append(edge['source'])
        
        # Find connected components
        visited = set()
        communities = []
        
        for node in graph['nodes']:
            node_id = node['id']
            if node_id not in visited:
                community = self._dfs(node_id, adj, visited)
                communities.append(community)
        
        return communities
    
    def _dfs(self, node: str, adj: Dict, visited: set) -> List[str]:
        """Depth-first search to find connected component."""
        stack = [node]
        component = []
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            
            visited.add(current)
            component.append(current)
            
            for neighbor in adj.get(current, []):
                if neighbor not in visited:
                    stack.append(neighbor)
        
        return component
    
    def export_to_rdf(self, graph: Dict) -> str:
        """Export graph to RDF Turtle format."""
        lines = ["@prefix : <http://example.org/> .", ""]
        
        # Export entities
        for node in graph['nodes']:
            lines.append(f":{node['id']}")
            lines.append(f"    a :{node['type']} ;")
            lines.append(f"    :name \"{node['name']}\" .")
            lines.append("")
        
        # Export relations
        for edge in graph['edges']:
            lines.append(
                f":{edge['source']} :{edge['relation']} :{edge['target']} ."
            )
        
        return "\n".join(lines)


# Usage
kg_builder = KnowledgeGraphBuilder(semantic_store)

# Build subgraph around entity
graph = kg_builder.build_subgraph("user_alice", depth=2)

print(kg_builder.visualize_graph(graph))

# Find communities
communities = kg_builder.find_communities(graph)
print(f"\nFound {len(communities)} communities")

# Export to RDF
rdf = kg_builder.export_to_rdf(graph)
print("\nRDF Export:")
print(rdf[:300] + "...")
```

## Domain Knowledge

Managing domain-specific knowledge.

### Domain Knowledge Manager

```python
class DomainKnowledgeManager:
    """Manage domain-specific knowledge."""
    
    def __init__(self, store: SemanticMemoryStore):
        self.store = store
        self.domains = {}
    
    def create_domain(self, domain_name: str, description: str):
        """Create a knowledge domain."""
        self.domains[domain_name] = {
            'name': domain_name,
            'description': description,
            'entities': set(),
            'concepts': set()
        }
    
    def add_to_domain(self, domain_name: str, entity_id: str):
        """Add entity to domain."""
        if domain_name not in self.domains:
            raise ValueError(f"Domain not found: {domain_name}")
        
        self.domains[domain_name]['entities'].add(entity_id)
    
    def get_domain_knowledge(self, domain_name: str) -> Dict:
        """Get all knowledge in a domain."""
        if domain_name not in self.domains:
            return None
        
        domain = self.domains[domain_name]
        
        # Get all entities in domain
        entities = []
        for entity_id in domain['entities']:
            entity = self.store.get_entity(entity_id)
            if entity:
                entities.append(entity)
        
        # Get relations between domain entities
        relations = []
        for entity_id in domain['entities']:
            entity_relations = self.store.get_relations(entity_id=entity_id)
            
            # Filter for relations within domain
            for rel in entity_relations:
                if (rel.subject_id in domain['entities'] and
                    rel.object_id in domain['entities']):
                    relations.append(rel)
        
        return {
            'domain': domain_name,
            'description': domain['description'],
            'entities': [e.to_dict() for e in entities],
            'relations': [r.to_triple() for r in relations],
            'entity_count': len(entities),
            'relation_count': len(relations)
        }
    
    def find_domain_experts(self, domain_name: str) -> List[Entity]:
        """Find entities that are experts in a domain."""
        domain_knowledge = self.get_domain_knowledge(domain_name)
        
        if not domain_knowledge:
            return []
        
        # Find entities with most connections in domain
        entity_connections = {}
        
        for entity_id in self.domains[domain_name]['entities']:
            relations = self.store.get_relations(entity_id=entity_id)
            entity_connections[entity_id] = len(relations)
        
        # Sort by connection count
        sorted_entities = sorted(
            entity_connections.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top entities
        experts = []
        for entity_id, _ in sorted_entities[:5]:
            entity = self.store.get_entity(entity_id)
            if entity and entity.entity_type == 'person':
                experts.append(entity)
        
        return experts


# Usage
domain_mgr = DomainKnowledgeManager(semantic_store)

# Create programming domain
domain_mgr.create_domain(
    "programming",
    "Knowledge about programming languages and concepts"
)

domain_mgr.add_to_domain("programming", "lang_python")
domain_mgr.add_to_domain("programming", "user_alice")

# Get domain knowledge
prog_knowledge = domain_mgr.get_domain_knowledge("programming")
print(f"Programming domain has {prog_knowledge['entity_count']} entities")
```

## Concept Learning

Learning abstract concepts from examples.

### Concept Learner

```python
from collections import Counter

class ConceptLearner:
    """Learn concepts from examples."""
    
    def __init__(self, store: SemanticMemoryStore):
        self.store = store
    
    def learn_concept_from_examples(self, concept_name: str,
                                   positive_examples: List[Entity],
                                   negative_examples: List[Entity] = None) -> Dict:
        """
        Learn concept from positive (and optionally negative) examples.
        
        Returns learned concept characteristics.
        """
        if not positive_examples:
            return None
        
        # Extract common attributes
        common_attributes = self._find_common_attributes(positive_examples)
        
        # Extract distinguishing features (if negative examples provided)
        distinguishing_features = {}
        if negative_examples:
            distinguishing_features = self._find_distinguishing_features(
                positive_examples,
                negative_examples
            )
        
        # Identify typical values
        typical_values = self._find_typical_values(positive_examples)
        
        return {
            'concept_name': concept_name,
            'common_attributes': common_attributes,
            'distinguishing_features': distinguishing_features,
            'typical_values': typical_values,
            'example_count': len(positive_examples)
        }
    
    def _find_common_attributes(self, examples: List[Entity]) -> List[str]:
        """Find attributes present in all examples."""
        if not examples:
            return []
        
        # Get attributes from first example
        common = set(examples[0].attributes.keys())
        
        # Intersect with all other examples
        for example in examples[1:]:
            common &= set(example.attributes.keys())
        
        return list(common)
    
    def _find_distinguishing_features(self, positive: List[Entity],
                                     negative: List[Entity]) -> Dict:
        """Find features that distinguish positive from negative examples."""
        distinguishing = {}
        
        # Get all attributes
        all_positive_attrs = {}
        for example in positive:
            for key, value in example.attributes.items():
                if key not in all_positive_attrs:
                    all_positive_attrs[key] = []
                all_positive_attrs[key].append(value)
        
        all_negative_attrs = {}
        for example in negative:
            for key, value in example.attributes.items():
                if key not in all_negative_attrs:
                    all_negative_attrs[key] = []
                all_negative_attrs[key].append(value)
        
        # Find attributes that differ
        for key in all_positive_attrs:
            positive_values = set(str(v) for v in all_positive_attrs[key])
            negative_values = set(str(v) for v in all_negative_attrs.get(key, []))
            
            if positive_values != negative_values:
                distinguishing[key] = {
                    'positive_values': list(positive_values),
                    'negative_values': list(negative_values)
                }
        
        return distinguishing
    
    def _find_typical_values(self, examples: List[Entity]) -> Dict:
        """Find typical values for each attribute."""
        typical = {}
        
        # Collect all values for each attribute
        attribute_values = {}
        for example in examples:
            for key, value in example.attributes.items():
                if key not in attribute_values:
                    attribute_values[key] = []
                attribute_values[key].append(str(value))
        
        # Find most common value for each attribute
        for key, values in attribute_values.items():
            counter = Counter(values)
            most_common = counter.most_common(1)[0]
            typical[key] = {
                'value': most_common[0],
                'frequency': most_common[1] / len(values)
            }
        
        return typical
    
    def classify_by_concept(self, entity: Entity,
                           concept: Dict) -> float:
        """
        Classify entity against learned concept.
        
        Returns confidence score (0-1).
        """
        if not concept:
            return 0.0
        
        score = 0.0
        checks = 0
        
        # Check common attributes
        common_attrs = concept.get('common_attributes', [])
        if common_attrs:
            matching = sum(
                1 for attr in common_attrs
                if attr in entity.attributes
            )
            score += matching / len(common_attrs)
            checks += 1
        
        # Check typical values
        typical_values = concept.get('typical_values', {})
        if typical_values:
            matching = 0
            for key, typical in typical_values.items():
                if key in entity.attributes:
                    if str(entity.attributes[key]) == typical['value']:
                        matching += typical['frequency']
            
            if typical_values:
                score += matching / len(typical_values)
                checks += 1
        
        return score / checks if checks > 0 else 0.0


# Usage
concept_learner = ConceptLearner(semantic_store)

# Learn "programming_language" concept
python = semantic_store.get_entity("lang_python")
# Would add more examples...

concept = concept_learner.learn_concept_from_examples(
    "programming_language",
    positive_examples=[python] if python else []
)

if concept:
    print(f"Learned concept: {concept['concept_name']}")
    print(f"Common attributes: {concept['common_attributes']}")
```

## Knowledge Updates

Managing knowledge updates and conflicts.

### Knowledge Updater

```python
class KnowledgeUpdater:
    """Handle knowledge updates and conflicts."""
    
    def __init__(self, store: SemanticMemoryStore):
        self.store = store
        self.update_log = []
    
    def update_fact(self, fact_id: str, new_statement: str = None,
                   new_confidence: float = None):
        """Update an existing fact."""
        # Find fact
        facts = self.store.search_facts(limit=1000)
        target_fact = None
        
        for fact in facts:
            if fact.id == fact_id:
                target_fact = fact
                break
        
        if not target_fact:
            raise ValueError(f"Fact not found: {fact_id}")
        
        # Log update
        self.update_log.append({
            'type': 'fact_update',
            'fact_id': fact_id,
            'old_statement': target_fact.statement,
            'new_statement': new_statement,
            'old_confidence': target_fact.confidence,
            'new_confidence': new_confidence,
            'timestamp': datetime.now()
        })
        
        # Apply updates
        if new_statement:
            target_fact.statement = new_statement
        if new_confidence is not None:
            target_fact.confidence = new_confidence
        
        self.store.add_fact(target_fact)
    
    def resolve_conflict(self, fact1_id: str, fact2_id: str,
                        resolution: str = 'highest_confidence') -> str:
        """
        Resolve conflict between two facts.
        
        resolution options:
        - 'highest_confidence': Keep fact with higher confidence
        - 'newest': Keep most recently created
        - 'merge': Create merged fact
        """
        facts = self.store.search_facts(limit=1000)
        
        fact1 = next((f for f in facts if f.id == fact1_id), None)
        fact2 = next((f for f in facts if f.id == fact2_id), None)
        
        if not fact1 or not fact2:
            raise ValueError("Both facts must exist")
        
        if resolution == 'highest_confidence':
            winner = fact1 if fact1.confidence >= fact2.confidence else fact2
            return winner.id
        
        elif resolution == 'newest':
            winner = fact1 if fact1.created_at >= fact2.created_at else fact2
            return winner.id
        
        elif resolution == 'merge':
            # Create merged fact
            merged_statement = f"{fact1.statement}; {fact2.statement}"
            avg_confidence = (fact1.confidence + fact2.confidence) / 2
            
            merged = Fact(
                id=str(uuid.uuid4()),
                statement=merged_statement,
                confidence=avg_confidence,
                source="merged",
                tags=list(set(fact1.tags + fact2.tags))
            )
            
            self.store.add_fact(merged)
            return merged.id
        
        else:
            raise ValueError(f"Unknown resolution: {resolution}")
    
    def deprecate_knowledge(self, entity_id: str = None, fact_id: str = None):
        """Mark knowledge as deprecated."""
        if entity_id:
            entity = self.store.get_entity(entity_id)
            if entity:
                entity.confidence = 0.1  # Low confidence = deprecated
                entity.attributes['deprecated'] = True
                self.store.add_entity(entity)
        
        elif fact_id:
            facts = self.store.search_facts(limit=1000)
            for fact in facts:
                if fact.id == fact_id:
                    fact.confidence = 0.1
                    fact.tags.append('deprecated')
                    self.store.add_fact(fact)
                    break


# Usage
updater = KnowledgeUpdater(semantic_store)

# Update fact confidence
# updater.update_fact(fact_id, new_confidence=0.95)

# Resolve conflict
# winner_id = updater.resolve_conflict(fact1_id, fact2_id, resolution='highest_confidence')
```

## Querying Semantic Memory

Advanced querying capabilities.

### Semantic Query Engine

```python
class SemanticQueryEngine:
    """Query semantic memory."""
    
    def __init__(self, store: SemanticMemoryStore):
        self.store = store
    
    def query(self, query_str: str) -> List[Dict]:
        """
        Natural language query.
        
        Parses simple queries like:
        - "What does Alice know?"
        - "Who created Python?"
        - "What is Python?"
        """
        # Simple pattern matching (would use NLU in practice)
        query_lower = query_str.lower()
        
        # Pattern: "what does X know/like/...?"
        if 'what does' in query_lower and 'know' in query_lower:
            entity_name = self._extract_entity_name(query_str, 'what does', 'know')
            return self._query_entity_knowledge(entity_name, 'knows')
        
        # Pattern: "who created X?"
        elif 'who created' in query_lower:
            entity_name = self._extract_entity_name(query_str, 'who created', '?')
            return self._query_entity_creator(entity_name)
        
        # Pattern: "what is X?"
        elif 'what is' in query_lower:
            entity_name = self._extract_entity_name(query_str, 'what is', '?')
            return self._query_entity_definition(entity_name)
        
        # Fallback: keyword search
        return self._keyword_search(query_str)
    
    def _extract_entity_name(self, query: str, start_marker: str,
                            end_marker: str) -> str:
        """Extract entity name from query."""
        start_idx = query.lower().find(start_marker) + len(start_marker)
        end_idx = query.find(end_marker, start_idx)
        
        if end_idx == -1:
            end_idx = len(query)
        
        entity_name = query[start_idx:end_idx].strip()
        return entity_name
    
    def _query_entity_knowledge(self, entity_name: str,
                               predicate: str) -> List[Dict]:
        """Query what an entity knows."""
        # Find entity
        entities = self.store.search_entities(query=entity_name, limit=1)
        
        if not entities:
            return [{'error': f'Entity not found: {entity_name}'}]
        
        entity = entities[0]
        
        # Get relations
        relations = self.store.get_relations(
            entity_id=entity.id,
            predicate=predicate
        )
        
        results = []
        for rel in relations:
            if rel.subject_id == entity.id:
                target = self.store.get_entity(rel.object_id)
                if target:
                    results.append({
                        'entity': entity.name,
                        'relation': rel.predicate,
                        'target': target.name,
                        'confidence': rel.confidence
                    })
        
        return results
    
    def _query_entity_creator(self, entity_name: str) -> List[Dict]:
        """Query who created an entity."""
        entities = self.store.search_entities(query=entity_name, limit=1)
        
        if not entities:
            return [{'error': f'Entity not found: {entity_name}'}]
        
        entity = entities[0]
        
        # Look for 'created_by' relation
        relations = self.store.get_relations(
            entity_id=entity.id,
            predicate='created_by'
        )
        
        results = []
        for rel in relations:
            if rel.object_id == entity.id:
                creator = self.store.get_entity(rel.subject_id)
            else:
                creator = self.store.get_entity(rel.object_id)
            
            if creator:
                results.append({
                    'entity': entity.name,
                    'created_by': creator.name,
                    'confidence': rel.confidence
                })
        
        return results
    
    def _query_entity_definition(self, entity_name: str) -> List[Dict]:
        """Query definition of an entity."""
        entities = self.store.search_entities(query=entity_name, limit=1)
        
        if not entities:
            # Try facts
            facts = self.store.search_facts(query=entity_name, limit=5)
            return [{'fact': f.statement, 'confidence': f.confidence} for f in facts]
        
        entity = entities[0]
        
        return [{
            'entity': entity.name,
            'type': entity.entity_type,
            'attributes': entity.attributes,
            'confidence': entity.confidence
        }]
    
    def _keyword_search(self, query: str) -> List[Dict]:
        """Fallback keyword search."""
        # Search entities
        entities = self.store.search_entities(query=query, limit=3)
        
        # Search facts
        facts = self.store.search_facts(query=query, limit=3)
        
        results = []
        
        for entity in entities:
            results.append({
                'type': 'entity',
                'name': entity.name,
                'entity_type': entity.entity_type,
                'confidence': entity.confidence
            })
        
        for fact in facts:
            results.append({
                'type': 'fact',
                'statement': fact.statement,
                'confidence': fact.confidence
            })
        
        return results


# Usage
query_engine = SemanticQueryEngine(semantic_store)

# Query examples
results = query_engine.query("What does Alice know?")
print("Query: What does Alice know?")
for result in results:
    print(f"  {result}")

results = query_engine.query("What is Python?")
print("\nQuery: What is Python?")
for result in results:
    print(f"  {result}")
```

## Reasoning with Knowledge

Using semantic memory for reasoning.

### Knowledge Reasoner

```python
class KnowledgeReasoner:
    """Reason over semantic knowledge."""
    
    def __init__(self, store: SemanticMemoryStore,
                 relation_mgr: RelationManager):
        self.store = store
        self.relation_mgr = relation_mgr
    
    def can_infer(self, subject: str, predicate: str, object: str) -> tuple:
        """
        Check if relation can be inferred.
        
        Returns (can_infer: bool, confidence: float, path: List)
        """
        # Check if direct relation exists
        relations = self.store.get_relations(entity_id=subject, predicate=predicate)
        
        for rel in relations:
            if rel.object_id == object and rel.subject_id == subject:
                return (True, rel.confidence, [(subject, predicate, object)])
        
        # Try to find path
        path = self.relation_mgr.find_path(subject, object, max_depth=3)
        
        if path:
            # Calculate confidence based on path
            confidence = 1.0
            for step in path:
                # Would look up actual relation confidence
                confidence *= 0.9  # Decay for each hop
            
            return (True, confidence, path)
        
        return (False, 0.0, [])
    
    def answer_question(self, question: str) -> str:
        """Answer question using knowledge reasoning."""
        # Simple question answering
        question_lower = question.lower()
        
        # "Is X a Y?"
        if question_lower.startswith('is '):
            parts = question[3:].replace('?', '').split(' a ')
            if len(parts) == 2:
                subject, obj = parts[0].strip(), parts[1].strip()
                
                can_infer, confidence, path = self.can_infer(subject, 'is_a', obj)
                
                if can_infer:
                    return f"Yes, {subject} is a {obj} (confidence: {confidence:.2f})"
                else:
                    return f"I don't know if {subject} is a {obj}"
        
        # Default
        return "I don't understand that question"


# Usage
reasoner = KnowledgeReasoner(semantic_store, relation_mgr)

# Check inference
can_infer, confidence, path = reasoner.can_infer(
    "user_alice",
    "knows",
    "formal_language"
)

print(f"Can infer: {can_infer}")
print(f"Confidence: {confidence:.2f}")
if path:
    print(f"Path: {path}")
```

## Summary

Semantic memory enables agents to store and reason over factual knowledge:

**Key Concepts**:

- **Entities**: People, places, things, concepts
- **Relations**: Connections between entities
- **Facts**: Standalone statements of truth
- **Knowledge graphs**: Networks of interconnected knowledge
- **Concepts**: Abstract ideas learned from examples

**Implementation Strategies**:

- **Structured storage**: SQL or graph databases
- **Entity management**: Create, update, merge entities
- **Relation modeling**: Symmetric, transitive, typed relations
- **Fact storage**: Confidence-scored statements
- **Knowledge graphs**: Visualizable network structures

**Use Cases**:

- Answering factual questions
- Understanding relationships
- Building domain expertise
- Learning concepts from examples
- Making inferences from known facts

## Next Steps

- **[Memory Retrieval](memory-retrieval.md)**: Advanced retrieval strategies
- **[Memory Consolidation](memory-consolidation.md)**: Extracting semantic knowledge from episodes
- **[Episodic Memory](episodic-memory.md)**: Experiences that feed semantic learning
- **[Memory Architecture](memory-architecture.md)**: Overall system design
