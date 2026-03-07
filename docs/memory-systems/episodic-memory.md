# Episodic Memory

## Table of Contents

- [Introduction](#introduction)
- [What is Episodic Memory](#what-is-episodic-memory)
- [Storing Episodes](#storing-episodes)
- [Episode Structure](#episode-structure)
- [Temporal Organization](#temporal-organization)
- [Retrieving Past Experiences](#retrieving-past-experiences)
- [Learning from Episodes](#learning-from-episodes)
- [Episode Similarity](#episode-similarity)
- [Task Episodes](#task-episodes)
- [Interaction Episodes](#interaction-episodes)
- [Success and Failure Analysis](#success-and-failure-analysis)
- [Episode Summarization](#episode-summarization)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Episodic memory stores experiences and events - the "what happened when" of an agent's life. While semantic memory stores facts ("Paris is the capital of France"), episodic memory stores experiences ("I helped a user plan a trip to Paris on Tuesday").

> "Episodic memory is the autobiographical record of an agent's experiences."

Episodic memory enables agents to:

- **Remember what happened** in past interactions
- **Learn from experience** by analyzing successes and failures
- **Maintain temporal awareness** of when events occurred
- **Recognize patterns** across similar situations
- **Avoid repeating mistakes** by recalling what didn't work

### Key Characteristics

**Temporal**: Episodes are tied to specific points in time

**Contextual**: Include the situation and circumstances

**Sequential**: Maintain order of events within episodes

**Experiential**: Capture the "lived experience" of the agent

## What is Episodic Memory

Episodic memory differs from other memory types in its focus on specific experiences.

### Memory Type Comparison

```
┌───────────────────────────────────────────────────────────┐
│ Memory Type  │ Content              │ Example             │
├──────────────┼──────────────────────┼─────────────────────┤
│ Semantic     │ Facts, knowledge     │ "Python is a        │
│              │                      │  programming lang"  │
├──────────────┼──────────────────────┼─────────────────────┤
│ Episodic     │ Experiences, events  │ "User asked about   │
│              │                      │  Python on Monday"  │
├──────────────┼──────────────────────┼─────────────────────┤
│ Procedural   │ How to do things     │ "Steps to debug     │
│              │                      │  Python code"       │
└───────────────────────────────────────────────────────────┘
```

### Episode Components

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

class EpisodeType(Enum):
    """Types of episodes."""
    INTERACTION = "interaction"
    TASK = "task"
    TOOL_USE = "tool_use"
    ERROR = "error"
    LEARNING = "learning"
    DECISION = "decision"

class Outcome(Enum):
    """Episode outcomes."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"

@dataclass
class Episode:
    """A single episode in agent memory."""
    
    # Identity
    id: str
    episode_type: EpisodeType
    
    # Temporal
    timestamp: datetime
    duration_seconds: Optional[float] = None
    
    # Content
    description: str
    context: Dict[str, Any] = None
    actions: List[Dict[str, Any]] = None
    
    # Outcome
    outcome: Outcome = Outcome.UNKNOWN
    result: Optional[Any] = None
    
    # Metadata
    importance: float = 0.5
    tags: List[str] = None
    related_episodes: List[str] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.actions is None:
            self.actions = []
        if self.tags is None:
            self.tags = []
        if self.related_episodes is None:
            self.related_episodes = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'type': self.episode_type.value,
            'timestamp': self.timestamp.isoformat(),
            'duration_seconds': self.duration_seconds,
            'description': self.description,
            'context': self.context,
            'actions': self.actions,
            'outcome': self.outcome.value,
            'result': self.result,
            'importance': self.importance,
            'tags': self.tags,
            'related_episodes': self.related_episodes
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Episode':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            episode_type=EpisodeType(data['type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            duration_seconds=data.get('duration_seconds'),
            description=data['description'],
            context=data.get('context', {}),
            actions=data.get('actions', []),
            outcome=Outcome(data.get('outcome', 'unknown')),
            result=data.get('result'),
            importance=data.get('importance', 0.5),
            tags=data.get('tags', []),
            related_episodes=data.get('related_episodes', [])
        )
```

## Storing Episodes

Efficient storage of episodic memories.

### Episode Store Implementation

```python
import sqlite3
import json
from typing import List, Optional

class EpisodicMemoryStore:
    """Storage for episodic memories."""
    
    def __init__(self, db_path: str = "episodes.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()
    
    def _create_schema(self):
        """Create database schema for episodes."""
        
        # Episodes table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                episode_type TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                duration_seconds REAL,
                description TEXT NOT NULL,
                context TEXT,
                actions TEXT,
                outcome TEXT NOT NULL,
                result TEXT,
                importance REAL DEFAULT 0.5,
                tags TEXT,
                related_episodes TEXT,
                session_id TEXT
            )
        """)
        
        # Episode relationships table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS episode_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT NOT NULL,
                related_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                FOREIGN KEY (episode_id) REFERENCES episodes(id),
                FOREIGN KEY (related_id) REFERENCES episodes(id)
            )
        """)
        
        # Indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_ep_timestamp ON episodes(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_ep_type ON episodes(episode_type)",
            "CREATE INDEX IF NOT EXISTS idx_ep_outcome ON episodes(outcome)",
            "CREATE INDEX IF NOT EXISTS idx_ep_importance ON episodes(importance DESC)",
            "CREATE INDEX IF NOT EXISTS idx_ep_session ON episodes(session_id)",
        ]
        
        for index in indexes:
            self.conn.execute(index)
        
        self.conn.commit()
    
    def store_episode(self, episode: Episode):
        """Store an episode."""
        self.conn.execute("""
            INSERT INTO episodes (
                id, episode_type, timestamp, duration_seconds,
                description, context, actions, outcome, result,
                importance, tags, related_episodes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            episode.id,
            episode.episode_type.value,
            episode.timestamp.isoformat(),
            episode.duration_seconds,
            episode.description,
            json.dumps(episode.context),
            json.dumps(episode.actions),
            episode.outcome.value,
            json.dumps(episode.result),
            episode.importance,
            json.dumps(episode.tags),
            json.dumps(episode.related_episodes)
        ))
        
        self.conn.commit()
    
    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Retrieve an episode by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM episodes WHERE id = ?",
            (episode_id,)
        )
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return self._row_to_episode(row)
    
    def get_recent_episodes(self, limit: int = 10,
                           episode_type: Optional[EpisodeType] = None) -> List[Episode]:
        """Get recent episodes."""
        if episode_type:
            cursor = self.conn.execute("""
                SELECT * FROM episodes
                WHERE episode_type = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (episode_type.value, limit))
        else:
            cursor = self.conn.execute("""
                SELECT * FROM episodes
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
        
        return [self._row_to_episode(row) for row in cursor.fetchall()]
    
    def search_episodes(self, query: str, limit: int = 10) -> List[Episode]:
        """Search episodes by description."""
        cursor = self.conn.execute("""
            SELECT * FROM episodes
            WHERE description LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (f"%{query}%", limit))
        
        return [self._row_to_episode(row) for row in cursor.fetchall()]
    
    def get_episodes_by_outcome(self, outcome: Outcome,
                                limit: int = 10) -> List[Episode]:
        """Get episodes with specific outcome."""
        cursor = self.conn.execute("""
            SELECT * FROM episodes
            WHERE outcome = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (outcome.value, limit))
        
        return [self._row_to_episode(row) for row in cursor.fetchall()]
    
    def get_episodes_in_range(self, start: datetime, end: datetime) -> List[Episode]:
        """Get episodes in time range."""
        cursor = self.conn.execute("""
            SELECT * FROM episodes
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
        """, (start.isoformat(), end.isoformat()))
        
        return [self._row_to_episode(row) for row in cursor.fetchall()]
    
    def _row_to_episode(self, row) -> Episode:
        """Convert database row to Episode."""
        return Episode(
            id=row['id'],
            episode_type=EpisodeType(row['episode_type']),
            timestamp=datetime.fromisoformat(row['timestamp']),
            duration_seconds=row['duration_seconds'],
            description=row['description'],
            context=json.loads(row['context']) if row['context'] else {},
            actions=json.loads(row['actions']) if row['actions'] else [],
            outcome=Outcome(row['outcome']),
            result=json.loads(row['result']) if row['result'] else None,
            importance=row['importance'],
            tags=json.loads(row['tags']) if row['tags'] else [],
            related_episodes=json.loads(row['related_episodes']) if row['related_episodes'] else []
        )


# Usage
store = EpisodicMemoryStore()

# Create and store an episode
import uuid

episode = Episode(
    id=str(uuid.uuid4()),
    episode_type=EpisodeType.TASK,
    timestamp=datetime.now(),
    duration_seconds=45.3,
    description="User asked to summarize research papers on RAG",
    context={
        'user_id': 'alice',
        'session_id': 'session_123',
        'query': 'summarize RAG papers'
    },
    actions=[
        {'action': 'search_papers', 'query': 'RAG', 'results': 15},
        {'action': 'read_papers', 'count': 5},
        {'action': 'generate_summary', 'length': 500}
    ],
    outcome=Outcome.SUCCESS,
    result={'summary': 'RAG combines retrieval with generation...'},
    importance=0.8,
    tags=['research', 'RAG', 'summarization']
)

store.store_episode(episode)

# Retrieve recent episodes
recent = store.get_recent_episodes(limit=5)
for ep in recent:
    print(f"{ep.timestamp}: {ep.description}")
```

## Episode Structure

Organizing information within episodes.

### Structured Episode Recording

```python
class EpisodeRecorder:
    """Record structured episodes as they happen."""
    
    def __init__(self, store: EpisodicMemoryStore):
        self.store = store
        self.current_episode: Optional[Episode] = None
        self.episode_stack: List[Episode] = []  # For nested episodes
    
    def start_episode(self, episode_type: EpisodeType,
                     description: str, context: Dict = None) -> str:
        """Start recording a new episode."""
        episode_id = str(uuid.uuid4())
        
        episode = Episode(
            id=episode_id,
            episode_type=episode_type,
            timestamp=datetime.now(),
            description=description,
            context=context or {}
        )
        
        # Handle nested episodes
        if self.current_episode:
            self.episode_stack.append(self.current_episode)
        
        self.current_episode = episode
        return episode_id
    
    def record_action(self, action_type: str, **kwargs):
        """Record an action within the current episode."""
        if not self.current_episode:
            raise ValueError("No active episode")
        
        action = {
            'type': action_type,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        
        self.current_episode.actions.append(action)
    
    def update_context(self, **kwargs):
        """Update episode context."""
        if not self.current_episode:
            raise ValueError("No active episode")
        
        self.current_episode.context.update(kwargs)
    
    def end_episode(self, outcome: Outcome, result: Any = None,
                   importance: float = None):
        """End the current episode."""
        if not self.current_episode:
            raise ValueError("No active episode")
        
        # Calculate duration
        duration = (datetime.now() - self.current_episode.timestamp).total_seconds()
        self.current_episode.duration_seconds = duration
        
        # Set outcome
        self.current_episode.outcome = outcome
        self.current_episode.result = result
        
        # Set importance if provided
        if importance is not None:
            self.current_episode.importance = importance
        else:
            # Auto-calculate importance
            self.current_episode.importance = self._calculate_importance(
                self.current_episode
            )
        
        # Store episode
        self.store.store_episode(self.current_episode)
        
        episode_id = self.current_episode.id
        
        # Restore parent episode if nested
        if self.episode_stack:
            parent = self.episode_stack.pop()
            # Link to parent
            parent.related_episodes.append(episode_id)
            self.current_episode = parent
        else:
            self.current_episode = None
        
        return episode_id
    
    def _calculate_importance(self, episode: Episode) -> float:
        """Auto-calculate episode importance."""
        importance = 0.5
        
        # Failed episodes are important (learn from mistakes)
        if episode.outcome == Outcome.FAILURE:
            importance += 0.2
        
        # Long episodes are typically important
        if episode.duration_seconds and episode.duration_seconds > 60:
            importance += 0.1
        
        # Many actions suggest complexity/importance
        if len(episode.actions) > 5:
            importance += 0.1
        
        # Certain types are more important
        if episode.episode_type in [EpisodeType.ERROR, EpisodeType.LEARNING]:
            importance += 0.1
        
        return min(1.0, importance)


# Usage
recorder = EpisodeRecorder(store)

# Record a task episode
episode_id = recorder.start_episode(
    EpisodeType.TASK,
    "Debug Python script",
    context={'user': 'alice', 'file': 'script.py'}
)

recorder.record_action('read_file', file='script.py', lines=150)
recorder.record_action('analyze_errors', errors_found=3)
recorder.record_action('suggest_fixes', fixes=3)

recorder.update_context(errors_fixed=2)

recorder.end_episode(
    Outcome.PARTIAL,
    result={'fixed': 2, 'remaining': 1},
    importance=0.7
)
```

## Temporal Organization

Organizing episodes by time.

### Timeline Manager

```python
from collections import defaultdict
from datetime import timedelta

class EpisodeTimeline:
    """Organize episodes temporally."""
    
    def __init__(self, store: EpisodicMemoryStore):
        self.store = store
    
    def get_timeline(self, start: datetime, end: datetime,
                    episode_type: Optional[EpisodeType] = None) -> List[Episode]:
        """Get episodes in time range."""
        episodes = self.store.get_episodes_in_range(start, end)
        
        if episode_type:
            episodes = [ep for ep in episodes if ep.episode_type == episode_type]
        
        return episodes
    
    def get_episodes_by_day(self, date: datetime.date) -> Dict[str, List[Episode]]:
        """Get episodes grouped by hour for a specific day."""
        start = datetime.combine(date, datetime.min.time())
        end = start + timedelta(days=1)
        
        episodes = self.store.get_episodes_in_range(start, end)
        
        # Group by hour
        by_hour = defaultdict(list)
        for episode in episodes:
            hour = episode.timestamp.hour
            by_hour[f"{hour:02d}:00"].append(episode)
        
        return dict(by_hour)
    
    def get_session_episodes(self, session_id: str) -> List[Episode]:
        """Get all episodes from a session."""
        # Would query by session_id in practice
        all_episodes = self.store.get_recent_episodes(limit=1000)
        return [
            ep for ep in all_episodes
            if ep.context.get('session_id') == session_id
        ]
    
    def get_episode_sequence(self, episode_id: str) -> List[Episode]:
        """Get episode and all related episodes in order."""
        main_episode = self.store.get_episode(episode_id)
        if not main_episode:
            return []
        
        # Get related episodes
        sequence = [main_episode]
        
        for related_id in main_episode.related_episodes:
            related = self.store.get_episode(related_id)
            if related:
                sequence.append(related)
        
        # Sort by timestamp
        sequence.sort(key=lambda ep: ep.timestamp)
        
        return sequence
    
    def visualize_timeline(self, episodes: List[Episode]) -> str:
        """Create ASCII timeline visualization."""
        if not episodes:
            return "No episodes to visualize"
        
        lines = ["Timeline:", "=" * 60]
        
        for episode in episodes:
            time_str = episode.timestamp.strftime("%H:%M:%S")
            outcome_symbol = {
                Outcome.SUCCESS: "✓",
                Outcome.FAILURE: "✗",
                Outcome.PARTIAL: "◐",
                Outcome.UNKNOWN: "?"
            }.get(episode.outcome, "?")
            
            duration_str = (
                f"({episode.duration_seconds:.1f}s)"
                if episode.duration_seconds else ""
            )
            
            lines.append(
                f"{time_str} {outcome_symbol} [{episode.episode_type.value}] "
                f"{episode.description[:40]} {duration_str}"
            )
        
        return "\n".join(lines)


# Usage
timeline = EpisodeTimeline(store)

# Get today's episodes
import datetime
today = datetime.date.today()
today_episodes = timeline.get_episodes_by_day(today)

print("Today's episodes by hour:")
for hour, episodes in sorted(today_episodes.items()):
    print(f"\n{hour}:")
    for ep in episodes:
        print(f"  - {ep.description}")

# Visualize timeline
all_episodes = timeline.get_timeline(
    datetime.datetime.combine(today, datetime.time.min),
    datetime.datetime.now()
)
print("\n" + timeline.visualize_timeline(all_episodes))
```

## Retrieving Past Experiences

Finding relevant episodes from the past.

### Episode Retrieval

```python
class EpisodeRetriever:
    """Retrieve relevant past episodes."""
    
    def __init__(self, store: EpisodicMemoryStore, embedder=None):
        self.store = store
        self.embedder = embedder
    
    def find_similar_episodes(self, current_context: Dict,
                             episode_type: Optional[EpisodeType] = None,
                             limit: int = 5) -> List[Episode]:
        """Find episodes with similar context."""
        # Get candidates
        if episode_type:
            candidates = self.store.get_recent_episodes(
                limit=100,
                episode_type=episode_type
            )
        else:
            candidates = self.store.get_recent_episodes(limit=100)
        
        # Score by context similarity
        scored = []
        for episode in candidates:
            similarity = self._context_similarity(
                current_context,
                episode.context
            )
            scored.append((similarity, episode))
        
        # Sort by similarity
        scored.sort(reverse=True)
        
        return [ep for _, ep in scored[:limit]]
    
    def find_by_outcome(self, outcome: Outcome,
                       episode_type: Optional[EpisodeType] = None,
                       limit: int = 10) -> List[Episode]:
        """Find episodes with specific outcome."""
        episodes = self.store.get_episodes_by_outcome(outcome, limit=100)
        
        if episode_type:
            episodes = [ep for ep in episodes if ep.episode_type == episode_type]
        
        return episodes[:limit]
    
    def find_successful_patterns(self, episode_type: EpisodeType,
                                 min_occurrences: int = 3) -> List[Dict]:
        """Find patterns in successful episodes."""
        successful = self.find_by_outcome(
            Outcome.SUCCESS,
            episode_type=episode_type,
            limit=100
        )
        
        if len(successful) < min_occurrences:
            return []
        
        # Analyze patterns in actions
        action_sequences = defaultdict(int)
        
        for episode in successful:
            # Extract action sequence
            sequence = tuple(
                action.get('type') for action in episode.actions
            )
            action_sequences[sequence] += 1
        
        # Find common patterns
        patterns = [
            {
                'sequence': list(seq),
                'count': count,
                'frequency': count / len(successful)
            }
            for seq, count in action_sequences.items()
            if count >= min_occurrences
        ]
        
        patterns.sort(key=lambda p: p['count'], reverse=True)
        
        return patterns
    
    def get_learning_episodes(self, topic: str, limit: int = 5) -> List[Episode]:
        """Get episodes where agent learned about a topic."""
        episodes = self.store.search_episodes(topic, limit=50)
        
        # Filter for learning episodes or episodes with topic in tags
        relevant = [
            ep for ep in episodes
            if (ep.episode_type == EpisodeType.LEARNING or
                topic.lower() in [tag.lower() for tag in ep.tags])
        ]
        
        return relevant[:limit]
    
    def _context_similarity(self, context1: Dict, context2: Dict) -> float:
        """Calculate similarity between contexts."""
        if not context1 or not context2:
            return 0.0
        
        # Simple Jaccard similarity on keys
        keys1 = set(context1.keys())
        keys2 = set(context2.keys())
        
        if not keys1 or not keys2:
            return 0.0
        
        intersection = keys1 & keys2
        union = keys1 | keys2
        
        key_similarity = len(intersection) / len(union)
        
        # Value similarity for shared keys
        value_similarity = 0.0
        shared_keys = intersection
        
        if shared_keys:
            matches = sum(
                1 for key in shared_keys
                if context1[key] == context2[key]
            )
            value_similarity = matches / len(shared_keys)
        
        # Combined score
        return 0.5 * key_similarity + 0.5 * value_similarity


# Usage
retriever = EpisodeRetriever(store)

# Find similar past episodes
current_context = {
    'user': 'alice',
    'task': 'debug',
    'language': 'python'
}

similar = retriever.find_similar_episodes(
    current_context,
    episode_type=EpisodeType.TASK,
    limit=3
)

print("Similar past episodes:")
for ep in similar:
    print(f"- {ep.description} (outcome: {ep.outcome.value})")

# Find successful patterns
patterns = retriever.find_successful_patterns(EpisodeType.TASK)
print("\nSuccessful patterns:")
for pattern in patterns[:3]:
    print(f"- {' -> '.join(pattern['sequence'])}")
    print(f"  Occurred {pattern['count']} times ({pattern['frequency']:.1%})")
```

## Learning from Episodes

Extract lessons from past experiences.

### Episode Analyzer

```python
class EpisodeAnalyzer:
    """Analyze episodes to extract insights."""
    
    def __init__(self, retriever: EpisodeRetriever):
        self.retriever = retriever
    
    def analyze_failures(self, episode_type: Optional[EpisodeType] = None,
                        limit: int = 10) -> Dict:
        """Analyze failure episodes to find common causes."""
        failures = self.retriever.find_by_outcome(
            Outcome.FAILURE,
            episode_type=episode_type,
            limit=limit
        )
        
        if not failures:
            return {'failures': 0, 'patterns': []}
        
        # Analyze common patterns in failures
        common_contexts = defaultdict(int)
        common_actions = defaultdict(int)
        
        for episode in failures:
            # Context patterns
            for key, value in episode.context.items():
                common_contexts[f"{key}={value}"] += 1
            
            # Action patterns
            for action in episode.actions:
                action_type = action.get('type', 'unknown')
                common_actions[action_type] += 1
        
        return {
            'failures': len(failures),
            'common_contexts': dict(sorted(
                common_contexts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]),
            'common_actions': dict(sorted(
                common_actions.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]),
            'examples': [
                {
                    'description': ep.description,
                    'timestamp': ep.timestamp.isoformat(),
                    'context': ep.context
                }
                for ep in failures[:3]
            ]
        }
    
    def compare_success_vs_failure(self, episode_type: EpisodeType) -> Dict:
        """Compare successful vs failed episodes."""
        successes = self.retriever.find_by_outcome(
            Outcome.SUCCESS,
            episode_type=episode_type,
            limit=50
        )
        
        failures = self.retriever.find_by_outcome(
            Outcome.FAILURE,
            episode_type=episode_type,
            limit=50
        )
        
        def analyze_group(episodes):
            if not episodes:
                return {}
            
            return {
                'count': len(episodes),
                'avg_duration': sum(
                    ep.duration_seconds or 0 for ep in episodes
                ) / len(episodes),
                'avg_actions': sum(
                    len(ep.actions) for ep in episodes
                ) / len(episodes),
                'common_tags': self._most_common_tags(episodes, 5)
            }
        
        return {
            'success': analyze_group(successes),
            'failure': analyze_group(failures),
            'insights': self._generate_insights(successes, failures)
        }
    
    def _most_common_tags(self, episodes: List[Episode], limit: int) -> List[str]:
        """Find most common tags."""
        tag_counts = defaultdict(int)
        
        for episode in episodes:
            for tag in episode.tags:
                tag_counts[tag] += 1
        
        sorted_tags = sorted(
            tag_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [tag for tag, _ in sorted_tags[:limit]]
    
    def _generate_insights(self, successes: List[Episode],
                          failures: List[Episode]) -> List[str]:
        """Generate insights from comparison."""
        insights = []
        
        if successes and failures:
            success_avg_duration = sum(
                ep.duration_seconds or 0 for ep in successes
            ) / len(successes)
            
            failure_avg_duration = sum(
                ep.duration_seconds or 0 for ep in failures
            ) / len(failures)
            
            if success_avg_duration < failure_avg_duration * 0.8:
                insights.append(
                    "Successful episodes tend to complete faster"
                )
            
            success_actions = sum(len(ep.actions) for ep in successes) / len(successes)
            failure_actions = sum(len(ep.actions) for ep in failures) / len(failures)
            
            if success_actions > failure_actions:
                insights.append(
                    "Successful episodes involve more actions (more thorough)"
                )
        
        return insights


# Usage
analyzer = EpisodeAnalyzer(retriever)

# Analyze failures
failure_analysis = analyzer.analyze_failures(EpisodeType.TASK)
print("Failure Analysis:")
print(f"Total failures: {failure_analysis['failures']}")
print("\nCommon context patterns:")
for pattern, count in list(failure_analysis['common_contexts'].items())[:3]:
    print(f"  {pattern}: {count} times")

# Compare success vs failure
comparison = analyzer.compare_success_vs_failure(EpisodeType.TASK)
print("\nSuccess vs Failure:")
print(f"Successes: {comparison['success']['count']}")
print(f"  Avg duration: {comparison['success']['avg_duration']:.1f}s")
print(f"Failures: {comparison['failure']['count']}")
print(f"  Avg duration: {comparison['failure']['avg_duration']:.1f}s")
print("\nInsights:")
for insight in comparison['insights']:
    print(f"  - {insight}")
```

## Episode Similarity

Finding similar past episodes.

### Similarity Metrics

```python
import numpy as np

class EpisodeSimilarityCalculator:
    """Calculate similarity between episodes."""
    
    def __init__(self, embedder=None):
        self.embedder = embedder
    
    def calculate_similarity(self, episode1: Episode,
                           episode2: Episode) -> Dict[str, float]:
        """Calculate multi-faceted similarity."""
        return {
            'type_match': self._type_similarity(episode1, episode2),
            'context_similarity': self._context_similarity(
                episode1.context,
                episode2.context
            ),
            'action_similarity': self._action_similarity(
                episode1.actions,
                episode2.actions
            ),
            'temporal_proximity': self._temporal_proximity(
                episode1.timestamp,
                episode2.timestamp
            ),
            'outcome_match': self._outcome_similarity(
                episode1.outcome,
                episode2.outcome
            )
        }
    
    def overall_similarity(self, episode1: Episode,
                          episode2: Episode,
                          weights: Dict[str, float] = None) -> float:
        """Calculate weighted overall similarity."""
        if weights is None:
            weights = {
                'type_match': 0.2,
                'context_similarity': 0.3,
                'action_similarity': 0.3,
                'temporal_proximity': 0.1,
                'outcome_match': 0.1
            }
        
        similarities = self.calculate_similarity(episode1, episode2)
        
        return sum(
            sim * weights.get(key, 0)
            for key, sim in similarities.items()
        )
    
    def _type_similarity(self, ep1: Episode, ep2: Episode) -> float:
        """Check if episode types match."""
        return 1.0 if ep1.episode_type == ep2.episode_type else 0.0
    
    def _context_similarity(self, ctx1: Dict, ctx2: Dict) -> float:
        """Calculate context similarity."""
        if not ctx1 or not ctx2:
            return 0.0
        
        # Jaccard similarity on keys
        keys1 = set(ctx1.keys())
        keys2 = set(ctx2.keys())
        
        if not keys1 or not keys2:
            return 0.0
        
        intersection = keys1 & keys2
        union = keys1 | keys2
        
        return len(intersection) / len(union)
    
    def _action_similarity(self, actions1: List[Dict],
                          actions2: List[Dict]) -> float:
        """Calculate action sequence similarity."""
        if not actions1 or not actions2:
            return 0.0
        
        # Extract action types
        seq1 = [a.get('type', '') for a in actions1]
        seq2 = [a.get('type', '') for a in actions2]
        
        # Use longest common subsequence
        lcs_length = self._lcs_length(seq1, seq2)
        max_length = max(len(seq1), len(seq2))
        
        return lcs_length / max_length if max_length > 0 else 0.0
    
    def _lcs_length(self, seq1: List, seq2: List) -> int:
        """Calculate longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _temporal_proximity(self, time1: datetime,
                          time2: datetime) -> float:
        """Calculate temporal proximity (closer = more similar)."""
        delta = abs((time1 - time2).total_seconds())
        
        # Exponential decay: similarity = e^(-delta / half_life)
        half_life = 7 * 24 * 3600  # 1 week in seconds
        
        return np.exp(-delta / half_life)
    
    def _outcome_similarity(self, outcome1: Outcome,
                          outcome2: Outcome) -> float:
        """Check if outcomes match."""
        return 1.0 if outcome1 == outcome2 else 0.0


# Usage
calculator = EpisodeSimilarityCalculator()

# Compare two episodes
ep1 = store.get_episode(episode_id)
similar_eps = retriever.find_similar_episodes(ep1.context)

if similar_eps:
    ep2 = similar_eps[0]
    
    similarity = calculator.calculate_similarity(ep1, ep2)
    print("Similarity breakdown:")
    for aspect, score in similarity.items():
        print(f"  {aspect}: {score:.2f}")
    
    overall = calculator.overall_similarity(ep1, ep2)
    print(f"\nOverall similarity: {overall:.2f}")
```

## Task Episodes

Specialized handling for task-based episodes.

### Task Episode Manager

```python
class TaskEpisodeManager:
    """Manage task-specific episodes."""
    
    def __init__(self, recorder: EpisodeRecorder):
        self.recorder = recorder
        self.active_tasks: Dict[str, str] = {}  # task_id -> episode_id
    
    def start_task(self, task_description: str, 
                  context: Dict = None) -> str:
        """Start a task episode."""
        task_id = str(uuid.uuid4())
        
        episode_id = self.recorder.start_episode(
            EpisodeType.TASK,
            task_description,
            context={
                'task_id': task_id,
                **(context or {})
            }
        )
        
        self.active_tasks[task_id] = episode_id
        
        return task_id
    
    def record_subtask(self, task_id: str, subtask: str,
                      **kwargs):
        """Record a subtask as an action."""
        if task_id not in self.active_tasks:
            raise ValueError(f"No active task: {task_id}")
        
        self.recorder.record_action(
            'subtask',
            description=subtask,
            **kwargs
        )
    
    def complete_task(self, task_id: str, success: bool,
                     result: Any = None):
        """Complete a task episode."""
        if task_id not in self.active_tasks:
            raise ValueError(f"No active task: {task_id}")
        
        outcome = Outcome.SUCCESS if success else Outcome.FAILURE
        
        episode_id = self.recorder.end_episode(
            outcome=outcome,
            result=result
        )
        
        del self.active_tasks[task_id]
        
        return episode_id
    
    def get_task_history(self, task_type: str, limit: int = 10) -> List[Episode]:
        """Get history of similar tasks."""
        all_episodes = self.recorder.store.get_recent_episodes(
            episode_type=EpisodeType.TASK,
            limit=100
        )
        
        # Filter by task type in description or context
        relevant = [
            ep for ep in all_episodes
            if (task_type.lower() in ep.description.lower() or
                task_type.lower() in str(ep.context.get('task_type', '')).lower())
        ]
        
        return relevant[:limit]


# Usage
task_manager = TaskEpisodeManager(recorder)

# Start a task
task_id = task_manager.start_task(
    "Generate unit tests for Python module",
    context={'module': 'calculator.py', 'task_type': 'testing'}
)

# Record subtasks
task_manager.record_subtask(task_id, "Analyze module functions", functions_found=5)
task_manager.record_subtask(task_id, "Generate test cases", tests_generated=15)
task_manager.record_subtask(task_id, "Write test file", file='test_calculator.py')

# Complete task
task_manager.complete_task(
    task_id,
    success=True,
    result={'tests': 15, 'coverage': 0.92}
)

# Get task history
history = task_manager.get_task_history('testing')
print(f"Found {len(history)} similar testing tasks")
```

## Interaction Episodes

Recording user interactions.

### Interaction Episode Tracker

```python
class InteractionEpisodeTracker:
    """Track user interaction episodes."""
    
    def __init__(self, recorder: EpisodeRecorder):
        self.recorder = recorder
        self.session_episodes: Dict[str, List[str]] = defaultdict(list)
    
    def record_interaction(self, user_id: str, session_id: str,
                          user_input: str, agent_response: str,
                          metadata: Dict = None):
        """Record a single interaction."""
        episode_id = self.recorder.start_episode(
            EpisodeType.INTERACTION,
            f"User: {user_input[:50]}...",
            context={
                'user_id': user_id,
                'session_id': session_id,
                'user_input': user_input,
                **(metadata or {})
            }
        )
        
        self.recorder.record_action(
            'user_message',
            content=user_input
        )
        
        self.recorder.record_action(
            'agent_response',
            content=agent_response
        )
        
        # Determine outcome based on response quality
        outcome = self._assess_interaction_quality(user_input, agent_response)
        
        episode_id = self.recorder.end_episode(
            outcome=outcome,
            result={'response': agent_response}
        )
        
        self.session_episodes[session_id].append(episode_id)
        
        return episode_id
    
    def get_session_summary(self, session_id: str) -> Dict:
        """Summarize a session's interactions."""
        episode_ids = self.session_episodes.get(session_id, [])
        
        if not episode_ids:
            return {'session_id': session_id, 'interactions': 0}
        
        episodes = [
            self.recorder.store.get_episode(ep_id)
            for ep_id in episode_ids
        ]
        episodes = [ep for ep in episodes if ep]  # Filter None
        
        return {
            'session_id': session_id,
            'interactions': len(episodes),
            'duration': self._calculate_session_duration(episodes),
            'topics': self._extract_topics(episodes),
            'outcome_distribution': self._outcome_distribution(episodes)
        }
    
    def _assess_interaction_quality(self, user_input: str,
                                   agent_response: str) -> Outcome:
        """Assess interaction quality (simplified)."""
        # In practice, would use more sophisticated assessment
        
        if len(agent_response) < 10:
            return Outcome.FAILURE
        
        # Check for error indicators
        error_phrases = ['error', 'cannot', "don't understand", 'failed']
        if any(phrase in agent_response.lower() for phrase in error_phrases):
            return Outcome.PARTIAL
        
        return Outcome.SUCCESS
    
    def _calculate_session_duration(self, episodes: List[Episode]) -> float:
        """Calculate total session duration."""
        if not episodes:
            return 0.0
        
        start = min(ep.timestamp for ep in episodes)
        end = max(ep.timestamp for ep in episodes)
        
        return (end - start).total_seconds()
    
    def _extract_topics(self, episodes: List[Episode]) -> List[str]:
        """Extract topics from episodes."""
        all_tags = []
        for ep in episodes:
            all_tags.extend(ep.tags)
        
        # Count and return most common
        tag_counts = defaultdict(int)
        for tag in all_tags:
            tag_counts[tag] += 1
        
        sorted_tags = sorted(
            tag_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [tag for tag, _ in sorted_tags[:5]]
    
    def _outcome_distribution(self, episodes: List[Episode]) -> Dict[str, int]:
        """Get distribution of outcomes."""
        distribution = defaultdict(int)
        
        for ep in episodes:
            distribution[ep.outcome.value] += 1
        
        return dict(distribution)


# Usage
interaction_tracker = InteractionEpisodeTracker(recorder)

# Record interactions
session_id = "session_456"

interaction_tracker.record_interaction(
    user_id="alice",
    session_id=session_id,
    user_input="How do I sort a list in Python?",
    agent_response="You can use the sorted() function or the .sort() method...",
    metadata={'intent': 'question'}
)

interaction_tracker.record_interaction(
    user_id="alice",
    session_id=session_id,
    user_input="Can you show an example?",
    agent_response="Sure! Here's an example: numbers = [3, 1, 4, 1, 5]...",
    metadata={'intent': 'clarification'}
)

# Get session summary
summary = interaction_tracker.get_session_summary(session_id)
print(f"Session summary:")
print(f"  Interactions: {summary['interactions']}")
print(f"  Duration: {summary['duration']:.1f}s")
print(f"  Outcomes: {summary['outcome_distribution']}")
```

## Success and Failure Analysis

Learning from outcomes.

### Outcome Analyzer

```python
class OutcomeAnalyzer:
    """Analyze episode outcomes."""
    
    def __init__(self, store: EpisodicMemoryStore):
        self.store = store
    
    def get_success_rate(self, episode_type: Optional[EpisodeType] = None,
                        time_range_days: int = 30) -> Dict:
        """Calculate success rate."""
        cutoff = datetime.now() - timedelta(days=time_range_days)
        
        # Get recent episodes
        if episode_type:
            episodes = self.store.get_recent_episodes(
                limit=1000,
                episode_type=episode_type
            )
        else:
            episodes = self.store.get_recent_episodes(limit=1000)
        
        # Filter by time
        episodes = [ep for ep in episodes if ep.timestamp >= cutoff]
        
        if not episodes:
            return {'total': 0, 'success_rate': 0.0}
        
        # Count outcomes
        outcome_counts = defaultdict(int)
        for ep in episodes:
            outcome_counts[ep.outcome] += 1
        
        total = len(episodes)
        successes = outcome_counts[Outcome.SUCCESS]
        
        return {
            'total': total,
            'successes': successes,
            'failures': outcome_counts[Outcome.FAILURE],
            'partial': outcome_counts[Outcome.PARTIAL],
            'success_rate': successes / total if total > 0 else 0.0,
            'outcome_distribution': dict(outcome_counts)
        }
    
    def identify_failure_patterns(self, limit: int = 20) -> List[Dict]:
        """Identify common failure patterns."""
        failures = self.store.get_episodes_by_outcome(
            Outcome.FAILURE,
            limit=limit
        )
        
        if not failures:
            return []
        
        # Group by similar contexts
        patterns = defaultdict(list)
        
        for episode in failures:
            # Create pattern key from context
            pattern_key = self._create_pattern_key(episode.context)
            patterns[pattern_key].append(episode)
        
        # Format patterns
        formatted = []
        for pattern_key, episodes in patterns.items():
            if len(episodes) >= 2:  # At least 2 occurrences
                formatted.append({
                    'pattern': pattern_key,
                    'occurrences': len(episodes),
                    'examples': [
                        {
                            'description': ep.description,
                            'timestamp': ep.timestamp.isoformat()
                        }
                        for ep in episodes[:3]
                    ]
                })
        
        # Sort by occurrences
        formatted.sort(key=lambda x: x['occurrences'], reverse=True)
        
        return formatted
    
    def _create_pattern_key(self, context: Dict) -> str:
        """Create pattern key from context."""
        # Use important context keys
        important_keys = ['task_type', 'language', 'tool', 'action_type']
        
        parts = []
        for key in important_keys:
            if key in context:
                parts.append(f"{key}={context[key]}")
        
        return "|".join(parts) if parts else "generic"


# Usage
outcome_analyzer = OutcomeAnalyzer(store)

# Get success rate
success_stats = outcome_analyzer.get_success_rate(
    episode_type=EpisodeType.TASK,
    time_range_days=7
)

print("Success Rate (last 7 days):")
print(f"  Total tasks: {success_stats['total']}")
print(f"  Success rate: {success_stats['success_rate']:.1%}")
print(f"  Distribution: {success_stats['outcome_distribution']}")

# Identify failure patterns
patterns = outcome_analyzer.identify_failure_patterns()
print("\nCommon failure patterns:")
for pattern in patterns[:3]:
    print(f"  {pattern['pattern']}: {pattern['occurrences']} occurrences")
```

## Episode Summarization

Condensing episode information.

### Episode Summarizer

```python
class EpisodeSummarizer:
    """Create summaries of episodes."""
    
    def summarize_episode(self, episode: Episode) -> str:
        """Create a concise summary of an episode."""
        parts = []
        
        # Basic info
        outcome_symbol = {
            Outcome.SUCCESS: "✓",
            Outcome.FAILURE: "✗",
            Outcome.PARTIAL: "◐"
        }.get(episode.outcome, "?")
        
        parts.append(
            f"{outcome_symbol} [{episode.episode_type.value}] {episode.description}"
        )
        
        # Duration
        if episode.duration_seconds:
            parts.append(f"  Duration: {episode.duration_seconds:.1f}s")
        
        # Key actions
        if episode.actions:
            action_summary = f"  Actions: {len(episode.actions)} steps"
            if episode.actions:
                first_actions = [a.get('type', 'unknown') for a in episode.actions[:3]]
                action_summary += f" ({', '.join(first_actions)}...)"
            parts.append(action_summary)
        
        # Result summary
        if episode.result:
            result_str = str(episode.result)
            if len(result_str) > 50:
                result_str = result_str[:50] + "..."
            parts.append(f"  Result: {result_str}")
        
        return "\n".join(parts)
    
    def summarize_session(self, episodes: List[Episode]) -> str:
        """Create a summary of multiple episodes."""
        if not episodes:
            return "No episodes in session"
        
        lines = [
            f"Session Summary ({len(episodes)} episodes)",
            "=" * 50
        ]
        
        # Time range
        start = min(ep.timestamp for ep in episodes)
        end = max(ep.timestamp for ep in episodes)
        duration = (end - start).total_seconds()
        
        lines.append(f"Duration: {duration:.1f}s")
        
        # Outcome distribution
        outcomes = defaultdict(int)
        for ep in episodes:
            outcomes[ep.outcome] += 1
        
        lines.append("\nOutcomes:")
        for outcome, count in outcomes.items():
            lines.append(f"  {outcome.value}: {count}")
        
        # Key episodes
        lines.append("\nKey Episodes:")
        important = sorted(
            episodes,
            key=lambda ep: ep.importance,
            reverse=True
        )[:5]
        
        for ep in important:
            lines.append(f"  - {ep.description[:60]}")
        
        return "\n".join(lines)


# Usage
summarizer = EpisodeSummarizer()

# Summarize single episode
episode = store.get_recent_episodes(limit=1)[0]
print(summarizer.summarize_episode(episode))

# Summarize session
session_episodes = timeline.get_session_episodes("session_123")
print("\n" + summarizer.summarize_session(session_episodes))
```

## Summary

Episodic memory enables agents to remember and learn from experiences:

**Key Concepts**:

- **Episodes**: Structured records of specific experiences
- **Temporal organization**: Events ordered by time
- **Context preservation**: Capturing the circumstances
- **Outcome tracking**: Recording success, failure, partial success
- **Pattern recognition**: Finding similarities across episodes

**Implementation Strategies**:

- **Structured storage**: SQL or document databases
- **Rich metadata**: Type, context, actions, outcomes
- **Similarity metrics**: Finding related past experiences
- **Timeline management**: Organizing by time
- **Analysis tools**: Learning from successes and failures

**Use Cases**:

- Learning from past mistakes
- Recognizing recurring patterns
- Retrieving relevant past experiences
- Maintaining conversation context
- Analyzing task performance trends

## Next Steps

- **[Semantic Memory](semantic-memory.md)**: Storing facts and knowledge
- **[Memory Retrieval](memory-retrieval.md)**: Finding relevant information efficiently
- **[Memory Consolidation](memory-consolidation.md)**: Moving from episodes to knowledge
- **[Memory Architecture](memory-architecture.md)**: Overall system design
