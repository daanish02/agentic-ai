# Memory Consolidation

## Table of Contents

- [Introduction](#introduction)
- [Consolidation Fundamentals](#consolidation-fundamentals)
- [When to Consolidate](#when-to-consolidate)
- [What to Consolidate](#what-to-consolidate)
- [Summarization Strategies](#summarization-strategies)
- [Importance Scoring](#importance-scoring)
- [Transition Management](#transition-management)
- [Consolidation Policies](#consolidation-policies)
- [Incremental Consolidation](#incremental-consolidation)
- [Compression Techniques](#compression-techniques)
- [Semantic Extraction](#semantic-extraction)
- [Memory Lifecycle](#memory-lifecycle)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Memory consolidation is the process of transferring information from short-term (working memory, context window) to long-term persistent storage. It determines what knowledge the agent retains permanently and how that knowledge is structured.

> "Like human sleep consolidates daily experiences into lasting memories, agent consolidation transforms transient interactions into permanent knowledge."

Key challenges:

- **Selectivity**: What deserves long-term storage?
- **Timing**: When should consolidation occur?
- **Summarization**: How to compress without losing meaning?
- **Organization**: How to structure consolidated memories?
- **Evolution**: How to update existing memories?

### Consolidation Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Short-Term Memory                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │
│  │ Context    │  │ Working    │  │ Recent             │ │
│  │ Window     │  │ Memory     │  │ Interactions       │ │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────────────┘ │
│        │               │               │                 │
│        └───────────────┴───────────────┘                 │
│                        │                                 │
│                        ▼                                 │
│          ┌─────────────────────────────┐                │
│          │ Consolidation Trigger       │                │
│          │ - Time elapsed              │                │
│          │ - Interaction ended         │                │
│          │ - Memory full               │                │
│          │ - Explicit signal           │                │
│          └──────────┬──────────────────┘                │
│                     │                                    │
│                     ▼                                    │
│          ┌─────────────────────────────┐                │
│          │ Selection & Filtering       │                │
│          │ - Importance scoring        │                │
│          │ - Novelty detection         │                │
│          │ - Redundancy removal        │                │
│          └──────────┬──────────────────┘                │
│                     │                                    │
│                     ▼                                    │
│          ┌─────────────────────────────┐                │
│          │ Transformation              │                │
│          │ - Summarization             │                │
│          │ - Abstraction               │                │
│          │ - Fact extraction           │                │
│          └──────────┬──────────────────┘                │
│                     │                                    │
│                     ▼                                    │
│          ┌─────────────────────────────┐                │
│          │ Integration                 │                │
│          │ - Merge with existing       │                │
│          │ - Update relations          │                │
│          │ - Build connections         │                │
│          └──────────┬──────────────────┘                │
│                     │                                    │
│                     ▼                                    │
└─────────────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│                  Long-Term Memory                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │
│  │ Episodic   │  │ Semantic   │  │ Procedural         │ │
│  │ Memory     │  │ Memory     │  │ Memory             │ │
│  └────────────┘  └────────────┘  └────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

## Consolidation Fundamentals

Core concepts and data structures.

### Consolidation Event

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class ConsolidationTrigger(Enum):
    """Triggers for consolidation."""
    TIME_ELAPSED = "time_elapsed"
    INTERACTION_ENDED = "interaction_ended"
    MEMORY_FULL = "memory_full"
    EXPLICIT = "explicit"
    PERIODIC = "periodic"

class ConsolidationType(Enum):
    """Types of consolidation."""
    EPISODIC = "episodic"  # Store as episode
    SEMANTIC = "semantic"  # Extract facts
    SUMMARIZE = "summarize"  # Compress information
    DISCARD = "discard"  # Not worth keeping

@dataclass
class ConsolidationCandidate:
    """A memory item being considered for consolidation."""
    id: str
    content: str
    source: str  # 'context_window', 'working_memory', etc.
    timestamp: datetime
    importance: float = 0.5
    novelty: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not hasattr(self, 'metadata') or self.metadata is None:
            self.metadata = {}

@dataclass
class ConsolidationResult:
    """Result of consolidation process."""
    candidate_id: str
    decision: ConsolidationType
    consolidated_content: Optional[str]
    storage_location: Optional[str]
    importance_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Consolidation Manager

```python
class ConsolidationManager:
    """Manages memory consolidation process."""

    def __init__(self,
                 importance_threshold: float = 0.3,
                 novelty_threshold: float = 0.2):
        self.importance_threshold = importance_threshold
        self.novelty_threshold = novelty_threshold
        self.consolidation_history: List[ConsolidationResult] = []

    def consolidate(self,
                   candidates: List[ConsolidationCandidate]) -> List[ConsolidationResult]:
        """
        Consolidate a batch of memory candidates.

        Args:
            candidates: Memories to consolidate

        Returns:
            List of consolidation results
        """
        results = []

        for candidate in candidates:
            # Score importance and novelty
            importance = self._score_importance(candidate)
            novelty = self._score_novelty(candidate)

            # Update scores
            candidate.importance = importance
            candidate.novelty = novelty

            # Decide what to do with this candidate
            decision = self._make_decision(candidate)

            # Execute consolidation
            result = self._execute_consolidation(candidate, decision)

            results.append(result)
            self.consolidation_history.append(result)

        return results

    def _score_importance(self, candidate: ConsolidationCandidate) -> float:
        """Score how important this memory is."""
        score = 0.0

        # Length (longer = potentially more important)
        length_score = min(len(candidate.content) / 500, 0.3)
        score += length_score

        # Contains specific markers
        if any(marker in candidate.content.lower() for marker in
               ['important', 'critical', 'remember', 'always', 'never']):
            score += 0.3

        # User preferences/instructions
        if 'preference' in candidate.metadata or 'instruction' in candidate.metadata:
            score += 0.4

        return min(score, 1.0)

    def _score_novelty(self, candidate: ConsolidationCandidate) -> float:
        """Score how novel this memory is."""
        # Check against recent consolidations
        recent_contents = [
            r.consolidated_content
            for r in self.consolidation_history[-50:]
            if r.consolidated_content
        ]

        # Simple overlap check (would use embeddings in practice)
        max_similarity = 0.0
        candidate_words = set(candidate.content.lower().split())

        for existing in recent_contents:
            existing_words = set(existing.lower().split())
            if not candidate_words or not existing_words:
                continue

            intersection = candidate_words & existing_words
            union = candidate_words | existing_words

            similarity = len(intersection) / len(union) if union else 0
            max_similarity = max(max_similarity, similarity)

        # Novelty is inverse of similarity
        return 1.0 - max_similarity

    def _make_decision(self, candidate: ConsolidationCandidate) -> ConsolidationType:
        """Decide what type of consolidation to perform."""
        # High importance and novelty -> store as episode
        if (candidate.importance >= self.importance_threshold and
            candidate.novelty >= self.novelty_threshold):

            # Check if it's a fact vs experience
            if self._is_factual(candidate):
                return ConsolidationType.SEMANTIC
            else:
                return ConsolidationType.EPISODIC

        # Medium importance -> summarize
        elif candidate.importance >= self.importance_threshold * 0.5:
            return ConsolidationType.SUMMARIZE

        # Low importance -> discard
        else:
            return ConsolidationType.DISCARD

    def _is_factual(self, candidate: ConsolidationCandidate) -> bool:
        """Check if memory is factual knowledge."""
        # Simple heuristic (would use NLU in practice)
        factual_patterns = [
            'is', 'are', 'was', 'were',
            'defined as', 'means', 'refers to',
            'can', 'cannot', 'always', 'never'
        ]

        content_lower = candidate.content.lower()
        return any(pattern in content_lower for pattern in factual_patterns)

    def _execute_consolidation(self,
                              candidate: ConsolidationCandidate,
                              decision: ConsolidationType) -> ConsolidationResult:
        """Execute the consolidation decision."""
        if decision == ConsolidationType.DISCARD:
            return ConsolidationResult(
                candidate_id=candidate.id,
                decision=decision,
                consolidated_content=None,
                storage_location=None,
                importance_score=candidate.importance
            )

        # Transform content based on decision
        if decision == ConsolidationType.SUMMARIZE:
            consolidated = self._summarize(candidate.content)
        else:
            consolidated = candidate.content

        # Determine storage location
        storage_location = self._get_storage_location(decision)

        return ConsolidationResult(
            candidate_id=candidate.id,
            decision=decision,
            consolidated_content=consolidated,
            storage_location=storage_location,
            importance_score=candidate.importance,
            metadata={'novelty': candidate.novelty}
        )

    def _summarize(self, content: str) -> str:
        """Summarize content (simple version)."""
        # In practice, would use LLM or extractive summarization
        sentences = content.split('.')

        # Keep first and most important sentences
        if len(sentences) <= 2:
            return content

        # Simple: keep first and last sentence
        return f"{sentences[0].strip()}. {sentences[-1].strip()}."

    def _get_storage_location(self, decision: ConsolidationType) -> str:
        """Get storage location for consolidated memory."""
        if decision == ConsolidationType.EPISODIC:
            return "episodic_memory"
        elif decision == ConsolidationType.SEMANTIC:
            return "semantic_memory"
        elif decision == ConsolidationType.SUMMARIZE:
            return "compressed_memory"
        else:
            return None


# Usage
consolidation_mgr = ConsolidationManager(
    importance_threshold=0.4,
    novelty_threshold=0.3
)

# Create candidates
import uuid

candidates = [
    ConsolidationCandidate(
        id=str(uuid.uuid4()),
        content="User mentioned they prefer Python over JavaScript for data science work",
        source="context_window",
        timestamp=datetime.now(),
        metadata={'type': 'preference'}
    ),
    ConsolidationCandidate(
        id=str(uuid.uuid4()),
        content="The weather was discussed",
        source="context_window",
        timestamp=datetime.now()
    ),
    ConsolidationCandidate(
        id=str(uuid.uuid4()),
        content="TensorFlow is a machine learning framework developed by Google",
        source="context_window",
        timestamp=datetime.now(),
        metadata={'type': 'fact'}
    )
]

# Consolidate
results = consolidation_mgr.consolidate(candidates)

print("Consolidation Results:")
for result in results:
    print(f"\n  Decision: {result.decision.value}")
    print(f"  Importance: {result.importance_score:.3f}")
    if result.consolidated_content:
        print(f"  Content: {result.consolidated_content[:60]}...")
        print(f"  Storage: {result.storage_location}")
```

## When to Consolidate

Determining consolidation timing.

### Consolidation Triggers

```python
class ConsolidationTriggerManager:
    """Manage when consolidation should occur."""

    def __init__(self):
        self.last_consolidation = datetime.now()
        self.interaction_count = 0
        self.memory_size = 0
        self.triggers_enabled = {
            ConsolidationTrigger.TIME_ELAPSED: True,
            ConsolidationTrigger.INTERACTION_ENDED: True,
            ConsolidationTrigger.MEMORY_FULL: True,
            ConsolidationTrigger.PERIODIC: True,
        }

    def check_triggers(self,
                      current_memory_size: int,
                      max_memory_size: int,
                      time_threshold: int = 300,  # 5 minutes
                      interaction_ended: bool = False) -> List[ConsolidationTrigger]:
        """
        Check which consolidation triggers are active.

        Args:
            current_memory_size: Current size of short-term memory
            max_memory_size: Maximum allowed size
            time_threshold: Seconds before time trigger activates
            interaction_ended: Whether current interaction has ended

        Returns:
            List of active triggers
        """
        active_triggers = []

        # Time elapsed trigger
        if self.triggers_enabled[ConsolidationTrigger.TIME_ELAPSED]:
            elapsed = (datetime.now() - self.last_consolidation).total_seconds()
            if elapsed >= time_threshold:
                active_triggers.append(ConsolidationTrigger.TIME_ELAPSED)

        # Interaction ended trigger
        if (self.triggers_enabled[ConsolidationTrigger.INTERACTION_ENDED] and
            interaction_ended):
            active_triggers.append(ConsolidationTrigger.INTERACTION_ENDED)

        # Memory full trigger
        if self.triggers_enabled[ConsolidationTrigger.MEMORY_FULL]:
            usage_ratio = current_memory_size / max_memory_size
            if usage_ratio >= 0.9:  # 90% full
                active_triggers.append(ConsolidationTrigger.MEMORY_FULL)

        return active_triggers

    def should_consolidate(self, *args, **kwargs) -> bool:
        """Check if any trigger is active."""
        return len(self.check_triggers(*args, **kwargs)) > 0

    def mark_consolidated(self):
        """Mark that consolidation occurred."""
        self.last_consolidation = datetime.now()


# Usage
trigger_mgr = ConsolidationTriggerManager()

# Check if should consolidate
should_consolidate = trigger_mgr.should_consolidate(
    current_memory_size=1800,
    max_memory_size=2000,
    time_threshold=300,
    interaction_ended=True
)

print(f"\nShould consolidate: {should_consolidate}")
if should_consolidate:
    triggers = trigger_mgr.check_triggers(1800, 2000, 300, True)
    print(f"Active triggers: {[t.value for t in triggers]}")
```

### Scheduled Consolidation

```python
from threading import Thread, Event
import time

class ScheduledConsolidation:
    """Run consolidation on a schedule."""

    def __init__(self,
                 consolidation_fn: callable,
                 interval_seconds: int = 300):
        self.consolidation_fn = consolidation_fn
        self.interval = interval_seconds
        self.running = False
        self.stop_event = Event()
        self.thread = None

    def start(self):
        """Start scheduled consolidation."""
        if self.running:
            return

        self.running = True
        self.stop_event.clear()
        self.thread = Thread(target=self._run_schedule)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop scheduled consolidation."""
        self.running = False
        self.stop_event.set()
        if self.thread:
            self.thread.join()

    def _run_schedule(self):
        """Run consolidation loop."""
        while self.running:
            # Wait for interval
            if self.stop_event.wait(self.interval):
                break

            # Run consolidation
            try:
                self.consolidation_fn()
            except Exception as e:
                print(f"Consolidation error: {e}")


# Usage (would call actual consolidation function)
def mock_consolidation():
    print(f"[{datetime.now()}] Running scheduled consolidation...")

scheduler = ScheduledConsolidation(
    consolidation_fn=mock_consolidation,
    interval_seconds=60  # Every minute
)

# Would call: scheduler.start()
```

## What to Consolidate

Selecting which memories to keep.

### Memory Selector

```python
class MemorySelector:
    """Select which memories to consolidate."""

    def __init__(self):
        self.selection_criteria = {
            'importance': 0.4,
            'novelty': 0.3,
            'recency': 0.2,
            'relevance': 0.1
        }

    def select(self,
              candidates: List[ConsolidationCandidate],
              max_select: int = None) -> List[ConsolidationCandidate]:
        """
        Select memories for consolidation.

        Args:
            candidates: All candidate memories
            max_select: Maximum number to select (None = all above threshold)

        Returns:
            Selected candidates
        """
        # Score each candidate
        scored_candidates = []

        for candidate in candidates:
            score = self._calculate_selection_score(candidate)
            scored_candidates.append((candidate, score))

        # Sort by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Select top candidates
        if max_select:
            selected = scored_candidates[:max_select]
        else:
            # Select all above threshold
            threshold = 0.5
            selected = [(c, s) for c, s in scored_candidates if s >= threshold]

        return [candidate for candidate, score in selected]

    def _calculate_selection_score(self,
                                   candidate: ConsolidationCandidate) -> float:
        """Calculate selection score for a candidate."""
        score = 0.0

        # Importance
        score += candidate.importance * self.selection_criteria['importance']

        # Novelty
        score += candidate.novelty * self.selection_criteria['novelty']

        # Recency (more recent = higher score)
        age_hours = (datetime.now() - candidate.timestamp).total_seconds() / 3600
        recency_score = math.exp(-age_hours / 24)  # Decay over days
        score += recency_score * self.selection_criteria['recency']

        # Relevance (would calculate based on current context)
        relevance_score = 0.5  # Placeholder
        score += relevance_score * self.selection_criteria['relevance']

        return score


# Usage
import math

selector = MemorySelector()
selected = selector.select(candidates, max_select=2)

print(f"\nSelected {len(selected)} memories for consolidation:")
for candidate in selected:
    print(f"  - {candidate.content[:50]}...")
```

## Summarization Strategies

Compressing information effectively.

### Extractive Summarizer

```python
class ExtractiveSummarizer:
    """Extract key sentences from content."""

    def summarize(self,
                 content: str,
                 num_sentences: int = 2) -> str:
        """
        Extractive summarization.

        Args:
            content: Text to summarize
            num_sentences: Number of sentences to extract

        Returns:
            Summary with extracted sentences
        """
        # Split into sentences
        sentences = [s.strip() for s in content.split('.') if s.strip()]

        if len(sentences) <= num_sentences:
            return content

        # Score sentences
        scored_sentences = []

        for i, sentence in enumerate(sentences):
            score = self._score_sentence(sentence, i, len(sentences))
            scored_sentences.append((sentence, score, i))

        # Select top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = scored_sentences[:num_sentences]

        # Sort by original position
        top_sentences.sort(key=lambda x: x[2])

        # Reconstruct summary
        summary = '. '.join([s[0] for s in top_sentences]) + '.'

        return summary

    def _score_sentence(self, sentence: str, position: int, total: int) -> float:
        """Score a sentence for extraction."""
        score = 0.0

        # Position (first and last sentences often important)
        if position == 0:
            score += 0.3
        elif position == total - 1:
            score += 0.2

        # Length (not too short, not too long)
        words = sentence.split()
        if 5 <= len(words) <= 20:
            score += 0.2

        # Contains important keywords
        important_words = ['important', 'critical', 'key', 'main', 'essential',
                          'always', 'never', 'must', 'should', 'prefer']

        if any(word in sentence.lower() for word in important_words):
            score += 0.3

        # Has numbers or specific details
        if any(char.isdigit() for char in sentence):
            score += 0.2

        return score


# Usage
summarizer = ExtractiveSummarizer()

long_content = """
The user has a strong preference for Python in data science contexts.
They mentioned having 5 years of experience with the language.
They particularly enjoy using pandas and numpy for data manipulation.
Recently they've been exploring machine learning with scikit-learn.
They asked about the weather but didn't seem very interested.
Their current project involves building a recommendation system.
"""

summary = summarizer.summarize(long_content, num_sentences=2)
print(f"\nExtracted Summary:\n{summary}")
```

### Abstractive Summarizer

```python
class AbstractiveSummarizer:
    """Generate abstractive summaries."""

    def summarize(self, content: str, max_length: int = 100) -> str:
        """
        Abstractive summarization (simplified).

        In practice, would use LLM for true abstractive summarization.
        This version does template-based abstraction.

        Args:
            content: Text to summarize
            max_length: Maximum length of summary

        Returns:
            Abstracted summary
        """
        # Extract key information
        entities = self._extract_entities(content)
        actions = self._extract_actions(content)
        topics = self._extract_topics(content)

        # Generate abstract summary
        if entities and actions:
            summary = f"{entities[0]} {actions[0]}"
            if topics:
                summary += f" regarding {topics[0]}"
        elif topics:
            summary = f"Discussion about {', '.join(topics)}"
        else:
            # Fallback to first sentence
            summary = content.split('.')[0]

        # Ensure length limit
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."

        return summary + "."

    def _extract_entities(self, content: str) -> List[str]:
        """Extract entity mentions (simplified)."""
        # Simple capitalized word detection
        words = content.split()
        entities = []

        for word in words:
            if word[0].isupper() and word.lower() not in ['the', 'a', 'an']:
                entities.append(word)

        # Add "user" if mentioned
        if 'user' in content.lower():
            entities.insert(0, "User")

        return entities[:3]

    def _extract_actions(self, content: str) -> List[str]:
        """Extract action verbs."""
        action_verbs = ['prefer', 'like', 'use', 'mention', 'discuss',
                       'ask', 'learn', 'build', 'create', 'explore']

        content_lower = content.lower()
        found_actions = []

        for verb in action_verbs:
            if verb in content_lower:
                found_actions.append(f"{verb}s" if not verb.endswith('s') else verb)

        return found_actions[:2]

    def _extract_topics(self, content: str) -> List[str]:
        """Extract main topics."""
        # Simple noun phrase extraction (would use NLP in practice)
        topic_keywords = ['python', 'javascript', 'data science', 'machine learning',
                         'programming', 'code', 'project', 'system']

        content_lower = content.lower()
        found_topics = []

        for topic in topic_keywords:
            if topic in content_lower:
                found_topics.append(topic)

        return found_topics[:2]


# Usage
abstractive = AbstractiveSummarizer()
abstract_summary = abstractive.summarize(long_content, max_length=80)
print(f"\nAbstract Summary:\n{abstract_summary}")
```

## Importance Scoring

Determining what matters.

### Importance Scorer

```python
class ImportanceScorer:
    """Score memory importance using multiple signals."""

    def __init__(self):
        self.signal_weights = {
            'content_markers': 0.3,
            'user_signals': 0.3,
            'context': 0.2,
            'metadata': 0.2
        }

    def score(self, candidate: ConsolidationCandidate,
             context: Dict[str, Any] = None) -> float:
        """
        Score importance of a memory candidate.

        Args:
            candidate: Memory to score
            context: Additional context information

        Returns:
            Importance score (0-1)
        """
        context = context or {}

        score = 0.0

        # Content markers
        content_score = self._score_content_markers(candidate.content)
        score += content_score * self.signal_weights['content_markers']

        # User signals
        user_score = self._score_user_signals(candidate.content, candidate.metadata)
        score += user_score * self.signal_weights['user_signals']

        # Context relevance
        context_score = self._score_context(candidate, context)
        score += context_score * self.signal_weights['context']

        # Metadata signals
        metadata_score = self._score_metadata(candidate.metadata)
        score += metadata_score * self.signal_weights['metadata']

        return min(score, 1.0)

    def _score_content_markers(self, content: str) -> float:
        """Score based on content markers."""
        content_lower = content.lower()
        score = 0.0

        # High importance markers
        high_markers = ['critical', 'important', 'essential', 'always', 'never',
                       'must', 'required', 'forbidden']

        if any(marker in content_lower for marker in high_markers):
            score += 0.5

        # Medium importance markers
        medium_markers = ['should', 'prefer', 'recommend', 'suggest', 'like']

        if any(marker in content_lower for marker in medium_markers):
            score += 0.3

        # Specific details (numbers, names, etc.)
        if any(char.isdigit() for char in content):
            score += 0.2

        return min(score, 1.0)

    def _score_user_signals(self, content: str, metadata: Dict) -> float:
        """Score based on user signals."""
        score = 0.0

        # Explicit user markers
        if 'remember' in content.lower():
            score += 0.4

        if 'don\'t forget' in content.lower() or 'do not forget' in content.lower():
            score += 0.5

        # User preferences or instructions
        if metadata.get('type') in ['preference', 'instruction', 'constraint']:
            score += 0.4

        # User corrections
        if 'actually' in content.lower() or 'correction' in content.lower():
            score += 0.3

        return min(score, 1.0)

    def _score_context(self, candidate: ConsolidationCandidate,
                      context: Dict) -> float:
        """Score based on context relevance."""
        score = 0.0

        # Mentioned in multiple turns
        if context.get('mention_count', 0) > 1:
            score += 0.4

        # Related to current task
        if context.get('task_relevant', False):
            score += 0.3

        # Part of longer conversation
        if context.get('conversation_depth', 0) > 3:
            score += 0.3

        return min(score, 1.0)

    def _score_metadata(self, metadata: Dict) -> float:
        """Score based on metadata."""
        score = 0.0

        # Has structured data
        if any(key in metadata for key in ['preference', 'constraint', 'fact']):
            score += 0.3

        # Has tags
        if 'tags' in metadata and len(metadata['tags']) > 0:
            score += 0.2

        # Has user ID or personalization
        if 'user_id' in metadata:
            score += 0.2

        # Explicitly marked important
        if metadata.get('important', False):
            score += 0.5

        return min(score, 1.0)


# Usage
scorer = ImportanceScorer()

test_candidate = ConsolidationCandidate(
    id=str(uuid.uuid4()),
    content="Remember: User never wants Windows solutions, always prefers Linux",
    source="context_window",
    timestamp=datetime.now(),
    metadata={'type': 'preference', 'important': True}
)

importance = scorer.score(
    test_candidate,
    context={'mention_count': 3, 'task_relevant': True}
)

print(f"\nImportance score: {importance:.3f}")
```

## Transition Management

Managing the transition from short to long-term memory.

### Transition Manager

```python
class TransitionManager:
    """Manage transition from short-term to long-term memory."""

    def __init__(self,
                 semantic_store,
                 episodic_store):
        self.semantic_store = semantic_store
        self.episodic_store = episodic_store
        self.transition_log = []

    def transition(self,
                  candidate: ConsolidationCandidate,
                  decision: ConsolidationType) -> bool:
        """
        Execute transition to long-term memory.

        Args:
            candidate: Memory to transition
            decision: Type of consolidation

        Returns:
            Success status
        """
        try:
            if decision == ConsolidationType.SEMANTIC:
                self._store_semantic(candidate)
            elif decision == ConsolidationType.EPISODIC:
                self._store_episodic(candidate)
            elif decision == ConsolidationType.SUMMARIZE:
                self._store_compressed(candidate)

            # Log transition
            self.transition_log.append({
                'candidate_id': candidate.id,
                'decision': decision,
                'timestamp': datetime.now(),
                'success': True
            })

            return True

        except Exception as e:
            print(f"Transition error: {e}")
            self.transition_log.append({
                'candidate_id': candidate.id,
                'decision': decision,
                'timestamp': datetime.now(),
                'success': False,
                'error': str(e)
            })
            return False

    def _store_semantic(self, candidate: ConsolidationCandidate):
        """Store as semantic memory (fact)."""
        # Extract facts from content
        facts = self._extract_facts(candidate.content)

        for fact in facts:
            # Store fact (using semantic memory store interface)
            # self.semantic_store.add_fact(fact)
            pass

    def _store_episodic(self, candidate: ConsolidationCandidate):
        """Store as episodic memory (experience)."""
        # Create episode
        episode = {
            'content': candidate.content,
            'timestamp': candidate.timestamp,
            'metadata': candidate.metadata,
            'importance': candidate.importance
        }

        # Store episode
        # self.episodic_store.add_episode(episode)
        pass

    def _store_compressed(self, candidate: ConsolidationCandidate):
        """Store in compressed form."""
        # Summarize
        summarizer = ExtractiveSummarizer()
        summary = summarizer.summarize(candidate.content, num_sentences=1)

        # Store summary
        # Would store in appropriate location
        pass

    def _extract_facts(self, content: str) -> List[str]:
        """Extract factual statements."""
        # Simple sentence splitting (would use NLP in practice)
        sentences = [s.strip() + '.' for s in content.split('.') if s.strip()]

        # Filter for factual-looking sentences
        facts = []
        for sentence in sentences:
            if self._looks_factual(sentence):
                facts.append(sentence)

        return facts

    def _looks_factual(self, sentence: str) -> bool:
        """Check if sentence looks factual."""
        factual_verbs = ['is', 'are', 'was', 'were', 'prefer', 'like',
                        'always', 'never', 'can', 'cannot']

        sentence_lower = sentence.lower()
        return any(verb in sentence_lower for verb in factual_verbs)


# Usage (with mock stores)
class MockStore:
    def add_fact(self, fact): pass
    def add_episode(self, episode): pass

transition_mgr = TransitionManager(
    semantic_store=MockStore(),
    episodic_store=MockStore()
)

success = transition_mgr.transition(
    test_candidate,
    ConsolidationType.SEMANTIC
)

print(f"\nTransition successful: {success}")
```

## Consolidation Policies

Different policies for different needs.

### Policy-Based Consolidator

```python
from abc import ABC, abstractmethod

class ConsolidationPolicy(ABC):
    """Base class for consolidation policies."""

    @abstractmethod
    def should_consolidate(self, candidate: ConsolidationCandidate) -> bool:
        """Check if candidate should be consolidated."""
        pass

    @abstractmethod
    def get_consolidation_type(self,
                              candidate: ConsolidationCandidate) -> ConsolidationType:
        """Determine type of consolidation."""
        pass

class AggressivePolicy(ConsolidationPolicy):
    """Consolidate most memories."""

    def should_consolidate(self, candidate: ConsolidationCandidate) -> bool:
        """Consolidate if importance > 0.2."""
        return candidate.importance > 0.2

    def get_consolidation_type(self,
                              candidate: ConsolidationCandidate) -> ConsolidationType:
        """Always store fully."""
        if candidate.importance > 0.5:
            return ConsolidationType.EPISODIC
        else:
            return ConsolidationType.SUMMARIZE

class ConservativePolicy(ConsolidationPolicy):
    """Only consolidate very important memories."""

    def should_consolidate(self, candidate: ConsolidationCandidate) -> bool:
        """Consolidate only if importance > 0.7."""
        return candidate.importance > 0.7

    def get_consolidation_type(self,
                              candidate: ConsolidationCandidate) -> ConsolidationType:
        """Always store as episode (high quality)."""
        return ConsolidationType.EPISODIC

class BalancedPolicy(ConsolidationPolicy):
    """Balance storage with quality."""

    def should_consolidate(self, candidate: ConsolidationCandidate) -> bool:
        """Consolidate if importance > 0.4 or novelty > 0.6."""
        return candidate.importance > 0.4 or candidate.novelty > 0.6

    def get_consolidation_type(self,
                              candidate: ConsolidationCandidate) -> ConsolidationType:
        """Vary based on importance."""
        if candidate.importance > 0.7:
            return ConsolidationType.EPISODIC
        elif candidate.importance > 0.4:
            return ConsolidationType.SUMMARIZE
        else:
            return ConsolidationType.SEMANTIC


# Usage
policies = {
    'aggressive': AggressivePolicy(),
    'conservative': ConservativePolicy(),
    'balanced': BalancedPolicy()
}

active_policy = policies['balanced']

for candidate in candidates:
    should = active_policy.should_consolidate(candidate)
    if should:
        cons_type = active_policy.get_consolidation_type(candidate)
        print(f"Policy says: consolidate as {cons_type.value}")
```

## Incremental Consolidation

Consolidating continuously rather than in batches.

### Incremental Consolidator

```python
class IncrementalConsolidator:
    """Consolidate memories incrementally."""

    def __init__(self, consolidation_mgr: ConsolidationManager):
        self.consolidation_mgr = consolidation_mgr
        self.pending_candidates = []
        self.batch_size = 5

    def add_candidate(self, candidate: ConsolidationCandidate):
        """Add candidate for incremental consolidation."""
        self.pending_candidates.append(candidate)

        # If batch full, consolidate
        if len(self.pending_candidates) >= self.batch_size:
            self.consolidate_batch()

    def consolidate_batch(self):
        """Consolidate current batch."""
        if not self.pending_candidates:
            return

        # Consolidate
        results = self.consolidation_mgr.consolidate(self.pending_candidates)

        print(f"Consolidated batch of {len(self.pending_candidates)} memories")
        print(f"  Kept: {sum(1 for r in results if r.decision != ConsolidationType.DISCARD)}")
        print(f"  Discarded: {sum(1 for r in results if r.decision == ConsolidationType.DISCARD)}")

        # Clear batch
        self.pending_candidates = []

    def flush(self):
        """Consolidate remaining candidates."""
        if self.pending_candidates:
            self.consolidate_batch()


# Usage
incremental = IncrementalConsolidator(consolidation_mgr)

# Add candidates one by one
for candidate in candidates:
    incremental.add_candidate(candidate)

# Flush remaining
incremental.flush()
```

## Compression Techniques

Advanced compression strategies.

### Hierarchical Compression

```python
class HierarchicalCompressor:
    """Compress memories in hierarchical layers."""

    def __init__(self):
        self.layers = {
            'raw': [],       # Original memories
            'hourly': [],    # Hour summaries
            'daily': [],     # Day summaries
            'weekly': []     # Week summaries
        }

    def add_memory(self, content: str, timestamp: datetime):
        """Add raw memory."""
        self.layers['raw'].append({
            'content': content,
            'timestamp': timestamp
        })

    def compress_to_hourly(self):
        """Compress raw memories into hourly summaries."""
        # Group by hour
        hour_groups = {}

        for memory in self.layers['raw']:
            hour_key = memory['timestamp'].strftime('%Y-%m-%d-%H')
            if hour_key not in hour_groups:
                hour_groups[hour_key] = []
            hour_groups[hour_key].append(memory)

        # Summarize each hour
        for hour, memories in hour_groups.items():
            combined = ' '.join([m['content'] for m in memories])

            summarizer = ExtractiveSummarizer()
            summary = summarizer.summarize(combined, num_sentences=3)

            self.layers['hourly'].append({
                'period': hour,
                'summary': summary,
                'memory_count': len(memories)
            })

    def compress_to_daily(self):
        """Compress hourly into daily summaries."""
        # Group hourly summaries by day
        day_groups = {}

        for hourly in self.layers['hourly']:
            day_key = hourly['period'][:10]  # YYYY-MM-DD
            if day_key not in day_groups:
                day_groups[day_key] = []
            day_groups[day_key].append(hourly)

        # Summarize each day
        for day, hourlies in day_groups.items():
            combined = ' '.join([h['summary'] for h in hourlies])

            summarizer = ExtractiveSummarizer()
            summary = summarizer.summarize(combined, num_sentences=2)

            self.layers['daily'].append({
                'date': day,
                'summary': summary,
                'hour_count': len(hourlies)
            })

    def get_summary(self, timeframe: str = 'daily') -> str:
        """Get summary for timeframe."""
        if timeframe not in self.layers:
            return ""

        summaries = self.layers[timeframe]
        if not summaries:
            return ""

        return '\n'.join([
            f"{s.get('period', s.get('date', 'summary'))}: {s.get('summary', s.get('content', ''))}"
            for s in summaries[-5:]  # Last 5
        ])


# Usage
compressor = HierarchicalCompressor()

# Add memories
from datetime import timedelta

for i in range(10):
    compressor.add_memory(
        f"Memory {i}: User interaction about topic {i}",
        datetime.now() - timedelta(hours=i)
    )

# Compress
compressor.compress_to_hourly()
compressor.compress_to_daily()

print("\nHierarchical compression:")
print(compressor.get_summary('hourly'))
```

## Semantic Extraction

Extracting structured knowledge from unstructured memories.

### Semantic Extractor

```python
class SemanticExtractor:
    """Extract semantic knowledge from memories."""

    def extract_facts(self, content: str) -> List[Dict]:
        """Extract factual statements."""
        sentences = [s.strip() for s in content.split('.') if s.strip()]

        facts = []

        for sentence in sentences:
            # Check if factual
            if not self._is_factual(sentence):
                continue

            # Extract fact structure
            fact = self._parse_fact(sentence)
            if fact:
                facts.append(fact)

        return facts

    def extract_preferences(self, content: str) -> List[Dict]:
        """Extract user preferences."""
        preferences = []

        content_lower = content.lower()

        # Pattern: "prefer X over Y"
        if 'prefer' in content_lower:
            # Simple extraction (would use NLP in practice)
            preferences.append({
                'type': 'preference',
                'content': content,
                'strength': 'strong' if 'strongly' in content_lower else 'moderate'
            })

        # Pattern: "like/dislike"
        if 'like' in content_lower or 'love' in content_lower:
            preferences.append({
                'type': 'positive_preference',
                'content': content,
                'strength': 'strong' if 'love' in content_lower else 'moderate'
            })

        if 'dislike' in content_lower or 'hate' in content_lower:
            preferences.append({
                'type': 'negative_preference',
                'content': content,
                'strength': 'strong' if 'hate' in content_lower else 'moderate'
            })

        return preferences

    def extract_entities(self, content: str) -> List[Dict]:
        """Extract entity mentions."""
        # Simple named entity recognition
        words = content.split()

        entities = []

        for i, word in enumerate(words):
            # Capitalized words
            if word[0].isupper() and word not in ['I', 'The', 'A', 'An']:
                entities.append({
                    'text': word,
                    'type': 'entity',
                    'context': ' '.join(words[max(0, i-2):min(len(words), i+3)])
                })

        return entities

    def _is_factual(self, sentence: str) -> bool:
        """Check if sentence is factual."""
        factual_patterns = ['is', 'are', 'was', 'were', 'can', 'cannot',
                           'always', 'never', 'must', 'defined as']

        return any(pattern in sentence.lower() for pattern in factual_patterns)

    def _parse_fact(self, sentence: str) -> Optional[Dict]:
        """Parse fact structure (simplified)."""
        # Pattern: "X is Y"
        if ' is ' in sentence:
            parts = sentence.split(' is ', 1)
            return {
                'subject': parts[0].strip(),
                'predicate': 'is',
                'object': parts[1].strip(),
                'confidence': 0.8
            }

        # Pattern: "X can Y"
        if ' can ' in sentence:
            parts = sentence.split(' can ', 1)
            return {
                'subject': parts[0].strip(),
                'predicate': 'can',
                'object': parts[1].strip(),
                'confidence': 0.7
            }

        return None


# Usage
extractor = SemanticExtractor()

content = "Python is a high-level programming language. User prefers Python over JavaScript for data tasks."

facts = extractor.extract_facts(content)
preferences = extractor.extract_preferences(content)
entities = extractor.extract_entities(content)

print("\nExtracted semantic knowledge:")
print(f"Facts: {len(facts)}")
for fact in facts:
    print(f"  {fact}")
print(f"Preferences: {len(preferences)}")
for pref in preferences:
    print(f"  {pref['type']}")
print(f"Entities: {len(entities)}")
for entity in entities:
    print(f"  {entity['text']}")
```

## Memory Lifecycle

Managing the complete lifecycle of memories.

### Lifecycle Manager

```python
class MemoryLifecycleManager:
    """Manage complete memory lifecycle."""

    def __init__(self):
        self.stages = {
            'working': [],      # Active working memory
            'consolidating': [], # Being consolidated
            'archived': [],     # Long-term storage
            'forgotten': []     # Expired/removed
        }
        self.max_age_days = 90

    def add_working_memory(self, content: str) -> str:
        """Add to working memory."""
        memory_id = str(uuid.uuid4())

        self.stages['working'].append({
            'id': memory_id,
            'content': content,
            'stage': 'working',
            'created': datetime.now(),
            'accessed': datetime.now(),
            'access_count': 0
        })

        return memory_id

    def move_to_consolidation(self, memory_ids: List[str]):
        """Move memories to consolidation stage."""
        for memory_id in memory_ids:
            memory = self._find_memory(memory_id, 'working')
            if memory:
                memory['stage'] = 'consolidating'
                memory['consolidation_started'] = datetime.now()

                self.stages['working'].remove(memory)
                self.stages['consolidating'].append(memory)

    def complete_consolidation(self, memory_id: str, archived: bool = True):
        """Complete consolidation."""
        memory = self._find_memory(memory_id, 'consolidating')

        if not memory:
            return

        if archived:
            memory['stage'] = 'archived'
            memory['archived_at'] = datetime.now()

            self.stages['consolidating'].remove(memory)
            self.stages['archived'].append(memory)
        else:
            # Discard
            memory['stage'] = 'forgotten'
            memory['forgotten_at'] = datetime.now()
            memory['reason'] = 'low_importance'

            self.stages['consolidating'].remove(memory)
            self.stages['forgotten'].append(memory)

    def expire_old_memories(self):
        """Move old memories to forgotten."""
        cutoff = datetime.now() - timedelta(days=self.max_age_days)

        for stage in ['working', 'consolidating', 'archived']:
            expired = []

            for memory in self.stages[stage]:
                if memory['created'] < cutoff:
                    expired.append(memory)

            for memory in expired:
                memory['stage'] = 'forgotten'
                memory['forgotten_at'] = datetime.now()
                memory['reason'] = 'expired'

                self.stages[stage].remove(memory)
                self.stages['forgotten'].append(memory)

    def get_statistics(self) -> Dict:
        """Get lifecycle statistics."""
        return {
            stage: len(memories)
            for stage, memories in self.stages.items()
        }

    def _find_memory(self, memory_id: str, stage: str) -> Optional[Dict]:
        """Find memory in a stage."""
        for memory in self.stages[stage]:
            if memory['id'] == memory_id:
                return memory
        return None


# Usage
lifecycle = MemoryLifecycleManager()

# Add working memories
mem1 = lifecycle.add_working_memory("User prefers Python")
mem2 = lifecycle.add_working_memory("Weather discussion")

# Move to consolidation
lifecycle.move_to_consolidation([mem1, mem2])

# Complete consolidation
lifecycle.complete_consolidation(mem1, archived=True)
lifecycle.complete_consolidation(mem2, archived=False)

stats = lifecycle.get_statistics()
print(f"\nMemory Lifecycle Statistics:")
for stage, count in stats.items():
    print(f"  {stage}: {count}")
```

## Summary

Memory consolidation transforms transient information into lasting knowledge:

**Key Decisions**:

- **When**: Time-based, interaction-based, or capacity-based triggers
- **What**: Importance and novelty scoring for selection
- **How**: Summarization strategies (extractive vs abstractive)
- **Where**: Episodic, semantic, or compressed storage

**Consolidation Strategies**:

- **Aggressive**: Store most memories, prioritize retention
- **Conservative**: Only store highly important memories
- **Balanced**: Optimize storage vs quality trade-off
- **Incremental**: Continuous consolidation vs batch processing

**Key Techniques**:

- **Importance scoring**: Multiple signals (content, user, context)
- **Summarization**: Extractive, abstractive, hierarchical
- **Semantic extraction**: Facts, preferences, entities
- **Lifecycle management**: Working → Consolidating → Archived → Forgotten

**Trade-offs**:

- Storage cost vs information retention
- Compression vs fidelity
- Immediate vs delayed consolidation
- Selectivity vs completeness

## Next Steps

- **[Short-Term Memory](short-term-memory.md)**: Source of consolidation candidates
- **[Long-Term Memory](long-term-memory.md)**: Destination for consolidated memories
- **[Episodic Memory](episodic-memory.md)**: Storing consolidated experiences
- **[Semantic Memory](semantic-memory.md)**: Extracting and storing facts
- **[Memory Retrieval](memory-retrieval.md)**: Accessing consolidated memories
