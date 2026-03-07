# Short-Term Memory

## Table of Contents

- [Introduction](#introduction)
- [Context Window Fundamentals](#context-window-fundamentals)
- [Working Memory](#working-memory)
- [Conversation History Management](#conversation-history-management)
- [Token Budget Management](#token-budget-management)
- [Context Window Strategies](#context-window-strategies)
- [Recency Effects](#recency-effects)
- [Relevance Filtering](#relevance-filtering)
- [Context Compression](#context-compression)
- [Multi-Turn Conversations](#multi-turn-conversations)
- [Context Overflow Handling](#context-overflow-handling)
- [Performance Optimization](#performance-optimization)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Short-term memory (STM) is the agent's immediate workspace - the information actively in use right now. Unlike long-term memory that persists across sessions, short-term memory is transient, bounded, and immediately accessible.

> "Short-term memory is to an agent what RAM is to a computer - fast, limited, and essential for current operations."

In language model-based agents, short-term memory is primarily implemented through the **context window** - the input text the model sees when generating a response. Managing this limited resource effectively is crucial for agent performance.

### Key Characteristics

**Immediate Access**: Information in short-term memory is directly available to the model without retrieval operations.

**Limited Capacity**: Bounded by the model's context window (typically 4K-200K tokens).

**High Volatility**: Lost when the conversation ends or context resets.

**Recency Bias**: More recent information is naturally prioritized.

### Why Short-Term Memory Matters

**Example: Without Proper STM Management**

```python
# Context exceeds window - model loses important information
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    # ... 100 previous messages ...
    {"role": "user", "content": "Remember I said I'm allergic to peanuts?"},
    # ... 50 more messages ...
    {"role": "user", "content": "Suggest a snack for me"}
]

# Model might suggest peanuts because allergy info was pushed out of context
```

**Example: With Proper STM Management**

```python
# Keep critical information in context
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "system", "content": "User info: Allergic to peanuts"},  # Important fact
    # ... recent conversation ...
    {"role": "user", "content": "Suggest a snack for me"}
]

# Model correctly avoids peanuts
```

## Context Window Fundamentals

The context window is the foundation of short-term memory in LLM-based agents.

### Context Window Sizes

```
Model           Context Window    Use Case
─────────────────────────────────────────────────────────
GPT-3.5         4,096 tokens      Short conversations
GPT-4           8,192 tokens      Standard tasks
GPT-4-32K       32,768 tokens     Long documents
Claude 3        200,000 tokens    Entire codebases
Gemini 1.5      1,000,000 tokens  Massive context
```

### Token Counting

Understanding token usage is essential:

```python
import tiktoken

class TokenCounter:
    """Count tokens for different models."""
    
    def __init__(self, model: str = "gpt-4"):
        self.encoding = tiktoken.encoding_for_model(model)
    
    def count(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def count_messages(self, messages: list[dict]) -> int:
        """Count tokens in message list."""
        total = 0
        
        for message in messages:
            # Role overhead (typically 3-4 tokens)
            total += 4
            
            # Content tokens
            total += self.count(message.get("content", ""))
            
            # Name tokens if present
            if "name" in message:
                total += self.count(message["name"])
        
        # Format overhead
        total += 2
        
        return total
    
    def estimate_completion_tokens(self, prompt_tokens: int,
                                   max_tokens: int = 4096) -> int:
        """Estimate available tokens for completion."""
        return max_tokens - prompt_tokens


# Usage
counter = TokenCounter()

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"}
]

token_count = counter.count_messages(messages)
print(f"Total tokens: {token_count}")
print(f"Available for response: {counter.estimate_completion_tokens(token_count)}")
```

### Basic Context Window Management

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Message:
    """A message in the conversation."""
    role: str  # 'system', 'user', 'assistant'
    content: str
    name: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to API format."""
        msg = {"role": self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        return msg


class ContextWindow:
    """Manage the context window."""
    
    def __init__(self, max_tokens: int = 4096, 
                 model: str = "gpt-4"):
        self.max_tokens = max_tokens
        self.messages: List[Message] = []
        self.token_counter = TokenCounter(model)
        self.system_message: Optional[Message] = None
    
    def set_system_message(self, content: str):
        """Set the system message (always kept)."""
        self.system_message = Message(role="system", content=content)
    
    def add_message(self, role: str, content: str, name: Optional[str] = None):
        """Add a message to the context window."""
        message = Message(role=role, content=content, name=name)
        self.messages.append(message)
        self._ensure_within_limit()
    
    def _ensure_within_limit(self):
        """Remove old messages if over token limit."""
        while self._count_tokens() > self.max_tokens and len(self.messages) > 1:
            # Remove oldest non-system message
            self.messages.pop(0)
    
    def _count_tokens(self) -> int:
        """Count total tokens in context."""
        all_messages = []
        
        if self.system_message:
            all_messages.append(self.system_message.to_dict())
        
        all_messages.extend([m.to_dict() for m in self.messages])
        
        return self.token_counter.count_messages(all_messages)
    
    def get_messages(self) -> list[dict]:
        """Get all messages for API call."""
        messages = []
        
        if self.system_message:
            messages.append(self.system_message.to_dict())
        
        messages.extend([m.to_dict() for m in self.messages])
        
        return messages
    
    def get_token_count(self) -> int:
        """Get current token count."""
        return self._count_tokens()
    
    def get_available_tokens(self) -> int:
        """Get available tokens for completion."""
        return self.max_tokens - self._count_tokens()
    
    def clear(self):
        """Clear conversation history (keep system message)."""
        self.messages.clear()
```

## Working Memory

Working memory stores temporary variables and state during task execution.

### Working Memory Implementation

```python
from typing import Any, Dict
from datetime import datetime, timedelta

class WorkingMemory:
    """Temporary storage for task execution."""
    
    def __init__(self):
        self.variables: Dict[str, Any] = {}
        self.timestamps: Dict[str, datetime] = {}
        self.task_state: Dict[str, Any] = {}
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Store a variable in working memory.
        
        Args:
            key: Variable name
            value: Variable value
            ttl: Time to live in seconds (None = no expiration)
        """
        self.variables[key] = value
        self.timestamps[key] = datetime.now()
        
        if ttl:
            # Would need background cleanup in production
            pass
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a variable from working memory."""
        return self.variables.get(key, default)
    
    def has(self, key: str) -> bool:
        """Check if variable exists."""
        return key in self.variables
    
    def delete(self, key: str):
        """Remove a variable."""
        self.variables.pop(key, None)
        self.timestamps.pop(key, None)
    
    def clear(self):
        """Clear all variables."""
        self.variables.clear()
        self.timestamps.clear()
    
    def update_task_state(self, **kwargs):
        """Update task-specific state."""
        self.task_state.update(kwargs)
    
    def get_task_state(self) -> Dict[str, Any]:
        """Get current task state."""
        return self.task_state.copy()
    
    def clear_task_state(self):
        """Clear task state."""
        self.task_state.clear()
    
    def to_context_string(self) -> str:
        """Convert working memory to context string."""
        if not self.variables and not self.task_state:
            return ""
        
        parts = []
        
        if self.variables:
            parts.append("Variables:")
            for key, value in self.variables.items():
                parts.append(f"  {key} = {value}")
        
        if self.task_state:
            parts.append("Task State:")
            for key, value in self.task_state.items():
                parts.append(f"  {key}: {value}")
        
        return "\n".join(parts)
    
    def get_summary(self) -> str:
        """Get concise summary of working memory."""
        var_count = len(self.variables)
        task_keys = list(self.task_state.keys())
        
        summary = f"{var_count} variables"
        if task_keys:
            summary += f", task state: {', '.join(task_keys)}"
        
        return summary


# Usage example
working_mem = WorkingMemory()

# Store intermediate results
working_mem.set("user_query", "Find papers on RAG")
working_mem.set("search_results", ["paper1", "paper2", "paper3"])
working_mem.update_task_state(
    step="searching",
    progress=0.3,
    next_action="read_papers"
)

# Get context string
print(working_mem.to_context_string())
# Output:
# Variables:
#   user_query = Find papers on RAG
#   search_results = ['paper1', 'paper2', 'paper3']
# Task State:
#   step: searching
#   progress: 0.3
#   next_action: read_papers
```

### Working Memory in Agent Loop

```python
class AgentWithWorkingMemory:
    """Agent that uses working memory during execution."""
    
    def __init__(self, max_context_tokens: int = 4096):
        self.context = ContextWindow(max_context_tokens)
        self.working_memory = WorkingMemory()
        self.context.set_system_message(
            "You are a helpful assistant. "
            "Use working memory to track task progress."
        )
    
    def execute_task(self, task: str) -> str:
        """Execute task using working memory."""
        # Initialize task state
        self.working_memory.clear_task_state()
        self.working_memory.update_task_state(
            task=task,
            step="initializing",
            attempts=0
        )
        
        # Add task to context with working memory state
        self._add_to_context(
            "user",
            f"Task: {task}\n\nWorking Memory:\n"
            f"{self.working_memory.to_context_string()}"
        )
        
        # Execute task steps
        result = self._execute_steps()
        
        # Clear task state when done
        self.working_memory.clear_task_state()
        
        return result
    
    def _execute_steps(self) -> str:
        """Execute task steps (simplified)."""
        # In practice, this would be a loop with tool calls
        
        # Update working memory as task progresses
        self.working_memory.update_task_state(step="executing")
        self.working_memory.set("intermediate_result", "...")
        
        self.working_memory.update_task_state(step="finalizing")
        
        return "Task completed"
    
    def _add_to_context(self, role: str, content: str):
        """Add message to context."""
        self.context.add_message(role, content)
```

## Conversation History Management

Managing conversation history effectively is critical for maintaining coherent interactions.

### History Storage Strategies

**Strategy 1: Keep Recent N Turns**

```python
class RecentTurnStrategy:
    """Keep only the N most recent conversation turns."""
    
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
    
    def filter_messages(self, messages: List[Message]) -> List[Message]:
        """Keep only recent turns."""
        # Count turns (user-assistant pairs)
        turns = []
        current_turn = []
        
        for msg in messages:
            if msg.role == "system":
                continue
            
            current_turn.append(msg)
            
            if msg.role == "assistant":
                turns.append(current_turn)
                current_turn = []
        
        # Add incomplete turn
        if current_turn:
            turns.append(current_turn)
        
        # Keep recent turns
        recent_turns = turns[-self.max_turns:]
        
        # Flatten
        filtered = []
        for turn in recent_turns:
            filtered.extend(turn)
        
        return filtered
```

**Strategy 2: Token-Based Pruning**

```python
class TokenBasedStrategy:
    """Keep messages that fit in token budget."""
    
    def __init__(self, max_tokens: int, token_counter: TokenCounter):
        self.max_tokens = max_tokens
        self.token_counter = token_counter
    
    def filter_messages(self, messages: List[Message]) -> List[Message]:
        """Keep messages that fit in token budget."""
        # Always keep system messages
        system_messages = [m for m in messages if m.role == "system"]
        other_messages = [m for m in messages if m.role != "system"]
        
        # Calculate system message tokens
        system_tokens = sum(
            self.token_counter.count(m.content) + 4  # Role overhead
            for m in system_messages
        )
        
        available = self.max_tokens - system_tokens
        
        # Add messages from most recent backwards
        kept = []
        tokens_used = 0
        
        for message in reversed(other_messages):
            msg_tokens = self.token_counter.count(message.content) + 4
            
            if tokens_used + msg_tokens > available:
                break
            
            kept.insert(0, message)
            tokens_used += msg_tokens
        
        return system_messages + kept
```

**Strategy 3: Importance-Based Selection**

```python
class ImportanceBasedStrategy:
    """Keep most important messages."""
    
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
    
    def score_importance(self, message: Message, position: int, 
                        total: int) -> float:
        """Score message importance."""
        score = 0.0
        
        # Recency score (more recent = higher)
        recency = position / total
        score += recency * 0.5
        
        # Length score (longer = more important)
        length_score = min(1.0, len(message.content) / 500)
        score += length_score * 0.2
        
        # Keyword score
        important_keywords = [
            'important', 'remember', 'always', 'never',
            'critical', 'essential', 'must', 'should not'
        ]
        
        content_lower = message.content.lower()
        keyword_matches = sum(
            1 for keyword in important_keywords
            if keyword in content_lower
        )
        score += min(1.0, keyword_matches * 0.2) * 0.3
        
        return score
    
    def filter_messages(self, messages: List[Message]) -> List[Message]:
        """Keep most important messages."""
        # Always keep system messages
        system_messages = [m for m in messages if m.role == "system"]
        other_messages = [m for m in messages if m.role != "system"]
        
        # Score all messages
        scored = []
        for i, msg in enumerate(other_messages):
            score = self.score_importance(msg, i, len(other_messages))
            scored.append((score, i, msg))
        
        # Sort by score
        scored.sort(reverse=True)
        
        # Keep top N, but maintain chronological order
        kept_indices = sorted([i for _, i, _ in scored[:self.max_messages]])
        kept_messages = [other_messages[i] for i in kept_indices]
        
        return system_messages + kept_messages
```

### Combining Strategies

```python
class HybridHistoryManager:
    """Combine multiple history management strategies."""
    
    def __init__(self, max_tokens: int, max_turns: int = 10):
        self.token_counter = TokenCounter()
        
        # Multiple strategies
        self.strategies = [
            RecentTurnStrategy(max_turns),
            TokenBasedStrategy(max_tokens, self.token_counter),
            ImportanceBasedStrategy(max_messages=15)
        ]
    
    def filter_messages(self, messages: List[Message]) -> List[Message]:
        """Apply all strategies and keep intersection."""
        filtered = messages
        
        for strategy in self.strategies:
            filtered = strategy.filter_messages(filtered)
        
        return filtered
```

## Token Budget Management

Allocating tokens across different context components.

### Token Budget Allocator

```python
from typing import Dict

class TokenBudgetAllocator:
    """Manage token budget across context components."""
    
    def __init__(self, total_budget: int):
        self.total_budget = total_budget
        
        # Default allocations (percentages)
        self.allocations = {
            "system_prompt": 0.15,      # 15% for system instructions
            "working_memory": 0.10,     # 10% for working variables
            "conversation_history": 0.40, # 40% for history
            "retrieved_context": 0.20,  # 20% for retrieved memories
            "current_input": 0.10,      # 10% for current query
            "output_buffer": 0.05       # 5% buffer for output
        }
    
    def set_allocation(self, component: str, percentage: float):
        """Set allocation percentage for a component."""
        if percentage < 0 or percentage > 1:
            raise ValueError("Percentage must be between 0 and 1")
        
        self.allocations[component] = percentage
    
    def get_allocation(self, component: str) -> int:
        """Get token allocation for a component."""
        percentage = self.allocations.get(component, 0)
        return int(self.total_budget * percentage)
    
    def get_budget_summary(self) -> Dict[str, int]:
        """Get token allocation for all components."""
        return {
            component: self.get_allocation(component)
            for component in self.allocations
        }
    
    def adjust_allocations(self, needs: Dict[str, float]):
        """
        Dynamically adjust allocations based on needs.
        
        Args:
            needs: Dictionary of component needs (0-1 scale)
        """
        total_need = sum(needs.values())
        
        if total_need == 0:
            return
        
        # Redistribute based on relative needs
        for component, need in needs.items():
            self.allocations[component] = need / total_need
```

### Adaptive Budget Management

```python
class AdaptiveBudgetManager:
    """Dynamically adjust token budget based on context."""
    
    def __init__(self, total_budget: int):
        self.allocator = TokenBudgetAllocator(total_budget)
        self.token_counter = TokenCounter()
    
    def build_context(self, components: Dict[str, str]) -> List[Message]:
        """Build context with adaptive budget management."""
        # Assess needs
        needs = self._assess_needs(components)
        
        # Adjust allocations
        self.allocator.adjust_allocations(needs)
        
        # Build context respecting allocations
        messages = []
        
        # System prompt (mandatory)
        if "system_prompt" in components:
            system_budget = self.allocator.get_allocation("system_prompt")
            system_content = self._fit_to_budget(
                components["system_prompt"],
                system_budget
            )
            messages.append(Message("system", system_content))
        
        # Working memory
        if "working_memory" in components:
            wm_budget = self.allocator.get_allocation("working_memory")
            wm_content = self._fit_to_budget(
                components["working_memory"],
                wm_budget
            )
            if wm_content:
                messages.append(Message("system", f"Working Memory:\n{wm_content}"))
        
        # Retrieved context
        if "retrieved_context" in components:
            rc_budget = self.allocator.get_allocation("retrieved_context")
            rc_content = self._fit_to_budget(
                components["retrieved_context"],
                rc_budget
            )
            if rc_content:
                messages.append(Message("system", f"Relevant Context:\n{rc_content}"))
        
        # Conversation history
        if "conversation_history" in components:
            history = components["conversation_history"]  # List of messages
            history_budget = self.allocator.get_allocation("conversation_history")
            fitted_history = self._fit_history_to_budget(history, history_budget)
            messages.extend(fitted_history)
        
        # Current input
        if "current_input" in components:
            input_budget = self.allocator.get_allocation("current_input")
            input_content = self._fit_to_budget(
                components["current_input"],
                input_budget
            )
            messages.append(Message("user", input_content))
        
        return messages
    
    def _assess_needs(self, components: Dict[str, str]) -> Dict[str, float]:
        """Assess relative importance of each component."""
        needs = {}
        
        # System prompt is always important
        needs["system_prompt"] = 1.0
        
        # Working memory need based on size
        if "working_memory" in components:
            wm_size = len(components["working_memory"])
            needs["working_memory"] = min(1.0, wm_size / 500)
        
        # History need based on conversation length
        if "conversation_history" in components:
            history_length = len(components["conversation_history"])
            needs["conversation_history"] = min(1.0, history_length / 20)
        
        # Retrieved context need based on availability
        if "retrieved_context" in components:
            rc_size = len(components["retrieved_context"])
            needs["retrieved_context"] = min(1.0, rc_size / 1000)
        
        # Current input is always important
        needs["current_input"] = 1.0
        needs["output_buffer"] = 0.5
        
        return needs
    
    def _fit_to_budget(self, text: str, budget: int) -> str:
        """Truncate text to fit budget."""
        tokens = self.token_counter.count(text)
        
        if tokens <= budget:
            return text
        
        # Truncate
        words = text.split()
        while tokens > budget and words:
            words.pop()
            text = " ".join(words) + "..."
            tokens = self.token_counter.count(text)
        
        return text
    
    def _fit_history_to_budget(self, history: List[Message],
                               budget: int) -> List[Message]:
        """Fit conversation history to token budget."""
        fitted = []
        tokens_used = 0
        
        # Add from most recent backwards
        for message in reversed(history):
            msg_tokens = self.token_counter.count(message.content) + 4
            
            if tokens_used + msg_tokens > budget:
                break
            
            fitted.insert(0, message)
            tokens_used += msg_tokens
        
        return fitted
```

## Context Window Strategies

Different strategies for different use cases.

### Strategy 1: Sliding Window

```python
class SlidingWindowStrategy:
    """
    Keep a sliding window of recent messages.
    Simple and predictable.
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.messages: List[Message] = []
        self.system_message: Optional[Message] = None
    
    def add_message(self, message: Message):
        """Add message to window."""
        if message.role == "system":
            self.system_message = message
        else:
            self.messages.append(message)
            
            # Keep only recent messages
            if len(self.messages) > self.window_size:
                self.messages.pop(0)
    
    def get_context(self) -> List[Message]:
        """Get current context window."""
        context = []
        
        if self.system_message:
            context.append(self.system_message)
        
        context.extend(self.messages)
        
        return context
```

### Strategy 2: Summarization Window

```python
class SummarizationWindowStrategy:
    """
    Summarize old messages when context fills.
    Preserves information in compressed form.
    """
    
    def __init__(self, max_tokens: int, summarizer):
        self.max_tokens = max_tokens
        self.summarizer = summarizer
        self.messages: List[Message] = []
        self.summary: Optional[str] = None
        self.token_counter = TokenCounter()
    
    def add_message(self, message: Message):
        """Add message, summarizing old ones if needed."""
        self.messages.append(message)
        
        # Check if we need to summarize
        if self._count_tokens() > self.max_tokens:
            self._summarize_old_messages()
    
    def _summarize_old_messages(self):
        """Summarize older messages to save space."""
        # Keep recent messages, summarize old ones
        messages_to_keep = 5
        
        if len(self.messages) <= messages_to_keep:
            return
        
        # Messages to summarize
        to_summarize = self.messages[:-messages_to_keep]
        
        # Create summary
        conversation_text = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in to_summarize
        ])
        
        new_summary = self.summarizer.summarize(conversation_text)
        
        # Update summary
        if self.summary:
            # Combine with existing summary
            combined = f"{self.summary}\n\nRecent conversation:\n{new_summary}"
            self.summary = self.summarizer.summarize(combined)
        else:
            self.summary = new_summary
        
        # Keep only recent messages
        self.messages = self.messages[-messages_to_keep:]
    
    def get_context(self) -> List[Message]:
        """Get context with summary."""
        context = []
        
        # Add summary if exists
        if self.summary:
            context.append(Message(
                "system",
                f"Previous conversation summary:\n{self.summary}"
            ))
        
        # Add recent messages
        context.extend(self.messages)
        
        return context
    
    def _count_tokens(self) -> int:
        """Count total tokens."""
        total = 0
        
        if self.summary:
            total += self.token_counter.count(self.summary)
        
        for msg in self.messages:
            total += self.token_counter.count(msg.content) + 4
        
        return total
```

### Strategy 3: Hierarchical Context

```python
class HierarchicalContextStrategy:
    """
    Organize context in hierarchical layers.
    Different retention policies for different layers.
    """
    
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.token_counter = TokenCounter()
        
        # Three layers
        self.l1_immediate = []    # Last 3 turns (always kept)
        self.l2_recent = []       # Last 10 turns (kept if space)
        self.l3_historical = []   # Older turns (summarized)
        
        self.summary = None
    
    def add_message(self, message: Message):
        """Add message to hierarchical context."""
        self.l1_immediate.append(message)
        
        # Promote to L2 when L1 exceeds 6 messages (3 turns)
        if len(self.l1_immediate) > 6:
            # Move oldest to L2
            self.l2_recent.extend(self.l1_immediate[:2])
            self.l1_immediate = self.l1_immediate[2:]
        
        # Promote to L3 when L2 exceeds 20 messages
        if len(self.l2_recent) > 20:
            # Move oldest to L3
            self.l3_historical.extend(self.l2_recent[:10])
            self.l2_recent = self.l2_recent[10:]
        
        # Summarize L3 when it gets large
        if len(self.l3_historical) > 30:
            self._summarize_historical()
    
    def _summarize_historical(self):
        """Summarize historical messages."""
        # Would use actual summarizer
        conversation = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in self.l3_historical
        ])
        
        self.summary = f"Historical context (summarized): {conversation[:200]}..."
        self.l3_historical.clear()
    
    def get_context(self) -> List[Message]:
        """Build context from layers."""
        context = []
        tokens_used = 0
        
        # L1: Always include (immediate context)
        context.extend(self.l1_immediate)
        tokens_used += sum(
            self.token_counter.count(m.content) + 4
            for m in self.l1_immediate
        )
        
        # L3: Add summary if exists
        if self.summary:
            summary_tokens = self.token_counter.count(self.summary)
            if tokens_used + summary_tokens < self.max_tokens * 0.8:
                context.insert(0, Message("system", self.summary))
                tokens_used += summary_tokens
        
        # L2: Add as many recent messages as fit
        available = self.max_tokens - tokens_used
        l2_to_add = []
        
        for msg in reversed(self.l2_recent):
            msg_tokens = self.token_counter.count(msg.content) + 4
            if tokens_used + msg_tokens > self.max_tokens * 0.9:
                break
            
            l2_to_add.insert(0, msg)
            tokens_used += msg_tokens
        
        # Insert L2 between summary and L1
        if l2_to_add:
            insert_pos = 1 if self.summary else 0
            context[insert_pos:insert_pos] = l2_to_add
        
        return context
```

## Recency Effects

More recent information is typically more relevant.

### Recency-Weighted Selection

```python
class RecencyWeightedSelector:
    """Select messages with recency bias."""
    
    def __init__(self, decay_factor: float = 0.95):
        self.decay_factor = decay_factor
    
    def calculate_score(self, message: Message, position: int,
                       total: int) -> float:
        """
        Calculate score for message.
        
        More recent messages get higher scores.
        """
        # Position score (0 = oldest, 1 = newest)
        position_score = position / max(1, total - 1)
        
        # Apply exponential decay for older messages
        recency_score = self.decay_factor ** (total - position - 1)
        
        # Combine scores
        return position_score * 0.3 + recency_score * 0.7
    
    def select_messages(self, messages: List[Message],
                       max_count: int) -> List[Message]:
        """Select messages with recency bias."""
        # Score all messages
        scored = [
            (self.calculate_score(msg, i, len(messages)), i, msg)
            for i, msg in enumerate(messages)
        ]
        
        # Sort by score
        scored.sort(reverse=True)
        
        # Keep top N, maintaining order
        selected_indices = sorted([i for _, i, _ in scored[:max_count]])
        
        return [messages[i] for i in selected_indices]
```

### Time-Based Decay

```python
from datetime import datetime, timedelta

class TimeBasedDecay:
    """Apply time-based decay to message importance."""
    
    def __init__(self, half_life_hours: float = 24.0):
        self.half_life = timedelta(hours=half_life_hours)
    
    def calculate_decay(self, timestamp: datetime) -> float:
        """
        Calculate decay factor based on age.
        
        Returns value between 0 and 1.
        """
        age = datetime.now() - timestamp
        
        # Exponential decay: value = (1/2)^(age/half_life)
        decay = 0.5 ** (age.total_seconds() / self.half_life.total_seconds())
        
        return max(0.0, min(1.0, decay))
    
    def score_message(self, message: Message, timestamp: datetime,
                     base_score: float = 0.5) -> float:
        """Score message with time decay."""
        decay = self.calculate_decay(timestamp)
        
        # Combine base score with decay
        return base_score * (0.5 + 0.5 * decay)


# Usage
decay = TimeBasedDecay(half_life_hours=12.0)

# Message from 6 hours ago
timestamp_6h = datetime.now() - timedelta(hours=6)
score_6h = decay.calculate_decay(timestamp_6h)
print(f"6 hours ago: decay factor = {score_6h:.2f}")  # ~0.71

# Message from 24 hours ago
timestamp_24h = datetime.now() - timedelta(hours=24)
score_24h = decay.calculate_decay(timestamp_24h)
print(f"24 hours ago: decay factor = {score_24h:.2f}")  # ~0.25
```

## Relevance Filtering

Keep only relevant messages in context.

### Semantic Relevance Filter

```python
class SemanticRelevanceFilter:
    """Filter messages by semantic relevance to current query."""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.message_embeddings = {}
    
    def add_message(self, message_id: str, message: Message):
        """Add message and compute embedding."""
        embedding = self.embedding_model.embed(message.content)
        self.message_embeddings[message_id] = (message, embedding)
    
    def filter_by_relevance(self, query: str, threshold: float = 0.5,
                           max_messages: int = 10) -> List[Message]:
        """Filter messages by relevance to query."""
        if not self.message_embeddings:
            return []
        
        # Compute query embedding
        query_embedding = self.embedding_model.embed(query)
        
        # Calculate relevance scores
        scored = []
        for msg_id, (message, embedding) in self.message_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            
            if similarity >= threshold:
                scored.append((similarity, msg_id, message))
        
        # Sort by relevance
        scored.sort(reverse=True)
        
        # Return top messages
        return [msg for _, _, msg in scored[:max_messages]]
    
    def _cosine_similarity(self, emb1: list, emb2: list) -> float:
        """Calculate cosine similarity."""
        import numpy as np
        
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        
        return np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2)
        )
```

### Keyword-Based Filter

```python
import re
from typing import Set

class KeywordRelevanceFilter:
    """Filter messages by keyword relevance."""
    
    def __init__(self):
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was'
        }
    
    def extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        # Simple tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stopwords
        keywords = {w for w in words if w not in self.stopwords}
        
        return keywords
    
    def calculate_overlap(self, keywords1: Set[str],
                         keywords2: Set[str]) -> float:
        """Calculate keyword overlap."""
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = keywords1 & keywords2
        union = keywords1 | keywords2
        
        # Jaccard similarity
        return len(intersection) / len(union)
    
    def filter_messages(self, messages: List[Message], query: str,
                       threshold: float = 0.1,
                       max_messages: int = 10) -> List[Message]:
        """Filter messages by keyword overlap."""
        query_keywords = self.extract_keywords(query)
        
        scored = []
        for msg in messages:
            msg_keywords = self.extract_keywords(msg.content)
            overlap = self.calculate_overlap(query_keywords, msg_keywords)
            
            if overlap >= threshold:
                scored.append((overlap, msg))
        
        # Sort by overlap
        scored.sort(reverse=True)
        
        return [msg for _, msg in scored[:max_messages]]
```

## Context Compression

Compress context to fit more information.

### Template-Based Compression

```python
class TemplateCompressor:
    """Compress messages using templates."""
    
    def __init__(self):
        self.templates = {
            "greeting": "User greeted",
            "question": "User asked: {topic}",
            "answer": "Assistant explained {topic}",
            "confirmation": "User confirmed",
            "negation": "User declined"
        }
    
    def detect_message_type(self, content: str) -> str:
        """Detect message type."""
        content_lower = content.lower()
        
        if any(g in content_lower for g in ["hello", "hi", "hey"]):
            return "greeting"
        elif content.endswith("?"):
            return "question"
        elif any(c in content_lower for c in ["yes", "confirm", "agree"]):
            return "confirmation"
        elif any(n in content_lower for n in ["no", "don't", "not"]):
            return "negation"
        else:
            return "answer"
    
    def compress_message(self, message: Message) -> str:
        """Compress message to template."""
        msg_type = self.detect_message_type(message.content)
        template = self.templates.get(msg_type, message.content)
        
        if "{topic}" in template:
            # Extract topic (simplified)
            words = message.content.split()
            topic = " ".join(words[:5])
            return template.format(topic=topic)
        
        return template
    
    def compress_conversation(self, messages: List[Message]) -> str:
        """Compress entire conversation."""
        compressed = []
        
        for msg in messages:
            compressed_msg = self.compress_message(msg)
            compressed.append(f"{msg.role}: {compressed_msg}")
        
        return "\n".join(compressed)
```

### Extractive Summarization

```python
class ExtractiveSummarizer:
    """Extract most important sentences."""
    
    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences
    
    def score_sentence(self, sentence: str, all_words: Set[str]) -> float:
        """Score sentence importance."""
        words = set(sentence.lower().split())
        
        # Simple importance: how many unique words
        importance = len(words & all_words) / len(all_words)
        
        return importance
    
    def summarize(self, text: str) -> str:
        """Extract important sentences."""
        # Split into sentences
        sentences = text.split(". ")
        
        if len(sentences) <= self.max_sentences:
            return text
        
        # Get all words
        all_words = set(text.lower().split())
        
        # Score sentences
        scored = [
            (self.score_sentence(sent, all_words), sent)
            for sent in sentences
        ]
        
        # Sort by score
        scored.sort(reverse=True)
        
        # Take top sentences
        top_sentences = [sent for _, sent in scored[:self.max_sentences]]
        
        return ". ".join(top_sentences) + "."
```

## Multi-Turn Conversations

Managing context across multiple conversation turns.

### Turn Tracking

```python
class ConversationTurnTracker:
    """Track and manage conversation turns."""
    
    def __init__(self):
        self.turns = []
        self.current_turn = None
    
    def start_turn(self, user_input: str):
        """Start a new conversation turn."""
        self.current_turn = {
            'turn_id': len(self.turns),
            'user_input': user_input,
            'timestamp': datetime.now(),
            'agent_response': None,
            'metadata': {}
        }
    
    def complete_turn(self, agent_response: str, metadata: dict = None):
        """Complete the current turn."""
        if self.current_turn:
            self.current_turn['agent_response'] = agent_response
            self.current_turn['metadata'] = metadata or {}
            
            self.turns.append(self.current_turn)
            self.current_turn = None
    
    def get_turn(self, turn_id: int) -> dict:
        """Get a specific turn."""
        if 0 <= turn_id < len(self.turns):
            return self.turns[turn_id]
        return None
    
    def get_recent_turns(self, count: int = 5) -> List[dict]:
        """Get recent turns."""
        return self.turns[-count:]
    
    def find_turns_by_topic(self, topic: str) -> List[dict]:
        """Find turns discussing a topic."""
        topic_lower = topic.lower()
        
        matching = []
        for turn in self.turns:
            if (topic_lower in turn['user_input'].lower() or
                topic_lower in turn.get('agent_response', '').lower()):
                matching.append(turn)
        
        return matching
    
    def to_messages(self, turn_ids: List[int] = None) -> List[Message]:
        """Convert turns to messages."""
        if turn_ids is None:
            turns_to_convert = self.turns
        else:
            turns_to_convert = [self.get_turn(i) for i in turn_ids]
            turns_to_convert = [t for t in turns_to_convert if t]
        
        messages = []
        for turn in turns_to_convert:
            messages.append(Message("user", turn['user_input']))
            if turn['agent_response']:
                messages.append(Message("assistant", turn['agent_response']))
        
        return messages
```

### Context Continuity

```python
class ContinuityManager:
    """Maintain context continuity across turns."""
    
    def __init__(self):
        self.entity_tracker = {}
        self.topic_stack = []
        self.unresolved_references = []
    
    def track_entities(self, text: str):
        """Track entities mentioned in conversation."""
        # Simplified entity tracking
        words = text.split()
        
        for i, word in enumerate(words):
            if word[0].isupper() and word.lower() not in {'i', 'the'}:
                # Likely a named entity
                self.entity_tracker[word.lower()] = {
                    'canonical': word,
                    'last_mention': datetime.now(),
                    'mention_count': self.entity_tracker.get(
                        word.lower(), {}
                    ).get('mention_count', 0) + 1
                }
    
    def resolve_references(self, text: str) -> str:
        """Resolve pronouns and references."""
        # Simple reference resolution
        resolved = text
        
        # Replace "it" with last mentioned entity
        if "it" in text.lower() and self.topic_stack:
            last_topic = self.topic_stack[-1]
            resolved = resolved.replace(" it ", f" {last_topic} ")
        
        # Replace "they" with recent entities
        if "they" in text.lower():
            recent_entities = [
                entity['canonical']
                for entity in sorted(
                    self.entity_tracker.values(),
                    key=lambda x: x['last_mention'],
                    reverse=True
                )[:2]
            ]
            
            if recent_entities:
                entities_str = " and ".join(recent_entities)
                resolved = resolved.replace(" they ", f" {entities_str} ")
        
        return resolved
    
    def update_topic_stack(self, text: str):
        """Update current topic stack."""
        # Extract potential topic (simplified)
        words = text.split()
        
        # Find noun phrases (very simplified)
        for i, word in enumerate(words):
            if word[0].isupper():
                self.topic_stack.append(word)
        
        # Keep stack bounded
        if len(self.topic_stack) > 5:
            self.topic_stack.pop(0)
    
    def get_context_string(self) -> str:
        """Get continuity context as string."""
        parts = []
        
        if self.entity_tracker:
            recent_entities = sorted(
                self.entity_tracker.items(),
                key=lambda x: x[1]['last_mention'],
                reverse=True
            )[:3]
            
            parts.append("Recent entities: " + ", ".join([
                e[1]['canonical'] for e in recent_entities
            ]))
        
        if self.topic_stack:
            parts.append("Current topics: " + ", ".join(self.topic_stack[-3:]))
        
        return "\n".join(parts) if parts else ""
```

## Context Overflow Handling

Gracefully handle context overflow.

### Overflow Detection

```python
class OverflowHandler:
    """Handle context window overflow."""
    
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.token_counter = TokenCounter()
        self.warning_threshold = 0.9  # Warn at 90% capacity
    
    def check_overflow(self, messages: List[Message]) -> dict:
        """Check if context is overflowing."""
        msg_dicts = [m.to_dict() for m in messages]
        token_count = self.token_counter.count_messages(msg_dicts)
        
        utilization = token_count / self.max_tokens
        
        return {
            'is_overflow': token_count > self.max_tokens,
            'is_near_overflow': utilization > self.warning_threshold,
            'token_count': token_count,
            'max_tokens': self.max_tokens,
            'utilization': utilization,
            'tokens_over': max(0, token_count - self.max_tokens)
        }
    
    def handle_overflow(self, messages: List[Message],
                       strategy: str = "prune") -> List[Message]:
        """Handle context overflow with specified strategy."""
        status = self.check_overflow(messages)
        
        if not status['is_overflow']:
            return messages
        
        if strategy == "prune":
            return self._prune_oldest(messages, status['tokens_over'])
        elif strategy == "summarize":
            return self._summarize_old(messages)
        elif strategy == "compress":
            return self._compress_messages(messages)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _prune_oldest(self, messages: List[Message],
                     tokens_to_remove: int) -> List[Message]:
        """Remove oldest messages to fit."""
        # Keep system messages
        system_msgs = [m for m in messages if m.role == "system"]
        other_msgs = [m for m in messages if m.role != "system"]
        
        # Remove oldest messages until under limit
        tokens_removed = 0
        keep_from = 0
        
        for i, msg in enumerate(other_msgs):
            msg_tokens = self.token_counter.count(msg.content) + 4
            tokens_removed += msg_tokens
            
            if tokens_removed >= tokens_to_remove:
                keep_from = i + 1
                break
        
        return system_msgs + other_msgs[keep_from:]
    
    def _summarize_old(self, messages: List[Message]) -> List[Message]:
        """Summarize old messages."""
        # Keep recent 5 messages, summarize older ones
        if len(messages) <= 5:
            return messages
        
        old_messages = messages[:-5]
        recent_messages = messages[-5:]
        
        # Create summary
        summarizer = ExtractiveSummarizer()
        old_text = "\n".join([
            f"{m.role}: {m.content}" for m in old_messages
        ])
        summary = summarizer.summarize(old_text)
        
        # Return summary + recent messages
        return [
            Message("system", f"Previous conversation:\n{summary}")
        ] + recent_messages
    
    def _compress_messages(self, messages: List[Message]) -> List[Message]:
        """Compress messages to fit."""
        compressor = TemplateCompressor()
        
        # Compress older messages
        compressed = []
        
        for i, msg in enumerate(messages):
            if i < len(messages) - 3:  # Compress all but last 3
                compressed_content = compressor.compress_message(msg)
                compressed.append(Message(msg.role, compressed_content))
            else:
                compressed.append(msg)
        
        return compressed
```

## Performance Optimization

Optimize short-term memory operations.

### Message Caching

```python
class MessageCache:
    """Cache computed message properties."""
    
    def __init__(self, max_cache_size: int = 1000):
        self.max_cache_size = max_cache_size
        self.token_cache = {}
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_token_count(self, message: Message,
                       counter: TokenCounter) -> int:
        """Get cached token count or compute."""
        cache_key = hash(message.content)
        
        if cache_key in self.token_cache:
            self.cache_hits += 1
            return self.token_cache[cache_key]
        
        self.cache_misses += 1
        count = counter.count(message.content) + 4
        
        # Cache result
        if len(self.token_cache) < self.max_cache_size:
            self.token_cache[cache_key] = count
        
        return count
    
    def get_embedding(self, message: Message, embedder) -> list:
        """Get cached embedding or compute."""
        cache_key = hash(message.content)
        
        if cache_key in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[cache_key]
        
        self.cache_misses += 1
        embedding = embedder.embed(message.content)
        
        # Cache result
        if len(self.embedding_cache) < self.max_cache_size:
            self.embedding_cache[cache_key] = embedding
        
        return embedding
    
    def clear(self):
        """Clear cache."""
        self.token_cache.clear()
        self.embedding_cache.clear()
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (
            self.cache_hits / total_requests if total_requests > 0 else 0
        )
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.token_cache) + len(self.embedding_cache)
        }
```

### Lazy Evaluation

```python
class LazyContextBuilder:
    """Build context lazily to avoid unnecessary computation."""
    
    def __init__(self):
        self.components = {}
        self.computed = {}
    
    def register_component(self, name: str, generator):
        """Register a component generator."""
        self.components[name] = generator
    
    def get_component(self, name: str):
        """Get component, computing if needed."""
        if name in self.computed:
            return self.computed[name]
        
        if name not in self.components:
            raise ValueError(f"Unknown component: {name}")
        
        # Compute component
        result = self.components[name]()
        self.computed[name] = result
        
        return result
    
    def build_context(self, needed_components: List[str]) -> dict:
        """Build context with only needed components."""
        context = {}
        
        for component in needed_components:
            context[component] = self.get_component(component)
        
        return context
    
    def invalidate(self, component: str):
        """Invalidate cached component."""
        self.computed.pop(component, None)
    
    def clear(self):
        """Clear all computed components."""
        self.computed.clear()
```

## Summary

Short-term memory is the agent's immediate workspace, crucial for maintaining coherent behavior in the moment:

**Key Concepts**:

- **Context window**: The model's immediate attention span
- **Token budget**: Limited resource that must be managed carefully
- **Working memory**: Temporary variables and state during execution
- **Conversation history**: Past interactions that inform current behavior
- **Recency effects**: Recent information is typically more relevant

**Management Strategies**:

- **Sliding window**: Simple, predictable approach
- **Summarization**: Compress old information
- **Hierarchical**: Multiple layers with different retention policies
- **Importance-based**: Keep most important messages
- **Adaptive**: Dynamically adjust based on needs

**Key Trade-offs**:

- **Simplicity vs. sophistication**: Simple strategies are more predictable
- **Completeness vs. efficiency**: Can't keep everything, must prioritize
- **Recency vs. relevance**: Recent isn't always most relevant
- **Compression vs. information loss**: Summarization loses details

## Next Steps

- **[Long-Term Memory](long-term-memory.md)**: Persistent storage across sessions
- **[Memory Retrieval](memory-retrieval.md)**: Finding relevant information efficiently
- **[Memory Consolidation](memory-consolidation.md)**: Moving information from short-term to long-term storage
- **[Memory Architecture](memory-architecture.md)**: Overall memory system design
