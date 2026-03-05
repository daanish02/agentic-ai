# Context Management

## Table of Contents

- [Introduction](#introduction)
- [What is Context?](#what-is-context)
- [The Context Window Problem](#the-context-window-problem)
- [Context Components](#context-components)
- [Building Context](#building-context)
- [Context Strategies](#context-strategies)
- [Context Compression](#context-compression)
- [Context Prioritization](#context-prioritization)
- [Rolling Context](#rolling-context)
- [Hierarchical Context](#hierarchical-context)
- [Context Retrieval](#context-retrieval)
- [Managing Tool Outputs](#managing-tool-outputs)
- [Conversation History Management](#conversation-history-management)
- [Context Overflow Handling](#context-overflow-handling)
- [Context-Aware Prompting](#context-aware-prompting)
- [Best Practices](#best-practices)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Context is the memory of your agent. It's everything the agent knows at any given moment: the conversation so far, tool outputs, retrieved information, system instructions, and more. Managing context effectively is the difference between an agent that works and one that falls apart.

The challenge: context windows are limited. Modern LLMs might have 100K or even 200K token limits, but that fills up fast when you're dealing with conversation history, tool outputs, documents, and system prompts. Poor context management leads to:

- Lost information
- Degraded reasoning
- Failed tasks
- Wasted tokens and costs

Good context management ensures your agent always has the right information at the right time without overwhelming the model.

> "An agent is only as good as its context."

This guide covers context management from basics to advanced techniques: what context is, how to build it, how to compress it, how to prioritize it, and how to keep your agent working even as conversations grow long and tool outputs accumulate.

## What is Context?

**Context** is all the information available to the language model when generating a response. For agents, this includes:

### Context Layers

```
┌─────────────────────────────────────────────┐
│        SYSTEM PROMPT (Static)               │
│  - Agent identity                           │
│  - Instructions                             │
│  - Tool descriptions                        │
│  - Guidelines                               │
├─────────────────────────────────────────────┤
│     CONVERSATION HISTORY (Growing)          │
│  - User messages                            │
│  - Agent responses                          │
│  - Timestamps                               │
├─────────────────────────────────────────────┤
│      TOOL OUTPUTS (Variable)                │
│  - Function results                         │
│  - API responses                            │
│  - Database queries                         │
├─────────────────────────────────────────────┤
│   RETRIEVED INFORMATION (Dynamic)           │
│  - Document snippets                        │
│  - Database records                         │
│  - Memory recalls                           │
├─────────────────────────────────────────────┤
│    CURRENT STATE (Ephemeral)                │
│  - Active tasks                             │
│  - Temporary variables                      │
│  - Working memory                           │
└─────────────────────────────────────────────┘
```

### Context Example

```python
# Complete context for a single LLM call
context = """
SYSTEM: You are a research assistant with web search capabilities.
Available tools: web_search, summarize, save_note

CONVERSATION:
User: Find recent papers on RAG
Assistant: I'll search for papers on RAG (Retrieval Augmented Generation).
[TOOL: web_search("RAG papers 2025")]
Tool result: Found 10 papers...
Assistant: Here are the key papers...

User: Summarize the first one

CURRENT TASK: Summarize the first paper from previous results
"""
```

### Why Context Matters

**Without proper context**:

```
User: "Summarize the first one"
Agent: "I don't know what you're referring to."
```

**With proper context**:

```
User: "Summarize the first one"
Agent: [Has previous search results in context]
      "The first paper, 'RAG 2.0' by Smith et al., proposes..."
```

## The Context Window Problem

Understanding context limits is fundamental to agent design.

### Context Window Sizes

```python
# Approximate token limits (as of 2025)
CONTEXT_LIMITS = {
    "gpt-3.5-turbo": 16_000,
    "gpt-4": 8_000,
    "gpt-4-turbo": 128_000,
    "gpt-4o": 128_000,
    "claude-2": 100_000,
    "claude-3": 200_000,
    "gemini-1.5": 1_000_000,
}
```

### The Problem

```python
def token_usage_breakdown():
    """Typical token usage in an agent conversation."""

    return {
        "system_prompt": 500,        # Agent instructions + tools
        "conversation_history": 3000, # 10 turns
        "tool_outputs": 5000,         # Search results, API responses
        "retrieved_docs": 4000,       # RAG documents
        "working_memory": 500,        # Current state
        "response_buffer": 1000,      # Reserve for generation
        # ─────────────────
        "total": 14_000               # Already using 14K tokens!
    }
```

After just a few interactions with rich tool outputs, you're consuming significant context. Without management, you hit limits and the agent breaks.

### Context Overflow

```python
def what_happens_on_overflow():
    """When context exceeds limits..."""

    # Option 1: Truncate from the beginning
    # ❌ PROBLEM: Lose conversation history

    # Option 2: Truncate from the middle
    # ❌ PROBLEM: Lose coherence

    # Option 3: Truncate everything
    # ❌ PROBLEM: Lose critical information

    # Option 4: Intelligent management
    # ✅ SOLUTION: Selective retention
```

## Context Components

Understanding each component helps you manage them effectively.

### 1. System Prompt

**Characteristics**:

- Static or rarely changing
- Highest priority (never remove)
- Defines agent identity and capabilities
- Relatively small (100-1000 tokens)

```python
class SystemPrompt:
    """Manage system prompt."""

    def __init__(self, agent_role: str, tools: List[str]):
        self.agent_role = agent_role
        self.tools = tools

    def build(self) -> str:
        """Build system prompt."""
        prompt = f"You are a {self.agent_role}.\n\n"
        prompt += "Available tools:\n"
        for tool in self.tools:
            prompt += f"- {tool}\n"
        prompt += "\nGuidelines:\n"
        prompt += "- Be precise and factual\n"
        prompt += "- Cite sources\n"
        prompt += "- Use tools when needed\n"
        return prompt

    def token_count(self) -> int:
        """Estimate token count."""
        # Rough estimate: ~4 chars per token
        return len(self.build()) // 4
```

### 2. Conversation History

**Characteristics**:

- Grows over time
- Variable importance
- Can be compressed or summarized
- Largest consumer of context

```python
class ConversationHistory:
    """Manage conversation history."""

    def __init__(self):
        self.messages = []

    def add_message(self, role: str, content: str):
        """Add a message to history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "tokens": len(content) // 4  # Rough estimate
        })

    def get_recent(self, max_tokens: int) -> List[dict]:
        """Get recent messages within token budget."""
        selected = []
        token_count = 0

        # Take from most recent backwards
        for msg in reversed(self.messages):
            if token_count + msg["tokens"] > max_tokens:
                break
            selected.insert(0, msg)
            token_count += msg["tokens"]

        return selected

    def total_tokens(self) -> int:
        """Get total token count."""
        return sum(msg["tokens"] for msg in self.messages)
```

### 3. Tool Outputs

**Characteristics**:

- Variable size (can be very large)
- Time-sensitive relevance
- Often needs processing
- Critical for task completion

```python
class ToolOutputManager:
    """Manage tool outputs in context."""

    def __init__(self):
        self.outputs = []

    def add_output(self, tool_name: str, result: Any):
        """Add tool output."""
        self.outputs.append({
            "tool": tool_name,
            "result": result,
            "timestamp": time.time(),
            "raw_size": len(str(result)),
            "summarized": None
        })

    def summarize_large_outputs(self, max_size: int = 500):
        """Summarize large tool outputs."""
        for output in self.outputs:
            if output["raw_size"] > max_size and not output["summarized"]:
                # Truncate or summarize
                result_str = str(output["result"])
                output["summarized"] = result_str[:max_size] + "..."

    def get_relevant_outputs(self, query: str) -> List[dict]:
        """Get outputs relevant to current query."""
        # Simple relevance: has overlapping keywords
        query_words = set(query.lower().split())
        relevant = []

        for output in self.outputs:
            output_words = set(str(output["result"]).lower().split())
            overlap = len(query_words & output_words)

            if overlap > 0:
                relevant.append({
                    **output,
                    "relevance": overlap
                })

        return sorted(relevant, key=lambda x: x["relevance"], reverse=True)
```

### 4. Retrieved Information

**Characteristics**:

- Dynamically fetched
- Relevant to current task
- Can be large (documents, records)
- Needs to be condensed

```python
class RetrievalManager:
    """Manage retrieved information."""

    def __init__(self):
        self.retrieved = []

    def add_retrieval(self, source: str, content: str, relevance: float):
        """Add retrieved information."""
        self.retrieved.append({
            "source": source,
            "content": content,
            "relevance": relevance,
            "timestamp": time.time()
        })

    def get_top_k(self, k: int = 5) -> List[dict]:
        """Get top k most relevant retrievals."""
        sorted_items = sorted(
            self.retrieved,
            key=lambda x: x["relevance"],
            reverse=True
        )
        return sorted_items[:k]

    def compress_retrievals(self, max_tokens: int) -> str:
        """Compress retrievals to fit budget."""
        top_items = self.get_top_k()

        compressed = "Retrieved information:\n"
        token_count = len(compressed) // 4

        for item in top_items:
            item_text = f"[{item['source']}] {item['content']}\n"
            item_tokens = len(item_text) // 4

            if token_count + item_tokens > max_tokens:
                # Truncate this item
                remaining = (max_tokens - token_count) * 4
                item_text = item_text[:remaining] + "...\n"

            compressed += item_text
            token_count += len(item_text) // 4

            if token_count >= max_tokens:
                break

        return compressed
```

## Building Context

How you construct context affects agent performance.

### Basic Context Builder

```python
class ContextBuilder:
    """Build context for LLM from components."""

    def __init__(
        self,
        system_prompt: str,
        max_tokens: int = 8000
    ):
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.history = ConversationHistory()
        self.tools = ToolOutputManager()
        self.retrievals = RetrievalManager()

    def build(self, current_input: str) -> str:
        """Build complete context."""

        components = []
        token_budget = self.max_tokens

        # 1. System prompt (always include)
        components.append(("system", self.system_prompt))
        token_budget -= len(self.system_prompt) // 4

        # 2. Reserve space for response
        response_buffer = 1000
        token_budget -= response_buffer

        # 3. Current input (always include)
        components.append(("user", current_input))
        token_budget -= len(current_input) // 4

        # 4. Allocate remaining budget
        history_budget = int(token_budget * 0.5)
        tool_budget = int(token_budget * 0.3)
        retrieval_budget = int(token_budget * 0.2)

        # 5. Add conversation history
        recent_history = self.history.get_recent(history_budget)
        for msg in recent_history:
            components.append((msg["role"], msg["content"]))

        # 6. Add relevant tool outputs
        self.tools.summarize_large_outputs()
        relevant_tools = self.tools.get_relevant_outputs(current_input)
        tool_text = self._format_tool_outputs(relevant_tools, tool_budget)
        if tool_text:
            components.append(("system", tool_text))

        # 7. Add relevant retrievals
        retrieval_text = self.retrievals.compress_retrievals(retrieval_budget)
        if retrieval_text:
            components.append(("system", retrieval_text))

        # 8. Assemble final context
        return self._assemble(components)

    def _format_tool_outputs(
        self,
        outputs: List[dict],
        max_tokens: int
    ) -> str:
        """Format tool outputs for context."""
        if not outputs:
            return ""

        text = "Recent tool outputs:\n"
        token_count = len(text) // 4

        for output in outputs:
            content = output.get("summarized") or str(output["result"])
            output_text = f"[{output['tool']}] {content}\n"
            output_tokens = len(output_text) // 4

            if token_count + output_tokens > max_tokens:
                break

            text += output_text
            token_count += output_tokens

        return text

    def _assemble(self, components: List[tuple]) -> str:
        """Assemble components into final prompt."""
        sections = []

        for role, content in components:
            sections.append(f"{role.upper()}: {content}")

        return "\n\n".join(sections)
```

### Advanced Context Builder

```python
class AdvancedContextBuilder:
    """Advanced context builder with prioritization."""

    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.components = {}

    def add_component(
        self,
        name: str,
        content: str,
        priority: int,
        required: bool = False
    ):
        """Add a context component with priority."""
        self.components[name] = {
            "content": content,
            "priority": priority,
            "required": required,
            "tokens": len(content) // 4
        }

    def build(self) -> str:
        """Build context respecting priorities and budget."""

        # Separate required and optional
        required = {
            k: v for k, v in self.components.items()
            if v["required"]
        }
        optional = {
            k: v for k, v in self.components.items()
            if not v["required"]
        }

        # Check if required components fit
        required_tokens = sum(c["tokens"] for c in required.values())
        if required_tokens > self.max_tokens:
            raise ValueError("Required components exceed token budget")

        # Build with required components
        selected = required.copy()
        remaining_tokens = self.max_tokens - required_tokens

        # Add optional by priority
        sorted_optional = sorted(
            optional.items(),
            key=lambda x: x[1]["priority"],
            reverse=True
        )

        for name, component in sorted_optional:
            if component["tokens"] <= remaining_tokens:
                selected[name] = component
                remaining_tokens -= component["tokens"]

        # Assemble in priority order
        sorted_selected = sorted(
            selected.items(),
            key=lambda x: x[1]["priority"],
            reverse=True
        )

        parts = [comp["content"] for name, comp in sorted_selected]
        return "\n\n".join(parts)

    def get_utilization(self) -> dict:
        """Get token utilization stats."""
        total_available = self.max_tokens
        total_used = sum(c["tokens"] for c in self.components.values())

        return {
            "total_available": total_available,
            "total_used": total_used,
            "utilization": total_used / total_available,
            "remaining": total_available - total_used
        }
```

## Context Strategies

Different strategies for different scenarios.

### 1. Full Context Strategy

**Use when**: Context fits comfortably within limits

```python
class FullContextStrategy:
    """Include everything - no compression."""

    def build_context(self, components: dict) -> str:
        """Simply concatenate all components."""
        return "\n\n".join([
            components["system_prompt"],
            components["conversation_history"],
            components["tool_outputs"],
            components["retrievals"]
        ])
```

### 2. Sliding Window Strategy

**Use when**: Long conversations but tasks are independent

```python
class SlidingWindowStrategy:
    """Keep only recent N messages."""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.messages = []

    def add_message(self, role: str, content: str):
        """Add message and maintain window."""
        self.messages.append({"role": role, "content": content})

        # Keep only recent messages
        if len(self.messages) > self.window_size:
            self.messages = self.messages[-self.window_size:]

    def build_context(self, system_prompt: str) -> str:
        """Build context with sliding window."""
        parts = [system_prompt]

        for msg in self.messages:
            parts.append(f"{msg['role']}: {msg['content']}")

        return "\n\n".join(parts)
```

### 3. Summarization Strategy

**Use when**: Need historical context but it's too large

```python
class SummarizationStrategy:
    """Summarize old context, keep recent full."""

    def __init__(
        self,
        keep_recent: int = 5,
        summarize_older: bool = True
    ):
        self.keep_recent = keep_recent
        self.summarize_older = summarize_older
        self.messages = []
        self.summary = None

    def add_message(self, role: str, content: str):
        """Add message."""
        self.messages.append({"role": role, "content": content})

        # Summarize if too many messages
        if len(self.messages) > self.keep_recent * 2:
            self._create_summary()

    def _create_summary(self):
        """Create summary of old messages."""
        # Messages to summarize
        to_summarize = self.messages[:-self.keep_recent]

        # Simple summary (in production, use LLM)
        summary_text = f"Previous conversation summary:\n"
        summary_text += f"- {len(to_summarize)} messages exchanged\n"

        # Extract key topics
        all_text = " ".join(m["content"] for m in to_summarize)
        words = all_text.split()
        summary_text += f"- Topics discussed: {', '.join(words[:10])}...\n"

        self.summary = summary_text
        self.messages = self.messages[-self.keep_recent:]

    def build_context(self, system_prompt: str) -> str:
        """Build context with summary."""
        parts = [system_prompt]

        if self.summary:
            parts.append(self.summary)

        for msg in self.messages:
            parts.append(f"{msg['role']}: {msg['content']}")

        return "\n\n".join(parts)
```

### 4. Relevance-Based Strategy

**Use when**: Have lots of context but only some is relevant

```python
class RelevanceStrategy:
    """Include only relevant context."""

    def __init__(self):
        self.all_context = []

    def add_context(self, content: str, tags: List[str]):
        """Add context with tags."""
        self.all_context.append({
            "content": content,
            "tags": set(tags),
            "timestamp": time.time()
        })

    def build_context(
        self,
        system_prompt: str,
        current_query: str,
        max_tokens: int
    ) -> str:
        """Build context using relevance."""

        # Extract keywords from query
        query_keywords = set(current_query.lower().split())

        # Score each context item
        scored_items = []
        for item in self.all_context:
            # Keyword overlap
            overlap = len(query_keywords & item["tags"])

            # Recency bonus
            age = time.time() - item["timestamp"]
            recency_score = 1.0 / (1.0 + age / 3600)  # Decay over hours

            score = overlap + recency_score
            scored_items.append((score, item))

        # Sort by relevance
        scored_items.sort(reverse=True, key=lambda x: x[0])

        # Build context within budget
        parts = [system_prompt]
        token_count = len(system_prompt) // 4

        for score, item in scored_items:
            item_tokens = len(item["content"]) // 4
            if token_count + item_tokens > max_tokens:
                break

            parts.append(item["content"])
            token_count += item_tokens

        return "\n\n".join(parts)
```

## Context Compression

Reduce context size without losing critical information.

### Token-Level Compression

```python
class TokenCompressor:
    """Compress text at token level."""

    def compress(self, text: str, target_ratio: float = 0.5) -> str:
        """Compress text to target ratio of original."""

        sentences = text.split('. ')
        target_count = int(len(sentences) * target_ratio)

        if target_count >= len(sentences):
            return text

        # Score sentences by information density
        scored = []
        for sent in sentences:
            # Simple scoring: length and unique words
            words = sent.split()
            unique_words = len(set(words))
            score = unique_words / max(len(words), 1)
            scored.append((score, sent))

        # Keep highest scoring sentences
        scored.sort(reverse=True, key=lambda x: x[0])
        kept = [sent for score, sent in scored[:target_count]]

        return '. '.join(kept) + '.'

    def compress_to_tokens(self, text: str, max_tokens: int) -> str:
        """Compress to specific token count."""

        current_tokens = len(text) // 4  # Rough estimate

        if current_tokens <= max_tokens:
            return text

        target_ratio = max_tokens / current_tokens
        return self.compress(text, target_ratio)
```

### Semantic Compression

```python
class SemanticCompressor:
    """Compress preserving semantic meaning."""

    def __init__(self):
        self.important_patterns = [
            r'\d+',  # Numbers
            r'\b[A-Z][A-Za-z]+\b',  # Proper nouns
            r'http[s]?://\S+',  # URLs
        ]

    def extract_key_information(self, text: str) -> str:
        """Extract key information from text."""

        sentences = text.split('. ')
        key_sentences = []

        for sent in sentences:
            # Keep if contains important patterns
            if self._is_important(sent):
                key_sentences.append(sent)
            # Or if it's a short, definitive statement
            elif len(sent.split()) < 15 and any(
                word in sent.lower()
                for word in ['is', 'are', 'was', 'were', 'will']
            ):
                key_sentences.append(sent)

        return '. '.join(key_sentences)

    def _is_important(self, text: str) -> bool:
        """Check if text contains important patterns."""
        import re
        for pattern in self.important_patterns:
            if re.search(pattern, text):
                return True
        return False

    def bullet_point_summary(self, text: str, max_points: int = 5) -> str:
        """Create bullet point summary."""

        sentences = text.split('. ')

        # Score by importance
        scored = []
        for sent in sentences:
            score = 0

            # Has numbers
            if re.search(r'\d+', sent):
                score += 2

            # Has proper nouns
            if re.search(r'\b[A-Z][A-Za-z]+\b', sent):
                score += 1

            # Contains keywords
            keywords = ['important', 'key', 'significant', 'main', 'critical']
            if any(kw in sent.lower() for kw in keywords):
                score += 2

            scored.append((score, sent))

        # Keep top sentences
        scored.sort(reverse=True, key=lambda x: x[0])
        top_sentences = [sent for score, sent in scored[:max_points]]

        # Format as bullets
        return '\n'.join(f"- {sent}" for sent in top_sentences)
```

### Structured Compression

```python
class StructuredCompressor:
    """Compress by extracting structure."""

    def compress_conversation(
        self,
        messages: List[dict],
        max_messages: int = 10
    ) -> dict:
        """Compress conversation to summary + recent."""

        if len(messages) <= max_messages:
            return {"messages": messages, "summary": None}

        # Split into old and recent
        old_messages = messages[:-max_messages]
        recent_messages = messages[-max_messages:]

        # Summarize old messages
        summary = self._summarize_messages(old_messages)

        return {
            "summary": summary,
            "messages": recent_messages
        }

    def _summarize_messages(self, messages: List[dict]) -> str:
        """Summarize a list of messages."""

        user_messages = [m for m in messages if m["role"] == "user"]
        agent_messages = [m for m in messages if m["role"] == "assistant"]

        summary = f"Earlier conversation ({len(messages)} messages):\n"
        summary += f"User made {len(user_messages)} requests:\n"

        # Extract main topics from user messages
        all_user_text = " ".join(m["content"] for m in user_messages)
        words = all_user_text.split()
        # Simple keyword extraction (in production, use proper NLP)
        keywords = list(set(words))[:10]
        summary += f"Topics: {', '.join(keywords)}\n"

        return summary

    def compress_tool_outputs(
        self,
        outputs: List[dict],
        max_outputs: int = 5
    ) -> List[dict]:
        """Compress tool outputs."""

        if len(outputs) <= max_outputs:
            return outputs

        # Keep most recent
        compressed = outputs[-max_outputs:]

        # Add summary of older outputs
        old_count = len(outputs) - max_outputs
        summary = {
            "tool": "summary",
            "result": f"[{old_count} earlier tool outputs omitted]"
        }

        return [summary] + compressed
```

## Context Prioritization

Decide what to keep when space is limited.

### Priority System

```python
class ContextPriority:
    """Priority levels for context components."""

    CRITICAL = 100    # System prompt, current input
    HIGH = 75         # Recent messages, current tool outputs
    MEDIUM = 50       # Older messages, past tool outputs
    LOW = 25          # Retrieved docs, background info
    OPTIONAL = 10     # Nice-to-have information


class PriorityManager:
    """Manage context with priorities."""

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.items = []

    def add_item(
        self,
        content: str,
        priority: int,
        category: str
    ):
        """Add item with priority."""
        self.items.append({
            "content": content,
            "priority": priority,
            "category": category,
            "tokens": len(content) // 4
        })

    def build_context(self) -> str:
        """Build context respecting priorities."""

        # Sort by priority
        sorted_items = sorted(
            self.items,
            key=lambda x: x["priority"],
            reverse=True
        )

        # Add items until budget exhausted
        selected = []
        token_count = 0

        for item in sorted_items:
            if token_count + item["tokens"] <= self.max_tokens:
                selected.append(item)
                token_count += item["tokens"]
            elif item["priority"] >= ContextPriority.CRITICAL:
                # Truncate critical items rather than drop them
                remaining = self.max_tokens - token_count
                truncated = item["content"][:remaining * 4]
                selected.append({
                    **item,
                    "content": truncated + "...[truncated]"
                })
                break

        # Group by category for better organization
        by_category = {}
        for item in selected:
            cat = item["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(item["content"])

        # Build final context
        parts = []
        for category in ["system", "conversation", "tools", "retrieval"]:
            if category in by_category:
                parts.extend(by_category[category])

        return "\n\n".join(parts)

    def get_stats(self) -> dict:
        """Get prioritization statistics."""
        total_tokens = sum(item["tokens"] for item in self.items)
        by_priority = {}

        for item in self.items:
            pri = item["priority"]
            if pri not in by_priority:
                by_priority[pri] = 0
            by_priority[pri] += item["tokens"]

        return {
            "total_items": len(self.items),
            "total_tokens": total_tokens,
            "max_tokens": self.max_tokens,
            "utilization": total_tokens / self.max_tokens,
            "by_priority": by_priority
        }
```

### Dynamic Prioritization

```python
class DynamicPriority:
    """Dynamically adjust priorities based on context."""

    def __init__(self):
        self.items = []

    def add_item(
        self,
        content: str,
        base_priority: int,
        metadata: dict
    ):
        """Add item with metadata for dynamic prioritization."""
        self.items.append({
            "content": content,
            "base_priority": base_priority,
            "metadata": metadata,
            "final_priority": base_priority
        })

    def compute_priorities(self, current_task: str):
        """Compute final priorities based on current task."""

        task_keywords = set(current_task.lower().split())

        for item in self.items:
            priority = item["base_priority"]

            # Recency bonus
            age = time.time() - item["metadata"].get("timestamp", 0)
            if age < 300:  # Last 5 minutes
                priority += 20
            elif age < 3600:  # Last hour
                priority += 10

            # Relevance bonus
            content_keywords = set(item["content"].lower().split())
            overlap = len(task_keywords & content_keywords)
            priority += overlap * 5

            # Task-specific adjustments
            if "error" in item["content"].lower():
                priority += 15  # Errors are important

            if item["metadata"].get("from_tool"):
                priority += 10  # Tool outputs are valuable

            item["final_priority"] = priority

    def get_prioritized_items(self) -> List[dict]:
        """Get items sorted by final priority."""
        return sorted(
            self.items,
            key=lambda x: x["final_priority"],
            reverse=True
        )
```

## Rolling Context

Maintain a rolling window of context.

### Basic Rolling Window

```python
class RollingContext:
    """Maintain rolling window of context."""

    def __init__(
        self,
        max_tokens: int,
        system_prompt: str
    ):
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.messages = []
        self.system_tokens = len(system_prompt) // 4

    def add_message(self, role: str, content: str):
        """Add message and maintain window."""

        message = {
            "role": role,
            "content": content,
            "tokens": len(content) // 4
        }

        self.messages.append(message)
        self._maintain_window()

    def _maintain_window(self):
        """Keep context within token limit."""

        # Calculate total tokens
        total = self.system_tokens
        total += sum(m["tokens"] for m in self.messages)

        # Remove oldest messages if over limit
        while total > self.max_tokens and len(self.messages) > 1:
            removed = self.messages.pop(0)
            total -= removed["tokens"]

    def get_context(self) -> str:
        """Get current context."""
        parts = [self.system_prompt]

        for msg in self.messages:
            parts.append(f"{msg['role']}: {msg['content']}")

        return "\n\n".join(parts)

    def get_stats(self) -> dict:
        """Get window statistics."""
        current_tokens = self.system_tokens
        current_tokens += sum(m["tokens"] for m in self.messages)

        return {
            "max_tokens": self.max_tokens,
            "current_tokens": current_tokens,
            "utilization": current_tokens / self.max_tokens,
            "message_count": len(self.messages)
        }
```

### Adaptive Rolling Window

```python
class AdaptiveRollingContext:
    """Rolling window that adapts to conversation needs."""

    def __init__(
        self,
        max_tokens: int,
        min_messages: int = 3
    ):
        self.max_tokens = max_tokens
        self.min_messages = min_messages
        self.messages = []

    def add_message(
        self,
        role: str,
        content: str,
        importance: float = 1.0
    ):
        """Add message with importance score."""

        self.messages.append({
            "role": role,
            "content": content,
            "tokens": len(content) // 4,
            "importance": importance,
            "timestamp": time.time()
        })

        self._adapt_window()

    def _adapt_window(self):
        """Adapt window size based on message importance."""

        # Always keep minimum messages
        if len(self.messages) <= self.min_messages:
            return

        # Calculate current tokens
        total_tokens = sum(m["tokens"] for m in self.messages)

        if total_tokens <= self.max_tokens:
            return

        # Remove least important old messages
        # Keep recent messages even if less important
        cutoff_index = max(len(self.messages) - self.min_messages, 0)
        old_messages = self.messages[:cutoff_index]
        recent_messages = self.messages[cutoff_index:]

        # Sort old messages by importance
        old_messages.sort(key=lambda x: x["importance"])

        # Remove lowest importance until within budget
        while total_tokens > self.max_tokens and old_messages:
            removed = old_messages.pop(0)
            total_tokens -= removed["tokens"]

        # Update messages list
        self.messages = old_messages + recent_messages
```

## Hierarchical Context

Organize context in layers.

### Layered Context Manager

```python
class HierarchicalContext:
    """Manage context in hierarchical layers."""

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.layers = {
            "system": {"content": [], "priority": 100, "required": True},
            "recent": {"content": [], "priority": 80, "required": True},
            "working": {"content": [], "priority": 60, "required": False},
            "background": {"content": [], "priority": 40, "required": False},
            "archive": {"content": [], "priority": 20, "required": False}
        }

    def add_to_layer(self, layer: str, content: str):
        """Add content to a specific layer."""
        if layer in self.layers:
            self.layers[layer]["content"].append(content)

    def build_context(self) -> str:
        """Build context from layers."""

        # Sort layers by priority
        sorted_layers = sorted(
            self.layers.items(),
            key=lambda x: x[1]["priority"],
            reverse=True
        )

        parts = []
        token_count = 0

        for layer_name, layer_data in sorted_layers:
            layer_content = "\n".join(layer_data["content"])
            layer_tokens = len(layer_content) // 4

            # Required layers must be included (truncated if needed)
            if layer_data["required"]:
                if token_count + layer_tokens > self.max_tokens:
                    # Truncate to fit
                    remaining = (self.max_tokens - token_count) * 4
                    layer_content = layer_content[:remaining] + "...[truncated]"
                    layer_tokens = len(layer_content) // 4

                parts.append(f"=== {layer_name.upper()} ===\n{layer_content}")
                token_count += layer_tokens

                if token_count >= self.max_tokens:
                    break

            # Optional layers only if space available
            else:
                if token_count + layer_tokens <= self.max_tokens:
                    parts.append(f"=== {layer_name.upper()} ===\n{layer_content}")
                    token_count += layer_tokens

        return "\n\n".join(parts)

    def promote_to_layer(self, content: str, from_layer: str, to_layer: str):
        """Move content between layers."""
        if from_layer in self.layers and to_layer in self.layers:
            if content in self.layers[from_layer]["content"]:
                self.layers[from_layer]["content"].remove(content)
                self.layers[to_layer]["content"].append(content)

    def archive_old_content(self, max_age: float = 3600):
        """Move old content from working to archive."""
        # This would need timestamp tracking per content item
        # Simplified version:
        for layer_name in ["working", "background"]:
            if layer_name in self.layers:
                content = self.layers[layer_name]["content"]
                if len(content) > 10:  # Arbitrary threshold
                    # Move oldest to archive
                    archived = content[:5]
                    self.layers[layer_name]["content"] = content[5:]
                    self.layers["archive"]["content"].extend(archived)
```

## Context Retrieval

Retrieve relevant context from storage.

### Context Retriever

```python
class ContextRetriever:
    """Retrieve relevant context from history."""

    def __init__(self):
        self.stored_context = []

    def store(self, content: str, metadata: dict):
        """Store context for later retrieval."""
        self.stored_context.append({
            "content": content,
            "metadata": metadata,
            "timestamp": time.time()
        })

    def retrieve_by_keywords(
        self,
        keywords: List[str],
        max_results: int = 5
    ) -> List[dict]:
        """Retrieve context matching keywords."""

        scored = []
        for item in self.stored_context:
            score = 0
            content_lower = item["content"].lower()

            for keyword in keywords:
                if keyword.lower() in content_lower:
                    score += 1

            if score > 0:
                scored.append((score, item))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [item for score, item in scored[:max_results]]

    def retrieve_by_time(
        self,
        start_time: float,
        end_time: float
    ) -> List[dict]:
        """Retrieve context from time range."""
        return [
            item for item in self.stored_context
            if start_time <= item["timestamp"] <= end_time
        ]

    def retrieve_similar(
        self,
        query: str,
        max_results: int = 5
    ) -> List[dict]:
        """Retrieve semantically similar context."""
        # Simple word overlap (in production, use embeddings)
        query_words = set(query.lower().split())

        scored = []
        for item in self.stored_context:
            content_words = set(item["content"].lower().split())
            overlap = len(query_words & content_words)

            if overlap > 0:
                scored.append((overlap, item))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [item for score, item in scored[:max_results]]
```

## Managing Tool Outputs

Tool outputs can be large and need special handling.

### Tool Output Manager

```python
class ToolOutputContextManager:
    """Specialized manager for tool outputs in context."""

    def __init__(self, max_total_tokens: int = 2000):
        self.max_total_tokens = max_total_tokens
        self.outputs = []

    def add_output(
        self,
        tool_name: str,
        arguments: dict,
        result: Any
    ):
        """Add tool output."""

        output = {
            "tool": tool_name,
            "arguments": arguments,
            "result": result,
            "timestamp": time.time(),
            "raw_tokens": len(str(result)) // 4
        }

        # Summarize if too large
        if output["raw_tokens"] > 500:
            output["summary"] = self._summarize_result(result)
            output["summary_tokens"] = len(output["summary"]) // 4
        else:
            output["summary"] = None
            output["summary_tokens"] = 0

        self.outputs.append(output)
        self._maintain_limit()

    def _summarize_result(self, result: Any) -> str:
        """Summarize a large result."""
        result_str = str(result)

        # If it's a list, summarize structure
        if isinstance(result, list):
            return f"List of {len(result)} items: {result[:3]}..."

        # If it's a dict, summarize keys
        elif isinstance(result, dict):
            keys = list(result.keys())[:5]
            return f"Dict with keys: {keys}..."

        # Otherwise truncate
        else:
            return result_str[:500] + "..."

    def _maintain_limit(self):
        """Keep outputs within token limit."""

        # Calculate total tokens (using summaries when available)
        total = sum(
            o.get("summary_tokens", o["raw_tokens"])
            for o in self.outputs
        )

        # Remove oldest until within limit
        while total > self.max_total_tokens and self.outputs:
            removed = self.outputs.pop(0)
            total -= removed.get("summary_tokens", removed["raw_tokens"])

    def get_context_string(self) -> str:
        """Get tool outputs as context string."""

        if not self.outputs:
            return ""

        parts = ["Recent tool outputs:"]

        for output in self.outputs:
            tool_str = f"\n[{output['tool']}("
            tool_str += ", ".join(f"{k}={v}" for k, v in output['arguments'].items())
            tool_str += ")]"

            # Use summary if available, otherwise raw result
            if output["summary"]:
                tool_str += f"\nResult (summarized): {output['summary']}"
            else:
                tool_str += f"\nResult: {output['result']}"

            parts.append(tool_str)

        return "\n".join(parts)

    def get_relevant_outputs(self, query: str, max_outputs: int = 3) -> str:
        """Get outputs relevant to current query."""

        query_words = set(query.lower().split())
        scored = []

        for output in self.outputs:
            # Score by keyword overlap
            tool_text = f"{output['tool']} {output['arguments']} {output['result']}"
            tool_words = set(tool_text.lower().split())
            overlap = len(query_words & tool_words)

            # Recency bonus
            age = time.time() - output["timestamp"]
            recency = 1.0 / (1.0 + age / 300)  # Decay over 5 minutes

            score = overlap + recency
            scored.append((score, output))

        # Return top outputs
        scored.sort(reverse=True, key=lambda x: x[0])
        relevant = [output for score, output in scored[:max_outputs]]

        # Format as string
        if not relevant:
            return ""

        parts = ["Relevant tool outputs:"]
        for output in relevant:
            parts.append(
                f"[{output['tool']}] {output.get('summary') or output['result']}"
            )

        return "\n".join(parts)
```

## Conversation History Management

Strategies specifically for managing conversation history.

### Smart History Manager

```python
class ConversationManager:
    """Intelligent conversation history management."""

    def __init__(
        self,
        max_tokens: int = 4000,
        always_keep: int = 3
    ):
        self.max_tokens = max_tokens
        self.always_keep = always_keep  # Recent messages to always keep
        self.messages = []
        self.summary = None

    def add_exchange(self, user_message: str, agent_response: str):
        """Add a complete exchange."""
        self.messages.append({
            "user": user_message,
            "agent": agent_response,
            "timestamp": time.time(),
            "tokens": (len(user_message) + len(agent_response)) // 4
        })

        self._manage_history()

    def _manage_history(self):
        """Manage history to stay within limits."""

        total_tokens = sum(m["tokens"] for m in self.messages)

        if total_tokens <= self.max_tokens:
            return

        # Keep recent messages
        recent = self.messages[-self.always_keep:]
        older = self.messages[:-self.always_keep]

        # Summarize older messages
        if older and not self.summary:
            self.summary = self._create_summary(older)
            self.messages = recent

        # If still over limit, compress recent
        total_tokens = sum(m["tokens"] for m in self.messages)
        if self.summary:
            total_tokens += len(self.summary) // 4

        if total_tokens > self.max_tokens:
            # More aggressive compression needed
            self.messages = self.messages[-self.always_keep//2:]

    def _create_summary(self, messages: List[dict]) -> str:
        """Create summary of messages."""
        summary = f"Previous conversation ({len(messages)} exchanges):\n"

        # Extract key topics
        all_text = " ".join(
            m["user"] + " " + m["agent"]
            for m in messages
        )

        # Simple keyword extraction
        words = all_text.split()
        word_freq = {}
        for word in words:
            if len(word) > 4:  # Only longer words
                word_lower = word.lower()
                word_freq[word_lower] = word_freq.get(word_lower, 0) + 1

        # Top keywords
        top_words = sorted(
            word_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        summary += "Topics discussed: " + ", ".join(w for w, c in top_words)
        return summary

    def get_context(self) -> str:
        """Get conversation history as context."""
        parts = []

        if self.summary:
            parts.append(self.summary)
            parts.append("\n--- Recent Conversation ---")

        for msg in self.messages:
            parts.append(f"User: {msg['user']}")
            parts.append(f"Assistant: {msg['agent']}")

        return "\n\n".join(parts)
```

## Context Overflow Handling

What to do when context is about to overflow.

### Overflow Handler

```python
class ContextOverflowHandler:
    """Handle context overflow scenarios."""

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.overflow_count = 0

    def handle_overflow(
        self,
        current_context: str,
        strategy: str = "compress"
    ) -> str:
        """Handle context overflow with specified strategy."""

        current_tokens = len(current_context) // 4

        if current_tokens <= self.max_tokens:
            return current_context

        self.overflow_count += 1

        if strategy == "compress":
            return self._compress_context(current_context)
        elif strategy == "truncate_old":
            return self._truncate_old(current_context)
        elif strategy == "truncate_middle":
            return self._truncate_middle(current_context)
        elif strategy == "summarize":
            return self._summarize_context(current_context)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _compress_context(self, context: str) -> str:
        """Compress context to fit."""
        target_chars = self.max_tokens * 4

        # Split into sections
        sections = context.split("\n\n")

        # Compress each section proportionally
        compression_ratio = target_chars / len(context)

        compressed_sections = []
        for section in sections:
            target_len = int(len(section) * compression_ratio)
            if target_len < len(section):
                # Take first part of section
                compressed_sections.append(section[:target_len] + "...")
            else:
                compressed_sections.append(section)

        return "\n\n".join(compressed_sections)

    def _truncate_old(self, context: str) -> str:
        """Remove oldest content."""
        target_chars = self.max_tokens * 4

        if len(context) <= target_chars:
            return context

        # Keep most recent content
        truncated = context[-target_chars:]
        return "[...earlier content truncated...]\n\n" + truncated

    def _truncate_middle(self, context: str) -> str:
        """Remove middle content, keep start and end."""
        target_chars = self.max_tokens * 4

        if len(context) <= target_chars:
            return context

        # Keep first and last portions
        keep_each = target_chars // 2

        start = context[:keep_each]
        end = context[-keep_each:]

        return start + "\n\n[...middle content truncated...]\n\n" + end

    def _summarize_context(self, context: str) -> str:
        """Summarize context (simplified version)."""
        # In production, use LLM to summarize

        lines = context.split("\n")

        # Keep important lines
        important_lines = []
        for line in lines:
            # Keep lines with keywords
            if any(kw in line.lower() for kw in [
                "user:", "assistant:", "tool:", "result:", "error:"
            ]):
                important_lines.append(line)

        summary = "\n".join(important_lines)

        # If still too long, truncate
        target_chars = self.max_tokens * 4
        if len(summary) > target_chars:
            summary = summary[:target_chars] + "..."

        return "[Context summarized]\n\n" + summary

    def get_overflow_stats(self) -> dict:
        """Get overflow statistics."""
        return {
            "overflow_count": self.overflow_count,
            "max_tokens": self.max_tokens
        }
```

## Context-Aware Prompting

Adjust prompting based on available context.

### Context-Aware Prompter

```python
class ContextAwarePrompter:
    """Generate prompts adapted to context availability."""

    def generate_prompt(
        self,
        task: str,
        available_context: dict
    ) -> str:
        """Generate prompt based on available context."""

        prompt_parts = []

        # Assess what context we have
        has_history = len(available_context.get("history", [])) > 0
        has_tools = len(available_context.get("tool_outputs", [])) > 0
        has_docs = len(available_context.get("documents", [])) > 0

        # Adjust instructions based on context
        if has_history:
            prompt_parts.append(
                "Continue the conversation considering previous exchanges."
            )
        else:
            prompt_parts.append(
                "Start a new conversation. You have no prior context."
            )

        if has_tools:
            prompt_parts.append(
                "Use the tool outputs provided to inform your response."
            )

        if has_docs:
            prompt_parts.append(
                "Reference the documents provided when relevant."
            )

        # Add the actual task
        prompt_parts.append(f"\nTask: {task}")

        # If context is limited, be explicit
        total_context_size = sum(
            len(str(v)) for v in available_context.values()
        )

        if total_context_size < 1000:  # Limited context
            prompt_parts.append(
                "\nNote: Context is limited. If you need more information, "
                "please ask specific questions or use available tools."
            )

        return "\n\n".join(prompt_parts)

    def adapt_for_long_context(self, context: str) -> str:
        """Adapt prompt when context is very long."""
        return """
The context provided is extensive. Focus on:
1. The most recent information
2. Information directly relevant to the current task
3. Any explicit user requests or questions

You don't need to reference all provided context in your response.
"""

    def adapt_for_short_context(self) -> str:
        """Adapt prompt when context is minimal."""
        return """
Context is limited. If you need additional information:
1. Ask clarifying questions
2. Use available tools to gather information
3. Be explicit about what assumptions you're making
"""
```

## Best Practices

Guidelines for effective context management.

### 1. Always Reserve Space for Response

```python
def build_context_with_buffer(
    components: List[str],
    max_tokens: int,
    response_buffer: int = 1000
) -> str:
    """Build context always reserving space for response."""

    available_tokens = max_tokens - response_buffer

    # Build context within available budget
    # ...

    return context
```

### 2. Prioritize Critical Information

```python
def build_with_priorities():
    """Always include critical context."""

    # CRITICAL (always include)
    critical = [
        system_prompt,
        current_user_input,
        active_tool_outputs
    ]

    # OPTIONAL (include if space)
    optional = [
        old_conversation,
        background_docs
    ]
```

### 3. Compress Proactively

```python
def compress_before_adding(content: str, max_size: int) -> str:
    """Compress content before adding to context."""

    if len(content) // 4 > max_size:
        # Compress first
        compressor = TokenCompressor()
        content = compressor.compress_to_tokens(content, max_size)

    return content
```

### 4. Monitor Context Usage

```python
class ContextMonitor:
    """Monitor context usage over time."""

    def __init__(self):
        self.usage_history = []

    def log_usage(self, usage: dict):
        """Log context usage."""
        self.usage_history.append({
            "timestamp": time.time(),
            **usage
        })

    def get_average_usage(self) -> float:
        """Get average context utilization."""
        if not self.usage_history:
            return 0.0

        return sum(
            u.get("utilization", 0)
            for u in self.usage_history
        ) / len(self.usage_history)

    def alert_if_high_usage(self, threshold: float = 0.9):
        """Alert if usage is consistently high."""
        recent = self.usage_history[-10:]

        if not recent:
            return

        avg = sum(u.get("utilization", 0) for u in recent) / len(recent)

        if avg > threshold:
            print(f"WARNING: Context usage at {avg:.1%}")
```

### 5. Test with Edge Cases

```python
def test_context_management():
    """Test context management with edge cases."""

    manager = ContextBuilder(max_tokens=1000)

    # Test 1: Empty context
    context = manager.build("")
    assert len(context) // 4 < 1000

    # Test 2: Exactly at limit
    large_input = "word " * 250  # ~1000 tokens
    context = manager.build(large_input)
    assert len(context) // 4 <= 1000

    # Test 3: Way over limit
    huge_input = "word " * 10000
    context = manager.build(huge_input)
    assert len(context) // 4 <= 1000

    # Test 4: Multiple components
    for i in range(50):
        manager.history.add_message("user", f"Message {i}")
    context = manager.build("current query")
    assert len(context) // 4 <= 1000
```

### 6. Provide Context Summary to User

```python
def show_context_summary(context_manager):
    """Show user what context is being used."""

    summary = context_manager.get_summary()

    print("Context being used:")
    print(f"- Conversation: {summary['conversation_messages']} messages")
    print(f"- Tool outputs: {summary['tool_outputs']} results")
    print(f"- Documents: {summary['documents']} items")
    print(f"- Total: {summary['total_tokens']} tokens ({summary['utilization']:.1%})")
```

## Summary

Context management is fundamental to agent reliability and performance. Key principles:

**Core Concepts**:

- Context is everything the LLM sees when generating a response
- Context windows are limited and fill up quickly
- Poor management leads to lost information and degraded performance
- Good management ensures the right information at the right time

**Components**:

- System prompts (static, critical)
- Conversation history (growing, variable importance)
- Tool outputs (large, time-sensitive)
- Retrieved information (dynamic, relevant)
- Current state (ephemeral, task-specific)

**Strategies**:

- Full context: Use everything when it fits
- Sliding window: Keep recent N messages
- Summarization: Compress old, keep recent full
- Relevance-based: Include only relevant context
- Hierarchical: Organize in priority layers

**Techniques**:

- Compression: Reduce size while preserving meaning
- Prioritization: Keep most important content
- Rolling windows: Maintain fixed-size buffer
- Retrieval: Fetch relevant context on demand
- Overflow handling: Gracefully manage limits

**Best Practices**:

- Always reserve space for responses
- Prioritize critical information
- Compress proactively
- Monitor usage
- Test edge cases
- Be transparent with users

Effective context management is the difference between an agent that works for 3 exchanges and one that works for 300. Master it.

## Next Steps

Now that you understand context management, explore:

- **[State Basics](state-basics.md)** - Maintaining state across interactions
- **[Simple Agent Loops](simple-agent-loops.md)** - Integrating context into loops
- **[Basic Error Handling](basic-error-handling.md)** - Handling context-related errors
- **Memory Systems** - Long-term context storage and retrieval
- **RAG Patterns** - Augmenting context with retrieval

**Practice exercises**:

1. Implement a complete context manager with compression
2. Build a rolling window system with adaptive sizing
3. Create a priority-based context builder
4. Implement semantic compression using embeddings
5. Build a context overflow handler with multiple strategies

**Advanced topics**:

- Semantic chunking for better context division
- Cross-session context persistence
- Distributed context management
- Context caching strategies
- Multi-modal context handling

Context is memory. Manage it well, and your agent remembers what matters.
