# MCP Context Management

## Table of Contents

- [Introduction](#introduction)
- [What Is Context in MCP?](#what-is-context-in-mcp)
- [Context Sources](#context-sources)
- [Context Priority](#context-priority)
- [Combining Multiple Context Providers](#combining-multiple-context-providers)
- [Context Windows and Budgets](#context-windows-and-budgets)
- [Context Selection Strategies](#context-selection-strategies)
- [Dynamic Context](#dynamic-context)
- [Context Caching](#context-caching)
- [Context Tracking](#context-tracking)
- [Context Quality](#context-quality)
- [Advanced Patterns](#advanced-patterns)
- [Context Optimization](#context-optimization)
- [Best Practices](#best-practices)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

In agentic AI, **context is everything**. LLMs need the right information at the right time to perform effectively. MCP provides a **standardized way** to aggregate, prioritize, and manage context from multiple sources.

> "Context management is the art of giving the LLM exactly what it needs, nothing more, nothing less."

MCP transforms context management from ad-hoc approaches to a **systematic protocol**:

- **Multiple sources** - Aggregate from files, databases, APIs, memory
- **Intelligent prioritization** - Surface most relevant context
- **Budget management** - Fit within token limits
- **Dynamic updates** - Context evolves as conversation progresses

## What Is Context in MCP?

### Definition

**Context** is information provided to an LLM to inform its responses:

```python
# Context includes:
context = {
    "conversation_history": [...],      # Prior messages
    "relevant_documents": [...],        # Retrieved information
    "system_state": {...},             # Application state
    "user_preferences": {...},         # Personalization
    "tool_results": [...],             # Tool outputs
    "resources": [...]                 # MCP resources
}
```

### Context vs Prompts vs Resources

Clear distinction:

| Concept      | Purpose                 | Example                        |
| ------------ | ----------------------- | ------------------------------ |
| **Resource** | Data source definition  | "file://docs/manual.pdf"       |
| **Prompt**   | Instruction template    | "Summarize this document..."   |
| **Context**  | Actual data sent to LLM | The PDF content + instructions |

**Context** is the assembled information package sent to the LLM.

### The Context Challenge

Without MCP:

```python
# Manually gathering context
doc1 = read_file("file1.txt")
doc2 = fetch_from_db(query)
doc3 = api_call(endpoint)
history = get_conversation()

# How to combine? How to prioritize? How to fit in token limit?
context = doc1 + doc2 + doc3 + history  # ❌ Naive approach
```

With MCP:

```python
# MCP aggregates and manages context
context = await mcp_client.aggregate_context(
    sources=["file-server", "db-server", "api-server"],
    query="user's question",
    max_tokens=8000
)

# ✅ Intelligent aggregation with prioritization
```

## Context Sources

### MCP Resources as Context

```python
class ContextFromResources:
    """Build context from MCP resources"""

    async def get_resource_context(self, resource_uris):
        """Fetch resources as context"""
        contexts = []

        for uri in resource_uris:
            # Read resource
            resource = await self.mcp.read_resource(uri)

            # Convert to context
            contexts.append({
                "source": uri,
                "content": resource['contents'][0]['text'],
                "metadata": resource.get('metadata', {})
            })

        return contexts

# Usage
contexts = await context_mgr.get_resource_context([
    "file:///docs/manual.pdf",
    "db://products/inventory",
    "api://weather/current"
])
```

### Conversation History as Context

```python
class ConversationContext:
    """Manage conversation history context"""

    def __init__(self):
        self.messages = []

    def add_message(self, role, content):
        """Add message to history"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })

    def get_context(self, max_messages=None, max_tokens=None):
        """Get conversation context within constraints"""

        if max_messages:
            # Take most recent N messages
            recent = self.messages[-max_messages:]
        else:
            recent = self.messages

        if max_tokens:
            # Truncate to fit token budget
            recent = self.fit_to_token_budget(recent, max_tokens)

        return recent

    def fit_to_token_budget(self, messages, max_tokens):
        """Truncate messages to fit token budget"""
        total_tokens = 0
        result = []

        # Start from most recent
        for msg in reversed(messages):
            msg_tokens = count_tokens(msg['content'])

            if total_tokens + msg_tokens <= max_tokens:
                result.insert(0, msg)
                total_tokens += msg_tokens
            else:
                break

        return result
```

### Tool Results as Context

```python
class ToolResultContext:
    """Context from tool invocations"""

    def __init__(self):
        self.tool_results = []

    async def execute_tool_for_context(self, tool_name, arguments):
        """Execute tool and capture result as context"""

        result = await self.mcp.call_tool(tool_name, arguments)

        context = {
            "source": f"tool:{tool_name}",
            "content": result['content'][0]['text'],
            "timestamp": datetime.now(),
            "relevance_score": self.score_relevance(result)
        }

        self.tool_results.append(context)
        return context

    def get_relevant_tool_results(self, query, top_k=5):
        """Get most relevant tool results"""
        # Score each result against query
        scored = [
            (result, self.similarity(query, result['content']))
            for result in self.tool_results
        ]

        # Sort by relevance
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top K
        return [result for result, score in scored[:top_k]]
```

### External Knowledge as Context

```python
class ExternalKnowledgeContext:
    """Context from external knowledge bases"""

    async def retrieve_knowledge(self, query, sources):
        """Retrieve relevant knowledge from multiple sources"""

        results = await asyncio.gather(*[
            self.search_source(source, query)
            for source in sources
        ])

        # Combine and rank results
        all_results = []
        for source_results in results:
            all_results.extend(source_results)

        # Rank by relevance
        ranked = self.rank_by_relevance(query, all_results)

        return ranked

# Usage
knowledge = await context_mgr.retrieve_knowledge(
    query="How does photosynthesis work?",
    sources=["wikipedia", "pubmed", "textbooks"]
)
```

## Context Priority

### Relevance Scoring

```python
class ContextPrioritizer:
    """Prioritize context by relevance"""

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    async def score_relevance(self, query, context_items):
        """Score each context item for relevance to query"""

        # Get query embedding
        query_embedding = await self.embedding_model.embed(query)

        scored_items = []
        for item in context_items:
            # Get item embedding
            item_embedding = await self.embedding_model.embed(
                item['content']
            )

            # Compute similarity
            similarity = cosine_similarity(query_embedding, item_embedding)

            scored_items.append({
                "item": item,
                "relevance_score": similarity
            })

        # Sort by relevance
        scored_items.sort(key=lambda x: x['relevance_score'], reverse=True)

        return scored_items

# Usage
scored = await prioritizer.score_relevance(
    query="What are the system requirements?",
    context_items=[doc1, doc2, doc3, history]
)
```

### Multi-Factor Prioritization

```python
class MultiFactor Prioritizer:
    """Prioritize using multiple factors"""

    def compute_priority(self, context_item, query, weights=None):
        """Compute priority score from multiple factors"""

        if weights is None:
            weights = {
                'relevance': 0.4,
                'recency': 0.2,
                'importance': 0.2,
                'source_trust': 0.2
            }

        # Compute individual scores
        relevance = self.relevance_score(query, context_item)
        recency = self.recency_score(context_item)
        importance = self.importance_score(context_item)
        trust = self.trust_score(context_item['source'])

        # Weighted combination
        priority = (
            weights['relevance'] * relevance +
            weights['recency'] * recency +
            weights['importance'] * importance +
            weights['source_trust'] * trust
        )

        return priority

    def relevance_score(self, query, item):
        """Semantic relevance to query"""
        return cosine_similarity(
            embed(query),
            embed(item['content'])
        )

    def recency_score(self, item):
        """Recency score (newer = higher)"""
        age = datetime.now() - item['timestamp']
        # Exponential decay
        return math.exp(-age.total_seconds() / (24 * 3600))

    def importance_score(self, item):
        """Intrinsic importance"""
        # Could be based on source, length, explicit marking, etc.
        return item.get('importance', 0.5)

    def trust_score(self, source):
        """Trust level of source"""
        trust_levels = {
            'user_input': 1.0,
            'verified_db': 0.9,
            'internal_api': 0.8,
            'external_api': 0.6,
            'web_search': 0.5
        }
        return trust_levels.get(source, 0.5)
```

### Priority Tiers

```python
class TieredContext:
    """Organize context into priority tiers"""

    TIERS = {
        'critical': 1,
        'high': 2,
        'medium': 3,
        'low': 4
    }

    def __init__(self):
        self.context_by_tier = {
            tier: [] for tier in self.TIERS.keys()
        }

    def add_context(self, content, tier='medium'):
        """Add context to a tier"""
        self.context_by_tier[tier].append(content)

    def get_context_by_budget(self, token_budget):
        """Get context fitting in budget, prioritizing higher tiers"""
        result = []
        remaining_budget = token_budget

        # Process tiers in priority order
        for tier in sorted(self.TIERS.keys(), key=lambda t: self.TIERS[t]):
            for item in self.context_by_tier[tier]:
                item_tokens = count_tokens(item)

                if item_tokens <= remaining_budget:
                    result.append(item)
                    remaining_budget -= item_tokens

                if remaining_budget <= 0:
                    return result

        return result

# Usage
context = TieredContext()

context.add_context("User's direct question", tier='critical')
context.add_context("Recent conversation history", tier='high')
context.add_context("Relevant documentation", tier='medium')
context.add_context("General background info", tier='low')

# Get context within budget
final_context = context.get_context_by_budget(token_budget=4000)
```

## Combining Multiple Context Providers

### Context Aggregation

```python
class ContextAggregator:
    """Aggregate context from multiple MCP servers"""

    def __init__(self, mcp_client):
        self.mcp = mcp_client
        self.providers = {}

    def register_provider(self, name, server_id, weight=1.0):
        """Register a context provider"""
        self.providers[name] = {
            'server_id': server_id,
            'weight': weight
        }

    async def aggregate_context(self, query, max_tokens):
        """Aggregate context from all providers"""

        # Gather context from all providers
        provider_contexts = await asyncio.gather(*[
            self.get_provider_context(name, query)
            for name in self.providers.keys()
        ])

        # Flatten and score
        all_contexts = []
        for provider_name, contexts in zip(self.providers.keys(), provider_contexts):
            provider_weight = self.providers[provider_name]['weight']

            for ctx in contexts:
                ctx['priority'] *= provider_weight
                ctx['provider'] = provider_name
                all_contexts.append(ctx)

        # Sort by priority
        all_contexts.sort(key=lambda x: x['priority'], reverse=True)

        # Select contexts within token budget
        selected = self.select_within_budget(all_contexts, max_tokens)

        return selected

    async def get_provider_context(self, provider_name, query):
        """Get context from specific provider"""
        provider = self.providers[provider_name]

        # List relevant resources
        resources = await self.mcp.list_resources(provider['server_id'])

        # Score resources for relevance
        scored = [
            {
                'uri': r['uri'],
                'priority': self.score_resource(r, query)
            }
            for r in resources
        ]

        # Fetch top resources
        top_resources = sorted(scored, key=lambda x: x['priority'], reverse=True)[:5]

        contexts = []
        for resource in top_resources:
            content = await self.mcp.read_resource(resource['uri'])
            contexts.append({
                'content': content,
                'priority': resource['priority']
            })

        return contexts

# Usage
aggregator = ContextAggregator(mcp_client)

aggregator.register_provider("documentation", "docs-server", weight=1.0)
aggregator.register_provider("codebase", "code-server", weight=0.8)
aggregator.register_provider("history", "history-server", weight=0.6)

context = await aggregator.aggregate_context(
    query="How do I configure authentication?",
    max_tokens=6000
)
```

### Context Fusion

```python
class ContextFusion:
    """Fuse overlapping context from multiple sources"""

    async def fuse_contexts(self, contexts):
        """Combine contexts, handling overlaps"""

        # Detect overlapping content
        overlap_groups = self.detect_overlaps(contexts)

        fused = []
        for group in overlap_groups:
            if len(group) == 1:
                # No overlap, include as-is
                fused.append(group[0])
            else:
                # Multiple sources with overlapping content
                # Merge into single context with combined metadata
                merged = self.merge_overlapping(group)
                fused.append(merged)

        return fused

    def detect_overlaps(self, contexts):
        """Detect overlapping contexts"""
        # Group contexts with high similarity
        groups = []
        processed = set()

        for i, ctx1 in enumerate(contexts):
            if i in processed:
                continue

            group = [ctx1]
            processed.add(i)

            for j, ctx2 in enumerate(contexts[i+1:], start=i+1):
                if j in processed:
                    continue

                similarity = self.compute_similarity(
                    ctx1['content'],
                    ctx2['content']
                )

                if similarity > 0.8:  # High overlap
                    group.append(ctx2)
                    processed.add(j)

            groups.append(group)

        return groups

    def merge_overlapping(self, contexts):
        """Merge overlapping contexts"""
        # Take longest version
        longest = max(contexts, key=lambda c: len(c['content']))

        # Combine metadata from all sources
        merged = {
            'content': longest['content'],
            'sources': [c['source'] for c in contexts],
            'priority': max(c['priority'] for c in contexts),
            'timestamp': max(c.get('timestamp', 0) for c in contexts)
        }

        return merged
```

### Context Deduplication

```python
class ContextDeduplicator:
    """Remove duplicate context"""

    def deduplicate(self, contexts):
        """Remove duplicate contexts"""
        unique = []
        seen_content = set()

        for ctx in contexts:
            # Create content signature
            signature = self.compute_signature(ctx['content'])

            if signature not in seen_content:
                unique.append(ctx)
                seen_content.add(signature)
            else:
                # Duplicate found - merge metadata
                self.merge_into_existing(unique, ctx)

        return unique

    def compute_signature(self, content):
        """Compute content signature for deduplication"""
        # Normalize content
        normalized = content.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)

        # Hash for efficient comparison
        return hashlib.md5(normalized.encode()).hexdigest()

    def merge_into_existing(self, unique_list, duplicate):
        """Merge duplicate's metadata into existing entry"""
        signature = self.compute_signature(duplicate['content'])

        for item in unique_list:
            if self.compute_signature(item['content']) == signature:
                # Found the original - enhance with duplicate's metadata
                if 'sources' not in item:
                    item['sources'] = [item.get('source')]
                item['sources'].append(duplicate.get('source'))

                # Take higher priority
                item['priority'] = max(
                    item.get('priority', 0),
                    duplicate.get('priority', 0)
                )
                break
```

## Context Windows and Budgets

### Token Budget Management

```python
class TokenBudgetManager:
    """Manage context within token budget"""

    def __init__(self, model_max_tokens, response_buffer=1000):
        self.model_max_tokens = model_max_tokens
        self.response_buffer = response_buffer
        self.available_for_context = model_max_tokens - response_buffer

    def select_context(self, candidates, prioritizer):
        """Select contexts that fit in budget"""
        # Sort by priority
        sorted_candidates = sorted(
            candidates,
            key=lambda c: prioritizer.compute_priority(c),
            reverse=True
        )

        selected = []
        total_tokens = 0

        for candidate in sorted_candidates:
            candidate_tokens = count_tokens(candidate['content'])

            if total_tokens + candidate_tokens <= self.available_for_context:
                selected.append(candidate)
                total_tokens += candidate_tokens
            else:
                # Try truncating
                if self.can_truncate(candidate):
                    truncated = self.truncate_to_fit(
                        candidate,
                        self.available_for_context - total_tokens
                    )
                    if truncated:
                        selected.append(truncated)
                        break

        return selected, total_tokens

    def can_truncate(self, context):
        """Check if context can be meaningfully truncated"""
        return context.get('truncatable', False)

    def truncate_to_fit(self, context, max_tokens):
        """Truncate context to fit in remaining budget"""
        content = context['content']

        # Binary search for truncation point
        tokens = self.tokenize(content)

        if len(tokens) <= max_tokens:
            return context

        # Truncate and add indicator
        truncated_tokens = tokens[:max_tokens-10]  # Leave room for indicator
        truncated_content = self.detokenize(truncated_tokens)
        truncated_content += "\n\n[Content truncated...]"

        return {
            **context,
            'content': truncated_content,
            'truncated': True
        }
```

### Dynamic Budget Allocation

```python
class DynamicBudgetAllocator:
    """Dynamically allocate budget across context types"""

    def __init__(self, total_budget):
        self.total_budget = total_budget

    def allocate_budget(self, context_types):
        """Allocate budget based on context importance"""

        # Default allocations
        allocations = {
            'conversation_history': 0.3,
            'relevant_documents': 0.4,
            'tool_results': 0.2,
            'system_state': 0.1
        }

        # Adjust based on what's available
        available_types = [t for t in context_types if t in allocations]
        total_weight = sum(allocations[t] for t in available_types)

        # Normalize and convert to tokens
        budget_by_type = {
            ctx_type: int(self.total_budget * (allocations[ctx_type] / total_weight))
            for ctx_type in available_types
        }

        return budget_by_type

    def reallocate_unused(self, usage_by_type, budget_by_type):
        """Reallocate unused budget to other types"""
        total_unused = sum(
            budget_by_type[t] - usage_by_type.get(t, 0)
            for t in budget_by_type.keys()
        )

        if total_unused > 0:
            # Redistribute to types that want more
            for ctx_type in budget_by_type.keys():
                if usage_by_type.get(ctx_type, 0) >= budget_by_type[ctx_type]:
                    # This type used all its budget, give it more
                    additional = total_unused // len(budget_by_type)
                    budget_by_type[ctx_type] += additional

        return budget_by_type

# Usage
allocator = DynamicBudgetAllocator(total_budget=8000)

budget_by_type = allocator.allocate_budget([
    'conversation_history',
    'relevant_documents',
    'tool_results'
])

# Later, after initial allocation:
usage = {
    'conversation_history': 1500,  # Used less than allocated
    'relevant_documents': 3200,    # Used all allocated
    'tool_results': 1600           # Used all allocated
}

# Reallocate unused budget
budget_by_type = allocator.reallocate_unused(usage, budget_by_type)
```

### Sliding Window Context

```python
class SlidingWindowContext:
    """Maintain sliding window of context"""

    def __init__(self, window_size):
        self.window_size = window_size
        self.items = deque(maxlen=window_size)

    def add(self, context_item):
        """Add item to window (oldest drops out)"""
        self.items.append(context_item)

    def get_window(self, max_tokens=None):
        """Get current window contents"""
        if not max_tokens:
            return list(self.items)

        # Fit within token budget
        result = []
        total_tokens = 0

        # Start from most recent
        for item in reversed(self.items):
            item_tokens = count_tokens(item['content'])

            if total_tokens + item_tokens <= max_tokens:
                result.insert(0, item)
                total_tokens += item_tokens
            else:
                break

        return result

    def summarize_old_context(self, llm, keep_recent=5):
        """Summarize old context to save tokens"""
        if len(self.items) <= keep_recent:
            return

        # Take old items
        old_items = list(self.items)[:-keep_recent]

        # Summarize
        summary = llm.summarize([item['content'] for item in old_items])

        # Replace old items with summary
        for _ in range(len(old_items)):
            self.items.popleft()

        self.items.appendleft({
            'content': summary,
            'type': 'summary',
            'timestamp': datetime.now()
        })
```

## Context Selection Strategies

### Greedy Selection

```python
def greedy_context_selection(candidates, max_tokens):
    """Greedily select highest priority contexts"""

    # Sort by priority
    sorted_candidates = sorted(
        candidates,
        key=lambda c: c['priority'],
        reverse=True
    )

    selected = []
    total_tokens = 0

    for candidate in sorted_candidates:
        tokens = count_tokens(candidate['content'])

        if total_tokens + tokens <= max_tokens:
            selected.append(candidate)
            total_tokens += tokens

    return selected, total_tokens
```

### Knapsack-Based Selection

```python
def knapsack_context_selection(candidates, max_tokens):
    """Optimal context selection (0/1 knapsack)"""

    n = len(candidates)

    # DP table: dp[i][w] = max value using first i items with weight limit w
    dp = [[0] * (max_tokens + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        candidate = candidates[i - 1]
        tokens = count_tokens(candidate['content'])
        value = candidate['priority']

        for w in range(max_tokens + 1):
            # Don't include this item
            dp[i][w] = dp[i - 1][w]

            # Include this item if it fits
            if tokens <= w:
                dp[i][w] = max(
                    dp[i][w],
                    dp[i - 1][w - tokens] + value
                )

    # Backtrack to find selected items
    selected = []
    w = max_tokens
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(candidates[i - 1])
            w -= count_tokens(candidates[i - 1]['content'])

    selected.reverse()
    return selected, sum(count_tokens(c['content']) for c in selected)
```

### Diversity-Aware Selection

```python
class DiversityAwareSelector:
    """Select diverse context to avoid redundancy"""

    def select_diverse_context(self, candidates, max_tokens, diversity_weight=0.3):
        """Select contexts balancing priority and diversity"""

        selected = []
        total_tokens = 0

        while candidates and total_tokens < max_tokens:
            # Score each candidate
            scores = []
            for candidate in candidates:
                tokens = count_tokens(candidate['content'])

                if total_tokens + tokens > max_tokens:
                    continue

                # Priority score
                priority_score = candidate['priority']

                # Diversity score (dissimilarity to already selected)
                diversity_score = self.compute_diversity(candidate, selected)

                # Combined score
                combined = (
                    (1 - diversity_weight) * priority_score +
                    diversity_weight * diversity_score
                )

                scores.append((candidate, combined, tokens))

            if not scores:
                break

            # Select best
            best_candidate, best_score, tokens = max(scores, key=lambda x: x[1])

            selected.append(best_candidate)
            total_tokens += tokens
            candidates.remove(best_candidate)

        return selected, total_tokens

    def compute_diversity(self, candidate, selected):
        """Compute how different candidate is from selected items"""
        if not selected:
            return 1.0  # Maximally diverse if nothing selected yet

        # Average dissimilarity to selected items
        dissimilarities = [
            1 - self.similarity(candidate['content'], s['content'])
            for s in selected
        ]

        return sum(dissimilarities) / len(dissimilarities)

    def similarity(self, text1, text2):
        """Compute similarity between texts"""
        # Simple word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0
```

## Dynamic Context

### Adaptive Context Updates

```python
class AdaptiveContextManager:
    """Dynamically update context during conversation"""

    def __init__(self, mcp_client):
        self.mcp = mcp_client
        self.current_context = []
        self.conversation_history = []

    async def update_context(self, new_message, llm_response=None):
        """Update context based on conversation progression"""

        # Add to history
        self.conversation_history.append(new_message)
        if llm_response:
            self.conversation_history.append(llm_response)

        # Analyze what context is needed
        needed_context = await self.analyze_context_needs(
            self.conversation_history
        )

        # Fetch new context if needed
        if needed_context:
            new_contexts = await self.fetch_context(needed_context)
            self.current_context.extend(new_contexts)

        # Remove stale context
        self.current_context = self.remove_stale(self.current_context)

        # Reprioritize
        self.current_context = self.reprioritize(
            self.current_context,
            new_message
        )

    async def analyze_context_needs(self, history):
        """Analyze what context is needed"""
        # Extract entities, topics, questions from recent messages
        recent = history[-3:]

        needs = {
            'entities': self.extract_entities(recent),
            'topics': self.extract_topics(recent),
            'questions': self.extract_questions(recent)
        }

        return needs

    def remove_stale(self, contexts):
        """Remove context that's no longer relevant"""
        current_time = datetime.now()

        return [
            ctx for ctx in contexts
            if self.is_still_relevant(ctx, current_time)
        ]

    def is_still_relevant(self, context, current_time):
        """Check if context is still relevant"""
        # Time-based decay
        age = current_time - context['timestamp']
        if age.total_seconds() > 3600:  # 1 hour
            return False

        # Topic-based relevance
        current_topics = self.extract_topics(self.conversation_history[-3:])
        context_topics = self.extract_topics([context['content']])

        overlap = set(current_topics) & set(context_topics)
        return len(overlap) > 0
```

### Context Subscription Updates

```python
class SubscriptionContextManager:
    """Manage context from MCP resource subscriptions"""

    def __init__(self, mcp_client):
        self.mcp = mcp_client
        self.subscriptions = {}
        self.context_updates = asyncio.Queue()

    async def subscribe_to_resource(self, resource_uri):
        """Subscribe to resource updates"""

        await self.mcp.subscribe(resource_uri)

        self.subscriptions[resource_uri] = {
            'uri': resource_uri,
            'last_update': datetime.now(),
            'content': await self.mcp.read_resource(resource_uri)
        }

    async def handle_update(self, notification):
        """Handle resource update notification"""
        resource_uri = notification['params']['uri']

        # Fetch updated content
        updated_content = await self.mcp.read_resource(resource_uri)

        # Update subscription
        self.subscriptions[resource_uri]['content'] = updated_content
        self.subscriptions[resource_uri]['last_update'] = datetime.now()

        # Queue for context refresh
        await self.context_updates.put({
            'type': 'resource_update',
            'uri': resource_uri,
            'content': updated_content
        })

    async def get_current_context(self):
        """Get current context from all subscriptions"""
        return [
            {
                'source': uri,
                'content': sub['content'],
                'timestamp': sub['last_update']
            }
            for uri, sub in self.subscriptions.items()
        ]
```

## Context Caching

### LRU Context Cache

```python
from functools import lru_cache

class ContextCache:
    """Cache frequently accessed context"""

    def __init__(self, max_size=100):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size

    def get(self, key):
        """Get cached context"""
        if key in self.cache:
            self.access_times[key] = datetime.now()
            return self.cache[key]
        return None

    def put(self, key, value):
        """Cache context"""
        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            self.evict_lru()

        self.cache[key] = value
        self.access_times[key] = datetime.now()

    def evict_lru(self):
        """Evict least recently used"""
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]

# Usage
cache = ContextCache()

async def get_context_with_cache(resource_uri):
    """Get context with caching"""

    # Check cache
    cached = cache.get(resource_uri)
    if cached:
        return cached

    # Fetch from source
    context = await mcp.read_resource(resource_uri)

    # Cache it
    cache.put(resource_uri, context)

    return context
```

### Semantic Context Cache

```python
class SemanticContextCache:
    """Cache context based on semantic similarity"""

    def __init__(self, embedding_model, similarity_threshold=0.9):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.cache = []

    async def get(self, query):
        """Get cached context for similar query"""

        if not self.cache:
            return None

        query_embedding = await self.embedding_model.embed(query)

        # Find most similar cached query
        best_match = None
        best_similarity = 0

        for cached_item in self.cache:
            similarity = cosine_similarity(
                query_embedding,
                cached_item['query_embedding']
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = cached_item

        # Return if similar enough
        if best_similarity >= self.similarity_threshold:
            return best_match['context']

        return None

    async def put(self, query, context):
        """Cache context with query embedding"""
        query_embedding = await self.embedding_model.embed(query)

        self.cache.append({
            'query': query,
            'query_embedding': query_embedding,
            'context': context,
            'timestamp': datetime.now()
        })

        # Limit cache size
        if len(self.cache) > 50:
            self.cache.pop(0)
```

## Context Tracking

### Context Provenance

```python
class ContextProvenance:
    """Track where context came from"""

    def __init__(self):
        self.provenance_db = {}

    def record_context(self, context_id, source, query, timestamp):
        """Record context provenance"""
        self.provenance_db[context_id] = {
            'source': source,
            'query': query,
            'timestamp': timestamp,
            'access_count': 0
        }

    def access(self, context_id):
        """Record context access"""
        if context_id in self.provenance_db:
            self.provenance_db[context_id]['access_count'] += 1

    def get_provenance(self, context_id):
        """Get provenance information"""
        return self.provenance_db.get(context_id)

    def audit_trail(self, start_time, end_time):
        """Get audit trail of context usage"""
        return [
            {
                'context_id': ctx_id,
                **info
            }
            for ctx_id, info in self.provenance_db.items()
            if start_time <= info['timestamp'] <= end_time
        ]
```

### Context Attribution

```python
class ContextAttribution:
    """Attribute LLM responses to context sources"""

    async def attribute_response(self, llm_response, context_items):
        """Determine which context influenced the response"""

        attributions = []

        for item in context_items:
            # Compute overlap between response and context item
            overlap = self.compute_overlap(
                llm_response,
                item['content']
            )

            if overlap > 0.1:  # Significant influence
                attributions.append({
                    'source': item['source'],
                    'influence_score': overlap,
                    'excerpts': self.extract_excerpts(
                        llm_response,
                        item['content']
                    )
                })

        # Sort by influence
        attributions.sort(key=lambda x: x['influence_score'], reverse=True)

        return attributions

    def compute_overlap(self, response, context):
        """Compute how much context influenced response"""
        # Simple word overlap
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())

        overlap = response_words & context_words

        return len(overlap) / len(response_words) if response_words else 0
```

## Context Quality

### Context Validation

```python
class ContextValidator:
    """Validate context quality"""

    def validate(self, context):
        """Validate context meets quality criteria"""
        issues = []

        # Check completeness
        if not context.get('content'):
            issues.append("Missing content")

        # Check freshness
        if 'timestamp' in context:
            age = datetime.now() - context['timestamp']
            if age.total_seconds() > 86400:  # 24 hours
                issues.append(f"Stale content (age: {age})")

        # Check length
        if len(context.get('content', '')) < 10:
            issues.append("Content too short")

        # Check encoding
        try:
            context['content'].encode('utf-8')
        except UnicodeEncodeError:
            issues.append("Invalid encoding")

        return {
            'valid': len(issues) == 0,
            'issues': issues
        }

# Usage
validator = ContextValidator()

for context in contexts:
    validation = validator.validate(context)
    if not validation['valid']:
        print(f"Invalid context: {validation['issues']}")
```

### Context Relevance Scoring

```python
class RelevanceScorer:
    """Score context relevance"""

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    async def score_batch(self, query, contexts):
        """Score multiple contexts"""
        query_embedding = await self.embedding_model.embed(query)

        scores = []
        for context in contexts:
            context_embedding = await self.embedding_model.embed(
                context['content']
            )

            similarity = cosine_similarity(query_embedding, context_embedding)

            scores.append({
                'context': context,
                'relevance_score': similarity
            })

        return sorted(scores, key=lambda x: x['relevance_score'], reverse=True)
```

## Advanced Patterns

### Hierarchical Context

```python
class HierarchicalContext:
    """Organize context hierarchically"""

    def __init__(self):
        self.hierarchy = {
            'global': [],      # Always included
            'session': [],     # For current session
            'conversation': [], # For current conversation
            'turn': []         # For current turn only
        }

    def add_context(self, content, level='conversation'):
        """Add context at specified level"""
        self.hierarchy[level].append(content)

    def get_context(self, include_levels=['global', 'session', 'conversation', 'turn']):
        """Get context from specified levels"""
        context = []
        for level in include_levels:
            context.extend(self.hierarchy[level])
        return context

    def clear_level(self, level):
        """Clear context at specified level"""
        self.hierarchy[level] = []
```

### Context Compression

```python
class ContextCompressor:
    """Compress context to save tokens"""

    def __init__(self, llm):
        self.llm = llm

    async def compress(self, contexts, target_tokens):
        """Compress contexts to target token count"""
        current_tokens = sum(count_tokens(c['content']) for c in contexts)

        if current_tokens <= target_tokens:
            return contexts

        compression_ratio = target_tokens / current_tokens

        if compression_ratio > 0.7:
            # Light compression: truncate least important
            return self.selective_truncation(contexts, target_tokens)
        else:
            # Heavy compression: summarize
            return await self.summarize_contexts(contexts, target_tokens)

    def selective_truncation(self, contexts, target_tokens):
        """Truncate lower priority contexts"""
        sorted_contexts = sorted(
            contexts,
            key=lambda c: c.get('priority', 0),
            reverse=True
        )

        result = []
        total_tokens = 0

        for ctx in sorted_contexts:
            ctx_tokens = count_tokens(ctx['content'])

            if total_tokens + ctx_tokens <= target_tokens:
                result.append(ctx)
                total_tokens += ctx_tokens
            else:
                # Truncate this one to fit
                remaining = target_tokens - total_tokens
                if remaining > 100:  # Only if meaningful space left
                    truncated = self.truncate(ctx, remaining)
                    result.append(truncated)
                break

        return result

    async def summarize_contexts(self, contexts, target_tokens):
        """Summarize contexts"""
        # Combine contexts
        combined = "\n\n".join(c['content'] for c in contexts)

        # Summarize
        summary = await self.llm.summarize(
            combined,
            max_tokens=target_tokens
        )

        return [{
            'content': summary,
            'type': 'summary',
            'original_count': len(contexts),
            'priority': max(c.get('priority', 0) for c in contexts)
        }]
```

## Context Optimization

### Query-Specific Optimization

```python
class QuerySpecificOptimizer:
    """Optimize context for specific query types"""

    def optimize_for_query(self, query, contexts):
        """Optimize context based on query type"""

        query_type = self.classify_query(query)

        if query_type == 'factual':
            return self.optimize_factual(contexts)
        elif query_type == 'analytical':
            return self.optimize_analytical(contexts)
        elif query_type == 'creative':
            return self.optimize_creative(contexts)
        else:
            return contexts

    def classify_query(self, query):
        """Classify query type"""
        query_lower = query.lower()

        if any(word in query_lower for word in ['what', 'when', 'where', 'who']):
            return 'factual'
        elif any(word in query_lower for word in ['why', 'how', 'analyze', 'explain']):
            return 'analytical'
        elif any(word in query_lower for word in ['create', 'write', 'generate', 'design']):
            return 'creative'
        else:
            return 'general'

    def optimize_factual(self, contexts):
        """Optimize for factual queries - prefer precise, authoritative sources"""
        return sorted(
            contexts,
            key=lambda c: c.get('authority_score', 0) * c.get('precision_score', 0),
            reverse=True
        )

    def optimize_analytical(self, contexts):
        """Optimize for analytical queries - prefer detailed, multi-perspective content"""
        return sorted(
            contexts,
            key=lambda c: c.get('detail_score', 0) * c.get('diversity_score', 0),
            reverse=True
        )

    def optimize_creative(self, contexts):
        """Optimize for creative queries - prefer examples and diverse inspiration"""
        return sorted(
            contexts,
            key=lambda c: c.get('example_score', 0) * c.get('inspiration_score', 0),
            reverse=True
        )
```

## Best Practices

### Context Quality Over Quantity

```python
# Bad: Include everything
all_context = [doc1, doc2, doc3, history, state, logs, ...]

# Good: Curate high-quality relevant context
quality_context = select_top_k_relevant(
    all_possible_context,
    query,
    k=5,
    min_relevance=0.7
)
```

### Clear Context Structure

```python
# Bad: Unstructured context dump
context = doc1 + "\n\n" + doc2 + "\n\n" + history

# Good: Clearly structured context
context = f"""
## Current Question
{query}

## Relevant Documentation
{doc1}

## Recent Conversation
{format_conversation(history)}

## System State
{format_state(state)}
"""
```

### Context Freshness

```python
# Check and update stale context
def refresh_context_if_needed(context, max_age_seconds=3600):
    """Refresh context if too old"""
    for item in context:
        age = (datetime.now() - item['timestamp']).total_seconds()

        if age > max_age_seconds:
            # Refresh from source
            item['content'] = fetch_fresh_content(item['source'])
            item['timestamp'] = datetime.now()

    return context
```

### Attribution and Citations

```python
# Include source information with context
context_with_sources = [
    {
        'content': content,
        'source': 'documentation/api.md',
        'section': 'Authentication',
        'last_updated': '2024-01-15'
    }
    for content, source in zip(contents, sources)
]

# LLM can cite sources in response
```

## Summary

MCP transforms context management from ad-hoc to systematic:

**Key Concepts**:

- **Context aggregation** - Combine from multiple sources
- **Prioritization** - Surface most relevant information
- **Budget management** - Fit within token limits
- **Dynamic updates** - Context evolves with conversation
- **Quality control** - Validate and optimize

**Benefits**:

- **Better responses** - LLM has right information
- **Efficiency** - Optimal token usage
- **Transparency** - Track context provenance
- **Scalability** - Handle multiple context sources

**Best Practices**:

- Quality over quantity
- Clear structure
- Fresh content
- Attribution
- Budget awareness

MCP's context management enables agents to work with rich, multi-source context effectively.

## Next Steps

- **[Building MCP Servers](building-mcp-servers.md)** - Implement context-providing servers
- **[MCP Prompts](mcp-prompts.md)** - Use prompts with context
- **[MCP Resources](mcp-resources.md)** - Understand resource-based context
- **[MCP Ecosystem](mcp-ecosystem.md)** - See how context enables the ecosystem
- **[Memory Systems](../memory-systems/memory-architecture.md)** - Long-term context management
