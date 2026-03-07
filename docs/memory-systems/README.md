# Memory Systems

## About This Section

This section covers how agents remember: storing information, retrieving relevant context, and maintaining state across interactions. Memory is what enables agents to learn from experience, maintain coherence over long conversations, and operate effectively beyond the constraints of context windows. Understanding memory architectures is crucial for building agents that can handle complex, long-running tasks.

## Contents

### [Memory Types and Architecture](memory-architecture.md)

Overview of different memory types: short-term (working memory, context window), long-term (persistent storage), episodic (experiences and events), and semantic (facts and knowledge). Covers how these memory types work together in agent systems.

### [Short-Term Memory](short-term-memory.md)

Conversation history, working memory, and context window management. Covers context limitations, recency effects, managing token budgets, and strategies for maintaining relevant context in the immediate conversation.

### [Long-Term Memory](long-term-memory.md)

Persistent memory systems for knowledge storage and retrieval. Covers vector databases, traditional databases, file systems, and hybrid approaches. Includes when to persist information and retrieval strategies.

### [Episodic Memory](episodic-memory.md)

Remembering experiences, events, and temporal sequences. Covers storing interaction history, retrieving past experiences, learning from previous attempts, and maintaining temporal awareness.

### [Semantic Memory](semantic-memory.md)

Factual knowledge, learned concepts, and structured information. Covers knowledge graphs, entity memory, fact storage, and managing accumulated knowledge about domains and entities.

### [Memory Retrieval](memory-retrieval.md)

Strategies for finding relevant information: semantic search, keyword search, recency-based retrieval, importance weighting, and hybrid retrieval approaches. Covers balancing relevance vs. recency vs. importance.

### [Memory Consolidation](memory-consolidation.md)

Moving information from short-term to long-term memory: when to persist, what to persist, summarization strategies, and managing the transition from working memory to permanent storage.
