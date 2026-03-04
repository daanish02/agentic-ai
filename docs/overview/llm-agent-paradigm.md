# The LLM Agent Paradigm

## Table of Contents

- [The LLM as Reasoning Engine](#the-llm-as-reasoning-engine)
- [The Agent Loop](#the-agent-loop)
- [Prompt-Based Agency](#prompt-based-agency)
- [Tool Use and Function Calling](#tool-use-and-function-calling)
- [Memory and Context](#memory-and-context)
- [Key Capabilities](#key-capabilities)
- [Unique Characteristics](#unique-characteristics)
- [Summary](#summary)
- [Next Steps](#next-steps)

## The LLM as Reasoning Engine

Large language models represent a paradigm shift in how agents reason and act. Unlike traditional agents with hand-coded logic or learned neural policies, LLM agents use **natural language as their reasoning substrate**.

### What Makes LLMs Different

**Traditional agents**:
- Reasoning encoded in code or neural weights
- Task-specific training
- Opaque decision-making

**LLM agents**:
- Reasoning expressed in language
- General-purpose pre-training
- Interpretable thought processes

The key insight: **Language is a universal interface** for reasoning, tool use, and human-agent interaction.

## The Agent Loop

LLM agents operate in an iterative observe-reason-act cycle:

```
1. Observe: Receive input (user request, tool output, environment state)
2. Reason: Generate thoughts, plans, or decisions in natural language
3. Act: Choose and execute a tool/action
4. Observe: Receive result of action
5. Repeat until goal is achieved
```

### Example: Research Task

**User**: "Find recent papers on chain-of-thought reasoning and summarize key findings"

**Agent Loop**:
```
Observation: User wants recent papers on chain-of-thought

Reasoning: I need to:
1. Search for relevant papers
2. Read and analyze them
3. Synthesize findings

Action: search_papers(query="chain-of-thought reasoning", year_min=2023)

Observation: Found 15 papers...

Reasoning: I'll read the top 5 most cited papers first

Action: read_paper(paper_id="arxiv:2023.12345")

Observation: Paper discusses self-consistency improvements...

[continues until task complete]
```

### Key Characteristics

- **Iterative**: Multiple steps toward goal
- **Adaptive**: Adjusts based on observations
- **Autonomous**: Decides which actions to take
- **Transparent**: Reasoning visible in language

## Prompt-Based Agency

LLM agents are configured and controlled through **prompts** - natural language instructions that define:

### 1. Role and Capabilities

```
You are a research assistant agent with access to:
- search_papers: Search academic databases
- read_paper: Extract content from papers
- summarize: Generate concise summaries
```

### 2. Reasoning Patterns

```
Before acting, always:
1. State what you know
2. Identify what you need to find out
3. Plan your next action
4. Execute and observe
```

### 3. Constraints and Guidelines

```
- Always cite sources
- If information is unavailable, say so explicitly
- Prefer recent papers over older ones
- Maximum 3 tool calls per reasoning step
```

### Prompting Strategies

**Zero-shot**: Agent reasons without examples
```
Task: Analyze code and suggest improvements
```

**Few-shot**: Agent learns from examples
```
Example 1: [input] → [reasoning] → [action]
Example 2: [input] → [reasoning] → [action]
Now: [new input]
```

**Chain-of-thought**: Explicit reasoning steps
```
Let's think step by step:
1. What is the goal?
2. What information do I have?
3. What do I need to find?
4. Which tool should I use?
```

## Tool Use and Function Calling

LLM agents extend their capabilities through **tools** - external functions they can invoke.

### Tool Schema

Tools are described to the LLM using structured schemas:

```json
{
  "name": "search_papers",
  "description": "Search academic papers by query and filters",
  "parameters": {
    "query": {
      "type": "string",
      "description": "Search query"
    },
    "year_min": {
      "type": "integer",
      "description": "Minimum publication year"
    }
  }
}
```

### Tool Selection

The LLM:
1. Understands available tools from descriptions
2. Reasons about which tool fits the current need
3. Generates a function call with appropriate parameters
4. Interprets the tool's output
5. Decides whether to use another tool or return a result

### Tool Chaining

Agents compose multiple tools to accomplish complex tasks:

```
Goal: "What's the weather like in the city where Python was created?"

1. search("where was Python created") → "Netherlands"
2. search("capital of Netherlands") → "Amsterdam"  
3. get_weather("Amsterdam") → "15°C, partly cloudy"
4. Return: "Python was created in the Netherlands. The weather in Amsterdam is currently 15°C and partly cloudy."
```

## Memory and Context

LLM agents maintain context through:

### Short-Term Memory (Context Window)

- Recent conversation history
- Tool outputs
- Intermediate reasoning
- Limited by context length (4k-128k tokens)

### Long-Term Memory

- Vector databases for semantic retrieval
- File systems for persistent storage
- Databases for structured information
- Requires explicit memory management

### Context Management Strategies

**Summarization**: Compress older context
```
[full history] → [summarized history] + [recent details]
```

**Retrieval**: Fetch relevant past information
```
Current task → Search memory → Retrieve relevant context
```

**Hierarchical**: Different memory levels
```
Working memory (immediate context)
Episodic memory (past interactions)
Semantic memory (learned facts)
```

## Key Capabilities

### 1. Natural Language Understanding

- Parse complex, ambiguous requests
- Handle varied phrasings of the same intent
- Understand context and nuance

### 2. Reasoning and Planning

- **Chain-of-thought**: Step-by-step reasoning
- **Tree-of-thought**: Exploring multiple paths
- **Plan-and-execute**: Upfront planning, then execution

### 3. Tool Use

- Select appropriate tools for tasks
- Compose multiple tools
- Handle tool failures and retries

### 4. Adaptation

- Adjust approach based on feedback
- Recover from errors
- Learn from corrections (within conversation)

### 5. Communication

- Explain reasoning and decisions
- Ask clarifying questions
- Provide status updates

## Unique Characteristics

### Strengths

**Flexibility**: Handles diverse tasks without retraining
```
Same agent can: write code, analyze data, search information, draft documents
```

**Generalization**: Applies knowledge to new situations
```
Trained on general text → Solves specific novel problems
```

**Interpretability**: Reasoning visible in language
```
Agent: "I need to search first, then analyze the results"
```

**Few-shot learning**: Learns from minimal examples
```
One example → Adapts to similar tasks
```

### Weaknesses

**Hallucination**: Can generate plausible but incorrect information
```
Agent: "The capital of Mars is Olympus City"
```

**Non-determinism**: Same input may produce different outputs
```
Run 1: [approach A] → success
Run 2: [approach B] → failure
```

**Cost**: Expensive inference
```
1000 tokens = $0.01 - $0.10 depending on model
```

**Latency**: Slower than traditional software
```
LLM reasoning: 1-5 seconds per step
Traditional code: milliseconds
```

**Limited reasoning**: Struggles with complex logic
```
Multi-step math, formal reasoning, consistency checking
```

## Summary

The LLM agent paradigm represents a fundamental shift in how we build autonomous systems:

- **Language as substrate**: Reasoning expressed in natural language
- **Tool use**: Extends capabilities through external functions
- **Iterative operation**: Observe-reason-act loops
- **Prompt-based control**: Configured through natural language
- **General-purpose**: One agent handles diverse tasks

This paradigm enables unprecedented flexibility and generality, but introduces new challenges around reliability, consistency, and cost.

## Next Steps

Continue to [Challenges and Considerations](challenges.md) to understand the hard problems in agentic AI - robustness, hallucination, safety, and reliability - and how to design systems that address these concerns.
