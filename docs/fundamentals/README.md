# Fundamentals

## About This Section

This section covers the essential building blocks of agentic systems - the practical fundamentals you need before tackling complex architectures. While the Overview section establishes conceptual understanding, this section focuses on **concrete implementation patterns** and **core techniques** that underpin all agent systems.

These are the foundations: basic prompting, simple loops, tool calling mechanics, context management, and error handling. Master these before moving to advanced architectures.

## Contents

### [Prompting Basics](prompting-basics.md)

Fundamental prompting techniques for agents: system prompts, user prompts, instruction clarity, role definition, and basic prompt engineering. Covers how to communicate intent to LLMs and structure effective agent prompts.

### [Simple Agent Loops](simple-agent-loops.md)

Building your first agent loop: the basic observe-reason-act cycle, iteration control, stopping conditions, and maintaining state across steps. The simplest form of agency.

### [Basic Tool Calling](basic-tool-calling.md)

Tool calling mechanics: defining simple tools, invoking functions, parsing tool outputs, and handling results. The fundamental pattern for extending agent capabilities.

### [Context and Message Management](context-management.md)

Managing conversation context: message history, system vs user messages, formatting context, context window limitations, and basic context compression strategies.

### [Structured Outputs](structured-outputs.md)

Getting reliable structured data from LLMs: JSON schemas, parsing outputs, validation, handling malformed responses, and ensuring agents produce usable data.

### [Basic Error Handling](basic-error-handling.md)

Fundamental error handling: catching failures, retry logic, graceful degradation, error messages, and preventing agent crashes. Essential resilience patterns.

### [State Management Basics](state-basics.md)

Maintaining agent state: tracking progress, storing intermediate results, conversation state, and basic persistence patterns.
