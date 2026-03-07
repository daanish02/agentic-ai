# Tool Use

## About This Section

This section covers how agents extend their capabilities through tools - external functions, APIs, and systems that agents can invoke to accomplish tasks beyond pure language generation. Tool use is what transforms a language model from a text generator into an action-taking agent. Understanding tool patterns, selection strategies, and integration techniques is essential for building capable agent systems.

## Contents

### [Function Calling Fundamentals](function-calling.md)

The basics of how LLMs invoke functions: structured output generation, function schemas, parameter extraction, and the mechanics of translating natural language intent into function calls. Covers both native function calling APIs and prompt-based approaches.

### [Tool Schemas and Discovery](tool-schemas.md)

Designing effective tool descriptions that help agents understand when and how to use tools. Covers schema design, parameter specifications, describing tool capabilities, and enabling dynamic tool discovery at runtime.

### [Tool Selection and Reasoning](tool-selection.md)

How agents choose which tool to use: reasoning about tool capabilities, matching tools to tasks, handling ambiguity, and strategies for effective tool selection in complex scenarios with many available tools.

### [Tool Chaining and Composition](tool-chaining.md)

Combining multiple tools to accomplish complex tasks. Covers sequential composition, parallel tool use, data flow between tools, and orchestrating multi-tool workflows.

### [Error Handling and Resilience](error-handling.md)

Dealing with tool failures: retry strategies, fallback mechanisms, handling malformed parameters, rate limiting, timeouts, and building resilient tool integration that gracefully handles failures.

### [API Integration Patterns](api-integration.md)

Practical patterns for integrating external APIs as agent tools. Covers authentication, response parsing, pagination, rate limiting, error codes, and translating between API contracts and agent tool interfaces.
