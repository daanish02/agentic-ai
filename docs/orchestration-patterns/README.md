# Orchestration Patterns

## About This Section

This section explores how to orchestrate agent workflows - coordinating multiple steps, managing state, controlling execution flow, and composing complex behaviors from simpler components. Orchestration patterns determine how agents execute tasks, handle concurrency, manage long-running processes, and coordinate work. These patterns are essential for building sophisticated agent systems that go beyond simple request-response interactions.

## Contents

### [Workflow Patterns](workflow-patterns.md)

Fundamental workflow patterns for agent orchestration: chains (sequential composition), graphs (DAGs and conditional flows), loops, and branches. Covers when to use each pattern and how to compose them.

### [State Management](state-management.md)

Managing agent state, conversation state, and workflow state. Covers state persistence, state transitions, maintaining context across steps, and handling stateful interactions.

### [Control Flow](control-flow.md)

Conditionals, loops, branching, and control structures in agent workflows. Covers dynamic routing, conditional execution, iteration patterns, and error-driven control flow.

### [Parallelization and Concurrency](parallelization.md)

Running multiple operations concurrently: parallel tool calls, concurrent reasoning, fan-out/fan-in patterns, and managing asynchronous execution. Covers when parallelization helps and coordination challenges.

### [Background Processes](background-processes.md)

Long-running tasks, async execution, and process management. Covers running agents in the background, handling long-duration operations, status tracking, and coordinating foreground-background work.

### [Artifacts and Work Products](artifacts.md)

Generated outputs, intermediate results, and persistent work products. Covers artifact patterns, versioning, artifact storage, and how agents produce and manage tangible outputs.

### [Filesystem-Based Context](filesystem-context.md)

Using file systems for memory, context persistence, and coordination. Covers file-based state, reading and writing context files, and filesystem patterns for agent memory.

### [Subagent Patterns](subagent-patterns.md)

Delegating to specialized sub-agents and hierarchical execution. Covers when to spawn subagents, parent-child coordination, task delegation, and managing subagent lifecycles.

### [Human-in-the-Loop](human-in-the-loop.md)

Approval workflows, human oversight, and interactive verification. Covers when to involve humans, approval patterns, feedback integration, and designing for human-agent collaboration.

### [Visual Workflow Builders](visual-workflows.md)

No-code/low-code orchestration and drag-and-drop composition. Covers visual workflow design, the role of no-code tools in agent orchestration, and when visual builders are appropriate.

### [Event-Driven Patterns](event-driven.md)

Triggers, webhooks, scheduled workflows, and event-driven automation. Covers reactive orchestration, event sources, event handlers, and building agents that respond to external events.
