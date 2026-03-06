# Agent Architectures

## About This Section

This section explores fundamental agent architecture patterns - the blueprints for how agents observe, reason, plan, and act. Each architecture represents a different approach to organizing agent behavior, from simple reactive loops to complex planning systems. Understanding these patterns helps you choose the right architecture for your specific use case and design robust, effective agent systems.

These are not library-specific implementations but **universal patterns** that apply regardless of framework or language.

## Contents

### [The Agent Loop](agent-loop.md)

The fundamental observe-reason-act cycle that underlies all agent systems. Covers iterative decision-making, perception-action cycles, and how agents maintain autonomy through continuous operation.

### [ReAct: Reasoning and Acting](react.md)

The ReAct pattern interleaves reasoning and acting, where agents explicitly think through their approach before taking each action. Covers thought-action pairs, when to use ReAct, and handling multi-step reasoning with tool use.

### [Planning Architectures](planning-architectures.md)

Comprehensive coverage of planning-based agent patterns including Plan-and-Execute (upfront planning, sequential execution), ReWOO (reasoning without observation), and other planning strategies. Explores when to plan ahead vs. react dynamically.

### [Reflection and Self-Critique](reflection.md)

Agents that evaluate their own performance and learn from mistakes. Covers self-critique patterns, reflection loops, iterative refinement, and how agents can improve their outputs through self-evaluation.

### [Autonomous Agent Patterns](autonomous-agents.md)

Agents designed for open-ended goal pursuit and continuous operation. Covers long-running agents, goal-directed autonomy, self-directed task decomposition, and architectural considerations for agents that operate independently over extended periods.

### [Advanced Patterns](advanced-patterns.md)

Subagent architectures, filesystem-aware agents, artifact-generating agents, and other sophisticated patterns. Covers task delegation, spawning specialized sub-agents, context persistence through filesystems, and producing versioned work products.

### [Design Principles](design-principles.md)

Core principles for building robust agent architectures: modularity, observability, controllability, composability, and graceful degradation. Covers architectural trade-offs and how to make design decisions.
