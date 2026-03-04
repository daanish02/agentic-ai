# What is Agency?

## Table of Contents

- [Defining Agency](#defining-agency)
- [Core Components of Agency](#core-components-of-agency)
- [Agent vs Non-Agent Systems](#agent-vs-non-agent-systems)
- [The Spectrum of Agency](#the-spectrum-of-agency)
- [Why Agency Matters](#why-agency-matters)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Defining Agency

**Agency** is the capacity of a system to act autonomously in pursuit of goals, making decisions based on observations of its environment and adapting its behavior to achieve desired outcomes.

An agent is not just a program that responds to inputs - it is a system that:
- **Perceives** its environment
- **Reasons** about what to do
- **Acts** to change its environment or state
- **Adapts** based on feedback and outcomes

Agency exists on a spectrum. A thermostat has minimal agency (perceives temperature, acts to maintain setpoint). A self-driving car has substantial agency (perceives roads, reasons about routes, acts to navigate safely). A research assistant agent has even more complex agency (perceives questions, reasons about information needs, acts to gather and synthesize knowledge).

## Core Components of Agency

### 1. Autonomy

The ability to operate without constant human intervention. An autonomous agent can:
- Make decisions independently
- Choose which actions to take
- Determine when to act or wait
- Recover from failures without explicit instruction

**Non-autonomous**: A script that executes a fixed sequence of steps.  
**Autonomous**: An agent that decides which tools to use based on the current situation.

### 2. Goal-Directedness

The orientation toward achieving specific objectives. A goal-directed agent:
- Has explicit or implicit goals
- Evaluates progress toward those goals
- Adjusts behavior to maximize goal achievement
- Can prioritize among multiple competing goals

**Non-goal-directed**: A reactive system that always responds the same way to inputs.  
**Goal-directed**: An agent that plans multiple steps to accomplish a user's request.

### 3. Perception and Observation

The ability to gather information about the environment. An agent must:
- Observe the current state
- Detect changes and events
- Process and interpret observations
- Maintain awareness of context

**Limited perception**: An agent that only sees the current user message.  
**Rich perception**: An agent that observes file systems, API responses, tool outputs, and conversation history.

### 4. Reasoning and Decision-Making

The ability to think about what to do next. This involves:
- Evaluating options
- Predicting outcomes
- Weighing trade-offs
- Selecting actions strategically

**Simple reasoning**: If-then rules.  
**Complex reasoning**: Chain-of-thought reasoning, planning multiple steps ahead, considering alternatives.

### 5. Action and Execution

The ability to affect the environment or change state. An agent can:
- Execute tool calls
- Make API requests
- Write to files or databases
- Invoke other agents or systems

**Read-only**: An agent that can only observe and report.  
**Active**: An agent that can modify files, trigger workflows, and cause real-world effects.

### 6. Adaptation and Learning

The ability to improve over time. An adaptive agent:
- Learns from feedback
- Refines strategies based on outcomes
- Remembers successful approaches
- Avoids repeating mistakes

**Static**: An agent that always behaves the same way.  
**Adaptive**: An agent that adjusts its approach based on success/failure patterns.

## Agent vs Non-Agent Systems

### Not Agents

**Function calls**: Deterministic, no autonomy, no goals.
```
input → function → output
```

**API endpoints**: Stateless request-response, no memory or planning.
```
request → process → response
```

**Traditional software**: Follows predefined logic, no adaptation.
```
state → transition rules → new state
```

### Agents

**Simple agent**: Observes, reasons, acts in a loop.
```
observe → reason → act → observe → ...
```

**Planning agent**: Decomposes goals, plans multiple steps, executes.
```
goal → plan → step₁ → step₂ → ... → goal achieved
```

**Reflective agent**: Evaluates its own performance, learns, adapts.
```
act → reflect → learn → act better
```

## The Spectrum of Agency

### Reactive Agents
- Respond directly to immediate inputs
- No planning or memory
- Example: A chatbot that answers questions without context

### Deliberative Agents
- Reason before acting
- Plan multiple steps
- Example: A coding agent that breaks down a task into subtasks

### Learning Agents
- Improve over time
- Adapt strategies
- Example: An agent that remembers which approaches worked best

### Autonomous Agents
- Operate independently for extended periods
- Self-directed goal pursuit
- Example: A research agent that autonomously gathers and synthesizes information

## Why Agency Matters

### 1. Automation of Complex Tasks

Agents can handle tasks that require:
- Multiple steps and tool interactions
- Conditional logic and decision-making
- Adaptation to unexpected situations

Traditional automation requires predefined workflows. Agents can figure out the workflow.

### 2. Handling Uncertainty

Real-world problems are rarely deterministic:
- Information may be incomplete
- Goals may be ambiguous
- Environments may change unexpectedly

Agents can navigate uncertainty through reasoning and adaptation.

### 3. Scalability of Intelligence

Agents enable AI systems to:
- Tackle open-ended problems
- Work on tasks without extensive prompt engineering
- Generalize across different domains

### 4. Human-AI Collaboration

Agents can act as:
- Assistants that augment human capabilities
- Collaborators that handle routine subtasks
- Tools that extend human reach and effectiveness

## Summary

Agency is the capacity for autonomous, goal-directed action. It involves:
- **Autonomy**: Acting without constant supervision
- **Goals**: Pursuing objectives
- **Perception**: Observing the environment
- **Reasoning**: Deciding what to do
- **Action**: Affecting the world
- **Adaptation**: Learning and improving

Agents exist on a spectrum from simple reactive systems to complex autonomous entities. Understanding agency as a concept - not just a technical pattern - is essential for designing effective agent systems.

## Next Steps

Continue to [Evolution of Agent Systems](evolution-of-agents.md) to understand how we arrived at modern LLM-based agents, and the key architectural innovations along the way.
