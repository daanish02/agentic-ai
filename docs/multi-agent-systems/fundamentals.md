# Multi-Agent Fundamentals

## Table of Contents

- [Introduction](#introduction)
- [What are Multi-Agent Systems?](#what-are-multi-agent-systems)
- [Why Multiple Agents?](#why-multiple-agents)
- [Benefits of Multi-Agent Systems](#benefits-of-multi-agent-systems)
- [When to Use Multi-Agent Systems](#when-to-use-multi-agent-systems)
- [Single-Agent vs Multi-Agent](#single-agent-vs-multi-agent)
- [Coordination Overhead](#coordination-overhead)
- [Multi-Agent Design Principles](#multi-agent-design-principles)
- [Common Multi-Agent Patterns](#common-multi-agent-patterns)
- [Multi-Agent System Components](#multi-agent-system-components)
- [Building Your First Multi-Agent System](#building-your-first-multi-agent-system)
- [Challenges and Trade-offs](#challenges-and-trade-offs)
- [Real-World Multi-Agent Applications](#real-world-multi-agent-applications)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Multi-agent systems represent a fundamental shift in how we build AI applications. Instead of relying on a single, monolithic agent to handle all tasks, we create **ecosystems of specialized agents** that work together, each contributing its unique capabilities.

> "The whole is greater than the sum of its parts."  
> -- Aristotle (still relevant in 2026)

Think of it like building a company versus hiring one super-employee. A single brilliant person can do amazing things, but a well-coordinated team of specialists can tackle problems of a completely different scale and complexity.

### The Core Insight

The power of multi-agent systems comes from:

**Specialization**: Each agent becomes an expert in its domain  
**Parallelism**: Multiple agents work simultaneously  
**Modularity**: Replace or upgrade individual agents without rebuilding everything  
**Robustness**: System continues if one agent fails  
**Emergent Capabilities**: Agents working together unlock new possibilities

But this power comes with a cost: **coordination complexity**. Managing multiple agents requires careful design of communication protocols, coordination mechanisms, and task allocation strategies.

This guide covers the fundamentals: why multi-agent systems exist, when to use them, how they differ from single-agent approaches, and the essential patterns for making agents work together effectively.

## What are Multi-Agent Systems?

A **multi-agent system** (MAS) is a computational system where multiple autonomous agents interact to achieve individual or collective goals.

### Defining Characteristics

**1. Multiple Agents**

The system contains two or more distinct agents, each with:

- Its own identity and state
- Decision-making capabilities
- Ability to perceive and act

**2. Autonomy**

Each agent operates independently:

- Makes its own decisions
- Controls its own actions
- Has its own goals or objectives

**3. Interaction**

Agents don't work in isolation:

- Communicate with each other
- Coordinate activities
- Share information or resources

**4. Environment**

Agents exist in a shared environment:

- Physical (robots, devices)
- Digital (software systems, data)
- Hybrid (cyber-physical systems)

### Simple Multi-Agent Example

```python
from typing import Dict, List, Any

class Agent:
    """Base class for agents in a multi-agent system."""

    def __init__(self, agent_id: str, role: str):
        self.agent_id = agent_id
        self.role = role
        self.inbox: List[Dict[str, Any]] = []

    def perceive(self, observation: Any) -> None:
        """Receive information from environment."""
        pass

    def decide(self) -> str:
        """Make decision based on current state."""
        pass

    def act(self, action: str) -> Any:
        """Execute action in environment."""
        pass

    def receive_message(self, message: Dict[str, Any]) -> None:
        """Receive message from another agent."""
        self.inbox.append(message)

    def send_message(self, recipient_id: str, content: Any) -> Dict[str, Any]:
        """Create message to send to another agent."""
        return {
            "from": self.agent_id,
            "to": recipient_id,
            "content": content,
            "timestamp": self._current_time()
        }

class MultiAgentSystem:
    """Manages multiple agents and their interactions."""

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.message_bus: List[Dict[str, Any]] = []

    def register_agent(self, agent: Agent) -> None:
        """Add agent to the system."""
        self.agents[agent.agent_id] = agent

    def send_message(self, message: Dict[str, Any]) -> None:
        """Route message to recipient agent."""
        recipient_id = message["to"]
        if recipient_id in self.agents:
            self.agents[recipient_id].receive_message(message)

    def broadcast_message(self, sender_id: str, content: Any) -> None:
        """Send message to all agents except sender."""
        for agent_id, agent in self.agents.items():
            if agent_id != sender_id:
                message = {
                    "from": sender_id,
                    "to": agent_id,
                    "content": content,
                    "broadcast": True
                }
                agent.receive_message(message)

    def step(self) -> None:
        """Execute one step for all agents."""
        for agent in self.agents.values():
            action = agent.decide()
            result = agent.act(action)
            agent.perceive(result)

# Example usage
system = MultiAgentSystem()

# Create specialized agents
researcher = Agent("researcher_1", "research")
writer = Agent("writer_1", "writing")
reviewer = Agent("reviewer_1", "review")

# Register agents
system.register_agent(researcher)
system.register_agent(writer)
system.register_agent(reviewer)

# Agents communicate
msg = researcher.send_message("writer_1", {
    "task": "write_summary",
    "findings": ["Finding 1", "Finding 2"]
})
system.send_message(msg)
```

### Multi-Agent Architecture

```
┌─────────────────────────────────────────────────────┐
│              Multi-Agent System                      │
│                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│  │ Agent A  │◄───►│ Agent B  │◄───►│ Agent C  │     │
│  │          │    │          │    │          │     │
│  │ Research │    │  Writer  │    │ Reviewer │     │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘     │
│       │               │               │            │
│       └───────────────┼───────────────┘            │
│                       │                            │
│              ┌────────▼────────┐                   │
│              │   Environment   │                   │
│              │  Shared State   │                   │
│              │  Message Bus    │                   │
│              └─────────────────┘                   │
└──────────────────────────────────────────────────────┘
```

## Why Multiple Agents?

Why create complexity by managing multiple agents instead of one powerful agent? Several compelling reasons:

### 1. Specialization

**Problem**: One agent trying to do everything becomes a "jack of all trades, master of none."

**Solution**: Specialized agents become experts in narrow domains.

```python
# Single agent trying to do everything
class UniversalAgent:
    def handle_request(self, request):
        if request.type == "research":
            return self.do_research(request)  # Mediocre at research
        elif request.type == "coding":
            return self.write_code(request)    # Mediocre at coding
        elif request.type == "design":
            return self.create_design(request) # Mediocre at design
        # Agent must be competent at everything

# Multi-agent with specialists
class ResearchAgent:
    def handle_request(self, request):
        return self.do_deep_research(request)  # Expert at research

class CodingAgent:
    def handle_request(self, request):
        return self.write_optimized_code(request)  # Expert at coding

class DesignAgent:
    def handle_request(self, request):
        return self.create_professional_design(request)  # Expert at design
```

**Benefits**:

- Each agent has focused prompts and capabilities
- Easier to optimize individual agents
- Clear expertise boundaries
- Better quality in each domain

### 2. Parallelism

**Problem**: A single agent processes tasks sequentially.

**Solution**: Multiple agents work simultaneously.

```python
import asyncio
from datetime import datetime

class TaskCoordinator:
    def __init__(self, agents: List[Agent]):
        self.agents = {agent.role: agent for agent in agents}

    async def process_parallel(self, tasks: List[Dict]):
        """Process tasks in parallel using multiple agents."""

        # Distribute tasks to appropriate agents
        agent_tasks = {}
        for task in tasks:
            agent_type = task["requires"]
            if agent_type not in agent_tasks:
                agent_tasks[agent_type] = []
            agent_tasks[agent_type].append(task)

        # Execute all agent tasks in parallel
        start_time = datetime.now()

        results = await asyncio.gather(*[
            self.agents[agent_type].process_batch(tasks)
            for agent_type, tasks in agent_tasks.items()
        ])

        duration = (datetime.now() - start_time).total_seconds()
        print(f"Completed {len(tasks)} tasks in {duration}s (parallel)")

        return results

# Example: Processing research paper
coordinator = TaskCoordinator([
    ResearchAgent("researcher"),
    SummarizerAgent("summarizer"),
    FactCheckerAgent("fact_checker")
])

tasks = [
    {"requires": "researcher", "query": "recent developments"},
    {"requires": "summarizer", "text": "long paper content"},
    {"requires": "fact_checker", "claims": ["claim 1", "claim 2"]}
]

# All tasks run concurrently
results = await coordinator.process_parallel(tasks)
# Much faster than sequential processing!
```

**Benefits**:

- Reduced latency for complex tasks
- Better resource utilization
- Scales with number of agents
- Can handle higher throughput

### 3. Modularity

**Problem**: Monolithic agents are hard to maintain and upgrade.

**Solution**: Modular agents can be replaced or updated independently.

```python
class ModularMultiAgentSystem:
    """System where agents can be swapped out."""

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.agent_registry: Dict[str, type] = {}

    def register_agent_type(self, role: str, agent_class: type):
        """Register an agent implementation for a role."""
        self.agent_registry[role] = agent_class

    def add_agent(self, role: str, agent_id: str, **kwargs):
        """Instantiate and add agent of specified role."""
        if role not in self.agent_registry:
            raise ValueError(f"No agent type registered for role: {role}")

        agent_class = self.agent_registry[role]
        agent = agent_class(agent_id, **kwargs)
        self.agents[agent_id] = agent

    def replace_agent(self, agent_id: str, new_agent: Agent):
        """Hot-swap an agent without restarting system."""
        if agent_id in self.agents:
            old_agent = self.agents[agent_id]
            # Transfer state if needed
            new_agent.restore_state(old_agent.get_state())
            self.agents[agent_id] = new_agent
            print(f"Replaced agent {agent_id}")

# Example: Upgrading individual agents
system = ModularMultiAgentSystem()

# Register agent types
system.register_agent_type("researcher", ResearchAgentV1)
system.register_agent_type("writer", WriterAgentV1)

# Create agents
system.add_agent("researcher", "r1")
system.add_agent("writer", "w1")

# Later: Upgrade just the researcher to V2
upgraded_researcher = ResearchAgentV2("r1")
system.replace_agent("r1", upgraded_researcher)
# Writer agent continues unchanged
```

**Benefits**:

- Upgrade individual components without system-wide changes
- A/B test different agent implementations
- Easier debugging and maintenance
- Clear separation of concerns

### 4. Robustness

**Problem**: Single agent failure means complete system failure.

**Solution**: Multiple agents provide redundancy and graceful degradation.

```python
class RobustMultiAgentSystem:
    """System that handles agent failures gracefully."""

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.agent_health: Dict[str, bool] = {}

    def execute_task_with_fallback(self, task: Dict, primary_role: str,
                                   fallback_roles: List[str]) -> Any:
        """Try primary agent, fall back to alternatives on failure."""

        # Try primary agent
        primary_agents = [a for a in self.agents.values()
                         if a.role == primary_role and self.agent_health.get(a.agent_id, True)]

        if primary_agents:
            try:
                return primary_agents[0].execute(task)
            except Exception as e:
                print(f"Primary agent failed: {e}")
                self.agent_health[primary_agents[0].agent_id] = False

        # Try fallback agents
        for fallback_role in fallback_roles:
            fallback_agents = [a for a in self.agents.values()
                             if a.role == fallback_role and self.agent_health.get(a.agent_id, True)]

            if fallback_agents:
                try:
                    print(f"Using fallback agent with role: {fallback_role}")
                    return fallback_agents[0].execute(task)
                except Exception as e:
                    print(f"Fallback agent failed: {e}")
                    self.agent_health[fallback_agents[0].agent_id] = False

        raise RuntimeError("All agents failed to execute task")

# Example: Research task with fallbacks
system = RobustMultiAgentSystem()
system.register_agent(ExpertResearcher("expert_r1", "research_expert"))
system.register_agent(GeneralResearcher("general_r1", "research_general"))
system.register_agent(WebSearcher("searcher_1", "web_search"))

result = system.execute_task_with_fallback(
    task={"query": "find papers on transformers"},
    primary_role="research_expert",
    fallback_roles=["research_general", "web_search"]
)
# If expert fails, uses general researcher
# If that fails, falls back to web search
```

**Benefits**:

- System continues operating despite failures
- Redundancy for critical functions
- Graceful degradation of capabilities
- Easier recovery and error handling

## Benefits of Multi-Agent Systems

Beyond the fundamental reasons, multi-agent systems provide several architectural and operational benefits:

### 1. Improved Task Decomposition

Complex problems can be naturally decomposed into agent responsibilities:

```python
class SoftwareDevTeam:
    """Multi-agent system for software development."""

    def __init__(self):
        # Each agent handles a phase of development
        self.product_manager = ProductManagerAgent()
        self.architect = ArchitectAgent()
        self.developer = DeveloperAgent()
        self.tester = TesterAgent()
        self.devops = DevOpsAgent()

    async def develop_feature(self, requirements: str):
        """Develop feature using agent pipeline."""

        # Product manager breaks down requirements
        user_stories = await self.product_manager.analyze(requirements)

        # Architect designs solution
        architecture = await self.architect.design(user_stories)

        # Developer implements
        code = await self.developer.implement(architecture)

        # Tester validates
        test_results = await self.tester.test(code)

        if not test_results.passed:
            # Developer fixes issues
            code = await self.developer.fix(test_results.issues)
            test_results = await self.tester.test(code)

        # DevOps deploys
        deployment = await self.devops.deploy(code)

        return deployment

# Each agent focuses on its specialty
# Natural decomposition of software development process
```

### 2. Better Quality Through Diversity

Different agents bring different perspectives:

```python
class ContentCreationTeam:
    """Multiple agents improve content quality."""

    def __init__(self):
        self.creative_writer = CreativeWriterAgent()  # Optimizes for engagement
        self.technical_writer = TechnicalWriterAgent()  # Optimizes for clarity
        self.editor = EditorAgent()  # Optimizes for correctness
        self.seo_specialist = SEOAgent()  # Optimizes for discoverability

    async def create_article(self, topic: str) -> str:
        """Create high-quality article through agent collaboration."""

        # Creative writer drafts engaging content
        draft = await self.creative_writer.write(topic)

        # Technical writer ensures clarity and accuracy
        clarified = await self.technical_writer.revise(draft)

        # Editor polishes grammar and style
        edited = await self.editor.polish(clarified)

        # SEO specialist optimizes for search
        final = await self.seo_specialist.optimize(edited)

        return final

    async def create_with_voting(self, topic: str) -> str:
        """Create multiple versions and vote on best."""

        # Each agent creates their version
        versions = await asyncio.gather(
            self.creative_writer.write(topic),
            self.technical_writer.write(topic)
        )

        # Agents vote on which is better
        votes = []
        for version in versions:
            score = (
                await self.editor.rate(version) +
                await self.seo_specialist.rate(version)
            )
            votes.append(score)

        # Select best version
        best_version = versions[votes.index(max(votes))]

        # Final polish by editor
        return await self.editor.polish(best_version)
```

### 3. Continuous Operation

Agents can work continuously and hand off tasks:

```python
class ContinuousMonitoringSystem:
    """Agents monitor and respond 24/7."""

    def __init__(self):
        self.monitor = MonitorAgent()  # Always watching
        self.analyzer = AnalyzerAgent()  # Analyzes issues
        self.responder = ResponderAgent()  # Takes action
        self.notifier = NotifierAgent()  # Alerts humans

    async def run_continuous(self):
        """Run continuous monitoring loop."""

        while True:
            # Monitor agent watches for issues
            issues = await self.monitor.check_systems()

            if issues:
                # Analyzer evaluates severity
                analysis = await self.analyzer.analyze(issues)

                if analysis.severity == "critical":
                    # Responder takes immediate action
                    actions = await self.responder.handle_critical(analysis)

                    # Notifier alerts team
                    await self.notifier.alert_urgent(analysis, actions)

                elif analysis.severity == "warning":
                    # Responder mitigates
                    await self.responder.handle_warning(analysis)

                    # Notifier logs for review
                    await self.notifier.log_for_review(analysis)

            # Continuous operation
            await asyncio.sleep(60)  # Check every minute

# Agents work together continuously
# Each agent can operate independently
# System never stops
```

### 4. Scalability

Add more agents to handle increased load:

```python
class ScalableCustomerSupport:
    """Multi-agent customer support that scales."""

    def __init__(self, num_agents: int = 3):
        # Create pool of support agents
        self.support_agents = [
            SupportAgent(f"support_{i}")
            for i in range(num_agents)
        ]
        self.supervisor = SupervisorAgent()
        self.task_queue = asyncio.Queue()

    async def handle_customer_query(self, query: Dict):
        """Add query to queue for processing."""
        await self.task_queue.put(query)

    async def agent_worker(self, agent: SupportAgent):
        """Worker loop for support agent."""
        while True:
            # Get next query from queue
            query = await self.task_queue.get()

            try:
                # Agent handles query
                response = await agent.handle(query)

                # Supervisor reviews if needed
                if response.needs_review:
                    response = await self.supervisor.review(response)

                # Send response to customer
                await self.send_response(query.customer_id, response)

            except Exception as e:
                # Escalate to supervisor
                await self.supervisor.handle_error(query, e)

            finally:
                self.task_queue.task_done()

    async def run(self):
        """Start all agent workers."""
        workers = [
            asyncio.create_task(self.agent_worker(agent))
            for agent in self.support_agents
        ]
        await asyncio.gather(*workers)

# Scale by adding more agents
support_system = ScalableCustomerSupport(num_agents=10)  # 10 concurrent agents

# Handle high volume of queries
for query in customer_queries:
    await support_system.handle_customer_query(query)
```

### 5. Emergent Capabilities

Agents interacting can produce behaviors beyond individual capabilities:

```python
class EmergentProblemSolving:
    """Agents collaborate to solve complex problems."""

    def __init__(self):
        self.hypothesis_generator = HypothesisAgent()
        self.experiment_designer = ExperimentAgent()
        self.data_analyzer = AnalyzerAgent()
        self.critic = CriticAgent()

    async def scientific_discovery(self, question: str) -> Dict:
        """Iterative scientific process through agent interaction."""

        hypotheses = []
        evidence = []

        for iteration in range(10):
            # Generate hypotheses
            new_hypotheses = await self.hypothesis_generator.generate(
                question=question,
                existing=hypotheses,
                evidence=evidence
            )

            # Critic evaluates hypotheses
            critique = await self.critic.evaluate(new_hypotheses)
            promising = critique.promising_hypotheses

            # Design experiments for promising hypotheses
            experiments = await self.experiment_designer.design(promising)

            # Run experiments (simulated or real)
            results = await self.run_experiments(experiments)

            # Analyze results
            analysis = await self.data_analyzer.analyze(results)

            # Update knowledge
            hypotheses.extend(promising)
            evidence.append(analysis)

            # Check if solved
            if analysis.confidence > 0.95:
                return {
                    "solution": analysis.conclusion,
                    "evidence": evidence,
                    "iterations": iteration + 1
                }

        return {"solution": "no_consensus", "evidence": evidence}

# Emergent scientific reasoning
# No single agent could do this alone
# Collaboration produces discovery
```

## When to Use Multi-Agent Systems

Multi-agent systems aren't always the right choice. Here's when they make sense:

### Use Multi-Agent When:

**1. Task has natural decomposition**

```
Good: "Analyze market trends and create investment report"
  → Analyst agent: Gather market data
  → Researcher agent: Analyze trends
  → Writer agent: Create report
  → Reviewer agent: Ensure accuracy

Bad: "Calculate 2 + 2"
  → Single operation, no decomposition needed
```

**2. Different capabilities needed**

```
Good: "Plan vacation itinerary"
  → Flight agent: Search flights
  → Hotel agent: Find accommodations
  → Activity agent: Suggest activities
  → Budget agent: Optimize costs

Bad: "Search for a restaurant"
  → Single capability needed
```

**3. Parallelism provides value**

```
Good: "Process 1000 customer support tickets"
  → 10 support agents handle 100 each (parallel)
  → Much faster than sequential

Bad: "Write a short email"
  → No parallelism benefit
```

**4. Specialization improves quality**

```
Good: "Develop and deploy software feature"
  → Specialist agents for each phase
  → Higher quality than generalist

Bad: "Format a phone number"
  → Too simple to benefit from specialization
```

**5. Robustness is critical**

```
Good: "Monitor production systems 24/7"
  → Multiple monitor agents
  → Redundancy ensures coverage

Bad: "One-time data query"
  → No need for redundancy
```

### Don't Use Multi-Agent When:

**1. Task is simple and atomic**

```python
# Don't do this
class SimpleTaskMultiAgent:
    def add_numbers(self, a: int, b: int) -> int:
        # Agent 1: Reads inputs
        reader_agent = ReaderAgent()
        inputs = reader_agent.read(a, b)

        # Agent 2: Performs addition
        calculator_agent = CalculatorAgent()
        result = calculator_agent.add(inputs)

        # Agent 3: Formats output
        formatter_agent = FormatterAgent()
        return formatter_agent.format(result)

# Just do this
def add_numbers(a: int, b: int) -> int:
    return a + b
```

**2. Coordination overhead exceeds benefits**

```
Cost of coordination: 2 seconds per agent message
Task duration: 1 second per agent
Number of agents: 5
Messages needed: 10

Total time = 5 * 1 + 10 * 2 = 25 seconds
Single agent time: 5 seconds

❌ Multi-agent is 5x slower due to coordination overhead
```

**3. State must be tightly synchronized**

```python
# Problematic: Multiple agents updating shared counter
class BadCounterSystem:
    def __init__(self):
        self.counter = 0
        self.agents = [CounterAgent() for _ in range(10)]

    async def increment_counter(self):
        # All agents try to increment simultaneously
        # Race conditions! Final count is unpredictable
        await asyncio.gather(*[
            agent.increment(self.counter)
            for agent in self.agents
        ])

# Better: Single agent for tightly coupled state
class GoodCounterSystem:
    def __init__(self):
        self.counter = 0

    def increment_counter(self):
        self.counter += 1  # No coordination needed
```

**4. Debugging complexity is too high**

```
Multi-agent debugging challenges:
- Which agent caused the error?
- What was the message sequence?
- How did agents interact?
- Why did coordination fail?

Single-agent debugging:
- Clear execution path
- Predictable behavior
- Easier to trace
```

## Single-Agent vs Multi-Agent

A direct comparison helps clarify when each approach shines:

### Single-Agent Strengths

```python
class SingleAgentApproach:
    """One agent handles everything."""

    async def process_request(self, request: str) -> str:
        # Simpler architecture
        result = await self.agent.process(request)
        return result

# Advantages:
# ✓ Simple architecture
# ✓ No coordination overhead
# ✓ Easier to debug
# ✓ Predictable execution
# ✓ Lower latency for simple tasks
# ✓ Easier state management

# Best for:
# - Simple, sequential tasks
# - Tightly coupled operations
# - When speed is critical
# - Prototyping and experiments
```

### Multi-Agent Strengths

```python
class MultiAgentApproach:
    """Specialized agents collaborate."""

    def __init__(self):
        self.researcher = ResearchAgent()
        self.analyzer = AnalyzerAgent()
        self.writer = WriterAgent()

    async def process_request(self, request: str) -> str:
        # More complex but more capable
        data = await self.researcher.gather(request)
        insights = await self.analyzer.analyze(data)
        result = await self.writer.compose(insights)
        return result

# Advantages:
# ✓ Specialization improves quality
# ✓ Parallelism improves speed
# ✓ Modularity improves maintainability
# ✓ Scalability for complex tasks
# ✓ Robustness through redundancy
# ✓ Natural task decomposition

# Best for:
# - Complex, multi-phase tasks
# - When parallelism helps
# - Systems requiring high quality
# - Long-running applications
# - Tasks with clear role boundaries
```

### Comparison Table

| Aspect           | Single-Agent            | Multi-Agent                   |
| ---------------- | ----------------------- | ----------------------------- |
| **Complexity**   | Low                     | High                          |
| **Setup Time**   | Minutes                 | Hours to Days                 |
| **Coordination** | None needed             | Critical                      |
| **Latency**      | Lower for simple tasks  | Higher due to coordination    |
| **Throughput**   | Limited by one agent    | Scales with agents            |
| **Quality**      | Good for general tasks  | Better through specialization |
| **Debugging**    | Easier                  | More challenging              |
| **Maintenance**  | Simpler                 | Requires more effort          |
| **Scalability**  | Limited                 | High                          |
| **Robustness**   | Single point of failure | Redundancy possible           |
| **Cost**         | Lower                   | Higher                        |

### Decision Framework

```python
def should_use_multi_agent(task_characteristics: Dict) -> bool:
    """Decision logic for choosing architecture."""

    score = 0

    # Factors favoring multi-agent
    if task_characteristics["complexity"] > 7:  # 1-10 scale
        score += 2

    if task_characteristics["can_parallelize"]:
        score += 2

    if task_characteristics["requires_specialization"]:
        score += 2

    if task_characteristics["long_running"]:
        score += 1

    if task_characteristics["needs_redundancy"]:
        score += 1

    # Factors against multi-agent
    if task_characteristics["tight_coupling"]:
        score -= 2

    if task_characteristics["requires_low_latency"]:
        score -= 1

    if task_characteristics["simple_workflow"]:
        score -= 2

    return score >= 3

# Example usage
task = {
    "complexity": 8,
    "can_parallelize": True,
    "requires_specialization": True,
    "long_running": True,
    "needs_redundancy": False,
    "tight_coupling": False,
    "requires_low_latency": False,
    "simple_workflow": False
}

if should_use_multi_agent(task):
    print("Use multi-agent system")
else:
    print("Use single-agent system")
```

## Coordination Overhead

The elephant in the room: multi-agent systems require coordination, and coordination isn't free.

### Sources of Overhead

**1. Communication Latency**

```python
import time

class CoordinationOverheadDemo:
    """Demonstrates cost of inter-agent communication."""

    async def single_agent_task(self):
        """All processing in one agent."""
        start = time.time()

        # Agent does all steps internally (fast)
        data = self.process_step_1()
        result = self.process_step_2(data)
        output = self.process_step_3(result)

        duration = time.time() - start
        print(f"Single agent: {duration:.3f}s")
        return output

    async def multi_agent_task(self):
        """Processing across multiple agents."""
        start = time.time()

        # Agent 1 processes
        data = await self.agent1.process_step_1()
        # Communication delay: 50ms
        await self.send_message(self.agent1, self.agent2, data)

        # Agent 2 processes
        result = await self.agent2.process_step_2(data)
        # Communication delay: 50ms
        await self.send_message(self.agent2, self.agent3, result)

        # Agent 3 processes
        output = await self.agent3.process_step_3(result)

        duration = time.time() - start
        print(f"Multi agent: {duration:.3f}s")
        # Includes 100ms+ of communication overhead
        return output

# Output:
# Single agent: 0.150s
# Multi agent: 0.250s  (communication adds overhead)
```

**2. Synchronization Costs**

```python
class SynchronizationOverhead:
    """Agents must coordinate their timing."""

    async def process_with_sync(self, tasks: List[Dict]):
        """All agents must wait for slowest."""

        # Start all agents
        results = await asyncio.gather(
            self.fast_agent.process(tasks[0]),      # Finishes in 1s
            self.medium_agent.process(tasks[1]),    # Finishes in 2s
            self.slow_agent.process(tasks[2])       # Finishes in 5s
        )

        # All agents wait for slow agent
        # Total time: 5s (limited by slowest)

        # Fast and medium agents idle waiting
        # Wasted capacity: 4s + 3s = 7s of agent time

        return results
```

**3. Conflict Resolution**

```python
class ConflictResolution:
    """Handling conflicting agent decisions."""

    async def get_consensus(self, question: str) -> str:
        """Multiple agents must agree."""

        # Get responses from multiple agents
        responses = await asyncio.gather(
            self.agent1.answer(question),
            self.agent2.answer(question),
            self.agent3.answer(question)
        )

        # Responses disagree - need to resolve
        if len(set(responses)) > 1:
            # Conflict resolution takes time
            # Options: voting, majority, escalation, debate
            consensus = await self.resolve_conflict(responses)
            return consensus

        return responses[0]

    async def resolve_conflict(self, responses: List[str]) -> str:
        """Resolve disagreement between agents."""

        # Simple voting
        vote_counts = {}
        for response in responses:
            vote_counts[response] = vote_counts.get(response, 0) + 1

        # If no clear winner, escalate to supervisor
        max_votes = max(vote_counts.values())
        if list(vote_counts.values()).count(max_votes) > 1:
            # Escalation adds more overhead
            return await self.supervisor.decide(responses)

        return max(vote_counts, key=vote_counts.get)
```

**4. State Management**

```python
class StateManagementOverhead:
    """Keeping agent states consistent."""

    def __init__(self):
        self.shared_state = {}
        self.state_lock = asyncio.Lock()

    async def update_shared_state(self, agent_id: str, updates: Dict):
        """Agents must coordinate access to shared state."""

        # Lock prevents simultaneous access
        async with self.state_lock:
            # Only one agent can update at a time
            # Other agents wait (overhead)
            self.shared_state.update(updates)
            await self.notify_state_change(agent_id, updates)

    async def notify_state_change(self, source_agent: str, changes: Dict):
        """Notify all agents of state changes."""

        # Broadcasting changes to all agents (overhead)
        notifications = [
            self.send_notification(agent_id, changes)
            for agent_id in self.agents.keys()
            if agent_id != source_agent
        ]

        await asyncio.gather(*notifications)
```

### Minimizing Coordination Overhead

**Strategy 1: Asynchronous Communication**

```python
class AsynchronousCoordination:
    """Reduce overhead with async communication."""

    def __init__(self):
        self.message_queue = asyncio.Queue()

    async def send_async(self, recipient: str, message: Dict):
        """Non-blocking message send."""
        await self.message_queue.put({
            "recipient": recipient,
            "message": message
        })
        # Sender doesn't wait for processing
        return "sent"

    async def message_processor(self):
        """Background process handles messages."""
        while True:
            msg = await self.message_queue.get()
            recipient = self.agents[msg["recipient"]]
            await recipient.receive(msg["message"])
            self.message_queue.task_done()

# Agents don't block waiting for responses
# Messages processed in background
# Lower latency for senders
```

**Strategy 2: Minimize Inter-Agent Dependencies**

```python
class LooselyCoupledAgents:
    """Design agents to work independently."""

    async def process_independently(self, task: Dict):
        """Agents work without coordination."""

        # Decompose task so agents don't need to communicate
        subtasks = self.decompose(task)

        # Each agent works independently
        results = await asyncio.gather(*[
            self.agents[i].process(subtasks[i])
            for i in range(len(subtasks))
        ])

        # Only aggregate at the end (minimal coordination)
        return self.aggregate(results)

    def decompose(self, task: Dict) -> List[Dict]:
        """Break task into independent pieces."""
        # Design decomposition to minimize dependencies
        return [
            {"subtask": "part1", "independent": True},
            {"subtask": "part2", "independent": True},
            {"subtask": "part3", "independent": True}
        ]
```

**Strategy 3: Batch Communications**

```python
class BatchedCommunication:
    """Reduce message overhead by batching."""

    def __init__(self):
        self.message_buffer = {}
        self.batch_interval = 0.1  # 100ms

    async def send_batched(self, recipient: str, message: Dict):
        """Add message to batch buffer."""
        if recipient not in self.message_buffer:
            self.message_buffer[recipient] = []

        self.message_buffer[recipient].append(message)

    async def flush_batches(self):
        """Send all batched messages periodically."""
        while True:
            await asyncio.sleep(self.batch_interval)

            for recipient, messages in self.message_buffer.items():
                if messages:
                    # Send all messages at once
                    await self.agents[recipient].receive_batch(messages)
                    messages.clear()

# Reduces number of communication round-trips
# Trades latency for throughput
```

**Strategy 4: Hierarchical Organization**

```python
class HierarchicalCoordination:
    """Reduce coordination through hierarchy."""

    def __init__(self):
        self.manager = ManagerAgent()
        self.workers = [WorkerAgent(i) for i in range(10)]

    async def execute_task(self, task: Dict):
        """Manager coordinates workers."""

        # Only manager coordinates (reduces O(n²) to O(n))
        # Workers don't communicate with each other

        subtasks = await self.manager.decompose(task)

        # Manager assigns work
        assignments = [
            (worker, subtask)
            for worker, subtask in zip(self.workers, subtasks)
        ]

        # Workers execute independently
        results = await asyncio.gather(*[
            worker.execute(subtask)
            for worker, subtask in assignments
        ])

        # Manager aggregates
        return await self.manager.aggregate(results)

# Coordination complexity: O(n) instead of O(n²)
# Clear hierarchy reduces confusion
```

## Multi-Agent Design Principles

Designing effective multi-agent systems requires following key principles:

### 1. Clear Roles and Responsibilities

```python
class RoleBasedDesign:
    """Each agent has a well-defined role."""

    class ResearchAgent:
        """Responsible for gathering information."""

        capabilities = [
            "search_papers",
            "access_databases",
            "extract_data"
        ]

        responsibilities = [
            "Find relevant sources",
            "Extract key information",
            "Provide raw data to analysts"
        ]

        not_responsible_for = [
            "Analyzing data (Analyst's job)",
            "Writing reports (Writer's job)",
            "Making decisions (Manager's job)"
        ]

    class AnalystAgent:
        """Responsible for analysis."""

        capabilities = [
            "statistical_analysis",
            "pattern_recognition",
            "insight_generation"
        ]

        responsibilities = [
            "Analyze data from researchers",
            "Identify patterns and trends",
            "Generate insights"
        ]

        not_responsible_for = [
            "Gathering data (Researcher's job)",
            "Writing reports (Writer's job)"
        ]

# Clear boundaries prevent conflicts
# Each agent knows its job
# No overlap or gaps in coverage
```

### 2. Explicit Communication Protocols

```python
from enum import Enum
from pydantic import BaseModel
from typing import Optional

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"

class AgentMessage(BaseModel):
    """Standardized message format."""

    msg_id: str
    msg_type: MessageType
    sender: str
    recipient: str
    content: Dict
    reply_to: Optional[str] = None
    timestamp: float

class CommunicationProtocol:
    """Defines how agents communicate."""

    async def send_request(self, sender: str, recipient: str,
                          request: Dict) -> str:
        """Send request and get response."""

        msg = AgentMessage(
            msg_id=self.generate_id(),
            msg_type=MessageType.REQUEST,
            sender=sender,
            recipient=recipient,
            content=request,
            timestamp=time.time()
        )

        await self.deliver_message(msg)

        # Wait for response
        response = await self.wait_for_response(msg.msg_id)
        return response

    async def send_notification(self, sender: str, content: Dict):
        """Broadcast notification (no response expected)."""

        msg = AgentMessage(
            msg_id=self.generate_id(),
            msg_type=MessageType.NOTIFICATION,
            sender=sender,
            recipient="all",
            content=content,
            timestamp=time.time()
        )

        await self.broadcast_message(msg)

# Standardized messages prevent confusion
# Type safety ensures correct handling
# Protocols make debugging easier
```

### 3. Graceful Degradation

```python
class GracefulDegradation:
    """System continues despite failures."""

    async def execute_with_fallback(self, task: Dict):
        """Try optimal path, fall back if needed."""

        try:
            # Optimal: Use specialist agent
            return await self.specialist_agent.execute(task)

        except AgentUnavailableError:
            # Fallback 1: Use generalist agent
            try:
                return await self.generalist_agent.execute(task)

            except AgentUnavailableError:
                # Fallback 2: Use simple heuristic
                return self.simple_heuristic(task)

    async def partial_results(self, task: Dict):
        """Return partial results if some agents fail."""

        results = []
        agents = [self.agent1, self.agent2, self.agent3]

        for agent in agents:
            try:
                result = await agent.execute(task)
                results.append(result)
            except Exception as e:
                # Log error but continue
                self.log_error(agent, e)

        if not results:
            raise AllAgentsFailedError()

        # Return what we got
        return self.aggregate_partial(results)

# System doesn't completely fail
# Users get something even if not perfect
# Critical for production systems
```

### 4. Observable Behavior

```python
class ObservableMultiAgentSystem:
    """Make agent behavior visible."""

    def __init__(self):
        self.event_log = []
        self.metrics = {}

    async def execute_with_logging(self, task: Dict):
        """Execute task with full observability."""

        task_id = self.generate_task_id()

        # Log task start
        self.log_event({
            "type": "task_start",
            "task_id": task_id,
            "task": task,
            "timestamp": time.time()
        })

        try:
            # Execute with agent tracking
            result = await self.tracked_execute(task_id, task)

            # Log success
            self.log_event({
                "type": "task_complete",
                "task_id": task_id,
                "result": result,
                "timestamp": time.time()
            })

            return result

        except Exception as e:
            # Log failure
            self.log_event({
                "type": "task_failed",
                "task_id": task_id,
                "error": str(e),
                "timestamp": time.time()
            })
            raise

    async def tracked_execute(self, task_id: str, task: Dict):
        """Execute with per-agent tracking."""

        for step in self.plan_execution(task):
            agent_id = step["agent"]

            # Log agent start
            self.log_event({
                "type": "agent_start",
                "task_id": task_id,
                "agent_id": agent_id,
                "step": step,
                "timestamp": time.time()
            })

            result = await self.agents[agent_id].execute(step)

            # Log agent complete
            self.log_event({
                "type": "agent_complete",
                "task_id": task_id,
                "agent_id": agent_id,
                "result": result,
                "timestamp": time.time()
            })

    def get_task_trace(self, task_id: str) -> List[Dict]:
        """Get complete trace of task execution."""
        return [
            event for event in self.event_log
            if event.get("task_id") == task_id
        ]

# Full visibility into system behavior
# Easy debugging and monitoring
# Track performance metrics
```

### 5. Composable Agents

```python
class ComposableAgents:
    """Agents can be combined in different ways."""

    class BaseAgent:
        """Base agent interface."""

        async def execute(self, task: Dict) -> Dict:
            raise NotImplementedError

    class SequentialComposition:
        """Chain agents sequentially."""

        def __init__(self, agents: List[BaseAgent]):
            self.agents = agents

        async def execute(self, task: Dict) -> Dict:
            result = task
            for agent in self.agents:
                result = await agent.execute(result)
            return result

    class ParallelComposition:
        """Run agents in parallel."""

        def __init__(self, agents: List[BaseAgent]):
            self.agents = agents

        async def execute(self, task: Dict) -> List[Dict]:
            results = await asyncio.gather(*[
                agent.execute(task)
                for agent in self.agents
            ])
            return results

    class ConditionalComposition:
        """Route to agents based on conditions."""

        def __init__(self, router: Callable, agents: Dict[str, BaseAgent]):
            self.router = router
            self.agents = agents

        async def execute(self, task: Dict) -> Dict:
            agent_key = self.router(task)
            agent = self.agents[agent_key]
            return await agent.execute(task)

# Example: Build complex systems from simple agents
researcher = ResearchAgent()
analyzer = AnalyzerAgent()
writer = WriterAgent()

# Sequential pipeline
pipeline = SequentialComposition([researcher, analyzer, writer])

# Parallel execution
parallel = ParallelComposition([researcher, analyzer])

# Conditional routing
router = lambda task: "expert" if task["difficulty"] > 7 else "general"
conditional = ConditionalComposition(router, {
    "expert": ExpertAgent(),
    "general": GeneralAgent()
})
```

## Common Multi-Agent Patterns

Several patterns recur in multi-agent systems:

### 1. Pipeline Pattern

```python
class PipelinePattern:
    """Agents process in sequence, each transforming output."""

    def __init__(self):
        self.stages = [
            DataIngestionAgent(),
            CleaningAgent(),
            TransformationAgent(),
            ValidationAgent(),
            StorageAgent()
        ]

    async def execute_pipeline(self, input_data: Any) -> Any:
        """Pass data through pipeline stages."""

        data = input_data

        for i, agent in enumerate(self.stages):
            print(f"Stage {i+1}: {agent.__class__.__name__}")
            data = await agent.process(data)

        return data

# Use case: Data processing, ETL, content creation
# Benefit: Clear flow, easy to understand
# Drawback: Sequential, no parallelism
```

### 2. Hub-and-Spoke Pattern

```python
class HubAndSpokePattern:
    """Central coordinator with specialist workers."""

    def __init__(self):
        self.coordinator = CoordinatorAgent()
        self.specialists = {
            "research": ResearchAgent(),
            "analysis": AnalysisAgent(),
            "writing": WritingAgent()
        }

    async def execute_task(self, task: Dict) -> Dict:
        """Coordinator manages specialists."""

        # Coordinator plans execution
        plan = await self.coordinator.plan(task)

        # Coordinator delegates to specialists
        results = {}
        for step in plan.steps:
            specialist = self.specialists[step.specialist_type]
            result = await specialist.execute(step.task)
            results[step.id] = result

        # Coordinator aggregates results
        final_result = await self.coordinator.aggregate(results)

        return final_result

# Use case: Complex tasks requiring coordination
# Benefit: Clear control flow, easy scaling
# Drawback: Coordinator can become bottleneck
```

### 3. Peer-to-Peer Pattern

```python
class PeerToPeerPattern:
    """Agents collaborate as equals."""

    def __init__(self):
        self.agents = [
            CollaborativeAgent(f"agent_{i}")
            for i in range(5)
        ]

    async def solve_collaboratively(self, problem: Dict) -> Dict:
        """Agents work together without central control."""

        # Each agent proposes solution
        proposals = await asyncio.gather(*[
            agent.propose_solution(problem)
            for agent in self.agents
        ])

        # Agents review each other's proposals
        reviews = []
        for agent in self.agents:
            agent_reviews = await agent.review_proposals(proposals)
            reviews.append(agent_reviews)

        # Agents reach consensus
        consensus = await self.reach_consensus(proposals, reviews)

        return consensus

    async def reach_consensus(self, proposals: List, reviews: List) -> Dict:
        """Democratic decision making."""
        scores = [sum(review.scores[i] for review in reviews)
                 for i in range(len(proposals))]
        best_idx = scores.index(max(scores))
        return proposals[best_idx]

# Use case: Democratic decision making, brainstorming
# Benefit: No single point of failure, creative
# Drawback: Can be slow to reach consensus
```

### 4. Hierarchical Pattern

```python
class HierarchicalPattern:
    """Multi-level agent organization."""

    def __init__(self):
        # Top level: Executive
        self.executive = ExecutiveAgent()

        # Middle level: Managers
        self.managers = {
            "research": ResearchManagerAgent(),
            "development": DevelopmentManagerAgent(),
            "operations": OperationsManagerAgent()
        }

        # Bottom level: Workers
        self.workers = {
            "research": [ResearcherAgent(i) for i in range(3)],
            "development": [DeveloperAgent(i) for i in range(5)],
            "operations": [OperatorAgent(i) for i in range(2)]
        }

    async def execute_project(self, project: Dict) -> Dict:
        """Execute through hierarchy."""

        # Executive creates high-level plan
        strategic_plan = await self.executive.plan_strategy(project)

        # Managers create tactical plans
        tactical_plans = {}
        for dept, manager in self.managers.items():
            tactical_plan = await manager.plan_tactics(
                strategic_plan.departments[dept]
            )
            tactical_plans[dept] = tactical_plan

        # Workers execute tasks
        results = {}
        for dept, plan in tactical_plans.items():
            dept_results = await asyncio.gather(*[
                worker.execute_task(task)
                for worker, task in zip(self.workers[dept], plan.tasks)
            ])
            results[dept] = dept_results

        # Roll up results through hierarchy
        dept_summaries = {
            dept: await self.managers[dept].summarize(results[dept])
            for dept in results
        }

        final_report = await self.executive.compile_report(dept_summaries)

        return final_report

# Use case: Large-scale projects, organizations
# Benefit: Scales well, clear authority
# Drawback: Can be slow, rigid
```

### 5. Market Pattern

```python
class MarketPattern:
    """Agents compete for tasks through bidding."""

    def __init__(self):
        self.agents = [
            BiddingAgent(f"agent_{i}")
            for i in range(10)
        ]
        self.task_queue = []

    async def allocate_task(self, task: Dict) -> Dict:
        """Auction task to best bidder."""

        # Agents submit bids
        bids = await asyncio.gather(*[
            agent.submit_bid(task)
            for agent in self.agents
        ])

        # Select winning bid (lowest cost, highest quality)
        winning_bid = self.select_winner(bids)
        winner = winning_bid.agent

        # Winner executes task
        result = await winner.execute_task(task)

        # Pay winner
        await self.process_payment(winner, winning_bid.price)

        return result

    def select_winner(self, bids: List) -> Any:
        """Select best bid based on criteria."""

        # Score each bid
        scores = []
        for bid in bids:
            score = (
                bid.estimated_quality * 0.6 +  # Quality weight
                (1.0 - bid.price / max_price) * 0.3 +  # Cost weight
                (1.0 - bid.estimated_time / max_time) * 0.1  # Speed weight
            )
            scores.append(score)

        best_idx = scores.index(max(scores))
        return bids[best_idx]

# Use case: Resource allocation, competitive environments
# Benefit: Efficient allocation, incentive alignment
# Drawback: Requires careful mechanism design
```

## Multi-Agent System Components

Essential components for building multi-agent systems:

### 1. Agent Registry

```python
class AgentRegistry:
    """Central registry of all agents."""

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.capabilities: Dict[str, List[str]] = {}

    def register(self, agent: Agent, capabilities: List[str]):
        """Register agent with its capabilities."""
        self.agents[agent.agent_id] = agent
        self.capabilities[agent.agent_id] = capabilities

    def find_by_capability(self, capability: str) -> List[Agent]:
        """Find agents with specific capability."""
        agent_ids = [
            agent_id for agent_id, caps in self.capabilities.items()
            if capability in caps
        ]
        return [self.agents[aid] for aid in agent_ids]

    def get_agent(self, agent_id: str) -> Agent:
        """Get agent by ID."""
        return self.agents.get(agent_id)

    def deregister(self, agent_id: str):
        """Remove agent from registry."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            del self.capabilities[agent_id]
```

### 2. Message Bus

```python
class MessageBus:
    """Centralized message routing."""

    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_log: List[Dict] = []

    def subscribe(self, topic: str, handler: Callable):
        """Subscribe to topic."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(handler)

    async def publish(self, topic: str, message: Dict):
        """Publish message to topic."""

        # Log message
        self.message_log.append({
            "topic": topic,
            "message": message,
            "timestamp": time.time()
        })

        # Notify subscribers
        if topic in self.subscribers:
            await asyncio.gather(*[
                handler(message)
                for handler in self.subscribers[topic]
            ])

    async def send_direct(self, sender: str, recipient: str, message: Dict):
        """Send message directly to specific agent."""
        topic = f"agent.{recipient}"
        await self.publish(topic, {
            "from": sender,
            "content": message
        })
```

### 3. Shared Memory/Blackboard

```python
class SharedMemory:
    """Shared knowledge space for agents."""

    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.access_log: List[Dict] = []
        self.locks: Dict[str, asyncio.Lock] = {}

    async def write(self, key: str, value: Any, agent_id: str):
        """Write data to shared memory."""

        # Get lock for key
        if key not in self.locks:
            self.locks[key] = asyncio.Lock()

        async with self.locks[key]:
            self.data[key] = value

            # Log access
            self.access_log.append({
                "operation": "write",
                "key": key,
                "agent": agent_id,
                "timestamp": time.time()
            })

    async def read(self, key: str, agent_id: str) -> Any:
        """Read data from shared memory."""

        value = self.data.get(key)

        # Log access
        self.access_log.append({
            "operation": "read",
            "key": key,
            "agent": agent_id,
            "timestamp": time.time()
        })

        return value

    async def read_pattern(self, pattern: str, agent_id: str) -> Dict[str, Any]:
        """Read all keys matching pattern."""
        import re
        regex = re.compile(pattern)

        matching = {
            key: value
            for key, value in self.data.items()
            if regex.match(key)
        }

        return matching
```

### 4. Task Queue

```python
class TaskQueue:
    """Distributed task queue for agents."""

    def __init__(self):
        self.queue = asyncio.Queue()
        self.in_progress: Dict[str, Dict] = {}
        self.completed: Dict[str, Any] = {}

    async def enqueue(self, task: Dict) -> str:
        """Add task to queue."""
        task_id = self.generate_task_id()
        task["task_id"] = task_id
        await self.queue.put(task)
        return task_id

    async def dequeue(self, agent_id: str) -> Optional[Dict]:
        """Get next task from queue."""

        try:
            task = await asyncio.wait_for(
                self.queue.get(),
                timeout=1.0
            )

            # Mark as in progress
            self.in_progress[task["task_id"]] = {
                "task": task,
                "agent": agent_id,
                "started_at": time.time()
            }

            return task

        except asyncio.TimeoutError:
            return None

    async def complete_task(self, task_id: str, result: Any):
        """Mark task as completed."""

        if task_id in self.in_progress:
            del self.in_progress[task_id]

        self.completed[task_id] = {
            "result": result,
            "completed_at": time.time()
        }

    async def retry_failed(self, task_id: str):
        """Re-queue failed task."""

        if task_id in self.in_progress:
            task = self.in_progress[task_id]["task"]
            del self.in_progress[task_id]
            await self.queue.put(task)
```

## Building Your First Multi-Agent System

Let's build a complete multi-agent system from scratch:

```python
"""
Complete Multi-Agent System Example
=====================================
A research assistant system with multiple specialized agents.
"""

import asyncio
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
import json

# ============================================================================
# Core Infrastructure
# ============================================================================

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"

@dataclass
class Message:
    sender: str
    recipient: str
    msg_type: MessageType
    content: Dict[str, Any]
    msg_id: Optional[str] = None

class MessageBus:
    """Handles all inter-agent communication."""

    def __init__(self):
        self.handlers: Dict[str, Any] = {}

    def register_handler(self, agent_id: str, handler: Any):
        self.handlers[agent_id] = handler

    async def send(self, message: Message):
        if message.recipient in self.handlers:
            await self.handlers[message.recipient].receive_message(message)

    async def broadcast(self, message: Message, exclude: Optional[str] = None):
        for agent_id, handler in self.handlers.items():
            if agent_id != exclude:
                msg = Message(
                    sender=message.sender,
                    recipient=agent_id,
                    msg_type=message.msg_type,
                    content=message.content
                )
                await handler.receive_message(msg)

# ============================================================================
# Base Agent
# ============================================================================

class BaseAgent:
    """Base class for all agents."""

    def __init__(self, agent_id: str, message_bus: MessageBus):
        self.agent_id = agent_id
        self.message_bus = message_bus
        self.inbox: List[Message] = []

        # Register with message bus
        message_bus.register_handler(agent_id, self)

    async def receive_message(self, message: Message):
        """Receive incoming message."""
        self.inbox.append(message)
        await self.process_message(message)

    async def process_message(self, message: Message):
        """Process received message (override in subclasses)."""
        pass

    async def send_message(self, recipient: str, content: Dict,
                          msg_type: MessageType = MessageType.REQUEST):
        """Send message to another agent."""
        message = Message(
            sender=self.agent_id,
            recipient=recipient,
            msg_type=msg_type,
            content=content
        )
        await self.message_bus.send(message)

    async def broadcast(self, content: Dict):
        """Broadcast message to all agents."""
        message = Message(
            sender=self.agent_id,
            recipient="all",
            msg_type=MessageType.NOTIFICATION,
            content=content
        )
        await self.message_bus.broadcast(message, exclude=self.agent_id)

# ============================================================================
# Specialized Agents
# ============================================================================

class ResearchAgent(BaseAgent):
    """Agent specialized in finding information."""

    async def process_message(self, message: Message):
        if message.msg_type == MessageType.REQUEST:
            query = message.content.get("query")
            results = await self.search(query)

            # Send results back
            await self.send_message(
                message.sender,
                {"results": results},
                MessageType.RESPONSE
            )

    async def search(self, query: str) -> List[Dict]:
        """Simulate research/search."""
        print(f"[{self.agent_id}] Researching: {query}")
        await asyncio.sleep(0.5)  # Simulate API call

        # Simulated results
        return [
            {"title": f"Result 1 for {query}", "relevance": 0.9},
            {"title": f"Result 2 for {query}", "relevance": 0.8},
            {"title": f"Result 3 for {query}", "relevance": 0.7}
        ]

class AnalysisAgent(BaseAgent):
    """Agent specialized in analyzing data."""

    async def process_message(self, message: Message):
        if message.msg_type == MessageType.REQUEST:
            data = message.content.get("data")
            analysis = await self.analyze(data)

            await self.send_message(
                message.sender,
                {"analysis": analysis},
                MessageType.RESPONSE
            )

    async def analyze(self, data: Any) -> Dict:
        """Simulate analysis."""
        print(f"[{self.agent_id}] Analyzing data...")
        await asyncio.sleep(0.3)

        return {
            "summary": "Analysis complete",
            "insights": ["Insight 1", "Insight 2"],
            "confidence": 0.85
        }

class WriterAgent(BaseAgent):
    """Agent specialized in generating text."""

    async def process_message(self, message: Message):
        if message.msg_type == MessageType.REQUEST:
            content = message.content.get("content")
            text = await self.write(content)

            await self.send_message(
                message.sender,
                {"text": text},
                MessageType.RESPONSE
            )

    async def write(self, content: Dict) -> str:
        """Simulate writing."""
        print(f"[{self.agent_id}] Writing content...")
        await asyncio.sleep(0.4)

        return f"Written content based on: {json.dumps(content, indent=2)}"

class CoordinatorAgent(BaseAgent):
    """Agent that coordinates other agents."""

    def __init__(self, agent_id: str, message_bus: MessageBus):
        super().__init__(agent_id, message_bus)
        self.pending_tasks: Dict[str, Dict] = {}

    async def execute_task(self, task_description: str) -> str:
        """Coordinate execution of a complex task."""

        print(f"\n[{self.agent_id}] Starting task: {task_description}\n")

        # Step 1: Research
        await self.send_message(
            "researcher",
            {"query": task_description},
            MessageType.REQUEST
        )

        # Wait for research results
        research_response = await self.wait_for_response("researcher")
        research_results = research_response.content["results"]

        # Step 2: Analyze
        await self.send_message(
            "analyst",
            {"data": research_results},
            MessageType.REQUEST
        )

        # Wait for analysis
        analysis_response = await self.wait_for_response("analyst")
        analysis = analysis_response.content["analysis"]

        # Step 3: Write
        await self.send_message(
            "writer",
            {"content": {
                "research": research_results,
                "analysis": analysis
            }},
            MessageType.REQUEST
        )

        # Wait for final text
        writer_response = await self.wait_for_response("writer")
        final_text = writer_response.content["text"]

        print(f"\n[{self.agent_id}] Task completed!\n")
        return final_text

    async def wait_for_response(self, sender: str, timeout: float = 5.0) -> Message:
        """Wait for response from specific agent."""

        start_time = asyncio.get_event_loop().time()

        while True:
            # Check inbox for response
            for msg in self.inbox:
                if msg.sender == sender and msg.msg_type == MessageType.RESPONSE:
                    self.inbox.remove(msg)
                    return msg

            # Check timeout
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError(f"No response from {sender}")

            await asyncio.sleep(0.1)

# ============================================================================
# Multi-Agent System
# ============================================================================

class MultiAgentResearchSystem:
    """Complete multi-agent research assistant system."""

    def __init__(self):
        # Create message bus
        self.message_bus = MessageBus()

        # Create agents
        self.researcher = ResearchAgent("researcher", self.message_bus)
        self.analyst = AnalysisAgent("analyst", self.message_bus)
        self.writer = WriterAgent("writer", self.message_bus)
        self.coordinator = CoordinatorAgent("coordinator", self.message_bus)

        print("Multi-Agent Research System initialized")
        print("Agents: researcher, analyst, writer, coordinator\n")

    async def process_query(self, query: str) -> str:
        """Process user query through multi-agent system."""
        result = await self.coordinator.execute_task(query)
        return result

# ============================================================================
# Example Usage
# ============================================================================

async def main():
    """Demo of the multi-agent system."""

    # Create system
    system = MultiAgentResearchSystem()

    # Process query
    query = "recent developments in multi-agent AI systems"
    result = await system.process_query(query)

    print("=" * 60)
    print("FINAL RESULT:")
    print("=" * 60)
    print(result)
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
```

**Output:**

```
Multi-Agent Research System initialized
Agents: researcher, analyst, writer, coordinator

[coordinator] Starting task: recent developments in multi-agent AI systems

[researcher] Researching: recent developments in multi-agent AI systems
[analyst] Analyzing data...
[writer] Writing content...

[coordinator] Task completed!

============================================================
FINAL RESULT:
============================================================
Written content based on: {
  "research": [
    {
      "title": "Result 1 for recent developments...",
      "relevance": 0.9
    },
    ...
  ],
  "analysis": {
    "summary": "Analysis complete",
    "insights": ["Insight 1", "Insight 2"],
    "confidence": 0.85
  }
}
============================================================
```

## Challenges and Trade-offs

Multi-agent systems face several challenges:

### 1. Complexity

**Challenge**: More agents = exponentially more complexity

**Trade-off**: Power vs. Simplicity

```python
# Complexity growth
agents = 3
interactions = agents * (agents - 1)  # 6 possible interactions

agents = 10
interactions = agents * (agents - 1)  # 90 possible interactions!

# Each interaction is a potential failure point
# Each interaction needs testing
# Debugging becomes exponentially harder
```

### 2. Cost

**Challenge**: Multiple agents = multiple LLM calls

**Trade-off**: Quality vs. Cost

```python
class CostAnalysis:
    """Analyze cost of multi-agent vs single-agent."""

    def calculate_cost(self, num_agents: int, calls_per_agent: int,
                      cost_per_call: float) -> float:
        """Calculate total cost."""
        return num_agents * calls_per_agent * cost_per_call

    def compare_approaches(self):
        # Single agent
        single_cost = self.calculate_cost(
            num_agents=1,
            calls_per_agent=10,
            cost_per_call=0.01
        )
        # $0.10

        # Multi-agent
        multi_cost = self.calculate_cost(
            num_agents=5,
            calls_per_agent=4,
            cost_per_call=0.01
        )
        # $0.20 (2x more expensive)

        # But multi-agent might provide 3x better quality
        # So cost/quality ratio might be better
```

### 3. Latency

**Challenge**: Sequential agent calls add latency

**Trade-off**: Completeness vs. Speed

```python
# Single agent: 2 seconds
# Multi-agent (sequential): 2s + 2s + 2s = 6 seconds
# Multi-agent (parallel): max(2s, 2s, 2s) = 2 seconds

# Parallelism helps but isn't always possible
```

### 4. Consistency

**Challenge**: Agents might contradict each other

**Trade-off**: Diversity vs. Consistency

```python
# Agent 1: "The answer is A"
# Agent 2: "The answer is B"
# Agent 3: "The answer is C"

# Need conflict resolution
# Might lose valuable diverse perspectives
# Consensus might be wrong (groupthink)
```

## Real-World Multi-Agent Applications

### 1. Software Development

```
Product Manager Agent: Requirements
Architect Agent: Design
Developer Agent: Implementation
Tester Agent: Quality Assurance
DevOps Agent: Deployment
```

### 2. Customer Support

```
Classifier Agent: Route queries
Knowledge Base Agent: Find answers
Conversation Agent: Interact with customer
Escalation Agent: Handle complex cases
Feedback Agent: Learn from interactions
```

### 3. Content Creation

```
Research Agent: Gather information
Outline Agent: Structure content
Writer Agent: Generate draft
Editor Agent: Improve quality
SEO Agent: Optimize for search
```

### 4. Trading Systems

```
Data Agent: Market data collection
Analysis Agent: Technical/fundamental analysis
Strategy Agent: Trading decisions
Risk Agent: Risk management
Execution Agent: Order execution
```

## Summary

Multi-agent systems extend single agents with:

**Core Benefits:**

- **Specialization**: Expert agents for specific domains
- **Parallelism**: Simultaneous execution
- **Modularity**: Independent upgrades
- **Robustness**: Graceful degradation
- **Emergence**: Capabilities beyond individual agents

**When to Use:**

- Complex tasks with natural decomposition
- Different capabilities needed
- Parallelism provides value
- Specialization improves quality
- Robustness is critical

**When Not to Use:**

- Simple, atomic tasks
- Tight coupling required
- Coordination overhead exceeds benefits
- Low latency critical
- Debugging complexity too high

**Key Principles:**

- Define clear roles and responsibilities
- Use explicit communication protocols
- Design for graceful degradation
- Make behavior observable
- Build composable agents

**Common Challenges:**

- Coordination overhead
- Increased complexity
- Higher costs
- Latency from sequential execution
- Consistency across agents

Multi-agent systems are powerful but complex. Start simple, add agents as needed, and always measure whether the additional complexity provides sufficient value.

## Next Steps

- [Agent Roles and Specialization](agent-roles.md) - Design specialized agents
- [Communication Patterns](communication.md) - Inter-agent communication
- [Coordination and Task Allocation](coordination.md) - Make agents work together
- [Delegation and Hierarchies](delegation.md) - Hierarchical organization
- [Collaboration Patterns](collaboration.md) - Cooperative problem-solving
