# Advanced Patterns

## Table of Contents

- [Introduction](#introduction)
- [Subagent Architectures](#subagent-architectures)
- [Filesystem-Aware Agents](#filesystem-aware-agents)
- [Artifact-Generating Agents](#artifact-generating-agents)
- [Multi-Modal Agents](#multi-modal-agents)
- [Conversational Agents](#conversational-agents)
- [Tool-Forging Agents](#tool-forging-agents)
- [Meta-Agents](#meta-agents)
- [Swarm Intelligence](#swarm-intelligence)
- [Evolutionary Agents](#evolutionary-agents)
- [Symbolic-Neural Hybrid Agents](#symbolic-neural-hybrid-agents)
- [Human-in-the-Loop Patterns](#human-in-the-loop-patterns)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Advanced patterns represent sophisticated agent architectures that combine multiple concepts, handle complex scenarios, or push the boundaries of what agents can do.

These patterns are typically used when:
- Simple architectures are insufficient
- Complex coordination is required
- Multiple modalities must be handled
- High sophistication is needed

> "Simple patterns get you far. Advanced patterns get you further."

This guide covers cutting-edge agent architectures that solve complex real-world problems.

## Subagent Architectures

Agents that spawn and coordinate specialized sub-agents.

### Master-Worker Pattern

```python
from typing import List, Dict, Any
from dataclasses import dataclass
import asyncio


@dataclass
class SubAgent:
    """Specialized sub-agent."""
    id: str
    specialty: str
    capabilities: List[str]
    status: str = "idle"  # idle, working, completed, failed


class MasterAgent:
    """
    Master agent coordinating sub-agents.
    """
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.subagents: List[SubAgent] = []
    
    def solve_with_subagents(self, task: str) -> str:
        """
        Solve task by coordinating sub-agents.
        """
        # 1. Decompose task
        subtasks = self.decompose_task(task)
        
        # 2. Spawn specialized sub-agents
        agents = self.spawn_subagents(subtasks)
        
        # 3. Assign work
        assignments = self.assign_work(subtasks, agents)
        
        # 4. Execute in parallel
        results = asyncio.run(self.execute_parallel(assignments))
        
        # 5. Synthesize results
        final_result = self.synthesize_results(results, task)
        
        # 6. Cleanup
        self.cleanup_subagents(agents)
        
        return final_result
    
    def decompose_task(self, task: str) -> List[Dict]:
        """Break task into specialized subtasks."""
        prompt = f"""
Decompose this task into specialized subtasks:

Task: {task}

For each subtask, specify:
- Description
- Required specialty (e.g., "research", "analysis", "coding")
- Dependencies

Subtasks:"""
        
        response = self.llm(prompt)
        return self.parse_subtasks(response)
    
    def spawn_subagents(self, subtasks: List[Dict]) -> List[SubAgent]:
        """Create specialized sub-agents."""
        agents = []
        
        for subtask in subtasks:
            specialty = subtask["specialty"]
            
            agent = SubAgent(
                id=self.generate_agent_id(),
                specialty=specialty,
                capabilities=self.get_capabilities_for_specialty(specialty)
            )
            
            agents.append(agent)
            self.subagents.append(agent)
        
        return agents
    
    async def execute_parallel(
        self,
        assignments: List[Dict]
    ) -> List[Dict]:
        """Execute subagent tasks in parallel."""
        tasks = [
            self.execute_subagent_task(assignment)
            for assignment in assignments
        ]
        
        results = await asyncio.gather(*tasks)
        return results
    
    async def execute_subagent_task(self, assignment: Dict) -> Dict:
        """Execute single subagent task."""
        agent = assignment["agent"]
        subtask = assignment["subtask"]
        
        agent.status = "working"
        
        try:
            # Execute with agent's specialized capabilities
            result = await self.execute_specialized(agent, subtask)
            
            agent.status = "completed"
            
            return {
                "agent_id": agent.id,
                "subtask": subtask,
                "result": result,
                "success": True
            }
        
        except Exception as e:
            agent.status = "failed"
            
            return {
                "agent_id": agent.id,
                "subtask": subtask,
                "error": str(e),
                "success": False
            }
    
    def synthesize_results(
        self,
        results: List[Dict],
        original_task: str
    ) -> str:
        """Combine subagent results."""
        prompt = f"""
Original task: {original_task}

Subagent results:
{self.format_results(results)}

Synthesize into final answer:"""
        
        return self.llm(prompt)


# Example usage
def example_master_worker():
    """Example of master-worker pattern."""
    
    def mock_llm(prompt):
        if "Decompose this task" in prompt:
            return """
Subtask 1:
Description: Research current state
Specialty: research
Dependencies: none

Subtask 2:
Description: Analyze findings
Specialty: analysis
Dependencies: Subtask 1

Subtask 3:
Description: Write summary
Specialty: writing
Dependencies: Subtask 2
"""
        return "Synthesized result"
    
    master = MasterAgent(llm=mock_llm, tools={})
    result = master.solve_with_subagents(
        "Research and analyze trends in AI, then write a summary"
    )
    
    print(result)
```

### Hierarchical Subagents

```python
class HierarchicalAgentSystem:
    """
    Multi-level agent hierarchy.
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.root_agent = None
    
    def create_hierarchy(self, task: str, max_depth: int = 3):
        """
        Create hierarchical agent structure.
        """
        self.root_agent = self.create_agent_recursive(
            task=task,
            depth=0,
            max_depth=max_depth
        )
    
    def create_agent_recursive(
        self,
        task: str,
        depth: int,
        max_depth: int
    ) -> Dict:
        """
        Recursively create agent hierarchy.
        """
        agent = {
            "id": self.generate_id(),
            "task": task,
            "depth": depth,
            "subagents": []
        }
        
        # Base case: atomic task
        if depth >= max_depth or self.is_atomic(task):
            agent["type"] = "worker"
            return agent
        
        # Recursive case: decompose and create subagents
        agent["type"] = "coordinator"
        subtasks = self.decompose_task(task)
        
        for subtask in subtasks:
            subagent = self.create_agent_recursive(
                task=subtask,
                depth=depth + 1,
                max_depth=max_depth
            )
            agent["subagents"].append(subagent)
        
        return agent
    
    def execute_hierarchically(self) -> str:
        """Execute hierarchical agent system."""
        return self.execute_agent(self.root_agent)
    
    def execute_agent(self, agent: Dict) -> Any:
        """Execute agent and its subagents."""
        if agent["type"] == "worker":
            # Atomic execution
            return self.execute_task(agent["task"])
        
        else:  # coordinator
            # Execute subagents
            subresults = []
            for subagent in agent["subagents"]:
                result = self.execute_agent(subagent)
                subresults.append(result)
            
            # Coordinate results
            return self.coordinate_results(agent["task"], subresults)
```

## Filesystem-Aware Agents

Agents that use filesystem for context and memory.

### File-Based Memory

```python
import os
import json
from pathlib import Path


class FilesystemAgent:
    """
    Agent using filesystem for persistent memory.
    """
    
    def __init__(self, workspace_dir: str, llm):
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(exist_ok=True)
        self.llm = llm
        
        # Directory structure
        (self.workspace / "memory").mkdir(exist_ok=True)
        (self.workspace / "artifacts").mkdir(exist_ok=True)
        (self.workspace / "logs").mkdir(exist_ok=True)
    
    def remember(self, key: str, value: Any):
        """Store information in filesystem."""
        memory_file = self.workspace / "memory" / f"{key}.json"
        
        with open(memory_file, 'w') as f:
            json.dump(value, f, indent=2, default=str)
    
    def recall(self, key: str) -> Any:
        """Retrieve information from filesystem."""
        memory_file = self.workspace / "memory" / f"{key}.json"
        
        if not memory_file.exists():
            return None
        
        with open(memory_file, 'r') as f:
            return json.load(f)
    
    def create_artifact(self, name: str, content: str):
        """Create versioned artifact."""
        artifact_dir = self.workspace / "artifacts" / name
        artifact_dir.mkdir(exist_ok=True)
        
        # Find next version
        existing = list(artifact_dir.glob("v*.txt"))
        version = len(existing) + 1
        
        # Save new version
        artifact_file = artifact_dir / f"v{version}.txt"
        with open(artifact_file, 'w') as f:
            f.write(content)
        
        # Update latest
        latest = artifact_dir / "latest.txt"
        with open(latest, 'w') as f:
            f.write(content)
        
        return str(artifact_file)
    
    def read_artifact(self, name: str, version: str = "latest") -> str:
        """Read artifact version."""
        artifact_file = self.workspace / "artifacts" / name / f"{version}.txt"
        
        if not artifact_file.exists():
            return None
        
        with open(artifact_file, 'r') as f:
            return f.read()
    
    def list_artifacts(self) -> List[str]:
        """List all artifacts."""
        artifact_dir = self.workspace / "artifacts"
        return [d.name for d in artifact_dir.iterdir() if d.is_dir()]
    
    def log_action(self, action: str, details: Dict):
        """Log action to file."""
        log_file = self.workspace / "logs" / "agent.log"
        
        entry = {
            "timestamp": time.time(),
            "action": action,
            "details": details
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(entry, default=str) + "\n")


# Example: Code generation agent with filesystem
class CodeGenerationAgent(FilesystemAgent):
    """Agent that generates and manages code artifacts."""
    
    def generate_code(self, specification: str) -> str:
        """Generate code and save as artifact."""
        # Remember specification
        self.remember("current_spec", specification)
        
        # Generate code
        code = self.llm(f"Generate Python code for:\n{specification}\n\nCode:")
        
        # Create artifact
        artifact_path = self.create_artifact("generated_code", code)
        
        # Log action
        self.log_action("code_generation", {
            "specification": specification,
            "artifact": artifact_path
        })
        
        return code
    
    def iterate_on_code(self, feedback: str) -> str:
        """Improve code based on feedback."""
        # Recall specification and current code
        spec = self.recall("current_spec")
        current_code = self.read_artifact("generated_code")
        
        # Generate improved version
        improved = self.llm(f"""
Original specification: {spec}

Current code:
{current_code}

Feedback: {feedback}

Improved code:""")
        
        # Save new version
        self.create_artifact("generated_code", improved)
        
        return improved
```

## Artifact-Generating Agents

Agents that produce versioned work products.

### Artifact System

```python
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Artifact:
    """Versioned work product."""
    name: str
    version: int
    content: str
    created_at: datetime
    created_by: str
    metadata: Dict[str, Any]


class ArtifactManager:
    """Manage artifact lifecycle."""
    
    def __init__(self):
        self.artifacts: Dict[str, List[Artifact]] = {}
    
    def create_artifact(
        self,
        name: str,
        content: str,
        agent_id: str,
        metadata: Dict = None
    ) -> Artifact:
        """Create new artifact version."""
        if name not in self.artifacts:
            self.artifacts[name] = []
        
        version = len(self.artifacts[name]) + 1
        
        artifact = Artifact(
            name=name,
            version=version,
            content=content,
            created_at=datetime.now(),
            created_by=agent_id,
            metadata=metadata or {}
        )
        
        self.artifacts[name].append(artifact)
        
        return artifact
    
    def get_latest(self, name: str) -> Artifact:
        """Get latest version of artifact."""
        if name not in self.artifacts or not self.artifacts[name]:
            return None
        
        return self.artifacts[name][-1]
    
    def get_version(self, name: str, version: int) -> Artifact:
        """Get specific version."""
        if name not in self.artifacts:
            return None
        
        for artifact in self.artifacts[name]:
            if artifact.version == version:
                return artifact
        
        return None
    
    def get_history(self, name: str) -> List[Artifact]:
        """Get all versions of artifact."""
        return self.artifacts.get(name, [])


class ArtifactGeneratingAgent:
    """Agent that produces versioned artifacts."""
    
    def __init__(self, llm, artifact_manager: ArtifactManager):
        self.llm = llm
        self.artifacts = artifact_manager
        self.agent_id = self.generate_id()
    
    def generate_with_refinement(
        self,
        task: str,
        max_iterations: int = 3
    ) -> Artifact:
        """
        Generate artifact with iterative refinement.
        """
        # Initial generation
        content = self.generate_initial(task)
        artifact = self.artifacts.create_artifact(
            name=task,
            content=content,
            agent_id=self.agent_id,
            metadata={"iteration": 1}
        )
        
        # Iterative refinement
        for iteration in range(2, max_iterations + 1):
            # Evaluate current version
            evaluation = self.evaluate_artifact(artifact, task)
            
            if evaluation["quality"] >= 0.9:
                break
            
            # Refine
            refined_content = self.refine_artifact(
                artifact.content,
                task,
                evaluation
            )
            
            artifact = self.artifacts.create_artifact(
                name=task,
                content=refined_content,
                agent_id=self.agent_id,
                metadata={
                    "iteration": iteration,
                    "improvements": evaluation["improvements"]
                }
            )
        
        return artifact
```

## Multi-Modal Agents

Agents handling multiple input/output modalities.

### Cross-Modal Agent

```python
class MultiModalAgent:
    """
    Agent handling text, images, audio, etc.
    """
    
    def __init__(self, text_model, vision_model, audio_model):
        self.text_model = text_model
        self.vision_model = vision_model
        self.audio_model = audio_model
    
    def process_multi_modal_input(self, inputs: Dict) -> str:
        """
        Process inputs from multiple modalities.
        """
        processed = {}
        
        # Process each modality
        if "text" in inputs:
            processed["text"] = self.process_text(inputs["text"])
        
        if "image" in inputs:
            processed["image"] = self.process_image(inputs["image"])
        
        if "audio" in inputs:
            processed["audio"] = self.process_audio(inputs["audio"])
        
        # Cross-modal fusion
        result = self.fuse_modalities(processed)
        
        return result
    
    def process_image(self, image) -> Dict:
        """Process image input."""
        # Vision model analysis
        description = self.vision_model.describe(image)
        objects = self.vision_model.detect_objects(image)
        
        return {
            "description": description,
            "objects": objects
        }
    
    def process_audio(self, audio) -> Dict:
        """Process audio input."""
        # Transcription
        transcript = self.audio_model.transcribe(audio)
        
        # Audio features
        features = self.audio_model.extract_features(audio)
        
        return {
            "transcript": transcript,
            "features": features
        }
    
    def fuse_modalities(self, processed: Dict) -> str:
        """Combine information from multiple modalities."""
        prompt = f"""
Synthesize information from multiple sources:

Text: {processed.get('text', 'N/A')}
Image: {processed.get('image', 'N/A')}
Audio: {processed.get('audio', 'N/A')}

Integrated understanding:"""
        
        return self.text_model(prompt)
```

## Conversational Agents

Agents designed for natural dialogue.

### Dialogue Manager

```python
class ConversationalAgent:
    """Agent maintaining conversation context."""
    
    def __init__(self, llm):
        self.llm = llm
        self.conversation_history = []
        self.user_model = {}  # Model of user preferences/context
    
    def chat(self, user_message: str) -> str:
        """Process user message and respond."""
        # Update conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Analyze user intent
        intent = self.analyze_intent(user_message)
        
        # Update user model
        self.update_user_model(user_message, intent)
        
        # Generate contextual response
        response = self.generate_response(user_message, intent)
        
        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def generate_response(self, message: str, intent: Dict) -> str:
        """Generate context-aware response."""
        # Build context from history
        context = self.build_context()
        
        prompt = f"""
Conversation context:
{context}

User: {message}

Intent: {intent}

User model: {self.user_model}

Response:"""
        
        return self.llm(prompt)
    
    def build_context(self, max_turns: int = 5) -> str:
        """Build conversation context."""
        recent = self.conversation_history[-max_turns*2:]
        
        context_lines = []
        for turn in recent:
            role = turn["role"].title()
            context_lines.append(f"{role}: {turn['content']}")
        
        return "\n".join(context_lines)
```

## Tool-Forging Agents

Agents that create their own tools.

### Dynamic Tool Creation

```python
class ToolForgingAgent:
    """Agent that creates custom tools."""
    
    def __init__(self, llm):
        self.llm = llm
        self.custom_tools = {}
    
    def forge_tool(self, need: str) -> Callable:
        """Create custom tool for specific need."""
        # Generate tool code
        tool_code = self.llm(f"""
Create a Python function for this need:
{need}

Requirements:
- Function name
- Parameters with type hints
- Docstring
- Implementation

Python code:""")
        
        # Parse and create function
        tool_func = self.parse_and_create_function(tool_code)
        
        # Register tool
        tool_name = tool_func.__name__
        self.custom_tools[tool_name] = tool_func
        
        return tool_func
    
    def identify_tool_need(self, task: str) -> str:
        """Identify what tool is needed."""
        prompt = f"""
Task: {task}

Available tools: {list(self.custom_tools.keys())}

Is a new tool needed? If yes, describe what it should do.

Analysis:"""
        
        return self.llm(prompt)
    
    def solve_with_tool_forging(self, task: str) -> str:
        """Solve task by creating tools as needed."""
        # Check if new tool needed
        tool_need = self.identify_tool_need(task)
        
        if "yes" in tool_need.lower():
            # Forge new tool
            tool = self.forge_tool(tool_need)
            
            # Use new tool
            result = self.use_tool(tool, task)
        else:
            # Use existing tools
            result = self.solve_with_existing_tools(task)
        
        return result
```

## Meta-Agents

Agents that reason about and control other agents.

### Meta-Controller

```python
class MetaAgent:
    """Agent controlling other agents."""
    
    def __init__(self, agents: List, llm):
        self.agents = agents
        self.llm = llm
        self.performance_history = {}
    
    def solve_meta(self, task: str) -> str:
        """
        Solve task by selecting and coordinating agents.
        """
        # 1. Analyze task requirements
        requirements = self.analyze_task(task)
        
        # 2. Select best agent for task
        agent = self.select_agent(requirements)
        
        # 3. Monitor execution
        result = self.execute_with_monitoring(agent, task)
        
        # 4. Evaluate performance
        self.evaluate_and_learn(agent, task, result)
        
        return result
    
    def select_agent(self, requirements: Dict) -> Any:
        """Select most suitable agent."""
        scores = []
        
        for agent in self.agents:
            # Score based on capabilities
            capability_score = self.score_capabilities(
                agent,
                requirements
            )
            
            # Score based on past performance
            performance_score = self.get_performance_score(agent)
            
            total_score = capability_score * 0.6 + performance_score * 0.4
            scores.append((agent, total_score))
        
        # Select best
        best_agent = max(scores, key=lambda x: x[1])[0]
        
        return best_agent
    
    def execute_with_monitoring(self, agent, task):
        """Execute with performance monitoring."""
        start_time = time.time()
        
        try:
            result = agent.execute(task)
            duration = time.time() - start_time
            
            return {
                "result": result,
                "duration": duration,
                "success": True
            }
        
        except Exception as e:
            duration = time.time() - start_time
            
            return {
                "error": str(e),
                "duration": duration,
                "success": False
            }
```

## Swarm Intelligence

Multiple simple agents exhibiting collective intelligence.

### Swarm Pattern

```python
class SwarmAgent:
    """Individual agent in swarm."""
    
    def __init__(self, agent_id: str):
        self.id = agent_id
        self.position = None
        self.best_solution = None
        self.best_score = -float('inf')
    
    def explore(self, search_space):
        """Explore solution space."""
        # Random exploration
        solution = self.generate_random_solution(search_space)
        score = self.evaluate_solution(solution)
        
        # Update personal best
        if score > self.best_score:
            self.best_solution = solution
            self.best_score = score
        
        return solution, score


class SwarmIntelligence:
    """Swarm of agents solving problems collectively."""
    
    def __init__(self, num_agents: int = 10):
        self.agents = [
            SwarmAgent(f"agent_{i}")
            for i in range(num_agents)
        ]
        self.global_best = None
        self.global_best_score = -float('inf')
    
    def solve_swarm(self, problem, max_iterations: int = 100):
        """Solve problem using swarm."""
        for iteration in range(max_iterations):
            # Each agent explores
            for agent in self.agents:
                solution, score = agent.explore(problem)
                
                # Update global best
                if score > self.global_best_score:
                    self.global_best = solution
                    self.global_best_score = score
            
            # Agents communicate and adjust
            self.swarm_communication()
            
            # Check convergence
            if self.has_converged():
                break
        
        return self.global_best
    
    def swarm_communication(self):
        """Agents share information."""
        for agent in self.agents:
            # Learn from global best
            agent.adjust_strategy(self.global_best)
```

## Evolutionary Agents

Agents that evolve strategies over time.

### Genetic Algorithm Pattern

```python
class EvolvingAgent:
    """Agent with evolvable strategy."""
    
    def __init__(self, genome):
        self.genome = genome  # Strategy encoding
        self.fitness = 0
    
    def mutate(self):
        """Randomly modify strategy."""
        # Simple mutation: modify random gene
        import random
        
        mutated_genome = self.genome.copy()
        mutation_point = random.randint(0, len(mutated_genome) - 1)
        mutated_genome[mutation_point] = random.random()
        
        return EvolvingAgent(mutated_genome)
    
    def crossover(self, other):
        """Combine strategies from two agents."""
        # Single-point crossover
        import random
        
        point = random.randint(1, len(self.genome) - 1)
        child_genome = self.genome[:point] + other.genome[point:]
        
        return EvolvingAgent(child_genome)


class EvolutionarySystem:
    """Evolve population of agents."""
    
    def __init__(self, population_size: int = 50):
        self.population = self.initialize_population(population_size)
    
    def evolve(self, task, generations: int = 100):
        """Evolve agents over generations."""
        for generation in range(generations):
            # Evaluate fitness
            for agent in self.population:
                agent.fitness = self.evaluate_fitness(agent, task)
            
            # Selection
            selected = self.select_parents(self.population)
            
            # Create next generation
            next_generation = []
            
            # Elitism: keep best
            best = max(self.population, key=lambda a: a.fitness)
            next_generation.append(best)
            
            # Crossover and mutation
            while len(next_generation) < len(self.population):
                parent1, parent2 = random.sample(selected, 2)
                child = parent1.crossover(parent2)
                
                if random.random() < 0.1:  # Mutation rate
                    child = child.mutate()
                
                next_generation.append(child)
            
            self.population = next_generation
        
        # Return best agent
        return max(self.population, key=lambda a: a.fitness)
```

## Symbolic-Neural Hybrid Agents

Combining symbolic reasoning with neural networks.

### Hybrid Architecture

```python
class HybridAgent:
    """Agent combining symbolic and neural approaches."""
    
    def __init__(self, neural_model, symbolic_engine):
        self.neural = neural_model
        self.symbolic = symbolic_engine
    
    def solve_hybrid(self, problem: str) -> str:
        """Solve using both neural and symbolic."""
        # 1. Neural perception
        perception = self.neural.perceive(problem)
        
        # 2. Symbolic reasoning
        symbolic_solution = self.symbolic.reason(perception)
        
        # 3. Neural refinement
        refined = self.neural.refine(symbolic_solution)
        
        # 4. Symbolic verification
        verified = self.symbolic.verify(refined, problem)
        
        if verified:
            return refined
        else:
            return self.resolve_discrepancy(
                neural=refined,
                symbolic=symbolic_solution
            )
```

## Human-in-the-Loop Patterns

Involving humans at critical points.

### Approval Gates

```python
class HumanInTheLoopAgent:
    """Agent with human approval gates."""
    
    def __init__(self, llm):
        self.llm = llm
        self.approval_threshold = 0.8  # Confidence threshold
    
    def execute_with_approval(self, task: str) -> str:
        """Execute with human approval for critical actions."""
        # Plan actions
        plan = self.create_plan(task)
        
        # Execute with approval gates
        for action in plan:
            # Assess risk
            risk = self.assess_risk(action)
            confidence = self.assess_confidence(action)
            
            # Require approval for risky/uncertain actions
            if risk > 0.5 or confidence < self.approval_threshold:
                approved = self.request_human_approval(action, risk)
                
                if not approved:
                    return "Action rejected by human"
            
            # Execute
            result = self.execute_action(action)
        
        return "Task completed"
    
    def request_human_approval(self, action: Dict, risk: float) -> bool:
        """Request human approval."""
        print(f"\n⚠️  Approval Required")
        print(f"Action: {action}")
        print(f"Risk level: {risk:.2f}")
        
        response = input("Approve? (yes/no): ").strip().lower()
        
        return response == "yes"
```

## Summary

Advanced agent patterns for sophisticated scenarios:

1. **Subagents**: Master-worker and hierarchical coordination
2. **Filesystem-Aware**: Using filesystem for memory and artifacts
3. **Artifact-Generating**: Producing versioned work products
4. **Multi-Modal**: Handling text, images, audio, etc.
5. **Conversational**: Natural dialogue with context
6. **Tool-Forging**: Creating custom tools dynamically
7. **Meta-Agents**: Controlling and coordinating other agents
8. **Swarm Intelligence**: Collective problem-solving
9. **Evolutionary**: Evolving strategies over time
10. **Symbolic-Neural**: Combining reasoning paradigms
11. **Human-in-the-Loop**: Critical human oversight

These patterns represent the cutting edge of agent architectures, enabling sophisticated real-world applications.

## Next Steps

- Review [Design Principles](design-principles.md) for building robust agent systems
- Study [Autonomous Agents](autonomous-agents.md) for long-running agents
- Explore [Planning Architectures](planning-architectures.md) for structured approaches
- Continue to [Reflection](reflection.md) for self-improvement patterns
