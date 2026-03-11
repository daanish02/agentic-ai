# Coordination and Task Allocation

## Table of Contents

- [Introduction](#introduction)
- [What is Coordination?](#what-is-coordination)
- [Task Allocation Strategies](#task-allocation-strategies)
- [Work Distribution](#work-distribution)
- [Synchronization](#synchronization)
- [Conflict Resolution](#conflict-resolution)
- [Resource Management](#resource-management)
- [Load Balancing](#load-balancing)
- [Dependency Management](#dependency-management)
- [Progress Tracking](#progress-tracking)
- [Coordination Patterns](#coordination-patterns)
- [Dynamic Coordination](#dynamic-coordination)
- [Coordination Overhead](#coordination-overhead)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Coordination is the art of making multiple agents work together effectively. Without coordination, agents might duplicate work, conflict with each other, or fail to accomplish shared goals. Good coordination ensures agents complement rather than interfere with each other.

> "Coordination is the hidden complexity cost of distribution."

This guide covers:

- How to allocate tasks among agents
- Strategies for distributing work
- Synchronizing agent activities
- Resolving conflicts when they arise
- Managing shared resources
- Tracking progress and ensuring completion

## What is Coordination?

**Coordination** is the process of organizing agent activities so they work together toward common goals.

### Coordination Challenges

```python
"""
Without Coordination:

Agent 1: "I'll handle task A"
Agent 2: "I'll handle task A"  # Duplicate work!

Agent 1: Writes to database
Agent 2: Writes to database  # Race condition!

Agent 1: Needs resource X
Agent 2: Needs resource X  # Resource conflict!

Agent 1: Waiting for Agent 2
Agent 2: Waiting for Agent 1  # Deadlock!
"""

class UncoordinatedSystem:
    """Problems without coordination."""

    async def parallel_work(self, task):
        # Both agents work independently
        result1 = await self.agent1.work(task)
        result2 = await self.agent2.work(task)

        # Problems:
        # - Duplicate effort (both did same work)
        # - Conflicting results
        # - Wasted resources
        # - No division of labor

"""
With Coordination:

Coordinator: "Agent 1 handles A, Agent 2 handles B"
Agent 1: Works on A
Agent 2: Works on B
Coordinator: Combines results
"""

class CoordinatedSystem:
    """Organized work distribution."""

    async def coordinated_work(self, tasks):
        # Coordinator assigns distinct work
        assignment = {
            "agent1": tasks[0],
            "agent2": tasks[1]
        }

        # Agents work in parallel on different tasks
        results = await asyncio.gather(
            self.agent1.work(assignment["agent1"]),
            self.agent2.work(assignment["agent2"])
        )

        # Benefits:
        # - No duplication
        # - Parallel progress
        # - Clear responsibilities
        # - Efficient resource use

        return self.combine_results(results)
```

### Levels of Coordination

```python
class CoordinationLevels:
    """Different levels of coordination complexity."""

    # Level 1: No Coordination
    # Agents work completely independently
    # Suitable when tasks are fully independent

    async def level1_independent(self, tasks):
        """No coordination needed."""
        return await asyncio.gather(*[
            agent.work(task)
            for agent, task in zip(self.agents, tasks)
        ])

    # Level 2: Simple Allocation
    # Central coordinator assigns tasks
    # No inter-agent communication needed

    async def level2_allocation(self, tasks):
        """Coordinator assigns work."""
        assignments = self.coordinator.allocate(tasks, self.agents)

        return await asyncio.gather(*[
            agent.work(assignment)
            for agent, assignment in assignments.items()
        ])

    # Level 3: Synchronized Execution
    # Agents must synchronize at checkpoints
    # Some inter-agent dependencies

    async def level3_synchronized(self, tasks):
        """Agents synchronize at points."""
        # Phase 1
        results1 = await self.synchronized_phase(tasks, phase=1)

        # Synchronization point
        await self.barrier.wait()

        # Phase 2 (uses Phase 1 results)
        results2 = await self.synchronized_phase(results1, phase=2)

        return results2

    # Level 4: Full Collaboration
    # Agents constantly communicate and coordinate
    # Complex dependencies and interactions

    async def level4_collaborative(self, task):
        """Agents collaborate continuously."""
        result = None

        while not self.is_complete(result):
            # Agents coordinate next steps
            plan = await self.coordinate_next_steps(result)

            # Agents execute in coordination
            result = await self.execute_coordinated(plan)

        return result
```

## Task Allocation Strategies

How to assign tasks to agents:

### 1. Centralized Allocation

```python
class CentralizedAllocator:
    """Central coordinator assigns all tasks."""

    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.task_queue = asyncio.Queue()
        self.assignments = {}

    async def allocate(self, tasks: List[Dict]) -> Dict[str, List[Dict]]:
        """Allocate tasks to agents."""

        assignments = {agent.id: [] for agent in self.agents}

        # Simple round-robin allocation
        for i, task in enumerate(tasks):
            agent = self.agents[i % len(self.agents)]
            assignments[agent.id].append(task)

        return assignments

    async def smart_allocate(self, tasks: List[Dict]) -> Dict[str, List[Dict]]:
        """Allocate based on agent capabilities and load."""

        assignments = {agent.id: [] for agent in self.agents}

        for task in tasks:
            # Find best agent for this task
            best_agent = self.find_best_agent(task)
            assignments[best_agent.id].append(task)

        return assignments

    def find_best_agent(self, task: Dict) -> Agent:
        """Find most suitable agent for task."""

        scores = []

        for agent in self.agents:
            score = (
                agent.get_capability_score(task) * 0.4 +  # Capability match
                (1.0 - agent.current_load()) * 0.3 +      # Availability
                agent.get_success_rate() * 0.3            # Reliability
            )
            scores.append((agent, score))

        # Return highest scoring agent
        return max(scores, key=lambda x: x[1])[0]

# Example usage
allocator = CentralizedAllocator([agent1, agent2, agent3])

tasks = [
    {"type": "research", "query": "topic1"},
    {"type": "analysis", "data": [1, 2, 3]},
    {"type": "research", "query": "topic2"}
]

# Allocate based on agent capabilities
assignments = await allocator.smart_allocate(tasks)

# Execute assignments
for agent_id, agent_tasks in assignments.items():
    agent = get_agent(agent_id)
    await agent.execute_tasks(agent_tasks)
```

**Pros:**

- Simple to implement and understand
- Global view enables optimal allocation
- Easy to reason about system state
- Clear accountability

**Cons:**

- Single point of failure (coordinator)
- Scalability bottleneck
- Coordinator can be overwhelmed
- Latency from centralization

### 2. Decentralized Allocation

```python
class DecentralizedAllocation:
    """Agents self-organize to claim tasks."""

    def __init__(self):
        self.task_board = SharedTaskBoard()
        self.agents = []

    async def publish_tasks(self, tasks: List[Dict]):
        """Post tasks to shared board."""

        for task in tasks:
            await self.task_board.post(task)

    async def agent_worker(self, agent: Agent):
        """Agent claims and executes tasks."""

        while True:
            # Look for suitable task
            task = await self.find_suitable_task(agent)

            if task:
                # Try to claim task
                claimed = await self.task_board.claim(task.id, agent.id)

                if claimed:
                    # Execute task
                    result = await agent.execute(task)

                    # Mark complete
                    await self.task_board.complete(task.id, result)
            else:
                # No suitable tasks, wait
                await asyncio.sleep(1)

    async def find_suitable_task(self, agent: Agent) -> Optional[Dict]:
        """Find task matching agent capabilities."""

        available_tasks = await self.task_board.get_available()

        # Filter to tasks agent can handle
        suitable = [
            task for task in available_tasks
            if agent.can_handle(task)
        ]

        if not suitable:
            return None

        # Select highest priority/best match
        return max(suitable, key=lambda t: (
            t.get("priority", 0),
            agent.get_capability_score(t)
        ))

class SharedTaskBoard:
    """Shared board for task coordination."""

    def __init__(self):
        self.tasks = {}
        self.locks = {}

    async def post(self, task: Dict) -> str:
        """Post task to board."""
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            "task": task,
            "status": "available",
            "claimed_by": None
        }
        self.locks[task_id] = asyncio.Lock()
        return task_id

    async def claim(self, task_id: str, agent_id: str) -> bool:
        """Try to claim task."""

        async with self.locks[task_id]:
            if self.tasks[task_id]["status"] == "available":
                self.tasks[task_id]["status"] = "claimed"
                self.tasks[task_id]["claimed_by"] = agent_id
                return True
            return False

    async def complete(self, task_id: str, result: Any):
        """Mark task complete."""
        self.tasks[task_id]["status"] = "complete"
        self.tasks[task_id]["result"] = result

    async def get_available(self) -> List[Dict]:
        """Get all available tasks."""
        return [
            {"id": tid, **data["task"]}
            for tid, data in self.tasks.items()
            if data["status"] == "available"
        ]

# Example usage
system = DecentralizedAllocation()

# Post tasks
await system.publish_tasks([
    {"type": "research", "query": "topic1"},
    {"type": "research", "query": "topic2"},
    {"type": "analysis", "data": [1, 2, 3]}
])

# Start agent workers
workers = [
    asyncio.create_task(system.agent_worker(agent))
    for agent in system.agents
]

# Agents self-organize to complete tasks
await asyncio.gather(*workers)
```

**Pros:**

- No single point of failure
- Scales better with many agents
- Agents select best-fit tasks
- Natural load balancing

**Cons:**

- More complex
- Potential for conflicts
- Harder to get global optimum
- Requires synchronization primitives

### 3. Auction-Based Allocation

```python
class AuctionBasedAllocation:
    """Agents bid for tasks."""

    async def allocate_task(self, task: Dict) -> Agent:
        """Allocate task through auction."""

        # Request bids from all agents
        bids = await self.request_bids(task)

        # Select winning bid
        winner = self.select_winner(bids)

        # Award task to winner
        await self.award_task(task, winner)

        return winner

    async def request_bids(self, task: Dict) -> List[Dict]:
        """Request bids from agents."""

        bid_requests = [
            agent.submit_bid(task)
            for agent in self.agents
        ]

        bids = await asyncio.gather(*bid_requests)
        return bids

    def select_winner(self, bids: List[Dict]) -> Agent:
        """Select winning bid."""

        # Score each bid
        scores = []
        for bid in bids:
            score = (
                bid["confidence"] * 0.4 +        # Confidence
                (1.0 - bid["cost"]) * 0.3 +      # Low cost
                (1.0 - bid["time"]) * 0.3        # Fast completion
            )
            scores.append((bid["agent"], score))

        # Return highest scorer
        return max(scores, key=lambda x: x[1])[0]

    async def award_task(self, task: Dict, winner: Agent):
        """Award task to winning agent."""

        await winner.receive_task(task)

class BiddingAgent:
    """Agent that can bid for tasks."""

    async def submit_bid(self, task: Dict) -> Dict:
        """Submit bid for task."""

        # Evaluate if we can handle task
        if not self.can_handle(task):
            return {
                "agent": self,
                "confidence": 0.0,
                "cost": 1.0,
                "time": 1.0
            }

        # Estimate our performance
        confidence = self.estimate_confidence(task)
        cost = self.estimate_cost(task)
        time = self.estimate_time(task)

        return {
            "agent": self,
            "confidence": confidence,
            "cost": cost,
            "time": time
        }

    def estimate_confidence(self, task: Dict) -> float:
        """Estimate confidence in handling task."""
        # Based on past performance, capabilities, etc.
        return 0.8

    def estimate_cost(self, task: Dict) -> float:
        """Estimate cost (normalized 0-1)."""
        # Based on resource requirements
        return 0.3

    def estimate_time(self, task: Dict) -> float:
        """Estimate time (normalized 0-1)."""
        # Based on complexity and current load
        return 0.4

# Example usage
allocator = AuctionBasedAllocation()

task = {"type": "analysis", "complexity": 7}

# Run auction
winner = await allocator.allocate_task(task)
print(f"Task awarded to {winner.id}")
```

**Pros:**

- Efficient market-based allocation
- Agents have incentive to bid accurately
- Good for heterogeneous agents
- Natural cost optimization

**Cons:**

- Overhead of bidding process
- Requires careful mechanism design
- Agents might game the system
- More complex implementation

## Work Distribution

Strategies for dividing work among agents:

### Parallel Distribution

```python
class ParallelDistribution:
    """Distribute work for parallel execution."""

    async def distribute_parallel(self, tasks: List[Dict]) -> List[Any]:
        """Execute tasks in parallel across agents."""

        # Divide tasks among available agents
        agent_tasks = self.divide_tasks(tasks, self.agents)

        # Each agent executes their tasks in parallel
        results = await asyncio.gather(*[
            agent.execute_batch(tasks)
            for agent, tasks in agent_tasks.items()
        ])

        # Flatten results
        return [r for batch in results for r in batch]

    def divide_tasks(self, tasks: List[Dict],
                    agents: List[Agent]) -> Dict[Agent, List[Dict]]:
        """Divide tasks evenly among agents."""

        tasks_per_agent = len(tasks) // len(agents)
        remainder = len(tasks) % len(agents)

        agent_tasks = {}
        task_idx = 0

        for i, agent in enumerate(agents):
            # Give extra task to first 'remainder' agents
            count = tasks_per_agent + (1 if i < remainder else 0)
            agent_tasks[agent] = tasks[task_idx:task_idx + count]
            task_idx += count

        return agent_tasks

# Example
distributor = ParallelDistribution()

tasks = [{"id": i, "work": f"task_{i}"} for i in range(100)]

# Distribute across 10 agents
# Each agent gets ~10 tasks
results = await distributor.distribute_parallel(tasks)
```

### Pipeline Distribution

```python
class PipelineDistribution:
    """Distribute work as a pipeline."""

    def __init__(self, stages: List[Agent]):
        self.stages = stages

    async def process_pipeline(self, items: List[Any]) -> List[Any]:
        """Process items through pipeline."""

        # Stage 1 processes all items
        results = await self.stages[0].process_batch(items)

        # Each subsequent stage processes previous results
        for stage in self.stages[1:]:
            results = await stage.process_batch(results)

        return results

    async def streaming_pipeline(self, items: List[Any]) -> List[Any]:
        """Stream items through pipeline for lower latency."""

        # Create queues between stages
        queues = [asyncio.Queue() for _ in range(len(self.stages) + 1)]

        # Put items in first queue
        for item in items:
            await queues[0].put(item)
        await queues[0].put(None)  # Sentinel

        # Start pipeline stages
        tasks = [
            asyncio.create_task(
                self.pipeline_stage(self.stages[i], queues[i], queues[i+1])
            )
            for i in range(len(self.stages))
        ]

        # Collect results from final queue
        results = []
        while True:
            result = await queues[-1].get()
            if result is None:
                break
            results.append(result)

        await asyncio.gather(*tasks)
        return results

    async def pipeline_stage(self, agent: Agent,
                            input_queue: asyncio.Queue,
                            output_queue: asyncio.Queue):
        """Process items from input to output queue."""

        while True:
            item = await input_queue.get()

            if item is None:  # Sentinel
                await output_queue.put(None)
                break

            # Process item
            result = await agent.process(item)

            # Pass to next stage
            await output_queue.put(result)

# Example
pipeline = PipelineDistribution([
    DataCleaningAgent(),
    FeatureExtractionAgent(),
    ModelPredictionAgent()
])

# Stream data through pipeline
data = load_dataset()
predictions = await pipeline.streaming_pipeline(data)
```

### MapReduce Distribution

```python
class MapReduceDistribution:
    """MapReduce-style work distribution."""

    def __init__(self, mappers: List[Agent], reducers: List[Agent]):
        self.mappers = mappers
        self.reducers = reducers

    async def map_reduce(self, data: List[Any]) -> Any:
        """Execute MapReduce."""

        # Map phase: Distribute data to mappers
        map_results = await self.map_phase(data)

        # Shuffle phase: Group by key
        shuffled = self.shuffle_phase(map_results)

        # Reduce phase: Aggregate by key
        final_result = await self.reduce_phase(shuffled)

        return final_result

    async def map_phase(self, data: List[Any]) -> List[List[Tuple]]:
        """Distribute data to mappers."""

        # Divide data among mappers
        chunks = self.chunk_data(data, len(self.mappers))

        # Each mapper processes its chunk
        map_results = await asyncio.gather(*[
            mapper.map(chunk)
            for mapper, chunk in zip(self.mappers, chunks)
        ])

        return map_results

    def shuffle_phase(self, map_results: List[List[Tuple]]) -> Dict[Any, List[Any]]:
        """Group mapped results by key."""

        grouped = {}

        for results in map_results:
            for key, value in results:
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(value)

        return grouped

    async def reduce_phase(self, grouped: Dict[Any, List[Any]]) -> Dict[Any, Any]:
        """Reduce grouped values."""

        # Distribute keys to reducers
        reducer_keys = self.distribute_keys(grouped.keys(), len(self.reducers))

        # Each reducer processes its keys
        reduce_tasks = []
        for reducer, keys in zip(self.reducers, reducer_keys):
            reducer_data = {k: grouped[k] for k in keys}
            reduce_tasks.append(reducer.reduce(reducer_data))

        results = await asyncio.gather(*reduce_tasks)

        # Merge all results
        final = {}
        for result in results:
            final.update(result)

        return final

    def chunk_data(self, data: List[Any], n: int) -> List[List[Any]]:
        """Split data into n chunks."""
        chunk_size = len(data) // n
        return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

    def distribute_keys(self, keys: List[Any], n: int) -> List[List[Any]]:
        """Distribute keys among n reducers."""
        keys = list(keys)
        chunk_size = len(keys) // n
        return [keys[i:i+chunk_size] for i in range(0, len(keys), chunk_size)]

# Example: Word count
class MapperAgent:
    async def map(self, documents: List[str]) -> List[Tuple[str, int]]:
        """Map documents to (word, 1) pairs."""
        results = []
        for doc in documents:
            for word in doc.split():
                results.append((word, 1))
        return results

class ReducerAgent:
    async def reduce(self, grouped: Dict[str, List[int]]) -> Dict[str, int]:
        """Reduce to word counts."""
        return {word: sum(counts) for word, counts in grouped.items()}

# Execute MapReduce word count
mapreduce = MapReduceDistribution(
    mappers=[MapperAgent() for _ in range(4)],
    reducers=[ReducerAgent() for _ in range(2)]
)

documents = ["hello world", "hello python", "world python"]
word_counts = await mapreduce.map_reduce(documents)
print(word_counts)  # {"hello": 2, "world": 2, "python": 2}
```

## Synchronization

Coordinating agent timing:

### Barriers

```python
class Barrier:
    """Synchronization barrier for agents."""

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.count = 0
        self.condition = asyncio.Condition()

    async def wait(self):
        """Wait for all agents to reach barrier."""

        async with self.condition:
            self.count += 1

            if self.count == self.n_agents:
                # Last agent arrives - release all
                self.count = 0
                self.condition.notify_all()
            else:
                # Wait for others
                await self.condition.wait()

# Example usage
class SynchronizedWorkflow:
    """Workflow with synchronization points."""

    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.barrier = Barrier(len(agents))

    async def execute_synchronized(self, tasks: List[Dict]):
        """Execute with synchronization points."""

        # Phase 1: All agents work independently
        phase1_tasks = [
            agent.execute_phase1(tasks[i])
            for i, agent in enumerate(self.agents)
        ]
        results1 = await asyncio.gather(*phase1_tasks)

        # Synchronization point
        await self.barrier.wait()

        # Phase 2: Use Phase 1 results
        phase2_tasks = [
            agent.execute_phase2(results1)
            for agent in self.agents
        ]
        results2 = await asyncio.gather(*phase2_tasks)

        # Synchronization point
        await self.barrier.wait()

        # Phase 3: Final processing
        final = await self.combine_results(results2)

        return final

# All agents wait at barriers
# Ensures phase completion before next phase
```

### Locks and Semaphores

```python
class ResourceCoordination:
    """Coordinate access to shared resources."""

    def __init__(self):
        self.locks = {}
        self.semaphores = {}

    def get_lock(self, resource: str) -> asyncio.Lock:
        """Get lock for exclusive resource access."""

        if resource not in self.locks:
            self.locks[resource] = asyncio.Lock()

        return self.locks[resource]

    def get_semaphore(self, resource: str, capacity: int) -> asyncio.Semaphore:
        """Get semaphore for limited resource access."""

        if resource not in self.semaphores:
            self.semaphores[resource] = asyncio.Semaphore(capacity)

        return self.semaphores[resource]

    async def exclusive_access(self, resource: str, agent: Agent,
                              operation: Callable):
        """Execute operation with exclusive access."""

        lock = self.get_lock(resource)

        async with lock:
            print(f"{agent.id} acquired exclusive access to {resource}")
            result = await operation()
            print(f"{agent.id} released {resource}")
            return result

    async def limited_access(self, resource: str, capacity: int,
                            agent: Agent, operation: Callable):
        """Execute operation with limited concurrent access."""

        semaphore = self.get_semaphore(resource, capacity)

        async with semaphore:
            print(f"{agent.id} acquired access to {resource}")
            result = await operation()
            print(f"{agent.id} released {resource}")
            return result

# Example
coordinator = ResourceCoordination()

# Exclusive access to database
async def write_to_db():
    await database.write(data)

await coordinator.exclusive_access("database", agent1, write_to_db)

# Limited concurrent API calls (max 5)
async def call_api():
    return await api.request(query)

await coordinator.limited_access("api", 5, agent2, call_api)
```

## Conflict Resolution

Handle conflicts when agents disagree or compete:

### Voting-Based Resolution

```python
class VotingResolution:
    """Resolve conflicts through voting."""

    async def resolve_by_vote(self, proposals: List[Any],
                             agents: List[Agent]) -> Any:
        """Agents vote on proposals."""

        # Collect votes
        votes = {}
        for proposal in proposals:
            votes[str(proposal)] = 0

        # Each agent votes
        for agent in agents:
            vote = await agent.vote(proposals)
            votes[str(vote)] += 1

        # Return proposal with most votes
        winner = max(votes.items(), key=lambda x: x[1])
        return eval(winner[0])  # Convert back from string

    async def weighted_vote(self, proposals: List[Any],
                           agents: List[Agent]) -> Any:
        """Weighted voting based on agent expertise."""

        votes = {}
        for proposal in proposals:
            votes[str(proposal)] = 0.0

        # Weighted votes
        for agent in agents:
            vote = await agent.vote(proposals)
            weight = agent.get_expertise_weight()
            votes[str(vote)] += weight

        # Return highest weighted proposal
        winner = max(votes.items(), key=lambda x: x[1])
        return eval(winner[0])

# Example
resolver = VotingResolution()

# Agents propose different approaches
proposals = [
    {"approach": "method_a", "cost": 100},
    {"approach": "method_b", "cost": 200},
    {"approach": "method_c", "cost": 150}
]

# Vote to resolve
chosen = await resolver.resolve_by_vote(proposals, agents)
print(f"Chosen approach: {chosen}")
```

### Priority-Based Resolution

```python
class PriorityResolution:
    """Resolve conflicts using priorities."""

    def __init__(self):
        self.agent_priorities = {}

    def set_priority(self, agent_id: str, priority: int):
        """Set agent priority (higher = more authority)."""
        self.agent_priorities[agent_id] = priority

    async def resolve_by_priority(self, conflicts: List[Tuple[str, Any]]) -> Any:
        """Resolve conflict using agent priorities."""

        # conflicts: [(agent_id, proposal), ...]

        # Find highest priority agent
        highest_priority = -1
        winner = None

        for agent_id, proposal in conflicts:
            priority = self.agent_priorities.get(agent_id, 0)
            if priority > highest_priority:
                highest_priority = priority
                winner = proposal

        return winner

    async def resolve_with_escalation(self, conflicts: List[Tuple[str, Any]],
                                     supervisor: Agent) -> Any:
        """Escalate unresolvable conflicts to supervisor."""

        # Try priority resolution
        if len(set(self.agent_priorities[a] for a, _ in conflicts)) > 1:
            # Different priorities - use priority
            return await self.resolve_by_priority(conflicts)

        # Same priorities - escalate to supervisor
        proposals = [p for _, p in conflicts]
        return await supervisor.resolve(proposals)

# Example
resolver = PriorityResolution()

# Set priorities
resolver.set_priority("expert_agent", 10)
resolver.set_priority("junior_agent", 5)

# Resolve conflict
conflicts = [
    ("expert_agent", "approach_a"),
    ("junior_agent", "approach_b")
]

chosen = await resolver.resolve_by_priority(conflicts)
print(f"Chosen: {chosen}")  # "approach_a" (expert wins)
```

### Consensus-Based Resolution

```python
class ConsensusResolution:
    """Resolve through consensus building."""

    async def build_consensus(self, agents: List[Agent],
                            topic: Dict, threshold: float = 0.8) -> Optional[Any]:
        """Build consensus on topic."""

        max_rounds = 5

        for round_num in range(max_rounds):
            # Each agent proposes
            proposals = await asyncio.gather(*[
                agent.propose(topic)
                for agent in agents
            ])

            # Check for consensus
            agreement = self.measure_agreement(proposals)

            if agreement >= threshold:
                # Consensus reached
                return self.merge_proposals(proposals)

            # Share proposals for next round
            for agent in agents:
                await agent.see_proposals(proposals)

        # No consensus reached
        return None

    def measure_agreement(self, proposals: List[Any]) -> float:
        """Measure level of agreement."""

        # Simple: percentage of identical proposals
        if not proposals:
            return 0.0

        most_common = max(set(str(p) for p in proposals),
                         key=lambda p: [str(x) for x in proposals].count(p))

        agreement = [str(p) for p in proposals].count(most_common) / len(proposals)

        return agreement

    def merge_proposals(self, proposals: List[Any]) -> Any:
        """Merge similar proposals."""

        # In practice, this would intelligently merge
        # For now, return most common
        return max(set(proposals), key=proposals.count)

# Example
resolver = ConsensusResolution()

topic = {"question": "What approach should we use?"}

# Build consensus
consensus = await resolver.build_consensus(agents, topic, threshold=0.8)

if consensus:
    print(f"Consensus reached: {consensus}")
else:
    print("No consensus - escalating")
```

## Summary

Coordination is essential for multi-agent systems:

**Key Strategies:**

- Centralized allocation for simplicity and control
- Decentralized allocation for scalability
- Auction-based for efficiency

**Work Distribution:**

- Parallel for independent tasks
- Pipeline for sequential processing
- MapReduce for large-scale data

**Synchronization:**

- Barriers for phase transitions
- Locks for exclusive access
- Semaphores for limited resources

**Conflict Resolution:**

- Voting for democratic decisions
- Priority for hierarchical authority
- Consensus for collaborative agreement

Effective coordination enables agents to work together productively without interference or duplication.

## Next Steps

- [Delegation and Hierarchies](delegation.md) - Hierarchical coordination
- [Collaboration Patterns](collaboration.md) - Cooperative work
- [Communication Patterns](communication.md) - Inter-agent messaging
- [Agent Roles and Specialization](agent-roles.md) - Role design
- [Multi-Agent Fundamentals](fundamentals.md) - Core principles
