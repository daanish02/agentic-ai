# Swarm Intelligence

## Table of Contents

- [Introduction](#introduction)
- [Swarm Principles](#swarm-principles)
- [Emergent Behavior](#emergent-behavior)
- [Decentralized Coordination](#decentralized-coordination)
- [Self-Organization](#self-organization)
- [Stigmergy](#stigmergy)
- [Particle Swarm Optimization](#particle-swarm-optimization)
- [Ant Colony Optimization](#ant-colony-optimization)
- [Flocking and Boids](#flocking-and-boids)
- [Consensus in Swarms](#consensus-in-swarms)
- [Robustness and Scalability](#robustness-and-scalability)
- [When to Use Swarms](#when-to-use-swarms)
- [Implementation Patterns](#implementation-patterns)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**Swarm intelligence** is collective behavior that emerges from many simple agents following local rules, without centralized control. Like ant colonies or bird flocks, swarm systems achieve complex global behavior through simple local interactions.

> "The whole is greater than the sum of its parts."  
> -- Aristotle

**Key Characteristics:**

- Many simple agents (not a few complex ones)
- Local interactions only (no global view)
- Decentralized control (no leader)
- Emergent global behavior
- Robust and scalable

This guide explores:

- How global behavior emerges from local rules
- Classic swarm algorithms
- When swarm approaches are appropriate
- Implementing swarm systems

## Swarm Principles

Core principles of swarm intelligence:

### Local Rules, Global Behavior

```python
class SwarmAgent:
    """Simple agent following local rules."""

    def __init__(self, position: Tuple[float, float], agent_id: str):
        self.position = position
        self.velocity = (0.0, 0.0)
        self.id = agent_id
        self.neighbors = []

    def sense_environment(self, all_agents: List['SwarmAgent'],
                         sensor_range: float):
        """Sense nearby agents (local perception)."""

        self.neighbors = []

        for agent in all_agents:
            if agent.id == self.id:
                continue

            distance = self.distance_to(agent)

            if distance < sensor_range:
                self.neighbors.append({
                    'agent': agent,
                    'distance': distance,
                    'direction': self.direction_to(agent)
                })

    def update_behavior(self):
        """Update behavior based on local information only."""

        if not self.neighbors:
            # No neighbors - random walk
            self.velocity = self.random_velocity()
            return

        # Simple flocking rules (local)
        separation = self.compute_separation()
        alignment = self.compute_alignment()
        cohesion = self.compute_cohesion()

        # Combine behaviors
        self.velocity = self.combine_vectors([
            (separation, 1.5),  # Weight
            (alignment, 1.0),
            (cohesion, 1.0)
        ])

    def compute_separation(self) -> Tuple[float, float]:
        """Avoid crowding neighbors."""

        separation = [0.0, 0.0]

        for neighbor in self.neighbors:
            if neighbor['distance'] < 1.0:  # Too close
                # Move away
                direction = neighbor['direction']
                separation[0] -= direction[0] / neighbor['distance']
                separation[1] -= direction[1] / neighbor['distance']

        return tuple(separation)

    def compute_alignment(self) -> Tuple[float, float]:
        """Align with neighbors' average heading."""

        if not self.neighbors:
            return (0.0, 0.0)

        avg_velocity = [0.0, 0.0]

        for neighbor in self.neighbors:
            agent = neighbor['agent']
            avg_velocity[0] += agent.velocity[0]
            avg_velocity[1] += agent.velocity[1]

        avg_velocity[0] /= len(self.neighbors)
        avg_velocity[1] /= len(self.neighbors)

        return tuple(avg_velocity)

    def compute_cohesion(self) -> Tuple[float, float]:
        """Move toward neighbors' average position."""

        if not self.neighbors:
            return (0.0, 0.0)

        center = [0.0, 0.0]

        for neighbor in self.neighbors:
            agent = neighbor['agent']
            center[0] += agent.position[0]
            center[1] += agent.position[1]

        center[0] /= len(self.neighbors)
        center[1] /= len(self.neighbors)

        # Direction toward center
        direction = (
            center[0] - self.position[0],
            center[1] - self.position[1]
        )

        return direction

    def move(self):
        """Update position."""

        self.position = (
            self.position[0] + self.velocity[0],
            self.position[1] + self.velocity[1]
        )

    def distance_to(self, other: 'SwarmAgent') -> float:
        """Distance to another agent."""

        dx = other.position[0] - self.position[0]
        dy = other.position[1] - self.position[1]

        return math.sqrt(dx*dx + dy*dy)

    def direction_to(self, other: 'SwarmAgent') -> Tuple[float, float]:
        """Unit vector toward another agent."""

        dx = other.position[0] - self.position[0]
        dy = other.position[1] - self.position[1]

        distance = math.sqrt(dx*dx + dy*dy)

        if distance > 0:
            return (dx / distance, dy / distance)
        else:
            return (0.0, 0.0)

# Swarm simulation
class SwarmSimulation:
    """Simulate swarm behavior."""

    def __init__(self, num_agents: int):
        self.agents = [
            SwarmAgent(
                position=(random.uniform(0, 100), random.uniform(0, 100)),
                agent_id=f"agent_{i}"
            )
            for i in range(num_agents)
        ]
        self.sensor_range = 10.0

    def step(self):
        """One simulation step."""

        # Agents sense environment
        for agent in self.agents:
            agent.sense_environment(self.agents, self.sensor_range)

        # Agents update behavior
        for agent in self.agents:
            agent.update_behavior()

        # Agents move
        for agent in self.agents:
            agent.move()

    def run(self, steps: int):
        """Run simulation."""

        for _ in range(steps):
            self.step()

# Example
swarm = SwarmSimulation(num_agents=50)
swarm.run(steps=100)
```

## Emergent Behavior

Global patterns from local rules:

### Pattern Formation

```python
class PatternFormingSwarm:
    """Swarm that forms spatial patterns."""

    def __init__(self, agents: List[SwarmAgent]):
        self.agents = agents
        self.pattern_detected = None

    async def evolve(self, steps: int):
        """Evolve swarm and detect patterns."""

        for step in range(steps):
            # Agents act based on local rules
            for agent in self.agents:
                await agent.act_locally()

            # Detect emergent pattern (global observer)
            pattern = self.detect_pattern()

            if pattern != self.pattern_detected:
                print(f"Step {step}: New pattern detected: {pattern}")
                self.pattern_detected = pattern

    def detect_pattern(self) -> str:
        """Detect global pattern from agent positions."""

        positions = [agent.position for agent in self.agents]

        # Analyze spatial distribution
        clustering = self.measure_clustering(positions)

        if clustering > 0.8:
            return "clustered"
        elif clustering < 0.2:
            return "dispersed"
        else:
            return "random"

    def measure_clustering(self, positions: List[Tuple]) -> float:
        """Measure how clustered positions are."""

        # Average distance to nearest neighbor
        avg_nearest_distance = 0

        for i, pos1 in enumerate(positions):
            nearest_dist = float('inf')

            for j, pos2 in enumerate(positions):
                if i == j:
                    continue

                dist = self.distance(pos1, pos2)
                nearest_dist = min(nearest_dist, dist)

            avg_nearest_distance += nearest_dist

        avg_nearest_distance /= len(positions)

        # Normalize to 0-1 (low distance = high clustering)
        # This is simplified - use proper spatial statistics in practice
        max_dist = 100.0  # Assuming 100x100 space
        clustering = 1.0 - (avg_nearest_distance / max_dist)

        return clustering

class EmergentLeader:
    """Leader emerges without explicit designation."""

    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.influence_scores = {agent.id: 0.0 for agent in agents}

    async def evolve(self):
        """Agents interact, leader emerges."""

        for round_num in range(10):
            # Agents interact
            for agent in self.agents:
                # Agent shares opinion
                opinion = await agent.share_opinion()

                # Others respond
                for other in self.agents:
                    if other.id == agent.id:
                        continue

                    # Other considers opinion
                    adoption = await other.consider_opinion(opinion)

                    if adoption:
                        # Increase influence score
                        self.influence_scores[agent.id] += 1

            # Who's most influential?
            leader = max(self.influence_scores.items(), key=lambda x: x[1])

            print(f"Round {round_num}: Leader is {leader[0]} "
                  f"(influence: {leader[1]})")

# Example
agents = [Agent(f"agent_{i}") for i in range(20)]
leader_emergence = EmergentLeader(agents)
await leader_emergence.evolve()
```

## Decentralized Coordination

Coordinating without central control:

### Decentralized Task Allocation

```python
class DecentralizedTaskAllocation:
    """Agents self-allocate to tasks without coordinator."""

    def __init__(self, agents: List[Agent], tasks: List[Dict]):
        self.agents = agents
        self.tasks = tasks
        self.allocations = {task['id']: [] for task in tasks}

    async def allocate(self):
        """Agents allocate themselves to tasks."""

        # Each agent independently decides
        for agent in self.agents:
            # Agent observes task states (distributed knowledge)
            task_states = await self.get_task_states()

            # Agent selects task based on local decision rule
            selected_task = await agent.select_task(
                task_states,
                self.allocations
            )

            if selected_task:
                # Agent commits to task
                self.allocations[selected_task].append(agent.id)

                # Broadcast commitment (so others see)
                await self.broadcast_commitment(agent.id, selected_task)

    async def get_task_states(self) -> List[Dict]:
        """Get current state of all tasks."""

        states = []

        for task in self.tasks:
            states.append({
                'id': task['id'],
                'difficulty': task['difficulty'],
                'assigned_agents': len(self.allocations[task['id']]),
                'required_agents': task.get('required_agents', 1)
            })

        return states

class SwarmAgent:
    """Agent in decentralized swarm."""

    async def select_task(self, task_states: List[Dict],
                         allocations: Dict) -> Optional[str]:
        """Select task using local decision rule."""

        # Response threshold model
        # Agent has threshold for each task type

        for task in task_states:
            # How many agents needed?
            needed = task['required_agents'] - task['assigned_agents']

            if needed <= 0:
                continue  # Task full

            # Probability of selecting task increases with need
            stimulus = needed / task['required_agents']

            # Agent's response threshold
            threshold = self.get_threshold(task['id'])

            # Probabilistic selection
            if stimulus > threshold:
                return task['id']

        return None

    def get_threshold(self, task_id: str) -> float:
        """Agent's response threshold for task."""
        # Based on agent's specialization
        # Lower threshold = more likely to respond
        return 0.5  # Simplified

# Example
agents = [SwarmAgent(f"agent_{i}") for i in range(20)]
tasks = [
    {'id': 'task_1', 'difficulty': 1, 'required_agents': 3},
    {'id': 'task_2', 'difficulty': 2, 'required_agents': 5},
    {'id': 'task_3', 'difficulty': 1, 'required_agents': 2}
]

allocator = DecentralizedTaskAllocation(agents, tasks)
await allocator.allocate()

print("Allocations:", allocator.allocations)
```

## Self-Organization

Systems organizing themselves:

### Self-Organizing Map

```python
class SelfOrganizingSwarm:
    """Swarm that self-organizes into structures."""

    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.connections = {}  # Emergent network structure

    async def organize(self, iterations: int):
        """Swarm self-organizes."""

        for iteration in range(iterations):
            # Agents form/break connections based on similarity
            for agent in self.agents:
                # Find similar nearby agents
                nearby = await self.find_nearby_agents(agent)

                for other in nearby:
                    similarity = await agent.compute_similarity(other)

                    # Form connection if similar enough
                    if similarity > 0.7:
                        self.form_connection(agent.id, other.id)
                    # Break connection if too dissimilar
                    elif similarity < 0.3:
                        self.break_connection(agent.id, other.id)

            # Agents adapt based on connections
            for agent in self.agents:
                connected = self.get_connected_agents(agent.id)

                if connected:
                    # Agent adjusts to be more like connected agents
                    await agent.adapt_to_neighbors(connected)

    def form_connection(self, agent1_id: str, agent2_id: str):
        """Form connection between agents."""

        if agent1_id not in self.connections:
            self.connections[agent1_id] = set()

        self.connections[agent1_id].add(agent2_id)

    def get_connected_agents(self, agent_id: str) -> List[Agent]:
        """Get agents connected to this agent."""

        if agent_id not in self.connections:
            return []

        connected_ids = self.connections[agent_id]

        return [a for a in self.agents if a.id in connected_ids]

# Example
agents = [Agent(f"agent_{i}") for i in range(50)]
swarm = SelfOrganizingSwarm(agents)
await swarm.organize(iterations=100)

print(f"Network formed with {len(swarm.connections)} nodes")
```

## Stigmergy

Indirect coordination through environment:

### Stigmergic Communication

```python
class StigmergicEnvironment:
    """Environment that mediates agent coordination."""

    def __init__(self):
        self.pheromones = {}  # Location -> pheromone intensity
        self.evaporation_rate = 0.1

    def deposit_pheromone(self, location: Tuple[int, int],
                         amount: float):
        """Agent deposits pheromone."""

        if location not in self.pheromones:
            self.pheromones[location] = 0.0

        self.pheromones[location] += amount

    def sense_pheromone(self, location: Tuple[int, int]) -> float:
        """Sense pheromone at location."""

        return self.pheromones.get(location, 0.0)

    def evaporate(self):
        """Pheromones evaporate over time."""

        for location in list(self.pheromones.keys()):
            self.pheromones[location] *= (1 - self.evaporation_rate)

            # Remove negligible pheromones
            if self.pheromones[location] < 0.01:
                del self.pheromones[location]

class StigmergicAgent:
    """Agent that communicates through environment."""

    def __init__(self, position: Tuple[int, int]):
        self.position = position
        self.carrying_food = False

    async def act(self, environment: StigmergicEnvironment):
        """Act based on environmental cues."""

        if self.carrying_food:
            # Follow pheromone trail home
            next_pos = self.follow_pheromone_gradient(
                environment,
                maximize=False  # Go toward lower pheromone (home)
            )

            # Deposit pheromone
            environment.deposit_pheromone(self.position, amount=1.0)

        else:
            # Explore, following pheromones to food
            next_pos = self.follow_pheromone_gradient(
                environment,
                maximize=True  # Go toward higher pheromone (food)
            )

        self.position = next_pos

    def follow_pheromone_gradient(self,
                                  environment: StigmergicEnvironment,
                                  maximize: bool) -> Tuple[int, int]:
        """Move toward higher (or lower) pheromone concentration."""

        # Check adjacent locations
        adjacent = self.get_adjacent_positions()

        # Find location with highest/lowest pheromone
        best_pos = self.position
        best_pheromone = environment.sense_pheromone(self.position)

        for pos in adjacent:
            pheromone = environment.sense_pheromone(pos)

            if maximize and pheromone > best_pheromone:
                best_pos = pos
                best_pheromone = pheromone
            elif not maximize and pheromone < best_pheromone:
                best_pos = pos
                best_pheromone = pheromone

        return best_pos

    def get_adjacent_positions(self) -> List[Tuple[int, int]]:
        """Get adjacent grid positions."""

        x, y = self.position

        return [
            (x-1, y), (x+1, y),
            (x, y-1), (x, y+1)
        ]

# Example: Ant colony foraging
environment = StigmergicEnvironment()

ants = [StigmergicAgent(position=(0, 0)) for _ in range(100)]

for step in range(1000):
    # Ants act
    for ant in ants:
        await ant.act(environment)

    # Pheromones evaporate
    environment.evaporate()
```

## Particle Swarm Optimization

Optimization through swarm:

### PSO Implementation

```python
class Particle:
    """Particle in swarm optimization."""

    def __init__(self, dimensions: int):
        # Random initial position and velocity
        self.position = [random.uniform(-10, 10) for _ in range(dimensions)]
        self.velocity = [random.uniform(-1, 1) for _ in range(dimensions)]

        # Best position found by this particle
        self.best_position = self.position.copy()
        self.best_fitness = float('-inf')

    def update(self, global_best_position: List[float],
               w: float = 0.7,  # Inertia
               c1: float = 1.5,  # Cognitive component
               c2: float = 1.5):  # Social component
        """Update particle velocity and position."""

        for i in range(len(self.position)):
            # Random factors
            r1 = random.random()
            r2 = random.random()

            # Velocity update
            cognitive = c1 * r1 * (self.best_position[i] - self.position[i])
            social = c2 * r2 * (global_best_position[i] - self.position[i])

            self.velocity[i] = (
                w * self.velocity[i] +  # Inertia
                cognitive +              # Personal best
                social                   # Global best
            )

            # Position update
            self.position[i] += self.velocity[i]

class ParticleSwarmOptimizer:
    """Particle Swarm Optimization algorithm."""

    def __init__(self, num_particles: int, dimensions: int,
                 objective_function: Callable):
        self.particles = [Particle(dimensions) for _ in range(num_particles)]
        self.objective_function = objective_function

        self.global_best_position = None
        self.global_best_fitness = float('-inf')

    async def optimize(self, max_iterations: int) -> Dict:
        """Run PSO optimization."""

        for iteration in range(max_iterations):
            # Evaluate all particles
            for particle in self.particles:
                fitness = await self.objective_function(particle.position)

                # Update particle's best
                if fitness > particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()

                # Update global best
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()

            # Update all particles
            for particle in self.particles:
                particle.update(self.global_best_position)

            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Best fitness = "
                      f"{self.global_best_fitness:.4f}")

        return {
            'best_position': self.global_best_position,
            'best_fitness': self.global_best_fitness
        }

# Example: Optimize Rosenbrock function
async def rosenbrock(x: List[float]) -> float:
    """Rosenbrock function (minimize = find [1, 1, ...])."""

    result = 0
    for i in range(len(x) - 1):
        result += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2

    return -result  # Negative because we maximize

pso = ParticleSwarmOptimizer(
    num_particles=30,
    dimensions=5,
    objective_function=rosenbrock
)

result = await pso.optimize(max_iterations=100)
print(f"Optimum found: {result['best_position']}")
```

## Ant Colony Optimization

Path optimization through ant behavior:

### ACO Implementation

```python
class AntColonyOptimizer:
    """Ant Colony Optimization for TSP."""

    def __init__(self, cities: List[Tuple[float, float]],
                 num_ants: int = 20):
        self.cities = cities
        self.num_cities = len(cities)
        self.num_ants = num_ants

        # Distance matrix
        self.distances = self.compute_distances()

        # Pheromone matrix
        self.pheromones = [[1.0] * self.num_cities
                           for _ in range(self.num_cities)]

        # Parameters
        self.alpha = 1.0  # Pheromone importance
        self.beta = 2.0   # Distance importance
        self.evaporation = 0.5
        self.Q = 100      # Pheromone deposit factor

    def compute_distances(self) -> List[List[float]]:
        """Compute distance matrix."""

        distances = []

        for i, city1 in enumerate(self.cities):
            row = []
            for j, city2 in enumerate(self.cities):
                if i == j:
                    row.append(0)
                else:
                    dist = math.sqrt(
                        (city1[0] - city2[0])**2 +
                        (city1[1] - city2[1])**2
                    )
                    row.append(dist)
            distances.append(row)

        return distances

    async def optimize(self, iterations: int) -> Dict:
        """Run ACO."""

        best_path = None
        best_length = float('inf')

        for iteration in range(iterations):
            # Each ant constructs a solution
            paths = []
            lengths = []

            for ant in range(self.num_ants):
                path = self.construct_path()
                length = self.path_length(path)

                paths.append(path)
                lengths.append(length)

                # Update best
                if length < best_length:
                    best_length = length
                    best_path = path

            # Update pheromones
            self.update_pheromones(paths, lengths)

            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Best length = {best_length:.2f}")

        return {
            'best_path': best_path,
            'best_length': best_length
        }

    def construct_path(self) -> List[int]:
        """Ant constructs a path."""

        path = [0]  # Start at city 0
        visited = {0}

        while len(visited) < self.num_cities:
            current = path[-1]
            next_city = self.select_next_city(current, visited)

            path.append(next_city)
            visited.add(next_city)

        return path

    def select_next_city(self, current: int, visited: Set[int]) -> int:
        """Probabilistically select next city."""

        # Compute probabilities
        probabilities = []

        for city in range(self.num_cities):
            if city in visited:
                probabilities.append(0)
            else:
                pheromone = self.pheromones[current][city]
                distance = self.distances[current][city]

                # Probability proportional to pheromone^alpha / distance^beta
                prob = (pheromone ** self.alpha) * ((1.0 / distance) ** self.beta)
                probabilities.append(prob)

        # Normalize
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        # Select city
        return random.choices(range(self.num_cities), weights=probabilities)[0]

    def path_length(self, path: List[int]) -> float:
        """Compute total path length."""

        length = 0

        for i in range(len(path)):
            current = path[i]
            next_city = path[(i + 1) % len(path)]  # Return to start
            length += self.distances[current][next_city]

        return length

    def update_pheromones(self, paths: List[List[int]],
                         lengths: List[float]):
        """Update pheromone trails."""

        # Evaporation
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                self.pheromones[i][j] *= (1 - self.evaporation)

        # Deposit new pheromones
        for path, length in zip(paths, lengths):
            deposit = self.Q / length

            for i in range(len(path)):
                current = path[i]
                next_city = path[(i + 1) % len(path)]

                self.pheromones[current][next_city] += deposit
                self.pheromones[next_city][current] += deposit

# Example: TSP with 10 cities
cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(10)]

aco = AntColonyOptimizer(cities, num_ants=20)
result = await aco.optimize(iterations=100)

print(f"Best path: {result['best_path']}")
print(f"Length: {result['best_length']:.2f}")
```

## Flocking and Boids

Collective movement:

### Boids Algorithm

```python
class Boid:
    """Agent in flocking simulation."""

    def __init__(self, position: np.ndarray, velocity: np.ndarray):
        self.position = position
        self.velocity = velocity
        self.max_speed = 2.0
        self.max_force = 0.1

    def update(self, boids: List['Boid']):
        """Update boid based on three rules."""

        # Find neighbors
        neighbors = self.find_neighbors(boids, radius=10.0)

        if not neighbors:
            return

        # Three flocking rules
        separation = self.separation(neighbors) * 1.5
        alignment = self.alignment(neighbors) * 1.0
        cohesion = self.cohesion(neighbors) * 1.0

        # Apply forces
        self.apply_force(separation)
        self.apply_force(alignment)
        self.apply_force(cohesion)

        # Update position
        self.velocity = self.limit_magnitude(self.velocity, self.max_speed)
        self.position += self.velocity

    def separation(self, neighbors: List['Boid']) -> np.ndarray:
        """Steer to avoid crowding."""

        steer = np.zeros(2)

        for boid in neighbors:
            distance = np.linalg.norm(self.position - boid.position)

            if distance < 2.0:  # Too close
                diff = self.position - boid.position
                diff = diff / distance  # Weight by distance
                steer += diff

        if len(neighbors) > 0:
            steer = steer / len(neighbors)

        return self.normalize(steer)

    def alignment(self, neighbors: List['Boid']) -> np.ndarray:
        """Steer toward average heading."""

        avg_velocity = np.zeros(2)

        for boid in neighbors:
            avg_velocity += boid.velocity

        avg_velocity = avg_velocity / len(neighbors)

        return self.normalize(avg_velocity)

    def cohesion(self, neighbors: List['Boid']) -> np.ndarray:
        """Steer toward average position."""

        center = np.zeros(2)

        for boid in neighbors:
            center += boid.position

        center = center / len(neighbors)

        desired = center - self.position

        return self.normalize(desired)

    def apply_force(self, force: np.ndarray):
        """Apply force to velocity."""

        force = self.limit_magnitude(force, self.max_force)
        self.velocity += force

    def find_neighbors(self, boids: List['Boid'],
                      radius: float) -> List['Boid']:
        """Find boids within radius."""

        neighbors = []

        for boid in boids:
            if boid is self:
                continue

            distance = np.linalg.norm(self.position - boid.position)

            if distance < radius:
                neighbors.append(boid)

        return neighbors

    @staticmethod
    def normalize(vector: np.ndarray) -> np.ndarray:
        """Normalize vector."""

        magnitude = np.linalg.norm(vector)

        if magnitude > 0:
            return vector / magnitude
        else:
            return vector

    @staticmethod
    def limit_magnitude(vector: np.ndarray, max_mag: float) -> np.ndarray:
        """Limit vector magnitude."""

        magnitude = np.linalg.norm(vector)

        if magnitude > max_mag:
            return (vector / magnitude) * max_mag
        else:
            return vector

# Simulation
boids = [
    Boid(
        position=np.array([random.uniform(0, 100), random.uniform(0, 100)]),
        velocity=np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
    )
    for _ in range(50)
]

for step in range(100):
    for boid in boids:
        boid.update(boids)
```

## Consensus in Swarms

Distributed consensus:

### Swarm Consensus Protocol

```python
class SwarmConsensus:
    """Achieve consensus in swarm."""

    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.opinions = {agent.id: random.choice([0, 1])
                        for agent in agents}

    async def reach_consensus(self, max_iterations: int = 100) -> int:
        """Reach consensus through local interactions."""

        for iteration in range(max_iterations):
            # Random pairwise interactions
            agent1, agent2 = random.sample(self.agents, 2)

            # Agents compare opinions
            opinion1 = self.opinions[agent1.id]
            opinion2 = self.opinions[agent2.id]

            if opinion1 != opinion2:
                # Agents try to convince each other
                # Simple majority rule in neighborhood
                neighbors1 = await self.get_neighbors(agent1)
                neighbors2 = await self.get_neighbors(agent2)

                count1 = sum(1 for n in neighbors1
                           if self.opinions[n.id] == opinion1)
                count2 = sum(1 for n in neighbors2
                           if self.opinions[n.id] == opinion2)

                # Agent with more supporting neighbors convinces the other
                if count1 > count2:
                    self.opinions[agent2.id] = opinion1
                elif count2 > count1:
                    self.opinions[agent1.id] = opinion2

            # Check if consensus reached
            if self.has_consensus():
                print(f"Consensus reached at iteration {iteration}")
                break

        return list(self.opinions.values())[0]

    def has_consensus(self) -> bool:
        """Check if all agents agree."""

        opinions = list(self.opinions.values())
        return all(o == opinions[0] for o in opinions)

# Example
agents = [Agent(f"agent_{i}") for i in range(50)]
consensus = SwarmConsensus(agents)

result = await consensus.reach_consensus()
print(f"Swarm consensus: {result}")
```

## When to Use Swarms

Decision guide:

```
Problem Characteristics
├── Large search space → PSO, ACO
├── Optimization problem → Swarm optimization
├── Routing/path finding → ACO
├── Decentralized control needed → Swarm
└── Robustness critical → Swarm

System Requirements
├── No central coordinator → Swarm
├── Scalability important → Swarm
├── Fault tolerance needed → Swarm
└── Simple agents preferred → Swarm

vs. Other Approaches
├── Need complex reasoning → NOT swarm
├── Need guarantees → NOT swarm (usually)
├── Need explainability → NOT swarm
└── Small, structured problem → Centralized better
```

## Summary

Swarm intelligence achieves complex behavior through simple local rules:

**Core Principles:**

- Local perception and action only
- No centralized control or communication
- Emergent global behavior
- Self-organization
- Robustness through redundancy

**Key Patterns:**

- Flocking/boids (collective movement)
- Stigmergy (environment-mediated coordination)
- Particle swarm optimization (solution search)
- Ant colony optimization (path optimization)

**Advantages:**

- Highly scalable (add more agents)
- Robust to agent failures
- No single point of failure
- Adaptive to changing environments
- Parallel exploration

**Limitations:**

- Hard to predict/control precisely
- May converge slowly
- Less suitable for complex reasoning
- Difficult to debug

**Best Use Cases:**

- Optimization in large search spaces
- Decentralized control scenarios
- Problems inspired by nature
- When robustness > optimality

Swarm systems trade individual agent complexity for collective intelligence, achieving sophisticated behavior through the interaction of many simple components.

## Next Steps

- [Collaboration Patterns](collaboration.md) - Cooperative agents
- [Coordination](coordination.md) - Task allocation and synchronization
- [Debate and Consensus](debate.md) - Deliberative decision-making
- [Communication](communication.md) - Agent interaction patterns
- [Multi-Agent Fundamentals](fundamentals.md) - Core concepts
