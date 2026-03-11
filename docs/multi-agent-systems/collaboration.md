# Collaboration Patterns

## Table of Contents

- [Introduction](#introduction)
- [Cooperative Problem-Solving](#cooperative-problem-solving)
- [Joint Planning](#joint-planning)
- [Shared Goals](#shared-goals)
- [Distributed Problem Solving](#distributed-problem-solving)
- [Shared Memory](#shared-memory)
- [Team Formation](#team-formation)
- [Collaboration Protocols](#collaboration-protocols)
- [Negotiation](#negotiation)
- [Resource Sharing](#resource-sharing)
- [Collaborative Learning](#collaborative-learning)
- [Conflict in Collaboration](#conflict-in-collaboration)
- [Success Metrics](#success-metrics)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Collaboration is when agents work **together** toward shared goals, pooling their knowledge, capabilities, and efforts. Unlike hierarchical delegation where one agent commands others, collaboration involves agents as peers working cooperatively.

> "Alone we can do so little; together we can do so much."  
> -- Helen Keller

**Collaboration** differs from mere **coordination**:
- **Coordination**: Agents avoid interference (negative)
- **Collaboration**: Agents actively help each other (positive)

This guide explores:
- How agents collaborate effectively
- Joint problem-solving strategies
- Shared memory and knowledge
- Building collaborative teams
- Managing collaborative dynamics

## Cooperative Problem-Solving

Agents working together to solve problems:

### Problem Decomposition and Sharing

```python
class CollaborativeProblemSolving:
    """Agents collaboratively solve problems."""
    
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.shared_workspace = SharedWorkspace()
    
    async def solve_collaboratively(self, problem: Dict) -> Any:
        """Solve problem through agent collaboration."""
        
        # Phase 1: Collective understanding
        understanding = await self.build_shared_understanding(problem)
        
        # Phase 2: Collaborative decomposition
        subproblems = await self.decompose_problem(problem, understanding)
        
        # Phase 3: Distributed solving
        partial_solutions = await self.solve_subproblems(subproblems)
        
        # Phase 4: Solution integration
        final_solution = await self.integrate_solutions(
            partial_solutions,
            problem
        )
        
        return final_solution
    
    async def build_shared_understanding(self, problem: Dict) -> Dict:
        """Agents build common understanding of problem."""
        
        # Each agent analyzes problem
        analyses = await asyncio.gather(*[
            agent.analyze_problem(problem)
            for agent in self.agents
        ])
        
        # Share and discuss analyses
        shared_understanding = {}
        
        for i in range(3):  # 3 rounds of discussion
            # Agents review each other's analyses
            for agent, analysis in zip(self.agents, analyses):
                # Agent sees others' analyses
                others_analyses = [a for a in analyses if a != analysis]
                
                # Agent updates their understanding
                updated = await agent.refine_understanding(
                    analysis,
                    others_analyses
                )
                
                analyses[analyses.index(analysis)] = updated
            
            # Check for convergence
            if self.has_converged(analyses):
                break
        
        # Synthesize shared understanding
        shared_understanding = await self.synthesize_analyses(analyses)
        
        return shared_understanding
    
    async def decompose_problem(self, problem: Dict, 
                               understanding: Dict) -> List[Dict]:
        """Collaboratively decompose problem."""
        
        # Agents propose decompositions
        proposals = await asyncio.gather(*[
            agent.propose_decomposition(problem, understanding)
            for agent in self.agents
        ])
        
        # Agents vote on best decomposition
        votes = []
        for proposal in proposals:
            vote_count = await self.collect_votes(proposal, self.agents)
            votes.append((proposal, vote_count))
        
        # Select highest-voted decomposition
        best_decomposition = max(votes, key=lambda x: x[1])[0]
        
        return best_decomposition['subproblems']
    
    async def solve_subproblems(self, subproblems: List[Dict]) -> List[Any]:
        """Agents solve subproblems, helping each other."""
        
        # Assign subproblems (agents can work on multiple)
        assignments = await self.assign_subproblems(subproblems)
        
        # Solve with mutual assistance
        solutions = {}
        
        for subproblem in subproblems:
            # Primary agent(s) for this subproblem
            primary_agents = assignments[subproblem['id']]
            
            # Solve with help from others
            solution = await self.solve_with_assistance(
                subproblem,
                primary_agents,
                supporting_agents=[a for a in self.agents 
                                 if a not in primary_agents]
            )
            
            solutions[subproblem['id']] = solution
            
            # Share solution with all agents
            await self.shared_workspace.post({
                'type': 'solution',
                'subproblem': subproblem,
                'solution': solution
            })
        
        return list(solutions.values())
    
    async def solve_with_assistance(self, subproblem: Dict,
                                    primary_agents: List[Agent],
                                    supporting_agents: List[Agent]) -> Any:
        """Solve subproblem with assistance from others."""
        
        # Primary agents attempt solution
        attempts = await asyncio.gather(*[
            agent.solve(subproblem)
            for agent in primary_agents
        ])
        
        # If solution is incomplete, request help
        for attempt in attempts:
            if attempt.get('confidence', 1.0) < 0.7:
                # Request assistance
                help_requests = await asyncio.gather(*[
                    agent.request_help(subproblem, attempt)
                    for agent in supporting_agents
                ])
                
                # Incorporate suggestions
                for suggestion in help_requests:
                    attempt = await self.incorporate_suggestion(
                        attempt,
                        suggestion
                    )
        
        # Merge attempts from primary agents
        final_solution = await self.merge_solutions(attempts)
        
        return final_solution
    
    async def integrate_solutions(self, partial_solutions: List[Any],
                                 problem: Dict) -> Any:
        """Integrate partial solutions into final solution."""
        
        # Agents collaboratively integrate
        integration_proposals = await asyncio.gather(*[
            agent.propose_integration(partial_solutions, problem)
            for agent in self.agents
        ])
        
        # Discuss and refine integration
        for round_num in range(3):
            # Agents critique each proposal
            critiques = []
            for proposal in integration_proposals:
                proposal_critiques = await asyncio.gather(*[
                    agent.critique_integration(proposal)
                    for agent in self.agents
                ])
                critiques.append(proposal_critiques)
            
            # Proposers refine based on critiques
            integration_proposals = await asyncio.gather(*[
                agent.refine_proposal(proposal, critique)
                for agent, proposal, critique in zip(
                    self.agents, integration_proposals, critiques
                )
            ])
        
        # Select best integration
        best = await self.select_best_integration(integration_proposals)
        
        return best

# Example usage
agents = [
    SpecialistAgent("algorithms"),
    SpecialistAgent("data_structures"),
    SpecialistAgent("optimization")
]

collaborative_system = CollaborativeProblemSolving(agents)

problem = {
    'description': 'Design efficient search system',
    'requirements': ['fast', 'scalable', 'accurate']
}

solution = await collaborative_system.solve_collaboratively(problem)
```

### Brainstorming and Ideation

```python
class CollaborativeBrainstorming:
    """Agents brainstorm together."""
    
    async def brainstorm(self, topic: str, 
                        agents: List[Agent],
                        rounds: int = 3) -> List[Dict]:
        """Collaborative brainstorming session."""
        
        ideas = []
        
        for round_num in range(rounds):
            print(f"\n=== Brainstorming Round {round_num + 1} ===\n")
            
            # Each agent generates ideas
            round_ideas = await asyncio.gather(*[
                agent.generate_ideas(topic, existing_ideas=ideas)
                for agent in agents
            ])
            
            # Flatten and add to pool
            for agent_ideas in round_ideas:
                ideas.extend(agent_ideas)
            
            # Agents build on each other's ideas
            if round_num < rounds - 1:
                extended_ideas = await asyncio.gather(*[
                    agent.extend_ideas(ideas)
                    for agent in agents
                ])
                
                for extensions in extended_ideas:
                    ideas.extend(extensions)
            
            # Collaborative filtering
            ideas = await self.filter_ideas(ideas, agents)
        
        # Final selection of best ideas
        best_ideas = await self.select_best_ideas(ideas, agents)
        
        return best_ideas
    
    async def filter_ideas(self, ideas: List[Dict], 
                          agents: List[Agent]) -> List[Dict]:
        """Collaboratively filter ideas."""
        
        # Each agent evaluates all ideas
        evaluations = {}
        
        for idea in ideas:
            idea_scores = await asyncio.gather(*[
                agent.evaluate_idea(idea)
                for agent in agents
            ])
            
            evaluations[idea['id']] = {
                'avg_score': sum(idea_scores) / len(idea_scores),
                'scores': idea_scores
            }
        
        # Keep ideas with good average scores
        threshold = 0.6
        filtered = [
            idea for idea in ideas
            if evaluations[idea['id']]['avg_score'] >= threshold
        ]
        
        return filtered
    
    async def select_best_ideas(self, ideas: List[Dict],
                               agents: List[Agent],
                               top_n: int = 5) -> List[Dict]:
        """Collaboratively select best ideas."""
        
        # Collaborative ranking
        rankings = await asyncio.gather(*[
            agent.rank_ideas(ideas)
            for agent in agents
        ])
        
        # Aggregate rankings (Borda count)
        scores = {idea['id']: 0 for idea in ideas}
        
        for ranking in rankings:
            for i, idea_id in enumerate(ranking):
                # Higher rank = more points
                scores[idea_id] += len(ranking) - i
        
        # Select top N
        sorted_ideas = sorted(
            ideas,
            key=lambda idea: scores[idea['id']],
            reverse=True
        )
        
        return sorted_ideas[:top_n]

# Example
brainstorm = CollaborativeBrainstorming()

agents = [
    CreativeAgent("creative_1"),
    AnalyticalAgent("analytical_1"),
    PracticalAgent("practical_1")
]

best_ideas = await brainstorm.brainstorm(
    topic="New features for mobile app",
    agents=agents,
    rounds=3
)

print("Best Ideas:")
for idea in best_ideas:
    print(f"- {idea['description']}")
```

## Joint Planning

Agents planning together:

### Collaborative Planning

```python
class JointPlanning:
    """Agents create plans together."""
    
    async def create_joint_plan(self, goal: Dict, 
                               agents: List[Agent]) -> Dict:
        """Create plan collaboratively."""
        
        # Phase 1: Goal clarification
        clarified_goal = await self.clarify_goal(goal, agents)
        
        # Phase 2: Constraint identification
        constraints = await self.identify_constraints(clarified_goal, agents)
        
        # Phase 3: Option generation
        options = await self.generate_options(clarified_goal, constraints, agents)
        
        # Phase 4: Collaborative evaluation
        evaluated = await self.evaluate_options(options, agents)
        
        # Phase 5: Plan selection
        selected_plan = await self.select_plan(evaluated, agents)
        
        # Phase 6: Plan refinement
        final_plan = await self.refine_plan(selected_plan, agents)
        
        return final_plan
    
    async def clarify_goal(self, goal: Dict, agents: List[Agent]) -> Dict:
        """Agents collaboratively clarify goal."""
        
        clarifications = await asyncio.gather(*[
            agent.clarify(goal)
            for agent in agents
        ])
        
        # Agents discuss and reach shared understanding
        shared_understanding = await self.reach_consensus(
            clarifications,
            agents
        )
        
        return shared_understanding
    
    async def generate_options(self, goal: Dict, constraints: List[Dict],
                              agents: List[Agent]) -> List[Dict]:
        """Generate plan options."""
        
        # Each agent proposes plan options
        all_options = []
        
        for agent in agents:
            options = await agent.propose_plans(goal, constraints)
            all_options.extend(options)
        
        # Agents refine each other's proposals
        for _ in range(2):  # 2 refinement rounds
            refined_options = []
            
            for option in all_options:
                # Other agents suggest improvements
                improvements = await asyncio.gather(*[
                    agent.suggest_improvements(option)
                    for agent in agents
                ])
                
                # Incorporate improvements
                refined = await self.incorporate_improvements(
                    option,
                    improvements
                )
                refined_options.append(refined)
            
            all_options = refined_options
        
        return all_options
    
    async def evaluate_options(self, options: List[Dict],
                              agents: List[Agent]) -> List[Dict]:
        """Collaboratively evaluate plan options."""
        
        evaluations = []
        
        for option in options:
            # Each agent evaluates
            agent_evals = await asyncio.gather(*[
                agent.evaluate_plan(option)
                for agent in agents
            ])
            
            # Aggregate evaluation
            option['evaluation'] = {
                'feasibility': sum(e['feasibility'] for e in agent_evals) / len(agents),
                'efficiency': sum(e['efficiency'] for e in agent_evals) / len(agents),
                'robustness': sum(e['robustness'] for e in agent_evals) / len(agents),
                'agent_evaluations': agent_evals
            }
            
            evaluations.append(option)
        
        return evaluations
    
    async def select_plan(self, evaluated_options: List[Dict],
                         agents: List[Agent]) -> Dict:
        """Select plan collaboratively."""
        
        # Agents vote on options
        votes = {}
        for option in evaluated_options:
            votes[option['id']] = 0
        
        for agent in agents:
            vote = await agent.vote_for_plan(evaluated_options)
            votes[vote] += 1
        
        # Select winner
        winner_id = max(votes.items(), key=lambda x: x[1])[0]
        winner = next(opt for opt in evaluated_options if opt['id'] == winner_id)
        
        return winner

# Example
planner = JointPlanning()

agents = [
    PlannerAgent("efficiency"),
    PlannerAgent("reliability"),
    PlannerAgent("innovation")
]

goal = {
    'objective': 'Launch new product',
    'deadline': '6 months',
    'budget': 1000000
}

plan = await planner.create_joint_plan(goal, agents)
```

## Shared Goals

Agents working toward common objectives:

### Goal Alignment

```python
class SharedGoalSystem:
    """System where agents share common goals."""
    
    def __init__(self):
        self.shared_goals = []
        self.agents = []
    
    def add_shared_goal(self, goal: Dict):
        """Add goal that all agents work toward."""
        
        goal_id = str(uuid.uuid4())
        goal['id'] = goal_id
        goal['progress'] = 0.0
        goal['contributions'] = {}
        
        self.shared_goals.append(goal)
        
        # Notify all agents
        for agent in self.agents:
            agent.new_shared_goal(goal)
    
    async def work_toward_goals(self):
        """Agents collaboratively work toward shared goals."""
        
        while any(g['progress'] < 1.0 for g in self.shared_goals):
            # Each agent decides how to contribute
            contributions = await asyncio.gather(*[
                agent.decide_contribution(self.shared_goals)
                for agent in self.agents
            ])
            
            # Execute contributions
            for agent, contribution in zip(self.agents, contributions):
                if contribution:
                    result = await agent.execute_contribution(contribution)
                    
                    # Update goal progress
                    goal = next(g for g in self.shared_goals 
                              if g['id'] == contribution['goal_id'])
                    
                    goal['progress'] += result['progress_delta']
                    goal['contributions'][agent.id] = result
                    
                    # Share progress with all agents
                    await self.broadcast_progress(goal, agent.id, result)
            
            await asyncio.sleep(1)
    
    async def broadcast_progress(self, goal: Dict, 
                                agent_id: str, result: Dict):
        """Share progress with all agents."""
        
        for agent in self.agents:
            await agent.see_progress(goal['id'], agent_id, result)

class CollaborativeAgent:
    """Agent that works toward shared goals."""
    
    def __init__(self, agent_id: str):
        self.id = agent_id
        self.known_goals = []
        self.capabilities = []
    
    def new_shared_goal(self, goal: Dict):
        """Learn about new shared goal."""
        self.known_goals.append(goal)
    
    async def decide_contribution(self, goals: List[Dict]) -> Optional[Dict]:
        """Decide how to contribute to shared goals."""
        
        # Filter to goals I can help with
        actionable_goals = [
            g for g in goals
            if self.can_contribute(g) and g['progress'] < 1.0
        ]
        
        if not actionable_goals:
            return None
        
        # Select most urgent/important goal
        selected = max(actionable_goals, 
                      key=lambda g: (g.get('priority', 0), 1.0 - g['progress']))
        
        # Decide on specific contribution
        contribution = await self.plan_contribution(selected)
        
        return {
            'goal_id': selected['id'],
            'action': contribution
        }
    
    def can_contribute(self, goal: Dict) -> bool:
        """Check if I can contribute to this goal."""
        
        required_capabilities = goal.get('required_capabilities', [])
        return any(cap in self.capabilities for cap in required_capabilities)
    
    async def see_progress(self, goal_id: str, 
                          contributor_id: str, result: Dict):
        """See another agent's contribution."""
        
        # Learn from others' contributions
        # Adjust own strategy if needed
        pass

# Example
system = SharedGoalSystem()

# Add agents
agents = [
    CollaborativeAgent("agent1"),
    CollaborativeAgent("agent2"),
    CollaborativeAgent("agent3")
]
system.agents = agents

# Add shared goals
system.add_shared_goal({
    'description': 'Build knowledge base',
    'target': 1000,  # 1000 entries
    'required_capabilities': ['research', 'writing']
})

system.add_shared_goal({
    'description': 'Process user requests',
    'target': 500,  # 500 requests
    'required_capabilities': ['customer_service']
})

# Agents work together
await system.work_toward_goals()
```

## Distributed Problem Solving

Solving problems distributedly:

### Distributed Search

```python
class DistributedSearch:
    """Agents search solution space together."""
    
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.shared_findings = []
        self.best_solution = None
    
    async def search(self, problem: Dict) -> Any:
        """Distributed search for solution."""
        
        # Divide search space among agents
        search_regions = self.partition_search_space(
            problem,
            len(self.agents)
        )
        
        # Agents search their regions
        search_tasks = [
            self.agent_search(agent, region, problem)
            for agent, region in zip(self.agents, search_regions)
        ]
        
        # Monitor and coordinate search
        await self.coordinate_search(search_tasks)
        
        return self.best_solution
    
    async def agent_search(self, agent: Agent, region: Dict, 
                          problem: Dict):
        """Agent searches assigned region."""
        
        for iteration in range(problem.get('max_iterations', 100)):
            # Agent explores region
            candidate = await agent.explore(region)
            
            # Evaluate candidate
            quality = await agent.evaluate(candidate, problem)
            
            # Share if promising
            if quality > 0.7:
                await self.share_finding(agent.id, candidate, quality)
                
                # Update best solution
                if not self.best_solution or quality > self.best_solution['quality']:
                    self.best_solution = {
                        'solution': candidate,
                        'quality': quality,
                        'found_by': agent.id
                    }
                    
                    # Notify all agents of new best
                    await self.broadcast_best(candidate, quality)
            
            # Check if should stop
            if quality > 0.95:  # Good enough
                break
            
            # See others' findings
            await agent.learn_from_others(self.shared_findings)
    
    async def share_finding(self, agent_id: str, 
                           finding: Any, quality: float):
        """Share finding with all agents."""
        
        self.shared_findings.append({
            'agent': agent_id,
            'finding': finding,
            'quality': quality,
            'timestamp': time.time()
        })
    
    async def broadcast_best(self, solution: Any, quality: float):
        """Broadcast new best solution."""
        
        for agent in self.agents:
            await agent.see_best_solution(solution, quality)

# Example: Collaborative optimization
class OptimizationAgent:
    """Agent that optimizes solutions."""
    
    async def explore(self, region: Dict) -> Any:
        """Explore region of solution space."""
        # Generate candidate solution in region
        pass
    
    async def evaluate(self, candidate: Any, problem: Dict) -> float:
        """Evaluate solution quality."""
        # Return quality score 0-1
        pass
    
    async def learn_from_others(self, findings: List[Dict]):
        """Learn from others' findings."""
        # Adjust search strategy based on others' success
        pass

searcher = DistributedSearch([
    OptimizationAgent("opt1"),
    OptimizationAgent("opt2"),
    OptimizationAgent("opt3")
])

problem = {
    'objective': 'minimize_cost',
    'constraints': [...],
    'max_iterations': 100
}

best_solution = await searcher.search(problem)
```

## Shared Memory

Agents collaborating through shared knowledge:

### Shared Knowledge Base

```python
class SharedKnowledgeBase:
    """Shared knowledge repository for agents."""
    
    def __init__(self):
        self.knowledge = {}
        self.contributors = {}
        self.access_log = []
    
    async def contribute(self, agent_id: str, knowledge_item: Dict):
        """Agent contributes knowledge."""
        
        item_id = str(uuid.uuid4())
        
        self.knowledge[item_id] = {
            'content': knowledge_item,
            'contributed_by': agent_id,
            'contributed_at': time.time(),
            'confirmations': 0,
            'contradictions': 0
        }
        
        # Track contributor
        if agent_id not in self.contributors:
            self.contributors[agent_id] = []
        self.contributors[agent_id].append(item_id)
        
        return item_id
    
    async def query(self, agent_id: str, query: Dict) -> List[Dict]:
        """Query knowledge base."""
        
        # Log access
        self.access_log.append({
            'agent': agent_id,
            'query': query,
            'timestamp': time.time()
        })
        
        # Find matching knowledge
        matches = []
        for item_id, item in self.knowledge.items():
            if self.matches_query(item['content'], query):
                matches.append({
                    'id': item_id,
                    'content': item['content'],
                    'confidence': self.compute_confidence(item)
                })
        
        return matches
    
    async def confirm(self, agent_id: str, item_id: str):
        """Agent confirms knowledge item."""
        
        if item_id in self.knowledge:
            self.knowledge[item_id]['confirmations'] += 1
    
    async def contradict(self, agent_id: str, item_id: str, 
                        alternative: Dict):
        """Agent contradicts knowledge item."""
        
        if item_id in self.knowledge:
            self.knowledge[item_id]['contradictions'] += 1
            
            # Add alternative
            alt_id = await self.contribute(agent_id, alternative)
            self.knowledge[item_id]['alternatives'] = \
                self.knowledge[item_id].get('alternatives', [])
            self.knowledge[item_id]['alternatives'].append(alt_id)
    
    def compute_confidence(self, item: Dict) -> float:
        """Compute confidence in knowledge item."""
        
        confirmations = item['confirmations']
        contradictions = item['contradictions']
        
        if confirmations + contradictions == 0:
            return 0.5  # Neutral
        
        confidence = confirmations / (confirmations + contradictions)
        return confidence
    
    def matches_query(self, content: Dict, query: Dict) -> bool:
        """Check if content matches query."""
        # Simple matching - in practice, use embeddings
        return any(
            str(v).lower() in str(content).lower()
            for v in query.values()
        )

# Example usage
kb = SharedKnowledgeBase()

# Agent 1 contributes
await kb.contribute("agent1", {
    'fact': 'Python supports async/await',
    'category': 'programming'
})

# Agent 2 queries
results = await kb.query("agent2", {
    'category': 'programming'
})

# Agent 3 confirms
await kb.confirm("agent3", results[0]['id'])
```

## Team Formation

Forming effective teams:

### Dynamic Team Formation

```python
class TeamFormation:
    """Form teams dynamically based on task needs."""
    
    def __init__(self, agent_pool: List[Agent]):
        self.agent_pool = agent_pool
    
    async def form_team(self, task: Dict) -> List[Agent]:
        """Form optimal team for task."""
        
        # Analyze task requirements
        requirements = await self.analyze_requirements(task)
        
        # Find agents matching requirements
        candidates = self.find_candidates(requirements)
        
        # Evaluate team compositions
        teams = self.generate_team_combinations(candidates, requirements)
        scored_teams = await self.score_teams(teams, task)
        
        # Select best team
        best_team = max(scored_teams, key=lambda t: t['score'])
        
        return best_team['members']
    
    def find_candidates(self, requirements: Dict) -> List[Agent]:
        """Find agents matching requirements."""
        
        candidates = []
        
        for agent in self.agent_pool:
            if agent.matches_requirements(requirements):
                candidates.append(agent)
        
        return candidates
    
    async def score_teams(self, teams: List[List[Agent]], 
                         task: Dict) -> List[Dict]:
        """Score potential teams."""
        
        scored = []
        
        for team in teams:
            score = await self.evaluate_team(team, task)
            scored.append({
                'members': team,
                'score': score
            })
        
        return scored
    
    async def evaluate_team(self, team: List[Agent], 
                           task: Dict) -> float:
        """Evaluate team effectiveness."""
        
        # Capability coverage
        required_capabilities = task.get('required_capabilities', [])
        team_capabilities = set()
        for agent in team:
            team_capabilities.update(agent.capabilities)
        
        coverage = len(required_capabilities & team_capabilities) / \
                   max(len(required_capabilities), 1)
        
        # Skill complementarity
        complementarity = self.measure_complementarity(team)
        
        # Team chemistry (past collaboration success)
        chemistry = await self.measure_chemistry(team)
        
        # Combined score
        score = (
            coverage * 0.5 +
            complementarity * 0.3 +
            chemistry * 0.2
        )
        
        return score
    
    def measure_complementarity(self, team: List[Agent]) -> float:
        """Measure how well skills complement each other."""
        
        # Higher score for diverse, non-overlapping skills
        all_skills = [skill for agent in team for skill in agent.skills]
        unique_skills = set(all_skills)
        
        if not all_skills:
            return 0.0
        
        return len(unique_skills) / len(all_skills)

# Example
team_former = TeamFormation(agent_pool=[
    Agent("researcher", skills=["research", "analysis"]),
    Agent("developer", skills=["coding", "testing"]),
    Agent("designer", skills=["design", "ux"]),
    Agent("writer", skills=["writing", "editing"])
])

task = {
    'description': 'Create new feature',
    'required_capabilities': ['research', 'design', 'coding', 'writing']
}

team = await team_former.form_team(task)
print(f"Formed team: {[a.id for a in team]}")
```

## Summary

Collaboration enables agents to achieve more together than they could alone:

**Key Patterns:**
- Cooperative problem-solving through decomposition and sharing
- Joint planning for coordinated action
- Shared goals for aligned incentives
- Distributed problem solving for scalability

**Critical Elements:**
- Shared understanding
- Mutual assistance
- Knowledge sharing
- Trust and coordination

**Collaborative Processes:**
- Brainstorming and ideation
- Joint planning and decision-making
- Distributed search and optimization
- Team formation and dynamics

**Success Factors:**
- Clear communication
- Aligned goals
- Complementary capabilities
- Effective conflict resolution
- Shared rewards

Collaboration transforms independent agents into a cohesive team capable of solving complex problems that no single agent could handle.

## Next Steps

- [Debate and Consensus](debate.md) - Deliberative collaboration
- [Coordination and Task Allocation](coordination.md) - Organizing work
- [Communication Patterns](communication.md) - Inter-agent messaging
- [Delegation and Hierarchies](delegation.md) - Structured collaboration
- [Multi-Agent Fundamentals](fundamentals.md) - Core concepts
