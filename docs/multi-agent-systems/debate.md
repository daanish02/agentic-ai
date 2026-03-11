# Debate and Deliberation

## Table of Contents

- [Introduction](#introduction)
- [Multi-Agent Debate](#multi-agent-debate)
- [Voting Mechanisms](#voting-mechanisms)
- [Consensus Building](#consensus-building)
- [Diverse Perspectives](#diverse-perspectives)
- [Decision Quality](#decision-quality)
- [Debate Protocols](#debate-protocols)
- [Argument Evaluation](#argument-evaluation)
- [Devil's Advocate](#devils-advocate)
- [Resolving Disagreements](#resolving-disagreements)
- [Iterative Refinement](#iterative-refinement)
- [Collective Intelligence](#collective-intelligence)
- [When to Use Debate](#when-to-use-debate)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Multi-agent debate leverages **diverse perspectives** to improve decision quality. Instead of relying on a single agent's judgment, debate systems elicit arguments from multiple agents, evaluate competing viewpoints, and synthesize better solutions.

> "In the multitude of counselors there is safety."  
> -- Proverbs 11:14

**Key Ideas:**
- Multiple agents propose and defend solutions
- Structured debate refines proposals
- Voting or consensus determines final decision
- Disagreement is productive, not problematic

This guide explores:
- How to structure multi-agent debates
- Voting and consensus mechanisms
- Leveraging diverse perspectives
- Improving decision quality through deliberation

## Multi-Agent Debate

The core debate pattern:

### Basic Debate Structure

```python
class MultiAgentDebate:
    """Multi-agent debate system."""
    
    def __init__(self, agents: List[Agent], moderator: Optional[Agent] = None):
        self.agents = agents
        self.moderator = moderator or Agent("moderator")
        self.debate_history = []
    
    async def debate(self, question: str, rounds: int = 3) -> Dict:
        """Conduct multi-round debate."""
        
        print(f"\nQuestion: {question}\n")
        
        # Round 1: Initial positions
        positions = await self.initial_positions(question)
        
        # Rounds 2-N: Debate and refinement
        for round_num in range(2, rounds + 1):
            positions = await self.debate_round(question, positions, round_num)
        
        # Final decision
        decision = await self.reach_decision(question, positions)
        
        return decision
    
    async def initial_positions(self, question: str) -> List[Dict]:
        """Agents state initial positions."""
        
        positions = []
        
        for agent in self.agents:
            position = await agent.state_position(question)
            
            positions.append({
                'agent': agent.id,
                'position': position['answer'],
                'reasoning': position['reasoning'],
                'confidence': position.get('confidence', 0.5)
            })
            
            self.debate_history.append({
                'round': 1,
                'agent': agent.id,
                'type': 'position',
                'content': position
            })
        
        return positions
    
    async def debate_round(self, question: str, 
                          positions: List[Dict], 
                          round_num: int) -> List[Dict]:
        """One round of debate."""
        
        print(f"\n=== Round {round_num} ===\n")
        
        updated_positions = []
        
        for agent, position in zip(self.agents, positions):
            # Agent sees others' positions
            other_positions = [p for p in positions if p['agent'] != agent.id]
            
            # Agent responds/updates position
            updated = await agent.respond_to_debate(
                question=question,
                own_position=position,
                other_positions=other_positions
            )
            
            updated_positions.append({
                'agent': agent.id,
                'position': updated['answer'],
                'reasoning': updated['reasoning'],
                'confidence': updated.get('confidence', 0.5),
                'changed': updated['answer'] != position['position']
            })
            
            self.debate_history.append({
                'round': round_num,
                'agent': agent.id,
                'type': 'response',
                'content': updated
            })
        
        return updated_positions
    
    async def reach_decision(self, question: str, 
                            positions: List[Dict]) -> Dict:
        """Reach final decision from positions."""
        
        # Moderator synthesizes debate
        synthesis = await self.moderator.synthesize_debate(
            question=question,
            positions=positions,
            history=self.debate_history
        )
        
        return {
            'question': question,
            'decision': synthesis['answer'],
            'reasoning': synthesis['reasoning'],
            'positions': positions,
            'confidence': synthesis.get('confidence', 0.5)
        }

# Example usage
agents = [
    Agent("optimist"),
    Agent("pessimist"),
    Agent("realist")
]

debate = MultiAgentDebate(agents)

result = await debate.debate(
    question="Should we invest in the new technology?",
    rounds=3
)

print(f"\nFinal Decision: {result['decision']}")
print(f"Reasoning: {result['reasoning']}")
```

### Structured Debate Format

```python
class StructuredDebate:
    """Debate with formal structure."""
    
    async def conduct_debate(self, proposition: str,
                            proponents: List[Agent],
                            opponents: List[Agent],
                            judges: List[Agent]) -> Dict:
        """Formal debate structure."""
        
        # Opening statements
        pro_opening = await self.opening_statements(proponents, proposition, "for")
        con_opening = await self.opening_statements(opponents, proposition, "against")
        
        # Cross-examination
        pro_questions = await self.cross_examine(proponents, opponents)
        con_questions = await self.cross_examine(opponents, proponents)
        
        # Rebuttals
        pro_rebuttals = await self.rebuttals(proponents, con_opening, con_questions)
        con_rebuttals = await self.rebuttals(opponents, pro_opening, pro_questions)
        
        # Closing statements
        pro_closing = await self.closing_statements(proponents, all_arguments=True)
        con_closing = await self.closing_statements(opponents, all_arguments=True)
        
        # Judges deliberate
        verdict = await self.judges_deliberate(
            judges,
            pro_arguments={
                'opening': pro_opening,
                'rebuttals': pro_rebuttals,
                'closing': pro_closing
            },
            con_arguments={
                'opening': con_opening,
                'rebuttals': con_rebuttals,
                'closing': con_closing
            }
        )
        
        return verdict
    
    async def opening_statements(self, agents: List[Agent], 
                                proposition: str, stance: str) -> List[Dict]:
        """Opening statements from team."""
        
        statements = []
        
        for agent in agents:
            statement = await agent.opening_statement(proposition, stance)
            statements.append({
                'agent': agent.id,
                'statement': statement
            })
        
        return statements
    
    async def cross_examine(self, questioners: List[Agent],
                           responders: List[Agent]) -> List[Dict]:
        """Cross-examination phase."""
        
        qa_pairs = []
        
        for questioner in questioners:
            # Generate question
            question = await questioner.generate_question(responders)
            
            # Responders answer
            for responder in responders:
                answer = await responder.answer_question(question)
                
                qa_pairs.append({
                    'questioner': questioner.id,
                    'responder': responder.id,
                    'question': question,
                    'answer': answer
                })
        
        return qa_pairs
    
    async def judges_deliberate(self, judges: List[Agent],
                               pro_arguments: Dict,
                               con_arguments: Dict) -> Dict:
        """Judges reach verdict."""
        
        # Each judge evaluates
        evaluations = []
        
        for judge in judges:
            eval_result = await judge.evaluate_debate(
                pro_arguments,
                con_arguments
            )
            
            evaluations.append({
                'judge': judge.id,
                'winner': eval_result['winner'],  # 'pro' or 'con'
                'reasoning': eval_result['reasoning'],
                'confidence': eval_result.get('confidence', 0.5)
            })
        
        # Majority vote
        pro_votes = sum(1 for e in evaluations if e['winner'] == 'pro')
        con_votes = sum(1 for e in evaluations if e['winner'] == 'con')
        
        return {
            'verdict': 'pro' if pro_votes > con_votes else 'con',
            'vote_count': {'pro': pro_votes, 'con': con_votes},
            'judge_evaluations': evaluations
        }

# Example
debate = StructuredDebate()

result = await debate.conduct_debate(
    proposition="AI will benefit humanity",
    proponents=[Agent("pro1"), Agent("pro2")],
    opponents=[Agent("con1"), Agent("con2")],
    judges=[Agent("judge1"), Agent("judge2"), Agent("judge3")]
)
```

## Voting Mechanisms

Different ways to aggregate preferences:

### Voting Systems

```python
class VotingSystem:
    """Various voting mechanisms."""
    
    async def plurality_vote(self, agents: List[Agent], 
                            options: List[str]) -> str:
        """Simple plurality voting (most votes wins)."""
        
        votes = {}
        for option in options:
            votes[option] = 0
        
        # Each agent votes for one option
        for agent in agents:
            vote = await agent.vote(options)
            votes[vote] += 1
        
        # Winner is option with most votes
        winner = max(votes.items(), key=lambda x: x[1])[0]
        
        return winner
    
    async def ranked_choice_vote(self, agents: List[Agent],
                                options: List[str]) -> str:
        """Ranked choice voting (instant runoff)."""
        
        # Agents rank all options
        rankings = []
        for agent in agents:
            ranking = await agent.rank_options(options)
            rankings.append(ranking)
        
        remaining_options = set(options)
        
        while len(remaining_options) > 1:
            # Count first-choice votes
            first_choice_counts = {opt: 0 for opt in remaining_options}
            
            for ranking in rankings:
                # Find first choice still in race
                for option in ranking:
                    if option in remaining_options:
                        first_choice_counts[option] += 1
                        break
            
            # Check for majority
            total_votes = len(agents)
            for option, count in first_choice_counts.items():
                if count > total_votes / 2:
                    return option  # Majority winner
            
            # Eliminate option with fewest votes
            loser = min(first_choice_counts.items(), key=lambda x: x[1])[0]
            remaining_options.remove(loser)
        
        return list(remaining_options)[0]
    
    async def borda_count(self, agents: List[Agent],
                         options: List[str]) -> str:
        """Borda count (points for ranking positions)."""
        
        scores = {opt: 0 for opt in options}
        
        for agent in agents:
            ranking = await agent.rank_options(options)
            
            # Award points based on position
            points = len(options)
            for option in ranking:
                scores[option] += points
                points -= 1
        
        winner = max(scores.items(), key=lambda x: x[1])[0]
        
        return winner
    
    async def approval_vote(self, agents: List[Agent],
                           options: List[str]) -> str:
        """Approval voting (approve multiple options)."""
        
        approvals = {opt: 0 for opt in options}
        
        for agent in agents:
            # Agent approves any acceptable options
            approved = await agent.approve_options(options)
            
            for option in approved:
                approvals[option] += 1
        
        winner = max(approvals.items(), key=lambda x: x[1])[0]
        
        return winner
    
    async def quadratic_vote(self, agents: List[Agent],
                            options: List[str],
                            credits_per_agent: int = 100) -> str:
        """Quadratic voting (express preference intensity)."""
        
        scores = {opt: 0 for opt in options}
        
        for agent in agents:
            # Agent allocates votes (cost is votes^2)
            allocation = await agent.allocate_quadratic_votes(
                options,
                total_credits=credits_per_agent
            )
            
            for option, votes in allocation.items():
                scores[option] += votes
        
        winner = max(scores.items(), key=lambda x: x[1])[0]
        
        return winner

# Example
voting = VotingSystem()

agents = [Agent(f"voter{i}") for i in range(5)]
options = ["Option A", "Option B", "Option C"]

# Different voting methods
plurality_winner = await voting.plurality_vote(agents, options)
ranked_winner = await voting.ranked_choice_vote(agents, options)
borda_winner = await voting.borda_count(agents, options)
approval_winner = await voting.approval_vote(agents, options)

print(f"Plurality: {plurality_winner}")
print(f"Ranked Choice: {ranked_winner}")
print(f"Borda Count: {borda_winner}")
print(f"Approval: {approval_winner}")
```

## Consensus Building

Building agreement:

### Consensus Protocol

```python
class ConsensusBuilder:
    """Build consensus among agents."""
    
    async def reach_consensus(self, agents: List[Agent],
                             question: str,
                             threshold: float = 0.8) -> Dict:
        """Iteratively build consensus."""
        
        iteration = 0
        max_iterations = 10
        
        # Initial proposals
        proposals = await asyncio.gather(*[
            agent.propose(question)
            for agent in agents
        ])
        
        while iteration < max_iterations:
            iteration += 1
            
            # Check if consensus reached
            consensus_level = self.measure_consensus(proposals)
            
            if consensus_level >= threshold:
                # Consensus reached!
                consensus_proposal = self.synthesize_consensus(proposals)
                return {
                    'consensus': True,
                    'proposal': consensus_proposal,
                    'support_level': consensus_level,
                    'iterations': iteration
                }
            
            # Refine proposals based on others' views
            proposals = await asyncio.gather(*[
                agent.refine_proposal(
                    own_proposal=proposals[i],
                    other_proposals=[p for j, p in enumerate(proposals) if j != i]
                )
                for i, agent in enumerate(agents)
            ])
        
        # Consensus not reached
        return {
            'consensus': False,
            'proposals': proposals,
            'support_level': consensus_level,
            'iterations': iteration
        }
    
    def measure_consensus(self, proposals: List[Dict]) -> float:
        """Measure how much proposals agree."""
        
        # Simple measure: similarity of proposals
        # In practice, use semantic similarity
        
        if len(proposals) <= 1:
            return 1.0
        
        similarities = []
        
        for i, prop1 in enumerate(proposals):
            for prop2 in proposals[i+1:]:
                sim = self.compute_similarity(prop1, prop2)
                similarities.append(sim)
        
        return sum(similarities) / len(similarities)
    
    def compute_similarity(self, prop1: Dict, prop2: Dict) -> float:
        """Compute similarity between proposals."""
        # Simplified - use embeddings in practice
        str1 = str(prop1.get('content', ''))
        str2 = str(prop2.get('content', ''))
        
        # Jaccard similarity of words
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def synthesize_consensus(self, proposals: List[Dict]) -> Dict:
        """Synthesize consensus from similar proposals."""
        
        # Take elements common to most proposals
        # Simplified implementation
        
        return {
            'content': 'Synthesized consensus',
            'supported_by': len(proposals),
            'source_proposals': proposals
        }

# Example
consensus = ConsensusBuilder()

agents = [Agent(f"agent{i}") for i in range(5)]

result = await consensus.reach_consensus(
    agents,
    question="What should our product roadmap prioritize?",
    threshold=0.8
)

if result['consensus']:
    print(f"Consensus reached after {result['iterations']} iterations")
    print(f"Support level: {result['support_level']:.2%}")
else:
    print("Consensus not reached")
```

### Delphi Method

```python
class DelphiMethod:
    """Structured consensus building through anonymous rounds."""
    
    async def conduct_delphi(self, experts: List[Agent],
                            question: str,
                            rounds: int = 3) -> Dict:
        """Conduct Delphi method."""
        
        responses = []
        
        for round_num in range(1, rounds + 1):
            print(f"\n=== Delphi Round {round_num} ===\n")
            
            if round_num == 1:
                # Initial responses
                round_responses = await asyncio.gather(*[
                    expert.respond(question)
                    for expert in experts
                ])
            else:
                # Show summary of previous round
                summary = self.summarize_responses(responses[-1])
                
                # Experts revise based on summary
                round_responses = await asyncio.gather(*[
                    expert.revise_response(
                        question,
                        previous_response=responses[-1][i],
                        group_summary=summary
                    )
                    for i, expert in enumerate(experts)
                ])
            
            responses.append(round_responses)
            
            # Check for convergence
            if round_num > 1:
                convergence = self.measure_convergence(
                    responses[-2],
                    responses[-1]
                )
                
                if convergence > 0.9:
                    print("Convergence reached")
                    break
        
        # Final synthesis
        consensus = self.synthesize_delphi(responses[-1])
        
        return {
            'consensus': consensus,
            'rounds': len(responses),
            'all_responses': responses
        }
    
    def summarize_responses(self, responses: List[Dict]) -> Dict:
        """Summarize round of responses (anonymous)."""
        
        # Aggregate statistics
        if all('numeric_estimate' in r for r in responses):
            estimates = [r['numeric_estimate'] for r in responses]
            
            summary = {
                'median': statistics.median(estimates),
                'mean': statistics.mean(estimates),
                'std': statistics.stdev(estimates) if len(estimates) > 1 else 0,
                'range': (min(estimates), max(estimates)),
                'quartiles': [
                    statistics.quantiles(estimates, n=4)
                ]
            }
        else:
            # Qualitative summary
            summary = {
                'themes': self.extract_themes(responses),
                'count': len(responses)
            }
        
        return summary

# Example
delphi = DelphiMethod()

experts = [Agent(f"expert{i}") for i in range(7)]

result = await delphi.conduct_delphi(
    experts,
    question="What will AI's impact be in 10 years?",
    rounds=3
)
```

## Diverse Perspectives

Leveraging diversity:

### Perspective Diversity

```python
class DiverseDebate:
    """Debate system that ensures diverse perspectives."""
    
    def __init__(self):
        self.agents = []
    
    def add_perspective(self, agent: Agent, perspective_type: str):
        """Add agent with specific perspective."""
        
        agent.perspective_type = perspective_type
        self.agents.append(agent)
    
    async def multi_perspective_analysis(self, problem: Dict) -> Dict:
        """Analyze problem from multiple perspectives."""
        
        analyses = {}
        
        # Each perspective analyzes
        for agent in self.agents:
            analysis = await agent.analyze_from_perspective(
                problem,
                perspective=agent.perspective_type
            )
            
            analyses[agent.perspective_type] = analysis
        
        # Cross-perspective discussion
        for agent in self.agents:
            # See other perspectives
            other_perspectives = {
                k: v for k, v in analyses.items()
                if k != agent.perspective_type
            }
            
            # Respond to other perspectives
            response = await agent.respond_to_perspectives(
                own_analysis=analyses[agent.perspective_type],
                other_analyses=other_perspectives
            )
            
            analyses[agent.perspective_type]['response'] = response
        
        # Synthesize perspectives
        synthesis = await self.synthesize_perspectives(analyses)
        
        return {
            'perspectives': analyses,
            'synthesis': synthesis
        }
    
    async def synthesize_perspectives(self, analyses: Dict) -> Dict:
        """Synthesize insights from diverse perspectives."""
        
        # Find common ground
        common_points = self.find_common_points(analyses)
        
        # Identify unique insights from each perspective
        unique_insights = {}
        for perspective, analysis in analyses.items():
            unique = self.extract_unique_insights(
                analysis,
                [a for p, a in analyses.items() if p != perspective]
            )
            unique_insights[perspective] = unique
        
        # Resolve conflicts
        conflicts = self.identify_conflicts(analyses)
        resolutions = await self.resolve_conflicts(conflicts)
        
        return {
            'common_ground': common_points,
            'unique_insights': unique_insights,
            'conflicts': conflicts,
            'resolutions': resolutions
        }

# Example: Different analytical perspectives
debate = DiverseDebate()

debate.add_perspective(Agent("technical"), "technical")
debate.add_perspective(Agent("business"), "business")
debate.add_perspective(Agent("user"), "user_experience")
debate.add_perspective(Agent("ethical"), "ethics")
debate.add_perspective(Agent("legal"), "legal")

problem = {
    'description': 'Should we add facial recognition?',
    'context': {...}
}

result = await debate.multi_perspective_analysis(problem)

print("Common ground:", result['synthesis']['common_ground'])
print("Unique insights:", result['synthesis']['unique_insights'])
```

## Decision Quality

Improving decisions through debate:

### Quality Metrics

```python
class DebateQualityAssessment:
    """Assess quality improvement from debate."""
    
    async def measure_improvement(self, 
                                  question: str,
                                  single_agent: Agent,
                                  debate_agents: List[Agent]) -> Dict:
        """Compare single agent vs debate performance."""
        
        # Single agent decision
        single_decision = await single_agent.decide(question)
        
        # Multi-agent debate decision
        debate = MultiAgentDebate(debate_agents)
        debate_decision = await debate.debate(question)
        
        # Evaluate both (requires ground truth or proxy)
        single_quality = await self.evaluate_decision(
            question,
            single_decision
        )
        
        debate_quality = await self.evaluate_decision(
            question,
            debate_decision
        )
        
        return {
            'single_agent': {
                'decision': single_decision,
                'quality': single_quality
            },
            'debate': {
                'decision': debate_decision,
                'quality': debate_quality
            },
            'improvement': debate_quality - single_quality
        }
    
    async def evaluate_decision(self, question: str, decision: Dict) -> float:
        """Evaluate decision quality."""
        
        # Multiple dimensions of quality
        scores = {
            'correctness': await self.assess_correctness(decision),
            'completeness': await self.assess_completeness(decision),
            'coherence': await self.assess_coherence(decision),
            'robustness': await self.assess_robustness(decision)
        }
        
        # Weighted combination
        quality = (
            scores['correctness'] * 0.4 +
            scores['completeness'] * 0.2 +
            scores['coherence'] * 0.2 +
            scores['robustness'] * 0.2
        )
        
        return quality

# Example
assessment = DebateQualityAssessment()

result = await assessment.measure_improvement(
    question="Should we expand to Europe?",
    single_agent=Agent("expert"),
    debate_agents=[Agent("agent1"), Agent("agent2"), Agent("agent3")]
)

print(f"Single agent quality: {result['single_agent']['quality']:.2f}")
print(f"Debate quality: {result['debate']['quality']:.2f}")
print(f"Improvement: {result['improvement']:.2f}")
```

## Devil's Advocate

Deliberate contrarian:

### Devil's Advocate Pattern

```python
class DevilsAdvocate:
    """Agent that argues against proposals."""
    
    def __init__(self, critical_agent: Agent):
        self.agent = critical_agent
    
    async def challenge_proposal(self, proposal: Dict) -> Dict:
        """Challenge a proposal."""
        
        # Find weaknesses
        weaknesses = await self.agent.identify_weaknesses(proposal)
        
        # Generate counter-arguments
        counter_arguments = await self.agent.generate_counter_arguments(
            proposal,
            weaknesses
        )
        
        # Propose alternatives
        alternatives = await self.agent.propose_alternatives(proposal)
        
        return {
            'weaknesses': weaknesses,
            'counter_arguments': counter_arguments,
            'alternatives': alternatives
        }
    
    async def stress_test(self, proposal: Dict) -> Dict:
        """Stress test proposal with edge cases."""
        
        # Generate challenging scenarios
        scenarios = await self.agent.generate_edge_cases(proposal)
        
        # Test proposal against scenarios
        test_results = []
        
        for scenario in scenarios:
            result = await self.agent.test_proposal(proposal, scenario)
            test_results.append({
                'scenario': scenario,
                'passes': result['passes'],
                'issues': result.get('issues', [])
            })
        
        return {
            'scenarios_tested': len(scenarios),
            'passed': sum(1 for r in test_results if r['passes']),
            'failed': sum(1 for r in test_results if not r['passes']),
            'details': test_results
        }

class DebateWithDevil:
    """Debate system with devil's advocate."""
    
    async def evaluate_proposal(self, proposal: Dict,
                                proponents: List[Agent],
                                devils_advocate: DevilsAdvocate) -> Dict:
        """Evaluate proposal with devil's advocate."""
        
        # Proponents present
        presentation = await self.present_proposal(proposal, proponents)
        
        # Devil's advocate challenges
        challenge = await devils_advocate.challenge_proposal(proposal)
        
        # Proponents respond to challenges
        response = await self.respond_to_challenges(
            proponents,
            challenge
        )
        
        # Devil's advocate stress tests
        stress_test = await devils_advocate.stress_test(proposal)
        
        # Proponents address weaknesses
        improvements = await self.address_weaknesses(
            proponents,
            challenge['weaknesses'],
            stress_test
        )
        
        # Final evaluation
        return {
            'original_proposal': proposal,
            'challenges': challenge,
            'responses': response,
            'stress_test': stress_test,
            'improved_proposal': improvements,
            'recommendation': await self.make_recommendation(
                proposal,
                challenge,
                response,
                improvements
            )
        }

# Example
devils_advocate = DevilsAdvocate(Agent("skeptic"))

debate = DebateWithDevil()

proposal = {
    'description': 'Launch product in Q2',
    'details': {...}
}

result = await debate.evaluate_proposal(
    proposal,
    proponents=[Agent("pm"), Agent("engineer")],
    devils_advocate=devils_advocate
)
```

## Iterative Refinement

Improving through debate cycles:

### Refinement Through Debate

```python
class IterativeRefinement:
    """Refine solutions through debate iterations."""
    
    async def refine(self, initial_solution: Dict,
                    agents: List[Agent],
                    max_iterations: int = 5,
                    convergence_threshold: float = 0.95) -> Dict:
        """Iteratively refine solution."""
        
        current_solution = initial_solution
        history = [initial_solution]
        
        for iteration in range(max_iterations):
            print(f"\n=== Refinement Iteration {iteration + 1} ===\n")
            
            # Agents critique current solution
            critiques = await asyncio.gather(*[
                agent.critique(current_solution)
                for agent in agents
            ])
            
            # Agents propose improvements
            improvements = await asyncio.gather(*[
                agent.propose_improvement(current_solution, critiques)
                for agent in agents
            ])
            
            # Agents vote on improvements
            selected_improvements = await self.select_improvements(
                improvements,
                agents
            )
            
            # Apply improvements
            improved_solution = await self.apply_improvements(
                current_solution,
                selected_improvements
            )
            
            # Evaluate improvement
            improvement_score = await self.measure_improvement(
                current_solution,
                improved_solution
            )
            
            history.append(improved_solution)
            
            # Check convergence
            if improvement_score < (1 - convergence_threshold):
                print("Converged - minimal improvement")
                break
            
            current_solution = improved_solution
        
        return {
            'final_solution': current_solution,
            'iterations': len(history) - 1,
            'history': history
        }
    
    async def select_improvements(self, improvements: List[Dict],
                                 agents: List[Agent]) -> List[Dict]:
        """Select which improvements to apply."""
        
        selected = []
        
        for improvement in improvements:
            # Agents vote on each improvement
            votes_for = 0
            votes_against = 0
            
            for agent in agents:
                vote = await agent.vote_on_improvement(improvement)
                
                if vote == 'for':
                    votes_for += 1
                elif vote == 'against':
                    votes_against += 1
            
            # Accept if majority approves
            if votes_for > votes_against:
                selected.append(improvement)
        
        return selected

# Example
refinement = IterativeRefinement()

agents = [
    Agent("performance"),
    Agent("usability"),
    Agent("security")
]

initial_solution = {
    'algorithm': 'basic_search',
    'parameters': {...}
}

result = await refinement.refine(
    initial_solution,
    agents,
    max_iterations=5
)

print(f"Refined after {result['iterations']} iterations")
```

## Collective Intelligence

Emerging from debate:

### Wisdom of Crowds

```python
class CollectiveIntelligence:
    """Harness collective intelligence through aggregation."""
    
    async def aggregate_estimates(self, agents: List[Agent],
                                  question: str) -> Dict:
        """Aggregate numerical estimates."""
        
        # Each agent provides estimate
        estimates = await asyncio.gather(*[
            agent.estimate(question)
            for agent in agents
        ])
        
        values = [e['value'] for e in estimates]
        
        # Various aggregation methods
        aggregations = {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'trimmed_mean': self.trimmed_mean(values, trim=0.1),
            'weighted_mean': self.weighted_mean(estimates)
        }
        
        return {
            'individual_estimates': estimates,
            'aggregations': aggregations,
            'diversity': statistics.stdev(values),
            'range': (min(values), max(values))
        }
    
    def trimmed_mean(self, values: List[float], trim: float = 0.1) -> float:
        """Mean after removing extreme values."""
        
        sorted_values = sorted(values)
        n_trim = int(len(values) * trim)
        
        if n_trim > 0:
            trimmed = sorted_values[n_trim:-n_trim]
        else:
            trimmed = sorted_values
        
        return statistics.mean(trimmed)
    
    def weighted_mean(self, estimates: List[Dict]) -> float:
        """Weighted by confidence."""
        
        total_weight = 0
        weighted_sum = 0
        
        for estimate in estimates:
            weight = estimate.get('confidence', 1.0)
            weighted_sum += estimate['value'] * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0

# Example
ci = CollectiveIntelligence()

agents = [Agent(f"estimator{i}") for i in range(20)]

result = await ci.aggregate_estimates(
    agents,
    question="How many users will we have next year?"
)

print(f"Mean estimate: {result['aggregations']['mean']:.0f}")
print(f"Median estimate: {result['aggregations']['median']:.0f}")
print(f"Diversity: {result['diversity']:.0f}")
```

## When to Use Debate

When debate helps:

```
Decision Context Analysis
├── High stakes → Use debate
├── Uncertain outcome → Use debate  
├── Multiple valid approaches → Use debate
├── Risk of bias → Use devil's advocate
└── Need creativity → Use brainstorming debate

Decision Complexity
├── Simple, clear-cut → Single agent
├── Nuanced trade-offs → Debate
├── Multiple dimensions → Multi-perspective debate
└── Novel problem → Diverse debate

Resource Constraints
├── Time-critical → Limited debate rounds
├── Cost-sensitive → Smaller group
└── Abundant resources → Full deliberation
```

## Summary

Multi-agent debate leverages diverse perspectives to improve decision quality:

**Core Mechanisms:**
- Structured debate formats (opening, rebuttal, closing)
- Voting systems (plurality, ranked choice, Borda, approval)
- Consensus building (iterative refinement, Delphi method)
- Devil's advocate (challenge assumptions, stress test)

**Key Benefits:**
- Better decisions through diverse perspectives
- Reduced bias from single viewpoints
- More robust solutions through criticism
- Improved creativity through ideation

**Deliberation Patterns:**
- Multi-round debate with position updates
- Cross-examination and questioning
- Collaborative refinement
- Collective intelligence aggregation

**Quality Improvements:**
- Multiple viewpoints catch blind spots
- Criticism identifies weaknesses
- Iteration refines solutions
- Aggregation smooths individual errors

**When to Use:**
- High-stakes decisions
- Uncertain or novel problems
- Multiple valid approaches
- Risk of individual bias

Debate transforms disagreement from a problem into a strength, using diversity to arrive at better solutions than any single agent could achieve.

## Next Steps

- [Collaboration Patterns](collaboration.md) - Cooperative agents
- [Swarm Intelligence](swarm-intelligence.md) - Emergent collective behavior
- [Coordination](coordination.md) - Organizing multi-agent work
- [Agent Roles](agent-roles.md) - Designing specialized agents
- [Multi-Agent Fundamentals](fundamentals.md) - Core concepts
