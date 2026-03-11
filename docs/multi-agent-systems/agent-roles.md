# Agent Roles and Specialization

## Table of Contents

- [Introduction](#introduction)
- [Why Specialize Agents?](#why-specialize-agents)
- [Role-Based Decomposition](#role-based-decomposition)
- [Designing Agent Roles](#designing-agent-roles)
- [Expert Agents](#expert-agents)
- [Generalist vs Specialist Trade-offs](#generalist-vs-specialist-trade-offs)
- [Role Boundaries](#role-boundaries)
- [Responsibility Assignment](#responsibility-assignment)
- [Role Hierarchies](#role-hierarchies)
- [Dynamic Role Assignment](#dynamic-role-assignment)
- [Role Composition](#role-composition)
- [Common Agent Roles](#common-agent-roles)
- [Role Evolution](#role-evolution)
- [Anti-Patterns](#anti-patterns)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

In multi-agent systems, **roles** define what agents do and **specialization** determines how well they do it. Just as a successful company doesn't hire one person to do everything, effective multi-agent systems assign specific roles to individual agents.

> "The right agent for the right job."

An agent's role encompasses:

- **Capabilities**: What the agent can do
- **Responsibilities**: What the agent should do
- **Boundaries**: What the agent should not do
- **Interfaces**: How the agent interacts with others

Specialization allows agents to develop deep expertise in narrow domains, leading to higher quality outputs than generalist agents attempting everything.

### The Specialization Advantage

```python
# Generalist agent (mediocre at everything)
class UniversalAgent:
    def handle(self, task):
        if task.type == "research":
            return self.mediocre_research(task)  # Competent but not great
        elif task.type == "coding":
            return self.mediocre_coding(task)     # Competent but not great
        elif task.type == "design":
            return self.mediocre_design(task)     # Competent but not great

# Specialist agents (excellent at one thing)
class ResearchExpert:
    def handle(self, task):
        return self.excellent_research(task)  # Expert-level quality

class CodingExpert:
    def handle(self, task):
        return self.excellent_coding(task)    # Expert-level quality

class DesignExpert:
    def handle(self, task):
        return self.excellent_design(task)    # Expert-level quality
```

This guide explores how to design effective agent roles, divide responsibilities, balance specialization with flexibility, and create agent teams that leverage the power of expertise.

## Why Specialize Agents?

Specialization provides several critical advantages:

### 1. Expertise Depth

```python
class SpecializationDemo:
    """Compare generalist vs specialist performance."""

    def __init__(self):
        # Generalist: Basic prompt for all tasks
        self.generalist = Agent(
            system_prompt="You are a helpful AI assistant."
        )

        # Specialist: Deep, focused prompt
        self.specialist = Agent(
            system_prompt="""
            You are an expert Python performance optimization engineer.

            Expertise:
            - Profiling and benchmarking
            - Algorithm complexity analysis
            - Memory optimization
            - Cython and PyPy
            - Vectorization with NumPy
            - Async programming

            When analyzing code:
            1. Profile before optimizing
            2. Focus on bottlenecks
            3. Measure impact
            4. Consider readability trade-offs
            5. Suggest multiple approaches

            Tools available:
            - cProfile, line_profiler
            - memory_profiler
            - py-spy
            - timeit
            """
        )

    async def compare_quality(self, code: str):
        """Compare optimization quality."""

        # Generalist attempt
        gen_result = await self.generalist.optimize(code)
        # Generic suggestions, may miss key optimizations

        # Specialist attempt
        spec_result = await self.specialist.optimize(code)
        # Detailed analysis, expert-level optimizations

        return {
            "generalist": gen_result,
            "specialist": spec_result,
            "quality_improvement": self.measure_quality(spec_result, gen_result)
        }

# Specialist typically 2-3x better quality due to:
# - Focused prompt engineering
# - Domain-specific knowledge
# - Specialized tools
# - Expert reasoning patterns
```

### 2. Prompt Efficiency

```python
class PromptEfficiency:
    """Specialists use context more efficiently."""

    def create_prompts(self):
        # Generalist must include instructions for everything
        generalist_prompt = """
        You can help with:
        - Research (search, summarize, cite)
        - Coding (write, debug, optimize)
        - Writing (articles, emails, reports)
        - Analysis (data, trends, insights)
        - Design (UI, UX, graphics)

        For research: [500 words of instructions]
        For coding: [500 words of instructions]
        For writing: [500 words of instructions]
        ...
        Total: ~2500 words
        """

        # Specialist only needs focused instructions
        research_specialist_prompt = """
        You are a research specialist.

        Capabilities:
        - Academic paper search
        - Citation management
        - Literature review

        [500 words of detailed research instructions]
        Total: ~500 words
        """

        # Specialist uses 5x less prompt space
        # More room for task-specific context
        # Clearer, more focused instructions

# Example with context limits
context_limit = 8000  # tokens

# Generalist
generalist_instructions = 2500  # tokens
task_context_space = 8000 - 2500  # = 5500 tokens

# Specialist
specialist_instructions = 500  # tokens
task_context_space = 8000 - 500  # = 7500 tokens

# Specialist has 36% more space for actual task context!
```

### 3. Tool Specialization

```python
class ToolSpecialization:
    """Specialists have focused toolsets."""

    class GeneralistAgent:
        """Many tools, not expert with any."""
        tools = [
            "web_search", "calculator", "python_repl",
            "file_read", "file_write", "sql_query",
            "image_generate", "send_email", "calendar"
        ]  # 9 tools - complex decision making

        def select_tool(self, task):
            # Must reason about all 9 tools
            # Higher chance of choosing wrong tool
            # More tokens spent on tool selection
            pass

    class ResearchSpecialist:
        """Few tools, expert usage."""
        tools = [
            "academic_search",
            "paper_retrieval",
            "citation_extraction"
        ]  # 3 tools - clear choice

        def select_tool(self, task):
            # Only 3 relevant tools to consider
            # Clear expertise with each
            # Quick, confident selection
            pass

# Specialist advantages:
# - Fewer tools = simpler decision making
# - Each tool deeply integrated into workflow
# - Better tool usage patterns
# - Lower latency in tool selection
```

### 4. Quality Assurance

```python
class QualityAssurance:
    """Specialists produce more consistent quality."""

    def compare_consistency(self):
        # Generalist: Variable quality across domains
        generalist_scores = {
            "research": 7.0,  # Average
            "coding": 6.5,    # Below average
            "writing": 8.0,   # Good
            "analysis": 6.0,  # Below average
            "design": 5.5     # Poor
        }
        avg_quality = 6.6
        variance = 1.2

        # Specialists: High quality in their domain
        specialist_scores = {
            "research_specialist": 9.5,  # Excellent
            "coding_specialist": 9.0,     # Excellent
            "writing_specialist": 9.5,    # Excellent
            "analysis_specialist": 8.5,   # Very good
            "design_specialist": 9.0      # Excellent
        }
        avg_quality = 9.1
        variance = 0.3

        # Specialists: 38% higher quality, 75% less variance

# Predictable quality is crucial for production
# Specialists deliver consistent excellence
```

## Role-Based Decomposition

Breaking down systems into roles is an art. Here's how to do it effectively:

### Decomposition Strategies

**1. Functional Decomposition**

```python
class FunctionalDecomposition:
    """Decompose by function/activity."""

    def create_content_pipeline(self):
        """
        Task: Create blog post
        Functions: Research → Write → Edit → Publish
        """

        agents = {
            "researcher": ResearchAgent(
                function="gather_information",
                inputs=["topic"],
                outputs=["sources", "facts", "data"]
            ),

            "writer": WriterAgent(
                function="create_content",
                inputs=["sources", "facts", "outline"],
                outputs=["draft"]
            ),

            "editor": EditorAgent(
                function="improve_quality",
                inputs=["draft"],
                outputs=["polished_article"]
            ),

            "publisher": PublisherAgent(
                function="distribute_content",
                inputs=["polished_article"],
                outputs=["published_url"]
            )
        }

        return agents

# Clear functional boundaries
# Each agent does one thing well
# Easy to understand and maintain
```

**2. Domain Decomposition**

```python
class DomainDecomposition:
    """Decompose by knowledge domain."""

    def create_medical_system(self):
        """
        System: Medical diagnosis assistant
        Domains: Cardiology, Neurology, Radiology, etc.
        """

        agents = {
            "cardiologist": MedicalAgent(
                domain="cardiology",
                expertise=[
                    "heart_conditions",
                    "cardiovascular_tests",
                    "cardiac_medications"
                ]
            ),

            "neurologist": MedicalAgent(
                domain="neurology",
                expertise=[
                    "brain_conditions",
                    "neurological_tests",
                    "neuro_medications"
                ]
            ),

            "radiologist": MedicalAgent(
                domain="radiology",
                expertise=[
                    "image_interpretation",
                    "scan_analysis",
                    "imaging_recommendations"
                ]
            ),

            "general_practitioner": MedicalAgent(
                domain="general_medicine",
                expertise=[
                    "initial_assessment",
                    "referral_decisions",
                    "general_care"
                ]
            )
        }

        return agents

# Domain experts provide depth
# GP routes to specialists as needed
# Mirrors real medical practice
```

**3. Responsibility Decomposition**

```python
class ResponsibilityDecomposition:
    """Decompose by area of responsibility."""

    def create_software_team(self):
        """
        System: Software development
        Responsibilities: Requirements, Design, Code, Test, Deploy
        """

        agents = {
            "product_manager": ProductAgent(
                responsibilities=[
                    "gather_requirements",
                    "prioritize_features",
                    "define_acceptance_criteria"
                ],
                authority="requirements_approval"
            ),

            "architect": ArchitectAgent(
                responsibilities=[
                    "system_design",
                    "technology_selection",
                    "architecture_decisions"
                ],
                authority="design_approval"
            ),

            "developer": DeveloperAgent(
                responsibilities=[
                    "implement_features",
                    "write_tests",
                    "fix_bugs"
                ],
                authority="code_changes"
            ),

            "qa_engineer": QAAgent(
                responsibilities=[
                    "test_planning",
                    "quality_validation",
                    "bug_reporting"
                ],
                authority="quality_approval"
            )
        }

        return agents

# Clear ownership
# Accountability for outcomes
# Authority aligned with responsibility
```

**4. Temporal Decomposition**

```python
class TemporalDecomposition:
    """Decompose by time/phase."""

    def create_project_lifecycle(self):
        """
        System: Project management
        Phases: Planning → Execution → Monitoring → Closure
        """

        agents = {
            "planner": PlanningAgent(
                phase="planning",
                activities=[
                    "create_project_plan",
                    "estimate_resources",
                    "identify_risks"
                ],
                handoff_to="executor"
            ),

            "executor": ExecutionAgent(
                phase="execution",
                activities=[
                    "coordinate_tasks",
                    "manage_team",
                    "track_progress"
                ],
                handoff_to="monitor"
            ),

            "monitor": MonitoringAgent(
                phase="monitoring",
                activities=[
                    "track_metrics",
                    "identify_issues",
                    "suggest_adjustments"
                ],
                handoff_to="executor"  # Or closer
            ),

            "closer": ClosureAgent(
                phase="closure",
                activities=[
                    "finalize_deliverables",
                    "document_lessons",
                    "release_resources"
                ]
            )
        }

        return agents

# Agents specialize in project phases
# Natural handoff points
# Supports iterative processes
```

## Designing Agent Roles

Effective role design requires careful consideration:

### Role Definition Template

```python
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class AgentCapability(Enum):
    """Types of capabilities an agent can have."""
    TOOL_USE = "tool_use"
    KNOWLEDGE = "knowledge"
    REASONING = "reasoning"
    COMMUNICATION = "communication"

@dataclass
class AgentRole:
    """Complete specification of an agent role."""

    # Identity
    role_name: str
    role_description: str

    # Capabilities
    capabilities: List[Dict[str, any]]
    tools: List[str]
    knowledge_domains: List[str]

    # Responsibilities
    primary_responsibilities: List[str]
    secondary_responsibilities: List[str]

    # Boundaries
    out_of_scope: List[str]
    escalation_conditions: List[str]

    # Interactions
    communicates_with: List[str]
    reports_to: Optional[str]
    manages: List[str]

    # Constraints
    response_time_target: float  # seconds
    quality_threshold: float  # 0-1
    resource_limits: Dict[str, any]

    # Performance metrics
    success_criteria: List[str]
    kpis: Dict[str, str]

# Example: Research Agent Role
research_agent_role = AgentRole(
    role_name="Senior Research Analyst",
    role_description="Expert at finding, analyzing, and synthesizing information",

    capabilities=[
        {"type": "tool_use", "tools": ["search", "scrape", "extract"]},
        {"type": "knowledge", "domains": ["research_methods", "citation"]},
        {"type": "reasoning", "skills": ["analysis", "synthesis"]}
    ],

    tools=[
        "academic_search",
        "web_search",
        "paper_retrieval",
        "citation_extraction",
        "summarization"
    ],

    knowledge_domains=[
        "research_methodology",
        "information_literacy",
        "source_evaluation",
        "citation_standards"
    ],

    primary_responsibilities=[
        "Find relevant sources for given topics",
        "Extract key information from sources",
        "Synthesize findings into coherent summaries",
        "Provide properly formatted citations"
    ],

    secondary_responsibilities=[
        "Evaluate source credibility",
        "Identify knowledge gaps",
        "Suggest related topics"
    ],

    out_of_scope=[
        "Writing full articles (Writer's job)",
        "Making strategic decisions (Manager's job)",
        "Publishing content (Publisher's job)"
    ],

    escalation_conditions=[
        "Unable to find credible sources",
        "Contradictory information from reliable sources",
        "Topic requires specialized expertise"
    ],

    communicates_with=["writer", "editor", "coordinator"],
    reports_to="coordinator",
    manages=[],

    response_time_target=30.0,
    quality_threshold=0.8,
    resource_limits={
        "max_searches": 10,
        "max_papers": 20,
        "max_time": 300
    },

    success_criteria=[
        "Sources are credible and relevant",
        "Information is accurate and up-to-date",
        "Citations are properly formatted",
        "Summary is clear and comprehensive"
    ],

    kpis={
        "avg_response_time": "< 30s",
        "source_quality": "> 0.9",
        "citation_accuracy": "> 0.95",
        "user_satisfaction": "> 4.5/5"
    }
)
```

### Role Design Checklist

```python
class RoleDesignChecklist:
    """Validate role design quality."""

    def validate_role(self, role: AgentRole) -> Dict[str, bool]:
        """Check if role is well-designed."""

        checks = {}

        # 1. Clear purpose
        checks["has_clear_purpose"] = (
            len(role.role_description) > 20 and
            len(role.primary_responsibilities) > 0
        )

        # 2. Appropriate scope
        checks["appropriate_scope"] = (
            2 <= len(role.primary_responsibilities) <= 5  # Not too narrow or broad
        )

        # 3. Clear boundaries
        checks["clear_boundaries"] = (
            len(role.out_of_scope) > 0 and
            len(role.escalation_conditions) > 0
        )

        # 4. Sufficient capabilities
        checks["sufficient_capabilities"] = (
            len(role.tools) >= len(role.primary_responsibilities) and
            len(role.knowledge_domains) > 0
        )

        # 5. Defined interactions
        checks["defined_interactions"] = (
            len(role.communicates_with) > 0
        )

        # 6. Measurable success
        checks["measurable_success"] = (
            len(role.success_criteria) > 0 and
            len(role.kpis) > 0
        )

        # 7. Realistic constraints
        checks["realistic_constraints"] = (
            role.response_time_target > 0 and
            0 < role.quality_threshold <= 1
        )

        return checks

    def suggest_improvements(self, role: AgentRole,
                           checks: Dict[str, bool]) -> List[str]:
        """Suggest improvements for failed checks."""

        suggestions = []

        if not checks["has_clear_purpose"]:
            suggestions.append(
                "Define clearer purpose and primary responsibilities"
            )

        if not checks["appropriate_scope"]:
            suggestions.append(
                "Adjust scope - too narrow or too broad"
            )

        if not checks["clear_boundaries"]:
            suggestions.append(
                "Define what's out of scope and escalation conditions"
            )

        if not checks["sufficient_capabilities"]:
            suggestions.append(
                "Add more tools or knowledge domains to support responsibilities"
            )

        if not checks["defined_interactions"]:
            suggestions.append(
                "Specify which agents this role communicates with"
            )

        if not checks["measurable_success"]:
            suggestions.append(
                "Add measurable success criteria and KPIs"
            )

        if not checks["realistic_constraints"]:
            suggestions.append(
                "Set realistic response time and quality thresholds"
            )

        return suggestions
```

## Expert Agents

Expert agents are specialists that excel in narrow domains:

### Building Expert Agents

```python
class ExpertAgent:
    """Agent with deep expertise in specific domain."""

    def __init__(self, domain: str):
        self.domain = domain
        self.expertise_level = "expert"

        # Expert-level system prompt
        self.system_prompt = self._build_expert_prompt()

        # Specialized tools
        self.tools = self._get_domain_tools()

        # Knowledge base
        self.knowledge = self._load_domain_knowledge()

    def _build_expert_prompt(self) -> str:
        """Create expert-level system prompt."""

        if self.domain == "python_performance":
            return """
            You are a world-class Python performance optimization expert.

            EXPERTISE:
            - 15+ years optimizing Python code
            - Deep knowledge of CPython internals
            - Expert in profiling tools (cProfile, py-spy, memory_profiler)
            - Mastery of optimization techniques:
              * Algorithm complexity reduction
              * Data structure selection
              * Memory optimization
              * Cython and PyPy usage
              * NumPy vectorization
              * Async/await patterns
              * C extensions

            METHODOLOGY:
            1. PROFILE FIRST - Never optimize without data
            2. FOCUS ON BOTTLENECKS - Amdahl's Law applies
            3. MEASURE IMPACT - Verify improvements
            4. CONSIDER TRADE-OFFS - Performance vs readability
            5. DOCUMENT DECISIONS - Explain why optimizations work

            BEST PRACTICES:
            - Use appropriate data structures (dict vs list vs set)
            - Leverage built-ins (they're optimized C code)
            - Minimize allocations in hot loops
            - Use generators for large datasets
            - Profile in production-like environment
            - Consider memory vs CPU trade-offs

            ANTI-PATTERNS TO AVOID:
            - Premature optimization
            - Optimizing cold paths
            - Sacrificing maintainability for micro-optimizations
            - Ignoring algorithmic complexity
            - Not measuring impact

            When analyzing code:
            1. Identify bottlenecks through profiling
            2. Determine root cause (algorithm, I/O, memory)
            3. Propose 2-3 optimization strategies
            4. Estimate impact of each strategy
            5. Recommend best approach with rationale
            6. Provide optimized code with benchmarks
            """

        # Other domain prompts...
        return ""

    def _get_domain_tools(self) -> List[str]:
        """Get tools specific to domain."""

        domain_tools = {
            "python_performance": [
                "run_cprofile",
                "run_line_profiler",
                "memory_profiler",
                "run_benchmark",
                "analyze_complexity"
            ],
            "machine_learning": [
                "load_dataset",
                "train_model",
                "evaluate_model",
                "hyperparameter_tune",
                "visualize_results"
            ],
            "web_scraping": [
                "fetch_page",
                "parse_html",
                "extract_data",
                "handle_javascript",
                "manage_session"
            ]
        }

        return domain_tools.get(self.domain, [])

    def _load_domain_knowledge(self) -> Dict:
        """Load domain-specific knowledge."""

        # In practice, this would load from a knowledge base
        knowledge = {
            "python_performance": {
                "algorithms": {
                    "O(1)": ["dict lookup", "set membership", "deque append"],
                    "O(log n)": ["bisect", "heapq", "sorted search"],
                    "O(n)": ["list iteration", "sum", "any/all"],
                    "O(n log n)": ["sorted", "heapq.merge"],
                    "O(n²)": ["nested loops", "naive search"]
                },
                "data_structures": {
                    "list": "Dynamic array, O(1) append, O(n) insert",
                    "dict": "Hash table, O(1) average lookup",
                    "set": "Hash table, O(1) membership",
                    "deque": "Double-ended queue, O(1) both ends",
                    "heapq": "Binary heap, O(log n) push/pop"
                },
                "profiling_tools": {
                    "cProfile": "Function-level profiling",
                    "line_profiler": "Line-by-line profiling",
                    "memory_profiler": "Memory usage per line",
                    "py-spy": "Sampling profiler (no code changes)"
                }
            }
        }

        return knowledge.get(self.domain, {})

    async def analyze(self, code: str) -> Dict:
        """Expert analysis of code."""

        # Run profiling tools
        profile_data = await self.profile_code(code)

        # Apply expert knowledge
        bottlenecks = self.identify_bottlenecks(profile_data)

        # Generate expert recommendations
        recommendations = self.generate_recommendations(bottlenecks)

        # Provide optimized code
        optimized = await self.optimize_code(code, recommendations)

        return {
            "analysis": profile_data,
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "optimized_code": optimized,
            "expected_improvement": self.estimate_improvement(profile_data, optimized)
        }

# Expert agent provides deep, domain-specific insights
# Far better than generalist agent
```

### Expert Agent Examples

```python
class SecurityExpertAgent:
    """Security expert specialized in vulnerability detection."""

    system_prompt = """
    You are a cybersecurity expert specializing in application security.

    EXPERTISE:
    - OWASP Top 10
    - Secure coding practices
    - Penetration testing
    - Vulnerability assessment
    - Security code review

    FOCUS AREAS:
    1. Injection attacks (SQL, NoSQL, Command, LDAP)
    2. Authentication/Authorization flaws
    3. Sensitive data exposure
    4. XML External Entities (XXE)
    5. Broken access control
    6. Security misconfiguration
    7. Cross-Site Scripting (XSS)
    8. Insecure deserialization
    9. Using components with known vulnerabilities
    10. Insufficient logging and monitoring

    METHODOLOGY:
    - Threat modeling
    - Static code analysis
    - Dynamic testing
    - Security by design
    - Defense in depth
    """

    tools = [
        "run_static_analysis",
        "check_dependencies",
        "scan_for_secrets",
        "test_authentication",
        "test_authorization",
        "fuzzing",
        "vulnerability_database_lookup"
    ]

class DataScienceExpertAgent:
    """Data science expert for ML/analysis tasks."""

    system_prompt = """
    You are a senior data scientist with expertise in:

    STATISTICAL ANALYSIS:
    - Hypothesis testing
    - A/B testing
    - Regression analysis
    - Time series analysis
    - Bayesian inference

    MACHINE LEARNING:
    - Supervised learning (classification, regression)
    - Unsupervised learning (clustering, dimensionality reduction)
    - Feature engineering
    - Model selection and validation
    - Hyperparameter tuning
    - Model interpretation (SHAP, LIME)

    BEST PRACTICES:
    - Exploratory Data Analysis (EDA) first
    - Train/val/test split
    - Cross-validation
    - Handling imbalanced data
    - Feature scaling and normalization
    - Preventing overfitting
    - Model monitoring in production
    """

    tools = [
        "load_data",
        "exploratory_analysis",
        "statistical_test",
        "train_model",
        "cross_validate",
        "feature_importance",
        "hyperparameter_search",
        "model_interpretation"
    ]
```

## Generalist vs Specialist Trade-offs

When to use each approach:

### Generalist Agents

```python
class GeneralistAgent:
    """Agent that handles variety of tasks."""

    advantages = [
        "Flexible - handles unexpected tasks",
        "Simpler architecture - fewer agents to manage",
        "Better context - sees full picture",
        "Lower coordination overhead",
        "Easier to deploy - single agent"
    ]

    disadvantages = [
        "Lower quality - not expert at anything",
        "Longer prompts - instructions for everything",
        "Tool confusion - many tools to choose from",
        "Inconsistent performance across domains",
        "Harder to optimize"
    ]

    best_for = [
        "Prototyping and experimentation",
        "Low-stakes applications",
        "Simple, varied tasks",
        "When speed matters more than quality",
        "Small-scale applications"
    ]

# Example: Personal assistant
class PersonalAssistant(GeneralistAgent):
    """Handles varied personal tasks."""

    tasks = [
        "Schedule meetings",
        "Draft emails",
        "Search information",
        "Set reminders",
        "Answer questions",
        "Manage tasks"
    ]

    # All tasks are simple enough for generalist
    # User wants single interface
    # Quality bar is moderate
```

### Specialist Agents

```python
class SpecialistAgent:
    """Agent focused on specific domain."""

    advantages = [
        "Higher quality - expert in domain",
        "Efficient prompts - focused instructions",
        "Tool mastery - expert tool usage",
        "Consistent performance in domain",
        "Easier to optimize domain-specific behavior"
    ]

    disadvantages = [
        "Inflexible - only handles specific tasks",
        "Complex architecture - many agents to coordinate",
        "Limited context - sees only their part",
        "Higher coordination overhead",
        "More complex deployment"
    ]

    best_for = [
        "Production applications",
        "High-stakes domains",
        "Complex, specialized tasks",
        "When quality matters most",
        "Large-scale applications"
    ]

# Example: Medical diagnosis system
class MedicalDiagnosisSystem:
    """Multiple specialist agents."""

    specialists = {
        "cardiologist": HeartSpecialist(),
        "neurologist": BrainSpecialist(),
        "radiologist": ImagingSpecialist(),
        "pathologist": LabSpecialist()
    }

    # Each agent is expert in their field
    # High-stakes domain requires expertise
    # Quality is critical
```

### Hybrid Approach

```python
class HybridApproach:
    """Combine generalists and specialists."""

    def __init__(self):
        # Generalist for routing and simple tasks
        self.coordinator = GeneralistAgent(role="coordinator")

        # Specialists for complex tasks
        self.specialists = {
            "coding": CodingExpert(),
            "research": ResearchExpert(),
            "analysis": AnalysisExpert()
        }

    async def handle_task(self, task: Dict) -> Any:
        """Route to specialist or handle directly."""

        # Coordinator evaluates complexity
        complexity = await self.coordinator.assess_complexity(task)

        if complexity < 5:  # Simple task
            # Generalist handles directly
            return await self.coordinator.handle(task)

        else:  # Complex task
            # Route to specialist
            specialist_type = await self.coordinator.identify_specialist(task)
            specialist = self.specialists[specialist_type]
            return await specialist.handle(task)

# Best of both worlds:
# - Generalist for routing and simple tasks
# - Specialists for complex, important tasks
# - Efficient resource usage
```

### Decision Matrix

```python
def choose_agent_type(task_characteristics: Dict) -> str:
    """Decide between generalist and specialist."""

    score = 0

    # Factors favoring specialist
    if task_characteristics["complexity"] > 7:
        score += 2
    if task_characteristics["stakes"] == "high":
        score += 2
    if task_characteristics["volume"] > 100:  # Many similar tasks
        score += 1
    if task_characteristics["domain_depth_needed"]:
        score += 2
    if task_characteristics["quality_critical"]:
        score += 2

    # Factors favoring generalist
    if task_characteristics["variety"] == "high":
        score -= 2
    if task_characteristics["simplicity"] > 7:
        score -= 1
    if task_characteristics["prototype"]:
        score -= 2
    if task_characteristics["low_volume"]:
        score -= 1

    if score >= 4:
        return "specialist"
    elif score <= -2:
        return "generalist"
    else:
        return "hybrid"

# Example usage
medical_diagnosis = {
    "complexity": 9,
    "stakes": "high",
    "volume": 1000,
    "domain_depth_needed": True,
    "quality_critical": True,
    "variety": "low",
    "simplicity": 2,
    "prototype": False,
    "low_volume": False
}
print(choose_agent_type(medical_diagnosis))  # "specialist"

personal_assistant = {
    "complexity": 4,
    "stakes": "low",
    "volume": 100,
    "domain_depth_needed": False,
    "quality_critical": False,
    "variety": "high",
    "simplicity": 7,
    "prototype": False,
    "low_volume": False
}
print(choose_agent_type(personal_assistant))  # "generalist"
```

## Role Boundaries

Clear boundaries prevent conflicts and overlap:

### Defining Boundaries

```python
class RoleBoundaries:
    """Define what each role should and shouldn't do."""

    @dataclass
    class Boundary:
        role: str
        in_scope: List[str]
        out_of_scope: List[str]
        escalate_when: List[str]
        consult_with: List[str]

    def define_boundaries(self):
        """Example boundary definitions."""

        researcher = self.Boundary(
            role="Researcher",

            in_scope=[
                "Search for information",
                "Extract key facts",
                "Cite sources",
                "Summarize findings"
            ],

            out_of_scope=[
                "Writing full articles (Writer's role)",
                "Making strategic decisions (Manager's role)",
                "Fact-checking writer's work (Editor's role)",
                "Publishing content (Publisher's role)"
            ],

            escalate_when=[
                "Can't find credible sources",
                "Contradictory information found",
                "Topic requires specialized expertise",
                "Information is paywalled"
            ],

            consult_with=[
                "Writer (for clarification on research needs)",
                "Subject matter experts (for technical topics)",
                "Manager (for scope questions)"
            ]
        )

        writer = self.Boundary(
            role="Writer",

            in_scope=[
                "Create article drafts",
                "Structure content logically",
                "Write clearly and engagingly",
                "Incorporate research findings"
            ],

            out_of_scope=[
                "Conducting research (Researcher's role)",
                "Final editing (Editor's role)",
                "SEO optimization (SEO Specialist's role)",
                "Publishing (Publisher's role)"
            ],

            escalate_when=[
                "Insufficient research provided",
                "Topic outside writing expertise",
                "Contradictory requirements",
                "Deadline cannot be met"
            ],

            consult_with=[
                "Researcher (for more information)",
                "Editor (for style guidance)",
                "Manager (for direction)"
            ]
        )

        return {"researcher": researcher, "writer": writer}

# Benefits of clear boundaries:
# - Prevents role confusion
# - Avoids duplicate work
# - Enables accountability
# - Facilitates coordination
```

### Boundary Enforcement

```python
class BoundaryEnforcement:
    """Enforce role boundaries in multi-agent system."""

    def __init__(self, boundaries: Dict[str, RoleBoundaries.Boundary]):
        self.boundaries = boundaries

    async def validate_task(self, agent_role: str, task: Dict) -> Dict:
        """Check if task is within agent's boundaries."""

        boundary = self.boundaries[agent_role]
        task_type = task["type"]

        # Check if in scope
        if task_type in boundary.in_scope:
            return {"allowed": True, "action": "proceed"}

        # Check if out of scope
        if task_type in boundary.out_of_scope:
            # Find correct agent for this task
            correct_agent = self.find_responsible_agent(task_type)
            return {
                "allowed": False,
                "action": "redirect",
                "redirect_to": correct_agent,
                "reason": f"{task_type} is outside {agent_role}'s scope"
            }

        # Check if escalation needed
        if task_type in boundary.escalate_when:
            return {
                "allowed": False,
                "action": "escalate",
                "escalate_to": "manager",
                "reason": f"{task_type} requires escalation"
            }

        # Unknown task type - seek clarification
        return {
            "allowed": False,
            "action": "clarify",
            "consult_with": boundary.consult_with,
            "reason": f"Unclear if {task_type} is in scope"
        }

    async def handle_boundary_violation(self, agent: str, task: Dict,
                                       violation: Dict):
        """Handle when agent attempts out-of-scope task."""

        if violation["action"] == "redirect":
            # Redirect to appropriate agent
            correct_agent = violation["redirect_to"]
            print(f"Redirecting task from {agent} to {correct_agent}")
            await self.send_task_to(correct_agent, task)

        elif violation["action"] == "escalate":
            # Escalate to manager
            print(f"Escalating task from {agent} to manager")
            await self.escalate_task(task, violation["reason"])

        elif violation["action"] == "clarify":
            # Seek clarification
            print(f"Seeking clarification for {agent}'s task")
            await self.request_clarification(agent, task, violation["consult_with"])

# Boundary enforcement prevents:
# - Agents doing work outside expertise
# - Duplicate work across agents
# - Quality issues from wrong agent handling task
```

## Responsibility Assignment

Assign responsibilities systematically:

### RACI Matrix

```python
class RACIMatrix:
    """Responsibility Assignment Matrix for multi-agent system."""

    # RACI:
    # R = Responsible (does the work)
    # A = Accountable (final authority)
    # C = Consulted (provides input)
    # I = Informed (kept in the loop)

    def create_raci(self):
        """Example RACI matrix for content creation."""

        matrix = {
            #                R  W  E  M  P    (Researcher, Writer, Editor, Manager, Publisher)
            "Define topic": ["I","I","I","RA","I"],
            "Research":     ["RA","C","I","I","I"],
            "Write draft":  ["C","RA","I","I","I"],
            "Review":       ["I","I","RA","C","I"],
            "Edit":         ["I","C","RA","I","I"],
            "Approve":      ["I","I","I","RA","I"],
            "Publish":      ["I","I","I","I","RA"],
            "Monitor":      ["I","I","I","C","RA"]
        }

        return matrix

    def check_task_assignment(self, task: str, agent: str,
                            matrix: Dict) -> str:
        """Check agent's role for this task."""

        agents = ["Researcher", "Writer", "Editor", "Manager", "Publisher"]
        agent_idx = agents.index(agent)

        assignment = matrix[task][agent_idx]

        roles = {
            "R": "You are responsible for doing this task",
            "A": "You are accountable - you have final authority",
            "C": "You should be consulted - provide input",
            "I": "You should be informed - stay aware",
            "RA": "You are both responsible AND accountable"
        }

        return roles.get(assignment, "No role in this task")

# RACI ensures:
# - Clear accountability (only one A per task)
# - No confusion about who does what
# - Appropriate consultation
# - Proper information flow
```

### Task Assignment Algorithm

```python
class TaskAssignment:
    """Intelligent task assignment to agents."""

    def __init__(self, agents: Dict[str, Agent]):
        self.agents = agents
        self.agent_capabilities = self._build_capability_map()

    def _build_capability_map(self) -> Dict:
        """Map capabilities to agents."""

        capability_map = {}

        for agent_id, agent in self.agents.items():
            for capability in agent.capabilities:
                if capability not in capability_map:
                    capability_map[capability] = []

                capability_map[capability].append({
                    "agent_id": agent_id,
                    "proficiency": agent.get_proficiency(capability),
                    "availability": agent.is_available(),
                    "load": agent.current_load()
                })

        return capability_map

    def assign_task(self, task: Dict) -> str:
        """Assign task to best available agent."""

        required_capability = task["requires"]

        # Get agents with this capability
        candidates = self.capability_map.get(required_capability, [])

        if not candidates:
            raise ValueError(f"No agent has capability: {required_capability}")

        # Score each candidate
        scores = []
        for candidate in candidates:
            score = (
                candidate["proficiency"] * 0.5 +  # Ability weight
                candidate["availability"] * 0.3 +  # Availability weight
                (1.0 - candidate["load"]) * 0.2    # Load weight
            )
            scores.append((candidate["agent_id"], score))

        # Return best agent
        best_agent = max(scores, key=lambda x: x[1])
        return best_agent[0]

    def assign_complex_task(self, task: Dict) -> Dict[str, List[str]]:
        """Assign complex task requiring multiple capabilities."""

        subtasks = self.decompose_task(task)

        assignments = {}

        for subtask in subtasks:
            agent_id = self.assign_task(subtask)

            if agent_id not in assignments:
                assignments[agent_id] = []

            assignments[agent_id].append(subtask)

        return assignments

# Intelligent assignment considers:
# - Agent capabilities
# - Agent proficiency
# - Current availability
# - Current workload
```

## Common Agent Roles

Standard roles that appear in many systems:

### 1. Coordinator/Manager Role

```python
class CoordinatorAgent:
    """Manages and coordinates other agents."""

    responsibilities = [
        "Decompose complex tasks",
        "Assign work to specialists",
        "Monitor progress",
        "Aggregate results",
        "Handle errors and exceptions"
    ]

    capabilities = [
        "Task planning",
        "Work allocation",
        "Progress tracking",
        "Result synthesis"
    ]

    system_prompt = """
    You are a coordinator agent managing a team of specialists.

    Your role:
    1. Understand user requests
    2. Break down into subtasks
    3. Assign to appropriate specialists
    4. Track execution
    5. Synthesize results
    6. Ensure quality

    Available specialists:
    - Researcher: Information gathering
    - Analyst: Data analysis
    - Writer: Content creation
    - Coder: Software development

    When coordinating:
    - Delegate rather than doing yourself
    - Provide clear instructions to specialists
    - Monitor for issues
    - Integrate specialist outputs
    - Maintain big picture view
    """
```

### 2. Research/Information Gathering Role

```python
class ResearcherAgent:
    """Finds and extracts information."""

    responsibilities = [
        "Search for relevant information",
        "Extract key facts and data",
        "Evaluate source credibility",
        "Cite sources properly"
    ]

    capabilities = [
        "Web search",
        "Database queries",
        "Document parsing",
        "Information extraction"
    ]

    tools = [
        "web_search",
        "academic_search",
        "database_query",
        "document_reader",
        "citation_formatter"
    ]
```

### 3. Analysis/Processing Role

```python
class AnalystAgent:
    """Analyzes data and generates insights."""

    responsibilities = [
        "Process and clean data",
        "Perform statistical analysis",
        "Identify patterns and trends",
        "Generate insights"
    ]

    capabilities = [
        "Statistical analysis",
        "Data visualization",
        "Pattern recognition",
        "Predictive modeling"
    ]

    tools = [
        "data_processor",
        "statistical_test",
        "visualization",
        "model_trainer"
    ]
```

### 4. Creation/Generation Role

```python
class CreatorAgent:
    """Creates content, code, or artifacts."""

    responsibilities = [
        "Generate content based on inputs",
        "Follow style and format guidelines",
        "Incorporate feedback",
        "Ensure quality and consistency"
    ]

    capabilities = [
        "Content generation",
        "Code writing",
        "Design creation",
        "Format conversion"
    ]

    tools = [
        "text_generator",
        "code_generator",
        "image_generator",
        "formatter"
    ]
```

### 5. Review/Quality Assurance Role

```python
class ReviewerAgent:
    """Reviews and validates outputs."""

    responsibilities = [
        "Check for errors and issues",
        "Validate against requirements",
        "Provide constructive feedback",
        "Approve or reject outputs"
    ]

    capabilities = [
        "Error detection",
        "Quality assessment",
        "Compliance checking",
        "Feedback generation"
    ]

    tools = [
        "linter",
        "validator",
        "spell_checker",
        "style_checker"
    ]
```

## Role Evolution

Roles can evolve over time:

```python
class EvolvingRole:
    """Role that adapts based on experience."""

    def __init__(self, role_spec: AgentRole):
        self.role = role_spec
        self.performance_history = []
        self.capability_scores = {}

    async def update_from_feedback(self, task: Dict, result: Any,
                                   feedback: Dict):
        """Learn from task performance."""

        # Record performance
        self.performance_history.append({
            "task": task,
            "result": result,
            "feedback": feedback,
            "timestamp": time.time()
        })

        # Update capability scores
        capability = task["requires"]
        score = feedback["quality_score"]

        if capability not in self.capability_scores:
            self.capability_scores[capability] = []

        self.capability_scores[capability].append(score)

        # Adjust role if needed
        await self.adjust_role()

    async def adjust_role(self):
        """Adjust role based on performance."""

        # Find strong capabilities
        strong_capabilities = [
            cap for cap, scores in self.capability_scores.items()
            if sum(scores) / len(scores) > 0.8
        ]

        # Find weak capabilities
        weak_capabilities = [
            cap for cap, scores in self.capability_scores.items()
            if sum(scores) / len(scores) < 0.5
        ]

        # Strengthen role in strong areas
        for cap in strong_capabilities:
            if cap not in self.role.primary_responsibilities:
                print(f"Promoting {cap} to primary responsibility")
                self.role.primary_responsibilities.append(cap)

        # Reduce role in weak areas
        for cap in weak_capabilities:
            if cap in self.role.primary_responsibilities:
                print(f"Demoting {cap} to secondary responsibility")
                self.role.primary_responsibilities.remove(cap)
                self.role.secondary_responsibilities.append(cap)

# Roles adapt to agent performance
# System optimizes over time
```

## Summary

Agent roles and specialization are fundamental to effective multi-agent systems:

**Key Principles:**

- **Specialize for quality**: Expert agents outperform generalists in specific domains
- **Define clear boundaries**: Prevent overlap and confusion
- **Assign responsibilities systematically**: Use frameworks like RACI
- **Choose appropriate specialization level**: Balance expertise vs flexibility
- **Design roles carefully**: Use role definition templates and validation

**Role Design Process:**

1. Decompose system into natural functions/domains
2. Define responsibilities for each role
3. Specify capabilities and tools needed
4. Set clear boundaries (in/out of scope)
5. Define interactions with other roles
6. Establish success criteria and metrics

**Common Patterns:**

- Coordinator/Manager for orchestration
- Researchers for information gathering
- Analysts for data processing
- Creators for content/code generation
- Reviewers for quality assurance

**Trade-offs:**

- Specialists: Higher quality, less flexible
- Generalists: More flexible, lower quality
- Hybrid: Combine both approaches

Well-designed roles enable agents to excel in their domains while collaborating effectively to solve complex problems.

## Next Steps

- [Communication Patterns](communication.md) - How agents exchange information
- [Coordination and Task Allocation](coordination.md) - Organizing agent work
- [Delegation and Hierarchies](delegation.md) - Hierarchical organization
- [Collaboration Patterns](collaboration.md) - Cooperative problem-solving
- [Multi-Agent Fundamentals](fundamentals.md) - Core concepts and principles
