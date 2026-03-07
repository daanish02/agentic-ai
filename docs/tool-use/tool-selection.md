# Tool Selection and Reasoning

## Table of Contents

- [Introduction](#introduction)
- [The Tool Selection Problem](#the-tool-selection-problem)
- [How Agents Choose Tools](#how-agents-choose-tools)
- [Reasoning About Capabilities](#reasoning-about-capabilities)
- [Matching Tools to Tasks](#matching-tools-to-tasks)
- [Handling Ambiguity](#handling-ambiguity)
- [Multiple Candidate Tools](#multiple-candidate-tools)
- [Strategies for Many Tools](#strategies-for-many-tools)
- [Context-Based Selection](#context-based-selection)
- [Learning from Selection History](#learning-from-selection-history)
- [Tool Selection Confidence](#tool-selection-confidence)
- [Fallback Strategies](#fallback-strategies)
- [Tool Combinations](#tool-combinations)
- [Performance-Based Selection](#performance-based-selection)
- [Cost-Aware Selection](#cost-aware-selection)
- [Testing Tool Selection](#testing-tool-selection)
- [Common Patterns](#common-patterns)
- [Best Practices](#best-practices)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Tool selection is the decision-making process where an agent chooses which tool to use for a given task. It's not just about finding **a** tool that works--it's about finding the **best** tool for the specific context.

Poor tool selection leads to:

- Using inefficient tools when better options exist
- Applying tools to inappropriate tasks
- Getting stuck when unable to decide between options
- Wasting resources on trial-and-error

> "The right tool makes the job easy; the wrong tool makes it impossible."

This guide covers tool selection from basic decision-making to advanced strategies for handling dozens or hundreds of tools, from simple pattern matching to sophisticated reasoning about capabilities and constraints.

## The Tool Selection Problem

Understanding the challenges of choosing tools.

### The Decision Space

```
User Request
     |
     ▼
┌─────────────────────────────────┐
│  Available Tools                │
│  - search_web                   │
│  - search_documents             │
│  - search_papers                │
│  - query_database               │
│  - get_weather                  │
│  - send_email                   │
│  ... 50 more tools              │
└─────────────────────────────────┘
     |
     ▼
┌─────────────────────────────────┐
│  Which tool should we use?      │
│  - Multiple might work          │
│  - Some are better than others  │
│  - Some might fail              │
│  - Cost/time trade-offs         │
└─────────────────────────────────┘
```

### Selection Challenges

**1. Ambiguity**

```python
# User request is ambiguous
user_input = "Find information about neural networks"

# Multiple tools could work:
# - search_web: general internet search
# - search_papers: academic papers
# - search_documents: internal docs
# - query_database: structured data

# Which is best?
```

**2. Similar Tools**

```python
# Tools with overlapping capabilities
tools = [
    "search_web",  # General web search
    "search_news",  # News articles only
    "search_wikipedia",  # Wikipedia only
    "search_scholar"  # Academic papers
]

# For "Find information about Python", which search tool?
```

**3. Capability Uncertainty**

```python
# Unclear if tool can handle the request
user_input = "Find papers from last week"

# Questions:
# - Does search_papers support date filtering?
# - How recent is "last week"?
# - Is this data even available?
```

**4. Scale**

```python
# Too many tools to consider
available_tools = load_all_tools()  # Returns 200+ tools

# Evaluating all tools is:
# - Slow (increases latency)
# - Expensive (more tokens)
# - Error-prone (more options = more confusion)
```

## How Agents Choose Tools

The mechanics of tool selection.

### Basic Selection Flow

```python
def select_tool(
    user_input: str,
    available_tools: List[Dict[str, Any]]
) -> str:
    """
    Basic tool selection.
    """
    # 1. Understand user intent
    intent = parse_intent(user_input)

    # 2. Filter relevant tools
    relevant = filter_by_keywords(available_tools, intent)

    # 3. Rank by relevance
    ranked = rank_tools(relevant, user_input)

    # 4. Select top tool
    if ranked:
        return ranked[0]["name"]

    return None
```

### LLM-Based Selection

```python
def llm_select_tool(
    user_input: str,
    tools: List[Dict[str, Any]]
) -> str:
    """
    Use LLM to select tool.
    """
    tool_descriptions = format_tools(tools)

    prompt = f"""
Given this user request:
"{user_input}"

And these available tools:
{tool_descriptions}

Which tool is most appropriate? Think step by step:

1. What is the user trying to accomplish?
2. Which tools could potentially help?
3. Which tool is the best fit?
4. Why is it better than alternatives?

Output only the tool name.
"""

    response = llm(prompt)
    return extract_tool_name(response)


def format_tools(tools: List[Dict[str, Any]]) -> str:
    """Format tools for prompt."""
    lines = []
    for tool in tools:
        lines.append(f"- {tool['name']}: {tool['description']}")
    return "\n".join(lines)
```

### Reasoning-Based Selection

```python
def reasoning_based_selection(
    user_input: str,
    tools: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Select tool with explicit reasoning.
    """
    prompt = f"""
User request: "{user_input}"

Available tools:
{format_tools_detailed(tools)}

Analyze each tool:

For each potentially relevant tool:
1. Can it accomplish the goal? (yes/no/maybe)
2. What are the pros of using this tool?
3. What are the cons?
4. What's the expected outcome?

Then select the best tool and explain why.

Format:
ANALYSIS:
[Your analysis of each tool]

SELECTION: tool_name
REASON: [Why this tool is best]
CONFIDENCE: [low/medium/high]
"""

    response = llm(prompt)

    return {
        "tool": extract_tool_name(response),
        "reasoning": extract_reasoning(response),
        "confidence": extract_confidence(response)
    }
```

## Reasoning About Capabilities

Understanding what tools can and cannot do.

### Capability Matching

```python
from typing import Set, List, Dict, Any

class ToolCapability:
    """Represent a tool's capabilities."""

    def __init__(self, tool_schema: Dict[str, Any]):
        self.name = tool_schema["name"]
        self.description = tool_schema["description"]
        self.capabilities = self._extract_capabilities(tool_schema)
        self.limitations = self._extract_limitations(tool_schema)

    def _extract_capabilities(self, schema: Dict[str, Any]) -> Set[str]:
        """Extract what the tool can do."""
        capabilities = set()

        # Parse from description
        desc = schema.get("description", "").lower()

        # Action verbs
        if "search" in desc:
            capabilities.add("search")
        if "create" in desc or "add" in desc:
            capabilities.add("create")
        if "update" in desc or "modify" in desc:
            capabilities.add("update")
        if "delete" in desc or "remove" in desc:
            capabilities.add("delete")
        if "read" in desc or "get" in desc or "fetch" in desc:
            capabilities.add("read")

        # Data types
        if "email" in desc:
            capabilities.add("email")
        if "document" in desc or "file" in desc:
            capabilities.add("documents")
        if "database" in desc:
            capabilities.add("database")

        return capabilities

    def _extract_limitations(self, schema: Dict[str, Any]) -> List[str]:
        """Extract tool limitations."""
        limitations = []

        desc = schema.get("description", "")

        # Look for limitation keywords
        limitation_markers = [
            "cannot", "does not", "limited to", "only",
            "maximum", "up to", "restricted"
        ]

        for marker in limitation_markers:
            if marker in desc.lower():
                # Extract sentence containing limitation
                sentences = desc.split('.')
                for sentence in sentences:
                    if marker in sentence.lower():
                        limitations.append(sentence.strip())

        return limitations

    def can_handle(self, task_requirements: Set[str]) -> bool:
        """Check if tool can handle task requirements."""
        return task_requirements.issubset(self.capabilities)

    def fitness_score(self, task_requirements: Set[str]) -> float:
        """
        Score how well tool fits task (0.0 to 1.0).
        """
        if not task_requirements:
            return 0.0

        # How many requirements can we satisfy?
        satisfied = len(task_requirements & self.capabilities)
        total = len(task_requirements)

        return satisfied / total


def select_by_capabilities(
    task_requirements: Set[str],
    tools: List[Dict[str, Any]]
) -> List[str]:
    """
    Select tools based on capability matching.
    """
    tool_capabilities = [ToolCapability(tool) for tool in tools]

    # Score each tool
    scored = []
    for tc in tool_capabilities:
        score = tc.fitness_score(task_requirements)
        if score > 0:
            scored.append((score, tc.name))

    # Sort by score
    scored.sort(reverse=True)

    return [name for score, name in scored]


# Example usage
task = {"search", "documents", "internal"}

tools = [
    {
        "name": "search_web",
        "description": "Search the internet for information"
    },
    {
        "name": "search_documents",
        "description": "Search internal documents and files"
    },
    {
        "name": "query_database",
        "description": "Query database tables"
    }
]

best_tools = select_by_capabilities(task, tools)
# Returns: ["search_documents", ...] - best match for internal document search
```

### Constraint Checking

```python
def check_constraints(
    tool: Dict[str, Any],
    context: Dict[str, Any]
) -> tuple[bool, List[str]]:
    """
    Check if tool constraints are satisfied.
    """
    issues = []

    # Check permissions
    required_permission = tool.get("required_permission")
    if required_permission:
        user_permissions = context.get("permissions", [])
        if required_permission not in user_permissions:
            issues.append(f"Missing permission: {required_permission}")

    # Check integrations
    required_integration = tool.get("requires_integration")
    if required_integration:
        available = context.get("integrations", [])
        if required_integration not in available:
            issues.append(f"Integration not enabled: {required_integration}")

    # Check rate limits
    if "rate_limit" in tool:
        usage = context.get("tool_usage", {}).get(tool["name"], 0)
        limit = tool["rate_limit"]
        if usage >= limit:
            issues.append(f"Rate limit exceeded: {usage}/{limit}")

    # Check prerequisites
    prerequisites = tool.get("prerequisites", [])
    for prereq in prerequisites:
        if not check_prerequisite(prereq, context):
            issues.append(f"Prerequisite not met: {prereq}")

    can_use = len(issues) == 0
    return can_use, issues


def filter_by_constraints(
    tools: List[Dict[str, Any]],
    context: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Filter tools by constraint satisfaction.
    """
    usable_tools = []

    for tool in tools:
        can_use, issues = check_constraints(tool, context)
        if can_use:
            usable_tools.append(tool)
        else:
            logger.debug(f"Tool {tool['name']} not usable: {issues}")

    return usable_tools
```

## Matching Tools to Tasks

Strategies for matching tools to user requests.

### Keyword Matching

```python
def keyword_match_score(
    user_input: str,
    tool: Dict[str, Any]
) -> float:
    """
    Score tool based on keyword overlap.
    """
    # Extract keywords from user input
    user_keywords = set(user_input.lower().split())

    # Extract keywords from tool
    tool_text = (
        tool["name"] + " " +
        tool.get("description", "") + " " +
        " ".join(tool.get("tags", []))
    ).lower()

    tool_keywords = set(tool_text.split())

    # Calculate overlap
    overlap = user_keywords & tool_keywords

    if not user_keywords:
        return 0.0

    return len(overlap) / len(user_keywords)
```

### Semantic Matching

```python
def semantic_match_score(
    user_input: str,
    tool: Dict[str, Any],
    embeddings_model
) -> float:
    """
    Score tool using semantic similarity.
    """
    # Get embeddings
    user_embedding = embeddings_model.embed(user_input)

    tool_text = f"{tool['name']}: {tool.get('description', '')}"
    tool_embedding = embeddings_model.embed(tool_text)

    # Cosine similarity
    similarity = cosine_similarity(user_embedding, tool_embedding)

    return similarity


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between vectors."""
    import numpy as np

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    return dot_product / (norm1 * norm2)
```

### Intent-Based Matching

```python
from enum import Enum

class Intent(Enum):
    """User intent categories."""
    SEARCH = "search"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    READ = "read"
    ANALYZE = "analyze"
    COMMUNICATE = "communicate"


def classify_intent(user_input: str) -> Intent:
    """Classify user intent."""
    user_lower = user_input.lower()

    # Search intent
    if any(word in user_lower for word in ["find", "search", "look for", "show me"]):
        return Intent.SEARCH

    # Create intent
    if any(word in user_lower for word in ["create", "make", "add", "new"]):
        return Intent.CREATE

    # Update intent
    if any(word in user_lower for word in ["update", "change", "modify", "edit"]):
        return Intent.UPDATE

    # Delete intent
    if any(word in user_lower for word in ["delete", "remove", "cancel"]):
        return Intent.DELETE

    # Read intent
    if any(word in user_lower for word in ["read", "get", "show", "display"]):
        return Intent.READ

    # Communicate intent
    if any(word in user_lower for word in ["send", "email", "message", "notify"]):
        return Intent.COMMUNICATE

    return Intent.READ  # Default


def match_by_intent(
    intent: Intent,
    tools: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Filter tools matching intent.
    """
    matched = []

    for tool in tools:
        tool_name = tool["name"].lower()
        tool_desc = tool.get("description", "").lower()

        intent_match = False

        if intent == Intent.SEARCH:
            intent_match = "search" in tool_name or "find" in tool_name
        elif intent == Intent.CREATE:
            intent_match = "create" in tool_name or "add" in tool_name
        elif intent == Intent.UPDATE:
            intent_match = "update" in tool_name or "modify" in tool_name
        elif intent == Intent.DELETE:
            intent_match = "delete" in tool_name or "remove" in tool_name
        elif intent == Intent.COMMUNICATE:
            intent_match = any(w in tool_name for w in ["send", "email", "message"])

        if intent_match:
            matched.append(tool)

    return matched
```

### Example-Based Matching

```python
def find_similar_examples(
    user_input: str,
    tool: Dict[str, Any]
) -> float:
    """
    Score tool based on similarity to its examples.
    """
    examples = tool.get("examples", [])

    if not examples:
        return 0.0

    max_similarity = 0.0

    for example in examples:
        # Get example input
        example_input = example.get("user_input", "")

        if not example_input:
            continue

        # Calculate similarity
        similarity = calculate_text_similarity(user_input, example_input)
        max_similarity = max(max_similarity, similarity)

    return max_similarity


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts.
    Simple word overlap for this example.
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union)  # Jaccard similarity
```

## Handling Ambiguity

Strategies for dealing with unclear requests.

### Disambiguation Prompts

```python
def disambiguate_request(
    user_input: str,
    candidate_tools: List[Dict[str, Any]]
) -> str:
    """
    Ask LLM to disambiguate between candidate tools.
    """
    prompt = f"""
User request: "{user_input}"

This request is ambiguous. Multiple tools could potentially help:

{format_tools_with_details(candidate_tools)}

Questions to consider:
1. What is the user's primary goal?
2. What type of information are they looking for?
3. Where should we look for this information?
4. What output format do they expect?

Based on these considerations, which tool is most likely what the user wants?

Output format:
TOOL: tool_name
REASONING: [Why this tool is the best match]
ALTERNATIVES: [What other tools could work and when to use them instead]
"""

    response = llm(prompt)
    return extract_tool_name(response)
```

### Clarification Questions

```python
def ask_clarification(
    user_input: str,
    ambiguous_aspect: str,
    options: List[str]
) -> str:
    """
    Generate clarification question for user.
    """
    questions = {
        "scope": f"Where would you like to search? Options: {', '.join(options)}",
        "output": f"What kind of output do you need? Options: {', '.join(options)}",
        "action": f"What would you like me to do? Options: {', '.join(options)}"
    }

    return questions.get(ambiguous_aspect, "Could you please clarify your request?")


def handle_ambiguity(
    user_input: str,
    tools: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Handle ambiguous requests.
    """
    # Find candidate tools
    candidates = find_candidate_tools(user_input, tools)

    if len(candidates) == 0:
        return {
            "action": "no_tools",
            "message": "I don't have a tool for that request."
        }

    if len(candidates) == 1:
        return {
            "action": "use_tool",
            "tool": candidates[0]["name"]
        }

    # Multiple candidates - need to disambiguate
    if len(candidates) <= 3:
        # Few options - ask user directly
        return {
            "action": "clarify",
            "message": ask_clarification(
                user_input,
                "scope",
                [c["name"] for c in candidates]
            ),
            "options": candidates
        }
    else:
        # Many options - use LLM to decide
        selected = disambiguate_request(user_input, candidates)
        return {
            "action": "use_tool",
            "tool": selected
        }
```

### Confidence Thresholds

```python
def select_with_confidence(
    user_input: str,
    tools: List[Dict[str, Any]],
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Only select tool if confidence is high enough.
    """
    # Score all tools
    scored_tools = []
    for tool in tools:
        score = calculate_match_score(user_input, tool)
        scored_tools.append((score, tool))

    scored_tools.sort(reverse=True, key=lambda x: x[0])

    if not scored_tools:
        return {"action": "no_match", "confidence": 0.0}

    top_score, top_tool = scored_tools[0]

    if top_score >= confidence_threshold:
        return {
            "action": "use_tool",
            "tool": top_tool["name"],
            "confidence": top_score
        }
    else:
        # Not confident enough
        if len(scored_tools) > 1:
            second_score = scored_tools[1][0]
            if top_score - second_score < 0.1:
                # Two tools are very close in score
                return {
                    "action": "ambiguous",
                    "candidates": [t for s, t in scored_tools[:3]],
                    "confidence": top_score
                }

        return {
            "action": "uncertain",
            "best_guess": top_tool["name"],
            "confidence": top_score
        }
```

## Multiple Candidate Tools

Handling situations with many possible tools.

### Scoring System

```python
from typing import NamedTuple

class ToolScore(NamedTuple):
    """Score components for a tool."""
    relevance: float  # How relevant to task (0-1)
    capability: float  # Can it do the job? (0-1)
    efficiency: float  # How efficient? (0-1)
    reliability: float  # How reliable? (0-1)
    cost: float  # Cost factor (0-1, lower is better)

    def total(self, weights: Dict[str, float] = None) -> float:
        """Calculate weighted total score."""
        if weights is None:
            weights = {
                "relevance": 0.3,
                "capability": 0.3,
                "efficiency": 0.2,
                "reliability": 0.15,
                "cost": 0.05
            }

        return (
            self.relevance * weights["relevance"] +
            self.capability * weights["capability"] +
            self.efficiency * weights["efficiency"] +
            self.reliability * weights["reliability"] +
            (1 - self.cost) * weights["cost"]  # Lower cost is better
        )


def score_tool(
    tool: Dict[str, Any],
    user_input: str,
    context: Dict[str, Any]
) -> ToolScore:
    """
    Comprehensively score a tool.
    """
    # Relevance: keyword/semantic matching
    relevance = calculate_relevance(user_input, tool)

    # Capability: can it do the job?
    capability = calculate_capability_match(user_input, tool)

    # Efficiency: expected performance
    efficiency = estimate_efficiency(tool, context)

    # Reliability: historical success rate
    reliability = get_reliability_score(tool, context)

    # Cost: resource usage
    cost = estimate_cost(tool, context)

    return ToolScore(relevance, capability, efficiency, reliability, cost)


def select_best_tool(
    user_input: str,
    tools: List[Dict[str, Any]],
    context: Dict[str, Any],
    weights: Dict[str, float] = None
) -> str:
    """
    Select best tool using comprehensive scoring.
    """
    scored_tools = []

    for tool in tools:
        score = score_tool(tool, user_input, context)
        total = score.total(weights)
        scored_tools.append((total, tool, score))

    # Sort by total score
    scored_tools.sort(reverse=True, key=lambda x: x[0])

    # Return best tool
    if scored_tools:
        return scored_tools[0][1]["name"]

    return None
```

### Ranking Strategies

```python
def rank_tools_multi_criteria(
    tools: List[Dict[str, Any]],
    user_input: str,
    context: Dict[str, Any]
) -> List[tuple[Dict[str, Any], Dict[str, float]]]:
    """
    Rank tools considering multiple criteria.
    """
    rankings = []

    for tool in tools:
        scores = {
            "keyword_match": keyword_match_score(user_input, tool),
            "semantic_match": semantic_match_score(user_input, tool),
            "capability_fit": capability_fitness(tool, user_input),
            "past_success": historical_success_rate(tool, context),
            "execution_speed": speed_score(tool),
            "cost_efficiency": cost_score(tool)
        }

        # Calculate weighted average
        weights = {
            "keyword_match": 0.15,
            "semantic_match": 0.25,
            "capability_fit": 0.30,
            "past_success": 0.15,
            "execution_speed": 0.10,
            "cost_efficiency": 0.05
        }

        total_score = sum(
            scores[criterion] * weights[criterion]
            for criterion in scores
        )

        rankings.append((tool, scores, total_score))

    # Sort by total score
    rankings.sort(reverse=True, key=lambda x: x[2])

    return [(tool, scores) for tool, scores, _ in rankings]
```

### Top-K Selection

```python
def select_top_k_tools(
    user_input: str,
    tools: List[Dict[str, Any]],
    k: int = 3
) -> List[str]:
    """
    Select top K tools for potential use.
    """
    # Score all tools
    scored = []
    for tool in tools:
        score = calculate_match_score(user_input, tool)
        scored.append((score, tool))

    # Sort and take top K
    scored.sort(reverse=True)

    return [tool["name"] for score, tool in scored[:k]]


def try_tools_sequentially(
    user_input: str,
    tools: List[Dict[str, Any]],
    max_attempts: int = 3
) -> Any:
    """
    Try top tools sequentially until one succeeds.
    """
    top_tools = select_top_k_tools(user_input, tools, k=max_attempts)

    for tool_name in top_tools:
        try:
            # Extract parameters and execute
            params = extract_parameters(user_input, tool_name)
            result = execute_tool(tool_name, params)

            if is_successful(result):
                return result
        except Exception as e:
            logger.warning(f"Tool {tool_name} failed: {e}")
            continue

    raise Exception("All candidate tools failed")
```

## Strategies for Many Tools

Managing large tool catalogs.

### Hierarchical Filtering

```python
def hierarchical_tool_selection(
    user_input: str,
    all_tools: List[Dict[str, Any]]
) -> str:
    """
    Select tool using hierarchical filtering.
    """
    # Level 1: Filter by category
    intent = classify_intent(user_input)
    category = intent_to_category(intent)

    category_tools = [
        t for t in all_tools
        if t.get("category") == category
    ]

    logger.info(f"Filtered to {len(category_tools)} tools in category '{category}'")

    # Level 2: Filter by tags
    user_tags = extract_tags(user_input)
    tagged_tools = [
        t for t in category_tools
        if any(tag in t.get("tags", []) for tag in user_tags)
    ]

    if tagged_tools:
        tools_to_rank = tagged_tools
    else:
        tools_to_rank = category_tools

    logger.info(f"Filtered to {len(tools_to_rank)} tools by tags")

    # Level 3: Rank remaining tools
    ranked = rank_tools(tools_to_rank, user_input)

    # Return best tool
    return ranked[0]["name"] if ranked else None


def intent_to_category(intent: Intent) -> str:
    """Map intent to tool category."""
    mapping = {
        Intent.SEARCH: "search",
        Intent.CREATE: "creation",
        Intent.UPDATE: "modification",
        Intent.DELETE: "modification",
        Intent.READ: "retrieval",
        Intent.COMMUNICATE: "communication"
    }
    return mapping.get(intent, "general")
```

### Two-Stage Selection

```python
def two_stage_selection(
    user_input: str,
    all_tools: List[Dict[str, Any]]
) -> str:
    """
    Two-stage selection: coarse filtering then fine ranking.
    """
    # Stage 1: Fast, coarse filtering
    # Use simple keyword matching to quickly filter
    candidates = []

    user_keywords = set(user_input.lower().split())

    for tool in all_tools:
        # Quick check: any keyword overlap?
        tool_text = (tool["name"] + " " + tool.get("description", "")).lower()
        tool_words = set(tool_text.split())

        overlap = user_keywords & tool_words

        if overlap:
            candidates.append(tool)

    logger.info(f"Stage 1: Filtered to {len(candidates)} candidates")

    # Stage 2: Detailed ranking of candidates
    # Use expensive methods only on filtered set
    if len(candidates) <= 10:
        # Few candidates - use comprehensive scoring
        ranked = rank_tools_multi_criteria(candidates, user_input, {})
        return ranked[0][0]["name"] if ranked else None
    else:
        # Still many candidates - use LLM
        return llm_select_tool(user_input, candidates[:20])
```

### Tool Clustering

```python
from typing import List, Dict, Set

class ToolCluster:
    """Group of related tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.tools: List[str] = []

    def add_tool(self, tool_name: str):
        """Add tool to cluster."""
        self.tools.append(tool_name)


def create_tool_clusters(
    tools: List[Dict[str, Any]]
) -> List[ToolCluster]:
    """
    Organize tools into clusters.
    """
    clusters = {
        "web_search": ToolCluster(
            "web_search",
            "Tools for searching the internet"
        ),
        "document_search": ToolCluster(
            "document_search",
            "Tools for searching documents"
        ),
        "data_access": ToolCluster(
            "data_access",
            "Tools for accessing data"
        ),
        "communication": ToolCluster(
            "communication",
            "Tools for communication"
        ),
        "computation": ToolCluster(
            "computation",
            "Tools for calculations"
        )
    }

    # Assign tools to clusters
    for tool in tools:
        name = tool["name"]
        desc = tool.get("description", "").lower()

        if "web" in desc or "internet" in desc:
            clusters["web_search"].add_tool(name)
        elif "document" in desc or "file" in desc:
            clusters["document_search"].add_tool(name)
        elif "database" in desc or "query" in desc:
            clusters["data_access"].add_tool(name)
        elif "email" in desc or "message" in desc:
            clusters["communication"].add_tool(name)
        elif "calculate" in desc or "compute" in desc:
            clusters["computation"].add_tool(name)

    return list(clusters.values())


def cluster_based_selection(
    user_input: str,
    tool_clusters: List[ToolCluster],
    all_tools: Dict[str, Dict[str, Any]]
) -> str:
    """
    Select tool by first choosing cluster.
    """
    # Step 1: Choose cluster
    cluster_scores = []

    for cluster in tool_clusters:
        score = keyword_match_score(
            user_input,
            {"name": cluster.name, "description": cluster.description}
        )
        cluster_scores.append((score, cluster))

    cluster_scores.sort(reverse=True)
    best_cluster = cluster_scores[0][1]

    logger.info(f"Selected cluster: {best_cluster.name}")

    # Step 2: Choose tool within cluster
    cluster_tools = [all_tools[name] for name in best_cluster.tools]
    return select_best_tool(user_input, cluster_tools, {})
```

### Adaptive Filtering

```python
class AdaptiveToolFilter:
    """
    Adaptively filter tools based on context and history.
    """

    def __init__(self):
        self.tool_usage_stats: Dict[str, int] = {}
        self.tool_success_rate: Dict[str, float] = {}

    def filter_tools(
        self,
        user_input: str,
        all_tools: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Adaptively filter tools.
        """
        # Start with all tools
        candidates = all_tools.copy()

        # Filter 1: Remove tools user doesn't have access to
        candidates = [
            t for t in candidates
            if self._has_access(t, context)
        ]

        # Filter 2: Prefer previously successful tools
        if len(candidates) > 50:
            # Prioritize tools with good success rate
            candidates = self._prioritize_successful(candidates)

        # Filter 3: Keyword filtering
        if len(candidates) > 20:
            candidates = self._keyword_filter(candidates, user_input)

        # Filter 4: Semantic filtering if still too many
        if len(candidates) > 10:
            candidates = self._semantic_filter(candidates, user_input)

        return candidates

    def _prioritize_successful(
        self,
        tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prioritize tools with good track record."""
        scored = []

        for tool in tools:
            name = tool["name"]
            success_rate = self.tool_success_rate.get(name, 0.5)
            usage_count = self.tool_usage_stats.get(name, 0)

            # Prefer tools that are both successful and used
            score = success_rate * (1 + min(usage_count, 10) / 10)

            scored.append((score, tool))

        scored.sort(reverse=True)

        # Return top 50 or all if fewer
        return [tool for score, tool in scored[:50]]

    def _keyword_filter(
        self,
        tools: List[Dict[str, Any]],
        user_input: str
    ) -> List[Dict[str, Any]]:
        """Filter by keyword matching."""
        user_keywords = set(user_input.lower().split())

        filtered = []
        for tool in tools:
            tool_text = (
                tool["name"] + " " +
                tool.get("description", "")
            ).lower()

            tool_words = set(tool_text.split())
            overlap = user_keywords & tool_words

            if overlap:
                filtered.append(tool)

        return filtered if filtered else tools

    def _semantic_filter(
        self,
        tools: List[Dict[str, Any]],
        user_input: str
    ) -> List[Dict[str, Any]]:
        """Filter by semantic similarity."""
        # This would use embeddings in practice
        # Simplified version:
        scored = []

        for tool in tools:
            score = calculate_match_score(user_input, tool)
            scored.append((score, tool))

        scored.sort(reverse=True)

        return [tool for score, tool in scored[:10]]
```

## Context-Based Selection

Using context to improve tool selection.

### User Context

```python
def context_aware_selection(
    user_input: str,
    tools: List[Dict[str, Any]],
    user_context: Dict[str, Any]
) -> str:
    """
    Select tool considering user context.
    """
    # Consider user's role
    user_role = user_context.get("role", "user")

    # Consider user's recent activity
    recent_tools = user_context.get("recent_tools", [])

    # Consider user's preferences
    preferred_tools = user_context.get("preferred_tools", [])

    # Score tools
    scored = []

    for tool in tools:
        score = calculate_match_score(user_input, tool)

        # Boost score for preferred tools
        if tool["name"] in preferred_tools:
            score *= 1.2

        # Boost for recently used tools (context continuity)
        if tool["name"] in recent_tools[-3:]:
            score *= 1.1

        # Adjust for role-specific tools
        if tool.get("recommended_for") == user_role:
            score *= 1.15

        scored.append((score, tool))

    scored.sort(reverse=True)

    return scored[0][1]["name"] if scored else None
```

### Conversation Context

```python
def conversation_aware_selection(
    user_input: str,
    tools: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]]
) -> str:
    """
    Select tool based on conversation context.
    """
    # Extract context from conversation
    recent_topics = extract_topics(conversation_history)
    recent_entities = extract_entities(conversation_history)

    # Find tools relevant to conversation topics
    scored = []

    for tool in tools:
        base_score = calculate_match_score(user_input, tool)

        # Boost if tool relates to recent topics
        tool_topics = extract_topics_from_description(tool)
        topic_overlap = len(set(recent_topics) & set(tool_topics))
        topic_boost = 1.0 + (topic_overlap * 0.1)

        # Boost if tool can handle mentioned entities
        entity_boost = 1.0
        if tool_can_handle_entities(tool, recent_entities):
            entity_boost = 1.15

        final_score = base_score * topic_boost * entity_boost

        scored.append((final_score, tool))

    scored.sort(reverse=True)

    return scored[0][1]["name"] if scored else None
```

### Temporal Context

```python
def time_aware_selection(
    user_input: str,
    tools: List[Dict[str, Any]],
    current_time: datetime
) -> str:
    """
    Select tool considering temporal context.
    """
    from datetime import datetime

    # Time of day might affect tool choice
    hour = current_time.hour

    scored = []

    for tool in tools:
        score = calculate_match_score(user_input, tool)

        # Some tools are better at certain times
        if tool.get("time_sensitive"):
            # Example: don't send emails late at night
            if tool["name"] == "send_email" and (hour < 6 or hour > 22):
                score *= 0.5

        # Consider data freshness requirements
        if "real-time" in user_input.lower():
            if tool.get("data_freshness") == "real-time":
                score *= 1.3
            elif tool.get("data_freshness") == "cached":
                score *= 0.7

        scored.append((score, tool))

    scored.sort(reverse=True)

    return scored[0][1]["name"] if scored else None
```

## Learning from Selection History

Improving selection over time.

### Success Tracking

```python
class ToolSelectionLearner:
    """
    Learn from tool selection outcomes.
    """

    def __init__(self):
        self.selection_history: List[Dict[str, Any]] = []
        self.tool_stats: Dict[str, Dict[str, Any]] = {}

    def record_selection(
        self,
        user_input: str,
        selected_tool: str,
        success: bool,
        execution_time: float
    ):
        """Record a tool selection outcome."""
        record = {
            "timestamp": datetime.now(),
            "user_input": user_input,
            "tool": selected_tool,
            "success": success,
            "execution_time": execution_time
        }

        self.selection_history.append(record)

        # Update stats
        if selected_tool not in self.tool_stats:
            self.tool_stats[selected_tool] = {
                "total_uses": 0,
                "successes": 0,
                "total_time": 0.0,
                "patterns": []
            }

        stats = self.tool_stats[selected_tool]
        stats["total_uses"] += 1
        if success:
            stats["successes"] += 1
        stats["total_time"] += execution_time

    def get_success_rate(self, tool_name: str) -> float:
        """Get success rate for a tool."""
        stats = self.tool_stats.get(tool_name)

        if not stats or stats["total_uses"] == 0:
            return 0.5  # Unknown - assume 50%

        return stats["successes"] / stats["total_uses"]

    def get_avg_execution_time(self, tool_name: str) -> float:
        """Get average execution time."""
        stats = self.tool_stats.get(tool_name)

        if not stats or stats["total_uses"] == 0:
            return 0.0

        return stats["total_time"] / stats["total_uses"]

    def find_similar_successful_cases(
        self,
        user_input: str,
        top_k: int = 5
    ) -> List[str]:
        """
        Find tools that worked for similar requests.
        """
        similar_cases = []

        for record in self.selection_history:
            if not record["success"]:
                continue

            similarity = calculate_text_similarity(
                user_input,
                record["user_input"]
            )

            if similarity > 0.3:  # Threshold for "similar"
                similar_cases.append((similarity, record["tool"]))

        similar_cases.sort(reverse=True)

        # Return top K tools
        tools = [tool for sim, tool in similar_cases[:top_k]]

        # Remove duplicates while preserving order
        seen = set()
        unique_tools = []
        for tool in tools:
            if tool not in seen:
                seen.add(tool)
                unique_tools.append(tool)

        return unique_tools


# Usage
learner = ToolSelectionLearner()

# Record outcomes
learner.record_selection(
    "Search for Python tutorials",
    "search_web",
    success=True,
    execution_time=1.2
)

# Use learned information in future selections
def learned_selection(
    user_input: str,
    tools: List[Dict[str, Any]],
    learner: ToolSelectionLearner
) -> str:
    """
    Select tool using learned information.
    """
    # Check if we've seen similar requests before
    similar_tools = learner.find_similar_successful_cases(user_input)

    if similar_tools:
        # Filter to similar successful tools
        candidate_tools = [
            t for t in tools if t["name"] in similar_tools
        ]

        if candidate_tools:
            # Rank by success rate
            scored = []
            for tool in candidate_tools:
                success_rate = learner.get_success_rate(tool["name"])
                scored.append((success_rate, tool))

            scored.sort(reverse=True)
            return scored[0][1]["name"]

    # Fall back to standard selection
    return select_best_tool(user_input, tools, {})
```

## Tool Selection Confidence

Quantifying confidence in tool selection.

### Confidence Calculation

```python
def calculate_selection_confidence(
    user_input: str,
    selected_tool: Dict[str, Any],
    all_tools: List[Dict[str, Any]]
) -> float:
    """
    Calculate confidence in tool selection (0-1).
    """
    # Score selected tool
    selected_score = calculate_match_score(user_input, selected_tool)

    # Score all tools
    all_scores = [
        calculate_match_score(user_input, tool)
        for tool in all_tools
    ]

    all_scores.sort(reverse=True)

    # Confidence factors:
    # 1. How well does selected tool match?
    match_confidence = selected_score

    # 2. How much better is it than alternatives?
    if len(all_scores) > 1:
        second_best = all_scores[1]
        if second_best > 0:
            gap_confidence = (selected_score - second_best) / selected_score
        else:
            gap_confidence = 1.0
    else:
        gap_confidence = 1.0

    # 3. Historical success rate
    historical_confidence = get_reliability_score(selected_tool, {})

    # Combine factors
    confidence = (
        match_confidence * 0.4 +
        gap_confidence * 0.3 +
        historical_confidence * 0.3
    )

    return min(confidence, 1.0)
```

### Confidence-Based Behavior

```python
def act_on_confidence(
    user_input: str,
    selected_tool: str,
    confidence: float
) -> Dict[str, Any]:
    """
    Determine action based on confidence level.
    """
    if confidence >= 0.8:
        # High confidence - proceed directly
        return {
            "action": "execute",
            "tool": selected_tool,
            "confirmation_needed": False
        }

    elif confidence >= 0.5:
        # Medium confidence - explain choice
        return {
            "action": "execute_with_explanation",
            "tool": selected_tool,
            "message": f"I'll use {selected_tool}. Let me know if you'd prefer a different approach.",
            "confirmation_needed": False
        }

    elif confidence >= 0.3:
        # Low confidence - ask for confirmation
        return {
            "action": "request_confirmation",
            "tool": selected_tool,
            "message": f"I think {selected_tool} might work, but I'm not certain. Should I try it?",
            "confirmation_needed": True
        }

    else:
        # Very low confidence - ask for help
        return {
            "action": "request_clarification",
            "message": "I'm not sure which tool to use. Could you provide more details?",
            "confirmation_needed": True
        }
```

## Fallback Strategies

What to do when tool selection fails.

### Sequential Fallbacks

```python
def select_tool_with_fallback(
    user_input: str,
    tools: List[Dict[str, Any]]
) -> str:
    """
    Try multiple selection strategies with fallbacks.
    """
    # Strategy 1: Semantic matching
    try:
        tool = semantic_based_selection(user_input, tools)
        if tool and calculate_selection_confidence(user_input, tool, tools) > 0.7:
            return tool["name"]
    except Exception as e:
        logger.warning(f"Semantic selection failed: {e}")

    # Strategy 2: LLM-based selection
    try:
        tool_name = llm_select_tool(user_input, tools)
        if tool_name:
            return tool_name
    except Exception as e:
        logger.warning(f"LLM selection failed: {e}")

    # Strategy 3: Keyword matching
    try:
        candidates = keyword_filter(user_input, tools)
        if candidates:
            return candidates[0]["name"]
    except Exception as e:
        logger.warning(f"Keyword selection failed: {e}")

    # Strategy 4: Use most popular tool
    return get_most_used_tool(tools)
```

### Graceful Degradation

```python
def select_or_degrade(
    user_input: str,
    tools: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Select tool or gracefully degrade if not possible.
    """
    # Try to select a tool
    selected = select_best_tool(user_input, tools, {})

    if selected:
        confidence = calculate_selection_confidence(
            user_input,
            next(t for t in tools if t["name"] == selected),
            tools
        )

        if confidence > 0.5:
            return {
                "status": "success",
                "tool": selected,
                "confidence": confidence
            }

    # Can't confidently select - try alternatives

    # Alternative 1: Suggest multiple tools
    top_3 = select_top_k_tools(user_input, tools, k=3)
    if len(top_3) > 1:
        return {
            "status": "multiple_options",
            "options": top_3,
            "message": "I found several tools that might help. Which would you like to use?"
        }

    # Alternative 2: Request more information
    return {
        "status": "need_clarification",
        "message": "I need more information to choose the right tool. Could you clarify what you're trying to accomplish?",
        "suggestions": [
            "What type of information are you looking for?",
            "Where should I search?",
            "What output format do you need?"
        ]
    }
```

## Tool Combinations

Selecting multiple complementary tools.

### Complementary Selection

```python
def select_complementary_tools(
    user_input: str,
    tools: List[Dict[str, Any]]
) -> List[str]:
    """
    Select multiple complementary tools for a complex task.
    """
    # Identify task components
    components = decompose_task(user_input)

    selected_tools = []

    for component in components:
        # Select best tool for this component
        component_tools = [
            t for t in tools
            if can_handle_component(t, component)
        ]

        if component_tools:
            best = select_best_tool(component, component_tools, {})
            selected_tools.append(best)

    # Remove duplicates
    return list(set(selected_tools))


def decompose_task(user_input: str) -> List[str]:
    """
    Break complex task into components.
    """
    # Simple decomposition based on keywords
    components = []

    if "search" in user_input.lower():
        components.append("search for information")

    if "analyze" in user_input.lower() or "summary" in user_input.lower():
        components.append("analyze data")

    if "send" in user_input.lower() or "email" in user_input.lower():
        components.append("send communication")

    if not components:
        components = [user_input]

    return components
```

## Performance-Based Selection

Selecting based on expected performance.

### Performance Estimation

```python
def estimate_performance(
    tool: Dict[str, Any],
    user_input: str,
    context: Dict[str, Any]
) -> Dict[str, float]:
    """
    Estimate tool performance metrics.
    """
    # Estimate execution time
    base_time = tool.get("avg_execution_time", 1.0)

    # Adjust based on input complexity
    input_complexity = len(user_input.split())
    time_multiplier = 1.0 + (input_complexity / 100)

    estimated_time = base_time * time_multiplier

    # Estimate success probability
    historical_success = context.get("tool_success_rates", {}).get(
        tool["name"],
        0.8
    )

    # Estimate resource usage
    estimated_cost = tool.get("cost_per_call", 0.01)

    return {
        "execution_time": estimated_time,
        "success_probability": historical_success,
        "estimated_cost": estimated_cost
    }


def select_by_performance(
    user_input: str,
    tools: List[Dict[str, Any]],
    context: Dict[str, Any],
    optimize_for: str = "speed"
) -> str:
    """
    Select tool optimizing for specific performance metric.
    """
    scored = []

    for tool in tools:
        # Get performance estimates
        perf = estimate_performance(tool, user_input, context)

        # Score based on optimization goal
        if optimize_for == "speed":
            score = 1.0 / perf["execution_time"]
        elif optimize_for == "reliability":
            score = perf["success_probability"]
        elif optimize_for == "cost":
            score = 1.0 / (perf["estimated_cost"] + 0.001)
        else:
            # Balanced
            score = (
                (1.0 / perf["execution_time"]) * 0.4 +
                perf["success_probability"] * 0.4 +
                (1.0 / (perf["estimated_cost"] + 0.001)) * 0.2
            )

        scored.append((score, tool))

    scored.sort(reverse=True)

    return scored[0][1]["name"] if scored else None
```

## Cost-Aware Selection

Considering cost in tool selection.

### Cost Optimization

```python
def cost_aware_selection(
    user_input: str,
    tools: List[Dict[str, Any]],
    budget: float
) -> str:
    """
    Select tool considering cost constraints.
    """
    # Filter tools within budget
    affordable = [
        t for t in tools
        if t.get("cost_per_call", 0) <= budget
    ]

    if not affordable:
        # All tools too expensive - return cheapest
        cheapest = min(tools, key=lambda t: t.get("cost_per_call", float('inf')))
        return cheapest["name"]

    # Among affordable, select best
    return select_best_tool(user_input, affordable, {})


def cost_benefit_selection(
    user_input: str,
    tools: List[Dict[str, Any]]
) -> str:
    """
    Select tool with best cost-benefit ratio.
    """
    scored = []

    for tool in tools:
        # Expected value
        quality = calculate_match_score(user_input, tool)
        success_rate = get_reliability_score(tool, {})
        expected_value = quality * success_rate

        # Cost
        cost = tool.get("cost_per_call", 0.01)

        # Cost-benefit ratio
        if cost > 0:
            ratio = expected_value / cost
        else:
            ratio = expected_value * 1000  # Free tool, very high ratio

        scored.append((ratio, tool))

    scored.sort(reverse=True)

    return scored[0][1]["name"] if scored else None
```

## Testing Tool Selection

Ensuring selection works correctly.

### Unit Tests

```python
import pytest

def test_tool_selection_accuracy():
    """Test that selection chooses correct tool."""

    tools = [
        {
            "name": "search_web",
            "description": "Search the internet"
        },
        {
            "name": "search_documents",
            "description": "Search internal documents"
        }
    ]

    # Should select search_web for internet query
    selected = select_best_tool("Find information about Python", tools, {})
    assert selected == "search_web"

    # Should select search_documents for internal query
    selected = select_best_tool("Find our internal Python guidelines", tools, {})
    assert selected == "search_documents"


def test_confidence_calculation():
    """Test confidence is calculated correctly."""

    tools = [
        {"name": "tool_a", "description": "Does A"},
        {"name": "tool_b", "description": "Does B"}
    ]

    # High match should give high confidence
    confidence = calculate_selection_confidence(
        "Do A",
        tools[0],
        tools
    )

    assert confidence > 0.7

    # Low match should give low confidence
    confidence = calculate_selection_confidence(
        "Do C",
        tools[0],
        tools
    )

    assert confidence < 0.5
```

## Common Patterns

Frequent tool selection patterns.

### Pattern: Search Then Process

```python
def search_then_process_pattern(user_input: str) -> List[str]:
    """
    Common pattern: search for data, then process it.
    """
    if "analyze" in user_input.lower() or "summarize" in user_input.lower():
        # Need search tool + analysis tool
        return ["search_documents", "analyze_text"]

    return select_single_tool(user_input)
```

### Pattern: Fetch Then Transform

```python
def fetch_then_transform_pattern(user_input: str) -> List[str]:
    """
    Pattern: fetch data then transform it.
    """
    if "convert" in user_input.lower() or "transform" in user_input.lower():
        # Need data source + transformation tool
        source = select_data_source(user_input)
        transformer = select_transformer(user_input)
        return [source, transformer]

    return select_single_tool(user_input)
```

## Best Practices

Guidelines for effective tool selection.

**1. Prefer Specificity**: Choose the most specific tool for the task

**2. Consider Context**: Use all available context to inform selection

**3. Handle Uncertainty**: Have strategies for ambiguous cases

**4. Learn from History**: Track and learn from selection outcomes

**5. Optimize Appropriately**: Balance speed, cost, and quality

**6. Provide Transparency**: Explain selection reasoning when confidence is low

**7. Enable Fallbacks**: Have multiple strategies if primary selection fails

**8. Test Thoroughly**: Test selection with diverse inputs and scenarios

## Summary

Tool selection is a critical decision-making process in agent systems:

**Key Concepts**:

- Tool selection matches tools to tasks
- Multiple strategies: keyword matching, semantic similarity, LLM reasoning
- Handle ambiguity through disambiguation and confidence thresholds
- Scale with hierarchical filtering and clustering

**Selection Strategies**:

- Keyword and semantic matching for relevance
- Capability checking for fitness
- Performance and cost optimization
- Learning from history

**Best Practices**:

- Use multi-stage selection for large tool sets
- Consider context and constraints
- Provide confidence scores
- Have fallback strategies
- Test selection accuracy

Effective tool selection is what turns a collection of tools into a capable agent system.

## Next Steps

Continue your tool use journey:

- [Function Calling](function-calling.md) - Mechanics of invoking selected tools
- [Tool Chaining](tool-chaining.md) - Combining multiple tools in workflows
- [Error Handling](error-handling.md) - Handling tool selection and execution failures
- [ReAct](../agent-architectures/react.md) - Reasoning about tool selection explicitly
