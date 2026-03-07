# Tool Chaining and Composition

## Table of Contents

- [Introduction](#introduction)
- [What is Tool Chaining?](#what-is-tool-chaining)
- [Sequential Composition](#sequential-composition)
- [Parallel Tool Use](#parallel-tool-use)
- [Data Flow Between Tools](#data-flow-between-tools)
- [Orchestrating Multi-Tool Workflows](#orchestrating-multi-tool-workflows)
- [Dependency Management](#dependency-management)
- [Conditional Chaining](#conditional-chaining)
- [Iterative Tool Chains](#iterative-tool-chains)
- [Pipeline Patterns](#pipeline-patterns)
- [Error Recovery in Chains](#error-recovery-in-chains)
- [Optimization Strategies](#optimization-strategies)
- [State Management](#state-management)
- [Branching and Merging](#branching-and-merging)
- [Testing Tool Chains](#testing-tool-chains)
- [Common Patterns](#common-patterns)
- [Best Practices](#best-practices)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Tool chaining is the practice of combining multiple tools to accomplish complex tasks that no single tool can handle. It's the difference between having a toolbox and knowing how to build something with those tools.

Simple tasks need one tool. Complex tasks need **orchestrated sequences** of tools working together.

> "The whole is greater than the sum of its parts - when the parts work together."

This guide covers tool chaining from basic sequential execution to sophisticated workflows with parallel execution, conditional logic, and error recovery.

## What is Tool Chaining?

Tool chaining connects tools where the output of one becomes the input to another.

### Basic Chain

```
User: "Find papers about RAG and summarize the top 3"

┌─────────────────┐
│  search_papers  │
│  query="RAG"    │
└────────┬────────┘
         │ Returns: [paper1, paper2, paper3, ...]
         ▼
┌─────────────────┐
│  read_paper     │
│  paper_id=1     │
└────────┬────────┘
         │ Returns: full_text_1
         ▼
┌─────────────────┐
│  summarize_text │
│  text=full_text │
└────────┬────────┘
         │ Returns: summary_1
         ▼
    [Repeat for papers 2 and 3]
         │
         ▼
┌─────────────────┐
│  combine_summaries
└────────┬────────┘
         │
         ▼
    Final Result
```

### Why Chain Tools?

**1. Divide and Conquer**

```python
# Single tool (impossible)
result = do_everything(user_request)

# Chained tools (possible)
data = search(user_request)
analysis = analyze(data)
visualization = visualize(analysis)
result = format_report(visualization)
```

**2. Reusability**

```python
# Each tool is reusable in different chains
summary_chain = [fetch_document, summarize]
translation_chain = [fetch_document, translate]
analysis_chain = [fetch_document, analyze, visualize]
```

**3. Modularity**

```python
# Easy to modify, extend, or replace tools
old_chain = [search_v1, process, output]
new_chain = [search_v2, process, output]  # Upgraded search
```

## Sequential Composition

The simplest form: execute tools one after another.

### Basic Sequential Chain

```python
from typing import Any, List, Dict, Callable

def sequential_chain(
    tools: List[Callable],
    initial_input: Any
) -> Any:
    """
    Execute tools sequentially, passing output to next input.
    """
    current_input = initial_input

    for tool in tools:
        current_input = tool(current_input)

    return current_input


# Example usage
def search(query: str) -> List[Dict]:
    """Search for documents."""
    return [{"id": 1, "text": "Document about " + query}]

def extract_entities(docs: List[Dict]) -> List[str]:
    """Extract entities from documents."""
    entities = []
    for doc in docs:
        # Simple extraction
        entities.extend(doc["text"].split())
    return list(set(entities))

def filter_entities(entities: List[str]) -> List[str]:
    """Filter to important entities."""
    return [e for e in entities if len(e) > 5]


# Chain execution
result = sequential_chain(
    tools=[search, extract_entities, filter_entities],
    initial_input="machine learning"
)
```

### Explicit Chaining with Context

```python
class ChainContext:
    """Context object that accumulates information across chain."""

    def __init__(self, initial_input: Any):
        self.initial_input = initial_input
        self.history: List[Dict[str, Any]] = []
        self.intermediate_results: Dict[str, Any] = {}

    def add_result(self, tool_name: str, result: Any):
        """Add a tool's result to context."""
        self.history.append({
            "tool": tool_name,
            "result": result
        })
        self.intermediate_results[tool_name] = result


def execute_chain_with_context(
    tools: List[Dict[str, Any]],
    initial_input: Any
) -> ChainContext:
    """
    Execute chain with full context tracking.
    """
    context = ChainContext(initial_input)
    current_input = initial_input

    for tool_spec in tools:
        tool_name = tool_spec["name"]
        tool_func = tool_spec["function"]

        # Execute tool
        result = tool_func(current_input)

        # Store result
        context.add_result(tool_name, result)

        # Result becomes next input
        current_input = result

    return context


# Example
tools = [
    {"name": "search", "function": search},
    {"name": "extract", "function": extract_entities},
    {"name": "filter", "function": filter_entities}
]

context = execute_chain_with_context(tools, "machine learning")

# Access all intermediate results
for step in context.history:
    print(f"{step['tool']}: {step['result']}")
```

### ReAct-Style Sequential Chaining

```python
def react_sequential_chain(
    initial_goal: str,
    tools: Dict[str, Callable],
    max_steps: int = 10
) -> Any:
    """
    Sequential chain with explicit reasoning at each step.
    """
    context = {
        "goal": initial_goal,
        "history": []
    }

    for step in range(max_steps):
        # Think: decide next tool
        thought = generate_thought(context)
        print(f"Thought: {thought}")

        # Act: select and execute tool
        action = extract_action(thought)
        if action["type"] == "FINISH":
            return action["value"]

        tool_name = action["tool"]
        tool_args = action["args"]

        result = tools[tool_name](**tool_args)

        # Observe: record result
        context["history"].append({
            "thought": thought,
            "action": action,
            "result": result
        })

        print(f"Action: {tool_name}({tool_args})")
        print(f"Result: {result}\n")

    return context["history"][-1]["result"]
```

## Parallel Tool Use

Execute independent tools simultaneously for efficiency.

### Basic Parallel Execution

```python
import asyncio
from typing import List, Callable, Any

async def parallel_execution(
    tools: List[Callable],
    inputs: List[Any]
) -> List[Any]:
    """
    Execute tools in parallel.
    """
    tasks = [
        asyncio.create_task(tool(input_data))
        for tool, input_data in zip(tools, inputs)
    ]

    results = await asyncio.gather(*tasks)

    return results


# Example
async def fetch_url(url: str) -> str:
    """Simulate fetching URL."""
    await asyncio.sleep(1)  # Simulate network delay
    return f"Content from {url}"

async def search_database(query: str) -> List[Dict]:
    """Simulate database search."""
    await asyncio.sleep(0.5)
    return [{"result": query}]

async def call_api(endpoint: str) -> Dict:
    """Simulate API call."""
    await asyncio.sleep(0.8)
    return {"data": endpoint}


# Execute in parallel
async def main():
    tools = [fetch_url, search_database, call_api]
    inputs = ["https://example.com", "machine learning", "/api/data"]

    results = await parallel_execution(tools, inputs)
    print(results)

# asyncio.run(main())
```

### Fan-Out Fan-In Pattern

```python
async def fan_out_fan_in(
    input_data: Any,
    parallel_tools: List[Callable],
    combine_function: Callable
) -> Any:
    """
    Fan out to multiple tools, then fan in to combine results.

    ┌─────┐
    │Input│
    └──┬──┘
       │ Fan Out
       ├────────────┬────────────┐
       ▼            ▼            ▼
    ┌─────┐      ┌─────┐      ┌─────┐
    │Tool1│      │Tool2│      │Tool3│
    └──┬──┘      └──┬──┘      └──┬──┘
       │            │            │
       └────────────┴────────────┘ Fan In
                    │
                    ▼
              ┌──────────┐
              │ Combine  │
              └──────────┘
    """
    # Fan out: execute all tools in parallel
    tasks = [tool(input_data) for tool in parallel_tools]
    results = await asyncio.gather(*tasks)

    # Fan in: combine results
    combined = combine_function(results)

    return combined


# Example: Search multiple sources and combine
async def search_web(query: str) -> List[Dict]:
    await asyncio.sleep(1)
    return [{"source": "web", "result": f"Web results for {query}"}]

async def search_papers(query: str) -> List[Dict]:
    await asyncio.sleep(1.5)
    return [{"source": "papers", "result": f"Papers about {query}"}]

async def search_news(query: str) -> List[Dict]:
    await asyncio.sleep(0.8)
    return [{"source": "news", "result": f"News on {query}"}]

def combine_search_results(results: List[List[Dict]]) -> Dict:
    """Combine results from multiple sources."""
    combined = {
        "total_results": sum(len(r) for r in results),
        "by_source": {}
    }

    for result_list in results:
        for item in result_list:
            source = item["source"]
            if source not in combined["by_source"]:
                combined["by_source"][source] = []
            combined["by_source"][source].append(item["result"])

    return combined


# Usage
async def multi_source_search(query: str):
    return await fan_out_fan_in(
        input_data=query,
        parallel_tools=[search_web, search_papers, search_news],
        combine_function=combine_search_results
    )
```

### Parallel with Different Inputs

```python
async def parallel_different_inputs(
    tool_specs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Execute multiple tools with different inputs in parallel.
    """
    tasks = {}

    for spec in tool_specs:
        tool_name = spec["name"]
        tool_func = spec["function"]
        tool_input = spec["input"]

        tasks[tool_name] = asyncio.create_task(tool_func(tool_input))

    # Wait for all tasks
    results = {}
    for name, task in tasks.items():
        results[name] = await task

    return results


# Example: Gather information from different sources
async def gather_user_info(user_id: str):
    specs = [
        {
            "name": "profile",
            "function": fetch_user_profile,
            "input": user_id
        },
        {
            "name": "activity",
            "function": fetch_user_activity,
            "input": user_id
        },
        {
            "name": "preferences",
            "function": fetch_user_preferences,
            "input": user_id
        }
    ]

    return await parallel_different_inputs(specs)
```

## Data Flow Between Tools

Managing how data moves through tool chains.

### Output to Input Mapping

```python
class ToolChain:
    """
    Tool chain with explicit output-to-input mapping.
    """

    def __init__(self):
        self.tools: List[Dict[str, Any]] = []

    def add_tool(
        self,
        name: str,
        function: Callable,
        input_mapping: Callable = None
    ):
        """
        Add tool to chain with optional input mapping.

        input_mapping: Function to transform previous output to this tool's input
        """
        self.tools.append({
            "name": name,
            "function": function,
            "input_mapping": input_mapping or (lambda x: x)
        })

    def execute(self, initial_input: Any) -> Any:
        """Execute the chain."""
        current_data = initial_input

        for tool_spec in self.tools:
            # Map previous output to current input
            mapped_input = tool_spec["input_mapping"](current_data)

            # Execute tool
            result = tool_spec["function"](mapped_input)

            current_data = result

        return current_data


# Example: Chain with data transformation
chain = ToolChain()

# Tool 1: Search returns list of documents
chain.add_tool(
    name="search",
    function=search_documents
)

# Tool 2: Extract needs just the text from documents
chain.add_tool(
    name="extract",
    function=extract_key_points,
    input_mapping=lambda docs: [doc["text"] for doc in docs]
)

# Tool 3: Summarize needs concatenated text
chain.add_tool(
    name="summarize",
    function=summarize_text,
    input_mapping=lambda points: "\n".join(points)
)

result = chain.execute("machine learning")
```

### Named Output Access

```python
class DataFlowChain:
    """
    Chain where tools can access any previous output by name.
    """

    def __init__(self):
        self.tools: List[Dict[str, Any]] = []
        self.results: Dict[str, Any] = {}

    def add_tool(
        self,
        name: str,
        function: Callable,
        input_spec: Dict[str, str]
    ):
        """
        Add tool with specification of which previous outputs to use.

        input_spec maps function parameter names to output names.
        Example: {"query": "user_input", "docs": "search_results"}
        """
        self.tools.append({
            "name": name,
            "function": function,
            "input_spec": input_spec
        })

    def execute(self, initial_input: Any, input_name: str = "input") -> Any:
        """Execute chain."""
        # Store initial input
        self.results[input_name] = initial_input

        for tool_spec in self.tools:
            # Build arguments from previous results
            kwargs = {}
            for param_name, output_name in tool_spec["input_spec"].items():
                if output_name in self.results:
                    kwargs[param_name] = self.results[output_name]
                else:
                    raise ValueError(f"Output '{output_name}' not found")

            # Execute tool
            result = tool_spec["function"](**kwargs)

            # Store result
            self.results[tool_spec["name"]] = result

        # Return final result
        return result


# Example: Complex data flow
def search(query: str) -> List[Dict]:
    return [{"id": 1, "text": "Document about " + query}]

def filter_docs(docs: List[Dict], min_relevance: float) -> List[Dict]:
    # Simplified filtering
    return docs

def summarize(docs: List[Dict], original_query: str) -> str:
    return f"Summary of {len(docs)} documents about {original_query}"


chain = DataFlowChain()

chain.add_tool(
    name="search",
    function=search,
    input_spec={"query": "input"}
)

chain.add_tool(
    name="filter",
    function=filter_docs,
    input_spec={"docs": "search"}
)

chain.add_tool(
    name="summary",
    function=summarize,
    input_spec={"docs": "filter", "original_query": "input"}
)

result = chain.execute("machine learning", input_name="input")
```

### Type-Safe Data Flow

```python
from typing import TypeVar, Generic
from dataclasses import dataclass

T = TypeVar('T')
U = TypeVar('U')

@dataclass
class TypedOutput(Generic[T]):
    """Type-safe output container."""
    data: T
    metadata: Dict[str, Any]


class TypedChain:
    """
    Type-safe tool chain.
    """

    def __init__(self):
        self.steps: List[Callable] = []

    def add_step(self, func: Callable[[T], U]) -> 'TypedChain':
        """Add a typed step to chain."""
        self.steps.append(func)
        return self  # For chaining

    def execute(self, initial_input: T) -> Any:
        """Execute with type checking."""
        current = initial_input

        for step in self.steps:
            current = step(current)

        return current


# Example with type hints
def search(query: str) -> List[Dict[str, str]]:
    """Search returns list of documents."""
    return [{"text": "Document"}]

def extract(docs: List[Dict[str, str]]) -> List[str]:
    """Extract returns list of strings."""
    return [doc["text"] for doc in docs]

def count_words(texts: List[str]) -> int:
    """Count returns integer."""
    return sum(len(text.split()) for text in texts)


# Build typed chain
chain = TypedChain()
chain.add_step(search).add_step(extract).add_step(count_words)

# Execute
word_count = chain.execute("machine learning")
```

## Orchestrating Multi-Tool Workflows

Complex workflows with branching, merging, and coordination.

### Workflow Definition

```python
from enum import Enum
from typing import Any, Dict, List, Optional

class StepType(Enum):
    """Types of workflow steps."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"


class WorkflowStep:
    """A step in a workflow."""

    def __init__(
        self,
        name: str,
        step_type: StepType,
        tool: Optional[Callable] = None,
        substeps: Optional[List['WorkflowStep']] = None,
        condition: Optional[Callable] = None
    ):
        self.name = name
        self.step_type = step_type
        self.tool = tool
        self.substeps = substeps or []
        self.condition = condition


class WorkflowOrchestrator:
    """
    Orchestrate complex multi-tool workflows.
    """

    async def execute_step(
        self,
        step: WorkflowStep,
        context: Dict[str, Any]
    ) -> Any:
        """Execute a workflow step."""

        if step.step_type == StepType.SEQUENTIAL:
            return await self._execute_sequential(step, context)

        elif step.step_type == StepType.PARALLEL:
            return await self._execute_parallel(step, context)

        elif step.step_type == StepType.CONDITIONAL:
            return await self._execute_conditional(step, context)

        elif step.step_type == StepType.LOOP:
            return await self._execute_loop(step, context)

    async def _execute_sequential(
        self,
        step: WorkflowStep,
        context: Dict[str, Any]
    ) -> Any:
        """Execute substeps sequentially."""
        results = []

        for substep in step.substeps:
            result = await self.execute_step(substep, context)
            results.append(result)
            context[substep.name] = result

        return results[-1] if results else None

    async def _execute_parallel(
        self,
        step: WorkflowStep,
        context: Dict[str, Any]
    ) -> List[Any]:
        """Execute substeps in parallel."""
        tasks = [
            self.execute_step(substep, context)
            for substep in step.substeps
        ]

        results = await asyncio.gather(*tasks)

        # Store results in context
        for substep, result in zip(step.substeps, results):
            context[substep.name] = result

        return results

    async def _execute_conditional(
        self,
        step: WorkflowStep,
        context: Dict[str, Any]
    ) -> Any:
        """Execute conditionally."""
        if step.condition and step.condition(context):
            return await self.execute_step(step.substeps[0], context)
        elif len(step.substeps) > 1:
            return await self.execute_step(step.substeps[1], context)

        return None

    async def _execute_loop(
        self,
        step: WorkflowStep,
        context: Dict[str, Any]
    ) -> List[Any]:
        """Execute substeps in a loop."""
        results = []
        max_iterations = context.get("max_iterations", 10)

        for i in range(max_iterations):
            if step.condition and not step.condition(context):
                break

            result = await self.execute_step(step.substeps[0], context)
            results.append(result)
            context[f"{step.name}_iteration_{i}"] = result

        return results


# Example workflow
async def example_workflow():
    """
    Complex workflow:
    1. Search (parallel across sources)
    2. If results > 10: filter, else: expand search
    3. Analyze results
    4. Generate report
    """

    orchestrator = WorkflowOrchestrator()

    # Define workflow
    workflow = WorkflowStep(
        name="main",
        step_type=StepType.SEQUENTIAL,
        substeps=[
            # Parallel search
            WorkflowStep(
                name="search",
                step_type=StepType.PARALLEL,
                substeps=[
                    WorkflowStep("web", StepType.SEQUENTIAL, tool=search_web),
                    WorkflowStep("papers", StepType.SEQUENTIAL, tool=search_papers)
                ]
            ),

            # Conditional processing
            WorkflowStep(
                name="process",
                step_type=StepType.CONDITIONAL,
                condition=lambda ctx: len(ctx.get("search", [])) > 10,
                substeps=[
                    WorkflowStep("filter", StepType.SEQUENTIAL, tool=filter_results),
                    WorkflowStep("expand", StepType.SEQUENTIAL, tool=expand_search)
                ]
            ),

            # Analysis
            WorkflowStep(
                name="analyze",
                step_type=StepType.SEQUENTIAL,
                tool=analyze_results
            ),

            # Report
            WorkflowStep(
                name="report",
                step_type=StepType.SEQUENTIAL,
                tool=generate_report
            )
        ]
    )

    # Execute
    context = {"query": "machine learning"}
    result = await orchestrator.execute_step(workflow, context)

    return result
```

## Dependency Management

Managing dependencies between tools in chains.

### Dependency Graph

```python
from collections import defaultdict, deque
from typing import Set

class DependencyGraph:
    """
    Manage tool dependencies.
    """

    def __init__(self):
        self.graph: Dict[str, Set[str]] = defaultdict(set)
        self.tools: Dict[str, Callable] = {}

    def add_tool(self, name: str, tool: Callable, depends_on: List[str] = None):
        """Add tool with its dependencies."""
        self.tools[name] = tool

        if depends_on:
            for dependency in depends_on:
                self.graph[name].add(dependency)

    def topological_sort(self) -> List[str]:
        """
        Return tools in order that respects dependencies.
        """
        # Calculate in-degree
        in_degree = defaultdict(int)
        for node in self.graph:
            for neighbor in self.graph[node]:
                in_degree[neighbor] += 1

        # Find all nodes with no incoming edges
        queue = deque([node for node in self.tools if in_degree[node] == 0])

        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            # Reduce in-degree for neighbors
            for neighbor in self.graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.tools):
            raise ValueError("Circular dependency detected")

        return result

    async def execute_with_dependencies(
        self,
        initial_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute tools respecting dependencies, parallelizing where possible.
        """
        execution_order = self.topological_sort()
        results = initial_context.copy()
        executed = set()

        while len(executed) < len(self.tools):
            # Find tools ready to execute (dependencies satisfied)
            ready = [
                tool_name for tool_name in execution_order
                if tool_name not in executed
                and all(dep in executed for dep in self.graph[tool_name])
            ]

            if not ready:
                break  # No more tools can execute

            # Execute ready tools in parallel
            tasks = []
            for tool_name in ready:
                tool = self.tools[tool_name]
                tasks.append(self._execute_tool(tool, results))

            parallel_results = await asyncio.gather(*tasks)

            # Store results
            for tool_name, result in zip(ready, parallel_results):
                results[tool_name] = result
                executed.add(tool_name)

        return results

    async def _execute_tool(self, tool: Callable, context: Dict[str, Any]) -> Any:
        """Execute a single tool."""
        # In practice, would extract needed inputs from context
        return await tool(context)


# Example
graph = DependencyGraph()

# Define tools and dependencies
graph.add_tool("search", search_tool, depends_on=[])
graph.add_tool("extract", extract_tool, depends_on=["search"])
graph.add_tool("translate", translate_tool, depends_on=["search"])
graph.add_tool("summarize", summarize_tool, depends_on=["extract", "translate"])
graph.add_tool("report", report_tool, depends_on=["summarize"])

# Execute - will automatically parallelize extract and translate
# results = await graph.execute_with_dependencies({"query": "ML"})
```

## Conditional Chaining

Chains that adapt based on intermediate results.

### If-Then-Else Chains

```python
def conditional_chain(
    initial_input: Any,
    condition: Callable[[Any], bool],
    then_tools: List[Callable],
    else_tools: List[Callable]
) -> Any:
    """
    Execute different tool chains based on condition.
    """
    if condition(initial_input):
        # Execute 'then' branch
        result = initial_input
        for tool in then_tools:
            result = tool(result)
        return result
    else:
        # Execute 'else' branch
        result = initial_input
        for tool in else_tools:
            result = tool(result)
        return result


# Example: Different processing based on data size
def process_data(data: List[Dict]) -> Any:
    """Process data differently based on size."""

    return conditional_chain(
        initial_input=data,
        condition=lambda d: len(d) > 1000,  # Large dataset?
        then_tools=[
            sample_data,  # Sample for large dataset
            parallel_process,
            aggregate_results
        ],
        else_tools=[
            process_all,  # Process all for small dataset
            detailed_analysis
        ]
    )
```

### Switch-Case Chains

```python
def switch_chain(
    initial_input: Any,
    selector: Callable[[Any], str],
    cases: Dict[str, List[Callable]],
    default: List[Callable] = None
) -> Any:
    """
    Execute different chains based on selector value.
    """
    case_key = selector(initial_input)

    tools = cases.get(case_key, default or [])

    result = initial_input
    for tool in tools:
        result = tool(result)

    return result


# Example: Process based on document type
def process_document(doc: Dict) -> Any:
    """Process document based on its type."""

    return switch_chain(
        initial_input=doc,
        selector=lambda d: d.get("type", "unknown"),
        cases={
            "pdf": [extract_pdf_text, parse_pdf, analyze_text],
            "image": [ocr_image, extract_text, analyze_text],
            "html": [parse_html, extract_content, analyze_text],
            "json": [parse_json, validate_schema, process_data]
        },
        default=[generic_processor]
    )
```

### Early Exit Chains

```python
def chain_with_early_exit(
    initial_input: Any,
    tools: List[Callable],
    should_continue: Callable[[Any], bool]
) -> Any:
    """
    Chain that can exit early based on intermediate results.
    """
    current = initial_input

    for tool in tools:
        current = tool(current)

        # Check if should continue
        if not should_continue(current):
            break

    return current


# Example: Search until sufficient results
def search_until_sufficient(query: str) -> List[Dict]:
    """
    Search using progressively broader strategies until enough results.
    """

    return chain_with_early_exit(
        initial_input={"query": query, "results": []},
        tools=[
            search_exact_match,
            search_fuzzy_match,
            search_semantic,
            search_related_terms
        ],
        should_continue=lambda data: len(data["results"]) < 10
    )
```

## Iterative Tool Chains

Chains that repeat until a condition is met.

### Retry Loop

```python
def retry_chain(
    tool: Callable,
    input_data: Any,
    max_retries: int = 3,
    is_success: Callable[[Any], bool] = None
) -> Any:
    """
    Retry tool execution until success or max retries.
    """
    if is_success is None:
        is_success = lambda result: result is not None

    for attempt in range(max_retries):
        try:
            result = tool(input_data)

            if is_success(result):
                return result

            print(f"Attempt {attempt + 1} unsuccessful, retrying...")

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")

            if attempt == max_retries - 1:
                raise

    raise Exception(f"Failed after {max_retries} attempts")
```

### Iterative Refinement

```python
def iterative_refinement_chain(
    initial_input: Any,
    process_tool: Callable,
    evaluate_tool: Callable,
    refine_tool: Callable,
    max_iterations: int = 5,
    quality_threshold: float = 0.8
) -> Any:
    """
    Iteratively refine output until quality threshold is met.

    Process: Generate → Evaluate → Refine → Repeat
    """
    current = initial_input

    for iteration in range(max_iterations):
        # Generate/process
        output = process_tool(current)

        # Evaluate quality
        quality = evaluate_tool(output)

        print(f"Iteration {iteration + 1}: Quality = {quality:.2f}")

        if quality >= quality_threshold:
            return output

        # Refine input based on output
        current = refine_tool(output, quality)

    # Return best effort after max iterations
    return output


# Example: Iteratively improve query
def improve_search_query(initial_query: str) -> str:
    """
    Iteratively refine search query based on result quality.
    """

    def process(query: str) -> Dict:
        results = search(query)
        return {"query": query, "results": results}

    def evaluate(output: Dict) -> float:
        # Score based on number and relevance of results
        results = output["results"]
        if not results:
            return 0.0
        return min(len(results) / 10, 1.0)  # Target 10+ results

    def refine(output: Dict, quality: float) -> str:
        if quality < 0.3:
            # Too few results - broaden query
            return broaden_query(output["query"])
        elif quality > 0.9:
            # Too many results - narrow query
            return narrow_query(output["query"])
        else:
            # Add related terms
            return add_related_terms(output["query"])

    result = iterative_refinement_chain(
        initial_input=initial_query,
        process_tool=process,
        evaluate_tool=evaluate,
        refine_tool=refine,
        max_iterations=5,
        quality_threshold=0.8
    )

    return result["query"]
```

### Convergence Loop

```python
def convergence_loop(
    initial_state: Any,
    update_tool: Callable,
    converged: Callable[[Any, Any], bool],
    max_iterations: int = 100
) -> Any:
    """
    Iterate until state converges.
    """
    current_state = initial_state
    previous_state = None

    for iteration in range(max_iterations):
        # Check convergence
        if previous_state is not None and converged(current_state, previous_state):
            print(f"Converged after {iteration} iterations")
            return current_state

        # Update state
        previous_state = current_state
        current_state = update_tool(current_state)

    print(f"Did not converge after {max_iterations} iterations")
    return current_state


# Example: Iterative clustering
def iterative_clustering(data: List[Any]) -> Dict:
    """
    Cluster data until assignments stabilize.
    """

    def update(state: Dict) -> Dict:
        # Update cluster assignments and centroids
        assignments = assign_to_clusters(data, state["centroids"])
        new_centroids = update_centroids(data, assignments)
        return {"centroids": new_centroids, "assignments": assignments}

    def converged(current: Dict, previous: Dict) -> bool:
        # Check if assignments changed
        return current["assignments"] == previous["assignments"]

    initial_state = {
        "centroids": initialize_centroids(data),
        "assignments": []
    }

    return convergence_loop(
        initial_state=initial_state,
        update_tool=update,
        converged=converged,
        max_iterations=100
    )
```

## Pipeline Patterns

Common pipeline patterns for tool chains.

### ETL Pipeline

```python
class ETLPipeline:
    """Extract, Transform, Load pipeline."""

    def __init__(self):
        self.extractors: List[Callable] = []
        self.transformers: List[Callable] = []
        self.loaders: List[Callable] = []

    def add_extractor(self, extractor: Callable):
        """Add data extraction step."""
        self.extractors.append(extractor)

    def add_transformer(self, transformer: Callable):
        """Add data transformation step."""
        self.transformers.append(transformer)

    def add_loader(self, loader: Callable):
        """Add data loading step."""
        self.loaders.append(loader)

    async def execute(self, source: Any) -> Any:
        """Execute ETL pipeline."""

        # Extract
        extracted_data = []
        for extractor in self.extractors:
            data = await extractor(source)
            extracted_data.append(data)

        # Combine extracted data
        combined = self._combine_extracted(extracted_data)

        # Transform
        transformed = combined
        for transformer in self.transformers:
            transformed = await transformer(transformed)

        # Load
        results = []
        for loader in self.loaders:
            result = await loader(transformed)
            results.append(result)

        return results

    def _combine_extracted(self, data_list: List[Any]) -> Any:
        """Combine data from multiple extractors."""
        # Simple concatenation - can be customized
        if all(isinstance(d, list) for d in data_list):
            return [item for sublist in data_list for item in sublist]
        return data_list


# Example usage
pipeline = ETLPipeline()

# Extract from multiple sources
pipeline.add_extractor(extract_from_api)
pipeline.add_extractor(extract_from_database)
pipeline.add_extractor(extract_from_files)

# Transform
pipeline.add_transformer(clean_data)
pipeline.add_transformer(normalize_format)
pipeline.add_transformer(enrich_data)

# Load to destinations
pipeline.add_loader(load_to_database)
pipeline.add_loader(load_to_cache)

# Execute
# result = await pipeline.execute("data_source")
```

### Map-Reduce Pipeline

```python
async def map_reduce_pipeline(
    data: List[Any],
    map_func: Callable,
    reduce_func: Callable
) -> Any:
    """
    Map-reduce pattern for parallel processing.
    """
    # Map phase: process each item independently
    map_tasks = [map_func(item) for item in data]
    mapped_results = await asyncio.gather(*map_tasks)

    # Reduce phase: combine results
    final_result = reduce_func(mapped_results)

    return final_result


# Example: Count word frequencies across documents
async def count_word_frequencies(documents: List[str]) -> Dict[str, int]:
    """Count word frequencies using map-reduce."""

    async def map_words(doc: str) -> Dict[str, int]:
        """Map: count words in single document."""
        await asyncio.sleep(0.1)  # Simulate processing
        words = doc.lower().split()
        counts = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
        return counts

    def reduce_counts(count_list: List[Dict[str, int]]) -> Dict[str, int]:
        """Reduce: combine word counts."""
        total_counts = {}
        for counts in count_list:
            for word, count in counts.items():
                total_counts[word] = total_counts.get(word, 0) + count
        return total_counts

    return await map_reduce_pipeline(documents, map_words, reduce_counts)
```

### Filter-Map-Reduce Pipeline

```python
async def filter_map_reduce_pipeline(
    data: List[Any],
    filter_func: Callable[[Any], bool],
    map_func: Callable,
    reduce_func: Callable
) -> Any:
    """
    Extended pipeline: Filter → Map → Reduce
    """
    # Filter
    filtered_data = [item for item in data if filter_func(item)]

    if not filtered_data:
        return None

    # Map
    map_tasks = [map_func(item) for item in filtered_data]
    mapped_results = await asyncio.gather(*map_tasks)

    # Reduce
    final_result = reduce_func(mapped_results)

    return final_result
```

## Error Recovery in Chains

Handling failures in tool chains.

### Try-Catch Chaining

```python
def chain_with_error_handling(
    tools: List[Callable],
    input_data: Any,
    on_error: str = "stop"  # "stop", "skip", or "retry"
) -> tuple[Any, List[str]]:
    """
    Execute chain with error handling.

    on_error:
        "stop" - stop on first error
        "skip" - skip failed tools, continue chain
        "retry" - retry failed tools once
    """
    current = input_data
    errors = []

    for i, tool in enumerate(tools):
        try:
            current = tool(current)

        except Exception as e:
            error_msg = f"Tool {i} failed: {e}"
            errors.append(error_msg)

            if on_error == "stop":
                raise

            elif on_error == "skip":
                print(f"Skipping failed tool: {error_msg}")
                continue

            elif on_error == "retry":
                try:
                    print(f"Retrying tool {i}")
                    current = tool(current)
                except Exception as retry_error:
                    errors.append(f"Retry failed: {retry_error}")
                    if i == len(tools) - 1:
                        # Last tool, can't skip
                        raise
                    continue

    return current, errors
```

### Fallback Chains

```python
def chain_with_fallbacks(
    primary_chain: List[Callable],
    fallback_chains: List[List[Callable]],
    input_data: Any
) -> tuple[Any, str]:
    """
    Try primary chain, fall back to alternatives if it fails.
    """
    # Try primary chain
    try:
        result = sequential_chain(primary_chain, input_data)
        return result, "primary"
    except Exception as e:
        print(f"Primary chain failed: {e}")

    # Try fallback chains
    for i, fallback_chain in enumerate(fallback_chains):
        try:
            result = sequential_chain(fallback_chain, input_data)
            return result, f"fallback_{i}"
        except Exception as e:
            print(f"Fallback {i} failed: {e}")
            continue

    raise Exception("All chains failed")


# Example
result, used = chain_with_fallbacks(
    primary_chain=[
        search_primary_source,
        extract_data,
        format_result
    ],
    fallback_chains=[
        # Fallback 1: Try secondary source
        [search_secondary_source, extract_data, format_result],

        # Fallback 2: Use cached data
        [load_cached_data, format_result],

        # Fallback 3: Return placeholder
        [return_placeholder]
    ],
    input_data="query"
)
```

### Compensating Actions

```python
class TransactionalChain:
    """
    Chain with compensating actions for rollback.
    """

    def __init__(self):
        self.steps: List[tuple[Callable, Callable]] = []
        self.executed: List[tuple[str, Any]] = []

    def add_step(
        self,
        name: str,
        action: Callable,
        compensation: Callable
    ):
        """
        Add step with its compensation action.
        """
        self.steps.append((name, action, compensation))

    def execute(self, input_data: Any) -> Any:
        """
        Execute chain with automatic rollback on failure.
        """
        current = input_data

        try:
            for name, action, compensation in self.steps:
                result = action(current)
                self.executed.append((name, result, compensation))
                current = result

            return current

        except Exception as e:
            print(f"Error during execution: {e}")
            print("Rolling back...")

            # Execute compensations in reverse order
            for name, result, compensation in reversed(self.executed):
                try:
                    compensation(result)
                    print(f"Rolled back: {name}")
                except Exception as comp_error:
                    print(f"Compensation failed for {name}: {comp_error}")

            raise


# Example: Database operations with rollback
def example_transactional():
    chain = TransactionalChain()

    chain.add_step(
        name="create_user",
        action=lambda data: create_user_in_db(data),
        compensation=lambda result: delete_user_from_db(result["user_id"])
    )

    chain.add_step(
        name="create_profile",
        action=lambda user: create_profile_in_db(user["user_id"]),
        compensation=lambda result: delete_profile_from_db(result["profile_id"])
    )

    chain.add_step(
        name="send_welcome_email",
        action=lambda profile: send_email(profile["email"]),
        compensation=lambda result: None  # Can't unsend email
    )

    try:
        result = chain.execute({"name": "John", "email": "john@example.com"})
    except Exception:
        print("Operation failed and was rolled back")
```

## Optimization Strategies

Making tool chains more efficient.

### Caching Intermediate Results

```python
from functools import lru_cache
from hashlib import md5
import json

class CachedChain:
    """Tool chain with result caching."""

    def __init__(self):
        self.tools: List[tuple[str, Callable]] = []
        self.cache: Dict[str, Any] = {}

    def add_tool(self, name: str, tool: Callable):
        """Add tool to chain."""
        self.tools.append((name, tool))

    def _cache_key(self, tool_name: str, input_data: Any) -> str:
        """Generate cache key."""
        input_str = json.dumps(input_data, sort_keys=True)
        return f"{tool_name}:{md5(input_str.encode()).hexdigest()}"

    def execute(self, input_data: Any, use_cache: bool = True) -> Any:
        """Execute chain with caching."""
        current = input_data

        for tool_name, tool in self.tools:
            # Check cache
            cache_key = self._cache_key(tool_name, current)

            if use_cache and cache_key in self.cache:
                print(f"Cache hit for {tool_name}")
                current = self.cache[cache_key]
            else:
                # Execute tool
                result = tool(current)

                # Store in cache
                self.cache[cache_key] = result
                current = result

        return current

    def clear_cache(self):
        """Clear all cached results."""
        self.cache.clear()
```

### Parallel Execution Where Possible

```python
async def optimize_chain_execution(
    chain_spec: Dict[str, Any]
) -> Any:
    """
    Automatically parallelize independent operations.
    """
    # Analyze dependencies
    dependencies = analyze_dependencies(chain_spec)

    # Group by dependency level
    levels = topological_levels(dependencies)

    results = {}

    for level_tools in levels:
        # Execute tools in same level in parallel
        tasks = []
        for tool_name in level_tools:
            tool = chain_spec["tools"][tool_name]
            # Get inputs from previous results
            inputs = {
                k: results[v]
                for k, v in tool["inputs"].items()
            }
            tasks.append(tool["function"](**inputs))

        # Wait for level to complete
        level_results = await asyncio.gather(*tasks)

        # Store results
        for tool_name, result in zip(level_tools, level_results):
            results[tool_name] = result

    return results
```

### Lazy Evaluation

```python
class LazyChain:
    """
    Chain that only executes tools when result is needed.
    """

    def __init__(self):
        self.tools: List[Callable] = []

    def add_tool(self, tool: Callable):
        """Add tool to chain."""
        self.tools.append(tool)

    def build(self, input_data: Any) -> 'LazyResult':
        """Build lazy execution plan."""
        return LazyResult(self.tools, input_data)


class LazyResult:
    """Lazy evaluation result."""

    def __init__(self, tools: List[Callable], input_data: Any):
        self.tools = tools
        self.input_data = input_data
        self._result = None
        self._evaluated = False

    def evaluate(self) -> Any:
        """Force evaluation."""
        if not self._evaluated:
            current = self.input_data
            for tool in self.tools:
                current = tool(current)
            self._result = current
            self._evaluated = True

        return self._result

    def __repr__(self):
        if self._evaluated:
            return f"LazyResult({self._result})"
        return "LazyResult(not evaluated)"


# Usage
chain = LazyChain()
chain.add_tool(expensive_operation_1)
chain.add_tool(expensive_operation_2)

# Build plan (doesn't execute yet)
result = chain.build(input_data)

# Only evaluate if needed
if some_condition:
    actual_result = result.evaluate()
```

## State Management

Tracking state across tool executions.

### Stateful Chain

```python
class StatefulChain:
    """
    Chain that maintains state across executions.
    """

    def __init__(self):
        self.tools: List[Callable] = []
        self.state: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

    def add_tool(self, tool: Callable):
        """Add tool to chain."""
        self.tools.append(tool)

    def execute(self, input_data: Any) -> Any:
        """Execute chain with state."""
        current = input_data

        for tool in self.tools:
            # Pass state to tool
            result = tool(current, self.state)

            # Record in history
            self.history.append({
                "tool": tool.__name__,
                "input": current,
                "output": result,
                "state": self.state.copy()
            })

            current = result

        return current

    def reset_state(self):
        """Reset state."""
        self.state.clear()
        self.history.clear()


# Example: Accumulating state
def accumulating_search(query: str, state: Dict[str, Any]) -> List[Dict]:
    """Search and accumulate results in state."""
    results = search(query)

    if "all_results" not in state:
        state["all_results"] = []

    state["all_results"].extend(results)
    state["total_count"] = len(state["all_results"])

    return results


chain = StatefulChain()
chain.add_tool(accumulating_search)
chain.add_tool(filter_duplicates)
chain.add_tool(rank_results)

# State persists across executions
chain.execute("machine learning")
chain.execute("deep learning")  # Can access previous results from state
```

## Branching and Merging

Chains that split and rejoin.

### Branch and Merge

```python
async def branch_and_merge(
    input_data: Any,
    branches: List[List[Callable]],
    merge_function: Callable
) -> Any:
    """
    Split execution into branches, then merge results.

    ┌─────────┐
    │  Input  │
    └────┬────┘
         │
    ┌────┴────┬────────────┬────────────┐
    │Branch 1 │  Branch 2  │  Branch 3  │
    │ Tool A  │   Tool X   │   Tool M   │
    │ Tool B  │   Tool Y   │   Tool N   │
    └────┬────┴────┬───────┴──────┬─────┘
         │         │              │
         └─────────┴──────────────┘
                   │
            ┌──────┴──────┐
            │    Merge    │
            └─────────────┘
    """
    # Execute branches in parallel
    branch_tasks = []

    for branch in branches:
        # Each branch is a sequential chain
        async def execute_branch(tools):
            current = input_data
            for tool in tools:
                current = await tool(current)
            return current

        branch_tasks.append(execute_branch(branch))

    # Wait for all branches
    branch_results = await asyncio.gather(*branch_tasks)

    # Merge results
    final_result = merge_function(branch_results)

    return final_result


# Example: Multi-perspective analysis
async def multi_perspective_analysis(document: str):
    """Analyze document from multiple perspectives."""

    return await branch_and_merge(
        input_data=document,
        branches=[
            # Technical analysis
            [extract_technical_terms, analyze_complexity],

            # Sentiment analysis
            [extract_sentiment, classify_tone],

            # Topic analysis
            [extract_topics, categorize_content]
        ],
        merge_function=lambda results: {
            "technical": results[0],
            "sentiment": results[1],
            "topics": results[2]
        }
    )
```

## Testing Tool Chains

Ensuring chains work correctly.

### Unit Tests for Chains

```python
import pytest

def test_sequential_chain():
    """Test sequential chain execution."""

    def double(x):
        return x * 2

    def add_ten(x):
        return x + 10

    def to_string(x):
        return str(x)

    result = sequential_chain([double, add_ten, to_string], 5)

    assert result == "20"  # (5 * 2) + 10 = "20"


@pytest.mark.asyncio
async def test_parallel_execution():
    """Test parallel tool execution."""

    async def tool1(x):
        await asyncio.sleep(0.1)
        return x * 2

    async def tool2(x):
        await asyncio.sleep(0.1)
        return x + 5

    async def tool3(x):
        await asyncio.sleep(0.1)
        return x ** 2

    results = await parallel_execution(
        [tool1, tool2, tool3],
        [5, 5, 5]
    )

    assert results == [10, 10, 25]


def test_conditional_chain():
    """Test conditional chain execution."""

    data_small = [1, 2, 3]
    data_large = list(range(100))

    def process_small(data):
        return sum(data)

    def process_large(data):
        return len(data)

    # Small data
    result = conditional_chain(
        data_small,
        condition=lambda d: len(d) > 10,
        then_tools=[process_large],
        else_tools=[process_small]
    )

    assert result == 6

    # Large data
    result = conditional_chain(
        data_large,
        condition=lambda d: len(d) > 10,
        then_tools=[process_large],
        else_tools=[process_small]
    )

    assert result == 100
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_complex_workflow():
    """Test complex multi-tool workflow."""

    # Mock tools
    async def mock_search(query):
        return [{"id": 1, "text": "Result 1"}]

    async def mock_analyze(results):
        return {"count": len(results), "summary": "Analysis"}

    async def mock_report(analysis):
        return f"Report: {analysis['count']} items"

    # Build workflow
    workflow = WorkflowStep(
        name="main",
        step_type=StepType.SEQUENTIAL,
        substeps=[
            WorkflowStep("search", StepType.SEQUENTIAL, tool=mock_search),
            WorkflowStep("analyze", StepType.SEQUENTIAL, tool=mock_analyze),
            WorkflowStep("report", StepType.SEQUENTIAL, tool=mock_report)
        ]
    )

    # Execute
    orchestrator = WorkflowOrchestrator()
    context = {"query": "test"}
    result = await orchestrator.execute_step(workflow, context)

    assert result == "Report: 1 items"
```

## Common Patterns

Frequently used tool chain patterns.

### Search-Process-Report

```python
async def search_process_report_pattern(query: str) -> str:
    """
    Common pattern: Search → Process → Report
    """
    # Search
    search_results = await search_tool(query)

    # Process
    processed = await process_tool(search_results)

    # Report
    report = await report_tool(processed)

    return report
```

### Gather-Aggregate-Distribute

```python
async def gather_aggregate_distribute_pattern(sources: List[str]) -> Dict:
    """
    Pattern: Gather from sources → Aggregate → Distribute to targets
    """
    # Gather from all sources in parallel
    gather_tasks = [fetch_from_source(source) for source in sources]
    gathered_data = await asyncio.gather(*gather_tasks)

    # Aggregate
    aggregated = aggregate_data(gathered_data)

    # Distribute to targets
    distribute_tasks = [
        send_to_target(aggregated, target)
        for target in ["cache", "database", "notification"]
    ]
    await asyncio.gather(*distribute_tasks)

    return aggregated
```

## Best Practices

Guidelines for effective tool chaining.

**1. Keep Chains Simple**: Avoid deeply nested chains

**2. Make Dependencies Explicit**: Clearly document what each tool needs

**3. Handle Errors Gracefully**: Every chain should have error handling

**4. Enable Observability**: Log each step for debugging

**5. Optimize Carefully**: Parallelize independent operations

**6. Test Thoroughly**: Test each tool and the full chain

**7. Document Data Flow**: Show how data transforms through chain

**8. Version Chains**: Track changes to chain definitions

## Summary

Tool chaining enables complex agent behaviors through tool composition:

**Key Concepts**:
- Sequential: tools execute one after another
- Parallel: independent tools execute simultaneously
- Conditional: different chains based on conditions
- Iterative: repeat tools until condition met

**Patterns**:
- ETL pipelines for data processing
- Map-reduce for parallel computation
- Branch-merge for multi-perspective analysis
- Retry and fallback for resilience

**Best Practices**:
- Explicit dependencies and data flow
- Comprehensive error handling
- Optimization through parallelization
- Thorough testing of chains

Mastering tool chaining transforms individual tools into powerful, composable workflows.

## Next Steps

Continue exploring tool use:

- [Function Calling](function-calling.md) - Invoking tools correctly
- [Tool Selection](tool-selection.md) - Choosing the right tools for chains
- [Error Handling](error-handling.md) - Comprehensive error strategies
- [API Integration](api-integration.md) - Chaining API calls effectively
