# LangChain Agent Patterns

## Table of Contents

- [Introduction](#introduction)
- [ReAct Agent Pattern](#react-agent-pattern)
- [Agent Components](#agent-components)
- [Execution Flow](#execution-flow)
- [ReAct vs LangGraph](#react-vs-langgraph)
- [Best Practices](#best-practices)
- [Migration Guide](#migration-guide)
- [Summary](#summary)

## Introduction

LangChain Classic provides agent frameworks that autonomously decide which tools to use and when. The most common pattern is **ReAct** (Reasoning + Acting), which interleaves thought, action, and observation steps.

> "ReAct agents combine reasoning traces and task-specific actions to solve complex problems."

This guide covers practical patterns for building ReAct agents using LangChain Classic, understanding their execution model, and knowing when to migrate to LangGraph for more control.

### Agent vs. Workflow

**Agents** (LangChain Classic):
- LLM decides execution flow
- Built-in reasoning loop
- Black box decision making
- Limited observability

**Workflows** (LangGraph):
- Developer defines execution flow
- Explicit graph structure
- Full state visibility
- Checkpoint/replay capabilities

## ReAct Agent Pattern

The ReAct pattern enables LLMs to generate reasoning traces alongside actions, creating a synergistic approach to problem-solving.

### Basic Implementation

```python
import requests
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic import hub

model = ChatOpenAI()

# Pull pre-built ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")

# Built-in search tool
search_tool = DuckDuckGoSearchRun()

# Custom tool with @tool decorator
@tool
def get_weather_data(city: str) -> str:
    """
    This function fetches the current weather data for a given city
    """
    url = f"https://api.weatherstack.com/current"
    params = {
        "access_key": "YOUR_API_KEY",
        "query": city
    }
    response = requests.get(url, params=params)
    return response.json()

# Create ReAct agent
agent = create_react_agent(
    llm=model,
    tools=[search_tool, get_weather_data],
    prompt=prompt,
)

# Wrap in executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
)

# Execute query
response = agent_executor.invoke({
    "input": "What's the weather like in Paris?"
})

print(response["output"])
```

## Agent Components

### 1. LLM (Language Model)

The decision-making engine that reasons about tasks and selects tools.

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# OpenAI models
model = ChatOpenAI(model="gpt-4", temperature=0)

# Anthropic models
model = ChatAnthropic(model="claude-3-sonnet", temperature=0)
```

### 2. Tools

External capabilities the agent can invoke.

```python
from langchain.tools import tool

@tool
def calculate_shipping_cost(weight: float, distance: int) -> float:
    """
    Calculate shipping cost based on package weight (kg) and distance (km).
    """
    base_rate = 5.0
    weight_rate = 2.5 * weight
    distance_rate = 0.1 * distance
    return base_rate + weight_rate + distance_rate

@tool
def get_inventory_status(product_id: str) -> dict:
    """
    Get current inventory status for a product.
    """
    # Simulate database lookup
    inventory = {
        "PROD123": {"in_stock": 45, "warehouse": "NYC"},
        "PROD456": {"in_stock": 0, "warehouse": "LA"},
    }
    return inventory.get(product_id, {"in_stock": 0, "warehouse": "Unknown"})
```

### 3. Prompt Template

The ReAct prompt from LangChain Hub structures the reasoning loop.

```python
from langchain_classic import hub

# Pre-built ReAct prompt
prompt = hub.pull("hwchase17/react")

# Custom prompt structure (simplified)
"""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
"""
```

### 4. Agent Executor

Manages the execution loop, tool invocation, and output parsing.

```python
from langchain_classic.agents import AgentExecutor

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,           # Print reasoning steps
    max_iterations=5,       # Limit loop iterations
    handle_parsing_errors=True,  # Gracefully handle errors
    return_intermediate_steps=True,  # Return full reasoning trace
)
```

## Execution Flow

### Reasoning Loop

The agent follows this cycle until reaching a final answer:

```
┌─────────────────────────────────────────┐
│  1. Thought: Analyze the question       │
│     "I need to get weather data"        │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  2. Action: Select tool                 │
│     Action: get_weather_data            │
│     Action Input: "Paris"               │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  3. Observation: Tool result            │
│     {"temperature": 18, "conditions"..} │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  4. Thought: Evaluate observation       │
│     "I now have the information needed" │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  5. Final Answer: Synthesize response   │
│     "Weather in Paris: 18°C, cloudy"    │
└─────────────────────────────────────────┘
```

### Example Execution

```python
response = agent_executor.invoke({
    "input": "What's the weather like in Paris and how much to ship a 5kg package there from NYC?"
})
```

**Output:**

```
> Entering new AgentExecutor chain...

Thought: I need to get weather data for Paris and calculate shipping cost
Action: get_weather_data
Action Input: Paris
Observation: {"temperature": 18, "weather_descriptions": ["Partly cloudy"], "wind_speed": 15}

Thought: Now I need to calculate shipping cost for 5kg package from NYC to Paris (roughly 5,800 km)
Action: calculate_shipping_cost
Action Input: weight=5, distance=5800
Observation: 593.5

Thought: I now have all the information needed
Final Answer: The weather in Paris is currently 18°C and partly cloudy with 15 km/h wind. Shipping a 5kg package from NYC to Paris would cost approximately $593.50.

> Finished chain.
```

### Accessing Intermediate Steps

```python
response = agent_executor.invoke(
    {"input": "What's the weather in Tokyo?"},
    return_intermediate_steps=True
)

# Access reasoning trace
for step in response["intermediate_steps"]:
    action, observation = step
    print(f"Action: {action.tool}")
    print(f"Input: {action.tool_input}")
    print(f"Output: {observation}")
    print("---")
```

## ReAct vs LangGraph

Understanding when to use each approach:

| Aspect | Classic ReAct | LangGraph |
|--------|---------------|-----------|
| **Control** | Agent controls flow | Developer defines graph |
| **Debugging** | Black box execution | Inspect each node |
| **Loops** | Automatic reasoning loop | Explicit conditional edges |
| **State** | Internal agent state | TypedDict state management |
| **Recovery** | Limited retry logic | Full checkpoint/recovery |
| **Flexibility** | Fixed ReAct pattern | Custom workflow patterns |
| **Human-in-Loop** | Not built-in | Native interrupt() support |
| **Observability** | Verbose output only | State inspection, time travel |
| **Complexity** | Simple setup | More boilerplate |

### When to Use ReAct Agents

**Use Classic ReAct When:**
- Simple tool-calling loops
- Standard question-answering with tool access
- Minimal control requirements
- Quick prototyping
- Research/experimentation
- Single-session interactions

**Example Use Cases:**
- Research assistant with search + summarization
- Customer support with knowledge base lookup
- Data analysis with calculation tools
- Content generation with fact-checking

### When to Use LangGraph

**Use LangGraph When:**
- Complex multi-step workflows
- Custom decision logic beyond tool selection
- Human-in-the-loop requirements
- State persistence across sessions
- Production-grade error handling
- Multiple parallel execution paths
- Need for debugging/replay capabilities

**Example Use Cases:**
- Document processing pipelines
- Multi-stage approval workflows
- Long-running background tasks
- Stateful conversational agents
- Complex routing logic
- Enterprise integrations

## Best Practices

### 1. Tool Design

```python
# Good: Clear, focused tool
@tool
def get_product_price(product_id: str) -> float:
    """
    Get the current price for a product by its ID.
    Returns price in USD.
    """
    return database.query(f"SELECT price FROM products WHERE id = '{product_id}'")

# Avoid: Vague, multi-purpose tool
@tool
def do_product_stuff(action: str, data: dict) -> Any:
    """Does various product-related things."""  # Too vague!
    pass
```

### 2. Error Handling

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,              # Prevent infinite loops
    handle_parsing_errors=True,     # Graceful error handling
    early_stopping_method="generate" # Continue even if parsing fails
)
```

### 3. Prompt Engineering

```python
# Provide clear instructions in custom prompts
custom_instructions = """
You are a helpful assistant that answers questions about products.

IMPORTANT: 
- Always check inventory before suggesting products
- If a product is out of stock, suggest alternatives
- Include pricing information in your final answer
"""

# Combine with ReAct prompt
prompt = hub.pull("hwchase17/react")
prompt.messages[0].prompt.template = custom_instructions + "\n\n" + prompt.messages[0].prompt.template
```

### 4. Monitoring and Debugging

```python
# Enable verbose mode during development
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # See reasoning steps
    return_intermediate_steps=True  # Access full trace
)

# Log executions
import logging
logging.basicConfig(level=logging.INFO)

# Handle failures gracefully
try:
    response = agent_executor.invoke({"input": user_query})
except Exception as e:
    print(f"Agent failed: {e}")
    # Fallback to simpler response
```

## Migration Guide

### From ReAct to LangGraph

When you need more control, migrate to LangGraph:

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

# Define state
class AgentState(TypedDict):
    messages: list

# Define tools (same as before)
tools = [search_tool, get_weather_data]
model = ChatOpenAI()

# Agent node makes decisions
def agent_node(state: AgentState):
    messages = state["messages"]
    response = model.bind_tools(tools).invoke(messages)
    return {"messages": [response]}

# Tool execution node
tool_node = ToolNode(tools)

# Routing logic (explicit control)
def should_continue(state: AgentState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"

# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END}
)
graph.add_edge("tools", "agent")

# Compile
workflow = graph.compile()

# Execute with same interface
result = workflow.invoke({
    "messages": [("user", "What's the weather like in Paris?")]
})

print(result["messages"][-1].content)
```

**Benefits of LangGraph Version:**

1. **Explicit State Management**: See exactly what's stored
2. **Visible Control Flow**: Understand execution paths
3. **Easy Debugging**: Inspect each node independently
4. **Checkpoint/Replay**: Save and resume from any point
5. **Custom Routing Logic**: Beyond simple tool selection
6. **Production Ready**: Built-in error handling and monitoring

## Summary

LangChain Classic's ReAct agents provide a simple, powerful pattern for autonomous tool-calling:

**Key Takeaways:**

1. **ReAct Pattern**: Interleaves thought, action, and observation
2. **Components**: LLM + Tools + Prompt + AgentExecutor
3. **Execution**: Autonomous reasoning loop until final answer
4. **Use Cases**: Simple Q&A, research, prototyping
5. **Limitations**: Black box, limited control, no persistence
6. **Migration**: Move to LangGraph for complex workflows

**When to Use:**

- ✅ Quick prototypes and research
- ✅ Simple tool-calling scenarios
- ✅ Single-session interactions
- ❌ Complex multi-step workflows
- ❌ Human-in-the-loop requirements
- ❌ Production systems needing recovery

**Remember**: Start with ReAct for simplicity, migrate to LangGraph when you need control, observability, and production-grade features.

## Next Steps

- See [tool-definition-patterns.md](tool-definition-patterns.md) for advanced tool design
- See [../langgraph/graph-workflows.md](../langgraph/graph-workflows.md) for LangGraph migration
- Explore LangChain Hub for more agent prompts
- Experiment with different models and tool combinations
