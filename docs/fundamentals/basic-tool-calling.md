# Basic Tool Calling

## Table of Contents

- [Introduction](#introduction)
- [What is Tool Calling?](#what-is-tool-calling)
- [Tool Calling Architecture](#tool-calling-architecture)
- [Defining Tools](#defining-tools)
- [Tool Schemas](#tool-schemas)
- [Basic Tool Implementation](#basic-tool-implementation)
- [The Tool Calling Flow](#the-tool-calling-flow)
- [Parsing Tool Requests](#parsing-tool-requests)
- [Executing Tools](#executing-tools)
- [Returning Tool Results](#returning-tool-results)
- [Multiple Tool Calls](#multiple-tool-calls)
- [Tool Selection Strategies](#tool-selection-strategies)
- [Common Tool Patterns](#common-tool-patterns)
- [Tool Calling Failures](#tool-calling-failures)
- [Best Practices](#best-practices)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Tools are what give agents their superpowers. Without tools, a language model can only generate text. With tools, an agent can search the web, query databases, manipulate files, call APIs, and interact with the real world. Tool calling is the bridge between language and action.

At its core, tool calling is simple: the agent decides it needs to perform an action, requests a tool with specific parameters, and the system executes it. But this simplicity hides crucial details: how tools are defined, how the model learns about them, how requests are parsed, how execution is handled, and how results are fed back.

> "A language model without tools is a brain without hands."

This guide covers the fundamentals of tool calling: from defining your first tool to handling complex multi-tool workflows, from schemas to execution, and from basic patterns to common pitfalls.

## What is Tool Calling?

**Tool calling** (also known as function calling or action execution) is the mechanism by which language models request and execute external operations. It transforms models from text generators into action-taking agents.

### The Basic Concept

```
Agent: "I need to search for information"
       ↓
       [Generates tool request]
       ↓
       {
         "tool": "web_search",
         "parameters": {
           "query": "tool calling in LLMs",
           "limit": 5
         }
       }
       ↓
System: [Executes tool]
       ↓
       [Returns results]
       ↓
Agent: "Based on the search results..."
```

### Why Tool Calling?

Without tools:
- Agent can only reason and generate text
- No access to real-time information
- Can't perform actions
- Limited to training data knowledge

With tools:
- Access external data sources
- Perform computations
- Manipulate state
- Interact with APIs and systems
- Ground responses in reality

### Tool vs. No Tool

**Without tools**:

```python
user: "What's the weather in Tokyo?"
agent: "I don't have access to real-time weather data. 
        As of my training cutoff, I cannot provide 
        current weather information."
```

**With tools**:

```python
user: "What's the weather in Tokyo?"
agent: [calls: get_weather(city="Tokyo")]
system: {"temp": 18, "condition": "cloudy", "humidity": 65}
agent: "The current weather in Tokyo is cloudy with 
        a temperature of 18°C and 65% humidity."
```

## Tool Calling Architecture

Understanding the architecture helps you build robust tool-calling systems.

### High-Level Architecture

```
┌──────────────────────────────────────────────────────┐
│                    USER INPUT                         │
└─────────────────────┬────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────┐
│                 LANGUAGE MODEL                        │
│  - Understands available tools                        │
│  - Decides when to use them                          │
│  - Generates tool requests                           │
└─────────────────────┬────────────────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │ Tool Request? │
              └───┬───────┬───┘
                  │ Yes   │ No
                  │       │
                  ▼       ▼
        ┌──────────────┐  ┌──────────────┐
        │ TOOL PARSER  │  │   RESPONSE   │
        └──────┬───────┘  └──────────────┘
               │
               ▼
        ┌──────────────┐
        │ VALIDATION   │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │  EXECUTION   │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │   RESULTS    │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │ BACK TO LLM  │
        └──────────────┘
```

### Component Breakdown

**1. Tool Registry**

Stores tool definitions and metadata:

```python
{
  "web_search": {
    "description": "Search the web",
    "parameters": {...},
    "function": web_search_impl
  },
  "calculator": {
    "description": "Perform calculations",
    "parameters": {...},
    "function": calculator_impl
  }
}
```

**2. Tool Prompt Generator**

Converts tool definitions into LLM-readable format:

```python
Available tools:
1. web_search(query: str, limit: int = 10)
   - Search the web for information
   
2. calculator(expression: str)
   - Evaluate mathematical expressions
```

**3. Request Parser**

Extracts tool calls from LLM output:

```python
# Input: LLM output with tool call
# Output: Structured tool request
{
  "tool_name": "web_search",
  "arguments": {"query": "...", "limit": 5}
}
```

**4. Validator**

Ensures tool requests are valid:

```python
- Tool exists?
- Required parameters present?
- Parameter types correct?
- Values within constraints?
```

**5. Executor**

Runs the actual tool:

```python
result = tools[tool_name](**arguments)
```

**6. Result Formatter**

Formats results for the LLM:

```python
# Convert tool output to LLM-readable format
"Tool 'web_search' returned: [results...]"
```

## Defining Tools

How you define tools determines how well your agent can use them. Good tool definitions are clear, specific, and well-documented.

### Basic Tool Definition

**Python function as tool**:

```python
def get_weather(city: str, units: str = "celsius") -> dict:
    """
    Get current weather for a city.
    
    Args:
        city: Name of the city
        units: Temperature units ("celsius" or "fahrenheit")
        
    Returns:
        Dictionary with weather data
    """
    # Implementation
    return {
        "temperature": 22,
        "condition": "sunny",
        "humidity": 45
    }
```

### Tool Metadata

```python
tool_definition = {
    "name": "get_weather",
    "description": "Get current weather information for a specified city",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city name (e.g., 'Tokyo', 'New York')"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature units",
                "default": "celsius"
            }
        },
        "required": ["city"]
    },
    "function": get_weather  # The actual implementation
}
```

### Tool Description Best Practices

**Bad description**:

```python
"description": "Gets weather"
```

**Good description**:

```python
"description": "Retrieves current weather conditions including temperature, "
               "humidity, and general conditions for a specified city. "
               "Returns real-time data from weather APIs."
```

**Why it matters**: The LLM uses descriptions to decide when and how to use tools. Clear descriptions lead to better tool selection.

### Parameter Descriptions

```python
"parameters": {
    "type": "object",
    "properties": {
        "city": {
            "type": "string",
            "description": "City name. Use common English names "
                          "(e.g., 'Tokyo' not '東京'). "
                          "Include country for ambiguous names "
                          "(e.g., 'Paris, France' vs 'Paris, Texas')."
        },
        "units": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "Temperature units for the response. "
                          "Defaults to celsius if not specified."
        }
    }
}
```

### Tool Naming Conventions

**Good names**:

- `get_weather` - Clear verb + noun
- `search_documents` - Action + target
- `calculate_distance` - Specific purpose
- `send_email` - Simple and direct

**Avoid**:

- `tool1` - No semantic meaning
- `do_thing` - Too vague
- `getWeatherDataFromAPI` - Too verbose
- `wthr` - Unclear abbreviation

## Tool Schemas

Tool schemas define the structure of tool calls. Most modern systems use JSON Schema, OpenAI's function calling format, or similar standards.

### JSON Schema Format

```python
{
    "type": "function",
    "function": {
        "name": "search_database",
        "description": "Search a database with a query",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL-like query string"
                },
                "database": {
                    "type": "string",
                    "enum": ["users", "products", "orders"],
                    "description": "Which database to query"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10,
                    "description": "Maximum number of results"
                }
            },
            "required": ["query", "database"]
        }
    }
}
```

### Type System

**Basic types**:

```python
"string"    # Text
"integer"   # Whole numbers
"number"    # Floating point
"boolean"   # true/false
"object"    # Nested structure
"array"     # List of items
```

**Advanced constraints**:

```python
{
    "type": "string",
    "minLength": 1,
    "maxLength": 100,
    "pattern": "^[a-zA-Z0-9_]+$"
}

{
    "type": "integer",
    "minimum": 0,
    "maximum": 100,
    "multipleOf": 5
}

{
    "type": "array",
    "items": {"type": "string"},
    "minItems": 1,
    "maxItems": 10,
    "uniqueItems": true
}
```

### Complex Tool Schemas

**Tool with nested objects**:

```python
{
    "name": "create_user",
    "parameters": {
        "type": "object",
        "properties": {
            "username": {"type": "string"},
            "email": {"type": "string", "format": "email"},
            "profile": {
                "type": "object",
                "properties": {
                    "first_name": {"type": "string"},
                    "last_name": {"type": "string"},
                    "age": {"type": "integer", "minimum": 0}
                },
                "required": ["first_name", "last_name"]
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["username", "email"]
    }
}
```

### Schema Validation

```python
import jsonschema

def validate_tool_call(tool_call, schema):
    """Validate a tool call against its schema."""
    try:
        jsonschema.validate(
            instance=tool_call["arguments"],
            schema=schema["parameters"]
        )
        return True, None
    except jsonschema.ValidationError as e:
        return False, str(e)

# Example usage
tool_call = {
    "name": "get_weather",
    "arguments": {"city": "Tokyo", "units": "celsius"}
}

valid, error = validate_tool_call(tool_call, weather_schema)
if not valid:
    print(f"Invalid tool call: {error}")
```

## Basic Tool Implementation

Let's build a complete tool calling system from scratch.

### Step 1: Define Tools

```python
from typing import Dict, Any, Callable, List
import json

class Tool:
    """Represents a callable tool."""
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        function: Callable
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
    
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments."""
        return self.function(**kwargs)


# Example tools
def calculator(expression: str) -> float:
    """Safely evaluate a mathematical expression."""
    try:
        # In production, use a safe eval library
        result = eval(expression, {"__builtins__": {}}, {})
        return float(result)
    except Exception as e:
        return f"Error: {str(e)}"


def web_search(query: str, limit: int = 5) -> List[Dict[str, str]]:
    """Simulate a web search."""
    # In production, call actual search API
    return [
        {"title": f"Result {i}", "url": f"http://example.com/{i}"}
        for i in range(limit)
    ]


# Register tools
TOOLS = {
    "calculator": Tool(
        name="calculator",
        description="Evaluate mathematical expressions",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        },
        function=calculator
    ),
    "web_search": Tool(
        name="web_search",
        description="Search the web for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of results (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        },
        function=web_search
    )
}
```

### Step 2: Generate Tool Descriptions

```python
def generate_tool_prompt(tools: Dict[str, Tool]) -> str:
    """Generate a prompt describing available tools."""
    prompt = "You have access to the following tools:\n\n"
    
    for tool_name, tool in tools.items():
        prompt += f"Tool: {tool.name}\n"
        prompt += f"Description: {tool.description}\n"
        prompt += f"Parameters:\n"
        
        params = tool.parameters.get("properties", {})
        required = tool.parameters.get("required", [])
        
        for param_name, param_info in params.items():
            req_marker = " (required)" if param_name in required else ""
            prompt += f"  - {param_name}{req_marker}: "
            prompt += f"{param_info.get('description', 'No description')}\n"
        
        prompt += "\n"
    
    prompt += """To use a tool, respond with:
TOOL_CALL: {"name": "tool_name", "arguments": {"param": "value"}}

You can make multiple tool calls by including multiple TOOL_CALL lines.
After tool execution, you'll receive the results.
"""
    
    return prompt
```

### Step 3: Parse Tool Calls

```python
import re

def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract tool calls from LLM output."""
    tool_calls = []
    
    # Find all TOOL_CALL: {...} patterns
    pattern = r'TOOL_CALL:\s*(\{[^}]+\})'
    matches = re.finditer(pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            call_data = json.loads(match.group(1))
            tool_calls.append(call_data)
        except json.JSONDecodeError as e:
            print(f"Failed to parse tool call: {e}")
    
    return tool_calls


# Example
output = """
I need to search for information and do a calculation.

TOOL_CALL: {"name": "web_search", "arguments": {"query": "AI agents", "limit": 3}}
TOOL_CALL: {"name": "calculator", "arguments": {"expression": "42 * 2"}}

After getting these results, I'll provide an answer.
"""

calls = parse_tool_calls(output)
print(calls)
# [
#     {"name": "web_search", "arguments": {"query": "AI agents", "limit": 3}},
#     {"name": "calculator", "arguments": {"expression": "42 * 2"}}
# ]
```

### Step 4: Execute Tools

```python
def execute_tool_call(
    tool_call: Dict[str, Any],
    tools: Dict[str, Tool]
) -> Dict[str, Any]:
    """Execute a single tool call."""
    tool_name = tool_call.get("name")
    arguments = tool_call.get("arguments", {})
    
    # Validate tool exists
    if tool_name not in tools:
        return {
            "success": False,
            "error": f"Tool '{tool_name}' not found"
        }
    
    tool = tools[tool_name]
    
    # Validate arguments (simplified)
    required = tool.parameters.get("required", [])
    for req_param in required:
        if req_param not in arguments:
            return {
                "success": False,
                "error": f"Missing required parameter: {req_param}"
            }
    
    # Execute
    try:
        result = tool.execute(**arguments)
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Execution error: {str(e)}"
        }


def execute_all_tool_calls(
    tool_calls: List[Dict[str, Any]],
    tools: Dict[str, Tool]
) -> List[Dict[str, Any]]:
    """Execute multiple tool calls."""
    results = []
    for call in tool_calls:
        result = execute_tool_call(call, tools)
        results.append({
            "tool": call["name"],
            "arguments": call["arguments"],
            **result
        })
    return results
```

### Step 5: Format Results

```python
def format_tool_results(results: List[Dict[str, Any]]) -> str:
    """Format tool results for LLM consumption."""
    formatted = "Tool execution results:\n\n"
    
    for i, result in enumerate(results, 1):
        formatted += f"[{i}] Tool: {result['tool']}\n"
        formatted += f"    Arguments: {result['arguments']}\n"
        
        if result["success"]:
            formatted += f"    Result: {result['result']}\n"
        else:
            formatted += f"    Error: {result['error']}\n"
        
        formatted += "\n"
    
    return formatted
```

### Complete Example

```python
def agent_with_tools(user_input: str, tools: Dict[str, Tool]) -> str:
    """Simple agent with tool calling."""
    
    # Build system prompt with tools
    system_prompt = generate_tool_prompt(tools)
    
    # Simulate LLM call (in practice, call actual LLM API)
    prompt = f"{system_prompt}\n\nUser: {user_input}\nAssistant:"
    
    # Get LLM response (simulated)
    llm_response = """
I'll search for information and calculate the result.

TOOL_CALL: {"name": "web_search", "arguments": {"query": "AI agents"}}
TOOL_CALL: {"name": "calculator", "arguments": {"expression": "100 / 4"}}
"""
    
    # Parse tool calls
    tool_calls = parse_tool_calls(llm_response)
    
    if not tool_calls:
        return llm_response
    
    # Execute tools
    results = execute_all_tool_calls(tool_calls, tools)
    
    # Format results
    tool_output = format_tool_results(results)
    
    # Send back to LLM (in practice, this continues the conversation)
    print("Tool Results:")
    print(tool_output)
    
    return "Agent would continue with tool results..."


# Usage
result = agent_with_tools("Search for AI agents and divide 100 by 4", TOOLS)
```

## The Tool Calling Flow

Understanding the complete flow helps debug issues and optimize performance.

### End-to-End Flow

```
1. USER REQUEST
   ↓
   "Find weather in Tokyo and calculate 15 + 27"
   
2. SYSTEM PROMPT + TOOLS
   ↓
   [Tools descriptions added to context]
   
3. LLM REASONING
   ↓
   "I need to use get_weather and calculator tools"
   
4. LLM GENERATES TOOL CALLS
   ↓
   TOOL_CALL: {"name": "get_weather", "arguments": {"city": "Tokyo"}}
   TOOL_CALL: {"name": "calculator", "arguments": {"expression": "15 + 27"}}
   
5. PARSER EXTRACTS CALLS
   ↓
   [List of 2 tool calls]
   
6. VALIDATOR CHECKS CALLS
   ↓
   [Both valid]
   
7. EXECUTOR RUNS TOOLS
   ↓
   Weather: {"temp": 18, "condition": "cloudy"}
   Calculator: 42
   
8. FORMATTER PREPARES RESULTS
   ↓
   "Tool results: 1. Weather in Tokyo: 18°C, cloudy..."
   
9. RESULTS SENT TO LLM
   ↓
   [Results added to conversation]
   
10. LLM FINAL RESPONSE
    ↓
    "The weather in Tokyo is 18°C and cloudy. 15 + 27 equals 42."
    
11. USER RECEIVES ANSWER
```

### Detailed Flow with State

```python
class ToolCallingAgent:
    """Agent with detailed tool calling flow."""
    
    def __init__(self, tools: Dict[str, Tool]):
        self.tools = tools
        self.conversation_history = []
    
    def run(self, user_input: str, max_iterations: int = 5) -> str:
        """Run agent with tool calling support."""
        
        # Add user message
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Step 1: Build prompt
            prompt = self._build_prompt()
            print(f"Prompt length: {len(prompt)} chars")
            
            # Step 2: Call LLM
            response = self._call_llm(prompt)
            print(f"LLM response: {response[:100]}...")
            
            # Step 3: Parse tool calls
            tool_calls = parse_tool_calls(response)
            print(f"Found {len(tool_calls)} tool calls")
            
            # Step 4: If no tool calls, return response
            if not tool_calls:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                return response
            
            # Step 5: Execute tools
            print("Executing tools...")
            results = execute_all_tool_calls(tool_calls, self.tools)
            
            # Step 6: Format and add results
            tool_output = format_tool_results(results)
            print(f"Tool results: {tool_output[:100]}...")
            
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            self.conversation_history.append({
                "role": "system",
                "content": tool_output
            })
        
        return "Max iterations reached"
    
    def _build_prompt(self) -> str:
        """Build prompt from conversation history."""
        prompt = generate_tool_prompt(self.tools)
        for msg in self.conversation_history:
            prompt += f"\n{msg['role']}: {msg['content']}"
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM (placeholder)."""
        # In production, call actual LLM API
        return "Simulated response with tool call"
```

## Parsing Tool Requests

Parsing is critical - if you can't extract tool calls reliably, your agent breaks.

### Format Options

**1. JSON Format**

```python
TOOL_CALL: {"name": "search", "arguments": {"query": "test"}}
```

**2. XML Format**

```python
<tool_call>
  <name>search</name>
  <arguments>
    <query>test</query>
  </arguments>
</tool_call>
```

**3. Structured Format**

```python
FUNCTION: search
ARGUMENTS:
  query: test
  limit: 5
```

### Robust Parsing

```python
import json
import re
from typing import Optional, List, Dict, Any

def robust_parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse tool calls with multiple fallback strategies."""
    
    # Strategy 1: JSON with TOOL_CALL prefix
    calls = []
    
    # Try JSON format
    pattern = r'TOOL_CALL:\s*(\{[^}]+\})'
    for match in re.finditer(pattern, text, re.DOTALL):
        try:
            call = json.loads(match.group(1))
            if "name" in call:
                calls.append(call)
        except json.JSONDecodeError:
            pass
    
    if calls:
        return calls
    
    # Strategy 2: Try to find JSON objects anywhere
    json_pattern = r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}[^{}]*\}'
    for match in re.finditer(json_pattern, text):
        try:
            call = json.loads(match.group(0))
            calls.append(call)
        except json.JSONDecodeError:
            pass
    
    if calls:
        return calls
    
    # Strategy 3: Parse structured format
    structured_pattern = r'FUNCTION:\s*(\w+)\s*ARGUMENTS:\s*\{([^}]+)\}'
    for match in re.finditer(structured_pattern, text, re.DOTALL):
        try:
            name = match.group(1)
            args_str = match.group(2)
            arguments = {}
            
            for line in args_str.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    arguments[key.strip()] = value.strip()
            
            calls.append({"name": name, "arguments": arguments})
        except Exception:
            pass
    
    return calls


def extract_json_objects(text: str) -> List[Dict[str, Any]]:
    """Extract all valid JSON objects from text."""
    objects = []
    depth = 0
    start = None
    
    for i, char in enumerate(text):
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start:i+1])
                    objects.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None
    
    return objects
```

### Handling Malformed Calls

```python
def fix_common_json_errors(json_str: str) -> Optional[str]:
    """Attempt to fix common JSON formatting errors."""
    
    # Fix single quotes
    json_str = json_str.replace("'", '"')
    
    # Fix trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Fix unquoted keys
    json_str = re.sub(r'(\w+):', r'"\1":', json_str)
    
    # Fix unquoted values (careful with this)
    # json_str = re.sub(r':\s*([^",\[\{}\]]+)([,}\]])', r': "\1"\2', json_str)
    
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        return None


def parse_with_recovery(text: str) -> List[Dict[str, Any]]:
    """Parse tool calls with error recovery."""
    
    # Try normal parsing
    calls = robust_parse_tool_calls(text)
    if calls:
        return calls
    
    # Try to fix and parse
    pattern = r'TOOL_CALL:\s*(\{[^}]+\})'
    for match in re.finditer(pattern, text, re.DOTALL):
        json_str = match.group(1)
        fixed = fix_common_json_errors(json_str)
        
        if fixed:
            try:
                call = json.loads(fixed)
                calls.append(call)
            except json.JSONDecodeError:
                pass
    
    return calls
```

## Executing Tools

Execution must be safe, reliable, and observable.

### Safe Execution

```python
import asyncio
import signal
from contextlib import contextmanager
import time

class TimeoutError(Exception):
    """Tool execution timeout."""
    pass


@contextmanager
def time_limit(seconds: int):
    """Context manager for timing out code."""
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def safe_execute_tool(
    tool: Tool,
    arguments: Dict[str, Any],
    timeout: int = 30
) -> Dict[str, Any]:
    """Execute tool with safety measures."""
    
    start_time = time.time()
    
    try:
        # Validate arguments
        required = tool.parameters.get("required", [])
        for param in required:
            if param not in arguments:
                return {
                    "success": False,
                    "error": f"Missing required parameter: {param}"
                }
        
        # Execute with timeout
        with time_limit(timeout):
            result = tool.execute(**arguments)
        
        execution_time = time.time() - start_time
        
        return {
            "success": True,
            "result": result,
            "execution_time": execution_time
        }
        
    except TimeoutError:
        return {
            "success": False,
            "error": f"Tool execution timed out after {timeout}s"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Execution error: {type(e).__name__}: {str(e)}"
        }
```

### Async Tool Execution

```python
async def async_execute_tool(
    tool: Tool,
    arguments: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute tool asynchronously."""
    
    try:
        # If tool function is async
        if asyncio.iscoroutinefunction(tool.function):
            result = await tool.function(**arguments)
        else:
            # Run sync function in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: tool.function(**arguments)
            )
        
        return {"success": True, "result": result}
        
    except Exception as e:
        return {"success": False, "error": str(e)}


async def execute_tools_parallel(
    tool_calls: List[Dict[str, Any]],
    tools: Dict[str, Tool]
) -> List[Dict[str, Any]]:
    """Execute multiple tools in parallel."""
    
    tasks = []
    for call in tool_calls:
        tool_name = call["name"]
        if tool_name in tools:
            task = async_execute_tool(
                tools[tool_name],
                call["arguments"]
            )
            tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

### Execution Logging

```python
class ToolExecutor:
    """Tool executor with comprehensive logging."""
    
    def __init__(self, tools: Dict[str, Tool]):
        self.tools = tools
        self.execution_log = []
    
    def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute tool with logging."""
        
        log_entry = {
            "timestamp": time.time(),
            "tool": tool_name,
            "arguments": arguments,
            "success": False,
            "result": None,
            "error": None,
            "duration": 0
        }
        
        start_time = time.time()
        
        try:
            if tool_name not in self.tools:
                raise ValueError(f"Tool '{tool_name}' not found")
            
            tool = self.tools[tool_name]
            result = tool.execute(**arguments)
            
            log_entry["success"] = True
            log_entry["result"] = result
            
        except Exception as e:
            log_entry["error"] = str(e)
        
        finally:
            log_entry["duration"] = time.time() - start_time
            self.execution_log.append(log_entry)
        
        return log_entry
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        total = len(self.execution_log)
        successful = sum(1 for log in self.execution_log if log["success"])
        
        return {
            "total_executions": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / total if total > 0 else 0,
            "tools_used": list(set(log["tool"] for log in self.execution_log)),
            "avg_duration": sum(log["duration"] for log in self.execution_log) / total if total > 0 else 0
        }
```

## Returning Tool Results

How you return results affects the agent's ability to use them effectively.

### Result Format Options

**1. Simple String**

```python
"Tool 'web_search' returned 5 results about AI agents."
```

**2. Structured Format**

```python
"""
TOOL RESULT: web_search
STATUS: Success
RESULTS:
- Result 1: ...
- Result 2: ...
"""
```

**3. JSON Format**

```python
{
    "tool": "web_search",
    "status": "success",
    "data": [...]
}
```

### Smart Result Formatting

```python
def format_result_intelligently(
    tool_name: str,
    result: Any,
    max_length: int = 1000
) -> str:
    """Format tool result based on content type."""
    
    # Handle different result types
    if isinstance(result, dict):
        return format_dict_result(tool_name, result, max_length)
    elif isinstance(result, list):
        return format_list_result(tool_name, result, max_length)
    elif isinstance(result, str):
        return format_string_result(tool_name, result, max_length)
    else:
        return f"Tool '{tool_name}' returned: {str(result)}"


def format_dict_result(
    tool_name: str,
    result: dict,
    max_length: int
) -> str:
    """Format dictionary results."""
    formatted = f"Tool '{tool_name}' results:\n"
    
    for key, value in result.items():
        formatted += f"  {key}: {value}\n"
    
    if len(formatted) > max_length:
        formatted = formatted[:max_length] + f"\n... (truncated)"
    
    return formatted


def format_list_result(
    tool_name: str,
    result: list,
    max_length: int
) -> str:
    """Format list results."""
    formatted = f"Tool '{tool_name}' found {len(result)} results:\n\n"
    
    for i, item in enumerate(result[:10], 1):  # Limit to 10 items
        if isinstance(item, dict):
            formatted += f"[{i}] "
            formatted += ", ".join(f"{k}: {v}" for k, v in item.items())
            formatted += "\n"
        else:
            formatted += f"[{i}] {item}\n"
    
    if len(result) > 10:
        formatted += f"\n... and {len(result) - 10} more results"
    
    return formatted


def format_string_result(
    tool_name: str,
    result: str,
    max_length: int
) -> str:
    """Format string results."""
    if len(result) <= max_length:
        return f"Tool '{tool_name}' returned:\n{result}"
    else:
        return f"Tool '{tool_name}' returned:\n{result[:max_length]}\n... (truncated)"
```

### Handling Large Results

```python
def truncate_result(result: Any, max_size: int = 5000) -> Any:
    """Truncate large results intelligently."""
    
    result_str = str(result)
    
    if len(result_str) <= max_size:
        return result
    
    # For lists, truncate number of items
    if isinstance(result, list):
        items_kept = []
        current_size = 0
        
        for item in result:
            item_size = len(str(item))
            if current_size + item_size > max_size:
                break
            items_kept.append(item)
            current_size += item_size
        
        return items_kept + [f"... ({len(result) - len(items_kept)} more items)"]
    
    # For strings, truncate with context
    if isinstance(result, str):
        return result[:max_size] + f"\n... (truncated {len(result) - max_size} chars)"
    
    # For dicts, keep most important keys
    if isinstance(result, dict):
        truncated = {}
        current_size = 0
        
        for key, value in result.items():
            item_size = len(str(key)) + len(str(value))
            if current_size + item_size > max_size:
                truncated["_truncated"] = f"{len(result) - len(truncated)} keys omitted"
                break
            truncated[key] = value
            current_size += item_size
        
        return truncated
    
    return result
```

## Multiple Tool Calls

Agents often need to call multiple tools in sequence or parallel.

### Sequential Tool Calls

```python
def execute_sequential(
    tool_calls: List[Dict[str, Any]],
    tools: Dict[str, Tool]
) -> List[Dict[str, Any]]:
    """Execute tools one after another."""
    
    results = []
    
    for i, call in enumerate(tool_calls):
        print(f"Executing tool {i+1}/{len(tool_calls)}: {call['name']}")
        
        result = execute_tool_call(call, tools)
        results.append(result)
        
        # Stop on error if desired
        if not result.get("success", False):
            print(f"Tool {call['name']} failed, stopping execution")
            break
    
    return results
```

### Parallel Tool Calls

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def execute_parallel(
    tool_calls: List[Dict[str, Any]],
    tools: Dict[str, Tool],
    max_workers: int = 5
) -> List[Dict[str, Any]]:
    """Execute tools in parallel."""
    
    results = [None] * len(tool_calls)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(execute_tool_call, call, tools): i
            for i, call in enumerate(tool_calls)
        }
        
        # Collect results
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as e:
                results[index] = {
                    "success": False,
                    "error": f"Execution exception: {str(e)}"
                }
    
    return results
```

### Dependency-Based Execution

```python
class DependencyExecutor:
    """Execute tools respecting dependencies."""
    
    def __init__(self, tools: Dict[str, Tool]):
        self.tools = tools
    
    def execute_with_dependencies(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute tools, automatically detecting dependencies.
        If tool B uses output from tool A, execute A first.
        """
        
        # Build dependency graph
        dependencies = self._build_dependency_graph(tool_calls)
        
        # Topological sort
        execution_order = self._topological_sort(dependencies)
        
        # Execute in order
        results = {}
        final_results = []
        
        for call_id in execution_order:
            call = tool_calls[call_id]
            
            # Replace references to previous results
            arguments = self._resolve_arguments(
                call["arguments"],
                results
            )
            
            # Execute
            result = execute_tool_call(
                {"name": call["name"], "arguments": arguments},
                self.tools
            )
            
            results[call_id] = result
            final_results.append(result)
        
        return final_results
    
    def _build_dependency_graph(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> Dict[int, List[int]]:
        """Build dependency graph between tool calls."""
        # Implementation depends on how you reference previous results
        # E.g., arguments like {"input": "$result_0"}
        pass
    
    def _topological_sort(
        self,
        graph: Dict[int, List[int]]
    ) -> List[int]:
        """Topological sort of dependency graph."""
        # Standard topological sort implementation
        pass
    
    def _resolve_arguments(
        self,
        arguments: Dict[str, Any],
        results: Dict[int, Any]
    ) -> Dict[str, Any]:
        """Replace references with actual results."""
        resolved = {}
        for key, value in arguments.items():
            if isinstance(value, str) and value.startswith("$result_"):
                result_id = int(value.split("_")[1])
                resolved[key] = results[result_id]["result"]
            else:
                resolved[key] = value
        return resolved
```

## Tool Selection Strategies

How the agent decides which tools to use.

### Rule-Based Selection

```python
def select_tool_by_rules(
    query: str,
    tools: Dict[str, Tool]
) -> Optional[str]:
    """Select tool using keyword matching."""
    
    query_lower = query.lower()
    
    # Define keyword mappings
    keyword_map = {
        "calculator": ["calculate", "compute", "math", "add", "subtract"],
        "web_search": ["search", "find", "look up", "google"],
        "get_weather": ["weather", "temperature", "forecast"],
        "send_email": ["email", "send message", "notify"]
    }
    
    # Find matching tools
    matches = []
    for tool_name, keywords in keyword_map.items():
        if any(kw in query_lower for kw in keywords):
            matches.append(tool_name)
    
    return matches[0] if matches else None
```

### LLM-Based Selection

This is the typical approach - the LLM itself decides which tools to use based on tool descriptions.

```python
def llm_select_tools(
    query: str,
    tools: Dict[str, Tool],
    llm_call: Callable
) -> List[str]:
    """Let LLM select appropriate tools."""
    
    tool_list = "\n".join([
        f"- {name}: {tool.description}"
        for name, tool in tools.items()
    ])
    
    prompt = f"""
Given the following tools:
{tool_list}

And the user query: "{query}"

Which tools are needed? List tool names, one per line.
If no tools are needed, respond with "NONE".
"""
    
    response = llm_call(prompt)
    
    # Parse tool names
    selected = []
    for line in response.strip().split('\n'):
        tool_name = line.strip('- ').strip()
        if tool_name in tools:
            selected.append(tool_name)
    
    return selected
```

### Scoring-Based Selection

```python
def score_tool_relevance(
    query: str,
    tool: Tool
) -> float:
    """Score how relevant a tool is to a query."""
    
    score = 0.0
    query_lower = query.lower()
    
    # Check name match
    if tool.name.lower() in query_lower:
        score += 3.0
    
    # Check description match
    description_words = set(tool.description.lower().split())
    query_words = set(query_lower.split())
    common_words = description_words & query_words
    score += len(common_words) * 0.5
    
    # Check parameter names
    params = tool.parameters.get("properties", {})
    for param_name in params.keys():
        if param_name.lower() in query_lower:
            score += 1.0
    
    return score


def select_tools_by_scoring(
    query: str,
    tools: Dict[str, Tool],
    threshold: float = 1.0,
    max_tools: int = 3
) -> List[str]:
    """Select tools by relevance scoring."""
    
    scores = {}
    for name, tool in tools.items():
        scores[name] = score_tool_relevance(query, tool)
    
    # Sort by score
    sorted_tools = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Filter by threshold and limit
    selected = [
        name for name, score in sorted_tools
        if score >= threshold
    ][:max_tools]
    
    return selected
```

## Common Tool Patterns

Reusable patterns for tool calling.

### Search-and-Process Pattern

```python
def search_and_process_pattern(
    search_query: str,
    tools: Dict[str, Tool]
) -> str:
    """Search for information, then process it."""
    
    # Step 1: Search
    search_result = tools["web_search"].execute(
        query=search_query,
        limit=5
    )
    
    # Step 2: Process each result
    summaries = []
    for item in search_result:
        summary = tools["summarize"].execute(
            text=item["content"],
            max_length=100
        )
        summaries.append(summary)
    
    # Step 3: Combine
    final_summary = tools["combine_texts"].execute(
        texts=summaries
    )
    
    return final_summary
```

### Validate-Execute-Verify Pattern

```python
def validate_execute_verify_pattern(
    tool_name: str,
    arguments: Dict[str, Any],
    tools: Dict[str, Tool]
) -> Dict[str, Any]:
    """Validate before execution, verify after."""
    
    # Step 1: Validate
    validation_result = tools["validate_input"].execute(
        tool=tool_name,
        arguments=arguments
    )
    
    if not validation_result["valid"]:
        return {
            "success": False,
            "error": f"Validation failed: {validation_result['error']}"
        }
    
    # Step 2: Execute
    result = tools[tool_name].execute(**arguments)
    
    # Step 3: Verify
    verification = tools["verify_result"].execute(
        tool=tool_name,
        result=result
    )
    
    if not verification["valid"]:
        return {
            "success": False,
            "error": "Result verification failed",
            "result": result
        }
    
    return {
        "success": True,
        "result": result
    }
```

### Fallback Pattern

```python
def tool_with_fallback(
    primary_tool: str,
    fallback_tool: str,
    arguments: Dict[str, Any],
    tools: Dict[str, Tool]
) -> Dict[str, Any]:
    """Try primary tool, fall back to alternative on failure."""
    
    # Try primary
    result = execute_tool_call(
        {"name": primary_tool, "arguments": arguments},
        tools
    )
    
    if result["success"]:
        return result
    
    # Try fallback
    print(f"{primary_tool} failed, trying {fallback_tool}")
    return execute_tool_call(
        {"name": fallback_tool, "arguments": arguments},
        tools
    )
```

### Retry Pattern

```python
def tool_with_retry(
    tool_name: str,
    arguments: Dict[str, Any],
    tools: Dict[str, Tool],
    max_retries: int = 3,
    backoff: float = 1.0
) -> Dict[str, Any]:
    """Execute tool with exponential backoff retry."""
    
    for attempt in range(max_retries):
        result = execute_tool_call(
            {"name": tool_name, "arguments": arguments},
            tools
        )
        
        if result["success"]:
            return result
        
        if attempt < max_retries - 1:
            wait_time = backoff * (2 ** attempt)
            print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s")
            time.sleep(wait_time)
    
    return {
        "success": False,
        "error": f"Failed after {max_retries} attempts"
    }
```

## Tool Calling Failures

Understanding and handling failures is crucial for reliability.

### Common Failure Modes

**1. Parsing Failures**

```python
# LLM output doesn't match expected format
"I'll search for that: web_search('AI agents')"  # Wrong format!
# Expected: TOOL_CALL: {"name": "web_search", "arguments": {"query": "AI agents"}}
```

**2. Invalid Tool Names**

```python
TOOL_CALL: {"name": "websearch", "arguments": {...}}  # Should be "web_search"
```

**3. Missing Required Parameters**

```python
TOOL_CALL: {"name": "web_search", "arguments": {}}  # Missing "query"
```

**4. Type Mismatches**

```python
TOOL_CALL: {"name": "calculator", "arguments": {"expression": 42}}  # Should be string
```

**5. Execution Errors**

```python
# Tool executes but hits an error
TOOL_CALL: {"name": "divide", "arguments": {"a": 10, "b": 0}}  # Division by zero
```

### Handling Failures

```python
class RobustToolExecutor:
    """Tool executor with comprehensive error handling."""
    
    def __init__(self, tools: Dict[str, Tool]):
        self.tools = tools
        self.error_counts = {}
    
    def execute_with_error_handling(
        self,
        tool_call: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute tool with detailed error handling."""
        
        # Extract tool info
        tool_name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})
        
        # Track errors
        error_key = tool_name or "unknown"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0)
        
        # Handle missing tool name
        if not tool_name:
            return self._error_response(
                "Missing tool name in call",
                error_type="missing_tool_name"
            )
        
        # Handle unknown tool
        if tool_name not in self.tools:
            suggestion = self._suggest_tool(tool_name)
            error_msg = f"Unknown tool: {tool_name}"
            if suggestion:
                error_msg += f". Did you mean '{suggestion}'?"
            
            return self._error_response(
                error_msg,
                error_type="unknown_tool"
            )
        
        tool = self.tools[tool_name]
        
        # Validate arguments
        validation_error = self._validate_arguments(tool, arguments)
        if validation_error:
            return self._error_response(
                validation_error,
                error_type="invalid_arguments"
            )
        
        # Execute with exception handling
        try:
            result = tool.execute(**arguments)
            self.error_counts[error_key] = 0  # Reset on success
            
            return {
                "success": True,
                "result": result
            }
            
        except Exception as e:
            self.error_counts[error_key] += 1
            
            return self._error_response(
                f"Execution error: {type(e).__name__}: {str(e)}",
                error_type="execution_error",
                exception=e
            )
    
    def _error_response(
        self,
        message: str,
        error_type: str,
        exception: Optional[Exception] = None
    ) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "success": False,
            "error": message,
            "error_type": error_type,
            "exception": str(exception) if exception else None,
            "suggestion": self._get_error_suggestion(error_type)
        }
    
    def _validate_arguments(
        self,
        tool: Tool,
        arguments: Dict[str, Any]
    ) -> Optional[str]:
        """Validate tool arguments."""
        
        required = tool.parameters.get("required", [])
        properties = tool.parameters.get("properties", {})
        
        # Check required parameters
        for param in required:
            if param not in arguments:
                return f"Missing required parameter: {param}"
        
        # Check parameter types
        for param, value in arguments.items():
            if param not in properties:
                continue
            
            expected_type = properties[param].get("type")
            if expected_type == "string" and not isinstance(value, str):
                return f"Parameter '{param}' should be string, got {type(value).__name__}"
            elif expected_type == "integer" and not isinstance(value, int):
                return f"Parameter '{param}' should be integer, got {type(value).__name__}"
            # Add more type checks as needed
        
        return None
    
    def _suggest_tool(self, tool_name: str) -> Optional[str]:
        """Suggest a tool name based on similarity."""
        from difflib import get_close_matches
        
        matches = get_close_matches(tool_name, self.tools.keys(), n=1, cutoff=0.6)
        return matches[0] if matches else None
    
    def _get_error_suggestion(self, error_type: str) -> str:
        """Get suggestion based on error type."""
        suggestions = {
            "missing_tool_name": "Ensure tool call includes 'name' field",
            "unknown_tool": "Check tool name spelling against available tools",
            "invalid_arguments": "Verify all required parameters are provided with correct types",
            "execution_error": "Check tool arguments are valid for the operation"
        }
        return suggestions.get(error_type, "Check tool call format and arguments")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for debugging."""
        return {
            "error_counts": self.error_counts,
            "most_problematic": max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None
        }
```

### Graceful Degradation

```python
def execute_with_degradation(
    tool_calls: List[Dict[str, Any]],
    tools: Dict[str, Tool]
) -> List[Dict[str, Any]]:
    """Execute tools with graceful degradation."""
    
    results = []
    
    for call in tool_calls:
        result = execute_tool_call(call, tools)
        
        if not result["success"]:
            # Try to provide partial/fallback results
            fallback = attempt_fallback(call, tools)
            if fallback:
                result = {
                    "success": True,
                    "result": fallback,
                    "warning": f"Used fallback due to error: {result['error']}"
                }
        
        results.append(result)
    
    return results


def attempt_fallback(
    failed_call: Dict[str, Any],
    tools: Dict[str, Tool]
) -> Optional[Any]:
    """Attempt to provide fallback result."""
    
    tool_name = failed_call["name"]
    
    # Define fallback strategies
    fallbacks = {
        "web_search": lambda: {"results": [], "note": "Search unavailable"},
        "calculator": lambda: "Calculation failed",
        "get_weather": lambda: {"condition": "unknown", "note": "Weather data unavailable"}
    }
    
    if tool_name in fallbacks:
        return fallbacks[tool_name]()
    
    return None
```

## Best Practices

Guidelines for effective tool calling.

### 1. Tool Definition Best Practices

```python
# ✅ GOOD: Clear, descriptive, well-documented
{
    "name": "search_documents",
    "description": "Search through internal documents using keywords. "
                  "Returns up to 10 most relevant documents with excerpts. "
                  "Useful for finding specific information in company knowledge base.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query. Use specific keywords for better results. "
                              "Example: 'Q4 2025 revenue report'",
                "minLength": 1,
                "maxLength": 200
            },
            "category": {
                "type": "string",
                "enum": ["reports", "policies", "projects", "all"],
                "description": "Document category to search within. "
                              "Use 'all' to search everything.",
                "default": "all"
            }
        },
        "required": ["query"]
    }
}

# ❌ BAD: Vague, poorly documented
{
    "name": "search",
    "description": "Searches",
    "parameters": {
        "type": "object",
        "properties": {
            "q": {"type": "string"},
            "cat": {"type": "string"}
        }
    }
}
```

### 2. Keep Tools Focused

```python
# ✅ GOOD: Focused single-responsibility tools
def get_weather(city: str) -> dict:
    """Get weather for a city."""
    pass

def get_forecast(city: str, days: int) -> dict:
    """Get weather forecast."""
    pass

# ❌ BAD: Too many responsibilities
def weather_operations(
    city: str,
    operation: str,  # "current", "forecast", "historical", "alerts"
    days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> dict:
    """Do various weather operations."""
    pass
```

### 3. Provide Good Examples

```python
tool_definition = {
    "name": "calculate",
    "description": "Evaluate mathematical expressions",
    "parameters": {
        "expression": {
            "type": "string",
            "description": "Mathematical expression to evaluate. "
                          "Examples: '2 + 2', '15 * 3 - 7', 'sqrt(16)'"
        }
    },
    "examples": [
        {"expression": "10 + 5", "result": 15},
        {"expression": "20 / 4", "result": 5.0}
    ]
}
```

### 4. Validate Early

```python
def execute_tool(tool_name: str, arguments: dict) -> dict:
    """Execute tool with early validation."""
    
    # Validate before execution
    if tool_name not in TOOLS:
        return {"error": "Unknown tool"}
    
    tool = TOOLS[tool_name]
    
    # Validate required parameters
    required = tool.parameters.get("required", [])
    missing = [p for p in required if p not in arguments]
    if missing:
        return {"error": f"Missing parameters: {missing}"}
    
    # Validate types
    for param, value in arguments.items():
        expected_type = tool.parameters["properties"][param].get("type")
        # ... type checking ...
    
    # Execute only after validation passes
    return tool.execute(**arguments)
```

### 5. Handle Errors Gracefully

```python
def safe_tool_call(tool_name: str, arguments: dict) -> dict:
    """Safe tool execution with error handling."""
    
    try:
        result = execute_tool(tool_name, arguments)
        return {"success": True, "result": result}
        
    except TimeoutError:
        return {
            "success": False,
            "error": "Tool execution timed out",
            "suggestion": "Try with smaller input or simpler parameters"
        }
        
    except ValueError as e:
        return {
            "success": False,
            "error": f"Invalid value: {e}",
            "suggestion": "Check parameter values and types"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {e}",
            "suggestion": "Report this error to system administrator"
        }
```

### 6. Log Tool Usage

```python
import logging

logger = logging.getLogger(__name__)

def logged_tool_call(tool_name: str, arguments: dict) -> dict:
    """Execute tool with comprehensive logging."""
    
    logger.info(f"Tool call: {tool_name} with arguments {arguments}")
    
    start_time = time.time()
    result = execute_tool(tool_name, arguments)
    duration = time.time() - start_time
    
    logger.info(f"Tool {tool_name} completed in {duration:.2f}s")
    
    if not result.get("success"):
        logger.error(f"Tool {tool_name} failed: {result.get('error')}")
    
    return result
```

### 7. Rate Limit and Throttle

```python
from collections import defaultdict
import time

class RateLimitedExecutor:
    """Execute tools with rate limiting."""
    
    def __init__(self, max_calls_per_minute: int = 60):
        self.max_calls_per_minute = max_calls_per_minute
        self.call_times = defaultdict(list)
    
    def execute(self, tool_name: str, arguments: dict) -> dict:
        """Execute with rate limiting."""
        
        now = time.time()
        
        # Clean old entries
        cutoff = now - 60
        self.call_times[tool_name] = [
            t for t in self.call_times[tool_name]
            if t > cutoff
        ]
        
        # Check rate limit
        if len(self.call_times[tool_name]) >= self.max_calls_per_minute:
            return {
                "success": False,
                "error": "Rate limit exceeded",
                "retry_after": 60 - (now - self.call_times[tool_name][0])
            }
        
        # Execute
        result = execute_tool(tool_name, arguments)
        self.call_times[tool_name].append(now)
        
        return result
```

### 8. Test Tools Independently

```python
import unittest

class TestWeatherTool(unittest.TestCase):
    """Test weather tool independently."""
    
    def test_valid_city(self):
        """Test with valid city name."""
        result = get_weather(city="Tokyo")
        self.assertIn("temperature", result)
        self.assertIn("condition", result)
    
    def test_invalid_city(self):
        """Test with invalid city name."""
        result = get_weather(city="InvalidCity12345")
        self.assertEqual(result.get("error"), "City not found")
    
    def test_units_parameter(self):
        """Test temperature units."""
        result_c = get_weather(city="Tokyo", units="celsius")
        result_f = get_weather(city="Tokyo", units="fahrenheit")
        
        # Fahrenheit should be higher than Celsius for same temp
        self.assertGreater(
            result_f["temperature"],
            result_c["temperature"]
        )
```

## Summary

Tool calling is the mechanism that transforms language models from text generators into action-taking agents. Effective tool calling requires:

**Key Concepts**:
- Tools are external functions that agents can invoke
- Tool schemas define how to call tools
- Parsing extracts tool calls from LLM output
- Execution runs the actual tool code
- Results are fed back to the LLM

**Architecture**:
- Tool registry stores definitions
- Tool prompt generator describes tools to LLM
- Request parser extracts structured calls
- Validator ensures correctness
- Executor runs tools safely
- Result formatter prepares output

**Implementation**:
- Define tools with clear schemas
- Generate tool descriptions for LLM
- Parse tool requests robustly
- Validate before execution
- Execute safely with timeouts
- Format results appropriately
- Handle errors gracefully

**Patterns**:
- Search-and-process
- Validate-execute-verify
- Fallback and retry
- Sequential and parallel execution
- Dependency-based execution

**Best Practices**:
- Keep tools focused and single-purpose
- Provide clear descriptions and examples
- Validate early and often
- Handle errors gracefully
- Log tool usage
- Rate limit expensive operations
- Test tools independently

Tool calling is fundamental to agent capabilities. Master it, and your agents can interact with the world beyond language.

## Next Steps

Now that you understand tool calling fundamentals, explore:

- **[Context Management](context-management.md)** - Managing tool outputs in context
- **[State Basics](state-basics.md)** - Maintaining state across tool calls
- **[Basic Error Handling](basic-error-handling.md)** - Robust error handling strategies
- **[Simple Agent Loops](simple-agent-loops.md)** - Integrating tools into agent loops
- **Tool Use** - Advanced tool patterns and orchestration

**Practice exercises**:
1. Implement a complete tool calling system from scratch
2. Build a set of 5 complementary tools (search, process, store, retrieve, analyze)
3. Create a robust parser that handles malformed tool calls
4. Implement parallel tool execution with proper error handling
5. Build a tool selection system that scores relevance

**Advanced topics**:
- Dynamic tool loading and unloading
- Tool composition and chaining
- Tool sandboxing and security
- Cross-language tool calling
- Distributed tool execution

Tool calling bridges thought and action. Build this bridge well.
