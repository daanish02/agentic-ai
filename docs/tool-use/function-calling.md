# Function Calling

## Table of Contents

- [Introduction](#introduction)
- [What is Function Calling?](#what-is-function-calling)
- [The Mechanics of Function Calling](#the-mechanics-of-function-calling)
- [Structured Output Generation](#structured-output-generation)
- [Function Schemas](#function-schemas)
- [Parameter Extraction](#parameter-extraction)
- [Native Function Calling APIs](#native-function-calling-apis)
- [Prompt-Based Function Calling](#prompt-based-function-calling)
- [Translating Natural Language to Function Calls](#translating-natural-language-to-function-calls)
- [Function Calling Patterns](#function-calling-patterns)
- [Type Safety and Validation](#type-safety-and-validation)
- [Complex Parameter Types](#complex-parameter-types)
- [Optional vs Required Parameters](#optional-vs-required-parameters)
- [Function Calling Errors](#function-calling-errors)
- [Best Practices](#best-practices)
- [Common Pitfalls](#common-pitfalls)
- [Testing Function Calls](#testing-function-calls)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Function calling is the foundational mechanism that enables language models to interact with the external world. Instead of just generating text, models can **invoke functions** with specific parameters, enabling them to take actions, retrieve information, or manipulate data.

This capability transforms LLMs from conversational chatbots into **action-taking agents**. A model can go from understanding "What's the weather in London?" to actually calling `get_weather(city="London", units="celsius")`.

> "Function calling bridges the gap between language understanding and programmatic action."

This guide covers the fundamentals of function calling: from basic mechanics to advanced patterns, from schema design to parameter extraction, and from native APIs to prompt-based approaches.

## What is Function Calling?

Function calling is the process by which a language model generates **structured function invocations** in response to natural language input.

### The Basic Flow

```
┌─────────────────┐
│ Natural Language│
│    Input        │
│ "Get weather in │
│     London"     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM decides   │
│  which function │
│    to call      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Structured Call │
│ get_weather(    │
│   city="London",│
│   units="C"     │
│ )               │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Function        │
│ Executed        │
│ Returns result  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Result back to  │
│  LLM for final  │
│   response      │
└─────────────────┘
```

### Why Function Calling?

**1. Deterministic Actions**

```python
# Without function calling (unreliable)
response = llm("Calculate 15% tip on $45.50")
# Output: "The tip would be $6.83" (might be wrong)

# With function calling (precise)
function_call = llm_function_call("Calculate 15% tip on $45.50")
# Output: calculate_tip(amount=45.50, percentage=15)
result = calculate_tip(45.50, 15)  # Exact: 6.825
```

**2. External Tool Access**

```python
# Can't generate this data, must fetch it
weather = get_current_weather(city="London")
stock_price = get_stock_price(symbol="AAPL")
db_results = query_database(sql="SELECT * FROM users WHERE active=true")
```

**3. Structured Data**

```python
# Parsing text is error-prone
text_response = "I think you should search for 'AI agents' and then..."

# Structured calls are reliable
function_call = FunctionCall(
    name="search",
    arguments={"query": "AI agents", "limit": 10}
)
```

## The Mechanics of Function Calling

How does a language model generate function calls?

### Conceptual Process

**Step 1: Understanding Intent**

The model processes the user's input and determines:

- What does the user want to accomplish?
- Can this be done with available functions?
- Which function(s) are relevant?

**Step 2: Parameter Extraction**

The model identifies the function parameters from context:

- Required values explicitly mentioned
- Optional values if specified
- Reasonable defaults for missing values

**Step 3: Generating Structured Output**

The model outputs the function call in a specific format:

- JSON structure
- XML format
- Custom schema

**Step 4: Execution and Response**

The function is executed, and results are fed back to the model for final response generation.

### Behind the Scenes

```python
# What the model "sees" (simplified)
system_prompt = """
You have access to these functions:

{
  "name": "get_weather",
  "description": "Get current weather for a location",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string", "description": "City name"},
      "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["city"]
  }
}

When you want to call a function, output JSON:
{"name": "function_name", "arguments": {...}}
"""

user_message = "What's the weather in Paris?"

# Model generates
function_call = {
    "name": "get_weather",
    "arguments": {
        "city": "Paris",
        "units": "celsius"
    }
}
```

## Structured Output Generation

The core of function calling is generating structured, parseable output.

### JSON Output

Most common format for function calls:

```python
{
    "name": "send_email",
    "arguments": {
        "to": "user@example.com",
        "subject": "Meeting Reminder",
        "body": "Don't forget our meeting at 3pm",
        "cc": ["manager@example.com"],
        "priority": "normal"
    }
}
```

### Ensuring Valid JSON

**Problem**: LLMs sometimes generate malformed JSON

```python
# Bad outputs that can happen
bad_1 = '{"name": "search", arguments: {"query": "test"}}'  # Missing quotes
bad_2 = '{"name": "search", "arguments": {"query": "test",}}'  # Trailing comma
bad_3 = '{"name": "search", "arguments": {"query": test}}'  # Unquoted string
```

**Solution**: Multiple strategies

```python
import json
import re
from typing import Dict, Any, Optional

def extract_function_call(llm_output: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extract function call from LLM output.
    """
    # Strategy 1: Try direct JSON parsing
    try:
        return json.loads(llm_output)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract JSON from text
    json_pattern = r'\{[^{}]*\{[^{}]*\}[^{}]*\}'  # Nested JSON
    matches = re.findall(json_pattern, llm_output)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Strategy 3: Fix common issues
    cleaned = llm_output.strip()
    cleaned = re.sub(r',\s*}', '}', cleaned)  # Remove trailing commas
    cleaned = re.sub(r',\s*]', ']', cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def validate_function_call(call: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate function call against schema.
    """
    if "name" not in call or "arguments" not in call:
        return False

    if call["name"] != schema["name"]:
        return False

    args = call["arguments"]
    params = schema["parameters"]

    # Check required parameters
    required = params.get("required", [])
    for param in required:
        if param not in args:
            return False

    # Check types
    properties = params.get("properties", {})
    for key, value in args.items():
        if key not in properties:
            return False  # Unknown parameter

        expected_type = properties[key].get("type")
        if expected_type == "string" and not isinstance(value, str):
            return False
        elif expected_type == "number" and not isinstance(value, (int, float)):
            return False
        elif expected_type == "boolean" and not isinstance(value, bool):
            return False
        elif expected_type == "array" and not isinstance(value, list):
            return False

    return True
```

### Alternative Formats

**XML Format** (used by some models):

```xml
<function_call>
    <name>search_database</name>
    <arguments>
        <table>users</table>
        <query>SELECT * FROM users WHERE age > 25</query>
        <limit>100</limit>
    </arguments>
</function_call>
```

**Custom DSL** (domain-specific language):

```
CALL search_database
  WITH table="users"
  WITH query="SELECT * FROM users WHERE age > 25"
  WITH limit=100
```

## Function Schemas

Schemas describe functions to the model in a structured way.

### Basic Schema Structure

```python
function_schema = {
    "name": "function_name",
    "description": "What the function does",
    "parameters": {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "What this parameter is"
            },
            "param2": {
                "type": "integer",
                "description": "Another parameter"
            }
        },
        "required": ["param1"]
    }
}
```

### Comprehensive Example

```python
search_papers_schema = {
    "name": "search_papers",
    "description": """
    Search for academic papers in a research database.
    Returns paper metadata including title, authors, abstract, and citations.
    Use this when the user asks about research, papers, or academic topics.
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string. Use keywords and topics, not full sentences."
            },
            "year_range": {
                "type": "object",
                "description": "Optional year range to filter papers",
                "properties": {
                    "start": {"type": "integer", "description": "Starting year (inclusive)"},
                    "end": {"type": "integer", "description": "Ending year (inclusive)"}
                }
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of papers to return",
                "default": 10,
                "minimum": 1,
                "maximum": 100
            },
            "sort_by": {
                "type": "string",
                "description": "How to sort results",
                "enum": ["relevance", "citations", "date"],
                "default": "relevance"
            },
            "include_abstracts": {
                "type": "boolean",
                "description": "Whether to include paper abstracts in results",
                "default": true
            }
        },
        "required": ["query"]
    }
}
```

### Schema Design Principles

**1. Clear Descriptions**

```python
# Bad: Vague
"description": "Search function"

# Good: Specific
"description": "Search academic papers by keywords, topics, or authors. Returns paper metadata including title, authors, publication year, and citation count."
```

**2. Parameter Guidance**

```python
# Bad: Unclear format
"query": {"type": "string", "description": "The search query"}

# Good: Format guidance
"query": {
    "type": "string",
    "description": "Search query using keywords (e.g., 'neural networks attention mechanism'). Avoid full sentences."
}
```

**3. Constraints and Validation**

```python
"parameters": {
    "properties": {
        "age": {
            "type": "integer",
            "minimum": 0,
            "maximum": 150,
            "description": "User age in years"
        },
        "email": {
            "type": "string",
            "pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
            "description": "Valid email address"
        },
        "priority": {
            "type": "string",
            "enum": ["low", "medium", "high", "urgent"],
            "description": "Task priority level"
        }
    }
}
```

## Parameter Extraction

How models extract parameter values from natural language.

### Direct Mention

```python
User: "Search for papers by Jane Smith"

# Model extracts:
{
    "name": "search_papers",
    "arguments": {
        "query": "Jane Smith",
        "search_type": "author"  # Inferred from context
    }
}
```

### Contextual Understanding

```python
User: "What's the weather like right now?"
# Location not specified

# Model might use:
# 1. User's profile/settings
# 2. Previous conversation
# 3. Ask for clarification
# 4. Use a default

{
    "name": "get_weather",
    "arguments": {
        "city": "user_location",  # From context
        "time": "current"  # Inferred from "right now"
    }
}
```

### Type Conversion

```python
User: "Set a reminder for tomorrow at 3pm"

# Model converts natural language to appropriate types
{
    "name": "set_reminder",
    "arguments": {
        "time": "2026-01-23T15:00:00",  # ISO format
        "message": "Reminder"
    }
}
```

### Implementation

```python
from datetime import datetime, timedelta
from typing import Any, Dict, List
import re

class ParameterExtractor:
    """Extract and convert parameters from natural language."""

    def extract_datetime(self, text: str) -> str:
        """Convert relative time expressions to ISO format."""
        now = datetime.now()

        # Relative dates
        if "tomorrow" in text.lower():
            date = now + timedelta(days=1)
        elif "next week" in text.lower():
            date = now + timedelta(weeks=1)
        elif "today" in text.lower():
            date = now
        else:
            date = now

        # Time extraction
        time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', text.lower())
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2)) if time_match.group(2) else 0
            period = time_match.group(3)

            if period == 'pm' and hour != 12:
                hour += 12
            elif period == 'am' and hour == 12:
                hour = 0

            date = date.replace(hour=hour, minute=minute, second=0, microsecond=0)

        return date.isoformat()

    def extract_location(self, text: str) -> Dict[str, str]:
        """Extract location information."""
        # City patterns
        city_pattern = r'in\s+([A-Z][a-zA-Z\s]+?)(?:\s|,|$)'
        city_match = re.search(city_pattern, text)

        if city_match:
            return {"city": city_match.group(1).strip()}

        return {}

    def extract_quantity(self, text: str, unit: str) -> float:
        """Extract numeric quantities with units."""
        pattern = rf'(\d+(?:\.\d+)?)\s*{unit}'
        match = re.search(pattern, text.lower())

        if match:
            return float(match.group(1))

        return None

    def extract_email(self, text: str) -> List[str]:
        """Extract email addresses."""
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return re.findall(email_pattern, text)


# Usage example
extractor = ParameterExtractor()

text = "Send email to john@example.com tomorrow at 2pm about the meeting"

params = {
    "to": extractor.extract_email(text)[0],
    "scheduled_time": extractor.extract_datetime(text),
    "subject": "Meeting"  # Would need more sophisticated extraction
}
```

## Native Function Calling APIs

Modern LLM APIs provide built-in function calling support.

### OpenAI Function Calling

```python
import openai
from typing import List, Dict, Any

def call_llm_with_functions(
    messages: List[Dict[str, str]],
    functions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Call OpenAI with function definitions.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        functions=functions,
        function_call="auto"  # Let model decide when to call
    )

    return response


# Define functions
functions = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or zip code"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
]

# User message
messages = [
    {"role": "user", "content": "What's the weather in Tokyo?"}
]

# Call LLM
response = call_llm_with_functions(messages, functions)

# Handle response
if response.choices[0].message.get("function_call"):
    function_call = response.choices[0].message["function_call"]
    function_name = function_call["name"]
    function_args = json.loads(function_call["arguments"])

    print(f"Model wants to call: {function_name}")
    print(f"With arguments: {function_args}")

    # Execute the function
    if function_name == "get_weather":
        result = get_weather(**function_args)
    elif function_name == "search_web":
        result = search_web(**function_args)

    # Send result back to model
    messages.append(response.choices[0].message)
    messages.append({
        "role": "function",
        "name": function_name,
        "content": json.dumps(result)
    })

    # Get final response
    final_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )

    print(final_response.choices[0].message["content"])
```

### Anthropic Tool Use

```python
import anthropic

def call_claude_with_tools(
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Call Claude with tool definitions.
    """
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

    return response


# Define tools
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }
    }
]

# Use tools
messages = [{"role": "user", "content": "What's the weather in Paris?"}]
response = call_claude_with_tools(messages, tools)

# Process tool use
for block in response.content:
    if block.type == "tool_use":
        print(f"Tool: {block.name}")
        print(f"Input: {block.input}")

        # Execute tool
        result = execute_tool(block.name, block.input)

        # Continue conversation with result
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result)
                }
            ]
        })

        # Get final response
        final = call_claude_with_tools(messages, tools)
```

### Comparison: OpenAI vs Anthropic

```python
# OpenAI style
openai_function = {
    "name": "search",
    "description": "Search for information",
    "parameters": {  # Key: "parameters"
        "type": "object",
        "properties": {...}
    }
}

# Anthropic style
anthropic_tool = {
    "name": "search",
    "description": "Search for information",
    "input_schema": {  # Key: "input_schema"
        "type": "object",
        "properties": {...}
    }
}
```

## Prompt-Based Function Calling

For models without native function calling, use prompts.

### Basic Pattern

```python
FUNCTION_CALLING_PROMPT = """
You are an assistant that can use tools to help users.

Available tools:

<tools>
{tools_description}
</tools>

When you need to use a tool, output EXACTLY this format:
TOOL_CALL: function_name(arg1="value1", arg2="value2")

Do not add any extra text before or after the TOOL_CALL line.

Examples:
User: What's the weather in London?
Assistant: TOOL_CALL: get_weather(city="London", units="celsius")

User: Search for Python tutorials
Assistant: TOOL_CALL: search_web(query="Python tutorials", max_results=10)
"""

def format_tools_for_prompt(tools: List[Dict]) -> str:
    """Format tool schemas for prompt."""
    descriptions = []

    for tool in tools:
        desc = f"- {tool['name']}: {tool['description']}\n"
        desc += "  Parameters:\n"

        for param, info in tool['parameters']['properties'].items():
            required = param in tool['parameters'].get('required', [])
            req_str = "(required)" if required else "(optional)"
            desc += f"    - {param} {req_str}: {info['description']}\n"

        descriptions.append(desc)

    return "\n".join(descriptions)


def extract_tool_call_from_text(response: str) -> Optional[Dict[str, Any]]:
    """Extract tool call from text response."""
    pattern = r'TOOL_CALL:\s*(\w+)\((.*?)\)'
    match = re.search(pattern, response)

    if not match:
        return None

    function_name = match.group(1)
    args_string = match.group(2)

    # Parse arguments
    arguments = {}
    for arg in args_string.split(','):
        if '=' in arg:
            key, value = arg.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"\'')
            arguments[key] = value

    return {
        "name": function_name,
        "arguments": arguments
    }
```

### JSON Output Prompting

````python
JSON_FUNCTION_PROMPT = """
You are an assistant with access to tools. When you need to use a tool,
output ONLY a JSON object with this structure:

{
  "tool": "tool_name",
  "arguments": {
    "param1": "value1",
    "param2": "value2"
  }
}

Available tools:
{tools}

Output ONLY valid JSON. No other text.
"""

def enforce_json_output(llm_output: str) -> Dict[str, Any]:
    """
    Enforce JSON output from free-form LLM response.
    """
    # Try to find JSON in the output
    try:
        return json.loads(llm_output)
    except json.JSONDecodeError:
        pass

    # Extract JSON block if wrapped in markdown
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', llm_output, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Find JSON-like structure
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', llm_output, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from: {llm_output}")
````

## Translating Natural Language to Function Calls

The core challenge: mapping informal language to precise function invocations.

### Ambiguity Resolution

```python
def resolve_ambiguity(user_input: str, candidate_functions: List[str]) -> str:
    """
    When multiple functions could work, choose the best one.
    """
    disambiguation_prompt = f"""
User said: "{user_input}"

Multiple functions could potentially help:
{candidate_functions}

Which function is most appropriate? Consider:
1. Directness: Does it directly accomplish the goal?
2. Efficiency: Fewest steps?
3. Specificity: Most targeted to the request?

Output only the function name.
"""

    return llm_call(disambiguation_prompt)


# Example
user_input = "Find me some information about neural networks"

candidates = [
    "search_web(query) - Search the internet",
    "search_papers(query) - Search academic papers",
    "query_database(table, query) - Query local database"
]

best = resolve_ambiguity(user_input, candidates)
# Likely chooses "search_papers" for technical topic
```

### Implicit Parameter Inference

```python
def infer_missing_parameters(
    function_call: Dict[str, Any],
    schema: Dict[str, Any],
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Fill in missing optional parameters with sensible defaults.
    """
    function_name = function_call["name"]
    args = function_call["arguments"].copy()

    # Get parameter definitions
    params = schema["parameters"]["properties"]
    required = schema["parameters"].get("required", [])

    for param_name, param_def in params.items():
        if param_name in args:
            continue  # Already provided

        if param_name in required:
            # Required but missing - need to ask user
            raise ValueError(f"Required parameter missing: {param_name}")

        # Optional - try to infer
        if "default" in param_def:
            args[param_name] = param_def["default"]
        elif param_name in context:
            args[param_name] = context[param_name]
        elif param_name == "user_id" and "user_id" in context:
            args[param_name] = context["user_id"]
        # Add more inference rules as needed

    return {
        "name": function_name,
        "arguments": args
    }


# Example usage
call = {
    "name": "send_email",
    "arguments": {
        "to": "user@example.com",
        "subject": "Hello"
    }
}

schema = {
    "name": "send_email",
    "parameters": {
        "properties": {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string", "default": ""},
            "from": {"type": "string"},
            "priority": {"type": "string", "default": "normal"}
        },
        "required": ["to", "subject"]
    }
}

context = {
    "user_email": "sender@example.com"
}

complete_call = infer_missing_parameters(call, schema, context)
# Adds: body="", from="sender@example.com", priority="normal"
```

### Natural Language to Type Conversion

```python
from typing import Any, Union
from datetime import datetime, timedelta
import re

class TypeConverter:
    """Convert natural language to typed parameters."""

    def convert(self, value: str, target_type: str) -> Any:
        """Convert value to target type."""
        if target_type == "string":
            return str(value)
        elif target_type == "integer":
            return self.to_integer(value)
        elif target_type == "number":
            return self.to_number(value)
        elif target_type == "boolean":
            return self.to_boolean(value)
        elif target_type == "array":
            return self.to_array(value)
        elif target_type == "datetime":
            return self.to_datetime(value)
        else:
            return value

    def to_integer(self, value: str) -> int:
        """Convert to integer."""
        # Handle words
        word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }

        value_lower = value.lower()
        if value_lower in word_to_num:
            return word_to_num[value_lower]

        # Extract number from text
        match = re.search(r'\d+', value)
        if match:
            return int(match.group())

        raise ValueError(f"Cannot convert '{value}' to integer")

    def to_number(self, value: str) -> float:
        """Convert to number."""
        match = re.search(r'\d+(?:\.\d+)?', value)
        if match:
            return float(match.group())

        raise ValueError(f"Cannot convert '{value}' to number")

    def to_boolean(self, value: str) -> bool:
        """Convert to boolean."""
        true_values = {"yes", "true", "on", "1", "enable", "enabled"}
        false_values = {"no", "false", "off", "0", "disable", "disabled"}

        value_lower = value.lower().strip()

        if value_lower in true_values:
            return True
        elif value_lower in false_values:
            return False

        raise ValueError(f"Cannot convert '{value}' to boolean")

    def to_array(self, value: str) -> list:
        """Convert to array."""
        # Split by common delimiters
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        elif ';' in value:
            return [item.strip() for item in value.split(';')]
        elif ' and ' in value:
            return [item.strip() for item in value.split(' and ')]

        return [value]

    def to_datetime(self, value: str) -> str:
        """Convert natural language to ISO datetime."""
        now = datetime.now()
        value_lower = value.lower()

        # Relative times
        if "now" in value_lower or "current" in value_lower:
            return now.isoformat()
        elif "tomorrow" in value_lower:
            return (now + timedelta(days=1)).isoformat()
        elif "yesterday" in value_lower:
            return (now - timedelta(days=1)).isoformat()
        elif "next week" in value_lower:
            return (now + timedelta(weeks=1)).isoformat()

        # Try parsing standard formats
        for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"]:
            try:
                return datetime.strptime(value, fmt).isoformat()
            except ValueError:
                continue

        return value  # Return as-is if can't parse


# Usage
converter = TypeConverter()

examples = {
    "five": converter.to_integer("five"),  # 5
    "3.14": converter.to_number("3.14"),  # 3.14
    "yes": converter.to_boolean("yes"),  # True
    "apple, banana, orange": converter.to_array("apple, banana, orange"),  # ["apple", "banana", "orange"]
    "tomorrow": converter.to_datetime("tomorrow")  # ISO datetime for tomorrow
}
```

## Function Calling Patterns

Common patterns for organizing function calls.

### Single Function Call

```python
def single_function_call(user_input: str, functions: List[Dict]) -> Any:
    """
    Simple pattern: one input, one function call, one result.
    """
    # 1. Model decides which function
    function_call = llm_decide_function(user_input, functions)

    # 2. Execute function
    result = execute_function(
        function_call["name"],
        function_call["arguments"]
    )

    # 3. Format result for user
    response = llm_format_response(user_input, result)

    return response
```

### Sequential Function Calls

```python
def sequential_function_calls(user_input: str, functions: List[Dict]) -> Any:
    """
    Chain multiple functions where output of one feeds into next.
    """
    context = {"user_input": user_input}
    max_steps = 10

    for step in range(max_steps):
        # Decide next function based on context
        function_call = llm_decide_next_function(context, functions)

        if function_call["name"] == "FINISH":
            break

        # Execute function
        result = execute_function(
            function_call["name"],
            function_call["arguments"]
        )

        # Add result to context
        context[f"step_{step}_result"] = result

    # Generate final response
    return llm_format_response(user_input, context)
```

### Parallel Function Calls

```python
import asyncio
from typing import List, Dict, Any

async def parallel_function_calls(
    user_input: str,
    functions: List[Dict]
) -> Any:
    """
    Execute multiple independent functions in parallel.
    """
    # Model identifies all needed functions
    function_calls = llm_identify_all_functions(user_input, functions)

    # Execute in parallel
    tasks = []
    for call in function_calls:
        task = execute_function_async(call["name"], call["arguments"])
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    # Combine results
    combined = {
        call["name"]: result
        for call, result in zip(function_calls, results)
    }

    # Generate response from all results
    return llm_format_response(user_input, combined)
```

### Conditional Function Calling

```python
def conditional_function_call(
    user_input: str,
    functions: List[Dict],
    context: Dict[str, Any]
) -> Any:
    """
    Call functions based on conditions.
    """
    # Check if function call is needed
    needs_function = llm_check_if_function_needed(user_input, context)

    if not needs_function:
        # Can answer from existing knowledge
        return llm_generate_response(user_input, context)

    # Determine which function
    function_call = llm_decide_function(user_input, functions)

    # Execute
    result = execute_function(
        function_call["name"],
        function_call["arguments"]
    )

    # Check if more functions needed
    if llm_check_if_more_functions_needed(user_input, result):
        # Recursive call
        context.update({"previous_result": result})
        return conditional_function_call(user_input, functions, context)

    return llm_format_response(user_input, result)
```

## Type Safety and Validation

Ensuring function calls are valid and safe.

### Schema Validation

```python
from typing import Any, Dict, List, Optional
from enum import Enum

class ValidationError(Exception):
    """Raised when validation fails."""
    pass

class TypeValidator:
    """Validate function arguments against schema."""

    def validate(self, arguments: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Validate arguments against schema."""
        errors = []

        # Check required parameters
        required = schema.get("required", [])
        for param in required:
            if param not in arguments:
                errors.append(f"Missing required parameter: {param}")

        # Validate each argument
        properties = schema.get("properties", {})
        for key, value in arguments.items():
            if key not in properties:
                errors.append(f"Unknown parameter: {key}")
                continue

            param_schema = properties[key]
            param_errors = self.validate_value(value, param_schema, key)
            errors.extend(param_errors)

        if errors:
            raise ValidationError("; ".join(errors))

        return True

    def validate_value(
        self,
        value: Any,
        schema: Dict[str, Any],
        param_name: str
    ) -> List[str]:
        """Validate a single value."""
        errors = []

        # Type validation
        expected_type = schema.get("type")
        if expected_type:
            if not self.check_type(value, expected_type):
                errors.append(
                    f"{param_name}: expected {expected_type}, got {type(value).__name__}"
                )
                return errors  # Don't validate further if type is wrong

        # Enum validation
        if "enum" in schema:
            if value not in schema["enum"]:
                errors.append(
                    f"{param_name}: must be one of {schema['enum']}, got {value}"
                )

        # Numeric constraints
        if expected_type in ["integer", "number"]:
            if "minimum" in schema and value < schema["minimum"]:
                errors.append(
                    f"{param_name}: must be >= {schema['minimum']}, got {value}"
                )
            if "maximum" in schema and value > schema["maximum"]:
                errors.append(
                    f"{param_name}: must be <= {schema['maximum']}, got {value}"
                )

        # String constraints
        if expected_type == "string":
            if "minLength" in schema and len(value) < schema["minLength"]:
                errors.append(
                    f"{param_name}: must be at least {schema['minLength']} characters"
                )
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                errors.append(
                    f"{param_name}: must be at most {schema['maxLength']} characters"
                )
            if "pattern" in schema:
                import re
                if not re.match(schema["pattern"], value):
                    errors.append(
                        f"{param_name}: does not match required pattern"
                    )

        # Array constraints
        if expected_type == "array":
            if "minItems" in schema and len(value) < schema["minItems"]:
                errors.append(
                    f"{param_name}: must have at least {schema['minItems']} items"
                )
            if "maxItems" in schema and len(value) > schema["maxItems"]:
                errors.append(
                    f"{param_name}: must have at most {schema['maxItems']} items"
                )

        return errors

    def check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict
        }

        expected_python_type = type_map.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, skip check

        return isinstance(value, expected_python_type)


# Usage
validator = TypeValidator()

schema = {
    "properties": {
        "age": {
            "type": "integer",
            "minimum": 0,
            "maximum": 150
        },
        "email": {
            "type": "string",
            "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        },
        "tags": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5
        }
    },
    "required": ["age", "email"]
}

# Valid
try:
    validator.validate({
        "age": 25,
        "email": "user@example.com",
        "tags": ["python", "coding"]
    }, schema)
    print("Valid!")
except ValidationError as e:
    print(f"Invalid: {e}")

# Invalid
try:
    validator.validate({
        "age": 200,  # Too high
        "email": "invalid-email"  # Bad format
    }, schema)
except ValidationError as e:
    print(f"Invalid: {e}")
```

## Complex Parameter Types

Handling nested and complex parameter structures.

### Nested Objects

```python
nested_schema = {
    "name": "create_user",
    "parameters": {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "address": {
                        "type": "object",
                        "properties": {
                            "street": {"type": "string"},
                            "city": {"type": "string"},
                            "country": {"type": "string"}
                        }
                    }
                }
            }
        }
    }
}

# Model generates
function_call = {
    "name": "create_user",
    "arguments": {
        "user": {
            "name": "John Doe",
            "email": "john@example.com",
            "address": {
                "street": "123 Main St",
                "city": "London",
                "country": "UK"
            }
        }
    }
}
```

### Arrays of Objects

```python
array_schema = {
    "name": "batch_create_tasks",
    "parameters": {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                        "assignee": {"type": "string"}
                    },
                    "required": ["title"]
                }
            }
        }
    }
}

# Model generates
function_call = {
    "name": "batch_create_tasks",
    "arguments": {
        "tasks": [
            {"title": "Review PR", "priority": "high", "assignee": "alice"},
            {"title": "Update docs", "priority": "medium", "assignee": "bob"},
            {"title": "Fix bug", "priority": "high", "assignee": "alice"}
        ]
    }
}
```

### Union Types

```python
# Using oneOf for union types
union_schema = {
    "name": "schedule_event",
    "parameters": {
        "type": "object",
        "properties": {
            "time": {
                "oneOf": [
                    {
                        "type": "string",
                        "description": "ISO datetime string"
                    },
                    {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string"},
                            "time": {"type": "string"}
                        },
                        "description": "Separate date and time"
                    }
                ]
            }
        }
    }
}

# Can accept either format
call1 = {
    "name": "schedule_event",
    "arguments": {
        "time": "2026-01-23T15:00:00"
    }
}

call2 = {
    "name": "schedule_event",
    "arguments": {
        "time": {
            "date": "2026-01-23",
            "time": "15:00"
        }
    }
}
```

## Optional vs Required Parameters

Managing parameter optionality effectively.

### Clear Defaults

```python
schema_with_defaults = {
    "name": "search",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results to return",
                "default": 10
            },
            "offset": {
                "type": "integer",
                "description": "Number of results to skip",
                "default": 0
            },
            "sort_by": {
                "type": "string",
                "enum": ["relevance", "date", "popularity"],
                "default": "relevance",
                "description": "How to sort results"
            }
        },
        "required": ["query"]
    }
}

def apply_defaults(arguments: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """Apply default values to missing optional parameters."""
    properties = schema["parameters"]["properties"]
    result = arguments.copy()

    for param, spec in properties.items():
        if param not in result and "default" in spec:
            result[param] = spec["default"]

    return result
```

### Context-Dependent Optionality

```python
def check_parameter_requirements(
    function_name: str,
    arguments: Dict[str, Any],
    context: Dict[str, Any]
) -> List[str]:
    """
    Check if parameters are sufficient given context.
    Some parameters might be optional if context provides them.
    """
    missing = []

    if function_name == "send_email":
        if "to" not in arguments:
            missing.append("to")
        if "subject" not in arguments:
            missing.append("subject")
        # "from" is optional if user is authenticated
        if "from" not in arguments and "user_email" not in context:
            missing.append("from")

    elif function_name == "get_weather":
        # "location" is optional if user location is known
        if "location" not in arguments and "user_location" not in context:
            missing.append("location")

    return missing
```

## Function Calling Errors

Common errors and how to handle them.

### Error Types

```python
from enum import Enum
from typing import Optional

class FunctionCallErrorType(Enum):
    """Types of function calling errors."""
    INVALID_JSON = "invalid_json"
    UNKNOWN_FUNCTION = "unknown_function"
    MISSING_PARAMETER = "missing_parameter"
    INVALID_PARAMETER_TYPE = "invalid_parameter_type"
    INVALID_PARAMETER_VALUE = "invalid_parameter_value"
    FUNCTION_EXECUTION_ERROR = "function_execution_error"


class FunctionCallError(Exception):
    """Base class for function calling errors."""

    def __init__(
        self,
        error_type: FunctionCallErrorType,
        message: str,
        function_name: Optional[str] = None,
        parameter_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.error_type = error_type
        self.message = message
        self.function_name = function_name
        self.parameter_name = parameter_name
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "error_type": self.error_type.value,
            "message": self.message,
            "function_name": self.function_name,
            "parameter_name": self.parameter_name,
            "details": self.details
        }


def safe_function_call(
    function_call_json: str,
    available_functions: Dict[str, callable],
    schemas: Dict[str, Dict]
) -> Dict[str, Any]:
    """
    Safely execute a function call with comprehensive error handling.
    """
    try:
        # Parse JSON
        try:
            call = json.loads(function_call_json)
        except json.JSONDecodeError as e:
            raise FunctionCallError(
                FunctionCallErrorType.INVALID_JSON,
                f"Could not parse function call as JSON: {e}",
                details={"raw_output": function_call_json}
            )

        # Extract function name
        function_name = call.get("name")
        if not function_name:
            raise FunctionCallError(
                FunctionCallErrorType.UNKNOWN_FUNCTION,
                "Function name not specified in call"
            )

        # Check if function exists
        if function_name not in available_functions:
            raise FunctionCallError(
                FunctionCallErrorType.UNKNOWN_FUNCTION,
                f"Unknown function: {function_name}",
                function_name=function_name,
                details={"available": list(available_functions.keys())}
            )

        # Validate against schema
        schema = schemas.get(function_name)
        if schema:
            validator = TypeValidator()
            try:
                validator.validate(call.get("arguments", {}), schema["parameters"])
            except ValidationError as e:
                raise FunctionCallError(
                    FunctionCallErrorType.INVALID_PARAMETER_VALUE,
                    str(e),
                    function_name=function_name
                )

        # Execute function
        try:
            func = available_functions[function_name]
            result = func(**call.get("arguments", {}))
            return {"success": True, "result": result}
        except TypeError as e:
            # Wrong parameters
            raise FunctionCallError(
                FunctionCallErrorType.MISSING_PARAMETER,
                f"Invalid parameters for {function_name}: {e}",
                function_name=function_name
            )
        except Exception as e:
            # Execution error
            raise FunctionCallError(
                FunctionCallErrorType.FUNCTION_EXECUTION_ERROR,
                f"Error executing {function_name}: {e}",
                function_name=function_name,
                details={"exception_type": type(e).__name__}
            )

    except FunctionCallError as e:
        return {
            "success": False,
            "error": e.to_dict()
        }
```

## Best Practices

Guidelines for effective function calling.

### 1. Clear Function Names

```python
# Bad: Vague names
def process(data):
    pass

def do_thing(x, y):
    pass

# Good: Descriptive names
def calculate_shipping_cost(weight_kg, destination_country):
    pass

def send_password_reset_email(user_email, reset_token):
    pass
```

### 2. Detailed Descriptions

```python
# Bad: Minimal description
{
    "name": "search",
    "description": "Search for stuff"
}

# Good: Comprehensive description
{
    "name": "search_products",
    "description": """
    Search the product catalog by keywords, category, or filters.
    Returns product listings with name, price, availability, and images.
    Use this when user wants to find or browse products.
    Do not use for searching orders or user accounts.
    """
}
```

### 3. Parameter Descriptions

```python
# Bad: No context
{
    "query": {"type": "string"}
}

# Good: Clear guidance
{
    "query": {
        "type": "string",
        "description": "Search keywords (e.g., 'wireless headphones'). Use specific terms for better results."
    }
}
```

### 4. Validation at Every Layer

```python
def robust_function_execution(call: Dict[str, Any]) -> Any:
    """Validate at multiple levels."""

    # 1. Schema validation
    validate_schema(call)

    # 2. Business logic validation
    validate_business_rules(call)

    # 3. Security validation
    validate_security(call)

    # 4. Execute
    result = execute(call)

    # 5. Validate result
    validate_result(result)

    return result
```

### 5. Comprehensive Examples

```python
# Include examples in schema
function_schema = {
    "name": "create_calendar_event",
    "description": "Create a new calendar event",
    "parameters": {...},
    "examples": [
        {
            "user_input": "Schedule a meeting with John tomorrow at 2pm",
            "function_call": {
                "name": "create_calendar_event",
                "arguments": {
                    "title": "Meeting with John",
                    "start_time": "2026-01-23T14:00:00",
                    "duration_minutes": 60,
                    "attendees": ["john@example.com"]
                }
            }
        }
    ]
}
```

## Common Pitfalls

Mistakes to avoid when implementing function calling.

### 1. Ambiguous Function Names

```python
# Bad: What does "get" get?
def get(id):
    pass

# Good: Specific names
def get_user_by_id(user_id):
    pass

def get_product_by_sku(sku):
    pass
```

### 2. Too Many Functions

```python
# Bad: Overwhelming the model
functions = [
    "search_users", "search_products", "search_orders",
    "create_user", "update_user", "delete_user",
    "create_product", "update_product", "delete_product",
    # ... 50 more functions
]

# Good: Group and organize
user_functions = ["manage_users", "search_users"]
product_functions = ["manage_products", "search_products"]

# Only pass relevant subset to model
relevant_functions = determine_relevant_functions(user_intent)
```

### 3. Missing Error Context

```python
# Bad: Generic error
raise Exception("Function failed")

# Good: Detailed error
raise FunctionCallError(
    FunctionCallErrorType.INVALID_PARAMETER_VALUE,
    f"Email address '{email}' is invalid: must contain @ symbol",
    function_name="send_email",
    parameter_name="to",
    details={"provided_value": email}
)
```

### 4. No Validation

```python
# Bad: Trust model output blindly
def execute_function(call):
    return eval(f"{call['name']}(**{call['arguments']})")  # DANGEROUS!

# Good: Validate everything
def execute_function(call):
    validate_function_exists(call['name'])
    validate_arguments(call['arguments'])
    validate_user_permissions(call['name'])
    return safely_execute(call)
```

## Testing Function Calls

Ensuring function calling works reliably.

### Unit Tests

````python
import pytest

def test_function_call_extraction():
    """Test extracting function calls from LLM output."""

    # Valid JSON
    output1 = '{"name": "search", "arguments": {"query": "test"}}'
    call1 = extract_function_call(output1)
    assert call1["name"] == "search"
    assert call1["arguments"]["query"] == "test"

    # JSON in markdown
    output2 = '''
    I'll search for that.
    ```json
    {"name": "search", "arguments": {"query": "test"}}
    ```
    '''
    call2 = extract_function_call(output2)
    assert call2["name"] == "search"

    # Invalid JSON
    output3 = '{"name": "search", invalid json}'
    call3 = extract_function_call(output3)
    assert call3 is None


def test_parameter_validation():
    """Test parameter validation."""
    validator = TypeValidator()

    schema = {
        "properties": {
            "age": {"type": "integer", "minimum": 0, "maximum": 150}
        },
        "required": ["age"]
    }

    # Valid
    validator.validate({"age": 25}, schema)

    # Missing required
    with pytest.raises(ValidationError):
        validator.validate({}, schema)

    # Invalid type
    with pytest.raises(ValidationError):
        validator.validate({"age": "25"}, schema)

    # Out of range
    with pytest.raises(ValidationError):
        validator.validate({"age": 200}, schema)


def test_type_conversion():
    """Test natural language to type conversion."""
    converter = TypeConverter()

    assert converter.to_integer("five") == 5
    assert converter.to_boolean("yes") == True
    assert converter.to_array("a, b, c") == ["a", "b", "c"]
````

### Integration Tests

```python
def test_end_to_end_function_calling():
    """Test complete function calling flow."""

    # Setup
    functions = [
        {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    ]

    # Simulate LLM call
    user_input = "What's the weather in Paris?"
    llm_output = call_llm_with_functions(user_input, functions)

    # Verify function call generated
    assert "function_call" in llm_output
    assert llm_output["function_call"]["name"] == "get_weather"
    assert llm_output["function_call"]["arguments"]["city"] == "Paris"

    # Execute function
    result = execute_function(
        llm_output["function_call"]["name"],
        llm_output["function_call"]["arguments"]
    )

    assert "temperature" in result
    assert "conditions" in result
```

## Summary

Function calling is the bridge between language understanding and programmatic action. Key takeaways:

**Core Concepts**:

- Function calling translates natural language to structured function invocations
- Requires clear schemas, parameter descriptions, and validation
- Can be implemented via native APIs or prompt-based approaches

**Implementation**:

- Define comprehensive schemas with detailed descriptions
- Validate at multiple levels: syntax, types, business logic
- Handle errors gracefully with clear error messages
- Support complex types: nested objects, arrays, unions

**Best Practices**:

- Use descriptive function names and clear descriptions
- Provide parameter guidance and examples
- Implement robust validation and error handling
- Test thoroughly with various inputs and edge cases

**Common Patterns**:

- Single function call: simple one-shot execution
- Sequential calls: chain functions with data flow
- Parallel calls: execute independent functions simultaneously
- Conditional calls: decide whether to call based on context

Function calling is foundational to agent systems. Master it well, as all other tool use patterns build on these fundamentals.

## Next Steps

Now that you understand function calling fundamentals, explore:

- [Tool Schemas](tool-schemas.md) - Design effective tool descriptions and enable dynamic discovery
- [Tool Selection](tool-selection.md) - How agents choose which tool to use in complex scenarios
- [Error Handling](error-handling.md) - Building resilient tool integration with comprehensive error handling
- [Agent Loop](../agent-architectures/agent-loop.md) - Integrating function calling into agent architectures
- [ReAct](../agent-architectures/react.md) - Combining reasoning with function calling
