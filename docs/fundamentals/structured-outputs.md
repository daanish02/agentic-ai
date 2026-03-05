# Structured Outputs

## Table of Contents

- [Introduction](#introduction)
- [What are Structured Outputs?](#what-are-structured-outputs)
- [Why Structured Outputs Matter](#why-structured-outputs-matter)
- [Output Formats](#output-formats)
- [JSON Outputs](#json-outputs)
- [JSON Schema Validation](#json-schema-validation)
- [Constraining LLM Outputs](#constraining-llm-outputs)
- [Parsing Structured Responses](#parsing-structured-responses)
- [Type Safety](#type-safety)
- [Structured Tool Calls](#structured-tool-calls)
- [Complex Nested Structures](#complex-nested-structures)
- [Handling Parse Failures](#handling-parse-failures)
- [Output Templates](#output-templates)
- [Streaming Structured Outputs](#streaming-structured-outputs)
- [Best Practices](#best-practices)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Language models are excellent at generating natural language, but agents need more than prose--they need data they can act on. Structured outputs transform free-form text into predictable, parseable data structures that your code can reliably process.

Without structured outputs, you're stuck parsing unpredictable text:

```python
response = "The temperature is probably around 72 degrees, maybe 73..."
# How do you extract the temperature reliably?
```

With structured outputs:

```python
response = {"temperature": 72, "unit": "fahrenheit", "confidence": 0.95}
# Clean, parseable, actionable
```

Structured outputs are critical for:

- Reliable parsing
- Type safety
- Programmatic processing
- Integration with downstream systems
- Deterministic behavior

> "An agent without structured outputs is a liability."

This guide covers structured outputs from fundamentals to advanced patterns: JSON schemas, parsing strategies, validation, type safety, error handling, and best practices for building robust agent systems.

## What are Structured Outputs?

**Structured outputs** are LLM responses formatted according to a predefined schema. Instead of free-form text, the model returns data in a specific format (usually JSON) that matches an expected structure.

### Unstructured vs. Structured

**Unstructured output**:

```python
user: "What's the weather in Tokyo?"
llm: "Looking at current conditions, Tokyo is experiencing
      mostly cloudy weather with a temperature of around 18
      degrees Celsius. The humidity is at 65% and there's a
      light breeze from the east."
```

Parsing this reliably is hard. Where's the temperature? What units? What's the humidity value?

**Structured output**:

```python
user: "What's the weather in Tokyo?"
llm: {
    "city": "Tokyo",
    "temperature": 18,
    "unit": "celsius",
    "condition": "mostly cloudy",
    "humidity": 65,
    "wind": {
        "speed": 12,
        "direction": "east"
    }
}
```

Now you can reliably access `response["temperature"]` and `response["humidity"]`.

### Structure Types

**1. Flat structure**:

```json
{
  "name": "John Doe",
  "age": 30,
  "city": "Tokyo"
}
```

**2. Nested structure**:

```json
{
  "person": {
    "name": "John Doe",
    "age": 30,
    "address": {
      "city": "Tokyo",
      "country": "Japan"
    }
  }
}
```

**3. Lists**:

```json
{
  "results": [
    { "id": 1, "title": "First" },
    { "id": 2, "title": "Second" }
  ]
}
```

**4. Mixed types**:

```json
{
  "summary": "Text summary here",
  "confidence": 0.95,
  "tags": ["tag1", "tag2"],
  "metadata": {
    "processed": true,
    "timestamp": 1234567890
  }
}
```

## Why Structured Outputs Matter

### 1. Reliability

**Without structure**:

```python
# Fragile parsing
response = "The answer is probably 42, or maybe 43"
try:
    answer = int(re.search(r'\d+', response).group())
except:
    answer = None  # Parsing failed
```

**With structure**:

```python
# Reliable access
response = {"answer": 42, "confidence": 0.95}
answer = response["answer"]  # Always works
```

### 2. Type Safety

```python
# With structured outputs, you know the types
class WeatherResponse:
    temperature: float
    condition: str
    humidity: int

# Your IDE can provide autocomplete and type checking
```

### 3. Validation

```python
# Validate against schema
schema = {
    "type": "object",
    "properties": {
        "temperature": {"type": "number"},
        "condition": {"type": "string"}
    },
    "required": ["temperature"]
}

# If validation passes, you know the data is correct
```

### 4. Composability

```python
# Outputs from one agent can be inputs to another
agent1_output = {"data": [...], "summary": "..."}
agent2_input = agent1_output  # Direct pass-through
```

### 5. Downstream Integration

```python
# Easy to insert into databases, APIs, etc.
weather_data = get_weather_structured()
db.insert(weather_data)  # Works directly

# vs. parsing text first
weather_text = get_weather_unstructured()
parsed = parse_somehow(weather_text)  # Fragile
db.insert(parsed)
```

## Output Formats

Common formats for structured outputs.

### JSON (Most Common)

```python
{
    "status": "success",
    "data": {
        "result": 42
    }
}
```

**Pros**:

- Universal support
- Human readable
- Well-defined spec
- Easy to parse

**Cons**:

- Can be verbose
- No comments
- LLMs sometimes make syntax errors

### XML

```xml
<response>
    <status>success</status>
    <data>
        <result>42</result>
    </data>
</response>
```

**Pros**:

- Self-describing
- Hierarchical
- Good for documents

**Cons**:

- Verbose
- Harder to parse
- Less common in modern APIs

### YAML

```yaml
status: success
data:
  result: 42
```

**Pros**:

- Human readable
- Less verbose than JSON
- Supports comments

**Cons**:

- More parsing complexity
- LLMs less familiar with it

### Key-Value Pairs

```python
status: success
result: 42
confidence: 0.95
```

**Pros**:

- Simple
- Easy to parse
- Less prone to syntax errors

**Cons**:

- Limited structure
- No nested data

### Structured Text

```python
ANSWER: 42
CONFIDENCE: 0.95
EXPLANATION: The answer was calculated based on...
```

**Pros**:

- Simple
- Less prone to JSON syntax errors
- Easy to parse with regex

**Cons**:

- Limited nesting
- Not standard

## JSON Outputs

JSON is the de facto standard for structured outputs.

### Basic JSON Output

**Prompt engineering for JSON**:

```python
def request_json_output(query: str) -> str:
    """Request JSON output from LLM."""

    prompt = f"""
{query}

Respond in JSON format with the following structure:
{{
    "answer": "your answer here",
    "confidence": 0.95,
    "reasoning": "explanation"
}}

Only output valid JSON, nothing else.
"""

    return call_llm(prompt)
```

### Example Implementation

```python
import json
from typing import Dict, Any

class StructuredOutputAgent:
    """Agent that returns structured JSON outputs."""

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

    def query(self, user_input: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Query agent and get structured response."""

        # Build prompt with schema
        prompt = self._build_prompt(user_input, schema)

        # Get response
        response = self._call_llm(prompt)

        # Parse JSON
        try:
            parsed = json.loads(response)
            return parsed
        except json.JSONDecodeError as e:
            # Try to extract JSON from response
            return self._extract_json(response)

    def _build_prompt(self, user_input: str, schema: Dict[str, Any]) -> str:
        """Build prompt requesting structured output."""

        prompt = f"{self.system_prompt}\n\n"
        prompt += f"User: {user_input}\n\n"
        prompt += "Respond with JSON matching this schema:\n"
        prompt += json.dumps(schema, indent=2)
        prompt += "\n\nJSON response:"

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call LLM (placeholder)."""
        # In production, call actual LLM
        return '{"answer": "42", "confidence": 0.95}'

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text that might contain other content."""

        # Find JSON object in text
        start = text.find('{')
        end = text.rfind('}')

        if start == -1 or end == -1:
            raise ValueError("No JSON found in response")

        json_str = text[start:end+1]
        return json.loads(json_str)
```

### Providing Examples

LLMs perform better with examples:

```python
def request_json_with_examples(query: str) -> str:
    """Request JSON with examples."""

    prompt = f"""
{query}

Respond in JSON format. Here are examples:

Example 1:
Input: "What's 2+2?"
Output: {{"answer": 4, "operation": "addition"}}

Example 2:
Input: "What's the capital of France?"
Output: {{"answer": "Paris", "country": "France"}}

Now respond to: {query}
Output:
"""

    return call_llm(prompt)
```

## JSON Schema Validation

Ensure outputs match expected structure.

### Basic Schema

```python
# Define schema
weather_schema = {
    "type": "object",
    "properties": {
        "temperature": {
            "type": "number",
            "description": "Temperature in celsius"
        },
        "condition": {
            "type": "string",
            "enum": ["sunny", "cloudy", "rainy", "snowy"]
        },
        "humidity": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100
        }
    },
    "required": ["temperature", "condition"]
}
```

### Validation Implementation

```python
import jsonschema
from jsonschema import validate, ValidationError

class SchemaValidator:
    """Validate JSON outputs against schemas."""

    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema

    def validate(self, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate data against schema."""

        try:
            validate(instance=data, schema=self.schema)
            return True, None
        except ValidationError as e:
            return False, str(e)

    def validate_and_fix(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and attempt to fix common issues."""

        valid, error = self.validate(data)

        if valid:
            return data

        # Try to fix common issues
        fixed_data = data.copy()

        # Add missing required fields with defaults
        required = self.schema.get("required", [])
        properties = self.schema.get("properties", {})

        for field in required:
            if field not in fixed_data:
                # Add default value based on type
                field_type = properties[field].get("type")

                if field_type == "string":
                    fixed_data[field] = ""
                elif field_type == "number":
                    fixed_data[field] = 0.0
                elif field_type == "integer":
                    fixed_data[field] = 0
                elif field_type == "boolean":
                    fixed_data[field] = False
                elif field_type == "array":
                    fixed_data[field] = []
                elif field_type == "object":
                    fixed_data[field] = {}

        # Validate again
        valid, error = self.validate(fixed_data)

        if valid:
            return fixed_data
        else:
            raise ValueError(f"Could not fix validation error: {error}")


# Usage
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0}
    },
    "required": ["name"]
}

validator = SchemaValidator(schema)

data = {"name": "Alice", "age": 30}
valid, error = validator.validate(data)

if not valid:
    print(f"Validation error: {error}")
```

### Complex Schema Example

```python
# Schema for complex nested structure
user_profile_schema = {
    "type": "object",
    "properties": {
        "user_id": {
            "type": "string",
            "pattern": "^[a-zA-Z0-9_]+$"
        },
        "personal_info": {
            "type": "object",
            "properties": {
                "first_name": {"type": "string"},
                "last_name": {"type": "string"},
                "email": {
                    "type": "string",
                    "format": "email"
                },
                "age": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 150
                }
            },
            "required": ["first_name", "last_name", "email"]
        },
        "preferences": {
            "type": "object",
            "properties": {
                "theme": {
                    "type": "string",
                    "enum": ["light", "dark", "auto"]
                },
                "notifications": {"type": "boolean"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 10
                }
            }
        }
    },
    "required": ["user_id", "personal_info"]
}
```

## Constraining LLM Outputs

Techniques to get LLMs to produce valid structured outputs.

### Prompt Engineering

**Clear instructions**:

```python
prompt = """
Extract information and respond in JSON format.

Rules:
1. Use only valid JSON
2. Include all required fields
3. Use correct data types
4. Don't add extra fields

Required JSON structure:
{
    "name": string,
    "age": integer,
    "email": string
}

Text to extract from: "{user_text}"

JSON:
"""
```

### Few-Shot Examples

```python
def build_prompt_with_examples(task: str) -> str:
    """Build prompt with few-shot examples."""

    examples = [
        {
            "input": "John Doe, 30 years old, john@example.com",
            "output": {
                "name": "John Doe",
                "age": 30,
                "email": "john@example.com"
            }
        },
        {
            "input": "Jane Smith, age 25, jane.smith@test.com",
            "output": {
                "name": "Jane Smith",
                "age": 25,
                "email": "jane.smith@test.com"
            }
        }
    ]

    prompt = "Extract information into JSON format.\n\n"
    prompt += "Examples:\n\n"

    for i, example in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Input: {example['input']}\n"
        prompt += f"Output: {json.dumps(example['output'])}\n\n"

    prompt += f"Now extract from:\nInput: {task}\nOutput:"

    return prompt
```

### Template Filling

```python
def template_filling_approach(data: str) -> str:
    """Use template that LLM fills in."""

    template = {
        "name": "<FILL>",
        "age": "<FILL>",
        "email": "<FILL>"
    }

    prompt = f"""
Fill in this JSON template with information from the text.
Replace <FILL> with actual values.

Template:
{json.dumps(template, indent=2)}

Text: {data}

Filled JSON:
"""

    return prompt
```

### Grammar-Based Constraints

Some modern LLMs support grammar constraints:

```python
# With constrained decoding (if supported by LLM API)
response = llm.generate(
    prompt="Extract user info",
    grammar="""
    root ::= "{" "\"name\":" string "," "\"age\":" number "}"
    string ::= "\"" [^"]* "\""
    number ::= [0-9]+
    """
)
```

### Structured Output APIs

Many modern LLMs have native structured output support:

```python
# OpenAI structured outputs (example)
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Extract user info"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "user_info",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "number"}
                },
                "required": ["name", "age"]
            }
        }
    }
)
```

## Parsing Structured Responses

Robust parsing is critical.

### Basic JSON Parser

````python
import json
import re
from typing import Optional, Dict, Any

class StructuredParser:
    """Parse structured outputs from LLM responses."""

    def parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from response."""

        # Try direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from text
        extracted = self._extract_json(text)
        if extracted:
            try:
                return json.loads(extracted)
            except json.JSONDecodeError:
                pass

        # Try fixing common JSON errors
        fixed = self._fix_json_errors(text)
        if fixed:
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

        return None

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON object from text."""

        # Look for JSON object
        pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            # Return the largest match (likely the complete object)
            return max(matches, key=len)

        return None

    def _fix_json_errors(self, json_str: str) -> Optional[str]:
        """Fix common JSON syntax errors."""

        # Remove markdown code blocks
        json_str = re.sub(r'```json\s*', '', json_str)
        json_str = re.sub(r'```\s*', '', json_str)

        # Fix single quotes
        json_str = json_str.replace("'", '"')

        # Fix trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        # Fix unquoted keys (simple cases)
        json_str = re.sub(r'(\w+):', r'"\1":', json_str)

        return json_str

    def parse_with_schema(
        self,
        text: str,
        schema: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Parse and validate against schema."""

        parsed = self.parse_json(text)

        if parsed is None:
            return None

        # Validate against schema
        validator = SchemaValidator(schema)
        valid, error = validator.validate(parsed)

        if not valid:
            print(f"Validation error: {error}")
            # Try to fix
            try:
                return validator.validate_and_fix(parsed)
            except:
                return None

        return parsed
````

### Robust Parser with Fallbacks

```python
class RobustParser:
    """Parser with multiple fallback strategies."""

    def parse(self, text: str, expected_fields: List[str]) -> Dict[str, Any]:
        """Parse with multiple strategies."""

        # Strategy 1: Direct JSON parsing
        result = self._try_json(text)
        if result and self._has_fields(result, expected_fields):
            return result

        # Strategy 2: Extract and parse JSON
        result = self._try_extract_json(text)
        if result and self._has_fields(result, expected_fields):
            return result

        # Strategy 3: Key-value parsing
        result = self._try_key_value(text, expected_fields)
        if result and self._has_fields(result, expected_fields):
            return result

        # Strategy 4: Regex extraction
        result = self._try_regex(text, expected_fields)
        if result and self._has_fields(result, expected_fields):
            return result

        raise ValueError("Could not parse response with any strategy")

    def _try_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Try direct JSON parsing."""
        try:
            return json.loads(text)
        except:
            return None

    def _try_extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Try extracting and parsing JSON."""
        # Find JSON object
        start = text.find('{')
        end = text.rfind('}')

        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except:
                pass

        return None

    def _try_key_value(
        self,
        text: str,
        fields: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Try parsing as key-value pairs."""

        result = {}

        for field in fields:
            # Look for patterns like "field: value" or "field = value"
            pattern = rf'{field}\s*[:=]\s*(.+?)(?:\n|$)'
            match = re.search(pattern, text, re.IGNORECASE)

            if match:
                value = match.group(1).strip()
                # Try to parse as number
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Keep as string

                result[field] = value

        return result if result else None

    def _try_regex(
        self,
        text: str,
        fields: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Try regex extraction for each field."""

        result = {}

        # Define patterns for common field types
        patterns = {
            "email": r'[\w\.-]+@[\w\.-]+\.\w+',
            "age": r'\b\d{1,3}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "url": r'https?://\S+',
        }

        for field in fields:
            if field in patterns:
                match = re.search(patterns[field], text)
                if match:
                    result[field] = match.group(0)

        return result if result else None

    def _has_fields(self, data: Dict[str, Any], fields: List[str]) -> bool:
        """Check if data has all required fields."""
        return all(field in data for field in fields)
```

## Type Safety

Ensure outputs match expected types.

### Type Definitions

```python
from typing import TypedDict, List
from dataclasses import dataclass

# Using TypedDict
class WeatherResponse(TypedDict):
    temperature: float
    condition: str
    humidity: int
    wind_speed: float


# Using dataclass
@dataclass
class UserInfo:
    name: str
    age: int
    email: str
    tags: List[str]

    def validate(self) -> bool:
        """Validate data."""
        if self.age < 0:
            return False
        if '@' not in self.email:
            return False
        return True
```

### Type-Safe Parser

```python
from typing import Type, TypeVar, get_type_hints
import json

T = TypeVar('T')

class TypeSafeParser:
    """Parse outputs into typed objects."""

    def parse(self, text: str, target_type: Type[T]) -> T:
        """Parse text into typed object."""

        # Parse JSON
        data = json.loads(text)

        # Get type hints
        hints = get_type_hints(target_type)

        # Validate and convert types
        validated = {}

        for field, field_type in hints.items():
            if field not in data:
                raise ValueError(f"Missing field: {field}")

            value = data[field]

            # Type checking and conversion
            if field_type == int:
                validated[field] = int(value)
            elif field_type == float:
                validated[field] = float(value)
            elif field_type == str:
                validated[field] = str(value)
            elif field_type == bool:
                validated[field] = bool(value)
            else:
                validated[field] = value

        # Create instance
        return target_type(**validated)


# Usage
parser = TypeSafeParser()

response_text = '{"temperature": 22.5, "condition": "sunny", "humidity": 45, "wind_speed": 12.3}'

weather = parser.parse(response_text, WeatherResponse)
print(weather["temperature"])  # Type-safe access
```

### Pydantic Integration

```python
from pydantic import BaseModel, Field, validator

class WeatherData(BaseModel):
    """Type-safe weather data model."""

    temperature: float = Field(..., ge=-100, le=100, description="Temperature in celsius")
    condition: str = Field(..., regex="^(sunny|cloudy|rainy|snowy)$")
    humidity: int = Field(..., ge=0, le=100)
    wind_speed: float = Field(default=0.0, ge=0)

    @validator('temperature')
    def check_temperature(cls, v):
        """Validate temperature is reasonable."""
        if v < -100 or v > 60:
            raise ValueError('Temperature out of reasonable range')
        return v

    class Config:
        schema_extra = {
            "example": {
                "temperature": 22.5,
                "condition": "sunny",
                "humidity": 45,
                "wind_speed": 12.3
            }
        }


# Usage with LLM
def get_weather_structured(city: str) -> WeatherData:
    """Get weather with type-safe structured output."""

    # Get schema from Pydantic model
    schema = WeatherData.schema()

    # Request from LLM
    prompt = f"Get weather for {city}. Return JSON matching: {json.dumps(schema)}"
    response = call_llm(prompt)

    # Parse and validate
    try:
        weather = WeatherData.parse_raw(response)
        return weather
    except Exception as e:
        raise ValueError(f"Failed to parse weather data: {e}")


# Type-safe usage
weather = get_weather_structured("Tokyo")
print(weather.temperature)  # IDE autocomplete works!
print(weather.condition)    # Type checking works!
```

## Structured Tool Calls

Ensuring tool calls are properly structured.

### Tool Call Schema

```python
tool_call_schema = {
    "type": "object",
    "properties": {
        "tool_name": {
            "type": "string",
            "description": "Name of the tool to call"
        },
        "arguments": {
            "type": "object",
            "description": "Arguments to pass to the tool"
        },
        "reason": {
            "type": "string",
            "description": "Why this tool is being called"
        }
    },
    "required": ["tool_name", "arguments"]
}
```

### Structured Tool Call Parser

```python
class ToolCallParser:
    """Parse structured tool calls from LLM."""

    def __init__(self, tool_registry: Dict[str, Any]):
        self.tool_registry = tool_registry

    def parse_tool_call(self, text: str) -> Dict[str, Any]:
        """Parse tool call from text."""

        # Extract JSON
        parser = StructuredParser()
        call_data = parser.parse_json(text)

        if not call_data:
            raise ValueError("Could not parse tool call")

        # Validate structure
        if "tool_name" not in call_data:
            raise ValueError("Missing tool_name in call")

        if "arguments" not in call_data:
            raise ValueError("Missing arguments in call")

        tool_name = call_data["tool_name"]

        # Validate tool exists
        if tool_name not in self.tool_registry:
            raise ValueError(f"Unknown tool: {tool_name}")

        # Validate arguments against tool schema
        tool = self.tool_registry[tool_name]
        arg_schema = tool.get("parameters", {})

        validator = SchemaValidator(arg_schema)
        valid, error = validator.validate(call_data["arguments"])

        if not valid:
            raise ValueError(f"Invalid arguments: {error}")

        return call_data

    def parse_multiple_calls(self, text: str) -> List[Dict[str, Any]]:
        """Parse multiple tool calls."""

        calls = []

        # Split by tool call markers
        sections = text.split("TOOL_CALL:")

        for section in sections[1:]:  # Skip first (before any call)
            try:
                call = self.parse_tool_call(section)
                calls.append(call)
            except ValueError as e:
                print(f"Failed to parse tool call: {e}")

        return calls
```

## Complex Nested Structures

Handling deeply nested outputs.

### Nested Structure Example

```python
complex_schema = {
    "type": "object",
    "properties": {
        "user": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "profile": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "contacts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "value": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            }
        },
        "preferences": {
            "type": "object",
            "additionalProperties": {"type": "string"}
        }
    }
}
```

### Nested Parser

```python
class NestedStructureHandler:
    """Handle complex nested structures."""

    def flatten_for_llm(self, schema: Dict[str, Any]) -> str:
        """Convert schema to LLM-friendly description."""

        def describe_property(name: str, prop: Dict[str, Any], level: int = 0) -> str:
            indent = "  " * level
            prop_type = prop.get("type", "any")
            description = prop.get("description", "")

            result = f"{indent}{name} ({prop_type})"
            if description:
                result += f": {description}"
            result += "\n"

            # Handle nested objects
            if prop_type == "object":
                properties = prop.get("properties", {})
                for sub_name, sub_prop in properties.items():
                    result += describe_property(sub_name, sub_prop, level + 1)

            # Handle arrays
            elif prop_type == "array":
                items = prop.get("items", {})
                result += f"{indent}  Each item:\n"
                if items.get("type") == "object":
                    for sub_name, sub_prop in items.get("properties", {}).items():
                        result += describe_property(sub_name, sub_prop, level + 2)

            return result

        output = "Structure:\n"
        for name, prop in schema.get("properties", {}).items():
            output += describe_property(name, prop)

        return output

    def validate_nested(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str = "root"
    ) -> List[str]:
        """Validate nested structure and return all errors."""

        errors = []
        schema_type = schema.get("type")

        # Check type
        if schema_type == "object":
            if not isinstance(data, dict):
                errors.append(f"{path}: Expected object, got {type(data).__name__}")
                return errors

            # Check required fields
            required = schema.get("required", [])
            for field in required:
                if field not in data:
                    errors.append(f"{path}.{field}: Required field missing")

            # Validate properties
            properties = schema.get("properties", {})
            for field, value in data.items():
                if field in properties:
                    field_errors = self.validate_nested(
                        value,
                        properties[field],
                        f"{path}.{field}"
                    )
                    errors.extend(field_errors)

        elif schema_type == "array":
            if not isinstance(data, list):
                errors.append(f"{path}: Expected array, got {type(data).__name__}")
                return errors

            items_schema = schema.get("items", {})
            for i, item in enumerate(data):
                item_errors = self.validate_nested(
                    item,
                    items_schema,
                    f"{path}[{i}]"
                )
                errors.extend(item_errors)

        elif schema_type == "string":
            if not isinstance(data, str):
                errors.append(f"{path}: Expected string, got {type(data).__name__}")

        elif schema_type == "number":
            if not isinstance(data, (int, float)):
                errors.append(f"{path}: Expected number, got {type(data).__name__}")

        elif schema_type == "integer":
            if not isinstance(data, int):
                errors.append(f"{path}: Expected integer, got {type(data).__name__}")

        elif schema_type == "boolean":
            if not isinstance(data, bool):
                errors.append(f"{path}: Expected boolean, got {type(data).__name__}")

        return errors
```

## Handling Parse Failures

Robust error handling for parsing failures.

### Retry with Different Prompts

```python
class ParseRetryHandler:
    """Handle parse failures with retries."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def parse_with_retry(
        self,
        query: str,
        schema: Dict[str, Any],
        llm_call: Callable
    ) -> Dict[str, Any]:
        """Parse with retries on failure."""

        parser = StructuredParser()

        for attempt in range(self.max_retries):
            # Adjust prompt based on attempt
            prompt = self._build_prompt(query, schema, attempt)

            # Get response
            response = llm_call(prompt)

            # Try to parse
            parsed = parser.parse_with_schema(response, schema)

            if parsed:
                return parsed

            print(f"Parse failed on attempt {attempt + 1}")

        raise ValueError(f"Failed to parse after {self.max_retries} attempts")

    def _build_prompt(
        self,
        query: str,
        schema: Dict[str, Any],
        attempt: int
    ) -> str:
        """Build prompt adjusted for retry attempt."""

        base_prompt = f"{query}\n\nRespond in JSON format: {json.dumps(schema)}"

        if attempt == 0:
            return base_prompt

        elif attempt == 1:
            return f"""{base_prompt}

IMPORTANT: Your previous response could not be parsed.
Ensure you output ONLY valid JSON, nothing else.
"""

        else:
            return f"""{base_prompt}

CRITICAL: Multiple parsing failures.
Rules:
1. Output ONLY JSON
2. No extra text before or after
3. Use double quotes for strings
4. No trailing commas
5. All required fields must be present

JSON response:
"""
```

### Fallback to Simpler Format

```python
def parse_with_fallback(text: str) -> Dict[str, Any]:
    """Try JSON, fall back to simpler formats."""

    # Try JSON
    try:
        return json.loads(text)
    except:
        pass

    # Try extracting JSON
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
    except:
        pass

    # Fall back to key-value pairs
    result = {}
    for line in text.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            result[key.strip()] = value.strip()

    if result:
        return result

    # Last resort: return as text
    return {"text": text}
```

### Error Recovery

```python
class ParseErrorRecovery:
    """Recover from parsing errors."""

    def recover(
        self,
        text: str,
        schema: Dict[str, Any],
        llm_call: Callable
    ) -> Dict[str, Any]:
        """Attempt to recover from parse error."""

        # Try to understand what went wrong
        error_analysis = self._analyze_error(text, schema)

        # Ask LLM to fix it
        fix_prompt = f"""
The following response could not be parsed:
{text}

Expected schema:
{json.dumps(schema, indent=2)}

Problem identified: {error_analysis}

Please provide a corrected version that matches the schema exactly.
Output only valid JSON:
"""

        fixed_response = llm_call(fix_prompt)

        # Try parsing fixed response
        try:
            return json.loads(fixed_response)
        except:
            # If still fails, use partial extraction
            return self._partial_extract(text, schema)

    def _analyze_error(self, text: str, schema: Dict[str, Any]) -> str:
        """Analyze what went wrong."""

        if not text.strip().startswith('{'):
            return "Response doesn't start with '{'"

        if not text.strip().endswith('}'):
            return "Response doesn't end with '}'"

        # Check for common JSON errors
        if "'" in text:
            return "Uses single quotes instead of double quotes"

        if text.count('{') != text.count('}'):
            return "Mismatched braces"

        return "Unknown JSON syntax error"

    def _partial_extract(
        self,
        text: str,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract partial data even if parsing fails."""

        result = {}
        properties = schema.get("properties", {})

        # Try to extract each field
        for field in properties:
            # Look for field in text
            pattern = rf'"{field}"\s*:\s*([^,}}]+)'
            match = re.search(pattern, text)

            if match:
                value_str = match.group(1).strip().strip('"')

                # Convert to appropriate type
                field_type = properties[field].get("type")
                if field_type == "integer":
                    try:
                        result[field] = int(value_str)
                    except:
                        pass
                elif field_type == "number":
                    try:
                        result[field] = float(value_str)
                    except:
                        pass
                else:
                    result[field] = value_str

        return result
```

## Output Templates

Provide templates for consistent outputs.

### Template System

```python
class OutputTemplate:
    """Template system for structured outputs."""

    def __init__(self, template: Dict[str, Any]):
        self.template = template

    def to_schema(self) -> Dict[str, Any]:
        """Convert template to JSON schema."""

        def infer_type(value):
            if isinstance(value, str):
                return "string"
            elif isinstance(value, int):
                return "integer"
            elif isinstance(value, float):
                return "number"
            elif isinstance(value, bool):
                return "boolean"
            elif isinstance(value, list):
                return "array"
            elif isinstance(value, dict):
                return "object"
            return "string"

        def build_schema(obj):
            if isinstance(obj, dict):
                properties = {}
                for key, value in obj.items():
                    if isinstance(value, dict):
                        properties[key] = {
                            "type": "object",
                            "properties": build_schema(value)
                        }
                    elif isinstance(value, list) and value:
                        properties[key] = {
                            "type": "array",
                            "items": {"type": infer_type(value[0])}
                        }
                    else:
                        properties[key] = {"type": infer_type(value)}
                return properties
            return {}

        return {
            "type": "object",
            "properties": build_schema(self.template)
        }

    def to_prompt(self) -> str:
        """Convert template to prompt description."""
        return f"Respond with JSON matching this template:\n{json.dumps(self.template, indent=2)}"

    def fill(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fill template with data."""

        def fill_recursive(template, data):
            if isinstance(template, dict):
                result = {}
                for key in template:
                    if key in data:
                        result[key] = fill_recursive(template[key], data[key])
                    else:
                        result[key] = template[key]  # Keep placeholder
                return result
            else:
                return data

        return fill_recursive(self.template, data)


# Usage
template = OutputTemplate({
    "user_name": "<NAME>",
    "age": 0,
    "email": "<EMAIL>",
    "address": {
        "city": "<CITY>",
        "country": "<COUNTRY>"
    }
})

# Get schema
schema = template.to_schema()

# Get prompt
prompt = template.to_prompt()

# Fill with data
filled = template.fill({
    "user_name": "John Doe",
    "age": 30,
    "email": "john@example.com",
    "address": {
        "city": "Tokyo",
        "country": "Japan"
    }
})
```

## Streaming Structured Outputs

Handle structured outputs in streaming mode.

### Stream Parser

```python
import json
from typing import Iterator, Optional

class StreamingStructuredParser:
    """Parse structured outputs from streaming responses."""

    def __init__(self):
        self.buffer = ""
        self.complete = False

    def add_chunk(self, chunk: str) -> Optional[Dict[str, Any]]:
        """Add a chunk and try to parse if complete."""

        self.buffer += chunk

        # Check if we have a complete JSON object
        if self.buffer.count('{') == self.buffer.count('}'):
            try:
                parsed = json.loads(self.buffer)
                self.complete = True
                return parsed
            except json.JSONDecodeError:
                pass  # Not complete yet

        return None

    def get_partial(self) -> Dict[str, Any]:
        """Get partial results from incomplete JSON."""

        # Extract complete fields
        partial = {}

        # Simple regex-based extraction
        pattern = r'"(\w+)"\s*:\s*"([^"]*)"'
        matches = re.findall(pattern, self.buffer)

        for key, value in matches:
            partial[key] = value

        # Extract numbers
        pattern = r'"(\w+)"\s*:\s*(\d+\.?\d*)'
        matches = re.findall(pattern, self.buffer)

        for key, value in matches:
            try:
                partial[key] = float(value) if '.' in value else int(value)
            except:
                pass

        return partial


# Usage with streaming
def process_streaming_response(stream: Iterator[str]) -> Dict[str, Any]:
    """Process streaming response into structured output."""

    parser = StreamingStructuredParser()

    for chunk in stream:
        result = parser.add_chunk(chunk)

        if result:
            return result

        # Optionally show partial results
        partial = parser.get_partial()
        if partial:
            print(f"Partial: {partial}")

    # If stream ended without complete JSON
    if not parser.complete:
        raise ValueError("Stream ended without complete JSON")

    return parser.get_partial()
```

## Best Practices

Guidelines for reliable structured outputs.

### 1. Always Provide Schema

```python
def good_structured_request(query: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Always include schema in request."""

    prompt = f"""
{query}

Respond with JSON matching this schema:
{json.dumps(schema, indent=2)}

Include all required fields.
Use correct types.
Output only valid JSON.
"""

    return call_llm(prompt)
```

### 2. Validate All Outputs

```python
def validated_output(response: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Always validate parsed outputs."""

    # Parse
    parsed = json.loads(response)

    # Validate
    validator = SchemaValidator(schema)
    valid, error = validator.validate(parsed)

    if not valid:
        raise ValueError(f"Validation failed: {error}")

    return parsed
```

### 3. Provide Examples

```python
def request_with_examples(task: str) -> Dict[str, Any]:
    """Include examples for clarity."""

    prompt = f"""
{task}

Example output:
{{
    "name": "John Doe",
    "age": 30,
    "city": "Tokyo"
}}

Your output:
"""

    return call_llm(prompt)
```

### 4. Handle Failures Gracefully

```python
def safe_structured_call(
    query: str,
    schema: Dict[str, Any],
    max_retries: int = 3
) -> Optional[Dict[str, Any]]:
    """Safe structured call with error handling."""

    for attempt in range(max_retries):
        try:
            response = call_llm(query)
            parsed = json.loads(response)

            validator = SchemaValidator(schema)
            valid, error = validator.validate(parsed)

            if valid:
                return parsed

            print(f"Validation failed: {error}")

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")

        time.sleep(1)  # Brief delay

    return None  # Failed after retries
```

### 5. Log Parsing Issues

```python
class StructuredOutputLogger:
    """Log parsing issues for debugging."""

    def __init__(self):
        self.failures = []

    def log_failure(
        self,
        response: str,
        schema: Dict[str, Any],
        error: str
    ):
        """Log a parsing failure."""
        self.failures.append({
            "timestamp": time.time(),
            "response": response,
            "schema": schema,
            "error": error
        })

    def get_failure_rate(self) -> float:
        """Get overall failure rate."""
        # Would need to track successes too
        pass

    def get_common_errors(self) -> List[str]:
        """Get most common error types."""
        from collections import Counter
        errors = [f["error"] for f in self.failures]
        return Counter(errors).most_common(5)
```

### 6. Test with Edge Cases

```python
def test_structured_outputs():
    """Test structured output handling."""

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    }

    # Test valid
    valid = '{"name": "Alice", "age": 30}'
    assert parse_and_validate(valid, schema) is not None

    # Test missing optional field
    missing_optional = '{"name": "Bob"}'
    assert parse_and_validate(missing_optional, schema) is not None

    # Test missing required field
    missing_required = '{"age": 25}'
    try:
        parse_and_validate(missing_required, schema)
        assert False, "Should have failed"
    except:
        pass  # Expected

    # Test wrong type
    wrong_type = '{"name": "Charlie", "age": "thirty"}'
    try:
        parse_and_validate(wrong_type, schema)
        assert False, "Should have failed"
    except:
        pass  # Expected
```

## Summary

Structured outputs transform unreliable text into actionable data. Key principles:

**Core Concepts**:

- Structured outputs follow predefined schemas
- JSON is the most common format
- Schemas define expected structure and types
- Validation ensures correctness
- Type safety prevents runtime errors

**Implementation**:

- Define clear schemas
- Request structured outputs explicitly
- Provide examples and templates
- Parse robustly with fallbacks
- Validate all outputs
- Handle errors gracefully

**Techniques**:

- Prompt engineering for structure
- Few-shot examples
- Template filling
- Grammar constraints
- Multi-strategy parsing
- Error recovery

**Best Practices**:

- Always provide schemas
- Validate all outputs
- Include examples
- Handle failures gracefully
- Log parsing issues
- Test edge cases
- Use type-safe models (Pydantic)

Structured outputs are the foundation of reliable agent systems. Master them, and your agents become predictable and trustworthy.

## Next Steps

Now that you understand structured outputs, explore:

- **[Basic Tool Calling](basic-tool-calling.md)** - Tools need structured inputs/outputs
- **[Basic Error Handling](basic-error-handling.md)** - Handling parsing failures
- **[State Basics](state-basics.md)** - Storing structured state
- **Type Systems** - Advanced type safety patterns
- **Schema Evolution** - Managing changing schemas over time

**Practice exercises**:

1. Implement a complete structured output system with validation
2. Build a multi-strategy parser that handles various error modes
3. Create a type-safe wrapper using Pydantic or similar
4. Implement streaming structured output parsing
5. Build a template system for consistent outputs

**Advanced topics**:

- Constrained decoding for guaranteed structure
- Schema inference from examples
- Multi-modal structured outputs
- Streaming partial results
- Cross-language type generation

Structured outputs bridge the gap between language and code. Build this bridge well.
