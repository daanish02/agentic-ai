# Tool Schemas and Discovery

## Table of Contents

- [Introduction](#introduction)
- [What is a Tool Schema?](#what-is-a-tool-schema)
- [Schema Anatomy](#schema-anatomy)
- [Writing Effective Descriptions](#writing-effective-descriptions)
- [Parameter Specifications](#parameter-specifications)
- [Describing Tool Capabilities](#describing-tool-capabilities)
- [Schema Design Patterns](#schema-design-patterns)
- [Dynamic Tool Discovery](#dynamic-tool-discovery)
- [Tool Categories and Organization](#tool-categories-and-organization)
- [Versioning Tool Schemas](#versioning-tool-schemas)
- [Schema Validation](#schema-validation)
- [Context-Aware Tool Descriptions](#context-aware-tool-descriptions)
- [Multi-Language Support](#multi-language-support)
- [Performance Considerations](#performance-considerations)
- [Testing Tool Schemas](#testing-tool-schemas)
- [Best Practices](#best-practices)
- [Common Pitfalls](#common-pitfalls)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

A tool schema is how you teach an agent what a tool does and how to use it. Think of it as the **instruction manual** that enables a language model to understand and invoke your tools correctly.

Poor schemas lead to confused agents that misuse tools, pass wrong parameters, or fail to recognize when a tool is relevant. Great schemas make tools feel **native** to the model - like they've always been part of its capabilities.

> "A tool is only as useful as the agent's understanding of it."

This guide covers everything about tool schemas: from basic structure to advanced patterns, from static definitions to dynamic discovery, and from single-tool scenarios to managing hundreds of tools.

## What is a Tool Schema?

A tool schema is a **structured description** of a tool that includes:

1. **Identity**: What is the tool called?
2. **Purpose**: What does it do?
3. **Interface**: What parameters does it accept?
4. **Constraints**: What are the requirements and limitations?
5. **Usage**: When and how should it be used?

### Basic Example

```python
tool_schema = {
    "name": "search_web",
    "description": "Search the internet for information",
    "parameters": {
        "query": "The search query string",
        "max_results": "Maximum number of results (default: 10)"
    }
}
```

### Comprehensive Example

```python
comprehensive_schema = {
    "name": "search_web",
    "description": """
    Search the internet for current information using a search engine.
    Returns a list of web pages with titles, URLs, and snippets.
    Use this when you need information not in your training data or when
    users ask about current events, recent news, or real-time information.
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query. Use keywords rather than full questions. Example: 'Python async tutorial' rather than 'How do I use async in Python?'",
                "minLength": 1,
                "maxLength": 200
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "minimum": 1,
                "maximum": 20,
                "default": 10
            },
            "search_type": {
                "type": "string",
                "description": "Type of search to perform",
                "enum": ["web", "news", "images", "videos"],
                "default": "web"
            },
            "language": {
                "type": "string",
                "description": "Language code for results (e.g., 'en', 'es', 'fr')",
                "pattern": "^[a-z]{2}$",
                "default": "en"
            },
            "safe_search": {
                "type": "boolean",
                "description": "Enable safe search filtering",
                "default": true
            }
        },
        "required": ["query"]
    },
    "returns": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "url": {"type": "string"},
                "snippet": {"type": "string"}
            }
        }
    },
    "examples": [
        {
            "query": "Python asyncio tutorial",
            "max_results": 5
        }
    ],
    "tags": ["search", "web", "information retrieval"],
    "version": "1.2.0"
}
```

## Schema Anatomy

Understanding each component of a tool schema.

### Core Components

```python
{
    # REQUIRED: Unique identifier for the tool
    "name": "string",

    # REQUIRED: What the tool does
    "description": "string",

    # REQUIRED: Input specification
    "parameters": {
        "type": "object",
        "properties": {...},
        "required": [...]
    },

    # OPTIONAL: Output specification
    "returns": {...},

    # OPTIONAL: Usage examples
    "examples": [...],

    # OPTIONAL: Metadata
    "tags": [...],
    "category": "string",
    "version": "string"
}
```

### Name

The tool's identifier - must be unique and follow conventions:

```python
# Good names
"search_web"
"get_weather"
"send_email"
"query_database"
"calculate_distance"

# Bad names
"search"  # Too generic
"doSearch"  # Use snake_case, not camelCase
"search-web"  # Use underscore, not hyphen
"SEARCH_WEB"  # Not all caps
```

**Naming conventions**:

- Use `snake_case`
- Start with a verb when possible
- Be specific but concise
- Avoid abbreviations unless very common

```python
def validate_tool_name(name: str) -> bool:
    """Validate tool name follows conventions."""
    import re

    # Must be snake_case
    if not re.match(r'^[a-z][a-z0-9_]*$', name):
        return False

    # Reasonable length
    if len(name) < 3 or len(name) > 50:
        return False

    # Should start with verb (soft check)
    common_verbs = [
        'get', 'set', 'create', 'update', 'delete', 'search',
        'find', 'list', 'calculate', 'send', 'receive', 'query',
        'fetch', 'load', 'save', 'read', 'write'
    ]

    starts_with_verb = any(name.startswith(verb) for verb in common_verbs)

    return True  # Name is valid even if doesn't start with verb
```

### Description

The tool's purpose and usage guidance - this is critical for tool selection:

```python
# Bad: Minimal information
"description": "Search function"

# Good: Comprehensive guidance
"description": """
Search the product catalog for items matching query terms.

Returns: List of products with name, price, availability, and images.

Use this when:
- User wants to find or browse products
- User asks about product availability
- User requests product recommendations

Do NOT use for:
- Searching order history (use search_orders instead)
- Searching user accounts (use search_users instead)
- General web search (use search_web instead)

Examples:
- 'wireless headphones' - find matching products
- 'coffee maker under $50' - search with price filter
- 'laptop for gaming' - find products by use case
"""
```

### Parameters

Detailed specification of inputs:

```python
"parameters": {
    "type": "object",  # Parameters are an object/dict

    # Define each parameter
    "properties": {
        "param_name": {
            "type": "string|integer|number|boolean|array|object",
            "description": "Clear description of this parameter",

            # Optional: Constraints
            "minimum": 0,  # For numbers
            "maximum": 100,
            "minLength": 1,  # For strings
            "maxLength": 200,
            "pattern": "regex",  # For strings
            "enum": ["option1", "option2"],  # Fixed choices
            "default": "value",  # Default if not provided

            # For arrays
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 10,

            # For objects
            "properties": {...},
            "required": [...]
        }
    },

    # Which parameters are required
    "required": ["param1", "param2"]
}
```

## Writing Effective Descriptions

Description writing is an art - it guides the agent's decision-making.

### Top-Level Description Structure

```python
description_template = """
[ONE-LINE SUMMARY]

[WHAT IT RETURNS / DOES]

[WHEN TO USE]
- Use case 1
- Use case 2
- Use case 3

[WHEN NOT TO USE]
- Alternative tool for scenario 1
- Alternative tool for scenario 2

[EXAMPLES / NOTES]
- Example usage pattern
- Important constraints or limitations
"""
```

### Real Example: Email Tool

```python
email_tool_schema = {
    "name": "send_email",
    "description": """
Send an email message to one or more recipients.

Returns: Message ID and delivery status for each recipient.

Use this when:
- User explicitly asks to send an email
- User wants to share information via email
- User requests a reminder or notification via email
- User wants to forward information to someone

Do NOT use:
- For instant messaging (use send_message instead)
- For SMS (use send_sms instead)
- For notifications to self (use create_reminder instead)

Requirements:
- Valid email addresses for recipients
- User must be authenticated with email permissions
- Attachments limited to 25MB total

Examples:
- "Send an email to john@example.com about tomorrow's meeting"
- "Email the report to my team"
- "Forward this information to alice@company.com"
""",
    "parameters": {...}
}
```

### Parameter Description Guidelines

```python
# Bad: Minimal
"email": {"type": "string", "description": "Email address"}

# Good: Detailed
"email": {
    "type": "string",
    "description": "Recipient's email address. Must be a valid email format (user@domain.com). Can use display name format: 'John Doe <john@example.com>'",
    "pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
}
```

```python
# Bad: Unclear format
"date": {"type": "string", "description": "The date"}

# Good: Format specified
"date": {
    "type": "string",
    "description": "Date in ISO 8601 format (YYYY-MM-DD). Example: '2026-01-23'",
    "pattern": "^\\d{4}-\\d{2}-\\d{2}$"
}
```

```python
# Bad: No context
"timeout": {"type": "integer", "description": "Timeout"}

# Good: Units and context
"timeout": {
    "type": "integer",
    "description": "Request timeout in seconds. How long to wait for response before giving up. Recommended: 30 for normal requests, 300 for long operations.",
    "minimum": 1,
    "maximum": 600,
    "default": 30
}
```

## Parameter Specifications

Deep dive into parameter definitions.

### Simple Types

```python
string_param = {
    "type": "string",
    "description": "A text value",
    "minLength": 1,
    "maxLength": 500,
    "pattern": "^[a-zA-Z0-9 ]+$",  # Regex validation
    "default": "default value"
}

integer_param = {
    "type": "integer",
    "description": "A whole number",
    "minimum": 0,
    "maximum": 100,
    "default": 10
}

number_param = {
    "type": "number",
    "description": "A numeric value (int or float)",
    "minimum": 0.0,
    "maximum": 100.0,
    "default": 10.5
}

boolean_param = {
    "type": "boolean",
    "description": "True or false value",
    "default": True
}
```

### Enumerations

```python
enum_param = {
    "type": "string",
    "description": "Priority level for the task",
    "enum": ["low", "medium", "high", "urgent"],
    "default": "medium"
}

# With descriptions for each option
enum_with_desc = {
    "type": "string",
    "description": """
Task priority level:
- 'low': No deadline, can be done anytime
- 'medium': Should be done this week
- 'high': Should be done within 2 days
- 'urgent': Must be done today
""",
    "enum": ["low", "medium", "high", "urgent"],
    "default": "medium"
}
```

### Arrays

```python
simple_array = {
    "type": "array",
    "description": "List of email addresses to send to",
    "items": {"type": "string"},
    "minItems": 1,
    "maxItems": 50,
    "uniqueItems": True  # No duplicates
}

array_of_objects = {
    "type": "array",
    "description": "List of tasks to create",
    "items": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "high"]},
            "due_date": {"type": "string"}
        },
        "required": ["title"]
    },
    "minItems": 1,
    "maxItems": 20
}
```

### Nested Objects

```python
nested_object = {
    "type": "object",
    "description": "User profile information",
    "properties": {
        "personal": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
                "phone": {"type": "string"}
            },
            "required": ["name", "email"]
        },
        "address": {
            "type": "object",
            "properties": {
                "street": {"type": "string"},
                "city": {"type": "string"},
                "country": {"type": "string"},
                "postal_code": {"type": "string"}
            }
        },
        "preferences": {
            "type": "object",
            "properties": {
                "language": {"type": "string", "default": "en"},
                "timezone": {"type": "string", "default": "UTC"},
                "notifications": {"type": "boolean", "default": True}
            }
        }
    },
    "required": ["personal"]
}
```

### Union Types (oneOf)

```python
# Parameter can be one of several types
union_param = {
    "oneOf": [
        {
            "type": "string",
            "description": "ISO datetime string (e.g., '2026-01-23T15:00:00')"
        },
        {
            "type": "integer",
            "description": "Unix timestamp in seconds"
        },
        {
            "type": "object",
            "properties": {
                "date": {"type": "string"},
                "time": {"type": "string"}
            },
            "description": "Separate date and time components"
        }
    ],
    "description": "Time specification in any supported format"
}
```

## Describing Tool Capabilities

Help agents understand what tools can and cannot do.

### Capability Statements

```python
database_tool = {
    "name": "query_database",
    "description": """
Execute SQL queries against the application database.

CAPABILITIES:
- Read data from any table user has access to
- Filter and sort results with WHERE and ORDER BY
- Join multiple tables
- Aggregate data with GROUP BY
- Limit results for performance

LIMITATIONS:
- Read-only access (no INSERT, UPDATE, DELETE)
- Maximum 10,000 rows returned per query
- Query timeout after 30 seconds
- Cannot access system tables or user credentials
- Cannot execute stored procedures

PERFORMANCE:
- Queries on indexed columns are fast (<1s)
- Full table scans may be slow (>5s for large tables)
- Consider using LIMIT for exploratory queries
""",
    "parameters": {...}
}
```

### Scope and Boundaries

```python
search_tool = {
    "name": "search_documents",
    "description": """
Search through internal company documents and knowledge base.

SCOPE:
- All company documentation
- Internal wikis and knowledge bases
- Uploaded documents in the system
- Shared team files

OUT OF SCOPE:
- External websites (use search_web instead)
- Email messages (use search_email instead)
- Code repositories (use search_code instead)
- Real-time data or live systems

DATA FRESHNESS:
- Documents indexed within 5 minutes of upload
- Updates reflected within 15 minutes
- Deleted documents removed from index within 1 hour
""",
    "parameters": {...}
}
```

### Prerequisites and Dependencies

```python
calendar_tool = {
    "name": "create_calendar_event",
    "description": """
Create a new event in the user's calendar.

PREREQUISITES:
- User must be authenticated with calendar access
- Calendar must be accessible (not deleted or suspended)
- User must have permission to create events

DEPENDENCIES:
- Requires valid event time (not in the past)
- If inviting attendees, requires valid email addresses
- If setting reminders, requires notification permissions

SIDE EFFECTS:
- Sends invitation emails to attendees
- Creates notifications based on reminder settings
- Updates free/busy status
- May trigger calendar sync to mobile devices
""",
    "parameters": {...}
}
```

## Schema Design Patterns

Common patterns for different tool types.

### Information Retrieval Tools

```python
retrieval_schema = {
    "name": "search_documents",
    "description": "Search for documents matching criteria",
    "parameters": {
        "type": "object",
        "properties": {
            # Query
            "query": {
                "type": "string",
                "description": "Search query"
            },

            # Filters
            "filters": {
                "type": "object",
                "properties": {
                    "document_type": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "date_range": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "string"},
                            "end": {"type": "string"}
                        }
                    },
                    "author": {"type": "string"}
                }
            },

            # Pagination
            "limit": {
                "type": "integer",
                "default": 20,
                "maximum": 100
            },
            "offset": {
                "type": "integer",
                "default": 0
            },

            # Sorting
            "sort_by": {
                "type": "string",
                "enum": ["relevance", "date", "title"],
                "default": "relevance"
            },
            "sort_order": {
                "type": "string",
                "enum": ["asc", "desc"],
                "default": "desc"
            }
        },
        "required": ["query"]
    }
}
```

### Action/Mutation Tools

```python
action_schema = {
    "name": "send_email",
    "description": "Send an email message",
    "parameters": {
        "type": "object",
        "properties": {
            # Required action parameters
            "to": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": "Recipient email addresses"
            },
            "subject": {
                "type": "string",
                "minLength": 1,
                "description": "Email subject line"
            },
            "body": {
                "type": "string",
                "description": "Email body content"
            },

            # Optional parameters
            "cc": {
                "type": "array",
                "items": {"type": "string"}
            },
            "bcc": {
                "type": "array",
                "items": {"type": "string"}
            },

            # Metadata
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high"],
                "default": "normal"
            },

            # Confirmation
            "send_immediately": {
                "type": "boolean",
                "default": True,
                "description": "If false, save as draft"
            }
        },
        "required": ["to", "subject", "body"]
    }
}
```

### Computation Tools

```python
computation_schema = {
    "name": "calculate_statistics",
    "description": "Calculate statistical measures for a dataset",
    "parameters": {
        "type": "object",
        "properties": {
            # Input data
            "data": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 1,
                "description": "Array of numeric values"
            },

            # What to calculate
            "measures": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "mean", "median", "mode", "std", "variance",
                        "min", "max", "range", "quartiles"
                    ]
                },
                "default": ["mean", "median", "std"],
                "description": "Which statistical measures to compute"
            },

            # Options
            "precision": {
                "type": "integer",
                "minimum": 0,
                "maximum": 10,
                "default": 2,
                "description": "Number of decimal places"
            }
        },
        "required": ["data"]
    }
}
```

## Dynamic Tool Discovery

Enabling agents to discover and learn about tools at runtime.

### Tool Registry

```python
from typing import Dict, List, Any, Optional

class ToolRegistry:
    """
    Registry for dynamic tool discovery.
    """

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.categories: Dict[str, List[str]] = {}

    def register_tool(self, schema: Dict[str, Any]) -> None:
        """Register a tool with the registry."""
        name = schema["name"]
        self.tools[name] = schema

        # Add to category index
        category = schema.get("category", "uncategorized")
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(name)

    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a tool schema by name."""
        return self.tools.get(name)

    def list_tools(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """List tools, optionally filtered by category or tags."""
        tools = self.tools.values()

        if category:
            tools = [t for t in tools if t.get("category") == category]

        if tags:
            tools = [
                t for t in tools
                if any(tag in t.get("tags", []) for tag in tags)
            ]

        return list(tools)

    def search_tools(self, query: str) -> List[Dict[str, Any]]:
        """Search tools by description."""
        query_lower = query.lower()
        results = []

        for tool in self.tools.values():
            # Search in name, description, and tags
            searchable = (
                tool["name"] +
                " " + tool.get("description", "") +
                " " + " ".join(tool.get("tags", []))
            ).lower()

            if query_lower in searchable:
                results.append(tool)

        return results

    def get_tools_summary(self) -> str:
        """Get a summary of all available tools."""
        lines = ["Available Tools:\n"]

        for category, tool_names in sorted(self.categories.items()):
            lines.append(f"\n{category.upper()}:")
            for name in tool_names:
                tool = self.tools[name]
                # First line of description
                desc = tool.get("description", "").split("\n")[0].strip()
                lines.append(f"  - {name}: {desc}")

        return "\n".join(lines)


# Usage
registry = ToolRegistry()

# Register tools
registry.register_tool({
    "name": "search_web",
    "description": "Search the internet",
    "category": "search",
    "tags": ["web", "internet", "search"],
    "parameters": {...}
})

registry.register_tool({
    "name": "search_documents",
    "description": "Search internal documents",
    "category": "search",
    "tags": ["documents", "search", "internal"],
    "parameters": {...}
})

# Discover tools
all_search_tools = registry.list_tools(category="search")
web_tools = registry.search_tools("internet")
summary = registry.get_tools_summary()
```

### Context-Aware Discovery

```python
def discover_relevant_tools(
    user_intent: str,
    user_context: Dict[str, Any],
    registry: ToolRegistry
) -> List[Dict[str, Any]]:
    """
    Discover tools relevant to user's current context and intent.
    """
    # Start with all tools
    candidate_tools = registry.list_tools()

    # Filter by user permissions
    user_permissions = user_context.get("permissions", [])
    candidate_tools = [
        tool for tool in candidate_tools
        if tool.get("required_permission") in user_permissions
        or "required_permission" not in tool
    ]

    # Filter by availability
    available_integrations = user_context.get("integrations", [])
    candidate_tools = [
        tool for tool in candidate_tools
        if tool.get("requires_integration") in available_integrations
        or "requires_integration" not in tool
    ]

    # Rank by relevance to intent
    scored_tools = []
    for tool in candidate_tools:
        score = calculate_relevance(user_intent, tool)
        scored_tools.append((score, tool))

    # Sort by score and return top tools
    scored_tools.sort(reverse=True, key=lambda x: x[0])

    # Return top 10 most relevant tools
    return [tool for score, tool in scored_tools[:10]]


def calculate_relevance(intent: str, tool: Dict[str, Any]) -> float:
    """Calculate how relevant a tool is to user intent."""
    score = 0.0

    # Keyword matching in description
    intent_words = set(intent.lower().split())
    tool_words = set(tool.get("description", "").lower().split())
    overlap = len(intent_words & tool_words)
    score += overlap * 2

    # Tag matching
    tool_tags = set(tool.get("tags", []))
    tag_overlap = len(intent_words & tool_tags)
    score += tag_overlap * 5

    # Exact name match
    if any(word in tool["name"] for word in intent_words):
        score += 10

    return score
```

### Progressive Disclosure

```python
def get_tool_description(
    tool_name: str,
    detail_level: str = "brief"
) -> str:
    """
    Get tool description at different levels of detail.
    """
    tool = registry.get_tool(tool_name)

    if detail_level == "brief":
        # Just name and one-line description
        first_line = tool["description"].split("\n")[0].strip()
        return f"{tool['name']}: {first_line}"

    elif detail_level == "medium":
        # Name, description, and parameter names
        params = list(tool["parameters"]["properties"].keys())
        return f"""
{tool['name']}: {tool['description']}
Parameters: {', '.join(params)}
"""

    elif detail_level == "full":
        # Complete schema
        return format_full_schema(tool)


def adaptive_tool_presentation(
    tools: List[Dict[str, Any]],
    context_window_size: int
) -> str:
    """
    Present tools adaptively based on available context.
    """
    # Estimate token usage
    total_tokens = 0
    presentation = []

    for tool in tools:
        # Start with brief description
        brief = get_tool_description(tool["name"], "brief")
        brief_tokens = len(brief.split())  # Rough estimate

        if total_tokens + brief_tokens < context_window_size * 0.8:
            # Plenty of room, try medium detail
            medium = get_tool_description(tool["name"], "medium")
            medium_tokens = len(medium.split())

            if total_tokens + medium_tokens < context_window_size * 0.9:
                presentation.append(medium)
                total_tokens += medium_tokens
            else:
                presentation.append(brief)
                total_tokens += brief_tokens
        else:
            # Running out of space, just list names
            presentation.append(tool["name"])

    return "\n\n".join(presentation)
```

## Tool Categories and Organization

Organizing tools for better discoverability.

### Category Structure

```python
TOOL_CATEGORIES = {
    "communication": {
        "description": "Tools for sending messages and communicating",
        "tools": ["send_email", "send_sms", "post_message"]
    },
    "search": {
        "description": "Tools for finding information",
        "tools": ["search_web", "search_documents", "search_database"]
    },
    "data": {
        "description": "Tools for data access and manipulation",
        "tools": ["query_database", "read_file", "write_file"]
    },
    "computation": {
        "description": "Tools for calculations and processing",
        "tools": ["calculate_statistics", "run_code", "transform_data"]
    },
    "external": {
        "description": "Tools for external API integration",
        "tools": ["call_api", "fetch_url", "webhook"]
    }
}
```

### Hierarchical Organization

```python
TOOL_HIERARCHY = {
    "data": {
        "description": "Data access and manipulation",
        "subcategories": {
            "read": {
                "description": "Reading data",
                "tools": ["read_file", "query_database", "fetch_url"]
            },
            "write": {
                "description": "Writing data",
                "tools": ["write_file", "insert_database", "post_data"]
            },
            "transform": {
                "description": "Transforming data",
                "tools": ["filter_data", "aggregate_data", "join_data"]
            }
        }
    },
    "communication": {
        "description": "Communication tools",
        "subcategories": {
            "email": {
                "description": "Email operations",
                "tools": ["send_email", "read_email", "search_email"]
            },
            "messaging": {
                "description": "Instant messaging",
                "tools": ["send_message", "create_channel", "invite_user"]
            }
        }
    }
}


def get_tools_by_category(category_path: str) -> List[str]:
    """
    Get tools by hierarchical category path.
    Example: 'data.read' returns reading tools
    """
    parts = category_path.split('.')
    current = TOOL_HIERARCHY

    for part in parts:
        current = current.get(part, {})
        if "subcategories" in current:
            current = current["subcategories"]

    return current.get("tools", [])
```

### Tag-Based Organization

```python
class TagBasedOrganization:
    """Organize tools using flexible tagging."""

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.tag_index: Dict[str, List[str]] = {}

    def add_tool(self, tool: Dict[str, Any]) -> None:
        """Add tool and index its tags."""
        name = tool["name"]
        self.tools[name] = tool

        # Index tags
        for tag in tool.get("tags", []):
            if tag not in self.tag_index:
                self.tag_index[tag] = []
            self.tag_index[tag].append(name)

    def find_by_tags(
        self,
        tags: List[str],
        match_all: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Find tools matching tags.
        match_all=True requires ALL tags, False requires ANY tag.
        """
        if match_all:
            # Tools must have all specified tags
            tool_names = set(self.tools.keys())
            for tag in tags:
                tool_names &= set(self.tag_index.get(tag, []))
        else:
            # Tools can have any of the specified tags
            tool_names = set()
            for tag in tags:
                tool_names |= set(self.tag_index.get(tag, []))

        return [self.tools[name] for name in tool_names]

    def get_related_tools(self, tool_name: str) -> List[Dict[str, Any]]:
        """Get tools with similar tags."""
        tool = self.tools.get(tool_name)
        if not tool:
            return []

        tags = tool.get("tags", [])
        related = self.find_by_tags(tags, match_all=False)

        # Remove the original tool
        related = [t for t in related if t["name"] != tool_name]

        return related


# Usage
org = TagBasedOrganization()

org.add_tool({
    "name": "search_web",
    "tags": ["search", "web", "internet", "external"]
})

org.add_tool({
    "name": "search_documents",
    "tags": ["search", "documents", "internal"]
})

org.add_tool({
    "name": "fetch_url",
    "tags": ["web", "internet", "external", "http"]
})

# Find all search tools
search_tools = org.find_by_tags(["search"])

# Find tools that are both web and external
web_external = org.find_by_tags(["web", "external"], match_all=True)

# Find related tools
related = org.get_related_tools("search_web")
```

## Versioning Tool Schemas

Managing schema evolution over time.

### Version Information

```python
versioned_schema = {
    "name": "search_products",
    "version": "2.1.0",
    "description": "Search product catalog",

    # Version history
    "changelog": [
        {
            "version": "2.1.0",
            "date": "2026-01-15",
            "changes": [
                "Added 'include_out_of_stock' parameter",
                "Improved relevance ranking"
            ]
        },
        {
            "version": "2.0.0",
            "date": "2025-12-01",
            "changes": [
                "BREAKING: Renamed 'max' parameter to 'max_results'",
                "Added pagination support",
                "Removed deprecated 'category_id' parameter"
            ]
        },
        {
            "version": "1.0.0",
            "date": "2025-06-01",
            "changes": ["Initial release"]
        }
    ],

    # Deprecation notices
    "deprecated_parameters": {
        "category_id": {
            "deprecated_in": "2.0.0",
            "removed_in": "3.0.0",
            "alternative": "Use 'category_name' instead"
        }
    },

    "parameters": {...}
}
```

### Backward Compatibility

```python
def handle_schema_version(
    tool_call: Dict[str, Any],
    schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Handle versioning and maintain backward compatibility.
    """
    arguments = tool_call["arguments"].copy()

    # Handle deprecated parameters
    deprecated = schema.get("deprecated_parameters", {})
    for old_param, info in deprecated.items():
        if old_param in arguments:
            # Warn about deprecation
            print(f"Warning: {old_param} is deprecated. {info['alternative']}")

            # Auto-migrate if possible
            if "maps_to" in info:
                new_param = info["maps_to"]
                arguments[new_param] = arguments.pop(old_param)

    # Handle renamed parameters
    renamed = schema.get("renamed_parameters", {})
    for old_name, new_name in renamed.items():
        if old_name in arguments:
            arguments[new_name] = arguments.pop(old_name)

    return {
        "name": tool_call["name"],
        "arguments": arguments
    }
```

### Migration Helpers

```python
def migrate_tool_call(
    tool_call: Dict[str, Any],
    from_version: str,
    to_version: str
) -> Dict[str, Any]:
    """Migrate tool call from one version to another."""

    migrations = {
        ("1.0.0", "2.0.0"): migrate_v1_to_v2,
        ("2.0.0", "2.1.0"): migrate_v2_to_v2_1,
    }

    migration_func = migrations.get((from_version, to_version))
    if migration_func:
        return migration_func(tool_call)

    return tool_call


def migrate_v1_to_v2(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate from v1 to v2."""
    args = tool_call["arguments"].copy()

    # Rename 'max' to 'max_results'
    if "max" in args:
        args["max_results"] = args.pop("max")

    # Convert category_id to category_name
    if "category_id" in args:
        category_id = args.pop("category_id")
        args["category_name"] = lookup_category_name(category_id)

    return {"name": tool_call["name"], "arguments": args}
```

## Schema Validation

Ensuring schemas are correct and complete.

### Schema Validator

```python
from typing import List, Any, Dict

class SchemaValidator:
    """Validate tool schemas for correctness and completeness."""

    def validate(self, schema: Dict[str, Any]) -> List[str]:
        """
        Validate schema and return list of issues.
        Empty list means schema is valid.
        """
        issues = []

        # Check required fields
        issues.extend(self._check_required_fields(schema))

        # Check naming conventions
        issues.extend(self._check_naming(schema))

        # Check description quality
        issues.extend(self._check_descriptions(schema))

        # Check parameter specs
        issues.extend(self._check_parameters(schema))

        # Check examples
        issues.extend(self._check_examples(schema))

        return issues

    def _check_required_fields(self, schema: Dict[str, Any]) -> List[str]:
        """Check for required fields."""
        issues = []

        required = ["name", "description", "parameters"]
        for field in required:
            if field not in schema:
                issues.append(f"Missing required field: {field}")

        return issues

    def _check_naming(self, schema: Dict[str, Any]) -> List[str]:
        """Check naming conventions."""
        issues = []

        name = schema.get("name", "")

        # Check format
        if not re.match(r'^[a-z][a-z0-9_]*$', name):
            issues.append(
                f"Tool name '{name}' should be snake_case lowercase"
            )

        # Check length
        if len(name) < 3:
            issues.append(f"Tool name '{name}' is too short")
        if len(name) > 50:
            issues.append(f"Tool name '{name}' is too long")

        return issues

    def _check_descriptions(self, schema: Dict[str, Any]) -> List[str]:
        """Check description quality."""
        issues = []

        desc = schema.get("description", "")

        # Check length
        if len(desc) < 20:
            issues.append("Description is too brief (< 20 chars)")

        if len(desc) > 2000:
            issues.append("Description is very long (> 2000 chars)")

        # Check for placeholder text
        placeholders = ["TODO", "TBD", "FIXME", "XXX"]
        for placeholder in placeholders:
            if placeholder in desc:
                issues.append(f"Description contains placeholder: {placeholder}")

        return issues

    def _check_parameters(self, schema: Dict[str, Any]) -> List[str]:
        """Check parameter specifications."""
        issues = []

        params = schema.get("parameters", {})
        properties = params.get("properties", {})

        for param_name, param_spec in properties.items():
            # Check parameter has description
            if "description" not in param_spec:
                issues.append(
                    f"Parameter '{param_name}' missing description"
                )

            # Check parameter has type
            if "type" not in param_spec:
                issues.append(
                    f"Parameter '{param_name}' missing type"
                )

            # Check enum values if enum type
            if "enum" in param_spec and len(param_spec["enum"]) == 0:
                issues.append(
                    f"Parameter '{param_name}' has empty enum"
                )

        return issues

    def _check_examples(self, schema: Dict[str, Any]) -> List[str]:
        """Check examples if present."""
        issues = []

        examples = schema.get("examples", [])

        if len(examples) == 0:
            issues.append("Consider adding usage examples")

        return issues


# Usage
validator = SchemaValidator()

schema = {
    "name": "search_products",
    "description": "Search for products",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"}  # Missing description!
        }
    }
}

issues = validator.validate(schema)
if issues:
    print("Schema validation issues:")
    for issue in issues:
        print(f"  - {issue}")
```

## Context-Aware Tool Descriptions

Adapting tool descriptions based on context.

### User-Specific Descriptions

```python
def get_contextual_schema(
    tool_name: str,
    user_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Get tool schema adapted to user context.
    """
    base_schema = registry.get_tool(tool_name).copy()

    # Adapt based on user role
    user_role = user_context.get("role")
    if user_role == "admin":
        # Show additional admin-only parameters
        base_schema["parameters"]["properties"]["admin_mode"] = {
            "type": "boolean",
            "description": "Enable admin features"
        }
    elif user_role == "viewer":
        # Remove write parameters
        read_only_params = {
            k: v for k, v in base_schema["parameters"]["properties"].items()
            if not k.startswith("update_") and not k.startswith("delete_")
        }
        base_schema["parameters"]["properties"] = read_only_params

    # Adapt based on user's integrations
    available_integrations = user_context.get("integrations", [])
    if "email" not in available_integrations:
        # Add note about missing integration
        base_schema["description"] += "\n\nNote: Email integration required but not enabled."

    return base_schema
```

### Task-Specific Descriptions

```python
def get_task_specific_schema(
    tool_name: str,
    current_task: str
) -> Dict[str, Any]:
    """
    Adapt tool schema based on current task.
    """
    schema = registry.get_tool(tool_name).copy()

    # Highlight relevant parameters for current task
    if "search" in current_task.lower():
        # Emphasize search-related parameters
        for param in schema["parameters"]["properties"].values():
            if "search" in param.get("description", "").lower():
                param["description"] = "⭐ " + param["description"]

    # Add task-specific examples
    schema["task_examples"] = generate_task_examples(tool_name, current_task)

    return schema
```

## Multi-Language Support

Supporting tool schemas in multiple languages.

### Localized Schemas

```python
def get_localized_schema(
    tool_name: str,
    language: str = "en"
) -> Dict[str, Any]:
    """Get tool schema in specified language."""

    base_schema = registry.get_tool(tool_name)

    # Load translations
    translations = load_translations(tool_name, language)

    if not translations:
        return base_schema  # Fall back to English

    # Create localized copy
    localized = base_schema.copy()
    localized["description"] = translations.get(
        "description",
        base_schema["description"]
    )

    # Localize parameter descriptions
    for param_name, param_spec in localized["parameters"]["properties"].items():
        translation_key = f"param_{param_name}_description"
        if translation_key in translations:
            param_spec["description"] = translations[translation_key]

    return localized


# Translation storage
TRANSLATIONS = {
    "search_web": {
        "es": {
            "description": "Buscar información en internet",
            "param_query_description": "Consulta de búsqueda"
        },
        "fr": {
            "description": "Rechercher des informations sur internet",
            "param_query_description": "Requête de recherche"
        }
    }
}
```

## Performance Considerations

Optimizing schema handling for performance.

### Schema Caching

```python
from functools import lru_cache
import time

class CachedToolRegistry(ToolRegistry):
    """Tool registry with caching for performance."""

    @lru_cache(maxsize=128)
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Get tool with caching."""
        return super().get_tool(name)

    @lru_cache(maxsize=32)
    def list_tools(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List tools with caching."""
        return super().list_tools(category)

    def register_tool(self, schema: Dict[str, Any]) -> None:
        """Register tool and clear relevant caches."""
        super().register_tool(schema)
        # Clear caches since tools changed
        self.get_tool.cache_clear()
        self.list_tools.cache_clear()
```

### Lazy Loading

```python
class LazyToolRegistry:
    """Load tool schemas lazily for better startup performance."""

    def __init__(self):
        self.tool_paths: Dict[str, str] = {}  # name -> file path
        self.loaded_tools: Dict[str, Dict[str, Any]] = {}

    def register_tool_path(self, name: str, path: str) -> None:
        """Register where to find a tool schema."""
        self.tool_paths[name] = path

    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Get tool, loading if necessary."""
        # Check if already loaded
        if name in self.loaded_tools:
            return self.loaded_tools[name]

        # Load from disk
        if name in self.tool_paths:
            schema = self._load_schema(self.tool_paths[name])
            self.loaded_tools[name] = schema
            return schema

        return None

    def _load_schema(self, path: str) -> Dict[str, Any]:
        """Load schema from file."""
        import json
        with open(path, 'r') as f:
            return json.load(f)
```

## Testing Tool Schemas

Ensuring schemas work correctly.

### Schema Tests

```python
import pytest

def test_schema_completeness():
    """Test that schema has all required fields."""
    schema = registry.get_tool("search_web")

    assert "name" in schema
    assert "description" in schema
    assert "parameters" in schema
    assert len(schema["description"]) > 20


def test_schema_validation():
    """Test schema validates correctly."""
    validator = SchemaValidator()

    schema = {
        "name": "test_tool",
        "description": "A test tool for validation",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "First parameter"
                }
            },
            "required": ["param1"]
        }
    }

    issues = validator.validate(schema)
    assert len(issues) == 0


def test_parameter_types():
    """Test that parameter types are correct."""
    schema = registry.get_tool("search_web")
    params = schema["parameters"]["properties"]

    assert params["query"]["type"] == "string"
    assert params["max_results"]["type"] == "integer"
    assert "minimum" in params["max_results"]
```

### Usage Tests

```python
def test_agent_can_call_tool():
    """Test that agent can successfully use tool schema."""

    schema = registry.get_tool("search_web")

    # Simulate agent generating a call
    user_input = "Search for Python tutorials"
    function_call = agent_generate_call(user_input, [schema])

    # Verify call is valid
    assert function_call["name"] == "search_web"
    assert "query" in function_call["arguments"]
    assert "python" in function_call["arguments"]["query"].lower()

    # Verify can execute
    result = execute_tool(
        function_call["name"],
        function_call["arguments"]
    )
    assert result is not None
```

## Best Practices

Guidelines for creating effective tool schemas.

### 1. Be Specific and Clear

```python
# Bad: Vague
"description": "Does a search"

# Good: Specific
"description": "Searches academic papers in the IEEE Xplore database by keywords, author, or publication year. Returns paper titles, abstracts, and citation counts."
```

### 2. Provide Context

```python
# Bad: No context
"timeout": {"type": "integer"}

# Good: With context
"timeout": {
    "type": "integer",
    "description": "Request timeout in seconds. Recommended values: 30 for normal queries, 60 for complex queries, 300 for bulk operations.",
    "minimum": 1,
    "maximum": 600,
    "default": 30
}
```

### 3. Include Examples

```python
"parameters": {
    "properties": {
        "date_range": {
            "type": "object",
            "description": "Date range for filtering results",
            "properties": {
                "start": {"type": "string"},
                "end": {"type": "string"}
            },
            "example": {
                "start": "2026-01-01",
                "end": "2026-01-31"
            }
        }
    }
}
```

### 4. Set Appropriate Defaults

```python
"parameters": {
    "properties": {
        "page_size": {
            "type": "integer",
            "description": "Number of results per page",
            "default": 20,  # Reasonable default
            "minimum": 1,
            "maximum": 100
        }
    }
}
```

### 5. Document Constraints

```python
"description": """
Send email to recipients.

CONSTRAINTS:
- Maximum 50 recipients per email
- Attachments limited to 25MB total
- Subject line max 200 characters
- Requires 'email.send' permission

RATE LIMITS:
- 100 emails per hour per user
- 1000 emails per day per organization
"""
```

## Common Pitfalls

Mistakes to avoid when designing schemas.

### 1. Overly Complex Schemas

```python
# Bad: Too complex for agent to understand
"parameters": {
    "properties": {
        "config": {
            "type": "object",
            "properties": {
                "advanced": {
                    "type": "object",
                    "properties": {
                        "optimization": {
                            "type": "object",
                            "properties": {
                                # 5 levels deep...
                            }
                        }
                    }
                }
            }
        }
    }
}

# Good: Flatten or split into multiple tools
"parameters": {
    "properties": {
        "optimization_level": {
            "type": "string",
            "enum": ["none", "basic", "aggressive"],
            "description": "Optimization level to apply"
        }
    }
}
```

### 2. Missing Critical Information

```python
# Bad: Doesn't explain side effects
"description": "Delete user account"

# Good: Explains consequences
"description": """
Permanently delete a user account and all associated data.

WARNING: This action cannot be undone!

Side effects:
- Deletes all user data (profile, settings, history)
- Removes user from all teams and projects
- Cancels all active subscriptions
- Sends account deletion confirmation email
"""
```

### 3. Inconsistent Naming

```python
# Bad: Inconsistent names across tools
"search_web"
"findDocuments"
"Query_Database"

# Good: Consistent snake_case
"search_web"
"search_documents"
"query_database"
```

### 4. No Validation Constraints

```python
# Bad: No constraints
"email": {"type": "string"}

# Good: With validation
"email": {
    "type": "string",
    "pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
    "maxLength": 254,
    "description": "Valid email address"
}
```

## Summary

Tool schemas are the bridge between tools and agents. Key takeaways:

**Core Principles**:

- Schemas teach agents what tools do and how to use them
- Clear descriptions enable better tool selection
- Comprehensive parameter specs prevent errors
- Good schemas make tools feel native to the model

**Design Guidelines**:

- Use descriptive names and clear descriptions
- Provide context, examples, and constraints
- Document capabilities and limitations
- Include validation rules and defaults

**Advanced Patterns**:

- Dynamic tool discovery for runtime flexibility
- Context-aware schemas for personalization
- Versioning for schema evolution
- Hierarchical organization for manageability

**Best Practices**:

- Test schemas thoroughly
- Keep schemas up to date
- Balance detail with simplicity
- Provide clear migration paths

Effective tool schemas are crucial for agent success. Invest time in creating clear, comprehensive schemas, and your agents will use tools more reliably and effectively.

## Next Steps

Now that you understand tool schemas, explore:

- [Function Calling](function-calling.md) - The mechanics of how agents invoke tools
- [Tool Selection](tool-selection.md) - How agents choose which tool to use
- [Tool Chaining](tool-chaining.md) - Combining multiple tools for complex tasks
- [API Integration](api-integration.md) - Practical patterns for integrating external APIs
