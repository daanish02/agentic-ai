# Tool Definition Patterns

## Table of Contents

- [Introduction](#introduction)
- [Why Tool Definition Patterns Matter](#why-tool-definition-patterns-matter)
- [The @tool Decorator Pattern](#the-tool-decorator-pattern)
- [StructuredTool Pattern](#structuredtool-pattern)
- [BaseTool Class Pattern](#basetool-class-pattern)
- [Pattern Comparison](#pattern-comparison)
- [Tool Schemas and Validation](#tool-schemas-and-validation)
- [Built-in Tools](#built-in-tools)
- [Custom Toolkits](#custom-toolkits)
- [Tool Binding to LLMs](#tool-binding-to-llms)
- [Tool Metadata and Introspection](#tool-metadata-and-introspection)
- [Tool Messages and Execution Flow](#tool-messages-and-execution-flow)
- [Dependency Injection Patterns](#dependency-injection-patterns)
- [Error Handling in Tools](#error-handling-in-tools)
- [Testing Tool Definitions](#testing-tool-definitions)
- [Best Practices](#best-practices)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Tool definition is the foundation of agentic AI systems. How you define tools determines how effectively language models can understand, select, and execute them. Poor tool definitions lead to confused agents, failed executions, and frustrated users. Good tool definitions enable agents to accomplish complex tasks reliably.

> "A well-defined tool is half executed."

This guide explores multiple patterns for defining tools in LangChain-based systems, from simple function decorators to sophisticated class-based definitions. We'll cover:

- **Three main patterns**: @tool decorator, StructuredTool, and BaseTool class
- **Schema design**: Using Pydantic for type safety and validation
- **Tool composition**: Building toolkits and tool collections
- **LLM integration**: Binding tools to language models
- **Advanced patterns**: Dependency injection, async tools, and error handling

### The Tool Definition Spectrum

```
┌─────────────────────────────────────────────────────────┐
│                                                          │
│  Simple          Flexible           Powerful            │
│    ↓                ↓                   ↓               │
│  @tool      StructuredTool        BaseTool              │
│  Quick          Explicit          Full Control          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Choose based on your needs**:

- **@tool**: Quick prototypes, simple functions
- **StructuredTool**: Explicit schemas, controlled construction
- **BaseTool**: Maximum control, complex logic, stateful tools

## Why Tool Definition Patterns Matter

### The Impact of Good Tool Definitions

**Poor tool definition**:

```python
def hash_function(x):
    """Hash something."""
    return hashlib.sha256(x).hexdigest()
```

Problems:

- Unclear what type x should be
- No description of what "something" means
- Will crash if x isn't bytes
- Agent can't understand when to use it

**Good tool definition**:

```python
@tool
def compute_sha256_hash(input_text: str) -> str:
    """Compute the SHA-256 cryptographic hash of a text string.

    Use this when you need to:
    - Generate a unique fingerprint of text content
    - Verify data integrity
    - Create secure identifiers

    Args:
        input_text: The text string to hash

    Returns:
        64-character hexadecimal hash string
    """
    return hashlib.sha256(input_text.encode()).hexdigest()
```

Benefits:

- Clear type hints
- Comprehensive documentation
- Explicit use cases
- Handles encoding automatically
- Agent knows exactly when and how to use it

### Tool Definition Quality Impact

```
Definition Quality → Agent Success Rate

Poor definition:    ████░░░░░░ 40% success
Okay definition:    ███████░░░ 70% success
Great definition:   █████████░ 90% success
```

## The @tool Decorator Pattern

The simplest and most common pattern for tool definition.

### Basic Usage

```python
from langchain_core.tools import tool
import hashlib

@tool
def compute_sha256_hash(input_text: str) -> str:
    """Compute the SHA-256 cryptographic hash of a text string."""
    return hashlib.sha256(input_text.encode()).hexdigest()
```

**What the decorator does**:

1. Extracts function signature for schema
2. Uses docstring as tool description
3. Creates a LangChain Tool object
4. Preserves original function behavior

### Realistic Example: Weather API Tool

```python
import requests
from langchain_core.tools import tool
from typing import Optional

@tool
def get_current_weather(city: str, units: str = "metric") -> dict:
    """Fetch current weather data for a specified city.

    This tool queries a weather API to retrieve real-time weather information
    including temperature, conditions, humidity, and wind speed.

    Use this when users ask about:
    - Current temperature in a location
    - Weather conditions ("Is it raining in...?")
    - Should I bring an umbrella
    - What to wear based on weather

    Args:
        city: Name of the city (e.g., "London", "Tokyo", "New York")
        units: Temperature units - "metric" (Celsius) or "imperial" (Fahrenheit)

    Returns:
        Dictionary containing:
        - temperature: Current temperature
        - condition: Weather condition (sunny, cloudy, rainy, etc.)
        - humidity: Humidity percentage
        - wind_speed: Wind speed in km/h or mph
        - timestamp: Time of observation
    """
    # Real API call (example with weatherstack)
    api_key = "your_api_key"
    url = f"https://api.weatherstack.com/current"
    params = {
        "access_key": api_key,
        "query": city,
        "units": "m" if units == "metric" else "f"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            return {"error": data["error"]["info"]}

        current = data.get("current", {})
        return {
            "temperature": current.get("temperature"),
            "condition": current.get("weather_descriptions", ["Unknown"])[0],
            "humidity": current.get("humidity"),
            "wind_speed": current.get("wind_speed"),
            "timestamp": current.get("observation_time")
        }
    except requests.RequestException as e:
        return {"error": f"Failed to fetch weather data: {str(e)}"}
```

### Invoking @tool Decorated Functions

```python
# Direct invocation (as a regular function)
result = get_current_weather("Tokyo")
print(result)

# Tool invocation (as a LangChain tool)
result = get_current_weather.invoke({"city": "Tokyo", "units": "metric"})
print(result)

# Async invocation
result = await get_current_weather.ainvoke({"city": "Tokyo"})
```

### Tool Metadata

The @tool decorator automatically creates metadata:

```python
# Tool has these attributes
print(get_current_weather.name)
# Output: "get_current_weather"

print(get_current_weather.description)
# Output: "Fetch current weather data for a specified city..."

print(get_current_weather.args)
# Output: {'city': {'type': 'string', 'description': '...'}, ...}

# Full schema
schema = get_current_weather.args_schema.model_json_schema()
print(schema)
```

### Advantages of @tool

✅ **Simplicity**: Minimal boilerplate
✅ **Readability**: Looks like a normal function
✅ **Quick iteration**: Easy to prototype
✅ **Type inference**: Automatic schema from type hints

### Limitations of @tool

❌ **Less control**: Can't customize initialization
❌ **Stateless**: Harder to maintain state
❌ **Limited validation**: Basic type checking only

## StructuredTool Pattern

For more control over tool definition without full class implementation.

### Basic Usage

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
import hashlib

class HashInput(BaseModel):
    """Input schema for hashing tool."""
    text: str = Field(..., description="Text to hash")
    algorithm: str = Field(
        default="sha256",
        description="Hash algorithm: sha256, md5, or sha1"
    )

def compute_hash(text: str, algorithm: str = "sha256") -> str:
    """Compute cryptographic hash of text."""
    hash_func = getattr(hashlib, algorithm)
    return hash_func(text.encode()).hexdigest()

# Create tool with explicit schema
hash_tool = StructuredTool.from_function(
    func=compute_hash,
    name="compute_hash",
    description="Compute cryptographic hash of a text string using specified algorithm",
    args_schema=HashInput,
)
```

### Realistic Example: Stock Price Tool

```python
import requests
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Optional, Literal

class StockPriceInput(BaseModel):
    """Input schema for stock price lookup."""
    symbol: str = Field(
        ...,
        description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')",
        pattern="^[A-Z]{1,5}$"
    )
    include_details: bool = Field(
        default=False,
        description="Include additional details like volume, high/low, etc."
    )

def fetch_stock_price(symbol: str, include_details: bool = False) -> dict:
    """Fetch current stock price and optionally detailed information.

    Args:
        symbol: Stock ticker symbol
        include_details: Whether to include volume, high/low, etc.

    Returns:
        Dictionary with stock information
    """
    api_key = "your_alpha_vantage_key"
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": api_key
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "Error Message" in data:
            return {"error": "Invalid stock symbol"}

        quote = data.get("Global Quote", {})

        result = {
            "symbol": quote.get("01. symbol"),
            "price": float(quote.get("05. price", 0)),
            "change": float(quote.get("09. change", 0)),
            "change_percent": quote.get("10. change percent"),
        }

        if include_details:
            result.update({
                "volume": int(quote.get("06. volume", 0)),
                "latest_trading_day": quote.get("07. latest trading day"),
                "previous_close": float(quote.get("08. previous close", 0)),
                "high": float(quote.get("03. high", 0)),
                "low": float(quote.get("04. low", 0)),
            })

        return result

    except requests.RequestException as e:
        return {"error": f"Failed to fetch stock data: {str(e)}"}
    except (ValueError, KeyError) as e:
        return {"error": f"Failed to parse stock data: {str(e)}"}

# Create structured tool
stock_price_tool = StructuredTool.from_function(
    func=fetch_stock_price,
    name="get_stock_price",
    description=(
        "Get real-time stock price and market data for a given ticker symbol. "
        "Use this when users ask about stock prices, market performance, "
        "or want to check how a company's stock is doing."
    ),
    args_schema=StockPriceInput,
)
```

### Using StructuredTool

```python
# Invoke with arguments
result = stock_price_tool.invoke({
    "symbol": "AAPL",
    "include_details": True
})
print(result)

# Schema introspection
print(stock_price_tool.args_schema.model_json_schema())
# Shows Pydantic schema with validation rules
```

### Advantages of StructuredTool

✅ **Explicit schemas**: Full control over input validation
✅ **Rich validation**: Pydantic constraints (regex, ranges, etc.)
✅ **Separation of concerns**: Schema separate from implementation
✅ **Documentation**: Schema serves as API documentation

### When to Use StructuredTool

Use StructuredTool when:

- You need complex input validation
- Schema needs to be defined separately from function
- Multiple functions share the same schema
- You want explicit control over field descriptions
- Input validation is critical (financial data, sensitive operations)

## BaseTool Class Pattern

Maximum control through class-based tool definition.

### Basic Usage

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional
import hashlib

class HashInput(BaseModel):
    """Input schema for hash tool."""
    text: str = Field(..., description="Text to hash")
    algorithm: str = Field(default="sha256", description="Hash algorithm")

class CryptoHashTool(BaseTool):
    """Tool for computing cryptographic hashes."""

    name: str = "compute_hash"
    description: str = (
        "Compute cryptographic hash of text. "
        "Supports sha256, sha1, and md5 algorithms."
    )
    args_schema: Type[BaseModel] = HashInput

    # Optional: Tool can have state
    call_count: int = 0

    def _run(self, text: str, algorithm: str = "sha256") -> str:
        """Synchronous implementation."""
        self.call_count += 1
        hash_func = getattr(hashlib, algorithm)
        return hash_func(text.encode()).hexdigest()

    async def _arun(self, text: str, algorithm: str = "sha256") -> str:
        """Async implementation (optional)."""
        # If not implemented, falls back to _run
        return self._run(text, algorithm)

# Create instance
hash_tool = CryptoHashTool()
```

### Realistic Example: Database Query Tool

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, List, Dict, Any
import sqlite3
from pathlib import Path

class QueryInput(BaseModel):
    """Input schema for database queries."""
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    category: str = Field(
        default=None,
        description="Optional: Filter by expense category"
    )

class ExpenseQueryTool(BaseTool):
    """Tool for querying expense database.

    This tool provides read-only access to expense records,
    allowing filtering by date range and category.
    """

    name: str = "query_expenses"
    description: str = (
        "Query expense records from the database. "
        "Can filter by date range and optionally by category. "
        "Returns detailed expense information including amounts, dates, and notes."
    )
    args_schema: Type[BaseModel] = QueryInput

    # Tool state - database connection
    db_path: Path = Field(default=Path("expenses.db"))
    connection_pool_size: int = 5

    def __init__(self, **kwargs):
        """Initialize tool with database setup."""
        super().__init__(**kwargs)
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        """Ensure database exists (idempotent)."""
        if not self.db_path.exists():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE expenses(
                        id INTEGER PRIMARY KEY,
                        date TEXT NOT NULL,
                        amount REAL NOT NULL,
                        category TEXT NOT NULL,
                        note TEXT
                    )
                """)

    def _run(
        self,
        start_date: str,
        end_date: str,
        category: str = None
    ) -> List[Dict[str, Any]]:
        """Execute database query synchronously."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                query = """
                    SELECT id, date, amount, category, note
                    FROM expenses
                    WHERE date BETWEEN ? AND ?
                """
                params = [start_date, end_date]

                if category:
                    query += " AND category = ?"
                    params.append(category)

                query += " ORDER BY date DESC, id DESC"

                cursor = conn.execute(query, params)
                results = [dict(row) for row in cursor.fetchall()]

                return {
                    "status": "success",
                    "count": len(results),
                    "expenses": results
                }

        except sqlite3.Error as e:
            return {
                "status": "error",
                "error": f"Database error: {str(e)}"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Unexpected error: {str(e)}"
            }

    async def _arun(self, start_date: str, end_date: str, category: str = None):
        """Async implementation - delegates to sync for SQLite."""
        # For async databases (PostgreSQL), implement true async here
        return self._run(start_date, end_date, category)

# Create tool instance
expense_query_tool = ExpenseQueryTool(
    db_path=Path("./data/expenses.db")
)
```

### Stateful Tools

BaseTool allows maintaining state across invocations:

```python
class RateLimitedSearchTool(BaseTool):
    """Search tool with rate limiting."""

    name: str = "web_search"
    description: str = "Search the web with rate limiting"

    # State
    calls_this_minute: List[float] = Field(default_factory=list)
    max_calls_per_minute: int = 10

    def _run(self, query: str) -> str:
        """Execute search with rate limiting."""
        import time

        # Clean old entries
        current_time = time.time()
        self.calls_this_minute = [
            t for t in self.calls_this_minute
            if current_time - t < 60
        ]

        # Check rate limit
        if len(self.calls_this_minute) >= self.max_calls_per_minute:
            return {"error": "Rate limit exceeded. Try again in a minute."}

        # Record call
        self.calls_this_minute.append(current_time)

        # Execute search
        return perform_search(query)

search_tool = RateLimitedSearchTool()
```

### Advantages of BaseTool

✅ **Maximum control**: Full control over initialization and execution
✅ **Stateful**: Can maintain state across calls
✅ **Lifecycle management**: Custom **init**, cleanup methods
✅ **Async support**: Explicit async/\_async separation
✅ **Complex logic**: Suitable for sophisticated tools

### When to Use BaseTool

Use BaseTool when:

- Tool needs to maintain state
- Complex initialization required
- Need lifecycle management (setup/teardown)
- Different sync/async implementations
- Building a toolkit/tool family

## Pattern Comparison

### Quick Comparison Table

```
┌──────────────┬─────────────┬──────────────────┬─────────────┐
│ Pattern      │ Complexity  │ Control          │ Use Case    │
├──────────────┼─────────────┼──────────────────┼─────────────┤
│ @tool        │ Low         │ Limited          │ Prototyping │
│ StructuredT  │ Medium      │ Schema control   │ Validation  │
│ BaseTool     │ High        │ Complete         │ Production  │
└──────────────┴─────────────┴──────────────────┴─────────────┘
```

### choosing a Pattern

**Use @tool when**:

- Prototyping quickly
- Simple stateless operations
- Type hints sufficient for schema
- Minimal validation needed

**Use StructuredTool when**:

- Need explicit input validation
- Schema is complex
- Want separation of schema and logic
- Need rich Pydantic features

**Use BaseTool when**:

- Tool maintains state
- Complex initialization
- Need lifecycle management
- Building production systems
- Different sync/async implementations

### Migration Path

```
1. Start with @tool for prototyping
        ↓
2. Upgrade to StructuredTool for validation
        ↓
3. Migrate to BaseTool for production needs
```

## Tool Schemas and Validation

Pydantic schemas enable rich validation and documentation.

### Basic Schema

```python
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(10, ge=1, le=100, description="Max results (1-100)")
```

### Advanced Validation

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal
from datetime import datetime

class ExpenseInput(BaseModel):
    """Validated expense input schema."""

    date: str = Field(
        ...,
        description="Date in YYYY-MM-DD format",
        pattern=r"^\d{4}-\d{2}-\d{2}$"
    )
    amount: float = Field(
        ...,
        gt=0,
        description="Amount in dollars (must be positive)"
    )
    category: Literal["Food", "Transport", "Housing", "Healthcare", "Entertainment"] = Field(
        ...,
        description="Expense category"
    )
    payment_method: Literal["cash", "card", "transfer"] = Field(
        default="card",
        description="Payment method"
    )

    @field_validator("date")
    @classmethod
    def validate_date_not_future(cls, v: str) -> str:
        """Ensure date is not in the future."""
        expense_date = datetime.strptime(v, "%Y-%m-%d").date()
        if expense_date > datetime.now().date():
            raise ValueError("Expense date cannot be in the future")
        return v

    @field_validator("amount")
    @classmethod
    def validate_amount_reasonable(cls, v: float) -> float:
        """Ensure amount is reasonable (< $10,000)."""
        if v > 10000:
            raise ValueError("Amount exceeds reasonable limit ($10,000)")
        return v
```

### Schema Introspection

```python
# Get JSON schema
schema = ExpenseInput.model_json_schema()
print(schema)

# Output:
{
    "type": "object",
    "properties": {
        "date": {
            "type": "string",
            "description": "Date in YYYY-MM-DD format",
            "pattern": "^\\d{4}-\\d{2}-\\d{2}$"
        },
        "amount": {
            "type": "number",
            "description": "Amount in dollars (must be positive)",
            "exclusiveMinimum": 0
        },
        # ...
    },
    "required": ["date", "amount", "category"]
}
```

## Built-in Tools

LangChain provides many built-in tools for common operations.

### Web Search Tools

```python
from langchain_community.tools import DuckDuckGoSearchRun

# Create search tool
search_tool = DuckDuckGoSearchRun()

# Use it
results = search_tool.invoke("latest AI news")
print(results)

# Tool metadata
print(search_tool.name)  # "duckduckgo_search"
print(search_tool.description)  # "A wrapper around DuckDuckGo Search..."
print(search_tool.args)  # {'query': {'type': 'string'}}
```

### Shell Tools

```python
from langchain_community.tools import ShellTool

# Create shell tool (use with caution!)
shell_tool = ShellTool()

# Execute command
result = shell_tool.invoke("ls -la")
print(result)

# Note: ShellTool is powerful but risky
# Only use in controlled environments
# Consider security implications
```

### Combining Built-in and Custom Tools

```python
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

# Built-in tool
search_tool = DuckDuckGoSearchRun()

# Custom tool
@tool
def get_weather(city: str) -> dict:
    """Get weather for a city."""
    return fetch_weather_data(city)

# Use together
tools = [search_tool, get_weather]

# Bind to LLM
llm_with_tools = llm.bind_tools(tools)
```

## Custom Toolkits

Group related tools into toolkits for organization.

### Simple Toolkit

```python
from langchain_core.tools import tool

@tool
def compute_sha256(text: str) -> str:
    """Compute SHA-256 hash."""
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()

@tool
def compute_md5(text: str) -> str:
    """Compute MD5 hash."""
    import hashlib
    return hashlib.md5(text.encode()).hexdigest()

class CryptoToolkit:
    """Toolkit for cryptographic operations."""

    def get_tools(self):
        """Return all tools in the toolkit."""
        return [compute_sha256, compute_md5]

# Use toolkit
toolkit = CryptoToolkit()
tools = toolkit.get_tools()
print([t.name for t in tools])
# Output: ['compute_sha256', 'compute_md5']
```

### Advanced Toolkit with Configuration

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Type
import sqlite3

class DatabaseToolkit:
    """Toolkit for database operations."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_tools(self) -> List[BaseTool]:
        """Create and return all database tools."""

        class QueryInput(BaseModel):
            start_date: str = Field(..., description="Start date")
            end_date: str = Field(..., description="End date")

        class QueryTool(BaseTool):
            name: str = "query_database"
            description: str = "Query expense records"
            args_schema: Type[BaseModel] = QueryInput
            db_path: str = self.db_path

            def _run(self, start_date: str, end_date: str):
                with sqlite3.connect(self.db_path) as conn:
                    return conn.execute(
                        "SELECT * FROM expenses WHERE date BETWEEN ? AND ?",
                        (start_date, end_date)
                    ).fetchall()

        class SummaryTool(BaseTool):
            name: str = "summarize_expenses"
            description: str = "Get expense summary"
            args_schema: Type[BaseModel] = QueryInput
            db_path: str = self.db_path

            def _run(self, start_date: str, end_date: str):
                with sqlite3.connect(self.db_path) as conn:
                    return conn.execute(
                        """SELECT category, SUM(amount) as total
                           FROM expenses
                           WHERE date BETWEEN ? AND ?
                           GROUP BY category""",
                        (start_date, end_date)
                    ).fetchall()

        return [QueryTool(), SummaryTool()]

# Use toolkit
toolkit = DatabaseToolkit("expenses.db")
tools = toolkit.get_tools()
```

## Tool Binding to LLMs

Tools become useful when bound to language models.

### Basic Binding

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def get_stock_price(symbol: str) -> dict:
    """Get current stock price."""
    return fetch_stock_data(symbol)

# Create model
model = ChatOpenAI(model="gpt-4")

# Bind tool to model
model_with_tools = model.bind_tools([get_stock_price])

# Use it
response = model_with_tools.invoke("What's the price of Apple stock?")
print(response.content)

# Check if tool was called
if hasattr(response, 'tool_calls') and response.tool_calls:
    print("Tool was requested!")
    print(response.tool_calls)
```

### Understanding Tool Calls vs Execution

**Important distinction**:

```python
response = model_with_tools.invoke("Get Apple stock price")

# Model returns a TOOL CALL REQUEST, not the result
# response.tool_calls = [
#     {
#         "name": "get_stock_price",
#         "args": {"symbol": "AAPL"},
#         "id": "call_1234"
#     }
# ]

# YOU must execute the tool
if response.tool_calls:
    for tool_call in response.tool_calls:
        # Execute tool
        result = get_stock_price.invoke(tool_call["args"])
        # Now you have the actual result
```

### Complete Tool Execution Flow

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
import json

@tool
def calculate_investment_return(
    initial: float,
    final: float,
) -> dict:
    """Calculate investment return percentage."""
    return_pct = ((final - initial) / initial) * 100
    return {
        "initial": initial,
        "final": final,
        "return_percentage": round(return_pct, 2),
        "profit": round(final - initial, 2)
    }

model = ChatOpenAI(model="gpt-4")
model_with_tools = model.bind_tools([calculate_investment_return])

# 1. Initial query
query = "If I invested $1000 and now have $1500, what's my return?"
response = model_with_tools.invoke(query)

# 2. Check for tool calls
if hasattr(response, 'tool_calls') and response.tool_calls:
    tool_messages = []

    # 3. Execute each tool call
    for tool_call in response.tool_calls:
        result = calculate_investment_return.invoke(tool_call["args"])

        # 4. Create tool message
        tool_messages.append(
            ToolMessage(
                tool_call_id=tool_call["id"],
                content=json.dumps(result)
            )
        )

    # 5. Get final response with tool results
    final_response = model_with_tools.invoke([query, response, *tool_messages])
    print(final_response.content)
else:
    # No tools needed, direct answer
    print(response.content)
```

## Tool Metadata and Introspection

Understanding tool metadata helps with debugging and tool selection.

### Accessing Metadata

```python
@tool
def example_tool(param1: str, param2: int = 10) -> str:
    """An example tool for demonstration."""
    return f"Processed: {param1} with {param2}"

# Name
print(example_tool.name)
# Output: "example_tool"

# Description
print(example_tool.description)
# Output: "An example tool for demonstration."

# Arguments
print(example_tool.args)
# Output: {'param1': {'type': 'string'}, 'param2': {'type': 'integer', 'default': 10}}

# Full schema
print(example_tool.args_schema.model_json_schema())
```

### Tool Selection by Metadata

```python
def find_tools_by_keyword(tools: List[BaseTool], keyword: str) -> List[BaseTool]:
    """Find tools whose description contains keyword."""
    return [
        tool for tool in tools
        if keyword.lower() in tool.description.lower()
    ]

def find_tools_by_capability(tools: List[BaseTool], capability: str) -> List[BaseTool]:
    """Find tools that mention a capability in their docstring."""
    matching = []
    for tool in tools:
        # Check description and args
        if capability.lower() in str(tool.description).lower():
            matching.append(tool)
        elif hasattr(tool, '__doc__') and tool.__doc__:
            if capability.lower() in tool.__doc__.lower():
                matching.append(tool)
    return matching

# Usage
all_tools = [search_tool, weather_tool, stock_tool]
finance_tools = find_tools_by_keyword(all_tools, "stock")
```

## Tool Messages and Execution Flow

Understanding the message flow is crucial for complex tool usage.

### Message Flow

```
User: "What's the weather in Tokyo?"
    ↓
LLM: [generates tool call request]
    {
        "tool_calls": [{
            "id": "call_123",
            "name": "get_weather",
            "args": {"city": "Tokyo"}
        }]
    }
    ↓
System: [executes tool]
    result = get_weather("Tokyo")
    ↓
System: [creates ToolMessage]
    ToolMessage(
        tool_call_id="call_123",
        content='{"temp": 18, "condition": "cloudy"}'
    )
    ↓
LLM: [generates final response using tool result]
    "The weather in Tokyo is currently cloudy with
     a temperature of 18°C."
```

### Tool Message Format

```python
from langchain_core.messages import ToolMessage

# Create tool message
tool_msg = ToolMessage(
    tool_call_id="call_123",  # Must match tool call ID
    content=json.dumps({      # Usually JSON-encoded
        "temperature": 18,
        "condition": "cloudy"
    })
)

# Or with error
error_msg = ToolMessage(
    tool_call_id="call_123",
    content=json.dumps({"error": "API rate limit exceeded"})
)
```

## Dependency Injection Patterns

Advanced pattern for tools that need runtime dependencies.

### InjectedToolArg

```python
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated

# Tool with injected dependency
@tool
def query_user_data(
    query: str,
    user_id: Annotated[str, InjectedToolArg]  # Injected, not from LLM
) -> dict:
    """Query user-specific data.

    The user_id is automatically injected at runtime
    and is NOT part of the schema exposed to the LLM.
    """
    # user_id is available but LLM didn't provide it
    return fetch_data_for_user(user_id, query)

# When calling, provide injected args
result = query_user_data.invoke({
    "query": "recent expenses",
    "user_id": "user_12345"  # Injected by system
})
```

### Use Cases for Injection

**1. User context**: Current user ID, permissions

```python
@tool
def get_my_expenses(
    date_range: str,
    user_id: Annotated[str, InjectedToolArg]
) -> list:
    """Get expenses for the current user."""
    return db.query(user_id=user_id, dates=date_range)
```

**2. Session state**: Current session, conversation ID

```python
@tool
def remember_preference(
    key: str,
    value: str,
    session: Annotated[Session, InjectedToolArg]
) -> dict:
    """Store user preference in current session."""
    session.store(key, value)
    return {"status": "saved"}
```

**3. System resources**: Database connections, API clients

```python
@tool
def execute_query(
    sql: str,
    db: Annotated[Database, InjectedToolArg]
) -> list:
    """Execute SQL query on database."""
    return db.execute(sql)
```

## Error Handling in Tools

Robust tools handle errors gracefully.

### Error Return Pattern

```python
@tool
def fetch_data(url: str) -> dict:
    """Fetch data from URL with error handling."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return {
            "status": "success",
            "data": response.json()
        }
    except requests.Timeout:
        return {
            "status": "error",
            "error": "Request timed out after 10 seconds"
        }
    except requests.HTTPError as e:
        return {
            "status": "error",
            "error": f"HTTP error: {e.response.status_code}"
        }
    except requests.RequestException as e:
        return {
            "status": "error",
            "error": f"Network error: {str(e)}"
        }
    except json.JSONDecodeError:
        return {
            "status": "error",
            "error": "Invalid JSON response"
        }
```

### Validation Error Pattern

```python
@tool
def process_expense(date: str, amount: str, category: str) -> dict:
    """Process expense with validation."""
    # Validate date format
    try:
        from datetime import datetime
        expense_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD"}

    # Validate amount
    try:
        amount_float = float(amount)
        if amount_float <= 0:
            return {"error": "Amount must be positive"}
    except ValueError:
        return {"error": "Invalid amount. Must be a number"}

    # Validate category
    valid_categories = ["Food", "Transport", "Housing", "Healthcare", "Entertainment"]
    if category not in valid_categories:
        return {
            "error": f"Invalid category. Must be one of: {', '.join(valid_categories)}"
        }

    # All valid, process
    return process_valid_expense(expense_date, amount_float, category)
```

## Testing Tool Definitions

### Unit Testing Tools

```python
import pytest

@tool
def calculate_percentage(value: float, total: float) -> dict:
    """Calculate percentage."""
    if total == 0:
        return {"error": "Total cannot be zero"}
    percentage = (value / total) * 100
    return {"percentage": round(percentage, 2)}

def test_calculate_percentage_normal():
    """Test normal percentage calculation."""
    result = calculate_percentage.invoke({"value": 25, "total": 100})
    assert result["percentage"] == 25.0

def test_calculate_percentage_zero_total():
    """Test with zero total."""
    result = calculate_percentage.invoke({"value": 10, "total": 0})
    assert "error" in result

def test_calculate_percentage_decimal():
    """Test with decimal result."""
    result = calculate_percentage.invoke({"value": 1, "total": 3})
    assert result["percentage"] == 33.33
```

### Testing Tool Schemas

```python
def test_tool_schema():
    """Test that tool schema is correct."""
    schema = calculate_percentage.args_schema.model_json_schema()

    # Check required fields
    assert "value" in schema["properties"]
    assert "total" in schema["properties"]

    # Check types
    assert schema["properties"]["value"]["type"] == "number"
    assert schema["properties"]["total"]["type"] == "number"
```

### Integration Testing with LLM

```python
import pytest
from langchain_openai import ChatOpenAI

@pytest.mark.integration
async def test_tool_with_llm():
    """Test tool integration with LLM."""
    llm = ChatOpenAI(model="gpt-4")
    llm_with_tools = llm.bind_tools([calculate_percentage])

    response = await llm_with_tools.ainvoke(
        "What percentage is 25 out of 100?"
    )

    # LLM should request the tool
    assert hasattr(response, 'tool_calls')
    assert len(response.tool_calls) > 0
    assert response.tool_calls[0]["name"] == "calculate_percentage"
```

## Best Practices

### 1. Descriptive Names

```python
# Good
@tool
def calculate_monthly_expense_average(...)

# Avoid
@tool
def calc(...)

@tool
def process(...)
```

### 2. Comprehensive Docstrings

```python
@tool
def add_expense(date: str, amount: float, category: str) -> dict:
    """Add a new expense entry to the database.

    This tool records financial transactions for expense tracking.
    All amounts are in USD. Dates must be in the past or today.

    Use this when:
    - User mentions spending money
    - Recording a purchase
    - Logging a transaction

    Args:
        date: Expense date in YYYY-MM-DD format (e.g., "2024-03-10")
        amount: Transaction amount in USD (must be positive)
        category: Expense category from: Food, Transport, Housing,
                 Healthcare, Entertainment

    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - id: Created expense ID (if successful)
        - error: Error message (if failed)

    Examples:
        >>> add_expense("2024-03-10", 45.50, "Food")
        {"status": "success", "id": 123}
    """
    # implementation...
```

### 3. Type Hints Everywhere

```python
from typing import Optional, List, Dict, Any

@tool
def query_expenses(
    start_date: str,
    end_date: str,
    category: Optional[str] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """Type hints improve schema generation."""
    pass
```

### 4. Input Validation

```python
from pydantic import BaseModel, Field, field_validator

class ToolInput(BaseModel):
    amount: float = Field(gt=0, description="Must be positive")
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        from datetime import datetime
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Invalid date format")
```

### 5. Error Handling

```python
@tool
def robust_tool(param: str) -> dict:
    """Tool with comprehensive error handling."""
    try:
        # Validate
        if not param:
            return {"error": "Parameter cannot be empty"}

        # Process
        result = process(param)

        # Return success
        return {"status": "success", "result": result}

    except SpecificError as e:
        logger.warning(f"Specific error in tool: {e}")
        return {"error": "Specific error occurred"}
    except Exception as e:
        logger.exception(f"Unexpected error in tool: {e}")
        return {"error": "Internal error"}
```

### 6. Consistent Return Format

```python
# Always return dict with status
@tool
def consistent_tool(param: str) -> dict:
    """Always return structured response."""
    try:
        result = process(param)
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
```

### 7. Tool Organization

```python
# Group related tools
class ExpenseTools:
    """Expense management tools."""

    @staticmethod
    @tool
    def add_expense(...):
        pass

    @staticmethod
    @tool
    def query_expenses(...):
        pass

    @staticmethod
    @tool
    def summarize_expenses(...):
        pass

    @classmethod
    def get_all_tools(cls):
        return [cls.add_expense, cls.query_expenses, cls.summarize_expenses]
```

## Summary

Tool definition patterns provide the foundation for effective agentic AI systems:

**Key Takeaways**:

1. **Three main patterns**: @tool (simple), StructuredTool (validated), BaseTool (full control)
2. **Schema design**: Use Pydantic for rich validation and documentation
3. **Built-in tools**: Leverage existing tools for common operations
4. **Toolkits**: Group related tools for organization
5. **LLM binding**: Understand tool call vs execution distinction
6. **Error handling**: Return structured errors, never raise exceptions
7. **Testing**: Unit test tools independently, integration test with LLMs
8. **Best practices**: Descriptive names, comprehensive docs, type hints, validation

**Tool Definition Checklist**:

```
Design:
☐ Clear, descriptive name
☐ Comprehensive docstring with use cases
☐ Type hints on all parameters
☐ Appropriate pattern (@tool, StructuredTool, BaseTool)

Validation:
☐ Input validation with Pydantic
☐ Custom validators for complex rules
☐ Clear error messages

Implementation:
☐ Robust error handling
☐ Consistent return format
☐ Idempotent operations
☐ Logging for debugging

Testing:
☐ Unit tests for logic
☐ Schema validation tests
☐ Integration tests with LLM
☐ Error case coverage
```

**Remember**: Well-defined tools are essential for agent success. Invest time in tool design, and your agents will be more capable, reliable, and maintainable.

## Next Steps

Now that you understand tool definition patterns, explore:

1. **[Tool Chaining](tool-chaining.md)**: Composing tools for complex operations
2. **[Tool Selection](tool-selection.md)**: How agents choose the right tool
3. **[Error Handling](error-handling.md)**: Advanced error recovery patterns
4. **[API Integration](api-integration.md)**: Integrating external APIs as tools
5. **[MCP Implementation Patterns](../model-context-protocol/mcp-implementation-patterns.md)**: Building MCP-compatible tools
6. **[Graph Workflows](../orchestration-patterns/graph-workflows.md)**: Using tools in workflow graphs

Start defining your tools with these patterns and build more capable agents.
