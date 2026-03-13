# MCP Implementation Patterns

## Table of Contents

- [Introduction](#introduction)
- [Server Implementation Basics](#server-implementation-basics)
- [Local MCP Servers](#local-mcp-servers)
- [Remote MCP Servers](#remote-mcp-servers)
- [MCP Proxy Patterns](#mcp-proxy-patterns)
- [Client Connection Patterns](#client-connection-patterns)
- [Database Integration](#database-integration)
- [Resource Management](#resource-management)
- [Multi-Server Client Setup](#multi-server-client-setup)
- [Transport Protocols](#transport-protocols)
- [Deployment Strategies](#deployment-strategies)
- [Error Handling and Reliability](#error-handling-and-reliability)
- [Testing MCP Servers](#testing-mcp-servers)
- [Best Practices](#best-practices)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

The Model Context Protocol (MCP) standardizes how AI agents connect to tools and data sources, but understanding the protocol specification is only half the battle. The other half is **implementation** - turning abstract protocol concepts into working code that agents can actually use.

This guide covers practical implementation patterns for building MCP servers and clients. We'll explore:

- **Server patterns**: Local servers, remote servers, and proxies
- **Connection patterns**: stdio, HTTP, and streamable transports
- **Integration patterns**: Databases, APIs, and external systems
- **Deployment patterns**: Development, staging, and production setups

> "Protocol standards enable interoperability. Implementation patterns enable productivity."

Whether you're building your first MCP server or architecting a production-grade multi-server setup, these patterns will help you implement MCP effectively.

### Why Implementation Patterns Matter

**Without patterns**:

- Reinventing solutions for common problems
- Inconsistent error handling
- Difficult to test and debug
- Hard to maintain and scale

**With patterns**:

- Proven solutions to common challenges
- Consistent, predictable behavior
- Easy to test and extend
- Production-ready from the start

### Implementation Overview

```
┌─────────────────────────────────────────┐
│          MCP Server Types                │
│                                          │
│  Local Server  →  stdio transport       │
│  Remote Server →  HTTP/SSE transport    │
│  Proxy Server  →  wraps remote server   │
└─────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────┐
│          MCP Clients                     │
│                                          │
│  Single Server    →  one connection     │
│  Multi-Server     →  multiple servers   │
│  Adaptive Client  →  dynamic discovery  │
└─────────────────────────────────────────┘
```

## Server Implementation Basics

Let's start with the fundamentals of implementing an MCP server using FastMCP.

### Minimal Server

The simplest possible MCP server:

```python
from fastmcp import FastMCP

# Create server instance
mcp = FastMCP("MinimalServer")

# Define a tool
@mcp.tool()
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"

# Run the server
if __name__ == "__main__":
    mcp.run()
```

**Running the server**:

```bash
# Activate the server via stdio
uv run server.py

# Inspect server capabilities
uv run fastmcp inspector server.py
```

### Server Anatomy

Every MCP server needs:

1. **Server instance**: The core MCP object
2. **Tools**: Functions the agent can call
3. **Resources** (optional): Data sources to read
4. **Prompts** (optional): Reusable prompt templates
5. **Transport**: How clients connect (stdio, HTTP, etc.)

```python
from fastmcp import FastMCP

mcp = FastMCP("ServerName")

# Tools - actions the agent can perform
@mcp.tool()
def action():
    pass

# Resources - data the agent can read
@mcp.resource("resource://path")
def data():
    pass

# Prompts - templates the agent can use
@mcp.prompt()
def template():
    pass
```

## Local MCP Servers

Local servers run on the same machine as the client and communicate via stdio (standard input/output).

### Use Cases

**Local servers are ideal for**:

- Development and testing
- Desktop applications (Claude Desktop, IDEs)
- Private data that shouldn't leave the machine
- Low-latency requirements
- Command-line tools

### Implementation Pattern: Expense Tracker

Let's implement a complete local MCP server that manages expenses with a SQLite database:

```python
from fastmcp import FastMCP
import sqlite3
from pathlib import Path

# Database and configuration
DB_PATH = Path(__file__).parent / "expenses.db"
CATEGORIES_PATH = Path(__file__).parent / "categories.json"

# Create MCP server
mcp = FastMCP("ExpenseTracker")

def init_db():
    """Initialize database schema."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS expenses(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                amount REAL NOT NULL,
                category TEXT NOT NULL,
                subcategory TEXT DEFAULT '',
                note TEXT DEFAULT ''
            )
        """)

# Initialize on startup
init_db()

@mcp.tool()
def add_expense(date: str, amount: float, category: str,
                subcategory: str = "", note: str = "") -> dict:
    """Add a new expense entry to the database.

    Args:
        date: Date in YYYY-MM-DD format
        amount: Expense amount
        category: Main expense category
        subcategory: Optional subcategory
        note: Optional note describing the expense

    Returns:
        Status and ID of created expense
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """INSERT INTO expenses(date, amount, category, subcategory, note)
               VALUES (?, ?, ?, ?, ?)""",
            (date, amount, category, subcategory, note),
        )
        return {"status": "success", "id": cursor.lastrowid}

@mcp.tool()
def list_expenses(start_date: str, end_date: str) -> list[dict]:
    """List expense entries within an inclusive date range.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        List of expense records matching the date range
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """SELECT id, date, amount, category, subcategory, note
               FROM expenses
               WHERE date BETWEEN ? AND ?
               ORDER BY date DESC, id DESC""",
            (start_date, end_date),
        )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

@mcp.tool()
def summarize_expenses(start_date: str, end_date: str,
                       category: str = None) -> list[dict]:
    """Summarize expenses by category within a date range.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        category: Optional filter by specific category

    Returns:
        Summary statistics grouped by category
    """
    with sqlite3.connect(DB_PATH) as conn:
        query = """
            SELECT category, SUM(amount) AS total_amount, COUNT(*) AS count
            FROM expenses
            WHERE date BETWEEN ? AND ?
        """
        params = [start_date, end_date]

        if category:
            query += " AND category = ?"
            params.append(category)

        query += " GROUP BY category ORDER BY total_amount DESC"

        cursor = conn.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

@mcp.resource("expense://categories", mime_type="application/json")
def expense_categories():
    """Provide available expense categories as a resource."""
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    mcp.run()
```

**Categories configuration** (`categories.json`):

```json
{
  "categories": [
    {
      "name": "Food & Dining",
      "subcategories": ["Groceries", "Restaurants", "Takeout", "Coffee"]
    },
    {
      "name": "Transportation",
      "subcategories": ["Gas", "Public Transit", "Parking", "Rideshare"]
    },
    {
      "name": "Housing",
      "subcategories": ["Rent", "Utilities", "Maintenance", "Insurance"]
    },
    {
      "name": "Healthcare",
      "subcategories": ["Doctor", "Pharmacy", "Insurance", "Gym"]
    },
    {
      "name": "Entertainment",
      "subcategories": ["Movies", "Streaming", "Games", "Books"]
    }
  ]
}
```

### Running and Testing

```bash
# Run the server
uv run local-server.py

# Inspect tools and resources
uv run fastmcp inspector local-server.py

# Use with Claude Desktop - add to claude_desktop_config.json:
{
  "mcpServers": {
    "expense-tracker": {
      "command": "uv",
      "args": ["run", "/path/to/local-server.py"]
    }
  }
}
```

### Key Patterns

**1. Database Initialization**: Initialize schema on startup

```python
def init_db():
    """Ensure database exists and has correct schema."""
    # Create tables if they don't exist
    # Idempotent - safe to run multiple times
```

**2. Connection Management**: Use context managers

```python
with sqlite3.connect(DB_PATH) as conn:
    # Connection automatically closed
    # Transactions automatically committed or rolled back
```

**3. Dynamic Queries**: Build queries safely with parameters

```python
query = "SELECT * FROM table WHERE date BETWEEN ? AND ?"
params = [start_date, end_date]

if category:
    query += " AND category = ?"
    params.append(category)

cursor.execute(query, params)  # Safe from SQL injection
```

**4. Resource Integration**: Expose static data as resources

```python
@mcp.resource("scheme://path", mime_type="application/json")
def provide_data():
    """Resources are read-only data sources."""
    return json_data
```

## Remote MCP Servers

Remote servers run separately from the client and communicate via HTTP or SSE (Server-Sent Events).

### Use Cases

**Remote servers are ideal for**:

- Cloud-deployed services
- Shared tools across multiple clients
- Heavy computation or specialized hardware
- Centralized data access
- Team collaboration

### Implementation Pattern: Cloud-Deployed Server

The same expense tracker, but deployed remotely:

```python
from fastmcp import FastMCP
import sqlite3
from pathlib import Path

# Same implementation as local server
DB_PATH = Path(__file__).parent / "expenses.db"
mcp = FastMCP("ExpenseTracker")

# ... (all the same tool definitions) ...

if __name__ == "__main__":
    # Remote servers typically use HTTP transport
    # FastMCP Cloud provides automatic deployment
    mcp.run()
```

### Deployment Options

**1. FastMCP Cloud**:

```bash
# Deploy to FastMCP Cloud
fastmcp deploy server.py

# Access via Streamable HTTP
# URL: https://your-server.fastmcp.app/mcp
```

**2. Custom Cloud Deployment**:

```python
# Deploy to any cloud provider (AWS, GCP, Azure, etc.)
# Use containerization for portability

# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY server.py .
CMD ["python", "server.py"]
```

**3. Self-Hosted with FastAPI Integration**:

```python
from fastapi import FastAPI
from fastmcp import FastMCP

# Create FastAPI app
app = FastAPI()

# Create MCP server from FastAPI
mcp = FastMCP.from_fastapi(app=app, name="ExpenseTracker")

# ... define tools ...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## MCP Proxy Patterns

Proxies wrap remote MCP servers to provide local stdio access. This is useful for integrating remote services with desktop applications.

### Use Case

Desktop applications like Claude Desktop expect stdio connections, but your MCP server is deployed remotely. A proxy bridges this gap.

### Implementation Pattern

```python
from fastmcp import FastMCP

# Create a proxy to your remote FastMCP Cloud server
# FastMCP Cloud uses Streamable HTTP (default transport)
mcp = FastMCP.as_proxy(
    "https://splendid-gold-dingo.fastmcp.app/mcp",
    name="Expense Tracker Proxy",
)

if __name__ == "__main__":
    # This runs via STDIO for local connections
    # But forwards requests to the remote HTTP server
    mcp.run()
```

### How Proxies Work

```
┌──────────────────┐
│  Claude Desktop  │
└────────┬─────────┘
         │ stdio
         ▼
┌──────────────────┐
│   Local Proxy    │
└────────┬─────────┘
         │ HTTP/SSE
         ▼
┌──────────────────┐
│  Remote Server   │
│  (FastMCP Cloud) │
└──────────────────┘
```

### Configuration Example

**Claude Desktop config** with proxy:

```json
{
  "mcpServers": {
    "expense-tracker": {
      "command": "uv",
      "args": ["run", "/path/to/proxy.py"]
    }
  }
}
```

The proxy handles:

- Protocol translation (stdio ↔ HTTP)
- Connection management
- Error handling and retries
- Credential management (if needed)

## Client Connection Patterns

Clients need to connect to one or more MCP servers. Let's explore different connection patterns.

### Single Server Client

Connect to one server:

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

async def main():
    # Connect to single server
    client = MultiServerMCPClient({
        "expense": {
            "transport": "stdio",
            "command": "uv",
            "args": ["run", "local-server.py"],
        }
    })

    # Get available tools
    tools = await client.get_tools()
    print(f"Available tools: {[t.name for t in tools]}")

    # Use with LLM
    llm = ChatOpenAI(model="gpt-4")
    llm_with_tools = llm.bind_tools(tools)

    response = await llm_with_tools.ainvoke(
        "Add an expense: $45.50 for groceries on 2024-03-10"
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

## Multi-Server Client Setup

Connect to multiple servers simultaneously:

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
import json

# Define multiple servers with different transports
SERVERS = {
    "expense": {
        "transport": "stdio",
        "command": "uv",
        "args": ["run", "local-expense-server.py"],
    },
    "weather": {
        "transport": "streamable_http",  # or "sse"
        "url": "https://weather-server.fastmcp.app/mcp",
    },
    "documents": {
        "transport": "stdio",
        "command": "python",
        "args": ["/path/to/document-server.py"],
        "env": {
            "DOC_ROOT": "/path/to/documents",
            "INDEX_PATH": "/path/to/index"
        },
    },
}

async def main():
    # Connect to all servers
    client = MultiServerMCPClient(SERVERS)

    # Get tools from all servers
    tools = await client.get_tools()

    # Create tool lookup
    tool_map = {tool.name: tool for tool in tools}

    print(f"Connected to {len(SERVERS)} servers")
    print(f"Available tools: {list(tool_map.keys())}")

    # Use tools with LLM
    llm = ChatOpenAI(model="gpt-4")
    llm_with_tools = llm.bind_tools(tools)

    prompt = "What's the weather in Tokyo and how much did I spend on food last month?"
    response = await llm_with_tools.ainvoke(prompt)

    # Handle tool calls
    if getattr(response, "tool_calls", None):
        tool_messages = []

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool_id = tool_call["id"]

            # Execute tool
            result = await tool_map[tool_name].ainvoke(tool_args)

            tool_messages.append(
                ToolMessage(
                    tool_call_id=tool_id,
                    content=json.dumps(result)
                )
            )

        # Get final response with tool results
        final_response = await llm_with_tools.ainvoke(
            [prompt, response, *tool_messages]
        )
        print(f"Final response: {final_response.content}")
    else:
        print(f"Direct response: {response.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Key Patterns in Multi-Server Setup

**1. Server Configuration**: Define each server with its transport

```python
SERVERS = {
    "server_name": {
        "transport": "stdio" | "http" | "sse" | "streamable_http",
        # transport-specific config...
    }
}
```

**2. Tool Namespacing**: Tools from different servers are merged

```python
# Tool names should be unique across servers
# Or use prefixes: "expense_add", "weather_get", etc.
```

**3. Environment Variables**: Pass configuration to servers

```python
{
    "env": {
        "API_KEY": "...",
        "CONFIG_PATH": "...",
        # Server-specific environment
    }
}
```

**4. Error Handling**: One server failure shouldn't break others

```python
try:
    result = await tool.ainvoke(args)
except Exception as e:
    # Log error, return error message
    # Other servers continue working
    result = {"error": str(e)}
```

## Transport Protocols

MCP supports multiple transport protocols. Choose based on your deployment model.

### stdio Transport

**Use for**: Local servers, desktop applications

**Characteristics**:

- Process-based communication
- Low latency
- Automatic lifecycle management
- Secure (no network exposure)

**Configuration**:

```python
{
    "transport": "stdio",
    "command": "uv",
    "args": ["run", "server.py"],
    "env": {...}  # Optional environment variables
}
```

**Pros**:

- Simple to set up
- No network configuration
- Automatic process cleanup

**Cons**:

- Single machine only
- One client per server instance
- Process overhead

### HTTP Transport

**Use for**: Remote servers, REST-style APIs

**Characteristics**:

- Request-response pattern
- Standard HTTP protocol
- Easy to deploy and monitor
- Firewall-friendly

**Configuration**:

```python
{
    "transport": "http",
    "url": "https://api.example.com/mcp"
}
```

**Pros**:

- Industry standard
- Easy debugging with curl/Postman
- Load balancing and scaling
- Works through proxies

**Cons**:

- Higher latency than stdio
- No streaming support
- Requires authentication setup

### Streamable HTTP / SSE

**Use for**: Remote servers with streaming support

**Characteristics**:

- Bidirectional communication
- Server can push updates
- Connection multiplexing
- Efficient for real-time data

**Configuration**:

```python
{
    "transport": "streamable_http",  # or "sse"
    "url": "https://api.example.com/mcp"
}
```

**Pros**:

- Real-time updates
- Efficient bandwidth usage
- Bidirectional messaging
- Better than polling

**Cons**:

- More complex than HTTP
- Some proxies may interfere
- Connection management overhead

### Transport Selection Guide

```
┌─────────────────────────────────────────────────────────┐
│ Deployment Model    │ Recommended Transport             │
├─────────────────────────────────────────────────────────┤
│ Local development   │ stdio                             │
│ Desktop app         │ stdio (or proxy to remote)        │
│ Remote API          │ HTTP                              │
│ Real-time data      │ Streamable HTTP / SSE             │
│ High-volume         │ Streamable HTTP with load balancer│
│ Behind firewall     │ stdio or HTTP with VPN            │
└─────────────────────────────────────────────────────────┘
```

## Database Integration

Most practical MCP servers integrate with databases. Here are patterns for different scenarios.

### SQLite for Local Data

**Use when**: Single-user, local data, simple persistence

```python
import sqlite3
from pathlib import Path
from contextlib import contextmanager

class DatabaseManager:
    """Manage SQLite database for MCP server."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize database schema."""
        with self.get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS expenses(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    amount REAL NOT NULL,
                    category TEXT NOT NULL,
                    note TEXT DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_expenses_date
                ON expenses(date);

                CREATE INDEX IF NOT EXISTS idx_expenses_category
                ON expenses(category);
            """)

    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

# Use in MCP server
db = DatabaseManager(Path(__file__).parent / "data.db")

@mcp.tool()
def add_expense(date: str, amount: float, category: str) -> dict:
    """Add expense with proper transaction handling."""
    with db.get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO expenses(date, amount, category) VALUES (?, ?, ?)",
            (date, amount, category)
        )
        return {"status": "success", "id": cursor.lastrowid}
```

### PostgreSQL for Production

**Use when**: Multi-user, remote access, complex queries

```python
import asyncpg
from typing import Optional

class PostgresManager:
    """Manage PostgreSQL connection for MCP server."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Create connection pool."""
        self.pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=2,
            max_size=10,
        )

    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()

    async def execute_query(self, query: str, *args):
        """Execute query with connection from pool."""
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

# Use in MCP server (async)
db = PostgresManager("postgresql://user:pass@localhost/db")

@mcp.tool()
async def add_expense(date: str, amount: float, category: str) -> dict:
    """Add expense with async PostgreSQL."""
    result = await db.execute_query(
        "INSERT INTO expenses(date, amount, category) VALUES ($1, $2, $3) RETURNING id",
        date, amount, category
    )
    return {"status": "success", "id": result[0]["id"]}
```

### Key Database Patterns

**1. Connection Pooling**: Reuse connections for efficiency

```python
# Create pool at startup
# Reuse connections across requests
# Close pool at shutdown
```

**2. Transaction Management**: Ensure data consistency

```python
with conn:
    # Multiple operations
    # Automatic commit or rollback
```

**3. Parameterized Queries**: Prevent SQL injection

```python
# Never: f"SELECT * FROM table WHERE id = {user_input}"
# Always: "SELECT * FROM table WHERE id = ?", (user_input,)
```

**4. Error Recovery**: Handle database errors gracefully

```python
try:
    result = execute_query(...)
except DatabaseError as e:
    logger.error(f"Database error: {e}")
    return {"error": "Database operation failed"}
```

## Resource Management

Resources provide read-only access to data. Common patterns:

### File-Based Resources

```python
@mcp.resource("config://settings", mime_type="application/json")
def get_settings():
    """Provide configuration as a resource."""
    with open("config.json") as f:
        return f.read()

@mcp.resource("docs://readme", mime_type="text/markdown")
def get_documentation():
    """Provide documentation."""
    with open("README.md") as f:
        return f.read()
```

### Dynamic Resources

```python
@mcp.resource("stats://daily", mime_type="application/json")
def get_daily_stats():
    """Generate statistics on demand."""
    with db.get_connection() as conn:
        stats = conn.execute("""
            SELECT
                date,
                SUM(amount) as total,
                COUNT(*) as count
            FROM expenses
            WHERE date >= date('now', '-30 days')
            GROUP BY date
            ORDER BY date DESC
        """).fetchall()

        return json.dumps([dict(row) for row in stats])
```

### Resource Best Practices

**1. Use appropriate MIME types**: Help clients understand data

```python
mime_type="application/json"  # JSON data
mime_type="text/plain"        # Plain text
mime_type="text/markdown"     # Markdown
mime_type="application/xml"   # XML
```

**2. Keep resources read-only**: Use tools for mutations

**3. Cache when appropriate**: Avoid recomputing expensive data

```python
from functools import lru_cache

@lru_cache(maxsize=1)
@mcp.resource("data://processed")
def get_processed_data():
    """Cached resource for expensive computation."""
    return expensive_computation()
```

**4. Use clear URI schemes**: Make resource purposes obvious

```python
# Good
"config://database"
"stats://monthly"
"docs://api-reference"

# Avoid
"resource1"
"data"
```

## Deployment Strategies

### Development Deployment

**Goal**: Fast iteration, easy debugging

```python
# Local stdio server
if __name__ == "__main__":
    mcp.run()

# Run directly
python server.py

# Or with auto-reload
uvicorn server:app --reload
```

**Configuration**:

- SQLite database
- Local file paths
- stdio transport
- Verbose logging

### Staging Deployment

**Goal**: Production-like environment for testing

```python
# HTTP server with staging database
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://staging...")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

if __name__ == "__main__":
    mcp.run(host="0.0.0.0", port=8000)
```

**Configuration**:

- Staging database (PostgreSQL)
- Environment variables
- HTTP transport
- Structured logging
- Basic monitoring

### Production Deployment

**Goal**: Reliability, scalability, security

```python
# Production server with all safeguards
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastmcp import FastMCP

# Logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app with middleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create MCP server
mcp = FastMCP.from_fastapi(app=app, name="ProductionServer")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check for load balancer."""
    return {"status": "healthy"}

# ... define tools with error handling ...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
```

**Configuration**:

- Production database with connection pooling
- Environment-based configuration
- HTTP transport with TLS
- Structured logging to aggregation service
- Health checks and metrics
- Rate limiting
- Authentication/authorization
- CORS policies
- Error tracking (Sentry, etc.)

### Deployment Checklist

```
Development:
☐ Local database
☐ stdio transport
☐ Debug logging
☐ No authentication

Staging:
☐ Staging database
☐ HTTP transport
☐ Info logging
☐ Basic auth
☐ Monitoring

Production:
☐ Production database with backups
☐ HTTPS with TLS
☐ Structured logging
☐ Full authentication
☐ Rate limiting
☐ Error tracking
☐ Health checks
☐ Load balancing
☐ Auto-scaling
☐ Disaster recovery plan
```

## Error Handling and Reliability

Robust MCP servers handle errors gracefully.

### Tool Error Handling

```python
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

@mcp.tool()
def add_expense(date: str, amount: float, category: str) -> Dict[str, Any]:
    """Add expense with comprehensive error handling."""
    try:
        # Validate inputs
        if amount < 0:
            return {"error": "Amount must be positive"}

        # Parse date
        from datetime import datetime
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return {"error": "Invalid date format. Use YYYY-MM-DD"}

        # Execute operation
        with db.get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO expenses(date, amount, category) VALUES (?, ?, ?)",
                (date, amount, category)
            )
            return {"status": "success", "id": cursor.lastrowid}

    except sqlite3.IntegrityError as e:
        logger.error(f"Database integrity error: {e}")
        return {"error": "Database constraint violation"}

    except sqlite3.OperationalError as e:
        logger.error(f"Database operational error: {e}")
        return {"error": "Database temporarily unavailable"}

    except Exception as e:
        logger.exception(f"Unexpected error in add_expense: {e}")
        return {"error": "Internal server error"}
```

### Retry Logic

```python
import asyncio
from typing import TypeVar, Callable

T = TypeVar('T')

async def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> T:
    """Retry function with exponential backoff."""
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise

            logger.warning(
                f"Attempt {attempt + 1} failed: {e}. "
                f"Retrying in {delay}s..."
            )
            await asyncio.sleep(delay)
            delay *= backoff_factor

# Usage
@mcp.tool()
async def fetch_external_data(query: str) -> dict:
    """Fetch data with retry logic."""
    async def _fetch():
        # Call external API
        return await external_api.get(query)

    return await retry_with_backoff(_fetch, max_retries=3)
```

### Circuit Breaker Pattern

```python
from datetime import datetime, timedelta

class CircuitBreaker:
    """Prevent cascading failures."""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker."""
        if self.state == "open":
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout):
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise

    def on_success(self):
        """Reset on successful call."""
        self.failures = 0
        self.state = "closed"

    def on_failure(self):
        """Track failure."""
        self.failures += 1
        self.last_failure_time = datetime.now()

        if self.failures >= self.failure_threshold:
            self.state = "open"
            logger.error("Circuit breaker OPENED")
```

## Testing MCP Servers

### Manual Testing with Inspector

```bash
# Inspect server capabilities
uv run fastmcp inspector server.py

# This shows:
# - Available tools
# - Tool schemas
# - Resources
# - Prompts
```

### Automated Testing

```python
import pytest
from fastmcp import FastMCP

@pytest.fixture
def mcp_server():
    """Create test server."""
    mcp = FastMCP("test-server")

    @mcp.tool()
    def test_tool(value: str) -> str:
        return f"processed: {value}"

    return mcp

def test_tool_execution(mcp_server):
    """Test tool can be called."""
    result = mcp_server.call_tool("test_tool", {"value": "hello"})
    assert result == "processed: hello"

def test_tool_schema(mcp_server):
    """Test tool schema is correct."""
    tools = mcp_server.list_tools()
    assert "test_tool" in tools
    assert tools["test_tool"]["input_schema"]["properties"]["value"]["type"] == "string"
```

### Integration Testing

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

async def test_end_to_end():
    """Test complete client-server interaction."""
    # Start test server
    client = MultiServerMCPClient({
        "test": {
            "transport": "stdio",
            "command": "python",
            "args": ["test_server.py"],
        }
    })

    # Get tools
    tools = await client.get_tools()
    assert len(tools) > 0

    # Execute tool
    result = await tools[0].ainvoke({"value": "test"})
    assert "processed" in result

asyncio.run(test_end_to_end())
```

## Best Practices

### Server Design

**1. Single Responsibility**: Each server should have a focused purpose

```python
# Good: Focused server
mcp = FastMCP("ExpenseTracker")  # Only expense-related tools

# Avoid: Kitchen sink server
mcp = FastMCP("Everything")  # Expenses, weather, email, etc.
```

**2. Descriptive Tool Names**: Make tools self-documenting

```python
# Good
@mcp.tool()
def add_expense(...):
    """Add a new expense entry."""

@mcp.tool()
def summarize_expenses_by_category(...):
    """Calculate expense totals grouped by category."""

# Avoid
@mcp.tool()
def add(...):
    """Add something."""

@mcp.tool()
def get(...):
    """Get data."""
```

**3. Comprehensive Docstrings**: Document parameters and return values

```python
@mcp.tool()
def list_expenses(start_date: str, end_date: str) -> list[dict]:
    """List expense entries within an inclusive date range.

    This tool retrieves all expense records where the expense date
    falls between start_date and end_date (inclusive).

    Args:
        start_date: Start date in YYYY-MM-DD format (e.g., "2024-01-01")
        end_date: End date in YYYY-MM-DD format (e.g., "2024-12-31")

    Returns:
        List of expense records, where each record is a dictionary containing:
        - id: Unique expense identifier
        - date: Expense date
        - amount: Expense amount
        - category: Expense category
        - subcategory: Optional subcategory
        - note: Optional descriptive note

    Example:
        >>> list_expenses("2024-03-01", "2024-03-31")
        [
            {
                "id": 1,
                "date": "2024-03-10",
                "amount": 45.50,
                "category": "Food & Dining",
                "subcategory": "Groceries",
                "note": "Weekly shopping"
            }
        ]
    """
    # implementation...
```

**4. Type Hints**: Use Python type hints for all parameters

```python
from typing import Optional, List, Dict, Any

@mcp.tool()
def process_data(
    input_data: str,
    options: Optional[Dict[str, Any]] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Type hints improve discoverability."""
    pass
```

**5. Idempotent Operations**: Make operations safe to retry

```python
@mcp.tool()
def create_or_update_expense(id: Optional[int], ...):
    """Idempotent - same result if called multiple times."""
    if id:
        # Update existing
        conn.execute("UPDATE expenses SET ... WHERE id = ?", ...)
    else:
        # Create new
        conn.execute("INSERT INTO expenses ...", ...)
```

### Performance

**1. Connection Pooling**: Reuse database connections

```python
# Create pool at startup
pool = create_connection_pool()

# Reuse in tools
@mcp.tool()
def query_data():
    with pool.get_connection() as conn:
        return conn.execute(...)
```

**2. Lazy Loading**: Load resources only when needed

```python
_cache = None

@mcp.resource("data://large-dataset")
def get_large_dataset():
    """Load expensive data only when accessed."""
    global _cache
    if _cache is None:
        _cache = load_expensive_data()
    return _cache
```

**3. Batch Operations**: Reduce round trips

```python
@mcp.tool()
def add_multiple_expenses(expenses: List[Dict]) -> Dict:
    """Add multiple expenses in one transaction."""
    with db.get_connection() as conn:
        conn.executemany(
            "INSERT INTO expenses(...) VALUES (?, ?, ?)",
            [(e["date"], e["amount"], e["category"]) for e in expenses]
        )
        return {"status": "success", "count": len(expenses)}
```

### Security

**1. Input Validation**: Never trust client input

```python
@mcp.tool()
def query_by_id(id: int) -> dict:
    """Validate all inputs."""
    if not isinstance(id, int) or id < 1:
        return {"error": "Invalid ID"}

    # Safe to use
    result = db.execute("SELECT * FROM table WHERE id = ?", (id,))
    return result
```

**2. Parameterized Queries**: Prevent SQL injection

```python
# NEVER
query = f"SELECT * FROM table WHERE id = {user_input}"

# ALWAYS
query = "SELECT * FROM table WHERE id = ?"
result = conn.execute(query, (user_input,))
```

**3. Error Message Sanitization**: Don't leak sensitive info

```python
try:
    result = operation()
except Exception as e:
    logger.exception(f"Error: {e}")  # Log full error
    return {"error": "Operation failed"}  # Generic message to client
```

**4. Rate Limiting**: Prevent abuse

```python
from functools import wraps
from time import time

def rate_limit(calls: int, period: int):
    """Rate limit decorator."""
    call_times = []

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time()
            call_times[:] = [t for t in call_times if now - t < period]

            if len(call_times) >= calls:
                raise Exception("Rate limit exceeded")

            call_times.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@mcp.tool()
@rate_limit(calls=10, period=60)  # 10 calls per minute
def expensive_operation():
    """Rate-limited tool."""
    pass
```

### Monitoring

**1. Structured Logging**: Enable searchable logs

```python
import logging
import json

logger = logging.getLogger(__name__)

@mcp.tool()
def add_expense(date: str, amount: float, category: str):
    """Log structured data."""
    logger.info(json.dumps({
        "event": "expense_added",
        "date": date,
        "amount": amount,
        "category": category,
        "timestamp": datetime.now().isoformat()
    }))
```

**2. Metrics**: Track key indicators

```python
from prometheus_client import Counter, Histogram

tool_calls = Counter('mcp_tool_calls_total', 'Total tool calls', ['tool_name'])
tool_duration = Histogram('mcp_tool_duration_seconds', 'Tool execution time', ['tool_name'])

@mcp.tool()
def monitored_tool():
    """Track metrics."""
    tool_calls.labels(tool_name='monitored_tool').inc()

    with tool_duration.labels(tool_name='monitored_tool').time():
        return execute_tool()
```

## Summary

MCP implementation patterns provide proven solutions for building production-ready MCP servers and clients:

**Key Takeaways**:

1. **Server Types**: Choose local (stdio), remote (HTTP), or proxy based on deployment needs
2. **Transport Protocols**: Match transport to use case (stdio for local, HTTP/SSE for remote)
3. **Database Integration**: Use appropriate database with proper connection management
4. **Multi-Server Clients**: Connect to multiple MCP servers simultaneously
5. **Error Handling**: Build robust servers with retries, circuit breakers, and graceful failures
6. **Deployment**: Progress from development (stdio, SQLite) to production (HTTP, PostgreSQL)
7. **Testing**: Use inspector for manual testing, pytest for automation
8. **Best Practices**: Focus on single responsibility, types, docs, security, and monitoring

**Implementation Checklist**:

```
Server Basics:
☐ FastMCP instance created
☐ Tools defined with docstrings
☐ Resources configured
☐ Type hints on all parameters

Reliability:
☐ Input validation
☐ Error handling
☐ Transaction management
☐ Retry logic

Production:
☐ Environment-based config
☐ Structured logging
☐ Health checks
☐ Monitoring
☐ Rate limiting
☐ Security review
```

**Remember**: Start simple (local stdio server), iterate quickly, and add complexity only as needed. The best MCP server is one that works reliably and serves its users well.

## Next Steps

Now that you understand MCP implementation patterns, explore:

1. **[MCP Context](mcp-context.md)**: Managing context across tools and resources
2. **[MCP Tools](mcp-tools.md)**: Advanced tool patterns and composition
3. **[MCP Ecosystem](mcp-ecosystem.md)**: Existing servers and discovery
4. **[Building MCP Servers](building-mcp-servers.md)**: Deep dive into server architecture
5. **[Tool Definition Patterns](../tool-use/tool-definition-patterns.md)**: How to define tools effectively
6. **[Graph Workflows](../orchestration-patterns/graph-workflows.md)**: Integrating MCP with workflow engines

Start building your first MCP server and join the composable AI ecosystem.
