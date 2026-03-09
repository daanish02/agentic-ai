# MCP Resources

## Table of Contents

- [Introduction](#introduction)
- [What Are Resources?](#what-are-resources)
- [Resource URIs](#resource-uris)
- [Resource Schemas](#resource-schemas)
- [Listing Resources](#listing-resources)
- [Reading Resources](#reading-resources)
- [Dynamic Resources](#dynamic-resources)
- [Resource Subscriptions](#resource-subscriptions)
- [Resource Templates](#resource-templates)
- [Common Resource Patterns](#common-resource-patterns)
- [Resource Providers](#resource-providers)
- [Resource Caching](#resource-caching)
- [Resource Security](#resource-security)
- [Resource Discovery](#resource-discovery)
- [Advanced Resource Techniques](#advanced-resource-techniques)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Resources are one of MCP's three core primitives. They represent **data sources that agents can read**. Think of resources as exposing your data through standardized URIs, similar to how HTTP exposes web pages.

Resources enable agents to:
- Access files and documents
- Query databases
- Fetch API data
- Read system information
- Access any external data source

The key innovation: **resources provide a uniform interface for accessing heterogeneous data sources**. Whether reading a file, querying a database, or calling an API, the agent uses the same MCP resource interface.

> "Resources make all data sources look the same to the agent."

This guide covers everything about MCP resources: from basic URI schemes to advanced subscription patterns, from static files to dynamic queries.

## What Are Resources?

### Definition

A **resource** in MCP is:
- A piece of data that can be read
- Identified by a URI
- Exposed by an MCP server
- Accessible to MCP clients

### Resource vs Tool

Understanding the distinction is important:

```
Resource:
- Represents DATA (read-only)
- "What information is available?"
- Examples: file, database table, API endpoint
- Operation: READ

Tool:
- Represents ACTION (side effects)
- "What can I do?"
- Examples: send email, create file, run calculation
- Operation: EXECUTE
```

**When to use resources**:
```python
# Reading data → Resource
content = await client.read_resource("file:///document.txt")
users = await client.read_resource("db://localhost/users")
weather = await client.read_resource("api://weather/forecast")

# Modifying data → Tool
await client.call_tool("send_email", {"to": "user@example.com"})
await client.call_tool("create_file", {"path": "/new.txt"})
await client.call_tool("delete_record", {"id": 123})
```

### Resource Structure

A resource has these components:

```python
{
    "uri": "file:///path/to/file.txt",           # Unique identifier
    "name": "Important Document",                 # Human-readable name
    "description": "Q4 planning document",        # What the resource contains
    "mimeType": "text/plain"                      # Content type
}
```

When read, resources return content:

```python
{
    "contents": [
        {
            "uri": "file:///path/to/file.txt",
            "mimeType": "text/plain",
            "text": "File contents here..."
        }
    ]
}
```

## Resource URIs

URIs (Uniform Resource Identifiers) uniquely identify resources.

### URI Structure

```
scheme://authority/path?query#fragment
  │        │         │     │      │
  │        │         │     │      └─ Optional fragment
  │        │         │     └─ Optional query parameters
  │        │         └─ Path to resource
  │        └─ Server/host (optional)
  └─ Resource type (file, db, api, etc.)
```

### Standard URI Schemes

**File Resources**:
```python
"file:///absolute/path/to/file.txt"
"file:///C:/Windows/path/on/windows.txt"
"file:///home/user/documents/report.pdf"
```

**Database Resources**:
```python
"db://localhost/database_name/table_name"
"db://server:5432/mydb/users"
"postgres://host/database/table/row/123"
```

**API Resources**:
```python
"api://service.com/endpoint"
"https://api.example.com/v1/data"
"http://internal-service/metrics"
```

**Custom Schemes**:
```python
"git://repo/path/to/file"
"s3://bucket/key"
"redis://localhost/key"
"custom://whatever/you/want"
```

### URI Best Practices

**1. Be Specific**:
```python
# Too vague
"file:///data"

# Better
"file:///data/users/2024/january/report.csv"

# Even better (with fragment)
"file:///data/report.csv#section_2"
```

**2. Use Query Parameters for Variations**:
```python
"db://localhost/users?active=true"
"api://weather/forecast?location=tokyo&days=7"
"file:///logs?date=2024-01-23&level=error"
```

**3. Make URIs Stable**:
```python
# Bad: includes timestamp that changes
"file:///export_20240123_143022.csv"

# Good: stable identifier
"file:///export_latest.csv"
"db://localhost/reports/latest"
```

**4. Use Hierarchy**:
```python
# Organized by hierarchy
"db://production/customers/active"
"db://production/customers/inactive"
"db://production/orders/pending"
"db://production/orders/completed"
```

## Resource Schemas

### Resource List Schema

When a server lists resources:

```json
{
    "resources": [
        {
            "uri": "file:///documents/plan.md",
            "name": "Project Plan",
            "description": "2024 Q1 project planning document",
            "mimeType": "text/markdown"
        },
        {
            "uri": "db://localhost/analytics",
            "name": "Analytics Data",
            "description": "Real-time analytics metrics",
            "mimeType": "application/json"
        }
    ]
}
```

### Resource Content Schema

When a resource is read:

```json
{
    "contents": [
        {
            "uri": "file:///documents/plan.md",
            "mimeType": "text/markdown",
            "text": "# Project Plan\n\n## Goals\n..."
        }
    ]
}
```

For binary content:

```json
{
    "contents": [
        {
            "uri": "file:///images/chart.png",
            "mimeType": "image/png",
            "blob": "base64_encoded_binary_data_here"
        }
    ]
}
```

### MIME Types

Common MIME types for resources:

```python
TEXT_TYPES = {
    "text/plain": "Plain text",
    "text/markdown": "Markdown",
    "text/html": "HTML",
    "text/csv": "CSV data",
    "application/json": "JSON data",
    "application/xml": "XML data"
}

BINARY_TYPES = {
    "image/png": "PNG image",
    "image/jpeg": "JPEG image",
    "application/pdf": "PDF document",
    "application/zip": "ZIP archive",
    "application/octet-stream": "Binary data"
}

DATABASE_TYPES = {
    "application/x-sqlite3": "SQLite database",
    "application/vnd.ms-excel": "Excel spreadsheet"
}
```

## Listing Resources

### Basic Listing

```python
# Client requests resource list
request = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "resources/list",
    "params": {}
}

# Server responds
response = {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "resources": [
            {
                "uri": "file:///data/users.json",
                "name": "User Data",
                "description": "All registered users",
                "mimeType": "application/json"
            },
            {
                "uri": "file:///data/orders.json",
                "name": "Order Data",
                "description": "All customer orders",
                "mimeType": "application/json"
            }
        ]
    }
}
```

### Server-Side Implementation

```python
class FileSystemResourceServer(MCPServer):
    def __init__(self, root_dir):
        super().__init__("filesystem")
        self.root_dir = Path(root_dir)
    
    async def list_resources(self):
        """List all files in root directory"""
        resources = []
        
        for file_path in self.root_dir.rglob('*'):
            if file_path.is_file():
                # Create file:// URI
                uri = f"file://{file_path.absolute()}"
                
                # Determine MIME type
                mime_type = self.get_mime_type(file_path)
                
                resources.append({
                    "uri": uri,
                    "name": file_path.name,
                    "description": f"File: {file_path.relative_to(self.root_dir)}",
                    "mimeType": mime_type
                })
        
        return {"resources": resources}
    
    def get_mime_type(self, file_path):
        """Determine MIME type from file extension"""
        import mimetypes
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"
```

### Client-Side Usage

```python
class ResourceDiscoveryClient:
    def __init__(self):
        self.mcp = MCPClient()
    
    async def discover_resources(self, server_id):
        """Discover all resources from a server"""
        response = await self.mcp.send_request(
            server_id,
            "resources/list",
            {}
        )
        
        resources = response['resources']
        
        # Organize by type
        by_type = {}
        for resource in resources:
            mime = resource['mimeType']
            if mime not in by_type:
                by_type[mime] = []
            by_type[mime].append(resource)
        
        return by_type
    
    async def print_resource_catalog(self, server_id):
        """Print organized resource catalog"""
        by_type = await self.discover_resources(server_id)
        
        for mime_type, resources in by_type.items():
            print(f"\n{mime_type}:")
            for resource in resources:
                print(f"  - {resource['name']}: {resource['uri']}")
```

## Reading Resources

### Basic Reading

```python
# Client requests resource content
request = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "resources/read",
    "params": {
        "uri": "file:///data/users.json"
    }
}

# Server responds with content
response = {
    "jsonrpc": "2.0",
    "id": 2,
    "result": {
        "contents": [
            {
                "uri": "file:///data/users.json",
                "mimeType": "application/json",
                "text": "{\"users\": [{\"id\": 1, \"name\": \"Alice\"}]}"
            }
        ]
    }
}
```

### Server-Side Implementation

```python
class FileSystemResourceServer(MCPServer):
    async def read_resource(self, uri):
        """Read resource content"""
        # Parse URI
        if not uri.startswith("file://"):
            raise ValueError(f"Unsupported URI scheme: {uri}")
        
        # Extract file path
        file_path = Path(uri[7:])  # Remove "file://"
        
        # Security check: must be under root_dir
        if not self.is_safe_path(file_path):
            raise PermissionError(f"Access denied: {uri}")
        
        # Check existence
        if not file_path.exists():
            raise FileNotFoundError(f"Resource not found: {uri}")
        
        # Read content
        mime_type = self.get_mime_type(file_path)
        
        if mime_type.startswith("text/") or mime_type == "application/json":
            # Text content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": mime_type,
                    "text": content
                }]
            }
        else:
            # Binary content
            with open(file_path, 'rb') as f:
                data = f.read()
            
            import base64
            encoded = base64.b64encode(data).decode('ascii')
            
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": mime_type,
                    "blob": encoded
                }]
            }
    
    def is_safe_path(self, file_path):
        """Check if path is under root directory"""
        try:
            file_path.resolve().relative_to(self.root_dir.resolve())
            return True
        except ValueError:
            return False
```

### Client-Side Usage

```python
class ResourceReader:
    def __init__(self, mcp_client):
        self.mcp = mcp_client
    
    async def read_text_resource(self, uri):
        """Read a text resource"""
        response = await self.mcp.send_request(
            "resources/read",
            {"uri": uri}
        )
        
        content = response['contents'][0]
        return content['text']
    
    async def read_json_resource(self, uri):
        """Read and parse JSON resource"""
        text = await self.read_text_resource(uri)
        import json
        return json.loads(text)
    
    async def read_binary_resource(self, uri):
        """Read a binary resource"""
        response = await self.mcp.send_request(
            "resources/read",
            {"uri": uri}
        )
        
        content = response['contents'][0]
        
        if 'blob' in content:
            import base64
            return base64.b64decode(content['blob'])
        else:
            raise ValueError("Resource is not binary")
```

## Dynamic Resources

Resources don't have to be static files. They can be generated dynamically.

### Database Resources

```python
class DatabaseResourceServer(MCPServer):
    def __init__(self, db_connection):
        super().__init__("database")
        self.db = db_connection
    
    async def list_resources(self):
        """List database tables as resources"""
        cursor = self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = cursor.fetchall()
        
        resources = []
        for (table_name,) in tables:
            resources.append({
                "uri": f"db://local/{table_name}",
                "name": table_name,
                "description": f"Database table: {table_name}",
                "mimeType": "application/json"
            })
        
        return {"resources": resources}
    
    async def read_resource(self, uri):
        """Read database table content"""
        # Parse URI: db://local/table_name
        if not uri.startswith("db://local/"):
            raise ValueError(f"Invalid database URI: {uri}")
        
        table_name = uri.split('/')[-1]
        
        # Query table
        cursor = self.db.execute(f"SELECT * FROM {table_name} LIMIT 100")
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        
        # Convert to JSON
        data = [
            dict(zip(columns, row))
            for row in rows
        ]
        
        import json
        content = json.dumps(data, indent=2)
        
        return {
            "contents": [{
                "uri": uri,
                "mimeType": "application/json",
                "text": content
            }]
        }
```

### API Resources

```python
class APIResourceServer(MCPServer):
    def __init__(self, base_url, api_key):
        super().__init__("api")
        self.base_url = base_url
        self.api_key = api_key
    
    async def list_resources(self):
        """List available API endpoints"""
        return {
            "resources": [
                {
                    "uri": "api://weather/forecast",
                    "name": "Weather Forecast",
                    "description": "7-day weather forecast",
                    "mimeType": "application/json"
                },
                {
                    "uri": "api://weather/current",
                    "name": "Current Weather",
                    "description": "Current weather conditions",
                    "mimeType": "application/json"
                }
            ]
        }
    
    async def read_resource(self, uri):
        """Fetch data from API"""
        import aiohttp
        
        # Map URI to API endpoint
        if uri == "api://weather/forecast":
            endpoint = f"{self.base_url}/forecast"
        elif uri == "api://weather/current":
            endpoint = f"{self.base_url}/current"
        else:
            raise ValueError(f"Unknown API resource: {uri}")
        
        # Make API request
        async with aiohttp.ClientSession() as session:
            async with session.get(
                endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"}
            ) as response:
                data = await response.text()
        
        return {
            "contents": [{
                "uri": uri,
                "mimeType": "application/json",
                "text": data
            }]
        }
```

### Computed Resources

```python
class ComputedResourceServer(MCPServer):
    """Resources that are computed on demand"""
    
    async def list_resources(self):
        return {
            "resources": [
                {
                    "uri": "computed://system/status",
                    "name": "System Status",
                    "description": "Current system metrics",
                    "mimeType": "application/json"
                },
                {
                    "uri": "computed://stats/summary",
                    "name": "Statistics Summary",
                    "description": "Aggregated statistics",
                    "mimeType": "application/json"
                }
            ]
        }
    
    async def read_resource(self, uri):
        if uri == "computed://system/status":
            # Compute system status
            import psutil
            
            status = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "timestamp": time.time()
            }
            
            import json
            content = json.dumps(status, indent=2)
        
        elif uri == "computed://stats/summary":
            # Compute statistics from database
            cursor = self.db.execute(
                "SELECT COUNT(*), AVG(value), MAX(value) FROM metrics"
            )
            count, avg, max_val = cursor.fetchone()
            
            summary = {
                "count": count,
                "average": avg,
                "maximum": max_val
            }
            
            import json
            content = json.dumps(summary, indent=2)
        
        else:
            raise ValueError(f"Unknown computed resource: {uri}")
        
        return {
            "contents": [{
                "uri": uri,
                "mimeType": "application/json",
                "text": content
            }]
        }
```

## Resource Subscriptions

Subscriptions allow clients to be notified when resources change.

### Server Capability

```python
{
    "capabilities": {
        "resources": {
            "subscribe": true,        # Server supports subscriptions
            "listChanged": true       # Server notifies when list changes
        }
    }
}
```

### Subscribe Request

```python
# Client subscribes to a resource
request = {
    "jsonrpc": "2.0",
    "id": 3,
    "method": "resources/subscribe",
    "params": {
        "uri": "file:///data/users.json"
    }
}

# Server acknowledges
response = {
    "jsonrpc": "2.0",
    "id": 3,
    "result": {}
}
```

### Update Notification

```python
# When resource changes, server sends notification
notification = {
    "jsonrpc": "2.0",
    "method": "notifications/resources/updated",
    "params": {
        "uri": "file:///data/users.json"
    }
}
```

### Server-Side Implementation

```python
class SubscribableResourceServer(MCPServer):
    def __init__(self):
        super().__init__("subscribable")
        self.subscriptions = defaultdict(set)  # uri -> set of client_ids
        self.watchers = {}  # uri -> file watcher
    
    async def handle_subscribe(self, client_id, uri):
        """Handle subscription request"""
        # Add client to subscribers
        self.subscriptions[uri].add(client_id)
        
        # Set up file watcher if not already watching
        if uri not in self.watchers:
            self.watchers[uri] = self.watch_resource(uri)
        
        return {}
    
    async def handle_unsubscribe(self, client_id, uri):
        """Handle unsubscription request"""
        self.subscriptions[uri].discard(client_id)
        
        # Stop watching if no more subscribers
        if not self.subscriptions[uri]:
            self.watchers[uri].stop()
            del self.watchers[uri]
        
        return {}
    
    def watch_resource(self, uri):
        """Set up file system watcher"""
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        
        class ChangeHandler(FileSystemEventHandler):
            def __init__(self, server, uri):
                self.server = server
                self.uri = uri
            
            def on_modified(self, event):
                if not event.is_directory:
                    asyncio.create_task(
                        self.server.notify_resource_updated(self.uri)
                    )
        
        # Extract file path from URI
        file_path = Path(uri[7:])  # Remove "file://"
        
        # Set up watcher
        observer = Observer()
        observer.schedule(
            ChangeHandler(self, uri),
            str(file_path.parent),
            recursive=False
        )
        observer.start()
        
        return observer
    
    async def notify_resource_updated(self, uri):
        """Notify all subscribers of resource update"""
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/resources/updated",
            "params": {"uri": uri}
        }
        
        # Send to all subscribers
        for client_id in self.subscriptions[uri]:
            await self.send_notification(client_id, notification)
```

### Client-Side Implementation

```python
class SubscribingResourceClient:
    def __init__(self):
        self.mcp = MCPClient()
        self.resource_callbacks = {}
    
    async def subscribe(self, uri, callback):
        """Subscribe to resource updates"""
        # Send subscription request
        await self.mcp.send_request(
            "resources/subscribe",
            {"uri": uri}
        )
        
        # Register callback
        self.resource_callbacks[uri] = callback
        
        # Set up notification handler
        self.mcp.on_notification(
            "notifications/resources/updated",
            self.handle_resource_updated
        )
    
    async def handle_resource_updated(self, notification):
        """Handle resource update notification"""
        uri = notification['params']['uri']
        
        if uri in self.resource_callbacks:
            # Re-read resource
            response = await self.mcp.send_request(
                "resources/read",
                {"uri": uri}
            )
            
            content = response['contents'][0]['text']
            
            # Call callback
            await self.resource_callbacks[uri](uri, content)
    
    async def unsubscribe(self, uri):
        """Unsubscribe from resource updates"""
        await self.mcp.send_request(
            "resources/unsubscribe",
            {"uri": uri}
        )
        
        del self.resource_callbacks[uri]

# Usage
client = SubscribingResourceClient()

async def handle_user_data_updated(uri, content):
    print(f"User data updated: {content[:100]}...")
    # Re-process data, update UI, etc.

await client.subscribe(
    "file:///data/users.json",
    handle_user_data_updated
)
```

## Resource Templates

Resources can be parameterized using URI templates.

### URI Templates

```python
# Template with parameters
"db://localhost/users?status={status}&limit={limit}"
"api://weather/forecast?city={city}&days={days}"
"file:///logs/{date}/{level}.log"
```

### Server Implementation

```python
class TemplatedResourceServer(MCPServer):
    async def list_resources(self):
        """List resource templates"""
        return {
            "resources": [
                {
                    "uri": "db://localhost/users?status={status}",
                    "name": "Filtered Users",
                    "description": "Users filtered by status (active/inactive)",
                    "mimeType": "application/json"
                },
                {
                    "uri": "api://weather?city={city}&days={days}",
                    "name": "Weather Forecast",
                    "description": "Weather forecast for specific city",
                    "mimeType": "application/json"
                }
            ]
        }
    
    async def read_resource(self, uri):
        """Read resource with template parameters"""
        from urllib.parse import urlparse, parse_qs
        
        parsed = urlparse(uri)
        params = parse_qs(parsed.query)
        
        if parsed.path == "/users":
            # Extract status parameter
            status = params.get('status', ['active'])[0]
            
            # Query with parameter
            cursor = self.db.execute(
                "SELECT * FROM users WHERE status = ?",
                (status,)
            )
            results = cursor.fetchall()
            
            # Format as JSON
            import json
            content = json.dumps([dict(row) for row in results], indent=2)
            
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": content
                }]
            }
```

### Client Usage

```python
# Use template with specific parameters
active_users = await client.read_resource(
    "db://localhost/users?status=active"
)

inactive_users = await client.read_resource(
    "db://localhost/users?status=inactive"
)

# Weather for different cities
tokyo_weather = await client.read_resource(
    "api://weather?city=tokyo&days=7"
)

london_weather = await client.read_resource(
    "api://weather?city=london&days=7"
)
```

## Common Resource Patterns

### Pattern 1: File System Navigation

```python
class NavigableFileSystem(MCPServer):
    """File system that supports directory navigation"""
    
    async def list_resources(self):
        """List top-level directories"""
        resources = []
        
        for item in self.root_dir.iterdir():
            if item.is_dir():
                uri = f"file://{item.absolute()}/"
                resources.append({
                    "uri": uri,
                    "name": item.name,
                    "description": f"Directory: {item.name}",
                    "mimeType": "inode/directory"
                })
            else:
                uri = f"file://{item.absolute()}"
                resources.append({
                    "uri": uri,
                    "name": item.name,
                    "description": f"File: {item.name}",
                    "mimeType": self.get_mime_type(item)
                })
        
        return {"resources": resources}
    
    async def read_resource(self, uri):
        """Read file or list directory contents"""
        path = Path(uri[7:])
        
        if path.is_dir():
            # List directory contents
            contents = []
            for item in path.iterdir():
                contents.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "uri": f"file://{item.absolute()}"
                })
            
            import json
            text = json.dumps(contents, indent=2)
            
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": text
                }]
            }
        else:
            # Read file content
            with open(path, 'r') as f:
                text = f.read()
            
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": self.get_mime_type(path),
                    "text": text
                }]
            }
```

### Pattern 2: Database Query Interface

```python
class QueryableDatabase(MCPServer):
    """Database with query-based resources"""
    
    async def list_resources(self):
        """List predefined queries"""
        return {
            "resources": [
                {
                    "uri": "query://active_users",
                    "name": "Active Users",
                    "description": "All currently active users",
                    "mimeType": "application/json"
                },
                {
                    "uri": "query://recent_orders",
                    "name": "Recent Orders",
                    "description": "Orders from last 24 hours",
                    "mimeType": "application/json"
                },
                {
                    "uri": "query://top_products",
                    "name": "Top Products",
                    "description": "Best selling products",
                    "mimeType": "application/json"
                }
            ]
        }
    
    async def read_resource(self, uri):
        """Execute predefined query"""
        queries = {
            "query://active_users": 
                "SELECT * FROM users WHERE active = 1",
            "query://recent_orders": 
                "SELECT * FROM orders WHERE created_at > datetime('now', '-1 day')",
            "query://top_products": 
                "SELECT product_id, COUNT(*) as count FROM orders "
                "GROUP BY product_id ORDER BY count DESC LIMIT 10"
        }
        
        if uri not in queries:
            raise ValueError(f"Unknown query: {uri}")
        
        # Execute query
        cursor = self.db.execute(queries[uri])
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        # Format results
        results = [dict(zip(columns, row)) for row in rows]
        
        import json
        text = json.dumps(results, indent=2)
        
        return {
            "contents": [{
                "uri": uri,
                "mimeType": "application/json",
                "text": text
            }]
        }
```

### Pattern 3: API Aggregation

```python
class AggregatingAPIServer(MCPServer):
    """Aggregates data from multiple APIs"""
    
    async def read_resource(self, uri):
        """Fetch and combine data from multiple sources"""
        if uri == "api://dashboard/summary":
            # Fetch from multiple APIs in parallel
            results = await asyncio.gather(
                self.fetch_api("users/count"),
                self.fetch_api("orders/count"),
                self.fetch_api("revenue/total"),
                return_exceptions=True
            )
            
            summary = {
                "users": results[0] if not isinstance(results[0], Exception) else None,
                "orders": results[1] if not isinstance(results[1], Exception) else None,
                "revenue": results[2] if not isinstance(results[2], Exception) else None,
                "timestamp": time.time()
            }
            
            import json
            text = json.dumps(summary, indent=2)
            
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": text
                }]
            }
```

### Pattern 4: Virtual Resources

```python
class VirtualResourceServer(MCPServer):
    """Resources that don't exist as files but are computed"""
    
    async def list_resources(self):
        return {
            "resources": [
                {
                    "uri": "virtual://config/merged",
                    "name": "Merged Configuration",
                    "description": "Configuration from all sources merged",
                    "mimeType": "application/json"
                },
                {
                    "uri": "virtual://search/index",
                    "name": "Search Index",
                    "description": "Searchable index of all content",
                    "mimeType": "application/json"
                }
            ]
        }
    
    async def read_resource(self, uri):
        if uri == "virtual://config/merged":
            # Merge configurations from multiple sources
            config = {}
            
            # Load from files
            config.update(self.load_json("config/default.json"))
            config.update(self.load_json("config/local.json"))
            
            # Load from environment
            config.update(self.load_from_env())
            
            # Load from database
            config.update(self.load_from_db())
            
            import json
            text = json.dumps(config, indent=2)
            
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": text
                }]
            }
```

## Resource Providers

### Provider Interface

```python
class ResourceProvider:
    """Abstract interface for resource providers"""
    
    async def list(self) -> List[Dict]:
        """List available resources"""
        raise NotImplementedError
    
    async def read(self, uri: str) -> str:
        """Read resource content"""
        raise NotImplementedError
    
    async def watch(self, uri: str, callback):
        """Watch for changes (optional)"""
        raise NotImplementedError
```

### Example Providers

**File Provider**:
```python
class FileResourceProvider(ResourceProvider):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
    
    async def list(self):
        resources = []
        for file_path in self.root_dir.rglob('*'):
            if file_path.is_file():
                resources.append({
                    "uri": f"file://{file_path.absolute()}",
                    "name": file_path.name,
                    "mimeType": self.get_mime_type(file_path)
                })
        return resources
    
    async def read(self, uri):
        path = Path(uri[7:])
        with open(path, 'r') as f:
            return f.read()
```

**Git Provider**:
```python
class GitResourceProvider(ResourceProvider):
    def __init__(self, repo_path):
        import git
        self.repo = git.Repo(repo_path)
    
    async def list(self):
        resources = []
        
        # List all files in HEAD
        for item in self.repo.head.commit.tree.traverse():
            if item.type == 'blob':  # File
                resources.append({
                    "uri": f"git://{self.repo.working_dir}/{item.path}",
                    "name": item.path,
                    "mimeType": "text/plain"
                })
        
        return resources
    
    async def read(self, uri):
        # Extract file path from URI
        file_path = uri.split('/')[-1]
        
        # Get file content from HEAD
        blob = self.repo.head.commit.tree / file_path
        return blob.data_stream.read().decode('utf-8')
```

**S3 Provider**:
```python
class S3ResourceProvider(ResourceProvider):
    def __init__(self, bucket_name):
        import boto3
        self.s3 = boto3.client('s3')
        self.bucket = bucket_name
    
    async def list(self):
        resources = []
        
        # List objects in bucket
        response = self.s3.list_objects_v2(Bucket=self.bucket)
        
        for obj in response.get('Contents', []):
            resources.append({
                "uri": f"s3://{self.bucket}/{obj['Key']}",
                "name": obj['Key'],
                "mimeType": self.guess_mime_type(obj['Key'])
            })
        
        return resources
    
    async def read(self, uri):
        # Extract key from URI
        key = uri.split('/', 3)[3]
        
        # Download object
        response = self.s3.get_object(Bucket=self.bucket, Key=key)
        return response['Body'].read().decode('utf-8')
```

## Resource Caching

### Client-Side Caching

```python
class CachingResourceClient:
    def __init__(self, mcp_client):
        self.mcp = mcp_client
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def read_resource(self, uri):
        """Read with caching"""
        now = time.time()
        
        # Check cache
        if uri in self.cache:
            content, timestamp = self.cache[uri]
            if now - timestamp < self.cache_ttl:
                return content
        
        # Fetch from server
        response = await self.mcp.send_request(
            "resources/read",
            {"uri": uri}
        )
        
        content = response['contents'][0]['text']
        
        # Update cache
        self.cache[uri] = (content, now)
        
        return content
    
    def invalidate(self, uri):
        """Manually invalidate cache"""
        self.cache.pop(uri, None)
    
    def clear_cache(self):
        """Clear entire cache"""
        self.cache.clear()
```

### Subscription-Based Cache Invalidation

```python
class SmartCachingClient:
    """Caching with automatic invalidation via subscriptions"""
    
    async def read_resource_cached(self, uri):
        """Read resource with smart caching"""
        
        # Subscribe to updates if not already subscribed
        if uri not in self.subscriptions:
            await self.subscribe_with_invalidation(uri)
        
        # Check cache
        if uri in self.cache and self.cache[uri]['valid']:
            return self.cache[uri]['content']
        
        # Fetch and cache
        content = await self.read_resource(uri)
        self.cache[uri] = {
            'content': content,
            'valid': True,
            'timestamp': time.time()
        }
        
        return content
    
    async def subscribe_with_invalidation(self, uri):
        """Subscribe and invalidate cache on updates"""
        await self.mcp.send_request(
            "resources/subscribe",
            {"uri": uri}
        )
        
        self.subscriptions.add(uri)
        
        # Register invalidation callback
        async def invalidate_callback(notification):
            if notification['params']['uri'] == uri:
                if uri in self.cache:
                    self.cache[uri]['valid'] = False
        
        self.mcp.on_notification(
            "notifications/resources/updated",
            invalidate_callback
        )
```

## Resource Security

### Access Control

```python
class SecureResourceServer(MCPServer):
    def __init__(self):
        super().__init__("secure")
        self.permissions = self.load_permissions()
    
    async def read_resource(self, client_id, uri):
        """Read resource with permission check"""
        
        # Check if client has permission
        if not self.has_permission(client_id, uri, 'read'):
            raise PermissionDenied(
                f"Client {client_id} cannot read {uri}"
            )
        
        # Proceed with read
        return await super().read_resource(uri)
    
    def has_permission(self, client_id, uri, action):
        """Check if client has permission for action on resource"""
        client_perms = self.permissions.get(client_id, {})
        
        # Check exact URI permission
        if uri in client_perms.get(action, []):
            return True
        
        # Check wildcard patterns
        import fnmatch
        for pattern in client_perms.get(action, []):
            if fnmatch.fnmatch(uri, pattern):
                return True
        
        return False
    
    def load_permissions(self):
        """Load permissions from configuration"""
        return {
            "client_1": {
                "read": [
                    "file:///public/*",
                    "db://localhost/public_data"
                ]
            },
            "client_2": {
                "read": [
                    "file:///public/*",
                    "file:///private/*",
                    "db://localhost/*"
                ]
            }
        }
```

### Path Traversal Protection

```python
def is_safe_path(requested_path, base_path):
    """Prevent path traversal attacks"""
    # Resolve to absolute paths
    base = Path(base_path).resolve()
    requested = Path(requested_path).resolve()
    
    # Check if requested is under base
    try:
        requested.relative_to(base)
        return True
    except ValueError:
        return False

# Usage in server
async def read_resource(self, uri):
    path = Path(uri[7:])  # Extract path from URI
    
    if not is_safe_path(path, self.root_dir):
        raise SecurityError("Path traversal attempt detected")
    
    # Proceed with read
    ...
```

## Resource Discovery

### Programmatic Discovery

```python
class ResourceDiscovery:
    """Discover and categorize resources"""
    
    async def discover_all_resources(self, servers):
        """Discover resources from multiple servers"""
        all_resources = {}
        
        for server_id, server in servers.items():
            response = await server.send_request(
                "resources/list", {}
            )
            
            all_resources[server_id] = response['resources']
        
        return all_resources
    
    def categorize_resources(self, resources):
        """Organize resources by type"""
        categories = {
            "documents": [],
            "data": [],
            "apis": [],
            "other": []
        }
        
        for resource in resources:
            mime = resource['mimeType']
            
            if mime in ['text/plain', 'text/markdown', 'application/pdf']:
                categories['documents'].append(resource)
            elif mime in ['application/json', 'text/csv', 'application/xml']:
                categories['data'].append(resource)
            elif mime.startswith('api/'):
                categories['apis'].append(resource)
            else:
                categories['other'].append(resource)
        
        return categories
    
    async def search_resources(self, servers, query):
        """Search for resources matching query"""
        all_resources = await self.discover_all_resources(servers)
        
        matching = []
        
        for server_id, resources in all_resources.items():
            for resource in resources:
                # Search in name and description
                if (query.lower() in resource['name'].lower() or
                    query.lower() in resource.get('description', '').lower()):
                    matching.append({
                        **resource,
                        'server_id': server_id
                    })
        
        return matching
```

## Advanced Resource Techniques

### Resource Streaming

For large resources, implement streaming:

```python
class StreamingResourceServer(MCPServer):
    """Server that supports streaming large resources"""
    
    async def read_resource(self, uri):
        """Read resource in chunks"""
        path = Path(uri[7:])
        
        if path.stat().st_size > 10 * 1024 * 1024:  # > 10 MB
            # Stream large file
            async def stream_chunks():
                with open(path, 'r') as f:
                    while True:
                        chunk = f.read(4096)
                        if not chunk:
                            break
                        yield {
                            "type": "text",
                            "text": chunk
                        }
            
            return {
                "contents": [chunk async for chunk in stream_chunks()]
            }
        else:
            # Read small file completely
            with open(path, 'r') as f:
                content = f.read()
            
            return {
                "contents": [{
                    "type": "text",
                    "text": content
                }]
            }
```

### Resource Transformation

Transform resources on the fly:

```python
class TransformingResourceServer(MCPServer):
    """Transform resources before returning"""
    
    async def read_resource(self, uri):
        """Read and transform resource"""
        # Parse URI with transformation hint
        # Example: file:///data.json?transform=flatten
        
        from urllib.parse import urlparse, parse_qs
        
        parsed = urlparse(uri)
        params = parse_qs(parsed.query)
        transform = params.get('transform', [None])[0]
        
        # Read base resource
        base_uri = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        content = await self.read_base_resource(base_uri)
        
        # Apply transformation
        if transform == 'flatten':
            content = self.flatten_json(content)
        elif transform == 'summarize':
            content = self.summarize_text(content)
        elif transform == 'extract_links':
            content = self.extract_links(content)
        
        return {
            "contents": [{
                "uri": uri,
                "mimeType": "application/json",
                "text": content
            }]
        }
```

### Resource Composition

Combine multiple resources:

```python
class ComposingResourceServer(MCPServer):
    """Compose multiple resources into one"""
    
    async def read_resource(self, uri):
        """Compose multiple resources"""
        if uri == "composite://dashboard":
            # Read multiple resources
            user_data = await self.read_base_resource(
                "db://localhost/users"
            )
            order_data = await self.read_base_resource(
                "db://localhost/orders"
            )
            analytics = await self.read_base_resource(
                "api://analytics/summary"
            )
            
            # Combine into single resource
            composite = {
                "users": json.loads(user_data),
                "orders": json.loads(order_data),
                "analytics": json.loads(analytics),
                "generated_at": time.time()
            }
            
            import json
            text = json.dumps(composite, indent=2)
            
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": text
                }]
            }
```

## Summary

Resources are MCP's mechanism for exposing data to agents:

**Key Concepts**:
- **Resources are read-only data sources**
- **URIs uniquely identify resources**
- **Support text and binary content**
- **Can be static or dynamic**
- **Subscriptions enable real-time updates**

**Resource Types**:
- **Files** - Documents, logs, configuration
- **Databases** - Tables, queries, results
- **APIs** - REST endpoints, web services
- **Computed** - Generated on demand
- **Virtual** - Aggregated or transformed

**Best Practices**:
- Use clear, stable URIs
- Provide good descriptions
- Implement subscriptions for changing data
- Cache appropriately
- Secure sensitive resources
- Handle errors gracefully

**Advanced Techniques**:
- Streaming for large resources
- On-the-fly transformation
- Resource composition
- Template-based access

Resources make heterogeneous data sources uniformly accessible to agents, enabling them to work with any data through a standard interface.

## Next Steps

- **[Tools and Tool Registration](mcp-tools.md)** - Learn about executing actions
- **[Prompts and Templates](mcp-prompts.md)** - Explore reusable prompts
- **[Context Management](mcp-context.md)** - Manage LLM context with resources
- **[Building MCP Servers](building-mcp-servers.md)** - Implement resource providers
- **[MCP Architecture](mcp-architecture.md)** - Understand protocol internals
