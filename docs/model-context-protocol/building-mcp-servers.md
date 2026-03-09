# Building MCP Servers

## Table of Contents

- [Introduction](#introduction)
- [Server Basics](#server-basics)
- [Setting Up a Server](#setting-up-a-server)
- [Implementing the Protocol](#implementing-the-protocol)
- [Exposing Resources](#exposing-resources)
- [Registering Tools](#registering-tools)
- [Providing Prompts](#providing-prompts)
- [Transport Implementation](#transport-implementation)
- [Error Handling](#error-handling)
- [Server Lifecycle](#server-lifecycle)
- [Testing Your Server](#testing-your-server)
- [Deployment Patterns](#deployment-patterns)
- [Performance Optimization](#performance-optimization)
- [Security Best Practices](#security-best-practices)
- [Real-World Examples](#real-world-examples)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Building an MCP server allows you to **expose your capabilities** to the MCP ecosystem. Whether you're providing:

- **Data access** (resources)
- **Executable actions** (tools)
- **Prompt templates** (prompts)

...an MCP server makes your functionality **discoverable and usable** by any MCP client.

> "An MCP server is your API to the agentic world."

This guide provides practical, step-by-step instructions for building production-ready MCP servers.

## Server Basics

### What Is an MCP Server?

An **MCP server**:
- Implements the MCP protocol (JSON-RPC 2.0)
- Exposes capabilities (resources, tools, prompts)
- Responds to client requests
- Handles transport (stdio, HTTP+SSE, WebSocket)

### Server Responsibilities

```python
# A server must:
# 1. Handle initialization
# 2. Expose capabilities
# 3. Process requests
# 4. Send notifications
# 5. Manage connections
```

### Minimal Server

```python
import asyncio
import json
from typing import Any, Dict

class MinimalMCPServer:
    """Minimal MCP server implementation"""
    
    def __init__(self, name: str):
        self.name = name
        self.running = False
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle JSON-RPC request"""
        method = request.get('method')
        params = request.get('params', {})
        request_id = request.get('id')
        
        # Route to handler
        if method == 'initialize':
            result = await self.handle_initialize(params)
        elif method == 'ping':
            result = await self.handle_ping(params)
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        }
    
    async def handle_initialize(self, params):
        """Handle initialization"""
        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": self.name,
                "version": "1.0.0"
            },
            "capabilities": {}
        }
    
    async def handle_ping(self, params):
        """Handle ping"""
        return {}
    
    async def run(self):
        """Run server (stdio transport)"""
        self.running = True
        
        while self.running:
            # Read request from stdin
            line = await asyncio.get_event_loop().run_in_executor(
                None, input
            )
            
            try:
                request = json.loads(line)
                response = await self.handle_request(request)
                
                # Write response to stdout
                print(json.dumps(response), flush=True)
            
            except json.JSONDecodeError:
                pass
            except Exception as e:
                print(json.dumps({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {str(e)}"
                    }
                }), flush=True)

# Run server
if __name__ == "__main__":
    server = MinimalMCPServer("minimal-server")
    asyncio.run(server.run())
```

## Setting Up a Server

### Project Structure

```
my-mcp-server/
├── mcp_server/
│   ├── __init__.py
│   ├── server.py        # Main server class
│   ├── resources.py     # Resource handlers
│   ├── tools.py         # Tool implementations
│   ├── prompts.py       # Prompt templates
│   └── transport.py     # Transport implementations
├── tests/
│   ├── test_server.py
│   ├── test_resources.py
│   └── test_tools.py
├── pyproject.toml
├── README.md
└── requirements.txt
```

### Dependencies

```toml
# pyproject.toml
[project]
name = "my-mcp-server"
version = "0.1.0"
dependencies = [
    "mcp>=0.1.0",        # Official MCP SDK
    "pydantic>=2.0.0",   # Data validation
    "httpx>=0.25.0",     # HTTP client (if needed)
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "mypy>=1.0.0",
]
```

### Base Server Class

```python
# server.py
from typing import Any, Callable, Dict, List, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)

class MCPServer:
    """Base MCP server implementation"""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        
        # Capability handlers
        self.resources = {}
        self.tools = {}
        self.prompts = {}
        
        # Request handlers
        self.request_handlers = {
            'initialize': self.handle_initialize,
            'ping': self.handle_ping,
            'resources/list': self.handle_list_resources,
            'resources/read': self.handle_read_resource,
            'tools/list': self.handle_list_tools,
            'tools/call': self.handle_call_tool,
            'prompts/list': self.handle_list_prompts,
            'prompts/get': self.handle_get_prompt,
        }
        
        # State
        self.initialized = False
        self.client_info = None
    
    def register_resource(self, uri: str, handler: Callable):
        """Register a resource handler"""
        self.resources[uri] = handler
        logger.info(f"Registered resource: {uri}")
    
    def register_tool(self, name: str, description: str, 
                     handler: Callable, input_schema: Dict):
        """Register a tool"""
        self.tools[name] = {
            'name': name,
            'description': description,
            'handler': handler,
            'inputSchema': input_schema
        }
        logger.info(f"Registered tool: {name}")
    
    def register_prompt(self, name: str, description: str,
                       template: Callable, arguments: List[Dict]):
        """Register a prompt"""
        self.prompts[name] = {
            'name': name,
            'description': description,
            'template': template,
            'arguments': arguments
        }
        logger.info(f"Registered prompt: {name}")
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming request"""
        method = request.get('method')
        params = request.get('params', {})
        request_id = request.get('id')
        
        try:
            # Find handler
            handler = self.request_handlers.get(method)
            
            if not handler:
                return self.error_response(
                    request_id,
                    -32601,
                    f"Method not found: {method}"
                )
            
            # Execute handler
            result = await handler(params)
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
        
        except Exception as e:
            logger.error(f"Error handling {method}: {e}", exc_info=True)
            return self.error_response(
                request_id,
                -32603,
                f"Internal error: {str(e)}"
            )
    
    async def handle_initialize(self, params):
        """Handle initialization"""
        self.client_info = params.get('clientInfo', {})
        self.initialized = True
        
        logger.info(f"Initialized for client: {self.client_info.get('name')}")
        
        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": self.name,
                "version": self.version
            },
            "capabilities": {
                "resources": {"listChanged": False} if self.resources else None,
                "tools": {} if self.tools else None,
                "prompts": {} if self.prompts else None,
            }
        }
    
    async def handle_ping(self, params):
        """Handle ping"""
        return {}
    
    async def handle_list_resources(self, params):
        """List available resources"""
        return {
            "resources": [
                {"uri": uri, "name": uri.split('/')[-1]}
                for uri in self.resources.keys()
            ]
        }
    
    async def handle_read_resource(self, params):
        """Read a resource"""
        uri = params.get('uri')
        
        if uri not in self.resources:
            raise ValueError(f"Resource not found: {uri}")
        
        handler = self.resources[uri]
        content = await handler()
        
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "text/plain",
                    "text": content
                }
            ]
        }
    
    async def handle_list_tools(self, params):
        """List available tools"""
        return {
            "tools": [
                {
                    "name": tool['name'],
                    "description": tool['description'],
                    "inputSchema": tool['inputSchema']
                }
                for tool in self.tools.values()
            ]
        }
    
    async def handle_call_tool(self, params):
        """Call a tool"""
        name = params.get('name')
        arguments = params.get('arguments', {})
        
        if name not in self.tools:
            raise ValueError(f"Tool not found: {name}")
        
        tool = self.tools[name]
        result = await tool['handler'](arguments)
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": str(result)
                }
            ]
        }
    
    async def handle_list_prompts(self, params):
        """List available prompts"""
        return {
            "prompts": [
                {
                    "name": prompt['name'],
                    "description": prompt['description'],
                    "arguments": prompt['arguments']
                }
                for prompt in self.prompts.values()
            ]
        }
    
    async def handle_get_prompt(self, params):
        """Get a prompt"""
        name = params.get('name')
        arguments = params.get('arguments', {})
        
        if name not in self.prompts:
            raise ValueError(f"Prompt not found: {name}")
        
        prompt = self.prompts[name]
        messages = await prompt['template'](arguments)
        
        return {
            "description": prompt['description'],
            "messages": messages
        }
    
    def error_response(self, request_id, code, message):
        """Create error response"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }
```

## Implementing the Protocol

### Request/Response Handling

```python
async def handle_request(self, request: Dict) -> Dict:
    """Handle request with proper protocol semantics"""
    
    # Validate JSON-RPC 2.0 format
    if request.get('jsonrpc') != '2.0':
        return self.error_response(None, -32600, "Invalid Request")
    
    # Extract fields
    method = request.get('method')
    params = request.get('params', {})
    request_id = request.get('id')
    
    # Notifications (no response expected)
    if request_id is None:
        await self.handle_notification(method, params)
        return None
    
    # Regular request
    try:
        result = await self.dispatch_method(method, params)
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        }
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": self.format_error(e)
        }
```

### Capability Negotiation

```python
class CapabilityNegotiator:
    """Negotiate capabilities during initialization"""
    
    def __init__(self, server):
        self.server = server
    
    async def negotiate(self, client_capabilities):
        """Negotiate capabilities with client"""
        
        server_capabilities = {}
        
        # Resources capability
        if self.server.resources:
            server_capabilities['resources'] = {
                "subscribe": False,  # We don't support subscriptions
                "listChanged": False  # We don't notify of list changes
            }
        
        # Tools capability
        if self.server.tools:
            server_capabilities['tools'] = {}
        
        # Prompts capability
        if self.server.prompts:
            server_capabilities['prompts'] = {
                "listChanged": False
            }
        
        # Check what client supports
        client_supports_notifications = (
            client_capabilities.get('experimental', {})
            .get('notifications', False)
        )
        
        if client_supports_notifications:
            server_capabilities['notifications'] = {}
        
        return server_capabilities
```

### Protocol Extensions

```python
class ExtendedMCPServer(MCPServer):
    """Server with protocol extensions"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add custom methods
        self.request_handlers.update({
            'custom/method': self.handle_custom_method,
            'batch/process': self.handle_batch_process,
        })
    
    async def handle_custom_method(self, params):
        """Custom protocol extension"""
        # Implement custom functionality
        return {"status": "ok"}
    
    async def handle_batch_process(self, params):
        """Batch processing extension"""
        operations = params.get('operations', [])
        
        results = await asyncio.gather(*[
            self.process_operation(op)
            for op in operations
        ])
        
        return {"results": results}
```

## Exposing Resources

### Simple File Resource

```python
# resources.py
import os
from pathlib import Path

class FileResourceProvider:
    """Expose files as MCP resources"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
    
    def register_with_server(self, server: MCPServer):
        """Register all files as resources"""
        
        for file_path in self.base_path.rglob('*'):
            if file_path.is_file():
                # Create URI
                relative = file_path.relative_to(self.base_path)
                uri = f"file:///{relative.as_posix()}"
                
                # Register handler
                server.register_resource(
                    uri,
                    self.create_handler(file_path)
                )
    
    def create_handler(self, file_path: Path):
        """Create handler for specific file"""
        async def handler():
            with open(file_path, 'r') as f:
                return f.read()
        
        return handler

# Usage
file_provider = FileResourceProvider("/path/to/docs")
file_provider.register_with_server(server)
```

### Database Resource

```python
import asyncpg

class DatabaseResourceProvider:
    """Expose database queries as resources"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
    
    async def initialize(self):
        """Initialize database connection pool"""
        self.pool = await asyncpg.create_pool(self.connection_string)
    
    def register_with_server(self, server: MCPServer):
        """Register database resources"""
        
        # Register queries as resources
        server.register_resource(
            "db://users/list",
            self.list_users_handler
        )
        
        server.register_resource(
            "db://products/inventory",
            self.inventory_handler
        )
    
    async def list_users_handler(self):
        """Handle users list query"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM users")
            
            # Format as JSON
            import json
            return json.dumps([dict(row) for row in rows], indent=2)
    
    async def inventory_handler(self):
        """Handle inventory query"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT product_id, name, quantity FROM inventory"
            )
            
            import json
            return json.dumps([dict(row) for row in rows], indent=2)

# Usage
db_provider = DatabaseResourceProvider("postgresql://...")
await db_provider.initialize()
db_provider.register_with_server(server)
```

### API Resource

```python
import httpx

class APIResourceProvider:
    """Expose API endpoints as resources"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.AsyncClient()
    
    def register_with_server(self, server: MCPServer):
        """Register API resources"""
        
        server.register_resource(
            "api://weather/current",
            self.current_weather_handler
        )
        
        server.register_resource(
            "api://weather/forecast",
            self.forecast_handler
        )
    
    async def current_weather_handler(self):
        """Fetch current weather"""
        response = await self.client.get(
            f"{self.base_url}/weather/current",
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        
        response.raise_for_status()
        return response.text
    
    async def forecast_handler(self):
        """Fetch weather forecast"""
        response = await self.client.get(
            f"{self.base_url}/weather/forecast",
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        
        response.raise_for_status()
        return response.text

# Usage
api_provider = APIResourceProvider("https://api.example.com", "key123")
api_provider.register_with_server(server)
```

## Registering Tools

### Simple Calculator Tool

```python
# tools.py

def register_calculator_tools(server: MCPServer):
    """Register calculator tools"""
    
    # Addition tool
    server.register_tool(
        name="add",
        description="Add two numbers",
        handler=add_handler,
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            },
            "required": ["a", "b"]
        }
    )
    
    # Multiplication tool
    server.register_tool(
        name="multiply",
        description="Multiply two numbers",
        handler=multiply_handler,
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        }
    )

async def add_handler(arguments):
    """Handle addition"""
    a = arguments['a']
    b = arguments['b']
    return a + b

async def multiply_handler(arguments):
    """Handle multiplication"""
    a = arguments['a']
    b = arguments['b']
    return a * b
```

### File Operations Tool

```python
class FileOperationTools:
    """File operation tools"""
    
    def __init__(self, allowed_path: str):
        self.allowed_path = Path(allowed_path)
    
    def register_with_server(self, server: MCPServer):
        """Register file tools"""
        
        server.register_tool(
            name="read_file",
            description="Read contents of a file",
            handler=self.read_file_handler,
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to file"
                    }
                },
                "required": ["path"]
            }
        )
        
        server.register_tool(
            name="write_file",
            description="Write content to a file",
            handler=self.write_file_handler,
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        )
    
    async def read_file_handler(self, arguments):
        """Read file"""
        path = self.validate_path(arguments['path'])
        
        with open(path, 'r') as f:
            return f.read()
    
    async def write_file_handler(self, arguments):
        """Write file"""
        path = self.validate_path(arguments['path'])
        content = arguments['content']
        
        with open(path, 'w') as f:
            f.write(content)
        
        return f"Wrote {len(content)} bytes to {path}"
    
    def validate_path(self, path: str) -> Path:
        """Validate path is within allowed directory"""
        full_path = (self.allowed_path / path).resolve()
        
        if not str(full_path).startswith(str(self.allowed_path)):
            raise ValueError("Path outside allowed directory")
        
        return full_path

# Usage
file_tools = FileOperationTools("/safe/directory")
file_tools.register_with_server(server)
```

### External Service Tool

```python
class EmailTool:
    """Email sending tool"""
    
    def __init__(self, smtp_config):
        self.smtp_config = smtp_config
    
    def register_with_server(self, server: MCPServer):
        """Register email tool"""
        
        server.register_tool(
            name="send_email",
            description="Send an email",
            handler=self.send_email_handler,
            input_schema={
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address"
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject"
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body"
                    }
                },
                "required": ["to", "subject", "body"]
            }
        )
    
    async def send_email_handler(self, arguments):
        """Send email via SMTP"""
        import smtplib
        from email.message import EmailMessage
        
        msg = EmailMessage()
        msg['To'] = arguments['to']
        msg['From'] = self.smtp_config['from']
        msg['Subject'] = arguments['subject']
        msg.set_content(arguments['body'])
        
        # Send via SMTP
        with smtplib.SMTP(
            self.smtp_config['host'],
            self.smtp_config['port']
        ) as smtp:
            smtp.starttls()
            smtp.login(
                self.smtp_config['username'],
                self.smtp_config['password']
            )
            smtp.send_message(msg)
        
        return f"Email sent to {arguments['to']}"

# Usage
email_tool = EmailTool({
    'host': 'smtp.example.com',
    'port': 587,
    'username': 'user',
    'password': 'pass',
    'from': 'server@example.com'
})
email_tool.register_with_server(server)
```

## Providing Prompts

### Template-Based Prompts

```python
# prompts.py
from jinja2 import Template

class PromptProvider:
    """Provide prompt templates"""
    
    def register_with_server(self, server: MCPServer):
        """Register prompts"""
        
        server.register_prompt(
            name="summarize",
            description="Summarize text",
            template=self.summarize_template,
            arguments=[
                {
                    "name": "text",
                    "description": "Text to summarize",
                    "required": True
                },
                {
                    "name": "max_length",
                    "description": "Maximum length",
                    "required": False
                }
            ]
        )
        
        server.register_prompt(
            name="analyze_code",
            description="Analyze code quality",
            template=self.code_analysis_template,
            arguments=[
                {
                    "name": "code",
                    "description": "Code to analyze",
                    "required": True
                },
                {
                    "name": "language",
                    "description": "Programming language",
                    "required": True
                }
            ]
        )
    
    async def summarize_template(self, arguments):
        """Generate summarization prompt"""
        text = arguments['text']
        max_length = arguments.get('max_length', 200)
        
        prompt_text = f"""Please summarize the following text in no more than {max_length} words:

{text}

Provide a concise summary focusing on the main points."""
        
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": prompt_text
                }
            }
        ]
    
    async def code_analysis_template(self, arguments):
        """Generate code analysis prompt"""
        code = arguments['code']
        language = arguments['language']
        
        template_str = """You are an expert {{ language }} code reviewer.

Analyze the following {{ language }} code:

```{{ language }}
{{ code }}
```

Evaluate:
1. Code quality and style
2. Potential bugs
3. Performance issues
4. Security concerns
5. Best practices

Provide specific, actionable feedback."""
        
        template = Template(template_str)
        prompt_text = template.render(code=code, language=language)
        
        return [
            {
                "role": "system",
                "content": {
                    "type": "text",
                    "text": f"You are an expert {language} code reviewer."
                }
            },
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": prompt_text
                }
            }
        ]

# Usage
prompt_provider = PromptProvider()
prompt_provider.register_with_server(server)
```

## Transport Implementation

### stdio Transport

```python
import sys
import asyncio
import json

class StdioTransport:
    """stdio-based transport"""
    
    def __init__(self, server: MCPServer):
        self.server = server
        self.running = False
    
    async def run(self):
        """Run server with stdio transport"""
        self.running = True
        
        # Read from stdin, write to stdout
        while self.running:
            try:
                # Read line
                line = await asyncio.get_event_loop().run_in_executor(
                    None,
                    sys.stdin.readline
                )
                
                if not line:
                    break
                
                # Parse request
                request = json.loads(line)
                
                # Handle request
                response = await self.server.handle_request(request)
                
                # Send response (if not notification)
                if response:
                    sys.stdout.write(json.dumps(response) + '\n')
                    sys.stdout.flush()
            
            except json.JSONDecodeError as e:
                # Invalid JSON
                error = {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32700,
                        "message": f"Parse error: {str(e)}"
                    }
                }
                sys.stdout.write(json.dumps(error) + '\n')
                sys.stdout.flush()
            
            except Exception as e:
                # Internal error
                error = {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {str(e)}"
                    }
                }
                sys.stdout.write(json.dumps(error) + '\n')
                sys.stdout.flush()

# Usage
transport = StdioTransport(server)
await transport.run()
```

### HTTP+SSE Transport

```python
from aiohttp import web
import asyncio

class HTTPSSETransport:
    """HTTP+SSE transport"""
    
    def __init__(self, server: MCPServer, host='0.0.0.0', port=8000):
        self.server = server
        self.host = host
        self.port = port
        self.app = web.Application()
        self.setup_routes()
        self.sse_connections = set()
    
    def setup_routes(self):
        """Setup HTTP routes"""
        self.app.router.add_post('/message', self.handle_message)
        self.app.router.add_get('/sse', self.handle_sse)
    
    async def handle_message(self, request):
        """Handle JSON-RPC message"""
        try:
            # Parse request
            body = await request.json()
            
            # Handle request
            response = await self.server.handle_request(body)
            
            # Return response
            if response:
                return web.json_response(response)
            else:
                return web.Response(status=204)  # No content for notifications
        
        except Exception as e:
            return web.json_response({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }, status=500)
    
    async def handle_sse(self, request):
        """Handle SSE connection for server-to-client messages"""
        response = web.StreamResponse()
        response.headers['Content-Type'] = 'text/event-stream'
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['Connection'] = 'keep-alive'
        
        await response.prepare(request)
        
        # Add to active connections
        self.sse_connections.add(response)
        
        try:
            # Keep connection alive
            while True:
                await asyncio.sleep(30)
                # Send keep-alive
                await response.write(b': keep-alive\n\n')
        
        except asyncio.CancelledError:
            pass
        
        finally:
            self.sse_connections.discard(response)
        
        return response
    
    async def send_notification(self, notification):
        """Send notification to all SSE clients"""
        data = json.dumps(notification)
        message = f"data: {data}\n\n"
        
        for conn in self.sse_connections:
            try:
                await conn.write(message.encode())
            except:
                self.sse_connections.discard(conn)
    
    async def run(self):
        """Run HTTP server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        print(f"Server running on http://{self.host}:{self.port}")
        
        # Keep running
        await asyncio.Event().wait()

# Usage
transport = HTTPSSETransport(server, port=8000)
await transport.run()
```

## Error Handling

### Comprehensive Error Handling

```python
class ErrorHandler:
    """Centralized error handling"""
    
    ERROR_CODES = {
        'PARSE_ERROR': -32700,
        'INVALID_REQUEST': -32600,
        'METHOD_NOT_FOUND': -32601,
        'INVALID_PARAMS': -32602,
        'INTERNAL_ERROR': -32603,
        'SERVER_ERROR': -32000,  # Application-specific
    }
    
    @staticmethod
    def format_error(exception: Exception, request_id=None):
        """Format exception as JSON-RPC error"""
        
        if isinstance(exception, ValidationError):
            code = ErrorHandler.ERROR_CODES['INVALID_PARAMS']
            message = str(exception)
        
        elif isinstance(exception, NotFoundError):
            code = ErrorHandler.ERROR_CODES['METHOD_NOT_FOUND']
            message = str(exception)
        
        elif isinstance(exception, ValueError):
            code = ErrorHandler.ERROR_CODES['INVALID_PARAMS']
            message = str(exception)
        
        else:
            code = ErrorHandler.ERROR_CODES['INTERNAL_ERROR']
            message = f"Internal error: {str(exception)}"
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message,
                "data": {
                    "type": type(exception).__name__
                }
            }
        }

# Usage in server
try:
    result = await handler(params)
except Exception as e:
    return ErrorHandler.format_error(e, request_id)
```

### Input Validation

```python
from pydantic import BaseModel, ValidationError

class ToolInput(BaseModel):
    """Validate tool inputs"""
    name: str
    arguments: dict

class InputValidator:
    """Validate inputs against schemas"""
    
    @staticmethod
    def validate_tool_input(tool_def, arguments):
        """Validate tool arguments against schema"""
        schema = tool_def['inputSchema']
        
        # Check required fields
        required = schema.get('required', [])
        for field in required:
            if field not in arguments:
                raise ValueError(f"Missing required field: {field}")
        
        # Check types
        properties = schema.get('properties', {})
        for field, value in arguments.items():
            if field not in properties:
                raise ValueError(f"Unknown field: {field}")
            
            expected_type = properties[field].get('type')
            actual_type = type(value).__name__
            
            type_mapping = {
                'string': 'str',
                'number': ('int', 'float'),
                'boolean': 'bool',
                'object': 'dict',
                'array': 'list'
            }
            
            expected = type_mapping.get(expected_type, expected_type)
            if isinstance(expected, tuple):
                if actual_type not in expected:
                    raise ValueError(
                        f"Field {field}: expected {expected}, got {actual_type}"
                    )
            elif actual_type != expected:
                raise ValueError(
                    f"Field {field}: expected {expected}, got {actual_type}"
                )
```

## Server Lifecycle

### Initialization and Shutdown

```python
class ManagedMCPServer(MCPServer):
    """Server with lifecycle management"""
    
    async def startup(self):
        """Server startup"""
        logger.info("Server starting up...")
        
        # Initialize connections
        await self.init_database()
        await self.init_external_services()
        
        # Register capabilities
        self.register_all_capabilities()
        
        logger.info("Server ready")
    
    async def shutdown(self):
        """Server shutdown"""
        logger.info("Server shutting down...")
        
        # Close connections
        await self.close_database()
        await self.close_external_services()
        
        logger.info("Server stopped")
    
    async def init_database(self):
        """Initialize database connection"""
        if hasattr(self, 'db_provider'):
            await self.db_provider.initialize()
    
    async def close_database(self):
        """Close database connection"""
        if hasattr(self, 'db_provider'):
            await self.db_provider.close()
    
    async def register_all_capabilities(self):
        """Register all capabilities"""
        # Register resources
        if hasattr(self, 'resource_providers'):
            for provider in self.resource_providers:
                provider.register_with_server(self)
        
        # Register tools
        if hasattr(self, 'tool_providers'):
            for provider in self.tool_providers:
                provider.register_with_server(self)
        
        # Register prompts
        if hasattr(self, 'prompt_providers'):
            for provider in self.prompt_providers:
                provider.register_with_server(self)

# Usage
server = ManagedMCPServer("my-server")

try:
    await server.startup()
    await server.run()
finally:
    await server.shutdown()
```

## Testing Your Server

### Unit Tests

```python
# tests/test_server.py
import pytest
from my_server import MCPServer

@pytest.fixture
def server():
    """Create test server"""
    return MCPServer("test-server")

@pytest.mark.asyncio
async def test_initialization(server):
    """Test server initialization"""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }
    
    response = await server.handle_request(request)
    
    assert response['jsonrpc'] == '2.0'
    assert response['id'] == 1
    assert 'result' in response
    assert response['result']['serverInfo']['name'] == 'test-server'

@pytest.mark.asyncio
async def test_tool_registration(server):
    """Test tool registration and invocation"""
    
    # Register tool
    async def add_handler(args):
        return args['a'] + args['b']
    
    server.register_tool(
        "add",
        "Add two numbers",
        add_handler,
        {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        }
    )
    
    # Call tool
    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "add",
            "arguments": {"a": 5, "b": 3}
        }
    }
    
    response = await server.handle_request(request)
    
    assert response['result']['content'][0]['text'] == '8'
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_full_workflow():
    """Test complete client-server interaction"""
    
    # Start server
    server = MCPServer("test-server")
    
    # Initialize
    init_response = await server.handle_request({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    })
    
    assert 'result' in init_response
    
    # List resources
    list_response = await server.handle_request({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "resources/list",
        "params": {}
    })
    
    assert 'result' in list_response
    
    # Read resource
    if list_response['result']['resources']:
        uri = list_response['result']['resources'][0]['uri']
        
        read_response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/read",
            "params": {"uri": uri}
        })
        
        assert 'result' in read_response
```

## Deployment Patterns

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY mcp_server/ ./mcp_server/

# Expose port (for HTTP transport)
EXPOSE 8000

# Run server
CMD ["python", "-m", "mcp_server"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  mcp-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DB_HOST=postgres
      - DB_NAME=mydb
    depends_on:
      - postgres
  
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=mydb
      - POSTGRES_PASSWORD=secret
```

### Systemd Service

```ini
# /etc/systemd/system/mcp-server.service
[Unit]
Description=MCP Server
After=network.target

[Service]
Type=simple
User=mcp
WorkingDirectory=/opt/mcp-server
ExecStart=/usr/bin/python3 -m mcp_server
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Performance Optimization

### Connection Pooling

```python
class OptimizedServer(MCPServer):
    """Server with connection pooling"""
    
    async def startup(self):
        """Initialize connection pools"""
        
        # Database pool
        self.db_pool = await asyncpg.create_pool(
            host='localhost',
            database='mydb',
            min_size=5,
            max_size=20
        )
        
        # HTTP client pool
        self.http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=20)
        )
```

### Caching

```python
from functools import lru_cache
from datetime import datetime, timedelta

class CachedResourceProvider:
    """Resource provider with caching"""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = timedelta(minutes=5)
    
    async def get_resource(self, uri):
        """Get resource with caching"""
        
        # Check cache
        if uri in self.cache:
            cached, timestamp = self.cache[uri]
            if datetime.now() - timestamp < self.cache_ttl:
                return cached
        
        # Fetch fresh data
        data = await self.fetch_resource(uri)
        
        # Cache it
        self.cache[uri] = (data, datetime.now())
        
        return data
```

## Security Best Practices

### Authentication

```python
class AuthenticatedServer(MCPServer):
    """Server with authentication"""
    
    def __init__(self, *args, api_keys=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid_api_keys = api_keys or set()
    
    async def handle_request(self, request, api_key=None):
        """Handle authenticated request"""
        
        # Verify API key
        if not self.verify_api_key(api_key):
            return self.error_response(
                request.get('id'),
                -32001,
                "Unauthorized"
            )
        
        # Process request
        return await super().handle_request(request)
    
    def verify_api_key(self, api_key):
        """Verify API key"""
        return api_key in self.valid_api_keys
```

### Input Sanitization

```python
class SecureFileTools:
    """File tools with security"""
    
    def __init__(self, allowed_path):
        self.allowed_path = Path(allowed_path).resolve()
    
    def validate_path(self, path):
        """Validate path for security"""
        
        # Resolve path
        full_path = (self.allowed_path / path).resolve()
        
        # Check if within allowed directory
        if not str(full_path).startswith(str(self.allowed_path)):
            raise ValueError("Path traversal detected")
        
        # Check for dangerous patterns
        if '..' in path or path.startswith('/'):
            raise ValueError("Invalid path")
        
        return full_path
```

## Real-World Examples

### Complete File Server

```python
# examples/file_server.py
import asyncio
from pathlib import Path
from mcp_server import MCPServer, StdioTransport

class FileServer(MCPServer):
    """MCP server for file operations"""
    
    def __init__(self, root_dir):
        super().__init__("file-server", "1.0.0")
        self.root = Path(root_dir).resolve()
        self.register_capabilities()
    
    def register_capabilities(self):
        """Register all capabilities"""
        
        # Resources: All files
        for file_path in self.root.rglob('*.txt'):
            uri = f"file:///{file_path.relative_to(self.root)}"
            self.register_resource(uri, self.create_file_handler(file_path))
        
        # Tools: File operations
        self.register_tool(
            "search_files",
            "Search files by pattern",
            self.search_files,
            {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"}
                },
                "required": ["pattern"]
            }
        )
    
    def create_file_handler(self, file_path):
        async def handler():
            with open(file_path) as f:
                return f.read()
        return handler
    
    async def search_files(self, args):
        """Search for files"""
        import fnmatch
        pattern = args['pattern']
        
        matches = [
            str(p.relative_to(self.root))
            for p in self.root.rglob('*')
            if fnmatch.fnmatch(p.name, pattern)
        ]
        
        return '\n'.join(matches)

# Run server
if __name__ == "__main__":
    server = FileServer("/path/to/docs")
    transport = StdioTransport(server)
    asyncio.run(transport.run())
```

## Summary

Building MCP servers:

**Key Steps**:
1. Set up base server class
2. Implement protocol handlers
3. Register capabilities (resources, tools, prompts)
4. Choose and implement transport
5. Add error handling
6. Test thoroughly
7. Deploy

**Best Practices**:
- Clear capability documentation
- Comprehensive error handling
- Input validation
- Security first
- Performance optimization
- Thorough testing

**Deployment**:
- Docker containers
- System services
- Cloud platforms
- Process managers

Your MCP server is now ready to join the ecosystem!

## Next Steps

- **[MCP Ecosystem](mcp-ecosystem.md)** - Understand ecosystem benefits
- **[MCP Fundamentals](mcp-fundamentals.md)** - Review core concepts
- **[Resources](mcp-resources.md)**, **[Tools](mcp-tools.md)**, **[Prompts](mcp-prompts.md)** - Deep dives
- **[Context Management](mcp-context.md)** - Handle context effectively
