# MCP Architecture and Components

## Table of Contents

- [Introduction](#introduction)
- [High-Level Architecture](#high-level-architecture)
- [Client-Server Model](#client-server-model)
- [Transport Mechanisms](#transport-mechanisms)
- [Communication Protocol](#communication-protocol)
- [Capability Discovery](#capability-discovery)
- [Message Flow Patterns](#message-flow-patterns)
- [Connection Lifecycle](#connection-lifecycle)
- [Server Components](#server-components)
- [Client Components](#client-components)
- [Protocol Layers](#protocol-layers)
- [Security Architecture](#security-architecture)
- [Scalability Patterns](#scalability-patterns)
- [Error Handling Architecture](#error-handling-architecture)
- [Extensibility Mechanisms](#extensibility-mechanisms)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Understanding MCP's architecture is essential for building robust agents and servers. The protocol is designed around a few core principles:

- **Simplicity**: Easy to implement and understand
- **Flexibility**: Supports multiple transport mechanisms
- **Extensibility**: New capabilities can be added without breaking changes
- **Interoperability**: Works across languages and platforms

This guide explores MCP's architectural components, from the high-level client-server model down to the specifics of message formatting and protocol layers.

> "Good architecture makes systems easy to understand, develop, test, deploy, and operate."
> -- Clean Architecture principles applied to MCP

The architecture is deliberately minimal - it provides just enough structure to ensure compatibility while allowing maximum flexibility in implementation.

## High-Level Architecture

### The Big Picture

```
┌─────────────────────────────────────────────────────┐
│                   Application Layer                 │
│  (Agent Logic, Business Rules, User Interaction)    │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│                   MCP Client                        │
│  - Connection Management                            │
│  - Capability Discovery                             │
│  - Request/Response Handling                        │
│  - State Management                                 │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│                 Transport Layer                     │
│  (stdio, HTTP+SSE, WebSocket, etc.)                 │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│                   MCP Server                        │
│  - Request Processing                               │
│  - Capability Exposure                              │
│  - Resource Management                              │
│  - Tool Execution                                   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│               Backend Systems                       │
│  (Databases, APIs, File Systems, Services)          │
└─────────────────────────────────────────────────────┘
```

### Key Design Decisions

**1. Client-Server (Not Peer-to-Peer)**

MCP uses a clear client-server model:

```
Client (Agent)  →  Initiates connections
                →  Requests capabilities
                →  Invokes operations

Server (Tool)   →  Exposes capabilities
                →  Processes requests
                →  Returns results
```

Why? Clear separation of concerns and simpler implementation.

**2. JSON-RPC 2.0 (Not REST or gRPC)**

```json
{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {"name": "search", "arguments": {...}}
}
```

Why? Simple, widely supported, and bidirectional communication.

**3. Transport Agnostic**

The protocol works over:
- Standard I/O (stdio)
- HTTP with Server-Sent Events
- WebSockets (future)
- Custom transports

Why? Different use cases need different transport mechanisms.

**4. Capability-Based**

Servers declare what they can do. Clients discover and use capabilities.

Why? Allows graceful degradation and forward compatibility.

## Client-Server Model

### Roles and Responsibilities

**Client Responsibilities**:

```python
class MCPClient:
    """
    The MCP client (AI assistant) is responsible for:
    """
    
    async def connect_to_server(self, server_config):
        """Establish connection to MCP server"""
        pass
    
    async def initialize_connection(self):
        """Exchange protocol version and capabilities"""
        pass
    
    async def discover_capabilities(self):
        """Find out what the server can do"""
        pass
    
    async def request_resources(self, uri):
        """Read data from the server"""
        pass
    
    async def invoke_tool(self, name, arguments):
        """Execute a tool on the server"""
        pass
    
    async def manage_context(self):
        """Track conversation state and context"""
        pass
    
    async def handle_notifications(self, notification):
        """React to server-initiated events"""
        pass
```

**Server Responsibilities**:

```python
class MCPServer:
    """
    The MCP server (tool provider) is responsible for:
    """
    
    async def accept_connection(self, client):
        """Accept client connection"""
        pass
    
    async def declare_capabilities(self):
        """Tell client what we can do"""
        return {
            "resources": {},
            "tools": {},
            "prompts": {}
        }
    
    async def list_resources(self):
        """Provide list of available resources"""
        pass
    
    async def read_resource(self, uri):
        """Provide resource content"""
        pass
    
    async def list_tools(self):
        """Provide list of available tools"""
        pass
    
    async def execute_tool(self, name, arguments):
        """Execute tool and return result"""
        pass
    
    async def send_notification(self, event):
        """Notify client of events"""
        pass
```

### Connection Cardinality

**One Client, Many Servers**:

```python
class Agent:
    def __init__(self):
        self.client = MCPClient()
        
        # Connect to multiple servers
        self.client.connect("filesystem-server")
        self.client.connect("database-server")
        self.client.connect("web-search-server")
        self.client.connect("email-server")
    
    async def accomplish_task(self, task):
        # Can use tools from any connected server
        data = await self.client.read_resource(
            server="filesystem-server",
            uri="file:///data.json"
        )
        
        results = await self.client.call_tool(
            server="web-search-server",
            tool="search",
            arguments={"query": task}
        )
        
        await self.client.call_tool(
            server="email-server",
            tool="send",
            arguments={"to": "user@example.com", "body": results}
        )
```

**One Server, Many Clients**:

```python
class SharedMCPServer:
    """
    A server can handle multiple client connections
    """
    
    def __init__(self):
        self.clients = {}
    
    async def handle_connection(self, client_id):
        """Handle a new client connection"""
        self.clients[client_id] = ClientSession()
        
        # Each client gets independent session
        async for request in self.receive_requests(client_id):
            response = await self.process_request(request)
            await self.send_response(client_id, response)
```

### State Management

**Client State**:

```python
class MCPClientState:
    def __init__(self):
        # Connection state
        self.connected_servers = {}
        self.server_capabilities = {}
        
        # Discovery cache
        self.available_resources = {}
        self.available_tools = {}
        self.available_prompts = {}
        
        # Request tracking
        self.pending_requests = {}
        self.request_id_counter = 0
        
        # Subscription state
        self.resource_subscriptions = set()
        
        # Context
        self.conversation_history = []
        self.accumulated_context = {}
```

**Server State**:

```python
class MCPServerState:
    def __init__(self):
        # Client connections
        self.active_clients = {}
        
        # Resource state
        self.resources = {}
        self.resource_subscribers = defaultdict(set)
        
        # Tool state
        self.registered_tools = {}
        
        # Prompt state
        self.prompt_templates = {}
        
        # Execution state
        self.running_operations = {}
```

## Transport Mechanisms

### Standard I/O (stdio)

The most common transport for local servers:

```python
# Server side
import sys
import json

async def stdio_server():
    """Server reads from stdin, writes to stdout"""
    
    while True:
        # Read JSON-RPC request from stdin
        line = sys.stdin.readline()
        if not line:
            break
        
        request = json.loads(line)
        
        # Process request
        response = await handle_request(request)
        
        # Write JSON-RPC response to stdout
        sys.stdout.write(json.dumps(response) + '\n')
        sys.stdout.flush()

# Client side
import subprocess
import json

class StdioClient:
    def __init__(self, server_command):
        # Launch server as subprocess
        self.process = subprocess.Popen(
            server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
    
    async def send_request(self, method, params):
        request = {
            "jsonrpc": "2.0",
            "id": self.next_id(),
            "method": method,
            "params": params
        }
        
        # Write to server's stdin
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        
        # Read from server's stdout
        line = self.process.stdout.readline()
        response = json.loads(line)
        
        return response
```

**Characteristics**:
- ✓ Simple and lightweight
- ✓ Works with any language
- ✓ No network configuration needed
- ✗ Server must be local
- ✗ No built-in encryption

### HTTP with SSE (Server-Sent Events)

For remote servers and web integration:

```python
# Server side (using FastAPI)
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI()

class SSEServer:
    def __init__(self):
        self.clients = {}
    
    @app.post("/mcp")
    async def handle_request(self, request: Request):
        """Handle JSON-RPC requests via HTTP POST"""
        body = await request.json()
        response = await self.process_request(body)
        return response
    
    @app.get("/mcp/events")
    async def event_stream(self, request: Request):
        """Server-Sent Events for notifications"""
        client_id = request.headers.get("X-Client-ID")
        
        async def generate():
            queue = asyncio.Queue()
            self.clients[client_id] = queue
            
            try:
                while True:
                    event = await queue.get()
                    yield f"data: {json.dumps(event)}\n\n"
            finally:
                del self.clients[client_id]
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
    
    async def send_notification(self, client_id, notification):
        """Send notification to specific client"""
        if client_id in self.clients:
            await self.clients[client_id].put(notification)

# Client side
import requests
import sseclient
import threading

class HTTPSSEClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.client_id = generate_uuid()
        self.event_thread = None
    
    def connect(self):
        """Start listening for server events"""
        self.event_thread = threading.Thread(
            target=self._listen_events
        )
        self.event_thread.start()
    
    def _listen_events(self):
        """Listen for SSE notifications"""
        headers = {"X-Client-ID": self.client_id}
        response = requests.get(
            f"{self.base_url}/events",
            headers=headers,
            stream=True
        )
        
        client = sseclient.SSEClient(response)
        for event in client.events():
            notification = json.loads(event.data)
            self.handle_notification(notification)
    
    async def send_request(self, method, params):
        """Send JSON-RPC request via HTTP POST"""
        request = {
            "jsonrpc": "2.0",
            "id": self.next_id(),
            "method": method,
            "params": params
        }
        
        response = requests.post(
            f"{self.base_url}/mcp",
            json=request
        )
        
        return response.json()
```

**Characteristics**:
- ✓ Works over network
- ✓ Can use HTTPS for encryption
- ✓ Bidirectional (via SSE for notifications)
- ✓ Firewall-friendly (standard HTTP ports)
- ✗ More complex than stdio
- ✗ Requires web server infrastructure

### Custom Transports

MCP is transport-agnostic, so you can implement custom transports:

```python
class CustomTransport:
    """
    Any transport that can:
    1. Send JSON-RPC messages
    2. Receive JSON-RPC messages
    3. Handle bidirectional communication
    """
    
    async def send(self, message: dict):
        """Send a JSON-RPC message"""
        raise NotImplementedError
    
    async def receive(self) -> dict:
        """Receive a JSON-RPC message"""
        raise NotImplementedError

# Example: Unix domain socket transport
class UnixSocketTransport(CustomTransport):
    def __init__(self, socket_path):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.connect(socket_path)
    
    async def send(self, message):
        data = json.dumps(message).encode('utf-8')
        length = len(data).to_bytes(4, 'big')
        self.socket.sendall(length + data)
    
    async def receive(self):
        length_bytes = self.socket.recv(4)
        length = int.from_bytes(length_bytes, 'big')
        data = self.socket.recv(length)
        return json.loads(data.decode('utf-8'))

# Example: Message queue transport
class RabbitMQTransport(CustomTransport):
    def __init__(self, queue_name):
        self.connection = pika.BlockingConnection()
        self.channel = self.connection.channel()
        self.queue_name = queue_name
        self.channel.queue_declare(queue=queue_name)
    
    async def send(self, message):
        self.channel.basic_publish(
            exchange='',
            routing_key=self.queue_name,
            body=json.dumps(message)
        )
    
    async def receive(self):
        method, properties, body = self.channel.basic_get(
            queue=self.queue_name
        )
        if body:
            return json.loads(body)
        return None
```

## Communication Protocol

### JSON-RPC 2.0 Basics

MCP uses JSON-RPC 2.0 for all communication:

**Request**:
```json
{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "method_name",
    "params": {
        "param1": "value1",
        "param2": "value2"
    }
}
```

**Response**:
```json
{
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "data": "result_value"
    }
}
```

**Error**:
```json
{
    "jsonrpc": "2.0",
    "id": 1,
    "error": {
        "code": -32600,
        "message": "Invalid Request",
        "data": {"details": "..."}
    }
}
```

**Notification** (no response expected):
```json
{
    "jsonrpc": "2.0",
    "method": "notification_name",
    "params": {"data": "value"}
}
```

### MCP-Specific Methods

**Initialization**:

```python
# Client → Server: initialize
{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "roots": {"listChanged": true},
            "sampling": {}
        },
        "clientInfo": {
            "name": "MyAgent",
            "version": "1.0.0"
        }
    }
}

# Server → Client: response
{
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "resources": {"subscribe": true},
            "tools": {},
            "prompts": {}
        },
        "serverInfo": {
            "name": "MyServer",
            "version": "1.0.0"
        }
    }
}

# Client → Server: initialized notification
{
    "jsonrpc": "2.0",
    "method": "notifications/initialized"
}
```

**Resource Operations**:

```python
# List resources
{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "resources/list",
    "params": {}
}

# Read resource
{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "resources/read",
    "params": {
        "uri": "file:///path/to/resource"
    }
}

# Subscribe to resource
{
    "jsonrpc": "2.0",
    "id": 4,
    "method": "resources/subscribe",
    "params": {
        "uri": "file:///path/to/resource"
    }
}

# Resource updated notification
{
    "jsonrpc": "2.0",
    "method": "notifications/resources/updated",
    "params": {
        "uri": "file:///path/to/resource"
    }
}
```

**Tool Operations**:

```python
# List tools
{
    "jsonrpc": "2.0",
    "id": 5,
    "method": "tools/list",
    "params": {}
}

# Call tool
{
    "jsonrpc": "2.0",
    "id": 6,
    "method": "tools/call",
    "params": {
        "name": "tool_name",
        "arguments": {
            "arg1": "value1",
            "arg2": "value2"
        }
    }
}
```

**Prompt Operations**:

```python
# List prompts
{
    "jsonrpc": "2.0",
    "id": 7,
    "method": "prompts/list",
    "params": {}
}

# Get prompt
{
    "jsonrpc": "2.0",
    "id": 8,
    "method": "prompts/get",
    "params": {
        "name": "prompt_name",
        "arguments": {
            "arg1": "value1"
        }
    }
}
```

### Message Format

**Content Types**:

MCP supports multiple content types in responses:

```python
# Text content
{
    "type": "text",
    "text": "This is a text response"
}

# Image content
{
    "type": "image",
    "data": "base64_encoded_image_data",
    "mimeType": "image/png"
}

# Resource reference
{
    "type": "resource",
    "resource": {
        "uri": "file:///path/to/resource",
        "text": "Resource content"
    }
}
```

**Complete Response Example**:

```json
{
    "jsonrpc": "2.0",
    "id": 6,
    "result": {
        "content": [
            {
                "type": "text",
                "text": "Search results:"
            },
            {
                "type": "text",
                "text": "1. First result\n2. Second result"
            },
            {
                "type": "resource",
                "resource": {
                    "uri": "search://results/123",
                    "text": "Detailed results..."
                }
            }
        ],
        "isError": false
    }
}
```

## Capability Discovery

### The Discovery Process

```
┌─────────────────────────────────────────────────┐
│  1. Client connects to server                   │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  2. Client sends initialize request             │
│     - Protocol version                          │
│     - Client capabilities                       │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  3. Server responds with:                       │
│     - Server capabilities                       │
│     - Supported protocol version                │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  4. Client discovers specific capabilities:     │
│     - List resources                            │
│     - List tools                                │
│     - List prompts                              │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  5. Client can now use discovered capabilities  │
└─────────────────────────────────────────────────┘
```

### Capability Negotiation

```python
class CapabilityNegotiation:
    """
    Client and server negotiate what features are available
    """
    
    async def negotiate(self, client_caps, server_caps):
        """Determine effective capabilities"""
        
        effective_caps = {}
        
        # Resources
        if 'resources' in server_caps:
            effective_caps['resources'] = {
                'subscribe': (
                    client_caps.get('resources', {}).get('subscribe', False)
                    and server_caps['resources'].get('subscribe', False)
                ),
                'listChanged': (
                    server_caps['resources'].get('listChanged', False)
                )
            }
        
        # Tools
        if 'tools' in server_caps:
            effective_caps['tools'] = server_caps['tools']
        
        # Prompts
        if 'prompts' in server_caps:
            effective_caps['prompts'] = server_caps['prompts']
        
        # Sampling (client must support it)
        if ('sampling' in client_caps and 'sampling' in server_caps):
            effective_caps['sampling'] = {}
        
        return effective_caps
```

### Dynamic Discovery

Capabilities can change at runtime:

```python
# Server notifies client that resources list changed
{
    "jsonrpc": "2.0",
    "method": "notifications/resources/list_changed"
}

# Client re-queries resources
{
    "jsonrpc": "2.0",
    "id": 10,
    "method": "resources/list"
}

# Server returns updated list
{
    "jsonrpc": "2.0",
    "id": 10,
    "result": {
        "resources": [
            {"uri": "file:///new_resource", "name": "New Resource"}
        ]
    }
}
```

### Capability-Based Programming

Clients should adapt to available capabilities:

```python
class AdaptiveClient:
    async def initialize(self):
        # Initialize connection
        result = await self.send_initialize()
        self.server_capabilities = result['capabilities']
    
    async def use_resources(self):
        # Check if server supports resources
        if 'resources' not in self.server_capabilities:
            raise NotSupported("Server doesn't support resources")
        
        # Check if server supports subscriptions
        if self.server_capabilities['resources'].get('subscribe'):
            # Use subscription feature
            await self.subscribe_to_resource(uri)
        else:
            # Fall back to polling
            await self.poll_resource(uri)
    
    async def handle_updates(self):
        # Check if server sends list_changed notifications
        if self.server_capabilities.get('resources', {}).get('listChanged'):
            # Listen for notifications
            self.on_notification('resources/list_changed', self.refresh_list)
        else:
            # Periodically refresh
            asyncio.create_task(self.periodic_refresh())
```

## Message Flow Patterns

### Request-Response Pattern

The standard pattern for most operations:

```
Client                          Server
   │                               │
   │  ──── Request ───────────►   │
   │                               │
   │                        [Process]
   │                               │
   │  ◄─── Response ──────────   │
   │                               │
```

Example:

```python
# Client sends request
async def call_tool(self, name, arguments):
    request = {
        "jsonrpc": "2.0",
        "id": self.next_id(),
        "method": "tools/call",
        "params": {
            "name": name,
            "arguments": arguments
        }
    }
    
    # Send and wait for response
    response = await self.send_and_wait(request)
    
    if 'error' in response:
        raise MCPError(response['error'])
    
    return response['result']
```

### Notification Pattern

One-way messages that don't expect a response:

```
Server                          Client
   │                               │
   │  ──── Notification ──────►   │
   │                               │
   │                        [Handle Event]
   │                               │
   │  (no response)                │
```

Example:

```python
# Server sends notification
async def notify_resource_updated(self, uri):
    notification = {
        "jsonrpc": "2.0",
        "method": "notifications/resources/updated",
        "params": {"uri": uri}
    }
    
    # Send without waiting for response
    await self.send_notification(notification)

# Client handles notification
async def handle_notification(self, notification):
    method = notification['method']
    params = notification['params']
    
    if method == 'notifications/resources/updated':
        # Refresh cached resource
        await self.refresh_resource(params['uri'])
```

### Subscribe Pattern

Client subscribes to updates:

```
Client                          Server
   │                               │
   │  ──── Subscribe ─────────►   │
   │  ◄─── OK ────────────────   │
   │                               │
   │                       [Resource changes]
   │                               │
   │  ◄─── Notification ──────   │
   │  ◄─── Notification ──────   │
   │  ◄─── Notification ──────   │
   │                               │
   │  ──── Unsubscribe ───────►   │
   │  ◄─── OK ────────────────   │
```

Example:

```python
# Client subscribes
async def subscribe_to_resource(self, uri):
    request = {
        "jsonrpc": "2.0",
        "id": self.next_id(),
        "method": "resources/subscribe",
        "params": {"uri": uri}
    }
    
    await self.send_and_wait(request)
    
    # Register callback for updates
    self.subscriptions[uri] = self.handle_resource_update

# Server manages subscriptions
class MCPServer:
    def __init__(self):
        self.subscriptions = defaultdict(set)
    
    async def handle_subscribe(self, client_id, uri):
        self.subscriptions[uri].add(client_id)
        return {"success": True}
    
    async def resource_changed(self, uri):
        # Notify all subscribers
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/resources/updated",
            "params": {"uri": uri}
        }
        
        for client_id in self.subscriptions[uri]:
            await self.send_notification(client_id, notification)
```

### Sampling Pattern

Server requests LLM completion from client:

```
Server                          Client
   │                               │
   │  ──── CreateMessage ─────►   │
   │                               │
   │                      [Call LLM]
   │                               │
   │  ◄─── Response ───────────   │
```

Example:

```python
# Server requests completion
async def ask_llm_for_help(self, prompt):
    request = {
        "jsonrpc": "2.0",
        "id": self.next_id(),
        "method": "sampling/createMessage",
        "params": {
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": prompt
                    }
                }
            ],
            "maxTokens": 1000
        }
    }
    
    response = await self.send_to_client(request)
    return response['result']['content']['text']

# Client handles sampling request
async def handle_sampling_request(self, request):
    messages = request['params']['messages']
    max_tokens = request['params']['maxTokens']
    
    # Call LLM
    completion = await self.llm.complete(
        messages=messages,
        max_tokens=max_tokens
    )
    
    # Return result
    return {
        "content": {
            "type": "text",
            "text": completion
        },
        "model": "claude-3-5-sonnet",
        "stopReason": "end_turn"
    }
```

## Connection Lifecycle

### Phase 1: Connection Establishment

```python
class ConnectionLifecycle:
    
    async def establish_connection(self):
        """
        Phase 1: Establish transport connection
        """
        
        # For stdio transport
        if self.transport_type == "stdio":
            self.process = subprocess.Popen(
                self.server_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE
            )
        
        # For HTTP transport
        elif self.transport_type == "http":
            # Test connection
            response = requests.get(f"{self.server_url}/health")
            if response.status_code != 200:
                raise ConnectionError("Server not available")
        
        self.state = "CONNECTED"
```

### Phase 2: Protocol Handshake

```python
    async def initialize_protocol(self):
        """
        Phase 2: Exchange protocol version and capabilities
        """
        
        # Send initialize request
        init_response = await self.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": self.client_capabilities,
                "clientInfo": {
                    "name": "MyClient",
                    "version": "1.0.0"
                }
            }
        )
        
        # Store server capabilities
        self.server_capabilities = init_response['capabilities']
        self.server_info = init_response['serverInfo']
        
        # Send initialized notification
        await self.send_notification("notifications/initialized", {})
        
        self.state = "INITIALIZED"
```

### Phase 3: Capability Discovery

```python
    async def discover_capabilities(self):
        """
        Phase 3: Discover available resources, tools, and prompts
        """
        
        # Discover resources
        if 'resources' in self.server_capabilities:
            resources_response = await self.send_request(
                "resources/list", {}
            )
            self.available_resources = resources_response['resources']
        
        # Discover tools
        if 'tools' in self.server_capabilities:
            tools_response = await self.send_request(
                "tools/list", {}
            )
            self.available_tools = tools_response['tools']
        
        # Discover prompts
        if 'prompts' in self.server_capabilities:
            prompts_response = await self.send_request(
                "prompts/list", {}
            )
            self.available_prompts = prompts_response['prompts']
        
        self.state = "READY"
```

### Phase 4: Active Operation

```python
    async def operate(self):
        """
        Phase 4: Active operation phase
        """
        
        while self.state == "READY":
            # Handle requests from application
            if self.has_pending_requests():
                request = self.get_next_request()
                result = await self.execute_request(request)
                self.return_result(result)
            
            # Handle notifications from server
            if self.has_pending_notifications():
                notification = self.get_next_notification()
                await self.handle_notification(notification)
            
            await asyncio.sleep(0.01)
```

### Phase 5: Graceful Shutdown

```python
    async def shutdown(self):
        """
        Phase 5: Clean disconnection
        """
        
        # Unsubscribe from all resources
        for uri in self.subscriptions:
            await self.send_request(
                "resources/unsubscribe",
                {"uri": uri}
            )
        
        # Cancel pending requests
        for request_id in self.pending_requests:
            self.pending_requests[request_id].cancel()
        
        # Close transport
        if self.transport_type == "stdio":
            self.process.terminate()
            self.process.wait(timeout=5)
        
        elif self.transport_type == "http":
            await self.http_session.close()
        
        self.state = "DISCONNECTED"
```

### Complete Lifecycle Example

```python
async def full_lifecycle_example():
    client = MCPClient()
    
    try:
        # Phase 1: Connect
        await client.establish_connection(
            transport="stdio",
            command="mcp-server-sqlite"
        )
        
        # Phase 2: Initialize
        await client.initialize_protocol()
        
        # Phase 3: Discover
        await client.discover_capabilities()
        
        print(f"Connected to {client.server_info['name']}")
        print(f"Available tools: {[t['name'] for t in client.available_tools]}")
        
        # Phase 4: Operate
        result = await client.call_tool(
            "query",
            {"sql": "SELECT * FROM users"}
        )
        print(f"Query result: {result}")
        
    finally:
        # Phase 5: Shutdown
        await client.shutdown()
```

## Server Components

### Core Server Structure

```python
class MCPServer:
    """
    Complete MCP server implementation structure
    """
    
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        
        # Component managers
        self.resource_manager = ResourceManager()
        self.tool_manager = ToolManager()
        self.prompt_manager = PromptManager()
        self.connection_manager = ConnectionManager()
        
        # State
        self.capabilities = self._define_capabilities()
        self.active_clients = {}
    
    def _define_capabilities(self):
        """Define server capabilities"""
        return {
            "resources": {
                "subscribe": True,
                "listChanged": True
            },
            "tools": {
                "listChanged": False
            },
            "prompts": {
                "listChanged": True
            },
            "logging": {}
        }
    
    async def handle_request(self, request: dict) -> dict:
        """Main request handler"""
        method = request['method']
        params = request.get('params', {})
        
        # Route to appropriate handler
        if method == 'initialize':
            return await self.handle_initialize(params)
        elif method.startswith('resources/'):
            return await self.resource_manager.handle(method, params)
        elif method.startswith('tools/'):
            return await self.tool_manager.handle(method, params)
        elif method.startswith('prompts/'):
            return await self.prompt_manager.handle(method, params)
        else:
            raise MethodNotFound(method)
```

### Resource Manager

```python
class ResourceManager:
    """Manages resource exposure and access"""
    
    def __init__(self):
        self.resources = {}
        self.subscribers = defaultdict(set)
        self.watchers = {}
    
    async def register_resource(self, uri, provider):
        """Register a resource provider"""
        self.resources[uri] = provider
        
        # Set up change watching if supported
        if hasattr(provider, 'watch'):
            self.watchers[uri] = provider.watch(
                lambda: self.notify_update(uri)
            )
    
    async def handle(self, method, params):
        """Handle resource-related requests"""
        if method == 'resources/list':
            return await self.list_resources()
        elif method == 'resources/read':
            return await self.read_resource(params['uri'])
        elif method == 'resources/subscribe':
            return await self.subscribe(params['uri'], params['client_id'])
        elif method == 'resources/unsubscribe':
            return await self.unsubscribe(params['uri'], params['client_id'])
    
    async def list_resources(self):
        """List all available resources"""
        return {
            "resources": [
                {
                    "uri": uri,
                    "name": provider.name,
                    "description": provider.description,
                    "mimeType": provider.mime_type
                }
                for uri, provider in self.resources.items()
            ]
        }
    
    async def read_resource(self, uri):
        """Read resource content"""
        if uri not in self.resources:
            raise ResourceNotFound(uri)
        
        provider = self.resources[uri]
        content = await provider.read()
        
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": provider.mime_type,
                    "text": content
                }
            ]
        }
    
    async def notify_update(self, uri):
        """Notify subscribers of resource update"""
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/resources/updated",
            "params": {"uri": uri}
        }
        
        for client_id in self.subscribers[uri]:
            await self.send_notification(client_id, notification)
```

### Tool Manager

```python
class ToolManager:
    """Manages tool registration and execution"""
    
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, name, handler, schema):
        """Register a tool"""
        self.tools[name] = {
            "handler": handler,
            "schema": schema
        }
    
    async def handle(self, method, params):
        """Handle tool-related requests"""
        if method == 'tools/list':
            return await self.list_tools()
        elif method == 'tools/call':
            return await self.call_tool(
                params['name'],
                params.get('arguments', {})
            )
    
    async def list_tools(self):
        """List all available tools"""
        return {
            "tools": [
                {
                    "name": name,
                    "description": tool['schema']['description'],
                    "inputSchema": tool['schema']['inputSchema']
                }
                for name, tool in self.tools.items()
            ]
        }
    
    async def call_tool(self, name, arguments):
        """Execute a tool"""
        if name not in self.tools:
            raise ToolNotFound(name)
        
        tool = self.tools[name]
        
        # Validate arguments
        self.validate_arguments(arguments, tool['schema']['inputSchema'])
        
        # Execute tool
        try:
            result = await tool['handler'](arguments)
            
            return {
                "content": [
                    {
                        "type": "text",
                        "text": str(result)
                    }
                ],
                "isError": False
            }
        
        except Exception as e:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: {str(e)}"
                    }
                ],
                "isError": True
            }
```

### Prompt Manager

```python
class PromptManager:
    """Manages prompt templates"""
    
    def __init__(self):
        self.prompts = {}
    
    def register_prompt(self, name, template, arguments_schema):
        """Register a prompt template"""
        self.prompts[name] = {
            "template": template,
            "arguments": arguments_schema
        }
    
    async def handle(self, method, params):
        """Handle prompt-related requests"""
        if method == 'prompts/list':
            return await self.list_prompts()
        elif method == 'prompts/get':
            return await self.get_prompt(
                params['name'],
                params.get('arguments', {})
            )
    
    async def list_prompts(self):
        """List all available prompts"""
        return {
            "prompts": [
                {
                    "name": name,
                    "description": prompt['template'].description,
                    "arguments": prompt['arguments']
                }
                for name, prompt in self.prompts.items()
            ]
        }
    
    async def get_prompt(self, name, arguments):
        """Get a prompt with arguments filled in"""
        if name not in self.prompts:
            raise PromptNotFound(name)
        
        prompt = self.prompts[name]
        rendered = prompt['template'].render(**arguments)
        
        return {
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": rendered
                    }
                }
            ]
        }
```

## Client Components

### Core Client Structure

```python
class MCPClient:
    """
    Complete MCP client implementation structure
    """
    
    def __init__(self, name: str):
        self.name = name
        
        # Component managers
        self.connection_manager = ClientConnectionManager()
        self.request_manager = RequestManager()
        self.cache_manager = CacheManager()
        
        # State
        self.servers = {}
        self.capabilities = self._define_capabilities()
    
    async def connect_server(self, server_id, transport_config):
        """Connect to an MCP server"""
        connection = await self.connection_manager.connect(
            server_id,
            transport_config
        )
        
        # Initialize protocol
        server_info = await self.initialize_connection(connection)
        
        # Store connection
        self.servers[server_id] = {
            "connection": connection,
            "info": server_info,
            "capabilities": server_info['capabilities']
        }
        
        # Discover capabilities
        await self.discover_server_capabilities(server_id)
        
        return server_id
    
    async def call_tool(self, server_id, tool_name, arguments):
        """Call a tool on a specific server"""
        server = self.servers[server_id]
        
        request = {
            "jsonrpc": "2.0",
            "id": self.request_manager.next_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        response = await server['connection'].send_request(request)
        return response['result']
```

### Request Manager

```python
class RequestManager:
    """Manages request/response correlation"""
    
    def __init__(self):
        self.next_request_id = 1
        self.pending_requests = {}
    
    def next_id(self):
        """Generate unique request ID"""
        request_id = self.next_request_id
        self.next_request_id += 1
        return request_id
    
    async def send_request(self, connection, method, params):
        """Send request and wait for response"""
        request_id = self.next_id()
        
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params
        }
        
        # Create future for response
        future = asyncio.Future()
        self.pending_requests[request_id] = future
        
        # Send request
        await connection.send(request)
        
        # Wait for response
        try:
            response = await asyncio.wait_for(future, timeout=30.0)
            return response
        finally:
            del self.pending_requests[request_id]
    
    def handle_response(self, response):
        """Handle incoming response"""
        request_id = response['id']
        
        if request_id in self.pending_requests:
            future = self.pending_requests[request_id]
            
            if 'error' in response:
                future.set_exception(
                    MCPError(response['error'])
                )
            else:
                future.set_result(response['result'])
```

### Cache Manager

```python
class CacheManager:
    """Caches discovered capabilities and resources"""
    
    def __init__(self):
        self.resource_cache = {}
        self.tool_cache = {}
        self.prompt_cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    async def get_tools(self, server_id, connection):
        """Get tools with caching"""
        cache_key = f"{server_id}:tools"
        
        # Check cache
        if cache_key in self.tool_cache:
            cached_data, timestamp = self.tool_cache[cache_key]
            if time.time() - timestamp < self.cache_timeout:
                return cached_data
        
        # Fetch from server
        response = await connection.send_request(
            "tools/list", {}
        )
        
        tools = response['tools']
        
        # Update cache
        self.tool_cache[cache_key] = (tools, time.time())
        
        return tools
    
    def invalidate_cache(self, server_id, cache_type):
        """Invalidate cache when list changes"""
        cache_key = f"{server_id}:{cache_type}"
        if cache_type == 'tools':
            self.tool_cache.pop(cache_key, None)
        elif cache_type == 'resources':
            self.resource_cache.pop(cache_key, None)
        elif cache_type == 'prompts':
            self.prompt_cache.pop(cache_key, None)
```

## Protocol Layers

The MCP architecture can be viewed as layers:

```
┌────────────────────────────────────────┐
│  Application Layer                     │
│  (Agent logic, business rules)         │
└─────────────────┬──────────────────────┘
                  │
┌─────────────────┴──────────────────────┐
│  MCP Semantic Layer                    │
│  (Resources, Tools, Prompts)           │
└─────────────────┬──────────────────────┘
                  │
┌─────────────────┴──────────────────────┐
│  MCP Protocol Layer                    │
│  (JSON-RPC methods, message formats)   │
└─────────────────┬──────────────────────┘
                  │
┌─────────────────┴──────────────────────┐
│  JSON-RPC Layer                        │
│  (Request/response, notifications)     │
└─────────────────┬──────────────────────┘
                  │
┌─────────────────┴──────────────────────┐
│  Transport Layer                       │
│  (stdio, HTTP, WebSocket, etc.)        │
└─────────────────┬──────────────────────┘
                  │
┌─────────────────┴──────────────────────┐
│  Physical Layer                        │
│  (TCP, IPC, local process, etc.)       │
└────────────────────────────────────────┘
```

### Layer Responsibilities

**Application Layer**:
- Agent reasoning
- Task planning
- Decision making
- User interaction

**MCP Semantic Layer**:
- Resource schemas
- Tool schemas
- Prompt templates
- Content types

**MCP Protocol Layer**:
- Method definitions
- Parameter structures
- Response formats
- Capability negotiation

**JSON-RPC Layer**:
- Message enveloping
- ID correlation
- Error codes
- Notification handling

**Transport Layer**:
- Connection management
- Message delivery
- Reliability
- Security (TLS)

**Physical Layer**:
- Network protocols
- Process communication
- Hardware interfaces

## Security Architecture

### Authentication

```python
class SecureM CPServer:
    """MCP server with authentication"""
    
    def __init__(self):
        self.authenticated_clients = {}
        self.api_keys = self.load_api_keys()
    
    async def handle_initialize(self, request):
        """Authenticate during initialization"""
        params = request['params']
        
        # Check for API key
        if 'auth' not in params:
            raise AuthenticationRequired()
        
        api_key = params['auth'].get('apiKey')
        if api_key not in self.api_keys:
            raise AuthenticationFailed()
        
        # Store authenticated client
        client_id = self.generate_client_id()
        self.authenticated_clients[client_id] = {
            "api_key": api_key,
            "permissions": self.api_keys[api_key]['permissions']
        }
        
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": self.get_capabilities_for_client(client_id),
            "serverInfo": self.server_info
        }
```

### Authorization

```python
class AuthorizedToolManager:
    """Tool manager with permission checks"""
    
    async def call_tool(self, client_id, name, arguments):
        """Execute tool with authorization check"""
        
        # Check authentication
        if client_id not in self.authenticated_clients:
            raise NotAuthenticated()
        
        # Check authorization
        client = self.authenticated_clients[client_id]
        tool = self.tools[name]
        
        if tool['required_permission'] not in client['permissions']:
            raise NotAuthorized(
                f"Tool '{name}' requires permission: {tool['required_permission']}"
            )
        
        # Execute tool
        return await tool['handler'](arguments)
```

### Secure Transport

```python
# Using HTTPS with TLS
class SecureHTTPTransport:
    def __init__(self, base_url, cert_path, key_path):
        # Use TLS for encryption
        context = ssl.create_default_context()
        context.load_cert_chain(cert_path, key_path)
        
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=context)
        )
    
    async def send_request(self, request):
        async with self.session.post(
            self.base_url,
            json=request,
            headers={"Content-Type": "application/json"}
        ) as response:
            return await response.json()
```

## Scalability Patterns

### Connection Pooling

```python
class ConnectionPool:
    """Pool of MCP server connections"""
    
    def __init__(self, server_config, pool_size=5):
        self.server_config = server_config
        self.pool_size = pool_size
        self.available = asyncio.Queue()
        self.connections = []
    
    async def initialize(self):
        """Create connection pool"""
        for _ in range(self.pool_size):
            conn = await self.create_connection()
            self.connections.append(conn)
            await self.available.put(conn)
    
    async def acquire(self):
        """Get a connection from pool"""
        return await self.available.get()
    
    async def release(self, connection):
        """Return connection to pool"""
        await self.available.put(connection)
    
    async def execute(self, method, params):
        """Execute request using pooled connection"""
        conn = await self.acquire()
        try:
            return await conn.send_request(method, params)
        finally:
            await self.release(conn)
```

### Load Balancing

```python
class LoadBalancedClient:
    """Client that load-balances across multiple servers"""
    
    def __init__(self):
        self.servers = []
        self.current_index = 0
    
    def add_server(self, server_connection):
        """Add server to rotation"""
        self.servers.append(server_connection)
    
    def get_next_server(self):
        """Round-robin server selection"""
        server = self.servers[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.servers)
        return server
    
    async def call_tool(self, name, arguments):
        """Call tool on next available server"""
        server = self.get_next_server()
        return await server.call_tool(name, arguments)
```

### Caching Layer

```python
class CachingProxy:
    """Caching proxy for MCP resources"""
    
    def __init__(self, backend_client):
        self.backend = backend_client
        self.cache = {}
        self.cache_ttl = 300
    
    async def read_resource(self, uri):
        """Read resource with caching"""
        now = time.time()
        
        # Check cache
        if uri in self.cache:
            content, timestamp = self.cache[uri]
            if now - timestamp < self.cache_ttl:
                return content
        
        # Fetch from backend
        content = await self.backend.read_resource(uri)
        
        # Update cache
        self.cache[uri] = (content, now)
        
        return content
    
    def invalidate(self, uri):
        """Invalidate cached resource"""
        self.cache.pop(uri, None)
```

## Error Handling Architecture

### Error Hierarchy

```python
class MCPError(Exception):
    """Base MCP error"""
    def __init__(self, code, message, data=None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(message)

class ParseError(MCPError):
    """JSON parse error"""
    def __init__(self):
        super().__init__(-32700, "Parse error")

class InvalidRequest(MCPError):
    """Invalid JSON-RPC request"""
    def __init__(self, details=None):
        super().__init__(-32600, "Invalid Request", details)

class MethodNotFound(MCPError):
    """Method not found"""
    def __init__(self, method):
        super().__init__(
            -32601,
            "Method not found",
            {"method": method}
        )

class InvalidParams(MCPError):
    """Invalid method parameters"""
    def __init__(self, details):
        super().__init__(-32602, "Invalid params", details)

class InternalError(MCPError):
    """Internal server error"""
    def __init__(self, details=None):
        super().__init__(-32603, "Internal error", details)
```

### Error Handling Patterns

```python
class RobustMCPClient:
    """MCP client with comprehensive error handling"""
    
    async def call_tool_with_retry(
        self,
        name,
        arguments,
        max_retries=3,
        backoff=1.0
    ):
        """Call tool with automatic retry"""
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                result = await self.call_tool(name, arguments)
                return result
            
            except MCPError as e:
                # Don't retry client errors
                if e.code in [-32600, -32601, -32602]:
                    raise
                
                last_error = e
                
                # Exponential backoff
                if attempt < max_retries - 1:
                    await asyncio.sleep(backoff * (2 ** attempt))
            
            except asyncio.TimeoutError:
                last_error = TimeoutError()
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(backoff * (2 ** attempt))
        
        raise last_error
    
    async def call_tool_with_fallback(
        self,
        primary_tool,
        fallback_tool,
        arguments
    ):
        """Call tool with fallback on failure"""
        try:
            return await self.call_tool(primary_tool, arguments)
        except MCPError:
            logger.warning(
                f"Primary tool {primary_tool} failed, "
                f"trying fallback {fallback_tool}"
            )
            return await self.call_tool(fallback_tool, arguments)
```

## Extensibility Mechanisms

### Custom Capabilities

```python
class ExtendedMCPServer(MCPServer):
    """Server with custom capabilities"""
    
    def _define_capabilities(self):
        """Add custom capabilities"""
        capabilities = super()._define_capabilities()
        
        # Add experimental capability
        capabilities['experimental/custom_feature'] = {
            "enabled": True,
            "version": "0.1.0"
        }
        
        return capabilities
    
    async def handle_request(self, request):
        """Handle including custom methods"""
        method = request['method']
        
        if method.startswith('experimental/custom_feature/'):
            return await self.handle_custom_feature(request)
        
        return await super().handle_request(request)
```

### Protocol Evolution

```python
class VersionAwareMCPClient:
    """Client that adapts to server protocol version"""
    
    async def initialize_connection(self, connection):
        """Initialize with version negotiation"""
        
        # Try latest version
        try:
            response = await connection.send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": self.capabilities
                }
            )
            
            self.protocol_version = response['protocolVersion']
        
        except InvalidRequest:
            # Fall back to older version
            response = await connection.send_request(
                "initialize",
                {
                    "protocolVersion": "2024-01-01",
                    "capabilities": self.legacy_capabilities
                }
            )
            
            self.protocol_version = response['protocolVersion']
        
        # Adapt behavior based on version
        if self.protocol_version >= "2024-11-05":
            self.use_modern_features = True
        else:
            self.use_modern_features = False
```

## Summary

MCP's architecture is designed for simplicity, flexibility, and extensibility:

**Key Architectural Principles**:
- **Client-server model** - Clear separation of concerns
- **JSON-RPC 2.0** - Simple, universal protocol
- **Transport-agnostic** - Works over stdio, HTTP, WebSocket
- **Capability-based** - Graceful feature negotiation
- **Layered design** - Clean abstraction boundaries

**Core Components**:
- **Clients** - Connect, discover, and invoke
- **Servers** - Expose resources, tools, and prompts
- **Transports** - stdio, HTTP+SSE, custom
- **Managers** - Resource, tool, prompt, connection management

**Communication Patterns**:
- **Request-response** - Standard operations
- **Notifications** - Asynchronous events
- **Subscriptions** - Real-time updates
- **Sampling** - Bidirectional LLM access

**Advanced Features**:
- **Security** - Authentication and authorization
- **Scalability** - Connection pooling, load balancing
- **Reliability** - Error handling, retries, fallbacks
- **Extensibility** - Custom capabilities, protocol evolution

Understanding this architecture enables you to build robust, scalable MCP implementations.

## Next Steps

- **[Resources](mcp-resources.md)** - Learn how resources expose data
- **[Tools and Tool Registration](mcp-tools.md)** - Understand tool integration
- **[Context Management](mcp-context.md)** - Manage LLM context effectively
- **[Building MCP Servers](building-mcp-servers.md)** - Practical implementation guide
- **[MCP Fundamentals](mcp-fundamentals.md)** - Review core concepts
