# MCP Tools and Tool Registration

## Table of Contents

- [Introduction](#introduction)
- [What Are MCP Tools?](#what-are-mcp-tools)
- [Tool Schemas](#tool-schemas)
- [Tool Discovery](#tool-discovery)
- [Tool Invocation](#tool-invocation)
- [Tool Registration](#tool-registration)
- [Input Validation](#input-validation)
- [Output Formatting](#output-formatting)
- [Error Handling](#error-handling)
- [Tool Categories](#tool-categories)
- [Tool Composition](#tool-composition)
- [Tool Lifecycle](#tool-lifecycle)
- [Advanced Tool Patterns](#advanced-tool-patterns)
- [Tool Performance](#tool-performance)
- [Tool Security](#tool-security)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Tools are the action primitives of MCP - they represent **operations that agents can execute** to interact with external systems. While resources provide read access to data, tools enable agents to **do things**: send emails, create files, make API calls, run calculations, and more.

MCP standardizes how tools are:
- **Discovered** - Agents find available tools dynamically
- **Described** - Tools declare their capabilities and requirements
- **Invoked** - Agents call tools with validated arguments
- **Executed** - Servers process tool calls and return results

> "Tools are verbs - they make things happen."

This comprehensive guide covers everything about MCP tools, from basic registration to advanced composition patterns.

## What Are MCP Tools?

### Definition

An **MCP tool** is:
- A named operation exposed by an MCP server
- Described by a JSON schema
- Invocable by MCP clients
- Executed server-side with defined inputs/outputs

### Tool vs Resource

The distinction is crucial:

```
Resource (data):
- READ operation
- No side effects
- Returns data
- Example: read_file("data.json")

Tool (action):
- EXECUTE operation
- Has side effects
- Performs action
- Example: send_email(to, subject, body)
```

**Decision matrix**:

| Operation | Use |
|-----------|-----|
| Get weather forecast | Resource |
| Send weather alert | Tool |
| Read database | Resource |
| Update database | Tool |
| View calendar | Resource |
| Create appointment | Tool |

### Tool Structure

A tool consists of:

**1. Metadata**:
```python
{
    "name": "send_email",
    "description": "Send an email to a recipient"
}
```

**2. Input Schema (JSON Schema)**:
```python
{
    "inputSchema": {
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": "Recipient email address"
            },
            "subject": {
                "type": "string",
                "description": "Email subject line"
            },
            "body": {
                "type": "string",
                "description": "Email body content"
            }
        },
        "required": ["to", "subject", "body"]
    }
}
```

**3. Handler (Server-side)**:
```python
async def send_email_handler(arguments):
    to = arguments['to']
    subject = arguments['subject']
    body = arguments['body']
    
    # Send email
    await email_service.send(to, subject, body)
    
    return "Email sent successfully"
```

## Tool Schemas

### JSON Schema Basics

MCP uses JSON Schema to describe tool inputs:

**Simple String Parameter**:
```json
{
    "type": "string",
    "description": "User's name"
}
```

**Number with Constraints**:
```json
{
    "type": "number",
    "description": "Temperature in Celsius",
    "minimum": -273.15,
    "maximum": 1000
}
```

**Enum (Fixed Choices)**:
```json
{
    "type": "string",
    "enum": ["draft", "published", "archived"],
    "description": "Document status"
}
```

**Array**:
```json
{
    "type": "array",
    "items": {
        "type": "string"
    },
    "description": "List of tags",
    "minItems": 1,
    "maxItems": 10
}
```

**Nested Object**:
```json
{
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string", "format": "email"}
    },
    "required": ["name", "email"]
}
```

### Complete Tool Schema Example

```python
{
    "name": "create_calendar_event",
    "description": "Create a new calendar event",
    "inputSchema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Event title",
                "minLength": 1,
                "maxLength": 200
            },
            "start_time": {
                "type": "string",
                "format": "date-time",
                "description": "Event start time (ISO 8601 format)"
            },
            "end_time": {
                "type": "string",
                "format": "date-time",
                "description": "Event end time (ISO 8601 format)"
            },
            "attendees": {
                "type": "array",
                "items": {
                    "type": "string",
                    "format": "email"
                },
                "description": "List of attendee email addresses",
                "uniqueItems": true
            },
            "location": {
                "type": "string",
                "description": "Event location (optional)"
            },
            "reminder_minutes": {
                "type": "integer",
                "description": "Minutes before event to send reminder",
                "minimum": 0,
                "maximum": 10080,
                "default": 15
            }
        },
        "required": ["title", "start_time", "end_time"]
    }
}
```

### Schema Validation Benefits

Schemas provide:
1. **Type safety** - Incorrect types rejected
2. **Constraint enforcement** - Min/max, patterns, etc.
3. **Documentation** - Self-documenting APIs
4. **IDE support** - Autocomplete in clients
5. **Error prevention** - Catch errors before execution

## Tool Discovery

### Listing Tools

```python
# Client requests tool list
request = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {}
}

# Server responds with tool list
response = {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "tools": [
            {
                "name": "web_search",
                "description": "Search the web for information",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 100
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "send_email",
                "description": "Send an email",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "format": "email"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"}
                    },
                    "required": ["to", "subject", "body"]
                }
            }
        ]
    }
}
```

### Client-Side Discovery

```python
class ToolDiscoveryClient:
    async def discover_tools(self, server_id):
        """Discover all tools from a server"""
        response = await self.mcp.send_request(
            server_id,
            "tools/list",
            {}
        )
        
        tools = response['tools']
        
        # Index by name
        self.tools[server_id] = {
            tool['name']: tool
            for tool in tools
        }
        
        return tools
    
    async def find_tools_by_keyword(self, keyword):
        """Search for tools matching keyword"""
        matching = []
        
        for server_id, tools in self.tools.items():
            for name, tool in tools.items():
                if (keyword.lower() in name.lower() or
                    keyword.lower() in tool['description'].lower()):
                    matching.append({
                        'server_id': server_id,
                        'name': name,
                        'tool': tool
                    })
        
        return matching
    
    async def get_tool_schema(self, server_id, tool_name):
        """Get detailed schema for a specific tool"""
        if server_id not in self.tools:
            await self.discover_tools(server_id)
        
        return self.tools[server_id].get(tool_name)
```

### Dynamic Tool Discovery

Tools can be added/removed at runtime:

```python
# Server notifies clients that tool list changed
notification = {
    "jsonrpc": "2.0",
    "method": "notifications/tools/list_changed"
}

# Client re-discovers tools
async def handle_tools_changed(notification):
    # Re-query all servers
    for server_id in self.connected_servers:
        await self.discover_tools(server_id)
    
    # Update agent's tool list
    self.agent.refresh_available_tools()
```

## Tool Invocation

### Basic Invocation

```python
# Client calls a tool
request = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
        "name": "web_search",
        "arguments": {
            "query": "Model Context Protocol",
            "max_results": 5
        }
    }
}

# Server executes and responds
response = {
    "jsonrpc": "2.0",
    "id": 2,
    "result": {
        "content": [
            {
                "type": "text",
                "text": "Found 5 results:\n1. MCP Documentation...\n2. GitHub Repository..."
            }
        ],
        "isError": false
    }
}
```

### Client-Side Invocation

```python
class MCPToolClient:
    async def call_tool(self, server_id, tool_name, arguments):
        """Call a tool on a specific server"""
        
        # Validate tool exists
        tool_schema = await self.get_tool_schema(server_id, tool_name)
        if not tool_schema:
            raise ToolNotFound(f"{server_id}/{tool_name}")
        
        # Validate arguments against schema
        self.validate_arguments(arguments, tool_schema['inputSchema'])
        
        # Call tool
        request = {
            "jsonrpc": "2.0",
            "id": self.next_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        response = await self.send_request(server_id, request)
        
        # Check for errors
        if response['result'].get('isError'):
            raise ToolExecutionError(
                response['result']['content'][0]['text']
            )
        
        return response['result']['content']
    
    def validate_arguments(self, arguments, schema):
        """Validate arguments match JSON schema"""
        import jsonschema
        
        try:
            jsonschema.validate(arguments, schema)
        except jsonschema.ValidationError as e:
            raise InvalidArguments(str(e))
```

### Server-Side Execution

```python
class MCPToolServer:
    async def handle_tool_call(self, tool_name, arguments):
        """Execute a tool call"""
        
        # Find tool handler
        if tool_name not in self.tools:
            raise ToolNotFound(tool_name)
        
        tool = self.tools[tool_name]
        
        # Validate arguments
        try:
            self.validate_arguments(arguments, tool['schema'])
        except ValidationError as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Invalid arguments: {e}"
                }],
                "isError": true
            }
        
        # Execute tool
        try:
            result = await tool['handler'](arguments)
            
            # Format result
            if isinstance(result, str):
                content = [{"type": "text", "text": result}]
            elif isinstance(result, dict):
                import json
                content = [{
                    "type": "text",
                    "text": json.dumps(result, indent=2)
                }]
            else:
                content = [{"type": "text", "text": str(result)}]
            
            return {
                "content": content,
                "isError": false
            }
        
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Tool execution failed: {str(e)}"
                }],
                "isError": true
            }
```

## Tool Registration

### Basic Registration

```python
class SimpleMCPServer(MCPServer):
    def __init__(self):
        super().__init__("simple-tools")
        self.register_tools()
    
    def register_tools(self):
        """Register all tools"""
        
        # Register search tool
        self.register_tool(
            name="web_search",
            description="Search the web for information",
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {
                        "type": "integer",
                        "default": 10
                    }
                },
                "required": ["query"]
            },
            handler=self.web_search_handler
        )
        
        # Register email tool
        self.register_tool(
            name="send_email",
            description="Send an email",
            schema={
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"}
                },
                "required": ["to", "subject", "body"]
            },
            handler=self.send_email_handler
        )
    
    async def web_search_handler(self, arguments):
        """Handle web search"""
        query = arguments['query']
        max_results = arguments.get('max_results', 10)
        
        # Perform search
        results = await self.search_service.search(query, max_results)
        
        # Format results
        output = f"Found {len(results)} results:\n"
        for i, result in enumerate(results, 1):
            output += f"{i}. {result['title']}: {result['url']}\n"
        
        return output
    
    async def send_email_handler(self, arguments):
        """Handle email sending"""
        await self.email_service.send(
            to=arguments['to'],
            subject=arguments['subject'],
            body=arguments['body']
        )
        
        return f"Email sent to {arguments['to']}"
```

### Decorator-Based Registration

```python
class DecoratorMCPServer(MCPServer):
    def __init__(self):
        super().__init__("decorator-tools")
        self.tools = {}
    
    def tool(self, name, description, schema):
        """Decorator for registering tools"""
        def decorator(func):
            self.tools[name] = {
                'description': description,
                'schema': schema,
                'handler': func
            }
            return func
        return decorator

# Usage
server = DecoratorMCPServer()

@server.tool(
    name="calculate",
    description="Perform a calculation",
    schema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate"
            }
        },
        "required": ["expression"]
    }
)
async def calculate(arguments):
    """Calculate mathematical expression"""
    expression = arguments['expression']
    
    # Safe evaluation
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")

@server.tool(
    name="translate",
    description="Translate text between languages",
    schema={
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "from_lang": {"type": "string"},
            "to_lang": {"type": "string"}
        },
        "required": ["text", "to_lang"]
    }
)
async def translate(arguments):
    """Translate text"""
    text = arguments['text']
    from_lang = arguments.get('from_lang', 'auto')
    to_lang = arguments['to_lang']
    
    result = await translation_service.translate(
        text, from_lang, to_lang
    )
    
    return result
```

### Class-Based Tools

```python
class Tool:
    """Base class for MCP tools"""
    name: str
    description: str
    
    def get_schema(self):
        """Return JSON schema for tool"""
        raise NotImplementedError
    
    async def execute(self, arguments):
        """Execute the tool"""
        raise NotImplementedError

class SearchTool(Tool):
    name = "web_search"
    description = "Search the web for information"
    
    def __init__(self, search_service):
        self.search_service = search_service
    
    def get_schema(self):
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 100
                }
            },
            "required": ["query"]
        }
    
    async def execute(self, arguments):
        query = arguments['query']
        max_results = arguments.get('max_results', 10)
        
        results = await self.search_service.search(query, max_results)
        
        return self.format_results(results)
    
    def format_results(self, results):
        output = []
        for i, result in enumerate(results, 1):
            output.append(
                f"{i}. {result['title']}\n"
                f"   {result['url']}\n"
                f"   {result['snippet']}\n"
            )
        return "\n".join(output)

# Register class-based tools
class ClassBasedServer(MCPServer):
    def __init__(self, search_service):
        super().__init__("class-based")
        
        # Instantiate tools
        self.tool_instances = [
            SearchTool(search_service),
            TranslateTool(),
            CalculatorTool(),
        ]
        
        # Register them
        for tool in self.tool_instances:
            self.register_tool(
                name=tool.name,
                description=tool.description,
                schema=tool.get_schema(),
                handler=tool.execute
            )
```

## Input Validation

### Schema Validation

```python
import jsonschema

class ValidatingMCPServer(MCPServer):
    def validate_tool_arguments(self, tool_name, arguments):
        """Validate arguments against tool schema"""
        tool = self.tools[tool_name]
        schema = tool['inputSchema']
        
        try:
            jsonschema.validate(arguments, schema)
        except jsonschema.ValidationError as e:
            raise InvalidArguments(
                f"Validation failed: {e.message}\n"
                f"Path: {'.'.join(str(p) for p in e.path)}"
            )
    
    async def call_tool(self, tool_name, arguments):
        # Validate first
        self.validate_tool_arguments(tool_name, arguments)
        
        # Then execute
        return await super().call_tool(tool_name, arguments)
```

### Custom Validation

```python
class CustomValidationServer(MCPServer):
    def register_tool_with_validator(self, name, schema, handler, validator=None):
        """Register tool with custom validator"""
        
        async def validated_handler(arguments):
            # JSON schema validation
            jsonschema.validate(arguments, schema)
            
            # Custom validation
            if validator:
                errors = validator(arguments)
                if errors:
                    raise ValidationError(
                        f"Custom validation failed: {', '.join(errors)}"
                    )
            
            # Execute if valid
            return await handler(arguments)
        
        self.register_tool(name, schema, validated_handler)

# Usage
def validate_email_send(arguments):
    """Custom validation for email sending"""
    errors = []
    
    # Check email format
    if '@' not in arguments['to']:
        errors.append("Invalid email address")
    
    # Check body length
    if len(arguments['body']) > 10000:
        errors.append("Email body too long (max 10000 chars)")
    
    # Check for spam patterns
    spam_words = ['FREE!!!', 'CLICK NOW', 'ACT FAST']
    if any(word in arguments['subject'].upper() for word in spam_words):
        errors.append("Subject contains spam-like content")
    
    return errors

server.register_tool_with_validator(
    name="send_email",
    schema={...},
    handler=send_email_handler,
    validator=validate_email_send
)
```

## Output Formatting

### Text Output

```python
async def search_handler(arguments):
    """Return formatted text output"""
    results = await search(arguments['query'])
    
    # Format as text
    output = "Search Results:\n\n"
    for i, result in enumerate(results, 1):
        output += f"{i}. {result['title']}\n"
        output += f"   URL: {result['url']}\n"
        output += f"   {result['snippet']}\n\n"
    
    return output  # Will be wrapped in text content type
```

### Structured Output

```python
async def analyze_handler(arguments):
    """Return structured output"""
    analysis = await analyze_data(arguments['data'])
    
    # Return as structured data
    return {
        "summary": {
            "total_records": analysis['count'],
            "average_value": analysis['avg'],
            "max_value": analysis['max'],
            "min_value": analysis['min']
        },
        "insights": analysis['insights'],
        "recommendations": analysis['recommendations']
    }
    # Will be JSON-formatted
```

### Multiple Content Types

```python
async def advanced_handler(arguments):
    """Return multiple content types"""
    
    # Generate different content types
    summary_text = "Analysis complete."
    detailed_data = {"results": [...]}
    chart_image = generate_chart_base64()
    
    # Return multiple content items
    return [
        {
            "type": "text",
            "text": summary_text
        },
        {
            "type": "text",
            "text": json.dumps(detailed_data, indent=2)
        },
        {
            "type": "image",
            "data": chart_image,
            "mimeType": "image/png"
        }
    ]
```

### Resource References

```python
async def report_handler(arguments):
    """Return resource reference in output"""
    
    # Generate report and save to file
    report_path = await generate_report(arguments)
    
    # Return reference to resource
    return {
        "content": [
            {
                "type": "text",
                "text": "Report generated successfully"
            },
            {
                "type": "resource",
                "resource": {
                    "uri": f"file://{report_path}",
                    "text": "View full report",
                    "mimeType": "application/pdf"
                }
            }
        ]
    }
```

## Error Handling

### Error Types

```python
class ToolError(Exception):
    """Base class for tool errors"""
    pass

class ToolNotFound(ToolError):
    """Tool doesn't exist"""
    pass

class InvalidArguments(ToolError):
    """Invalid arguments provided"""
    pass

class ToolExecutionError(ToolError):
    """Error during tool execution"""
    pass

class ToolTimeout(ToolError):
    """Tool execution timed out"""
    pass

class InsufficientPermissions(ToolError):
    """Caller doesn't have permission"""
    pass
```

### Graceful Error Handling

```python
class RobustMCPServer(MCPServer):
    async def safe_execute_tool(self, tool_name, arguments):
        """Execute tool with comprehensive error handling"""
        
        try:
            # Validate tool exists
            if tool_name not in self.tools:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Error: Tool '{tool_name}' not found"
                    }],
                    "isError": True
                }
            
            # Validate arguments
            try:
                self.validate_arguments(arguments, self.tools[tool_name]['schema'])
            except ValidationError as e:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Invalid arguments: {str(e)}"
                    }],
                    "isError": True
                }
            
            # Execute with timeout
            try:
                result = await asyncio.wait_for(
                    self.tools[tool_name]['handler'](arguments),
                    timeout=30.0
                )
                
                return {
                    "content": [{
                        "type": "text",
                        "text": str(result)
                    }],
                    "isError": False
                }
            
            except asyncio.TimeoutError:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Tool '{tool_name}' timed out after 30 seconds"
                    }],
                    "isError": True
                }
            
            except Exception as e:
                # Log error for debugging
                logger.error(f"Tool {tool_name} failed: {e}", exc_info=True)
                
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Tool execution failed: {str(e)}"
                    }],
                    "isError": True
                }
        
        except Exception as e:
            # Catch-all for unexpected errors
            logger.critical(f"Unexpected error in tool execution: {e}", exc_info=True)
            
            return {
                "content": [{
                    "type": "text",
                    "text": "An unexpected error occurred. Please try again."
                }],
                "isError": True
            }
```

### Partial Success

```python
async def batch_operation_handler(arguments):
    """Tool that processes multiple items, some may fail"""
    items = arguments['items']
    results = []
    errors = []
    
    for item in items:
        try:
            result = await process_item(item)
            results.append({
                "item": item,
                "status": "success",
                "result": result
            })
        except Exception as e:
            errors.append({
                "item": item,
                "status": "error",
                "error": str(e)
            })
    
    # Return combined results
    output = {
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors
    }
    
    return {
        "content": [{
            "type": "text",
            "text": json.dumps(output, indent=2)
        }],
        "isError": len(errors) > 0  # Partial failure
    }
```

## Tool Categories

### Information Retrieval Tools

```python
class InformationTools(MCPServer):
    """Tools for finding and retrieving information"""
    
    @tool("web_search")
    async def web_search(self, query, max_results=10):
        """Search the web"""
        pass
    
    @tool("database_query")
    async def database_query(self, sql):
        """Query database"""
        pass
    
    @tool("file_search")
    async def file_search(self, pattern, directory):
        """Search files"""
        pass
    
    @tool("api_fetch")
    async def api_fetch(self, url, headers=None):
        """Fetch from API"""
        pass
```

### Communication Tools

```python
class CommunicationTools(MCPServer):
    """Tools for communication"""
    
    @tool("send_email")
    async def send_email(self, to, subject, body):
        """Send email"""
        pass
    
    @tool("send_slack_message")
    async def send_slack_message(self, channel, text):
        """Post to Slack"""
        pass
    
    @tool("create_notification")
    async def create_notification(self, title, message, priority="normal"):
        """Create notification"""
        pass
```

### Data Manipulation Tools

```python
class DataTools(MCPServer):
    """Tools for data manipulation"""
    
    @tool("transform_data")
    async def transform_data(self, data, transformation):
        """Transform data structure"""
        pass
    
    @tool("aggregate_data")
    async def aggregate_data(self, data, group_by, aggregations):
        """Aggregate data"""
        pass
    
    @tool("filter_data")
    async def filter_data(self, data, conditions):
        """Filter data"""
        pass
```

### File Operations Tools

```python
class FileTools(MCPServer):
    """Tools for file operations"""
    
    @tool("create_file")
    async def create_file(self, path, content):
        """Create new file"""
        pass
    
    @tool("update_file")
    async def update_file(self, path, content):
        """Update existing file"""
        pass
    
    @tool("delete_file")
    async def delete_file(self, path):
        """Delete file"""
        pass
    
    @tool("move_file")
    async def move_file(self, source, destination):
        """Move/rename file"""
        pass
```

### Analysis Tools

```python
class AnalysisTools(MCPServer):
    """Tools for data analysis"""
    
    @tool("sentiment_analysis")
    async def sentiment_analysis(self, text):
        """Analyze sentiment"""
        pass
    
    @tool("summarize_text")
    async def summarize_text(self, text, max_length=100):
        """Summarize text"""
        pass
    
    @tool("extract_entities")
    async def extract_entities(self, text):
        """Extract named entities"""
        pass
    
    @tool("statistical_analysis")
    async def statistical_analysis(self, data, metrics):
        """Compute statistics"""
        pass
```

## Tool Composition

### Sequential Composition

```python
async def composed_workflow(client, query):
    """Compose multiple tools sequentially"""
    
    # Step 1: Search for information
    search_results = await client.call_tool(
        "web_search",
        {"query": query, "max_results": 5}
    )
    
    # Step 2: Extract URLs from results
    urls = extract_urls(search_results)
    
    # Step 3: Fetch content from each URL
    contents = []
    for url in urls:
        content = await client.call_tool(
            "fetch_url",
            {"url": url}
        )
        contents.append(content)
    
    # Step 4: Summarize all content
    combined = "\n\n".join(contents)
    summary = await client.call_tool(
        "summarize_text",
        {"text": combined, "max_length": 500}
    )
    
    # Step 5: Generate report
    report = await client.call_tool(
        "generate_report",
        {"summary": summary, "sources": urls}
    )
    
    return report
```

### Parallel Composition

```python
async def parallel_analysis(client, text):
    """Run multiple analyses in parallel"""
    
    # Execute multiple tools simultaneously
    results = await asyncio.gather(
        client.call_tool("sentiment_analysis", {"text": text}),
        client.call_tool("extract_entities", {"text": text}),
        client.call_tool("summarize_text", {"text": text}),
        client.call_tool("detect_language", {"text": text}),
        return_exceptions=True
    )
    
    sentiment, entities, summary, language = results
    
    # Combine results
    return {
        "sentiment": sentiment if not isinstance(sentiment, Exception) else None,
        "entities": entities if not isinstance(entities, Exception) else None,
        "summary": summary if not isinstance(summary, Exception) else None,
        "language": language if not isinstance(language, Exception) else None
    }
```

### Conditional Composition

```python
async def conditional_workflow(client, document):
    """Compose tools conditionally based on results"""
    
    # Analyze document type
    doc_type = await client.call_tool(
        "detect_document_type",
        {"document": document}
    )
    
    # Different processing based on type
    if doc_type == "legal":
        # Legal document processing
        entities = await client.call_tool(
            "extract_legal_entities",
            {"document": document}
        )
        summary = await client.call_tool(
            "legal_summarize",
            {"document": document}
        )
    
    elif doc_type == "financial":
        # Financial document processing
        entities = await client.call_tool(
            "extract_financial_data",
            {"document": document}
        )
        summary = await client.call_tool(
            "financial_summarize",
            {"document": document}
        )
    
    else:
        # Generic processing
        entities = await client.call_tool(
            "extract_entities",
            {"text": document}
        )
        summary = await client.call_tool(
            "summarize_text",
            {"text": document}
        )
    
    return {"type": doc_type, "entities": entities, "summary": summary}
```

### Tool Chaining Server-Side

```python
class ComposingMCPServer(MCPServer):
    """Server that provides composed tools"""
    
    @tool("research_and_summarize")
    async def research_and_summarize(self, topic):
        """Composite tool that chains other tools"""
        
        # Internal tool composition
        search_results = await self.web_search({"query": topic})
        
        urls = self.extract_urls(search_results)
        
        contents = []
        for url in urls[:3]:  # Top 3 results
            try:
                content = await self.fetch_url({"url": url})
                contents.append(content)
            except:
                continue
        
        combined = "\n\n".join(contents)
        summary = await self.summarize_text({"text": combined})
        
        return {
            "topic": topic,
            "sources": urls[:3],
            "summary": summary
        }
```

## Tool Lifecycle

### Tool Initialization

```python
class LifecycleAwareMCPServer(MCPServer):
    async def startup(self):
        """Initialize tools on server startup"""
        
        # Connect to external services
        self.db = await database.connect()
        self.cache = await redis.connect()
        self.api_client = APIClient(api_key=os.getenv('API_KEY'))
        
        # Initialize tool handlers
        self.register_tools()
        
        logger.info("All tools initialized")
    
    async def shutdown(self):
        """Cleanup on server shutdown"""
        
        # Close connections
        await self.db.close()
        await self.cache.close()
        await self.api_client.close()
        
        logger.info("All tools cleaned up")
```

### Tool State Management

```python
class StatefulTool:
    """Tool with persistent state"""
    
    def __init__(self):
        self.state = {}
        self.call_count = 0
    
    async def execute(self, arguments):
        self.call_count += 1
        
        # Use state from previous calls
        if 'context' in self.state:
            # Continue from previous context
            result = await self.process_with_context(
                arguments,
                self.state['context']
            )
        else:
            # First call
            result = await self.process_fresh(arguments)
        
        # Update state
        self.state['context'] = result['context']
        self.state['last_call'] = time.time()
        
        return result['output']
```

### Tool Versioning

```python
class VersionedToolServer(MCPServer):
    """Server with versioned tools"""
    
    def register_tools(self):
        # Register v1 of tool
        self.register_tool(
            name="analyze_v1",
            description="Analysis tool (v1)",
            schema=self.get_analyze_v1_schema(),
            handler=self.analyze_v1_handler
        )
        
        # Register v2 of tool (improved)
        self.register_tool(
            name="analyze_v2",
            description="Analysis tool (v2 - enhanced)",
            schema=self.get_analyze_v2_schema(),
            handler=self.analyze_v2_handler
        )
        
        # Register latest (alias to v2)
        self.register_tool(
            name="analyze",
            description="Analysis tool (latest)",
            schema=self.get_analyze_v2_schema(),
            handler=self.analyze_v2_handler
        )
```

## Advanced Tool Patterns

### Streaming Tools

```python
class StreamingMCPServer(MCPServer):
    """Tools that stream results"""
    
    async def streaming_search_handler(self, arguments):
        """Stream search results as they arrive"""
        query = arguments['query']
        
        async for result in self.search_service.stream_search(query):
            # Yield results incrementally
            yield {
                "type": "text",
                "text": f"Found: {result['title']}\n{result['url']}\n\n"
            }
```

### Progress Reporting Tools

```python
class ProgressReportingTool:
    """Tool that reports progress"""
    
    async def execute(self, arguments, progress_callback=None):
        items = arguments['items']
        total = len(items)
        
        results = []
        
        for i, item in enumerate(items):
            # Process item
            result = await self.process_item(item)
            results.append(result)
            
            # Report progress
            if progress_callback:
                await progress_callback({
                    "completed": i + 1,
                    "total": total,
                    "percent": ((i + 1) / total) * 100
                })
        
        return results
```

### Cacheable Tools

```python
class CacheableMCPServer(MCPServer):
    """Server with result caching"""
    
    def __init__(self):
        super().__init__("cacheable")
        self.cache = {}
    
    def cacheable_tool(self, name, schema, handler, ttl=300):
        """Register tool with caching"""
        
        async def cached_handler(arguments):
            # Generate cache key
            cache_key = f"{name}:{json.dumps(arguments, sort_keys=True)}"
            
            # Check cache
            if cache_key in self.cache:
                cached_result, timestamp = self.cache[cache_key]
                if time.time() - timestamp < ttl:
                    return cached_result
            
            # Execute tool
            result = await handler(arguments)
            
            # Cache result
            self.cache[cache_key] = (result, time.time())
            
            return result
        
        self.register_tool(name, schema, cached_handler)
```

### Retry Logic

```python
async def resilient_tool_call(client, tool_name, arguments, max_retries=3):
    """Call tool with automatic retry"""
    
    for attempt in range(max_retries):
        try:
            result = await client.call_tool(tool_name, arguments)
            return result
        
        except ToolExecutionError as e:
            if attempt == max_retries - 1:
                raise
            
            # Exponential backoff
            wait_time = 2 ** attempt
            logger.warning(
                f"Tool call failed (attempt {attempt + 1}/{max_retries}), "
                f"retrying in {wait_time}s: {e}"
            )
            await asyncio.sleep(wait_time)
```

## Tool Performance

### Performance Monitoring

```python
class MonitoredMCPServer(MCPServer):
    """Server with performance monitoring"""
    
    def __init__(self):
        super().__init__("monitored")
        self.metrics = {
            'call_count': {},
            'total_duration': {},
            'error_count': {}
        }
    
    async def monitored_tool_call(self, tool_name, arguments):
        """Execute tool with monitoring"""
        
        start_time = time.time()
        
        try:
            result = await self.tools[tool_name]['handler'](arguments)
            
            # Record success
            self.metrics['call_count'][tool_name] = \
                self.metrics['call_count'].get(tool_name, 0) + 1
            
            duration = time.time() - start_time
            self.metrics['total_duration'][tool_name] = \
                self.metrics['total_duration'].get(tool_name, 0) + duration
            
            return result
        
        except Exception as e:
            # Record error
            self.metrics['error_count'][tool_name] = \
                self.metrics['error_count'].get(tool_name, 0) + 1
            raise
    
    def get_tool_stats(self, tool_name):
        """Get performance statistics for a tool"""
        calls = self.metrics['call_count'].get(tool_name, 0)
        total_duration = self.metrics['total_duration'].get(tool_name, 0)
        errors = self.metrics['error_count'].get(tool_name, 0)
        
        return {
            'total_calls': calls,
            'average_duration': total_duration / calls if calls > 0 else 0,
            'error_rate': errors / calls if calls > 0 else 0
        }
```

### Rate Limiting

```python
class RateLimitedMCPServer(MCPServer):
    """Server with rate limiting"""
    
    def __init__(self):
        super().__init__("rate-limited")
        self.rate_limits = {}
    
    def rate_limited_tool(self, name, schema, handler, calls_per_minute=60):
        """Register tool with rate limiting"""
        
        self.rate_limits[name] = {
            'limit': calls_per_minute,
            'calls': [],
            'blocked_until': 0
        }
        
        async def limited_handler(arguments):
            now = time.time()
            limit_info = self.rate_limits[name]
            
            # Check if currently blocked
            if now < limit_info['blocked_until']:
                raise ToolError("Rate limit exceeded, please try again later")
            
            # Remove old calls
            limit_info['calls'] = [
                t for t in limit_info['calls']
                if now - t < 60  # Keep calls from last minute
            ]
            
            # Check limit
            if len(limit_info['calls']) >= limit_info['limit']:
                # Block for remainder of minute
                oldest_call = min(limit_info['calls'])
                limit_info['blocked_until'] = oldest_call + 60
                raise ToolError("Rate limit exceeded")
            
            # Record call
            limit_info['calls'].append(now)
            
            # Execute tool
            return await handler(arguments)
        
        self.register_tool(name, schema, limited_handler)
```

## Tool Security

### Input Sanitization

```python
class SecureMCPServer(MCPServer):
    """Server with security measures"""
    
    def sanitize_input(self, value, value_type):
        """Sanitize input values"""
        
        if value_type == 'string':
            # Remove potentially dangerous characters
            return re.sub(r'[<>\"\'&]', '', value)
        
        elif value_type == 'path':
            # Prevent path traversal
            path = Path(value).resolve()
            if '..' in path.parts:
                raise SecurityError("Path traversal attempt detected")
            return str(path)
        
        elif value_type == 'sql':
            # Basic SQL injection prevention
            if re.search(r';\s*(DROP|DELETE|UPDATE|INSERT)', value, re.I):
                raise SecurityError("Potentially malicious SQL detected")
            return value
        
        return value
    
    async def secure_tool_handler(self, arguments):
        """Tool handler with input sanitization"""
        # Sanitize all inputs
        sanitized = {
            k: self.sanitize_input(v, self.get_type(k))
            for k, v in arguments.items()
        }
        
        # Execute with sanitized inputs
        return await self.unsafe_tool_handler(sanitized)
```

### Permission-Based Access

```python
class PermissionBasedServer(MCPServer):
    """Server with permission-based tool access"""
    
    def __init__(self):
        super().__init__("permissions")
        self.tool_permissions = {}
    
    def register_protected_tool(self, name, schema, handler, required_permission):
        """Register tool with permission requirement"""
        
        self.tool_permissions[name] = required_permission
        
        async def protected_handler(client_id, arguments):
            # Check permission
            if not self.has_permission(client_id, required_permission):
                raise PermissionError(
                    f"Tool '{name}' requires permission: {required_permission}"
                )
            
            # Execute if authorized
            return await handler(arguments)
        
        self.register_tool(name, schema, protected_handler)
    
    def has_permission(self, client_id, permission):
        """Check if client has permission"""
        client_permissions = self.get_client_permissions(client_id)
        return permission in client_permissions
```

## Summary

MCP tools provide standardized action execution for agents:

**Core Concepts**:
- Tools represent **actions** (vs resources = data)
- Defined by **JSON schemas**
- **Discovered dynamically**
- **Invoked with validation**
- **Return structured results**

**Tool Lifecycle**:
1. **Registration** - Server exposes tools
2. **Discovery** - Client lists available tools
3. **Validation** - Arguments checked against schema
4. **Execution** - Server runs tool logic
5. **Response** - Results returned to client

**Advanced Patterns**:
- **Composition** - Chain multiple tools
- **Caching** - Optimize repeated calls
- **Rate limiting** - Control usage
- **Monitoring** - Track performance
- **Security** - Sanitize inputs, check permissions

**Best Practices**:
- Clear, descriptive tool names
- Comprehensive schemas with good descriptions
- Graceful error handling
- Performance monitoring
- Security-first design

Tools are the "verbs" of MCP - they enable agents to take action and interact with the world.

## Next Steps

- **[Prompts and Templates](mcp-prompts.md)** - Explore reusable prompts
- **[Resources](mcp-resources.md)** - Understand data access
- **[Context Management](mcp-context.md)** - Manage LLM context
- **[Building MCP Servers](building-mcp-servers.md)** - Implement tools in practice
- **[MCP Fundamentals](mcp-fundamentals.md)** - Review core concepts
