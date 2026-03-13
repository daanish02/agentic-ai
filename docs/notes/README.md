# Tutorial Notes

## About This Section

This section contains practical implementation examples and documentation derived from hands-on tutorials. Unlike the conceptual guides in other sections, these are **working code examples** with accompanying documentation that capture real-world patterns for building agentic systems.

The code here represents learning-by-doing: implementing MCP servers, building agents with LangChain, and orchestrating workflows with LangGraph. Each subfolder contains both the original tutorial code and consolidated documentation that extracts key patterns and best practices. These examples bridge the gap between theory and practice, showing how concepts from other sections translate into real implementations.

## Contents

### [FastMCP](fastmcp/)

Practical MCP (Model Context Protocol) server and client implementations using the FastMCP library. Covers local and remote server patterns, different transport mechanisms (stdio, HTTP, SSE), database integration, and deployment strategies. Includes an expense tracker example with SQLite, multi-server clients, and proxy patterns for production use.

### [LangChain](langchain/)

Classic LangChain patterns for agent and tool development. Covers the ReAct agent framework, tool definition approaches (@tool decorator, StructuredTool, BaseTool), and integration patterns using LangChain's established abstractions. Includes autonomous agents with search and weather APIs, comprehensive tool validation patterns, and migration guidance to LangGraph.

### [LangGraph](langgraph/)

Graph-based workflow orchestration with LangGraph. Covers state management with TypedDict, checkpointing (InMemorySaver and PostgresSaver), memory systems (short-term and long-term), human-in-the-loop patterns, subgraph composition, tool integration, RAG patterns, MCP integration, fault tolerance mechanisms with idempotent nodes, and comprehensive debugging approaches. Provides explicit control over execution flow with full observability and recovery capabilities.
