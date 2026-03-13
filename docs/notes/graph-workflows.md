# Graph-Based Workflow Patterns

## Table of Contents

- [Introduction](#introduction)
- [Workflows vs Agents](#workflows-vs-agents)
- [State Management Fundamentals](#state-management-fundamentals)
- [Sequential Workflows](#sequential-workflows)
- [Parallel Workflows](#parallel-workflows)
- [Conditional Routing](#conditional-routing)
- [Checkpointing and Persistence](#checkpointing-and-persistence)
- [Short-Term Memory Patterns](#short-term-memory-patterns)
- [Long-Term Memory Patterns](#long-term-memory-patterns)
- [Human-in-the-Loop](#human-in-the-loop)
- [Subgraph Patterns](#subgraph-patterns)
- [Tool Integration](#tool-integration)
- [RAG as a Tool](#rag-as-a-tool)
- [MCP Integration](#mcp-integration)
- [Debugging and Observability](#debugging-and-observability)
- [Fault Tolerance and Recovery](#fault-tolerance-and-recovery)
- [Best Practices](#best-practices)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Graph-based workflows are a powerful pattern for orchestrating complex multi-step AI operations. Unlike simple agent loops that rely purely on LLM decision-making, workflows provide explicit structure, deterministic control flow, and sophisticated state management.

> "Workflows bring structure to chaos, turning unpredictable agent behavior into reliable systems."

This guide covers practical patterns for building production-grade graph workflows using LangGraph. We'll explore:

- **State management**: TypedDict schemas, reducers, and state transitions
- **Workflow patterns**: Sequential, parallel, and conditional execution
- **Memory systems**: Short-term and long-term memory implementation
- **Integration patterns**: Tools, MCP servers, and RAG systems
- **Advanced features**: Human-in-the-loop, subgraphs, and debugging

### Why Graph Workflows?

**Without structured workflows**:

- Unpredictable execution paths
- Difficult to debug failures
- Hard to maintain conversation context
- Limited control over agent behavior

**With graph workflows**:

- Explicit, understandable control flow
- Debuggable with state inspection
- Sophisticated memory management
- Fine-grained control over execution

### Workflow Architecture

```
┌─────────────────────────────────────────┐
│         Graph Definition                 │
│  - Nodes (processing steps)              │
│  - Edges (transitions)                   │
│  - State schema                          │
└─────────────┬───────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│         Compiled Workflow                │
│  - Execution engine                      │
│  - State management                      │
│  - Checkpointing                         │
└─────────────┬───────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│         Runtime Execution                │
│  - Process inputs                        │
│  - Update state                          │
│  - Generate outputs                      │
└─────────────────────────────────────────┘
```

## Workflows vs Agents

Understanding the distinction helps you choose the right pattern.

### Workflows: Structured and Deterministic

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class BlogState(TypedDict):
    title: str
    outline: str
    content: str

# Workflow with explicit steps
graph = StateGraph(BlogState)

# Step 1: Create outline
graph.add_node("create_outline", create_outline_func)

# Step 2: Write blog
graph.add_node("write_blog", write_blog_func)

# Explicit flow
graph.add_edge(START, "create_outline")
graph.add_edge("create_outline", "write_blog")
graph.add_edge("write_blog", END)

# Deterministic execution
workflow = graph.compile()
```

**Characteristics**:

- Explicit control flow
- Predictable execution paths
- Structured state transitions
- Easier to debug and test

### Agents: Dynamic and Autonomous

```python
# Agent makes decisions at each step
while not done:
    # Agent decides what to do next
    action = agent.decide(state)

    # Execute chosen action
    result = execute(action)

    # Agent evaluates if done
    done = agent.is_complete(result)
```

**Characteristics**:

- LLM-driven decisions
- Dynamic execution paths
- Autonomous behavior
- Flexible but less predictable

### When to Use Each

**Use workflows when**:

- Need predictable execution
- Have well-defined steps
- Require strong guarantees
- Building production systems

**Use agents when**:

- Tasks are open-ended
- Need flexible problem-solving
- Don't know steps in advance
- Exploring new capabilities

**Hybrid approach** (best of both):

- Workflow structure for reliability
- Agent nodes for flexibility

```python
# Workflow with agent node
graph.add_node("plan", agent_planning_node)  # Agent decides
graph.add_node("execute", deterministic_execution)  # Workflow runs
graph.add_edge("plan", "execute")
```

## State Management Fundamentals

State is the backbone of graph workflows.

### TypedDict State Schema

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class ChatState(TypedDict):
    """State schema for chatbot workflow."""
    messages: Annotated[list[BaseMessage], add_messages]
```

**Key concepts**:

- `TypedDict`: Defines state structure
- `Annotated`: Adds metadata (reducers)
- `add_messages`: Reducer for message lists

### State Reducers

Reducers control how state updates are merged.

**add_messages reducer**:

```python
from langgraph.graph.message import add_messages

class State(TypedDict):
    # Messages are appended, not replaced
    messages: Annotated[list[BaseMessage], add_messages]

# Usage in node
def chat_node(state: State):
    response = llm.invoke(state["messages"])
    # This APPENDS to messages, doesn't replace
    return {"messages": [response]}
```

**Custom reducer**:

```python
import operator
from typing import Annotated

class ReviewState(TypedDict):
    review: str
    # Scores are added together
    individual_scores: Annotated[list[int], operator.add]
    avg_score: float

# Node 1 returns: {"individual_scores": [8]}
# Node 2 returns: {"individual_scores": [9]}
# Final state: {"individual_scores": [8, 9]}
```

### State Updates

Nodes can update state in different ways:

```python
def my_node(state: MyState):
    # Full update - all fields
    return {
        "field1": "value1",
        "field2": "value2"
    }

def my_node(state: MyState):
    # Partial update - only specified fields
    return {"field1": "value1"}
    # field2 remains unchanged

def my_node(state: MyState):
    # No update - return empty dict
    return {}
```

## Sequential Workflows

Sequential workflows execute steps in order.

### Basic Sequential Pattern

```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict

model = ChatOpenAI(model="gpt-4o-mini")

class BlogState(TypedDict):
    title: str
    outline: str
    content: str

def create_outline(state: BlogState) -> BlogState:
    """Step 1: Generate blog outline."""
    prompt = f"Create a detailed outline for a blog about: {state['title']}"
    outline = model.invoke(prompt).content
    return {"outline": outline}

def write_blog(state: BlogState) -> BlogState:
    """Step 2: Write blog from outline."""
    prompt = f"Write a detailed blog post about '{state['title']}' using this outline:\n{state['outline']}"
    content = model.invoke(prompt).content
    return {"content": content}

# Build graph
graph = StateGraph(BlogState)

# Add nodes in sequence
graph.add_node("create_outline", create_outline)
graph.add_node("write_blog", write_blog)

# Define sequential flow
graph.add_edge(START, "create_outline")
graph.add_edge("create_outline", "write_blog")
graph.add_edge("write_blog", END)

# Compile and run
workflow = graph.compile()

result = workflow.invoke({"title": "The Future of AI in Healthcare"})
print(result["content"])
```

### Multi-Step Sequential Workflow

```python
class ResearchState(TypedDict):
    topic: str
    research_notes: str
    outline: str
    draft: str
    final_article: str

def research(state: ResearchState):
    prompt = f"Research key points about: {state['topic']}"
    notes = model.invoke(prompt).content
    return {"research_notes": notes}

def outline(state: ResearchState):
    prompt = f"Create outline from research:\n{state['research_notes']}"
    outline = model.invoke(prompt).content
    return {"outline": outline}

def draft(state: ResearchState):
    prompt = f"Write draft using outline:\n{state['outline']}"
    draft = model.invoke(prompt).content
    return {"draft": draft}

def polish(state: ResearchState):
    prompt = f"Polish this draft:\n{state['draft']}"
    final = model.invoke(prompt).content
    return {"final_article": final}

# Sequential chain
graph = StateGraph(ResearchState)
graph.add_node("research", research)
graph.add_node("outline", outline)
graph.add_node("draft", draft)
graph.add_node("polish", polish)

graph.add_edge(START, "research")
graph.add_edge("research", "outline")
graph.add_edge("outline", "draft")
graph.add_edge("draft", "polish")
graph.add_edge("polish", END)

workflow = graph.compile()
```

## Parallel Workflows

Execute multiple operations simultaneously (fan-out/fan-in pattern).

### Basic Parallel Pattern

```python
import operator
from typing import Annotated
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

class EvaluationSchema(BaseModel):
    """Schema for essay evaluation."""
    feedback: str = Field(description="Detailed feedback")
    score: int = Field(description="Score out of 10", ge=0, le=10)

structured_model = model.with_structured_output(EvaluationSchema)

class EssayState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]  # Scores are added
    avg_score: float

# Parallel evaluation nodes
def evaluate_language(state: EssayState):
    """Evaluate language quality."""
    prompt = f"Evaluate language quality of:\n{state['essay']}"
    result = structured_model.invoke(prompt)
    return {
        "language_feedback": result.feedback,
        "individual_scores": [result.score]
    }

def evaluate_analysis(state: EssayState):
    """Evaluate depth of analysis."""
    prompt = f"Evaluate analysis depth of:\n{state['essay']}"
    result = structured_model.invoke(prompt)
    return {
        "analysis_feedback": result.feedback,
        "individual_scores": [result.score]
    }

def evaluate_clarity(state: EssayState):
    """Evaluate clarity of thought."""
    prompt = f"Evaluate clarity of thought in:\n{state['essay']}"
    result = structured_model.invoke(prompt)
    return {
        "clarity_feedback": result.feedback,
        "individual_scores": [result.score]
    }

def final_evaluation(state: EssayState):
    """Combine all evaluations."""
    prompt = f"""Summarize these evaluations:
    Language: {state['language_feedback']}
    Analysis: {state['analysis_feedback']}
    Clarity: {state['clarity_feedback']}"""

    overall = model.invoke(prompt).content
    avg = sum(state["individual_scores"]) / len(state["individual_scores"])

    return {
        "overall_feedback": overall,
        "avg_score": avg
    }

# Build parallel graph
graph = StateGraph(EssayState)

# Add evaluation nodes
graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_clarity", evaluate_clarity)
graph.add_node("final_evaluation", final_evaluation)

# Fan-out: START → all evaluation nodes (parallel)
graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_clarity")

# Fan-in: all evaluations → final (waits for all)
graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_clarity", "final_evaluation")

graph.add_edge("final_evaluation", END)

workflow = graph.compile()

# Execute - evaluations run in parallel
result = workflow.invoke({"essay": "Your essay text here..."})
print(f"Average Score: {result['avg_score']}")
print(f"Overall: {result['overall_feedback']}")
```

### Parallel Execution Behavior

```
         START
           │
     ┌─────┼─────┐
     ▼     ▼     ▼
  Node1  Node2  Node3  ← Run in parallel
     │     │     │
     └─────┼─────┘
           ▼
       Aggregator  ← Waits for all
           │
          END
```

## Conditional Routing

Route execution based on state or LLM decisions.

### Basic Conditional Pattern

```python
from typing import Literal
from pydantic import BaseModel, Field

class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the review"
    )

sentiment_model = model.with_structured_output(SentimentSchema)

class ReviewState(TypedDict):
    review: str
    sentiment: Literal["positive", "negative"]
    response: str

def analyze_sentiment(state: ReviewState):
    """Determine review sentiment."""
    prompt = f"Determine sentiment of:\n{state['review']}"
    result = sentiment_model.invoke(prompt)
    return {"sentiment": result.sentiment}

def route_by_sentiment(state: ReviewState) -> Literal["positive_response", "negative_response"]:
    """Route based on sentiment."""
    if state["sentiment"] == "positive":
        return "positive_response"
    else:
        return "negative_response"

def positive_response(state: ReviewState):
    """Handle positive review."""
    prompt = f"Write a thank-you message for:\n{state['review']}"
    response = model.invoke(prompt).content
    return {"response": response}

def negative_response(state: ReviewState):
    """Handle negative review."""
    prompt = f"Write an empathetic resolution message for:\n{state['review']}"
    response = model.invoke(prompt).content
    return {"response": response}

# Build conditional graph
graph = StateGraph(ReviewState)

graph.add_node("analyze_sentiment", analyze_sentiment)
graph.add_node("positive_response", positive_response)
graph.add_node("negative_response", negative_response)

graph.add_edge(START, "analyze_sentiment")

# Conditional routing
graph.add_conditional_edges(
    "analyze_sentiment",
    route_by_sentiment,  # Function returns target node name
    {
        "positive_response": "positive_response",
        "negative_response": "negative_response"
    }
)

graph.add_edge("positive_response", END)
graph.add_edge("negative_response", END)

workflow = graph.compile()
```

### Complex Conditional Logic

```python
class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"]
    urgency: Literal["low", "medium", "high"]

def route_by_diagnosis(state: ReviewState) -> str:
    """Route based on multiple factors."""
    diagnosis = state["diagnosis"]

    # High urgency always goes to escalation
    if diagnosis["urgency"] == "high":
        return "escalate"

    # Route by issue type
    if diagnosis["issue_type"] in ["Bug", "Performance"]:
        return "technical_response"
    elif diagnosis["issue_type"] == "Support":
        return "support_response"
    else:
        return "general_response"

graph.add_conditional_edges(
    "diagnose",
    route_by_diagnosis,
    {
        "escalate": "escalation_node",
        "technical_response": "technical_node",
        "support_response": "support_node",
        "general_response": "general_node"
    }
)
```

## Checkpointing and Persistence

Checkpointers enable conversation memory and fault recovery.

### In-Memory Checkpointer (Development)

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage

def chatbot(state: MessagesState):
    """Simple chatbot node."""
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# Create graph with checkpointer
graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

# Add in-memory checkpointing
checkpointer = InMemorySaver()
workflow = graph.compile(checkpointer=checkpointer)

# Use with thread ID
config = {"configurable": {"thread_id": "user_123"}}

# First conversation
workflow.invoke(
    {"messages": [HumanMessage(content="Hi, I'm Alice")]},
    config=config
)

# Later conversation - remembers context
response = workflow.invoke(
    {"messages": [HumanMessage(content="What's my name?")]},
    config=config
)
# Response: "Your name is Alice"
```

### PostgreSQL Checkpointer (Production)

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user:password@localhost:5432/db"

def chatbot(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

# Use PostgreSQL for persistence
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # Setup database tables
    checkpointer.setup()

    # Compile graph with PostgreSQL checkpointer
    workflow = graph.compile(checkpointer=checkpointer)

    # Conversations persist across restarts
    config = {"configurable": {"thread_id": "user_123"}}

    result = workflow.invoke(
        {"messages": [HumanMessage(content="Remember this: I like pizza")]},
        config=config
    )
```

### Thread Management

```python
# Different threads = different conversations
thread_1 = {"configurable": {"thread_id": "conversation_1"}}
thread_2 = {"configurable": {"thread_id": "conversation_2"}}

# Thread 1
workflow.invoke(
    {"messages": [HumanMessage(content="I'm learning Python")]},
    config=thread_1
)

# Thread 2 - separate context
workflow.invoke(
    {"messages": [HumanMessage(content="I'm learning JavaScript")]},
    config=thread_2
)

# Each thread maintains its own history
```

## Short-Term Memory Patterns

Managing conversation context within the context window.

### Token-Based Trimming

```python
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

MAX_TOKENS = 150

def chat_with_trimming(state: MessagesState):
    """Keep only recent messages that fit in context."""
    # Trim to last N tokens
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=MAX_TOKENS
    )

    print(f"Token count: {count_tokens_approximately(trimmed_messages)}")

    # Use trimmed history
    response = model.invoke(trimmed_messages)
    return {"messages": [response]}

graph = StateGraph(MessagesState)
graph.add_node("chat", chat_with_trimming)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

workflow = graph.compile(checkpointer=InMemorySaver())

# Long conversation - older messages automatically trimmed
```

### Message Deletion Pattern

```python
from langchain.messages import RemoveMessage

def delete_old_messages(state: MessagesState):
    """Delete oldest messages when history grows too long."""
    messages = state["messages"]

    # If more than 4 messages, delete the oldest 3
    if len(messages) >= 5:
        to_remove = messages[:3]
        return {"messages": [RemoveMessage(id=m.id) for m in to_remove]}

    return {}

def chat_node(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(MessagesState)
graph.add_node("chat", chat_node)
graph.add_node("cleanup", delete_old_messages)

graph.add_edge(START, "chat")
graph.add_edge("chat", "cleanup")  # Cleanup after each response
graph.add_edge("cleanup", END)

workflow = graph.compile(checkpointer=InMemorySaver())
```

### Summarization Pattern

```python
from langchain_core.messages import HumanMessage

class ChatState(MessagesState):
    summary: str

def should_summarize(state: ChatState) -> bool:
    """Check if conversation is getting too long."""
    return len(state["messages"]) > 6

def summarize_conversation(state: ChatState):
    """Summarize old messages, keep recent ones."""
    existing_summary = state.get("summary", "")

    # Create summarization prompt
    if existing_summary:
        prompt = f"Existing summary:\n{existing_summary}\n\nExtend the summary with the new messages above."
    else:
        prompt = "Summarize the conversation above."

    messages_for_summary = state["messages"] + [HumanMessage(content=prompt)]
    summary = model.invoke(messages_for_summary).content

    # Keep only last 2 messages verbatim, delete the rest
    messages_to_delete = state["messages"][:-2]

    return {
        "summary": summary,
        "messages": [RemoveMessage(id=m.id) for m in messages_to_delete]
    }

def chat_node(state: ChatState):
    """Chat with summary context."""
    messages = []

    # Add summary as system context
    if state.get("summary"):
        messages.append({
            "role": "system",
            "content": f"Conversation summary:\n{state['summary']}"
        })

    # Add recent messages
    messages.extend(state["messages"])

    response = model.invoke(messages)
    return {"messages": [response]}

# Build graph with conditional summarization
graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_node("summarize", summarize_conversation)

graph.add_edge(START, "chat")
graph.add_conditional_edges(
    "chat",
    should_summarize,
    {
        True: "summarize",
        False: END
    }
)
graph.add_edge("summarize", END)

workflow = graph.compile(checkpointer=InMemorySaver())
```

## Long-Term Memory Patterns

Persisting and retrieving information across sessions.

### Memory Store Pattern

```python
from langgraph.store.memory import InMemoryStore
from langchain_openai import OpenAIEmbeddings

# Create memory store with semantic search
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
store = InMemoryStore(index={"embed": embedding_model, "dims": 768})

# Store user information with namespaces
namespace = ("users", "u1")

store.put(namespace, "1", {"data": "User prefers concise answers"})
store.put(namespace, "2", {"data": "User likes Python examples"})
store.put(namespace, "3", {"data": "User is learning machine learning"})

# Semantic search
items = store.search(
    namespace,
    query="what are user's preferences about explanations",
    limit=3
)

for item in items:
    print(item.value["data"])
```

### Memory Extraction Pattern

```python
import uuid
from pydantic import BaseModel, Field
from typing import List
from langgraph.store.base import BaseStore
from langgraph.graph import MessagesState
from langchain_core.runnables import RunnableConfig

class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory")
    is_new: bool = Field(description="True if new, false if duplicate")

class MemoryDecision(BaseModel):
    should_write: bool
    memories: List[MemoryItem] = Field(default_factory=list)

memory_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
memory_extractor = memory_llm.with_structured_output(MemoryDecision)

MEMORY_PROMPT = """You are responsible for updating user memory.

CURRENT USER DETAILS:
{user_details}

TASK:
- Extract user-specific facts from the latest message
- Mark as is_new=true ONLY if it adds NEW information
- Keep memories atomic and factual
- Return should_write=false if nothing to remember
"""

def remember_node(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    """Extract and store memories from conversation."""
    user_id = config["configurable"]["user_id"]
    namespace = ("user", user_id, "details")

    # Get existing memories
    items = store.search(namespace)
    existing = "\n".join(it.value["data"] for it in items) if items else "(empty)"

    # Extract new memories
    last_message = state["messages"][-1].content
    decision = memory_extractor.invoke([
        {"role": "system", "content": MEMORY_PROMPT.format(user_details=existing)},
        {"role": "user", "content": last_message}
    ])

    # Store new memories
    if decision.should_write:
        for memory in decision.memories:
            if memory.is_new:
                store.put(namespace, str(uuid.uuid4()), {"data": memory.text})

    return {}

def chat_node(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    """Chat with memory context."""
    user_id = config["configurable"]["user_id"]
    namespace = ("user", user_id, "details")

    # Retrieve user memories
    items = store.search(namespace)
    user_details = "\n".join(it.value["data"] for it in items) if items else ""

    # Create prompt with memory context
    system_prompt = f"""You are a helpful assistant with memory.

    User information:
    {user_details or '(no information yet)'}

    Use this context to personalize your responses."""

    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = model.invoke(messages)

    return {"messages": [response]}

# Build graph with memory
graph = StateGraph(MessagesState)
graph.add_node("remember", remember_node)
graph.add_node("chat", chat_node)

graph.add_edge(START, "remember")
graph.add_edge("remember", "chat")
graph.add_edge("chat", END)

store = InMemoryStore()
workflow = graph.compile(store=store)

# Use with user ID
config = {"configurable": {"user_id": "alice"}}

workflow.invoke(
    {"messages": [HumanMessage(content="I'm learning LangGraph")]},
    config=config
)

workflow.invoke(
    {"messages": [HumanMessage(content="I prefer concise examples")]},
    config=config
)

# Later conversation - remembers context
workflow.invoke(
    {"messages": [HumanMessage(content="Show me an example")]},
    config=config
)
# Response uses stored preferences
```

## Human-in-the-Loop

Pause workflow for human approval or input.

### Interrupt Pattern

```python
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

class ChatState(MessagesState):
    pass

def chat_with_approval(state: ChatState):
    """Chat node that requires approval."""
    # Pause for approval
    decision = interrupt({
        "type": "approval",
        "reason": "Model is about to answer user question",
        "question": state["messages"][-1].content,
        "instruction": "Approve this question? yes/no"
    })

    # Check approval
    if decision.get("approved") == "no":
        return {"messages": [AIMessage(content="Question not approved.")]}

    # Approved - proceed
    response = model.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(ChatState)
graph.add_node("chat", chat_with_approval)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

# Checkpointer required for interrupts
checkpointer = InMemorySaver()
workflow = graph.compile(checkpointer=checkpointer)

# First invocation - hits interrupt
config = {"configurable": {"thread_id": "1"}}
result = workflow.invoke(
    {"messages": [HumanMessage(content="Explain quantum physics")]},
    config=config
)

# Check interrupt state
print(result["__interrupt__"])

# Resume with approval
final_result = workflow.invoke(
    Command(resume={"approved": "yes"}),
    config=config
)
print(final_result["messages"][-1].content)
```

### Use Cases for HITL

1. **Approval workflows**: Sensitive operations
2. **Data collection**: Ask user for missing information
3. **Disambiguation**: Multiple valid options
4. **Safety checks**: Review before execution

## Subgraph Patterns

Compose complex workflows from reusable subgraphs.

### Basic Subgraph

```python
from langgraph.graph import StateGraph, START, END

# Shared state
class TranslationState(TypedDict):
    question: str
    answer_eng: str
    answer_urdu: str

# Subgraph for translation
def translate_text(state: TranslationState):
    prompt = f"Translate to Urdu:\n{state['answer_eng']}"
    translated = model.invoke(prompt).content
    return {"answer_urdu": translated}

# Build subgraph
subgraph_builder = StateGraph(TranslationState)
subgraph_builder.add_node("translate", translate_text)
subgraph_builder.add_edge(START, "translate")
subgraph_builder.add_edge("translate", END)
subgraph = subgraph_builder.compile()

# Parent graph uses subgraph
def generate_answer(state: TranslationState):
    prompt = f"Answer this question: {state['question']}"
    answer = model.invoke(prompt).content
    return {"answer_eng": answer}

parent_builder = StateGraph(TranslationState)
parent_builder.add_node("answer", generate_answer)
parent_builder.add_node("translate", subgraph)  # Use subgraph as node

parent_builder.add_edge(START, "answer")
parent_builder.add_edge("answer", "translate")
parent_builder.add_edge("translate", END)

workflow = parent_builder.compile()

result = workflow.invoke({"question": "What is quantum physics?"})
```

### Different State Subgraphs

```python
# Subgraph has different state
class SubState(TypedDict):
    input_text: str
    translated_text: str

def translate(state: SubState):
    translated = model.invoke(f"Translate to Urdu: {state['input_text']}").content
    return {"translated_text": translated}

subgraph_builder = StateGraph(SubState)
subgraph_builder.add_node("translate", translate)
subgraph_builder.add_edge(START, "translate")
subgraph_builder.add_edge("translate", END)
subgraph = subgraph_builder.compile()

# Parent state is different
class ParentState(TypedDict):
    question: str
    answer_eng: str
    answer_urdu: str

def translate_answer(state: ParentState):
    """Adapter that calls subgraph with different state."""
    # Call subgraph with mapped state
    result = subgraph.invoke({"input_text": state["answer_eng"]})
    # Map result back to parent state
    return {"answer_urdu": result["translated_text"]}

parent_builder = StateGraph(ParentState)
parent_builder.add_node("translate", translate_answer)
# ... rest of parent graph
```

### When to Use Subgraphs

**Use subgraphs for**:

- Reusable workflow components
- Complex multi-step operations
- Logical separation of concerns
- Testing individual components

## Tool Integration

Integrate tools into workflows for external capabilities.

### Basic Tool Integration

```python
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
import requests

@tool
def get_stock_price(symbol: str) -> dict:
    """Get current stock price for a symbol."""
    api_key = "your_key"
    url = f"https://www.alphavantage.co/query"
    params = {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": api_key}

    response = requests.get(url, params=params)
    data = response.json()
    quote = data.get("Global Quote", {})

    return {
        "symbol": symbol,
        "price": float(quote.get("05. price", 0)),
        "change": quote.get("10. change percent", "N/A")
    }

@tool
def calculate(first: float, second: float, operation: str) -> dict:
    """Perform arithmetic operation."""
    if operation == "multiply":
        return {"result": first * second}
    elif operation == "add":
        return {"result": first + second}
    else:
        return {"error": "Unsupported operation"}

# Create tools list
tools = [get_stock_price, calculate]
model_with_tools = model.bind_tools(tools)

def chat_node(state: MessagesState):
    """Chat node that can call tools."""
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Build graph with tools
graph = StateGraph(MessagesState)
tool_node = ToolNode(tools)

graph.add_node("chat", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat")

# Conditional routing: if tools needed, execute them
graph.add_conditional_edges("chat", tools_condition)

# After tools, return to chat
graph.add_edge("tools", "chat")

workflow = graph.compile()

# Use with tool calls
result = workflow.invoke({
    "messages": [HumanMessage(
        content="Get Apple stock price then calculate the cost of 50 shares"
    )]
})
print(result["messages"][-1].content)
```

### How Tool Integration Works

```
User query
    ↓
Chat Node (LLM with tools)
    ↓
Decision: Need tool?
    ├─ No → END
    └─ Yes → Tool Node
              ↓
         Execute tools
              ↓
         Back to Chat Node
              ↓
         Generate response with results
```

## RAG as a Tool

Implement Retrieval-Augmented Generation as a tool.

### RAG Tool Pattern

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool

# Setup RAG
loader = PyPDFLoader("document.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

@tool
def rag_search(query: str) -> dict:
    """Retrieve relevant information from the document database.

    Use this when users ask factual or conceptual questions
    that might be answered from stored documents.
    """
    results = retriever.invoke(query)

    return {
        "query": query,
        "context": [doc.page_content for doc in results],
        "metadata": [doc.metadata for doc in results]
    }

# Integrate RAG tool into workflow
tools = [rag_search]
model_with_tools = model.bind_tools(tools)

def chat_node(state: MessagesState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(MessagesState)
tool_node = ToolNode(tools)

graph.add_node("chat", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")

workflow = graph.compile()

# Query uses RAG
result = workflow.invoke({
    "messages": [HumanMessage(
        content="Using the document, explain machine learning workflows"
    )]
})
```

## MCP Integration

Connect MCP servers to workflows.

### MCP Client Pattern

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

async def build_mcp_workflow():
    """Build workflow with MCP tools."""
    # Connect to MCP servers
    client = MultiServerMCPClient({
        "deepwiki": {
            "transport": "http",
            "url": "https://mcp.deepwiki.com/mcp"
        }
    })

    # Get tools from MCP servers
    tools = await client.get_tools()
    model_with_tools = model.bind_tools(tools)

    async def chat_node(state: MessagesState):
        response = await model_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    # Build graph with MCP tools
    graph = StateGraph(MessagesState)
    tool_node = ToolNode(tools)

    graph.add_node("chat", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")

    return graph.compile()

async def main():
    workflow = await build_mcp_workflow()

    result = await workflow.ainvoke({
        "messages": [HumanMessage(
            content="Show documentation topics for facebook/react"
        )]
    })

    print(result["messages"][-1].content)

asyncio.run(main())
```

## Debugging and Observability

Tools for understanding and debugging workflows.

### State Inspection

```python
# Get current state
config = {"configurable": {"thread_id": "1"}}
state = workflow.get_state(config)

print(state.values)  # Current state values
print(state.next)     # Next nodes to execute
print(state.config)   # Configuration
print(state.metadata) # Metadata
```

### State History (Time Travel)

```python
# Get state history
history = list(workflow.get_state_history(config))

for entry in history:
    print(f"Checkpoint: {entry.config['configurable']['checkpoint_id']}")
    print(f"State: {entry.values}")
    print("---")
```

### Replay from Checkpoint

```python
# Go back to specific checkpoint
checkpoint_config = {
    "configurable": {
        "thread_id": "1",
        "checkpoint_id": "specific_checkpoint_id"
    }
}

# Replay from that point
result = workflow.invoke(None, config=checkpoint_config)
```

### Update and Branch

```python
# Update state at checkpoint
workflow.update_state(
    checkpoint_config,
    {"field": "new_value"}
)

# Continue from updated state
result = workflow.invoke(None, config=checkpoint_config)
```

### Visualize Graph

```python
# Generate graph visualization
image_bytes = workflow.get_graph().draw_mermaid_png()

with open("workflow_graph.png", "wb") as f:
    f.write(image_bytes)
```

## Fault Tolerance and Recovery

Production workflows need robust error handling and recovery mechanisms. LangGraph's checkpointing system enables sophisticated fault tolerance patterns.

### Idempotent Node Design

Design nodes to safely re-execute after crashes by tracking completion state.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    step1_done: bool
    step2_done: bool
    attempt: int

def step1(state: State) -> dict:
    # Check if already completed
    if state["step1_done"]:
        print("Step 1 already done, skipping")
        return {}

    print("Executing step 1...")
    # Perform work
    # ...

    # Mark complete
    return {"step1_done": True}

def step2(state: State) -> dict:
    # Check if already completed
    if state["step2_done"]:
        print("Step 2 already done, skipping")
        return {}

    print(f"Executing step 2 (attempt {state['attempt']})...")

    # Simulate crash on first attempt
    if state["attempt"] == 0:
        raise RuntimeError("Simulated failure")

    # Mark complete
    return {"step2_done": True}

# Build graph
graph = StateGraph(State)
graph.add_node("step1", step1)
graph.add_node("step2", step2)

graph.set_entry_point("step1")
graph.add_edge("step1", "step2")
graph.add_edge("step2", END)

# Compile with checkpointer
workflow = graph.compile(checkpointer=InMemorySaver())

# First attempt - will crash at step2
try:
    workflow.invoke(
        {"step1_done": False, "step2_done": False, "attempt": 0},
        config={"configurable": {"thread_id": "t1"}},
    )
except RuntimeError:
    print("Workflow crashed, but checkpoint saved")

# Recovery - increment attempt counter and retry
result = workflow.invoke(
    {"attempt": 1},
    config={"configurable": {"thread_id": "t1"}},
)

print("Workflow recovered and completed successfully")
```

**Output:**

```
Executing step 1...
Executing step 2 (attempt 0)...
Workflow crashed, but checkpoint saved
Step 1 already done, skipping
Executing step 2 (attempt 1)...
Workflow recovered and completed successfully
```

### Key Fault Tolerance Patterns

#### 1. Completion Flags

Track which steps have completed to enable safe re-execution:

```python
class RobustState(TypedDict):
    data_fetched: bool
    data_processed: bool
    data_stored: bool
    raw_data: dict
    processed_data: dict

def fetch_data(state: RobustState) -> dict:
    if state.get("data_fetched"):
        return {}

    # Expensive API call
    data = expensive_api_call()
    return {"raw_data": data, "data_fetched": True}

def process_data(state: RobustState) -> dict:
    if state.get("data_processed"):
        return {}

    # CPU-intensive processing
    result = process(state["raw_data"])
    return {"processed_data": result, "data_processed": True}

def store_data(state: RobustState) -> dict:
    if state.get("data_stored"):
        return {}

    # Database write
    database.save(state["processed_data"])
    return {"data_stored": True}
```

#### 2. Retry Counter

Track attempt numbers for exponential backoff or circuit breakers:

```python
class RetryState(TypedDict):
    api_response: dict | None
    retry_count: int
    max_retries: int

def api_call(state: RetryState) -> dict:
    if state["api_response"] is not None:
        return {}  # Already succeeded

    if state["retry_count"] >= state["max_retries"]:
        raise Exception("Max retries exceeded")

    try:
        response = external_api.call()
        return {
            "api_response": response,
            "retry_count": 0  # Reset on success
        }
    except Exception as e:
        return {
            "retry_count": state["retry_count"] + 1,
            "error": str(e)
        }
```

#### 3. Partial Results

Save intermediate results to avoid losing work:

```python
class ProcessingState(TypedDict):
    items: list
    processed_items: list
    current_index: int

def process_batch(state: ProcessingState) -> dict:
    items = state["items"]
    processed = state["processed_items"]
    start_idx = state["current_index"]

    # Process in small batches
    batch_size = 10
    end_idx = min(start_idx + batch_size, len(items))

    for i in range(start_idx, end_idx):
        result = expensive_process(items[i])
        processed.append(result)

    # Update progress
    return {
        "processed_items": processed,
        "current_index": end_idx
    }

def should_continue(state: ProcessingState) -> str:
    if state["current_index"] >= len(state["items"]):
        return "done"
    return "continue"
```

#### 4. Checkpoint-Based Recovery

Use checkpointers for automatic state persistence:

```python
from langgraph.checkpoint.postgres import PostgresSaver

# Production checkpointer with PostgreSQL
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost/db"
)

workflow = graph.compile(checkpointer=checkpointer)

# Long-running workflow
config = {"configurable": {"thread_id": "production-job-123"}}

try:
    result = workflow.invoke(initial_state, config=config)
except Exception:
    # Crash anywhere - state is saved
    pass

# Days later - resume from exact point of failure
result = workflow.invoke(None, config=config)
```

### Error Handling Strategies

```python
class ErrorTolerantState(TypedDict):
    messages: list
    errors: list
    retry_count: int

def fallback_node(state: ErrorTolerantState) -> dict:
    """Execute with multiple fallback strategies."""
    errors = []

    # Try primary API
    try:
        result = primary_api.call()
        return {"messages": [result]}
    except Exception as e:
        errors.append(f"Primary failed: {e}")

    # Try secondary API
    try:
        result = secondary_api.call()
        return {"messages": [result]}
    except Exception as e:
        errors.append(f"Secondary failed: {e}")

    # Try cached response
    try:
        result = cache.get()
        return {"messages": [result]}
    except Exception as e:
        errors.append(f"Cache failed: {e}")

    # All failed
    return {"errors": errors}
```

### Testing Fault Tolerance

```python
import pytest

def test_node_idempotence():
    """Verify node can be executed multiple times safely."""
    state = {"step_done": False, "data": None}

    # First execution
    result1 = my_node(state)
    state.update(result1)

    # Second execution should be no-op
    result2 = my_node(state)

    assert result2 == {} or result2 == {"step_done": True}
    assert state["data"] is not None

def test_crash_recovery():
    """Verify workflow can recover from crashes."""
    workflow = build_workflow()
    config = {"configurable": {"thread_id": "test"}}

    # Induce crash
    with pytest.raises(RuntimeError):
        workflow.invoke(
            {"should_crash": True},
            config=config
        )

    # Verify checkpoint exists
    state = workflow.get_state(config)
    assert state.values is not None

    # Resume and complete
    result = workflow.invoke(
        {"should_crash": False},
        config=config
    )
    assert result["completed"] is True
```

### Best Practices for Fault Tolerance

1. **Design for Idempotence**: Every node should safely re-execute
2. **Track Completion**: Use boolean flags for each logical step
3. **Save Early, Save Often**: Update state before expensive operations
4. **Partial Progress**: Process in batches and save intermediate results
5. **Meaningful Retry Logic**: Track attempts and implement backoff
6. **Test Recovery Paths**: Verify workflows can resume from any checkpoint
7. **Use Production Checkpointers**: InMemorySaver is for development only

## Best Practices

### 1. State Design

```python
# Good: Clear, typed state
class MyState(TypedDict):
    user_input: str
    processed_data: dict
    final_output: str

# Avoid: Vague, untyped state
class MyState(TypedDict):
    data: Any
    stuff: dict
```

### 2. Node Responsibility

```python
# Good: Single responsibility
def validate_input(state):
    """Only validates input."""
    if not state["input"]:
        return {"error": "Input required"}
    return {}

def process_data(state):
    """Only processes data."""
    result = process(state["input"])
    return {"output": result}

# Avoid: Mixed responsibilities
def validate_and_process(state):
    """Does too much."""
    # validation + processing + error handling + logging...
```

### 3. Error Handling

```python
def robust_node(state: MyState):
    """Node with proper error handling."""
    try:
        result = risky_operation(state["input"])
        return {"output": result, "status": "success"}
    except SpecificException as e:
        logger.warning(f"Specific error: {e}")
        return {"error": str(e), "status": "failed"}
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return {"error": "Internal error", "status": "failed"}
```

### 4. Checkpointer Selection

```python
# Development: InMemorySaver
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()

# Production: PostgreSQL
from langgraph.checkpoint.postgres import PostgresSaver
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    workflow = graph.compile(checkpointer=checkpointer)
```

### 5. Memory Management

```python
# Implement memory cleanup
def should_cleanup(state):
    """Decide when to clean up memory."""
    return len(state["messages"]) > 10

graph.add_conditional_edges(
    "chat",
    should_cleanup,
    {True: "cleanup", False: END}
)
```

### 6. Testing

```python
def test_workflow_node():
    """Test individual nodes."""
    state = {"input": "test"}
    result = my_node(state)
    assert result["output"] == "expected"

def test_workflow_end_to_end():
    """Test complete workflow."""
    result = workflow.invoke({"input": "test"})
    assert result["status"] == "success"
```

## Summary

Graph-based workflows provide structured, observable, and maintainable patterns for complex AI operations:

**Key Takeaways**:

1. **State management**: TypedDict schemas with reducers for controlled updates
2. **Workflow patterns**: Sequential, parallel, and conditional execution
3. **Checkpointing**: InMemorySaver for dev, PostgresSaver for production
4. **Memory**: Short-term (trimming, deletion, summarization) and long-term (semantic stores)
5. **HITL**: Interrupt for human approval and input
6. **Subgraphs**: Reusable components for complex workflows
7. **Tools**: Integrate external capabilities with ToolNode
8. **Debugging**: State inspection, time travel, and visualization

**Workflow Design Checklist**:

```
Design:
☐ Clear state schema (TypedDict)
☐ Single-responsibility nodes
☐ Appropriate workflow pattern (sequential/parallel/conditional)
☐ Visualization for understanding

Memory:
☐ Short-term strategy (trim/delete/summarize)
☐ Long-term storage if needed
☐ Appropriate checkpointer (InMemory vs PostgreSQL)

Integration:
☐ Tools properly defined
☐ Error handling in all nodes
☐ HITL where needed
☐ MCP servers if applicable

Testing:
☐ Unit tests for nodes
☐ Integration tests for workflows
☐ State inspection for debugging
```

**Remember**: Start simple (sequential workflow), add complexity as needed (parallel, conditional, memory), and always maintain observability (checkpointing, state inspection).

## Next Steps

Now that you understand graph workflow patterns, explore:

1. **[State Management](state-management.md)**: Deep dive into state patterns
2. **[Control Flow](control-flow.md)**: Advanced routing and branching
3. **[Subagent Patterns](subagent-patterns.md)**: Orchestrating multiple agents
4. **[Tool Definition Patterns](../tool-use/tool-definition-patterns.md)**: Creating effective tools
5. **[MCP Implementation](../model-context-protocol/mcp-implementation-patterns.md)**: Building MCP servers
6. **[Memory Architecture](../memory-systems/memory-architecture.md)**: Designing memory systems

Start building structured, observable, and maintainable AI workflows.
