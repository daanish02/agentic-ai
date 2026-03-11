# Communication Patterns

## Table of Contents

- [Introduction](#introduction)
- [Why Communication Matters](#why-communication-matters)
- [Communication Paradigms](#communication-paradigms)
- [Message Passing](#message-passing)
- [Shared State](#shared-state)
- [Blackboard Systems](#blackboard-systems)
- [Event-Driven Communication](#event-driven-communication)
- [Synchronous vs Asynchronous](#synchronous-vs-asynchronous)
- [Communication Protocols](#communication-protocols)
- [Message Formats](#message-formats)
- [Ensuring Understanding](#ensuring-understanding)
- [Communication Patterns](#communication-patterns)
- [Error Handling](#error-handling)
- [Performance Considerations](#performance-considerations)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Communication is the lifeblood of multi-agent systems. Agents can be brilliant individually, but without effective communication, they cannot collaborate. How agents exchange information determines whether a multi-agent system succeeds or fails.

> "The single biggest problem in communication is the illusion that it has taken place."  
> -- George Bernard Shaw

In multi-agent systems, we face unique communication challenges:

- **Asynchronous timing**: Agents operate independently
- **Semantic understanding**: Ensuring messages mean the same thing to sender and receiver
- **Reliability**: Handling message loss, delays, and errors
- **Scalability**: Managing communication as agent count grows
- **Privacy**: Controlling information access

This guide explores communication paradigms, protocols, patterns, and best practices for making agents communicate effectively.

## Why Communication Matters

Communication enables coordination:

### Without Communication

```python
class IsolatedAgents:
    """Agents work in isolation - chaos results."""

    def __init__(self):
        self.agent1 = Agent("agent1")
        self.agent2 = Agent("agent2")

    async def parallel_work(self, task):
        """Both agents work without communicating."""

        # No communication - both agents might:
        # - Do duplicate work
        # - Make conflicting decisions
        # - Miss important context
        # - Unable to coordinate

        result1 = await self.agent1.work(task)
        result2 = await self.agent2.work(task)

        # Results might conflict!
        if result1 != result2:
            print("Agents disagree - no way to resolve")
            return None

        return result1

# Problems:
# - Duplicate effort
# - Conflicting results
# - No coordination
# - Wasted resources
```

### With Communication

```python
class CommunicatingAgents:
    """Agents coordinate through communication."""

    def __init__(self):
        self.agent1 = Agent("agent1")
        self.agent2 = Agent("agent2")
        self.message_bus = MessageBus()

    async def coordinated_work(self, task):
        """Agents communicate to coordinate."""

        # Agent 1 claims task
        await self.agent1.broadcast({
            "type": "claim",
            "task_id": task.id,
            "agent": "agent1"
        })

        # Agent 2 sees claim, works on something else
        await self.agent2.receive_claim("agent1", task.id)

        # Agent 1 completes work
        result = await self.agent1.work(task)

        # Agent 1 shares result
        await self.agent1.broadcast({
            "type": "complete",
            "task_id": task.id,
            "result": result
        })

        # Agent 2 knows task is done
        await self.agent2.receive_result(result)

        return result

# Benefits:
# - No duplicate work
# - Coordinated actions
# - Shared awareness
# - Efficient resource use
```

## Communication Paradigms

Several fundamental approaches to agent communication:

### 1. Message Passing

```python
"""
Message Passing: Agents send explicit messages to each other.

Characteristics:
- Direct communication
- Point-to-point or broadcast
- Explicit send/receive
- No shared state
"""

class MessagePassingSystem:
    """Message passing implementation."""

    def __init__(self):
        self.agents = {}
        self.message_queues = {}

    def register_agent(self, agent_id: str, agent: Agent):
        """Register agent in system."""
        self.agents[agent_id] = agent
        self.message_queues[agent_id] = asyncio.Queue()

    async def send_message(self, from_id: str, to_id: str, content: Dict):
        """Send message from one agent to another."""

        message = {
            "from": from_id,
            "to": to_id,
            "content": content,
            "timestamp": time.time()
        }

        # Add to recipient's queue
        if to_id in self.message_queues:
            await self.message_queues[to_id].put(message)

    async def receive_messages(self, agent_id: str) -> List[Dict]:
        """Get pending messages for agent."""

        messages = []
        queue = self.message_queues[agent_id]

        while not queue.empty():
            try:
                message = queue.get_nowait()
                messages.append(message)
            except asyncio.QueueEmpty:
                break

        return messages

    async def broadcast(self, from_id: str, content: Dict):
        """Broadcast message to all agents."""

        for agent_id in self.agents:
            if agent_id != from_id:
                await self.send_message(from_id, agent_id, content)

# Example usage
system = MessagePassingSystem()
system.register_agent("agent1", Agent("agent1"))
system.register_agent("agent2", Agent("agent2"))

# Agent 1 sends to Agent 2
await system.send_message("agent1", "agent2", {
    "type": "request",
    "task": "analyze data"
})

# Agent 2 receives
messages = await system.receive_messages("agent2")
```

**Pros:**

- Clear sender/receiver
- Easy to trace message flow
- No shared state conflicts
- Straightforward debugging

**Cons:**

- Tight coupling (sender must know recipient)
- Scalability challenges (O(n²) connections)
- Requires message routing

### 2. Shared State

```python
"""
Shared State: Agents communicate by reading/writing shared memory.

Characteristics:
- Indirect communication
- Through shared data structures
- No explicit messaging
- Requires synchronization
"""

class SharedStateSystem:
    """Shared state implementation."""

    def __init__(self):
        self.agents = {}
        self.shared_data = {}
        self.locks = {}

    async def write(self, agent_id: str, key: str, value: Any):
        """Agent writes to shared state."""

        # Get or create lock for this key
        if key not in self.locks:
            self.locks[key] = asyncio.Lock()

        # Acquire lock
        async with self.locks[key]:
            # Write data
            self.shared_data[key] = {
                "value": value,
                "updated_by": agent_id,
                "timestamp": time.time()
            }

            # Notify observers
            await self.notify_observers(key, value, agent_id)

    async def read(self, agent_id: str, key: str) -> Any:
        """Agent reads from shared state."""

        if key in self.shared_data:
            data = self.shared_data[key]
            return data["value"]

        return None

    async def subscribe(self, agent_id: str, key: str):
        """Agent subscribes to changes on key."""

        if key not in self.observers:
            self.observers[key] = []

        self.observers[key].append(agent_id)

    async def notify_observers(self, key: str, value: Any, updater: str):
        """Notify agents watching this key."""

        if key in self.observers:
            for observer_id in self.observers[key]:
                if observer_id != updater:
                    await self.agents[observer_id].notify_update(key, value)

# Example usage
system = SharedStateSystem()

# Agent 1 writes
await system.write("agent1", "user_preference", "dark_mode")

# Agent 2 reads
preference = await system.read("agent2", "user_preference")
print(preference)  # "dark_mode"

# Agent 3 subscribes to changes
await system.subscribe("agent3", "user_preference")
```

**Pros:**

- Loose coupling (agents don't need to know each other)
- Natural for shared data
- Easy to broadcast information
- Simpler than explicit messaging

**Cons:**

- Race conditions
- Requires synchronization
- Harder to debug
- Scalability issues with high contention

### 3. Blackboard Systems

```python
"""
Blackboard: Agents collaborate through shared knowledge space.

Characteristics:
- Shared repository of knowledge
- Agents contribute and consume knowledge
- Pattern-based access
- Opportunistic cooperation
"""

class Blackboard:
    """Blackboard system for agent collaboration."""

    def __init__(self):
        self.knowledge = {}  # knowledge items
        self.subscriptions = {}  # pattern subscriptions
        self.agents = {}

    async def post(self, agent_id: str, item: Dict):
        """Post knowledge item to blackboard."""

        item_id = self.generate_id()

        # Add metadata
        item["posted_by"] = agent_id
        item["posted_at"] = time.time()
        item["item_id"] = item_id

        # Store on blackboard
        self.knowledge[item_id] = item

        # Notify subscribers
        await self.notify_matches(item)

        return item_id

    async def query(self, pattern: Dict) -> List[Dict]:
        """Query blackboard for matching items."""

        matches = []

        for item_id, item in self.knowledge.items():
            if self.matches_pattern(item, pattern):
                matches.append(item)

        return matches

    def matches_pattern(self, item: Dict, pattern: Dict) -> bool:
        """Check if item matches pattern."""

        for key, value in pattern.items():
            if key not in item:
                return False

            if callable(value):
                if not value(item[key]):
                    return False
            elif item[key] != value:
                return False

        return True

    async def subscribe(self, agent_id: str, pattern: Dict,
                       handler: Callable):
        """Subscribe to items matching pattern."""

        pattern_key = str(pattern)

        if pattern_key not in self.subscriptions:
            self.subscriptions[pattern_key] = []

        self.subscriptions[pattern_key].append({
            "agent_id": agent_id,
            "pattern": pattern,
            "handler": handler
        })

    async def notify_matches(self, item: Dict):
        """Notify subscribers of matching items."""

        for pattern_key, subscriptions in self.subscriptions.items():
            for sub in subscriptions:
                if self.matches_pattern(item, sub["pattern"]):
                    await sub["handler"](item)

# Example usage
blackboard = Blackboard()

# Agent 1 posts finding
await blackboard.post("researcher", {
    "type": "finding",
    "topic": "climate change",
    "content": "Temperature rising 0.2°C per decade",
    "confidence": 0.95
})

# Agent 2 queries for climate findings
findings = await blackboard.query({
    "type": "finding",
    "topic": "climate change",
    "confidence": lambda x: x > 0.9
})

# Agent 3 subscribes to high-confidence findings
async def handle_finding(item):
    print(f"New finding: {item['content']}")

await blackboard.subscribe("analyst",
    {"type": "finding", "confidence": lambda x: x > 0.9},
    handle_finding
)
```

**Pros:**

- Flexible, opportunistic cooperation
- Loose coupling
- Natural for incremental problem solving
- Easy to add new agents

**Cons:**

- Can be hard to understand flow
- Potential for chaos without coordination
- Performance overhead from pattern matching

### 4. Event-Driven Communication

```python
"""
Event-Driven: Agents communicate via events and handlers.

Characteristics:
- Publish-subscribe model
- Event producers and consumers
- Asynchronous
- Loose coupling
"""

class EventBus:
    """Event-driven communication system."""

    def __init__(self):
        self.handlers = {}  # topic -> handlers
        self.event_log = []

    def subscribe(self, topic: str, handler: Callable,
                 agent_id: str):
        """Subscribe to topic."""

        if topic not in self.handlers:
            self.handlers[topic] = []

        self.handlers[topic].append({
            "handler": handler,
            "agent_id": agent_id
        })

    async def publish(self, topic: str, event: Dict,
                     publisher: str):
        """Publish event to topic."""

        # Log event
        event_record = {
            "topic": topic,
            "event": event,
            "publisher": publisher,
            "timestamp": time.time()
        }
        self.event_log.append(event_record)

        # Notify subscribers
        if topic in self.handlers:
            for subscription in self.handlers[topic]:
                # Don't send to self
                if subscription["agent_id"] != publisher:
                    try:
                        await subscription["handler"](event)
                    except Exception as e:
                        print(f"Handler error: {e}")

    def unsubscribe(self, topic: str, agent_id: str):
        """Unsubscribe from topic."""

        if topic in self.handlers:
            self.handlers[topic] = [
                h for h in self.handlers[topic]
                if h["agent_id"] != agent_id
            ]

# Example usage
bus = EventBus()

# Agent 1 subscribes to "task.complete" events
async def handle_complete(event):
    print(f"Task {event['task_id']} completed")

bus.subscribe("task.complete", handle_complete, "agent1")

# Agent 2 publishes completion event
await bus.publish("task.complete", {
    "task_id": "task_123",
    "result": "success"
}, "agent2")

# Agent 1's handler is called
```

**Pros:**

- Very loose coupling
- Easy to add subscribers
- Natural for distributed systems
- Scalable

**Cons:**

- Hard to guarantee delivery
- Difficult to trace message flow
- No built-in ordering guarantees

## Message Passing

Deep dive into message passing:

### Message Structure

```python
from dataclasses import dataclass
from typing import Any, Optional
from enum import Enum
import uuid

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    HEARTBEAT = "heartbeat"

@dataclass
class Message:
    """Standard message structure."""

    # Identity
    msg_id: str
    msg_type: MessageType

    # Routing
    sender: str
    recipient: str
    reply_to: Optional[str] = None

    # Content
    content: Any = None

    # Metadata
    timestamp: float = 0.0
    priority: int = 0
    ttl: Optional[float] = None  # Time to live

    # Tracking
    correlation_id: Optional[str] = None

    def __post_init__(self):
        if not self.msg_id:
            self.msg_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = time.time()

# Example messages
request = Message(
    msg_id="msg_001",
    msg_type=MessageType.REQUEST,
    sender="agent1",
    recipient="agent2",
    content={
        "action": "analyze",
        "data": [1, 2, 3, 4, 5]
    }
)

response = Message(
    msg_id="msg_002",
    msg_type=MessageType.RESPONSE,
    sender="agent2",
    recipient="agent1",
    reply_to="msg_001",
    content={
        "result": "average is 3.0"
    },
    correlation_id="msg_001"
)
```

### Request-Response Pattern

```python
class RequestResponsePattern:
    """Implement request-response communication."""

    def __init__(self):
        self.pending_requests = {}
        self.message_bus = MessageBus()

    async def request(self, sender: str, recipient: str,
                     content: Dict, timeout: float = 5.0) -> Dict:
        """Send request and wait for response."""

        # Create request message
        request = Message(
            msg_id=str(uuid.uuid4()),
            msg_type=MessageType.REQUEST,
            sender=sender,
            recipient=recipient,
            content=content
        )

        # Track pending request
        response_future = asyncio.Future()
        self.pending_requests[request.msg_id] = response_future

        # Send request
        await self.message_bus.send(request)

        # Wait for response with timeout
        try:
            response = await asyncio.wait_for(
                response_future,
                timeout=timeout
            )
            return response.content

        except asyncio.TimeoutError:
            # Cleanup
            del self.pending_requests[request.msg_id]
            raise TimeoutError(f"No response from {recipient}")

    async def respond(self, sender: str, request_msg: Message,
                     content: Dict):
        """Send response to request."""

        response = Message(
            msg_id=str(uuid.uuid4()),
            msg_type=MessageType.RESPONSE,
            sender=sender,
            recipient=request_msg.sender,
            reply_to=request_msg.msg_id,
            content=content,
            correlation_id=request_msg.msg_id
        )

        await self.message_bus.send(response)

    async def handle_response(self, response: Message):
        """Handle incoming response."""

        # Find pending request
        if response.correlation_id in self.pending_requests:
            future = self.pending_requests[response.correlation_id]
            future.set_result(response)
            del self.pending_requests[response.correlation_id]

# Example usage
pattern = RequestResponsePattern()

# Agent 1 requests analysis
result = await pattern.request(
    sender="agent1",
    recipient="agent2",
    content={"action": "analyze", "data": [1, 2, 3]},
    timeout=10.0
)

print(result)  # Response from agent2
```

### One-Way Messages

```python
class OneWayMessaging:
    """Fire-and-forget messaging."""

    async def send_notification(self, sender: str, recipient: str,
                               content: Dict):
        """Send notification (no response expected)."""

        notification = Message(
            msg_id=str(uuid.uuid4()),
            msg_type=MessageType.NOTIFICATION,
            sender=sender,
            recipient=recipient,
            content=content
        )

        await self.message_bus.send(notification)

    async def broadcast(self, sender: str, content: Dict):
        """Broadcast to all agents."""

        notification = Message(
            msg_id=str(uuid.uuid4()),
            msg_type=MessageType.NOTIFICATION,
            sender=sender,
            recipient="*",  # Broadcast
            content=content
        )

        await self.message_bus.broadcast(notification)

# Example
messaging = OneWayMessaging()

# Notify specific agent
await messaging.send_notification(
    sender="monitor",
    recipient="logger",
    content={"event": "system_start"}
)

# Broadcast to all
await messaging.broadcast(
    sender="coordinator",
    content={"status": "task_complete", "task_id": "123"}
)
```

## Synchronous vs Asynchronous

Critical distinction affecting system design:

### Synchronous Communication

```python
class SynchronousCommunication:
    """Blocking communication - sender waits for response."""

    async def call_agent(self, agent: str, request: Dict) -> Dict:
        """Synchronous call - blocks until response."""

        print(f"Calling {agent}...")
        start = time.time()

        # Send request
        response = await self.send_and_wait(agent, request)

        duration = time.time() - start
        print(f"Response received in {duration:.2f}s")

        return response

# Characteristics:
# - Caller blocks waiting for response
# - Simple to reason about
# - Sequential execution
# - Higher latency (cumulative waits)

# Example timing
async def sequential_calls():
    # Call agent 1 (blocks 1s)
    result1 = await call_agent("agent1", {...})  # 1s

    # Call agent 2 (blocks 1s)
    result2 = await call_agent("agent2", {...})  # 1s

    # Call agent 3 (blocks 1s)
    result3 = await call_agent("agent3", {...})  # 1s

    # Total: 3 seconds (sequential)
    return combine(result1, result2, result3)

# Pros:
# - Simple to understand
# - Easy error handling
# - Predictable ordering

# Cons:
# - High latency
# - Caller blocked
# - Poor resource utilization
```

### Asynchronous Communication

```python
class AsynchronousCommunication:
    """Non-blocking communication - sender continues."""

    async def send_async(self, agent: str, request: Dict,
                        callback: Optional[Callable] = None):
        """Async send - don't wait for response."""

        print(f"Sending to {agent}...")

        # Send request
        msg_id = await self.send(agent, request)

        # Register callback if provided
        if callback:
            self.callbacks[msg_id] = callback

        # Return immediately
        return msg_id

    async def handle_response(self, response: Message):
        """Handle response when it arrives."""

        if response.correlation_id in self.callbacks:
            callback = self.callbacks[response.correlation_id]
            await callback(response.content)
            del self.callbacks[response.correlation_id]

# Example timing
async def parallel_calls():
    # Send all requests without waiting
    future1 = asyncio.create_task(call_agent("agent1", {...}))
    future2 = asyncio.create_task(call_agent("agent2", {...}))
    future3 = asyncio.create_task(call_agent("agent3", {...}))

    # Wait for all to complete
    results = await asyncio.gather(future1, future2, future3)

    # Total: 1 second (parallel)
    return combine(*results)

# Pros:
# - Low latency
# - Caller not blocked
# - Better resource utilization
# - Natural parallelism

# Cons:
# - More complex
# - Harder error handling
# - Ordering not guaranteed
```

### Choosing Communication Mode

```python
def choose_communication_mode(scenario: Dict) -> str:
    """Decide sync vs async based on scenario."""

    # Use synchronous if:
    if (scenario["needs_response"] and
        scenario["dependent_operations"] and
        scenario["simple_flow"]):
        return "synchronous"

    # Use asynchronous if:
    if (scenario["high_throughput"] or
        scenario["independent_operations"] or
        scenario["long_running"]):
        return "asynchronous"

    # Default to async
    return "asynchronous"

# Examples
print(choose_communication_mode({
    "needs_response": True,
    "dependent_operations": True,
    "simple_flow": True,
    "high_throughput": False,
    "independent_operations": False,
    "long_running": False
}))  # "synchronous"

print(choose_communication_mode({
    "needs_response": True,
    "dependent_operations": False,
    "simple_flow": False,
    "high_throughput": True,
    "independent_operations": True,
    "long_running": True
}))  # "asynchronous"
```

## Communication Protocols

Define how agents interact:

### Protocol Definition

```python
from abc import ABC, abstractmethod

class CommunicationProtocol(ABC):
    """Abstract base for communication protocols."""

    @abstractmethod
    async def send(self, message: Message):
        """Send message."""
        pass

    @abstractmethod
    async def receive(self) -> Message:
        """Receive message."""
        pass

    @abstractmethod
    async def connect(self, agent_id: str):
        """Connect agent to protocol."""
        pass

    @abstractmethod
    async def disconnect(self, agent_id: str):
        """Disconnect agent."""
        pass

class DirectMessageProtocol(CommunicationProtocol):
    """Direct agent-to-agent messaging."""

    def __init__(self):
        self.mailboxes = {}

    async def connect(self, agent_id: str):
        self.mailboxes[agent_id] = asyncio.Queue()

    async def send(self, message: Message):
        if message.recipient in self.mailboxes:
            await self.mailboxes[message.recipient].put(message)
        else:
            raise ValueError(f"Unknown recipient: {message.recipient}")

    async def receive(self, agent_id: str) -> Message:
        if agent_id in self.mailboxes:
            return await self.mailboxes[agent_id].get()
        else:
            raise ValueError(f"Unknown agent: {agent_id}")

class BroadcastProtocol(CommunicationProtocol):
    """Broadcast messages to all agents."""

    def __init__(self):
        self.agents = set()
        self.mailboxes = {}

    async def connect(self, agent_id: str):
        self.agents.add(agent_id)
        self.mailboxes[agent_id] = asyncio.Queue()

    async def send(self, message: Message):
        # Send to all agents except sender
        for agent_id in self.agents:
            if agent_id != message.sender:
                await self.mailboxes[agent_id].put(message)

    async def receive(self, agent_id: str) -> Message:
        return await self.mailboxes[agent_id].get()

class PublishSubscribeProtocol(CommunicationProtocol):
    """Topic-based pub/sub messaging."""

    def __init__(self):
        self.subscriptions = {}  # topic -> agents
        self.mailboxes = {}

    async def connect(self, agent_id: str):
        self.mailboxes[agent_id] = asyncio.Queue()

    async def subscribe(self, agent_id: str, topic: str):
        if topic not in self.subscriptions:
            self.subscriptions[topic] = set()
        self.subscriptions[topic].add(agent_id)

    async def publish(self, topic: str, message: Message):
        if topic in self.subscriptions:
            for agent_id in self.subscriptions[topic]:
                if agent_id != message.sender:
                    await self.mailboxes[agent_id].put(message)

    async def receive(self, agent_id: str) -> Message:
        return await self.mailboxes[agent_id].get()
```

### Protocol Selection

```python
class ProtocolSelector:
    """Choose appropriate protocol for communication needs."""

    protocols = {
        "direct": DirectMessageProtocol(),
        "broadcast": BroadcastProtocol(),
        "pubsub": PublishSubscribeProtocol()
    }

    def select_protocol(self, requirements: Dict) -> CommunicationProtocol:
        """Select protocol based on requirements."""

        # Direct messaging for point-to-point
        if requirements["pattern"] == "point_to_point":
            return self.protocols["direct"]

        # Broadcast for all-to-all
        if requirements["pattern"] == "all_to_all":
            return self.protocols["broadcast"]

        # Pub/sub for topic-based
        if requirements["pattern"] == "topic_based":
            return self.protocols["pubsub"]

        # Default
        return self.protocols["direct"]
```

## Message Formats

Standardize message content:

### JSON Messages

```python
import json
from typing import Any

class JSONMessageFormat:
    """JSON-based message format."""

    @staticmethod
    def encode(message: Message) -> str:
        """Encode message as JSON."""

        return json.dumps({
            "msg_id": message.msg_id,
            "msg_type": message.msg_type.value,
            "sender": message.sender,
            "recipient": message.recipient,
            "content": message.content,
            "timestamp": message.timestamp,
            "reply_to": message.reply_to
        })

    @staticmethod
    def decode(json_str: str) -> Message:
        """Decode JSON to message."""

        data = json.loads(json_str)

        return Message(
            msg_id=data["msg_id"],
            msg_type=MessageType(data["msg_type"]),
            sender=data["sender"],
            recipient=data["recipient"],
            content=data["content"],
            timestamp=data["timestamp"],
            reply_to=data.get("reply_to")
        )

# Example
message = Message(
    msg_id="123",
    msg_type=MessageType.REQUEST,
    sender="agent1",
    recipient="agent2",
    content={"action": "analyze", "data": [1, 2, 3]}
)

# Encode
json_str = JSONMessageFormat.encode(message)
print(json_str)

# Decode
decoded = JSONMessageFormat.decode(json_str)
```

### Structured Content

```python
from pydantic import BaseModel

class TaskRequest(BaseModel):
    """Structured task request."""
    action: str
    parameters: Dict[str, Any]
    priority: int = 0
    deadline: Optional[float] = None

class TaskResponse(BaseModel):
    """Structured task response."""
    status: str
    result: Any
    error: Optional[str] = None
    execution_time: float

class StructuredMessaging:
    """Type-safe messaging with Pydantic."""

    async def send_task_request(self, sender: str, recipient: str,
                               task: TaskRequest) -> Message:
        """Send typed task request."""

        return Message(
            msg_id=str(uuid.uuid4()),
            msg_type=MessageType.REQUEST,
            sender=sender,
            recipient=recipient,
            content=task.dict()
        )

    async def parse_task_request(self, message: Message) -> TaskRequest:
        """Parse message as task request."""

        return TaskRequest(**message.content)

    async def send_task_response(self, sender: str, recipient: str,
                                response: TaskResponse,
                                request_id: str) -> Message:
        """Send typed task response."""

        return Message(
            msg_id=str(uuid.uuid4()),
            msg_type=MessageType.RESPONSE,
            sender=sender,
            recipient=recipient,
            content=response.dict(),
            reply_to=request_id
        )

# Example
messaging = StructuredMessaging()

# Create typed request
request = TaskRequest(
    action="analyze_sentiment",
    parameters={"text": "This is great!"},
    priority=1
)

# Send
msg = await messaging.send_task_request("agent1", "agent2", request)

# Receive and parse
parsed = await messaging.parse_task_request(msg)
print(parsed.action)  # "analyze_sentiment"
```

## Ensuring Understanding

Verify messages are understood correctly:

### Acknowledgments

```python
class AcknowledgmentSystem:
    """Ensure message delivery with ACKs."""

    def __init__(self):
        self.pending_acks = {}

    async def send_with_ack(self, message: Message,
                           timeout: float = 5.0) -> bool:
        """Send message and wait for acknowledgment."""

        # Track pending ACK
        ack_future = asyncio.Future()
        self.pending_acks[message.msg_id] = ack_future

        # Send message
        await self.send(message)

        # Wait for ACK
        try:
            await asyncio.wait_for(ack_future, timeout=timeout)
            return True
        except asyncio.TimeoutError:
            print(f"No ACK received for {message.msg_id}")
            return False
        finally:
            if message.msg_id in self.pending_acks:
                del self.pending_acks[message.msg_id]

    async def acknowledge(self, message_id: str, recipient: str):
        """Send acknowledgment."""

        ack = Message(
            msg_id=str(uuid.uuid4()),
            msg_type=MessageType.RESPONSE,
            sender=recipient,
            recipient=message_id,
            content={"ack": True},
            reply_to=message_id
        )

        await self.send(ack)

    async def handle_ack(self, ack: Message):
        """Handle incoming acknowledgment."""

        if ack.reply_to in self.pending_acks:
            future = self.pending_acks[ack.reply_to]
            future.set_result(True)

# Example
system = AcknowledgmentSystem()

# Send with ACK
success = await system.send_with_ack(message, timeout=10.0)

if success:
    print("Message delivered and acknowledged")
else:
    print("Message delivery failed")
```

### Confirmation and Checksums

```python
import hashlib

class MessageIntegrity:
    """Ensure message integrity."""

    @staticmethod
    def add_checksum(message: Message) -> Message:
        """Add checksum to message."""

        # Serialize content
        content_str = json.dumps(message.content, sort_keys=True)

        # Compute checksum
        checksum = hashlib.sha256(content_str.encode()).hexdigest()

        # Add to message
        message.content["__checksum__"] = checksum

        return message

    @staticmethod
    def verify_checksum(message: Message) -> bool:
        """Verify message checksum."""

        if "__checksum__" not in message.content:
            return False

        # Extract checksum
        received_checksum = message.content.pop("__checksum__")

        # Recompute checksum
        content_str = json.dumps(message.content, sort_keys=True)
        computed_checksum = hashlib.sha256(content_str.encode()).hexdigest()

        # Compare
        return received_checksum == computed_checksum

# Example
message = Message(
    msg_id="123",
    msg_type=MessageType.REQUEST,
    sender="agent1",
    recipient="agent2",
    content={"data": "important information"}
)

# Add checksum
message = MessageIntegrity.add_checksum(message)

# Later: verify
if MessageIntegrity.verify_checksum(message):
    print("Message integrity verified")
else:
    print("Message may be corrupted")
```

## Summary

Effective communication is essential for multi-agent systems:

**Key Paradigms:**

- Message passing for explicit communication
- Shared state for implicit coordination
- Blackboards for collaborative problem-solving
- Event-driven for loose coupling

**Communication Modes:**

- Synchronous: Simple but blocking
- Asynchronous: Complex but efficient

**Best Practices:**

- Define clear protocols
- Use structured message formats
- Verify message delivery and understanding
- Handle errors gracefully
- Monitor communication performance

**Design Decisions:**

- Choose paradigm based on coupling needs
- Select sync/async based on latency requirements
- Use appropriate protocols for communication patterns
- Structure messages for type safety
- Implement acknowledgments for critical messages

Well-designed communication enables agents to coordinate effectively, share knowledge, and collaborate to solve complex problems.

## Next Steps

- [Coordination and Task Allocation](coordination.md) - Organizing agent work
- [Agent Roles and Specialization](agent-roles.md) - Defining agent responsibilities
- [Delegation and Hierarchies](delegation.md) - Hierarchical communication
- [Collaboration Patterns](collaboration.md) - Cooperative communication
- [Multi-Agent Fundamentals](fundamentals.md) - Core concepts
