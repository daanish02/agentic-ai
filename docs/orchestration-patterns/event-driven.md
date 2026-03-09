# Event-Driven Orchestration

## Table of Contents

- [Introduction](#introduction)
- [Event-Driven Architecture](#event-driven-architecture)
- [Triggers and Events](#triggers-and-events)
- [Event Sources](#event-sources)
- [Event Handlers](#event-handlers)
- [Webhooks](#webhooks)
- [Scheduled Workflows](#scheduled-workflows)
- [Message Queues](#message-queues)
- [Pub/Sub Patterns](#pubsub-patterns)
- [Event Streaming](#event-streaming)
- [Reactive Agents](#reactive-agents)
- [Error Handling and Retry](#error-handling-and-retry)
- [Real-World Applications](#real-world-applications)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Event-driven orchestration enables agents to react to events rather than running continuously. Instead of polling or waiting, agents respond automatically when things happen:

- **Triggers**: Workflows start from external events
- **Webhooks**: HTTP callbacks from external systems
- **Schedules**: Time-based automation
- **Messages**: Queue-based event processing
- **Streams**: Real-time event processing

> "Don't call us, we'll call you" -- The essence of event-driven design

```
Traditional (Polling):              Event-Driven:

while True:                         def on_event(event):
    check_for_work()                    process(event)
    if work_found:
        do_work()                   wait_for_events()
    sleep(60)

↓ Wastes resources               ↓ Efficient, reactive
```

**Benefits**:

- **Resource Efficient**: No constant polling
- **Real-Time**: Immediate response to events
- **Scalable**: Handle spikes in events
- **Decoupled**: Components loosely connected
- **Flexible**: Easy to add new event handlers

## Event-Driven Architecture

Core concepts of event-driven systems.

### Event Model

```python
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Callable
from datetime import datetime
from enum import Enum
import uuid

class EventType(Enum):
    """Common event types"""
    HTTP_REQUEST = "http_request"
    DATABASE_CHANGE = "database_change"
    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"
    SCHEDULED = "scheduled"
    MESSAGE_RECEIVED = "message_received"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"

@dataclass
class Event:
    """Base event class"""
    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:8]}")
    event_type: EventType = EventType.SYSTEM_EVENT
    source: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "metadata": self.metadata
        }

class EventBus:
    """Central event bus for publishing and subscribing"""

    def __init__(self):
        self.handlers: Dict[EventType, List[Callable]] = {}
        self.event_history: List[Event] = []

    def subscribe(self, event_type: EventType, handler: Callable):
        """Subscribe to event type"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []

        self.handlers[event_type].append(handler)
        print(f"Handler subscribed to {event_type.value}")

    def unsubscribe(self, event_type: EventType, handler: Callable):
        """Unsubscribe from event type"""
        if event_type in self.handlers:
            self.handlers[event_type].remove(handler)

    def publish(self, event: Event):
        """Publish event to all subscribers"""
        print(f"\n{'='*60}")
        print(f"Event Published: {event.event_type.value}")
        print(f"Event ID: {event.event_id}")
        print(f"Source: {event.source}")
        print(f"Data: {event.data}")
        print(f"{'='*60}\n")

        # Store in history
        self.event_history.append(event)

        # Notify handlers
        handlers = self.handlers.get(event.event_type, [])

        for handler in handlers:
            try:
                print(f"  → Calling handler: {handler.__name__}")
                handler(event)
            except Exception as e:
                print(f"  ✗ Handler {handler.__name__} failed: {e}")

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 10
    ) -> List[Event]:
        """Get event history"""
        events = self.event_history

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events[-limit:]

# Example usage
bus = EventBus()

# Define handlers
def on_file_created(event: Event):
    """Handle file creation"""
    filename = event.data.get("filename")
    print(f"    Processing new file: {filename}")

def on_file_created_backup(event: Event):
    """Backup handler for file creation"""
    filename = event.data.get("filename")
    print(f"    Backing up file: {filename}")

# Subscribe handlers
bus.subscribe(EventType.FILE_CREATED, on_file_created)
bus.subscribe(EventType.FILE_CREATED, on_file_created_backup)

# Publish event
event = Event(
    event_type=EventType.FILE_CREATED,
    source="filesystem",
    data={"filename": "document.pdf", "size": 1024}
)

bus.publish(event)

# View history
history = bus.get_history()
print(f"\nEvent history: {len(history)} events")
```

## Triggers and Events

Defining triggers that start agent workflows.

### Trigger System

```python
@dataclass
class Trigger:
    """Workflow trigger definition"""
    trigger_id: str
    trigger_type: EventType
    conditions: Dict[str, Any]
    handler: Callable[[Event], Any]
    enabled: bool = True

    def matches(self, event: Event) -> bool:
        """Check if event matches trigger conditions"""
        if not self.enabled:
            return False

        if event.event_type != self.trigger_type:
            return False

        # Check conditions
        for key, expected_value in self.conditions.items():
            if key not in event.data:
                return False

            actual_value = event.data[key]

            # Support simple equality and callable conditions
            if callable(expected_value):
                if not expected_value(actual_value):
                    return False
            elif actual_value != expected_value:
                return False

        return True

class TriggerManager:
    """Manage workflow triggers"""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.triggers: Dict[str, Trigger] = {}

    def register_trigger(
        self,
        trigger_id: str,
        event_type: EventType,
        conditions: Dict[str, Any],
        handler: Callable[[Event], Any]
    ):
        """Register new trigger"""
        trigger = Trigger(
            trigger_id=trigger_id,
            trigger_type=event_type,
            conditions=conditions,
            handler=handler
        )

        self.triggers[trigger_id] = trigger

        # Subscribe to event bus
        self.event_bus.subscribe(event_type, self._handle_event)

        print(f"Registered trigger: {trigger_id}")

    def _handle_event(self, event: Event):
        """Handle incoming event"""
        # Check all triggers
        for trigger in self.triggers.values():
            if trigger.matches(event):
                print(f"  🎯 Trigger matched: {trigger.trigger_id}")
                try:
                    trigger.handler(event)
                except Exception as e:
                    print(f"  ✗ Trigger handler failed: {e}")

    def enable_trigger(self, trigger_id: str):
        """Enable trigger"""
        if trigger_id in self.triggers:
            self.triggers[trigger_id].enabled = True
            print(f"Enabled trigger: {trigger_id}")

    def disable_trigger(self, trigger_id: str):
        """Disable trigger"""
        if trigger_id in self.triggers:
            self.triggers[trigger_id].enabled = False
            print(f"Disabled trigger: {trigger_id}")

    def list_triggers(self) -> List[Trigger]:
        """List all triggers"""
        return list(self.triggers.values())

# Example usage
bus = EventBus()
trigger_mgr = TriggerManager(bus)

# Register triggers
def process_large_file(event: Event):
    """Process large file"""
    filename = event.data.get("filename")
    size = event.data.get("size")
    print(f"      → Processing large file: {filename} ({size} bytes)")

trigger_mgr.register_trigger(
    trigger_id="large_file_trigger",
    event_type=EventType.FILE_CREATED,
    conditions={
        "size": lambda s: s > 1000  # Files larger than 1000 bytes
    },
    handler=process_large_file
)

def process_pdf(event: Event):
    """Process PDF file"""
    filename = event.data.get("filename")
    print(f"      → Processing PDF: {filename}")

trigger_mgr.register_trigger(
    trigger_id="pdf_trigger",
    event_type=EventType.FILE_CREATED,
    conditions={
        "filename": lambda f: f.endswith(".pdf")
    },
    handler=process_pdf
)

# Publish events
print("\n" + "="*60)
print("Publishing Events")
print("="*60)

# Small text file - no triggers
bus.publish(Event(
    event_type=EventType.FILE_CREATED,
    source="filesystem",
    data={"filename": "small.txt", "size": 100}
))

# Large PDF - both triggers
bus.publish(Event(
    event_type=EventType.FILE_CREATED,
    source="filesystem",
    data={"filename": "large.pdf", "size": 2000}
))
```

## Event Sources

Different sources of events for agents.

### HTTP Webhooks

```python
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import json

class WebhookHandler(BaseHTTPRequestHandler):
    """Handle incoming webhook requests"""

    event_bus = None  # Set by WebhookServer

    def do_POST(self):
        """Handle POST request"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            # Parse JSON payload
            payload = json.loads(post_data.decode('utf-8'))

            # Create event
            event = Event(
                event_type=EventType.HTTP_REQUEST,
                source=f"webhook:{self.path}",
                data=payload
            )

            # Publish to event bus
            if self.event_bus:
                self.event_bus.publish(event)

            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            response = {
                "status": "success",
                "event_id": event.event_id
            }
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(str(e).encode())

    def log_message(self, format, *args):
        """Suppress default logging"""
        pass

class WebhookServer:
    """Webhook server for receiving events"""

    def __init__(self, event_bus: EventBus, port: int = 8080):
        self.event_bus = event_bus
        self.port = port
        self.server = None
        self.thread = None

    def start(self):
        """Start webhook server"""
        # Set event bus on handler class
        WebhookHandler.event_bus = self.event_bus

        self.server = HTTPServer(('localhost', self.port), WebhookHandler)

        def run_server():
            print(f"Webhook server listening on http://localhost:{self.port}")
            self.server.serve_forever()

        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop webhook server"""
        if self.server:
            self.server.shutdown()
            print("Webhook server stopped")

# Example usage
webhook_server = WebhookServer(bus, port=8080)
webhook_server.start()

# Register handler for webhook events
def on_webhook(event: Event):
    """Handle webhook"""
    print(f"    Received webhook from: {event.source}")
    print(f"    Payload: {event.data}")

bus.subscribe(EventType.HTTP_REQUEST, on_webhook)

# To test: POST to http://localhost:8080/webhook
# curl -X POST http://localhost:8080/webhook -d '{"message": "hello"}' -H "Content-Type: application/json"
```

### File System Watcher

```python
import os
import time
from pathlib import Path

class FileSystemWatcher:
    """Watch filesystem for changes"""

    def __init__(self, event_bus: EventBus, watch_path: str):
        self.event_bus = event_bus
        self.watch_path = Path(watch_path)
        self.known_files = set()
        self.running = False

    def start(self):
        """Start watching filesystem"""
        self.running = True
        self.known_files = set(self.watch_path.glob("*"))

        def watch():
            print(f"Watching filesystem: {self.watch_path}")

            while self.running:
                current_files = set(self.watch_path.glob("*"))

                # Check for new files
                new_files = current_files - self.known_files
                for file_path in new_files:
                    if file_path.is_file():
                        event = Event(
                            event_type=EventType.FILE_CREATED,
                            source="filesystem_watcher",
                            data={
                                "filename": file_path.name,
                                "path": str(file_path),
                                "size": file_path.stat().st_size
                            }
                        )
                        self.event_bus.publish(event)

                # Check for modified files
                for file_path in current_files & self.known_files:
                    if file_path.is_file():
                        # Check modification time
                        # (simplified - in practice, track mtimes)
                        pass

                self.known_files = current_files
                time.sleep(1)  # Poll interval

        threading.Thread(target=watch, daemon=True).start()

    def stop(self):
        """Stop watching"""
        self.running = False

# Example
# watcher = FileSystemWatcher(bus, "./watched_folder")
# watcher.start()
```

## Scheduled Workflows

Time-based event triggering.

### Scheduler

```python
import schedule
import time
from datetime import datetime, timedelta

class ScheduledEventPublisher:
    """Publish events on schedule"""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.running = False
        self.jobs = []

    def schedule_recurring(
        self,
        interval_seconds: int,
        event_data: Dict[str, Any],
        source: str = "scheduler"
    ):
        """Schedule recurring event"""
        def publish_event():
            event = Event(
                event_type=EventType.SCHEDULED,
                source=source,
                data=event_data
            )
            self.event_bus.publish(event)

        # Schedule using schedule library
        job = schedule.every(interval_seconds).seconds.do(publish_event)
        self.jobs.append(job)

        print(f"Scheduled recurring event every {interval_seconds}s")

    def schedule_at(
        self,
        time_str: str,
        event_data: Dict[str, Any],
        source: str = "scheduler"
    ):
        """Schedule event at specific time"""
        def publish_event():
            event = Event(
                event_type=EventType.SCHEDULED,
                source=source,
                data=event_data
            )
            self.event_bus.publish(event)

        job = schedule.every().day.at(time_str).do(publish_event)
        self.jobs.append(job)

        print(f"Scheduled daily event at {time_str}")

    def schedule_cron(
        self,
        cron_expression: str,
        event_data: Dict[str, Any],
        source: str = "scheduler"
    ):
        """Schedule using cron-like expression"""
        # Simplified cron parser
        # In practice, use croniter library
        print(f"Scheduled cron job: {cron_expression}")

    def start(self):
        """Start scheduler"""
        self.running = True

        def run_scheduler():
            print("Scheduler started")
            while self.running:
                schedule.run_pending()
                time.sleep(1)

        threading.Thread(target=run_scheduler, daemon=True).start()

    def stop(self):
        """Stop scheduler"""
        self.running = False
        schedule.clear()
        print("Scheduler stopped")

# Example
scheduler = ScheduledEventPublisher(bus)

# Schedule recurring backup
scheduler.schedule_recurring(
    interval_seconds=5,
    event_data={"task": "backup", "type": "incremental"}
)

# Schedule daily report
scheduler.schedule_at(
    time_str="09:00",
    event_data={"task": "generate_report", "report_type": "daily"}
)

# Start scheduler
scheduler.start()

# Handler for scheduled events
def on_scheduled_event(event: Event):
    """Handle scheduled event"""
    task = event.data.get("task")
    print(f"    Executing scheduled task: {task}")

bus.subscribe(EventType.SCHEDULED, on_scheduled_event)
```

## Message Queues

Queue-based event processing.

### Queue Processor

```python
from queue import Queue, Empty
from typing import Callable

class MessageQueue:
    """Message queue for events"""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.queue = Queue()
        self.running = False
        self.workers = []

    def enqueue(self, event: Event):
        """Add event to queue"""
        self.queue.put(event)
        print(f"Event queued: {event.event_id}")

    def start_workers(self, num_workers: int = 2):
        """Start worker threads"""
        self.running = True

        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)

        print(f"Started {num_workers} queue workers")

    def _worker(self, worker_id: int):
        """Worker thread"""
        print(f"Worker {worker_id} started")

        while self.running:
            try:
                # Get event from queue
                event = self.queue.get(timeout=1)

                print(f"Worker {worker_id} processing: {event.event_id}")

                # Publish to event bus
                self.event_bus.publish(event)

                self.queue.task_done()

            except Empty:
                continue

        print(f"Worker {worker_id} stopped")

    def stop(self):
        """Stop workers"""
        self.running = False

        for worker in self.workers:
            worker.join()

        print("Queue workers stopped")

    def get_queue_size(self) -> int:
        """Get queue size"""
        return self.queue.qsize()

# Example
message_queue = MessageQueue(bus)
message_queue.start_workers(num_workers=2)

# Enqueue events
for i in range(5):
    event = Event(
        event_type=EventType.MESSAGE_RECEIVED,
        source="queue",
        data={"message_id": i, "content": f"Message {i}"}
    )
    message_queue.enqueue(event)

time.sleep(2)  # Let workers process

print(f"Queue size: {message_queue.get_queue_size()}")
```

## Pub/Sub Patterns

Publisher-subscriber messaging.

### Pub/Sub System

```python
class Topic:
    """Pub/Sub topic"""

    def __init__(self, name: str):
        self.name = name
        self.subscribers: List[Callable[[Event], None]] = []

    def subscribe(self, handler: Callable[[Event], None]):
        """Subscribe to topic"""
        self.subscribers.append(handler)
        print(f"Subscribed to topic: {self.name}")

    def publish(self, event: Event):
        """Publish event to all subscribers"""
        print(f"\nPublishing to topic '{self.name}': {event.event_id}")

        for handler in self.subscribers:
            try:
                handler(event)
            except Exception as e:
                print(f"  Subscriber error: {e}")

class PubSubSystem:
    """Pub/Sub messaging system"""

    def __init__(self):
        self.topics: Dict[str, Topic] = {}

    def create_topic(self, topic_name: str) -> Topic:
        """Create new topic"""
        if topic_name not in self.topics:
            self.topics[topic_name] = Topic(topic_name)
            print(f"Created topic: {topic_name}")

        return self.topics[topic_name]

    def get_topic(self, topic_name: str) -> Optional[Topic]:
        """Get topic"""
        return self.topics.get(topic_name)

    def publish(self, topic_name: str, event: Event):
        """Publish to topic"""
        topic = self.topics.get(topic_name)
        if topic:
            topic.publish(event)
        else:
            print(f"Topic not found: {topic_name}")

    def subscribe(self, topic_name: str, handler: Callable[[Event], None]):
        """Subscribe to topic"""
        topic = self.create_topic(topic_name)
        topic.subscribe(handler)

# Example
pubsub = PubSubSystem()

# Create topics
pubsub.create_topic("user.created")
pubsub.create_topic("user.updated")
pubsub.create_topic("order.placed")

# Subscribe to topics
def send_welcome_email(event: Event):
    """Send welcome email"""
    user_id = event.data.get("user_id")
    print(f"  → Sending welcome email to user {user_id}")

def setup_user_account(event: Event):
    """Setup user account"""
    user_id = event.data.get("user_id")
    print(f"  → Setting up account for user {user_id}")

def process_order(event: Event):
    """Process order"""
    order_id = event.data.get("order_id")
    print(f"  → Processing order {order_id}")

pubsub.subscribe("user.created", send_welcome_email)
pubsub.subscribe("user.created", setup_user_account)
pubsub.subscribe("order.placed", process_order)

# Publish events
pubsub.publish("user.created", Event(
    event_type=EventType.USER_ACTION,
    source="api",
    data={"user_id": "user_123", "email": "user@example.com"}
))

pubsub.publish("order.placed", Event(
    event_type=EventType.USER_ACTION,
    source="api",
    data={"order_id": "order_456", "total": 99.99}
))
```

## Reactive Agents

Building agents that react to events.

### Reactive Agent

```python
class ReactiveAgent:
    """Agent that reacts to events"""

    def __init__(self, agent_id: str, event_bus: EventBus):
        self.agent_id = agent_id
        self.event_bus = event_bus
        self.subscriptions = []

    def register_reaction(
        self,
        event_type: EventType,
        condition: Optional[Callable[[Event], bool]] = None
    ):
        """Register reaction to event type"""
        def decorator(func):
            def handler(event: Event):
                # Check condition
                if condition and not condition(event):
                    return

                print(f"\n[{self.agent_id}] Reacting to {event.event_type.value}")
                func(self, event)

            self.event_bus.subscribe(event_type, handler)
            self.subscriptions.append((event_type, handler))

            return func

        return decorator

    def stop(self):
        """Stop agent"""
        for event_type, handler in self.subscriptions:
            self.event_bus.unsubscribe(event_type, handler)

        print(f"Agent {self.agent_id} stopped")

# Example: Customer support agent
class CustomerSupportAgent(ReactiveAgent):
    """Agent that handles customer support events"""

    def __init__(self, event_bus: EventBus):
        super().__init__("customer_support", event_bus)
        self.setup_reactions()

    def setup_reactions(self):
        """Setup event reactions"""

        @self.register_reaction(
            EventType.MESSAGE_RECEIVED,
            condition=lambda e: e.data.get("channel") == "support"
        )
        def handle_support_message(self, event: Event):
            """Handle support message"""
            message = event.data.get("content")
            customer_id = event.data.get("customer_id")

            print(f"  Processing support request from {customer_id}")
            print(f"  Message: {message}")

            # Analyze message
            if "urgent" in message.lower():
                print(f"  → Escalating urgent request")
            else:
                print(f"  → Generating automated response")

        @self.register_reaction(EventType.USER_ACTION)
        def handle_user_action(self, event: Event):
            """Handle user action"""
            action = event.data.get("action")
            user_id = event.data.get("user_id")

            print(f"  Logging user action: {action} by {user_id}")

# Example usage
support_agent = CustomerSupportAgent(bus)

# Publish support events
bus.publish(Event(
    event_type=EventType.MESSAGE_RECEIVED,
    source="chat",
    data={
        "channel": "support",
        "customer_id": "cust_123",
        "content": "I need urgent help with my order"
    }
))

bus.publish(Event(
    event_type=EventType.USER_ACTION,
    source="web_app",
    data={
        "user_id": "user_456",
        "action": "viewed_product",
        "product_id": "prod_789"
    }
))
```

## Error Handling and Retry

Handling failures in event processing.

### Retry Manager

```python
@dataclass
class FailedEvent:
    """Failed event record"""
    event: Event
    error: str
    attempt: int
    last_attempt: datetime

class RetryManager:
    """Manage event retry logic"""

    def __init__(self, event_bus: EventBus, max_retries: int = 3):
        self.event_bus = event_bus
        self.max_retries = max_retries
        self.failed_events: List[FailedEvent] = []
        self.dead_letter_queue: List[Event] = []

    def wrap_handler(self, handler: Callable[[Event], None]) -> Callable[[Event], None]:
        """Wrap handler with retry logic"""
        def wrapped_handler(event: Event):
            attempt = 0

            while attempt < self.max_retries:
                try:
                    handler(event)
                    return  # Success

                except Exception as e:
                    attempt += 1
                    print(f"  ✗ Handler failed (attempt {attempt}/{self.max_retries}): {e}")

                    if attempt < self.max_retries:
                        # Exponential backoff
                        delay = 2 ** attempt
                        print(f"  Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        # Max retries reached
                        print(f"  ✗ Max retries reached, moving to dead letter queue")

                        self.failed_events.append(FailedEvent(
                            event=event,
                            error=str(e),
                            attempt=attempt,
                            last_attempt=datetime.now()
                        ))

                        self.dead_letter_queue.append(event)

        return wrapped_handler

    def get_failed_events(self) -> List[FailedEvent]:
        """Get failed events"""
        return self.failed_events

    def retry_dead_letter(self):
        """Retry events in dead letter queue"""
        print(f"\nRetrying {len(self.dead_letter_queue)} failed events...")

        for event in self.dead_letter_queue:
            self.event_bus.publish(event)

        self.dead_letter_queue.clear()

# Example
retry_mgr = RetryManager(bus, max_retries=3)

# Unreliable handler
attempt_count = {}

def unreliable_handler(event: Event):
    """Handler that fails sometimes"""
    event_id = event.event_id

    if event_id not in attempt_count:
        attempt_count[event_id] = 0

    attempt_count[event_id] += 1

    # Fail first 2 attempts, succeed on 3rd
    if attempt_count[event_id] < 3:
        raise Exception(f"Simulated failure #{attempt_count[event_id]}")

    print(f"  ✓ Handler succeeded on attempt {attempt_count[event_id]}")

# Wrap handler with retry logic
wrapped_handler = retry_mgr.wrap_handler(unreliable_handler)

bus.subscribe(EventType.MESSAGE_RECEIVED, wrapped_handler)

# Publish event
bus.publish(Event(
    event_type=EventType.MESSAGE_RECEIVED,
    source="test",
    data={"message": "test"}
))
```

## Real-World Applications

### Pattern: Document Processing Pipeline

```python
class DocumentProcessingSystem:
    """Event-driven document processing"""

    def __init__(self):
        self.event_bus = EventBus()
        self.setup_pipeline()

    def setup_pipeline(self):
        """Setup event-driven pipeline"""

        # Stage 1: Document uploaded
        def on_document_uploaded(event: Event):
            """Handle document upload"""
            doc_id = event.data.get("doc_id")
            print(f"\n[Stage 1] Document uploaded: {doc_id}")

            # Publish next event
            self.event_bus.publish(Event(
                event_type=EventType.SYSTEM_EVENT,
                source="pipeline",
                data={"doc_id": doc_id, "stage": "extract_text"},
                metadata={"pipeline_stage": 1}
            ))

        # Stage 2: Extract text
        def on_extract_text(event: Event):
            """Extract text from document"""
            if event.data.get("stage") == "extract_text":
                doc_id = event.data.get("doc_id")
                print(f"[Stage 2] Extracting text from: {doc_id}")

                # Simulate text extraction
                extracted_text = "Document text content..."

                # Publish next event
                self.event_bus.publish(Event(
                    event_type=EventType.SYSTEM_EVENT,
                    source="pipeline",
                    data={
                        "doc_id": doc_id,
                        "stage": "analyze",
                        "text": extracted_text
                    },
                    metadata={"pipeline_stage": 2}
                ))

        # Stage 3: Analyze
        def on_analyze(event: Event):
            """Analyze document"""
            if event.data.get("stage") == "analyze":
                doc_id = event.data.get("doc_id")
                text = event.data.get("text")
                print(f"[Stage 3] Analyzing: {doc_id}")

                # Simulate analysis
                analysis = {"sentiment": "positive", "topics": ["AI", "automation"]}

                # Publish completion event
                self.event_bus.publish(Event(
                    event_type=EventType.SYSTEM_EVENT,
                    source="pipeline",
                    data={
                        "doc_id": doc_id,
                        "stage": "complete",
                        "analysis": analysis
                    },
                    metadata={"pipeline_stage": 3}
                ))

        # Subscribe handlers
        self.event_bus.subscribe(EventType.FILE_CREATED, on_document_uploaded)
        self.event_bus.subscribe(EventType.SYSTEM_EVENT, on_extract_text)
        self.event_bus.subscribe(EventType.SYSTEM_EVENT, on_analyze)

    def process_document(self, doc_id: str):
        """Start document processing"""
        print(f"\n{'='*60}")
        print(f"Starting document processing: {doc_id}")
        print(f"{'='*60}")

        # Trigger pipeline
        self.event_bus.publish(Event(
            event_type=EventType.FILE_CREATED,
            source="upload",
            data={"doc_id": doc_id, "filename": f"{doc_id}.pdf"}
        ))

# Usage
system = DocumentProcessingSystem()
system.process_document("doc_001")
```

## Summary

Event-driven orchestration enables reactive, scalable agent systems:

**Key Concepts**:

- **Events**: Things that happen
- **Triggers**: Start workflows from events
- **Handlers**: React to events
- **Pub/Sub**: Decouple components
- **Queues**: Buffer and process events

**Event Sources**:

- HTTP webhooks
- File system changes
- Scheduled times
- Message queues
- User actions

**Patterns**:

- **Event Bus**: Central event distribution
- **Pub/Sub**: Topic-based messaging
- **Message Queue**: Buffered processing
- **Reactive Agent**: Event-driven behavior
- **Retry Logic**: Handle failures

**Benefits**:

- Resource efficient (no polling)
- Real-time responsiveness
- Loosely coupled components
- Scalable architecture
- Easy to extend

**Best Practices**:

1. Define clear event schemas
2. Implement idempotent handlers
3. Add retry logic
4. Use dead letter queues
5. Monitor event flows
6. Version event formats
7. Implement circuit breakers

Event-driven architectures enable responsive, scalable agent orchestration.

## Next Steps

Explore related orchestration topics:

- **[Workflow Patterns](workflow-patterns.md)**: Event-driven workflows
- **[Background Processes](background-processes.md)**: Async event processing
- **[Parallelization](parallelization.md)**: Concurrent event handling

Related areas:

- **[Agent Architectures](../agent-architectures/)**: Building reactive agents
- **[Tool Use](../tool-use/)**: Event-triggered tool calls
