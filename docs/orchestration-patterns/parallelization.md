# Parallelization and Concurrency

## Table of Contents

- [Introduction](#introduction)
- [Why Parallelization Matters](#why-parallelization-matters)
- [Concurrency Models](#concurrency-models)
- [Parallel Tool Calls](#parallel-tool-calls)
- [Concurrent Reasoning](#concurrent-reasoning)
- [Fan-Out Fan-In Pattern](#fan-out-fan-in-pattern)
- [Asynchronous Execution](#asynchronous-execution)
- [Thread-Based Parallelism](#thread-based-parallelism)
- [Process-Based Parallelism](#process-based-parallelism)
- [Async/Await Patterns](#asyncawait-patterns)
- [Coordination Challenges](#coordination-challenges)
- [Synchronization Primitives](#synchronization-primitives)
- [Load Balancing](#load-balancing)
- [Performance Considerations](#performance-considerations)
- [Real-World Scenarios](#real-world-scenarios)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Parallelization enables agents to do multiple things at once, dramatically improving throughput and reducing latency. Instead of executing tasks sequentially, parallel execution can:

- **Speed up** independent operations by running them simultaneously
- **Improve responsiveness** by handling multiple requests concurrently
- **Increase throughput** by utilizing multiple cores/workers
- **Reduce latency** for operations that would otherwise block

> "Sequential execution is simple. Parallel execution is fast."

However, parallelization introduces complexity: coordination overhead, race conditions, resource contention, and debugging challenges. The art is knowing when parallelization helps and how to manage its complexity.

### When to Parallelize

```
Parallelize when:
  ✓ Operations are independent
  ✓ Operations are I/O-bound
  ✓ Speedup outweighs coordination cost
  ✓ Operations can fail independently

Don't parallelize when:
  ✗ Operations have dependencies
  ✗ Operations are CPU-bound (without multi-core)
  ✗ Coordination cost exceeds benefit
  ✗ Shared state requires heavy synchronization
```

### Parallel Execution Model

```
Sequential:
   Step 1 → Step 2 → Step 3 → Step 4
   Total time: T1 + T2 + T3 + T4

Parallel:
   ┌─ Step 1 ─┐
   ├─ Step 2 ─┤
   ├─ Step 3 ─┤
   └─ Step 4 ─┘
   Total time: max(T1, T2, T3, T4) + coordination overhead
```

## Why Parallelization Matters

### Performance Gains

```python
import time
from concurrent.futures import ThreadPoolExecutor

def sequential_execution(tasks):
    """Execute tasks sequentially"""
    start = time.time()
    results = []
    
    for task in tasks:
        result = task()
        results.append(result)
    
    duration = time.time() - start
    print(f"Sequential: {duration:.2f}s")
    return results

def parallel_execution(tasks, max_workers=4):
    """Execute tasks in parallel"""
    start = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda t: t(), tasks))
    
    duration = time.time() - start
    print(f"Parallel: {duration:.2f}s")
    return results

# Example: API calls
def make_api_call(api_id):
    """Simulate API call (I/O-bound)"""
    def call():
        time.sleep(1)  # Simulate network delay
        return {"api": api_id, "data": f"result_{api_id}"}
    return call

tasks = [make_api_call(i) for i in range(10)]

# Sequential: ~10 seconds
sequential_results = sequential_execution(tasks)

# Parallel: ~2.5 seconds (4 workers)
parallel_results = parallel_execution(tasks, max_workers=4)

# Speedup: ~4x
```

### Amdahl's Law

```python
def calculate_speedup(parallel_fraction: float, num_processors: int) -> float:
    """Calculate theoretical speedup using Amdahl's Law"""
    # Speedup = 1 / ((1 - P) + P / N)
    # P = parallel fraction, N = number of processors
    sequential_fraction = 1 - parallel_fraction
    speedup = 1 / (sequential_fraction + (parallel_fraction / num_processors))
    return speedup

# Example: 90% parallel code
for n_procs in [2, 4, 8, 16]:
    speedup = calculate_speedup(0.9, n_procs)
    print(f"{n_procs} processors: {speedup:.2f}x speedup")

# Output:
# 2 processors: 1.82x speedup
# 4 processors: 3.08x speedup
# 8 processors: 4.71x speedup
# 16 processors: 6.40x speedup
```

## Concurrency Models

Different approaches to concurrent execution.

### Threading Model

```python
import threading
from typing import List, Callable, Any

class ThreadedExecutor:
    """Execute tasks using threads"""
    
    def __init__(self, max_threads: int = 4):
        self.max_threads = max_threads
        self.results = []
        self.lock = threading.Lock()
    
    def execute(self, tasks: List[Callable]) -> List[Any]:
        """Execute tasks in threads"""
        threads = []
        
        def worker(task_id: int, task: Callable):
            result = task()
            with self.lock:
                self.results.append((task_id, result))
        
        # Create and start threads
        for i, task in enumerate(tasks):
            thread = threading.Thread(target=worker, args=(i, task))
            thread.start()
            threads.append(thread)
            
            # Limit concurrent threads
            if len(threads) >= self.max_threads:
                threads[0].join()
                threads.pop(0)
        
        # Wait for remaining threads
        for thread in threads:
            thread.join()
        
        # Sort by task ID to maintain order
        self.results.sort(key=lambda x: x[0])
        return [result for _, result in self.results]

# Usage
executor = ThreadedExecutor(max_threads=4)
tasks = [lambda: fetch_data(i) for i in range(10)]
results = executor.execute(tasks)
```

### Process Model

```python
from multiprocessing import Pool, cpu_count
from typing import List, Callable

class ProcessPoolExecutor:
    """Execute tasks using processes"""
    
    def __init__(self, max_processes: int = None):
        self.max_processes = max_processes or cpu_count()
    
    def execute(self, func: Callable, tasks: List[Any]) -> List[Any]:
        """Execute function on tasks in parallel"""
        with Pool(processes=self.max_processes) as pool:
            results = pool.map(func, tasks)
        return results

# Usage (for CPU-bound tasks)
def cpu_intensive_task(data):
    """CPU-intensive computation"""
    result = 0
    for i in range(1000000):
        result += data * i
    return result

executor = ProcessPoolExecutor(max_processes=4)
data = list(range(100))
results = executor.execute(cpu_intensive_task, data)
```

### Async/Await Model

```python
import asyncio
from typing import List, Callable, Any

class AsyncExecutor:
    """Execute tasks using async/await"""
    
    async def execute(self, tasks: List[Callable]) -> List[Any]:
        """Execute async tasks concurrently"""
        # Gather all tasks
        results = await asyncio.gather(*[task() for task in tasks])
        return results

# Usage
async def async_api_call(api_id: int):
    """Async API call"""
    await asyncio.sleep(1)  # Simulate async I/O
    return {"api": api_id, "data": f"result_{api_id}"}

async def main():
    executor = AsyncExecutor()
    tasks = [lambda i=i: async_api_call(i) for i in range(10)]
    results = await executor.execute(tasks)
    return results

# Run
# results = asyncio.run(main())
```

## Parallel Tool Calls

Executing multiple tool calls simultaneously.

### Parallel Tool Executor

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import time

class ParallelToolExecutor:
    """Execute multiple tool calls in parallel"""
    
    def __init__(self, tools: Dict[str, Callable], max_workers: int = 4):
        self.tools = tools
        self.max_workers = max_workers
    
    def execute_tools(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute multiple tool calls in parallel"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tool calls
            future_to_call = {
                executor.submit(
                    self._execute_tool,
                    call["tool"],
                    call.get("args", {}),
                    call.get("id", f"call_{i}")
                ): call
                for i, call in enumerate(tool_calls)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_call):
                call = future_to_call[future]
                call_id = call.get("id", "unknown")
                
                try:
                    result = future.result()
                    results[call_id] = {
                        "success": True,
                        "result": result
                    }
                except Exception as e:
                    results[call_id] = {
                        "success": False,
                        "error": str(e)
                    }
        
        return results
    
    def _execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        call_id: str
    ) -> Any:
        """Execute single tool call"""
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
        
        print(f"Executing {tool_name} (call_id: {call_id})")
        start = time.time()
        result = tool(**args)
        duration = time.time() - start
        print(f"Completed {tool_name} in {duration:.2f}s")
        
        return result

# Example tools
def search_web(query: str) -> dict:
    """Search the web"""
    time.sleep(1)  # Simulate network delay
    return {"results": [f"Result for {query}"]}

def query_database(table: str, filter: str) -> dict:
    """Query database"""
    time.sleep(1)
    return {"rows": [{"id": 1, "data": "..."}]}

def call_api(endpoint: str) -> dict:
    """Call external API"""
    time.sleep(1)
    return {"data": f"Data from {endpoint}"}

# Execute tools in parallel
tools = {
    "search_web": search_web,
    "query_database": query_database,
    "call_api": call_api
}

executor = ParallelToolExecutor(tools, max_workers=3)

tool_calls = [
    {"id": "search", "tool": "search_web", "args": {"query": "AI"}},
    {"id": "db", "tool": "query_database", "args": {"table": "users", "filter": "active"}},
    {"id": "api", "tool": "call_api", "args": {"endpoint": "/data"}}
]

results = executor.execute_tools(tool_calls)
# All three tools execute in parallel, total time ~1s instead of 3s
```

### Dependency-Aware Parallel Execution

```python
from typing import Set
from collections import defaultdict

class DependencyAwareExecutor:
    """Execute tool calls respecting dependencies"""
    
    def __init__(self, tools: Dict[str, Callable]):
        self.tools = tools
    
    def execute_with_dependencies(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute tool calls in parallel where possible"""
        # Build dependency graph
        dependencies = self._build_dependency_graph(tool_calls)
        
        results = {}
        executed = set()
        
        while len(executed) < len(tool_calls):
            # Find calls ready to execute (all dependencies met)
            ready = []
            for call in tool_calls:
                call_id = call["id"]
                if call_id in executed:
                    continue
                
                deps = dependencies.get(call_id, set())
                if deps.issubset(executed):
                    ready.append(call)
            
            if not ready:
                raise RuntimeError("Circular dependency or missing dependencies")
            
            # Execute ready calls in parallel
            batch_results = self._execute_batch(ready)
            results.update(batch_results)
            
            # Mark as executed
            for call in ready:
                executed.add(call["id"])
        
        return results
    
    def _build_dependency_graph(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> Dict[str, Set[str]]:
        """Build dependency graph from tool calls"""
        dependencies = {}
        
        for call in tool_calls:
            call_id = call["id"]
            deps = set(call.get("depends_on", []))
            dependencies[call_id] = deps
        
        return dependencies
    
    def _execute_batch(
        self,
        calls: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute batch of calls in parallel"""
        results = {}
        
        with ThreadPoolExecutor() as executor:
            future_to_call = {
                executor.submit(
                    self.tools[call["tool"]],
                    **call.get("args", {})
                ): call
                for call in calls
            }
            
            for future in as_completed(future_to_call):
                call = future_to_call[future]
                results[call["id"]] = future.result()
        
        return results

# Example with dependencies
tools = {
    "fetch_user": lambda user_id: {"id": user_id, "name": f"User{user_id}"},
    "fetch_posts": lambda user_id: {"posts": [f"post_{user_id}_1"]},
    "fetch_comments": lambda post_id: {"comments": ["comment_1"]},
}

executor = DependencyAwareExecutor(tools)

calls = [
    {
        "id": "user",
        "tool": "fetch_user",
        "args": {"user_id": 1},
        "depends_on": []
    },
    {
        "id": "posts",
        "tool": "fetch_posts",
        "args": {"user_id": 1},
        "depends_on": ["user"]  # Depends on user
    },
    {
        "id": "comments",
        "tool": "fetch_comments",
        "args": {"post_id": "post_1"},
        "depends_on": ["posts"]  # Depends on posts
    }
]

results = executor.execute_with_dependencies(calls)
# Execution order: user, then posts, then comments
# But independent calls at each level execute in parallel
```

## Concurrent Reasoning

Running multiple reasoning processes in parallel.

### Parallel Reasoning Paths

```python
class ParallelReasoning:
    """Explore multiple reasoning paths in parallel"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def reason_parallel(
        self,
        problem: str,
        num_paths: int = 3,
        temperature: float = 0.8
    ) -> dict:
        """Generate multiple reasoning paths in parallel"""
        
        def generate_reasoning():
            """Generate one reasoning path"""
            prompt = f"""
            Problem: {problem}
            
            Think step-by-step and provide your reasoning.
            """
            response = self.llm.complete(prompt, temperature=temperature)
            return response
        
        # Generate multiple paths in parallel
        with ThreadPoolExecutor(max_workers=num_paths) as executor:
            futures = [executor.submit(generate_reasoning) for _ in range(num_paths)]
            paths = [f.result() for f in futures]
        
        # Evaluate paths
        best_path = self._select_best_path(paths)
        
        return {
            "paths": paths,
            "best_path": best_path,
            "num_paths": num_paths
        }
    
    def _select_best_path(self, paths: List[str]) -> str:
        """Select best reasoning path"""
        # Could use voting, confidence scores, or another LLM call
        # For simplicity, return the longest path
        return max(paths, key=len)

# Usage
reasoner = ParallelReasoning(llm_client)
result = reasoner.reason_parallel(
    "How can we optimize database query performance?",
    num_paths=5
)

print(f"Generated {len(result['paths'])} reasoning paths")
print(f"Best path: {result['best_path']}")
```

### Self-Consistency via Parallel Sampling

```python
from collections import Counter

class SelfConsistencyReasoner:
    """Use parallel sampling for self-consistency"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def reason_with_self_consistency(
        self,
        question: str,
        num_samples: int = 5
    ) -> dict:
        """Sample multiple answers and use majority vote"""
        
        def generate_answer():
            """Generate one answer"""
            prompt = f"""
            Question: {question}
            
            Think step-by-step and provide your final answer.
            End with "Final Answer: [your answer]"
            """
            response = self.llm.complete(prompt, temperature=0.7)
            return self._extract_answer(response)
        
        # Generate multiple answers in parallel
        with ThreadPoolExecutor(max_workers=num_samples) as executor:
            futures = [executor.submit(generate_answer) for _ in range(num_samples)]
            answers = [f.result() for f in futures]
        
        # Count answers
        answer_counts = Counter(answers)
        most_common = answer_counts.most_common(1)[0]
        consensus_answer = most_common[0]
        confidence = most_common[1] / len(answers)
        
        return {
            "answer": consensus_answer,
            "confidence": confidence,
            "all_answers": answers,
            "distribution": dict(answer_counts)
        }
    
    def _extract_answer(self, response: str) -> str:
        """Extract final answer from response"""
        if "Final Answer:" in response:
            return response.split("Final Answer:")[-1].strip()
        return response.strip()

# Usage
reasoner = SelfConsistencyReasoner(llm_client)
result = reasoner.reason_with_self_consistency(
    "If a store sells apples for $2 each and oranges for $3 each, "
    "and you buy 5 apples and 3 oranges, how much do you spend?",
    num_samples=10
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Distribution: {result['distribution']}")
```

## Fan-Out Fan-In Pattern

Distributing work and collecting results.

### Scatter-Gather Implementation

```python
from typing import TypeVar, Generic

T = TypeVar('T')
R = TypeVar('R')

class ScatterGather(Generic[T, R]):
    """Scatter-gather pattern for parallel processing"""
    
    def __init__(
        self,
        scatter: Callable[[T], List[Any]],
        process: Callable[[Any], R],
        gather: Callable[[List[R]], Any]
    ):
        self.scatter = scatter
        self.process = process
        self.gather = gather
    
    def execute(self, input_data: T, max_workers: int = 4) -> Any:
        """Execute scatter-gather pattern"""
        # Scatter: split input into work units
        work_units = self.scatter(input_data)
        print(f"Scattered into {len(work_units)} work units")
        
        # Process: execute in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.process, work_units))
        
        # Gather: combine results
        final_result = self.gather(results)
        return final_result

# Example: Parallel document processing
def scatter_documents(documents: List[dict]) -> List[dict]:
    """Split documents into individual items"""
    return documents

def process_document(doc: dict) -> dict:
    """Process single document"""
    return {
        "id": doc["id"],
        "summary": summarize(doc["content"]),
        "keywords": extract_keywords(doc["content"]),
        "sentiment": analyze_sentiment(doc["content"])
    }

def gather_results(results: List[dict]) -> dict:
    """Gather processed documents"""
    return {
        "total": len(results),
        "documents": results,
        "aggregate_sentiment": sum(r["sentiment"] for r in results) / len(results)
    }

# Execute
sg = ScatterGather(scatter_documents, process_document, gather_results)
documents = load_documents()
result = sg.execute(documents, max_workers=8)
```

### Map-Reduce Pattern

```python
from functools import reduce
from typing import Callable, List, Any

class MapReduce:
    """Map-reduce pattern for parallel processing"""
    
    def __init__(
        self,
        map_func: Callable[[Any], Any],
        reduce_func: Callable[[Any, Any], Any],
        max_workers: int = 4
    ):
        self.map_func = map_func
        self.reduce_func = reduce_func
        self.max_workers = max_workers
    
    def execute(self, data: List[Any]) -> Any:
        """Execute map-reduce"""
        # Map phase (parallel)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            mapped = list(executor.map(self.map_func, data))
        
        # Reduce phase (sequential)
        result = reduce(self.reduce_func, mapped)
        
        return result

# Example: Word count
def map_word_count(text: str) -> dict:
    """Map: count words in text"""
    words = text.lower().split()
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts

def reduce_word_count(counts1: dict, counts2: dict) -> dict:
    """Reduce: merge word counts"""
    merged = counts1.copy()
    for word, count in counts2.items():
        merged[word] = merged.get(word, 0) + count
    return merged

# Execute
mr = MapReduce(map_word_count, reduce_word_count, max_workers=4)
documents = ["text one", "text two", "more text"]
word_counts = mr.execute(documents)
```

## Asynchronous Execution

Non-blocking execution patterns.

### Async Tool Executor

```python
import asyncio
from typing import List, Dict, Any

class AsyncToolExecutor:
    """Execute tools asynchronously"""
    
    def __init__(self, tools: Dict[str, Callable]):
        self.tools = tools
    
    async def execute_tools_async(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute tool calls asynchronously"""
        
        async def execute_tool(call: Dict[str, Any]) -> tuple:
            """Execute single tool call"""
            call_id = call["id"]
            tool_name = call["tool"]
            args = call.get("args", {})
            
            tool = self.tools.get(tool_name)
            if not tool:
                return call_id, {"error": f"Tool not found: {tool_name}"}
            
            try:
                # If tool is async, await it
                if asyncio.iscoroutinefunction(tool):
                    result = await tool(**args)
                else:
                    # Run sync tool in executor
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: tool(**args))
                
                return call_id, {"success": True, "result": result}
            except Exception as e:
                return call_id, {"success": False, "error": str(e)}
        
        # Execute all calls concurrently
        tasks = [execute_tool(call) for call in tool_calls]
        results_list = await asyncio.gather(*tasks)
        
        # Convert to dictionary
        results = {call_id: result for call_id, result in results_list}
        return results

# Example async tools
async def async_fetch_url(url: str) -> dict:
    """Async HTTP fetch"""
    # Simulate async I/O
    await asyncio.sleep(1)
    return {"url": url, "content": f"Content from {url}"}

async def async_query_db(query: str) -> dict:
    """Async database query"""
    await asyncio.sleep(0.5)
    return {"query": query, "results": ["row1", "row2"]}

# Usage
async def main():
    tools = {
        "fetch_url": async_fetch_url,
        "query_db": async_query_db
    }
    
    executor = AsyncToolExecutor(tools)
    
    calls = [
        {"id": "url1", "tool": "fetch_url", "args": {"url": "https://example.com/1"}},
        {"id": "url2", "tool": "fetch_url", "args": {"url": "https://example.com/2"}},
        {"id": "db1", "tool": "query_db", "args": {"query": "SELECT * FROM users"}}
    ]
    
    results = await executor.execute_tools_async(calls)
    return results

# Run
# results = asyncio.run(main())
```

### Background Task Queue

```python
import asyncio
from asyncio import Queue
from typing import Callable, Any

class BackgroundTaskQueue:
    """Async task queue for background execution"""
    
    def __init__(self, num_workers: int = 4):
        self.queue = Queue()
        self.num_workers = num_workers
        self.results = {}
    
    async def add_task(
        self,
        task_id: str,
        func: Callable,
        *args,
        **kwargs
    ):
        """Add task to queue"""
        await self.queue.put({
            "id": task_id,
            "func": func,
            "args": args,
            "kwargs": kwargs
        })
    
    async def worker(self, worker_id: int):
        """Worker that processes tasks from queue"""
        while True:
            task = await self.queue.get()
            
            if task is None:  # Poison pill
                break
            
            task_id = task["id"]
            func = task["func"]
            
            try:
                print(f"Worker {worker_id} processing {task_id}")
                
                # Execute task
                if asyncio.iscoroutinefunction(func):
                    result = await func(*task["args"], **task["kwargs"])
                else:
                    result = func(*task["args"], **task["kwargs"])
                
                self.results[task_id] = {"success": True, "result": result}
            
            except Exception as e:
                self.results[task_id] = {"success": False, "error": str(e)}
            
            finally:
                self.queue.task_done()
    
    async def process_all(self):
        """Process all tasks in queue"""
        # Start workers
        workers = [
            asyncio.create_task(self.worker(i))
            for i in range(self.num_workers)
        ]
        
        # Wait for queue to be empty
        await self.queue.join()
        
        # Stop workers
        for _ in range(self.num_workers):
            await self.queue.put(None)
        
        # Wait for workers to finish
        await asyncio.gather(*workers)
        
        return self.results

# Usage
async def example_task(task_num: int):
    """Example async task"""
    await asyncio.sleep(1)
    return f"Result from task {task_num}"

async def main():
    queue = BackgroundTaskQueue(num_workers=4)
    
    # Add tasks
    for i in range(10):
        await queue.add_task(f"task_{i}", example_task, i)
    
    # Process all
    results = await queue.process_all()
    return results

# results = asyncio.run(main())
```

## Thread-Based Parallelism

Using threads for I/O-bound operations.

### Thread Pool Pattern

```python
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Callable, Any
import queue
import threading

class ManagedThreadPool:
    """Managed thread pool with monitoring"""
    
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks = {}
        self.lock = threading.Lock()
    
    def submit(
        self,
        task_id: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Future:
        """Submit task to thread pool"""
        future = self.executor.submit(func, *args, **kwargs)
        
        with self.lock:
            self.active_tasks[task_id] = {
                "future": future,
                "submitted_at": time.time()
            }
        
        # Add callback to clean up on completion
        def cleanup(f):
            with self.lock:
                self.active_tasks.pop(task_id, None)
        
        future.add_done_callback(cleanup)
        
        return future
    
    def get_active_count(self) -> int:
        """Get number of active tasks"""
        with self.lock:
            return len(self.active_tasks)
    
    def shutdown(self, wait: bool = True):
        """Shutdown thread pool"""
        self.executor.shutdown(wait=wait)

# Usage
pool = ManagedThreadPool(max_workers=4)

# Submit tasks
futures = []
for i in range(10):
    future = pool.submit(f"task_{i}", time.sleep, 1)
    futures.append(future)

# Wait for completion
for future in futures:
    future.result()

print(f"Active tasks: {pool.get_active_count()}")
pool.shutdown()
```

## Process-Based Parallelism

Using processes for CPU-bound operations.

### Process Pool Pattern

```python
from multiprocessing import Pool, Manager, cpu_count
from typing import Callable, List, Any
import os

class ProcessPoolExecutor:
    """Execute CPU-bound tasks using processes"""
    
    def __init__(self, max_processes: int = None):
        self.max_processes = max_processes or cpu_count()
    
    def map(
        self,
        func: Callable,
        items: List[Any],
        chunksize: int = 1
    ) -> List[Any]:
        """Map function over items using processes"""
        with Pool(processes=self.max_processes) as pool:
            results = pool.map(func, items, chunksize=chunksize)
        return results
    
    def starmap(
        self,
        func: Callable,
        items: List[tuple],
        chunksize: int = 1
    ) -> List[Any]:
        """Map function with multiple arguments"""
        with Pool(processes=self.max_processes) as pool:
            results = pool.starmap(func, items, chunksize=chunksize)
        return results
    
    def map_with_progress(
        self,
        func: Callable,
        items: List[Any]
    ) -> List[Any]:
        """Map with progress tracking"""
        manager = Manager()
        progress = manager.Value('i', 0)
        total = len(items)
        
        def tracked_func(item):
            result = func(item)
            with progress.get_lock():
                progress.value += 1
                print(f"Progress: {progress.value}/{total}")
            return result
        
        with Pool(processes=self.max_processes) as pool:
            results = pool.map(tracked_func, items)
        
        return results

# Example: CPU-intensive computation
def compute_intensive(n: int) -> int:
    """CPU-intensive task"""
    result = 0
    for i in range(n):
        result += i ** 2
    return result

# Execute
executor = ProcessPoolExecutor(max_processes=4)
numbers = [1000000] * 8  # 8 tasks
results = executor.map(compute_intensive, numbers)
```

## Coordination Challenges

Managing coordination in parallel systems.

### Race Conditions

```python
import threading

class Counter:
    """Thread-safe counter"""
    
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()
    
    def increment_unsafe(self):
        """Unsafe increment (race condition)"""
        temp = self.value
        temp += 1
        self.value = temp
    
    def increment_safe(self):
        """Safe increment (with lock)"""
        with self.lock:
            self.value += 1

# Demonstrate race condition
def test_race_condition():
    """Test race condition"""
    counter = Counter()
    
    def worker():
        for _ in range(1000):
            counter.increment_unsafe()
    
    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    print(f"Expected: 10000, Got: {counter.value}")
    # Output will be < 10000 due to race condition

# Test with safe version
def test_thread_safe():
    """Test thread-safe version"""
    counter = Counter()
    
    def worker():
        for _ in range(1000):
            counter.increment_safe()
    
    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    print(f"Expected: 10000, Got: {counter.value}")
    # Output will be exactly 10000
```

### Deadlock Prevention

```python
import threading
import time

class DeadlockFreeExecutor:
    """Executor with deadlock prevention"""
    
    def __init__(self):
        self.locks = {}
        self.lock_order = []
    
    def register_lock(self, lock_id: str) -> threading.Lock:
        """Register lock with ordering"""
        lock = threading.Lock()
        self.locks[lock_id] = lock
        self.lock_order.append(lock_id)
        return lock
    
    def acquire_locks(self, lock_ids: List[str]):
        """Acquire multiple locks in consistent order"""
        # Sort lock IDs to ensure consistent ordering
        sorted_ids = sorted(lock_ids, key=lambda x: self.lock_order.index(x))
        
        acquired = []
        try:
            for lock_id in sorted_ids:
                lock = self.locks[lock_id]
                lock.acquire()
                acquired.append(lock)
            return acquired
        except:
            # Release all acquired locks on failure
            for lock in acquired:
                lock.release()
            raise
    
    def release_locks(self, locks: List[threading.Lock]):
        """Release locks"""
        for lock in reversed(locks):
            lock.release()

# Usage
executor = DeadlockFreeExecutor()
lock_a = executor.register_lock("A")
lock_b = executor.register_lock("B")

def task_1():
    """Task that needs both locks"""
    locks = executor.acquire_locks(["A", "B"])
    try:
        # Do work with both locks
        time.sleep(0.1)
    finally:
        executor.release_locks(locks)

def task_2():
    """Another task that needs both locks"""
    locks = executor.acquire_locks(["B", "A"])  # Different order, but handled safely
    try:
        # Do work with both locks
        time.sleep(0.1)
    finally:
        executor.release_locks(locks)

# No deadlock occurs due to consistent lock ordering
```

## Synchronization Primitives

Tools for coordinating parallel execution.

### Barriers

```python
import threading

class Barrier:
    """Synchronization barrier"""
    
    def __init__(self, num_parties: int):
        self.num_parties = num_parties
        self.count = 0
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
    
    def wait(self):
        """Wait for all parties to reach barrier"""
        with self.condition:
            self.count += 1
            
            if self.count == self.num_parties:
                # Last party arrived, release all
                self.count = 0
                self.condition.notify_all()
            else:
                # Wait for others
                self.condition.wait()

# Example: Synchronized workers
def parallel_workflow(worker_id: int, barrier: Barrier):
    """Worker that synchronizes at checkpoints"""
    print(f"Worker {worker_id}: Phase 1")
    time.sleep(worker_id * 0.1)  # Simulate work
    
    barrier.wait()  # Synchronize
    
    print(f"Worker {worker_id}: Phase 2")
    time.sleep(worker_id * 0.1)
    
    barrier.wait()  # Synchronize again
    
    print(f"Worker {worker_id}: Phase 3")

# Execute
barrier = Barrier(num_parties=4)
threads = [
    threading.Thread(target=parallel_workflow, args=(i, barrier))
    for i in range(4)
]

for t in threads:
    t.start()
for t in threads:
    t.join()
```

### Semaphores

```python
import threading
import time

class RateLimiter:
    """Rate limiter using semaphore"""
    
    def __init__(self, max_concurrent: int, time_window: float = 1.0):
        self.semaphore = threading.Semaphore(max_concurrent)
        self.time_window = time_window
        self.last_release_time = {}
        self.lock = threading.Lock()
    
    def acquire(self, token_id: str = None):
        """Acquire rate limit token"""
        self.semaphore.acquire()
        
        if token_id:
            with self.lock:
                self.last_release_time[token_id] = time.time()
    
    def release(self, token_id: str = None):
        """Release rate limit token"""
        if token_id:
            with self.lock:
                if token_id in self.last_release_time:
                    # Ensure minimum time between releases
                    elapsed = time.time() - self.last_release_time[token_id]
                    if elapsed < self.time_window:
                        time.sleep(self.time_window - elapsed)
        
        self.semaphore.release()

# Usage
limiter = RateLimiter(max_concurrent=3, time_window=1.0)

def rate_limited_task(task_id: int):
    """Task with rate limiting"""
    token = f"token_{task_id}"
    limiter.acquire(token)
    try:
        print(f"Task {task_id} executing")
        time.sleep(0.5)
    finally:
        limiter.release(token)

# Execute tasks (max 3 concurrent)
threads = [
    threading.Thread(target=rate_limited_task, args=(i,))
    for i in range(10)
]

for t in threads:
    t.start()
for t in threads:
    t.join()
```

## Load Balancing

Distributing work evenly across workers.

### Dynamic Load Balancer

```python
from queue import Queue
import threading
import time

class LoadBalancer:
    """Dynamic load balancer"""
    
    def __init__(self, num_workers: int = 4):
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.workers = []
        self.worker_stats = {}
        
        # Start workers
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker, args=(i,))
            worker.start()
            self.workers.append(worker)
            self.worker_stats[i] = {"processed": 0, "errors": 0}
    
    def _worker(self, worker_id: int):
        """Worker thread"""
        while True:
            task = self.task_queue.get()
            
            if task is None:  # Poison pill
                break
            
            try:
                task_id, func, args, kwargs = task
                result = func(*args, **kwargs)
                
                self.result_queue.put((task_id, {"success": True, "result": result}))
                self.worker_stats[worker_id]["processed"] += 1
            
            except Exception as e:
                self.result_queue.put((task_id, {"success": False, "error": str(e)}))
                self.worker_stats[worker_id]["errors"] += 1
            
            finally:
                self.task_queue.task_done()
    
    def submit(self, task_id: str, func: Callable, *args, **kwargs):
        """Submit task to load balancer"""
        self.task_queue.put((task_id, func, args, kwargs))
    
    def get_results(self, num_expected: int) -> dict:
        """Get all results"""
        results = {}
        for _ in range(num_expected):
            task_id, result = self.result_queue.get()
            results[task_id] = result
        return results
    
    def shutdown(self):
        """Shutdown workers"""
        # Send poison pills
        for _ in self.workers:
            self.task_queue.put(None)
        
        # Wait for workers
        for worker in self.workers:
            worker.join()
    
    def get_stats(self) -> dict:
        """Get worker statistics"""
        return self.worker_stats

# Usage
lb = LoadBalancer(num_workers=4)

# Submit tasks
for i in range(20):
    lb.submit(f"task_{i}", time.sleep, 0.1)

# Get results
results = lb.get_results(20)
print(f"Completed {len(results)} tasks")

# Print stats
print("Worker stats:", lb.get_stats())

lb.shutdown()
```

## Performance Considerations

### Measuring Parallel Performance

```python
import time
from typing import Callable, List
from concurrent.futures import ThreadPoolExecutor

class PerformanceBenchmark:
    """Benchmark parallel vs sequential execution"""
    
    def benchmark(
        self,
        task: Callable,
        num_iterations: int,
        parallel: bool = False,
        max_workers: int = 4
    ) -> dict:
        """Benchmark task execution"""
        
        if parallel:
            return self._benchmark_parallel(task, num_iterations, max_workers)
        else:
            return self._benchmark_sequential(task, num_iterations)
    
    def _benchmark_sequential(
        self,
        task: Callable,
        num_iterations: int
    ) -> dict:
        """Benchmark sequential execution"""
        start = time.time()
        
        for _ in range(num_iterations):
            task()
        
        duration = time.time() - start
        
        return {
            "mode": "sequential",
            "iterations": num_iterations,
            "duration": duration,
            "avg_per_iteration": duration / num_iterations
        }
    
    def _benchmark_parallel(
        self,
        task: Callable,
        num_iterations: int,
        max_workers: int
    ) -> dict:
        """Benchmark parallel execution"""
        start = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(task) for _ in range(num_iterations)]
            for future in futures:
                future.result()
        
        duration = time.time() - start
        
        return {
            "mode": "parallel",
            "iterations": num_iterations,
            "workers": max_workers,
            "duration": duration,
            "avg_per_iteration": duration / num_iterations,
            "speedup": None  # Calculated later
        }
    
    def compare(
        self,
        task: Callable,
        num_iterations: int,
        max_workers: int = 4
    ) -> dict:
        """Compare sequential vs parallel"""
        seq_result = self.benchmark(task, num_iterations, parallel=False)
        par_result = self.benchmark(task, num_iterations, parallel=True, max_workers=max_workers)
        
        speedup = seq_result["duration"] / par_result["duration"]
        par_result["speedup"] = speedup
        
        return {
            "sequential": seq_result,
            "parallel": par_result,
            "speedup": speedup
        }

# Usage
def io_bound_task():
    """Simulate I/O-bound task"""
    time.sleep(0.1)

benchmark = PerformanceBenchmark()
results = benchmark.compare(io_bound_task, num_iterations=20, max_workers=4)

print(f"Sequential: {results['sequential']['duration']:.2f}s")
print(f"Parallel: {results['parallel']['duration']:.2f}s")
print(f"Speedup: {results['speedup']:.2f}x")
```

## Real-World Scenarios

### Scenario 1: Parallel Research Pipeline

```python
class ParallelResearchPipeline:
    """Research pipeline with parallel execution"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
    
    def research(self, query: str) -> dict:
        """Conduct research with parallel operations"""
        
        # Stage 1: Parallel search across sources
        sources = ["academic", "web", "patents", "news"]
        
        def search_source(source: str):
            return {
                "source": source,
                "results": search(query, source)
            }
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            search_results = list(executor.map(search_source, sources))
        
        # Stage 2: Parallel analysis of top results
        top_results = self._select_top_results(search_results, top_n=10)
        
        def analyze_result(result):
            return {
                "id": result["id"],
                "summary": summarize(result["content"]),
                "key_points": extract_key_points(result["content"]),
                "relevance": score_relevance(result, query)
            }
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            analyses = list(executor.map(analyze_result, top_results))
        
        # Stage 3: Synthesize findings
        synthesis = self._synthesize_findings(analyses)
        
        return {
            "query": query,
            "sources_searched": sources,
            "results_analyzed": len(analyses),
            "synthesis": synthesis
        }
    
    def _select_top_results(self, search_results: List[dict], top_n: int) -> List[dict]:
        """Select top N results"""
        all_results = []
        for source_results in search_results:
            all_results.extend(source_results["results"])
        
        # Sort by relevance and take top N
        sorted_results = sorted(all_results, key=lambda x: x.get("score", 0), reverse=True)
        return sorted_results[:top_n]
    
    def _synthesize_findings(self, analyses: List[dict]) -> dict:
        """Synthesize analysis findings"""
        return {
            "summary": "Combined summary of all analyses",
            "key_insights": ["Insight 1", "Insight 2"],
            "sources_count": len(analyses)
        }

# Execute
pipeline = ParallelResearchPipeline(max_workers=8)
result = pipeline.research("quantum computing applications")
```

## Summary

Parallelization enables efficient agent orchestration:

**Key Concepts**:
- **Parallel Tool Calls**: Execute multiple tools simultaneously
- **Concurrent Reasoning**: Explore multiple reasoning paths
- **Fan-Out/Fan-In**: Distribute work and collect results
- **Async Execution**: Non-blocking operations
- **Coordination**: Managing parallel execution

**Concurrency Models**:
- **Threading**: For I/O-bound operations
- **Multiprocessing**: For CPU-bound operations
- **Async/Await**: For high-concurrency I/O

**Patterns**:
- **Scatter-Gather**: Distribute and collect
- **Map-Reduce**: Transform and aggregate
- **Pipeline**: Staged parallel processing
- **Work Queue**: Dynamic work distribution

**Challenges**:
- Race conditions and data races
- Deadlocks and livelocks
- Resource contention
- Coordination overhead

**Best Practices**:
1. Parallelize I/O-bound operations
2. Use processes for CPU-bound work
3. Minimize shared state
4. Use synchronization primitives correctly
5. Measure actual speedup
6. Handle errors in parallel contexts

Effective parallelization can dramatically improve agent performance, but requires careful design to manage complexity and avoid common pitfalls.

## Next Steps

Explore related orchestration topics:

- **[Workflow Patterns](workflow-patterns.md)**: Sequential and DAG patterns
- **[Background Processes](background-processes.md)**: Long-running async tasks
- **[State Management](state-management.md)**: Managing state in parallel contexts
- **[Control Flow](control-flow.md)**: Advanced control structures

Related topics:
- **[Tool Use](../tool-use/function-calling.md)**: Parallel tool execution
- **[Error Handling](../fundamentals/error-handling.md)**: Error handling in parallel systems
