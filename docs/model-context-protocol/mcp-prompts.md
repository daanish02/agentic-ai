# MCP Prompts and Templates

## Table of Contents

- [Introduction](#introduction)
- [What Are MCP Prompts?](#what-are-mcp-prompts)
- [Prompt Structure](#prompt-structure)
- [Prompt Discovery](#prompt-discovery)
- [Retrieving Prompts](#retrieving-prompts)
- [Parameterized Prompts](#parameterized-prompts)
- [Prompt Libraries](#prompt-libraries)
- [Prompt Composition](#prompt-composition)
- [Use Cases](#use-cases)
- [Best Practices](#best-practices)
- [Prompt Versioning](#prompt-versioning)
- [Dynamic Prompts](#dynamic-prompts)
- [Prompt Testing](#prompt-testing)
- [Sharing Prompts](#sharing-prompts)
- [Advanced Patterns](#advanced-patterns)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Prompts are MCP's third core primitive (alongside resources and tools). They enable **sharing reusable prompt templates** across the ecosystem. Instead of every agent reinventing prompts for common tasks, MCP prompts provide:

- **Standardized templates** for frequent operations
- **Best-practice patterns** shared across the community
- **Parameterized prompts** that adapt to specific contexts
- **Prompt composition** to build complex workflows

> "Prompts are the reusable blueprints for agent interactions."

Just as resources standardize data access and tools standardize actions, **prompts standardize high-quality agent instructions**.

## What Are MCP Prompts?

### Definition

An **MCP prompt** is:

- A reusable template for LLM interactions
- Exposed by MCP servers
- Discoverable by MCP clients
- Parameterized with arguments
- Rendered into final messages

### Why Prompts Matter

**Problem without MCP prompts**:

```python
# Every agent reimplements the same prompts
agent_a_summary_prompt = "Summarize this text..."
agent_b_summary_prompt = "Provide a summary of..."
agent_c_summary_prompt = "Create a brief summary..."

# All slightly different, all reinvented
```

**Solution with MCP prompts**:

```python
# One canonical prompt, shared by all
prompt = await client.get_prompt("summarize_text", {
    "text": document,
    "max_length": 200
})

# Everyone uses the same optimized template
```

### Prompts vs Resources vs Tools

Clear distinction:

| Primitive    | Purpose               | Example                         |
| ------------ | --------------------- | ------------------------------- |
| **Resource** | Access data           | Read file content               |
| **Tool**     | Execute action        | Send email                      |
| **Prompt**   | Generate instructions | "Analyze this data..." template |

Prompts are **metadata** - they don't fetch data or execute actions. They provide **instructions for the LLM**.

## Prompt Structure

### Basic Prompt

```json
{
  "name": "summarize_article",
  "description": "Generate a concise summary of an article",
  "arguments": [
    {
      "name": "article_url",
      "description": "URL of the article to summarize",
      "required": true
    },
    {
      "name": "max_sentences",
      "description": "Maximum number of sentences in summary",
      "required": false
    }
  ]
}
```

### Rendered Prompt

When retrieved with arguments, returns messages:

```json
{
  "messages": [
    {
      "role": "user",
      "content": {
        "type": "text",
        "text": "Please summarize the article at https://example.com/article.html\n\nProvide a concise summary in no more than 3 sentences. Focus on the main points and key takeaways."
      }
    }
  ],
  "description": "Summarize article prompt for https://example.com/article.html"
}
```

### Multi-Message Prompts

Prompts can include multiple messages:

```json
{
  "messages": [
    {
      "role": "system",
      "content": {
        "type": "text",
        "text": "You are an expert data analyst."
      }
    },
    {
      "role": "user",
      "content": {
        "type": "text",
        "text": "Analyze this dataset:\n\n{{dataset}}\n\nProvide insights on: {{focus_areas}}"
      }
    }
  ]
}
```

### Embedded Resources

Prompts can reference resources:

```json
{
  "messages": [
    {
      "role": "user",
      "content": {
        "type": "resource",
        "resource": {
          "uri": "file:///data/report.pdf",
          "text": "Review this report"
        }
      }
    },
    {
      "role": "user",
      "content": {
        "type": "text",
        "text": "What are the key findings?"
      }
    }
  ]
}
```

## Prompt Discovery

### Listing Prompts

```python
# Client requests prompt list
request = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "prompts/list",
    "params": {}
}

# Server responds
response = {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "prompts": [
            {
                "name": "summarize_text",
                "description": "Summarize any text input",
                "arguments": [
                    {
                        "name": "text",
                        "description": "Text to summarize",
                        "required": true
                    },
                    {
                        "name": "style",
                        "description": "Summary style: brief, detailed, bullet-points",
                        "required": false
                    }
                ]
            },
            {
                "name": "analyze_data",
                "description": "Analyze structured data",
                "arguments": [
                    {
                        "name": "data",
                        "description": "Data to analyze",
                        "required": true
                    },
                    {
                        "name": "focus",
                        "description": "What to focus on",
                        "required": false
                    }
                ]
            }
        ]
    }
}
```

### Client-Side Discovery

```python
class PromptDiscoveryClient:
    async def discover_prompts(self, server_id):
        """Discover all prompts from a server"""
        response = await self.mcp.send_request(
            server_id,
            "prompts/list",
            {}
        )

        prompts = response['prompts']

        # Index by name
        self.prompts[server_id] = {
            p['name']: p
            for p in prompts
        }

        return prompts

    async def search_prompts(self, keyword):
        """Find prompts matching keyword"""
        matching = []

        for server_id, prompts in self.prompts.items():
            for name, prompt in prompts.items():
                if (keyword.lower() in name.lower() or
                    keyword.lower() in prompt['description'].lower()):
                    matching.append({
                        'server_id': server_id,
                        'name': name,
                        'prompt': prompt
                    })

        return matching

    async def list_prompts_by_category(self):
        """Organize prompts by category"""
        categories = {
            'analysis': [],
            'writing': [],
            'coding': [],
            'other': []
        }

        for server_id, prompts in self.prompts.items():
            for name, prompt in prompts.items():
                # Categorize based on name/description
                if any(word in name for word in ['analyze', 'review', 'examine']):
                    categories['analysis'].append(prompt)
                elif any(word in name for word in ['write', 'compose', 'draft']):
                    categories['writing'].append(prompt)
                elif any(word in name for word in ['code', 'debug', 'refactor']):
                    categories['coding'].append(prompt)
                else:
                    categories['other'].append(prompt)

        return categories
```

## Retrieving Prompts

### Basic Retrieval

```python
# Client requests a prompt
request = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "prompts/get",
    "params": {
        "name": "summarize_text",
        "arguments": {
            "text": "Long article content here...",
            "style": "brief"
        }
    }
}

# Server responds with rendered prompt
response = {
    "jsonrpc": "2.0",
    "id": 2,
    "result": {
        "description": "Summarize text prompt",
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": "Please provide a brief summary of the following text:\n\n[Long article content here...]\n\nFocus on the key points and main ideas."
                }
            }
        ]
    }
}
```

### Client-Side Retrieval

```python
class PromptClient:
    async def get_prompt(self, server_id, prompt_name, arguments=None):
        """Retrieve and render a prompt"""
        request = {
            "jsonrpc": "2.0",
            "id": self.next_id(),
            "method": "prompts/get",
            "params": {
                "name": prompt_name,
                "arguments": arguments or {}
            }
        }

        response = await self.send_request(server_id, request)
        return response['result']

    async def use_prompt_with_llm(self, server_id, prompt_name, arguments, llm):
        """Get prompt and send directly to LLM"""
        # Get rendered prompt
        prompt = await self.get_prompt(server_id, prompt_name, arguments)

        # Send to LLM
        response = await llm.complete(
            messages=prompt['messages']
        )

        return response
```

### Server-Side Rendering

```python
class PromptServer(MCPServer):
    def __init__(self):
        super().__init__("prompts")
        self.prompt_templates = {}

    def register_prompt(self, name, description, template, arguments_schema):
        """Register a prompt template"""
        self.prompt_templates[name] = {
            'description': description,
            'template': template,
            'arguments': arguments_schema
        }

    async def get_prompt(self, name, arguments):
        """Render prompt with arguments"""
        if name not in self.prompt_templates:
            raise PromptNotFound(name)

        template_info = self.prompt_templates[name]

        # Validate arguments
        self.validate_arguments(arguments, template_info['arguments'])

        # Render template
        rendered = template_info['template'].render(**arguments)

        return {
            'description': template_info['description'],
            'messages': rendered
        }
```

## Parameterized Prompts

### String Interpolation

```python
class PromptTemplate:
    def __init__(self, template_string):
        self.template = template_string

    def render(self, **kwargs):
        """Render template with arguments"""
        # Simple string formatting
        text = self.template.format(**kwargs)

        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": text
                }
            }
        ]

# Usage
template = PromptTemplate(
    "Summarize this {doc_type}:\n\n{content}\n\n"
    "Provide a {style} summary in {max_length} words."
)

messages = template.render(
    doc_type="research paper",
    content="...",
    style="detailed",
    max_length=500
)
```

### Jinja2 Templates

```python
from jinja2 import Template

class Jinja2PromptTemplate:
    def __init__(self, template_string):
        self.template = Template(template_string)

    def render(self, **kwargs):
        """Render Jinja2 template"""
        text = self.template.render(**kwargs)

        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": text
                }
            }
        ]

# Complex template with conditionals
template_string = """
Analyze the following {{ data_type }}:

{{ data }}

{% if focus_areas %}
Focus on these aspects:
{% for area in focus_areas %}
- {{ area }}
{% endfor %}
{% endif %}

{% if comparison_data %}
Compare with:
{{ comparison_data }}
{% endif %}

Provide {{ detail_level }} analysis.
"""

template = Jinja2PromptTemplate(template_string)

messages = template.render(
    data_type="sales data",
    data="Q1 2024 sales...",
    focus_areas=["trends", "anomalies", "predictions"],
    detail_level="comprehensive"
)
```

### Multi-Message Templates

```python
class MultiMessagePromptTemplate:
    def __init__(self, messages_template):
        self.messages_template = messages_template

    def render(self, **kwargs):
        """Render each message in template"""
        rendered_messages = []

        for msg_template in self.messages_template:
            # Render content
            if isinstance(msg_template['content'], str):
                content = msg_template['content'].format(**kwargs)
            else:
                content = msg_template['content']

            rendered_messages.append({
                "role": msg_template['role'],
                "content": {
                    "type": "text",
                    "text": content
                }
            })

        return rendered_messages

# Usage
template = MultiMessagePromptTemplate([
    {
        "role": "system",
        "content": "You are a {expertise} expert."
    },
    {
        "role": "user",
        "content": "Please help me with:\n\n{task}"
    },
    {
        "role": "assistant",
        "content": "I'll be happy to help with {task}. Let me analyze it."
    },
    {
        "role": "user",
        "content": "Here's the data:\n\n{data}"
    }
])

messages = template.render(
    expertise="data analysis",
    task="interpreting this dataset",
    data="..."
)
```

## Prompt Libraries

### Building a Prompt Library

````python
class PromptLibraryServer(MCPServer):
    """Server that provides a library of prompts"""

    def __init__(self):
        super().__init__("prompt-library")
        self.register_prompts()

    def register_prompts(self):
        """Register all prompts in library"""

        # Analysis prompts
        self.register_prompt(
            "analyze_code",
            "Analyze code for quality and issues",
            self.code_analysis_template(),
            [
                {"name": "code", "required": True},
                {"name": "language", "required": True},
                {"name": "focus", "required": False}
            ]
        )

        self.register_prompt(
            "analyze_data",
            "Analyze structured data",
            self.data_analysis_template(),
            [
                {"name": "data", "required": True},
                {"name": "format", "required": False},
                {"name": "questions", "required": False}
            ]
        )

        # Writing prompts
        self.register_prompt(
            "write_documentation",
            "Generate documentation",
            self.documentation_template(),
            [
                {"name": "code", "required": True},
                {"name": "style", "required": False}
            ]
        )

        # Transformation prompts
        self.register_prompt(
            "translate_language",
            "Translate between languages",
            self.translation_template(),
            [
                {"name": "text", "required": True},
                {"name": "from_lang", "required": False},
                {"name": "to_lang", "required": True}
            ]
        )

        # And many more...

    def code_analysis_template(self):
        """Template for code analysis"""
        return Jinja2PromptTemplate("""
You are an expert code reviewer.

Analyze the following {{ language }} code:

```{{ language }}
{{ code }}
````

{% if focus %}
Focus on: {{ focus }}
{% else %}
Evaluate:

- Code quality
- Potential bugs
- Performance issues
- Security concerns
- Best practices
  {% endif %}

Provide detailed feedback with specific suggestions for improvement.
""")

    def data_analysis_template(self):
        """Template for data analysis"""
        return Jinja2PromptTemplate("""

Analyze the following data:

{{ data }}

{% if format %}
The data is in {{ format }} format.
{% endif %}

{% if questions %}
Answer these specific questions:
{% for q in questions %}
{{ loop.index }}. {{ q }}
{% endfor %}
{% else %}
Provide comprehensive analysis including:

- Summary statistics
- Patterns and trends
- Anomalies
- Key insights
- Recommendations
  {% endif %}
  """)

````

### Specialized Libraries

```python
# Research library
class ResearchPromptLibrary(MCPServer):
    """Prompts for research tasks"""

    def register_prompts(self):
        self.register_prompt("literature_review", ...)
        self.register_prompt("hypothesis_generation", ...)
        self.register_prompt("experiment_design", ...)
        self.register_prompt("results_interpretation", ...)

# Business library
class BusinessPromptLibrary(MCPServer):
    """Prompts for business tasks"""

    def register_prompts(self):
        self.register_prompt("swot_analysis", ...)
        self.register_prompt("business_plan", ...)
        self.register_prompt("market_research", ...)
        self.register_prompt("competitive_analysis", ...)

# Education library
class EducationPromptLibrary(MCPServer):
    """Prompts for educational tasks"""

    def register_prompts(self):
        self.register_prompt("explain_concept", ...)
        self.register_prompt("create_quiz", ...)
        self.register_prompt("lesson_plan", ...)
        self.register_prompt("provide_feedback", ...)
````

## Prompt Composition

### Sequential Composition

```python
async def multi_stage_analysis(client, data):
    """Use multiple prompts in sequence"""

    # Stage 1: Initial analysis
    stage1_prompt = await client.get_prompt(
        "initial_analysis",
        {"data": data}
    )
    stage1_result = await llm.complete(stage1_prompt['messages'])

    # Stage 2: Deep dive based on initial findings
    stage2_prompt = await client.get_prompt(
        "detailed_analysis",
        {
            "data": data,
            "initial_findings": stage1_result
        }
    )
    stage2_result = await llm.complete(stage2_prompt['messages'])

    # Stage 3: Generate recommendations
    stage3_prompt = await client.get_prompt(
        "generate_recommendations",
        {
            "analysis": stage2_result
        }
    )
    recommendations = await llm.complete(stage3_prompt['messages'])

    return recommendations
```

### Parallel Composition

```python
async def multi_perspective_analysis(client, document):
    """Analyze from multiple angles simultaneously"""

    # Get multiple specialized prompts
    prompts = await asyncio.gather(
        client.get_prompt("technical_analysis", {"doc": document}),
        client.get_prompt("business_analysis", {"doc": document}),
        client.get_prompt("legal_analysis", {"doc": document}),
        client.get_prompt("risk_analysis", {"doc": document})
    )

    # Execute all analyses in parallel
    results = await asyncio.gather(
        llm.complete(prompts[0]['messages']),
        llm.complete(prompts[1]['messages']),
        llm.complete(prompts[2]['messages']),
        llm.complete(prompts[3]['messages'])
    )

    technical, business, legal, risk = results

    # Combine perspectives
    synthesis_prompt = await client.get_prompt(
        "synthesize_analyses",
        {
            "technical": technical,
            "business": business,
            "legal": legal,
            "risk": risk
        }
    )

    final_analysis = await llm.complete(synthesis_prompt['messages'])
    return final_analysis
```

### Conditional Composition

```python
async def adaptive_prompting(client, content, content_type):
    """Choose prompts based on content type"""

    # Determine content type if not provided
    if not content_type:
        detect_prompt = await client.get_prompt(
            "detect_content_type",
            {"content": content}
        )
        content_type = await llm.complete(detect_prompt['messages'])

    # Select appropriate prompt based on type
    if content_type == "code":
        prompt = await client.get_prompt(
            "analyze_code",
            {"code": content}
        )
    elif content_type == "data":
        prompt = await client.get_prompt(
            "analyze_data",
            {"data": content}
        )
    elif content_type == "text":
        prompt = await client.get_prompt(
            "analyze_text",
            {"text": content}
        )
    else:
        prompt = await client.get_prompt(
            "general_analysis",
            {"content": content}
        )

    result = await llm.complete(prompt['messages'])
    return result
```

## Use Cases

### Code Review Automation

```python
class CodeReviewPrompts(MCPServer):
    def register_prompts(self):
        self.register_prompt(
            "review_pull_request",
            "Review a pull request",
            Jinja2PromptTemplate("""
Review this pull request:

**Title**: {{ title }}
**Author**: {{ author }}
**Changes**:
{{ changes }}

Evaluate:
1. Code quality and style
2. Potential bugs or issues
3. Performance implications
4. Security considerations
5. Test coverage
6. Documentation quality

Provide constructive feedback with specific suggestions.
            """),
            [
                {"name": "title", "required": True},
                {"name": "author", "required": True},
                {"name": "changes", "required": True}
            ]
        )

# Usage in CI/CD
async def automated_code_review(pr_data):
    prompt = await client.get_prompt(
        "review_pull_request",
        {
            "title": pr_data['title'],
            "author": pr_data['author'],
            "changes": pr_data['diff']
        }
    )

    review = await llm.complete(prompt['messages'])

    # Post review as comment
    await github.post_comment(pr_data['id'], review)
```

### Content Generation

```python
class ContentGenerationPrompts(MCPServer):
    def register_prompts(self):
        self.register_prompt(
            "generate_blog_post",
            "Generate blog post content",
            Jinja2PromptTemplate("""
Write a {{ tone }} blog post about: {{ topic }}

Target audience: {{ audience }}
Word count: approximately {{ word_count }} words

{% if key_points %}
Include these key points:
{% for point in key_points %}
- {{ point }}
{% endfor %}
{% endif %}

{% if seo_keywords %}
Optimize for these keywords: {{ seo_keywords | join(', ') }}
{% endif %}

Structure:
1. Engaging introduction
2. Main content with subheadings
3. Conclusion with call-to-action
            """),
            [
                {"name": "topic", "required": True},
                {"name": "audience", "required": True},
                {"name": "word_count", "required": False},
                {"name": "tone", "required": False},
                {"name": "key_points", "required": False},
                {"name": "seo_keywords", "required": False}
            ]
        )
```

### Data Analysis

```python
class DataAnalysisPrompts(MCPServer):
    def register_prompts(self):
        self.register_prompt(
            "exploratory_data_analysis",
            "Perform exploratory data analysis",
            Jinja2PromptTemplate("""
Perform exploratory data analysis on this dataset:

{{ dataset }}

Dataset description: {{ description }}

Analyze:
1. Data structure and types
2. Summary statistics
3. Missing values and data quality
4. Distributions and patterns
5. Correlations and relationships
6. Outliers and anomalies
7. Initial insights and hypotheses

{% if specific_questions %}
Also address these specific questions:
{% for question in specific_questions %}
- {{ question }}
{% endfor %}
{% endif %}

Provide visualizations recommendations and next steps for deeper analysis.
            """),
            [
                {"name": "dataset", "required": True},
                {"name": "description", "required": False},
                {"name": "specific_questions", "required": False}
            ]
        )
```

### Customer Support

```python
class SupportPrompts(MCPServer):
    def register_prompts(self):
        self.register_prompt(
            "handle_support_ticket",
            "Generate support response",
            Jinja2PromptTemplate("""
Customer support ticket:

**Customer**: {{ customer_name }}
**Issue**: {{ issue_description }}
**Priority**: {{ priority }}

{% if customer_history %}
**Previous interactions**:
{{ customer_history }}
{% endif %}

Generate a helpful, empathetic response that:
1. Acknowledges the issue
2. Provides clear solution steps
3. Offers additional help if needed
4. Maintains professional, friendly tone

{% if knowledge_base_articles %}
Reference these knowledge base articles:
{% for article in knowledge_base_articles %}
- {{ article }}
{% endfor %}
{% endif %}
            """),
            [
                {"name": "customer_name", "required": True},
                {"name": "issue_description", "required": True},
                {"name": "priority", "required": True},
                {"name": "customer_history", "required": False},
                {"name": "knowledge_base_articles", "required": False}
            ]
        )
```

## Best Practices

### Clear Descriptions

```python
# Bad: Vague description
self.register_prompt(
    "analyze",
    "Analyze something",
    ...
)

# Good: Clear, specific description
self.register_prompt(
    "analyze_python_code",
    "Perform comprehensive code review of Python code including style, "
    "bugs, performance, security, and best practices",
    ...
)
```

### Well-Documented Arguments

```python
# Bad: Minimal argument info
arguments = [
    {"name": "text", "required": True}
]

# Good: Detailed argument info
arguments = [
    {
        "name": "text",
        "description": "The text content to summarize. Can be any length, "
                      "but works best with 500-5000 words.",
        "required": True
    },
    {
        "name": "style",
        "description": "Summary style: 'brief' (2-3 sentences), "
                      "'detailed' (1-2 paragraphs), or 'bullets' (key points)",
        "required": False
    },
    {
        "name": "max_length",
        "description": "Maximum length in words. Overrides style defaults.",
        "required": False
    }
]
```

### Structured Output Guidance

```python
template = Jinja2PromptTemplate("""
Analyze the code and provide feedback in this exact structure:

## Overview
[Brief summary of the code's purpose and quality]

## Strengths
- [Positive aspect 1]
- [Positive aspect 2]
...

## Issues Found
### Critical
- [Critical issue 1]
...

### Major
- [Major issue 1]
...

### Minor
- [Minor issue 1]
...

## Recommendations
1. [Specific recommendation 1]
2. [Specific recommendation 2]
...

## Overall Rating
[Rating out of 10 with justification]
""")
```

### Versioning Prompts

```python
class VersionedPrompts(MCPServer):
    def register_prompts(self):
        # Version 1: Basic summarization
        self.register_prompt(
            "summarize_v1",
            "Basic text summarization",
            self.summarize_v1_template(),
            [{"name": "text", "required": True}]
        )

        # Version 2: Enhanced with style options
        self.register_prompt(
            "summarize_v2",
            "Enhanced text summarization with style control",
            self.summarize_v2_template(),
            [
                {"name": "text", "required": True},
                {"name": "style", "required": False}
            ]
        )

        # Latest: Point to current best version
        self.register_prompt(
            "summarize",
            "Text summarization (latest version)",
            self.summarize_v2_template(),  # Points to v2
            [
                {"name": "text", "required": True},
                {"name": "style", "required": False}
            ]
        )
```

## Prompt Versioning

### Semantic Versioning

```python
class SemanticVersioningPrompts(MCPServer):
    """Prompts with semantic versioning"""

    def register_prompt_versions(self, base_name, versions):
        """Register multiple versions of a prompt"""
        for version, (template, args) in versions.items():
            self.register_prompt(
                f"{base_name}_v{version}",
                f"{base_name} (version {version})",
                template,
                args
            )

        # Register latest
        latest_version = max(versions.keys())
        latest_template, latest_args = versions[latest_version]
        self.register_prompt(
            base_name,
            f"{base_name} (latest: v{latest_version})",
            latest_template,
            latest_args
        )

# Usage
server.register_prompt_versions(
    "analyze_code",
    {
        "1.0": (basic_template, basic_args),
        "2.0": (enhanced_template, enhanced_args),
        "2.1": (refined_template, refined_args)
    }
)
```

### Migration Helpers

```python
class MigrationPrompts(MCPServer):
    """Help clients migrate between prompt versions"""

    def register_prompt_with_migration(self, name, old_version, new_version):
        """Register prompt with migration path"""

        # Register old version (deprecated)
        self.register_prompt(
            f"{name}_v{old_version}",
            f"{name} v{old_version} (DEPRECATED - use v{new_version})",
            self.get_template(name, old_version),
            self.get_args(name, old_version)
        )

        # Register new version
        self.register_prompt(
            f"{name}_v{new_version}",
            f"{name} v{new_version}",
            self.get_template(name, new_version),
            self.get_args(name, new_version)
        )

        # Register migration prompt
        self.register_prompt(
            f"{name}_migration_guide",
            f"Guide for migrating from {name} v{old_version} to v{new_version}",
            self.migration_guide_template(name, old_version, new_version),
            []
        )
```

## Dynamic Prompts

### Context-Aware Prompts

```python
class ContextAwarePrompts(MCPServer):
    """Prompts that adapt to context"""

    async def get_prompt(self, name, arguments):
        """Generate prompt based on runtime context"""

        if name == "adaptive_analysis":
            # Determine data type
            data = arguments['data']
            data_type = self.detect_type(data)

            # Choose template based on type
            if data_type == "numeric":
                template = self.numeric_analysis_template()
            elif data_type == "textual":
                template = self.text_analysis_template()
            elif data_type == "categorical":
                template = self.categorical_analysis_template()
            else:
                template = self.general_analysis_template()

            # Render with context
            return {
                "description": f"Adaptive analysis for {data_type} data",
                "messages": template.render(**arguments)
            }
```

### User-Personalized Prompts

```python
class PersonalizedPrompts(MCPServer):
    """Prompts that adapt to user preferences"""

    def __init__(self):
        super().__init__("personalized")
        self.user_preferences = {}

    async def get_prompt(self, client_id, name, arguments):
        """Get personalized prompt for specific user"""

        # Load user preferences
        prefs = self.user_preferences.get(client_id, {})

        # Get base template
        base_template = self.templates[name]

        # Customize based on preferences
        customized = self.customize_template(
            base_template,
            prefs.get('tone', 'professional'),
            prefs.get('detail_level', 'moderate'),
            prefs.get('format', 'structured')
        )

        # Render with arguments
        return {
            "description": f"Personalized {name}",
            "messages": customized.render(**arguments)
        }
```

## Prompt Testing

### Validation Testing

```python
class PromptTester:
    """Test prompt templates"""

    def test_prompt_rendering(self, prompt_server, prompt_name, test_cases):
        """Test prompt renders correctly with various inputs"""
        results = []

        for test_case in test_cases:
            try:
                prompt = await prompt_server.get_prompt(
                    prompt_name,
                    test_case['arguments']
                )

                results.append({
                    "test_case": test_case['name'],
                    "status": "pass",
                    "output": prompt
                })

            except Exception as e:
                results.append({
                    "test_case": test_case['name'],
                    "status": "fail",
                    "error": str(e)
                })

        return results

# Usage
test_cases = [
    {
        "name": "basic_summary",
        "arguments": {
            "text": "Sample text...",
            "style": "brief"
        }
    },
    {
        "name": "missing_optional",
        "arguments": {
            "text": "Sample text..."
        }
    },
    {
        "name": "all_arguments",
        "arguments": {
            "text": "Sample text...",
            "style": "detailed",
            "max_length": 200
        }
    }
]

results = tester.test_prompt_rendering(
    prompt_server,
    "summarize_text",
    test_cases
)
```

### Quality Testing

```python
async def test_prompt_quality(prompt_server, prompt_name, llm, test_inputs):
    """Test prompt quality with real LLM"""

    results = []

    for test_input in test_inputs:
        # Get prompt
        prompt = await prompt_server.get_prompt(
            prompt_name,
            test_input['arguments']
        )

        # Run with LLM
        response = await llm.complete(prompt['messages'])

        # Evaluate response
        evaluation = {
            "input": test_input,
            "response": response,
            "metrics": {
                "length": len(response),
                "contains_required_sections": check_sections(response),
                "tone": analyze_tone(response),
                "clarity": rate_clarity(response)
            }
        }

        results.append(evaluation)

    return results
```

## Sharing Prompts

### Public Prompt Registries

```python
class PublicPromptRegistry(MCPServer):
    """Registry of community-contributed prompts"""

    def __init__(self):
        super().__init__("prompt-registry")
        self.load_prompts_from_registry()

    def load_prompts_from_registry(self):
        """Load prompts from community registry"""

        # Load from GitHub, S3, database, etc.
        prompt_definitions = self.fetch_registry()

        for prompt_def in prompt_definitions:
            self.register_prompt(
                name=prompt_def['name'],
                description=prompt_def['description'],
                template=self.parse_template(prompt_def['template']),
                arguments=prompt_def['arguments'],
                metadata={
                    'author': prompt_def['author'],
                    'version': prompt_def['version'],
                    'tags': prompt_def['tags'],
                    'rating': prompt_def['rating']
                }
            )
```

### Prompt Packages

```python
class PromptPackage:
    """Bundled collection of related prompts"""

    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.prompts = {}

    def add_prompt(self, name, template, arguments):
        """Add prompt to package"""
        self.prompts[name] = {
            'template': template,
            'arguments': arguments
        }

    def install(self, mcp_server):
        """Install all prompts in package to server"""
        for name, prompt_def in self.prompts.items():
            mcp_server.register_prompt(
                name=f"{self.name}::{name}",
                description=f"From {self.name} package",
                template=prompt_def['template'],
                arguments=prompt_def['arguments']
            )

# Usage
research_package = PromptPackage(
    "research-assistant",
    "Prompts for academic research"
)

research_package.add_prompt("literature_review", ...)
research_package.add_prompt("citation_analysis", ...)
research_package.add_prompt("hypothesis_generation", ...)

# Install to server
research_package.install(my_server)
```

## Advanced Patterns

### Chain-of-Thought Prompting

```python
cot_template = Jinja2PromptTemplate("""
{{ problem }}

Let's solve this step by step:

1. First, let's understand what we're being asked to do:
   [Your analysis of the problem]

2. What information do we have?
   [List the given information]

3. What approach should we take?
   [Outline your strategy]

4. Let's work through it:
   [Step-by-step solution]

5. Let's verify our answer:
   [Check your work]

Therefore, the answer is:
[Your final answer]
""")
```

### Few-Shot Prompting

```python
few_shot_template = Jinja2PromptTemplate("""
I'll show you some examples, then you do the same for a new input.

Example 1:
Input: {{ example_1_input }}
Output: {{ example_1_output }}

Example 2:
Input: {{ example_2_input }}
Output: {{ example_2_output }}

Example 3:
Input: {{ example_3_input }}
Output: {{ example_3_output }}

Now your turn:
Input: {{ new_input }}
Output:
""")
```

### Self-Consistency Prompting

```python
async def self_consistency_prompt(client, prompt_name, arguments, n=5):
    """Generate multiple responses and find consensus"""

    # Get prompt
    prompt = await client.get_prompt(prompt_name, arguments)

    # Generate multiple responses
    responses = await asyncio.gather(*[
        llm.complete(prompt['messages'])
        for _ in range(n)
    ])

    # Find most consistent answer
    consensus = find_consensus(responses)

    return consensus
```

## Summary

MCP prompts provide reusable, shareable prompt templates:

**Key Concepts**:

- **Prompts are templates** for LLM interactions
- **Parameterized** with arguments
- **Discoverable** like resources and tools
- **Composable** for complex workflows
- **Shareable** across the ecosystem

**Benefits**:

- **Consistency** - Same high-quality prompts everywhere
- **Best practices** - Learn from community expertise
- **Efficiency** - Don't reinvent prompts
- **Collaboration** - Share and improve together

**Use Cases**:

- Code review automation
- Content generation
- Data analysis
- Customer support
- Research assistance

**Best Practices**:

- Clear descriptions
- Well-documented arguments
- Structured output guidance
- Version management
- Quality testing

Prompts complete MCP's triad: **resources** (data), **tools** (actions), **prompts** (instructions).

## Next Steps

- **[Context Management](mcp-context.md)** - Use prompts with context effectively
- **[Building MCP Servers](building-mcp-servers.md)** - Implement prompt servers
- **[Tools and Tool Registration](mcp-tools.md)** - Combine prompts with tools
- **[Benefits and Ecosystem](mcp-ecosystem.md)** - Understand ecosystem impact
- **[MCP Fundamentals](mcp-fundamentals.md)** - Review core concepts
