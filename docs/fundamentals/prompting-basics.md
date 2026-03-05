# Prompting Basics

## Table of Contents

- [Introduction](#introduction)
- [What is a Prompt?](#what-is-a-prompt)
- [System Prompts vs User Prompts](#system-prompts-vs-user-prompts)
- [Role Definition](#role-definition)
- [Instruction Clarity](#instruction-clarity)
- [Prompt Structure](#prompt-structure)
- [Few-Shot Prompting](#few-shot-prompting)
- [Chain-of-Thought Prompting](#chain-of-thought-prompting)
- [Constraints and Guardrails](#constraints-and-guardrails)
- [Common Prompting Patterns](#common-prompting-patterns)
- [Prompt Engineering Principles](#prompt-engineering-principles)
- [Debugging Prompts](#debugging-prompts)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Prompts are how we communicate with language models. For agents, prompts are even more critical - they define not just what the agent should do, but **how it should think, when it should act, and how it should behave**. Effective prompting is the foundation of reliable agent systems.

Unlike traditional programming where you write explicit logic, agent programming involves **describing desired behavior in natural language**. This requires a different mindset: you're not commanding, you're guiding. You're not debugging code, you're refining instructions.

> "The prompt is the program."  
> -- Modern agent development wisdom

This guide covers the fundamentals of agent prompting: from basic structure to advanced techniques, from role definition to constraint setting, and from simple instructions to complex reasoning patterns.

## What is a Prompt?

A **prompt** is the input text provided to a language model that elicits a desired response or behavior. For agents, prompts serve multiple purposes:

### Purposes of Agent Prompts

**1. Define Identity**

```
You are a research assistant specialized in finding and analyzing academic papers.
```

**2. Specify Capabilities**

```
You have access to the following tools:
- search_papers(query, year_range)
- read_paper(paper_id)
- summarize(text, max_length)
```

**3. Set Behavioral Guidelines**

```
Always cite sources for factual claims.
If information is uncertain, say so explicitly.
Never fabricate references or data.
```

**4. Provide Context**

```
Previous conversation:
User: Find papers on chain-of-thought reasoning
Assistant: I'll search for recent papers...
Current task: Summarize the key findings
```

**5. Request Specific Actions**

```
Based on the papers found, create a summary of the three most important contributions to chain-of-thought research in 2025.
```

### Prompt Anatomy

```
┌─────────────────────────────────────┐
│ System Prompt (persistent)          │
│ - Role definition                   │
│ - Tool descriptions                 │
│ - General guidelines                │
├─────────────────────────────────────┤
│ Context (varies)                    │
│ - Conversation history              │
│ - Retrieved information             │
│ - Tool outputs                      │
├─────────────────────────────────────┤
│ User Input (current task)           │
│ - Current request                   │
│ - Specific instructions             │
└─────────────────────────────────────┘
```

## System Prompts vs User Prompts

Understanding the distinction between system and user prompts is crucial for agent design.

### System Prompts

**Characteristics**:

- Set once at agent initialization
- Persist across the entire conversation
- Define agent identity, capabilities, and behavior
- Typically not visible to end users

**Purpose**:

- Establish the agent's role and expertise
- Define available tools and how to use them
- Set constraints and safety guidelines
- Specify reasoning patterns

**Example**:

```
You are an autonomous coding assistant with access to file operations and code analysis tools.

Your capabilities:
- read_file(path): Read file contents
- write_file(path, content): Write or modify files
- run_tests(path): Execute test suites
- analyze_code(path): Get code quality metrics

Your behavior:
1. Always read existing code before making changes
2. Run tests after modifications
3. Explain your reasoning before acting
4. Ask for clarification when requirements are ambiguous
5. Never delete files without explicit confirmation

When editing code:
- Preserve existing style and patterns
- Add comments for non-obvious logic
- Consider edge cases
- Write tests for new functionality
```

### User Prompts

**Characteristics**:

- Change with each interaction
- Contain the specific task or question
- May include additional context
- Visible to end users (it's their input)

**Purpose**:

- Request specific actions
- Provide task-specific information
- Give feedback on agent actions
- Guide toward desired outcomes

**Example**:

```
Add input validation to the login function in auth.py.
It should check that username is alphanumeric and password
is at least 8 characters.
```

### The Interaction

```
System Prompt (persistent):
"You are a coding assistant..."

User: "Add input validation to login function"
Agent: [reads auth.py, reasons about validation, writes code]

User: "Also add rate limiting"
Agent: [builds on previous context, adds rate limiting]
        ^
        |
System prompt still active, guiding behavior
```

## Role Definition

Defining the agent's role is the first and most important part of the system prompt.

### Elements of Role Definition

**1. Identity**

```
You are a [role] with expertise in [domain].
```

Examples:

- "You are a data analysis agent specializing in exploratory data analysis and visualization."
- "You are a research assistant focused on finding and synthesizing academic literature."
- "You are a debugging assistant that helps identify and fix software bugs."

**2. Expertise Level**

```
You have [level] knowledge of [domain] and [level] knowledge of [domain].
```

Examples:

- "You have expert-level knowledge of Python and intermediate knowledge of JavaScript."
- "You understand machine learning concepts at a graduate level."

**3. Personality and Tone**

```
You communicate in a [style] manner, [characteristic].
```

Examples:

- "You communicate in a concise, technical manner, prioritizing accuracy."
- "You are helpful and patient, explaining concepts clearly for beginners."
- "You are thorough and detail-oriented, catching edge cases others might miss."

### Role Examples

**Research Assistant**:

```
You are an academic research assistant with expertise in computer science
and machine learning. You excel at finding relevant papers, extracting
key insights, and synthesizing information across multiple sources.

You approach research systematically:
1. Clarify the research question
2. Identify relevant sources
3. Extract and organize key findings
4. Synthesize insights
5. Cite sources rigorously

You communicate findings clearly, distinguishing between established facts
and emerging research. When evidence is limited or conflicting, you say so.
```

**Code Review Agent**:

```
You are a senior software engineer conducting code reviews. You have
deep expertise in software design, testing, security, and performance.

Your review process:
1. Understand the code's purpose and context
2. Check for correctness and edge cases
3. Evaluate design and architecture
4. Assess test coverage
5. Identify security vulnerabilities
6. Suggest performance improvements

You provide constructive feedback with specific examples and rationale.
You distinguish between critical issues (bugs, security) and suggestions
(style, refactoring). You recognize good patterns and practices.
```

**Data Analysis Agent**:

```
You are a data scientist specializing in exploratory data analysis.
You help users understand their data through statistical analysis
and visualization.

Your analytical approach:
1. Understand the data context and questions
2. Profile the data (types, distributions, missing values)
3. Explore relationships and patterns
4. Create informative visualizations
5. Draw evidence-based conclusions
6. Suggest further analyses

You explain statistical concepts clearly, avoiding jargon when possible.
You acknowledge limitations and don't overstate conclusions. You consider
data quality and potential biases.
```

## Instruction Clarity

Clear instructions are essential for predictable agent behavior.

### Be Specific

**Vague**:

```
Help me with the code.
```

**Specific**:

```
Review the authentication logic in auth.py and identify potential
security vulnerabilities. Focus on input validation, password handling,
and session management.
```

### Use Action Verbs

**Weak**:

```
Think about the best approach...
```

**Strong**:

```
Analyze the requirements, propose three implementation approaches,
and recommend the best one with justification.
```

### Define Success Criteria

**Without criteria**:

```
Make the function better.
```

**With criteria**:

```
Optimize the search function to:
1. Reduce time complexity from O(n²) to O(n log n) or better
2. Maintain existing functionality and test compatibility
3. Preserve the current API interface
```

### Provide Context

**Missing context**:

```
Fix the bug.
```

**With context**:

```
The user login is failing with a 500 error when the username contains
special characters. The error appears in auth.py around line 45.
Fix the input validation to properly handle special characters while
preventing SQL injection.
```

## Prompt Structure

Well-structured prompts improve agent performance and reliability.

### Template: Task-Oriented Structure

```
[CONTEXT]
What's the situation? What background is needed?

[GOAL]
What should be accomplished?

[CONSTRAINTS]
What are the limits and requirements?

[OUTPUT FORMAT]
How should results be presented?
```

**Example**:

```
CONTEXT:
We have a customer support chatbot that's receiving negative feedback
about response quality. Users report that answers are too vague and
don't address their specific issues.

GOAL:
Analyze the last 100 chat transcripts and identify the top 5 patterns
of unsatisfactory responses.

CONSTRAINTS:
- Focus on structural issues (vagueness, lack of specifics) not content
- Provide concrete examples for each pattern
- Suggest specific improvements for each pattern

OUTPUT FORMAT:
For each pattern:
1. Description of the problem
2. Example from transcripts
3. Improvement recommendation
```

### Template: Reasoning-Oriented Structure

```
[TASK]
Clear statement of what needs to be done

[APPROACH]
How should the agent think about it?

[STEPS]
What process should be followed?

[DECISION POINTS]
When should the agent pause and reconsider?
```

**Example**:

```
TASK:
Refactor the data processing pipeline to improve maintainability.

APPROACH:
Focus on separating concerns, reducing coupling, and improving testability.

STEPS:
1. Analyze the current pipeline architecture
2. Identify responsibilities and dependencies
3. Propose a modular structure
4. Create a refactoring plan that can be implemented incrementally
5. Highlight potential risks

DECISION POINTS:
- If current tests are insufficient, propose test improvements first
- If changes would break existing APIs, suggest a migration strategy
- If performance implications are unclear, recommend profiling first
```

## Few-Shot Prompting

Providing examples teaches the agent the desired behavior pattern.

### Structure

```
Here are examples of how to approach this task:

Example 1:
Input: [example input]
Process: [example reasoning]
Output: [example output]

Example 2:
Input: [example input]
Process: [example reasoning]
Output: [example output]

Now apply this pattern to:
[actual task]
```

### Example: Code Review Feedback

```
Here are examples of constructive code review feedback:

Example 1:
Code:
def calculate_average(numbers):
    return sum(numbers) / len(numbers)

Feedback:
**Issue**: Division by zero error when list is empty
**Severity**: High (will crash)
**Fix**:
def calculate_average(numbers):
    if not numbers:
        return 0  # or raise ValueError with clear message
    return sum(numbers) / len(numbers)

Example 2:
Code:
password = input("Enter password: ")
if password == "admin123":
    grant_access()

Feedback:
**Issue**: Hardcoded password is a security vulnerability
**Severity**: Critical
**Fix**: Store hashed passwords in secure configuration, use proper
authentication library like bcrypt
**Additional**: This also needs rate limiting to prevent brute force

Now review this code:
[actual code to review]
```

### When to Use Few-Shot

- Complex output formats
- Specific reasoning patterns
- Tone and style requirements
- Novel or unusual tasks
- When zero-shot performance is inadequate

## Chain-of-Thought Prompting

Encouraging explicit reasoning improves agent reliability.

### Basic Pattern

```
Let's approach this step by step:

1. First, [identify/understand/analyze]...
2. Then, [plan/consider/evaluate]...
3. Next, [implement/execute/apply]...
4. Finally, [verify/conclude/summarize]...
```

### Example: Debugging Prompt

```
Let's debug this systematically:

1. Understand the problem:
   - What is the expected behavior?
   - What is the actual behavior?
   - What are the error messages?

2. Identify potential causes:
   - What changed recently?
   - What are common causes of this error type?
   - What assumptions might be wrong?

3. Form hypotheses:
   - List 2-3 most likely causes
   - For each, what evidence would confirm it?

4. Test hypotheses:
   - Check evidence for each hypothesis
   - Narrow down to root cause

5. Implement fix:
   - Design fix that addresses root cause
   - Consider edge cases
   - Verify fix resolves issue

Now apply this process to: [error description]
```

### Explicit Reasoning Markers

```
Before taking action, always:

OBSERVE: What information do I have?
THINK: What does this tell me?
PLAN: What should I do next?
ACT: Execute the planned action
REFLECT: Did it work? What did I learn?
```

## Constraints and Guardrails

Constraints prevent unwanted behaviors and ensure safety.

### Types of Constraints

**1. Action Constraints**

```
Never:
- Delete files without explicit user confirmation
- Make API calls to production systems during testing
- Modify configuration files in /etc/ or /system/

Always:
- Create backups before destructive operations
- Validate input parameters before using tools
- Log all file modifications
```

**2. Information Constraints**

```
Do not:
- Share or log sensitive information (API keys, passwords)
- Include personal data in examples or logs
- Expose internal system details in error messages

Do:
- Redact sensitive data when displaying information
- Use placeholder values in examples
- Provide generic error messages to users
```

**3. Reasoning Constraints**

```
When uncertain:
- Say "I'm not certain, but..." rather than guessing
- Ask clarifying questions
- Request additional information

When you don't know:
- Admit it explicitly: "I don't have information about..."
- Suggest how to find the answer
- Don't fabricate information
```

**4. Output Constraints**

```
Output format requirements:
- Always provide reasoning before conclusions
- Structure responses with clear sections
- Use markdown formatting for code
- Keep explanations concise (< 3 paragraphs unless detail requested)
```

### Enforcement Patterns

**Checklist Pattern**:

```
Before modifying any file, verify:
[] The file path is correct
[] I have read the current contents
[] The modification is needed to complete the task
[] The change won't break existing functionality
[] I have a clear rollback plan

If any checkbox is unclear, ask for clarification first.
```

**Tiered Permissions**:

```
Actions are categorized by risk:

LOW RISK (proceed automatically):
- Reading files
- Searching documentation
- Analyzing code

MEDIUM RISK (explain then proceed):
- Writing new files
- Running tests
- Installing packages

HIGH RISK (require explicit approval):
- Modifying existing files
- Deleting anything
- Deploying changes
- Running in production

For MEDIUM/HIGH risk actions, explain what you'll do and why before acting.
```

## Common Prompting Patterns

### Pattern: Iterative Refinement

```
Your task will require multiple iterations. For each iteration:

1. Propose an approach
2. Implement it
3. Evaluate the results
4. Identify shortcomings
5. Propose improvements
6. Repeat until goals are met or 5 iterations reached

After each iteration, explicitly state:
- What worked
- What didn't work
- What to try next
```

### Pattern: Breadth-First Exploration

```
When exploring solutions:

1. First, generate 3-5 different approaches (breadth)
2. For each approach, list pros and cons
3. Select the most promising 1-2 approaches
4. Dive deep into implementation details (depth)

Don't commit to the first idea. Explore the solution space.
```

### Pattern: Defensive Programming

```
When writing code:

1. Consider edge cases:
   - Empty inputs
   - Null/undefined values
   - Extremely large or small values
   - Invalid types

2. Add input validation
3. Include error handling
4. Write defensive checks
5. Provide meaningful error messages

Example:
✗ result = data['key']
✓ result = data.get('key', default_value)
  if result is None:
      raise ValueError("Required field 'key' is missing")
```

### Pattern: Explain-Then-Execute

```
Before executing any significant action:

1. Explain what you plan to do
2. Explain why this approach makes sense
3. Highlight any risks or assumptions
4. Wait for confirmation (if HIGH RISK)
5. Execute the action
6. Verify the outcome

This pattern builds trust and prevents mistakes.
```

## Prompt Engineering Principles

### Principle 1: Be Clear and Specific

**Poor**:

```
You're an agent that helps with stuff.
```

**Good**:

```
You are a Python code analysis agent that identifies bugs,
suggests improvements, and explains code functionality.
```

### Principle 2: Provide Examples

**Poor**:

```
Format the output nicely.
```

**Good**:

```
Format output as JSON:
{
  "status": "success",
  "results": [...],
  "metadata": { "count": 10, "time": "2024-01-01" }
}
```

### Principle 3: Separate Concerns

**Poor** (everything mixed):

```
You're a helpful assistant that knows Python and can analyze code
and also you should never delete files and format output as JSON
with status and results fields and if you're uncertain say so...
```

**Good** (organized):

```
# Identity
You are a Python code analysis agent.

# Capabilities
- Static analysis
- Bug detection
- Performance profiling

# Constraints
- Never modify files
- Always validate inputs
- Request clarification when uncertain

# Output Format
JSON with fields: status, results, metadata
```

### Principle 4: Test and Iterate

Prompts are code. They need testing and iteration.

```
Version 1: "Analyze the code"
→ Too vague, inconsistent results

Version 2: "Check for bugs in the code"
→ Better, but misses other issues

Version 3: "Analyze code for: 1) Bugs 2) Performance 3) Style"
→ More comprehensive, structured

Version 4: Add examples of each analysis type
→ Consistent, high-quality output
```

### Principle 5: Make Implicit Explicit

**Implicit expectations**:

```
Review this code.
```

**Explicit expectations**:

```
Review this code for:
1. Correctness: Are there bugs or logic errors?
2. Performance: Any obvious inefficiencies?
3. Readability: Is it clear and maintainable?
4. Best practices: Does it follow Python conventions?

Provide specific examples and suggestions for each issue found.
```

## Debugging Prompts

When agents behave unexpectedly, debug the prompt.

### Common Prompt Problems

**Problem 1: Ambiguous Instructions**

Symptom: Inconsistent behavior across similar inputs

```
Bad: "Make it better"
Fix: "Reduce time complexity from O(n²) to O(n log n)"
```

**Problem 2: Conflicting Constraints**

Symptom: Agent gets stuck or behaves erratically

```
Conflict:
"Always explain reasoning" + "Keep responses under 50 words"

Fix: Prioritize constraints:
"Keep responses under 50 words. If detailed explanation is needed,
ask if the user wants more detail."
```

**Problem 3: Insufficient Context**

Symptom: Agent asks obvious questions or makes wrong assumptions

```
Missing context: "Fix the login bug"
Needed context: "Fix the login bug where special characters in
usernames cause SQL errors"
```

**Problem 4: Overly Complex Instructions**

Symptom: Agent ignores parts of the prompt

```
Too complex: [15 bullet points of requirements]

Simplified: Group related requirements:
- Input validation (5 rules)
- Error handling (3 rules)
- Output format (2 rules)
```

### Debugging Process

```
1. Identify the issue
   - What behavior is wrong?
   - Is it consistent or random?

2. Isolate the cause
   - Test with minimal prompt
   - Add complexity incrementally
   - Find where behavior changes

3. Hypothesis
   - What part of the prompt might cause this?
   - Is it ambiguity? Conflict? Missing info?

4. Test fix
   - Modify suspected prompt section
   - Test on multiple examples
   - Verify fix doesn't break other behaviors

5. Document
   - Note what went wrong
   - Note what fixed it
   - Update prompt guidelines
```

### Prompt Version Control

Treat prompts like code:

```python
# version: 1.2.1
# last modified: 2024-01-15
# changes: Added explicit file path validation requirement

SYSTEM_PROMPT = """
You are a file management agent...
[prompt content]
"""

# Changelog:
# 1.2.1 - Added file path validation (fixes issue #23)
# 1.2.0 - Restructured constraints section for clarity
# 1.1.0 - Added examples for each tool
# 1.0.0 - Initial version
```

## Summary

Effective prompting is the foundation of reliable agent systems:

1. **System vs User Prompts**: System prompts define persistent behavior, user prompts specify tasks
2. **Role Definition**: Clear identity, expertise, and communication style
3. **Instruction Clarity**: Specific, actionable, with success criteria
4. **Structure**: Organized sections (context, goal, constraints, format)
5. **Few-Shot Learning**: Examples teach desired patterns
6. **Chain-of-Thought**: Explicit reasoning improves reliability
7. **Constraints**: Guardrails prevent unwanted behaviors
8. **Common Patterns**: Iterative refinement, explain-then-execute, defensive programming
9. **Principles**: Clarity, specificity, organization, iteration
10. **Debugging**: Systematic approach to fixing prompt issues

Remember: **The prompt is the program**. Invest time in crafting clear, comprehensive prompts. Test them. Iterate on them. Version control them. Good prompts are the difference between unreliable agents and production-ready systems.

## Next Steps

Continue to [Simple Agent Loops](simple-agent-loops.md) to learn how to implement the basic observe-reason-act cycle that turns a language model into an autonomous agent.
