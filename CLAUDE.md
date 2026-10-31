# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Customer-support refund agent built with Python, LangChain, and Pydantic. The agent processes refund requests by gathering information, verifying eligibility against order data and policies, and creating support tickets when needed.

## Architecture

```
refund_agent/
├── agent/
│   ├── __init__.py
│   ├── models.py          # Pydantic models: Item, Order, Answer, ObservabilityEvent, ObservabilitySummary
│   ├── data.py            # Mock order data as Pydantic objects (MOCK_ORDERS) and dict (MOCK_ORDERS_DICT)
│   ├── tools.py           # Tool implementations (find_user, calculate_refund, get_current_date, create_ticket)
│   ├── prompts.py         # System prompt and agent instructions
│   ├── agent.py           # Simple agent with manual tool calling loop and structured output
│   └── observability.py   # Callback handler for tracking all agent operations
├── main.py                # Entry point - runs interactive conversation loop
├── pyproject.toml         # Project dependencies
├── uv.lock                # Locked dependencies
├── .env.example           # Example environment variables
└── README.md              # Project documentation
```

## Development Commands

```bash
# Install dependencies
uv sync

# Run the agent
uv run main.py

# Run with custom model
GROQ_MODEL=openai/gpt-oss-120b uv run main.py

# Lint/type-check (if configured)
# (no lint config yet — add when needed)

# Run tests
# (no test framework yet — add when needed)
```

## Data Models

- **Item**: name (str), date_bought (date), price (float)
- **Order**: user_id (str), items (list[Item])
- **Answer**: response (str), continue_conversation (bool) - **This is the structured output format**
- **ObservabilityEvent**: timestamp, event_type, thread_id, details, duration_ms
- **ObservabilitySummary**: thread_id, total_events, total_duration_ms, tool_calls, llm_calls, tools_used, errors

## Mock Data

4 sample users (`user_001`–`user_004`) with orders spanning 2023-11 to 2024-03. Access via:

- `from agent.data import MOCK_ORDERS` — list of Order objects
- `from agent.data import MOCK_ORDERS_DICT` — dict keyed by user_id for tool lookup

## Tools (agent/tools.py)

All tools use `@tool` decorator from `langchain.tools` and return dicts with `status` field.

| Tool               | Parameters                                            | Returns                                                                                                           |
| ------------------ | ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `find_user`        | `user_id: str`                                        | User dict or `{"status": "error", "message": "...", "user_id": "..."}`                                            |
| `calculate_refund` | `user_data: dict`, `items: list[str]`                 | `{"status": "success", "refund_amount": float, "items_found": [...], "items_not_found": [...], "message": "..."}` |
| `get_current_date` | (none)                                                | `{"status": "success", "current_date": "YYYY-MM-DD"}`                                                             |
| `create_ticket`    | `user_id: str`, `reason: str`, `refund_amount: float` | `{"status": "success", "ticket_id": "TKT-XXXXXX", "message": "..."}`                                              |

Refund policy: items must be within 30 days of purchase (checked via `get_current_date`).

## Agent Architecture (agent/agent.py)

The agent uses a **simple manual tool-calling loop** instead of complex frameworks:

### Core Components

1. **LLM Setup** (`_create_llm`):
   - Creates base `ChatGroq` instance
   - Creates `llm_with_tools` for tool calling (bound with TOOL_SCHEMAS)
   - Creates `llm_structured` with `with_structured_output(Answer, method="json_schema")` to enforce Answer schema

2. **Conversation Loop** (`process_message`):
   - Adds user message to memory
   - Builds messages: `[SystemMessage(SYSTEM_PROMPT)] + memory`
   - Loops up to 5 iterations:
     - Calls `llm_with_tools.invoke(messages)` - may return tool calls
     - If tool calls: executes each tool via `_call_tool`, adds ToolMessage to memory, continues loop
     - If content but no tool calls: calls `llm_structured.invoke(messages)` to get validated Answer object
     - If neither: returns error Answer
   - Returns Answer with `continue_conversation` flag

3. **Conversation Control**:
   - `Answer.continue_conversation` = `True` → conversation continues
   - `Answer.continue_conversation` = `False` → conversation ends
   - Auto-detected from keywords: "goodbye", "thank you", "bye", "ticket created"

### Why This Design?

- **No complex frameworks**: Avoids langgraph create_agent issues, simpler debugging
- **Structured output**: Uses Groq's JSON Schema mode to enforce Answer schema
- **Manual tool loop**: Explicit control over tool calling, easier to understand
- **Memory**: Simple list of messages, persisted per thread_id
- **Observability**: Built-in callback handler tracks all operations

## Observability (agent/observability.py)

The `ObservabilityCallbackHandler` extends `BaseCallbackHandler` to track all agent operations:

### Tracked Events

| Event Type     | When               | Data                                  |
| -------------- | ------------------ | ------------------------------------- |
| `llm_start`    | Before LLM call    | model, prompt_length                  |
| `llm_end`      | After LLM call     | duration_ms, tokens_used, generations |
| `llm_error`    | LLM error          | error, error_type                     |
| `tool_start`   | Before tool call   | tool, input                           |
| `tool_end`     | After tool call    | tool, output, duration_ms             |
| `tool_error`   | Tool error         | error, error_type                     |
| `agent_action` | Agent selects tool | tool, tool_input, log                 |
| `agent_finish` | Agent finishes     | return_values, log                    |

### Summary Data

```python
summary = {
    "thread_id": "...",
    "total_events": 10,
    "total_duration_ms": 1500.5,
    "tool_calls": 3,
    "llm_calls": 4,
    "tools_used": ["find_user", "calculate_refund", "get_current_date"],
    "errors": [...]
}
```

### Usage in Agent

```python
# In __init__
self.observability = ObservabilityCallbackHandler(self.thread_id)

# In ChatGroq
callbacks=[self.observability]

# Get summary
summary = self.observability.get_summary()
```

## Dependencies

- `langchain>=0.2.0`
- `langchain-tools>=0.1.34`
- `langchain-core>=0.2.0`
- `langgraph>=0.1.0` (for InMemorySaver)
- `python-dotenv>=1.2.4`
- `langchain-groq>=1.1.3`
- `pydantic>=2.0`

Python >=3.14

## Environment Variables

```bash
GROQ_API_KEY=your_groq_api_key  # Required
GROQ_MODEL=openai/gpt-oss-120b  # Optional, defaults to gpt-oss-120b
```

## Key Implementation Details

### Tool Schemas (TOOL_SCHEMAS)

Defined as OpenAI function calling format in agent.py for the LLM to understand available tools.

### Memory Management

- Each thread_id gets its own memory list
- Memory contains: HumanMessage, AIMessage, ToolMessage
- Cleared via `clear_memory()` method

### Error Handling

- Tool errors caught and returned as error dicts
- LLM errors caught and returned as error Answers
- Max 5 iterations prevents infinite loops

## Extending the Agent

To add a new tool:

1. Add tool function to `agent/tools.py` with `@tool` decorator
2. Add to `TOOLS` dict in `agent/agent.py`
3. Add schema to `TOOL_SCHEMAS` in `agent/agent.py`
4. Update SYSTEM_PROMPT in `agent/prompts.py` if needed
