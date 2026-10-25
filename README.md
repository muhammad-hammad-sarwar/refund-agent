# Refund Agent

A customer-support refund agent built with Python, LangChain, Groq, and Pydantic. The agent processes refund requests by gathering information, verifying eligibility against order data and policies, and creating support tickets when needed.

## Features

- **Structured Output**: Uses Groq's JSON Schema mode to enforce `Answer` response format
- **Tool Calling**: Manual tool-calling loop with 4 tools (find_user, calculate_refund, get_current_date, create_ticket)
- **Observability**: Built-in callback handler tracking all LLM calls, tool calls, and errors
- **Conversation Memory**: Per-thread conversation memory with `continue_conversation` flag
- **Refund Policy**: 30-day refund window from purchase date

## Quick Start

### Prerequisites

- Python 3.14+
- [uv](https://github.com/astral-sh/uv) package manager
- Groq API key

### Installation

```bash
# Clone the repository
cd refund_agent

# Install dependencies
uv sync

# Copy environment example and add your API key
cp .env.example .env
# Edit .env and add your GROK_API_KEY
```

### Running the Agent

```bash
# Run with default model
uv run main.py

# Run with custom model
GROK_MODEL=openai/gpt-oss-120b uv run main.py
```

### Example Conversation

```
============================================================
Refund Agent Conversation (Thread: abc-123)
Model: openai/gpt-oss-120b
Type 'exit' or 'quit' to end the conversation
============================================================

Agent: Hello! I'd be happy to help you with a refund request. To get started, could you please provide your user ID (e.g., user_001)?

You: user_001

Agent: Great! I found your account. You have the following items in your order:
- Wireless Headphones (purchased 2024-03-15, $99.99)
- Phone Case (purchased 2024-03-15, $24.99)

Which item(s) would you like to refund?

You: Wireless Headphones

Agent: I'll check the eligibility for Wireless Headphones. Let me get the current date and calculate the refund.

[Tool calls: get_current_date, calculate_refund]

Agent: The Wireless Headphones were purchased on 2024-03-15, which is within the 30-day refund window. The refund amount would be $99.99. Would you like me to create a support ticket to process this refund?

You: Yes, please create the ticket.

Agent: [Tool call: create_ticket]

Agent: I've created a support ticket for your refund request.

Ticket ID: TKT-A1B2C3
Refund Amount: $99.99
Items: Wireless Headphones

Your refund has been escalated to our support team. They will process it within 3-5 business days. Is there anything else I can help you with?

You: No, thank you. Goodbye!

Agent: You're welcome! Thank you for contacting customer support. Have a great day!

============================================================
CONVERSATION SUMMARY
============================================================
Thread ID: abc-123
Total Events: 12
Tool Calls: 4
LLM Calls: 5
Tools Used: find_user, get_current_date, calculate_refund, create_ticket
Total Duration: 3421ms
Errors: 0
============================================================
```

## Project Structure

```
refund_agent/
├── agent/
│   ├── __init__.py
│   ├── models.py          # Pydantic models
│   ├── data.py            # Mock order data
│   ├── tools.py           # Tool implementations
│   ├── prompts.py         # System prompt
│   ├── agent.py           # Main agent logic
│   └── observability.py   # Observability callback handler
├── main.py                # Entry point
├── pyproject.toml         # Dependencies
├── uv.lock                # Locked dependencies
├── .env.example           # Environment template
├── CLAUDE.md              # Development guidance
└── README.md              # This file
```

## Architecture

### Data Models (`agent/models.py`)

- **Item**: `name`, `date_bought`, `price`
- **Order**: `user_id`, `items` (list of Item)
- **Answer**: `response` (str), `continue_conversation` (bool) — **structured output**
- **ObservabilityEvent**: timestamp, event_type, thread_id, details, duration_ms
- **ObservabilitySummary**: Aggregated metrics for a conversation

### Tools (`agent/tools.py`)

| Tool | Description |
|------|-------------|
| `find_user` | Look up user by user_id |
| `calculate_refund` | Calculate refund for specific items |
| `get_current_date` | Get current date for eligibility check |
| `create_ticket` | Create support ticket for refund |

### Agent Loop (`agent/agent.py`)

```
User Input → Add to Memory → LLM with Tools
                                    ↓
                            Tool Calls? ──Yes──→ Execute Tools → Add Results → Loop
                                    ↓ No
                            Content? ──Yes──→ Structured LLM → Answer Object → Return
                                    ↓ No
                            Error → Return Error Answer
```

The loop runs up to 5 iterations to prevent infinite loops.

### Observability (`agent/observability.py`)

Extends `BaseCallbackHandler` to track:
- LLM start/end/error with token usage and duration
- Tool start/end/error with input/output and duration
- Agent actions and finishes
- Provides `get_summary()` for aggregated metrics

## Mock Data

4 sample users with orders:
- `user_001`: Wireless Headphones ($99.99), Phone Case ($24.99) — 2024-03-15
- `user_002`: Laptop Stand ($49.99), USB Cable ($12.99) — 2024-02-20
- `user_003`: Monitor ($299.99) — 2023-11-10
- `user_004`: Keyboard ($79.99), Mouse ($39.99) — 2024-01-05

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROK_API_KEY` | Yes | - | Groq API key |
| `GROK_MODEL` | No | `openai/gpt-oss-120b` | Model to use |

## Dependencies

See `pyproject.toml` for full list. Key dependencies:
- `langchain-groq>=1.1.3` — Groq LLM integration
- `langchain-core>=0.2.0` — Core LangChain abstractions
- `langchain-tools>=0.1.34` — Tool decorator
- `langgraph>=0.1.0` — InMemorySaver for checkpointer
- `pydantic>=2.0` — Data validation
- `python-dotenv>=1.2.4` — Environment loading

## Extending

### Adding a New Tool

1. Add function to `agent/tools.py` with `@tool` decorator
2. Add to `TOOLS` dict in `agent/agent.py`
3. Add schema to `TOOL_SCHEMAS` in `agent/agent.py`
4. Update `SYSTEM_PROMPT` in `agent/prompts.py`

### Adding Observability Events

The `ObservabilityCallbackHandler` automatically tracks all LangChain callbacks. Custom events can be added by calling `self.events.append(ObservabilityEvent(...))` directly.

## License

MIT