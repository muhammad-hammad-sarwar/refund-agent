"""Simple refund agent with manual tool calling loop and structured output."""

import json
import logging
import os
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq

from agent.models import Answer, ObservabilityEvent
from agent.observability import ObservabilityCallbackHandler
from agent.prompts import SYSTEM_PROMPT
from agent.tools import calculate_refund, create_ticket, find_user, get_current_date

logger = logging.getLogger("refund_agent")

# Tool map for easy lookup
TOOLS: dict[str, BaseTool] = {
    "find_user": find_user,
    "calculate_refund": calculate_refund,
    "get_current_date": get_current_date,
    "create_ticket": create_ticket,
}

# Tool descriptions for the LLM
TOOL_SCHEMAS = [
    {
        "name": "find_user",
        "description": "Search for a user in the system by user_id. Use this first to look up customer order details.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "The user ID to search for (e.g., 'user_001')",
                }
            },
            "required": ["user_id"],
        },
    },
    {
        "name": "calculate_refund",
        "description": "Calculate refund amount for specific items in a user's order. Use after finding the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_data": {
                    "type": "object",
                    "description": "The user dict returned by find_user",
                },
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of item names to calculate refund for",
                },
            },
            "required": ["user_data", "items"],
        },
    },
    {
        "name": "get_current_date",
        "description": "Get the current date in ISO format. Use to check 30-day refund eligibility.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "create_ticket",
        "description": "Create a support ticket for a refund request. Use after calculating refund or for ineligible items.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "The user ID"},
                "reason": {
                    "type": "string",
                    "description": "Reason for the refund request",
                },
                "refund_amount": {
                    "type": "number",
                    "description": "The refund amount to be processed",
                },
            },
            "required": ["user_id", "reason", "refund_amount"],
        },
    },
]


class RefundAgent:
    """Simple refund agent with manual tool calling and conversation management."""

    def __init__(
        self,
        thread_id: str | None = None,
        model_name: str = "",
        temperature: float = 0.1,
        api_key: str | None = None,
    ):
        """
        Initialize the refund agent.

        Args:
            thread_id: Conversation thread ID for memory
            model_name: LLM model to use
            temperature: Temperature for LLM responses
            api_key: Groq API key
        """
        self.thread_id = thread_id or str(datetime.now().timestamp())
        self.model_name = os.getenv("GROQ_MODEL")
        self.temperature = temperature

        # Conversation memory
        self.memory: list[Any] = []

        # Setup observability
        self.observability = ObservabilityCallbackHandler(self.thread_id)

        # Setup LLM with structured output
        self.llm, self.llm_with_tools, self.llm_structured = self._create_llm(api_key)

        logger.info(
            f"[{self.thread_id}] Agent initialized | model={model_name} | thread_id={self.thread_id}"
        )

    def _create_llm(self, api_key: str | None):
        """Create the Groq LLM instance with structured output for Answer."""
        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            logger.warning(
                "No API key found for Groq. Set GROQ_API_KEY environment variable."
            )

        base_llm = ChatGroq(
            model=self.model_name,
            temperature=self.temperature,
            api_key=api_key,
            callbacks=[self.observability],
        )

        # Wrap with structured output to enforce Answer schema
        llm_structured = base_llm.with_structured_output(Answer, method="json_schema")

        # Bind tools to base LLM for tool calling
        llm_with_tools = base_llm.bind_tools(TOOL_SCHEMAS)

        return base_llm, llm_with_tools, llm_structured

    def _call_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool and return its result."""
        logger.info(f"[{self.thread_id}] TOOL CALL | tool={tool_name} | args={args}")
        tool = TOOLS.get(tool_name)
        if not tool:
            return {"status": "error", "message": f"Unknown tool: {tool_name}"}

        # Manually trigger observability for tool start
        from langchain_core.tools import Tool
        serialized = {"name": tool_name} if isinstance(tool, Tool) else {"name": tool_name}
        self.observability.on_tool_start(serialized, str(args))

        try:
            result = tool.invoke(args)
            logger.info(
                f"[{self.thread_id}] TOOL RESULT | tool={tool_name} | result={result}"
            )
            # Manually trigger observability for tool end
            self.observability.on_tool_end(str(result))
            return result
        except Exception as e:
            logger.error(
                f"[{self.thread_id}] TOOL ERROR | tool={tool_name} | error={e}"
            )
            # Manually trigger observability for tool error
            self.observability.on_tool_error(e)
            return {"status": "error", "message": str(e)}

    def process_message(self, user_input: str) -> Answer:
        """
        Process a user message and return an Answer with continue_conversation flag.
        """
        logger.info(f"[{self.thread_id}] PROCESS MESSAGE | input={user_input[:100]}")

        # Add user message to memory
        self.memory.append(HumanMessage(content=user_input))

        # Build messages for LLM
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + self.memory

        try:
            # Loop to handle multiple tool calls
            max_iterations = 5
            iteration = 0

            while iteration < max_iterations:
                iteration += 1

                # LLM call - may request tool calls (use llm_with_tools for tool calling)
                response = self.llm_with_tools.invoke(messages)
                self.memory.append(response)

                # Check if response has content (final answer) or tool calls
                has_content = response.content and response.content.strip()
                has_tool_calls = bool(response.tool_calls)

                logger.info(
                    f"[{self.thread_id}] LLM RESPONSE | iteration={iteration} | has_content={has_content} | has_tool_calls={has_tool_calls} | content={response.content[:100] if response.content else 'None'}"
                )

                # Handle tool calls if any
                if has_tool_calls:
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        tool_call_id = tool_call["id"]

                        # Execute tool
                        tool_result = self._call_tool(tool_name, tool_args)

                        # Add tool result to memory
                        self.memory.append(
                            ToolMessage(
                                content=json.dumps(tool_result),
                                tool_call_id=tool_call_id,
                            )
                        )

                    # Continue loop to get next LLM response with tool results
                    messages = [SystemMessage(content=SYSTEM_PROMPT)] + self.memory
                    continue

                # If we have content but no tool calls, this is the final response
                # Use structured output LLM to get properly formatted Answer
                if has_content:
                    # Re-invoke with structured output to get Answer object
                    structured_response = self.llm_structured.invoke(messages)
                    output = structured_response.response
                    answer = structured_response
                    break

                # If no content and no tool calls, something went wrong
                if not has_content and not has_tool_calls:
                    output = "I apologize, but I couldn't process your request. Please try again."
                    answer = Answer(response=output, continue_conversation=True)
                    break
            else:
                # Max iterations reached
                output = "I apologize, but the conversation took too many steps. Please try again."
                answer = Answer(response=output, continue_conversation=True)

            # Log observability
            summary = self.observability.get_summary()
            logger.info(f"[{self.thread_id}] OBSERVABILITY SUMMARY | {summary}")

            return answer

        except Exception as e:
            logger.error(f"[{self.thread_id}] PROCESS ERROR | {type(e).__name__}: {e}")
            return Answer(
                response=f"I apologize, but I encountered an error: {e!s}. Please try again.",
                continue_conversation=True,
            )

    def get_memory(self) -> list[Any]:
        """Get the current conversation memory."""
        return self.memory

    def get_observability_events(self) -> list[ObservabilityEvent]:
        """Get all observability events for this conversation."""
        return self.observability.get_events()

    def get_observability_summary(self) -> dict[str, Any]:
        """Get observability summary for this conversation."""
        return self.observability.get_summary()

    def clear_memory(self) -> None:
        """Clear conversation memory."""
        self.memory.clear()
        logger.info(f"[{self.thread_id}] Memory cleared")


def create_agent(
    thread_id: str,
    model_name: str = "",
    temperature: float = 0.1,
    api_key: str | None = None,
) -> RefundAgent:
    """Factory function to create a RefundAgent instance."""
    return RefundAgent(
        thread_id=thread_id,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
    )


def run_conversation_loop(agent: RefundAgent) -> None:
    """
    Run the interactive conversation loop in the terminal.
    """
    print(f"\n{'=' * 60}")
    print(f"Refund Agent Conversation (Thread: {agent.thread_id})")
    print(f"Model: {agent.model_name}")
    print("Type 'exit' or 'quit' to end the conversation")
    print(f"{'=' * 60}\n")

    # Initial greeting
    initial_response = agent.process_message(
        "Hello, I'd like to start a refund request."
    )
    print(f"Agent: {initial_response.response}\n")

    while initial_response.continue_conversation:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ("exit", "quit", "bye"):
                print(
                    "\nAgent: Thank you for contacting customer support. Have a great day!"
                )
                break

            if not user_input:
                continue

            response = agent.process_message(user_input)
            print(f"\nAgent: {response.response}\n")

            # Print observability info
            summary = agent.get_observability_summary()
            print(
                f"[Observability] Tools: {summary['tool_calls']} | LLM calls: {summary['llm_calls']} | Duration: {summary['total_duration_ms']:.0f}ms"
            )

            initial_response = response

        except KeyboardInterrupt:
            print("\n\nConversation ended by user.")
            break
        except EOFError:
            print("\n\nConversation ended.")
            break

    # Final observability report
    print(f"\n{'=' * 60}")
    print("CONVERSATION SUMMARY")
    print(f"{'=' * 60}")
    summary = agent.get_observability_summary()
    print(f"Thread ID: {summary['thread_id']}")
    print(f"Total Events: {summary['total_events']}")
    print(f"Tool Calls: {summary['tool_calls']}")
    print(f"LLM Calls: {summary['llm_calls']}")
    print(
        f"Tools Used: {', '.join(summary['tools_used']) if summary['tools_used'] else 'None'}"
    )
    print(f"Total Duration: {summary['total_duration_ms']:.0f}ms")
    if summary["errors"]:
        print(f"Errors: {len(summary['errors'])}")
    print(f"{'=' * 60}")
