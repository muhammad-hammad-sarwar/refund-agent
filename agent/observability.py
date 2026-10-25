"""Observability module for the refund agent - tracks all agent operations."""

import uuid
import logging
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from agent.models import ObservabilityEvent

# Configure logging for observability
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("refund_agent.observability")


class ObservabilityCallbackHandler(BaseCallbackHandler):
    """Callback handler for observability - tracks all agent operations."""

    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self.events: List[ObservabilityEvent] = []
        self._start_times: Dict[str, datetime] = {}

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **_kwargs: Any) -> None:
        """Track LLM invocation start."""
        event_id = f"llm_{uuid.uuid4().hex[:8]}"
        self._start_times[event_id] = datetime.now()
        self.events.append(ObservabilityEvent(
            event_type="llm_start",
            thread_id=self.thread_id,
            details={
                "event_id": event_id,
                "model": serialized.get("name", "unknown"),
                "prompt_length": sum(len(p) for p in prompts),
            }
        ))
        logger.info(f"[{self.thread_id}] LLM START | model={serialized.get('name', 'unknown')} | prompt_chars={sum(len(p) for p in prompts)}")

    def on_llm_end(self, response: LLMResult, **_kwargs: Any) -> None:
        """Track LLM invocation end."""
        for event in reversed(self.events):
            if event.event_type == "llm_start" and event.details.get("event_id") in self._start_times:
                start_time = self._start_times.pop(event.details["event_id"])
                duration = (datetime.now() - start_time).total_seconds() * 1000
                event.duration_ms = duration
                token_usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
                self.events.append(ObservabilityEvent(
                    event_type="llm_end",
                    thread_id=self.thread_id,
                    details={
                        "event_id": event.details["event_id"],
                        "tokens_used": token_usage,
                        "generations": len(response.generations),
                    },
                    duration_ms=duration
                ))
                logger.info(f"[{self.thread_id}] LLM END | duration_ms={duration:.2f} | tokens={token_usage}")
                break

    def on_llm_error(self, error: BaseException, **_kwargs: Any) -> None:
        """Track LLM errors."""
        self.events.append(ObservabilityEvent(
            event_type="llm_error",
            thread_id=self.thread_id,
            details={"error": str(error), "error_type": type(error).__name__}
        ))
        logger.error(f"[{self.thread_id}] LLM ERROR | {type(error).__name__}: {error}")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **_kwargs: Any) -> None:
        """Track tool invocation start."""
        tool_name = serialized.get("name", "unknown")
        event_id = f"tool_{tool_name}_{uuid.uuid4().hex[:8]}"
        self._start_times[event_id] = datetime.now()
        self.events.append(ObservabilityEvent(
            event_type="tool_start",
            thread_id=self.thread_id,
            details={
                "event_id": event_id,
                "tool": tool_name,
                "input": input_str[:200] if input_str else "",
            }
        ))
        logger.info(f"[{self.thread_id}] TOOL START | tool={tool_name} | input={input_str[:100]}")

    def on_tool_end(self, output: str, **_kwargs: Any) -> None:
        """Track tool invocation end."""
        for event in reversed(self.events):
            if event.event_type == "tool_start" and event.details.get("event_id") in self._start_times:
                start_time = self._start_times.pop(event.details["event_id"])
                duration = (datetime.now() - start_time).total_seconds() * 1000
                self.events.append(ObservabilityEvent(
                    event_type="tool_end",
                    thread_id=self.thread_id,
                    details={
                        "event_id": event.details["event_id"],
                        "tool": event.details["tool"],
                        "output": output[:200] if output else "",
                    },
                    duration_ms=duration
                ))
                logger.info(f"[{self.thread_id}] TOOL END | tool={event.details['tool']} | duration_ms={duration:.2f} | output={output[:100]}")
                break

    def on_tool_error(self, error: BaseException, **_kwargs: Any) -> None:
        """Track tool errors."""
        self.events.append(ObservabilityEvent(
            event_type="tool_error",
            thread_id=self.thread_id,
            details={"error": str(error), "error_type": type(error).__name__}
        ))
        logger.error(f"[{self.thread_id}] TOOL ERROR | {type(error).__name__}: {error}")

    def on_agent_action(self, action: Any, **_kwargs: Any) -> None:
        """Track agent action (tool selection)."""
        self.events.append(ObservabilityEvent(
            event_type="agent_action",
            thread_id=self.thread_id,
            details={
                "tool": action.tool,
                "tool_input": action.tool_input,
                "log": action.log[:200] if action.log else "",
            }
        ))
        logger.info(f"[{self.thread_id}] AGENT ACTION | tool={action.tool} | input={action.tool_input}")

    def on_agent_finish(self, finish: Any, **_kwargs: Any) -> None:
        """Track agent finish."""
        self.events.append(ObservabilityEvent(
            event_type="agent_finish",
            thread_id=self.thread_id,
            details={
                "return_values": finish.return_values,
                "log": finish.log[:200] if finish.log else "",
            }
        ))
        logger.info(f"[{self.thread_id}] AGENT FINISH | return_values={finish.return_values}")

    def get_events(self) -> List[ObservabilityEvent]:
        """Get all recorded events."""
        return self.events

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all events."""
        total_duration = sum(e.duration_ms or 0 for e in self.events)
        tool_calls = [e for e in self.events if e.event_type == "tool_end"]
        llm_calls = [e for e in self.events if e.event_type == "llm_end"]

        return {
            "thread_id": self.thread_id,
            "total_events": len(self.events),
            "total_duration_ms": total_duration,
            "tool_calls": len(tool_calls),
            "llm_calls": len(llm_calls),
            "tools_used": list(set(e.details.get("tool") for e in tool_calls)),
            "errors": [e for e in self.events if e.event_type in ("llm_error", "tool_error")],
        }