"""Data models for the refund agent using Pydantic."""

from datetime import date, datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from pydantic import BaseModel, Field


class Item(BaseModel):
    """Represents a single item in an order."""
    name: str
    date_bought: date
    price: float


class Order(BaseModel):
    """Represents a user's order containing multiple items."""
    user_id: str
    items: List[Item] = Field(default_factory=list)


class Answer(BaseModel):
    """Agent response containing the answer and whether to continue the conversation."""
    response: str
    continue_conversation: bool = True


# Observability Models

@dataclass
class ObservabilityEvent:
    """Event for tracking agent operations."""
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = ""
    thread_id: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None


class ObservabilitySummary(BaseModel):
    """Summary of observability events for a conversation."""
    thread_id: str
    total_events: int
    total_duration_ms: float
    tool_calls: int
    llm_calls: int
    tools_used: List[str]
    errors: List[Dict[str, Any]]