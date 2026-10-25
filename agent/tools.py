"""Tool implementations for the refund agent."""

from datetime import date
from typing import Any

from langchain.tools import tool

from agent.data import MOCK_ORDERS_DICT


@tool
def find_user(user_id: str) -> dict[str, Any]:
    """
    Search for a user in MOCK_ORDERS_DICT by user_id.

    Args:
        user_id: The user ID to search for (e.g., "user_001")

    Returns:
        Dict with user data if found, or error dict if not found.
    """
    user_data = MOCK_ORDERS_DICT.get(user_id)
    if user_data:
        return user_data
    return {
        "status": "error",
        "message": f"User '{user_id}' not found",
        "user_id": user_id,
    }


@tool
def calculate_refund(user_data: dict[str, Any], items: list[str]) -> dict[str, Any]:
    """
    Calculate refund amount for specific items in a user's order.

    Args:
        user_data: The user dict returned by find_user
        items: List of item names to calculate refund for

    Returns:
        Dict with refund calculation details.
    """
    # Check if user_data has error status
    if user_data.get("status") == "error":
        return {
            "status": "error",
            "message": user_data.get("message", "Invalid user data"),
            "refund_amount": 0.0,
            "items_found": [],
            "items_not_found": items,
        }

    items_found = []
    items_not_found = []
    refund_amount = 0.0

    # Build a lookup dict for user's items
    user_items = {item["name"]: item["price"] for item in user_data.get("items", [])}

    for item_name in items:
        if item_name in user_items:
            items_found.append(item_name)
            refund_amount += user_items[item_name]
        else:
            items_not_found.append(item_name)

    message = f"Refund calculated for {len(items_found)} item(s)"
    if items_not_found:
        message += f", {len(items_not_found)} item(s) not found in order"

    return {
        "status": "success",
        "refund_amount": round(refund_amount, 2),
        "items_found": items_found,
        "items_not_found": items_not_found,
        "message": message,
    }


@tool
def get_current_date() -> dict[str, Any]:
    """
    Get the current date in ISO format.

    Returns:
        Dict with current date string.
    """
    today = date.today().isoformat()
    return {
        "status": "success",
        "current_date": today,
    }


@tool
def create_ticket(user_id: str, reason: str, refund_amount: float) -> dict[str, Any]:
    """
    Create a support ticket for a refund request.

    Args:
        user_id: The user ID
        reason: Reason for the refund request
        refund_amount: The refund amount to be processed

    Returns:
        Dict with ticket creation result.
    """
    import uuid

    # Generate a ticket ID
    ticket_id = f"TKT-{uuid.uuid4().hex[:6].upper()}"

    # In a real system, this would create a ticket in a ticketing system
    # For now, we just return a success response
    return {
        "status": "success",
        "ticket_id": ticket_id,
        "message": f"Ticket created successfully for user {user_id} with refund amount ${refund_amount:.2f}",
    }