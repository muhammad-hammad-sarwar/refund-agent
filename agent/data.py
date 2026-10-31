"""Mock data for the refund agent - JSON/dict format for tool access."""

from datetime import date

from agent.models import Item, Order

MOCK_ORDERS: list[Order] = [
    Order(
        user_id="user_001",
        items=[
            Item(
                name="Wireless Headphones", date_bought=date(2026, 10, 10), price=99.99
            ),
            Item(name="Phone Case", date_bought=date(2026, 10, 10), price=24.99),
        ],
    ),
    Order(
        user_id="user_002",
        items=[
            Item(name="Laptop Stand", date_bought=date(2026, 10, 5), price=49.99),
            Item(name="USB-C Hub", date_bought=date(2026, 10, 5), price=39.99),
            Item(
                name="Mechanical Keyboard", date_bought=date(2026, 10, 5), price=129.99
            ),
        ],
    ),
    Order(
        user_id="user_003",
        items=[
            Item(name="Smart Watch", date_bought=date(2026, 9, 20), price=299.99),
        ],
    ),
    Order(
        user_id="user_004",
        items=[
            Item(name="Bluetooth Speaker", date_bought=date(2026, 10, 15), price=79.99),
            Item(name="Portable Charger", date_bought=date(2023, 11, 5), price=34.99),
        ],
    ),
]


# Export as dict for easy tool access
MOCK_ORDERS_DICT: dict[str, dict] = {
    order.user_id: {
        "user_id": order.user_id,
        "items": [
            {
                "name": item.name,
                "date_bought": item.date_bought.isoformat(),
                "price": item.price,
            }
            for item in order.items
        ],
    }
    for order in MOCK_ORDERS
}
