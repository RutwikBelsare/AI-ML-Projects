from typing import Literal

from pydantic import BaseModel


class OrderStatusRequest(BaseModel):
    order_id: str


class OrderStatus(BaseModel):
    order_id: str
    status: Literal[
        "processing",
        "shipped",
        "out_for_delivery",
        "delivered",
        "cancelled",
    ]
    carrier: str
    estimated_delivery: str


# ---------------------------------------------------------
# Mock Order Database
# ---------------------------------------------------------

ORDERS = {
    "ORD1001": {
        "order_id": "ORD1001",
        "status": "shipped",
        "carrier": "DHL",
        "estimated_delivery": "2026-09-29",
    },
    "ORD1002": {
        "order_id": "ORD1002",
        "status": "processing",
        "carrier": "FedEx",
        "estimated_delivery": "2026-10-01",
    },
    "ORD1003": {
        "order_id": "ORD1003",
        "status": "out_for_delivery",
        "carrier": "UPS",
        "estimated_delivery": "2026-09-27",
    },
    "ORD1004": {
        "order_id": "ORD1004",
        "status": "delivered",
        "carrier": "DHL",
        "estimated_delivery": "2026-09-25",
    },
}


# ---------------------------------------------------------
# Tool Implementation
# ---------------------------------------------------------

def get_order_status(
    order_id: str,
) -> str:

    order_id = order_id.strip().upper()

    order = ORDERS.get(order_id)

    if not order:
        return (
            f"No order was found for order ID {order_id}."
        )

    return (
        f"Order {order['order_id']} is currently "
        f"{order['status']}. "
        f"Carrier: {order['carrier']}. "
        f"Estimated delivery: "
        f"{order['estimated_delivery']}."
    )