ORDER_STATUS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_order_status",
        "description": (
            "Get the current status of a customer order. "
            "Use this only when the customer provides an "
            "order ID and asks about order status or delivery."
        ),
        "parameters": {
            "type": "object",
            "required": [
                "order_id",
            ],
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": (
                        "The customer's order ID, "
                        "for example ORD1001."
                    ),
                },
            },
        },
    },
}


AVAILABLE_TOOLS = [
    ORDER_STATUS_TOOL,
]