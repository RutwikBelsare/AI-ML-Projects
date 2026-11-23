from backend.tools.order_tools import (
    get_order_status,
)


def test_existing_order():

    result = get_order_status(
        "ORD1001"
    )

    assert "ORD1001" in result
    assert "shipped" in result
    assert "DHL" in result


def test_order_id_is_case_insensitive():

    result = get_order_status(
        "ord1001"
    )

    assert "ORD1001" in result


def test_unknown_order():

    result = get_order_status(
        "ORD9999"
    )

    assert (
        result
        == "No order was found for order ID ORD9999."
    )