"""Tools use real sample data and a private database. The currency API is simulated."""

import json

import httpx
import pytest

from procurement_agent.config import Settings
from procurement_agent.purchase_orders import PurchaseOrderStore
from procurement_agent.tools import ToolBox

ORDER = {
    "sku": "LAP-STD-14",
    "vendor_id": "V-101",
    "requested_quantity": 20,
    "office": "Pune",
    "department": "Engineering",
    "idempotency_key": "request-0001",
    "approved_by": ["Department Head", "Finance Controller"],
}


def box_with(handler=None) -> ToolBox:
    handler = handler or (lambda request: httpx.Response(500))
    return ToolBox(
        settings=Settings(_env_file=None),
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )


@pytest.fixture
def box():
    return box_with()


def test_stock_tool_reports_the_plan(box):
    result = box.run(
        "check_stock", {"sku": "LAP-STD-14", "office": "Pune", "requested_quantity": 20}
    )
    assert result.ok
    assert result.data["from_local_stock"] == 4
    assert result.data["purchase_quantity"] == 16


def test_offers_only_list_vendors_who_deliver_there(box):
    result = box.run("list_vendor_offers", {"sku": "DOCK-USBC", "office": "Mumbai"})
    assert [o["vendor_id"] for o in result.data["offers"]] == ["V-103"]


def test_budget_tool(box):
    result = box.run("check_budget", {"department": "Operations"})
    assert result.data["remaining_inr"] == 240_000


@pytest.mark.parametrize(
    ("name", "arguments", "code"),
    [
        ("order_pizza", {}, "unknown_tool"),
        ("check_stock", {"sku": "LAP-STD-14", "office": "Pune"}, "invalid_input"),
        (
            "check_stock",
            {"sku": "LAP-STD-14", "office": "Pune", "requested_quantity": -1},
            "invalid_input",
        ),
        (
            "check_stock",
            {"sku": "GIFT-CARD", "office": "Pune", "requested_quantity": 1},
            "not_found",
        ),
        ("list_vendor_offers", {"sku": "MON-27", "office": "Delhi"}, "not_found"),
        ("check_budget", {"department": "Legal"}, "not_found"),
        ("get_exchange_rate", {"currency": "EUR"}, "invalid_input"),
    ],
)
def test_bad_calls_return_an_error_instead_of_raising(box, name, arguments, code):
    result = box.run(name, arguments)
    assert not result.ok
    assert result.error_code == code
    assert not result.retryable


def test_exchange_rate_success():
    box = box_with(
        lambda request: httpx.Response(200, json={"date": "2026-11-03", "rates": {"INR": 94.75}})
    )
    result = box.run("get_exchange_rate", {"currency": "USD"})
    assert result.data["rate"] == 94.75
    assert result.data["date"] == "2026-11-03"


def _timeout(request):
    raise httpx.ConnectTimeout("timed out")


@pytest.mark.parametrize(
    ("handler", "retryable"),
    [
        (_timeout, True),
        (lambda request: httpx.Response(503), True),
        (lambda request: httpx.Response(404), False),
        (lambda request: httpx.Response(200, text="not json"), False),
        (lambda request: httpx.Response(200, json={"rates": {}}), False),
        (lambda request: httpx.Response(200, json={"date": "x", "rates": {"INR": 0}}), False),
    ],
)
def test_exchange_rate_failures_are_reported(handler, retryable):
    result = box_with(handler).run("get_exchange_rate", {"currency": "USD"})
    assert not result.ok
    assert result.error_code == "unavailable"
    assert result.retryable is retryable


def test_purchase_order_created_with_figures_from_the_catalogue(box):
    result = box.run("create_purchase_order", ORDER)
    assert result.ok, result.error
    assert result.data["po_number"] == "PO-00001"
    assert result.data["quantity"] == 16
    assert result.data["total_inr"] == 1_359_360
    assert result.data["already_existed"] is False


def test_price_supplied_by_the_caller_is_ignored(box):
    result = box.run("create_purchase_order", ORDER | {"total_inr": 1, "unit_price": 1})
    assert result.data["total_inr"] == 1_359_360


def test_repeating_a_request_does_not_create_a_second_order(box):
    first = box.run("create_purchase_order", ORDER)
    second = box.run("create_purchase_order", ORDER)
    assert second.data["po_number"] == first.data["po_number"]
    assert second.data["already_existed"] is True
    assert box.run("check_budget", {"department": "Engineering"}).data["committed_inr"] == (
        950_000 + 1_359_360
    )


def test_reusing_a_key_for_a_different_order_is_refused(box):
    box.run("create_purchase_order", ORDER)
    result = box.run("create_purchase_order", ORDER | {"requested_quantity": 21})
    assert result.error_code == "blocked"
    assert "different order" in result.error


def test_order_needs_every_approver(box):
    result = box.run("create_purchase_order", ORDER | {"approved_by": ["Department Head"]})
    assert result.error_code == "approval_missing"
    assert "Finance Controller" in result.error


def test_order_without_approval_creates_nothing(box):
    box.run("create_purchase_order", ORDER | {"approved_by": []})
    assert box.orders.committed_inr("Engineering") == 0


@pytest.mark.parametrize("vendor", ["V-106", "V-104"])
def test_forbidden_vendor_is_refused_even_with_approvals(box, vendor):
    result = box.run("create_purchase_order", ORDER | {"vendor_id": vendor})
    assert result.error_code == "blocked"


def test_dearer_vendor_needs_a_reason(box):
    refused = box.run("create_purchase_order", ORDER | {"vendor_id": "V-103"})
    assert refused.error_code == "blocked"
    allowed = box.run(
        "create_purchase_order",
        ORDER | {"vendor_id": "V-103", "justification": "Needed within 6 days"},
    )
    assert allowed.ok


def test_nothing_to_buy_is_refused(box):
    result = box.run(
        "create_purchase_order",
        ORDER | {"office": "Mumbai", "requested_quantity": 10, "vendor_id": "V-101"},
    )
    assert result.error_code == "blocked"
    assert "Nothing to buy" in result.error


def test_new_orders_reduce_the_budget_for_the_next_one(box):
    box.run(
        "create_purchase_order",
        ORDER | {"department": "Operations", "approved_by": ORDER["approved_by"] + ["CFO"]},
    )
    assert box.run("check_budget", {"department": "Operations"}).data["remaining_inr"] < 0


def test_orders_survive_a_restart(tmp_path):
    path = tmp_path / "orders.db"
    first = ToolBox(settings=Settings(_env_file=None), orders=PurchaseOrderStore(path))
    created = first.run("create_purchase_order", ORDER)
    second = ToolBox(settings=Settings(_env_file=None), orders=PurchaseOrderStore(path))
    repeated = second.run("create_purchase_order", ORDER)
    assert repeated.data["po_number"] == created.data["po_number"]
    assert repeated.data["already_existed"] is True


def test_every_tool_describes_itself_for_the_ai(box):
    schemas = [tool.schema() for tool in box.tools.values()]
    assert len(schemas) == 6
    for schema in schemas:
        json.dumps(schema)
        assert schema["function"]["parameters"]["type"] == "object"
    assert [t.name for t in box.tools.values() if t.changes_data] == ["create_purchase_order"]
