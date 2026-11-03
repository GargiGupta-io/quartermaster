"""Every expected figure here was worked out by hand from the policy documents."""

import pytest

from procurement_agent.company_data import load_company_data
from procurement_agent.rules import (
    CFO,
    CIO,
    DEPARTMENT_HEAD,
    FINANCE_CONTROLLER,
    LINE_MANAGER,
    UnknownRecord,
    evaluate_order,
    plan_stock,
    quotes_required,
    required_approvers,
)


@pytest.fixture(scope="module")
def data():
    return load_company_data()


@pytest.mark.parametrize(
    ("total", "expected"),
    [
        (50_000, [LINE_MANAGER]),
        (50_001, [DEPARTMENT_HEAD]),
        (500_000, [DEPARTMENT_HEAD]),
        (500_001, [DEPARTMENT_HEAD, FINANCE_CONTROLLER]),
        (2_500_000, [DEPARTMENT_HEAD, FINANCE_CONTROLLER]),
        (2_500_001, [DEPARTMENT_HEAD, FINANCE_CONTROLLER, CFO]),
    ],
)
def test_approval_limits(total, expected):
    assert required_approvers(total, False, False, False) == expected


@pytest.mark.parametrize(
    ("total", "expected"),
    [(100_000, 1), (100_001, 2), (1_000_000, 2), (1_000_001, 3)],
)
def test_quote_limits(total, expected):
    assert quotes_required(total) == expected


def test_flagship_order(data):
    order = evaluate_order(data, "LAP-STD-14", "V-101", 20, "Pune", "Engineering")
    assert order.stock.from_local_stock == 4
    assert order.stock.purchase_quantity == 16
    assert order.subtotal_inr == 1_152_000
    assert order.gst_inr == 207_360
    assert order.total_inr == 1_359_360
    assert order.required_approvers == [DEPARTMENT_HEAD, FINANCE_CONTROLLER]
    assert order.quotes_required == 3
    assert not order.sole_source
    assert order.lowest_eligible_vendor == "V-101"
    assert order.blocking_problems == []
    assert order.justification_required == []


@pytest.mark.parametrize(
    ("vendor", "phrase"),
    [("V-106", "suspended"), ("V-104", "probation"), ("V-105", "POL-005")],
)
def test_tempting_vendors_are_blocked(data, vendor, phrase):
    order = evaluate_order(data, "LAP-STD-14", vendor, 20, "Pune", "Engineering", fx_rate=94.75)
    assert any(phrase in problem for problem in order.blocking_problems)
    assert order.lowest_eligible_vendor == "V-101"


def test_dearer_vendor_needs_a_recorded_reason(data):
    order = evaluate_order(data, "LAP-STD-14", "V-103", 20, "Pune", "Engineering")
    assert order.blocking_problems == []
    assert any("V-101 is cheaper" in reason for reason in order.justification_required)


def test_probation_vendor_wins_a_small_order(data):
    order = evaluate_order(data, "MON-27", "V-104", 13, "Pune", "Engineering")
    assert order.stock.from_local_stock == 12
    assert order.total_inr == 21_122
    assert order.lowest_eligible_vendor == "V-104"
    assert order.blocking_problems == []


def test_whole_shortfall_transfers_when_one_office_can_cover_it(data):
    plan = plan_stock(data, "LAP-STD-14", "Mumbai", 10)
    assert (plan.transfer_from, plan.transfer_quantity, plan.purchase_quantity) == (
        "Bengaluru",
        10,
        0,
    )


def test_no_partial_transfer(data):
    plan = plan_stock(data, "LAP-STD-14", "Mumbai", 13)
    assert plan.transfer_from is None
    assert plan.purchase_quantity == 13


def test_local_stock_alone_needs_no_purchase(data):
    plan = plan_stock(data, "LAP-STD-14", "Pune", 3)
    assert (plan.from_local_stock, plan.transfer_from, plan.purchase_quantity) == (3, None, 0)


def test_over_budget_adds_the_cfo(data):
    order = evaluate_order(data, "MON-27", "V-103", 20, "Mumbai", "Operations")
    assert order.total_inr == 443_680
    assert order.budget_remaining_inr == 240_000
    assert order.over_budget
    assert order.required_approvers == [DEPARTMENT_HEAD, CFO]


def test_foreign_only_item_uses_buffered_rate_and_is_sole_source(data):
    order = evaluate_order(data, "NET-SW48", "V-105", 2, "Pune", "Engineering", fx_rate=94.75)
    assert order.unit_price_inr == pytest.approx(209_823.88, abs=0.01)
    assert order.total_inr == pytest.approx(495_184.34, abs=0.01)
    assert order.blocking_problems == []
    assert order.sole_source
    assert order.required_approvers == [DEPARTMENT_HEAD, FINANCE_CONTROLLER]


def test_foreign_vendor_needs_an_exchange_rate(data):
    with pytest.raises(ValueError, match="exchange rate"):
        evaluate_order(data, "NET-SW48", "V-105", 2, "Pune", "Engineering")


def test_pro_laptop_always_needs_the_cio(data):
    order = evaluate_order(data, "LAP-PRO-16", "V-101", 1, "Bengaluru", "Sales")
    assert order.total_inr == 194_700
    assert order.hardware_exception
    assert order.required_approvers == [DEPARTMENT_HEAD, CIO]


def test_engineering_laptop_outside_engineering_needs_the_cio(data):
    allowed = evaluate_order(data, "LAP-ENG-16", "V-101", 2, "Bengaluru", "Engineering")
    refused = evaluate_order(data, "LAP-ENG-16", "V-101", 2, "Bengaluru", "Sales")
    assert not allowed.hardware_exception
    assert refused.hardware_exception


def test_committed_orders_reduce_the_budget(data):
    order = evaluate_order(
        data, "MON-27", "V-103", 13, "Pune", "Sales", committed_inr=1_000_000
    )
    assert order.budget_remaining_inr == 0
    assert order.over_budget


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sku": "GIFT-CARD"},
        {"office": "Delhi"},
        {"department": "Legal"},
        {"vendor_id": "V-999"},
    ],
)
def test_unknown_records_are_refused(data, kwargs):
    base = {
        "sku": "MON-27",
        "vendor_id": "V-103",
        "requested_quantity": 1,
        "office": "Pune",
        "department": "Sales",
    }
    with pytest.raises(UnknownRecord):
        evaluate_order(data, **(base | kwargs))
