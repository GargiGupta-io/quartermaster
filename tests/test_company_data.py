"""The sample data must be self-consistent and support the worked example."""

from pathlib import Path

import pytest

from procurement_agent.company_data import load_company_data

POLICY_DIR = Path(__file__).resolve().parents[1] / "data" / "policies"
GST = 0.18


@pytest.fixture(scope="module")
def data():
    return load_company_data()


def test_ids_are_unique(data):
    assert len({v.id for v in data.vendors}) == len(data.vendors)
    assert len({p.sku for p in data.products}) == len(data.products)
    assert len({(e.sku, e.vendor_id) for e in data.catalog}) == len(data.catalog)
    assert len({(s.office, s.sku) for s in data.inventory}) == len(data.inventory)


def test_references_resolve(data):
    vendor_ids = {v.id for v in data.vendors}
    skus = {p.sku for p in data.products}
    assert {e.vendor_id for e in data.catalog} <= vendor_ids
    assert {e.sku for e in data.catalog} <= skus
    assert {s.sku for s in data.inventory} <= skus


def test_stock_and_budgets_are_not_negative(data):
    assert all(s.available >= 0 for s in data.inventory)
    assert all(b.remaining >= 0 for b in data.budgets)


def test_every_policy_has_numbered_sections():
    policies = sorted(POLICY_DIR.glob("POL-*.md"))
    assert len(policies) == 6
    for path in policies:
        number = int(path.name[4:7])
        assert f"## {number}.1 " in path.read_text(encoding="utf-8")


def test_worked_example_20_laptops_for_pune(data):
    """Stock covers 4; the cheapest vendor allowed to take the rest is not the cheapest listed."""
    stock = next(s for s in data.inventory if (s.office, s.sku) == ("Pune", "LAP-STD-14"))
    shortfall = 20 - stock.available
    assert shortfall == 16

    vendors = {v.id: v for v in data.vendors}
    offers = [e for e in data.catalog if e.sku == "LAP-STD-14"]
    assert min(offers, key=lambda e: e.unit_price).vendor_id == "V-105"  # USD price, not comparable

    eligible = [
        e
        for e in offers
        if vendors[e.vendor_id].status == "approved"
        and vendors[e.vendor_id].currency == "INR"
        and "Pune" in vendors[e.vendor_id].serves
    ]
    assert len(eligible) == 3  # enough for the three quotes this order value requires
    best = min(eligible, key=lambda e: e.unit_price)
    assert best.vendor_id == "V-101"

    total = shortfall * best.unit_price * (1 + GST)
    assert total == pytest.approx(1_359_360)
    # The cheaper probation vendor is over its per-order cap at this size.
    assert shortfall * 69_500 * (1 + GST) > 500_000
    engineering = next(b for b in data.budgets if b.department == "Engineering")
    assert total < engineering.remaining
