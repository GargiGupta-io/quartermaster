"""Company rules as plain code, so totals and approvals never depend on the AI.

Each rule names the policy section it enforces. Orders cover one product at a time.
"""

from pydantic import BaseModel

from procurement_agent.company_data import CompanyData

GST_RATE = 0.18  # POL-001 1.4
FX_BUFFER = 0.03  # POL-005 5.2
PROBATION_ORDER_CAP = 500_000  # POL-003 3.1
HARDWARE_CAPS = {"LAP-STD-14": 85_000, "LAP-ENG-16": 140_000}  # POL-004 4.1
ENGINEERING_ONLY = {"LAP-ENG-16"}  # POL-004 4.1

LINE_MANAGER = "Line Manager"
DEPARTMENT_HEAD = "Department Head"
FINANCE_CONTROLLER = "Finance Controller"
CFO = "CFO"
CIO = "CIO"


class UnknownRecord(ValueError):
    """A product, office, department or vendor offer that does not exist."""


class StockPlan(BaseModel):
    """How much comes from stock, a transfer or a purchase (POL-001 1.2)."""

    requested: int
    from_local_stock: int
    transfer_from: str | None
    transfer_quantity: int
    purchase_quantity: int


class OrderEvaluation(BaseModel):
    stock: StockPlan
    vendor_id: str
    currency: str
    unit_price: float
    fx_rate: float | None
    unit_price_inr: float
    subtotal_inr: float
    gst_inr: float
    total_inr: float
    budget_remaining_inr: float
    over_budget: bool
    hardware_exception: bool
    required_approvers: list[str]
    quotes_required: int
    qualifying_vendors: list[str]
    sole_source: bool
    lowest_eligible_vendor: str | None
    blocking_problems: list[str]
    justification_required: list[str]


def required_approvers(
    total_inr: float, foreign: bool, hardware_exception: bool, over_budget: bool
) -> list[str]:
    """POL-002 2.1 and 2.2, plus the CFO budget exception in POL-001 1.3."""
    if total_inr <= 50_000:
        approvers = [LINE_MANAGER]
    elif total_inr <= 500_000:
        approvers = [DEPARTMENT_HEAD]
    elif total_inr <= 2_500_000:
        approvers = [DEPARTMENT_HEAD, FINANCE_CONTROLLER]
    else:
        approvers = [DEPARTMENT_HEAD, FINANCE_CONTROLLER, CFO]
    extras = (
        (foreign, FINANCE_CONTROLLER),
        (over_budget, CFO),
        (hardware_exception, CIO),
    )
    for needed, role in extras:
        if needed and role not in approvers:
            approvers.append(role)
    return approvers


def quotes_required(total_inr: float) -> int:
    """POL-003 3.2."""
    if total_inr <= 100_000:
        return 1
    return 2 if total_inr <= 1_000_000 else 3


def plan_stock(data: CompanyData, sku: str, office: str, requested: int) -> StockPlan:
    """Use local stock first, then a transfer if one office can cover the whole shortfall."""
    _require_known(data, sku=sku, office=office)
    available = {s.office: s.available for s in data.inventory if s.sku == sku}
    from_local = min(available.get(office, 0), requested)
    shortfall = requested - from_local
    donors = [o for o, units in available.items() if o != office and units >= shortfall > 0]
    donor = max(donors, key=lambda o: available[o]) if donors else None
    return StockPlan(
        requested=requested,
        from_local_stock=from_local,
        transfer_from=donor,
        transfer_quantity=shortfall if donor else 0,
        purchase_quantity=0 if donor else shortfall,
    )


def offices(data: CompanyData) -> set[str]:
    return {office for vendor in data.vendors for office in vendor.serves}


def _require_known(
    data: CompanyData,
    sku: str | None = None,
    office: str | None = None,
    department: str | None = None,
) -> None:
    checks = (
        (sku, {p.sku for p in data.products}, "product"),
        (office, offices(data), "office"),
        (department, {b.department for b in data.budgets}, "department"),
    )
    for value, known, label in checks:
        if value is not None and value not in known:
            raise UnknownRecord(f"Unknown {label} '{value}'. Valid: {', '.join(sorted(known))}")


def evaluate_order(
    data: CompanyData,
    sku: str,
    vendor_id: str,
    requested_quantity: int,
    office: str,
    department: str,
    fx_rate: float | None = None,
    committed_inr: float = 0,
) -> OrderEvaluation:
    """Work out the cost, approvals and rule breaches of buying from one vendor."""
    _require_known(data, sku=sku, office=office, department=department)
    vendors = {v.id: v for v in data.vendors}
    offers = {e.vendor_id: e for e in data.catalog if e.sku == sku}
    if vendor_id not in offers:
        raise UnknownRecord(
            f"Vendor '{vendor_id}' does not sell {sku}. Sellers: {', '.join(sorted(offers))}"
        )
    vendor = vendors[vendor_id]
    offer = offers[vendor_id]
    foreign = vendor.currency != "INR"
    if foreign and not fx_rate:
        raise ValueError(f"{vendor_id} bills in {vendor.currency}, so an exchange rate is required")

    stock = plan_stock(data, sku, office, requested_quantity)
    unit_inr = offer.unit_price * fx_rate * (1 + FX_BUFFER) if foreign else offer.unit_price
    subtotal = stock.purchase_quantity * unit_inr
    gst = subtotal * GST_RATE
    total = subtotal + gst

    budget = next(b for b in data.budgets if b.department == department)
    remaining = budget.remaining - committed_inr
    over_budget = total > remaining
    hardware_exception = (
        sku == "LAP-PRO-16"
        or (sku in ENGINEERING_ONLY and department != "Engineering")
        or (sku in HARDWARE_CAPS and unit_inr > HARDWARE_CAPS[sku])
    )

    serving = [vendors[v] for v in offers if office in vendors[v].serves]
    qualifying = sorted(v.id for v in serving if v.status == "approved")
    inr_approved = [v for v in serving if v.status == "approved" and v.currency == "INR"]

    def is_eligible_inr(candidate) -> bool:
        if candidate.currency != "INR" or candidate.status == "suspended":
            return False
        cost = stock.purchase_quantity * offers[candidate.id].unit_price * (1 + GST_RATE)
        return candidate.status == "approved" or cost <= PROBATION_ORDER_CAP

    eligible = [v for v in serving if is_eligible_inr(v)]
    if not inr_approved:
        eligible += [v for v in serving if v.currency != "INR" and v.status == "approved"]
    lowest = min(eligible, key=lambda v: offers[v.id].unit_price, default=None)

    problems = []
    if office not in vendor.serves:
        problems.append(f"{vendor_id} does not deliver to {office}.")
    if vendor.status == "suspended":
        problems.append(f"{vendor_id} is suspended and may not receive orders (POL-003 3.1).")
    if vendor.status == "probation" and total > PROBATION_ORDER_CAP:
        problems.append(
            f"{vendor_id} is on probation and this order is over its "
            f"₹{PROBATION_ORDER_CAP:,} limit (POL-003 3.1)."
        )
    if foreign and inr_approved:
        problems.append(
            f"{vendor_id} bills in {vendor.currency} but an Approved INR vendor can supply "
            f"this item (POL-005 5.1)."
        )

    needed_quotes = quotes_required(total)
    sole_source = stock.purchase_quantity > 0 and len(qualifying) < needed_quotes
    justification = []
    if sole_source:
        justification.append("Sole-source purchase: record why (POL-003 3.2).")
    if lowest and lowest.id != vendor_id and not problems:
        justification.append(f"{lowest.id} is cheaper: record why it was not used (POL-003 3.3).")
    if hardware_exception:
        justification.append("Hardware exception: record the business reason (POL-004 4.2).")

    return OrderEvaluation(
        stock=stock,
        vendor_id=vendor_id,
        currency=vendor.currency,
        unit_price=offer.unit_price,
        fx_rate=fx_rate if foreign else None,
        unit_price_inr=round(unit_inr, 2),
        subtotal_inr=round(subtotal, 2),
        gst_inr=round(gst, 2),
        total_inr=round(total, 2),
        budget_remaining_inr=round(remaining, 2),
        over_budget=over_budget,
        hardware_exception=hardware_exception,
        required_approvers=required_approvers(total, foreign, hardware_exception, over_budget),
        quotes_required=needed_quotes,
        qualifying_vendors=qualifying,
        sole_source=sole_source,
        lowest_eligible_vendor=lowest.id if lowest else None,
        blocking_problems=problems,
        justification_required=justification,
    )
