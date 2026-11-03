"""The actions the agent can take. A tool call never raises: it returns a ToolResult."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import httpx
from pydantic import BaseModel, Field, ValidationError

from procurement_agent.company_data import CompanyData, load_company_data
from procurement_agent.config import Settings
from procurement_agent.purchase_orders import KeyConflict, PurchaseOrderStore
from procurement_agent.rules import UnknownRecord, evaluate_order, plan_stock

ErrorCode = Literal[
    "invalid_input", "not_found", "unknown_tool", "unavailable", "blocked", "approval_missing"
]


class ToolError(Exception):
    def __init__(self, code: ErrorCode, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class ToolResult(BaseModel):
    ok: bool
    data: dict | None = None
    error_code: ErrorCode | None = None
    error: str | None = None
    retryable: bool = False


class CheckStockInput(BaseModel):
    sku: str
    office: str
    requested_quantity: int = Field(gt=0, le=10_000)


class ListOffersInput(BaseModel):
    sku: str
    office: str


class CheckBudgetInput(BaseModel):
    department: str


class ExchangeRateInput(BaseModel):
    currency: Literal["USD"]


class EvaluateOrderInput(BaseModel):
    sku: str
    vendor_id: str
    requested_quantity: int = Field(gt=0, le=10_000)
    office: str
    department: str
    fx_rate: float | None = Field(default=None, gt=0)


class CreatePurchaseOrderInput(EvaluateOrderInput):
    idempotency_key: str = Field(min_length=8)
    approved_by: list[str]
    justification: str = ""


_ORDER_IDENTITY = ("sku", "vendor_id", "requested_quantity", "office", "department")


@dataclass
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    run: Callable[[BaseModel], dict]
    changes_data: bool = False

    def schema(self) -> dict:
        """The tool's description in the format AI models expect."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }


@dataclass
class ToolBox:
    """All tools, wired to their data. Pass a client or store in tests to avoid real services."""

    settings: Settings = field(default_factory=Settings)
    data: CompanyData = field(default_factory=load_company_data)
    orders: PurchaseOrderStore = field(default_factory=PurchaseOrderStore)
    http: httpx.Client = field(default_factory=lambda: httpx.Client(timeout=10))

    def __post_init__(self):
        self.tools = {
            tool.name: tool
            for tool in (
                Tool(
                    "check_stock",
                    "How many units can come from local stock or a transfer between offices, "
                    "and how many must be purchased. Check this before looking at vendors.",
                    CheckStockInput,
                    self._check_stock,
                ),
                Tool(
                    "list_vendor_offers",
                    "All vendors selling a product who deliver to an office, with status, "
                    "currency, unit price excluding GST and lead time. Includes vendors who "
                    "may not be eligible: use evaluate_order to check.",
                    ListOffersInput,
                    self._list_offers,
                ),
                Tool(
                    "check_budget",
                    "A department's allocated, spent, committed and remaining budget in INR.",
                    CheckBudgetInput,
                    self._check_budget,
                ),
                Tool(
                    "get_exchange_rate",
                    "Today's market rate from a foreign currency to INR.",
                    ExchangeRateInput,
                    self._exchange_rate,
                ),
                Tool(
                    "evaluate_order",
                    "Work out the exact cost, required approvers, quotes and rule breaches of "
                    "buying from one vendor. Always use this for figures: never calculate them.",
                    EvaluateOrderInput,
                    self._evaluate,
                ),
                Tool(
                    "create_purchase_order",
                    "Create a purchase order. Refused unless every required approver has "
                    "approved. Repeating the same idempotency key returns the original order.",
                    CreatePurchaseOrderInput,
                    self._create_order,
                    changes_data=True,
                ),
            )
        }

    def run(self, name: str, arguments: dict) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(
                ok=False,
                error_code="unknown_tool",
                error=f"No tool named '{name}'. Available: {', '.join(self.tools)}",
            )
        try:
            return ToolResult(ok=True, data=tool.run(tool.input_model(**arguments)))
        except ValidationError as error:
            details = "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()
            )
            return ToolResult(ok=False, error_code="invalid_input", error=details)
        except UnknownRecord as error:
            return ToolResult(ok=False, error_code="not_found", error=str(error))
        except ToolError as error:
            return ToolResult(
                ok=False, error_code=error.code, error=error.message, retryable=error.retryable
            )

    def _check_stock(self, args: CheckStockInput) -> dict:
        return plan_stock(self.data, args.sku, args.office, args.requested_quantity).model_dump()

    def _list_offers(self, args: ListOffersInput) -> dict:
        plan_stock(self.data, args.sku, args.office, 1)  # refuses an unknown product or office
        vendors = {v.id: v for v in self.data.vendors}
        offers = [
            {
                "vendor_id": entry.vendor_id,
                "vendor_name": vendors[entry.vendor_id].name,
                "status": vendors[entry.vendor_id].status,
                "currency": vendors[entry.vendor_id].currency,
                "unit_price_excluding_gst": entry.unit_price,
                "lead_time_days": entry.lead_time_days,
            }
            for entry in self.data.catalog
            if entry.sku == args.sku and args.office in vendors[entry.vendor_id].serves
        ]
        return {"sku": args.sku, "office": args.office, "offers": offers}

    def _check_budget(self, args: CheckBudgetInput) -> dict:
        budget = next((b for b in self.data.budgets if b.department == args.department), None)
        if budget is None:
            known = ", ".join(sorted(b.department for b in self.data.budgets))
            raise UnknownRecord(f"Unknown department '{args.department}'. Valid: {known}")
        new_orders = self.orders.committed_inr(args.department)
        return {
            "department": budget.department,
            "allocated_inr": budget.allocated,
            "spent_inr": budget.spent,
            "committed_inr": budget.committed + new_orders,
            "remaining_inr": budget.remaining - new_orders,
        }

    def _exchange_rate(self, args: ExchangeRateInput) -> dict:
        try:
            response = self.http.get(
                f"{self.settings.fx_base_url}/latest",
                params={"base": args.currency, "symbols": "INR"},
            )
            response.raise_for_status()
            body = response.json()
            rate = float(body["rates"]["INR"])
            date = body["date"]
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            raise ToolError(
                "unavailable", f"Exchange rate service returned {status}.", retryable=status >= 500
            ) from error
        except httpx.HTTPError as error:
            raise ToolError(
                "unavailable", f"Exchange rate service unreachable: {error}", retryable=True
            ) from error
        except (KeyError, TypeError, ValueError) as error:
            raise ToolError("unavailable", "Exchange rate reply was malformed.") from error
        if rate <= 0:
            raise ToolError("unavailable", f"Exchange rate reply was not usable: {rate}.")
        return {"currency": args.currency, "to": "INR", "rate": rate, "date": date}

    def _evaluate_args(self, args: EvaluateOrderInput):
        try:
            return evaluate_order(
                self.data,
                args.sku,
                args.vendor_id,
                args.requested_quantity,
                args.office,
                args.department,
                fx_rate=args.fx_rate,
                committed_inr=self.orders.committed_inr(args.department),
            )
        except UnknownRecord:
            raise
        except ValueError as error:
            raise ToolError("invalid_input", str(error)) from error

    def _evaluate(self, args: EvaluateOrderInput) -> dict:
        return self._evaluate_args(args).model_dump()

    def _create_order(self, args: CreatePurchaseOrderInput) -> dict:
        """Recheck every rule here, whatever the agent believes it has already checked."""
        # A repeat must come first: the original order now counts against the budget,
        # so re-evaluating it could wrongly refuse a harmless retry.
        existing = self.orders.find(args.idempotency_key)
        if existing:
            if any(getattr(existing, f) != getattr(args, f) for f in _ORDER_IDENTITY):
                raise ToolError(
                    "blocked",
                    f"Key '{args.idempotency_key}' was already used for a different order.",
                )
            return existing.model_dump() | {"already_existed": True}

        order = self._evaluate_args(args)
        if order.stock.purchase_quantity == 0:
            raise ToolError("blocked", "Nothing to buy: stock or a transfer covers this request.")
        if order.blocking_problems:
            raise ToolError("blocked", " ".join(order.blocking_problems))
        if order.justification_required and not args.justification.strip():
            raise ToolError("blocked", " ".join(order.justification_required))
        missing = [r for r in order.required_approvers if r not in args.approved_by]
        if missing:
            raise ToolError("approval_missing", f"Still needs approval from: {', '.join(missing)}.")
        try:
            saved, existed = self.orders.create(
                {
                    "idempotency_key": args.idempotency_key,
                    "sku": args.sku,
                    "vendor_id": args.vendor_id,
                    "requested_quantity": args.requested_quantity,
                    "quantity": order.stock.purchase_quantity,
                    "office": args.office,
                    "department": args.department,
                    "total_inr": order.total_inr,
                    "approved_by": args.approved_by,
                    "justification": args.justification,
                }
            )
        except KeyConflict as error:
            raise ToolError("blocked", str(error)) from error
        return saved.model_dump() | {"already_existed": existed}
