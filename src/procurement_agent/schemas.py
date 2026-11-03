"""The structured shapes that go in and out of the agent."""

from typing import Literal

from pydantic import BaseModel, Field

from procurement_agent.rules import OrderEvaluation, StockPlan

Status = Literal[
    "ready_for_approval", "no_purchase_needed", "needs_clarification", "refused", "failed"
]


class ParsedRequest(BaseModel):
    is_purchase_request: bool
    item: str | None = None
    sku: str | None = None
    quantity: int | None = Field(default=None, gt=0)
    office: str | None = None
    department: str | None = None
    budget_inr: float | None = None
    business_reason: str | None = None


class Reason(BaseModel):
    statement: str
    citations: list[str] = []


class Draft(BaseModel):
    """What the model decides. Figures are never taken from here."""

    decision: Literal["purchase", "no_purchase", "refuse"]
    vendor_id: str | None = None
    summary: str
    reasons: list[Reason] = []


class ProcurementResponse(BaseModel):
    status: Status
    summary: str
    request: ParsedRequest | None = None
    stock: StockPlan | None = None
    order: OrderEvaluation | None = None
    reasons: list[Reason] = []
    questions: list[str] = []
    tools_used: list[str] = []
    tokens_used: int = 0
