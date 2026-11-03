"""The structured shapes that go in and out of the agent."""

from typing import Literal

from pydantic import BaseModel, Field

from procurement_agent.rules import OrderEvaluation, StockPlan

Status = Literal[
    "ready_for_approval",
    "needs_review",
    "no_purchase_needed",
    "needs_clarification",
    "refused",
    "failed",
    "rejected",
    "ordered",
]


class ParsedRequest(BaseModel):
    is_purchase_request: bool
    item: str | None = None
    sku: str | None = None
    candidate_skus: list[str] = []
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


class PolicyBasis(BaseModel):
    citation: str
    heading: str
    text: str


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class VerificationResult(BaseModel):
    passed: bool
    checks: list[CheckResult]


class Decision(BaseModel):
    """One person's decision on a request, kept as the audit trail."""

    role: str
    action: Literal["approve", "reject", "edit"]
    name: str = Field(min_length=1)
    note: str = ""
    justification: str = ""
    vendor_id: str | None = None
    requested_quantity: int | None = Field(default=None, gt=0)


class ProcurementResponse(BaseModel):
    status: Status
    summary: str
    request: ParsedRequest | None = None
    stock: StockPlan | None = None
    order: OrderEvaluation | None = None
    reasons: list[Reason] = []
    policy_basis: list[PolicyBasis] = []
    verification: VerificationResult | None = None
    warnings: list[str] = []
    questions: list[str] = []
    pending_approvers: list[str] = []
    decisions: list[Decision] = []
    approval_error: str | None = None
    purchase_order: dict | None = None
    tools_used: list[str] = []
    tokens_used: int = 0
