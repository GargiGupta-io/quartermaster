"""Check the model's words against the facts before anyone relies on them."""

import json
import re

from pydantic import BaseModel, ValidationError

from procurement_agent import prompts
from procurement_agent.company_data import CompanyData
from procurement_agent.knowledge import Passage, load_passages
from procurement_agent.llm import LLMClient, LLMError
from procurement_agent.rules import OrderEvaluation, StockPlan
from procurement_agent.schemas import PolicyBasis, Reason, Source

_CITATION = re.compile(
    r"\b[A-Z]{3}-[A-Z0-9]+\b(?!-)(?: (?:section|page|row|paragraph|message|record) [A-Z0-9.-]+"
    r"| \d+\.\d+)?"
)
_PRODUCT_CODE = re.compile(r"\b[A-Z]{2,}-[A-Z0-9-]+\b")
_VENDOR_ID = re.compile(r"\bV-\d+\b")
_NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(lakh|crore)?", re.IGNORECASE)
_ROUNDED_TOLERANCE = 0.01  # "13.6 lakh" for 13,59,360


class Check(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class Verification(BaseModel):
    passed: bool
    checks: list[Check]

    @property
    def problems(self) -> str:
        return " ".join(c.detail for c in self.checks if not c.passed)


def numbers_in(text: str) -> list[tuple[float, bool]]:
    """Every figure in the text, with whether it was written in rounded lakh or crore form."""
    text = _CITATION.sub(" ", text)
    text = _PRODUCT_CODE.sub(" ", text)
    found = []
    for match in _NUMBER.finditer(text):
        value = float(match.group(1).replace(",", ""))
        unit = (match.group(2) or "").lower()
        scale = {"lakh": 100_000, "crore": 10_000_000}.get(unit, 1)
        found.append((value * scale, bool(unit)))
    return found


def _flatten_numbers(value) -> set[float]:
    if isinstance(value, bool):
        return set()
    if isinstance(value, int | float):
        return {float(value)}
    if isinstance(value, str):
        return {number for number, _ in numbers_in(value)}
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        return {number for item in value for number in _flatten_numbers(item)}
    return set()


def check_numbers(words: str, known_facts) -> Check:
    known = _flatten_numbers(known_facts)
    unsupported = []
    for number, rounded in numbers_in(words):
        tolerance = number * _ROUNDED_TOLERANCE if rounded else 0.005
        if not any(abs(number - fact) <= tolerance for fact in known):
            unsupported.append(f"{number:g}")
    if unsupported:
        return Check(
            name="figures",
            passed=False,
            detail=f"These figures match no record: {', '.join(unsupported)}.",
        )
    return Check(name="figures", passed=True)


def check_citations(reasons: list[Reason], passages: dict[str, Passage]) -> Check:
    invented = sorted({c for r in reasons for c in r.citations if c not in passages})
    uncited = [r.statement for r in reasons if not r.citations]
    if invented:
        detail = f"These citations do not exist: {', '.join(invented)}."
        return Check(name="citations", passed=False, detail=detail)
    if uncited:
        return Check(
            name="citations", passed=False, detail=f"Reason with no citation: '{uncited[0]}'."
        )
    if not reasons:
        return Check(name="citations", passed=False, detail="No reasons were given.")
    return Check(name="citations", passed=True)


def check_vendors(words: str, data: CompanyData) -> Check:
    known = {v.id for v in data.vendors}
    invented = sorted(set(_VENDOR_ID.findall(words)) - known)
    if invented:
        detail = f"These vendors do not exist: {', '.join(invented)}."
        return Check(name="vendors", passed=False, detail=detail)
    return Check(name="vendors", passed=True)


def check_support(
    llm: LLMClient, reasons: list[Reason], passages: dict[str, Passage], facts: dict
) -> tuple[Check, int]:
    """Ask the model, separately, whether each reason is backed by its cited text."""
    statements = [
        {
            "index": index,
            "statement": reason.statement,
            "cited_text": {c: passages[c].text for c in reason.citations if c in passages},
        }
        for index, reason in enumerate(reasons)
    ]
    try:
        reply = llm.complete(
            [
                {"role": "system", "content": prompts.JUDGE},
                {
                    "role": "user",
                    "content": json.dumps({"facts": facts, "statements": statements}, default=str),
                },
            ],
            json_mode=True,
        )
        verdicts = _Verdicts.model_validate_json(reply.content or "")
    except (LLMError, ValidationError) as error:
        detail = f"The support check could not run: {str(error)[:120]}"
        return Check(name="support", passed=False, detail=detail), 0

    checked = {v.index for v in verdicts.results}
    unsupported = [v for v in verdicts.results if not v.supported]
    if unsupported or checked < set(range(len(reasons))):
        failures = "; ".join(
            f"'{reasons[v.index].statement}' ({v.why})"
            for v in unsupported
            if 0 <= v.index < len(reasons)
        )
        detail = f"Not supported by the cited text: {failures or 'not every reason was checked'}."
        return Check(name="support", passed=False, detail=detail), reply.tokens
    return Check(name="support", passed=True), reply.tokens


class _Verdict(BaseModel):
    index: int
    supported: bool
    why: str = ""


class _Verdicts(BaseModel):
    results: list[_Verdict]


def policy_basis(
    stock: StockPlan | None, order: OrderEvaluation | None, data: CompanyData
) -> list[PolicyBasis]:
    """The sections that applied, decided from the facts rather than the model's citations."""
    sections = []
    if stock:
        sections.append("POL-001 section 1.2")
    if order:
        vendor = next(v for v in data.vendors if v.id == order.vendor_id)
        sections += [
            "POL-001 section 1.4",
            "POL-002 section 2.1",
            "POL-003 section 3.2",
            "POL-003 section 3.3",
        ]
        if order.over_budget:
            sections.append("POL-001 section 1.3")
        if vendor.status != "approved":
            sections.append("POL-003 section 3.1")
        if order.currency != "INR":
            sections += ["POL-002 section 2.2", "POL-005 section 5.1", "POL-005 section 5.2"]
        if order.hardware_exception:
            sections += ["POL-002 section 2.2", "POL-004 section 4.2"]
    by_citation = {p.citation: p for p in load_passages()}
    return [
        PolicyBasis(citation=c, heading=by_citation[c].heading, text=by_citation[c].text)
        for c in dict.fromkeys(sections)
    ]


def sources(reasons: list[Reason], passages: dict[str, Passage]) -> list[Source]:
    """Every passage the reasons cite, labelled with its document type."""
    cited = dict.fromkeys(c for r in reasons for c in r.citations if c in passages)
    return [
        Source(
            citation=c,
            title=passages[c].doc_title,
            source_type=passages[c].source_type,
            structured=passages[c].structured,
            file=passages[c].source_file,
            text=passages[c].text,
        )
        for c in cited
    ]
