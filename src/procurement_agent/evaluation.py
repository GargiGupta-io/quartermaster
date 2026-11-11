"""Score the agent's answers against known right answers, and summarise the results.

Purchase answers are worked out from the policy rules in code, so the evaluation checks the
AI's choices (which vendor, what status, which sources) against what the rules require.
"""

import statistics
from dataclasses import dataclass, field

from procurement_agent.company_data import CompanyData
from procurement_agent.rules import evaluate_order, plan_stock

DECLINE_TEXT = "do not answer"
PLACEHOLDER_FX = 85.0  # only used to find the vendor; foreign totals are not compared


@dataclass
class Score:
    case_id: str
    kind: str
    passed: bool
    status: str
    seconds: float
    tokens: int
    checks_passed: bool | None
    problems: list[str] = field(default_factory=list)
    aspects: dict[str, bool] = field(default_factory=dict)


def expected_order(order: dict, data: CompanyData) -> dict:
    """What the policy rules say should happen for a purchase."""
    stock = plan_stock(data, order["sku"], order["office"], order["quantity"])
    expected = {"stock": stock.model_dump()}
    if stock.purchase_quantity == 0:
        return expected | {"status": "no_purchase_needed"}
    first = next(offer for offer in data.catalog if offer.sku == order["sku"])
    args = (order["sku"], order["quantity"], order["office"], order["department"])
    probe = evaluate_order(data, args[0], first.vendor_id, *args[1:], fx_rate=PLACEHOLDER_FX)
    best = evaluate_order(
        data, args[0], probe.lowest_eligible_vendor, *args[1:], fx_rate=PLACEHOLDER_FX
    )
    return expected | {
        "status": "ready_for_approval",
        "vendor_id": best.vendor_id,
        "approvers": sorted(best.required_approvers),
        "total_inr": best.total_inr if best.currency == "INR" else None,
    }


def score(case: dict, response: dict, seconds: float, data: CompanyData) -> Score:
    result = Score(
        case_id=case["id"],
        kind=case["kind"],
        passed=False,
        status=response["status"],
        seconds=seconds,
        tokens=response.get("tokens_used", 0),
        checks_passed=(response.get("verification") or {}).get("passed"),
    )
    if "status" in case:
        _expect(
            result,
            "status",
            response["status"] in case["status"],
            f"status {response['status']}, expected {' or '.join(case['status'])}",
        )
        if case.get("no_ai"):
            _expect(result, "no_ai", result.tokens == 0, f"used {result.tokens} AI tokens")
    elif "order" in case:
        _score_order(result, case, response, data)
    elif case.get("declined"):
        _expect(
            result,
            "declined",
            response["status"] == "answered"
            and DECLINE_TEXT in response["summary"]
            and not response.get("sources"),
            f"answered instead of declining: {response['summary'][:80]}",
        )
    else:
        _score_answer(result, case, response)
    result.passed = all(result.aspects.values())
    return result


def _expect(result: Score, aspect: str, ok: bool, problem: str) -> None:
    result.aspects[aspect] = ok
    if not ok:
        result.problems.append(problem)


def _score_order(result: Score, case: dict, response: dict, data: CompanyData) -> None:
    expected = expected_order(case["order"], data)
    _expect(
        result,
        "status",
        response["status"] == expected["status"],
        f"status {response['status']}, expected {expected['status']}",
    )
    stock = response.get("stock") or {}
    _expect(result, "stock", stock == expected["stock"], "stock plan differs from the rules")
    if expected["status"] != "ready_for_approval":
        return
    order = response.get("order") or {}
    _expect(
        result,
        "vendor",
        order.get("vendor_id") == expected["vendor_id"],
        f"vendor {order.get('vendor_id')}, expected {expected['vendor_id']}",
    )
    _expect(
        result,
        "approvers",
        sorted(order.get("required_approvers", [])) == expected["approvers"],
        f"approvers {order.get('required_approvers')}, expected {expected['approvers']}",
    )
    if expected["total_inr"] is not None:
        _expect(
            result,
            "figures",
            abs(order.get("total_inr", 0) - expected["total_inr"]) < 1,
            f"total {order.get('total_inr')}, expected {expected['total_inr']}",
        )
    _expect(result, "checks", result.checks_passed is True, "automatic checks did not pass")


def _score_answer(result: Score, case: dict, response: dict) -> None:
    _expect(result, "status", response["status"] == "answered", f"status {response['status']}")
    answer = response["summary"].lower()
    _expect(
        result,
        "answer",
        any(p.lower() in answer for p in case["answer_any"]),
        f"answer lacks {case['answer_any']}: {response['summary'][:100]}",
    )
    cited = [s["citation"] for s in response.get("sources", [])]
    _expect(
        result,
        "source",
        any(c.startswith(d) for c in cited for d in case["cite_any"]),
        f"cited {cited or 'nothing'}, expected {' or '.join(case['cite_any'])}",
    )
    _expect(result, "checks", result.checks_passed is True, "automatic checks did not pass")


def _rate(scores: list[Score], aspect: str) -> str:
    relevant = [s for s in scores if aspect in s.aspects]
    if not relevant:
        return "-"
    good = sum(s.aspects[aspect] for s in relevant)
    return f"{good}/{len(relevant)}"


def summary(scores: list[Score]) -> dict[str, str]:
    """The headline numbers for one run."""
    ai_runs = [s for s in scores if s.tokens]
    seconds = sorted(s.seconds for s in ai_runs) or [0.0]
    return {
        "Cases fully correct": f"{sum(s.passed for s in scores)}/{len(scores)}",
        "Purchases fully correct": _count(s for s in scores if s.kind == "purchase"),
        "Questions fully correct": _count(s for s in scores if s.kind == "question"),
        "Guardrail cases correct": _count(s for s in scores if s.kind == "guardrail"),
        "Right status or decision": _rate(scores, "status"),
        "Right vendor": _rate(scores, "vendor"),
        "Figures match the rules": _rate(scores, "figures"),
        "Right approvers": _rate(scores, "approvers"),
        "Answer contains the fact": _rate(scores, "answer"),
        "Right document cited": _rate(scores, "source"),
        "Declined when not covered": _rate(scores, "declined"),
        "Automatic checks passed": _rate(scores, "checks"),
        "Sent to a person (needs review)": str(sum(s.status == "needs_review" for s in scores)),
        "Median seconds per AI case": f"{statistics.median(seconds):.1f}",
        "Slowest case, seconds": f"{seconds[-1]:.1f}",
        "Average tokens per AI case": f"{statistics.mean([s.tokens for s in ai_runs] or [0]):,.0f}",
        "Total tokens": f"{sum(s.tokens for s in scores):,}",
    }


def _count(scores) -> str:
    scores = list(scores)
    return f"{sum(s.passed for s in scores)}/{len(scores)}"
