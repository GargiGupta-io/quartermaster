"""The evaluation scorer, on hand-made responses."""

import json
from pathlib import Path

from procurement_agent.company_data import load_company_data
from procurement_agent.evaluation import expected_order, score, summary

DATA = load_company_data()
CASES = {c["id"]: c for c in json.loads(Path("evals/cases.json").read_text())["cases"]}
LAPTOPS = CASES["P01"]


def purchase_response(**changes):
    expected = expected_order(LAPTOPS["order"], DATA)
    order = {
        "vendor_id": expected["vendor_id"],
        "required_approvers": expected["approvers"],
        "total_inr": expected["total_inr"],
    }
    response = {
        "status": "ready_for_approval",
        "summary": "Buy 16.",
        "stock": expected["stock"],
        "order": order,
        "verification": {"passed": True},
        "tokens_used": 16000,
        "sources": [],
    }
    for key, value in changes.items():
        if key in order:
            order[key] = value
        else:
            response[key] = value
    return response


def test_cases_have_unique_ids_and_a_known_kind():
    raw = json.loads(Path("evals/cases.json").read_text())["cases"]
    assert len({c["id"] for c in raw}) == len(raw) == 40
    assert {c["kind"] for c in raw} == {"purchase", "question", "guardrail"}


def test_expected_answer_comes_from_the_rules():
    expected = expected_order(LAPTOPS["order"], DATA)
    assert expected["vendor_id"] == "V-101"
    assert expected["total_inr"] == 1_359_360
    assert expected["stock"]["purchase_quantity"] == 16


def test_a_correct_purchase_passes():
    assert score(LAPTOPS, purchase_response(), 60, DATA).passed


def test_a_wrong_vendor_fails_with_a_reason():
    result = score(LAPTOPS, purchase_response(vendor_id="V-104"), 60, DATA)
    assert not result.passed
    assert result.aspects["vendor"] is False
    assert "expected V-101" in result.problems[0]


def test_a_flagged_purchase_fails_on_status():
    result = score(LAPTOPS, purchase_response(status="needs_review"), 60, DATA)
    assert result.aspects["status"] is False


def test_question_needs_the_fact_and_the_source():
    case = CASES["Q01"]
    good = {
        "status": "answered",
        "summary": "They carry a 24-month warranty.",
        "sources": [{"citation": "CON-104 page 2"}],
        "verification": {"passed": True},
        "tokens_used": 2000,
    }
    assert score(case, good, 3, DATA).passed
    wrong_source = good | {"sources": [{"citation": "MEM-011 paragraph 1"}]}
    assert score(case, wrong_source, 3, DATA).aspects["source"] is False


def test_declining_an_uncovered_question_passes():
    response = {
        "status": "answered",
        "sources": [],
        "tokens_used": 900,
        "summary": "The company documents do not answer this. Please ask the procurement desk.",
    }
    assert score(CASES["Q17"], response, 2, DATA).passed


def test_a_screened_request_must_use_no_ai():
    response = {"status": "refused", "summary": "x", "tokens_used": 0}
    assert score(CASES["G01"], response, 0, DATA).passed
    assert not score(CASES["G01"], response | {"tokens_used": 500}, 0, DATA).passed


def test_summary_counts_by_kind():
    results = [
        score(LAPTOPS, purchase_response(), 60, DATA),
        score(CASES["G01"], {"status": "refused", "summary": "", "tokens_used": 0}, 0, DATA),
    ]
    numbers = summary(results)
    assert numbers["Cases fully correct"] == "2/2"
    assert numbers["Right vendor"] == "1/1"
    assert numbers["Total tokens"] == "16,000"
