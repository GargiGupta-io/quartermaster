"""Each check, using the kinds of mistakes a model actually makes."""

import json

import pytest

from procurement_agent.company_data import load_company_data
from procurement_agent.knowledge import load_passages
from procurement_agent.llm import LLMError, LLMReply
from procurement_agent.rules import evaluate_order
from procurement_agent.schemas import Reason
from procurement_agent.verify import (
    canonical_citation,
    check_citations,
    check_numbers,
    check_support,
    check_vendors,
    numbers_in,
    policy_basis,
    sources,
)

DATA = load_company_data()
PASSAGES = {p.citation: p for p in load_passages()}
FACTS = {"total_inr": 1_359_360.0, "unit_price_inr": 72_000.0, "quantity": 16}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("costs ₹13,59,360 in all", [1_359_360.0]),
        ("about 13.59 lakh", [1_359_000.0]),
        ("about 1.5 crore", [15_000_000.0]),
        ("See POL-001 section 1.2 and POL-002 2.1", []),
        ("The LAP-STD-14 and DOCK-USBC", []),
        ("See CON-104 page 3, PRL-Q4 row 18 and EML-022 message 2", []),
        ("See MEM-011 paragraph 4 and REG-VND record V-106", []),
        ("18% GST", [18.0]),
    ],
)
def test_finding_numbers(text, expected):
    assert [number for number, _ in numbers_in(text)] == expected


@pytest.mark.parametrize(
    "words",
    [
        "The total is ₹13,59,360 for 16 laptops at ₹72,000 each.",
        "The total is about 13.6 lakh.",
        "No figures here at all.",
    ],
)
def test_true_figures_pass(words):
    assert check_numbers(words, FACTS).passed


@pytest.mark.parametrize(
    "words",
    [
        "The total is ₹12,00,000.",
        "Each laptop costs ₹65,000.",
        "The total is about 20 lakh.",
        "Buying 17 laptops.",
    ],
)
def test_invented_figures_fail(words):
    check = check_numbers(words, FACTS)
    assert not check.passed
    assert "match no record" in check.detail


def test_vendor_ids_are_not_figures():
    assert check_numbers("V-106 is suspended, so V-101 is used.", FACTS).passed


def test_short_policy_citations_are_normalised():
    assert canonical_citation("POL-003 3.1") == "POL-003 section 3.1"
    assert canonical_citation("CON-104 page 2") == "CON-104 page 2"
    assert canonical_citation("check_stock") == "check_stock"


def test_real_citation_passes():
    reasons = [Reason(statement="x", citations=["POL-001 section 1.2"])]
    assert check_citations(reasons, PASSAGES).passed


@pytest.mark.parametrize(
    "reasons",
    [
        [Reason(statement="x", citations=["POL-009 section 9.9"])],
        [Reason(statement="x", citations=["POL-001 section 1.99"])],
        [Reason(statement="x", citations=[])],
        [],
    ],
)
def test_invented_or_missing_citations_fail(reasons):
    assert not check_citations(reasons, PASSAGES).passed


def test_records_count_as_citations_only_when_looked_up():
    reasons = [Reason(statement="x", citations=["check_stock"])]
    assert check_citations(reasons, PASSAGES, {"check_stock": {"available": 4}}).passed
    assert not check_citations(reasons, PASSAGES, {}).passed
    assert not check_citations(reasons, PASSAGES).passed


def test_vendors_must_exist():
    assert check_vendors("Buy from V-101, not V-106.", DATA).passed
    assert not check_vendors("Buy from V-777.", DATA).passed


class Judge:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, messages, tools=None, json_mode=False):
        if isinstance(self.reply, Exception):
            raise self.reply
        return LLMReply(content=self.reply, tokens=7)


REASON = [Reason(statement="Stock first.", citations=["POL-001 section 1.2"])]


def test_supported_statement_passes():
    judge = Judge(json.dumps({"results": [{"index": 0, "supported": True}]}))
    check, tokens = check_support(judge, REASON, PASSAGES, {})
    assert check.passed
    assert tokens == 7


@pytest.mark.parametrize(
    "reply",
    [
        json.dumps({"results": [{"index": 0, "supported": False, "why": "not stated"}]}),
        json.dumps({"results": []}),
        "not json",
        LLMError("down"),
    ],
)
def test_unsupported_or_unchecked_statement_fails_safe(reply):
    check, _ = check_support(Judge(reply), REASON, PASSAGES, {})
    assert not check.passed


def test_policy_basis_follows_the_facts():
    plain = evaluate_order(DATA, "LAP-STD-14", "V-101", 20, "Pune", "Engineering")
    citations = [b.citation for b in policy_basis(plain.stock, plain, DATA)]
    assert "POL-002 section 2.1" in citations
    assert "POL-005 section 5.1" not in citations

    foreign = evaluate_order(DATA, "NET-SW48", "V-105", 2, "Pune", "Engineering", fx_rate=94.75)
    citations = [b.citation for b in policy_basis(foreign.stock, foreign, DATA)]
    assert {"POL-005 section 5.1", "POL-005 section 5.2", "POL-002 section 2.2"} <= set(citations)

    over = evaluate_order(DATA, "MON-27", "V-103", 20, "Mumbai", "Operations")
    assert "POL-001 section 1.3" in [b.citation for b in policy_basis(over.stock, over, DATA)]


def test_sources_label_each_cited_document_type():
    reasons = [
        Reason(statement="Lotus Peak orders are capped.", citations=["CON-104 page 3"]),
        Reason(
            statement="Finance accepted a single supplier.",
            citations=["EML-022 message 2", "PRL-Q4 row 18", "CON-104 page 3"],
        ),
        Reason(statement="Invented.", citations=["XYZ-999 page 9"]),
    ]
    found = sources(reasons, PASSAGES)
    assert [(s.citation, s.source_type, s.structured) for s in found] == [
        ("CON-104 page 3", "PDF", False),
        ("EML-022 message 2", "Email thread", False),
        ("PRL-Q4 row 18", "Spreadsheet", True),
    ]
    assert found[0].file == "CON-104-lotus-peak-supply-agreement.pdf"
