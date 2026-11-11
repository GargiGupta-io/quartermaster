"""The workflow, driven by a scripted model so each path can be tested exactly."""

import json

import httpx
import pytest

from procurement_agent.agent import MAX_DRAFT_ATTEMPTS, MAX_TOOL_ROUNDS, ProcurementAgent
from procurement_agent.config import Settings
from procurement_agent.knowledge import KnowledgeBase
from procurement_agent.llm import LLMError, LLMReply, ToolCall
from procurement_agent.prompts import JUDGE
from procurement_agent.tools import ToolBox

PARSED = {
    "is_purchase_request": True,
    "item": "standard laptops",
    "sku": "LAP-STD-14",
    "quantity": 20,
    "office": "Pune",
    "department": "Engineering",
    "budget_inr": 1_500_000,
    "business_reason": None,
}
CITED = [{"statement": "Stock is used first.", "citations": ["POL-001 section 1.2"]}]


def as_json(payload) -> LLMReply:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return LLMReply(content=text, tokens=10)


def calls(*items) -> LLMReply:
    tool_calls = [
        ToolCall(id=f"call-{i}", name=name, arguments=args) for i, (name, args) in enumerate(items)
    ]
    raw = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in tool_calls
        ],
    }
    return LLMReply(tool_calls=tool_calls, tokens=10, raw_message=raw)


DONE = LLMReply(content="Done.", tokens=10)


def draft(decision="purchase", vendor="V-101", **extra) -> LLMReply:
    return as_json(
        {"decision": decision, "vendor_id": vendor, "summary": "Plain summary.", "reasons": CITED}
        | extra
    )


class ScriptedLLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.judged = []
        self.verdicts = []

    def complete(self, messages, tools=None, json_mode=False):
        if messages and messages[0]["content"] == JUDGE:
            self.judged.append(messages)
            verdict = self.verdicts.pop(0) if self.verdicts else None
            if isinstance(verdict, Exception):
                raise verdict
            if verdict is None:
                count = len(json.loads(messages[1]["content"])["statements"])
                verdict = {"results": [{"index": i, "supported": True} for i in range(count)]}
            return as_json(verdict)
        self.requests.append({"messages": messages, "tools": tools, "json_mode": json_mode})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture(scope="module")
def knowledge(tmp_path_factory):
    return KnowledgeBase(index_dir=tmp_path_factory.mktemp("index"))


@pytest.fixture
def toolbox():
    def answer(request):
        return httpx.Response(200, json={"date": "2026-11-03", "rates": {"INR": 94.75}})

    return ToolBox(
        settings=Settings(_env_file=None),
        http=httpx.Client(transport=httpx.MockTransport(answer)),
    )


@pytest.fixture
def run(knowledge, toolbox):
    def _run(*replies, text="20 laptops for Pune engineering"):
        llm = ScriptedLLM(*replies)
        return ProcurementAgent(llm, toolbox, knowledge).run(text), llm

    return _run


def test_full_purchase_takes_figures_from_code(run):
    response, llm = run(
        as_json(PARSED),
        calls(("search_documents", {"query": "stock first"})),
        calls(("check_stock", PARSED | {"requested_quantity": 20})),
        DONE,
        draft(total_inr=1, required_approvers=["Nobody"]),
    )
    assert response.status == "ready_for_approval"
    assert response.order.total_inr == 1_359_360
    assert response.order.required_approvers == ["Department Head", "Finance Controller"]
    assert response.stock.purchase_quantity == 16
    assert response.reasons[0].citations == ["POL-001 section 1.2"]
    assert response.tools_used == ["search_documents", "check_stock"]
    assert response.tokens_used == 60
    assert llm.replies == []


def test_a_record_the_agent_looked_up_can_be_cited(run):
    stock_reason = {
        "statement": "4 laptops are already in stock in Pune, so stock is used first.",
        "citations": ["POL-001 section 1.2", "check_stock"],
    }
    response, _ = run(
        as_json(PARSED),
        calls(("check_stock", PARSED | {"requested_quantity": 20})),
        DONE,
        draft(reasons=[stock_reason]),
    )
    assert response.status == "ready_for_approval"
    assert response.verification.passed
    by_citation = {s.citation: s for s in response.sources}
    assert by_citation["check_stock"].source_type == "Company records"
    assert by_citation["check_stock"].title == "Stock levels by office"


def test_citing_a_record_that_was_never_looked_up_fails(run):
    reason = {"statement": "The budget allows it.", "citations": ["check_budget"]}
    response, _ = run(as_json(PARSED), DONE, draft(reasons=[reason]), draft(reasons=[reason]))
    assert response.status == "needs_review"
    assert "check_budget" in response.verification.checks[1].detail


def test_the_model_is_never_offered_purchase_order_creation(run):
    _, llm = run(as_json(PARSED), DONE, draft())
    offered = [t["function"]["name"] for t in llm.requests[1]["tools"]]
    assert "create_purchase_order" not in offered
    assert "search_documents" in offered


def test_a_requested_purchase_order_tool_call_is_refused(run, toolbox):
    response, llm = run(
        as_json(PARSED),
        calls(("create_purchase_order", {"sku": "LAP-STD-14"})),
        DONE,
        draft(),
    )
    assert toolbox.orders.committed_inr("Engineering") == 0
    tool_message = llm.requests[2]["messages"][-1]
    assert "not available" in tool_message["content"]
    assert response.status == "ready_for_approval"


def test_tool_errors_reach_the_model_so_it_can_correct_itself(run):
    _, llm = run(
        as_json(PARSED),
        calls(("check_budget", {"department": "Legal"})),
        DONE,
        draft(),
    )
    assert "Unknown department" in llm.requests[2]["messages"][-1]["content"]


def test_not_a_purchase_request(run):
    response, llm = run(
        as_json({"is_purchase_request": False}), text="What is the capital of France?"
    )
    assert response.status == "refused"
    assert len(llm.requests) == 1


def test_missing_details_become_questions(run):
    parsed = PARSED | {"department": None, "quantity": None, "office": "Delhi"}
    response, _ = run(as_json(parsed))
    assert response.status == "needs_clarification"
    assert len(response.questions) == 3
    assert any("Bengaluru" in q for q in response.questions)


def test_step_limit_stops_a_looping_model(run):
    loop = [calls(("check_budget", {"department": "Sales"})) for _ in range(MAX_TOOL_ROUNDS)]
    response, llm = run(as_json(PARSED), *loop)
    assert response.status == "failed"
    assert "Stopped after" in response.summary
    assert len(llm.requests) == MAX_TOOL_ROUNDS + 1


def test_forbidden_vendor_is_corrected(run):
    response, llm = run(as_json(PARSED), DONE, draft(vendor="V-106"), draft(vendor="V-101"))
    assert response.status == "ready_for_approval"
    assert response.order.vendor_id == "V-101"
    assert "suspended" in llm.requests[-1]["messages"][-1]["content"]


def test_dearer_vendor_without_reason_is_corrected(run):
    response, llm = run(as_json(PARSED), DONE, draft(vendor="V-103"), draft(vendor="V-101"))
    assert response.order.vendor_id == "V-101"
    assert "cheapest" in llm.requests[-1]["messages"][-1]["content"]


def test_dearer_vendor_with_reason_is_allowed(run):
    parsed = PARSED | {"business_reason": "Must arrive within 5 days"}
    response, _ = run(as_json(parsed), DONE, draft(vendor="V-103"))
    assert response.status == "ready_for_approval"
    assert response.order.vendor_id == "V-103"
    assert response.order.justification_required


def test_model_that_keeps_breaking_policy_fails_safely(run):
    response, _ = run(
        as_json(PARSED), DONE, *[draft(vendor="V-106") for _ in range(MAX_DRAFT_ATTEMPTS)]
    )
    assert response.status == "failed"
    assert response.order is None
    assert "suspended" in response.summary


def test_invented_vendor_is_rejected(run):
    response, _ = run(as_json(PARSED), DONE, draft(vendor="V-999"), draft(vendor="V-101"))
    assert response.order.vendor_id == "V-101"


def test_stock_decision_overrides_the_model(run):
    parsed = PARSED | {"quantity": 3}
    response, _ = run(as_json(parsed), DONE, draft())
    assert response.status == "no_purchase_needed"
    assert response.order is None
    assert response.stock.from_local_stock == 3


def test_transfer_needs_no_purchase(run):
    parsed = PARSED | {"quantity": 10, "office": "Mumbai"}
    response, _ = run(as_json(parsed), DONE, draft(decision="no_purchase", vendor=None))
    assert response.status == "no_purchase_needed"
    assert response.stock.transfer_from == "Bengaluru"


def test_claiming_stock_covers_it_when_it_does_not_is_corrected(run):
    response, _ = run(as_json(PARSED), DONE, draft(decision="no_purchase", vendor=None), draft())
    assert response.status == "ready_for_approval"


def test_unknown_product_can_only_be_refused(run):
    parsed = PARSED | {"item": "gift cards", "sku": None}
    response, _ = run(as_json(parsed), DONE, draft(), draft(decision="refuse", vendor=None))
    assert response.status == "refused"


def test_unlisted_item_is_refused_without_asking_for_details(run):
    parsed = {"is_purchase_request": True, "item": "gift cards", "department": "Sales"}
    response, _ = run(as_json(parsed), DONE, draft(decision="refuse", vendor=None))
    assert response.status == "refused"
    assert response.questions == []


def test_invented_product_code_is_treated_as_unlisted(run):
    parsed = PARSED | {"item": "yacht", "sku": "YACHT-1"}
    response, _ = run(as_json(parsed), DONE, draft(decision="refuse", vendor=None))
    assert response.status == "refused"
    assert response.request.sku is None


def test_ambiguous_item_asks_which_product(run):
    parsed = PARSED | {
        "item": "laptops",
        "sku": None,
        "candidate_skus": ["LAP-STD-14", "LAP-ENG-16", "NOT-REAL"],
    }
    response, _ = run(as_json(parsed))
    assert response.status == "needs_clarification"
    assert "LAP-ENG-16" in response.questions[0]
    assert "NOT-REAL" not in response.questions[0]


def test_refusing_a_stocked_product_is_corrected(run):
    parsed = PARSED | {"sku": "MON-27", "office": "Mumbai", "department": "Operations"}
    response, llm = run(
        as_json(parsed),
        DONE,
        draft(decision="refuse", vendor=None),
        draft(vendor="V-103"),
    )
    assert response.status == "ready_for_approval"
    assert response.order.over_budget
    assert "CFO" in response.order.required_approvers
    assert "not refusal" in llm.requests[-1]["messages"][-1]["content"]


def test_foreign_item_fetches_the_rate_if_the_model_did_not(run):
    parsed = PARSED | {"sku": "NET-SW48", "quantity": 2}
    response, _ = run(as_json(parsed), DONE, draft(vendor="V-105"))
    assert response.status == "ready_for_approval"
    assert response.order.total_inr == pytest.approx(495_184.34, abs=0.01)


def test_hardware_exception_without_reason_asks_for_one(run):
    parsed = PARSED | {"sku": "LAP-PRO-16", "quantity": 1, "office": "Bengaluru"}
    response, _ = run(as_json(parsed), DONE, draft())
    assert response.status == "needs_clarification"
    assert "business reason" in response.questions[0]


@pytest.mark.parametrize(
    "replies",
    [
        (as_json("not json"),),
        (LLMError("service down"),),
        (as_json(PARSED), LLMError("service down")),
        (as_json(PARSED), DONE, LLMError("service down")),
        (as_json(PARSED), DONE, as_json("no"), as_json("no"), as_json("still no")),
    ],
)
def test_model_failures_end_cleanly_with_no_order(run, replies):
    response, _ = run(*replies)
    assert response.status == "failed"
    assert response.order is None


QUESTION = {"is_purchase_request": False, "is_question": True, "item": None}
WARRANTY_QUESTION = "How long is the warranty on Lotus Peak laptops?"


def answer(reasons, found=True, text="Lotus Peak laptops carry a 24-month on-site warranty."):
    return as_json({"found": found, "answer": text, "reasons": reasons})


WARRANTY_REASON = [
    {"statement": "Laptops carry a 24-month on-site warranty.", "citations": ["CON-104 page 2"]}
]


def test_question_is_answered_from_a_cited_document(run):
    response, llm = run(as_json(QUESTION), answer(WARRANTY_REASON), text=WARRANTY_QUESTION)
    assert response.status == "answered"
    assert "24-month" in response.summary
    assert response.tools_used == ["search_documents"]
    assert [(s.citation, s.source_type) for s in response.sources] == [("CON-104 page 2", "PDF")]
    assert response.verification.passed
    assert len(llm.judged) == 1  # the support check still runs
    assert llm.requests[1]["tools"] is None  # answering is not offered any tools


def test_question_evidence_comes_from_the_search(run):
    _, llm = run(as_json(QUESTION), answer(WARRANTY_REASON), text=WARRANTY_QUESTION)
    prompt = llm.requests[1]["messages"][0]["content"]
    assert "CON-104 page 2" in prompt
    assert "POL-" in prompt  # policy sections are always searched too


def test_question_the_documents_do_not_cover(run):
    response, llm = run(
        as_json(QUESTION),
        answer([], found=False, text="Probably 3 years."),
        text="What is the CEO's favourite colour?",
    )
    assert response.status == "answered"
    assert response.summary.startswith("The company documents do not answer this.")
    assert "3 years" not in response.summary  # a guess is never shown
    assert llm.judged == []


def test_answer_with_an_invented_citation_is_corrected(run):
    invented = [
        {"statement": "Laptops carry a 24-month warranty.", "citations": ["CON-104 page 9"]}
    ]
    response, llm = run(
        as_json(QUESTION), answer(invented), answer(WARRANTY_REASON), text=WARRANTY_QUESTION
    )
    assert response.status == "answered"
    correction = llm.requests[2]["messages"][1]["content"]
    assert "CON-104 page 9" in correction


def test_answer_with_an_invented_figure_goes_to_review(run):
    wrong = "Lotus Peak laptops carry a 36-month on-site warranty."
    response, _ = run(
        as_json(QUESTION),
        answer(WARRANTY_REASON, text=wrong),
        answer(WARRANTY_REASON, text=wrong),
        text=WARRANTY_QUESTION,
    )
    assert response.status == "needs_review"
    assert any("36" in w for w in response.warnings)


def test_something_that_is_neither_a_purchase_nor_a_question_is_refused(run):
    response, _ = run(
        as_json({"is_purchase_request": False, "is_question": False}), text="Write me a poem"
    )
    assert response.status == "refused"
