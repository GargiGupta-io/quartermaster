"""Human approval and the checks that run before it."""

import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from procurement_agent.agent import NotWaitingForApproval, ProcurementAgent
from procurement_agent.config import Settings
from procurement_agent.knowledge import KnowledgeBase
from procurement_agent.llm import LLMError
from procurement_agent.purchase_orders import PurchaseOrderStore
from procurement_agent.schemas import Decision
from procurement_agent.tools import ToolBox
from test_agent import DONE, PARSED, ScriptedLLM, as_json, draft

HEAD = "Department Head"
FINANCE = "Finance Controller"
THREAD = "request-0001"


@pytest.fixture(scope="module")
def knowledge(tmp_path_factory):
    return KnowledgeBase(index_dir=tmp_path_factory.mktemp("index"))


@pytest.fixture
def toolbox():
    return ToolBox(settings=Settings(_env_file=None))


def start(knowledge, toolbox, *drafts, parsed=PARSED, thread=THREAD, verdicts=()):
    llm = ScriptedLLM(as_json(parsed), DONE, *(drafts or (draft(),)))
    llm.verdicts = list(verdicts)
    agent = ProcurementAgent(llm, toolbox, knowledge)
    return agent, agent.run("request", thread_id=thread), llm


def approve(role, **extra) -> Decision:
    return Decision(role=role, action="approve", name="A Person", **extra)


def test_nothing_is_ordered_until_every_approver_agrees(knowledge, toolbox):
    agent, first, _ = start(knowledge, toolbox)
    assert first.status == "ready_for_approval"
    assert first.pending_approvers == [HEAD, FINANCE]
    assert first.purchase_order is None
    assert toolbox.orders.committed_inr("Engineering") == 0

    second = agent.decide(THREAD, approve(HEAD))
    assert second.pending_approvers == [FINANCE]
    assert second.purchase_order is None
    assert toolbox.orders.committed_inr("Engineering") == 0

    final = agent.decide(THREAD, approve(FINANCE, note="Fine"))
    assert final.status == "ordered"
    assert final.purchase_order["total_inr"] == 1_359_360
    assert final.purchase_order["approved_by"] == [HEAD, FINANCE]
    assert [d.role for d in final.decisions] == [HEAD, FINANCE]
    assert toolbox.orders.committed_inr("Engineering") == 1_359_360


def test_rejection_ends_it_with_no_order(knowledge, toolbox):
    agent, _, _ = start(knowledge, toolbox)
    response = agent.decide(THREAD, Decision(role=HEAD, action="reject", name="A", note="No"))
    assert response.status == "rejected"
    assert response.purchase_order is None
    assert toolbox.orders.committed_inr("Engineering") == 0


def test_someone_who_is_not_an_approver_changes_nothing(knowledge, toolbox):
    agent, _, _ = start(knowledge, toolbox)
    response = agent.decide(THREAD, approve("CFO"))
    assert "not an outstanding approver" in response.approval_error
    assert response.pending_approvers == [HEAD, FINANCE]


def test_the_same_person_cannot_count_twice(knowledge, toolbox):
    agent, _, _ = start(knowledge, toolbox)
    agent.decide(THREAD, approve(HEAD))
    response = agent.decide(THREAD, approve(HEAD))
    assert response.pending_approvers == [FINANCE]
    assert response.approval_error


def test_decision_after_it_is_finished_is_refused(knowledge, toolbox):
    agent, _, _ = start(knowledge, toolbox)
    agent.decide(THREAD, Decision(role=HEAD, action="reject", name="A"))
    with pytest.raises(NotWaitingForApproval):
        agent.decide(THREAD, approve(HEAD))


def test_edit_recalculates_and_restarts_approval(knowledge, toolbox):
    agent, _, _ = start(knowledge, toolbox)
    agent.decide(THREAD, approve(HEAD))
    edited = agent.decide(
        THREAD, Decision(role=FINANCE, action="edit", name="A", requested_quantity=18)
    )
    assert edited.order.stock.purchase_quantity == 14
    assert edited.order.total_inr == 1_189_440
    assert edited.pending_approvers == [HEAD, FINANCE]  # the earlier approval no longer counts


@pytest.mark.parametrize(
    ("change", "phrase"),
    [
        ({"vendor_id": "V-106"}, "suspended"),
        ({"vendor_id": "V-999"}, "not possible"),
        ({"requested_quantity": 2}, "no purchase"),
    ],
)
def test_an_edit_cannot_break_policy(knowledge, toolbox, change, phrase):
    agent, _, _ = start(knowledge, toolbox)
    response = agent.decide(THREAD, Decision(role=HEAD, action="edit", name="A", **change))
    assert phrase in response.approval_error
    assert response.order.vendor_id == "V-101"
    assert response.order.stock.requested == 20


def test_pause_survives_a_restart(knowledge, tmp_path):
    def build(llm):
        connection = sqlite3.connect(tmp_path / "state.db", check_same_thread=False)
        toolbox = ToolBox(
            settings=Settings(_env_file=None), orders=PurchaseOrderStore(tmp_path / "orders.db")
        )
        return ProcurementAgent(llm, toolbox, knowledge, checkpointer=SqliteSaver(connection))

    build(ScriptedLLM(as_json(PARSED), DONE, draft())).run("request", thread_id=THREAD)

    restarted = build(ScriptedLLM())  # a new process: it must not need the model again
    assert restarted.is_waiting(THREAD)
    assert restarted.get(THREAD).pending_approvers == [HEAD, FINANCE]
    restarted.decide(THREAD, approve(HEAD))
    assert restarted.decide(THREAD, approve(FINANCE)).status == "ordered"


def test_justification_is_required_when_policy_says_so(knowledge, toolbox):
    parsed = PARSED | {"sku": "NET-SW48", "quantity": 1}
    agent, first, _ = start(knowledge, toolbox, draft(vendor="V-105"), parsed=parsed)
    assert first.order.sole_source
    refused = agent.decide(THREAD, approve(HEAD))
    assert "justification is required" in refused.approval_error
    assert refused.pending_approvers == [HEAD, FINANCE]
    agent.decide(THREAD, approve(HEAD, justification="Only supplier of this switch"))
    final = agent.decide(THREAD, approve(FINANCE))
    assert final.status == "ordered"
    assert final.purchase_order["justification"] == "Only supplier of this switch"


def test_invented_figure_is_corrected_once(knowledge, toolbox):
    _, response, llm = start(
        knowledge, toolbox, draft(summary="This costs ₹9,99,999."), draft(summary="Fine.")
    )
    assert response.status == "ready_for_approval"
    assert response.verification.passed
    assert "match no record" in llm.requests[-1]["messages"][-1]["content"]


def test_real_figure_passes_first_time(knowledge, toolbox):
    _, response, llm = start(
        knowledge, toolbox, draft(summary="This costs ₹13,59,360 for 16 laptops.")
    )
    assert response.verification.passed
    assert len(llm.requests) == 3


def test_failing_twice_is_flagged_for_a_person_and_stays_blocked(knowledge, toolbox):
    bad = draft(summary="This costs ₹9,99,999.")
    _, response, llm = start(knowledge, toolbox, bad, bad)
    assert response.status == "needs_review"
    assert not response.verification.passed
    assert any("match no record" in warning for warning in response.warnings)
    assert response.pending_approvers == [HEAD, FINANCE]
    assert response.purchase_order is None
    assert len(llm.requests) == 4


def test_invented_citation_is_caught(knowledge, toolbox):
    wrong = draft(reasons=[{"statement": "A rule.", "citations": ["POL-099 section 9.9"]}])
    _, response, _ = start(knowledge, toolbox, wrong, wrong)
    assert response.status == "needs_review"
    assert "do not exist" in response.warnings[0]


def test_unsupported_claim_is_caught(knowledge, toolbox):
    no = {"results": [{"index": 0, "supported": False, "why": "not in the text"}]}
    _, response, llm = start(knowledge, toolbox, draft(), draft(), verdicts=[no, no])
    assert response.status == "needs_review"
    assert len(llm.judged) == 2


def test_support_check_being_down_flags_rather_than_passes(knowledge, toolbox):
    down = LLMError("down")
    _, response, _ = start(knowledge, toolbox, draft(), draft(), verdicts=[down, down])
    assert response.status == "needs_review"
    assert "could not run" in response.warnings[0]


def test_unverifiable_refusal_needs_review_and_no_approval(knowledge, toolbox):
    parsed = {"is_purchase_request": True, "item": "gift cards"}
    bad = draft(decision="refuse", vendor=None, reasons=[])
    _, response, _ = start(knowledge, toolbox, bad, bad, parsed=parsed)
    assert response.status == "needs_review"
    assert response.pending_approvers == []


def test_policy_sections_are_attached_from_the_facts(knowledge, toolbox):
    _, response, _ = start(knowledge, toolbox)
    citations = [basis.citation for basis in response.policy_basis]
    assert "POL-002 section 2.1" in citations
    assert all(basis.text for basis in response.policy_basis)


def test_over_stated_budget_warns_the_approver(knowledge, toolbox):
    parsed = PARSED | {"budget_inr": 1_000_000}
    _, response, _ = start(knowledge, toolbox, parsed=parsed)
    assert any("stated budget" in warning for warning in response.warnings)
