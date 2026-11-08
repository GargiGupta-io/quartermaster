"""The workflow: understand the request, investigate with tools, then draft a decision."""

import json
import uuid
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ValidationError

from procurement_agent import prompts, verify
from procurement_agent.knowledge import KnowledgeBase, load_passages
from procurement_agent.llm import LLMClient, LLMError
from procurement_agent.rules import offices, plan_stock
from procurement_agent.schemas import (
    Answer,
    Decision,
    Draft,
    ParsedRequest,
    ProcurementResponse,
    Reason,
    VerificationResult,
)
from procurement_agent.tools import ToolBox

MAX_TOOL_ROUNDS = 8
MAX_DRAFT_ATTEMPTS = 3
MAX_CHECK_ATTEMPTS = 2  # the plan's "retry once": a first draft and one correction
POLICY_RESULTS = 3  # plus DOCUMENT_RESULTS from contracts, memos, emails and records
DOCUMENT_RESULTS = 2
READ_ONLY_TOOLS = (
    "check_stock",
    "list_vendor_offers",
    "check_budget",
    "get_exchange_rate",
    "evaluate_order",
)


class NotWaitingForApproval(Exception):
    """A decision arrived for a request that is not waiting for one."""


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


class SearchInput(BaseModel):
    query: str


SEARCH_TOOL = "search_documents"
SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": SEARCH_TOOL,
        "description": (
            "Search company documents: policies, contracts, memos, emails, the vendor rate card "
            "and vendor records. Returns passages with their citations."
        ),
        "parameters": SearchInput.model_json_schema(),
    },
}


class AgentState(TypedDict, total=False):
    request_text: str
    parsed: dict
    messages: list[dict]
    rounds: int
    draft_attempts: int
    correction: str
    evidence: dict[str, str]
    tool_log: list[dict]
    tokens: int
    response: dict | None
    decisions: list[dict]
    approved_roles: list[str]
    justification: str
    approval_error: str | None


def _strip_titles(schema: Any) -> Any:
    """Remove labels the model does not need, which saves tokens on free-tier limits."""
    if isinstance(schema, dict):
        return {k: _strip_titles(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_strip_titles(item) for item in schema]
    return schema


class ProcurementAgent:
    def __init__(
        self,
        llm: LLMClient,
        toolbox: ToolBox,
        knowledge: KnowledgeBase,
        checkpointer: Any = None,
    ):
        self.llm = llm
        self.toolbox = toolbox
        self.knowledge = knowledge
        self.tool_schemas = [
            _strip_titles(schema)
            for schema in (
                SEARCH_SCHEMA,
                *(toolbox.tools[name].schema() for name in READ_ONLY_TOOLS),
            )
        ]
        self._passages = {p.citation: p for p in load_passages()}
        self.graph = self._build().compile(checkpointer=checkpointer or InMemorySaver())

    def run(self, request_text: str, thread_id: str | None = None) -> ProcurementResponse:
        """Start a request. It stops at the first approval, if one is needed."""
        config = _config(thread_id or str(uuid.uuid4()))
        self.graph.invoke(
            {
                "request_text": request_text,
                "messages": [],
                "rounds": 0,
                "draft_attempts": 0,
                "evidence": {},
                "tool_log": [],
                "tokens": 0,
                "decisions": [],
                "approved_roles": [],
                "justification": "",
                "approval_error": None,
            },
            config,
        )
        return self.view(config)

    def decide(self, thread_id: str, decision: Decision) -> ProcurementResponse:
        """Record one person's decision and carry on from where the request stopped."""
        config = _config(thread_id)
        if not self.is_waiting(thread_id):
            raise NotWaitingForApproval(f"Request {thread_id} is not waiting for a decision.")
        self.graph.invoke(Command(resume=decision.model_dump()), config)
        return self.view(config)

    def is_waiting(self, thread_id: str) -> bool:
        return "approval" in self.graph.get_state(_config(thread_id)).next

    def get(self, thread_id: str) -> ProcurementResponse | None:
        config = _config(thread_id)
        if not self.graph.get_state(config).values:
            return None
        return self.view(config)

    def view(self, config: dict) -> ProcurementResponse:
        snapshot = self.graph.get_state(config)
        values = snapshot.values
        response = ProcurementResponse(**values["response"])
        response.decisions = [Decision(**d) for d in values.get("decisions", [])]
        response.approval_error = values.get("approval_error")
        if "approval" in snapshot.next and response.order:
            approved = values.get("approved_roles", [])
            response.pending_approvers = [
                role for role in response.order.required_approvers if role not in approved
            ]
        return response

    def _build(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("understand", self._guarded(self._understand))
        graph.add_node("investigate", self._guarded(self._investigate))
        graph.add_node("answer", self._guarded(self._answer))
        graph.add_node("draft", self._guarded(self._draft))
        graph.add_node("check", self._guarded(self._check))
        graph.add_node("approval", self._approval)
        graph.add_node("create_order", self._create_order)
        graph.add_edge(START, "understand")
        graph.add_conditional_edges("understand", self._after_understand)
        graph.add_conditional_edges("answer", self._after_answer)
        graph.add_conditional_edges("investigate", self._after_investigate)
        graph.add_conditional_edges("draft", lambda s: "check" if s.get("response") else "draft")
        graph.add_conditional_edges("check", self._after_check)
        graph.add_conditional_edges("approval", self._after_approval)
        graph.add_conditional_edges(
            "create_order", lambda s: END if s["response"]["status"] == "ordered" else "approval"
        )
        return graph

    def _guarded(self, node):
        """A model failure ends the run with a clear status rather than a crash."""

        def wrapped(state: AgentState) -> dict:
            try:
                return node(state)
            except LLMError as error:
                return {
                    "response": self._finish(
                        state,
                        "failed",
                        f"The AI service could not complete this request: {error}",
                    )
                }

        return wrapped

    def _finish(self, state: AgentState, status: str, summary: str, **fields) -> dict:
        parsed = state.get("parsed")
        return ProcurementResponse(
            status=status,
            summary=summary,
            request=ParsedRequest(**parsed) if parsed else None,
            tools_used=[entry["tool"] for entry in state.get("tool_log", [])],
            tokens_used=state.get("tokens", 0) + fields.pop("extra_tokens", 0),
            **fields,
        ).model_dump()

    def _understand(self, state: AgentState) -> dict:
        data = self.toolbox.data
        system = prompts.UNDERSTAND.format(
            products="\n".join(f"- {p.sku}: {p.name}" for p in data.products),
            offices=", ".join(sorted(offices(data))),
            departments=", ".join(b.department for b in data.budgets),
        )
        reply = self.llm.complete(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": state["request_text"]},
            ],
            json_mode=True,
        )
        tokens = state["tokens"] + reply.tokens
        try:
            parsed = ParsedRequest.model_validate_json(reply.content or "")
        except ValidationError as error:
            raise LLMError(
                f"could not read the request details: {error.error_count()} errors"
            ) from error

        known_skus = {p.sku for p in data.products}
        if parsed.sku not in known_skus:
            parsed.sku = None
        parsed.candidate_skus = [sku for sku in parsed.candidate_skus if sku in known_skus]

        update: dict = {"parsed": parsed.model_dump(), "tokens": tokens}
        current = {**state, **update}
        if not parsed.is_purchase_request:
            if not parsed.is_question:
                update["response"] = self._finish(
                    current,
                    "refused",
                    "This assistant handles purchase requests and questions about procurement.",
                )
            return update

        if parsed.item and not parsed.sku and not parsed.candidate_skus:
            return update  # not a product we stock: check policy and refuse, don't ask for details

        known_offices = offices(data)
        known_departments = {b.department for b in data.budgets}
        questions = []
        if not parsed.item:
            questions.append("What would you like to buy?")
        elif not parsed.sku:
            names = {p.sku: p.name for p in data.products}
            options = ", ".join(f"{names[sku]} ({sku})" for sku in parsed.candidate_skus)
            questions.append(f"Which product do you mean? {options}")
        if parsed.quantity is None:
            questions.append("How many do you need?")
        if parsed.office not in known_offices:
            questions.append(f"Which office is it for? ({', '.join(sorted(known_offices))})")
        if parsed.department not in known_departments:
            questions.append(
                f"Which department is paying? ({', '.join(sorted(known_departments))})"
            )
        if questions:
            update["response"] = self._finish(
                current,
                "needs_clarification",
                "A few details are needed before this can be checked.",
                questions=questions,
            )
        return update

    @staticmethod
    def _is_question(state: AgentState) -> bool:
        parsed = state.get("parsed") or {}
        return bool(parsed.get("is_question")) and not parsed.get("is_purchase_request")

    def _after_understand(self, state: AgentState) -> str:
        if state.get("response"):
            return END
        return "answer" if self._is_question(state) else "investigate"

    def _after_answer(self, state: AgentState) -> str:
        response = state.get("response")
        if not response:
            return "answer"
        return "check" if response["reasons"] else END

    def _answer(self, state: AgentState) -> dict:
        """A question about the documents: one search, then an answer that must cite it."""
        question = state["request_text"]
        tool_log = list(state["tool_log"])
        evidence = dict(state["evidence"])
        if not evidence:
            result = self._call_tool(SEARCH_TOOL, {"query": question}, evidence)
            tool_log.append(
                {"tool": SEARCH_TOOL, "arguments": {"query": question}, "result": result}
            )
        system = prompts.ANSWER.format(question=question, evidence=json.dumps(evidence))
        messages = [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": state.get("correction") or "Answer as JSON."})
        reply = self.llm.complete(messages, json_mode=True)
        attempts = state["draft_attempts"] + 1
        current = {
            **state,
            "evidence": evidence,
            "tool_log": tool_log,
            "tokens": state["tokens"] + reply.tokens,
            "draft_attempts": attempts,
        }
        update = {k: current[k] for k in ("evidence", "tool_log", "tokens", "draft_attempts")}
        try:
            answer = Answer.model_validate_json(reply.content or "")
        except ValidationError:
            if attempts >= MAX_DRAFT_ATTEMPTS:
                update["response"] = self._finish(
                    current, "failed", "Could not produce an answer in the expected format."
                )
            else:
                update["correction"] = "Your last reply was not valid JSON in the format."
            return update
        if not answer.found:
            answer.reasons = []
            answer.answer = (
                "The company documents do not answer this. Please ask the procurement desk."
            )
        update["response"] = self._finish(
            current, "answered", answer.answer, reasons=[r.model_dump() for r in answer.reasons]
        )
        return update

    def _investigate(self, state: AgentState) -> dict:
        messages = state["messages"] or [
            {
                "role": "system",
                "content": prompts.INVESTIGATE.format(request=json.dumps(state["parsed"])),
            },
            {"role": "user", "content": state["request_text"]},
        ]
        reply = self.llm.complete(messages, tools=self.tool_schemas)
        messages = [*messages, reply.raw_message or {"role": "assistant", "content": reply.content}]
        evidence = dict(state["evidence"])
        tool_log = list(state["tool_log"])
        for call in reply.tool_calls:
            result = self._call_tool(call.name, call.arguments, evidence)
            tool_log.append({"tool": call.name, "arguments": call.arguments, "result": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, default=str),
                }
            )
        return {
            "messages": messages,
            "evidence": evidence,
            "tool_log": tool_log,
            "rounds": state["rounds"] + 1,
            "tokens": state["tokens"] + reply.tokens,
        } | self._limit_response(state, reply, tool_log)

    def _limit_response(self, state: AgentState, reply, tool_log) -> dict:
        if reply.tool_calls and state["rounds"] + 1 >= MAX_TOOL_ROUNDS:
            return {
                "response": self._finish(
                    {**state, "tool_log": tool_log},
                    "failed",
                    f"Stopped after {MAX_TOOL_ROUNDS} rounds of checks without reaching a decision.",
                    extra_tokens=reply.tokens,
                )
            }
        return {}

    def _after_investigate(self, state: AgentState) -> str:
        if state.get("response"):
            return END
        last = state["messages"][-1]
        return "investigate" if last.get("role") == "tool" else "draft"

    def _call_tool(self, name: str, arguments: dict, evidence: dict[str, str]) -> dict:
        if name == SEARCH_TOOL:
            query = arguments.get("query", "")
            hits = self.knowledge.search_balanced(query, POLICY_RESULTS, DOCUMENT_RESULTS)
            for hit in hits:
                evidence[hit.passage.citation] = f"{hit.passage.heading}: {hit.passage.text}"
            return {
                "passages": [
                    {
                        "citation": h.passage.citation,
                        "source": h.passage.source_type,
                        "text": h.passage.text,
                    }
                    for h in hits
                ]
            }
        if name not in READ_ONLY_TOOLS:
            return {"ok": False, "error": f"Tool '{name}' is not available."}
        return self.toolbox.run(name, arguments).model_dump(exclude_none=True)

    def _draft(self, state: AgentState) -> dict:
        evidence = {
            "documents": state["evidence"],
            "tool_results": [
                {"tool": e["tool"], "arguments": e["arguments"], "result": e["result"]}
                for e in state["tool_log"]
                if e["tool"] != SEARCH_TOOL
            ],
        }
        system = prompts.DRAFT.format(
            request=json.dumps(state["parsed"]),
            evidence=json.dumps(evidence, default=str),
        )
        messages = [{"role": "system", "content": system}]
        if state.get("correction"):
            messages.append({"role": "user", "content": state["correction"]})
        else:
            messages.append({"role": "user", "content": "Return the decision as JSON."})
        reply = self.llm.complete(messages, json_mode=True)
        attempts = state["draft_attempts"] + 1
        current = {**state, "tokens": state["tokens"] + reply.tokens, "draft_attempts": attempts}
        try:
            draft = Draft.model_validate_json(reply.content or "")
        except ValidationError:
            outcome = self._retry_or_fail(
                current, "Your last reply was not valid JSON in the format."
            )
        else:
            outcome = self._apply_rules(current, draft)
        return {"tokens": current["tokens"], "draft_attempts": attempts} | outcome

    def _retry_or_fail(self, state: AgentState, problem: str) -> dict:
        if state["draft_attempts"] >= MAX_DRAFT_ATTEMPTS:
            return {
                "response": self._finish(
                    state, "failed", f"Could not reach a decision that follows policy: {problem}"
                )
            }
        return {
            "tokens": state["tokens"],
            "draft_attempts": state["draft_attempts"],
            "correction": f"Your last answer was rejected: {problem} Return corrected JSON.",
        }

    def _apply_rules(self, state: AgentState, draft: Draft) -> dict:
        """The model proposes; code decides whether the proposal is allowed and supplies figures."""
        parsed = ParsedRequest(**state["parsed"])
        reasons: list[Reason] = draft.reasons
        known_sku = parsed.sku in {p.sku for p in self.toolbox.data.products}

        if draft.decision == "refuse" and known_sku:
            return self._retry_or_fail(
                state,
                f"{parsed.sku} is a product the company buys. Going over budget or needing an "
                "exception means extra approval, not refusal. Decide purchase or no_purchase.",
            )
        if draft.decision == "refuse":
            return {"response": self._finish(state, "refused", draft.summary, reasons=reasons)}
        if not known_sku:
            return self._retry_or_fail(
                state, f"'{parsed.sku}' is not a product the company buys, so refuse this request."
            )

        stock = plan_stock(self.toolbox.data, parsed.sku, parsed.office, parsed.quantity)
        if stock.purchase_quantity == 0:
            return {
                "response": self._finish(
                    state, "no_purchase_needed", draft.summary, stock=stock, reasons=reasons
                )
            }
        if draft.decision == "no_purchase":
            return self._retry_or_fail(
                state, f"Stock does not cover this: {stock.purchase_quantity} must be purchased."
            )
        if not draft.vendor_id:
            return self._retry_or_fail(state, "A purchase needs a vendor_id.")

        fx_rate = self._latest_fx_rate(state)
        if fx_rate is None and self._bills_in_foreign_currency(draft.vendor_id):
            rate = self.toolbox.run("get_exchange_rate", {"currency": "USD"})
            if not rate.ok:
                return {
                    "response": self._finish(
                        state,
                        "failed",
                        f"The exchange rate is needed but unavailable: {rate.error}",
                        stock=stock,
                    )
                }
            fx_rate = rate.data["rate"]

        result = self.toolbox.run(
            "evaluate_order",
            {
                "sku": parsed.sku,
                "vendor_id": draft.vendor_id,
                "requested_quantity": parsed.quantity,
                "office": parsed.office,
                "department": parsed.department,
                "fx_rate": fx_rate,
            },
        )
        if not result.ok:
            return self._retry_or_fail(state, result.error or "That vendor cannot be evaluated.")
        order = result.data
        if order["blocking_problems"]:
            return self._retry_or_fail(state, " ".join(order["blocking_problems"]))
        lowest = order["lowest_eligible_vendor"]
        if lowest and lowest != draft.vendor_id and not parsed.business_reason:
            return self._retry_or_fail(state, f"{lowest} is the cheapest eligible vendor.")
        if order["hardware_exception"] and not parsed.business_reason:
            return {
                "response": self._finish(
                    state,
                    "needs_clarification",
                    "This laptop needs a hardware exception, which requires a business reason.",
                    stock=stock,
                    order=order,
                    reasons=reasons,
                    questions=["What is the business reason for this laptop?"],
                )
            }
        return {
            "response": self._finish(
                state,
                "ready_for_approval",
                draft.summary,
                stock=stock,
                order=order,
                reasons=reasons,
            )
        }

    def _check(self, state: AgentState) -> dict:
        """Check the draft's words, retry once on failure, then flag it for a person."""
        response = ProcurementResponse(**state["response"])
        if response.status in ("failed", "needs_clarification"):
            return {}

        words = " ".join([response.summary, *(reason.statement for reason in response.reasons)])
        facts = {
            "request_text": state["request_text"],
            "request": state["parsed"],
            "stock": response.stock.model_dump() if response.stock else None,
            "order": response.order.model_dump() if response.order else None,
            "tools": [e["result"] for e in state["tool_log"] if e["tool"] != SEARCH_TOOL],
        }
        cited_text = [
            self._passages[c].text
            for r in response.reasons
            for c in r.citations
            if c in self._passages
        ]
        checks = [
            verify.check_numbers(words, [facts, cited_text]),
            verify.check_citations(response.reasons, self._passages),
            verify.check_vendors(words, self.toolbox.data),
        ]
        tokens = state["tokens"]
        if all(check.passed for check in checks):
            support, used = verify.check_support(
                self.llm,
                response.reasons,
                self._passages,
                {k: v for k, v in facts.items() if k in ("request", "stock", "order")},
            )
            checks.append(support)
            tokens += used
        result = verify.Verification(passed=all(c.passed for c in checks), checks=checks)

        if not result.passed and state["draft_attempts"] < MAX_CHECK_ATTEMPTS:
            return {
                "response": None,
                "tokens": tokens,
                "correction": f"Your last answer failed checking: {result.problems} "
                "Correct it using only the evidence.",
            }

        response.verification = VerificationResult(**result.model_dump())
        response.tokens_used = tokens
        response.policy_basis = verify.policy_basis(
            response.stock, response.order, self.toolbox.data
        )
        response.sources = verify.sources(response.reasons, self._passages)
        parsed = ParsedRequest(**state["parsed"])
        if parsed.budget_inr and response.order and response.order.total_inr > parsed.budget_inr:
            response.warnings.append(
                f"The total of ₹{response.order.total_inr:,.2f} is over the requester's stated "
                f"budget of ₹{parsed.budget_inr:,.2f}."
            )
        if not result.passed:
            response.warnings.append(f"Failed automatic checking: {result.problems}")
            response.status = "needs_review"
            if not response.order:
                response.summary = (
                    "This answer could not be verified and needs a person to review it. "
                    + response.summary
                )
        return {"response": response.model_dump(), "tokens": tokens}

    def _after_check(self, state: AgentState) -> str:
        response = state.get("response")
        if not response:
            return "answer" if self._is_question(state) else "draft"
        if response["status"] in ("ready_for_approval", "needs_review") and response["order"]:
            return "approval"
        return END

    def _approval(self, state: AgentState) -> dict:
        """Pause until a person decides. The pause is saved, so it survives a restart."""
        order = state["response"]["order"]
        approved = state["approved_roles"]
        pending = [role for role in order["required_approvers"] if role not in approved]
        decision = Decision(**interrupt({"pending_approvers": pending}))

        if decision.role not in pending:
            return {
                "approval_error": f"{decision.role} is not an outstanding approver. "
                f"Waiting for: {', '.join(pending)}."
            }
        decisions = [*state["decisions"], decision.model_dump()]
        if decision.action == "reject":
            response = {**state["response"], "status": "rejected"}
            return {"decisions": decisions, "response": response, "approval_error": None}
        if decision.action == "edit":
            return self._apply_edit(state, decision, decisions)

        justification = decision.justification or state["justification"]
        reason = justification or (state["parsed"].get("business_reason") or "")
        if order["justification_required"] and not reason.strip():
            needed = " ".join(order["justification_required"])
            return {"approval_error": f"A justification is required to approve this: {needed}"}
        return {
            "decisions": decisions,
            "approved_roles": [*approved, decision.role],
            "justification": reason,
            "approval_error": None,
        }

    def _apply_edit(self, state: AgentState, decision: Decision, decisions: list[dict]) -> dict:
        """Changing an order recalculates it and restarts approval, as people approved the old one."""
        order = state["response"]["order"]
        parsed = state["parsed"]
        quantity = decision.requested_quantity or order["stock"]["requested"]
        vendor_id = decision.vendor_id or order["vendor_id"]
        fx_rate = self._latest_fx_rate(state)
        if fx_rate is None and self._bills_in_foreign_currency(vendor_id):
            rate = self.toolbox.run("get_exchange_rate", {"currency": "USD"})
            if not rate.ok:
                return {"approval_error": f"Exchange rate unavailable: {rate.error}"}
            fx_rate = rate.data["rate"]
        result = self.toolbox.run(
            "evaluate_order",
            {
                "sku": parsed["sku"],
                "vendor_id": vendor_id,
                "requested_quantity": quantity,
                "office": parsed["office"],
                "department": parsed["department"],
                "fx_rate": fx_rate,
            },
        )
        if not result.ok:
            return {"approval_error": f"That edit is not possible: {result.error}"}
        new_order = result.data
        if new_order["blocking_problems"]:
            problems = " ".join(new_order["blocking_problems"])
            return {"approval_error": f"That edit breaks policy: {problems}"}
        if new_order["stock"]["purchase_quantity"] == 0:
            return {"approval_error": "That edit needs no purchase, so reject the request instead."}
        response = {**state["response"], "order": new_order, "stock": new_order["stock"]}
        return {
            "decisions": decisions,
            "response": response,
            "approved_roles": [],
            "approval_error": None,
        }

    def _after_approval(self, state: AgentState) -> str:
        response = state["response"]
        if response["status"] == "rejected":
            return END
        required = response["order"]["required_approvers"]
        if all(role in state["approved_roles"] for role in required):
            return "create_order"
        return "approval"

    def _create_order(self, state: AgentState, config: RunnableConfig) -> dict:
        """Only reachable after every approval. The rules are checked once more by the tool."""
        response = state["response"]
        order = response["order"]
        result = self.toolbox.run(
            "create_purchase_order",
            {
                "sku": state["parsed"]["sku"],
                "vendor_id": order["vendor_id"],
                "requested_quantity": order["stock"]["requested"],
                "office": state["parsed"]["office"],
                "department": state["parsed"]["department"],
                "fx_rate": order["fx_rate"],
                "idempotency_key": config["configurable"]["thread_id"],
                "approved_by": state["approved_roles"],
                "justification": state["justification"],
            },
        )
        if not result.ok:
            return {
                "approved_roles": [],
                "approval_error": f"The order could not be created: {result.error}",
            }
        return {"response": {**response, "status": "ordered", "purchase_order": result.data}}

    def _bills_in_foreign_currency(self, vendor_id: str) -> bool:
        return any(v.id == vendor_id and v.currency != "INR" for v in self.toolbox.data.vendors)

    @staticmethod
    def _latest_fx_rate(state: AgentState) -> float | None:
        for entry in reversed(state["tool_log"]):
            if entry["tool"] == "get_exchange_rate" and entry["result"].get("ok"):
                return entry["result"]["data"]["rate"]
        return None
