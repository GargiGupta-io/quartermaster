"""The workflow: understand the request, investigate with tools, then draft a decision."""

import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from procurement_agent import prompts
from procurement_agent.knowledge import KnowledgeBase
from procurement_agent.llm import LLMClient, LLMError
from procurement_agent.rules import offices, plan_stock
from procurement_agent.schemas import Draft, ParsedRequest, ProcurementResponse, Reason
from procurement_agent.tools import ToolBox

MAX_TOOL_ROUNDS = 8
MAX_DRAFT_ATTEMPTS = 3
SEARCH_RESULTS = 4
READ_ONLY_TOOLS = (
    "check_stock",
    "list_vendor_offers",
    "check_budget",
    "get_exchange_rate",
    "evaluate_order",
)


class SearchInput(BaseModel):
    query: str


SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_policies",
        "description": "Search company policy documents. Returns sections with their citations.",
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
    response: dict


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
        self.graph = self._build().compile(checkpointer=checkpointer)

    def run(self, request_text: str, config: dict | None = None) -> ProcurementResponse:
        state = self.graph.invoke(
            {
                "request_text": request_text,
                "messages": [],
                "rounds": 0,
                "draft_attempts": 0,
                "evidence": {},
                "tool_log": [],
                "tokens": 0,
            },
            config or {},
        )
        return ProcurementResponse(**state["response"])

    def _build(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("understand", self._guarded(self._understand))
        graph.add_node("investigate", self._guarded(self._investigate))
        graph.add_node("draft", self._guarded(self._draft))
        graph.add_edge(START, "understand")
        graph.add_conditional_edges(
            "understand", lambda s: END if "response" in s else "investigate"
        )
        graph.add_conditional_edges("investigate", self._after_investigate)
        graph.add_conditional_edges("draft", lambda s: END if "response" in s else "draft")
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

        update: dict = {"parsed": parsed.model_dump(), "tokens": tokens}
        current = {**state, **update}
        if not parsed.is_purchase_request:
            update["response"] = self._finish(
                current, "refused", "This assistant only handles purchase requests."
            )
            return update

        known_offices = offices(data)
        known_departments = {b.department for b in data.budgets}
        questions = []
        if not parsed.item:
            questions.append("What would you like to buy?")
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
        if "response" in state:
            return END
        last = state["messages"][-1]
        return "investigate" if last.get("role") == "tool" else "draft"

    def _call_tool(self, name: str, arguments: dict, evidence: dict[str, str]) -> dict:
        if name == "search_policies":
            query = arguments.get("query", "")
            hits = self.knowledge.search(query, k=SEARCH_RESULTS)
            for hit in hits:
                evidence[hit.passage.citation] = f"{hit.passage.heading}: {hit.passage.text}"
            return {
                "sections": [{"citation": h.passage.citation, "text": h.passage.text} for h in hits]
            }
        if name not in READ_ONLY_TOOLS:
            return {"ok": False, "error": f"Tool '{name}' is not available."}
        return self.toolbox.run(name, arguments).model_dump(exclude_none=True)

    def _draft(self, state: AgentState) -> dict:
        evidence = {
            "policy_sections": state["evidence"],
            "tool_results": [
                {"tool": e["tool"], "arguments": e["arguments"], "result": e["result"]}
                for e in state["tool_log"]
                if e["tool"] != "search_policies"
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
            return self._retry_or_fail(current, "Your last reply was not valid JSON in the format.")
        return self._apply_rules(current, draft)

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

    def _bills_in_foreign_currency(self, vendor_id: str) -> bool:
        return any(v.id == vendor_id and v.currency != "INR" for v in self.toolbox.data.vendors)

    @staticmethod
    def _latest_fx_rate(state: AgentState) -> float | None:
        for entry in reversed(state["tool_log"]):
            if entry["tool"] == "get_exchange_rate" and entry["result"].get("ok"):
                return entry["result"]["data"]["rate"]
        return None
