"""A small client for the AI model. Other code depends only on the `complete` method."""

import json
import time
from typing import Protocol

import httpx
from pydantic import BaseModel

from procurement_agent.config import Settings

MAX_ATTEMPTS = 4
MAX_WAIT_SECONDS = 60


class LLMError(Exception):
    """The model could not produce a usable reply."""


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict


class LLMReply(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = []
    tokens: int = 0
    raw_message: dict = {}


class LLMClient(Protocol):
    def complete(
        self, messages: list[dict], tools: list[dict] | None = None, json_mode: bool = False
    ) -> LLMReply: ...


class GroqClient:
    def __init__(self, settings: Settings | None = None, http: httpx.Client | None = None):
        self._settings = settings or Settings()
        self._http = http or httpx.Client(timeout=60)

    @property
    def configured(self) -> bool:
        """Whether a key is set. Checked without calling the service, so health checks are free."""
        return bool(self._settings.groq_api_key)

    def complete(
        self, messages: list[dict], tools: list[dict] | None = None, json_mode: bool = False
    ) -> LLMReply:
        body: dict = {"model": self._settings.groq_model, "messages": messages, "temperature": 0}
        if tools:
            body["tools"] = tools
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        problem = "no attempt made"
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = self._http.post(
                    f"{self._settings.groq_base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._settings.groq_api_key}"},
                    json=body,
                )
            except httpx.HTTPError as error:
                problem = f"could not reach the model: {error}"
                time.sleep(2**attempt)
                continue

            if response.status_code == 200:
                return _parse_reply(response.json())

            problem = f"HTTP {response.status_code}: {response.text[:200]}"
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(_wait_seconds(response, attempt))
            elif "tool_use_failed" in response.text or "json_validate_failed" in response.text:
                continue  # the model produced malformed output; asking again usually works
            else:
                raise LLMError(problem)
        raise LLMError(f"gave up after {MAX_ATTEMPTS} attempts: {problem}")


def _wait_seconds(response: httpx.Response, attempt: int) -> float:
    try:
        wait = float(response.headers.get("retry-after", 2**attempt))
    except ValueError:
        wait = 2**attempt
    if wait > MAX_WAIT_SECONDS:
        raise LLMError(f"model is rate limited for {wait:.0f} seconds")
    return wait


def _parse_reply(body: dict) -> LLMReply:
    try:
        message = body["choices"][0]["message"]
        calls = [
            ToolCall(
                id=call["id"],
                name=call["function"]["name"],
                arguments=json.loads(call["function"]["arguments"] or "{}"),
            )
            for call in message.get("tool_calls") or []
        ]
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise LLMError(f"model reply was malformed: {error}") from error
    return LLMReply(
        content=message.get("content"),
        tool_calls=calls,
        tokens=body.get("usage", {}).get("total_tokens", 0),
        raw_message={k: v for k, v in message.items() if k in ("role", "content", "tool_calls")},
    )
