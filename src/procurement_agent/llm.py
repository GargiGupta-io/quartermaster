"""A small client for the AI model. Other code depends only on the `complete` method."""

import json
import logging
import time
from typing import Protocol

import httpx
from pydantic import BaseModel

from procurement_agent.config import Settings

MAX_ATTEMPTS = 4
log = logging.getLogger("procurement_agent")
MAX_WAIT_SECONDS = 60
# Free plans limit tokens per minute, so waiting for the minute to roll over usually works.
MAX_TOTAL_RATE_LIMIT_WAIT = 90


class LLMError(Exception):
    """The model could not produce a usable reply."""


class ModelUnavailable(LLMError):
    """The model is rate limited, down or unreachable, so another model may still work."""


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict


class LLMReply(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = []
    tokens: int = 0
    model: str = ""
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
        """Ask the main model, and the backup model if the main one is unavailable."""
        models = [self._settings.groq_model]
        backup = self._settings.groq_fallback_model
        if backup and backup != models[0]:
            models.append(backup)
        for position, model in enumerate(models):
            has_backup = position < len(models) - 1
            try:
                return self._complete(model, messages, tools, json_mode, has_backup)
            except ModelUnavailable as error:
                if not has_backup:
                    raise
                log.warning("model %s unavailable, using %s: %s", model, models[-1], error)
        raise AssertionError("unreachable")

    def _complete(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        json_mode: bool,
        has_backup: bool,
    ) -> LLMReply:
        body: dict = {"model": model, "messages": messages, "temperature": 0}
        if tools:
            body["tools"] = tools
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        problem = "no attempt made"
        attempt = 0
        waited = 0.0
        rate_limited = 0
        while attempt < MAX_ATTEMPTS:
            try:
                response = self._http.post(
                    f"{self._settings.groq_base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._settings.groq_api_key}"},
                    json=body,
                )
            except httpx.HTTPError as error:
                problem = f"could not reach the model: {error}"
                if has_backup and attempt >= 1:
                    raise ModelUnavailable(problem) from error
                time.sleep(2**attempt)
                attempt += 1
                continue

            if response.status_code == 200:
                reply = _parse_reply(response.json())
                reply.model = model
                return reply

            problem = f"HTTP {response.status_code}: {response.text[:200]}"
            if response.status_code == 429:
                if has_backup:
                    raise ModelUnavailable(f"{model} is rate limited")
                wait = _wait_seconds(response, rate_limited)
                rate_limited += 1
                if waited + wait > MAX_TOTAL_RATE_LIMIT_WAIT:
                    raise ModelUnavailable(f"still rate limited after waiting {waited:.0f} seconds")
                time.sleep(wait)
                waited += wait
                continue  # waiting for the limit to reset does not use up an attempt
            attempt += 1
            if response.status_code >= 500:
                if has_backup and attempt >= 2:
                    raise ModelUnavailable(problem)
                time.sleep(_wait_seconds(response, attempt))
            elif "tool_use_failed" in response.text or "json_validate_failed" in response.text:
                continue  # the model produced malformed output; asking again usually works
            else:
                raise LLMError(problem)
        raise ModelUnavailable(f"gave up after {MAX_ATTEMPTS} attempts: {problem}")


def _wait_seconds(response: httpx.Response, attempt: int) -> float:
    try:
        wait = float(response.headers.get("retry-after", 2**attempt))
    except ValueError:
        wait = 2**attempt
    if wait > MAX_WAIT_SECONDS:
        raise ModelUnavailable(f"model is rate limited for {wait:.0f} seconds")
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
