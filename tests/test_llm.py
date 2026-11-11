"""The AI client, against simulated replies."""

import json

import httpx
import pytest

from procurement_agent import llm as llm_module
from procurement_agent.config import Settings
from procurement_agent.llm import GroqClient, LLMError

OK = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {"name": "check_budget", "arguments": '{"department":"HR"}'},
                    }
                ],
            }
        }
    ],
    "usage": {"total_tokens": 42},
}


@pytest.fixture(autouse=True)
def no_real_waiting(monkeypatch):
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: None)


def client(*responses, backup="", models_asked=None) -> GroqClient:
    queue = list(responses)

    def handler(request):
        if models_asked is not None:
            models_asked.append(json.loads(request.content)["model"])
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return GroqClient(
        Settings(_env_file=None, groq_api_key="test", groq_fallback_model=backup),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_tool_call_is_parsed():
    reply = client(httpx.Response(200, json=OK)).complete([])
    assert reply.tool_calls[0].arguments == {"department": "HR"}
    assert reply.tokens == 42


@pytest.mark.parametrize(
    "first",
    [
        httpx.Response(429, headers={"retry-after": "1"}),
        httpx.Response(503),
        httpx.Response(400, text='{"error":{"code":"tool_use_failed"}}'),
        httpx.ConnectTimeout("slow"),
    ],
)
def test_temporary_failures_are_retried(first):
    assert client(first, httpx.Response(200, json=OK)).complete([]).tokens == 42


def test_bad_key_is_not_retried():
    with pytest.raises(LLMError, match="401"):
        client(httpx.Response(401, text="bad key")).complete([])


def test_long_rate_limit_gives_up_at_once():
    with pytest.raises(LLMError, match="rate limited"):
        client(httpx.Response(429, headers={"retry-after": "3600"})).complete([])


def test_short_rate_limits_are_waited_out_without_using_attempts():
    waits = [httpx.Response(429, headers={"retry-after": "5"}) for _ in range(6)]
    assert client(*waits, httpx.Response(200, json=OK)).complete([]).tokens == 42


def test_rate_limit_waiting_stops_after_90_seconds():
    waits = [httpx.Response(429, headers={"retry-after": "40"}) for _ in range(3)]
    with pytest.raises(LLMError, match="still rate limited"):
        client(*waits).complete([])


def test_gives_up_after_repeated_failures():
    with pytest.raises(LLMError, match="gave up"):
        client(*[httpx.Response(500) for _ in range(4)]).complete([])


def test_malformed_tool_arguments_are_reported():
    body = {
        "choices": [
            {
                "message": {
                    "tool_calls": [{"id": "1", "function": {"name": "x", "arguments": "{oops"}}]
                }
            }
        ]
    }
    with pytest.raises(LLMError, match="malformed"):
        client(httpx.Response(200, json=body)).complete([])


def test_rate_limit_switches_to_the_backup_model_at_once():
    asked = []
    reply = client(
        httpx.Response(429, headers={"retry-after": "20"}),
        httpx.Response(200, json=OK),
        backup="small-model",
        models_asked=asked,
    ).complete([])
    assert asked == ["openai/gpt-oss-120b", "small-model"]
    assert reply.model == "small-model"


def test_outage_switches_to_the_backup_after_a_retry():
    asked = []
    client(
        httpx.Response(503),
        httpx.Response(503),
        httpx.Response(200, json=OK),
        backup="small-model",
        models_asked=asked,
    ).complete([])
    assert asked == ["openai/gpt-oss-120b", "openai/gpt-oss-120b", "small-model"]


def test_bad_key_does_not_try_the_backup():
    with pytest.raises(LLMError, match="401"):
        client(httpx.Response(401, text="bad key"), backup="small-model").complete([])


def test_both_models_down_gives_a_clear_error():
    responses = [httpx.Response(429)] + [httpx.Response(500) for _ in range(4)]
    with pytest.raises(LLMError, match="gave up"):
        client(*responses, backup="small-model").complete([])


def test_main_model_is_tried_first_again_on_the_next_call():
    asked = []
    groq = client(
        httpx.Response(429),
        httpx.Response(200, json=OK),
        httpx.Response(200, json=OK),
        backup="small-model",
        models_asked=asked,
    )
    groq.complete([])
    assert groq.complete([]).model == "openai/gpt-oss-120b"
