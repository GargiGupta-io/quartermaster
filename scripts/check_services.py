"""Confirm the free outside services work: currency rates, AI replies and AI tool use."""

import json
import sys

import httpx

from procurement_agent.config import Settings

TOOL = {
    "type": "function",
    "function": {
        "name": "get_stock",
        "description": "Look up available stock for a product at an office.",
        "parameters": {
            "type": "object",
            "properties": {"office": {"type": "string"}, "sku": {"type": "string"}},
            "required": ["office", "sku"],
        },
    },
}


def check_currency(settings: Settings) -> str:
    response = httpx.get(
        f"{settings.fx_base_url}/latest", params={"base": "USD", "symbols": "INR"}, timeout=20
    )
    response.raise_for_status()
    body = response.json()
    return f"1 USD = {body['rates']['INR']} INR on {body['date']}"


def check_ai_tool_use(settings: Settings) -> str:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is empty. Add it to .env")
    response = httpx.post(
        f"{settings.groq_base_url}/chat/completions",
        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        json={
            "model": settings.groq_model,
            "messages": [
                {"role": "user", "content": "How many LAP-STD-14 are available in Pune?"}
            ],
            "tools": [TOOL],
            "tool_choice": "required",
        },
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
    call = response.json()["choices"][0]["message"]["tool_calls"][0]["function"]
    arguments = json.loads(call["arguments"])
    if call["name"] != "get_stock" or arguments.get("office") != "Pune":
        raise RuntimeError(f"Unexpected tool call: {call}")
    return f"{settings.groq_model} called {call['name']}({arguments})"


def main() -> int:
    settings = Settings()
    failed = False
    for label, check in (("Currency API", check_currency), ("AI tool use", check_ai_tool_use)):
        try:
            print(f"PASS  {label}: {check(settings)}")
        except (httpx.HTTPError, RuntimeError, KeyError, IndexError, ValueError) as error:
            print(f"FAIL  {label}: {error}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
