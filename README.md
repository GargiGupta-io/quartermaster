# Procurement Agent

An AI workflow that takes a plain-English purchase request, checks company policy, stock,
prices and budget, and prepares a purchase order for a manager to approve.

Work in progress. The company, its policies, vendors and figures are fictional.

## Setup

```
uv sync
cp .env.example .env    # add a free key from https://console.groq.com/keys
uv run python scripts/check_services.py
uv run pytest
```
