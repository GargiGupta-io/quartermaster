"""The web service, with a scripted model and temporary storage."""

import pytest
from fastapi.testclient import TestClient

from procurement_agent.api import Service, create_app
from procurement_agent.knowledge import KnowledgeBase
from test_agent import DONE, PARSED, ScriptedLLM, as_json, draft


@pytest.fixture(scope="module")
def knowledge(tmp_path_factory):
    return KnowledgeBase(index_dir=tmp_path_factory.mktemp("index"))


@pytest.fixture
def setup(knowledge, tmp_path):
    llm = ScriptedLLM(as_json(PARSED), DONE, draft())
    with TestClient(create_app(Service(tmp_path, llm, knowledge))) as client:
        yield client, llm, tmp_path


def submit(client):
    response = client.post("/requests", json={"text": "20 laptops for Pune engineering"})
    assert response.status_code == 201
    return response.json()


def decide(client, request_id, role, action="approve", **extra):
    return client.post(
        f"/requests/{request_id}/decisions",
        json={"role": role, "action": action, "name": "A Person"} | extra,
    )


def test_page_loads(setup):
    client, _, _ = setup
    page = client.get("/")
    assert page.status_code == 200
    assert "Procurement Agent" in page.text


def test_full_journey_to_a_purchase_order(setup):
    client, _, _ = setup
    created = submit(client)
    assert created["response"]["status"] == "ready_for_approval"

    listed = client.get("/requests").json()
    assert listed[0]["pending_approvers"] == ["Department Head", "Finance Controller"]

    assert decide(client, created["id"], "Department Head").status_code == 200
    final = decide(client, created["id"], "Finance Controller").json()
    assert final["response"]["status"] == "ordered"
    assert final["response"]["purchase_order"]["total_inr"] == 1_359_360
    assert client.get(f"/requests/{created['id']}").json()["response"]["status"] == "ordered"


def test_finished_request_returns_a_conflict(setup):
    client, _, _ = setup
    created = submit(client)
    decide(client, created["id"], "Department Head", "reject")
    assert decide(client, created["id"], "Department Head").status_code == 409


def test_unknown_request_is_not_found(setup):
    client, _, _ = setup
    assert client.get("/requests/nope").status_code == 404
    assert decide(client, "nope", "CFO").status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"role": "CFO", "action": "bribe", "name": "A"},
        {"role": "CFO", "action": "approve", "name": ""},
        {"action": "approve", "name": "A"},
    ],
)
def test_malformed_decisions_are_rejected(setup, body):
    client, _, _ = setup
    created = submit(client)
    response = client.post(f"/requests/{created['id']}/decisions", json=body)
    assert response.status_code == 422


@pytest.mark.parametrize("text", ["", "x" * 2001])
def test_bad_request_text_is_rejected(setup, text):
    client, _, _ = setup
    assert client.post("/requests", json={"text": text}).status_code == 422


def test_requests_survive_a_service_restart(setup, knowledge):
    client, _, data_dir = setup
    created = submit(client)
    with TestClient(create_app(Service(data_dir, ScriptedLLM(), knowledge))) as restarted:
        waiting = restarted.get(f"/requests/{created['id']}").json()
        assert waiting["response"]["pending_approvers"]
        assert decide(restarted, created["id"], "Department Head").status_code == 200
        final = decide(restarted, created["id"], "Finance Controller").json()
        assert final["response"]["status"] == "ordered"
