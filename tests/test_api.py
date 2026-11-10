"""The web service, with a scripted model and temporary storage."""

import json

import pytest
from fastapi.testclient import TestClient

from procurement_agent.api import Service, create_app
from procurement_agent.knowledge import KnowledgeBase
from test_agent import DONE, PARSED, ScriptedLLM, as_json, draft

KEY = "test-access-key"


@pytest.fixture(scope="module")
def knowledge(tmp_path_factory):
    return KnowledgeBase(index_dir=tmp_path_factory.mktemp("index"))


@pytest.fixture
def setup(knowledge, tmp_path):
    llm = ScriptedLLM(as_json(PARSED), DONE, draft())
    with TestClient(create_app(Service(tmp_path, llm, knowledge), api_key="")) as client:
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


@pytest.fixture
def locked(knowledge, tmp_path):
    llm = ScriptedLLM(as_json(PARSED), DONE, draft())
    with TestClient(create_app(Service(tmp_path, llm, knowledge), api_key=KEY)) as client:
        yield client


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}, {"X-API-Key": ""}])
def test_requests_need_the_access_key(locked, headers):
    for call in (
        lambda: locked.post("/requests", json={"text": "20 laptops"}, headers=headers),
        lambda: locked.get("/requests", headers=headers),
        lambda: locked.get("/requests/some-id", headers=headers),
        lambda: locked.post("/requests/some-id/decisions", json={}, headers=headers),
    ):
        assert call().status_code == 401


def test_the_right_key_is_accepted(locked):
    response = locked.post(
        "/requests", json={"text": "20 laptops for Pune"}, headers={"X-API-Key": KEY}
    )
    assert response.status_code == 201


def test_page_and_health_check_stay_open(locked):
    assert locked.get("/").status_code == 200
    assert locked.get("/health").status_code == 200


def test_health_reports_what_it_checked(setup):
    client, _, _ = setup
    report = client.get("/health").json()
    assert report["status"] == "ok"
    assert report["database"] is True
    assert report["documents_indexed"] > 50
    assert report["ai_key_configured"] is True


def test_health_fails_without_an_ai_key(knowledge, tmp_path):
    llm = ScriptedLLM()
    llm.configured = False
    with TestClient(create_app(Service(tmp_path, llm, knowledge), api_key="")) as client:
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["ai_key_configured"] is False


def test_finished_requests_are_logged_without_their_text(setup, caplog):
    client, _, _ = setup
    caplog.set_level("INFO", logger="procurement_agent")
    created = submit(client)
    finished = [json.loads(r.message) for r in caplog.records if '"request_finished"' in r.message]
    assert finished[0]["request_id"] == created["id"]
    assert finished[0]["status"] == "ready_for_approval"
    assert finished[0]["tokens"] > 0
    assert "laptops" not in caplog.text
