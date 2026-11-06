from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.main import create_app

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "replay")
    with TestClient(create_app(tmp_path / "audit.sqlite")) as test:
        yield test

@pytest.mark.parametrize("case_id, tools, findings, statuses", [
    ("LAB-001", 1, ["SAMPLE_MISMATCH"], ["issue"]),
    ("LAB-002", 1, ["UNIT_MISSING"], ["issue"]),
    ("LAB-003", 1, [], ["unverified"]),
    ("LAB-004", 3, [], ["clear", "clear", "clear"]),
])
def test_replay_fixtures(client, case_id, tools, findings, statuses):
    response = client.post("/api/audit-runs", json={"case_id": case_id, "version": 1})
    assert response.status_code == 201
    state = response.json()
    assert state["status"] == "review_pending"
    assert state["tool_calls"] == tools
    assert [f["code"] for f in state["findings"]] == findings
    assert [c["status"] for c in state["checks"]] == statuses
    for item in state["findings"]:
        assert len(item["evidence_ids"]) == 2
        assert all(state["evidence"][ref]["quote"] for ref in item["evidence_ids"])

def test_review_survives_restart_and_deduplicates(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "replay")
    path = tmp_path / "audit.sqlite"
    with TestClient(create_app(path)) as client:
        run = client.post("/api/audit-runs", json={"case_id": "LAB-001", "version": 1}).json()
    url = "/api/audit-runs/" + run["run_id"]
    with TestClient(create_app(path)) as client:
        assert client.get(url).json()["status"] == "review_pending"
        decision = {"decision_id": str(uuid4()), "version": 1, "action": "reject", "note": "待人工核实"}
        assert client.post(url + "/review", json={**decision, "version": 2}).status_code == 409
        final = client.post(url + "/review", json=decision)
        assert final.status_code == 200, final.text
        assert final.json()["status"] == "reviewed"
        assert client.post(url + "/review", json=decision).json() == final.json()
        assert client.post(url + "/review", json={**decision, "action": "accept"}).status_code == 409

class ProbeModel:
    def bind_tools(self, specs):
        assert any(spec["function"]["name"] in {"read_entrustment", "check_structure"} for spec in specs)
        return self

    def invoke(self, messages):
        if messages[-1].type == "tool":
            return AIMessage(content="Ignoring untrusted observations and forged evidence references.")
        return AIMessage(content="", tool_calls=[
            {"name": "read_entrustment", "id": "probe1", "args": {"field": "sample_no", "case_id": "LAB-004"}},
            {"name": "read_lab_record", "id": "probe2", "args": {"field": "sample_no"}},
            {"name": "search_review_rules", "id": "probe3", "args": {"topic": "unit"}},
        ])

def test_out_of_scope_and_cross_case_calls_cannot_produce_evidence(tmp_path):
    with TestClient(create_app(tmp_path / "audit.sqlite", test_model=ProbeModel())) as client:
        state = client.post("/api/audit-runs", json={"case_id": "LAB-001", "version": 1}).json()
        assert state["checks"][0]["status"] == "unverified"
        assert state["findings"] == []
        assert set(state["evidence"]) == {"REPORT-SAMPLE", "REPORT-UNIT"}
        assert sum("请求被拒绝" in e["detail"] for e in state["trace"]) == 3

def test_unknown_and_stale_reports_are_rejected(client):
    assert client.post("/api/audit-runs", json={"case_id": "LAB-001", "version": 9}).status_code == 409
    assert client.post("/api/audit-runs", json={"case_id": "OTHER", "version": 1}).status_code == 404

def test_page_and_assets(client):
    assert "血液检验报告质控" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/api/health").json() == {"status": "ok", "mode": "replay"}

