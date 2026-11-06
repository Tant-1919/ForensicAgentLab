from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from app.main import create_app

PATH = "/internal/agent/report-audit"
BODY = {"specialty": "XJ", "mode": "JSON", "content": '{"samples":[{"sampleNo":"A-1"},{"sampleNo":"A-2"}],"unit":""}'}

def test_auth_scoping_and_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "replay")
    monkeypatch.setenv("AGENT_BRIDGE_TOKEN", "test-key")
    with TestClient(create_app(tmp_path / "db.sqlite")) as client:
        assert client.post(PATH, json=BODY).status_code == 401
        headers = {"X-Agent-Bridge-Token": "test-key"}
        assert client.post(PATH, json={**BODY, "specialty": "BL"}, headers=headers).status_code == 422
        assert client.post(PATH, json={**BODY, "content": "[]"}, headers=headers).status_code == 422
        response = client.post(PATH, json=BODY, headers=headers)
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["agent"]["checks"] == {"check_structure": "checked", "check_identifiers": "issue", "check_unit": "issue", "lookup_review_guidance": "checked"}
        assert all(item["type"] == 2 for item in data["suggestions"])
        assert len(data["suggestions"]) == 2
        assert data["agent"]["knowledge"]
        assert "A-1" not in response.text

def test_file_and_review_modes_do_not_report_unverified_as_clear(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "replay")
    monkeypatch.setenv("AGENT_BRIDGE_TOKEN", "test-key")
    with TestClient(create_app(tmp_path / "db.sqlite")) as client:
        for mode in ("FILE", "TEXT", "TECH"):
            data = client.post(PATH, json={"specialty": "XJ", "mode": mode, "content": "一份没有字段定位的合成报告"},
                               headers={"X-Agent-Bridge-Token": "test-key"}).json()
            assert "仅供人工参考" in data["suggestionText"]
            assert data["agent"]["unverified"] == (0 if mode == "TEXT" else 2)

class Planner:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        assert all("PRIVATE-SNAPSHOT-MARKER" not in str(m.content) for m in messages)
        return AIMessage(content="", tool_calls=[])

def test_live_bridge_only_sends_check_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_BRIDGE_TOKEN", "test-key")
    with TestClient(create_app(tmp_path / "db.sqlite", test_model=Planner())) as client:
        data = client.post(PATH, json={**BODY, "content": '{"unit":"PRIVATE-SNAPSHOT-MARKER"}'},
                           headers={"X-Agent-Bridge-Token": "test-key"}).json()
        assert data["agent"]["mode"] == "live"
        assert data["agent"]["unverified"] == 4
        assert data["suggestions"] == []

