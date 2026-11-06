from pathlib import Path
import json
from fastapi.testclient import TestClient
from app.knowledge import slices, ensure_index, search
from app.main import create_app
from app.mcp_client import call_knowledge

def test_slices_are_bounded_and_offsets_recover_source():
    body = ("第一句说明样本、文书与复核过程。" * 200)
    parts = list(slices(body, limit=200, overlap=40))
    assert parts[0][0] == 0 and parts[-1][1] == len(body)
    for a, b, text in parts:
        assert text == body[a:b] and b-a <= 200
    assert all(left[1]-right[0] == 40 for left,right in zip(parts,parts[1:]))

def test_private_corpus_is_filtered_by_scope_date_and_public_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_KNOWLEDGE_DB", str(tmp_path/"kb.sqlite"))
    inbox=tmp_path/"inbox"
    inbox.mkdir()
    monkeypatch.setenv("AGENT_KNOWLEDGE_IMPORT", str(inbox))
    (inbox/"source.md").write_text("# 实验条款\n这里包含 solemarker 术语。",encoding="utf-8")
    (inbox/"source.json").write_text(json.dumps({"file":"source.md","doc_id":"TEST","title":"演示测试","version":"V2",
        "effective_from":"2025-01-01","effective_to":"2026-12-31","specialty":"XJ"}),encoding="utf-8")
    ensure_index()
    hit=search("solemarker","XJ","2026-06-01")[0]
    assert hit["section"]=="实验条款" and hit["kind"]=="imported_text" and hit["body"].startswith("# 实验条款")
    assert not search("solemarker","BL","2026-06-01")
    assert not search("solemarker","XJ","2024-06-01")
    assert not search("solemarker","XJ","2026-06-01", public_only=True)
    assert search("solemarker","XJ","2026-06-01")[0]["id"]==hit["id"]

def test_mcp_protocol_and_demo_audit(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_KNOWLEDGE_DB",str(tmp_path/"kb.sqlite"))
    monkeypatch.setenv("AGENT_MODE","replay")
    assert call_knowledge("knowledge_index_info")["sources"] >= 3
    with TestClient(create_app(tmp_path/"audit.sqlite")) as client:
        results=client.get("/api/knowledge/search",params={"q":"司法鉴定 内部复核","top_k":2}).json()
        assert results["hits"] and all(hit["url"].startswith("https://") for hit in results["hits"])
        result=client.post("/api/audit-runs",json={"case_id":"LAB-005","version":1}).json()
        assert result["status"]=="review_pending" and result["knowledge"]
        trace=client.get("/api/audit-runs/"+result["run_id"]+"/trace").json()
        events=[x for x in trace["events"] if x.get("transport")=="mcp/stdio"]
        assert events and events[0]["citations"] == [h["id"] for h in result["knowledge"]]

