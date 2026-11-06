"""Sidecar for the copied Java report workbench.

The Java API keeps ownership of login, report files and review writes. This
endpoint receives a read-only snapshot; it never opens a Java database or file.
Even in live mode, the planner only receives check names and result counts.
"""
from hmac import compare_digest
import json
import os
import re
import time
from app.mcp_client import call_knowledge
from typing import Literal, TypedDict

from fastapi import APIRouter, Header, HTTPException, Request
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import START, END, StateGraph
from langsmith import tracing_context
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(prefix="/internal/agent", tags=["Java bridge"])
ID_FIELDS = {"sampleno", "samplenumber", "specimenno", "样本编号", "检材编号"}
TOOL_DESCRIPTIONS = {
    "check_structure": "Check whether the snapshot has readable content.",
    "check_identifiers": "Look for multiple sample identifiers within the supplied snapshot.",
    "check_unit": "Check whether an explicit unit field is blank.",
    "lookup_review_guidance": "Use MCP to retrieve relevant local forensic review sources.",
}
# Text proofreading is intentionally limited to structure. Missing anchors do
# not imply a report is correct. A full review requires other authorized sources.
SCOPE = {
    "JSON": ["check_structure", "check_identifiers", "check_unit", "lookup_review_guidance"],
    "FILE": ["check_structure", "check_identifiers", "check_unit", "lookup_review_guidance"],
    "TEXT": ["check_structure"],
    "TECH": ["check_structure", "check_identifiers", "check_unit", "lookup_review_guidance"],
}

class Snapshot(BaseModel):
    # Java 已完成报告读取。这里收的是本次只读快照，不是路径、账号或数据库连接。
    model_config = ConfigDict(extra="forbid")
    mode: Literal["JSON", "FILE", "TEXT", "TECH"]
    specialty: Literal["XJ"]
    content: str = Field(max_length=120_000)

class State(TypedDict, total=False):
    snapshot: dict  # Java 已经加载的报告快照，只在本次请求内保留。
    pending: list[str]  # 剩余的检查工具；它同时是本次任务的权限清单。
    trace: list[dict]
    findings: list[dict]
    checks: dict
    messages: list
    selections: list
    guidance: list[dict]
    turns: int
    stop_reason: str

def walk_fields(value):
    """Iterate report form fields without building another tree in memory."""
    queue = [value]
    while queue:
        current = queue.pop()
        if isinstance(current, dict):
            for key, child in current.items():
                yield key, child
                if isinstance(child, (list, dict)):
                    queue.append(child)
        elif isinstance(current, list):
            queue.extend(child for child in current if isinstance(child, (list, dict)))

def tool_result(name, snapshot):
    """Return fixed messages; no unchecked report text enters browser HTML."""
    content = snapshot["content"].strip()
    form = json.loads(content) if snapshot["mode"] == "JSON" else {}
    fields = list(walk_fields(form))
    if name == "check_structure":
        return ("checked", []) if (form if snapshot["mode"] == "JSON" else content) else (
            "issue", [{"type": 2, "title": "报告内容为空", "text": "请补全可读取的报告内容后重新审核。"}])
    if name == "check_identifiers":
        # Check repeated values of the same field class, not report number vs.
        # sample number. Multiple specimens can be valid, so ask a human.
        groups = {}
        for label, value in fields:
            label = re.sub(r"[^\w]", "", label).casefold().replace("_", "")
            if label in ID_FIELDS and isinstance(value, str) and value.strip():
                groups.setdefault(label, set()).add(value.strip().casefold())
        if snapshot["mode"] != "JSON":
            for label, value in re.findall(r"(样本编号|检材编号)\s*[:：]\s*([A-Za-z0-9_-]{1,48})", content):
                groups.setdefault(label, set()).add(value.casefold())
        if not groups:
            return "unverified", []
        if any(len(values) > 1 for values in groups.values()):
            return "issue", [{"type": 2, "title": "存在多个样本编号，待复核", "text": "请人工区分多检材情形和同一检材编号冲突。当前仅比对报告内部字段。"}]
        return "checked", []
    units = [v for k, v in fields if k.casefold() in {"unit", "resultunit", "计量单位", "单位"}]
    if snapshot["mode"] != "JSON":
        units = re.findall(r"(?m)^\s*(?:计量单位|单位)[ \t]*[:：][ \t]*([^\r\n]*)", content)
    if not units:
        return "unverified", []
    if any(value is None or not str(value).strip() for value in units):
        return "issue", [{"type": 2, "title": "计量单位栏为空", "text": "请查阅原始记录并人工补填。"}]
    return "checked", []

def build_bridge_graph(model=None):
    """Java 快照审核独立编排，不写检查点；报告正文不交给模型。"""
    specs = [
        {"type": "function", "function": {"name": name, "description": desc,
          "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}}
        for name, desc in TOOL_DESCRIPTIONS.items()
    ]
    bound = model.bind_tools(specs) if model else None

    def decide(state):
        # 决策节点只发出调用意图，不在这里读取报告或写库。
        remaining = state["pending"]
        if not remaining or state["turns"] >= 4:
            return {"selections": [], "stop_reason": "partial" if remaining else "complete"}
        if not bound:
            # Offline policy follows the chosen audit scope; no model is called.
            return {"selections": [{"name": remaining[0], "args": {}, "id": "replay"}]}
        try:
            reply = bound.invoke([
                SystemMessage(content="Select the pending checks using tool calls with no arguments. You may request several pending checks at once. Do not repeat completed tools and do not generate audit findings. Stop once no checks are pending."),
                HumanMessage(content=json.dumps({"pending": remaining, "completed_check_status": state["checks"]})),
                *state["messages"],
            ])
            return {"selections": reply.tool_calls[:4], "messages": [*state["messages"], reply]}
        except Exception:
            # Fail closed. Remaining checks are returned as unverified.
            return {"selections": [], "stop_reason": "model_error"}

    def execute(state):
        # 先复制容器，再返回状态更新。工具执行权始终在 Python 应用。
        remaining, findings = list(state["pending"]), list(state["findings"])
        checks, trace, messages = dict(state["checks"]), list(state["trace"]), list(state["messages"])
        guidance = list(state.get("guidance", []))
        for call in state["selections"]:
            started = time.perf_counter()
            name = call["name"]
            if name not in remaining or call["args"]:
                status = "denied"
            else:
                if name == "lookup_review_guidance":
                    try:
                        hits = call_knowledge("search_forensic_knowledge",
                            {"query":"鉴定程序 鉴定意见 内部复核 检验记录", "specialty":"XJ", "top_k":3})["hits"]
                        guidance.extend(hits)
                        status = "checked" if hits else "unverified"
                    except Exception:
                        status = "unverified"
                else:
                    status, result = tool_result(name, state["snapshot"])
                    findings.extend(result)
                checks[name] = status
                remaining.remove(name)
            trace.append({"tool": name if name in TOOL_DESCRIPTIONS else "unknown", "status": status,
                          "transport": "mcp/stdio" if name == "lookup_review_guidance" else "local",
                          "call_id": call["id"], "elapsed_ms": round((time.perf_counter()-started)*1000)})
            if bound:
                messages.append(ToolMessage(content=json.dumps({"status": status}), tool_call_id=call["id"]))
        return {"pending": remaining, "findings": findings, "checks": checks, "trace": trace,
                "turns": state["turns"] + 1, "messages": messages, "guidance": guidance}

    graph = StateGraph(State)
    graph.add_node("decide", decide)
    graph.add_node("execute", execute)
    graph.add_edge(START, "decide")
    # 条件边决定继续调用工具还是结束本次取证。
    graph.add_conditional_edges("decide", lambda state: "execute" if state["selections"] else END)
    graph.add_edge("execute", "decide")
    return graph.compile()

@router.post("/report-audit")
def report_audit(body: Snapshot, request: Request,
                 token: str = Header(default="", alias="X-Agent-Bridge-Token")):
    # Loopback plus a shared secret prevents an accidental open bridge. This is
    # transport protection; the Java layer remains the application trust boundary.
    expected = os.getenv("AGENT_BRIDGE_TOKEN", "")
    if not expected:
        raise HTTPException(503, "Java bridge disabled")
    if not compare_digest(token.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(401, "Invalid bridge credentials")
    if body.mode == "JSON":
        try:
            data = json.loads(body.content)
        except (ValueError, RecursionError):
            raise HTTPException(422, "Expected a JSON report snapshot")
        if not isinstance(data, dict):
            raise HTTPException(422, "Expected an object for the report snapshot")
    # Even if external tracing was enabled in the shell, never export a report
    # snapshot as a graph input, intermediate node value, or diagnostic trace.
    with tracing_context(enabled=False):
        result = request.app.state.bridge.invoke({
            "snapshot": body.model_dump(), "pending": SCOPE[body.mode], "trace": [],
            "findings": [], "checks": {}, "messages": [], "turns": 0, "stop_reason": "complete",
        }, {"recursion_limit": 12})
    checks = {name: result["checks"].get(name, "unverified") for name in SCOPE[body.mode]}
    unverified = sum(status == "unverified" for status in checks.values())
    text = [item["title"] + "：" + item["text"] for item in result["findings"]]
    if not text:
        text = ["在已完成的本地检查中未发现需提示的问题。"]
    text.append(f"本次检查未核实 {unverified} 项；仅供人工参考，不代表完成专业复核。")
    return {"suggestions": result["findings"], "suggestionText": "\n".join(text),
            "agent": {"engine": "langgraph", "mode": request.app.state.mode, "checks": checks,
                      "trace": result["trace"], "knowledge": result.get("guidance", []), "unverified": unverified, "stop_reason": result["stop_reason"]}}

