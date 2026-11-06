import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from threading import RLock
from typing import Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from langsmith import tracing_context
from app.mcp_client import call_knowledge
from pydantic import BaseModel, ConfigDict, Field

from app.domain import CASES
from app.graph import build_agent_graph
from app.legacy_bridge import build_bridge_graph, router as bridge_router

ROOT = Path(__file__).resolve().parent.parent

class StartAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(max_length=32)
    version: int = Field(ge=1)

class ReviewAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision_id: UUID
    version: int = Field(ge=1)
    action: Literal["accept", "reject"]
    note: str = Field(default="", max_length=300)

def create_app(db_path: Path | None = None, test_model=None):
    # 入口文件只负责接请求、装配模型/图、返回接口数据。检查规则放在 domain.py。
    # db_path 与 test_model 可由测试注入，所以测试无需改本地数据库或连接模型。
    path = db_path or Path(os.getenv("AGENT_DB_PATH", str(ROOT / "data" / "checkpoints.sqlite")))
    lock = RLock()

    @asynccontextmanager
    async def lifespan(api):
        # FastAPI 每个进程启动时只编译一次工作流；关闭时释放 SQLite 连接。
        # live 才创建模型客户端，replay 模式仍使用同一套图和工具。
        mode = "live" if test_model else os.getenv("AGENT_MODE", "replay").lower()
        if mode not in {"replay", "live"}:
            raise RuntimeError("AGENT_MODE must be replay or live")
        model = test_model
        if mode == "live" and model is None:
            from langchain_openai import ChatOpenAI
            from app.model_settings import live_model_settings
            model = ChatOpenAI(**live_model_settings(), timeout=10, max_retries=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, check_same_thread=False)
        try:
            api.state.graph = build_agent_graph(model, SqliteSaver(connection))
            api.state.mode = mode
            # Bridge review is ephemeral; the demo workspace uses SQLite.
            api.state.bridge = build_bridge_graph(model)
            yield
        finally:
            connection.close()

    api = FastAPI(title="司法鉴定报告质控 Agent", version="0.1.0", lifespan=lifespan)
    api.include_router(bridge_router)
    api.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")

    def config(run_id, phase="audit"):
        return {"configurable": {"thread_id": str(run_id)}, "recursion_limit": 24,
                "run_name": "forensic-synthetic-" + phase, "metadata": {"audit_run_id": str(run_id)},
                "tags": ["forensic-lab", "synthetic"]}

    def demo_tracing():
        # Never allow a global tracing flag alone to export data. The Java
        # bridge independently disables tracing even when this is enabled.
        return tracing_context(enabled=os.getenv("AGENT_LANGSMITH_DEMO") == "true" and bool(os.getenv("LANGSMITH_API_KEY")))

    def snapshot(run_id):
        # SQLite 的 thread_id 用于恢复任务；它不是用户身份或鉴权凭据。
        current = api.state.graph.get_state(config(run_id))
        if not current.values.get("case_id"):
            raise HTTPException(404, "未找到审核任务")
        return current

    def view(run_id, state):
        keys = ("case_id", "version", "scope", "report", "mode", "status", "checks", "findings",
                "evidence", "trace", "knowledge", "guidance", "decision", "model_calls", "tool_calls", "stop_reason")
        return {"run_id": str(run_id), **{key: state[key] for key in keys if key in state}}

    @api.get("/", include_in_schema=False)
    def home():
        return FileResponse(ROOT / "web" / "index.html")

    @api.get("/api/health")
    def health():
        return {"status": "ok", "mode": api.state.mode}

    @api.get("/api/cases")
    def cases():
        return {"mode": api.state.mode, "cases": [
            {"id": case_id, **{key: case[key] for key in ("title", "version", "scope", "report")}}
            for case_id, case in CASES.items()
        ]}

    @api.post("/api/audit-runs", status_code=201)
    def start(body: StartAudit):
        # 当前是同步的本地演示：浏览器等图走到复核节点，再拿到一份完整结果。
        case = CASES.get(body.case_id)
        if not case:
            raise HTTPException(404, "未找到演示案件")
        if body.version != case["version"]:
            raise HTTPException(409, "报告版本已变化")
        run_id = uuid4()
        with lock, demo_tracing():
            api.state.graph.invoke({"case_id": body.case_id, "mode": api.state.mode}, config(run_id))
            return view(run_id, snapshot(run_id).values)

    @api.get("/api/audit-runs/{run_id}")
    def get_run(run_id: UUID):
        with lock:
            return view(run_id, snapshot(run_id).values)

    @api.get("/api/audit-runs/{run_id}/trace")
    def run_trace(run_id: UUID):
        with lock:
            state = snapshot(run_id).values
            return {"run_id":str(run_id), "events":state.get("trace", []),
                    "model_calls":state.get("model_calls", 0), "tool_calls":state.get("tool_calls", 0)}

    def knowledge_call(tool, arguments=None):
        try:
            return call_knowledge(tool, arguments)
        except Exception:
            raise HTTPException(503, "MCP 知识库暂不可用，请检查依赖、索引和导入清单") from None

    @api.get("/api/knowledge/stats")
    def knowledge_stats():
        return knowledge_call("knowledge_index_info")

    @api.get("/api/knowledge/search")
    def knowledge_search(q: str = Query(min_length=1, max_length=180),
                         specialty: Literal["XJ","CJ","LC","BL"] = "XJ",
                         top_k: int = Query(4, ge=1, le=8), as_of: date | None = None):
        return knowledge_call("search_forensic_knowledge", {"query":q, "specialty":specialty, "top_k":top_k,
                                                            "as_of": as_of.isoformat() if as_of else None})

    @api.post("/api/audit-runs/{run_id}/review")
    def submit_review(run_id: UUID, body: ReviewAudit):
        with lock:
            current = snapshot(run_id)
            state = current.values
            decision = body.model_dump(mode="json")
            if state["version"] != body.version or CASES[state["case_id"]]["version"] != body.version:
                raise HTTPException(409, "报告版本已变化，请重新审核")
            if state.get("decision") == decision:
                return view(run_id, state)
            if state["status"] != "review_pending" or not current.interrupts:
                raise HTTPException(409, "该任务不在待复核状态")
            # Review notes are user-entered text and are not sent to LangSmith.
            with tracing_context(enabled=False):
                api.state.graph.invoke(Command(resume=decision), config(run_id, "review"))
            return view(run_id, snapshot(run_id).values)

    return api

app = create_app()

