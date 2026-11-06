"""Synthetic fixture repository and deterministic evidence rules."""
from typing import Literal
from pydantic import BaseModel, ConfigDict
from app.mcp_client import call_knowledge

KNOWLEDGE_TOPICS = {
    "review": "鉴定程序 鉴定意见 人工复核 签名",
    "record": "检验记录 数据 检材 样本编号 计量单位"
}

CASES = {
    "LAB-001": {"title": "样本编号不一致", "report": {"sample_no": "DEMO-014", "unit": "mg/100mL"}, "entrustment": "DEMO-013", "lab": "DEMO-014", "scope": ["sample"], "version": 1},
    "LAB-002": {"title": "报告缺少计量单位", "report": {"sample_no": "DEMO-022", "unit": ""}, "entrustment": "DEMO-022", "lab": "DEMO-022", "scope": ["template"], "version": 1},
    "LAB-003": {"title": "检测记录缺失", "report": {"sample_no": "DEMO-032", "unit": "mg/100mL"}, "entrustment": "DEMO-032", "lab": None, "scope": ["lab"], "version": 1},
    "LAB-004": {"title": "未发现检查项异常", "report": {"sample_no": "DEMO-042", "unit": "mg/100mL"}, "entrustment": "DEMO-042", "lab": "DEMO-042", "scope": ["sample", "lab", "template"], "version": 1},
    "LAB-005": {"title": "MCP / RAG · 查找复核依据", "report": {"sample_no": "DEMO-050", "unit": "mg/100mL"}, "entrustment": "DEMO-050", "lab": "DEMO-050", "scope": ["knowledge"], "version": 1},
}

RULES = (
    {"id": "SOP-01", "source": "演示质控说明 v1 · 第 1 条", "quote": "检测结果栏应填写计量单位。", "aliases": ("单位", "计量", "unit", "结果栏")},
    {"id": "SOP-02", "source": "演示质控说明 v1 · 第 2 条", "quote": "文书应标明报告版本。", "aliases": ("版本", "version")},
)

class ScopedArgs(BaseModel):
    # extra=forbid 拒绝模型擅自增加参数，例如试图传入另一个案件 ID。
    model_config = ConfigDict(extra="forbid")

class EntrustmentArgs(ScopedArgs):
    field: Literal["sample_no"]

class LabArgs(ScopedArgs):
    field: Literal["sample_no"]

class RuleArgs(ScopedArgs):
    topic: Literal["unit"]

class KnowledgeArgs(ScopedArgs):
    topic: Literal["review", "record"]

TOOLS = {
    "search_forensic_knowledge": (KnowledgeArgs, "通过 MCP 检索司法鉴定知识库；review 查复核程序，record 查检验记录"),
    "read_entrustment": (EntrustmentArgs, "读取当前案件委托书中的样本编号"),
    "read_lab_record": (LabArgs, "读取当前案件检测图谱中的样本编号"),
    "search_review_rules": (RuleArgs, "检索演示质控说明中的计量单位要求"),
}

CHECKS = {
    "knowledge": ("search_forensic_knowledge", "司法知识检索", "KNOWLEDGE"),
    "sample": ("read_entrustment", "委托书样本编号", "SAMPLE_MISMATCH"),
    "lab": ("read_lab_record", "图谱样本编号", "LAB_MISMATCH"),
    "template": ("search_review_rules", "模板计量单位", "UNIT_MISSING"),
}

def tool_specs():
    return [
        {"type": "function", "function": {"name": name, "description": desc, "parameters": schema.model_json_schema()}}
        for name, (schema, desc) in TOOLS.items()
    ]

def record(evidence_id, source, quote):
    return {"id": evidence_id, "source": source, "quote": quote}

def execute_tool(case_id, name, args):
    # case_id comes from the server-owned run state, never from model arguments.
    TOOLS[name][0].model_validate(args)
    case = CASES[case_id]
    if name == "search_forensic_knowledge":
        hits = call_knowledge(name, {"query": KNOWLEDGE_TOPICS[args["topic"]], "specialty": "XJ", "top_k": 4, "public_only": True})["hits"]
        evidence = [dict(h, source=h["title"] + " · " + h["section"], quote=h["body"]) for h in hits]
        return {"status": "found" if hits else "missing", "evidence": evidence}
    if name == "search_review_rules":
        evidence = [
            record(rule["id"], rule["source"], rule["quote"])
            for rule in RULES if args["topic"] in rule["aliases"]
        ]
        return {"status": "found" if evidence else "missing", "evidence": evidence}
    key = "entrustment" if name == "read_entrustment" else "lab"
    value = case[key]
    if value is None:
        return {"status": "missing", "evidence": []}
    title = "委托书 · 样本信息" if key == "entrustment" else "检测图谱 · 样本信息"
    return {"status": "found", "value": value, "evidence": [record(key.upper(), title, f"样本编号：{value}")]}

def evaluate(state):
    # 将原报告与已取证字段比对，结果分为异常、已核对、无法核实。
    # 无证据不能输出“已核对”；问题所引用的 ID 由程序生成。
    report, observations = state["report"], state.get("observations", {})
    evidence = {
        "REPORT-SAMPLE": record("REPORT-SAMPLE", "报告草稿 · 样本信息", f"样本编号：{report['sample_no']}"),
        "REPORT-UNIT": record("REPORT-UNIT", "报告草稿 · 检测结果", f"计量单位：{report['unit'] or '（空白）'}"),
    }
    findings, checks, knowledge = [], [], []
    for check in state["scope"]:
        tool, title, issue = CHECKS[check]
        observation = observations.get(tool, {})
        sources = observation.get("evidence", [])
        if observation.get("status") != "found" or not sources:
            checks.append({"title": title, "status": "unverified", "detail": "没有取得所需证据"})
            continue
        evidence.update({source["id"]: source for source in sources})
        if check == "knowledge":
            knowledge.extend(sources)
            checks.append({"title": title, "status": "clear", "detail": f"已通过 MCP 取得 {len(sources)} 个知识切片"})
            continue
        if check == "template":
            mismatch = not report["unit"]
            detail = "报告的检测结果栏未填写计量单位。"
            source_id = "REPORT-UNIT"
            advice = "请查阅原始记录，人工补充正确单位。"
        else:
            value = observation["value"]
            mismatch = value != report["sample_no"]
            detail = f"报告编号为 {report['sample_no']}，{title}为 {value}。"
            source_id = "REPORT-SAMPLE"
            advice = "请对照原件核实编号及样本对应关系。"
        checks.append({"title": title, "status": "issue" if mismatch else "clear", "detail": detail if mismatch else "已取证并完成比对"})
        if mismatch:
            ids = [source_id, sources[0]["id"]]
            findings.append({"code": issue, "title": title + "待核对", "detail": detail, "suggestion": advice, "evidence_ids": ids})
    return {"findings": findings, "checks": checks, "evidence": evidence, "knowledge": knowledge}

