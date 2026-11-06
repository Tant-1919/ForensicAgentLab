"use strict";
const $ = (id) => document.getElementById(id);
const labels = {sample:"委托书编号",lab:"图谱编号",template:"报告模板",knowledge:"司法知识"};
const statuses = {issue:"有疑点",clear:"已核对",unverified:"无法核实"};
let fixtures = [], active = null, busy = false, pendingDecision = null;
function el(tag, value, cls) {
    const node = document.createElement(tag);
    if (value !== undefined) node.textContent = value;
    if (cls) node.className = cls;
    return node;
}
// 浏览器只发请求给本机 FastAPI；真正的模型请求由 Python 后端发出。
async function api(path, body) {
    const res = await fetch(path, body ? {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)} : {});
    const data = await res.json();
    if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "请求未完成，请重试。");
    return data;
}
function error(message = "") { $("error").textContent = message; $("error").hidden = !message; }
function busyState(value) {
    busy = value;
    $("case").disabled = value || !fixtures.length;
    $("run").disabled = value || !fixtures.length;
    for (const button of $("review-form").querySelectorAll("button")) button.disabled = value;
}
function selected() {return fixtures.find((c) => c.id === $("case").value);}
function preview() {
    const c = selected();
    if (!c) return;
    const facts = el("div", undefined, "facts");
    for (const line of ["报告版本：v" + c.version, "样本编号：" + c.report.sample_no,
        "计量单位：" + (c.report.unit || "未填写"), "检查范围：" + c.scope.map(x => labels[x]).join("、")]) facts.append(el("div", line));
    $("preview").replaceChildren(facts);
}
function citation(hit) {
    const row = el("article", undefined, "finding");
    const meta = [hit.version, hit.section, hit.page ? "第 " + hit.page + " 页" : null, hit.id].filter(Boolean).join(" · ");
    row.append(el("h3", hit.title), el("p", meta, "subtle"),
        el("p", hit.kind === "summary" ? "公开材料摘要（非原文）" : "本地导入切片", "subtle"),
        el("blockquote", hit.quote || hit.body));
    if (/^https?:\/\//i.test(hit.url || "")) {
        const link = el("a","核对官方来源");
        link.href=hit.url; link.target="_blank"; link.rel="noopener noreferrer";
        row.append(link);
    }
    return row;
}
// 根据后端返回的 run 渲染结果和可复核证据，不在浏览器里编造检查结果。
function render(run) {
    active = run;
    $("status").textContent = run.status === "reviewed" ? "已复核" : "待人工复核";
    const findings = run.findings || [];
    const result = $("result");
    result.replaceChildren(el("p", run.case_id + " · 报告 v" + run.version + " · " + findings.length + " 条问题", "subtle"));
    for (const item of findings) {
        const row = el("article", undefined, "finding"), sources = el("details");
        sources.open = true;
        sources.append(el("summary", "原文证据"));
        for (const id of item.evidence_ids) {
            const source = run.evidence[id];
            if (source) sources.append(el("blockquote", source.source + "：" + source.quote));
        }
        row.append(el("h3", item.title), el("p", item.detail), el("p", item.suggestion), sources);
        result.append(row);
    }
    const checks = el("ul", undefined, "checks");
    for (const check of run.checks || []) checks.append(el("li", check.title + " · " + statuses[check.status] + "：" + check.detail));
    result.append(checks);
    if (run.guidance) {
        result.append(el("h3", "有引用的复核提示"), el("p", run.guidance.summary),
            el("p", run.guidance.citation_ids.join(" · "), "subtle"));
    }
    if (run.knowledge?.length) {
        result.append(el("h3", "通过 MCP 检索的司法知识"));
        result.append(...run.knowledge.map(citation));
    }
    if (run.stop_reason !== "complete") result.append(el("p","本次取证未完整结束，请查看执行记录。","subtle"));
    $("review").hidden = false;
    $("review-form").hidden = run.status === "reviewed";
    $("review-done").hidden = run.status !== "reviewed";
    if (run.decision) $("review-done").textContent = (run.decision.action === "accept" ? "已采纳" : "已驳回") + (run.decision.note ? " · " + run.decision.note : "");
    $("trace-panel").hidden = false;
    $("trace-link").href = "/api/audit-runs/" + encodeURIComponent(run.run_id) + "/trace";
    $("trace").replaceChildren(...run.trace.map(step => {
        const detail = [step.node, step.detail, step.transport, step.elapsed_ms == null ? null : step.elapsed_ms + " ms"].filter(Boolean).join(" · ");
        const row = el("li", detail);
        if (step.citations?.length) row.append(el("div", step.citations.join(" · "), "subtle"));
        return row;
    }));
}
$("case").addEventListener("change", () => { error(); preview(); });
$("run").addEventListener("click", async () => {
    if (busy || !selected()) return;
    error(); busyState(true);
    $("run").textContent = "正在核对材料…";
    try {
        const c = selected();
        const run = await api("/api/audit-runs", {case_id:c.id,version:c.version});
        history.replaceState(null, "", "?run=" + encodeURIComponent(run.run_id));
        pendingDecision = null; $("note").value = ""; render(run);
    } catch (err) { error(err.message); }
    finally { $("run").textContent = "开始审核"; busyState(false); }
});
$("review-form").addEventListener("submit", async event => {
    event.preventDefault();
    if (busy || !active) return;
    const action = event.submitter?.value;
    if (!action) return;
    error(); busyState(true);
    try {
        const note = $("note").value.trim();
        if (!pendingDecision || pendingDecision.action !== action || pendingDecision.note !== note) {
            pendingDecision = {action, note, decision_id:crypto.randomUUID(), version:active.version};
        }
        render(await api("/api/audit-runs/" + active.run_id + "/review", pendingDecision));
    } catch (err) { error(err.message); }
    finally { busyState(false); }
});
(async () => {
    try {
        const data = await api("/api/cases");
        fixtures = data.cases;
        $("mode").textContent = data.mode === "live" ? "真实模型模式" : "回放模式 · 无需密钥";
        $("case").replaceChildren(...fixtures.map(c => {const o = el("option", c.id + " · " + c.title); o.value=c.id; return o;}));
        const existing = new URLSearchParams(location.search).get("run");
        if (existing) {
            const run = await api("/api/audit-runs/" + encodeURIComponent(existing));
            $("case").value = run.case_id;
            render(run);
        }
    } catch(err) { error(err.message); }
    finally { preview(); busyState(false); }
})();

$("knowledge-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = event.submitter;
    if (button) button.disabled = true;
    const list = $("knowledge-results");
    try {
        const result = await api("/api/knowledge/search?q=" + encodeURIComponent($("knowledge-query").value));
        list.replaceChildren(el("p", "命中 " + result.hits.length + " 条 · " + result.ranking, "subtle"), ...result.hits.map(citation));
    } catch (err) {
        list.replaceChildren(el("p", err.message, "error"));
    } finally { if (button) button.disabled = false; }
});
api("/api/knowledge/stats").then(s => {
    $("knowledge-stats").textContent = s.sources + " 份来源 · " + s.chunks + " 个切片 · " + s.ranking + " · 索引 " + s.revision;
}).catch(err => { $("knowledge-stats").textContent = err.message; });
