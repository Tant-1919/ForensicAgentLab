"""Local RAG: sources -> traceable chunks -> scoped SQLite FTS5 retrieval."""
from contextlib import closing
from hashlib import sha256
from pathlib import Path
from datetime import date
import json
import os
import re
import sqlite3
import threading

ROOT = Path(__file__).resolve().parent.parent
SEED = ROOT / "knowledge" / "seed.json"
INDEX_LOCK = threading.RLock()
INDEX_SCHEMA = "2"
ARTICLE = re.compile(r"(?m)^[ \t]*(第[零一二三四五六七八九十百0-9]+[条章节])")
TOKENS = re.compile(r"[\u3400-\u9fff]+|[a-z0-9]+", re.I)


def tokenize(text):
    """Chinese bigrams plus single characters; ASCII words stay whole."""
    result = []
    for word in TOKENS.findall(text.casefold()):
        if re.match(r"[\u3400-\u9fff]", word):
            result.extend(word)
            result.extend(word[i:i+2] for i in range(len(word)-1))
        else:
            result.append(word)
    return result


def slices(text, limit=420, overlap=55):
    """Split within one article/page; retain character offsets for verification."""
    if not 0 <= overlap < limit:
        raise ValueError("overlap must be less than limit")
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            boundary = max(text.rfind(x, start + limit//2, end) for x in "\n。；！？")
            if boundary >= 0:
                end = boundary + 1
        if text[start:end].strip():
            yield start, end, text[start:end]
        if end == len(text):
            break
        start = max(start+1, end-overlap)


def _markdown_sections(text):
    pattern = re.compile(r"(?m)^[ \t]*#{1,6}[ \t]+(.+)$")
    headings = list(pattern.finditer(text)) or list(ARTICLE.finditer(text))
    if not headings:
        return [("正文", None, text)]
    rows = [("前言", None, text[:headings[0].start()])] if text[:headings[0].start()].strip() else []
    for i, h in enumerate(headings):
        end = headings[i+1].start() if i+1 < len(headings) else len(text)
        rows.append((h.group(1).strip(), None, text[h.start():end]))
    return rows


def _pdf_sections(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    if len(reader.pages) > 300:
        raise ValueError("Each PDF may have at most 300 pages")
    rows = []
    for number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        for section, _, body in _markdown_sections(text):
            if body.strip():
                rows.append((section, number, body))
    if not rows:
        raise ValueError("PDF has no text layer. OCR locally before importing.")
    return rows


def files():
    """Private manifests point to files inside the private corpus, never arbitrary paths."""
    directory = Path(os.getenv("AGENT_KNOWLEDGE_IMPORT", ROOT / "data" / "knowledge-import")).resolve()
    imports = []
    for manifest in sorted(directory.glob("*.json")):
        d = json.loads(manifest.read_text(encoding="utf-8-sig"))
        source = (directory / d["file"]).resolve()
        if not source.is_relative_to(directory) or source.suffix.lower() not in {".pdf", ".md", ".txt"}:
            raise ValueError(f"Invalid knowledge input: {manifest.name}")
        if source.stat().st_size > 25_000_000:
            raise ValueError("Knowledge source exceeds 25 MB")
        imports.append((d, source, manifest))
    return imports


def records(imports):
    """Yield provenance records. Seed entries are editorial summaries, not legal quotations."""
    for d in json.loads(SEED.read_text(encoding="utf-8")):
        yield d, "summary", [(name, None, text) for name, text in d["sections"]]
    for d, path, _ in imports:
        if path.suffix.lower() == ".pdf":
            sections = _pdf_sections(path)
        else:
            sections = _markdown_sections(path.read_text(encoding="utf-8-sig"))
        yield d, "imported_text", sections


def _dbpath():
    return Path(os.getenv("AGENT_KNOWLEDGE_DB", ROOT / "data" / "knowledge.sqlite"))


def ensure_index(force=False):
    # 用资料内容计算索引版本。未改动就复用；变化时在 SQLite 事务里重建。
    # 存储切片正文与元数据，另用 FTS5 索引分词，便于定位来源和筛选版本。
    path = _dbpath()
    path.parent.mkdir(parents=True, exist_ok=True)
    imports = files()
    digest = sha256(INDEX_SCHEMA.encode())
    for file in [SEED, *(p for _, source, manifest in imports for p in (manifest, source))]:
        digest.update(file.read_bytes())
    revision = digest.hexdigest()
    with INDEX_LOCK, closing(sqlite3.connect(path, timeout=12)) as db:
        db.execute("CREATE TABLE IF NOT EXISTS index_meta (id INTEGER PRIMARY KEY, revision TEXT)")
        db.execute("""CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY, doc_id TEXT, title TEXT, version TEXT, section TEXT,
            page INTEGER, start INTEGER, end INTEGER, url TEXT, kind TEXT, body TEXT,
            effective_from TEXT, effective_to TEXT, specialty TEXT)""")
        db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunk_search USING fts5(id UNINDEXED, tokens)")
        if not force and db.execute("SELECT 1 FROM index_meta WHERE revision=?", (revision,)).fetchone():
            return path
        with db:
            db.execute("DELETE FROM chunks")
            db.execute("DELETE FROM chunk_search")
            for d, kind, sections in records(imports):
                if not all(d.get(field) for field in ("doc_id","title","version")):
                    raise ValueError("Missing doc_id, title or version")
                for key in ("effective_from", "effective_to"):
                    if d.get(key):
                        date.fromisoformat(d[key])
                for seq, (section, page, body) in enumerate(sections):
                    for start, end, part in slices(body):
                        identity = json.dumps([d["doc_id"],d["version"],page,section,seq,start,part], ensure_ascii=False)
                        cid = "KB-"+sha256(identity.encode()).hexdigest()[:16]
                        row = (cid, d["doc_id"], d["title"], d["version"], section, page, start, end,
                               d.get("url",""), kind, part, d.get("effective_from"), d.get("effective_to"),
                               d.get("specialty","all"))
                        db.execute("INSERT INTO chunks VALUES ("+",".join(["?"]*len(row))+")", row)
                        db.execute("INSERT INTO chunk_search VALUES (?,?)", (cid, " ".join(tokenize(d["title"]+" "+section+" "+part))))
            db.execute("INSERT OR REPLACE INTO index_meta VALUES (1,?)",(revision,))
    return path


def search(query, specialty="XJ", as_of=None, top_k=4, public_only=False):
    # 检索只排序候选切片，不能判定文献适用性或法律效力。public_only
    # 用于允许调用外部模型的合成流程，私有导入文献只在本机检索。
    effective_date = as_of or date.today().isoformat()
    date.fromisoformat(effective_date)
    path = ensure_index()
    terms = list(dict.fromkeys(t for t in tokenize(query[:180]) if len(t)>1))[:48]
    if not terms:
        terms = list(dict.fromkeys(tokenize(query[:60])))[:32]
    if not terms:
        return []
    expr = " OR ".join('"'+term+'"' for term in terms)
    with closing(sqlite3.connect(path)) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("""
            SELECT c.*, round(-bm25(chunk_search),6) AS score
            FROM chunk_search JOIN chunks c ON c.id=chunk_search.id
            WHERE chunk_search MATCH ? AND c.specialty IN ('all',?) AND (?=0 OR c.kind='summary')
              AND (c.effective_from IS NULL OR c.effective_from<=?)
              AND (c.effective_to IS NULL OR c.effective_to>=?)
            ORDER BY score DESC,c.id LIMIT ?""", (expr,specialty,int(public_only),effective_date,effective_date,min(8,max(1,top_k)))).fetchall()
        return [dict(row) for row in rows]


def stats():
    with closing(sqlite3.connect(ensure_index())) as db:
        count, sources = db.execute("SELECT count(*),count(DISTINCT doc_id) FROM chunks").fetchone()
        revision = db.execute("SELECT revision FROM index_meta WHERE id=1").fetchone()[0]
    return {"chunks":count,"sources":sources,"revision":revision[:12],"ranking":"FTS5/BM25"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build and inspect the local forensic knowledge index.")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--query")
    args=parser.parse_args()
    ensure_index(force=args.rebuild)
    print(json.dumps(search(args.query) if args.query else stats(), ensure_ascii=False, indent=2))

