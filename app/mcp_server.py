"""Read-only MCP stdio server. Its stdout carries protocol messages only."""
from mcp.server.mcpserver import MCPServer
from app.knowledge import search, stats
from pydantic import BaseModel

class SearchResult(BaseModel):
    hits: list[dict]
    ranking: str

class IndexInfo(BaseModel):
    chunks: int
    sources: int
    revision: str
    ranking: str

server = MCPServer("forensic-knowledge", version="1.0.0")

@server.tool(structured_output=True)
def search_forensic_knowledge(query: str, specialty: str = "XJ", top_k: int = 4, as_of: str | None = None, public_only: bool = False) -> SearchResult:
    """Search indexed forensic sources with citations, filtered by specialty and date."""
    if not 1 <= len(query) <= 180 or not 1 <= top_k <= 8 or specialty not in {"XJ","CJ","LC","BL"}:
        raise ValueError("Query or scope is out of bounds")
    return SearchResult(hits=search(query, specialty, as_of, top_k, public_only), ranking="FTS5/BM25")

@server.tool(structured_output=True)
def knowledge_index_info() -> IndexInfo:
    """Return corpus and chunk counts and the index revision."""
    return IndexInfo(**stats())

if __name__ == "__main__":
    server.run(transport="stdio")

