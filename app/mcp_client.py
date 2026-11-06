"""Dispatch a bounded MCP tool request to our known local stdio server."""
import asyncio
import os
import sys
from pathlib import Path
from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parent.parent

async def _call(tool, arguments):
    # No model credentials are passed to the subprocess.
    forwarded = {k: os.environ[k] for k in ("AGENT_KNOWLEDGE_DB","AGENT_KNOWLEDGE_IMPORT") if k in os.environ}
    server = StdioServerParameters(command=sys.executable, args=["-m","app.mcp_server"],
                                   cwd=ROOT, env=forwarded)
    async with Client(server, read_timeout_seconds=10) as client:
        result = await client.call_tool(tool, arguments)
        if result.is_error or not isinstance(result.structured_content, dict):
            raise RuntimeError("The MCP knowledge tool did not return a usable result")
        return result.structured_content

def call_knowledge(tool, arguments=None):
    # The FastAPI sync endpoints and graph tools run outside the async event loop.
    if tool not in {"search_forensic_knowledge","knowledge_index_info"}:
        raise ValueError("Unregistered MCP tool")
    return asyncio.run(asyncio.wait_for(_call(tool, arguments or {}), timeout=15))

