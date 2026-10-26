"""Tavily Remote MCP: live web search and page extraction shared by agents.

Connects over Streamable HTTP to Tavily's hosted MCP server, discovers its
tools with tools/list, and calls them with tools/call. If the server is
unreachable the agents keep working from the textbook alone, and the client
retries discovery at most once a minute instead of on every request.
"""

import asyncio
import json
import logging
import re
import time

from langchain_core.tools import BaseTool

log = logging.getLogger(__name__)

# The assignment specifies the key as a query parameter on the Remote MCP URL, so any
# error message or log line that includes the URL would include the key: redact first.
TAVILY_MCP_URL = "https://mcp.tavily.com/mcp/?tavilyApiKey={key}"
KEY_IN_URL = re.compile(r"(tavilyApiKey=)[^&\s'\"]+")
RETRY_AFTER_S = 60
CONNECT_TIMEOUT_S = 15
MAX_RESULT_CHARS = 1500


class TavilyMCP:
    clock = staticmethod(time.monotonic)  # replaceable in tests

    def __init__(self, api_key: str | None):
        self.api_key = api_key
        self.tools: dict[str, BaseTool] = {}
        # None until the first attempt. Not 0.0: self.clock() counts from boot, and a fresh
        # Cloud Run container is seconds old, so a 0.0 start looked like "tried moments ago"
        # and every cold start skipped the connection.
        self.last_attempt: float | None = None
        self.last_error: str | None = None if api_key else "TAVILY_API_KEY is not set"

    async def connect(self) -> bool:
        """Discover Tavily's tools (MCP tools/list). Returns True if web tools are available."""
        if self.tools:
            return True
        if not self.api_key:
            if self.last_attempt is None:  # say so once, not on every turn
                self.last_attempt = self.clock()
                log.warning("Tavily MCP unavailable: %s", self.last_error)
            return False
        if self.last_attempt is not None and self.clock() - self.last_attempt < RETRY_AFTER_S:
            return False
        self.last_attempt = self.clock()
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient({
            "tavily": {"transport": "streamable_http", "url": TAVILY_MCP_URL.format(key=self.api_key)},
        })
        try:
            tools = await asyncio.wait_for(client.get_tools(server_name="tavily"), CONNECT_TIMEOUT_S)
        except Exception as error:  # transport failure, bad key, malformed handshake
            self.last_error = redact(f"{type(error).__name__}: {error}")
            log.warning("Tavily MCP unavailable: %s", self.last_error)
            return False
        self.tools = {tool.name: tool for tool in tools}
        self.last_error = None
        log.info("Tavily MCP tools: %s", sorted(self.tools))
        return True

    def find(self, suffix: str) -> BaseTool | None:
        """Tool whose name ends with e.g. "search" (tavily_search or tavily-search)."""
        return next((t for name, t in self.tools.items() if name.replace("-", "_").endswith(suffix)), None)


def redact(text: str) -> str:
    """Hide the Tavily API key in any text that may contain the MCP URL."""
    return KEY_IN_URL.sub(r"\1***", text)


def accepted_args(tool: BaseTool, wanted: dict) -> dict:
    """Keep only the arguments the tool's JSON Schema declares."""
    schema = tool.args_schema if isinstance(tool.args_schema, dict) else tool.args_schema.model_json_schema()
    allowed = schema.get("properties", {})
    return {key: value for key, value in wanted.items() if key in allowed}


def content_text(content) -> str:
    """Flatten an MCP tool result (string or list of content blocks) to text."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, dict):
            parts.append(block.get("text") or "")
        else:
            parts.append(getattr(block, "text", "") or str(block))
    return "\n".join(p for p in parts if p)


def parse_results(text: str) -> list[dict]:
    """Parse Tavily output into [{title, url, content}], tolerating JSON or plain text."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        data = None
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return [
            {"title": r.get("title") or r.get("url", ""), "url": r.get("url"),
             "content": (r.get("content") or r.get("raw_content") or "")[:MAX_RESULT_CHARS]}
            for r in data["results"] if isinstance(r, dict)
        ]

    blocks = re.split(r"\n(?=Title: )", text)
    results = []
    for block in blocks:
        url = re.search(r"^URL: (\S+)", block, re.MULTILINE)
        if not url:
            continue
        title = re.search(r"^Title: (.+)$", block, re.MULTILINE)
        content = re.search(r"^(?:Content|Raw Content): (.*)", block, re.MULTILINE | re.DOTALL)
        results.append({
            "title": title.group(1).strip() if title else url.group(1),
            "url": url.group(1),
            "content": (content.group(1) if content else block).strip()[:MAX_RESULT_CHARS],
        })
    return results
