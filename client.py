"""
MCPClient — async context manager wrapping the MCP ClientSession over SSE.

Usage:
    async with MCPClient("http://localhost:8000/sse") as mcp:
        result = await mcp.call("connect_csv", {"file_path": "..."})
        tools  = await mcp.list_tools()
"""

import json
from mcp import ClientSession
from mcp.client.sse import sse_client


class MCPClient:
    def __init__(self, url: str) -> None:
        self._url = url
        self._streams_ctx = None
        self._session_ctx = None
        self.session: ClientSession | None = None

    async def __aenter__(self) -> "MCPClient":
        self._streams_ctx = sse_client(self._url)
        read, write = await self._streams_ctx.__aenter__()

        self._session_ctx = ClientSession(read, write)
        self.session = await self._session_ctx.__aenter__()

        await self.session.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._session_ctx is not None:
            await self._session_ctx.__aexit__(exc_type, exc_val, exc_tb)
        if self._streams_ctx is not None:
            await self._streams_ctx.__aexit__(exc_type, exc_val, exc_tb)

    async def call(self, tool_name: str, args: dict) -> dict:
        """
        Call a named MCP tool and return its result as a parsed dict.

        The MCP server returns content[0].text as a JSON string.
        Raises ValueError if the server returns an error or empty content.
        """
        result = await self.session.call_tool(tool_name, args)

        if not result.content:
            raise ValueError(f"Tool '{tool_name}' returned empty content")

        text = result.content[0].text
        return json.loads(text)

    async def list_tools(self) -> list[dict]:
        """
        Return all tools registered on the MCP server as a list of dicts.

        Each dict has: name (str), description (str), inputSchema (dict).
        """
        response = await self.session.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema if hasattr(t, "inputSchema") else {},
            }
            for t in response.tools
        ]
