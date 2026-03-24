"""
DirectClient — calls MCP tool functions in-process, bypassing SSE.

Used by main.py instead of MCPClient when the agent process is co-located
with the tool implementations. Avoids the anyio cross-task SSE issue that
occurs when LangGraph runs agent nodes in separate asyncio tasks.

Has the same interface as MCPClient:
    async with DirectClient() as client:
        result = await client.call("sample_data", {...})
        tools  = await client.list_tools()
"""

from mcp_server.server import TOOL_DEFINITIONS, TOOL_HANDLERS


class DirectClient:
    async def __aenter__(self) -> "DirectClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    async def call(self, tool_name: str, args: dict) -> dict:
        """Call a tool function directly and return its result dict."""
        handler = TOOL_HANDLERS.get(tool_name)
        if handler is None:
            raise ValueError(f"Unknown tool: {tool_name!r}")
        return await handler(**args)

    async def list_tools(self) -> list[dict]:
        """Return tool definitions in the same format as MCPClient.list_tools()."""
        tools = []
        for t in TOOL_DEFINITIONS:
            schema = {}
            if hasattr(t, "inputSchema"):
                raw = t.inputSchema
                # inputSchema may be a dict or a Pydantic model
                if hasattr(raw, "model_json_schema"):
                    schema = raw.model_json_schema()
                elif isinstance(raw, dict):
                    schema = raw
            tools.append({
                "name": t.name,
                "description": t.description or "",
                "inputSchema": schema,
            })
        return tools
