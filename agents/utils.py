"""
Shared LLM utilities for ReAct agents.

call_llm_with_tools — makes an Anthropic messages.create call with
    tool definitions filtered to only the names the agent wants.
extract_tool_calls  — pulls tool_use blocks from a response.
tool_result_message — builds the {"role": "user", "content": [tool_result, ...]}
    dict that Anthropic expects after tool_use blocks.
"""

import json
import os

import anthropic

_MODEL = lambda: os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")


def _mcp_to_anthropic(tool: dict) -> dict:
    """Convert an MCP tool dict to Anthropic tool format."""
    schema = tool.get("inputSchema") or {}
    # Ensure the schema has a type field — Anthropic requires it
    if "type" not in schema:
        schema = {"type": "object", "properties": schema}
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "input_schema": schema,
    }


async def call_llm_with_tools(
    messages: list[dict],
    available_tools: list[str],
    mcp_tools: list[dict],
    max_tokens: int = 4096,
    system_prompt: str | None = None,
):
    """
    Call the Anthropic API with a filtered set of tools.

    available_tools: names of tools the agent is allowed to call this turn.
    mcp_tools: full tool list from mcp.list_tools() — filtered here by name.
    system_prompt: static agent system prompt. When provided, it is sent with
        cache_control so the stable prefix (system + tool definitions) is cached
        across all ReAct loop turns within the same agent run (5-min TTL).

    Returns the raw Anthropic Message object. Callers check
    response.stop_reason and use extract_tool_calls(response).
    """
    client = anthropic.AsyncAnthropic()
    tools = [
        _mcp_to_anthropic(t)
        for t in mcp_tools
        if t["name"] in available_tools
    ]

    # Mark the last tool definition as the end of the cacheable prefix.
    # The cache covers: system prompt + all tool definitions. Turns 2+ in the
    # ReAct loop hit the cache because this prefix is identical each turn.
    if tools:
        tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}

    kwargs: dict = dict(
        model=_MODEL(),
        max_tokens=max_tokens,
        tools=tools,
        messages=messages,
        timeout=120.0,
    )
    if system_prompt:
        kwargs["system"] = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    return await client.messages.create(**kwargs)


def extract_tool_calls(response) -> list[dict]:
    """
    Return a list of {id, name, input} dicts from tool_use content blocks.
    Returns an empty list if the response has no tool calls.
    """
    return [
        {"id": b.id, "name": b.name, "input": b.input}
        for b in response.content
        if b.type == "tool_use"
    ]


def tool_result_message(tool_use_id: str, result: dict) -> dict:
    """
    Build a single tool_result user-turn message for one tool call.

    Note: if a response contained multiple tool_use blocks, callers should
    collect all results and build a single user message with all of them:

        user_msg = {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": id1, "content": json1},
                {"type": "tool_result", "tool_use_id": id2, "content": json2},
            ]
        }

    This helper is a convenience for the single-tool-per-turn case.
    """
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": json.dumps(result),
            }
        ],
    }


def trim_messages(
    messages: list[dict],
    keep_first: int = 1,
    keep_last: int = 10,
) -> list[dict]:
    """
    Trim a ReAct message list to prevent context window overflow.

    Keeps the first `keep_first` messages (the initial user context/goal, which
    the LLM must always see) and the last `keep_last` messages (the most recent
    tool calls and results). Middle turns are dropped.

    Call this at the top of each ReAct loop iteration if len(messages) is large:

        messages = trim_messages(messages)
        response = await call_llm_with_tools(messages, ...)

    The default of keep_first=1, keep_last=10 retains ~5 full tool-call rounds
    without growing the prompt past ~8K tokens in typical use.
    """
    total = keep_first + keep_last
    if len(messages) <= total:
        return messages
    return messages[:keep_first] + messages[-keep_last:]


def build_tool_results_message(results: list[tuple[str, dict]]) -> dict:
    """
    Build a single user message containing multiple tool_result blocks.

    results: list of (tool_use_id, result_dict) tuples, one per tool call
    in the preceding assistant message.
    """
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tid,
                "content": json.dumps(res),
            }
            for tid, res in results
        ],
    }
