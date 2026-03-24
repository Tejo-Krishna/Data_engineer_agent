"""
Domain detection agent — fixed 2-call sequence.

  1. detect_domain  — keyword heuristic first, LLM fallback if ambiguous
  2. load_domain_rules — reads YAML from domain_rules/{domain}.yaml

The graph handles the confidence threshold check (< 0.60 → HITL pause).
This agent only detects and loads; it does not enforce the threshold.

Writes to state: domain, domain_confidence, domain_context.
"""

from langgraph.types import RunnableConfig

from observability.tracing import log_domain_low_confidence
from orchestrator.state import PipelineState


async def run_domain_agent(
    state: PipelineState,
    config: RunnableConfig,
) -> PipelineState:
    mcp = config["configurable"]["mcp"]
    logger = config["configurable"]["logger"]
    span = logger.agent_start("domain")

    schema: dict = state.get("schema") or {}
    profile: dict = state.get("profile") or {}

    # ------------------------------------------------------------------
    # Tool call 1 — detect domain
    # ------------------------------------------------------------------
    print("  [domain] calling detect_domain ...", flush=True)
    domain_result = await mcp.call("detect_domain", {
        "column_names": list(schema.keys()),
        "sample_values": {
            col: profile[col].get("sample_values", [])
            for col in profile
            if col in schema
        },
        "user_goal": state["user_goal"],
    })

    print(f"  [domain] detect_domain done", flush=True)
    domain: str = domain_result.get("domain", "unknown")
    confidence: float = float(domain_result.get("confidence", 0.0))
    method: str = domain_result.get("method", "unknown")

    logger.tool_call(
        span, "domain", "detect_domain",
        {"column_count": len(schema)},
        {"domain": domain, "confidence": confidence, "method": method},
    )

    # Emit Langfuse event if confidence is low
    if confidence < 0.60:
        log_domain_low_confidence(state["run_id"], domain, confidence)

    # ------------------------------------------------------------------
    # Tool call 2 — load domain rules
    # ------------------------------------------------------------------
    print(f"  [domain] calling load_domain_rules ({domain}) ...", flush=True)
    rules_result = await mcp.call("load_domain_rules", {"domain": domain})
    domain_context: dict = rules_result.get("rules", {})

    logger.tool_call(
        span, "domain", "load_domain_rules",
        {"domain": domain},
        {
            "rules_loaded": bool(domain_context),
            "required_transforms": len(
                domain_context.get("required_transforms", [])
            ),
            "forbidden_transforms": len(
                domain_context.get("forbidden_transforms", [])
            ),
        },
    )

    logger.agent_end(span)

    return {
        **state,
        "domain": domain,
        "domain_confidence": confidence,
        "domain_context": domain_context,
    }
