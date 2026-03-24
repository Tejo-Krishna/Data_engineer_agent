"""
LangGraph pipeline graph.

Nodes:
  profiler         → fixed 5-call sequence
  drift_checkpoint → inline conditional node (no MCP calls)
  domain_detection → fixed 2-call sequence
  transformer      → ReAct loop, max 6 calls
  quality          → ReAct loop, max 6 calls
  catalogue        → fixed 5-call sequence

Edges:
  profiler → drift_checkpoint → domain_detection → transformer → quality
  quality → (conditional) → catalogue / transformer(retry) / END

No interrupt_before — HITL is handled via Redis polling inside the transformer
agent. The graph runs to completion asynchronously.
"""

from langgraph.graph import StateGraph, END
from langgraph.types import RunnableConfig

from agents.profiler_agent import run_profiler_agent
from agents.domain_agent import run_domain_agent
from agents.transformer_agent import run_transformer_agent
from agents.quality_agent import run_quality_agent
from agents.catalogue_agent import run_catalogue_agent
from hitl.checkpoint import hitl_drift_checkpoint
from orchestrator.router import route_after_quality
from orchestrator.state import PipelineState


def _wrap_node(agent_key: str, fn):
    """Wrap an agent node to publish live events to the SSE bus."""
    async def _wrapper(state: PipelineState, config: RunnableConfig) -> PipelineState:
        try:
            from api.event_bus import publish
            run_id = state.get("run_id", "")
            publish(run_id, "agent_started", {"agent": agent_key})
            result = await fn(state, config)
            publish(run_id, "agent_complete", {"agent": agent_key})
            return result
        except Exception as exc:
            try:
                from api.event_bus import publish
                publish(state.get("run_id", ""), "agent_failed", {"agent": agent_key, "error": str(exc)})
            except Exception:
                pass
            raise
    return _wrapper


# ---------------------------------------------------------------------------
# Drift checkpoint — inline node, no MCP calls
# ---------------------------------------------------------------------------


async def run_drift_checkpoint(
    state: PipelineState,
    config: RunnableConfig,
) -> PipelineState:
    """
    Lightweight conditional pause based on schema drift severity.

    none/warning → pass through (drift_checkpoint_approved=True)
    critical     → block on HITL drift approval via Redis polling
    """
    logger = config["configurable"]["logger"]
    span = logger.agent_start("drift_checkpoint")

    print("  [drift_checkpoint] entered", flush=True)
    drift: dict = state.get("schema_drift") or {}
    severity: str = drift.get("drift_severity", "none")

    if severity in ("none", "warning"):
        approved = True
    elif severity == "critical":
        approved = await hitl_drift_checkpoint(
            run_id=state["run_id"],
            drift_report=drift,
        )
    else:
        approved = True

    logger.tool_call(
        span, "drift_checkpoint", "evaluate_drift",
        {"severity": severity},
        {"approved": approved},
    )
    logger.agent_end(span)

    return {**state, "drift_checkpoint_approved": approved}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph():
    g = StateGraph(PipelineState)

    # Register nodes — wrapped to publish live events to the SSE bus
    g.add_node("profiler", _wrap_node("profiler", run_profiler_agent))
    g.add_node("drift_checkpoint", run_drift_checkpoint)
    g.add_node("domain_detection", _wrap_node("domain", run_domain_agent))
    g.add_node("transformer", _wrap_node("transformer", run_transformer_agent))
    g.add_node("quality", _wrap_node("quality", run_quality_agent))
    g.add_node("catalogue", _wrap_node("catalogue", run_catalogue_agent))

    # Entry point
    g.set_entry_point("profiler")

    # Fixed edges
    g.add_edge("profiler", "drift_checkpoint")
    g.add_edge("drift_checkpoint", "domain_detection")
    g.add_edge("domain_detection", "transformer")
    g.add_edge("transformer", "quality")
    g.add_edge("catalogue", END)

    # Conditional edge from quality
    g.add_conditional_edges(
        "quality",
        route_after_quality,
        {
            "catalogue": "catalogue",
            "retry": "transformer",
            "failed": END,
        },
    )

    return g.compile()
