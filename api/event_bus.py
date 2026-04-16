"""
Event bus — per-run asyncio.Queue for streaming pipeline events to the frontend.

The pipeline background task calls publish() after each agent/tool step.
The SSE endpoint calls subscribe() to get the queue and stream events to the browser.

Gap 4 fix: Events are also persisted to a Redis list (events:{run_id}) so that:
  - Subscribers that connect after events were published receive missed events (replay).
  - Process restarts don't lose the event history for in-flight runs.
  - Multiple browser tabs watching the same run each get the full stream.

The Redis list acts as a ring buffer: max 500 events, TTL 1 hour.
In-memory asyncio.Queue is still used for low-latency delivery within the
current process.
"""

import asyncio
import json
import time
from typing import Any

# run_id -> list of queues (multiple browser tabs can watch the same run)
_subscribers: dict[str, list[asyncio.Queue]] = {}

_EVENTS_KEY = "events:{run_id}"
_MAX_BUFFERED_EVENTS = 500
_EVENTS_TTL = 3600  # 1 hour


def _events_key(run_id: str) -> str:
    return f"events:{run_id}"


async def _redis_append(run_id: str, payload: dict) -> None:
    """
    Append event to Redis list for durability.

    Fire-and-forget — never blocks the pipeline on Redis errors.
    Uses RPUSH + LTRIM to keep only the last _MAX_BUFFERED_EVENTS entries.
    """
    try:
        from db import get_redis_client
        redis = await get_redis_client()
        key = _events_key(run_id)
        await redis.rpush(key, json.dumps(payload, default=str))
        await redis.ltrim(key, -_MAX_BUFFERED_EVENTS, -1)
        await redis.expire(key, _EVENTS_TTL)
    except Exception:
        pass  # Never block the pipeline on event persistence failures


def publish(run_id: str, event_type: str, data: dict) -> None:
    """
    Push an event to:
      1. All in-process subscribers watching this run (low-latency delivery).
      2. Redis list for durability across reconnects and process restarts.
    """
    payload: dict[str, Any] = {"type": event_type, "ts": time.time(), **data}

    # Deliver to in-memory subscribers
    for q in _subscribers.get(run_id, []):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass

    # Persist to Redis — schedule as a fire-and-forget task
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_redis_append(run_id, payload))
    except RuntimeError:
        pass


async def subscribe(run_id: str) -> asyncio.Queue:
    """
    Create and register a subscriber queue for this run.

    Replays any buffered events from Redis first so late subscribers
    (page reloads, reconnections, post-restart connections) receive the
    full event history.

    Caller must call unsubscribe() on cleanup.
    """
    q: asyncio.Queue = asyncio.Queue(maxsize=200)

    # Replay missed events from Redis before registering for new ones
    try:
        from db import get_redis_client
        redis = await get_redis_client()
        raw_events = await redis.lrange(_events_key(run_id), 0, -1)
        for raw in raw_events:
            try:
                q.put_nowait(json.loads(raw))
            except asyncio.QueueFull:
                break
    except Exception:
        pass  # Replay failure is non-fatal — subscriber still gets future events

    _subscribers.setdefault(run_id, []).append(q)
    return q


def unsubscribe(run_id: str, q: asyncio.Queue) -> None:
    """Remove a queue from the subscriber list."""
    subs = _subscribers.get(run_id, [])
    if q in subs:
        subs.remove(q)
    if not subs:
        _subscribers.pop(run_id, None)
