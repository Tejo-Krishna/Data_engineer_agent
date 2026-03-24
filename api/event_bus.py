"""
Event bus — per-run asyncio.Queue for streaming pipeline events to the frontend.

The pipeline background task calls publish() after each agent/tool step.
The SSE endpoint calls subscribe() to get the queue and stream events to the browser.
"""

import asyncio
import time
from typing import Any

# run_id -> list of queues (multiple browser tabs can watch the same run)
_subscribers: dict[str, list[asyncio.Queue]] = {}


def publish(run_id: str, event_type: str, data: dict) -> None:
    """Push an event to all subscribers watching this run."""
    payload = {"type": event_type, "ts": time.time(), **data}
    for q in _subscribers.get(run_id, []):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass


def subscribe(run_id: str) -> asyncio.Queue:
    """Create and register a new queue for this run. Caller must call unsubscribe() on cleanup."""
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _subscribers.setdefault(run_id, []).append(q)
    return q


def unsubscribe(run_id: str, q: asyncio.Queue) -> None:
    """Remove a queue from the subscriber list."""
    subs = _subscribers.get(run_id, [])
    if q in subs:
        subs.remove(q)
    if not subs:
        _subscribers.pop(run_id, None)
