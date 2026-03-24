"""
HITL approval helper — run in a second terminal while main.py is running.

Usage:
    python scripts/hitl_approve.py <run_id>          # auto-approve code
    python scripts/hitl_approve.py <run_id> --reject  # reject code
    python scripts/hitl_approve.py <run_id> --drift   # approve drift checkpoint
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add project root to path so `db` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from db import get_redis_client


def _code_key(run_id: str) -> str:
    return f"hitl:{run_id}:code"


def _drift_key(run_id: str) -> str:
    return f"hitl:{run_id}:drift"


async def poll_and_approve(run_id: str, reject: bool = False, drift: bool = False):
    redis = await get_redis_client()
    key = _drift_key(run_id) if drift else _code_key(run_id)
    kind = "drift" if drift else "code"

    print(f"Polling for {kind} checkpoint on run {run_id} ...")
    print("(Ctrl+C to cancel)\n")

    deadline = time.time() + 1800  # 30 min max wait
    while time.time() < deadline:
        raw = await redis.get(key)
        if raw:
            data = json.loads(raw)
            state = data.get("state")
            print(f"  State: {state}", end="")

            if state == "pending":
                if drift:
                    data["state"] = "confirmed" if not reject else "rejected"
                    await redis.setex(key, 3600, json.dumps(data))
                    print(f"  → {data['state']}")
                    print(f"\nDrift checkpoint {data['state']}.")
                    return
                else:
                    data["state"] = "rejected" if reject else "confirmed"
                    await redis.setex(key, 3600, json.dumps(data))
                    print(f"  → {data['state']}")
                    print(f"\nCode checkpoint {data['state']}.")
                    return
            else:
                print()  # newline
        else:
            print("  (no checkpoint yet)")

        await asyncio.sleep(3)

    print("\nTimed out waiting for checkpoint.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/hitl_approve.py <run_id> [--reject] [--drift]")
        sys.exit(1)

    run_id = sys.argv[1]
    reject = "--reject" in sys.argv
    drift = "--drift" in sys.argv
    asyncio.run(poll_and_approve(run_id, reject=reject, drift=drift))
