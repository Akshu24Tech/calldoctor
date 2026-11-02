"""Convert each platform's call payload into the common Call format.

Vapi    : assistant `server.url` webhook, message.type == "end-of-call-report"
Retell  : agent `webhook_url`, event == "call_ended" or "call_analyzed"
LiveKit : our own agent POSTs its session transcript to /ingest/livekit
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from calldoctor.models import Call, Turn


def _iso(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


# ------------------------------------------------------------------ Vapi
def from_vapi(payload: dict[str, Any]) -> Call | None:
    msg = payload.get("message") or payload
    if msg.get("type") not in (None, "end-of-call-report"):
        return None  # status updates etc. are ignored
    call = msg.get("call") or {}
    artifact = msg.get("artifact") or call.get("artifact") or {}
    messages = artifact.get("messages") or msg.get("messages") or []
    started = _iso(msg.get("startedAt") or call.get("startedAt"))
    ended = _iso(msg.get("endedAt") or call.get("endedAt"))

    turns = []
    for m in messages:
        role = {"bot": "agent", "assistant": "agent", "user": "user"}.get(m.get("role"))
        if not role or not (m.get("message") or "").strip():
            continue
        start = m.get("secondsFromStart")
        end = start + m["duration"] / 1000 if start is not None and m.get("duration") is not None else None
        turns.append(Turn(role=role, text=m["message"].strip(), start_s=start, end_s=end))

    meta = call.get("metadata") or (call.get("assistantOverrides") or {}).get("metadata") or {}
    return Call(
        id=f"vapi-{call.get('id') or int(time.time() * 1000)}",
        platform="vapi",
        source="webhook",
        turns=turns,
        agent_version=meta.get("agent_version"),
        created_at=started or time.time(),
        duration_s=round(ended - started, 1) if started and ended else msg.get("durationSeconds"),
        cost_usd=msg.get("cost") if msg.get("cost") is not None else call.get("cost"),
        ended_reason=msg.get("endedReason") or call.get("endedReason"),
        meta={"assistant_id": call.get("assistantId")},
    )


# ------------------------------------------------------------------ Retell
def from_retell(payload: dict[str, Any]) -> Call | None:
    if payload.get("event") not in ("call_ended", "call_analyzed"):
        return None
    call = payload.get("call") or {}
    turns = []
    for u in call.get("transcript_object") or []:
        if u.get("role") not in ("agent", "user") or not (u.get("content") or "").strip():
            continue
        words = u.get("words") or []
        turns.append(
            Turn(
                role=u["role"],
                text=u["content"].strip(),
                start_s=words[0].get("start") if words else None,
                end_s=words[-1].get("end") if words else None,
            )
        )
    cost = (call.get("call_cost") or {}).get("combined_cost")
    meta = call.get("metadata") or {}
    return Call(
        id=f"retell-{call.get('call_id') or int(time.time() * 1000)}",
        platform="retell",
        source="webhook",
        turns=turns,
        agent_version=meta.get("agent_version"),
        created_at=(call.get("start_timestamp") or time.time() * 1000) / 1000,
        duration_s=(call.get("duration_ms") or 0) / 1000 or None,
        cost_usd=cost / 100 if cost is not None else None,
        ended_reason=call.get("disconnection_reason"),
        meta={"agent_id": call.get("agent_id"), "event": payload.get("event")},
    )


# ------------------------------------------------------------------ LiveKit
def from_livekit(payload: dict[str, Any]) -> Call:
    t0 = payload.get("started_at") or time.time()
    turns = []
    for t in payload.get("turns", []):
        if t.get("role") not in ("assistant", "user") or not (t.get("text") or "").strip():
            continue
        m = t.get("metrics") or {}
        s, e = m.get("started_speaking_at"), m.get("stopped_speaking_at")
        turns.append(
            Turn(
                role="agent" if t["role"] == "assistant" else "user",
                text=t["text"].strip(),
                start_s=s - t0 if s else None,
                end_s=e - t0 if e else None,
                interrupted=bool(t.get("interrupted")),
            )
        )
    ended = payload.get("ended_at")
    return Call(
        id=f"livekit-{payload.get('call_id') or int(time.time() * 1000)}",
        platform="livekit",
        source="upload",
        turns=turns,
        agent_version=payload.get("agent_version"),
        created_at=t0,
        duration_s=round(ended - t0, 1) if ended else None,
        cost_usd=0.0,
        ended_reason=payload.get("ended_reason"),
    )
