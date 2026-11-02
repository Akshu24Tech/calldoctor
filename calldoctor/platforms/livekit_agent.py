"""The survey agent on LiveKit. When a session ends it POSTs the transcript
(with LiveKit's per-turn timing) to CallDoctor's /ingest/livekit.

    pip install -r requirements-agent.txt
    python -m calldoctor.platforms.livekit_agent dev      # then talk to it in the LiveKit console
    python -m calldoctor.platforms.livekit_agent console  # terminal only

CALLDOCTOR_URL defaults to PUBLIC_URL, or http://localhost:8000 when unset.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

import httpx
from livekit.agents import (Agent, AgentServer, AgentSession, JobContext,
                            TurnHandlingOptions, cli, inference)
from livekit.agents.voice.events import ConversationItemAddedEvent
from livekit.plugins import groq, sarvam

from calldoctor import config

log = logging.getLogger("calldoctor.livekit")
S = config.survey()
LK = S["platforms"]["livekit"]
TARGET = (os.environ.get("CALLDOCTOR_URL") or config.PUBLIC_URL or "http://localhost:8000").rstrip("/")


class SurveyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=config.system_prompt())


server = AgentServer()


@server.rtc_session(agent_name=LK["agent_name"])
async def entrypoint(ctx: JobContext) -> None:
    started, turns = time.time(), []

    session = AgentSession(
        stt=sarvam.STT(model=LK["stt"]["model"], language=LK["stt"]["language"]),
        llm=groq.LLM(model=config.agent_model(), temperature=S["llm"]["temperature"]),
        tts=sarvam.TTS(model=LK["tts"]["model"], speaker=LK["tts"]["speaker"],
                       target_language_code=LK["tts"]["language"]),
        turn_handling=TurnHandlingOptions(turn_detection=inference.TurnDetector(),
                                          preemptive_generation={"enabled": True}),
    )

    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        if getattr(item, "type", "message") != "message":
            return
        turns.append({
            "role": item.role,
            "text": item.text_content or "",
            "interrupted": bool(item.interrupted),
            "metrics": {k: v for k, v in dict(item.metrics or {}).items() if isinstance(v, (int, float))},
        })

    async def _upload() -> None:
        payload = {"call_id": ctx.room.name, "started_at": started, "ended_at": time.time(),
                   "agent_version": S["active_version"], "turns": turns}
        token = f"?token={config.INGEST_TOKEN}" if config.INGEST_TOKEN else ""
        try:
            async with httpx.AsyncClient(timeout=20) as c:
                r = await c.post(f"{TARGET}/ingest/livekit{token}", json=payload)
            log.info("uploaded %d turns to CallDoctor: %s", len(turns), r.status_code)
        except httpx.HTTPError as e:
            log.error("upload to CallDoctor failed: %s", e)

    ctx.add_shutdown_callback(_upload)

    await session.start(agent=SurveyAgent(), room=ctx.room)
    await ctx.connect()
    await session.say(S["first_message"].strip())

    async def _cap() -> None:
        await asyncio.sleep(S["max_call_seconds"])
        await session.aclose()
        ctx.shutdown(reason="max duration")

    cap = asyncio.create_task(_cap())

    async def _cancel() -> None:
        cap.cancel()

    ctx.add_shutdown_callback(_cancel)


if __name__ == "__main__":
    cli.run_app(server)
