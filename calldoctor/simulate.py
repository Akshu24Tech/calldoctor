"""Simulated callers: an LLM plays a difficult citizen against the survey agent.

Runs entirely in text on the free Groq tier, so a full regression suite
costs no voice minutes. The agent side uses exactly the prompt that is
deployed to the voice platforms.
"""

from __future__ import annotations

import time
import uuid
from collections import Counter
from typing import Any, Callable

from calldoctor import config, store
from calldoctor.analyze import analyze
from calldoctor.llm import chat
from calldoctor.models import Call, Turn

END_TAG = "[END_CALL]"
SIM_SUFFIX = (
    "\n\n# Simulation note\nThis call is a text simulation. When you end the call, "
    f"put {END_TAG} at the very end of your final message."
)
MAX_USER_TURNS = 12


def _caller_prompt(p: dict) -> str:
    return f"""You are an ordinary Indian citizen who just picked up a phone call from a survey agent.
Your facts (only reveal them when it makes sense): {p['profile']}
How you behave: {p['style']}

Rules:
- Reply as spoken words only, usually one short sentence. No stage directions.
- Stay in character. Never mention that you are an AI or a simulation.
- Optional markers, only when your behaviour calls for them:
  start with [INTERRUPT] to cut the agent off mid-sentence,
  reply exactly [SILENT] to say nothing,
  reply [HANGUP] to end the call.
- When the agent says goodbye or ends the call, reply [HANGUP]."""


def _cut(text: str) -> str:
    words = text.split()
    keep = max(3, int(len(words) * 0.45))
    return " ".join(words[:keep]) + "—" if len(words) > keep else text


def simulate(persona_id: str, version: str) -> Call:
    persona = config.personas()[persona_id]
    first = config.survey()["first_message"].strip()
    agent_msgs = [{"role": "system", "content": config.system_prompt(version) + SIM_SUFFIX},
                  {"role": "assistant", "content": first}]
    caller_msgs = [{"role": "system", "content": _caller_prompt(persona)},
                   {"role": "user", "content": first}]
    turns = [Turn("agent", first)]
    ended_reason = "max_turns"

    for _ in range(MAX_USER_TURNS):
        said = chat(caller_msgs, config.CALLER_MODEL, temperature=0.8, max_tokens=120).strip()
        caller_msgs.append({"role": "assistant", "content": said})

        hangup = "[HANGUP]" in said
        said = said.replace("[HANGUP]", "").strip()
        if said.startswith("[INTERRUPT]"):
            said = said.replace("[INTERRUPT]", "", 1).strip()
            turns[-1].text = _cut(turns[-1].text)
            turns[-1].interrupted = True
            agent_msgs[-1]["content"] = turns[-1].text
        silent = said == "[SILENT]" or not said
        if silent:
            said = ""
        if said or (silent and not hangup):
            turns.append(Turn("user", said, silent=silent))
        if hangup:
            ended_reason = "customer-ended-call"
            break

        agent_msgs.append({"role": "user", "content": said or "(the caller says nothing)"})
        reply = chat(agent_msgs, config.agent_model(), temperature=config.survey()["llm"]["temperature"],
                     max_tokens=220).strip()
        ended = END_TAG in reply
        reply = reply.replace(END_TAG, "").strip()
        agent_msgs.append({"role": "assistant", "content": reply})
        turns.append(Turn("agent", reply))
        caller_msgs.append({"role": "user", "content": reply})
        if ended:
            ended_reason = "assistant-ended-call"
            break

    return Call(
        id=f"sim-{version}-{persona_id}-{uuid.uuid4().hex[:6]}",
        platform="simulation",
        source="simulation",
        turns=turns,
        agent_version=version,
        persona=persona_id,
        created_at=time.time(),
        ended_reason=ended_reason,
        cost_usd=0.0,
        meta={"persona_targets": persona.get("targets", [])},
    )


def summarize(results: list[tuple[Call, dict]]) -> dict[str, Any]:
    cats: Counter = Counter()
    per_persona = {}
    for call, a in results:
        verified = [f for f in a["findings"] if f.get("verified")]
        cats.update(f["category"] for f in verified)
        per_persona[call.persona] = {"failed": a["failed"], "score": a["score"],
                                     "categories": sorted({f["category"] for f in verified})}
    n = len(results)
    return {
        "calls": n,
        "failed": sum(1 for _, a in results if a["failed"]),
        "failure_rate": round(100 * sum(1 for _, a in results if a["failed"]) / n) if n else 0,
        "avg_score": round(sum(a["score"] for _, a in results) / n) if n else 0,
        "categories": dict(cats.most_common()),
        "personas": per_persona,
        "unverified_judge_claims": sum(1 for _, a in results for f in a["findings"] if not f.get("verified")),
    }


def run_suite(version: str, persona_ids: list[str] | None = None,
              log: Callable[[str], None] = print, suite_id: int | None = None) -> dict[str, Any]:
    persona_ids = persona_ids or list(config.personas())
    suite_id = suite_id or store.create_suite(version)
    results = []
    try:
        for pid in persona_ids:
            log(f"[{version}] simulating {pid} …")
            call = simulate(pid, version)
            analysis = analyze(call)
            store.save_call(call, analysis, suite_id=suite_id)
            results.append((call, analysis))
            cats = sorted({f["category"] for f in analysis["findings"] if f.get("verified")})
            log(f"    {'FAIL' if analysis['failed'] else 'pass'}  score {analysis['score']:>3}  {', '.join(cats)}")
        summary = summarize(results)
        summary["suite_id"] = suite_id
        store.finish_suite(suite_id, summary)
        return summary
    except Exception as e:
        store.finish_suite(suite_id, {"error": str(e), **summarize(results)}, status="error")
        raise
