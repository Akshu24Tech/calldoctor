"""Deploy the survey agent to Vapi and Retell, wired to CallDoctor's webhooks.

    python -m calldoctor.platforms.deploy vapi
    python -m calldoctor.platforms.deploy retell
    python -m calldoctor.platforms.deploy all --dry-run

Requires PUBLIC_URL (your deployed CallDoctor URL) so the platforms know
where to send finished calls. Re-run after changing the prompt: updates in place.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

from calldoctor import config

STATE = config.ROOT / "data" / "deployed.json"


def _state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def _save(state: dict) -> None:
    Path(STATE).parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2))


def _key(name: str) -> str:
    v = os.environ.get(name, "")
    if not v:
        sys.exit(f"Missing {name} in .env")
    return v


def _req(method: str, url: str, key: str, body: dict) -> dict:
    r = httpx.request(method, url, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=30)
    if r.status_code >= 400:
        sys.exit(f"{method} {url} -> {r.status_code}: {r.text[:600]}")
    return r.json() if r.content else {}


def webhook(path: str) -> str:
    if not config.PUBLIC_URL:
        sys.exit("Set PUBLIC_URL in .env to your deployed CallDoctor URL first.")
    token = f"?token={config.INGEST_TOKEN}" if config.INGEST_TOKEN else ""
    return f"{config.PUBLIC_URL}{path}{token}"


# ---------------------------------------------------------------- Vapi
def vapi_payload() -> dict:
    s, v = config.survey(), config.survey()["platforms"]["vapi"]
    return {
        "name": f"calldoctor-{s['name']}-{s['active_version']}"[:40],
        "firstMessage": s["first_message"].strip(),
        "model": {
            "provider": "groq",
            "model": config.agent_model(),
            "temperature": s["llm"]["temperature"],
            "messages": [{"role": "system", "content": config.system_prompt()}],
        },
        "transcriber": v["transcriber"],
        "voice": v["voice"],
        "endCallFunctionEnabled": True,
        "maxDurationSeconds": s["max_call_seconds"],
        "server": {"url": webhook("/webhooks/vapi")},
        "serverMessages": ["end-of-call-report"],
        "metadata": {"agent_version": s["active_version"]},
    }


def deploy_vapi(dry: bool) -> None:
    body = vapi_payload()
    if dry:
        print(json.dumps(body, indent=2)[:1500])
        return
    key, state = _key("VAPI_API_KEY"), _state()
    aid = state.get("vapi_assistant_id")
    if aid:
        _req("PATCH", f"https://api.vapi.ai/assistant/{aid}", key, body)
        print("Updated Vapi assistant", aid)
    else:
        aid = _req("POST", "https://api.vapi.ai/assistant", key, body)["id"]
        print("Created Vapi assistant", aid)
    state["vapi_assistant_id"] = aid
    _save(state)


# ---------------------------------------------------------------- Retell
def retell_payloads(llm_id: str = "<llm_id>") -> tuple[dict, dict]:
    s, r = config.survey(), config.survey()["platforms"]["retell"]
    llm = {
        "model_config": {"type": "cascading", "primary": {"model": r["model"]}},
        "general_prompt": config.system_prompt(),
        "begin_message": s["first_message"].strip(),
        "start_speaker": "agent",
        "general_tools": [{"type": "end_call", "name": "end_call",
                           "description": "End the call when the survey is done or the caller wants to stop."}],
    }
    agent = {
        "agent_name": f"calldoctor-{s['name']}-{s['active_version']}",
        "response_engine": {"type": "retell-llm", "llm_id": llm_id},
        "voice_id": r["voice_id"],
        "language": r["language"],
        "webhook_url": webhook("/webhooks/retell"),
        "max_call_duration_ms": s["max_call_seconds"] * 1000,
    }
    return llm, agent


def deploy_retell(dry: bool) -> None:
    if dry:
        for p in retell_payloads():
            print(json.dumps(p, indent=2)[:1200])
        return
    key, state = _key("RETELL_API_KEY"), _state()
    base = "https://api.retellai.com"
    llm_body, _ = retell_payloads()
    llm_id = state.get("retell_llm_id")
    if llm_id:
        _req("PATCH", f"{base}/update-retell-llm/{llm_id}", key, llm_body)
    else:
        llm_id = _req("POST", f"{base}/create-retell-llm", key, llm_body)["llm_id"]
    _, agent_body = retell_payloads(llm_id)
    agent_id = state.get("retell_agent_id")
    if agent_id:
        _req("PATCH", f"{base}/update-agent/{agent_id}", key, agent_body)
        print("Updated Retell agent", agent_id)
    else:
        agent_id = _req("POST", f"{base}/create-agent", key, agent_body)["agent_id"]
        print("Created Retell agent", agent_id)
    state.update(retell_llm_id=llm_id, retell_agent_id=agent_id)
    _save(state)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("target", choices=["vapi", "retell", "all"])
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.target in ("vapi", "all"):
        deploy_vapi(a.dry_run)
    if a.target in ("retell", "all"):
        deploy_retell(a.dry_run)


if __name__ == "__main__":
    main()
