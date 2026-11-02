"""CallDoctor web app: webhooks in, failure reports out.

Webhooks (append ?token=<CALLDOCTOR_TOKEN>):
  POST /webhooks/vapi      Vapi end-of-call-report
  POST /webhooks/retell    Retell call_ended / call_analyzed
  POST /ingest/livekit     transcript posted by our LiveKit agent

Dashboard + JSON API:
  GET  /                   dashboard
  GET  /api/stats, /api/calls, /api/calls/{id}, /api/suites, /api/meta
  POST /api/simulate       run the persona suite (needs token)
"""

from __future__ import annotations

import hmac
import logging
import threading
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from calldoctor import config, ingest, store
from calldoctor.analyze import analyze
from calldoctor.models import Call

log = logging.getLogger("calldoctor")
STATIC = Path(__file__).parent / "static"
app = FastAPI(title="CallDoctor", version="1.0")
_sim_lock = threading.Lock()


# ------------------------------------------------------------------ auth
def check_token(token: str | None) -> None:
    expected = config.INGEST_TOKEN
    if not expected:
        return  # local dev without a token
    if not token or not hmac.compare_digest(token, expected):
        raise HTTPException(401, "bad or missing token")


def _process(call: Call) -> None:
    # Live calls are made against whatever prompt version is deployed.
    call.agent_version = call.agent_version or config.survey()["active_version"]
    try:
        store.save_call(call)  # store first, so nothing is lost if the judge fails
        store.save_call(call, analyze(call))
    except Exception:  # never let a background job crash the server
        log.exception("analysis failed for %s", call.id)


# ------------------------------------------------------------------ webhooks
@app.post("/webhooks/vapi")
async def vapi_webhook(request: Request, bg: BackgroundTasks, token: str | None = Query(None),
                       x_vapi_secret: str | None = Header(None)):
    check_token(token or x_vapi_secret)
    call = ingest.from_vapi(await request.json())
    if call and call.turns:
        bg.add_task(_process, call)
        return {"ok": True, "call_id": call.id}
    return {"ok": True, "ignored": True}


@app.post("/webhooks/retell")
async def retell_webhook(request: Request, bg: BackgroundTasks, token: str | None = Query(None)):
    check_token(token)
    call = ingest.from_retell(await request.json())
    if call and call.turns:
        bg.add_task(_process, call)
        return {"ok": True, "call_id": call.id}
    return {"ok": True, "ignored": True}


@app.post("/ingest/livekit")
async def livekit_ingest(request: Request, bg: BackgroundTasks, token: str | None = Query(None)):
    check_token(token)
    call = ingest.from_livekit(await request.json())
    if call.turns:
        bg.add_task(_process, call)
    return {"ok": True, "call_id": call.id}


# ------------------------------------------------------------------ API
@app.get("/api/meta")
def meta() -> dict[str, Any]:
    s = config.survey()
    return {
        "agent": s["display_name"],
        "active_version": s["active_version"],
        "versions": config.prompt_versions(),
        "personas": [{"id": p["id"], "targets": p.get("targets", [])} for p in config.personas().values()],
        "required_fields": list(s["required_fields"]),
        "token_required": bool(config.INGEST_TOKEN),
        "simulation_enabled": bool(config.GROQ_API_KEY),
    }


@app.get("/api/stats")
def stats():
    return store.stats()


@app.get("/api/calls")
def calls(platform: str | None = None, version: str | None = None, failed: bool | None = None,
          suite_id: int | None = None, limit: int = 200):
    return store.list_calls(platform, version, failed, suite_id, min(limit, 1000))


@app.get("/api/calls/{call_id}")
def call_detail(call_id: str):
    rec = store.get_call(call_id)
    if not rec:
        raise HTTPException(404, "call not found")
    return rec


@app.post("/api/calls/{call_id}/reanalyze")
def reanalyze(call_id: str, bg: BackgroundTasks, x_calldoctor_token: str | None = Header(None)):
    check_token(x_calldoctor_token)
    rec = store.get_call(call_id)
    if not rec:
        raise HTTPException(404, "call not found")
    bg.add_task(_process, Call.from_dict(rec["call"]))
    return {"ok": True}


@app.get("/api/suites")
def suites():
    return store.list_suites()


class SimRequest(BaseModel):
    version: str
    personas: list[str] | None = None


@app.post("/api/simulate")
def simulate(req: SimRequest, x_calldoctor_token: str | None = Header(None)):
    check_token(x_calldoctor_token)
    if not config.GROQ_API_KEY:
        raise HTTPException(400, "GROQ_API_KEY is not set on the server")
    if req.version not in config.prompt_versions():
        raise HTTPException(400, f"unknown version {req.version}")
    unknown = set(req.personas or []) - set(config.personas())
    if unknown:
        raise HTTPException(400, f"unknown personas {sorted(unknown)}")
    if not _sim_lock.acquire(blocking=False):
        raise HTTPException(409, "a simulation is already running")

    suite_id = store.create_suite(req.version)

    def job():
        from calldoctor.simulate import run_suite
        try:
            run_suite(req.version, req.personas, log=log.info, suite_id=suite_id)
        except Exception:
            log.exception("suite %s failed", suite_id)
        finally:
            _sim_lock.release()

    threading.Thread(target=job, daemon=True).start()
    return {"ok": True, "suite_id": suite_id}


@app.get("/healthz")
def healthz():
    return {"ok": True, "calls": store.count_calls()}


# ------------------------------------------------------------------ dashboard
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.exception_handler(ValueError)
async def value_error(_, exc: ValueError):
    return JSONResponse({"error": str(exc)}, status_code=400)
