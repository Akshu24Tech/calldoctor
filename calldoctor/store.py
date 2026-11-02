"""Storage: SQLite locally, Postgres in production (set DATABASE_URL)."""

from __future__ import annotations

import json
import time
from collections import Counter
from typing import Any

from sqlalchemy import (Column, Float, Integer, MetaData, String, Table, Text,
                        create_engine, delete, func, insert, select, update)
from sqlalchemy.pool import StaticPool

from calldoctor.config import database_url
from calldoctor.models import Call

meta = MetaData()

calls = Table(
    "calls", meta,
    Column("id", String(120), primary_key=True),
    Column("platform", String(20), index=True),
    Column("source", String(20)),
    Column("agent_version", String(40), index=True),
    Column("persona", String(60)),
    Column("suite_id", Integer, index=True),
    Column("created_at", Float, index=True),
    Column("call_json", Text),
    Column("analysis_json", Text),
    Column("failed", Integer),
    Column("score", Integer),
)

suites = Table(
    "suites", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("version", String(40)),
    Column("status", String(20)),          # running | done | error
    Column("created_at", Float),
    Column("summary_json", Text),
)

_engine = None


def engine():
    global _engine
    if _engine is None:
        url = database_url()
        if url.startswith("sqlite"):
            # one shared connection for in-memory DBs; allow use from the webhook/background threads
            kw = {"connect_args": {"check_same_thread": False}}
            if ":memory:" in url:
                kw["poolclass"] = StaticPool
            _engine = create_engine(url, **kw)
        else:
            _engine = create_engine(url, pool_pre_ping=True)
        meta.create_all(_engine)
    return _engine


def save_call(call: Call, analysis: dict | None = None, suite_id: int | None = None) -> None:
    row = {
        "platform": call.platform, "source": call.source, "agent_version": call.agent_version,
        "persona": call.persona, "suite_id": suite_id, "created_at": call.created_at or time.time(),
        "call_json": json.dumps(call.to_dict(), ensure_ascii=False),
        "analysis_json": json.dumps(analysis, ensure_ascii=False) if analysis else None,
        "failed": int(analysis["failed"]) if analysis else None,
        "score": analysis["score"] if analysis else None,
    }
    with engine().begin() as c:
        exists = c.execute(select(calls.c.id).where(calls.c.id == call.id)).first()
        if exists:
            c.execute(update(calls).where(calls.c.id == call.id).values(**row))
        else:
            c.execute(insert(calls).values(id=call.id, **row))


def get_call(call_id: str) -> dict | None:
    with engine().connect() as c:
        r = c.execute(select(calls).where(calls.c.id == call_id)).mappings().first()
    if not r:
        return None
    return {"call": json.loads(r["call_json"]),
            "analysis": json.loads(r["analysis_json"]) if r["analysis_json"] else None,
            "suite_id": r["suite_id"]}


def list_calls(platform: str | None = None, version: str | None = None, failed: bool | None = None,
               suite_id: int | None = None, limit: int = 200) -> list[dict]:
    q = select(calls.c.id, calls.c.platform, calls.c.source, calls.c.agent_version, calls.c.persona,
               calls.c.suite_id, calls.c.created_at, calls.c.failed, calls.c.score, calls.c.analysis_json)
    if platform:
        q = q.where(calls.c.platform == platform)
    if version:
        q = q.where(calls.c.agent_version == version)
    if failed is not None:
        q = q.where(calls.c.failed == int(failed))
    if suite_id is not None:
        q = q.where(calls.c.suite_id == suite_id)
    q = q.order_by(calls.c.created_at.desc()).limit(limit)
    out = []
    with engine().connect() as c:
        for r in c.execute(q).mappings():
            a = json.loads(r["analysis_json"]) if r["analysis_json"] else {}
            cats = Counter(f["category"] for f in a.get("findings", []) if f.get("verified"))
            row = {k: r[k] for k in ("id", "platform", "source", "agent_version", "persona",
                                     "suite_id", "created_at", "failed", "score")}
            row["categories"] = dict(cats)
            out.append(row)
    return out


def stats() -> dict[str, Any]:
    by_platform: dict[str, dict] = {}
    categories: Counter = Counter()
    for row in list_calls(limit=5000):
        p = by_platform.setdefault(row["platform"], {"calls": 0, "failed": 0, "score_sum": 0, "scored": 0})
        p["calls"] += 1
        p["failed"] += row["failed"] or 0
        if row["score"] is not None:
            p["score_sum"] += row["score"]
            p["scored"] += 1
        categories.update(row["categories"])
    for p in by_platform.values():
        p["failure_rate"] = round(100 * p["failed"] / p["calls"]) if p["calls"] else 0
        p["avg_score"] = round(p.pop("score_sum") / p["scored"]) if p["scored"] else None
        p.pop("scored")
    return {"platforms": by_platform, "categories": dict(categories.most_common())}


# ---------------------------------------------------------------- suites
def create_suite(version: str) -> int:
    with engine().begin() as c:
        res = c.execute(insert(suites).values(version=version, status="running", created_at=time.time(),
                                              summary_json=None))
        return int(res.inserted_primary_key[0])


def finish_suite(suite_id: int, summary: dict, status: str = "done") -> None:
    with engine().begin() as c:
        c.execute(update(suites).where(suites.c.id == suite_id)
                  .values(status=status, summary_json=json.dumps(summary)))


def list_suites() -> list[dict]:
    with engine().connect() as c:
        rows = c.execute(select(suites).order_by(suites.c.id.desc()).limit(50)).mappings().all()
    return [{"id": r["id"], "version": r["version"], "status": r["status"], "created_at": r["created_at"],
             "summary": json.loads(r["summary_json"]) if r["summary_json"] else None} for r in rows]


def latest_suite(version: str) -> dict | None:
    return next((s for s in list_suites() if s["version"] == version and s["status"] == "done"), None)


def count_calls() -> int:
    with engine().connect() as c:
        return c.execute(select(func.count()).select_from(calls)).scalar_one()


def delete_all() -> None:
    with engine().begin() as c:
        c.execute(delete(calls))
        c.execute(delete(suites))
