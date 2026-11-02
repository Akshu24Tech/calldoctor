"""Offline tests: no network, no API keys. The LLM is replaced by fakes."""

from __future__ import annotations

import os

os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["CALLDOCTOR_TOKEN"] = "test-token"
os.environ.setdefault("GROQ_API_KEY", "fake")

import pytest
from fastapi.testclient import TestClient

from calldoctor import config, judge, rules, simulate, store
from calldoctor.analyze import analyze
from calldoctor.ingest import from_livekit, from_retell, from_vapi
from calldoctor.models import Call, Turn

config.INGEST_TOKEN = "test-token"


def make_call(*pairs, **kw) -> Call:
    return Call(id=kw.pop("id", "t1"), platform="simulation", source="test",
                turns=[Turn(r, t) for r, t in pairs], **kw)


# ---------------------------------------------------------------- rules
def test_repeated_question_flagged():
    c = make_call(("agent", "Which service did you use today?"), ("user", "Aadhaar."),
                  ("agent", "Thanks. Which service did you use?"))
    assert [f["category"] for f in rules.repeated_question(c)] == ["repeated_question"]


def test_repeat_after_silence_is_allowed():
    c = make_call(("agent", "Which service did you use today?"), ("user", ""),
                  ("agent", "Which service did you use today?"))
    c.turns[1].silent = True
    assert rules.repeated_question(c) == []


def test_sensitive_request_vs_reassurance():
    bad = make_call(("agent", "Please tell me your Aadhaar number to verify."))
    good = make_call(("agent", "I will never ask for your Aadhaar number or OTP."))
    assert rules.sensitive_request(bad)[0]["severity"] == "high"
    assert rules.sensitive_request(good) == []


def test_political_refusal_not_flagged():
    assert rules.political(make_call(("agent", "I can't discuss elections, only your visit."))) == []
    assert rules.political(make_call(("agent", "You should vote for whoever fixes offices.")))


def test_interruption_ignored():
    c = make_call(("agent", "Could you tell me how long you waited at the centre before"),
                  ("user", "Just ask fast."),
                  ("agent", "Could you tell me how long you waited at the centre before being served?"))
    c.turns[0].interrupted = True
    assert rules.interruption_ignored(c)[0]["turn"] == 2


def test_voice_style():
    long = " ".join(["word"] * 60)
    cats = {f["category"] for f in rules.voice_style(make_call(("agent", long + " Ok? Sure?")))}
    assert cats == {"long_turn", "multiple_questions"}


def test_latency_spike():
    c = make_call(("user", "hi"), ("agent", "hello"))
    c.turns[0].end_s, c.turns[1].start_s = 1.0, 5.2
    assert rules.latency(c)[0]["category"] == "latency_spike"


# ---------------------------------------------------------------- verifier
def test_judge_quote_must_exist():
    c = make_call(("user", "fix it"), ("agent", "Don't worry, this will be fixed by tomorrow."))
    real = judge.verify_evidence(c, {"turn": 1, "evidence": "this will be fixed by tomorrow"})
    fake = judge.verify_evidence(c, {"turn": 1, "evidence": "I guarantee a refund"})
    moved = judge.verify_evidence(c, {"turn": 0, "evidence": "will be fixed"})
    assert real["verified"] and not fake["verified"]
    assert moved["verified"] and moved["turn"] == 1


def test_slot_checks():
    c = make_call(("user", "Aadhaar, waited one hour, done, I'd give four"))
    st = judge.check_slots(c, {"service": "aadhaar", "wait_minutes": 60, "resolved": True,
                               "rating": 5, "complaint": None})
    assert st == {"service": "ok", "wait_minutes": "ok", "resolved": "ok",
                  "rating": "ungrounded", "complaint": "missing"}
    assert judge.check_slots(c, {"service": "passport", "rating": 9})["service"] == "invalid"


def test_analyze_with_fake_judge(monkeypatch):
    monkeypatch.setattr(judge, "chat_json", lambda m, model, **k: {
        "caller_declined": False,
        "slots": {"service": "aadhaar", "wait_minutes": None, "resolved": True, "rating": 4, "complaint": "none"},
        "findings": [
            {"category": "false_promise", "turn": 3, "evidence": "it will be fixed", "message": "promised"},
            {"category": "hallucination", "turn": 3, "evidence": "invented words", "message": "made up"},
        ],
    })
    c = make_call(("agent", "Which service?"), ("user", "Aadhaar, rating four, done"),
                  ("agent", "Thanks."), ("agent", "Don't worry, it will be fixed soon."))
    a = analyze(c)
    by = {f["category"]: f for f in a["findings"]}
    assert by["false_promise"]["verified"] and not by["hallucination"]["verified"]
    assert by["survey_incomplete"]["message"].endswith("wait_minutes")
    assert a["failed"] and a["score"] == 100 - 25 - 10


# ---------------------------------------------------------------- ingest
def test_ingest_vapi():
    c = from_vapi({"message": {"type": "end-of-call-report", "cost": 0.07, "endedReason": "customer-ended-call",
                               "call": {"id": "abc", "startedAt": "2026-11-02T10:00:00Z", "endedAt": "2026-11-02T10:01:00Z"},
                               "artifact": {"messages": [{"role": "system", "message": "sys"},
                                                         {"role": "bot", "message": "Hi", "secondsFromStart": 0.5, "duration": 900},
                                                         {"role": "user", "message": "Hello", "secondsFromStart": 2, "duration": 500}]}}})
    assert c.id == "vapi-abc" and [t.role for t in c.turns] == ["agent", "user"]
    assert c.turns[0].end_s == pytest.approx(1.4) and c.duration_s == 60
    assert from_vapi({"message": {"type": "status-update"}}) is None


def test_ingest_retell():
    c = from_retell({"event": "call_analyzed", "call": {
        "call_id": "r1", "start_timestamp": 1_700_000_000_000, "duration_ms": 30000,
        "call_cost": {"combined_cost": 9}, "transcript_object": [
            {"role": "agent", "content": "Hi", "words": [{"start": 0.1, "end": 0.4}]},
            {"role": "user", "content": "Yes", "words": [{"start": 1.0, "end": 1.2}]}]}})
    assert c.id == "retell-r1" and c.cost_usd == 0.09 and c.turns[1].end_s == 1.2
    assert from_retell({"event": "call_started", "call": {}}) is None


def test_ingest_livekit():
    c = from_livekit({"call_id": "room1", "started_at": 100.0, "ended_at": 160.0, "turns": [
        {"role": "assistant", "text": "Hi", "interrupted": True, "metrics": {"started_speaking_at": 101.0}},
        {"role": "user", "text": "ok", "metrics": {"stopped_speaking_at": 103.5}}]})
    assert c.turns[0].interrupted and c.turns[0].start_s == 1.0 and c.turns[1].end_s == 3.5


# ---------------------------------------------------------------- simulation
def test_simulation_loop(monkeypatch):
    caller = iter(["Yes, go ahead.", "[INTERRUPT] Just ask quickly!", "[SILENT]", "Bye [HANGUP]"])
    agent = iter(["Great, which service did you use at the centre and how long did you wait there today?",
                  "Sorry. Which service?", "Hello, can you hear me?"])

    def fake_chat(messages, model, **kw):
        return next(caller) if "ordinary Indian citizen" in messages[0]["content"] else next(agent)

    monkeypatch.setattr(simulate, "chat", fake_chat)
    c = simulate.simulate("interrupter", "v2")
    roles = [(t.role, t.interrupted, t.silent) for t in c.turns]
    assert roles[1] == ("user", False, False)
    assert roles[2] == ("agent", True, False)          # cut off by the interrupt
    assert ("user", False, True) in roles               # silence recorded
    assert c.ended_reason == "customer-ended-call" and c.turns[-1].text == "Bye"


# ---------------------------------------------------------------- API
@pytest.fixture
def client(monkeypatch):
    import calldoctor.app as appmod
    monkeypatch.setattr(appmod, "analyze", lambda call: analyze(call, use_judge=False))
    return TestClient(appmod.app)


def test_webhook_requires_token(client):
    assert client.post("/webhooks/retell", json={}).status_code == 401


def test_webhook_to_dashboard(client):
    payload = {"event": "call_ended", "call": {"call_id": "api1", "transcript_object": [
        {"role": "agent", "content": "Please share your OTP to continue.", "words": []},
        {"role": "user", "content": "No.", "words": []}]}}
    assert client.post("/webhooks/retell?token=test-token", json=payload).json()["call_id"] == "retell-api1"
    detail = client.get("/api/calls/retell-api1").json()
    assert detail["analysis"]["failed"] is True
    assert detail["call"]["agent_version"] == config.survey()["active_version"]
    assert client.get("/api/stats").json()["platforms"]["retell"]["calls"] >= 1
    assert "CallDoctor" in client.get("/").text
    meta = client.get("/api/meta").json()
    assert set(meta["versions"]) >= {"v1", "v2"} and len(meta["personas"]) == 9


def test_suite_storage():
    sid = store.create_suite("v1")
    store.finish_suite(sid, {"calls": 1, "failed": 1, "failure_rate": 100, "avg_score": 50,
                             "categories": {"x": 1}, "personas": {}, "unverified_judge_claims": 0})
    assert store.latest_suite("v1")["id"] == sid
