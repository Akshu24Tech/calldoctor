"""Run every detector on a call and produce one analysis record."""

from __future__ import annotations

import time
from typing import Any

from calldoctor.judge import LLMError, check_slots, judge
from calldoctor.models import Call
from calldoctor.rules import run_rules

PENALTY = {"high": 25, "medium": 10, "low": 3}


def analyze(call: Call, use_judge: bool = True) -> dict[str, Any]:
    findings = run_rules(call)
    slots, slot_status, declined, judge_error = {}, {}, False, None

    if use_judge and call.turns:
        try:
            j = judge(call)
            declined, slots = j["caller_declined"], j["slots"]
            seen = {(f["category"], f["turn"]) for f in findings}
            findings += [f for f in j["findings"] if (f["category"], f["turn"]) not in seen]
            slot_status = check_slots(call, slots)
        except LLMError as e:
            judge_error = str(e)

    if slot_status and not declined:
        missing = [k for k, v in slot_status.items() if v == "missing"]
        if missing:
            findings.append({
                "detector": "verifier", "category": "survey_incomplete", "severity": "medium",
                "turn": None, "message": "Never collected: " + ", ".join(missing),
                "evidence": "", "verified": True,
            })
        for k, v in slot_status.items():
            if v in ("invalid", "ungrounded"):
                findings.append({
                    "detector": "verifier", "category": f"answer_{v}", "severity": "low",
                    "turn": None, "evidence": str(slots.get(k)), "verified": True,
                    "message": (f"'{k}' = {slots.get(k)!r} is not a valid value" if v == "invalid"
                                else f"'{k}' = {slots.get(k)!r} was never said by the caller"),
                })

    counted = [f for f in findings if f.get("verified")]
    score = max(0, 100 - sum(PENALTY[f["severity"]] for f in counted))
    return {
        "analyzed_at": time.time(),
        "findings": sorted(findings, key=lambda f: (f["turn"] is None, f["turn"] or 0)),
        "slots": slots,
        "slot_status": slot_status,
        "caller_declined": declined,
        "failed": any(f["severity"] in ("high", "medium") for f in counted),
        "score": score,
        "judge_error": judge_error,
    }
