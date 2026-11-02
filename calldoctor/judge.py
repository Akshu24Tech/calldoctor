"""LLM judge + the verifier that checks the judge.

The LLM does two jobs it is good at: extracting the survey answers and
spotting judgment-call failures (false promises, mishandled objections).
Code then checks everything the LLM claims:

  * every finding must quote words that really appear in that agent turn,
    otherwise it is marked unverified and does not count;
  * severity comes from a fixed table, not from the LLM;
  * every extracted answer is type- and range-checked against survey.yaml,
    and must be grounded in what the CALLER actually said.
"""

from __future__ import annotations

import re
from typing import Any

from calldoctor import config
from calldoctor.llm import LLMError, chat_json
from calldoctor.models import Call

SEVERITY = {
    "false_promise": "high",
    "hallucination": "high",
    "objection_mishandled": "medium",
    "ignored_caller": "medium",
    "wrong_language": "medium",
    "pushy_after_decline": "medium",
    "rude_or_argumentative": "medium",
    "other": "low",
}

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5,
}


def _judge_prompt() -> str:
    fields = config.survey()["required_fields"]
    field_lines = "\n".join(f'  "{k}": {v["description"]}'
                            + (f' (one of {v["allowed"]})' if "allowed" in v else "")
                            for k, v in fields.items())
    cats = "\n".join(f"- {c}" for c in SEVERITY)
    return f"""You review phone calls made by a survey voice agent for a government service feedback desk.
The agent should collect five answers, stay polite, handle objections, never promise outcomes,
never invent facts, and respect a caller who declines.

Return JSON only, with this shape:
{{
  "caller_declined": true|false,
  "slots": {{
{field_lines}
  }},
  "findings": [
    {{"category": "<one of the categories>", "turn": <turn number>,
      "evidence": "<exact words copied from that AGENT turn>", "message": "<one sentence>"}}
  ]
}}

Rules:
- slots: use the CALLER's answers only. Use null if the caller never gave it. wait_minutes and
  rating are integers, resolved is true/false, service is one of the allowed values.
- findings: only real agent failures. Categories:
{cats}
- evidence MUST be copied word for word from the agent turn you cite. No paraphrasing.
- Do not report style issues (long turns, several questions, repeated questions); code checks those.
- If the agent did nothing wrong, return an empty findings list."""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def verify_evidence(call: Call, finding: dict) -> dict:
    """Mark a judge finding verified only if its quote exists in an agent turn."""
    ev = _norm(finding.get("evidence") or "")
    turn = finding.get("turn")
    finding["verified"] = False
    if len(ev.split()) < 2:
        finding["verify_note"] = "no usable quote"
        return finding
    candidates = []
    if isinstance(turn, int) and 0 <= turn < len(call.turns):
        candidates.append(turn)
    candidates += [i for i, _ in call.agent_turns() if i != turn]
    for i in candidates:
        t = call.turns[i]
        if t.role == "agent" and ev in _norm(t.text):
            finding["verified"] = True
            if i != turn:
                finding["verify_note"] = f"quote found in turn {i}, not {turn}"
            finding["turn"] = i
            return finding
    finding["verify_note"] = "quote not found in any agent turn"
    return finding


def _user_text(call: Call) -> str:
    return " ".join(t.text for _, t in call.user_turns()).lower()


def _numbers_said(text: str) -> set[int]:
    nums = {int(n) for n in re.findall(r"\b\d+\b", text)}
    nums |= {v for w, v in NUMBER_WORDS.items() if re.search(rf"\b{w}\b", text)}
    return nums


def check_slots(call: Call, slots: dict[str, Any]) -> dict[str, str]:
    """Return a status per field: ok | missing | invalid | ungrounded."""
    spec = config.survey()["required_fields"]
    said = _user_text(call)
    nums = _numbers_said(said)
    status = {}
    for name, rule in spec.items():
        v = slots.get(name)
        if v in (None, "", "null"):
            status[name] = "missing"
            continue
        kind = rule.get("type", "str")
        if "allowed" in rule and v not in rule["allowed"]:
            status[name] = "invalid"
        elif kind == "int" and not (isinstance(v, int) and not isinstance(v, bool)
                                    and rule.get("min", -1e9) <= v <= rule.get("max", 1e9)):
            status[name] = "invalid"
        elif kind == "bool" and not isinstance(v, bool):
            status[name] = "invalid"
        elif name == "rating" and v not in nums:
            status[name] = "ungrounded"   # the caller never said this number
        elif name == "wait_minutes" and not (nums or re.search(r"hour|ghant|minute|min\b", said)):
            status[name] = "ungrounded"
        else:
            status[name] = "ok"
    return status


def judge(call: Call) -> dict[str, Any]:
    verdict = chat_json(
        [{"role": "system", "content": _judge_prompt()},
         {"role": "user", "content": "TRANSCRIPT:\n" + call.transcript()}],
        config.JUDGE_MODEL,
    )
    findings = []
    for f in verdict.get("findings") or []:
        category = f.get("category") if f.get("category") in SEVERITY else "other"
        finding = {
            "detector": "judge",
            "category": category,
            "severity": SEVERITY[category],
            "turn": f.get("turn"),
            "message": str(f.get("message") or "")[:300],
            "evidence": str(f.get("evidence") or "")[:300],
        }
        findings.append(verify_evidence(call, finding))
    return {
        "caller_declined": bool(verdict.get("caller_declined")),
        "slots": verdict.get("slots") or {},
        "findings": findings,
    }


__all__ = ["LLMError", "check_slots", "judge", "verify_evidence"]
