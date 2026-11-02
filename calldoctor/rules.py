"""Deterministic detectors. No LLM involved, so these findings are always
reproducible and need no verification. Each returns a list of findings:

    {detector, category, severity, turn, message, evidence, verified}
"""

from __future__ import annotations

import re

from calldoctor.models import Call

STOP = set(
    "a an the to of and or is are was were be you your i me my we our it this that "
    "for in on at with do did does can could would will please so just okay ok "
    "what which how".split()
)
NEGATION = re.compile(r"\b(never|not|don't|do not|won't|will not|no need|can't|cannot)\b", re.I)
SENSITIVE = re.compile(
    r"\b(aadhaar (number|no|card number)|otp|one[- ]time password|bank (account|details)|"
    r"account number|ifsc|cvv|atm pin|upi (id|pin)|password)\b",
    re.I,
)
POLITICAL = re.compile(
    r"\b(bjp|congress|aap|aam aadmi|samajwadi|bsp|tmc|shiv sena|vote for|voting for|"
    r"your vote|election|modi|rahul gandhi|kejriwal)\b",
    re.I,
)
REPAIR = re.compile(r"^\W*(hello|hello\?|what|sorry|huh|pardon|come again|repeat)\W*", re.I)

MAX_WORDS = 45
LATENCY_SPIKE_MS = 3000


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP}


def _questions(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip().endswith("?")]


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _f(category, severity, turn, message, evidence, detector="rules"):
    return {"detector": detector, "category": category, "severity": severity, "turn": turn,
            "message": message, "evidence": evidence, "verified": True}


def repeated_question(call: Call) -> list[dict]:
    """Agent asks (almost) the same question again, and it wasn't a repair after silence."""
    out, asked = [], []  # asked: (turn, tokens, text)
    for i, t in call.agent_turns():
        prev_user = call.turns[i - 1] if i > 0 and call.turns[i - 1].role == "user" else None
        is_repair = prev_user is not None and (prev_user.silent or not prev_user.text.strip()
                                               or REPAIR.match(prev_user.text))
        for q in _questions(t.text):
            toks = _tokens(q)
            if len(toks) < 2:
                continue
            match = next((a for a in asked if _jaccard(toks, a[1]) >= 0.6), None)
            if match and not is_repair:
                out.append(_f("repeated_question", "medium", i,
                              f"Asked again (first asked in turn {match[0]})", q))
            asked.append((i, toks, q))
    return out


def sensitive_request(call: Call) -> list[dict]:
    out = []
    for i, t in call.agent_turns():
        for s in _sentences(t.text):
            if SENSITIVE.search(s) and not NEGATION.search(s):
                out.append(_f("sensitive_data_request", "high", i,
                              "Agent mentions sensitive data without saying it won't ask for it", s))
    return out


def political(call: Call) -> list[dict]:
    out = []
    for i, t in call.agent_turns():
        for s in _sentences(t.text):
            if POLITICAL.search(s) and not NEGATION.search(s) and "only" not in s.lower():
                out.append(_f("political_content", "high", i, "Agent engages with political content", s))
    return out


def voice_style(call: Call) -> list[dict]:
    out = []
    for i, t in call.agent_turns():
        words = len(t.text.split())
        if words > MAX_WORDS:
            out.append(_f("long_turn", "low", i, f"{words} words in one spoken turn (limit {MAX_WORDS})",
                          " ".join(t.text.split()[:12]) + " …"))
        if t.text.count("?") > 1:
            out.append(_f("multiple_questions", "low", i, f"{t.text.count('?')} questions in one turn",
                          t.text))
        if re.search(r"(^|\n)\s*(\d+[.)]|[-*•])\s", t.text) or "**" in t.text:
            out.append(_f("formatting_in_speech", "low", i, "List or markdown formatting in a spoken reply",
                          t.text[:80]))
    return out


def interruption_ignored(call: Call) -> list[dict]:
    """After being cut off, the agent restarts the same sentence instead of answering."""
    out = []
    turns = call.turns
    for i, t in enumerate(turns):
        if t.role != "agent" or not t.interrupted:
            continue
        nxt = next((j for j in range(i + 2, len(turns)) if turns[j].role == "agent"), None)
        if nxt is None:
            continue
        cut, after = _tokens(t.text), _tokens(turns[nxt].text)
        if cut and len(cut & after) / len(cut) >= 0.6:
            out.append(_f("interruption_ignored", "medium", nxt,
                          f"Repeats the reply that was cut off in turn {i}", turns[nxt].text[:120]))
    return out


def latency(call: Call) -> list[dict]:
    out = []
    for i in range(1, len(call.turns)):
        prev, cur = call.turns[i - 1], call.turns[i]
        if prev.role == "user" and cur.role == "agent" and prev.end_s is not None and cur.start_s is not None:
            gap = (cur.start_s - prev.end_s) * 1000
            if gap > LATENCY_SPIKE_MS:
                out.append(_f("latency_spike", "low", i, f"{gap / 1000:.1f}s of silence before the agent replied",
                              cur.text[:80]))
    return out


def dropped_question(call: Call) -> list[dict]:
    """Call ends right after the caller asked something."""
    if call.turns and call.turns[-1].role == "user" and call.turns[-1].text.strip().endswith("?"):
        if call.ended_reason not in ("customer-ended-call", "user_hangup", "caller_hangup"):
            return [_f("unanswered_at_end", "medium", len(call.turns) - 1,
                       "Call ended without answering the caller's last question", call.turns[-1].text)]
    return []


ALL_RULES = [repeated_question, sensitive_request, political, voice_style,
             interruption_ignored, latency, dropped_question]


def run_rules(call: Call) -> list[dict]:
    findings = []
    for rule in ALL_RULES:
        findings.extend(rule(call))
    return findings
