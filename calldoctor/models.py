"""The one call format every platform is converted into."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Turn:
    role: str                      # "agent" | "user"
    text: str
    start_s: float | None = None   # seconds from call start, if the platform gives it
    end_s: float | None = None
    interrupted: bool = False      # this turn was cut off by the other side
    silent: bool = False           # caller said nothing (simulation)


@dataclass
class Call:
    id: str
    platform: str                  # livekit | vapi | retell | simulation
    source: str                    # webhook | upload | simulation
    turns: list[Turn]
    agent_version: str | None = None
    persona: str | None = None
    created_at: float = 0.0
    duration_s: float | None = None
    cost_usd: float | None = None
    ended_reason: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Call:
        d = dict(d)
        d["turns"] = [Turn(**t) for t in d.get("turns", [])]
        return cls(**d)

    def agent_turns(self) -> list[tuple[int, Turn]]:
        return [(i, t) for i, t in enumerate(self.turns) if t.role == "agent"]

    def user_turns(self) -> list[tuple[int, Turn]]:
        return [(i, t) for i, t in enumerate(self.turns) if t.role == "user"]

    def transcript(self) -> str:
        lines = []
        for i, t in enumerate(self.turns):
            flags = []
            if t.interrupted:
                flags.append("cut off")
            if t.silent:
                flags.append("silence")
            tag = f" ({', '.join(flags)})" if flags else ""
            lines.append(f"[{i}] {t.role.upper()}{tag}: {t.text}")
        return "\n".join(lines)
