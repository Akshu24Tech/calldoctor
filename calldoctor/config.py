"""Paths, environment and the survey agent config."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = ROOT / "agent"
load_dotenv(ROOT / ".env")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
AGENT_MODEL_OVERRIDE = os.environ.get("AGENT_MODEL", "")
CALLER_MODEL = os.environ.get("CALLER_MODEL", "llama-3.1-8b-instant")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "llama-3.3-70b-versatile")

# Shared secret for webhooks and admin actions. Put it in every webhook URL
# as ?token=..., and in the dashboard when you trigger a simulation.
INGEST_TOKEN = os.environ.get("CALLDOCTOR_TOKEN", "")
PUBLIC_URL = os.environ.get("PUBLIC_URL", "").rstrip("/")


def database_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        (ROOT / "data").mkdir(exist_ok=True)
        return f"sqlite:///{ROOT / 'data' / 'calldoctor.db'}"
    # Render / Neon hand out postgres:// URLs; SQLAlchemy wants a driver name.
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


@lru_cache
def survey() -> dict[str, Any]:
    return yaml.safe_load((AGENT_DIR / "survey.yaml").read_text(encoding="utf-8"))


@lru_cache
def personas() -> dict[str, dict[str, Any]]:
    data = yaml.safe_load((AGENT_DIR / "personas.yaml").read_text(encoding="utf-8"))
    return {p["id"]: p for p in data["personas"]}


def prompt_versions() -> list[str]:
    return sorted(p.stem for p in (AGENT_DIR / "prompts").glob("*.md"))


def system_prompt(version: str | None = None) -> str:
    version = version or survey()["active_version"]
    path = AGENT_DIR / "prompts" / f"{version}.md"
    if not path.exists():
        raise ValueError(f"No prompt version '{version}'. Have: {prompt_versions()}")
    return path.read_text(encoding="utf-8").strip()


def agent_model() -> str:
    return AGENT_MODEL_OVERRIDE or survey()["llm"]["model"]
