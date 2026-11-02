"""Minimal Groq (OpenAI-compatible) client with free-tier rate-limit retries."""

from __future__ import annotations

import json
import time

import httpx

from calldoctor import config

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class LLMError(RuntimeError):
    pass


def chat(messages: list[dict], model: str, temperature: float = 0.4,
         json_mode: bool = False, max_tokens: int = 400, retries: int = 5) -> str:
    if not config.GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY is not set")
    body: dict = {"model": model, "messages": messages,
                  "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {config.GROQ_API_KEY}"}
    for attempt in range(retries):
        try:
            resp = httpx.post(GROQ_URL, json=body, headers=headers, timeout=60)
        except httpx.HTTPError as e:
            if attempt == retries - 1:
                raise LLMError(str(e)) from e
            time.sleep(2 ** attempt)
            continue
        if resp.status_code in (429, 500, 502, 503) and attempt < retries - 1:
            wait = float(resp.headers.get("retry-after") or 2 ** (attempt + 1))
            time.sleep(min(wait, 30))
            continue
        if resp.status_code >= 400:
            raise LLMError(f"Groq {resp.status_code}: {resp.text[:300]}")
        return resp.json()["choices"][0]["message"]["content"] or ""
    raise LLMError("Groq rate limit, try again in a minute")


def chat_json(messages: list[dict], model: str, max_tokens: int = 1500) -> dict:
    raw = chat(messages, model, temperature=0, json_mode=True, max_tokens=max_tokens)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMError(f"Judge returned invalid JSON: {raw[:200]}") from e
