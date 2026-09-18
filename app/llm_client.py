"""
Thin LLM abstraction so the rest of the app never cares which backend answers.

Supported providers (both zero-cost):
  - "groq"   : Groq's free-tier API (fast Llama models). Needs GROQ_API_KEY.
  - "ollama" : A locally running Ollama model. Needs nothing but Ollama installed.

Set LLM_PROVIDER=groq|ollama in the environment (see .env.example).
"""
from __future__ import annotations

import json
import os
from typing import Any


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")


class LLMError(RuntimeError):
    """Raised when the configured LLM backend can't be reached or misbehaves."""


def _chat_groq(system_prompt: str, user_prompt: str) -> str:
    from groq import Groq  # imported lazily so the package is only required if used

    if not GROQ_API_KEY:
        raise LLMError(
            "LLM_PROVIDER=groq but GROQ_API_KEY is not set. "
            "Get a free key at https://console.groq.com/keys and put it in .env"
        )
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content


def _chat_ollama(system_prompt: str, user_prompt: str) -> str:
    import requests

    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=60,
        )
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        raise LLMError(
            f"Could not reach Ollama at {OLLAMA_HOST}. Is `ollama serve` running "
            f"and have you pulled the model (`ollama pull {OLLAMA_MODEL}`)? Original error: {e}"
        ) from e
    return r.json()["message"]["content"]


def chat(system_prompt: str, user_prompt: str) -> str:
    if LLM_PROVIDER == "groq":
        return _chat_groq(system_prompt, user_prompt)
    if LLM_PROVIDER == "ollama":
        return _chat_ollama(system_prompt, user_prompt)
    raise LLMError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use 'groq' or 'ollama'.")


def chat_json(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    """Call the LLM and parse its reply as JSON, tolerating markdown code fences."""
    raw = chat(system_prompt, user_prompt)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LLMError(
            f"LLM did not return valid JSON. Raw output: {raw!r}"
        ) from e

