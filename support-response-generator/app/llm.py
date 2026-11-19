"""Optional LLM helper: polish a finished draft.

The n-gram suggester works fully offline. This module is only used by
`python -m app.cli polish` and talks to any OpenAI-compatible endpoint
(Ollama, Groq, Gemini) configured in .env. The draft is anonymized before it
is sent, so no customer data leaves the machine.
"""
from __future__ import annotations

import json
import os
import urllib.request

from app.core import anonymize


def _settings() -> tuple[str, str, str]:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # python-dotenv is optional
        pass
    base = os.getenv("LLM_BASE_URL", "").rstrip("/")
    model = os.getenv("LLM_MODEL", "")
    key = os.getenv("LLM_API_KEY", "")
    if not base or not model:
        raise RuntimeError("LLM not configured: copy .env.example to .env and set LLM_BASE_URL and LLM_MODEL.")
    return base, model, key


def polish(draft: str, timeout: float = 30.0) -> str:
    base, model, key = _settings()
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Fix grammar and capitalisation of this customer support reply. "
                                          "Keep placeholders like <NAME> unchanged. Reply with the text only."},
            {"role": "user", "content": anonymize(draft)},
        ],
        "temperature": 0.2,
    }
    req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"].strip()
