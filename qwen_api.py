"""Qwen API client helpers used by the ReasonOPs app.

Contains a small chat client (`qwen_chat`) and two convenience wrappers
(`classify_problem`, `generate_schema`) used to classify OR problems and
produce adaptive JSON schemas.

This module uses the Dashscope/Aliyun text-generation endpoint by default
but allows overriding the full endpoint via the QWEN_API_BASE env var.
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Dict, Optional

import requests


DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def _get_api_key() -> Optional[str]:
    # Accept either a QWEN_API_KEY or an OPENROUTER_API_KEY for flexibility
    key = os.environ.get("QWEN_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    try:
        import streamlit as st

        # Accept either a dedicated OpenRouter secret or the older qwen_api_key
        return st.secrets.get("openrouter_api_key") or st.secrets.get("qwen_api_key")
    except Exception:
        return None


def _extract_text(resp_json: dict) -> str:
    """Try several common response shapes to extract generated text."""
    # Print raw response for debugging
    print("[qwen_api] raw response:", json.dumps(resp_json)[:400])
    # Common patterns
    if not isinstance(resp_json, dict):
        return str(resp_json)
    if "text" in resp_json and isinstance(resp_json["text"], str):
        return resp_json["text"]
    # choices -> message -> content
    if "choices" in resp_json and isinstance(resp_json["choices"], list):
        first = resp_json["choices"][0]
        # openai-like
        if isinstance(first, dict):
            msg = first.get("message") or first.get("text") or first.get("content")
            if isinstance(msg, dict) and "content" in msg:
                return msg["content"]
            if isinstance(msg, str):
                return msg
    # dashscope style: data -> [ { content: { text: ... } } ]
    if "data" in resp_json and isinstance(resp_json["data"], list):
        first = resp_json["data"][0]
        if isinstance(first, dict):
            cont = first.get("content") or {}
            if isinstance(cont, dict):
                t = cont.get("text") or cont.get("response")
                if isinstance(t, str):
                    return t
    # fallback to stringifying
    return json.dumps(resp_json)


def qwen_chat(
    messages: List[Dict[str, str]], *, max_tokens: int = 700, temperature: float = 0.3, timeout: int = 30
) -> str:
    """Send multi-turn chat messages to the Qwen reasoning API and return model text.

    messages is a list of {role: 'system'|'user'|'assistant', content: str}.
    Raises RuntimeError on network or API errors.
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list of role/content dicts")

    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Qwen API key not found. Set QWEN_API_KEY or .streamlit/secrets.toml")

    endpoint = os.environ.get("QWEN_API_BASE", os.environ.get("OPENROUTER_API_BASE", DEFAULT_ENDPOINT))

    # Allow callers to override the model name via env var (OPENROUTER_MODEL or QWEN_MODEL)
    model = os.environ.get("OPENROUTER_MODEL") or os.environ.get("QWEN_MODEL") or "gpt-4o-mini"

    # OpenRouter expects a chat-completions style payload including the model
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"Network error calling Qwen API: {e}") from e

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # Try to include body for debugging
        body = resp.text[:1000]
        raise RuntimeError(f"Qwen API returned error {resp.status_code}: {body}") from e

    try:
        data = resp.json()
    except Exception:
        raise RuntimeError("Qwen API returned non-JSON response")

    text = _extract_text(data)
    return text


def classify_problem(user_text: str) -> str:
    """Ask Qwen to label the OR problem type (e.g., profit_optimization, scheduling)."""
    system = {
        "role": "system",
        "content": (
            "You are an assistant that classifies short descriptions of operations research "
            "problems into concise labels such as: profit_optimization, scheduling, routing, "
            "inventory_management, allocation, or other. Reply with a single label."
        ),
    }
    user = {"role": "user", "content": user_text}
    resp = qwen_chat([system, user], max_tokens=30, temperature=0.0)
    return resp.strip()


def _find_json_substr(s: str) -> Optional[str]:
    # Find first curly-braced block
    match = re.search(r"(\{[\s\S]*\})", s)
    return match.group(1) if match else None


def generate_schema(context: List[Dict[str, str]], base_schema: dict | None = None) -> dict:
    """Ask Qwen to return/extend a JSON schema of inputs based on the conversation.

    Returns a dict parsed from JSON. On parse error returns {'error':'invalid json','raw':...}.
    """
    system = {
        "role": "system",
        "content": (
            "Given the conversation so far, decide what inputs are needed for this "
            "optimization problem. Output valid JSON with a fields array (each field = {label,type,default,unit,description}). "
            "If extending a previous schema, keep old fields and add new ones."
        ),
    }
    messages = [system] + list(context)
    if base_schema:
        # provide base schema as context for extension
        messages.append({"role": "system", "content": f"Base schema: {json.dumps(base_schema)}"})

    raw = qwen_chat(messages, max_tokens=800, temperature=0.2)

    # Try to parse JSON directly, else try to extract a JSON substring
    try:
        return json.loads(raw)
    except Exception:
        substr = _find_json_substr(raw)
        if substr:
            try:
                return json.loads(substr)
            except Exception:
                return {"error": "invalid json", "raw": raw}
        return {"error": "invalid json", "raw": raw}


if __name__ == "__main__":
    # Demo block
    from schema_library import get_base_schema

    user_text = "We produce A,B,C with limited labor and material; profits 10,15,25."
    print("Problem type:", classify_problem(user_text))
    schema = generate_schema([{"role": "user", "content": user_text}], base_schema=get_base_schema("profit_optimization"))
    print("Schema:", schema)
