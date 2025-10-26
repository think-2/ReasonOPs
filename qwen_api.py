"""Qwen reasoning API wrapper supporting DashScope and OpenRouter.

Provides qwen_reason(prompt) -> str that returns generated text, or a
descriptive error string. It reads API key from env var QWEN_API_KEY or
Streamlit secrets (QWEN_API_KEY/qwen_api_key).

Routing:
- If the key looks like an OpenRouter key ("sk-or-..."), use OpenRouter's
  OpenAI-compatible chat completions API.
- Otherwise, default to DashScope text generation API.
"""
import os
from typing import Optional

import requests

# Endpoints
DASHSCOPE_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
)
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def _get_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    if explicit_key:
        return explicit_key
    key = os.environ.get("QWEN_API_KEY")
    if key:
        return key.strip()
    # Try Streamlit secrets if available
    try:
        import streamlit as st  # type: ignore

        v = st.secrets.get("QWEN_API_KEY") or st.secrets.get("qwen_api_key")
        return v.strip() if isinstance(v, str) else v
    except Exception:
        return None


def qwen_reason(prompt: str, api_key: Optional[str] = None, model: Optional[str] = None) -> str:
    """Call Qwen provider to produce a reasoning response.

    Auto-detects provider from API key pattern. Returns the model's text output
    or an error string prefixed with 'Error:'.
    """
    key = _get_api_key(api_key)
    if not key:
        return (
            "Error: Qwen API key not found. Set env QWEN_API_KEY or "
            ".streamlit/secrets.toml (QWEN_API_KEY)."
        )

    # Provider override via env var
    provider = (os.environ.get("QWEN_PROVIDER") or "").strip().lower()
    if provider in {"openrouter", "or"}:
        use_openrouter = True
    elif provider in {"dashscope", "ds"}:
        use_openrouter = False
    else:
        # Auto-detect: OpenRouter if key looks like sk-or-..., else DashScope
        use_openrouter = key.startswith("sk-or-")

    if use_openrouter:
        # OpenRouter (OpenAI-compatible chat)
        model_name = (
            model
            or os.environ.get("QWEN_MODEL")
            or "qwen/qwen-2.5-7b-instruct"
        )
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        # Optional metadata headers improve rate limits/analytics on OpenRouter
        site_url = os.environ.get("OPENROUTER_SITE_URL") or "http://localhost"
        app_title = os.environ.get("OPENROUTER_APP_TITLE") or "ReasonOPs"
        headers["HTTP-Referer"] = site_url
        headers["X-Title"] = app_title

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
        }
        try:
            resp = requests.post(OPENROUTER_ENDPOINT, headers=headers, json=payload, timeout=45)
            try:
                resp.raise_for_status()
            except requests.HTTPError as he:
                return f"Error: {resp.status_code} {he}. Body: {resp.text[:400]}"
            data = resp.json()
            text = (
                (data.get("choices") or [{}])[0]
                .get("message", {})
                .get("content")
            )
            if not text:
                return "Error: Empty response from OpenRouter."
            return text
        except Exception as e:
            return f"Error: {e}"
    else:
        # DashScope
        model_name = model or os.environ.get("QWEN_MODEL") or "qwen-turbo"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "model": model_name,
            "input": {"prompt": prompt},
            "parameters": {"max_tokens": 500},
        }
        try:
            resp = requests.post(DASHSCOPE_ENDPOINT, headers=headers, json=payload, timeout=30)
            try:
                resp.raise_for_status()
            except requests.HTTPError as he:
                return f"Error: {resp.status_code} {he}. Body: {resp.text[:400]}"
            data = resp.json()
            text = data.get("output", {}).get("text")
            if not text:
                return "Error: Empty response from DashScope."
            return text
        except Exception as e:
            return f"Error: {e}"


# Backwards-compatible alias if earlier code imported call_qwen
def call_qwen(prompt: str, api_key: Optional[str] = None) -> dict:
    text = qwen_reason(prompt, api_key=api_key)
    return {"text": text}
