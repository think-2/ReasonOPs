"""Qwen reasoning API wrapper (DashScope-compatible minimal client).

Provides qwen_reason(prompt) -> str that returns generated text, or a
descriptive error string. It reads API key from env var QWEN_API_KEY or
Streamlit secrets (QWEN_API_KEY/qwen_api_key).
"""
import os
from typing import Optional

import requests

# DashScope text generation endpoint (adjust if your provider differs)
API_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
)


def _get_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    if explicit_key:
        return explicit_key
    key = os.environ.get("QWEN_API_KEY")
    if key:
        return key
    # Try Streamlit secrets if available
    try:
        import streamlit as st  # type: ignore

        return (
            st.secrets.get("QWEN_API_KEY")
            or st.secrets.get("qwen_api_key")
            or None
        )
    except Exception:
        return None


def qwen_reason(prompt: str, api_key: Optional[str] = None, model: str = "qwen-turbo") -> str:
    """Call Qwen to produce a reasoning response.

    Returns the model's text output or an error string prefixed with 'Error:'.
    """
    key = _get_api_key(api_key)
    if not key:
        return (
            "Error: Qwen API key not found. Set env QWEN_API_KEY or "
            ".streamlit/secrets.toml (QWEN_API_KEY)."
        )

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": {"prompt": prompt},
        "parameters": {"max_tokens": 500},
    }
    try:
        resp = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # DashScope example structure: {"output": {"text": "..."}}
        text = (
            data.get("output", {}).get("text")
            or data.get("choices", [{}])[0].get("message", {}).get("content")
        )
        if not text:
            return "Error: Empty response from Qwen API."
        return text
    except Exception as e:
        return f"Error: {e}"


# Backwards-compatible alias if earlier code imported call_qwen
def call_qwen(prompt: str, api_key: Optional[str] = None) -> dict:
    text = qwen_reason(prompt, api_key=api_key)
    return {"text": text}
