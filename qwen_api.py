"""Small wrapper for calling a Qwen-like reasoning API.
This is a placeholder; it expects an API key stored in environment or Streamlit secrets.
Replace endpoint and payload with the actual API details from the provider.
"""
import os
import requests
from typing import Optional

API_ENDPOINT = "https://api.example.com/v1/reason"  # Replace with real endpoint


def call_qwen(prompt: str, api_key: Optional[str] = None) -> dict:
    if api_key is None:
        api_key = os.environ.get("QWEN_API_KEY")
    if api_key is None:
        # Try Streamlit secrets if running inside Streamlit
        try:
            import streamlit as st
            api_key = st.secrets.get("qwen_api_key")
        except Exception:
            pass

    if api_key is None:
        raise RuntimeError("Qwen API key not found. Set QWEN_API_KEY or .streamlit/secrets.toml")

    payload = {"prompt": prompt, "max_tokens": 256}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(API_ENDPOINT, json=payload, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()
