"""Streamlit page to chat with configured OpenRouter models."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

# Ensure repository root is on PYTHONPATH and environment variables are loaded
REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from api.config import settings  # noqa: E402  pylint: disable=wrong-import-position
from api.llm import OpenRouterClient, OpenRouterError  # noqa: E402  pylint: disable=wrong-import-position


st.set_page_config(page_title="OpenRouter Chat Playground", page_icon="💬")
st.title("💬 OpenRouter Chat Playground")
st.caption(
    "Interact with the configured OpenRouter models to compare outputs and sanity-check prompts."
)


def _rerun_app() -> None:
    """Trigger a Streamlit rerun compatible with multiple versions."""

    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()  # type: ignore[attr-defined]
    else:  # pragma: no cover - new API
        st.rerun()


@st.cache_resource(show_spinner=False)
def get_openrouter_client() -> OpenRouterClient:
    """Instantiate and cache the OpenRouter client for the Streamlit session."""

    return OpenRouterClient()


try:
    client = get_openrouter_client()
except OpenRouterError as exc:
    st.error(f"OpenRouter configuration error: {exc}")
    st.stop()


MODEL_OPTIONS: Dict[str, str] = {
    f"Primary – {settings.openrouter_primary_model}": settings.openrouter_primary_model
}
if settings.openrouter_fallback_model:
    MODEL_OPTIONS[f"Fallback – {settings.openrouter_fallback_model}"] = settings.openrouter_fallback_model


with st.sidebar:
    st.header("⚙️ Chat Settings")
    model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.3, step=0.05)
    max_tokens_option = st.number_input(
        "Max tokens (0 = auto)", min_value=0, max_value=4096, value=400, step=50
    )
    max_tokens = max_tokens_option or None
    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful assistant. Answer clearly and concisely.",
        height=80,
    ).strip()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.setdefault("chat_history", {})
        st.session_state["chat_history"].pop(MODEL_OPTIONS[model_label], None)
        _rerun_app()


selected_model = MODEL_OPTIONS[model_label]


def ensure_history(model_name: str) -> List[Dict[str, Any]]:
    """Retrieve (or initialize) the chat history for a specific model."""

    st.session_state.setdefault("chat_history", {})
    history = st.session_state["chat_history"].setdefault(model_name, [])
    return history


history = ensure_history(selected_model)


def render_history(messages: List[Dict[str, Any]]) -> None:
    """Display chat history using Streamlit chat bubbles."""

    for message in messages:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        metadata = message.get("metadata")
        with st.chat_message(role):
            st.markdown(content)
            if metadata and role == "assistant":
                st.caption(
                    f"Model: {metadata.get('model')} · Tokens: {metadata.get('total_tokens')} · "
                    f"Latency: {metadata.get('latency', 0.0):.2f}s"
                )


render_history(history)


def run_async(coro):
    """Run an async coroutine in a synchronous context."""

    return asyncio.run(coro)


def build_message_payload(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Convert stored history into the OpenRouter message format."""

    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
        if msg.get("role") in {"system", "user", "assistant"}
    ]


user_input = st.chat_input("Ask a question or describe a task...")

if user_input:
    history.append({"role": "user", "content": user_input})

    message_payload = build_message_payload(history)
    if system_prompt:
        message_payload.insert(0, {"role": "system", "content": system_prompt})

    try:
        response = run_async(
            client.chat(
                message_payload,
                model=selected_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
    except OpenRouterError as exc:  # pragma: no cover - network errors
        st.session_state.setdefault("chat_errors", [])
        st.session_state["chat_errors"].append(str(exc))
        history.append(
            {
                "role": "assistant",
                "content": f"⚠️ Request failed: {exc}",
                "metadata": {"model": selected_model, "total_tokens": 0, "latency": 0.0},
            }
        )
    else:
        history.append(
            {
                "role": "assistant",
                "content": response.data,
                "metadata": {
                    "model": response.model,
                    "total_tokens": response.total_tokens,
                    "latency": response.latency,
                },
            }
        )

    _rerun_app()


if st.session_state.get("chat_errors"):
    with st.expander("View recent errors"):
        for err in st.session_state["chat_errors"][-5:]:
            st.write(f"- {err}")
