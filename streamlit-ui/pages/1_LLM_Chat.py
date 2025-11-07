"""Streamlit page to chat with Groq (primary) and OpenRouter (fallback) models."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import streamlit as st
from dotenv import load_dotenv

# Ensure repository root is on PYTHONPATH and environment variables are loaded
REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
API_ROOT = REPO_ROOT / "api"
if str(API_ROOT) not in sys.path:
    sys.path.append(str(API_ROOT))

from api.config import settings  # noqa: E402  pylint: disable=wrong-import-position
from api.llm import GroqClient, OpenRouterClient, OpenRouterError  # noqa: E402  pylint: disable=wrong-import-position


st.set_page_config(page_title="LLM Chat Playground", page_icon="💬")
st.title("💬 LLM Chat Playground")
st.caption("Compare Groq (primary) and OpenRouter (fallback) outputs with the same prompt.")


def _rerun_app() -> None:
    """Trigger a Streamlit rerun compatible with multiple versions."""

    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()  # type: ignore[attr-defined]
    else:  # pragma: no cover - new API
        st.rerun()


@st.cache_resource(show_spinner=False)
def get_llm_clients() -> Mapping[str, Optional[object]]:
    """Instantiate available LLM clients once per session."""

    groq_client: Optional[GroqClient] = None
    openrouter_client: Optional[OpenRouterClient] = None

    if settings.groq_api_key:
        try:
            groq_client = GroqClient()
        except OpenRouterError as exc:
            st.warning(f"Groq configuration issue: {exc}")

    if settings.openrouter_api_key:
        try:
            openrouter_client = OpenRouterClient()
        except OpenRouterError as exc:
            st.warning(f"OpenRouter configuration issue: {exc}")

    return {"groq": groq_client, "openrouter": openrouter_client}


clients = get_llm_clients()
available_providers: Dict[str, object] = {
    "Groq": clients["groq"],
    "OpenRouter": clients["openrouter"],
}
available_providers = {name: client for name, client in available_providers.items() if client}

if not available_providers:
    st.error("No LLM providers are configured. Set GROQ_API_KEY or OPENROUTER_API_KEY in your environment.")
    st.stop()


def _provider_models(provider: str) -> Dict[str, str]:
    if provider == "Groq":
        return {
            f"Groq · {settings.groq_primary_model}": settings.groq_primary_model,
        }
    models = {
        f"Primary · {settings.openrouter_primary_model}": settings.openrouter_primary_model,
    }
    if settings.openrouter_fallback_model:
        models[f"Fallback · {settings.openrouter_fallback_model}"] = settings.openrouter_fallback_model
    return models


with st.sidebar:
    st.header("⚙️ Chat Settings")
    provider = st.selectbox("Provider", list(available_providers.keys()))
    provider_models = _provider_models(provider)
    model_label = st.selectbox("Model", list(provider_models.keys()))
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

    history_key = f"{provider}:{provider_models[model_label]}"
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.setdefault("chat_history", {})
        st.session_state["chat_history"].pop(history_key, None)
        _rerun_app()


selected_model = _provider_models(provider)[model_label]
history_id = f"{provider}:{selected_model}"


def ensure_history(key: str) -> List[Dict[str, Any]]:
    """Retrieve (or initialize) the chat history for a specific provider/model pair."""

    st.session_state.setdefault("chat_history", {})
    history = st.session_state["chat_history"].setdefault(key, [])
    return history


history = ensure_history(history_id)


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
                    f"Provider: {metadata.get('provider')} · Model: {metadata.get('model')} · "
                    f"Tokens: {metadata.get('total_tokens')} · Latency: {metadata.get('latency', 0.0):.2f}s"
                )


render_history(history)


def run_async(coro):
    """Run an async coroutine in a synchronous context."""

    return asyncio.run(coro)


def build_message_payload(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Convert stored history into the chat completion message format."""

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

    client = available_providers[provider]
    try:
        response = run_async(
            client.chat(
                message_payload,
                model=selected_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
    except OpenRouterError as exc:  # pragma: no cover - provider/network errors
        st.session_state.setdefault("chat_errors", [])
        st.session_state["chat_errors"].append(f"{provider}: {exc}")
        history.append(
            {
                "role": "assistant",
                "content": f"⚠️ {provider} request failed: {exc}",
                "metadata": {"provider": provider, "model": selected_model, "total_tokens": 0, "latency": 0.0},
            }
        )
    else:
        history.append(
            {
                "role": "assistant",
                "content": response.data,
                "metadata": {
                    "provider": provider,
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
