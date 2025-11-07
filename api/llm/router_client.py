"""Router that prefers Groq and falls back to OpenRouter."""

from __future__ import annotations

import logging
from typing import Mapping, Optional, Sequence

from pydantic import BaseModel

from .groq_client import GroqClient
from .openrouter_client import OpenRouterClient, OpenRouterError, OpenRouterResponse
from .types import SupportsFullLLMClient

logger = logging.getLogger(__name__)


class LLMRouter(SupportsFullLLMClient):
    """Tries Groq first and falls back to OpenRouter."""

    def __init__(
        self,
        *,
        groq_client: Optional[GroqClient] = None,
        openrouter_client: Optional[OpenRouterClient] = None,
    ) -> None:
        providers: list[SupportsFullLLMClient] = []
        if groq_client is not None:
            providers.append(groq_client)
        if openrouter_client is not None:
            providers.append(openrouter_client)

        if not providers:
            raise OpenRouterError("No LLM providers configured. Add Groq or OpenRouter credentials.")

        self._providers = providers

    async def complete_json(
        self,
        prompt: str,
        *,
        schema: Optional[type[BaseModel]] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        extra_messages: Optional[Sequence[Mapping[str, str]]] = None,
    ) -> OpenRouterResponse:
        errors: list[str] = []
        for provider in self._providers:
            try:
                return await provider.complete_json(
                    prompt,
                    schema=schema,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_messages=extra_messages,
                )
            except OpenRouterError as exc:
                logger.warning("LLM provider %s failed: %s", provider.__class__.__name__, exc)
                errors.append(f"{provider.__class__.__name__}: {exc}")
            except Exception as exc:  # pragma: no cover - network/runtime surprises
                logger.exception("Unexpected error from LLM provider %s", provider.__class__.__name__)
                errors.append(f"{provider.__class__.__name__}: {exc}")

        raise OpenRouterError("All LLM providers failed. " + " | ".join(errors))

    async def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> OpenRouterResponse:
        errors: list[str] = []
        for provider in self._providers:
            try:
                return await provider.chat(
                    messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                )
            except OpenRouterError as exc:
                logger.warning("LLM provider %s chat failed: %s", provider.__class__.__name__, exc)
                errors.append(f"{provider.__class__.__name__}: {exc}")
            except Exception as exc:  # pragma: no cover
                logger.exception("Unexpected chat error from %s", provider.__class__.__name__)
                errors.append(f"{provider.__class__.__name__}: {exc}")

        raise OpenRouterError("All LLM providers failed. " + " | ".join(errors))

