from __future__ import annotations

from typing import Mapping, Optional, Protocol, Sequence

from pydantic import BaseModel

from .openrouter_client import OpenRouterResponse


class SupportsJSONCompletion(Protocol):
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
        ...


class SupportsChatCompletion(Protocol):
    async def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> OpenRouterResponse:
        ...


class SupportsFullLLMClient(SupportsJSONCompletion, SupportsChatCompletion, Protocol):
    """Protocol for clients that support both JSON completions and free-form chat."""
