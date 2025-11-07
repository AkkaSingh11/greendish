"""Async Groq client compatible with the OpenRouter interface."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import httpx
from pydantic import BaseModel, ValidationError

from config import settings

from .openrouter_client import OpenRouterError, OpenRouterResponse

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _GroqUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


class GroqClient:
    """Thin wrapper around Groq's OpenAI-compatible chat completions API."""

    _SYSTEM_PROMPT = (
        "You are a precise assistant. Always reply with a single valid JSON object "
        "matching the requested schema, without additional text."
    )

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        primary_model: Optional[str] = None,
        request_timeout: Optional[int] = None,
        app_name: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or settings.groq_api_key
        if not self.api_key:
            raise OpenRouterError("Groq API key is not configured.")

        self.base_url = (base_url or settings.groq_base_url).rstrip("/")
        if not self.base_url.endswith("/chat/completions"):
            self._completions_url = f"{self.base_url}/chat/completions"
        else:
            self._completions_url = self.base_url

        self.primary_model = primary_model or settings.groq_primary_model
        if not self.primary_model:
            raise OpenRouterError("Groq primary model is not configured.")

        self.request_timeout = request_timeout or settings.groq_request_timeout
        self.app_name = app_name or settings.openrouter_app_name

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
        messages: list[dict[str, str]] = [{"role": "system", "content": self._SYSTEM_PROMPT}]
        if extra_messages:
            messages.extend({"role": msg["role"], "content": msg["content"]} for msg in extra_messages)
        messages.append({"role": "user", "content": prompt})

        payload = await self._invoke_model(
            model or self.primary_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        structured = self._parse_structured_content(payload)
        data = self._validate_schema(structured, schema)
        usage = self._extract_usage(payload)
        latency = float(payload.get("_latency", 0.0))
        request_id = payload.get("id")

        return OpenRouterResponse(
            data=data,
            model=payload.get("model", model or self.primary_model),
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.prompt_tokens + usage.completion_tokens,
            latency=latency,
            request_id=request_id,
        )

    async def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> OpenRouterResponse:
        if not messages and not system_prompt:
            raise OpenRouterError("At least one message or a system prompt is required.")

        assembled: list[dict[str, str]] = []
        if system_prompt:
            assembled.append({"role": "system", "content": system_prompt})

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role not in {"system", "user", "assistant"}:
                continue
            if content is None:
                continue
            assembled.append({"role": role, "content": str(content)})

        if not any(m["role"] == "user" for m in assembled):
            raise OpenRouterError("Chat requires at least one user message.")

        payload = await self._invoke_model(
            model or self.primary_model,
            messages=assembled,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=None,
        )

        text = self._extract_text_content(payload)
        usage = self._extract_usage(payload)
        latency = float(payload.get("_latency", 0.0))
        request_id = payload.get("id")
        return OpenRouterResponse(
            data=text,
            model=payload.get("model", model or self.primary_model),
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.prompt_tokens + usage.completion_tokens,
            latency=latency,
            request_id=request_id,
        )

    async def _invoke_model(
        self,
        model: str,
        *,
        messages: Sequence[Mapping[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        response_format: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        body: Dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if response_format:
            body["response_format"] = dict(response_format)

        timeout = httpx.Timeout(self.request_timeout)
        headers = self._build_headers()

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                self._completions_url,
                headers=headers,
                json=body,
            )

        elapsed = time.perf_counter() - start_time
        if response.status_code >= 400:
            raise OpenRouterError(
                f"Groq returned status {response.status_code}: {response.text.strip()}"
            )

        payload = response.json()
        payload["_latency"] = elapsed
        return payload

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.app_name:
            headers["X-Title"] = self.app_name
        return headers

    @staticmethod
    def _parse_structured_content(payload: Dict[str, Any]) -> Dict[str, Any]:
        choices = payload.get("choices")
        if not choices:
            raise OpenRouterError("Groq response did not contain choices.")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise OpenRouterError("Groq response did not contain message content.")

        text = str(content).strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()
            else:
                text = ""
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover
            raise OpenRouterError(f"Failed to decode JSON response: {text}") from exc

    @staticmethod
    def _extract_text_content(payload: Dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not choices:
            raise OpenRouterError("Groq response did not contain choices.")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            raise OpenRouterError("Groq response did not contain message content.")
        return str(content).strip()

    @staticmethod
    def _extract_usage(payload: Dict[str, Any]) -> _GroqUsage:
        usage = payload.get("usage") or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        return _GroqUsage(prompt_tokens=prompt, completion_tokens=completion)

    @staticmethod
    def _validate_schema(
        data: Dict[str, Any],
        schema: Optional[type[BaseModel]],
    ) -> Dict[str, Any]:
        if schema is None:
            return data

        try:
            return schema.model_validate(data).model_dump()
        except ValidationError as exc:
            raise OpenRouterError(f"Response failed schema validation: {exc}") from exc
