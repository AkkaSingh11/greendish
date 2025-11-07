"""Async OpenRouter client with structured JSON responses."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Type, Union

import httpx
from pydantic import BaseModel, ValidationError

try:  # Lazy import to support both intra-package and external consumers
    from config import settings  # type: ignore
except ImportError:  # pragma: no cover - streamlit/imported contexts
    from api.config import settings  # type: ignore

logger = logging.getLogger(__name__)


JsonSchemaHint = Mapping[str, Union[Type[Any], Tuple[Type[Any], ...]]]


class OpenRouterError(RuntimeError):
    """Raised when the OpenRouter client cannot return a valid response."""


@dataclass(slots=True)
class OpenRouterResponse:
    """Structured payload returned from OpenRouter."""

    data: Any
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency: float
    request_id: Optional[str] = None


class OpenRouterClient:
    """Thin wrapper around the OpenRouter chat completions API."""

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
        fallback_model: Optional[str] = None,
        request_timeout: Optional[int] = None,
        app_name: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or settings.openrouter_api_key
        if not self.api_key:
            raise OpenRouterError("OpenRouter API key is not configured.")

        self.base_url = (base_url or settings.openrouter_base_url).rstrip("/")
        self.primary_model = primary_model or settings.openrouter_primary_model
        self.fallback_model = fallback_model or settings.openrouter_fallback_model
        self.request_timeout = request_timeout or settings.openrouter_request_timeout
        self.app_name = app_name or settings.openrouter_app_name

        if not self.primary_model:
            raise OpenRouterError("OpenRouter primary model is not configured.")

        if not self.base_url.endswith("/chat/completions"):
            self._completions_url = f"{self.base_url}/chat/completions"
        else:
            self._completions_url = self.base_url

    async def complete_json(
        self,
        prompt: str,
        *,
        schema: Optional[Union[Type[BaseModel], JsonSchemaHint]] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        extra_messages: Optional[Sequence[Mapping[str, str]]] = None,
    ) -> OpenRouterResponse:
        """Request a JSON response from OpenRouter with fallback handling."""

        messages: list[dict[str, str]] = [{"role": "system", "content": self._SYSTEM_PROMPT}]
        if extra_messages:
            messages.extend({"role": msg["role"], "content": msg["content"]} for msg in extra_messages)
        messages.append({"role": "user", "content": prompt})

        errors: list[Exception] = []
        for candidate_model in self._candidate_models(model):
            try:
                payload = await self._invoke_model(
                    candidate_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                structured = self._parse_structured_content(payload)
                data = self._validate_schema(structured, schema)
                usage = payload.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                completion_tokens = int(usage.get("completion_tokens") or 0)
                total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
                latency = float(payload.get("_latency", 0.0))
                request_id = payload.get("id")
                logger.debug(
                    "OpenRouter request succeeded with model %s (prompt_tokens=%d, completion_tokens=%d)",
                    candidate_model,
                    prompt_tokens,
                    completion_tokens,
                )
                return OpenRouterResponse(
                    data=data,
                    model=candidate_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency=latency,
                    request_id=request_id,
                )
            except Exception as exc:  # noqa: BLE001 - we re-raise below
                logger.warning("OpenRouter model %s failed: %s", candidate_model, exc)
                errors.append(exc)
                continue

        raise OpenRouterError(
            "; ".join(str(err) for err in errors) if errors else "OpenRouter request failed."
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
        """Send a general chat request and return text output."""

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

        errors: list[Exception] = []
        for candidate_model in self._candidate_models(model):
            try:
                payload = await self._invoke_model(
                    candidate_model,
                    messages=assembled,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=None,
                )
                text = self._extract_text_content(payload)
                usage = payload.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                completion_tokens = int(usage.get("completion_tokens") or 0)
                total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
                latency = float(payload.get("_latency", 0.0))
                request_id = payload.get("id")
                return OpenRouterResponse(
                    data=text,
                    model=candidate_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency=latency,
                    request_id=request_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("OpenRouter chat with model %s failed: %s", candidate_model, exc)
                errors.append(exc)
                continue

        raise OpenRouterError(
            "; ".join(str(err) for err in errors) if errors else "OpenRouter chat request failed."
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
                f"OpenRouter returned status {response.status_code}: {response.text.strip()}"
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
        if settings.openrouter_referer:
            headers["HTTP-Referer"] = settings.openrouter_referer
        return headers

    def _candidate_models(self, explicit_model: Optional[str]) -> Tuple[str, ...]:
        chosen = explicit_model or self.primary_model
        models = [chosen]
        if self.fallback_model and self.fallback_model != chosen:
            models.append(self.fallback_model)
        return tuple(models)

    @staticmethod
    def _parse_structured_content(payload: Dict[str, Any]) -> Dict[str, Any]:
        choices = payload.get("choices")
        if not choices:
            raise OpenRouterError("OpenRouter response did not contain choices.")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise OpenRouterError("OpenRouter response did not contain message content.")

        text = OpenRouterClient._strip_code_fence(str(content).strip())
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise OpenRouterError(f"Failed to decode JSON response: {text}") from exc

    @staticmethod
    def _strip_code_fence(value: str) -> str:
        if value.startswith("```") and value.endswith("```"):
            lines = value.splitlines()
            if len(lines) >= 3:
                return "\n".join(lines[1:-1]).strip()
            return ""
        return value

    @staticmethod
    def _extract_text_content(payload: Dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not choices:
            raise OpenRouterError("OpenRouter response did not contain choices.")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            raise OpenRouterError("OpenRouter response did not contain message content.")
        return str(content).strip()

    @staticmethod
    def _validate_schema(
        data: Dict[str, Any],
        schema: Optional[Union[Type[BaseModel], JsonSchemaHint]],
    ) -> Dict[str, Any]:
        if schema is None:
            return data

        if isinstance(schema, type) and issubclass(schema, BaseModel):
            try:
                return schema.model_validate(data).model_dump()
            except ValidationError as exc:
                raise OpenRouterError(f"Response failed schema validation: {exc}") from exc

        if isinstance(schema, Mapping):
            missing = [key for key in schema.keys() if key not in data]
            if missing:
                raise OpenRouterError(
                    f"Response missing required keys: {', '.join(missing)}"
                )
            for key, expected in schema.items():
                if expected is None:
                    continue
                expected_types = expected if isinstance(expected, tuple) else (expected,)
                if not isinstance(data[key], expected_types):
                    readable_types = ", ".join(t.__name__ for t in expected_types)
                    raise OpenRouterError(
                        f"Key '{key}' expected type {readable_types}, got {type(data[key]).__name__}"
                    )
            return {key: data[key] for key in schema.keys()}

        raise OpenRouterError("Unsupported schema specification provided.")
