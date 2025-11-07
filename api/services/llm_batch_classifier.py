"""Batch classifier that asks the configured LLM provider to label dishes in a single call."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List, Sequence

from llm.types import SupportsJSONCompletion
from models import Dish

from agents.utils import SignalsInput, normalize_signals


@dataclass
class BatchClassificationItem:
    """Single classification for a dish index."""

    index: int
    name: str
    is_vegetarian: bool
    confidence: float
    reasoning: str
    signals: SignalsInput = None


class LLMBatchClassifier:
    """Classifies a menu-worth of dishes with a single LLM request."""

    def __init__(self, client: SupportsJSONCompletion) -> None:
        self.client = client

    async def classify(self, dishes: Sequence[Dish], *, request_id: str) -> List[BatchClassificationItem]:
        if not dishes:
            return []

        dish_payload = [
            {"index": idx, "name": dish.name, "price": dish.price, "raw_text": dish.raw_text}
            for idx, dish in enumerate(dishes)
        ]

        prompt = self._build_prompt(dish_payload)
        response = await self.client.complete_json(
            prompt,
            schema=None,
            temperature=0.0,
            extra_messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert food analyst. Given structured menu entries, "
                        "return vegetarian classifications for each dish."
                    ),
                }
            ],
        )

        payload = response.data
        return self._coerce_results(payload)

    def _coerce_results(self, payload: Any) -> List[BatchClassificationItem]:
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                return []

        if isinstance(payload, dict):
            results = payload.get("results", payload)
        else:
            results = payload

        if not isinstance(results, list):
            return []

        normalized: List[BatchClassificationItem] = []
        for idx, entry in enumerate(results):
            if not isinstance(entry, dict):
                continue

            dish_index = int(entry.get("index", idx))
            name = str(entry.get("name") or f"Dish {dish_index}")
            is_veg = bool(entry.get("is_vegetarian"))
            try:
                confidence = float(entry.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            reasoning = str(entry.get("reasoning") or "")
            signals = normalize_signals(entry.get("signals"))

            normalized.append(
                BatchClassificationItem(
                    index=dish_index,
                    name=name,
                    is_vegetarian=is_veg,
                    confidence=confidence,
                    reasoning=reasoning,
                    signals=signals,
                )
            )

        return normalized

    def _build_prompt(self, payload: list[dict]) -> str:
        dishes_json = json.dumps(payload, ensure_ascii=False, indent=2)
        return (
            "Classify each dish in the following JSON array. For each entry, return a JSON object with:\n"
            "index (matching the input), name, is_vegetarian (true/false), confidence (0-1 float), "
            "reasoning, and optional signals explaining key evidence. "
            "Use menu context such as ingredients and descriptions to decide vegetarian status. "
            "Input dishes:\n"
            f"{dishes_json}\n"
            "Output format:\n"
            "{\n"
            '  "results": [\n'
            '    {"index": 0, "name": "...", "is_vegetarian": true, "confidence": 0.9, "reasoning": "...", "signals": {...}},\n'
            "    ...\n"
            "  ]\n"
            "}"
        )
