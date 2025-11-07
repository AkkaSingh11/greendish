"""Classifier node that uses the configured LLM client to classify dishes."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Union

from pydantic import BaseModel, Field

from config import settings
from llm import OpenRouterResponse
from llm.types import SupportsJSONCompletion
from models import ClassificationResult, Dish
from ..state import (
    MenuProcessorState,
    append_reasoning,
    clone_dish,
    get_current_dish,
    update_vegetarian_dishes,
    upsert_by_index,
)
from ..utils import normalize_signals

logger = logging.getLogger(__name__)


class ClassificationPayload(BaseModel):
    """Expected schema returned by the classification LLM."""

    is_vegetarian: bool = Field(description="Whether the dish should be considered vegetarian.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1.")
    reasoning: str = Field(description="Brief explanation for the decision.")
    signals: Optional[Union[Dict[str, Sequence[str]], Sequence[str], str]] = Field(
        default=None,
        description="Optional evidence such as ingredients or keywords.",
    )


class ClassifierNode:
    """Classifies a single dish and updates agent state."""

    def __init__(
        self,
        client: SupportsJSONCompletion,
        *,
        confidence_threshold: Optional[float] = None,
        temperature: float = 0.0,
        max_response_tokens: int = 250,
    ) -> None:
        self.client = client
        self.confidence_threshold = confidence_threshold or settings.confidence_threshold
        self.temperature = temperature
        self.max_response_tokens = max_response_tokens

    async def __call__(self, state: MenuProcessorState) -> Dict[str, Any]:
        dish = get_current_dish(state)
        if dish is None:
            logger.debug("Classifier node invoked with no remaining dishes.")
            return {}

        prompt = self._build_prompt(dish, state)
        logger.debug("Classifying dish '%s' via LLM provider.", dish.name)
        response = await self.client.complete_json(
            prompt,
            schema=ClassificationPayload,
            temperature=self.temperature,
            max_tokens=self.max_response_tokens,
        )

        payload = ClassificationPayload.model_validate(response.data)
        confidence = float(payload.confidence)
        is_vegetarian = bool(payload.is_vegetarian)
        reasoning = payload.reasoning.strip()
        signals = normalize_signals(payload.signals)

        had_context = bool(state.get("rag_context"))
        method = "rag" if had_context else "llm"
        updated_dish = clone_dish(
            dish,
            is_vegetarian=is_vegetarian,
            confidence=confidence,
            classification_method=method,
            reasoning=reasoning or None,
            signals=signals,
        )

        classification = ClassificationResult(
            dish=updated_dish,
            is_vegetarian=is_vegetarian,
            confidence=confidence,
            method=method,
            reasoning=reasoning or None,
            rag_matches=state.get("rag_lookups") or None,
        )

        index = state.get("current_dish_index", 0)
        classified = upsert_by_index(state.get("classified_dishes", []), index, updated_dish)
        classifications = upsert_by_index(
            state.get("classification_results", []),
            index,
            classification,
        )
        vegetarian = update_vegetarian_dishes(
            state.get("vegetarian_dishes", []),
            updated_dish,
            include=is_vegetarian,
        )
        confidence_scores = dict(state.get("confidence_scores", {}))
        confidence_scores[updated_dish.name] = confidence

        reasoning_line = (
            f"{updated_dish.name}: {reasoning or 'No reasoning provided.'} "
            f"(confidence={confidence:.2f}, vegetarian={is_vegetarian})"
        )

        requires_rag = confidence < self.confidence_threshold and not had_context

        return {
            "classified_dishes": classified,
            "classification_results": classifications,
            "vegetarian_dishes": vegetarian,
            "confidence_scores": confidence_scores,
            "reasoning_log": append_reasoning(state, reasoning_line),
            "requires_rag": requires_rag,
            "llm_model": response.model,
            "rag_context": None,
            "rag_lookups": state.get("rag_lookups", []),
        }

    def _build_prompt(self, dish: Dish, state: MenuProcessorState) -> str:
        """Assemble the structured classification prompt."""
        context_lines = [
            "You are classifying whether a restaurant menu item is vegetarian.",
            "Return strict JSON with fields is_vegetarian, confidence (0-1 float), reasoning, and optional signals.",
        ]
        if state.get("rag_context"):
            context_lines.append("Use the provided context to improve your answer.")
            context_lines.append(f"Context: {state['rag_context']}")

        dish_lines = [
            f"Dish name: {dish.name}",
            f"Raw menu text: {dish.raw_text}",
        ]
        if dish.price is not None:
            dish_lines.append(f"Listed price: {dish.price}")

        context_lines.extend(dish_lines)
        context_lines.append("Assume vegetarian dishes contain no meat, fish, or gelatin.")
        return "\n".join(context_lines)
