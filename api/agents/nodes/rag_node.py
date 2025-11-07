"""RAG node that enriches ambiguous dishes with retrieval evidence."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from services.rag_service import RAGService
from ..state import MenuProcessorState, append_reasoning, get_current_dish

logger = logging.getLogger(__name__)


class RAGNode:
    """Performs retrieval and supplies context back to the classifier."""

    def __init__(
        self,
        rag_service: Optional[RAGService],
        *,
        top_k: int = 3,
        min_score: float = 0.35,
    ) -> None:
        self.rag_service = rag_service
        self.top_k = top_k
        self.min_score = min_score

    async def __call__(self, state: MenuProcessorState) -> Dict[str, Any]:
        dish = get_current_dish(state)
        if dish is None or self.rag_service is None:
            logger.info("RAG skipped; service unavailable or no dish remaining.")
            return {
                "rag_context": None,
                "rag_lookups": state.get("rag_lookups", []),
                "requires_rag": False,
            }

        query_text = f"{dish.name}. {dish.raw_text}"
        matches = self.rag_service.search(query_text, top_k=self.top_k)
        filtered = [match for match in matches if match.score >= self.min_score]

        rag_records = list(state.get("rag_lookups", []))
        reasoning_message = ""

        if filtered:
            context = self.rag_service.build_context(filtered)
            rag_records.append(
                {
                    "dish": dish.name,
                    "query": query_text,
                    "matches": [match.as_metadata() for match in filtered],
                }
            )
            reasoning_message = (
                f"Retrieved {len(filtered)} contextual matches for {dish.name}; "
                "retrying classification with evidence."
            )
            logger.info("RAG retrieved context for '%s': %s", dish.name, ", ".join(m.name for m in filtered))
            return {
                "rag_context": context,
                "rag_lookups": rag_records,
                "requires_rag": False,
                "reasoning_log": append_reasoning(state, reasoning_message),
            }

        reasoning_message = (
            f"No relevant RAG evidence found for {dish.name}; keeping original classification."
        )
        logger.info("RAG found no relevant matches for '%s'.", dish.name)
        return {
            "rag_context": None,
            "rag_lookups": rag_records,
            "requires_rag": False,
            "reasoning_log": append_reasoning(state, reasoning_message),
        }
