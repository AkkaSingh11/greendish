"""Calculator node that aggregates vegetarian dishes via the MCP client."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional

from models import CalculationSummary, Dish
from services.mcp_client import MCPClient
from ..state import MenuProcessorState, append_reasoning

logger = logging.getLogger(__name__)


class CalculatorNode:
    """Finalizes totals for vegetarian dishes using the MCP calculator."""

    def __init__(
        self,
        mcp_client: Optional[MCPClient],
        *,
        confidence_threshold: float,
    ) -> None:
        self.mcp_client = mcp_client
        self.confidence_threshold = confidence_threshold

    async def __call__(self, state: MenuProcessorState) -> Dict[str, Any]:
        vegetarian_dishes = list(state.get("vegetarian_dishes") or [])
        request_id = state.get("request_id")

        if not vegetarian_dishes:
            logger.info("Calculator node received no vegetarian dishes.")
            summary = CalculationSummary(
                total_price=0.0,
                average_confidence=0.0,
                dish_count=0,
                uncertain_dishes=[],
                reasoning="No vegetarian dishes identified by the agent.",
            )
            return {
                "mcp_summary": summary,
                "total_price": 0.0,
                "reasoning_log": append_reasoning(state, summary.reasoning),
            }

        priced_dishes = _filter_priced_dishes(vegetarian_dishes)
        summary: Optional[CalculationSummary] = None

        if self.mcp_client and priced_dishes:
            try:
                summary = await self.mcp_client.calculate_total(
                    priced_dishes,
                    request_id=request_id,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("MCP calculation failed: %s", exc)

        if summary is None:
            summary = self._fallback_summary(vegetarian_dishes, priced_dishes)

        total_price = summary.total_price if summary else 0.0
        reasoning_line = (
            summary.reasoning
            if summary.reasoning
            else f"Calculated vegetarian total ${total_price:.2f} for {len(priced_dishes)} dishes."
        )

        return {
            "mcp_summary": summary,
            "total_price": total_price,
            "reasoning_log": append_reasoning(state, reasoning_line),
        }

    def _fallback_summary(
        self,
        vegetarian_dishes: Iterable[Dish],
        priced_dishes: Iterable[Dish],
    ) -> CalculationSummary:
        priced_list = list(priced_dishes)
        total_price = round(sum(dish.price or 0.0 for dish in priced_list), 2)
        confidences = [float(dish.confidence or 0.0) for dish in priced_list]
        avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
        missing = len(list(vegetarian_dishes)) - len(priced_list)

        uncertain = [
            dish.name
            for dish in priced_list
            if float(dish.confidence or 0.0) < self.confidence_threshold
        ]

        return CalculationSummary(
            total_price=total_price,
            average_confidence=avg_confidence,
            dish_count=len(priced_list),
            uncertain_dishes=uncertain,
            reasoning=(
                f"Fallback calculation used for {len(priced_list)} priced dishes; "
                f"{missing} dishes were missing prices."
            ),
            priced_dish_count=len(priced_list),
            missing_price_count=missing,
        )


def _filter_priced_dishes(dishes: Iterable[Dish]) -> list[Dish]:
    """Return only dishes with prices."""
    return [dish for dish in dishes if dish.price is not None]

