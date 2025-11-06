from typing import Iterable, List, Tuple

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

from config import settings
from models import CalculationRequest, CalculationResponse, DishSubtotal, VegetarianDish


def compute_totals(
    dishes: Iterable[VegetarianDish],
    *,
    confidence_threshold: float,
) -> Tuple[float, float, List[str], List[DishSubtotal], int]:
    """
    Aggregate totals and confidence metrics for vegetarian dishes.

    Args:
        dishes: Iterable of vegetarian dishes with price/confidence data.
        confidence_threshold: Threshold for flagging uncertain dishes.

    Returns:
        Tuple of (total_price, average_confidence, uncertain_names, breakdown, missing_prices)
    """
    totals: List[DishSubtotal] = []
    total_price = 0.0
    confidences: List[float] = []
    uncertain: List[str] = []
    missing_prices = 0

    for dish in dishes:
        if dish.price is None:
            missing_prices += 1
            continue

        total_price += dish.price
        confidences.append(dish.confidence)
        totals.append(
            DishSubtotal(
                name=dish.name,
                price=dish.price,
                confidence=dish.confidence,
            )
        )
        if dish.confidence < confidence_threshold:
            uncertain.append(dish.name)

    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return total_price, average_confidence, uncertain, totals, missing_prices


def register_calculator(mcp: FastMCP) -> None:
    """Register calculator tool and debugging resource with the FastMCP instance."""

    @mcp.tool()
    async def calculate_vegetarian_total(
        payload: CalculationRequest,
        ctx: Context[ServerSession, None],
    ) -> CalculationResponse:
        """Sum vegetarian dishes and report aggregate confidence."""
        await ctx.debug(
            f"Received {len(payload.dishes)} vegetarian dishes for request {payload.request_id or 'n/a'}"
        )

        total_price, avg_conf, uncertain, breakdown, missing_prices = compute_totals(
            payload.dishes,
            confidence_threshold=settings.confidence_threshold,
        )

        priced_count = len(payload.dishes) - missing_prices
        total_dishes = len(payload.dishes)
        reasoning = (
            f"Out of {total_dishes} vegetarian dishes, {priced_count} had prices and "
            f"{missing_prices} were missing prices. Total price ${total_price:.2f} with "
            f"average confidence {avg_conf:.2f}."
        )

        if uncertain:
            await ctx.info(
                f"{len(uncertain)} dish(es) fell below confidence threshold: {', '.join(uncertain[:5])}"
            )

        return CalculationResponse(
            total_price=round(total_price, 2),
            average_confidence=round(avg_conf, 3),
            dish_count=len(payload.dishes),
            priced_dish_count=priced_count,
            missing_price_count=missing_prices,
            uncertain_dishes=uncertain,
            reasoning=reasoning,
            breakdown=breakdown,
        )

    @mcp.resource("calculator://config")
    def calculator_config() -> str:
        """Expose basic calculator configuration for debugging."""
        return (
            f"{settings.app_name} running with confidence threshold "
            f"{settings.confidence_threshold}"
        )
