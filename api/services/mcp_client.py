from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel, Field

from config import settings
from models import CalculationSummary, Dish

logger = logging.getLogger(__name__)


class VegetarianDishPayload(BaseModel):
    """Payload forwarded to the MCP calculator tool."""

    name: str
    price: float = Field(..., ge=0.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    signals: Optional[Dict[str, List[str]]] = None
    reasoning: Optional[str] = None


class CalculationRequest(BaseModel):
    """Request envelope sent to the MCP calculator tool."""

    request_id: Optional[str] = None
    dishes: List[VegetarianDishPayload]


class MCPClient:
    """Thin MCP client wrapper for the vegetarian calculator tool."""

    def __init__(self, base_url: str, timeout: int = 30) -> None:
        self.endpoint = base_url.rstrip("/") + "/mcp"
        self.timeout = timeout

    async def calculate_total(
        self,
        dishes: List[Dish],
        request_id: Optional[str] = None,
    ) -> CalculationSummary:
        payload = CalculationRequest(
            request_id=request_id,
            dishes=[
                VegetarianDishPayload(
                    name=dish.name,
                    price=float(dish.price),
                    confidence=float(dish.confidence or 0.0),
                    signals=dish.signals,
                    reasoning=dish.reasoning,
                )
                for dish in dishes
                if dish.price is not None
            ],
        )

        if not payload.dishes:
            logger.info("No dishes with prices provided; returning zero total.")
            return CalculationSummary(
                total_price=0.0,
                average_confidence=0.0,
                dish_count=0,
                uncertain_dishes=[],
                reasoning="No vegetarian dishes with prices were supplied.",
            )

        async with streamablehttp_client(self.endpoint, timeout=self.timeout) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                logger.debug(
                    "Calling MCP tool calculate_vegetarian_total with %d dishes", len(payload.dishes)
                )
                result = await session.call_tool(
                    "calculate_vegetarian_total",
                    arguments={"payload": payload.model_dump()},
                )

        structured = getattr(result, "structuredContent", None)
        if structured:
            return CalculationSummary.model_validate(structured)

        # Fallback to parsing text blocks if structured output unavailable
        text_chunks = [
            block.text
            for block in getattr(result, "content", [])
            if isinstance(block, types.TextContent)
        ]
        reasoning = text_chunks[0] if text_chunks else "Calculation completed."
        return CalculationSummary(
            total_price=0.0,
            average_confidence=0.0,
            dish_count=len(payload.dishes),
            uncertain_dishes=[],
            reasoning=reasoning,
        )


def get_mcp_client() -> Optional[MCPClient]:
    """Factory to obtain a configured MCP client based on settings."""
    if not settings.mcp_server_url:
        return None

    return MCPClient(
        base_url=settings.mcp_server_url,
        timeout=settings.mcp_timeout,
    )
