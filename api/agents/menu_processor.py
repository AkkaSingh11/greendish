"""LangGraph menu processor assembly."""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from langgraph.graph import END, StateGraph

from config import settings
from models import Dish
from services.mcp_client import MCPClient, get_mcp_client
from services.rag_service import RAGService
from .nodes import CalculatorNode, ClassifierNode, RAGNode
from .state import MenuProcessorState, create_initial_state

logger = logging.getLogger(__name__)


class MenuProcessorAgent:
    """Stateful LangGraph agent that classifies dishes and calculates totals."""

    def __init__(
        self,
        *,
        openrouter_client,
        mcp_client: Optional[MCPClient] = None,
        rag_service: Optional[RAGService] = None,
        rag_node: Optional[RAGNode] = None,
        confidence_threshold: Optional[float] = None,
    ) -> None:
        if openrouter_client is None:
            raise ValueError("openrouter_client is required for the LangGraph agent.")

        self.confidence_threshold = confidence_threshold or settings.confidence_threshold
        self.openrouter_client = openrouter_client
        self.mcp_client = mcp_client or get_mcp_client()
        self.rag_service = rag_service
        if rag_node is not None:
            self.rag_node = rag_node
        else:
            if self.rag_service is None and settings.rag_enabled:
                self.rag_service = RAGService()
            self.rag_node = RAGNode(self.rag_service, top_k=settings.rag_top_k)

        classifier_node = ClassifierNode(
            openrouter_client,
            confidence_threshold=self.confidence_threshold,
        )
        calculator_node = CalculatorNode(
            self.mcp_client,
            confidence_threshold=self.confidence_threshold,
        )

        graph = StateGraph(MenuProcessorState)
        graph.add_node("classify_dish", classifier_node)
        graph.add_node("rag_lookup", self.rag_node)
        graph.add_node("advance", self._advance_node)
        graph.add_node("calculate_total", calculator_node)

        graph.set_entry_point("classify_dish")

        graph.add_conditional_edges(
            "classify_dish",
            self._post_classification_route,
            {
                "rag": "rag_lookup",
                "advance": "advance",
                "finalize": "calculate_total",
            },
        )
        graph.add_edge("rag_lookup", "classify_dish")
        graph.add_conditional_edges(
            "advance",
            self._advance_route,
            {
                "classify": "classify_dish",
                "finalize": "calculate_total",
            },
        )
        graph.add_edge("calculate_total", END)

        self._graph = graph.compile()

    async def run(
        self,
        dishes: Sequence[Dish],
        *,
        request_id: str,
        mode: str = "ai",
    ) -> MenuProcessorState:
        """Execute the LangGraph agent for the provided dishes."""
        initial_state = create_initial_state(dishes, request_id=request_id, mode=mode)
        logger.info(
            "Starting LangGraph menu processor for request %s with %d dishes.",
            request_id,
            len(dishes),
        )
        config = {
            "configurable": {
                "thread_id": request_id,
            },
            "metadata": {
                "request_id": request_id,
                "mode": mode,
            },
        }
        return await self._graph.ainvoke(initial_state, config=config)

    async def _advance_node(self, state: MenuProcessorState) -> MenuProcessorState:
        dishes = state.get("dishes") or []
        index = state.get("current_dish_index", 0)
        next_index = index + 1 if dishes else index
        return MenuProcessorState(
            current_dish_index=next_index,
            rag_context=None,
            requires_rag=False,
        )

    def _post_classification_route(self, state: MenuProcessorState) -> str:
        dishes = state.get("dishes") or []
        if not dishes:
            return "finalize"

        if state.get("requires_rag"):
            return "rag"

        index = state.get("current_dish_index", 0)
        if index >= len(dishes):
            return "finalize"
        return "advance"

    def _advance_route(self, state: MenuProcessorState) -> str:
        dishes = state.get("dishes") or []
        index = state.get("current_dish_index", 0)
        if index >= len(dishes):
            return "finalize"
        return "classify"
