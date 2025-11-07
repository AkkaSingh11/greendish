import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pytest

# Ensure application packages resolve via the repo root and API directory
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
API_ROOT = ROOT_DIR / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from api.agents import MenuProcessorAgent  # noqa: E402
from api.llm import OpenRouterResponse  # noqa: E402
from models import CalculationSummary, Dish  # noqa: E402


class StubLLMClient:
    """Deterministic LLM stub for unit tests."""

    def __init__(self, responses: Sequence[Dict[str, Any]], model: str = "stub-model") -> None:
        self._responses = list(responses)
        self._model = model
        self.calls: List[str] = []

    async def complete_json(self, prompt: str, **_: Any) -> OpenRouterResponse:
        if not self._responses:
            raise AssertionError("No more stub responses available.")
        payload = self._responses.pop(0)
        self.calls.append(prompt)
        return OpenRouterResponse(
            data=payload,
            model=self._model,
            prompt_tokens=10,
            completion_tokens=12,
            total_tokens=22,
            latency=0.01,
        )


class StubMCPClient:
    """MCP client stub that accumulates provided dishes."""

    def __init__(self) -> None:
        self.calls = 0
        self.seen_requests: List[str] = []
        self.seen_dish_counts: List[int] = []

    async def calculate_total(self, dishes: List[Dish], request_id: str | None = None) -> CalculationSummary:
        self.calls += 1
        if request_id:
            self.seen_requests.append(request_id)
        self.seen_dish_counts.append(len(dishes))
        total = round(sum(dish.price or 0.0 for dish in dishes), 2)
        avg_confidence = (
            sum(float(dish.confidence or 0.0) for dish in dishes) / len(dishes) if dishes else 0.0
        )
        return CalculationSummary(
            total_price=total,
            average_confidence=round(avg_confidence, 3),
            dish_count=len(dishes),
            uncertain_dishes=[],
            reasoning=f"Stub summary for {len(dishes)} priced dishes.",
            priced_dish_count=len(dishes),
            missing_price_count=0,
        )


class NoopRAGNode:
    """RAG node stub that does nothing (used to bypass real service)."""

    async def __call__(self, state) -> Dict[str, Any]:
        return {
            "rag_context": None,
            "rag_lookups": state.get("rag_lookups", []),
            "requires_rag": False,
        }


class RecordingRAGNode:
    """RAG node stub that records invocations."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, state) -> Dict[str, Any]:
        self.calls += 1
        return {
            "rag_context": "Evidence: vegetarian soup examples.",
            "rag_lookups": state.get("rag_lookups", []) + [{"source": "stub"}],
            "requires_rag": False,
        }


@pytest.mark.asyncio
async def test_agent_classifies_dishes_and_calculates_total() -> None:
    dishes = [
        Dish(name="Margherita Pizza", price=12.5, raw_text="Margherita Pizza - $12.50"),
        Dish(name="Chicken Wings", price=9.0, raw_text="Chicken Wings - $9.00"),
    ]

    llm_stub = StubLLMClient(
        responses=[
            {"is_vegetarian": True, "confidence": 0.92, "reasoning": "Classic vegetarian pizza."},
            {"is_vegetarian": False, "confidence": 0.95, "reasoning": "Contains chicken."},
        ]
    )
    mcp_stub = StubMCPClient()

    agent = MenuProcessorAgent(llm_client=llm_stub, mcp_client=mcp_stub, rag_node=NoopRAGNode())
    state = await agent.run(dishes, request_id="test-run-1")

    assert len(state["classified_dishes"]) == 2
    assert len(state["vegetarian_dishes"]) == 1
    veg_dish = state["vegetarian_dishes"][0]
    assert veg_dish.name == "Margherita Pizza"
    assert veg_dish.is_vegetarian is True
    assert veg_dish.classification_method == "llm"
    assert state["total_price"] == pytest.approx(12.5)
    assert state["mcp_summary"].total_price == pytest.approx(12.5)
    assert state["llm_model"] == "stub-model"
    assert state["confidence_scores"]["Margherita Pizza"] == pytest.approx(0.92)
    assert len(state["reasoning_log"]) >= 2
    assert mcp_stub.calls == 1
    assert mcp_stub.seen_dish_counts == [1]
    assert mcp_stub.seen_requests == ["test-run-1"]


@pytest.mark.asyncio
async def test_agent_triggers_rag_route_on_low_confidence() -> None:
    dishes = [
        Dish(name="Mystery Soup", price=None, raw_text="Soup of the day"),
    ]

    llm_stub = StubLLMClient(
        responses=[
            {"is_vegetarian": False, "confidence": 0.2, "reasoning": "Insufficient information."},
            {"is_vegetarian": True, "confidence": 0.75, "reasoning": "Soup described as vegetable-based."},
        ]
    )
    rag_stub = RecordingRAGNode()

    agent = MenuProcessorAgent(llm_client=llm_stub, rag_node=rag_stub)
    state = await agent.run(dishes, request_id="test-run-2")

    assert rag_stub.calls == 1
    assert state["classified_dishes"][0].confidence == pytest.approx(0.75)
    assert state["classified_dishes"][0].is_vegetarian is True
    assert state["rag_context"] is None  # Cleared after reclassification
    assert state["rag_lookups"][-1]["source"] == "stub"
    assert len(llm_stub.calls) == 2
    assert "Mystery Soup" in state["confidence_scores"]
