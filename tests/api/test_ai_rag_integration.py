import tempfile
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
API_ROOT = ROOT_DIR / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

import pytest

from agents import MenuProcessorAgent  # noqa: E402
from llm import OpenRouterResponse  # noqa: E402
from models import Dish  # noqa: E402
from services.rag_service import RAGService  # noqa: E402


class StubLLMClient:
    """Deterministic client that requires RAG evidence to boost confidence."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete_json(self, prompt: str, **kwargs) -> OpenRouterResponse:
        self.calls += 1
        if self.calls == 1:
            payload = {
                "is_vegetarian": False,
                "confidence": 0.2,
                "reasoning": "Initial pass unsure without additional evidence.",
            }
        else:
            payload = {
                "is_vegetarian": True,
                "confidence": 0.9,
                "reasoning": "Matches vegetarian evidence from RAG context.",
            }

        return OpenRouterResponse(
            data=payload,
            model="stub-model",
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            latency=0.01,
        )


@pytest.mark.asyncio
async def test_menu_processor_agent_uses_rag_for_uncertain_dish():
    dish = Dish(
        name="Margherita Pizza",
        price=12.0,
        raw_text="Margherita Pizza - Fresh mozzarella, basil, tomato sauce $12",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        rag_service = RAGService(
            persist_path=tmpdir,
            seed_path=API_ROOT / "data" / "vegetarian_db.json",
        )
        rag_service.reseed(force=True)

        agent = MenuProcessorAgent(
            llm_client=StubLLMClient(),
            rag_service=rag_service,
        )

        state = await agent.run([dish], request_id="rag-test")

        assert len(state["vegetarian_dishes"]) == 1
        veg_dish = state["vegetarian_dishes"][0]
        assert veg_dish.name == "Margherita Pizza"
        assert veg_dish.classification_method == "rag"
        assert veg_dish.confidence and veg_dish.confidence > 0.8
        assert rag_service.collection.count() > 0
