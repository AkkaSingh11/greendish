import io
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, UploadFile

# Ensure application modules are importable when running from repo root
ROOT_DIR = Path(__file__).resolve().parents[2]
API_ROOT = ROOT_DIR / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from models import Dish, OCRResult  # noqa: E402

with mock.patch("pytesseract.get_tesseract_version", return_value="Fake 0.0"):
    from routers.menu import get_mcp_tools, process_menu  # noqa: E402


def _dummy_image() -> bytes:
    """Return a minimal PNG header to satisfy the upload parser."""
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
        b"\x90wS\xde"
        b"\x00\x00\x00\x0bIDAT\x08\xd7c\xf8\x0f\x00\x01\x01\x01\x00"
        b"\x18\xdd\xdc\x94"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _upload_file() -> UploadFile:
    return UploadFile(
        filename="dummy.png",
        file=io.BytesIO(_dummy_image()),
    )


@pytest.mark.asyncio
async def test_process_menu_ai_mode_runs_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    upload = _upload_file()

    fake_ocr = [
        OCRResult(
            image_name="menu.png",
            raw_text="AI Soup - $12.00",
            confidence=95.0,
            processing_time_ms=12.3,
        )
    ]

    async def fake_extract_text(files):
        return fake_ocr

    def fake_parse_menu_text(raw_text: str):
        return [
            Dish(
                name="AI Soup",
                price=12.0,
                raw_text=raw_text,
            )
        ]

    def fake_stats(dishes):
        total = len(dishes)
        priced = sum(1 for dish in dishes if dish.price is not None)
        return {
            "total_dishes": total,
            "dishes_with_prices": priced,
            "dishes_without_prices": total - priced,
            "price_coverage": 1.0 if total else 0.0,
            "average_confidence": 0.9,
        }

    class StubBatchClassifier:
        async def classify(self, dishes, request_id):
            return [
                SimpleNamespace(
                    index=0,
                    name="AI Soup",
                    is_vegetarian=True,
                    confidence=0.92,
                    reasoning="Menu description mentions only vegetables.",
                    signals=["dish name implies vegetables"],
                )
            ]

    class StubAgent:
        def __init__(self) -> None:
            self.calls = []

        async def run(self, dishes, request_id, mode):
            self.calls.append((dishes, request_id, mode))
            classified = [
                Dish(
                    name="AI Soup",
                    price=12.0,
                    raw_text="AI Soup - $12.00",
                    is_vegetarian=True,
                    confidence=0.88,
                    classification_method="llm",
                    reasoning="Stubbed vegetarian classification.",
                )
            ]
            return {
                "classified_dishes": classified,
                "vegetarian_dishes": classified,
                "reasoning_log": ["Stub reasoning"],
            }

    stub_agent = StubAgent()
    stub_classifier = StubBatchClassifier()

    monkeypatch.setattr("routers.menu.extract_text", fake_extract_text)
    monkeypatch.setattr("routers.menu.parser_service.parse_menu_text", fake_parse_menu_text)
    monkeypatch.setattr("routers.menu.parser_service.get_parsing_stats", fake_stats)
    monkeypatch.setattr("routers.menu.get_ai_agent", lambda: stub_agent)
    monkeypatch.setattr("routers.menu.get_batch_classifier", lambda: stub_classifier)
    monkeypatch.setattr("routers.menu.mcp_client", None)

    response = await process_menu(files=[upload], mode="ai", use_rag=False)

    assert response.mode == "ai"
    assert len(response.dishes) == 1
    assert response.dishes[0].is_vegetarian is False
    assert response.total_price == pytest.approx(0.0)
    assert response.calculation_summary is None
    assert not stub_agent.calls, "Agent should not run when RAG is disabled"


@pytest.mark.asyncio
async def test_process_menu_ai_mode_triggers_rag(monkeypatch: pytest.MonkeyPatch) -> None:
    upload = _upload_file()

    fake_ocr = [
        OCRResult(
            image_name="menu.png",
            raw_text="AI Soup - $12.00",
            confidence=95.0,
            processing_time_ms=12.3,
        )
    ]

    async def fake_extract_text(files):
        return fake_ocr

    def fake_parse_menu_text(raw_text: str):
        return [Dish(name="AI Soup", price=12.0, raw_text=raw_text)]

    def fake_stats(_):
        return {
            "total_dishes": 1,
            "dishes_with_prices": 1,
            "dishes_without_prices": 0,
            "price_coverage": 1.0,
            "average_confidence": 0.9,
        }

    class StubBatchClassifier:
        async def classify(self, dishes, request_id):
            return [
                SimpleNamespace(
                    index=0,
                    name="AI Soup",
                    is_vegetarian=False,
                    confidence=0.2,
                    reasoning="Needs more context.",
                    signals=["vegetable hints"],
                )
            ]

    class StubAgent:
        def __init__(self) -> None:
            self.calls = 0

        async def run(self, dishes, request_id, mode):
            self.calls += 1
            return {
                "classified_dishes": [
                    Dish(
                        name="AI Soup",
                        price=12.0,
                        raw_text="AI Soup - $12.00",
                        is_vegetarian=True,
                        confidence=0.95,
                        classification_method="rag",
                        reasoning="Stubbed agent result.",
                    )
                ],
                "vegetarian_dishes": [
                    Dish(
                        name="AI Soup",
                        price=12.0,
                        raw_text="AI Soup - $12.00",
                        is_vegetarian=True,
                        confidence=0.95,
                        classification_method="rag",
                        reasoning="Stubbed agent result.",
                    )
                ],
                "reasoning_log": ["stub"],
            }

    stub_agent = StubAgent()
    stub_classifier = StubBatchClassifier()

    monkeypatch.setattr("routers.menu.extract_text", fake_extract_text)
    monkeypatch.setattr("routers.menu.parser_service.parse_menu_text", fake_parse_menu_text)
    monkeypatch.setattr("routers.menu.parser_service.get_parsing_stats", fake_stats)
    monkeypatch.setattr("routers.menu.get_ai_agent", lambda: stub_agent)
    monkeypatch.setattr("routers.menu.get_batch_classifier", lambda: stub_classifier)
    monkeypatch.setattr("routers.menu.mcp_client", None)

    response = await process_menu(files=[upload], mode="ai", use_rag=True)

    assert response.mode == "ai"
    assert response.dishes[0].classification_method == "rag"
    assert stub_agent.calls == 1, "Agent should run when RAG is enabled"


@pytest.mark.asyncio
async def test_process_menu_invalid_mode_bad_request() -> None:
    upload = _upload_file()

    with pytest.raises(HTTPException) as exc_info:
        await process_menu(files=[upload], mode="experimental")

    assert exc_info.value.status_code == 400
    assert "Unsupported processing mode" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_mcp_tools_success() -> None:
    mock_tools = [
        {
            "name": "calculate_vegetarian_total",
            "title": "Calculator",
            "description": "Sum vegetarian dishes",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
        }
    ]

    with mock.patch("routers.menu.mcp_client") as mock_client:
        mock_client.list_tools = AsyncMock(return_value=mock_tools)
        result = await get_mcp_tools()

    assert len(result) == 1
    assert result[0].name == "calculate_vegetarian_total"
    assert result[0].description == "Sum vegetarian dishes"


@pytest.mark.asyncio
async def test_get_mcp_tools_missing_client() -> None:
    with mock.patch("routers.menu.mcp_client", None):
        with pytest.raises(HTTPException) as exc_info:
            await get_mcp_tools()

    assert exc_info.value.status_code == 503
