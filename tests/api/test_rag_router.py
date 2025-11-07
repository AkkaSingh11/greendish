import sys
from pathlib import Path
from unittest import mock

import importlib.util
import httpx
import pytest
from fastapi import FastAPI

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
API_ROOT = ROOT_DIR / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

with mock.patch("services.ocr_service.OCRService", autospec=True):
    rag_module_path = API_ROOT / "routers" / "rag.py"
    spec = importlib.util.spec_from_file_location("test_rag_router_module", rag_module_path)
    rag_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(rag_module)
    rag_router = rag_module.router

app = FastAPI()
app.include_router(rag_router)


class StubRAGService:
    def __init__(self) -> None:
        self.search_calls = []
        self.reseed_calls = []

    def search(self, text: str, top_k: int = 3):
        self.search_calls.append((text, top_k))
        return [
            mock.Mock(name="Match", name_attr="Margherita Pizza"),
        ]

    def reseed(self, force: bool = False) -> int:
        self.reseed_calls.append(force)
        return 42


def _make_match(name: str, category: str, description: str, score: float, chunk_index: int):
    match = mock.Mock()
    match.name = name
    match.category = category
    match.description = description
    match.score = score
    match.chunk_index = chunk_index
    return match


@pytest.mark.asyncio
async def test_rag_search_returns_matches(monkeypatch):
    stub_service = StubRAGService()
    stub_service.search = mock.Mock(
        return_value=[
            _make_match("Margherita Pizza", "vegetarian", "Tomato, mozzarella, basil.", 0.92, 0),
            _make_match("Pepperoni Pizza", "non-vegetarian", "Includes pepperoni slices.", 0.65, 1),
        ]
    )

    monkeypatch.setattr(rag_module, "get_rag_service", lambda: stub_service)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/rag/search", params={"query": "pizza", "top_k": 2})
    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "pizza"
    assert len(payload["matches"]) == 2
    assert payload["matches"][0]["name"] == "Margherita Pizza"
    stub_service.search.assert_called_once_with("pizza", top_k=2)


@pytest.mark.asyncio
async def test_rag_reseed_endpoint(monkeypatch):
    stub_service = StubRAGService()
    monkeypatch.setattr(rag_module, "get_rag_service", lambda: stub_service)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/rag/reseed", json={"force": True})
    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["documents"] == 42
    assert stub_service.reseed_calls == [True]
