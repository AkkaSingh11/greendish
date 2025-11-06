import io
import sys
from pathlib import Path
from unittest import mock
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, UploadFile

# Ensure application modules are importable when running from repo root
ROOT_DIR = Path(__file__).resolve().parents[2]
API_ROOT = ROOT_DIR / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

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
async def test_process_menu_ai_mode_not_implemented() -> None:
    upload = _upload_file()

    with pytest.raises(HTTPException) as exc_info:
        await process_menu(files=[upload], mode="ai")

    assert exc_info.value.status_code == 501
    assert "not available yet" in exc_info.value.detail


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
