import importlib.util
import sys
from pathlib import Path

import pytest

MCP_ROOT = Path(__file__).resolve().parents[2] / "mcp-server"
MODELS_PATH = MCP_ROOT / "models.py"
CALCULATOR_PATH = MCP_ROOT / "tools" / "calculator.py"


def _load_attr(module_path: Path, attr: str):
    sys.path.insert(0, str(MCP_ROOT))
    module_name = f"_mcp_tests_{module_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    finally:
        try:
            sys.path.remove(str(MCP_ROOT))
        except ValueError:  # pragma: no cover - defensive
            pass
        sys.modules.pop("models", None)
        sys.modules.pop("config", None)
    return getattr(module, attr)


VegetarianDish = _load_attr(MODELS_PATH, "VegetarianDish")
compute_totals = _load_attr(CALCULATOR_PATH, "compute_totals")


def test_compute_totals_aggregates_values() -> None:
    dishes = [
        VegetarianDish(name="Paneer Tikka", price=12.5, confidence=0.9),
        VegetarianDish(name="Veg Biryani", price=10.0, confidence=0.6),
    ]

    total, avg_conf, uncertain, breakdown, missing = compute_totals(
        dishes,
        confidence_threshold=0.7,
    )

    assert pytest.approx(total) == 22.5
    assert pytest.approx(avg_conf, rel=1e-3) == 0.75
    assert uncertain == ["Veg Biryani"]
    assert len(breakdown) == 2
    assert missing == 0
    assert breakdown[0].name == "Paneer Tikka"


def test_compute_totals_handles_empty_iterable() -> None:
    total, avg_conf, uncertain, breakdown, missing = compute_totals(
        [],
        confidence_threshold=0.8,
    )

    assert total == 0.0
    assert avg_conf == 0.0
    assert uncertain == []
    assert breakdown == []
    assert missing == 0
