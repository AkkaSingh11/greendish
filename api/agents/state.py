"""Shared state definitions and helpers for the LangGraph menu processor."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar, TypedDict

from models import CalculationSummary, ClassificationResult, Dish

T = TypeVar("T")


def _dish_signature(dish: Dish) -> Tuple[str, Optional[float], str]:
    """Return a signature tuple used to uniquely identify a dish."""
    return dish.name, dish.price, dish.raw_text


class MenuProcessorState(TypedDict, total=False):
    """Graph state propagated between LangGraph nodes."""

    request_id: str
    mode: str
    dishes: List[Dish]
    current_dish_index: int
    classified_dishes: List[Dish]
    vegetarian_dishes: List[Dish]
    classification_results: List[ClassificationResult]
    confidence_scores: Dict[str, float]
    reasoning_log: List[str]
    rag_lookups: List[Dict[str, Any]]
    rag_context: Optional[str]
    requires_rag: bool
    llm_model: Optional[str]
    mcp_summary: Optional[CalculationSummary]
    total_price: float


def create_initial_state(
    dishes: Sequence[Dish],
    *,
    request_id: str,
    mode: str = "ai",
) -> MenuProcessorState:
    """Create the initial LangGraph state for agent execution."""
    return MenuProcessorState(
        request_id=request_id,
        mode=mode,
        dishes=list(dishes),
        current_dish_index=0,
        classified_dishes=[],
        vegetarian_dishes=[],
        classification_results=[],
        confidence_scores={},
        reasoning_log=[],
        rag_lookups=[],
        rag_context=None,
        requires_rag=False,
        llm_model=None,
        mcp_summary=None,
        total_price=0.0,
    )


def get_current_dish(state: MenuProcessorState) -> Optional[Dish]:
    """Return the dish currently being processed."""
    dishes = state.get("dishes") or []
    index = state.get("current_dish_index", 0)
    if 0 <= index < len(dishes):
        return dishes[index]
    return None


def is_last_dish(state: MenuProcessorState) -> bool:
    """Return True when all dishes have been processed."""
    dishes = state.get("dishes") or []
    index = state.get("current_dish_index", 0)
    return index >= len(dishes) - 1


def advance_index(state: MenuProcessorState) -> int:
    """Return the next dish index based on the current state."""
    return state.get("current_dish_index", 0) + 1


def clone_dish(dish: Dish, **updates: Any) -> Dish:
    """Return a copy of the dish with updates applied."""
    return dish.model_copy(update=updates)


def upsert_by_index(items: Sequence[T], index: int, value: T) -> List[T]:
    """Return a list where the item at index is replaced or appended."""
    mutable = list(items)
    if index < len(mutable):
        mutable[index] = value
    else:
        mutable.append(value)
    return mutable


def update_vegetarian_dishes(
    existing: Iterable[Dish],
    dish: Dish,
    *,
    include: bool,
) -> List[Dish]:
    """Return a vegetarian dish list with the dish added or removed."""
    signature = _dish_signature(dish)
    updated = [item for item in existing if _dish_signature(item) != signature]
    if include:
        updated.append(dish)
    return updated


def append_reasoning(state: MenuProcessorState, message: str) -> List[str]:
    """Append a reasoning entry to the state log."""
    log = list(state.get("reasoning_log", []))
    log.append(message)
    return log
