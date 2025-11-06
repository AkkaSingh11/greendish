from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class VegetarianDish(BaseModel):
    """Dish payload expected from upstream services."""

    name: str
    price: float = Field(..., ge=0.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    signals: Optional[Dict[str, List[str]]] = None
    reasoning: Optional[str] = None


class CalculationRequest(BaseModel):
    """Structured request for the calculator tool."""

    request_id: Optional[str] = None
    dishes: List[VegetarianDish]

    @field_validator("dishes")
    @classmethod
    def validate_non_empty(cls, value: List[VegetarianDish]) -> List[VegetarianDish]:
        if not value:
            raise ValueError("At least one dish is required for calculation")
        return value


class DishSubtotal(BaseModel):
    """Per-dish contribution in the calculation."""

    name: str
    price: float
    confidence: float


class CalculationResponse(BaseModel):
    """Structured response emitted by the calculator tool."""

    total_price: float
    average_confidence: float
    dish_count: int
    uncertain_dishes: List[str] = Field(default_factory=list)
    reasoning: str
    priced_dish_count: Optional[int] = None
    missing_price_count: Optional[int] = None
    breakdown: List[DishSubtotal] = Field(default_factory=list)
