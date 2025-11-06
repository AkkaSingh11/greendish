from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    timestamp: str
    version: str
    services: dict[str, str] = Field(default_factory=dict)


class OCRResult(BaseModel):
    """OCR extraction result for a single image."""
    image_name: str
    raw_text: str
    confidence: Optional[float] = None
    processing_time_ms: float


class Dish(BaseModel):
    """Structured dish information."""
    name: str
    price: Optional[float] = None
    raw_text: str
    is_vegetarian: Optional[bool] = None
    confidence: Optional[float] = None
    classification_method: Optional[str] = None  # "keyword", "llm", "rag"
    reasoning: Optional[str] = None


class ParsedDish(BaseModel):
    """Simplified dish payload returned to clients."""
    name: str
    price: Optional[float] = None
    raw_text: str


class ParsedMenu(BaseModel):
    """Structured menu payload produced after parsing."""
    dishes: List[ParsedDish] = Field(default_factory=list)
    total_dishes: int = 0
    dishes_with_prices: int = 0
    dishes_without_prices: int = 0
    price_coverage: float = 0.0
    average_confidence: float = 0.0
    average_ocr_confidence: Optional[float] = None


class ClassificationResult(BaseModel):
    """Classification result for dishes."""
    dish: Dish
    is_vegetarian: bool
    confidence: float
    method: str
    reasoning: Optional[str] = None
    rag_matches: Optional[List[dict]] = None


class ProcessMenuResponse(BaseModel):
    """Complete menu processing response."""
    request_id: str
    total_images: int
    ocr_results: List[OCRResult]
    dishes: List[Dish]
    parsed_menu: ParsedMenu
    vegetarian_dishes: List[Dish]
    total_price: float
    processing_time_ms: float
    langsmith_trace_url: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: str
