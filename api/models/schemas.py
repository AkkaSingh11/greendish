from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
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
    signals: Optional[Dict[str, List[str]]] = None


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
    calculation_summary: Optional["CalculationSummary"] = None
    mode: str = "non-ai"


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: str


class CalculationSummary(BaseModel):
    """Aggregated vegetarian totals returned from MCP server."""

    total_price: float
    average_confidence: float
    dish_count: int
    uncertain_dishes: List[str] = Field(default_factory=list)
    reasoning: str
    priced_dish_count: Optional[int] = None
    missing_price_count: Optional[int] = None


class MCPToolInfo(BaseModel):
    """Metadata describing an MCP tool exposed by the calculator service."""

    name: str
    title: Optional[str] = None
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None


class RagMatchModel(BaseModel):
    """Single RAG match returned from the retrieval service."""

    name: str
    category: str
    description: str
    score: float
    chunk_index: Optional[int] = None


class RagSearchResponse(BaseModel):
    """Response model for RAG search endpoint."""

    query: str
    top_k: int
    matches: List[RagMatchModel]


ProcessMenuResponse.model_rebuild()
