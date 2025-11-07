import logging
import time
import uuid
from typing import List, Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Query, status

from models import (
    OCRResult,
    ProcessMenuResponse,
    ErrorResponse,
    ParsedMenu,
    ParsedDish,
    Dish,
    CalculationSummary,
    MCPToolInfo,
)
from agents import MenuProcessorAgent
from llm.openrouter_client import OpenRouterClient, OpenRouterError
from services import (
    OCRService,
    KeywordClassifier,
    KeywordClassifierError,
    MCPClient,
    LLMBatchClassifier,
    get_mcp_client,
)
from services.parser_service import ParserService
from config import settings
from agents.utils import normalize_signals

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["menu"])

SUPPORTED_MODES = {"non-ai", "ai"}
DEFAULT_MODE = "non-ai"

# Initialize services
ocr_service = OCRService()
parser_service = ParserService()

keyword_classifier = None
if settings.keyword_mode_enabled:
    try:
        keyword_classifier = KeywordClassifier(
            confidence_threshold=settings.confidence_threshold,
            fuzzy_threshold=settings.keyword_fuzzy_threshold,
        )
    except KeywordClassifierError as exc:
        logger.warning("Keyword classifier unavailable: %s", exc)

mcp_client: Optional[MCPClient] = get_mcp_client()
openrouter_client: Optional[OpenRouterClient] = None
ai_agent: Optional[MenuProcessorAgent] = None
batch_classifier: Optional[LLMBatchClassifier] = None


def get_openrouter_client() -> OpenRouterClient:
    """Return a shared OpenRouter client instance."""
    global openrouter_client

    if not settings.openrouter_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENROUTER_API_KEY is not configured. AI processing mode is unavailable.",
        )

    if openrouter_client is None:
        try:
            openrouter_client = OpenRouterClient()
        except OpenRouterError as exc:
            logger.error("Failed to initialize OpenRouter client: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to initialize OpenRouter client. Check API key and configuration.",
            ) from exc

    return openrouter_client


def get_ai_agent() -> MenuProcessorAgent:
    """Return a shared LangGraph menu processor agent."""
    global ai_agent

    if ai_agent is not None:
        return ai_agent

    client = get_openrouter_client()
    ai_agent = MenuProcessorAgent(
        openrouter_client=client,
        mcp_client=mcp_client,
    )
    return ai_agent


def get_batch_classifier() -> LLMBatchClassifier:
    """Return a shared batch classifier instance."""
    global batch_classifier
    if batch_classifier is None:
        client = get_openrouter_client()
        batch_classifier = LLMBatchClassifier(client)
    return batch_classifier


async def _calculate_totals(request_id: str, vegetarian_dishes: List[Dish]) -> tuple[float, Optional[CalculationSummary]]:
    """Calculate vegetarian totals using the MCP service with local fallback."""
    if not vegetarian_dishes:
        return 0.0, None

    priced_dishes = [dish for dish in vegetarian_dishes if dish.price is not None]
    if mcp_client and priced_dishes:
        try:
            summary = await mcp_client.calculate_total(priced_dishes, request_id=request_id)
            return summary.total_price, summary
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to calculate total via MCP: %s", exc)

    confidences = [float(dish.confidence or 0.0) for dish in priced_dishes]
    average_confidence = (sum(confidences) / len(confidences)) if confidences else 0.0
    total_price = round(sum(float(dish.price or 0.0) for dish in priced_dishes), 2)
    missing_count = len(vegetarian_dishes) - len(priced_dishes)

    summary = CalculationSummary(
        total_price=total_price,
        average_confidence=round(average_confidence, 3),
        dish_count=len(priced_dishes),
        priced_dish_count=len(priced_dishes),
        missing_price_count=missing_count,
        uncertain_dishes=[
            dish.name
            for dish in priced_dishes
            if float(dish.confidence or 0.0) < settings.confidence_threshold
        ],
        reasoning=(
            f"Out of {len(vegetarian_dishes)} vegetarian dishes, "
            f"{len(priced_dishes)} had prices and {missing_count} were missing prices. "
            "Local fallback calculation executed."
        ),
    )

    return total_price, summary


@router.post(
    "/extract-text",
    response_model=List[OCRResult],
    summary="Extract text from menu images using OCR",
)
async def extract_text(
    files: List[UploadFile] = File(..., description="Menu images (1-5 files)")
):
    """
    Extract text from uploaded menu images using Tesseract OCR.

    - **files**: List of image files (JPEG, PNG, WEBP)
    - Returns OCR results with extracted text and confidence scores
    """
    # Validate number of files
    if len(files) > settings.max_images:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum {settings.max_images} images allowed.",
        )

    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided.")

    results = []

    for file in files:
        try:
            # Read file content
            content = await file.read()

            # Validate image
            is_valid, error_msg = ocr_service.validate_image(content, file.filename)
            if not is_valid:
                raise HTTPException(status_code=400, detail=error_msg)

            # Extract text
            text, confidence, processing_time = ocr_service.extract_text(
                content, file.filename
            )

            results.append(
                OCRResult(
                    image_name=file.filename,
                    raw_text=text,
                    confidence=confidence,
                    processing_time_ms=processing_time,
                )
            )

            logger.info(f"Successfully processed {file.filename}")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process {file.filename}: {str(e)}",
            )

    return results


@router.post(
    "/process-menu",
    response_model=ProcessMenuResponse,
    summary="Process menu images with OCR and keyword-based vegetarian classification",
)
async def process_menu(
    files: List[UploadFile] = File(..., description="Menu images (1-5 files)"),
    mode: str = Query(
        DEFAULT_MODE,
        description="Processing mode. Use 'non-ai' for the deterministic pipeline or 'ai' for the LangGraph agent.",
    ),
    use_rag: Optional[bool] = Query(
        None,
        description="Override whether the AI pipeline should invoke RAG for low-confidence dishes.",
    ),
):
    """
    Process menu images to extract dishes, classify vegetarian options, and calculate totals
    using the selected pipeline mode.

    - non-ai: deterministic keyword pipeline (OCR → Parse → Keyword Match → MCP totals)
    - ai: LangGraph agent pipeline leveraging LLM classification with RAG support
    """
    normalized_mode = mode.strip().lower()
    if normalized_mode not in SUPPORTED_MODES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported processing mode '{mode}'. Supported modes: {', '.join(sorted(SUPPORTED_MODES))}.",
        )

    if normalized_mode == "non-ai" and not settings.keyword_mode_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Keyword-based processing is disabled. Enable keyword_mode_enabled to use the non-ai pipeline.",
        )

    start_time = time.time()
    request_id = str(uuid.uuid4())

    logger.info(
        "Processing menu request %s with %d image(s) using mode=%s",
        request_id,
        len(files),
        normalized_mode,
    )

    # Extract text from images
    ocr_results_list = await extract_text(files)

    # Phase 2: Parse dishes from OCR text
    all_dishes = []
    for ocr_result in ocr_results_list:
        try:
            # Parse dishes from OCR text
            dishes = parser_service.parse_menu_text(ocr_result.raw_text)
            all_dishes.extend(dishes)

            # Log parsing results
            stats = parser_service.get_parsing_stats(dishes)
            logger.info(
                f"Parsed {stats['total_dishes']} dishes from {ocr_result.image_name} "
                f"({stats['dishes_with_prices']} with prices, "
                f"avg confidence: {stats['average_confidence']:.2f})"
            )
        except Exception as e:
            logger.error(f"Error parsing dishes from {ocr_result.image_name}: {str(e)}")
            # Continue processing other images even if one fails

    overall_stats = parser_service.get_parsing_stats(all_dishes)
    ocr_confidences = [res.confidence for res in ocr_results_list if res.confidence is not None]
    avg_ocr_confidence = (
        round(sum(ocr_confidences) / len(ocr_confidences), 2)
        if ocr_confidences
        else None
    )

    parsed_menu = ParsedMenu(
        dishes=[
            ParsedDish(
                name=dish.name,
                price=dish.price,
                raw_text=dish.raw_text,
            )
            for dish in all_dishes
        ],
        total_dishes=overall_stats["total_dishes"],
        dishes_with_prices=overall_stats["dishes_with_prices"],
        dishes_without_prices=overall_stats["dishes_without_prices"],
        price_coverage=overall_stats["price_coverage"],
        average_confidence=overall_stats["average_confidence"],
        average_ocr_confidence=avg_ocr_confidence,
    )
    final_dishes: List[Dish] = []
    vegetarian_dishes: List[Dish] = []
    calculation_summary: Optional[CalculationSummary] = None
    total_price = 0.0

    effective_rag = settings.rag_enabled if use_rag is None else use_rag

    if normalized_mode == "ai":
        classifier = get_batch_classifier()
        try:
            batch_items = await classifier.classify(all_dishes, request_id=request_id)
        except OpenRouterError as exc:
            logger.error("OpenRouter request failed for %s: %s", request_id, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"AI classification failed due to LLM provider error: {exc}",
            ) from exc

        batch_by_index = {item.index: item for item in batch_items}
        classified_map: dict[int, Dish] = {}
        uncertain_indices: List[int] = []
        uncertain_inputs: List[Dish] = []

        for idx, dish in enumerate(all_dishes):
            item = batch_by_index.get(idx)
            if item is None:
                uncertain_indices.append(idx)
                uncertain_inputs.append(dish)
                continue

            updated = dish.model_copy(
                update={
                    "is_vegetarian": item.is_vegetarian,
                    "confidence": item.confidence,
                    "classification_method": "llm",
                    "reasoning": item.reasoning,
                    "signals": normalize_signals(item.signals),
                }
            )
            classified_map[idx] = updated
            if item.confidence < settings.confidence_threshold:
                uncertain_indices.append(idx)
                uncertain_inputs.append(dish)

        agent_map: dict[int, Dish] = {}
        if effective_rag and uncertain_inputs:
            agent = get_ai_agent()
            try:
                agent_state = await agent.run(uncertain_inputs, request_id=request_id, mode=normalized_mode)
            except OpenRouterError as exc:
                logger.error("OpenRouter request failed for %s: %s", request_id, exc)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"AI classification failed due to LLM provider error: {exc}",
                ) from exc
            except HTTPException:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("AI pipeline failed for %s", request_id)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="AI processing pipeline encountered an unexpected error.",
                ) from exc

            classified_subset = agent_state.get("classified_dishes", [])
            for offset, idx in enumerate(uncertain_indices):
                if offset >= len(classified_subset):
                    break
                dish_obj = classified_subset[offset]
                if not isinstance(dish_obj, Dish):
                    dish_obj = Dish.model_validate(dish_obj)
                agent_map[idx] = dish_obj

        for idx, dish in enumerate(all_dishes):
            final = agent_map.get(idx) or classified_map.get(idx) or dish
            final_dishes.append(final)

        vegetarian_dishes = [dish for dish in final_dishes if dish.is_vegetarian]
    else:
        if settings.keyword_mode_enabled and keyword_classifier:
            uncertain: List[str] = []
            for dish in all_dishes:
                result = keyword_classifier.classify_and_update(dish)
                if result.is_vegetarian:
                    vegetarian_dishes.append(dish)
                if result.is_uncertain:
                    uncertain.append(f"{dish.name} ({result.confidence:.2f})")

            if uncertain:
                logger.info(
                    "Keyword classifier marked %d dish(es) as uncertain: %s",
                    len(uncertain),
                    ", ".join(uncertain[:5]),
                )

        final_dishes = list(all_dishes)

    total_price, calculation_summary = await _calculate_totals(request_id, vegetarian_dishes)

    processing_time = (time.time() - start_time) * 1000

    logger.info(
        "Request %s completed in %.2fms using mode=%s (%d vegetarian dishes identified)",
        request_id,
        processing_time,
        normalized_mode,
        len(vegetarian_dishes),
    )

    return ProcessMenuResponse(
        request_id=request_id,
        total_images=len(files),
        ocr_results=ocr_results_list,
        dishes=final_dishes,
        parsed_menu=parsed_menu,
        vegetarian_dishes=vegetarian_dishes,
        total_price=total_price,
        processing_time_ms=processing_time,
        langsmith_trace_url=None,  # Will be added in Phase 7
        calculation_summary=calculation_summary,
        mode=normalized_mode,
    )


@router.get(
    "/mcp/tools",
    response_model=List[MCPToolInfo],
    summary="List available MCP tools exposed by the calculator service",
)
async def get_mcp_tools() -> List[MCPToolInfo]:
    """Return the set of MCP tools currently registered with the server."""
    if not mcp_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MCP client is not configured. Check MCP server settings.",
        )

    try:
        tools = await mcp_client.list_tools()
    except Exception as exc:  # pragma: no cover - network failures
        logger.error("Failed to list MCP tools: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to retrieve MCP tool metadata.",
        )

    return [MCPToolInfo(**tool) for tool in tools]
