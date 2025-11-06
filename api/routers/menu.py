from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
import logging
import uuid
from datetime import datetime

from models import (
    OCRResult,
    ProcessMenuResponse,
    ErrorResponse,
    ParsedMenu,
    ParsedDish,
)
from services import OCRService
from services.parser_service import ParserService
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["menu"])

# Initialize services
ocr_service = OCRService()
parser_service = ParserService()


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
    summary="Process menu images and parse dishes (Phase 2: OCR + Parsing)",
)
async def process_menu(
    files: List[UploadFile] = File(..., description="Menu images (1-5 files)")
):
    """
    Process menu images to extract and parse dishes (Phase 2 implementation).

    - Phase 1: OCR text extraction ✅
    - Phase 2: Parse dishes with names and prices ✅
    - Phase 3+: Vegetarian classification (upcoming)
    - Phase 4+: Price calculation via MCP server (upcoming)

    Returns OCR results and parsed dishes.
    """
    import time

    start_time = time.time()
    request_id = str(uuid.uuid4())

    logger.info(f"Processing menu request {request_id} with {len(files)} images")

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

    processing_time = (time.time() - start_time) * 1000

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

    logger.info(
        f"Request {request_id} completed: {parsed_menu.total_dishes} dishes parsed "
        f"in {processing_time:.2f}ms"
    )

    # Phase 2: Return OCR results and parsed dishes
    # Future phases will add classification and calculation
    return ProcessMenuResponse(
        request_id=request_id,
        total_images=len(files),
        ocr_results=ocr_results_list,
        dishes=all_dishes,
        parsed_menu=parsed_menu,
        vegetarian_dishes=[],  # Will be populated in Phase 3+
        total_price=0.0,  # Will be calculated in Phase 4+
        processing_time_ms=processing_time,
        langsmith_trace_url=None,  # Will be added in Phase 7
    )
