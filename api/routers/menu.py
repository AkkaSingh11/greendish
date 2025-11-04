from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
import logging
import uuid
from datetime import datetime

from models import OCRResult, ProcessMenuResponse, ErrorResponse
from services import OCRService
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["menu"])

# Initialize OCR service
ocr_service = OCRService()


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
    summary="Process menu images and identify vegetarian dishes (Phase 1: OCR Only)",
)
async def process_menu(
    files: List[UploadFile] = File(..., description="Menu images (1-5 files)")
):
    """
    Process menu images to extract text (Phase 1 implementation).

    Future phases will add:
    - Text parsing and dish extraction
    - Vegetarian classification
    - Price calculation via MCP server

    Currently returns OCR results only.
    """
    import time

    start_time = time.time()
    request_id = str(uuid.uuid4())

    logger.info(f"Processing menu request {request_id} with {len(files)} images")

    # Extract text from images
    ocr_results_list = await extract_text(files)

    processing_time = (time.time() - start_time) * 1000

    # Phase 1: Return OCR results only
    # Future phases will add parsing, classification, and calculation
    return ProcessMenuResponse(
        request_id=request_id,
        total_images=len(files),
        ocr_results=ocr_results_list,
        dishes=[],  # Will be populated in Phase 2
        vegetarian_dishes=[],  # Will be populated in Phase 3+
        total_price=0.0,  # Will be calculated in Phase 4+
        processing_time_ms=processing_time,
        langsmith_trace_url=None,  # Will be added in Phase 7
    )
