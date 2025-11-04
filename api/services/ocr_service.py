import pytesseract
from PIL import Image
import time
import logging
from typing import Optional
from pathlib import Path
import io

from config import settings

logger = logging.getLogger(__name__)


class OCRService:
    """Service for extracting text from menu images using Tesseract OCR."""

    def __init__(self):
        """Initialize OCR service with Tesseract configuration."""
        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

        # Verify Tesseract installation
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract OCR initialized (version: {version})")
        except Exception as e:
            logger.error(f"Tesseract not found or not properly installed: {e}")
            raise RuntimeError(
                "Tesseract OCR is not installed or not in PATH. "
                "Please install it: brew install tesseract (macOS) or "
                "apt-get install tesseract-ocr (Linux)"
            )

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR accuracy.

        Args:
            image: PIL Image object

        Returns:
            Preprocessed PIL Image
        """
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize if image is too large (maintain aspect ratio)
        max_dimension = 3000
        if max(image.size) > max_dimension:
            ratio = max_dimension / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Image resized to {new_size}")

        return image

    def extract_text(
        self, image_data: bytes, image_name: str = "image"
    ) -> tuple[str, float, float]:
        """
        Extract text from image using Tesseract OCR.

        Args:
            image_data: Image file bytes
            image_name: Name of the image file

        Returns:
            Tuple of (extracted_text, confidence_score, processing_time_ms)
        """
        start_time = time.time()

        try:
            # Open image from bytes
            image = Image.open(io.BytesIO(image_data))
            logger.info(f"Processing image: {image_name} (size: {image.size}, mode: {image.mode})")

            # Preprocess image
            image = self.preprocess_image(image)

            # Extract text with detailed data (includes confidence)
            ocr_data = pytesseract.image_to_data(
                image,
                lang=settings.ocr_lang,
                output_type=pytesseract.Output.DICT,
            )

            # Extract text using image_to_string for cleaner output
            text = pytesseract.image_to_string(image, lang=settings.ocr_lang)

            # Calculate average confidence from OCR data
            confidences = [
                int(conf) for conf in ocr_data["conf"] if int(conf) > 0
            ]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            processing_time = (time.time() - start_time) * 1000  # Convert to ms

            logger.info(
                f"OCR completed for {image_name}: "
                f"{len(text)} chars, {avg_confidence:.1f}% confidence, "
                f"{processing_time:.2f}ms"
            )

            return text.strip(), avg_confidence, processing_time

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"OCR failed for {image_name}: {str(e)}")
            raise RuntimeError(f"Failed to extract text from {image_name}: {str(e)}")

    def validate_image(self, image_data: bytes, filename: str) -> tuple[bool, Optional[str]]:
        """
        Validate image file.

        Args:
            image_data: Image file bytes
            filename: Name of the file

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in settings.allowed_extensions:
            return False, f"Invalid file type. Allowed: {settings.allowed_extensions}"

        # Check file size
        size_mb = len(image_data) / (1024 * 1024)
        if size_mb > settings.max_file_size_mb:
            return False, f"File too large. Max size: {settings.max_file_size_mb}MB"

        # Try to open as image
        try:
            image = Image.open(io.BytesIO(image_data))
            image.verify()  # Verify it's a valid image
            return True, None
        except Exception as e:
            return False, f"Invalid or corrupted image file: {str(e)}"
