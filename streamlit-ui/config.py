import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
API_EXTRACT_TEXT_ENDPOINT = f"{API_BASE_URL}/api/v1/extract-text"
API_PROCESS_MENU_ENDPOINT = f"{API_BASE_URL}/api/v1/process-menu"

# UI Configuration
MAX_IMAGES = 5
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp"]
