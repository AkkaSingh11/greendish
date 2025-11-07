from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent
ROOT_ENV = BASE_DIR.parent / ".env"
SERVICE_ENV = BASE_DIR / ".env"

# Load environment variables from repo root first, then override with service-specific values
if ROOT_ENV.exists():
    load_dotenv(ROOT_ENV, override=False)
if SERVICE_ENV.exists():
    load_dotenv(SERVICE_ENV, override=True)


class Settings(BaseSettings):
    """Application settings and configuration."""

    # API Settings
    app_name: str = "ConvergeFi Menu Analyzer"
    app_version: str = "0.1.0"
    debug: bool = True

    # Image Upload Settings
    max_images: int = 5
    allowed_extensions: list[str] = [".jpg", ".jpeg", ".png", ".webp"]
    max_file_size_mb: int = 10

    # OCR Settings
    tesseract_cmd: Optional[str] = None  # Auto-detect if None
    ocr_lang: str = "eng"

    # MCP Server Settings
    mcp_server_url: str = "http://localhost:8001"
    mcp_timeout: int = 30

    # OpenRouter Settings
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_primary_model: str = "deepseek/deepseek-chat-v3.1"
    openrouter_fallback_model: Optional[str] = None
    openrouter_request_timeout: int = 30
    openrouter_app_name: str = "ConvergeFi-Menu Analyzer"
    openrouter_referer: Optional[str] = None

    # Logging Settings
    logs_dir: str = str(BASE_DIR.parent / "logs")

    # Classification Settings
    confidence_threshold: float = 0.7
    use_keyword_fallback: bool = True
    keyword_mode_enabled: bool = True
    keyword_data_path: str = str(BASE_DIR / "data/vegetarian_keywords.json")
    keyword_fuzzy_threshold: float = 0.82
    keyword_max_hint_bonus: float = 0.15

    # RAG Settings
    rag_enabled: bool = True
    rag_db_path: str = str(BASE_DIR / "rag_db")
    rag_collection_name: str = "vegetarian_dishes"
    rag_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_seed_path: str = str(BASE_DIR / "data/vegetarian_db.json")
    rag_top_k: int = 3

    # LangSmith Settings
    langchain_tracing_v2: bool = False
    langchain_api_key: Optional[str] = None
    langchain_project: str = "convergefi-menu-analyzer"

    model_config = SettingsConfigDict(
        env_file=str(SERVICE_ENV) if SERVICE_ENV.exists() else None,
        case_sensitive=False,
        extra="ignore",
    )


# Global settings instance
settings = Settings()
