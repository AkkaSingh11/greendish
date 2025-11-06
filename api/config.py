from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent


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

    # LLM Settings
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    max_tokens: int = 500
    temperature: float = 0.3

    # Classification Settings
    confidence_threshold: float = 0.7
    use_keyword_fallback: bool = True
    keyword_mode_enabled: bool = True
    keyword_data_path: str = str(BASE_DIR / "data/vegetarian_keywords.json")
    keyword_fuzzy_threshold: float = 0.82
    keyword_max_hint_bonus: float = 0.15

    # LangSmith Settings
    langchain_tracing_v2: bool = False
    langchain_api_key: Optional[str] = None
    langchain_project: str = "convergefi-menu-analyzer"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


# Global settings instance
settings = Settings()
