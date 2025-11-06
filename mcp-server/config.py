from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for the MCP calculator server."""

    app_name: str = "ConvergeFi MCP Calculator"
    host: str = "0.0.0.0"
    port: int = 8001
    stateless_http: bool = True
    confidence_threshold: float = 0.7

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
