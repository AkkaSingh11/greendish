import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import logging

from config import settings
from models import HealthResponse
from routers import menu_router, rag_router

log_dir = Path(settings.logs_dir)
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "api.log"

logging.basicConfig(
    level=logging.INFO if settings.debug else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode="a", encoding="utf-8"),
    ],
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Restaurant Menu Vegetarian Dish Analyzer API",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(menu_router)
app.include_router(rag_router)


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to docs."""
    return {
        "message": "ConvergeFi Menu Analyzer API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """
    Health check endpoint.

    Returns service status and version information.
    """
    services = {
        "api": "healthy",
        "ocr": "healthy",
    }

    # Check MCP server status (will be implemented in Phase 4)
    # try:
    #     async with httpx.AsyncClient() as client:
    #         response = await client.get(f"{settings.mcp_server_url}/health", timeout=5)
    #         services["mcp_server"] = "healthy" if response.status_code == 200 else "unhealthy"
    # except:
    #     services["mcp_server"] = "unavailable"

    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version=settings.app_version,
        services=services,
    )


def configure_langsmith() -> None:
    """Configure LangSmith tracing environment if enabled."""
    if not settings.langchain_tracing_v2:
        logger.info("LangSmith tracing disabled (LANGCHAIN_TRACING_V2=false).")
        return

    if not settings.langchain_api_key:
        logger.warning("LangSmith tracing requested but LANGCHAIN_API_KEY is not configured.")
        return

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.langchain_api_key)
    if settings.langchain_project:
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langchain_project)
        project_display = settings.langchain_project
    else:
        project_display = "(default)"
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    logger.info("LangSmith tracing enabled for project %s.", project_display)


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    configure_langsmith()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Max images per request: {settings.max_images}")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Shutting down application")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8005,
        reload=settings.debug,
        log_level="info" if settings.debug else "warning",
    )
