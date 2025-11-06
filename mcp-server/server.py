import contextlib
import logging

import uvicorn
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from config import settings
from mcp.server.fastmcp import FastMCP
from tools.calculator import register_calculator

logger = logging.getLogger(__name__)


def create_app() -> Starlette:
    """Create the ASGI application hosting the MCP server."""
    mcp = FastMCP(settings.app_name, stateless_http=settings.stateless_http)
    register_calculator(mcp)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        async with mcp.session_manager.run():
            logger.info("%s session manager started", settings.app_name)
            yield
            logger.info("%s session manager stopped", settings.app_name)

    async def health_check(request):
        return JSONResponse({"status": "healthy", "service": settings.app_name})

    starlette_app = Starlette(
        routes=[
            Route("/health", health_check),
            Mount("/", app=mcp.streamable_http_app()),
        ],
        lifespan=lifespan,
    )

    starlette_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
        expose_headers=["Mcp-Session-Id"],
    )

    return starlette_app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    app = create_app()
    logger.info("Starting %s on %s:%s", settings.app_name, settings.host, settings.port)
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
