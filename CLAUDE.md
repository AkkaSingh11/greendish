# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ConvergeFi is a microservices-based Restaurant Menu Vegetarian Dish Analyzer that processes menu photos to identify and calculate total prices of vegetarian dishes. The system uses:
- **OCR** (Tesseract) for text extraction
- **Structured parsing** that emits canonical `{name, price, raw_text}` JSON for each dish
- **LLM classification** via Groq (`openai/gpt-oss-20b` default) with OpenRouter fallback, surfaced through the shared `api.llm` router utilities
- **LangGraph menu-processor agent** to orchestrate dish classification, RAG fallback, and MCP tool usage
- **RAG** (ChromaDB + sentence-transformers) for confidence bolstering when classification is uncertain
- **MCP (Model Context Protocol)** server for deterministic price calculation logic
- **LangSmith** for observability

## Architecture

The system is split into three microservices that communicate over HTTP:

1. **API Service** (`/api`) - FastAPI REST API
   - Handles image upload, OCR, text parsing, and emits structured parsed menu JSON (`parsed_menu`)
   - Orchestrates the LangGraph `menu-processor` agent for dish classification and RAG fallback
   - Communicates with the MCP server tool for price aggregation

2. **MCP Server** (`/mcp-server`) - Calculation service
   - Implements deterministic MCP tools for vegetarian dish calculations
   - Receives pre-classified vegetarian dishes and returns totals/confidence

3. **Streamlit UI** (`/streamlit-ui`) - Testing interface
   - Multi-page app for testing each phase
   - Includes an “💬 LLM Chat Playground” page to compare Groq (primary) vs. OpenRouter (fallback) prompts
   - Mirrors API outputs, including the structured `parsed_menu` JSON for verification

## Import Structure

**CRITICAL**: All imports within the `/api` directory use **relative imports**, NOT absolute imports with the `api.` prefix:

```python
# Correct
from config import settings
from models import OCRResult
from services import OCRService

# Wrong
from api.config import settings
from api.models import OCRResult
```

This is because the API runs from within the `/api` directory. The same pattern applies to `mcp-server` and `streamlit-ui`. Subpackages should use explicit relative imports (`from ..config import settings`) so they remain importable from outside the service (e.g., the Streamlit chat page).

## Dependency Management

This project uses **uv** for fast, reliable Python package management. All dependencies are defined in `pyproject.toml` files (one per service).

### Installing uv

```bash
# Via curl (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Via pip
pip install uv

# Via Homebrew (macOS)
brew install uv
```

### Installing Dependencies

```bash
# Method 1: Using uv pip install (for existing environments)
cd api  # or streamlit-ui or mcp-server
uv pip install -r pyproject.toml

# Method 2: Using uv sync (recommended, creates/updates .venv)
cd api  # or streamlit-ui or mcp-server
uv sync
```

### Adding New Dependencies

```bash
# Edit pyproject.toml and add to dependencies array
# Then run:
uv sync

# Or use uv to add directly:
uv add package-name
```

**IMPORTANT**: Do NOT create `requirements.txt` files. All dependencies are managed via `pyproject.toml`.

## LLM Client Module

- Shared LLM router components live in `api/llm/` (`groq_client.py`, `openrouter_client.py`, and `router_client.py`) and are re-exported via `api/llm/__init__.py`.
- Within the API service, import using relative paths (`from llm import GroqClient, LLMRouter`). From external services (Streamlit), use `from api.llm import GroqClient` (or `OpenRouterClient`) after ensuring the repo root is on `sys.path`.
- `complete_json(...)` enforces schema-validated responses for classification nodes; `chat(...)` streams plain text output for playgrounds.
- The client automatically falls back from the primary model to the configured secondary model and returns token/latency telemetry for observability.

## Development Commands

### Running Services Locally

```bash
# API Service (from /api directory)
cd api
uv sync  # Install/update dependencies first
python main.py
# Runs on http://localhost:8005

# Alternative: Using uvicorn directly with auto-reload
uv run uvicorn main:app --reload --port 8005

# Streamlit UI (from /streamlit-ui directory)
cd streamlit-ui
uv sync  # Install/update dependencies first
streamlit run app.py
# Runs on http://localhost:8501

# Or using uv run:
uv run streamlit run app.py

# MCP Server (from /mcp-server directory) - Phase 4+
cd mcp-server
uv sync  # Install/update dependencies first
python server.py
# Runs on http://localhost:8001
```

### Docker

```bash
# Build and run all services
docker-compose up --build

# Run in detached mode
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f api
docker-compose logs -f streamlit
```

### Testing

```bash
# Quick OCR test with sample menus
python test_ocr.py

# Run pytest suite (Phase 9+)
pytest tests/

# Run specific test file
pytest tests/test_ocr.py

# Run with coverage
pytest --cov=api --cov=mcp-server tests/
```

### Verify Tesseract Installation

```bash
tesseract --version
# Should show version 5.x.x
```

## Configuration

All configuration is centralized in `/api/config.py` using Pydantic Settings. Environment variables are loaded from `.env` (copy from `.env.example`):

```bash
# Create .env from example
cp .env.example .env
```

Key settings:
- `DEBUG`: Enable debug mode and auto-reload
- `MAX_IMAGES`: Maximum images per request (default: 5)
- `TESSERACT_CMD`: Path to Tesseract binary (auto-detect if None)
- `MCP_SERVER_URL`: MCP server endpoint (Phase 4+)
- `OPENROUTER_API_KEY`: For LLM classification (Phase 5+)
- `OPENROUTER_PRIMARY_MODEL`: Default `deepseek/deepseek-chat-v3.1`
- `OPENROUTER_FALLBACK_MODEL`: Secondary model identifier (optional; blank disables fallback)
- `LANGCHAIN_TRACING_V2`: Enable LangSmith tracing (Phase 7+)
- `CONFIDENCE_THRESHOLD`: Classification confidence threshold (default: 0.7)

## Phased Development

This project is being built in **10 phases**. See `PHASE_WISE_PLAN.MD` for the complete roadmap.

### Completed Phases
- ✅ **Phase 0**: Git initialization, project structure
- ✅ **Phase 1**: OCR implementation with FastAPI and Streamlit

### Current Phase
- 🔄 **Phase 2**: Text parsing and dish structuring

### Key Phase Dependencies
- **Phase 1-3**: API service only (no MCP server needed)
- **Phase 4+**: MCP server required
- **Phase 5+**: Groq API key preferred (falls back to OpenRouter API key if provided)
- **Phase 6+**: ChromaDB integration required
- **Phase 7+**: LangSmith API key required

## API Structure

### Endpoint Versioning
All API endpoints are prefixed with `/api/v1/` to allow future versioning.

### Key Endpoints

**GET /health**
- Health check for service status
- Returns version and service availability

**POST /api/v1/extract-text**
- Extracts text from 1-5 menu images using OCR
- Returns OCR results with confidence scores

**POST /api/v1/process-menu**
- Full menu processing pipeline
- Phase 1: Returns only OCR results
- Phase 2+: Returns structured `parsed_menu` JSON (dishes + stats)
- Phase 3+: Returns classified vegetarian dishes
- Phase 4+: Returns total price via MCP calculation tool

### Response Models

All API responses use Pydantic models defined in `/api/models/schemas.py`:
- `HealthResponse`
- `OCRResult` - Single image OCR result
- `Dish` - Structured dish with name, price, classification
- `ParsedDish` / `ParsedMenu` - Canonical parsed menu payload returned post-OCR
- `ProcessMenuResponse` - Complete menu processing result

## Service Layer Pattern

Business logic is separated into service classes in `/api/services/`:
- `OCRService` - Tesseract OCR integration, image preprocessing
- `ParserService` - Text parsing (Phase 2)
- `ClassifierService` - Keyword/LLM classification (Phases 3-5)
- `RAGService` - Vector search (Phase 6)

Each service is instantiated once at module level in the router and reused across requests.

Agent workflows are defined in `/api/agents/`:
- `menu_processor.py` builds the LangGraph state machine that coordinates parsing output, LLM classification, RAG lookups, and MCP tool calls.
- Node modules (`classifier_node.py`, `rag_node.py`, `calculator_node.py`) should remain small and composable.

## MCP Server Architecture (Phase 4+)

The MCP server implements tools following the Model Context Protocol standard:

**Tools:**
- `calculate_vegetarian_total` - Sums prices of vegetarian dishes (primary tool)
- Additional calculation utilities may be added in later phases, but RAG stays within the LangGraph agent.

Communication between API/LangGraph and MCP server uses HTTP transport with JSON-RPC style requests.

## Streamlit UI Structure

The Streamlit UI uses a multi-page app structure where each page corresponds to a development phase:

```
streamlit-ui/
├── app.py                    # Main dashboard/overview
└── pages/
    ├── 1_OCR_Test.py        # Phase 1 testing
    ├── 2_Parser_Test.py     # Phase 2 testing
    └── ... (more pages added per phase)
```

Each page should be self-contained and demonstrate the capabilities added in that phase.

## Testing Strategy

Sample menu images are provided in `ocr_test_images/`:
- `menu1.jpeg` - Applebee's menu (complex, multi-column)
- `menu2.png` - Simple menu with prices
- `menu3.webp` - Cafe menu with descriptions
- `image_4.webp`, `image_6.png` - Additional menus for parser regression tests

Use `test_ocr.py` for quick validation that OCR is working. Full pytest suite will be added in Phase 9.

## LLM Classification (Phase 5)

When implementing LLM classification:
1. Always implement keyword fallback first (Phase 3)
2. Route requests through Groq using `openai/gpt-oss-20b` as the default model, falling back to OpenRouter (`deepseek/deepseek-chat-v3.1`) when configured via env / Streamlit UI
3. LLM responses must be strict JSON: `{is_vegetarian: bool, confidence: float, reasoning: str}`
4. Use LangGraph state to decide when to trigger RAG lookups and re-classify
5. Track token usage and costs via LangSmith
6. Cache LLM responses to minimize API calls

## RAG Implementation (Phase 6)

ChromaDB setup (invoked from the LangGraph agent within the API service):
- Use `sentence-transformers/all-MiniLM-L6-v2` for embeddings (fast, good quality)
- Persistent storage in `/api/rag_db/`
- Seed data from `/api/data/vegetarian_db.json`
- Retrieve top-3 similar dishes per query
- Feed retrieved context back into the classification node and merge similarity score with LLM confidence

## Common Pitfalls

1. **Import errors in API service**: Use relative imports, not `from api.module`
2. **Tesseract not found**: Ensure Tesseract is installed and in PATH
3. **Port conflicts**: Default ports are 8005 (API), 8001 (MCP), 8501 (Streamlit)
4. **File upload size**: Default max is 10MB per image, configurable via `MAX_FILE_SIZE_MB`
5. **Working directory**: FastAPI app must run from `/api` directory

## Git Workflow

Each phase completion should result in a clear commit:
- Format: "Phase X: [Brief Description]"
- Include what was implemented and test results
- Update `PHASE_WISE_PLAN.MD` to mark phase as complete

## Observability (Phase 7)

When adding LangSmith tracing:
- Use `@traceable` decorator on key functions
- Track: OCR time, parsing time, LLM calls, RAG queries, MCP tool calls
- Link all operations in a request with a common `request_id`
- Display trace URLs in Streamlit UI for debugging
