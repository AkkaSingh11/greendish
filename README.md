# GreenDish - AI-Powered Vegetarian Menu Analyzer

An intelligent microservices-based system that processes restaurant menu photos to identify and calculate total prices of vegetarian dishes using OCR, LangGraph AI agents, LLM classification, RAG, and MCP architecture.

![Project Status](https://img.shields.io/badge/status-Phase%209%20Complete-success)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

GreenDish combines cutting-edge AI technologies to help users identify vegetarian options from menu photos:

- **OCR (Tesseract)** - Extracts text from menu images with preprocessing
- **Structured Parsing** - Converts OCR output to canonical `{name, price, raw_text}` JSON
- **LangGraph Agent** - Orchestrates dish classification, RAG fallback, and MCP tool usage
- **LLM Classification** - Uses Groq (`openai/gpt-oss-20b`) with OpenRouter fallback
- **RAG (Retrieval-Augmented Generation)** - ChromaDB + sentence-transformers for confidence bolstering
- **MCP Server** - Deterministic price calculation via Model Context Protocol
- **LangSmith** - Complete observability and tracing
- **Interactive UI** - Streamlit testing interface with LLM chat playground

## System Architecture

```
┌─────────────────────────┐
│    Streamlit UI         │  (Testing + LLM Playground)
│  - Image Upload         │
│  - Results Viewer       │
│  - Chat Interface       │
└───────────┬─────────────┘
            │ HTTP
            ▼
┌───────────────────────────────────────────────┐
│           REST API Service                    │
│                                               │
│  ┌─────────────────────────────────────────┐ │
│  │      LangGraph Agent Workflow           │ │
│  │  ┌──────────┐  ┌──────────┐  ┌────────┐│ │
│  │  │Classifier│→ │RAG Check │→ │MCP Tool││ │
│  │  │  Node    │  │  Node    │  │  Node  ││ │
│  │  └──────────┘  └──────────┘  └────────┘│ │
│  └─────────────────────────────────────────┘ │
│                                               │
│  - OCR Service (Tesseract)                   │
│  - Parser Service (Structured JSON)          │
│  - LLM Router (Groq → OpenRouter)            │
└───────────┬───────────────────────────────────┘
            │ HTTP (MCP Protocol)
            ▼
┌───────────────────────────┐
│    MCP Server Service     │
│  - calculate_vegetarian   │
│    _total tool            │
│  - Deterministic logic    │
└───────────────────────────┘
            │
            ▼
┌───────────────────────────┐
│  LangSmith Observability  │
│  - Trace requests         │
│  - Token usage            │
│  - Performance metrics    │
└───────────────────────────┘
```

## Project Structure

```
GreenDish/
├── api/                          # REST API service (FastAPI)
│   ├── agents/                   # LangGraph agent workflows
│   │   ├── menu_processor.py    # Main agent state machine
│   │   └── nodes/                # Agent node implementations
│   │       ├── classifier_node.py
│   │       ├── rag_node.py
│   │       └── calculator_node.py
│   ├── llm/                      # LLM client utilities
│   │   ├── groq_client.py        # Groq integration (primary)
│   │   ├── openrouter_client.py  # OpenRouter fallback
│   │   └── router_client.py      # LLM router with auto-fallback
│   ├── routers/                  # API endpoints
│   ├── services/                 # Business logic
│   │   ├── ocr_service.py
│   │   ├── parser_service.py
│   │   └── classifier_service.py
│   ├── models/                   # Pydantic schemas
│   ├── data/                     # Seed data (vegetarian_db.json)
│   ├── rag_db/                   # ChromaDB persistent storage
│   ├── config.py                 # Centralized configuration
│   └── main.py                   # FastAPI app entry point
├── mcp-server/                   # MCP calculation service
│   ├── tools/                    # MCP tool implementations
│   └── server.py                 # MCP server entry point
├── streamlit-ui/                 # Interactive testing UI
│   ├── pages/                    # Multi-page app
│   │   ├── 1_OCR_Test.py
│   │   ├── 2_Parser_Test.py
│   │   └── ...
│   └── app.py                    # Main dashboard
├── tests/                        # Pytest test suite
│   ├── fixtures/images/          # Test menu images
│   ├── api/                      # API tests
│   ├── mcp/                      # MCP tests
│   └── services/                 # Service layer tests
├── scripts/                      # Utility scripts
│   └── test_ocr.py               # Quick OCR validation
├── docs/                         # Documentation
│   ├── phases/                   # Phase-wise planning docs
│   ├── architecture.md           # Detailed architecture
│   └── requirements.md           # Original requirements
├── docker-compose.yml            # Docker orchestration
├── .env.example                  # Environment template
└── CLAUDE.md                     # AI assistant guidelines
```

## Technology Stack

### Core Framework
- **Python 3.11+** - Modern Python with type hints
- **FastAPI** - High-performance async REST API
- **Streamlit** - Interactive web UI for testing
- **MCP SDK** - Model Context Protocol server
- **uv** - Fast, reliable Python package manager

### AI/ML Stack
- **LangGraph** - Agent workflow orchestration
- **Groq API** - Primary LLM provider (`openai/gpt-oss-20b`)
- **OpenRouter** - Fallback LLM provider (`deepseek/deepseek-chat-v3.1`)
- **ChromaDB** - Vector database for RAG
- **sentence-transformers** - Embeddings (`all-MiniLM-L6-v2`)
- **LangSmith** - Tracing and observability

### OCR & Processing
- **Tesseract OCR 5.x** - Text extraction
- **pytesseract** - Python wrapper
- **Pillow (PIL)** - Image preprocessing

### Infrastructure
- **Docker & Docker Compose** - Containerization
- **httpx** - Async HTTP client
- **pydantic** - Data validation
- **pydantic-settings** - Configuration management

## Quick Start

### Prerequisites

1. **Python 3.11+** - [Download](https://www.python.org/downloads/)
2. **uv** - Fast Python package installer
   ```bash
   # Via curl (recommended)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Via pip
   pip install uv

   # Via Homebrew (macOS)
   brew install uv
   ```
3. **Tesseract OCR**
   ```bash
   # macOS
   brew install tesseract

   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr

   # Windows
   # Download from: https://github.com/UB-Mannheim/tesseract/wiki
   ```
4. **Docker & Docker Compose** (optional) - [Install Docker](https://docs.docker.com/get-docker/)
5. **API Keys** (recommended for full functionality)
   - Groq API key - [Get key](https://console.groq.com/)
   - OpenRouter API key (optional fallback) - [Get key](https://openrouter.ai/)
   - LangSmith API key (optional, for tracing) - [Get key](https://smith.langchain.com/)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AkkaSingh11/greendish.git
   cd greendish
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Install dependencies**

   **Option A: Using uv sync (recommended)**
   ```bash
   # API service
   cd api
   uv sync

   # MCP server
   cd ../mcp-server
   uv sync

   # Streamlit UI
   cd ../streamlit-ui
   uv sync
   ```

   **Option B: Using uv pip**
   ```bash
   cd api
   uv pip install -r pyproject.toml
   ```

### Configuration

Edit `.env` with your settings:

```env
# Groq (Primary LLM Provider)
GROQ_API_KEY=your_groq_api_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_PRIMARY_MODEL=openai/gpt-oss-20b
GROQ_REQUEST_TIMEOUT=30

# OpenRouter (Fallback LLM Provider)
OPENROUTER_API_KEY=your_openrouter_key  # Optional
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_PRIMARY_MODEL=deepseek/deepseek-chat-v3.1
OPENROUTER_FALLBACK_MODEL=
OPENROUTER_REQUEST_TIMEOUT=30
OPENROUTER_APP_NAME=GreenDish-MenuAnalyzer

# API Configuration
MCP_SERVER_URL=http://localhost:8001
MAX_IMAGES=5
CONFIDENCE_THRESHOLD=0.4
DEBUG=false

# LangSmith (Optional - for observability)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=greendish
```

### Running the Application

#### Option 1: Docker Compose (Recommended for Production)

```bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f api
docker-compose logs -f streamlit

# Stop services
docker-compose down
```

#### Option 2: Manual Launch (Recommended for Development)

```bash
# Terminal 1: Start MCP Server
cd mcp-server
uv run python server.py
# Runs on http://localhost:8001

# Terminal 2: Start API Service
cd api
uv run uvicorn main:app --reload --port 8005
# Runs on http://localhost:8005

# Terminal 3: Start Streamlit UI
cd streamlit-ui
uv run streamlit run app.py
# Runs on http://localhost:8501
```

### Access Points

- **Streamlit UI**: http://localhost:8501 - Main testing interface
- **LLM Chat Playground**: http://localhost:8501 (sidebar: "💬 LLM Chat Playground")
- **API Documentation**: http://localhost:8005/docs - Interactive OpenAPI docs
- **API Health Check**: http://localhost:8005/health
- **MCP Server**: http://localhost:8001 - Calculation service

## Usage

### Via Streamlit UI (Recommended)

1. Open http://localhost:8501
2. Navigate to the desired phase test page (sidebar)
3. Upload 1-5 menu images (JPEG, PNG, WEBP)
4. Toggle AI mode on/off to compare approaches
5. View:
   - OCR extraction results
   - Parsed menu structure (`parsed_menu` JSON)
   - Vegetarian dish classifications with confidence scores
   - Total price calculation
   - LLM reasoning (when AI mode enabled)
6. Use the **LLM Chat Playground** to test Groq/OpenRouter prompts directly

### Via API

**Process Menu (Full Pipeline)**
```bash
curl -X POST "http://localhost:8005/api/v1/process-menu" \
  -F "files=@tests/fixtures/images/menu1.jpeg" \
  -F "files=@tests/fixtures/images/menu2.png" \
  -F "use_ai=true"
```

**Extract Text Only (OCR)**
```bash
curl -X POST "http://localhost:8005/api/v1/extract-text" \
  -F "files=@tests/fixtures/images/menu1.jpeg"
```

**Response Format**
```json
{
  "vegetarian_dishes": [
    {
      "name": "Veggie Burger",
      "price": 12.99,
      "is_vegetarian": true,
      "confidence": 0.95,
      "reasoning": "Contains vegetables and no meat products",
      "classification_method": "llm"
    }
  ],
  "total_vegetarian_price": 12.99,
  "total_dishes": 15,
  "vegetarian_count": 1,
  "processing_time": 2.34,
  "metadata": {
    "ai_mode": true,
    "ocr_time": 1.2,
    "classification_time": 0.8,
    "llm_provider": "groq"
  }
}
```

## Features by Phase

### ✅ Phase 1-2: OCR & Parsing (Complete)
- Multi-image upload (1-5 images)
- Tesseract OCR with image preprocessing
- Structured parsing to canonical `{name, price, raw_text}` JSON
- Confidence scoring for OCR results

### ✅ Phase 3: Keyword Classification (Complete)
- Keyword-based vegetarian detection
- Fallback mechanism for non-AI mode
- Configurable keyword lists

### ✅ Phase 4: MCP Integration (Complete)
- Separate calculation microservice
- MCP tool: `calculate_vegetarian_total`
- HTTP-based service communication

### ✅ Phase 5: LLM Classification (Complete)
- Groq API integration with `openai/gpt-oss-20b`
- OpenRouter fallback with `deepseek/deepseek-chat-v3.1`
- Structured JSON responses with confidence + reasoning
- Automatic provider failover
- Token usage tracking

### ✅ Phase 6: RAG Implementation (Complete)
- ChromaDB vector store with persistent storage
- Sentence-transformers embeddings (`all-MiniLM-L6-v2`)
- Top-3 semantic similarity retrieval
- Confidence score merging (LLM + RAG similarity)
- Seeded with vegetarian dish database

### ✅ Phase 7-8: LangGraph Agent & Observability (Complete)
- LangGraph state machine orchestration
- Multi-node workflow (classifier → RAG → calculator)
- LangSmith tracing for all operations
- Request ID tracking across services
- Performance metrics (OCR, parsing, classification, RAG)
- AI/non-AI mode toggle

### ✅ Phase 9: Testing & Validation (Complete)
- Comprehensive pytest suite
- API endpoint tests
- Service layer tests
- OCR regression tests
- Integration tests with multiple sample menus

## Development

### Current Status

**Phase 9+: Complete** - All core features implemented and tested

### Running Tests

```bash
# Run full test suite
pytest tests/

# Run with coverage
pytest --cov=api --cov=mcp-server tests/

# Run specific test file
pytest tests/test_ocr.py
pytest tests/test_classification.py

# Run with verbose output
pytest -v tests/
```

### Adding Dependencies

```bash
# Navigate to service directory
cd api  # or mcp-server or streamlit-ui

# Add a package
uv add package-name

# Or edit pyproject.toml and run
uv sync
```

**IMPORTANT**: Do NOT create `requirements.txt`. All dependencies are managed via `pyproject.toml`.

### Code Structure Guidelines

- **Relative imports** within each service (no `api.module` imports)
- **Service layer pattern** for business logic
- **Pydantic models** for all data validation
- **Type hints** throughout the codebase
- **Async/await** for I/O operations

## Configuration Reference

### Key Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key (primary LLM) | Required for AI mode |
| `GROQ_PRIMARY_MODEL` | Groq model identifier | `openai/gpt-oss-20b` |
| `OPENROUTER_API_KEY` | OpenRouter key (fallback) | Optional |
| `CONFIDENCE_THRESHOLD` | Min confidence for classification | `0.4` |
| `MAX_IMAGES` | Max images per request | `5` |
| `MCP_SERVER_URL` | MCP service endpoint | `http://localhost:8001` |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing | `false` |
| `DEBUG` | Debug mode with auto-reload | `false` |

See `.env.example` for complete list.

## Troubleshooting

### Tesseract Not Found
```bash
# Verify installation
tesseract --version

# macOS: Ensure in PATH
brew info tesseract

# Set TESSERACT_CMD in .env if needed
TESSERACT_CMD=/usr/local/bin/tesseract
```

### Import Errors in API Service
- Use **relative imports** (`from services import OCRService`)
- NOT absolute imports (`from api.services import OCRService`)
- See [CLAUDE.md](CLAUDE.md) for details

### Docker Port Conflicts
- API: 8005
- MCP Server: 8001
- Streamlit: 8501

```bash
# Check ports in use
lsof -i :8005
lsof -i :8001
lsof -i :8501
```

### LLM API Errors
- Verify API keys in `.env`
- Check Groq/OpenRouter account credits
- Review logs: `docker-compose logs api`
- Fallback to non-AI mode (keyword-based)

### RAG/ChromaDB Issues
```bash
# Delete and rebuild vector store
rm -rf api/rag_db/
# Restart API service to reinitialize
```

## Architecture Decisions

### Why LangGraph?
- Visual workflow debugging
- Easy state management
- Conditional routing (AI vs. non-AI mode)
- Built-in retry and error handling

### Why Groq + OpenRouter?
- Groq: Ultra-fast inference (sub-second)
- OpenRouter: Broad model selection for fallback
- Cost optimization with automatic failover
- Unified interface via LLM router

### Why ChromaDB?
- Simple local setup (no external services)
- Persistent storage with low overhead
- Metadata filtering support
- Better DX than FAISS

### Why MCP Protocol?
- Modern standard for LLM tool integration
- Clear service boundaries
- Network-based for multi-language support
- Future-proof for Claude/GPT tool use

### Why Hybrid (AI + Keywords)?
- Keywords: Fast, deterministic fallback
- LLM: Handles edge cases ("plant-based", "meatless")
- RAG: Boosts confidence on ambiguous dishes
- Cost control with configurable AI toggle

## Cost Estimates

### Development/Testing (Monthly)
- **Tesseract OCR**: Free (open-source)
- **ChromaDB**: Free (local storage)
- **Sentence Transformers**: Free (local inference)
- **LangSmith**: Free tier (5,000 traces/month)
- **Groq API**: Free tier or ~$0.005 per menu
- **OpenRouter**: ~$0.01 per menu (fallback only)

**Estimated**: $0-10/month for development

### Production (1,000 menus/month)
- **Groq API**: ~$5
- **OpenRouter** (fallback, 10% usage): ~$1
- **LangSmith**: Free or ~$5 (beyond free tier)
- **Hosting**: Variable (Docker-based, any cloud)

**Estimated**: $10-20/month

## Performance

Typical processing time for a single menu image:
- **OCR**: 0.5-1.5s (depends on image quality)
- **Parsing**: <0.1s
- **LLM Classification**: 0.5-2s (Groq) or 2-5s (OpenRouter)
- **RAG Retrieval**: 0.1-0.3s
- **MCP Calculation**: <0.05s

**Total end-to-end**: 1-4 seconds per menu

## Roadmap

### Completed
- ✅ Multi-service architecture with Docker
- ✅ LangGraph agent orchestration
- ✅ Groq + OpenRouter LLM routing
- ✅ RAG-enhanced classification
- ✅ LangSmith observability
- ✅ Comprehensive test suite
- ✅ Interactive Streamlit UI with chat playground

### Future Enhancements
- 🔲 Human-in-the-loop review interface (Phase 10)
- 🔲 Multi-language menu support
- 🔲 Mobile app integration
- 🔲 Dietary restriction customization (vegan, gluten-free, etc.)
- 🔲 Restaurant API integration
- 🔲 Batch processing for multiple restaurants
- 🔲 Analytics dashboard

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature`
3. **Follow** code style (use relative imports, type hints)
4. **Test** thoroughly: `pytest tests/`
5. **Commit** with clear messages (no AI badging per project rules)
6. **Push** and create a pull request

See [docs/phases/](docs/phases/) for development planning docs.

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Tesseract OCR** - Google's open-source OCR engine
- **LangGraph** - Agent workflow framework by LangChain
- **Groq** - Ultra-fast LLM inference
- **ChromaDB** - Open-source vector database
- **Sentence Transformers** - State-of-the-art embeddings

## Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/AkkaSingh11/greendish/issues)
- **Discussions**: [Ask questions or share ideas](https://github.com/AkkaSingh11/greendish/discussions)

---

**Status**: ✅ Phase 9+ Complete - Production Ready
**Last Updated**: 2025-11-07
**Maintainer**: [@AkkaSingh11](https://github.com/AkkaSingh11)
