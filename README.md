# ConvergeFi - Restaurant Menu Vegetarian Dish Analyzer

A microservices-based system that processes restaurant menu photos to identify and sum up prices of vegetarian dishes using OCR, LLM classification, and MCP (Model Context Protocol) server architecture.

## Overview

This system combines:
- **OCR (Tesseract)** for text extraction from menu images
- **LLM Classification** (GPT-4o-mini/Claude) for intelligent vegetarian dish detection
- **RAG (Retrieval-Augmented Generation)** with ChromaDB for confidence scoring
- **MCP Server** for calculation logic
- **LangSmith** for observability and tracing
- **Streamlit UI** for testing and demonstration

## System Architecture

```
┌─────────────────┐
│   Streamlit UI  │ (Testing Interface)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│      REST API Service       │
│  - Image Upload             │
│  - OCR (Tesseract)          │
│  - Text Parsing             │
│  - LLM Classification       │
└─────────┬───────────────────┘
          │ HTTP
          ▼
┌─────────────────────────────┐
│     MCP Server Service      │
│  - RAG Vector Search        │
│  - Confidence Scoring       │
│  - Price Calculation        │
└─────────────────────────────┘
          │
          ▼
┌─────────────────────────────┐
│  LangSmith Observability    │
└─────────────────────────────┘
```

## Project Structure

```
ConvergeFi/
├── api/                      # REST API service
│   ├── routers/             # API endpoints
│   ├── services/            # Business logic
│   ├── models/              # Pydantic models
│   ├── main.py              # FastAPI app
│   └── config.py            # Configuration
├── mcp-server/              # MCP calculation service
│   ├── tools/               # MCP tools
│   ├── rag/                 # RAG implementation
│   ├── data/                # Seed data
│   └── server.py            # MCP server
├── streamlit-ui/            # Testing UI
│   ├── pages/               # Multi-page app
│   └── app.py               # Main app
├── tests/                   # Test suite
├── menu*.{jpeg,png,webp}    # Sample menu images
└── docker-compose.yml       # Docker orchestration
```

## Technology Stack

### Core
- **Python 3.11+**
- **FastAPI** - High-performance REST API
- **MCP SDK** - Model Context Protocol server
- **Streamlit** - Interactive testing UI

### OCR & NLP
- **Tesseract OCR** - Text extraction
- **pytesseract** - Python wrapper
- **Pillow (PIL)** - Image processing

### AI/ML
- **OpenAI API** - GPT-4o-mini for classification
- **ChromaDB** - Vector database
- **sentence-transformers** - Embeddings (all-MiniLM-L6-v2)

### Observability
- **LangSmith** - Tracing and monitoring

### Infrastructure
- **Docker & Docker Compose** - Containerization
- **httpx** - Async HTTP client

## Getting Started

### Prerequisites

1. **Python 3.11+**
2. **uv** - Fast Python package installer
   - Install: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Or via pip: `pip install uv`
3. **Tesseract OCR** - Install via:
   - macOS: `brew install tesseract`
   - Ubuntu: `sudo apt-get install tesseract-ocr`
   - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
4. **Docker & Docker Compose** (optional, for containerized deployment)
5. **API Keys** (optional but recommended):
   - OpenAI API key (for LLM classification - Phase 5+)
   - LangSmith API key (for observability - Phase 7+)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd ConvergeFi
```

2. **Create environment file**
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. **Install dependencies using uv**
```bash
# For API service
cd api
uv pip install -r pyproject.toml

# For MCP server (Phase 4+)
cd ../mcp-server
uv pip install -r pyproject.toml

# For Streamlit UI
cd ../streamlit-ui
uv pip install -r pyproject.toml
```

**Alternative: Use uv sync (recommended)**
```bash
# Each service directory
cd api  # or streamlit-ui or mcp-server
uv sync
```

### Running the Application

#### Option 1: Docker Compose (Recommended)
```bash
docker-compose up --build
```

#### Option 2: Manual (Development)
```bash
# Terminal 1: Start MCP Server
cd mcp-server
python server.py

# Terminal 2: Start API Service
cd api
uvicorn main:app --reload --port 8000

# Terminal 3: Start Streamlit UI
cd streamlit-ui
streamlit run app.py
```

### Access Points

- **API Documentation**: http://localhost:8000/docs
- **Streamlit UI**: http://localhost:8501
- **MCP Server**: http://localhost:8001
- **Health Check**: http://localhost:8000/health

## Usage

### API Endpoint

**POST /process-menu**
- Upload 1-5 menu images
- Returns vegetarian dishes with total price

```bash
curl -X POST "http://localhost:8000/process-menu" \
  -F "files=@menu1.jpeg" \
  -F "files=@menu2.png"
```

### Streamlit UI

1. Navigate to http://localhost:8501
2. Upload menu images
3. View OCR extraction results
4. See classified vegetarian dishes
5. Review confidence scores and reasoning
6. Get total price calculation

## Features

### Phase 1: OCR Extraction ✓
- Upload and validate menu images
- Extract text using Tesseract OCR
- Display raw OCR output

### Phase 2: Text Parsing (In Progress)
- Parse OCR text into structured dishes
- Extract dish names and prices
- Handle various menu formats

### Phase 3: Keyword Classification (Planned)
- Basic vegetarian detection
- Keyword matching (vegetarian, veggie, vegan, etc.)
- Fallback mechanism

### Phase 4: MCP Integration (Planned)
- Separate calculation service
- MCP tool: `calculate_vegetarian_total`
- Service-to-service communication

### Phase 5: LLM Classification (Planned)
- Intelligent classification with GPT-4o-mini
- Confidence scoring
- Cost tracking

### Phase 6: RAG Implementation (Planned)
- ChromaDB vector store
- Semantic similarity search
- Enhanced confidence scoring

### Phase 7: Observability (Planned)
- LangSmith tracing integration
- Performance metrics
- Token usage tracking

### Phase 8: HITL Review (Planned)
- Human-in-the-loop for uncertain items
- Review and correction interface
- Model improvement feedback

## Configuration

### Environment Variables

```bash
# API Service
MCP_SERVER_URL=http://mcp-server:8001
OPENAI_API_KEY=your_openai_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
MAX_IMAGES=5
CONFIDENCE_THRESHOLD=0.7

# MCP Server
CHROMA_PERSIST_DIR=/app/chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K_RETRIEVAL=3
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_ocr.py
pytest tests/test_parser.py
pytest tests/test_classification.py

# Run with coverage
pytest --cov=api --cov=mcp-server tests/
```

## Development

### Current Phase: Phase 1 - OCR Implementation

**Completed:**
- ✅ Git repository initialization
- ✅ Project structure setup
- ✅ .gitignore configuration

**In Progress:**
- 🔄 FastAPI basic structure
- 🔄 OCR service implementation
- 🔄 Streamlit UI for testing

**Next Steps:**
- Text parsing and structuring
- Keyword-based classification
- MCP server setup

## Design Decisions

### Why FastAPI?
- Native async support
- Automatic OpenAPI documentation
- Type safety with Pydantic
- High performance (ASGI)

### Why ChromaDB?
- Easy local setup
- Persistent storage
- Metadata filtering
- Better developer experience than FAISS

### Why MCP Protocol?
- Modern standard for LLM tools
- Clear separation of concerns
- Network-based communication
- Future-proof architecture

### Why Hybrid LLM + Keywords?
- LLM provides intelligence
- Keywords ensure baseline accuracy
- Fallback for API failures
- Cost optimization

## Cost Estimates

All components use free tiers:
- **Tesseract**: Free, open-source
- **ChromaDB**: Free, local
- **Sentence Transformers**: Free, local
- **LangSmith**: Free tier (5,000 traces/month)
- **LLM API**: ~$0.01 per menu (GPT-4o-mini)

**Estimated Monthly Cost (Dev/Testing)**: $0-5

## Troubleshooting

### Tesseract Not Found
- Ensure Tesseract is installed and in PATH
- Check with: `tesseract --version`

### Docker Issues
- Ensure Docker daemon is running
- Try: `docker-compose down && docker-compose up --build`

### Import Errors
- Verify virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

## Contributing

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and test thoroughly
3. Commit with clear messages
4. Push and create pull request

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please create an issue in the GitHub repository.

---

**Status**: 🚧 In Active Development - Phase 1
**Last Updated**: 2025-11-04
