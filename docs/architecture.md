# GreenDish Architecture Documentation

## System Overview

GreenDish is an AI-powered microservices application that analyzes restaurant menu images to identify vegetarian dishes and calculate their total prices. The system combines OCR, LLM classification, RAG, and MCP architecture to provide accurate results with configurable AI/non-AI modes.

## High-Level Architecture

```
┌─────────────────────────┐
│    Streamlit UI         │  (Port 8501)
│  - Image Upload         │
│  - Results Viewer       │
│  - LLM Chat Playground  │
└───────────┬─────────────┘
            │ HTTP
            ▼
┌───────────────────────────────────────────────┐
│           REST API Service (Port 8005)        │
│                                               │
│  ┌─────────────────────────────────────────┐ │
│  │   1. Image Upload & Validation          │ │
│  │   2. OCR Processing (Tesseract)         │ │
│  │   3. Text Parsing → Structured JSON     │ │
│  └──────────────┬──────────────────────────┘ │
│                 │                             │
│  ┌──────────────▼──────────────────────────┐ │
│  │      LangGraph Agent Workflow           │ │
│  │  ┌──────────┐  ┌──────────┐  ┌────────┐│ │
│  │  │Classifier│→ │RAG Check │→ │MCP Tool││ │
│  │  │  Node    │  │  Node    │  │  Node  ││ │
│  │  └──────────┘  └──────────┘  └────────┘│ │
│  └─────────────────────────────────────────┘ │
│                                               │
│  Components:                                  │
│  - OCR Service (Tesseract)                   │
│  - Parser Service (Structured JSON)          │
│  - LLM Router (Groq → OpenRouter)            │
│  - RAG Service (ChromaDB)                    │
└───────────┬───────────────────────────────────┘
            │ HTTP (MCP Protocol)
            ▼
┌───────────────────────────┐
│  MCP Server (Port 8001)   │
│  - calculate_vegetarian   │
│    _total tool            │
│  - Deterministic logic    │
└───────────────────────────┘
            │
            ▼
┌───────────────────────────┐
│  LangSmith Observability  │
│  - Request tracing        │
│  - Token usage metrics    │
│  - Performance analytics  │
└───────────────────────────┘
```

## Processing Modes

### AI Mode (Default)

Uses LangGraph agent with LLM classification, RAG fallback, and MCP calculation:

```
Image → OCR → Parser → [LangGraph Agent] → Results
                        ├─ LLM Classifier (Groq/OpenRouter)
                        ├─ RAG Lookup (if low confidence)
                        └─ MCP Calculator Tool
```

**Flow:**
1. **OCR**: Extract text from menu images
2. **Parser**: Convert to structured `{name, price, raw_text}` JSON
3. **Classifier Node**: LLM classifies each dish as vegetarian/non-vegetarian
4. **Confidence Check**: If < threshold (default 0.4), proceed to RAG
5. **RAG Node**: Query ChromaDB for similar dishes, boost confidence
6. **Calculator Node**: Call MCP server tool to sum vegetarian prices
7. **Return**: Classified dishes + total + metadata

### Non-AI Mode

Uses keyword-based classification, bypassing LLM and RAG:

```
Image → OCR → Parser → Keyword Matcher → MCP → Results
```

**Flow:**
1. **OCR**: Same as AI mode
2. **Parser**: Same as AI mode
3. **Keyword Matcher**: Check for vegetarian keywords (vegetarian, veggie, vegan, plant-based, etc.)
4. **MCP Calculator**: Same tool call as AI mode
5. **Return**: Classified dishes + total + metadata (no LLM reasoning)

## Service Details

### 1. API Service (FastAPI)

**Responsibilities:**
- REST API endpoints (`/health`, `/api/v1/extract-text`, `/api/v1/process-menu`)
- Image upload validation (max 5 images, 10MB each)
- OCR orchestration via Tesseract
- Text parsing into canonical dish format
- LangGraph agent execution
- Response formatting

**Key Components:**
- **Routers**: `menu.py` (main endpoints)
- **Services**: `ocr_service.py`, `parser_service.py`, `classifier_service.py`
- **Agents**: `menu_processor.py` (LangGraph state machine)
- **LLM**: `groq_client.py`, `openrouter_client.py`, `router_client.py`
- **Models**: Pydantic schemas for validation

**Configuration:** `api/config.py` (Pydantic Settings)

**Dependencies:**
- FastAPI, Uvicorn
- pytesseract, Pillow
- httpx (async HTTP)
- langgraph, langchain-core
- chromadb, sentence-transformers

### 2. MCP Server

**Responsibilities:**
- Expose MCP tools for calculation logic
- Receive pre-classified vegetarian dishes
- Return total price + metadata
- Deterministic, no AI/LLM dependencies

**Tools:**
- `calculate_vegetarian_total(dishes: List[Dish]) -> CalculationResult`

**Protocol:** Model Context Protocol (MCP) over HTTP

**Dependencies:**
- mcp (MCP SDK)
- pydantic

### 3. Streamlit UI

**Responsibilities:**
- Multi-page testing interface
- Image upload and results display
- LLM chat playground (compare Groq vs. OpenRouter)
- Phase-by-phase testing pages

**Pages:**
- `app.py` - Dashboard
- `1_OCR_Test.py` - OCR testing
- `2_Parser_Test.py` - Parser testing
- `3_Classifier_Test.py` - Classification testing
- `4_MCP_Test.py` - MCP integration testing
- `LLM_Chat_Playground.py` - Interactive LLM testing

**Dependencies:**
- Streamlit
- httpx (API communication)
- Groq/OpenRouter clients (for playground)

## Data Flow

### Request Processing (AI Mode)

```
1. User uploads menu image(s) via Streamlit or API

2. API Service receives request
   ├─ Validate images (format, size, count)
   ├─ Save to temp directory
   └─ Call OCR Service

3. OCR Service (Tesseract)
   ├─ Preprocess images (grayscale, contrast, denoise)
   ├─ Run Tesseract OCR
   ├─ Extract raw text + confidence scores
   └─ Return OCRResult[]

4. Parser Service
   ├─ Parse raw text line-by-line
   ├─ Extract dish names and prices (regex + heuristics)
   ├─ Build ParsedDish objects {name, price, raw_text}
   └─ Return ParsedMenu {dishes, stats}

5. LangGraph Menu-Processor Agent
   ├─ Initialize state with parsed dishes
   ├─ Execute Classifier Node
   │  ├─ For each dish:
   │  │  ├─ Build classification prompt
   │  │  ├─ Call LLM via router (Groq → OpenRouter fallback)
   │  │  ├─ Parse JSON response {is_vegetarian, confidence, reasoning}
   │  │  └─ Update state
   │  └─ Return classified dishes
   │
   ├─ Execute Conditional Edge (confidence check)
   │  ├─ If any dish confidence < threshold → RAG Node
   │  └─ Else → Calculator Node
   │
   ├─ Execute RAG Node (if triggered)
   │  ├─ For low-confidence dishes:
   │  │  ├─ Query ChromaDB with dish name
   │  │  ├─ Retrieve top-3 similar vegetarian dishes
   │  │  ├─ Compute similarity scores
   │  │  ├─ Re-classify with RAG context
   │  │  └─ Boost confidence based on similarity
   │  └─ Update state
   │
   └─ Execute Calculator Node
      ├─ Extract vegetarian dishes from state
      ├─ Call MCP server tool: calculate_vegetarian_total
      ├─ Receive total price + metadata
      └─ Update final state

6. API Response
   ├─ Format ProcessMenuResponse
   ├─ Include vegetarian dishes, total, metadata
   └─ Return JSON to client

7. LangSmith (if enabled)
   ├─ Trace entire request lifecycle
   ├─ Log token usage, latency per LLM call
   ├─ Record agent state transitions
   └─ Upload to LangSmith dashboard
```

### LLM Router Logic

```python
class LLMRouter:
    def complete_json(prompt, schema):
        try:
            # Try primary provider (Groq)
            response = groq_client.complete_json(prompt, schema)
            return response
        except Exception as e:
            logger.warning(f"Groq failed: {e}")
            # Fallback to OpenRouter
            response = openrouter_client.complete_json(prompt, schema)
            return response
```

**Features:**
- Automatic failover
- Token/latency tracking
- Schema validation (Pydantic)
- Streaming support (for chat playground)

### RAG Service

**Workflow:**
1. **Initialization** (on API startup):
   - Load ChromaDB from `api/rag_db/`
   - If empty, seed with `api/data/vegetarian_db.json`
   - Use `sentence-transformers/all-MiniLM-L6-v2` for embeddings

2. **Query** (during classification):
   - Embed dish name: `embedding_fn(dish.name)`
   - Search ChromaDB: `collection.query(embedding, n_results=3)`
   - Filter by metadata if needed
   - Return similar dishes with scores

3. **Confidence Boosting**:
   - Merge LLM confidence with RAG similarity
   - Formula: `final_conf = (llm_conf + max_similarity) / 2`
   - Update classification reasoning

## LangGraph State Machine

### State Schema

```python
class MenuProcessorState(TypedDict):
    parsed_menu: ParsedMenu
    classified_dishes: List[ClassifiedDish]
    low_confidence_dishes: List[ClassifiedDish]
    rag_results: Dict[str, List[SimilarDish]]
    total_price: float
    metadata: Dict[str, Any]
    errors: List[str]
```

### Nodes

1. **Classifier Node** (`classifier_node.py`)
   - Input: `parsed_menu`
   - Process: LLM classification for each dish
   - Output: `classified_dishes`, `low_confidence_dishes`

2. **RAG Node** (`rag_node.py`)
   - Input: `low_confidence_dishes`
   - Process: ChromaDB queries + re-classification
   - Output: Updated `classified_dishes`, `rag_results`

3. **Calculator Node** (`calculator_node.py`)
   - Input: `classified_dishes`
   - Process: MCP tool call
   - Output: `total_price`, `metadata`

### Edges

- **START → Classifier**: Always
- **Classifier → RAG**: If `len(low_confidence_dishes) > 0`
- **Classifier → Calculator**: If `len(low_confidence_dishes) == 0`
- **RAG → Calculator**: Always
- **Calculator → END**: Always

### Conditional Routing

```python
def should_use_rag(state: MenuProcessorState) -> str:
    if len(state["low_confidence_dishes"]) > 0:
        return "rag_node"
    return "calculator_node"
```

## Configuration Management

### Environment Variables

**Core Settings:**
- `DEBUG`: Enable debug mode
- `MAX_IMAGES`: Max images per request (default: 5)
- `CONFIDENCE_THRESHOLD`: Min confidence for classification (default: 0.4)

**Groq (Primary LLM):**
- `GROQ_API_KEY`: Groq API key
- `GROQ_BASE_URL`: https://api.groq.com/openai/v1
- `GROQ_PRIMARY_MODEL`: openai/gpt-oss-20b
- `GROQ_REQUEST_TIMEOUT`: 30 seconds

**OpenRouter (Fallback):**
- `OPENROUTER_API_KEY`: OpenRouter API key
- `OPENROUTER_BASE_URL`: https://openrouter.ai/api/v1
- `OPENROUTER_PRIMARY_MODEL`: deepseek/deepseek-chat-v3.1
- `OPENROUTER_FALLBACK_MODEL`: (optional secondary)

**MCP Server:**
- `MCP_SERVER_URL`: http://localhost:8001

**LangSmith:**
- `LANGCHAIN_TRACING_V2`: true/false
- `LANGCHAIN_API_KEY`: LangSmith API key
- `LANGCHAIN_PROJECT`: greendish

### Configuration Class

```python
# api/config.py
class Settings(BaseSettings):
    # Loaded from .env or environment
    # Validated with Pydantic
    # Accessed via singleton: settings.GROQ_API_KEY

    class Config:
        env_file = ".env"
        case_sensitive = True
```

## Docker Deployment

### Services

**docker-compose.yml** defines 3 services:

1. **mcp-server** (starts first)
   - Port: 8001
   - Healthcheck: `/health`
   - No external dependencies

2. **api** (depends on mcp-server)
   - Port: 8005
   - Volumes: `./api`, `./api/rag_db` (persistent ChromaDB)
   - Healthcheck: `/health`
   - Env: All API keys passed from .env

3. **streamlit** (depends on api)
   - Port: 8501
   - Healthcheck: `/_stcore/health`
   - Env: API_BASE_URL, LLM keys (for playground)

### Network

- Custom bridge network: `greendish-network`
- Services communicate via container names (e.g., `http://api:8005`)

### Volumes

- **Code volumes**: Hot-reload during development
- **ChromaDB volume**: Persistent RAG database

## Observability

### LangSmith Integration

**Traced Operations:**
- OCR processing time
- Parser execution
- LLM API calls (prompt, response, tokens, latency)
- RAG queries (similarity scores)
- MCP tool invocations
- Total request duration

**Metadata:**
- Request ID (UUID)
- Image count
- AI mode (true/false)
- Model used (Groq/OpenRouter)
- Confidence scores

**Dashboard:** https://smith.langchain.com/

### Logging

**Levels:**
- `INFO`: Request lifecycle, service calls
- `WARNING`: Groq fallback to OpenRouter
- `ERROR`: API failures, OCR errors
- `DEBUG`: Detailed state transitions (when DEBUG=true)

**Format:**
```
[2025-11-07 12:34:56] INFO [request_id=abc123] OCR completed in 1.2s
[2025-11-07 12:34:57] WARNING [request_id=abc123] Groq API timeout, falling back to OpenRouter
[2025-11-07 12:34:58] INFO [request_id=abc123] Classification complete: 10 dishes, 3 vegetarian
```

## Security Considerations

### API Keys

- Never commit `.env` to git
- Use `.env.example` as template
- Rotate keys regularly
- Use environment variables in production

### Input Validation

- Max file size: 10MB per image
- Allowed formats: JPEG, PNG, WEBP
- Max images per request: 5
- Sanitize filenames (prevent path traversal)

### CORS

- Configure allowed origins in production
- Default: localhost only for development

### Rate Limiting

- Not implemented (future enhancement)
- Recommend: 10 requests/minute per IP in production

## Performance Optimization

### OCR

- **Preprocessing**: Grayscale, contrast enhancement, denoising
- **Parallel processing**: Process multiple images concurrently
- **Caching**: (future) Cache OCR results by image hash

### LLM

- **Batching**: Classify multiple dishes in single request (future)
- **Caching**: (future) Cache classification by dish name hash
- **Model selection**: Groq for speed, OpenRouter for fallback

### RAG

- **Embedding cache**: sentence-transformers caches locally
- **ChromaDB persistence**: Avoid re-seeding on restart
- **Top-K**: Limit to 3 results for speed

### MCP

- **Lightweight**: Pure Python calculation, no I/O
- **HTTP/2**: (future) Use for multiplexing

## Scalability

### Current Limits

- **Single instance**: API, MCP server, Streamlit
- **Local ChromaDB**: Single node
- **No queue**: Synchronous processing

### Scaling Path

1. **Horizontal scaling** (API service):
   - Load balancer (Nginx, Traefik)
   - Multiple API replicas
   - Shared ChromaDB (migrate to server mode)

2. **Async processing**:
   - Queue (Redis, RabbitMQ)
   - Background workers (Celery)
   - Webhook callbacks

3. **Caching**:
   - Redis for OCR/LLM results
   - CDN for static assets

4. **Database**:
   - PostgreSQL for request history
   - S3 for image storage

## Testing Strategy

### Unit Tests

- Service layer: `tests/services/`
- Models: `tests/models/`
- Utilities: `tests/utils/`

### Integration Tests

- API endpoints: `tests/api/`
- MCP tools: `tests/mcp/`
- LangGraph agent: `tests/agents/`

### End-to-End Tests

- Full pipeline with sample menus
- AI vs. non-AI mode comparison
- Error handling (bad images, API failures)

### Test Fixtures

- Sample menu images: `tests/fixtures/images/`
- Mock responses: `tests/fixtures/responses/`

### Coverage

```bash
pytest --cov=api --cov=mcp-server tests/
# Target: >80% coverage
```

## Monitoring (Production)

### Metrics

- Request rate (requests/second)
- Response time (p50, p95, p99)
- Error rate (%)
- LLM token usage (total, per request)
- OCR success rate (%)

### Alerts

- API downtime (>1 minute)
- High error rate (>5%)
- LLM API failures (fallback triggered)
- Memory/CPU usage (>80%)

### Tools

- Prometheus + Grafana (metrics)
- Sentry (error tracking)
- LangSmith (LLM observability)

## Cost Analysis

### Development (Monthly)

- **Tesseract**: Free
- **ChromaDB**: Free (local)
- **Groq API**: Free tier or ~$5
- **OpenRouter**: ~$1 (fallback only)
- **LangSmith**: Free tier (5,000 traces)

**Total:** $0-10/month

### Production (1,000 menus/month)

- **Groq API**: ~$5 (0.005/menu)
- **OpenRouter**: ~$1 (10% fallback usage)
- **LangSmith**: ~$5 (beyond free tier)
- **Hosting**: Variable (AWS/GCP/Azure)

**Total:** $10-20/month (excluding hosting)

## Future Enhancements

### Phase 10+

- **HITL (Human-in-the-Loop)**: Review interface for uncertain classifications
- **Multi-language**: Support for non-English menus
- **Dietary restrictions**: Vegan, gluten-free, nut-free, etc.
- **Restaurant API**: Direct integration with POS systems
- **Mobile app**: iOS/Android clients
- **Batch processing**: Upload multiple menus at once
- **Analytics dashboard**: Usage stats, accuracy trends

### Technical Debt

- Add request rate limiting
- Implement result caching
- Migrate to async processing queue
- Add comprehensive logging (structured JSON)
- Improve error messages
- Add API versioning strategy

## Conclusion

GreenDish demonstrates a modern, production-ready architecture combining:
- **Microservices** for separation of concerns
- **LangGraph agents** for complex workflow orchestration
- **Dual LLM providers** for reliability
- **RAG** for accuracy improvement
- **MCP protocol** for tool integration
- **Docker** for easy deployment
- **LangSmith** for observability

The system is designed to be maintainable, scalable, and cost-effective, with clear paths for future enhancement.
