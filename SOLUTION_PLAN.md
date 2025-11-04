# Restaurant Menu Vegetarian Dish Analyzer - Solution Plan

## Project Overview
A microservices-based system that processes restaurant menu photos to identify and sum up prices of vegetarian dishes using OCR, LLM classification, and MCP (Model Context Protocol) server architecture.

## Technology Stack

### Core Framework
- **Language**: Python 3.11+
- **REST API**: FastAPI
- **MCP Server**: Python MCP SDK (`mcp` package)

### Key Libraries & Tools
- **OCR**: Tesseract OCR + `pytesseract`
- **Image Processing**: Pillow (PIL)
- **LLM**: OpenAI API (gpt-4o-mini for cost efficiency) or Anthropic Claude via API
- **Vector Database**: ChromaDB (local, open-source, easy setup)
- **Embeddings**: `sentence-transformers` (all-MiniLM-L6-v2 model)
- **Observability**: LangSmith (free tier: 5,000 traces/month)
- **Containerization**: Docker + Docker Compose
- **HTTP Client**: `httpx` for async requests

### Why This Stack?
- All components support free tier usage
- Tesseract is completely free and open-source
- ChromaDB runs locally without external dependencies
- LangSmith free tier provides sufficient tracing for development
- FastAPI offers excellent performance and automatic documentation
- MCP is a modern standard for LLM tool integration

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Client                               │
└────────────────────┬────────────────────────────────────────┘
                     │ POST /process-menu
                     │ (multipart/form-data)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    REST API Service                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 1. Image Upload & Validation                         │   │
│  │ 2. OCR Processing (Tesseract)                        │   │
│  │ 3. Text Parsing & Structuring                        │   │
│  │ 4. LLM Classification (with keyword fallback)        │   │
│  └──────────────────────┬───────────────────────────────┘   │
└─────────────────────────┼───────────────────────────────────┘
                          │ HTTP POST
                          │ (vegetarian dishes data)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    MCP Server Service                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Tool: calculate_vegetarian_total                     │   │
│  │ - RAG Vector Search (ChromaDB)                       │   │
│  │ - Confidence Scoring                                 │   │
│  │ - Price Summation                                    │   │
│  │ - Return: total + confidence + reasoning             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│           LangSmith Observability (Tracing)                  │
│  - Request ID tracking across services                       │
│  - OCR latency, LLM calls, MCP tool calls                   │
│  - Token usage, confidence scores                            │
└─────────────────────────────────────────────────────────────┘
```

## Component Design

### 1. REST API Service (`/api`)

#### Endpoints
- `POST /process-menu` - Main endpoint to process menu images
- `POST /review` - (Optional) HITL endpoint for corrections
- `GET /health` - Health check endpoint

#### Flow
1. **Image Upload**: Accept 1-5 images (multipart/form-data)
2. **OCR Extraction**: Use Tesseract to extract text from images
3. **Text Parsing**: Parse OCR output into structured dish objects
   - Use regex patterns to identify dish names and prices
   - Format: `{name: str, price: float, raw_text: str}`
4. **Classification**:
   - Primary: LLM-based classification (GPT-4o-mini)
   - Fallback: Keyword matching (vegetarian, veggie, veg, tofu, salad, etc.)
5. **MCP Tool Call**: Send structured dishes to MCP server
6. **Response**: Return JSON with vegetarian items and total

#### Key Modules
```
api/
├── main.py                 # FastAPI app
├── routers/
│   ├── menu.py            # Menu processing endpoints
│   └── review.py          # HITL review endpoints (optional)
├── services/
│   ├── ocr_service.py     # Tesseract OCR wrapper
│   ├── parser_service.py  # Text to structured data
│   ├── llm_service.py     # LLM classification
│   └── mcp_client.py      # MCP server HTTP client
├── models/
│   ├── request.py         # Pydantic request models
│   └── response.py        # Pydantic response models
└── config.py              # Configuration
```

### 2. MCP Server Service (`/mcp-server`)

#### MCP Tools
- `calculate_vegetarian_total` - Main calculation tool
  - Input: List of candidate dishes with classifications
  - Output: Total sum + confidence + reasoning

#### Flow
1. **Receive Dishes**: Get structured dish data from REST API
2. **RAG Retrieval**:
   - Query ChromaDB vector store with dish names
   - Retrieve top-3 similar vegetarian ingredients/dishes
   - Calculate semantic similarity scores
3. **Confidence Calculation**:
   - LLM confidence (0-1)
   - Keyword match bonus (+0.2)
   - RAG similarity score (0-1)
   - Combined: weighted average
4. **Summation**: Calculate total for dishes above threshold (0.7)
5. **Return**: JSON with results and reasoning

#### Key Modules
```
mcp-server/
├── server.py              # MCP server with tools
├── tools/
│   └── calculator.py      # Calculation logic
├── rag/
│   ├── vector_store.py    # ChromaDB wrapper
│   ├── embeddings.py      # Sentence transformer embeddings
│   └── retriever.py       # RAG retrieval logic
├── data/
│   └── vegetarian_db.json # Seed data for vector store
└── config.py              # Configuration
```

### 3. RAG Memory Implementation

#### Vector Store Setup
- **Database**: ChromaDB (persistent local storage)
- **Embeddings Model**: `all-MiniLM-L6-v2` (384 dimensions, fast)
- **Collection**: "vegetarian_dishes"

#### Seed Data Structure
```json
{
  "dishes": [
    {"name": "Greek Salad", "category": "vegetarian", "common_ingredients": ["lettuce", "tomato", "cucumber", "feta"]},
    {"name": "Veggie Burger", "category": "vegetarian", "common_ingredients": ["patty", "bun", "lettuce"]},
    ...
  ],
  "ingredients": [
    {"name": "tofu", "category": "vegetarian"},
    {"name": "paneer", "category": "vegetarian"},
    ...
  ]
}
```

#### Retrieval Strategy
1. Embed dish name using sentence transformer
2. Query ChromaDB for top-3 similar entries
3. Calculate weighted similarity score
4. Use as additional confidence signal

### 4. Observability with LangSmith

#### Implementation
- Use `@traceable` decorator for key functions
- Environment variables:
  - `LANGCHAIN_TRACING_V2=true`
  - `LANGCHAIN_API_KEY=<key>`
  - `LANGCHAIN_ENDPOINT=https://api.smith.langchain.com`

#### Tracked Metrics
- **Request Level**: `request_id`, total latency, success/failure
- **OCR**: Processing time per image, character count extracted
- **Parsing**: Number of dishes identified, parsing errors
- **LLM Calls**: Token count, latency, model used, cost estimate
- **MCP Tool Calls**: Input size, response time, tool name
- **RAG**: Query time, number of retrievals, similarity scores
- **Classification**: Per-dish confidence, final count, total sum

### 5. HITL Loop (Optional)

#### Low Confidence Handling
- Threshold: Combined confidence < 0.7
- Response format:
```json
{
  "status": "needs_review",
  "uncertain_items": [
    {
      "name": "Garden Delight",
      "price": 12.5,
      "confidence": 0.65,
      "reasoning": "No clear vegetarian keywords; RAG found 'Garden Salad' (0.72 similarity)",
      "rag_evidence": ["Garden Salad", "Greek Salad", "Caesar Salad"]
    }
  ]
}
```

#### Review Endpoint
- `POST /review`
- Accept corrections: `{"corrections": [{"name": "Garden Delight", "is_vegetarian": true}]}`
- Recompute deterministically with user input
- Optionally update vector store with confirmed examples

## Docker Setup

### Services
1. **api**: REST API service (Port 8000)
2. **mcp-server**: MCP server (Port 8001)

### docker-compose.yml
```yaml
version: '3.8'
services:
  api:
    build: ./api
    ports:
      - "8000:8000"
    environment:
      - MCP_SERVER_URL=http://mcp-server:8001
      - LANGCHAIN_TRACING_V2=true
    depends_on:
      - mcp-server
    volumes:
      - ./api:/app

  mcp-server:
    build: ./mcp-server
    ports:
      - "8001:8001"
    environment:
      - LANGCHAIN_TRACING_V2=true
    volumes:
      - ./mcp-server:/app
      - chroma-data:/app/chroma_db
volumes:
  chroma-data:
```

## Implementation Steps

### Phase 1: Setup & Infrastructure (Day 1)
1. Initialize monorepo structure
2. Setup Docker configuration for both services
3. Install core dependencies (FastAPI, MCP SDK, Tesseract)
4. Configure LangSmith for observability
5. Create basic health check endpoints

### Phase 2: REST API - OCR & Parsing (Day 2)
1. Implement image upload endpoint with validation
2. Integrate Tesseract OCR for text extraction
3. Build parser to structure OCR output (regex-based)
4. Create keyword-based fallback classifier
5. Add LangSmith tracing to all functions

### Phase 3: MCP Server - Basic Calculation (Day 3)
1. Setup MCP server with HTTP transport
2. Implement `calculate_vegetarian_total` tool
3. Create basic summation logic
4. Test inter-service communication
5. Add tracing for MCP tool calls

### Phase 4: LLM Classification (Day 4)
1. Integrate OpenAI/Anthropic API
2. Design prompt for vegetarian classification
3. Implement confidence scoring
4. Add cost tracking and token counting
5. Handle API errors gracefully

### Phase 5: RAG Implementation (Day 5)
1. Setup ChromaDB with persistent storage
2. Create seed data for vegetarian dishes/ingredients
3. Implement embedding generation with sentence-transformers
4. Build retrieval logic (top-k similar items)
5. Integrate RAG scores into confidence calculation

### Phase 6: HITL & Refinement (Day 6)
1. Implement uncertainty detection
2. Create review endpoint for corrections
3. Add reasoning explanations to responses
4. Refine confidence thresholds
5. Add comprehensive error handling

### Phase 7: Testing & Documentation (Day 7)
1. Write unit tests for parsers and classifiers
2. Integration tests for API-MCP communication
3. End-to-end testing with sample menu images
4. Complete README with setup instructions
5. Document architecture and design choices

## Testing Strategy

### Unit Tests
- OCR extraction accuracy
- Parser regex patterns
- Keyword matching logic
- Price summation calculations
- Confidence score computation

### Integration Tests
- API → MCP server communication
- ChromaDB query/retrieval
- LLM API calls with mocking
- LangSmith trace generation

### End-to-End Tests
- Upload sample menu images
- Verify correct vegetarian identification
- Validate price totals
- Check confidence scores
- Confirm HITL flow for uncertain items

### Test Data
- Create 5-10 sample menu images with known answers
- Include edge cases: unclear names, missing prices, handwritten text
- Test single and multiple image uploads

## Cost Estimates (Free Tier)

- **Tesseract OCR**: Free, open-source
- **ChromaDB**: Free, local deployment
- **Sentence Transformers**: Free, runs locally
- **LangSmith**: Free tier (5,000 traces/month, sufficient for dev/testing)
- **LLM API**:
  - OpenAI GPT-4o-mini: ~$0.15/1M input tokens, $0.60/1M output tokens
  - Anthropic Claude Haiku: ~$0.25/1M input tokens, $1.25/1M output tokens
  - Estimate: ~50-100 tokens per dish → ~$0.01 per menu
- **Hosting**: Docker runs locally, $0

**Total Monthly Cost (Dev/Testing)**: ~$0-5

## Configuration Management

### Environment Variables
```env
# API Service
MCP_SERVER_URL=http://mcp-server:8001
OPENAI_API_KEY=<key>
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<key>
MAX_IMAGES=5
CONFIDENCE_THRESHOLD=0.7

# MCP Server
CHROMA_PERSIST_DIR=/app/chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K_RETRIEVAL=3
```

## Security Considerations
- Validate image file types and sizes
- Sanitize OCR output before parsing
- Rate limit API endpoints
- Don't log sensitive data (API keys)
- Use environment variables for secrets

## Performance Optimizations
- Cache OCR results for duplicate images
- Batch LLM requests when possible
- Use async/await for I/O operations
- Preload embedding model at startup
- Index optimization for ChromaDB

## Deliverables Checklist

- [ ] Git monorepo with `/api` and `/mcp-server` directories
- [ ] Docker Compose setup with both services
- [ ] README.md with:
  - [ ] System requirements
  - [ ] Installation instructions
  - [ ] Running instructions (`docker-compose up`)
  - [ ] API documentation
  - [ ] Architecture diagram
  - [ ] Design choices explanation
  - [ ] Testing approach
- [ ] Working `/process-menu` endpoint
- [ ] MCP server with calculation tool
- [ ] RAG vector store with seed data
- [ ] LangSmith tracing integration
- [ ] (Optional) HITL review endpoint
- [ ] Test suite with sample images

## Design Choices Rationale

### 1. FastAPI over Flask
- Better performance (ASGI vs WSGI)
- Native async support
- Automatic OpenAPI documentation
- Type safety with Pydantic

### 2. ChromaDB over FAISS
- Full database features (metadata filtering)
- Easier setup (no manual index management)
- Persistent storage out-of-the-box
- Better developer experience

### 3. MCP Protocol
- Modern standard for LLM tool integration
- Clear separation of concerns
- Network-based communication
- Future-proof architecture

### 4. Two-Service Architecture
- Clear responsibility separation
- Independent scaling potential
- Easier testing and debugging
- Follows microservices best practices

### 5. LLM + Keyword Hybrid
- LLM provides intelligent classification
- Keywords ensure baseline accuracy
- Fallback mechanism for API failures
- Cost optimization (skip LLM if keywords match)

### 6. RAG for Confidence
- Improves classification accuracy
- Provides explainability
- Handles ambiguous dish names
- Low overhead (local embeddings)

## Risk Mitigation

### OCR Accuracy Issues
- **Risk**: Poor quality images, handwriting, complex layouts
- **Mitigation**: Preprocessing (grayscale, contrast), multiple OCR configs, manual review for low confidence

### LLM API Failures
- **Risk**: Rate limits, downtime, cost overruns
- **Mitigation**: Fallback to keywords, caching, error handling, cost monitoring

### False Positives/Negatives
- **Risk**: Misclassifying non-vegetarian as vegetarian or vice versa
- **Mitigation**: HITL loop, confidence thresholds, RAG evidence, user review

### Service Communication Failures
- **Risk**: Network issues between API and MCP server
- **Mitigation**: Retry logic, timeouts, health checks, graceful degradation

## Future Enhancements (Out of Scope)
- Support for more dietary restrictions (vegan, gluten-free)
- Multiple language support
- Menu item photo recognition (computer vision)
- Price extraction from handwritten menus
- User feedback loop to improve model
- Database of restaurant menus
- Mobile app interface

---

## Summary
This solution provides a production-ready, microservices-based system using modern best practices. The architecture is simple yet extensible, uses only free/open-source components, and follows the MCP standard for LLM tool integration. The implementation prioritizes correctness, observability, and user experience with the optional HITL loop for handling uncertainty.
