# Restaurant Menu Vegetarian Dish Analyzer - Solution Plan

## Project Overview
A microservices-based system centered around a LangGraph `menu-processor` agent that consumes structured menu data, orchestrates vegetarian classification, and defers price calculations to a dedicated MCP (Model Context Protocol) server. The pipeline processes restaurant menu photos, normalizes them into JSON dishes with prices, and supports both LLM-driven and non-LLM keyword workflows to accurately sum vegetarian dish totals.

## Technology Stack

### Core Framework
- **Language**: Python 3.11+
- **REST API**: FastAPI
- **MCP Server**: Python MCP SDK (`mcp` package) leveraged via FastMCP for defining tools/resources over stdio and HTTP transports

### Key Libraries & Tools
- **OCR**: Tesseract OCR + `pytesseract`
- **Image Processing**: Pillow (PIL)
- **LLM Router**: Groq API (primary) with OpenRouter fallback
  - Groq default: `openai/gpt-oss-20b` (fast OSS frontier model)
  - OpenRouter fallback: `deepseek/deepseek-chat-v3.1` or any OpenRouter-supported model
  - Provider/model choices configurable from both API and Streamlit UI
- **Agent Framework**: LangGraph (for menu-processor agent)
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
- Groq delivers high-throughput OSS models while OpenRouter still provides unified fallback access
- LangGraph enables complex, stateful agent workflows with graph-based orchestration
- Dual approach (AI agent + non-AI keyword matching) provides flexibility and cost control

## System Architecture

### Two Processing Approaches

The system supports two distinct processing modes:
1. **AI Agent Mode (LangGraph)**: Uses LLM-based classification with RAG fallback
2. **Non-AI Mode (Keyword Matching)**: Uses deterministic keyword-based classification

Regardless of mode, the OCR pipeline standardizes menu content into a JSON array of `{name, price, raw_text}` dishes. That structured payload feeds either the LangGraph `menu-processor` agent—which delegates to a dish classification agent, triggers RAG lookups on uncertainty, and finally calls the MCP calculator tool for vegetarian totals—or the lightweight keyword matcher that skips LLM usage entirely before invoking the same MCP service.

```
┌─────────────────────────────────────────────────────────────┐
│                         Client                               │
└────────────────────┬────────────────────────────────────────┘
                     │ POST /process-menu?mode=[ai|non-ai]
                     │ (multipart/form-data)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    REST API Service                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 1. Image Upload & Validation                         │   │
│  │ 2. OCR Processing (Tesseract)                        │   │
│  │ 3. Text Parsing & Structuring                        │   │
│  │    → Returns: {dishes: [{name, price, raw_text}]}   │   │
│  └──────────────────────┬───────────────────────────────┘   │
└─────────────────────────┼───────────────────────────────────┘
                          │
         ┌────────────────┴────────────────┐
         │                                  │
         ▼                                  ▼
┌────────────────────┐          ┌─────────────────────────┐
│   AI Agent Mode    │          │   Non-AI Mode           │
│  (LangGraph)       │          │   (Keyword Matching)    │
└────────┬───────────┘          └──────────┬──────────────┘
         │                                  │
         ▼                                  ▼
┌─────────────────────────────────────────────────────────────┐
│            LangGraph Menu-Processor Agent                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ State Graph:                                         │   │
│  │                                                       │   │
│  │  START                                               │   │
│  │    ↓                                                 │   │
│  │  [Classify Dishes]                                   │   │
│  │    ├─→ Try LLM Classification (Groq → OpenRouter)   │   │
│  │    │   ├─ deepseek/deepseek-chat-v3.1               │   │
│  │    │   └─ (optional fallback model)                 │   │
│  │    │                                                 │   │
│  │    ├─→ [Confidence Check]                           │   │
│  │    │      ├─ High confidence → Continue             │   │
│  │    │      └─ Low confidence → RAG Search            │   │
│  │    │                                                 │   │
│  │    └─→ [RAG Tool] (if needed)                       │   │
│  │         ├─ Query ChromaDB for similar dishes        │   │
│  │         └─ Re-classify with RAG context             │   │
│  │                                                       │   │
│  │  [Calculate Total]                                   │   │
│  │    └─→ Call MCP Server Tool                         │   │
│  │         (calculate_vegetarian_total)                 │   │
│  │                                                       │   │
│  │  END → Return final result                           │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                                │
         ▼                                ▼
┌───────────────────┐          ┌──────────────────────────┐
│  MCP Server       │          │  Keyword Classifier       │
│  (Calculation)    │          │  (Non-AI)                 │
│                   │          │                           │
│  Tool:            │          │  - Load keyword dataset   │
│  calculate_       │          │  - Match dish names       │
│  vegetarian_total │          │  - Direct classification  │
│                   │          │  - Call MCP as function   │
└───────────────────┘          └──────────────────────────┘
         │                                │
         └────────────────┬───────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Final Response                              │
│  {                                                           │
│    "dishes": [...],                                          │
│    "vegetarian_dishes": [...],                              │
│    "total": 45.50,                                           │
│    "confidence": 0.92,                                       │
│    "mode": "ai|non-ai",                                      │
│    "reasoning": "..."                                        │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│           LangSmith Observability (Tracing)                  │
│  - Request ID tracking across services                       │
│  - OCR latency, LLM calls, agent steps, MCP tool calls      │
│  - Token usage, confidence scores, mode selection            │
└─────────────────────────────────────────────────────────────┘
```

## Component Design

### 1. REST API Service (`/api`)

#### Endpoints
- `POST /process-menu?mode=[ai|non-ai]` - Main endpoint to process menu images
  - Query parameter `mode` selects processing approach (default: "ai")
- `POST /review` - (Optional) HITL endpoint for corrections
- `GET /health` - Health check endpoint

#### Flow
1. **Image Upload**: Accept 1-5 images (multipart/form-data)
2. **OCR Extraction**: Use Tesseract to extract text from images
3. **Text Parsing**: Parse OCR output into structured dish objects
   - Use regex patterns to identify dish names and prices
   - **CRITICAL**: Return structured JSON format that becomes the canonical payload for downstream agents and tools:
   ```json
   {
     "dishes": [
       {"name": "Greek Salad", "price": 12.50, "raw_text": "Greek Salad ... $12.50"},
       {"name": "Chicken Tikka", "price": 18.99, "raw_text": "Chicken Tikka ... $18.99"}
     ],
     "total_dishes": 2,
     "ocr_confidence": 0.95
   }
   ```
4. **Mode Selection**:
   - **AI Mode**: Pass parsed dishes to LangGraph menu-processor agent
   - **Non-AI Mode**: Pass parsed dishes to keyword classifier
5. **Response**: Return JSON with vegetarian items, total, and processing metadata

#### Key Modules
```
api/
├── main.py                      # FastAPI app
├── routers/
│   ├── menu.py                 # Menu processing endpoints
│   └── review.py               # HITL review endpoints (optional)
├── services/
│   ├── ocr_service.py          # Tesseract OCR wrapper
│   ├── parser_service.py       # Text to structured JSON
│   ├── keyword_classifier.py   # Non-AI keyword-based classification
│   └── mcp_client.py           # MCP server HTTP client
├── agents/
│   ├── menu_processor.py       # LangGraph agent definition
│   ├── nodes/
│   │   ├── classifier_node.py  # LLM classification node
│   │   ├── rag_node.py         # RAG search node
│   │   └── calculator_node.py  # MCP calculation node
│   └── state.py                # Agent state definition
├── llm/
│   ├── groq_client.py          # Groq API wrapper (primary)
│   ├── openrouter_client.py    # OpenRouter API wrapper (fallback)
│   └── router_client.py        # Provider router (Groq → OpenRouter)
├── models/
│   ├── request.py              # Pydantic request models
│   └── response.py             # Pydantic response models
└── config.py                   # Shared configuration (Groq/OpenRouter env + models)
```

### 2. LangGraph Menu-Processor Agent (`/api/agents`)

#### Agent Architecture
The menu-processor agent is a LangGraph StateGraph that receives parsed menu JSON and coordinates a dedicated dish classification agent alongside downstream calculation tooling.

#### State Definition
```python
class MenuProcessorState(TypedDict):
    dishes: List[Dish]                    # Input: parsed dishes
    current_dish_index: int               # Tracking progress
    classified_dishes: List[ClassifiedDish]
    vegetarian_dishes: List[Dish]
    rag_lookups: List[RAGResult]          # RAG evidence trail
    total_price: float
    confidence_scores: Dict[str, float]
    mode: str                             # "ai" or "non-ai"
    llm_model: str                        # Selected OpenRouter model
    reasoning: str
```

The `menu-processor` agent forwards each dish payload to an embedded dish classification agent. This classifier selects an OpenRouter model (primary Deepseek, fallback GPT OSS) that is configurable from both the API service and the Streamlit console. When the classifier returns low confidence, the menu-processor automatically launches a RAG retrieval pass, enriches the context, and re-invokes the classification agent before proceeding.

#### Graph Nodes

The agent maintains dish-by-dish state while guaranteeing that every final total is produced by invoking the external MCP calculation tool, keeping arithmetic out of the LLM loop.

1. **Classifier Node** (`classify_dish`)
   - Uses OpenRouter API with configurable model
   - Prompts LLM: "Is '{dish_name}' vegetarian? Return JSON with confidence."
   - Returns: `{is_vegetarian: bool, confidence: float, reasoning: str}`
   - If confidence < threshold → trigger RAG node

2. **RAG Node** (`rag_search`)
   - Queries ChromaDB for top-3 similar dishes
   - Provides context to re-run LLM classification
   - Increases confidence with semantic evidence

3. **Calculator Node** (`calculate_total`)
   - Calls MCP server tool: `calculate_vegetarian_total`
   - Input: List of classified vegetarian dishes
   - Returns: Total sum + aggregated confidence

#### Graph Flow
```python
graph = StateGraph(MenuProcessorState)
graph.add_node("classify", classifier_node)
graph.add_node("rag_search", rag_node)
graph.add_node("calculate", calculator_node)

# Edges
graph.set_entry_point("classify")
graph.add_conditional_edges(
    "classify",
    should_use_rag,  # Check confidence threshold
    {True: "rag_search", False: "calculate"}
)
graph.add_edge("rag_search", "classify")  # Re-classify with RAG
graph.add_edge("calculate", END)
```

### 3. Non-AI Keyword Classifier (`/api/services/keyword_classifier.py`)

#### Functionality
- Loads keyword dataset from `/data/vegetarian_keywords.json`
- Performs exact and fuzzy matching on dish names
- Dataset structure:
  ```json
  {
    "vegetarian_keywords": ["vegetarian", "veg", "veggie", "tofu", "paneer", "salad", ...],
    "non_vegetarian_keywords": ["chicken", "beef", "pork", "fish", "meat", "lamb", ...],
    "vegetarian_dishes": ["Greek Salad", "Veggie Burger", "Tofu Scramble", ...]
  }
  ```
- Returns classification with keyword-based confidence (0.0-1.0)
- Directly calls MCP server as a Python function (not through agent)
- Never invokes an LLM; relies solely on deterministic keyword heuristics before delegating totals to the same MCP calculation tool.

### 4. MCP Server Service (`/mcp-server`)

#### MCP Tools
- `calculate_vegetarian_total` - Main calculation tool
  - Input: List of classified vegetarian dishes `[{name, price, confidence}]`
  - Output: `{total: float, confidence: float, reasoning: str}`
  - Pure calculation logic - no classification
  - Declared with type-hinted return models so FastMCP emits structured tool responses automatically

#### Flow
1. **Receive Dishes**: Get list of pre-classified vegetarian dishes
2. **Validate Input**: Ensure all dishes have price and confidence
3. **Summation**: Calculate total price
4. **Confidence Aggregation**: Average/weighted confidence across dishes
5. **Return**: JSON with total and reasoning

#### Protocol Usage Notes
- Stand up the service with `FastMCP` from the MCP Python SDK to manage connection lifecycle, stdio/HTTP transports, and capability negotiation.
- MCP divides functionality into **Resources** (read-only context), **Tools** (actions), and **Prompts** (reusable LLM instructions); we will prioritise tools for calculations and optionally expose resources like `resource://classifier/keywords` for debugging.
- The calculator tool can accept a `Context` parameter when we need progress reporting or logging back to clients (e.g., LangGraph nodes or MCP Inspector).
- Host the server via Streamable HTTP so the FastAPI service and LangGraph agent can call the MCP endpoint over the internal network while still supporting stdio for local inspection.
- Keep implementation aligned with the published specification at `https://modelcontextprotocol.io/specification/latest` for forward compatibility.

#### Key Modules
```
mcp-server/
├── server.py              # MCP server with tools
├── tools/
│   └── calculator.py      # Pure calculation logic (no classification)
└── config.py              # Configuration
```

Note: RAG functionality is moved to the API service's LangGraph agent for better separation of concerns.

### 5. RAG Memory Implementation (AI Mode Only)

#### Vector Store Setup
- **Database**: ChromaDB (persistent local storage in `/api/rag_db/`)
- **Embeddings Model**: `all-MiniLM-L6-v2` (384 dimensions, fast)
- **Collection**: "vegetarian_dishes"
- **Location**: Embedded within LangGraph RAG node

#### Seed Data Structure
```json
{
  "dishes": [
    {"name": "Greek Salad", "category": "vegetarian", "common_ingredients": ["lettuce", "tomato", "cucumber", "feta"]},
    {"name": "Veggie Burger", "category": "vegetarian", "common_ingredients": ["patty", "bun", "lettuce"]},
    {"name": "Chicken Caesar Salad", "category": "non-vegetarian", "common_ingredients": ["chicken", "lettuce", "caesar dressing"]},
    ...
  ],
  "ingredients": [
    {"name": "tofu", "category": "vegetarian"},
    {"name": "paneer", "category": "vegetarian"},
    {"name": "chicken", "category": "non-vegetarian"},
    {"name": "beef", "category": "non-vegetarian"},
    ...
  ]
}
```

#### Retrieval Strategy (Within RAG Node)
1. Triggered only when LLM classification confidence < threshold (0.7)
2. Embed uncertain dish name using sentence transformer
3. Query ChromaDB for top-3 similar dishes with known classifications
4. Provide RAG context to LLM for re-classification
5. Boost confidence if RAG evidence supports classification

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
1. **api**: REST API service (Port 8005)
2. **mcp-server**: MCP server (Port 8001)

### docker-compose.yml
```yaml
version: '3.8'
services:
  api:
    build: ./api
    ports:
      - "8005:8005"
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

### Phase 1: Setup & Infrastructure ✅ (Completed)
1. Initialize monorepo structure
2. Setup Docker configuration for services
3. Install core dependencies (FastAPI, Tesseract)
4. Create basic health check endpoints
5. Implement OCR service

### Phase 2: REST API - Parsing & Structured Output ✅ (Completed)
1. Build parser to structure OCR output (regex-based)
2. **Return structured JSON**: `{dishes: [{name, price, raw_text}], total_dishes, ocr_confidence}`
3. Add parsing validation and error handling
4. Create Pydantic models for structured responses
5. Test with sample menu images

### Phase 3: Non-AI Keyword Classifier (Day 3)
1. Create keyword dataset (`/data/vegetarian_keywords.json`)
2. Implement `KeywordClassifier` service
3. Build exact and fuzzy matching logic
4. Add confidence scoring based on keyword matches
5. Test non-AI classification pipeline

### Phase 4: MCP Server - Pure Calculation (Day 4)
1. Setup FastMCP server with Streamable HTTP transport (while retaining stdio for MCP Inspector)
2. Implement `calculate_vegetarian_total` tool (pure calculation) with typed models for structured responses
3. Accept pre-classified dishes as input and optionally expose read-only resources (e.g., keyword catalog metadata) for debugging
4. Create summation and confidence aggregation logic plus optional `Context`-driven logging
5. Test inter-service communication (FastAPI ↔ MCP) via httpx client and MCP Inspector

### Phase 5: Non-AI End-to-End Pipeline (Day 5)
1. Integrate keyword classifier with MCP server
2. Implement `/process-menu?mode=non-ai` endpoint
3. Create complete non-AI flow: OCR → Parse → Keyword Match → Calculate
4. Add comprehensive error handling
5. Validate results with test menus

### Phase 6: OpenRouter Integration (Day 6)
1. Setup OpenRouter API client
2. Configure `deepseek/deepseek-chat-v3.1` (optional fallback additional model)
3. Implement model selection and fallback logic
4. Design classification prompts
5. Add token tracking and cost monitoring

### Phase 7: LangGraph Agent Setup (Day 7)
1. Install LangGraph and dependencies
2. Define `MenuProcessorState` schema
3. Create basic state graph with nodes
4. Implement classifier node (OpenRouter LLM)
5. Test basic agent execution

### Phase 8: RAG Node Implementation (Day 8)
1. Setup ChromaDB within API service
2. Create vegetarian dishes seed data
3. Implement RAG node for low-confidence lookups
4. Add conditional edge: low confidence → RAG
5. Test RAG-enhanced classification

### Phase 9: Calculator Node & Agent Integration (Day 9)
1. Implement calculator node (calls MCP server)
2. Complete LangGraph state transitions
3. Add agent execution to `/process-menu?mode=ai` endpoint
4. Implement error handling and retries
5. Test full AI pipeline

### Phase 10: LangSmith Observability (Day 10)
1. Configure LangSmith tracing for both modes
2. Add trace decorators to key functions
3. Track: OCR time, agent steps, LLM calls, RAG queries, MCP calls
4. Display trace URLs in API responses
5. Create observability dashboard

### Phase 11: HITL & Refinement (Day 11)
1. Implement uncertainty detection
2. Create review endpoint for corrections
3. Add detailed reasoning to responses
4. Refine confidence thresholds
5. Add comprehensive error handling

### Phase 12: Testing & Documentation (Day 12)
1. Write unit tests for parsers, classifiers, and agents
2. Integration tests for both AI and non-AI modes
3. End-to-end testing with diverse menu images
4. Complete README with architecture diagrams
5. Document design choices and trade-offs

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
  - Groq GPT-OSS-20B: ~$0.59/1M input tokens, ~$0.79/1M output tokens (subject to Groq pricing updates)
  - OpenRouter DeepSeek (fallback): ~$0.14/1K input tokens, ~$0.28/1K output tokens
  - Estimate: ~50-100 tokens per dish → ~$0.01 per menu
- **Hosting**: Docker runs locally, $0

**Total Monthly Cost (Dev/Testing)**: ~$0-5

## Configuration Management

### Environment Variables
```env
# API Service
MCP_SERVER_URL=http://mcp-server:8001
PROCESSING_MODE=ai  # Options: ai, non-ai

# Groq Configuration
GROQ_API_KEY=<key>
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_PRIMARY_MODEL=openai/gpt-oss-20b
GROQ_REQUEST_TIMEOUT=30

# OpenRouter Configuration (fallback)
OPENROUTER_API_KEY=<key>
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_PRIMARY_MODEL=deepseek/deepseek-chat-v3.1
OPENROUTER_FALLBACK_MODEL=
OPENROUTER_APP_NAME=ConvergeFi-MenuAnalyzer

# LangSmith Observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<key>
LANGCHAIN_PROJECT=convergeFi-menu-processor

# Processing Configuration
MAX_IMAGES=5
CONFIDENCE_THRESHOLD=0.7
USE_RAG_FALLBACK=true

# RAG Configuration (AI Mode)
RAG_DB_PATH=/app/rag_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K_RETRIEVAL=3

# Keyword Classifier (Non-AI Mode)
KEYWORD_DB_PATH=/app/data/vegetarian_keywords.json
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
This solution delivers a production-ready, microservices-based system built around a LangGraph `menu-processor` agent that accepts structured `{name, price}` dish JSON, delegates vegetarian classification to a Groq-powered agent (with OpenRouter fallback) plus RAG, and offloads all price summations to an MCP calculator service. The architecture remains extensible, cost-aware, and observable, while the non-LLM keyword mode guarantees a deterministic alternative that still leverages the shared MCP tooling.
