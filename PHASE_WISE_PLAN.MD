Phase-Wise Implementation Plan for ConvergeFi
Phase 0: Git Initialization & Project Setup
Initialize git repository
Create .gitignore for Python, Docker, and IDE files
Create project directory structure (/api, /mcp-server, /streamlit-ui)
Create initial README.md with project overview
Phase 1: Basic Infrastructure & OCR (with Streamlit UI)
Goal: Image upload → OCR extraction → Display text Backend (/api):
Setup FastAPI with basic structure
Implement /health endpoint
Create OCR service using Tesseract
Create /extract-text endpoint (simplified version)
Add image validation and preprocessing
Streamlit UI (/streamlit-ui):
Image upload component (1-5 images)
Display OCR extracted text
Show processing status and errors
Docker: Basic Dockerfile for API service Testing: Use menu1.jpeg, menu2.png, menu3.webp
Phase 2: Text Parsing & Structuring (with Streamlit UI)
Goal: Raw OCR text → Structured dishes with prices Backend:
Implement parser service with regex patterns
Extract dish names and prices
Create Pydantic models for structured data
Update API to return structured dishes
Streamlit UI:
Display parsed dishes in table format
Show dish name, price, and raw text
Highlight parsing errors
Testing: Verify all 3 menu images parse correctly
Phase 3: Keyword-Based Classification (with Streamlit UI)
Goal: Basic vegetarian detection using keywords Backend:
Implement keyword matcher (vegetarian, veggie, veg, tofu, salad, etc.)
Add classification to dish objects
Create /classify-keyword endpoint
Calculate total for keyword-matched dishes
Streamlit UI:
Show classification results with color coding
Display vegetarian vs non-vegetarian items
Show total price for vegetarian dishes
Allow manual keyword configuration
Testing: Validate accuracy on test menus
Phase 4: MCP Server Setup (with Streamlit UI)
Goal: Separate calculation service using MCP protocol Backend:
Setup MCP server with HTTP transport
Implement calculate_vegetarian_total tool
Create MCP client in API service
Integrate API → MCP communication
Streamlit UI:
Show MCP server status
Display calculation results from MCP
Show request/response timing
Docker: Add mcp-server service to docker-compose.yml Testing: End-to-end API → MCP flow
Phase 5: LLM Classification (with Streamlit UI)
Goal: Intelligent classification using GPT-4o-mini/Claude Backend:
Integrate OpenAI or Anthropic API
Design classification prompt
Implement confidence scoring
Add fallback to keyword matching
Track token usage and costs
Streamlit UI:
Toggle between keyword/LLM/hybrid modes
Display confidence scores per dish
Show LLM reasoning
Display token usage and cost estimates
Show classification time
Testing: Compare LLM vs keyword accuracy
Phase 6: RAG Implementation (with Streamlit UI)
Goal: Vector store for improved confidence Backend:
Setup ChromaDB with persistent storage
Create seed data (vegetarian_db.json)
Implement sentence-transformers embeddings
Build retrieval logic (top-3 similar)
Integrate RAG scores into confidence calculation
Streamlit UI:
Display RAG retrieval results
Show similar dishes found
Visualize similarity scores
Allow viewing/editing seed data
Show combined confidence scores
Testing: Verify RAG improves accuracy on ambiguous dishes
Phase 7: LangSmith Observability (with Streamlit UI)
Goal: Full tracing and monitoring Backend:
Configure LangSmith environment
Add @traceable decorators
Track all key metrics (OCR time, LLM calls, RAG queries)
Generate request IDs
Streamlit UI:
Display trace IDs and LangSmith links
Show latency breakdown (OCR/parsing/LLM/MCP/RAG)
Display success/failure metrics
Show token usage over time
Testing: Verify all traces appear in LangSmith dashboard
Phase 8: HITL Review System (with Streamlit UI)
Goal: Human review for low-confidence items Backend:
Implement uncertainty detection (threshold < 0.7)
Create /review endpoint for corrections
Add reasoning explanations
Update vector store with corrections
Streamlit UI:
Show uncertain items requiring review
Provide approve/reject buttons
Display RAG evidence for decisions
Allow bulk corrections
Show before/after comparisons
Testing: Simulate low-confidence scenarios
Phase 9: Testing & Documentation
Goal: Production-ready system Tasks:
Write unit tests for all services
Integration tests for API ↔ MCP
End-to-end tests with all 3 menu images
Complete README with architecture diagrams
Document API endpoints (OpenAPI/Swagger)
Add setup/installation instructions
Create troubleshooting guide
Phase 10: Optimization & Refinement
Goal: Performance and reliability improvements Tasks:
Add caching for OCR results
Implement retry logic for service communication
Add rate limiting
Optimize ChromaDB queries
Add comprehensive error handling
Security hardening (input validation, sanitization)
Streamlit UI Structure
streamlit-ui/
├── app.py (main Streamlit app)
├── pages/
│   ├── 1_OCR_Test.py
│   ├── 2_Parser_Test.py
│   ├── 3_Keyword_Classification.py
│   ├── 4_MCP_Integration.py
│   ├── 5_LLM_Classification.py
│   ├── 6_RAG_Explorer.py
│   ├── 7_Observability.py
│   └── 8_HITL_Review.py
├── requirements.txt
└── config.py
Key Decision Points
LLM Provider: OpenAI (GPT-4o-mini) vs Anthropic (Claude Haiku) - need API key
LangSmith Setup: Need API key for tracing
Confidence Threshold: Default 0.7, adjustable in UI
RAG Seed Data: Start with basic vegetarian dishes/ingredients, expand based on testing
Success Criteria Per Phase
Each phase completion requires: ✅ Backend functionality working ✅ Streamlit UI showing results ✅ Tests passing with sample menus ✅ Git commit with clear message ✅ Documentation updated Ready to begin implementation?