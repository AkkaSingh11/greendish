# Phase-Wise Implementation Plan for ConvergeFi

## Project Overview
ConvergeFi is a Restaurant Menu Vegetarian Dish Analyzer that processes menu photos to identify and sum up prices of vegetarian dishes using OCR, LLM classification, and MCP server architecture.

---

## Phase 0: Git Initialization & Project Setup ✅ COMPLETED

### Goals
- Initialize repository structure
- Setup version control
- Create base documentation

### Tasks
- [x] Initialize git repository
- [x] Create `.gitignore` for Python, Docker, and IDE files
- [x] Create project directory structure (`/api`, `/mcp-server`, `/streamlit-ui`, `/tests`)
- [x] Create initial README.md with project overview

### Deliverables
- Git repository initialized
- Project structure created
- README with architecture overview

### Commit
- ✅ Commit: "Phase 0: Initialize project structure and documentation"

---

## Phase 1: Basic Infrastructure & OCR (with Streamlit UI) ✅ COMPLETED

### Goals
- Image upload → OCR extraction → Display text
- Working API with OCR capabilities
- Interactive Streamlit UI for testing

### Backend (`/api`)
- [x] Setup FastAPI with basic structure
- [x] Implement `/health` endpoint
- [x] Create OCR service using Tesseract
- [x] Create `/api/v1/extract-text` endpoint
- [x] Create `/api/v1/process-menu` endpoint (Phase 1 version)
- [x] Add image validation and preprocessing
- [x] Pydantic models for API responses
- [x] Confidence score calculation
- [x] Processing time metrics

### Streamlit UI (`/streamlit-ui`)
- [x] Image upload component (1-5 images)
- [x] Display OCR extracted text
- [x] Show processing status and errors
- [x] Confidence scores and metrics
- [x] API health check display

### Docker
- [x] Dockerfile for API service (with Tesseract)
- [x] Dockerfile for Streamlit UI
- [x] docker-compose.yml for orchestration

### Testing
- [x] Test with menu1.jpeg (88.7% confidence, 2940 chars)
- [x] Test with menu2.png (95.2% confidence, 399 chars)
- [x] Test with menu3.webp (93.7% confidence, 2487 chars)
- [x] Python test script (`test_ocr.py`)

### Deliverables
- Working FastAPI server with OCR
- Streamlit testing interface
- Docker containerization
- Setup guide (PHASE_1_SETUP.md)
- All sample menus processing successfully

### Commit
- ✅ Commit: "Phase 1: OCR Implementation Complete"

---

## Phase 2: Text Parsing & Structuring (with Streamlit UI) 🔄 NEXT

### Goals
- Raw OCR text → Structured dishes with prices
- Parse dish names and prices from extracted text
- Handle various menu formats

### Backend
- [ ] Implement parser service with regex patterns
- [ ] Extract dish names from text
- [ ] Extract prices (support $X.XX, XX.XX, etc.)
- [ ] Create Pydantic models for structured dishes
- [ ] Update `/api/v1/process-menu` to return parsed dishes
- [ ] Handle multi-line dishes
- [ ] Handle dishes without prices
- [ ] Add parser confidence scores

### Streamlit UI
- [ ] Display parsed dishes in table format
- [ ] Show dish name, price, and raw text
- [ ] Highlight parsing errors
- [ ] Show parsing statistics
- [ ] Allow manual correction of parsing

### Testing
- [ ] Verify all 3 menu images parse correctly
- [ ] Test edge cases (missing prices, special chars)
- [ ] Validate dish count matches manual count

### Deliverables
- Parser service with regex engine
- Structured dish data models
- Enhanced Streamlit UI with tables
- Parsing accuracy > 90%

### Commit
- ⏳ Commit: "Phase 2: Text Parsing and Dish Structuring"

---

## Phase 3-10: Future Phases

### Phase 3: Keyword-Based Classification
- Basic vegetarian detection using keywords
- Calculate total price for vegetarian dishes

### Phase 4: MCP Server Setup
- Separate calculation service using MCP protocol
- Service-to-service communication

### Phase 5: LLM Classification
- Intelligent classification using GPT-4o-mini/Claude
- Confidence scoring and fallback

### Phase 6: RAG Implementation
- Vector store for improved confidence
- Semantic similarity search

### Phase 7: LangSmith Observability
- Full tracing and monitoring
- Performance metrics

### Phase 8: HITL Review System
- Human review for low-confidence items
- Feedback loop for model improvement

### Phase 9: Testing & Documentation
- Production-ready system
- Comprehensive test coverage

### Phase 10: Optimization & Refinement
- Performance and reliability improvements
- Production hardening

---

## Project Status

### Completed Phases
- ✅ Phase 0: Project Setup
- ✅ Phase 1: OCR Implementation

### Current Phase
- 🔄 Phase 2: Text Parsing (NEXT)

---

## Quick Start

### Running Phase 1

```bash
# Start API server
cd api
python main.py

# In another terminal, run tests
cd ..
python test_ocr.py

# Or use Streamlit UI
cd streamlit-ui
streamlit run app.py
```

### Docker

```bash
docker-compose up --build
```

Access:
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501

---

## Success Criteria Per Phase

Each phase completion requires:
- ✅ Backend functionality working
- ✅ Streamlit UI showing results
- ✅ Tests passing with sample menus
- ✅ Git commit with clear message
- ✅ Documentation updated
