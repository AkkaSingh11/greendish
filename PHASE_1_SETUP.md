# Phase 1: OCR Implementation - Setup Guide

## Prerequisites

1. **Python 3.11+** installed
2. **uv** - Fast Python package installer
   - Install: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Or: `brew install uv` (macOS) / `pip install uv`
3. **Tesseract OCR** installed:
   - macOS: `brew install tesseract`
   - Ubuntu: `sudo apt-get install tesseract-ocr`
   - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
4. **Docker** (optional, for containerized setup)

## Quick Start (Local Development)

### 1. Setup Environment with uv

```bash
# Install API dependencies
cd api
uv sync  # Creates .venv and installs all dependencies
cd ..

# Install Streamlit UI dependencies
cd streamlit-ui
uv sync  # Creates .venv and installs all dependencies
cd ..
```

**Alternative: Use existing Python environment**
```bash
# Install dependencies into current environment
cd api
uv pip install -r pyproject.toml
cd ..

cd streamlit-ui
uv pip install -r pyproject.toml
cd ..
```

### 2. Verify Tesseract Installation

```bash
tesseract --version
```

You should see output like:
```
tesseract 5.x.x
```

### 3. Start the API Server

```bash
# Terminal 1
cd api
python main.py
```

API will be available at:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 4. Start the Streamlit UI

```bash
# Terminal 2
cd streamlit-ui
streamlit run app.py
```

Streamlit UI will be available at:
- **UI**: http://localhost:8501

### 5. Test with Sample Menus

1. Open http://localhost:8501 in your browser
2. Upload the sample menu images:
   - `menu1.jpeg`
   - `menu2.png`
   - `menu3.webp`
3. Click "Extract Text"
4. Review the OCR results

## Docker Setup (Alternative)

```bash
# Build and start all services
docker-compose up --build

# Stop services
docker-compose down
```

Access:
- API: http://localhost:8000
- Streamlit UI: http://localhost:8501

## Testing the API Directly

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Extract text from menu
curl -X POST "http://localhost:8000/api/v1/extract-text" \
  -F "files=@menu1.jpeg" \
  -F "files=@menu2.png"
```

### Using Python

```python
import requests

# Upload images
with open('menu1.jpeg', 'rb') as f1, open('menu2.png', 'rb') as f2:
    files = [
        ('files', ('menu1.jpeg', f1, 'image/jpeg')),
        ('files', ('menu2.png', f2, 'image/png'))
    ]
    response = requests.post(
        'http://localhost:8000/api/v1/extract-text',
        files=files
    )
    print(response.json())
```

## Phase 1 Features

✅ **Implemented:**
- Image upload (1-5 images, JPEG/PNG/WEBP)
- OCR text extraction using Tesseract
- Image validation and preprocessing
- Confidence score calculation
- Processing time metrics
- RESTful API with FastAPI
- Interactive Streamlit UI
- Docker containerization
- Health check endpoint
- Comprehensive error handling

❌ **Not Yet Implemented (Future Phases):**
- Text parsing into dishes
- Vegetarian classification
- Price extraction and calculation
- MCP server integration
- LLM classification
- RAG with ChromaDB
- LangSmith observability
- HITL review system

## Project Structure (Phase 1)

```
ConvergeFi/
├── api/
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py          # Pydantic models
│   ├── routers/
│   │   ├── __init__.py
│   │   └── menu.py             # API endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   └── ocr_service.py      # Tesseract OCR logic
│   ├── config.py               # Configuration
│   ├── main.py                 # FastAPI app
│   ├── requirements.txt
│   └── Dockerfile
├── streamlit-ui/
│   ├── app.py                  # Main UI
│   ├── config.py               # UI config
│   ├── requirements.txt
│   └── Dockerfile
├── menu1.jpeg                  # Sample menu 1
├── menu2.png                   # Sample menu 2
├── menu3.webp                  # Sample menu 3
└── docker-compose.yml
```

## Troubleshooting

### Tesseract Not Found

**Error:** `TesseractNotFoundError`

**Solution:**
1. Install Tesseract OCR
2. If installed but not in PATH, set in `.env`:
   ```
   TESSERACT_CMD=/usr/local/bin/tesseract  # Your Tesseract path
   ```

### Import Errors

**Error:** `ModuleNotFoundError`

**Solution:**
```bash
# Make sure you're in the correct directory
cd api  # or streamlit-ui

# Reinstall dependencies
pip install -r requirements.txt
```

### API Not Accessible

**Error:** Connection refused to http://localhost:8000

**Solution:**
1. Check if API is running: `ps aux | grep uvicorn`
2. Check for port conflicts: `lsof -i :8000`
3. Restart API server

### Docker Issues

**Error:** Docker build fails

**Solution:**
```bash
# Clean rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up
```

## API Endpoints

### GET /health
Returns API health status and version.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-04T10:30:00",
  "version": "0.1.0",
  "services": {
    "api": "healthy",
    "ocr": "healthy"
  }
}
```

### POST /api/v1/extract-text
Extract text from uploaded menu images.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: files (1-5 images)

**Response:**
```json
[
  {
    "image_name": "menu1.jpeg",
    "raw_text": "Extracted text from menu...",
    "confidence": 87.5,
    "processing_time_ms": 245.3
  }
]
```

### POST /api/v1/process-menu
Process menu images (Phase 1: OCR only).

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: files (1-5 images)

**Response:**
```json
{
  "request_id": "uuid-here",
  "total_images": 2,
  "ocr_results": [...],
  "dishes": [],
  "vegetarian_dishes": [],
  "total_price": 0.0,
  "processing_time_ms": 523.7,
  "langsmith_trace_url": null
}
```

## Next Steps (Phase 2)

Phase 2 will add:
- Text parsing to extract dish names and prices
- Regex patterns for various menu formats
- Structured dish data models
- Enhanced Streamlit UI with parsed results

## Testing Checklist

- [ ] API starts without errors
- [ ] Health endpoint returns 200
- [ ] Streamlit UI loads successfully
- [ ] Can upload single image
- [ ] Can upload multiple images (2-5)
- [ ] OCR extracts text from menu1.jpeg
- [ ] OCR extracts text from menu2.png
- [ ] OCR extracts text from menu3.webp
- [ ] Confidence scores are displayed
- [ ] Processing times are reasonable (<1s per image)
- [ ] Error handling works (upload invalid file)
- [ ] Error handling works (upload too many files)

## Support

If you encounter issues:
1. Check this troubleshooting guide
2. Review API logs
3. Check Streamlit console output
4. Create an issue in the repository
