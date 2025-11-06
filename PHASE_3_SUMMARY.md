# Phase 3 Summary — Non-AI Keyword Classifier

## Implementation Overview
- Added a deterministic `KeywordClassifier` service (`api/services/keyword_classifier.py`) with curated positive/negative keyword datasets, fuzzy matching, and provenance signals.
- Extended API configuration (`api/config.py`) for keyword classifier toggles, data path, and thresholds while tolerating extra environment settings.
- Enriched the `Dish` schema to carry classification reasoning and matched signal metadata (`api/models/schemas.py`).
- Wired keyword classification into the `/api/v1/process-menu` workflow so parsed dishes return vegetarian flags and confidence scores (`api/routers/menu.py`).
- Exported classifier utilities for reuse across services (`api/services/__init__.py`) and ensured package discovery via `api/__init__.py`.
- Seeded the initial keyword dataset (`api/data/vegetarian_keywords.json`) using open food corpora with documented sources.
- Added regression tests covering exact, negative override, fuzzy, and uncertain scenarios (`tests/services/test_keyword_classifier.py`).

## Streamlit Integration
- Main dashboard now calls `/api/v1/process-menu`, rendering OCR output, parsed dishes, and vegetarian metrics (`streamlit-ui/app.py`).
- Sidebar copy reflects Phase 3 completion; users now click “Process Menu” to view vegetarian counts.
- Classification details per dish (reasoning, signals, raw text) are available via expanders for manual QA.

## Phase3_Testing
### Setup
- Environment: macOS (Tesseract located at `/opt/homebrew/bin/tesseract`, exported as `TESSERACT_CMD` for tests).
- Pipeline executed via local script mirroring API flow: OCR → Parser → KeywordClassifier.
- Dataset updated with additional positive dish keywords (`french fries`) and broader seafood negatives (e.g., `shrimp`, `lobster`, `scallop`, `prawn`, `crawfish`) to curb false positives in sample menus.

### Results (OCR Test Images)
| Image | Total Dishes | Vegetarian Dishes | Vegetarian % |
|-------|--------------|-------------------|--------------|
| `image_4.webp` | 14 | 2 | 14.29% |
| `image_6.png` | 52 | 9 | 17.31% |
| `menu1.jpeg` | 84 | 10 | 11.90% |
| `menu2.png` | 15 | 1 | 6.67% |
| `menu3.webp` | 65 | 10 | 15.38% |

### Observations
- Added seafood negatives successfully flipped prior false positives such as “Garlic Shrimp” back to non-vegetarian.
- `French Fries` now registers as vegetarian after augmenting dish keywords.
- Remaining non-vegetarian defaults predominantly arise from ambiguous dish names lacking explicit vegetarian markers; no additional manual tuning applied pending product guidance.

---

# Phase 4 Summary — MCP Calculator Service

## Implementation Overview
- Created a dedicated FastMCP server (`mcp-server/server.py`) with a `calculate_vegetarian_total` tool and debugging resource.
- Added reusable Pydantic data contracts for calculator input/output (`mcp-server/models.py`) and core aggregation logic (`mcp-server/tools/calculator.py`).
- Integrated an MCP client in the API (`api/services/mcp_client.py`) that leverages the streamable HTTP transport to call the calculator tool.
- Extended the menu processing endpoint to invoke the MCP calculator, populate `calculation_summary`, and fall back gracefully if the MCP service is unavailable (`api/routers/menu.py`).
- Updated domain models to carry calculation metadata back to clients (`api/models/schemas.py`) and supplied Docker orchestration for the new service (`docker-compose.yml`).

## Streamlit & UX Updates
- Streamlit dashboard automatically reflects MCP totals through the existing vegetarian summary, so end users now see aggregated totals alongside confidence metrics.

## Phase4_Testing
### Unit Tests
- `pytest tests/mcp/test_calculator.py` — validates aggregation logic for totals, averages, and low-confidence detection.
- `pytest tests/services/test_keyword_classifier.py` — regression to ensure Phase 3 classifier remains stable after dependency upgrades.

### Manual Validation
- Launched the MCP server locally (`python mcp-server/server.py`) and confirmed the `/health` endpoint plus successful tool invocation via the updated API pipeline.
- Re-ran the OCR → parse → classify flow against `ocr_test_images/*`, verifying identical vegetarian counts with MCP-derived totals.

### Results Snapshot
- MCP calculator outputs include total price, average confidence, and low-confidence dish names (visible in API/Streamlit responses via `calculation_summary`).
- Fallback handling confirmed by temporarily stopping the MCP service: API logs an error and returns a locally computed total with an explanatory reasoning string.
