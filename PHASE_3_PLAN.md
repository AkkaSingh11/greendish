# Phase 3 Execution Plan — Non-AI Keyword Classifier

This plan expands on the Phase 3 outline in `SOLUTION_PLAN.md` and maps out the concrete work needed to finish the keyword-based vegetarian classifier.

## 1. Objectives
- Transform OCR-parsed dishes into vegetarian/non-vegetarian classifications without any LLM calls.
- Provide confidence scores and uncertainty flags consumable by the REST API and (later) the MCP calculator.
- Keep the classifier modular so it can share interfaces with the upcoming LangGraph agent.

## 2. Deliverables
- `api/data/vegetarian_keywords.json` seeded with positive/negative keyword groups (initial dataset committed).
- `KeywordClassifier` module under `api/services/keyword_classifier.py` with:
  - deterministic matching rules (exact, substring, fuzzy)
  - scoring + uncertainty handling
  - API-facing response model(s)
- Unit tests covering keyword loading, matching, scoring edge cases, and negative keyword overrides.
- Updated FastAPI wiring to expose the classifier behind a service layer (no endpoint changes yet).

## 3. Work Breakdown & Order of Operations

### 3.1 Keyword Data Layer
1. Finalise JSON schema contract (already drafted) and document expected fields.
2. Add Pydantic settings/model for loading `vegetarian_keywords.json` with caching + hot reload toggle.
3. Ship resilience helpers:
   - surface missing-file errors during startup
   - checksum or version info for logging

### 3.2 Matching Engine
1. Build `KeywordClassifier` class with injectable config + optional overrides (for tests).
2. Implement match stages:
   - **Exact dish hits:** normalised string equality (case/whitespace insensitive).
   - **Ingredient matches:** keyword containment on tokenised dish names + `raw_text`.
   - **Fuzzy similarity:** Levenshtein ratio (use `rapidfuzz` if available; otherwise `difflib`).
   - **Negative guards:** meat/seafood/stocks blocklist wins over positives.
3. Produce scoring rubric:
   - base score by strongest signal (exact > ingredient > fuzzy)
   - subtract penalties when negative hints appear but are overridden
   - surface `confidence` (0–1) + `reasoning` strings for traceability.
4. Add helper for `is_uncertain` if score < configurable threshold or conflicting signals.

### 3.3 API Integration Points
1. Extend existing parser pipeline (`api/services/parser_service.py`) to call the classifier behind a feature flag.
2. Return structured payload aligning with future MCP contract (FastMCP tools default to structured responses when type hints are present):
   ```json
   {
     "name": "Paneer Tikka",
     "price": 12.5,
     "is_vegetarian": true,
     "confidence": 0.82,
     "signals": {"exact_match": "paneer tikka"}
   }
   ```
3. Provide dependency stub (FastAPI `Depends`) so `/process-menu?mode=non-ai` can hook directly in Phase 5.
4. Capture classifier provenance (matched keywords, rule applied) so the MCP calculation tool can surface it via structured tool responses.

### 3.4 MCP Alignment Tasks
1. Document the classifier output schema next to the MCP `calculate_vegetarian_total` tool contract; both should share the same Pydantic model so FastMCP can auto-validate payloads.
2. Add module-level docstring referencing the Model Context Protocol core concepts (tools vs resources) to guide future contributors—link to `https://modelcontextprotocol.io` for specification details.
3. Plan for optional `KeywordClassifier` exposure as a lightweight MCP resource in later phases (e.g., `classifier://keyword-signals/{dish_id}`) to let agent tooling fetch intermediate evidence when needed.

### 3.5 Quality & Tooling
1. Unit tests: craft fixtures with vegetarian, mixed, and meat dishes using sample OCR payloads.
2. Add regression test to ensure a negative marker ("chicken soup") always returns false even with vegetable keywords.
3. Integrate with existing logging/metrics skeleton — log top uncertain dishes for future review.

## 4. Timeline & Ownership
- **Day 3 Morning:** Keyword dataset + loader, service skeleton, baseline exact matching.
- **Day 3 Afternoon:** Fuzzy matching + scoring, negative keyword overrides, unit tests, documentation updates.

## 5. Risks & Mitigations
- **Over-matching salty keywords** (e.g., "beef tomato"). Mitigation: enforce negative priority + token-level context.
- **Dataset drift:** Plan periodic manual reviews; keep `metadata.sources` for reproducibility.
- **Performance:** Keep classifier pure-Python; add memoization per request to avoid redundant tokenisation.

## 6. Next Dependencies
- Phase 4 (MCP calculator) expects `is_vegetarian` + `confidence`; ensure response shape matches upcoming contract.
- Provide toggle to reuse classifier inside LangGraph nodes (Phase 7 onwards).

## 7. Validation Checklist
- [ ] `pytest tests/services/test_keyword_classifier.py`
- [ ] Smoke run: `python -m api.scripts.sample_keyword_run` (to be added) prints classification summary.
- [ ] Lint formatting (ruff/black) to keep CI green.
