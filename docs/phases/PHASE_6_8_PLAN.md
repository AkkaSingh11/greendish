# Phases 6–8 Technical Plan (LangGraph + OpenRouter + RAG)

This document details the implementation plan for phases 6–8, grounded in the current repo state and the architecture described in SOLUTION_PLAN.md. It focuses on concrete module boundaries, state and node design for LangGraph, and the LLM/RAG glue code. No code is included here — this is a precise build plan.

## Current Status Snapshot

- Non‑AI pipeline is implemented end‑to‑end (OCR → Parse → Keyword Match → MCP totals) and wired to the API.
  - `/api/v1/process-menu` currently rejects `mode=ai` with a 501, confirming phases 6–8 are not implemented yet: `api/routers/menu.py:146`.
  - Keyword pipeline and models are live: `api/services/keyword_classifier.py`, `api/models/schemas.py:24`.
  - MCP calculator tool exists and is invokable via client: `mcp-server/tools/calculator.py:55`, `api/services/mcp_client.py:34`.
- Config wiring needs OpenRouter-specific settings/endpoints for LangGraph.

## Phase 6 — OpenRouter Integration

Goal: Add an OpenRouter client thin wrapper with model selection, retry/fallback, and structured JSON outputs for dish classification. Expose it as a library module to be reused by the LangGraph classifier node.

- Configuration
  - Add settings and env vars:
    - `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL` (default `https://openrouter.ai/api/v1`), `OPENROUTER_PRIMARY_MODEL` (`deepseek/deepseek-chat-v3.1`), optional `OPENROUTER_FALLBACK_MODEL`.
  - Map these into `api/config.py` and `.env.example`.

- Client module (`api/llm/openrouter_client.py`)
  - Provide an async `OpenRouterClient` with:
    - `complete_json(prompt: str, schema: dict | pydantic-model, *, temperature: float=0.0, max_tokens: int, model: str|None=None) -> dict`
    - Token accounting fields (request/response token counts if returned in headers), latency, and model used.
    - Error handling: 429 with exponential backoff, timeouts, non‑200 responses; fallback to configured secondary model on model‑specific failures.
  - Use `httpx.AsyncClient` with robust timeouts and retries (idempotent JSON POST). Include user‑agent `OpenRouter-Referer`/`X-Title` headers if desired by policy.

- Classification prompt and schema
  - Define a compact system+user prompt for deterministic structured output, targeting JSON only:
    - Fields: `is_vegetarian: bool`, `confidence: float (0..1)`, `reasoning: str`.
    - Add optional `signals` bag for surfaced features (keywords, ingredients) when available.
  - Enforce schema via parser/validator in the client wrapper; drop any extra fields.

- Model selection
  - Prefer `OPENROUTER_PRIMARY_MODEL`, lazy fallback to `OPENROUTER_FALLBACK_MODEL` only when:
    - Transport/HTTP errors, server 5xx, model‑specific 4xx (not auth), or structured parse failure.
  - Surface which model produced the result for observability.

- Test strategy
  - Mock OpenRouter responses with httpx mock transport; verify JSON coercion, fallback, retry/backoff paths, and failure surfaces.
  - Unit tests for prompt/schema validation edge cases (missing keys, out‑of‑range confidence).

Acceptance
- `OpenRouterClient` can return a valid `ClassificationResult`‑compatible dict for a given dish name/context, with fallback behavior and telemetry fields available to the graph.

## Phase 7 — LangGraph Agent Setup

Goal: Implement a dish‑wise stateful LangGraph agent that classifies dishes with the OpenRouter client, conditionally triggers RAG, and accumulates vegetarian dishes for downstream calculation.

References
- LangGraph StateGraph patterns: define a `TypedDict` state, add node functions, wire with `add_conditional_edges`, finalize with `compile()`, then `app.invoke(state)` or `app.stream(state)`.

- Dependencies
  - Add `langgraph` and `langchain-core` to `api/pyproject.toml`.

- Module layout (`api/agents`)
  - `state.py`: `TypedDict` for `MenuProcessorState` (matches SOLUTION_PLAN fields) and light helpers for mutations.
  - `nodes/classifier_node.py`: single‑dish classifier leveraging `OpenRouterClient` (produces per‑dish result and flags low‑confidence for RAG).
  - `nodes/rag_node.py`: enrich context for the current dish (Phase 8), writes `rag_lookups` and a `rag_context` payload.
  - `nodes/calculator_node.py`: terminal node that calls `MCPClient.calculate_total` on accumulated vegetarian dishes.
  - `menu_processor.py`: builds and compiles the `StateGraph` and exposes `run(dishes, config)` helpers for the API.

- State shape (no code, conceptual)
  - Required fields (subset):
    - `dishes: List[Dish]`, `current_dish_index: int`, `classified_dishes: List[ClassifiedDish]`, `vegetarian_dishes: List[Dish]`, `rag_lookups: List[RAGResult]`, `confidence_scores: Dict[str, float]`, `llm_model: str`, `mode: str`, `reasoning: str`.
  - Invariants: `0 <= current_dish_index <= len(dishes)`; graph terminates only when all dishes processed and calculator has run.

- Node contracts
  - `classify_dish(state) -> state_delta`
    - Inputs: `dishes[current_dish_index]`, optional `rag_context`.
    - Calls `OpenRouterClient.complete_json()` with deterministic prompt; writes `classified_dishes[i]`, updates `vegetarian_dishes` if true, increments `current_dish_index`, writes `confidence_scores[name]`, `reasoning` append.
    - Emits a transient flag: `needs_rag: bool` when `confidence < CONFIDENCE_THRESHOLD` and no `rag_context` used.
  - `rag_search(state) -> state_delta`
    - Inputs: `dishes[current_dish_index]` (same index), embeds/query top‑K (Phase 8 below), writes `rag_context`, appends `rag_lookups`.
  - `calculate_total(state) -> state_delta`
    - Inputs: `vegetarian_dishes` with prices; calls `MCPClient.calculate_total`, writes `total_price` and summary; clears transient flags.

- Graph wiring (control flow)
  - Entry: `classify`.
  - Conditional edges from `classify` using a guard:
    - If `needs_rag` → `rag_search`.
    - Else if `current_dish_index < len(dishes)` → loop back to `classify`.
    - Else → `calculate`.
  - Edge: `rag_search` → `classify` (re‑classification with context).
  - Edge: `calculate` → `END`.

- Execution API (for Phase 9 to consume later)
  - `run_menu_processor(dishes, *, llm_model, threshold, request_id) -> { vegetarian_dishes, total_price, per_dish, calc_summary, trace_url? }`.
  - Allow optional streaming via `app.stream()` for progress in UI/logs (kept internal for now).

- Error handling and fallbacks
  - If OpenRouter fails twice (primary+fallback), set `is_vegetarian=None`, mark dish uncertain; optionally call keyword classifier when `settings.use_keyword_fallback` is true to produce a best‑effort label and confidence.
  - Ensure the graph keeps advancing `current_dish_index` to avoid deadlocks when a dish repeatedly fails.

- Tests
  - Unit tests for guard function and loop termination with synthetic dishes.
  - Node‑level tests with mocked `OpenRouterClient` and `MCPClient` to validate state deltas and conditional edges.

Acceptance
- A compiled LangGraph `menu-processor` that deterministically classifies each dish, conditionally performs RAG, and aggregates vegetarian dishes ready for calculator invocation.

## Phase 8 — RAG Node Implementation

Goal: Add a lightweight local vector store using ChromaDB and `sentence-transformers` to provide retrieval context only when classification confidence is low.

- Vector store
  - Backend: ChromaDB (persistent path `api/rag_db/`).
  - Encoder: `all-MiniLM-L6-v2`.
  - Collections: `vegetarian_dishes` with documents containing dish names and short ingredient hints.
  - Seeding sources: bootstrap from existing keyword dataset `api/data/vegetarian_keywords.json` (positive dish keywords + ingredient terms). Optionally include a small curated CSV/JSON for known non‑veg dishes to balance retrieval.

- Retrieval policy
  - Triggered only when `confidence < settings.confidence_threshold` and no prior `rag_context`.
  - Query: embed dish name (and optional description snippet from `raw_text`), fetch top‑K (`K=3` default) with similarity scores.
  - Context construction: a compact bulleted summary of top hits with `category` labels (veg/non‑veg) and salient ingredient cues.

- Re‑classification with context
  - Re‑call `OpenRouterClient.complete_json()` with an augmented prompt that includes RAG evidence. If confidence improves and crosses threshold, accept; else persist low confidence and mark uncertain.
  - Cap confidence boost to prevent over‑trusting weak retrieval; e.g., clamp delta to `<= 0.15` above the initial score when the evidence is mixed.

- Caching and warmup
  - Persist embeddings; warm the collection at startup if empty; de‑duplicate by normalized name.
  - Optional async pre‑embed during application boot to avoid first‑query latency spikes.

- Tests
  - Build a tiny in‑memory collection; verify that ambiguous names (e.g., “Garden Delight”) receive improved confidence with appropriate evidence.
  - Validate guard logic: with/without `rag_context` and threshold boundaries.

Acceptance
- `rag_search` supplies relevant evidence that measurably improves classification confidence for ambiguous dishes without affecting the deterministic non‑AI pipeline.

## Integration Notes and Interfaces

- API integration (deferred to Phase 9)
  - Keep `process-menu?mode=ai` returning 501 until Phase 9 wires `menu_processor.run()` into the endpoint.
  - Graph results should be convertible into `ProcessMenuResponse` fields already defined (`api/models/schemas.py:64`).

- MCP contract
  - `MCPClient.calculate_total` expects vegetarian dishes with `price` and `confidence`; already implemented (`api/services/mcp_client.py:34`). The agent passes through this contract unchanged.

- Observability
  - While LangSmith wiring is Phase 10, retain hooks to capture request IDs and model names inside state for future trace annotations.

## Definition of Done (Phases 6–8)

- Phase 6
  - OpenRouter client exists with optional model fallback, schema‑validated JSON outputs, and tests.
  - Config/env supports OpenRouter without legacy key requirements.

- Phase 7
  - LangGraph agent compiles; classifier and calculator nodes function with deterministic dish iteration; guard logic drives RAG only when needed.
  - Unit tests cover edge routing and failure fallbacks.

- Phase 8
  - ChromaDB seeded and queried by a `rag_search` node; re‑classification with context can lift low confidence cases; tests demonstrate improvement on samples.

---

This plan follows LangGraph StateGraph patterns (typed state, node functions, conditional edges, `END`) and keeps MCP math external. It slots cleanly into the existing FastAPI surface while preserving the deterministic non‑AI path.
