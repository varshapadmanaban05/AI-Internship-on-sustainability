# Design Document: RePurpose AI

## Overview

RePurpose AI is a conversational, AI-powered circular reuse and repurposing advisor built as a student internship prototype aligned with SDG 12 (Responsible Consumption & Production). Users describe an unwanted item in plain language, and the system recommends the most appropriate circular pathway from a 7-level hierarchy (Reuse As-Is → Repair → Repurpose → Donate → Refurbish → Recycle → Responsible Disposal).

Recommendations are grounded in a curated sustainability knowledge base via a Retrieval-Augmented Generation (RAG) pipeline, ensuring advice is cited and traceable rather than hallucinated. The interface is a Streamlit web application accessible on localhost:8501.

The system is intentionally scoped as a student internship prototype: it runs fully locally, uses a pre-loaded knowledge base of 30–50 documents, and relies on free or low-cost API tiers for the LLM.

### Goals

- Provide actionable, hierarchy-aware circular economy recommendations grounded in real references
- Detect hazardous items pre-LLM and surface safety warnings prominently
- Support multi-turn conversation to refine advice without restarting
- Remain simple to run: `pip install -r requirements.txt`, set API key, `streamlit run app.py`

### Non-Goals

- Real-time web scraping or live knowledge base updates
- User authentication or persistent user accounts
- Production deployment, scaling, or multi-user concurrency
- Mobile-specific UI optimisation

---

## Architecture

The system follows a layered pipeline architecture:

```
┌─────────────────────────────────────────────────┐
│                  app.py (Streamlit UI)           │
│  - Input validation (min 5, max 1000 chars)      │
│  - Session state: conversation history           │
│  - Renders Recommendation + Safety Warning       │
└───────────────────┬─────────────────────────────┘
                    │ calls
┌───────────────────▼─────────────────────────────┐
│             rag/pipeline.py (Orchestrator)       │
│  1. Invoke Retriever                             │
│  2. Keyword-based hazard check                   │
│  3. Build message list (history + context)       │
│  4. Invoke Generator                             │
│  Returns: {recommendation, is_hazardous,         │
│            safety_warning, chunks_used}          │
└──────┬────────────────────────┬──────────────────┘
       │                        │
┌──────▼──────────┐    ┌────────▼──────────────────┐
│ rag/retriever.py│    │    rag/generator.py        │
│                 │    │                            │
│ sentence-       │    │ OpenAI API                 │
│ transformers    │    │ (gpt-3.5-turbo / gpt-4o)   │
│ all-MiniLM-L6-v2│    │ IBM watsonx.ai (alt)       │
│                 │    │                            │
│ ChromaDB        │    │ System prompt encodes:     │
│ (local, persist)│    │ - 7-level hierarchy        │
│ cosine sim ≥0.50│    │ - Citation instructions    │
│ top 3–5 chunks  │    │ - Confidence Note rules    │
└─────────────────┘    └────────────────────────────┘

┌─────────────────────────────────────────────────┐
│            ingest.py (Standalone Script)         │
│  - Reads data/docs/ (PDF + TXT)                  │
│  - Splits to 300–500 token chunks, 50-token OL   │
│  - Embeds + stores to data/chroma_db/            │
└─────────────────────────────────────────────────┘
```

### Data Flow

1. User types an item description in the Streamlit UI
2. `app.py` validates the input (length 5–1000 chars) and calls `pipeline.run()`
3. `pipeline.run()` calls `retriever.retrieve(description)` → returns top 3–5 chunks
4. `pipeline.run()` performs keyword hazard check on the raw item description
5. `pipeline.run()` assembles the LLM message list: system prompt + last 5 conversation history turns + new user message (description + retrieved chunk context)
6. `pipeline.run()` calls `generator.generate(messages)` → returns structured markdown
7. `app.py` renders the response: Safety Warning (if triggered) followed by the Recommendation body
8. `app.py` appends the exchange to `st.session_state.history`

---

## Components and Interfaces

### app.py — Streamlit UI

Responsibilities:
- Render the chat interface (title, description, usage instruction, scrollable chat log)
- Validate item description input (min 5 chars, max 1000 chars, live character counter)
- Show spinner while pipeline is running; disable submission during processing
- Render Safety Warning (styled `st.warning` or custom `st.markdown` block) before Recommendation body when `is_hazardous=True`
- Render Recommendation using `st.markdown` with Streamlit formatting
- Maintain and display `st.session_state.history` (full history in UI; only last 5 passed to pipeline)
- Provide "Clear Conversation" button that resets session state and clears display

Key functions:
```python
def validate_input(text: str) -> tuple[bool, str]:
    """Returns (is_valid, error_message). Checks length [5, 1000]."""

def render_recommendation(result: dict) -> None:
    """Renders Safety Warning + Recommendation body from pipeline result."""

def run_pipeline_with_spinner(description: str, history: list) -> dict:
    """Calls pipeline.run() with Streamlit spinner context."""
```

Session state keys:
- `st.session_state.history`: `list[dict]` — `[{"role": "user"|"assistant", "content": str}, ...]`
- `st.session_state.processing`: `bool` — blocks submit button when True

---

### rag/retriever.py — Retrieval Engine

Responsibilities:
- Load the sentence-transformers `all-MiniLM-L6-v2` model (cached after first load)
- Embed a query string to a 384-dimensional float vector
- Query ChromaDB with cosine distance filter (distance ≤ 0.50, equivalent to similarity ≥ 0.50)
- Return top 3–5 chunks with metadata attached
- Handle embedding model errors gracefully (raise `RetrieverUnavailableError`)

Interface:
```python
class Retriever:
    def __init__(self, chroma_path: str = "data/chroma_db", collection_name: str = "knowledge_base"):
        ...

    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        """
        Returns list of Chunk objects with similarity >= 0.50.
        Returns empty list if no chunks meet threshold.
        Raises RetrieverUnavailableError if embedding model fails.
        """

@dataclass
class Chunk:
    text: str
    source_title: str   # non-empty; from metadata
    source_file: str    # non-empty; filename or URL from metadata
    similarity: float   # cosine similarity score [0.0, 1.0]
```

ChromaDB note: ChromaDB returns L2 distance by default. The collection is configured with `cosine` distance metric at creation time. A returned distance `d` maps to similarity `s = 1 - d` (for cosine distance). The filter `where_document` or `query` distance threshold is set to `≤ 0.50` (meaning similarity ≥ 0.50).

---

### rag/generator.py — LLM Generator

Responsibilities:
- Accept a pre-built message list (`list[dict]` in OpenAI chat format)
- Call the LLM API (OpenAI or watsonx.ai)
- Return the raw response string
- Handle API errors and timeouts (raise `GeneratorUnavailableError` after 30-second timeout)

Interface:
```python
class Generator:
    def __init__(self, model: str = "gpt-3.5-turbo", timeout: int = 30):
        ...

    def generate(self, messages: list[dict]) -> str:
        """
        Calls LLM with messages list.
        Returns structured markdown recommendation string.
        Raises GeneratorUnavailableError on API error or timeout.
        """
```

System prompt (embedded in `generator.py`):
```
You are RePurpose AI, a circular economy advisor. Given a description of an unwanted item 
and relevant knowledge base excerpts, recommend the best circular pathway from this priority 
order (highest to lowest): Reuse As-Is, Repair, Repurpose, Donate, Refurbish, Recycle, 
Responsible Disposal.

Always structure your response with these exact sections in order:
**Recommended Pathway**: [pathway name]
**Reasoning**: [why this pathway; if not Reuse As-Is, explain why higher options do not apply]
**Source Citations**:
1. [source_title from retrieved chunk]
2. [source_title from retrieved chunk]
...
**Next Steps**:
- [concrete action the user can take immediately]
- [additional steps if applicable]
**Confidence Note**: [high/moderate/low coverage; note if condition was assumed]

Rules:
- Never recommend home disposal of hazardous materials (batteries, electronics, chemicals).
- If no knowledge base chunks were provided, state "low knowledge base coverage" in Confidence Note.
- Derive item condition from explicit keywords ("broken", "working", etc.); if absent, state assumption.
- Cite only the source titles from the provided chunks; do not fabricate citations.
```

---

### rag/pipeline.py — Orchestrator

Responsibilities:
- Coordinate the full request/response cycle
- Invoke `Retriever.retrieve()` and handle `RetrieverUnavailableError`
- Perform keyword-based hazard detection on the raw item description
- Assemble the LLM message list (system prompt handled inside `Generator`; pipeline sends user + history)
- Invoke `Generator.generate()` and handle `GeneratorUnavailableError`
- Return a structured result dict to `app.py`

Hazard keyword list (checked via case-insensitive substring match):
```python
HAZARD_KEYWORDS = {
    "battery", "batteries", "e-waste", "electronic", "phone", "laptop",
    "computer", "tv", "monitor", "chemical", "paint", "solvent",
    "fluorescent", "cfl", "led bulb", "mercury", "refrigerant", "aerosol"
}
```

Interface:
```python
@dataclass
class PipelineResult:
    recommendation: str         # full structured markdown from Generator
    is_hazardous: bool          # True if hazard keyword detected
    safety_warning: str | None  # pre-composed warning text, or None
    chunks_used: list[Chunk]    # retrieved chunks (may be empty)
    error: str | None           # non-None if pipeline failed

def run(description: str, history: list[dict]) -> PipelineResult:
    """
    Full pipeline: retrieve → hazard check → generate.
    history is a list of the last min(len(history), 5) exchanges.
    """
```

Pipeline execution logic:
```
1. try: chunks = retriever.retrieve(description)
   except RetrieverUnavailableError: return PipelineResult(error="Retrieval service unavailable...")

2. is_hazardous = any(kw in description.lower() for kw in HAZARD_KEYWORDS)
   safety_warning = SAFETY_WARNING_TEXT if is_hazardous else None

3. context_text = format_chunks_as_context(chunks)  # "Source: {title}\n{text}\n---\n"
   user_message = f"Item description: {description}\n\nKnowledge base context:\n{context_text}"

4. messages = [{"role": "user" if e["role"]=="user" else "assistant", "content": e["content"]} 
               for e in history[-5:]]
   messages.append({"role": "user", "content": user_message})

5. try: recommendation = generator.generate(messages)
   except GeneratorUnavailableError: return PipelineResult(error="Request could not be completed...")

6. return PipelineResult(recommendation=recommendation, is_hazardous=is_hazardous,
                         safety_warning=safety_warning, chunks_used=chunks, error=None)
```

---

### ingest.py — Knowledge Base Ingestion Script

Responsibilities:
- Read all `.pdf` and `.txt` files from a configurable `DOCS_FOLDER` path
- Parse PDFs using `PyPDF2`, plain text using standard file read
- Split each document into chunks of 300–500 tokens with 50-token overlap using LangChain's `RecursiveCharacterTextSplitter`
- Embed chunks using the same `all-MiniLM-L6-v2` model
- Store chunks in ChromaDB at `CHROMA_PATH` with metadata: `source_file` (filename), `source_title` (from configurable title mapping or derived from filename)
- Log errors for unparseable files and continue; print summary on completion

```python
# Configuration (top of ingest.py)
DOCS_FOLDER = "data/docs"
CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "knowledge_base"
CHUNK_SIZE_TOKENS = 400       # target; splitter uses chars, calibrated to ~400 tokens
CHUNK_OVERLAP_TOKENS = 50
TITLE_MAPPING = {}            # optional: {"filename.pdf": "Human Readable Title"}
```

---

## Data Models

### Chunk

```python
@dataclass
class Chunk:
    text: str           # 300–500 token text segment
    source_title: str   # human-readable document title (non-empty)
    source_file: str    # filename or URL (non-empty)
    similarity: float   # cosine similarity [0.0, 1.0]; only set on retrieval
```

### PipelineResult

```python
@dataclass
class PipelineResult:
    recommendation: str         # structured markdown string
    is_hazardous: bool
    safety_warning: str | None  # pre-composed safety text
    chunks_used: list[Chunk]
    error: str | None           # if non-None, display error to user instead of recommendation
```

### Conversation History Entry

```python
# Stored in st.session_state.history as list of dicts:
{
    "role": "user" | "assistant",
    "content": str
}
```

### ChromaDB Document Metadata Schema

Every document stored in ChromaDB carries:
```json
{
  "source_file": "circular_economy_guide.pdf",
  "source_title": "Ellen MacArthur Foundation Circular Economy Guide"
}
```

Both fields are required non-empty strings. The ingestion script enforces this at write time.

### Recommendation Markdown Structure

The LLM is instructed to return a response in the following exact structure:

```markdown
**Recommended Pathway**: Repair

**Reasoning**: The item is described as non-functional with a broken screen, making Reuse As-Is
unsuitable. Repurposing is possible but Repair is attempted first given it restores full value...

**Source Citations**:
1. iFixit Repair Guide for Consumer Electronics
2. Ellen MacArthur Foundation Circular Economy Principles

**Next Steps**:
- Find a local electronics repair shop or check manufacturer warranty
- If repair is uneconomical, consider repurposing the case/components

**Confidence Note**: Moderate knowledge base coverage. Condition assessed from explicit keyword
"broken screen" in description.
```

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Input Validation Rejects Short Strings

*For any* string of length less than 5 characters, the input validation function SHALL return an invalid result and SHALL NOT proceed to invoke the RAG pipeline.

**Validates: Requirements 1.2**

---

### Property 2: Input Validation Accepts Valid Strings

*For any* string of length between 5 and 1000 characters (inclusive), the input validation function SHALL return a valid result.

**Validates: Requirements 1.3, 1.4**

---

### Property 3: Character Limit Enforcement

*For any* string of length greater than 1000 characters, the accepted/processed input length SHALL be exactly 1000 characters.

**Validates: Requirements 1.5**

---

### Property 4: Embedding Produces Fixed-Dimension Vector

*For any* valid item description string (length 1–500 characters), the sentence-transformers embedding function SHALL return a non-null, non-empty float vector of exactly 384 dimensions.

**Validates: Requirements 2.1**

---

### Property 5: Retrieved Chunks Meet Similarity Threshold

*For any* query against the knowledge base, every chunk returned by the retriever SHALL have a cosine similarity score of 0.50 or above.

**Validates: Requirements 2.2**

---

### Property 6: Metadata Integrity Over All Chunks

*For any* chunk returned by the retriever or stored by the ingestion script, both `source_title` and `source_file` metadata fields SHALL be non-empty strings.

**Validates: Requirements 2.3, 2.6, 8.3**

---

### Property 7: Recommendation Contains All Five Required Sections

*For any* item description and any set of retrieved chunks (including empty), the generator output SHALL contain all five required section headers: `Recommended Pathway`, `Reasoning`, `Source Citations`, `Next Steps`, and `Confidence Note`.

**Validates: Requirements 3.1**

---

### Property 8: Recommended Pathway Appears First

*For any* recommendation output, the character index of the `Recommended Pathway` section header SHALL be less than the character index of every other section header (`Reasoning`, `Source Citations`, `Next Steps`, `Confidence Note`).

**Validates: Requirements 3.2**

---

### Property 9: Citation Count Matches Retrieved Chunk Count

*For any* retrieval result returning N chunks with cosine similarity ≥ 0.50, the recommendation's `Source Citations` section SHALL contain exactly N numbered citation entries.

**Validates: Requirements 3.3**

---

### Property 10: Hazard Detection Coverage

*For any* item description containing at least one keyword from the defined hazard keyword list, the pipeline result SHALL have `is_hazardous = True` and SHALL include a non-null `safety_warning` string.

**Validates: Requirements 5.1**

---

### Property 11: Safety Warning Contains Required Content

*For any* pipeline result where `is_hazardous = True`, the `safety_warning` text SHALL contain both (a) a specialist disposal statement and (b) at least one specific guidance point referencing a safe disposal channel.

**Validates: Requirements 5.2**

---

### Property 12: Conversation Context Windowing

*For any* conversation history of length N, the message list passed to the generator SHALL contain exactly `min(N, 5)` prior exchanges.

**Validates: Requirements 6.2, 6.5**

---

### Property 13: Conversation History Ordering

*For any* sequence of messages submitted by the user, the messages stored in `st.session_state.history` SHALL be in the same chronological order as their submission.

**Validates: Requirements 6.4**

---

### Property 14: Hierarchy Monotonicity — Broken Items

*For any* item description containing explicit condition indicators of non-functionality (e.g., "broken", "non-functional", "cracked", "damaged"), the recommended pathway SHALL NOT be `Reuse As-Is`.

**Validates: Requirements 4.1, 4.4**

---

### Property 15: No Fabricated Citations on Zero Retrieval

*For any* pipeline invocation where the retriever returns zero chunks above the similarity threshold, the recommendation SHALL NOT contain any numbered entries under `Source Citations` that reference a document not provided as context.

**Validates: Requirements 3.3, 2.4**

---

### Property 16: Ingestion Chunk Size Invariant

*For any* document processed by the ingestion script, all produced chunks (except the final chunk of the document) SHALL have a token count in the range [300, 500].

**Validates: Requirements 8.2**

---

### Property 17: Ingestion Error Resilience

*For any* set of input files containing a mix of valid documents and unparseable files, the ingestion script SHALL successfully process all valid documents and SHALL NOT raise an unhandled exception due to the unparseable files.

**Validates: Requirements 8.4**

---

## Error Handling

### Retriever Unavailable

Condition: `sentence-transformers` model fails to load or raises an exception during embedding.

Handling: `retriever.py` raises `RetrieverUnavailableError`. `pipeline.py` catches this, returns a `PipelineResult` with `error="The retrieval service is currently unavailable. Please try again shortly."`. `app.py` displays this as a Streamlit error message.

### Zero Chunks Retrieved (Above Threshold)

Condition: No documents in ChromaDB have cosine similarity ≥ 0.50 for the given query.

Handling: `retriever.py` returns an empty list. `pipeline.py` proceeds to `generator.py` with empty context. System prompt instructs the LLM to include `"low knowledge base coverage"` in the Confidence Note and produce a recommendation based solely on the item description.

### Fewer Than 3 Chunks Retrieved

Condition: 1 or 2 chunks meet the threshold.

Handling: `pipeline.py` proceeds with available chunks. System prompt instructs the LLM to include `"limited reference material found"` in the Confidence Note.

### Generator Timeout / API Error

Condition: LLM API call fails or does not respond within 30 seconds.

Handling: `generator.py` raises `GeneratorUnavailableError`. `pipeline.py` returns a `PipelineResult` with `error="The request could not be completed. Please resubmit your item description."`. `app.py` displays the error message.

### Knowledge Base Not Ingested

Condition: `data/chroma_db/` does not exist or is empty when the app starts.

Handling: `retriever.py` detects an empty or missing collection and returns an empty list (same as zero chunks case). The app continues to function but all recommendations will have low confidence. A startup warning is printed to the console: `"WARNING: ChromaDB collection is empty. Run ingest.py to populate the knowledge base."`

### Ingestion Parse Errors

Condition: A file in `data/docs/` is a corrupted PDF or unsupported file type.

Handling: `ingest.py` wraps each file's processing in a try/except block. On failure, it logs: `"ERROR: Could not process {filename}: {reason}"` and continues to the next file. The final summary line always prints regardless of errors.

---

## Testing Strategy

### Unit Tests

Unit tests are written using `pytest`. They focus on specific examples, edge cases, and error conditions for pure-logic components.

**`tests/test_validation.py`** — Input validation logic
- `test_empty_string_rejected` — empty string fails validation
- `test_four_chars_rejected` — 4-char string fails
- `test_five_chars_accepted` — 5-char string passes
- `test_thousand_chars_accepted` — 1000-char string passes
- `test_over_limit_truncated` — string > 1000 chars is clamped to 1000

**`tests/test_pipeline.py`** — Pipeline orchestration with mocked retriever and generator
- `test_hazard_keyword_sets_flag` — "battery" in description → `is_hazardous=True`
- `test_non_hazard_clears_flag` — no keywords → `is_hazardous=False`
- `test_retriever_error_returns_error_result` — mocked `RetrieverUnavailableError` → error in result
- `test_generator_error_returns_error_result` — mocked `GeneratorUnavailableError` → error in result
- `test_zero_chunks_proceeds_to_generator` — empty retrieval → generator called with empty context
- `test_history_truncated_to_five` — 8-exchange history → only 5 passed to generator
- `test_condition_assumption_note` — description with no condition indicator → Confidence Note contains assumption language (integration-style example)

**`tests/test_retriever.py`** — Retriever with mocked ChromaDB
- `test_returns_only_above_threshold` — mock returns 3 results, 2 above threshold → only 2 returned
- `test_returns_empty_list_on_no_results` — mock returns empty → empty list returned
- `test_metadata_non_empty_on_results` — all returned chunks have non-empty metadata

**`tests/test_generator.py`** — Generator with mocked API
- `test_structured_output_contains_all_sections` — mock LLM response has all 5 headers
- `test_pathway_appears_before_confidence_note` — ordering in mock response

**`tests/test_ingest.py`** — Ingestion script
- `test_summary_line_printed` — verify console output on completion
- `test_bad_file_skipped` — place a corrupted PDF; verify script completes and logs error

### Property-Based Tests

Property tests use [**Hypothesis**](https://hypothesis.readthedocs.io/) for Python. Each test is configured to run a minimum of 100 iterations (`settings(max_examples=100)`).

**`tests/test_properties.py`**

```python
# Feature: repurpose-ai, Property 1: Input validation rejects short strings
# Feature: repurpose-ai, Property 2: Input validation accepts valid strings
# Feature: repurpose-ai, Property 3: Character limit enforcement
# Feature: repurpose-ai, Property 5: Retrieved chunks meet similarity threshold
# Feature: repurpose-ai, Property 6: Metadata integrity over all chunks
# Feature: repurpose-ai, Property 10: Hazard detection coverage
# Feature: repurpose-ai, Property 12: Conversation context windowing
# Feature: repurpose-ai, Property 13: Conversation history ordering
# Feature: repurpose-ai, Property 14: Hierarchy monotonicity — broken items
# Feature: repurpose-ai, Property 16: Ingestion chunk size invariant
# Feature: repurpose-ai, Property 17: Ingestion error resilience
```

Properties 7, 8, 9, 11, 15 (generator output structure) are tested by generating random item descriptions and chunk sets, calling the generator with a mocked or sandboxed LLM call, and asserting structural properties of the output string.

Properties 4 (embedding dimension) and 6 (metadata integrity) require the sentence-transformers model and ChromaDB to be available; they are tagged `@pytest.mark.integration` and excluded from the fast unit test run.

### Integration Tests

Integration tests are tagged `@pytest.mark.integration` and run separately (require live ChromaDB + LLM API):

- `test_full_pipeline_returns_recommendation` — end-to-end: real description, real retrieval, real LLM call; verify all 5 sections present and response within 30 seconds
- `test_hazardous_item_end_to_end` — "old laptop battery" → Safety Warning present, Responsible Disposal in Next Steps
- `test_knowledge_base_has_minimum_documents` — count documents in ChromaDB collection ≥ 30

### Smoke Tests

- `test_app_launches` — `streamlit run app.py` responds at localhost:8501 within 10 seconds
- `test_knowledge_base_populated` — `data/chroma_db/` directory exists and is non-empty after running `ingest.py`

### Test Run Commands

```bash
# Fast tests (unit + property, no external dependencies)
pytest tests/ -m "not integration" --tb=short

# All tests including integration (requires API keys and populated KB)
pytest tests/ --tb=short

# Property tests only (Hypothesis)
pytest tests/test_properties.py -v
```
