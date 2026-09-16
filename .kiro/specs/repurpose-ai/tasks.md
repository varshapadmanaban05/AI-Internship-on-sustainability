# Implementation Plan: RePurpose AI

## Overview

Implement a conversational, RAG-powered circular economy advisor as a Streamlit web application. The build follows a bottom-up pipeline order: project scaffolding → ingestion script → retriever → generator → pipeline orchestrator → Streamlit UI → tests.

## Tasks

- [x] 1. Set up project structure and configuration files
  - Create the following directory layout at the project root:
    ```
    data/docs/          # drop PDF/TXT knowledge base files here
    data/chroma_db/     # auto-created by ingest.py
    rag/
    tests/
    ```
  - Create `rag/__init__.py` (empty package marker)
  - Create `tests/__init__.py` (empty package marker)
  - Create `requirements.txt` with pinned versions:
    ```
    streamlit==1.35.0
    langchain==0.2.6
    langchain-community==0.2.6
    chromadb==0.5.3
    sentence-transformers==3.0.1
    openai==1.35.0
    PyPDF2==3.0.1
    python-dotenv==1.0.1
    hypothesis==6.103.1
    pytest==8.2.2
    pytest-mock==3.14.0
    ```
  - Create `.env.example`:
    ```
    OPENAI_API_KEY=your_openai_api_key_here
    OPENAI_MODEL=gpt-3.5-turbo
    CHROMA_PATH=data/chroma_db
    DOCS_FOLDER=data/docs
    ```
  - Create `.gitignore` entries for `.env`, `data/chroma_db/`, `__pycache__/`, `.pytest_cache/`
  - _Requirements: 7.1, 8.1_

- [x] 2. Implement the knowledge base ingestion script (`ingest.py`)
  - [x] 2.1 Implement `ingest.py` document loading and chunking
    - At the top of `ingest.py`, define configuration constants:
      ```python
      DOCS_FOLDER = os.getenv("DOCS_FOLDER", "data/docs")
      CHROMA_PATH = os.getenv("CHROMA_PATH", "data/chroma_db")
      COLLECTION_NAME = "knowledge_base"
      CHUNK_SIZE_TOKENS = 400   # ~1600 chars
      CHUNK_OVERLAP_TOKENS = 50  # ~200 chars
      TITLE_MAPPING = {}  # optional override: {"filename.pdf": "Human Title"}
      ```
    - Implement `load_documents(docs_folder: str) -> list[tuple[str, str, str]]` that walks the folder, reads `.pdf` files via `PyPDF2.PdfReader` and `.txt` files via `open()`, returns list of `(text, source_file, source_title)` tuples; wraps each file in try/except and logs `ERROR: Could not process {filename}: {reason}` on failure
    - Derive `source_title` from `TITLE_MAPPING.get(filename, filename.replace("_", " ").replace("-", " ").removesuffix(".pdf").removesuffix(".txt").title())`
    - Implement `chunk_documents(docs)` using `langchain.text_splitter.RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)` to split each document's text; attach metadata `{"source_file": ..., "source_title": ...}` to each chunk
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

  - [x] 2.2 Implement `ingest.py` embedding and ChromaDB storage
    - Implement `store_chunks(chunks, chroma_path, collection_name)` that initialises a `chromadb.PersistentClient(path=chroma_path)`, creates or gets a collection with `metadata={"hnsw:space": "cosine"}`, embeds all chunk texts using `sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")`, and calls `collection.add()` with unique IDs, embeddings, documents, and metadata
    - Enforce that `source_title` and `source_file` metadata fields are non-empty strings before calling `collection.add()`; raise `ValueError` if either is empty
    - Implement `main()` that calls `load_documents`, `chunk_documents`, `store_chunks`, and prints the summary line: `"Ingestion complete: {n_docs} documents processed, {n_chunks} chunks stored."`
    - _Requirements: 8.2, 8.3, 8.5_

  - [ ]* 2.3 Write property test for ingestion chunk size invariant
    - **Property 16: Ingestion Chunk Size Invariant**
    - **Validates: Requirements 8.2**
    - In `tests/test_properties.py`, using `@given(st.text(min_size=2000))` generate long text strings, run them through `RecursiveCharacterTextSplitter`, and assert every chunk except the last has a token count ≤ 500 (use `len(chunk.split())` as a proxy)

  - [ ]* 2.4 Write property test for ingestion error resilience
    - **Property 17: Ingestion Error Resilience**
    - **Validates: Requirements 8.4**
    - In `tests/test_properties.py`, using `@given(st.lists(st.binary()))` generate a mix of arbitrary byte sequences as fake file contents; mock `open()` so some raise `Exception`; assert `load_documents` never raises an unhandled exception and the returned list contains only the successfully parsed entries

- [x] 3. Implement the retrieval engine (`rag/retriever.py`)
  - [x] 3.1 Define `Chunk` dataclass and `RetrieverUnavailableError` in `rag/retriever.py`
    - ```python
      from dataclasses import dataclass

      @dataclass
      class Chunk:
          text: str
          source_title: str
          source_file: str
          similarity: float

      class RetrieverUnavailableError(Exception):
          pass
      ```
    - _Requirements: 2.3, 2.6_

  - [x] 3.2 Implement `Retriever` class with `retrieve()` method
    - Constructor loads `SentenceTransformer("all-MiniLM-L6-v2")` (cached on first call) and opens a `chromadb.PersistentClient` at `chroma_path`; wraps model load in try/except, raises `RetrieverUnavailableError` on failure
    - `retrieve(query: str, top_k: int = 5) -> list[Chunk]`: embeds query, queries the collection with `n_results=top_k`, filters results to those with `distance <= 0.50` (cosine distance, equivalent to similarity ≥ 0.50), converts each result to a `Chunk` with `similarity = 1 - distance`, returns the filtered list (empty list if none pass threshold)
    - Print console warning `"WARNING: ChromaDB collection is empty. Run ingest.py to populate the knowledge base."` if the collection has zero documents
    - _Requirements: 2.1, 2.2, 2.3, 2.6, 2.7_

  - [ ]* 3.3 Write property test for retrieved chunks similarity threshold
    - **Property 5: Retrieved Chunks Meet Similarity Threshold**
    - **Validates: Requirements 2.2**
    - In `tests/test_properties.py`, mock ChromaDB to return arbitrary distances in `[0.0, 2.0]`; using `@given(st.lists(st.floats(min_value=0.0, max_value=2.0), min_size=1, max_size=10))` generate distance lists; assert every `Chunk` in the return has `similarity >= 0.50` (i.e., distance ≤ 0.50 filter is applied)

  - [ ]* 3.4 Write property test for metadata integrity
    - **Property 6: Metadata Integrity Over All Chunks**
    - **Validates: Requirements 2.3, 2.6, 8.3**
    - In `tests/test_properties.py`, using `@given(st.lists(st.fixed_dictionaries({"source_title": st.text(min_size=1), "source_file": st.text(min_size=1), "distance": st.floats(0.0, 0.5)})))` generate mock ChromaDB result metadata; assert all returned `Chunk` objects have non-empty `source_title` and `source_file`

- [x] 4. Implement the LLM generator (`rag/generator.py`)
  - [x] 4.1 Define `GeneratorUnavailableError` and system prompt constant in `rag/generator.py`
    - Define `SYSTEM_PROMPT` as the exact string specified in the design (the RePurpose AI persona prompt with the 7-level hierarchy, structured output sections, and citation/confidence rules)
    - ```python
      class GeneratorUnavailableError(Exception):
          pass
      ```
    - _Requirements: 3.1, 4.1_

  - [x] 4.2 Implement `Generator` class with `generate()` method
    - Constructor: `__init__(self, model: str = None, timeout: int = 30)` reads `model` from `os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")` if not provided; initialises `openai.OpenAI()` client (reads `OPENAI_API_KEY` from environment via `python-dotenv`)
    - `generate(self, messages: list[dict]) -> str`: prepends `{"role": "system", "content": SYSTEM_PROMPT}` to the messages list, calls `self.client.chat.completions.create(model=..., messages=..., timeout=self.timeout)`, returns `response.choices[0].message.content`; wraps call in try/except `(openai.APIError, openai.Timeout, Exception)` and raises `GeneratorUnavailableError` on any failure
    - _Requirements: 3.1, 3.6, 3.7_

- [ ] 5. Implement the pipeline orchestrator (`rag/pipeline.py`)
  - [x] 5.1 Define `PipelineResult` dataclass and `HAZARD_KEYWORDS` set in `rag/pipeline.py`
    - ```python
      from dataclasses import dataclass, field
      from rag.retriever import Chunk

      HAZARD_KEYWORDS = {
          "battery", "batteries", "e-waste", "electronic", "phone", "laptop",
          "computer", "tv", "monitor", "chemical", "paint", "solvent",
          "fluorescent", "cfl", "led bulb", "mercury", "refrigerant", "aerosol"
      }

      SAFETY_WARNING_TEXT = (
          "⚠️ **Safety Warning**: This item requires specialist disposal. "
          "Do not place in general waste — take to a certified e-waste collection point "
          "or manufacturer take-back scheme. Contact your local council for chemical/hazardous waste drop-off."
      )

      @dataclass
      class PipelineResult:
          recommendation: str = ""
          is_hazardous: bool = False
          safety_warning: str | None = None
          chunks_used: list[Chunk] = field(default_factory=list)
          error: str | None = None
      ```
    - _Requirements: 5.1, 5.2_

  - [ ] 5.2 Implement `run()` function in `rag/pipeline.py`
    - Implement `format_chunks_as_context(chunks: list[Chunk]) -> str` that joins chunks as `"Source: {title}\n{text}\n---\n"`
    - Implement `run(description: str, history: list[dict]) -> PipelineResult` following the exact 6-step logic from the design:
      1. Call `retriever.retrieve(description)`; on `RetrieverUnavailableError` return error result
      2. Hazard check: `is_hazardous = any(kw in description.lower() for kw in HAZARD_KEYWORDS)`
      3. Format chunks as context and assemble `user_message`
      4. Build `messages` from `history[-5:]` + new user message
      5. Call `generator.generate(messages)`; on `GeneratorUnavailableError` return error result
      6. Return full `PipelineResult`
    - Instantiate `Retriever` and `Generator` as module-level singletons (lazy init acceptable)
    - _Requirements: 2.1–2.7, 3.1–3.8, 4.1–4.5, 5.1–5.4, 6.2, 6.5_

  - [ ]* 5.3 Write property test for hazard detection coverage
    - **Property 10: Hazard Detection Coverage**
    - **Validates: Requirements 5.1**
    - In `tests/test_properties.py`, using `@given(st.sampled_from(list(HAZARD_KEYWORDS)), st.text())` construct descriptions containing a hazard keyword; mock retriever and generator; assert `result.is_hazardous == True` and `result.safety_warning is not None`

  - [ ]* 5.4 Write property test for conversation context windowing
    - **Property 12: Conversation Context Windowing**
    - **Validates: Requirements 6.2, 6.5**
    - In `tests/test_properties.py`, using `@given(st.lists(st.fixed_dictionaries({"role": st.sampled_from(["user", "assistant"]), "content": st.text()}), min_size=0, max_size=20))` generate history lists of varying lengths; capture the `messages` argument passed to `generator.generate()`; assert the number of prior-history messages equals `min(len(history), 5)`

- [ ] 6. Checkpoint — Core pipeline complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Implement the Streamlit application (`app.py`)
  - [ ] 7.1 Implement input validation and session state initialisation in `app.py`
    - Load environment with `python-dotenv` at module top
    - Initialise `st.session_state.history` (list) and `st.session_state.processing` (bool, False) if not already set
    - Implement `validate_input(text: str) -> tuple[bool, str]`:
      - Returns `(False, "Please provide a more detailed description (at least 5 characters).")` if `len(text) < 5`
      - Returns `(True, "")` if `5 <= len(text) <= 1000`
      - Truncates to 1000 chars before validation if longer (so `validate_input` always sees ≤ 1000)
    - In the chat input widget, set `max_chars=1000`; display a live character counter beneath the input showing `{len(text)}/1000`
    - _Requirements: 1.1–1.6_

  - [ ]* 7.2 Write property test for input validation — short strings rejected
    - **Property 1: Input Validation Rejects Short Strings**
    - **Validates: Requirements 1.2**
    - In `tests/test_properties.py`, using `@given(st.text(max_size=4))` assert `validate_input(text)[0] == False`

  - [ ]* 7.3 Write property test for input validation — valid strings accepted
    - **Property 2: Input Validation Accepts Valid Strings**
    - **Validates: Requirements 1.3, 1.4**
    - In `tests/test_properties.py`, using `@given(st.text(min_size=5, max_size=1000))` assert `validate_input(text)[0] == True`

  - [ ]* 7.4 Write property test for character limit enforcement
    - **Property 3: Character Limit Enforcement**
    - **Validates: Requirements 1.5**
    - In `tests/test_properties.py`, using `@given(st.text(min_size=1001))` assert that the text processed by the app is exactly 1000 characters (test the truncation logic directly, not via `validate_input`)

  - [ ] 7.5 Implement `render_recommendation()` and main chat UI in `app.py`
    - Render page header: `st.title("♻️ RePurpose AI")`, a `st.markdown` description (≤ 100 words), and a short usage instruction
    - Implement `render_recommendation(result: PipelineResult) -> None`:
      - If `result.error` is non-None, call `st.error(result.error)` and return
      - If `result.is_hazardous`, render `result.safety_warning` in a `st.warning()` block before the recommendation body
      - Render `result.recommendation` using `st.markdown()`
    - Implement `run_pipeline_with_spinner(description: str, history: list) -> PipelineResult` that sets `st.session_state.processing = True`, calls `pipeline.run()` inside a `with st.spinner("Thinking…"):` block, sets `processing = False`, and returns the result
    - Render the scrollable chat log from `st.session_state.history` using `st.chat_message("user")` / `st.chat_message("assistant")` containers
    - Disable the submit button when `st.session_state.processing == True`
    - _Requirements: 3.2, 3.5, 5.1, 5.4, 6.1, 6.4, 7.2, 7.4, 7.5_

  - [ ] 7.6 Implement "Clear Conversation" button and history appending in `app.py`
    - Add a `st.button("Clear Conversation")` in the sidebar or above the chat log; on click, set `st.session_state.history = []` and call `st.rerun()`
    - After a successful pipeline run, append `{"role": "user", "content": description}` and `{"role": "assistant", "content": result.recommendation}` to `st.session_state.history`
    - Pass only `st.session_state.history[-5:]` to `pipeline.run()` (full history kept in session state for UI display)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ]* 7.7 Write property test for conversation history ordering
    - **Property 13: Conversation History Ordering**
    - **Validates: Requirements 6.4**
    - In `tests/test_properties.py`, using `@given(st.lists(st.text(min_size=1), min_size=1, max_size=10))` simulate submitting N messages in sequence; assert that `st.session_state.history` entries appear in the same order as submission (test the append logic in isolation, mocking Streamlit session state as a plain list)

- [ ] 8. Write unit tests
  - [ ] 8.1 Write unit tests for input validation (`tests/test_validation.py`)
    - `test_empty_string_rejected` — `validate_input("")` returns `(False, ...)`
    - `test_four_chars_rejected` — `validate_input("abcd")` returns `(False, ...)`
    - `test_five_chars_accepted` — `validate_input("hello")` returns `(True, "")`
    - `test_thousand_chars_accepted` — `validate_input("a" * 1000)` returns `(True, "")`
    - `test_over_limit_string_is_handled` — a 1001-char string passed through the truncation path yields a 1000-char string
    - _Requirements: 1.2, 1.3, 1.4, 1.5_

  - [ ] 8.2 Write unit tests for pipeline orchestration (`tests/test_pipeline.py`)
    - `test_hazard_keyword_sets_flag` — mock retriever + generator; "battery" in description → `is_hazardous=True`, `safety_warning` is not None
    - `test_non_hazard_clears_flag` — "old wooden chair" → `is_hazardous=False`, `safety_warning` is None
    - `test_retriever_error_returns_error_result` — mock `RetrieverUnavailableError` → `result.error` is non-None, `result.recommendation == ""`
    - `test_generator_error_returns_error_result` — mock `GeneratorUnavailableError` → `result.error` is non-None
    - `test_zero_chunks_proceeds_to_generator` — retriever returns `[]`; assert generator is still called
    - `test_history_truncated_to_five` — pass 8-entry history; capture `messages` arg to generator mock; assert prior-history entries count == 5
    - _Requirements: 2.4, 2.7, 3.7, 5.1, 6.2, 6.5_

  - [ ] 8.3 Write unit tests for the retriever (`tests/test_retriever.py`)
    - `test_returns_only_above_threshold` — mock ChromaDB returning distances `[0.3, 0.6, 0.8]`; assert only the 2 chunks with distance ≤ 0.50 are returned
    - `test_returns_empty_list_on_no_results` — mock returns empty lists; assert `retrieve()` returns `[]`
    - `test_metadata_non_empty_on_results` — mock returns results with non-empty metadata; assert all returned `Chunk` objects have non-empty `source_title` and `source_file`
    - _Requirements: 2.2, 2.3, 2.6_

  - [ ] 8.4 Write unit tests for the generator (`tests/test_generator.py`)
    - `test_structured_output_contains_all_sections` — mock OpenAI response containing all 5 section headers; assert all headers present in returned string
    - `test_pathway_appears_before_confidence_note` — in mock response, assert index of `"Recommended Pathway"` < index of `"Confidence Note"`
    - `test_api_error_raises_generator_unavailable` — mock `openai.APIError`; assert `GeneratorUnavailableError` is raised
    - _Requirements: 3.1, 3.2, 3.7_

  - [ ] 8.5 Write unit tests for the ingestion script (`tests/test_ingest.py`)
    - `test_summary_line_printed` — run `ingest.main()` with a mocked folder containing 2 valid TXT files; capture stdout; assert summary line matches `"Ingestion complete: 2 documents processed, N chunks stored."`
    - `test_bad_file_skipped` — place one valid TXT and one file that raises `Exception` on read; assert `main()` completes without raising and logs an error containing the bad filename
    - _Requirements: 8.4, 8.5_

- [ ] 9. Checkpoint — All tests pass
  - Run `pytest tests/ -m "not integration" --tb=short` and ensure all tests pass. Ask the user if any questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Run `python ingest.py` before launching the app — the ChromaDB store must be populated first
- Launch the app with `streamlit run app.py`
- Fast test suite (unit + property, no live APIs): `pytest tests/ -m "not integration" --tb=short`
- Full test suite (requires `OPENAI_API_KEY` and populated ChromaDB): `pytest tests/ --tb=short`
- Each task references specific requirements for traceability
- Property tests use Hypothesis with `@settings(max_examples=100)`

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["2.1", "3.1", "4.1"] },
    { "id": 2, "tasks": ["2.2", "3.2", "4.2"] },
    { "id": 3, "tasks": ["2.3", "2.4", "3.3", "3.4", "5.1"] },
    { "id": 4, "tasks": ["5.2"] },
    { "id": 5, "tasks": ["5.3", "5.4", "7.1"] },
    { "id": 6, "tasks": ["7.2", "7.3", "7.4", "7.5"] },
    { "id": 7, "tasks": ["7.6"] },
    { "id": 8, "tasks": ["7.7", "8.1", "8.2", "8.3", "8.4", "8.5"] }
  ]
}
```
