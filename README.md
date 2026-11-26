# ForgeAI — The Agent That Builds Agents

ForgeAI is a domain-independent agent factory. Given an unseen business brief and a folder of source documents (`.pdf`, `.csv`, `.txt`, `.md`), it orchestrates requirements intake, corpus groundability verification, declarative agent synthesis, automated 20+ case evaluation generation, hybrid judging, targeted Critic–Optimizer iterations (maximum 3), and honest evidence-backed reporting.

---

## Architecture & Workflow

```text
Unseen Brief + Document Folder
   │
   ▼
[ingest_documents] ──► PDF/CSV/TXT/MD Loaders ──► Chunker ──► Local SentenceTransformer ──► ChromaDB
   │
   ▼
[intake_interview] (<= 5 clarifying questions; explicit unattended policy) ──► StructuredRequirements
   │
   ▼
[groundability_guard] ──(unsupported & unapproved)──► [build_report]
   │ (supported or requester-approved gaps)
   ▼
[generate_agent] ──► System Prompt + Retrieval/Chunking + Least-Privilege ToolRegistry + OutputContract
   │
   ▼
[generate_evals] (>= 20 grounded cases; duplicate & fabricated-source rejection)
   │
   ▼
[run_and_judge] ◄───(iter < 3 & below target)─── [optimize_agent] (SHA-256 suite lock + regression guard)
   │ (target met, no actionable proposal, or iter == 3)
   ▼
[build_report] ──► FactoryRunReport + IterationScorecards + Persisted Artifacts
```

### Technology Stack

- **LLM Provider:** Google Gemini via official `google-genai` SDK (`GEMINI_API_KEY`, `GEMINI_MODEL` via `.env`)
- **Orchestration:** LangGraph (`StateGraph` with explicit conditional edges and 0–3 iteration budget)
- **Vector Store & Embeddings:** ChromaDB + local Sentence Transformers (`all-MiniLM-L6-v2`, no paid embedding API)
- **Schemas & Validation:** Pydantic v2 & `pydantic-settings`
- **Document Parsing:** `pypdf` (PDF), stdlib `csv` (1 row per `DocumentUnit` by default, configurable), UTF-8 text/Markdown
- **UI & CLI:** Streamlit (`app.py`) and generic CLI (`python -m forgeai.cli`)
- **Testing:** Pytest (deterministic mocked unit tests + isolated `@pytest.mark.live_gemini` connectivity test)

---

## Environment Setup & Gemini Configuration (Windows PowerShell)

1. **Activate the existing virtual environment:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   *(If setting up from scratch on Python 3.11+: `python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -e ".[dev]"`)*

2. **Configure `.env`:**
   ```powershell
   Copy-Item .env.example .env
   ```
   Set your key and model in `.env` (never commit `.env`):
   ```env
   GEMINI_API_KEY=your_real_key_here
   GEMINI_MODEL=gemini-3.8-flash
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   CHROMA_PERSIST_DIR=.forgeai/chroma
   ARTIFACTS_DIR=artifacts/runs
   ```

---

## Running Tests, Streamlit & Unseen Briefs

### 1. Run the Deterministic Test Suite
Ordinary `pytest` runs use deterministic mocked LLM responses and do **not** require live Gemini API availability or consume tokens:
```powershell
.\.venv\Scripts\pytest.exe -v
```

### 2. Run Explicit Live Gemini Connectivity Verification
```powershell
$env:FORGEAI_RUN_LIVE_TESTS="1"; .\.venv\Scripts\pytest.exe -m live_gemini -v
```

### 3. Launch the Streamlit Application
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

### 4. Run the Generic Factory CLI on an Unseen Brief & Document Folder
```powershell
.\.venv\Scripts\python.exe -m forgeai.cli run `
  --brief path\to\brief.txt `
  --docs path\to\documents_folder `
  --out artifacts\runs\my_unseen_run `
  --unattended-policy proceed_with_assumptions `
  --max-iterations 3 `
  --min-eval-cases 20
```
- `--unattended-policy stop_for_clarification` pauses and outputs clarifying questions without inventing answers.
- `--unattended-policy proceed_with_assumptions` records unanswered questions transparently in `unattended_assumptions.json` and `requirements.json`.

### 5. Re-run Demonstrations
- **Deterministic / offline-LLM pipeline mode** (used when live Gemini returns `503 UNAVAILABLE`; runs real PDF/CSV ingestion, `all-MiniLM-L6-v2` embeddings, ChromaDB, `DeclarativeAgentRunner`, `HybridJudge`, and `CriticOptimizer`):
  ```powershell
  .\.venv\Scripts\python.exe scripts/run_demos.py --mode deterministic
  ```
- **Live Gemini mode** (once `gemini-3.8-flash` is available):
  ```powershell
  .\.venv\Scripts\python.exe scripts/run_demos.py --mode live
  ```

---

## Demonstrations & Generated Artifacts

- **Synthetic Data Notice:** [demos/SYNTHETIC_DATA_NOTICE.md](demos/SYNTHETIC_DATA_NOTICE.md)
- **Demo 1 (HR Policy Assistant):**
  - Corpus & brief: `demos/hr_policy/` (`brief.txt`, `synthesis_prompt.md`, 2 PDFs + 3 Markdown policies in `documents/`)
  - Committed artifacts: `artifacts/demos/hr_policy/`
  - Selected tools: `retrieve_documents`, `list_corpus_documents`
- **Demo 2 (Ship Spares & Supplier Assistant):**
  - Corpus & brief: `demos/ship_spares/` (`brief.txt`, `synthesis_prompt.md`, 18-row `inventory_catalogue.csv` + 2 supplier PDFs in `documents/`)
  - Committed artifacts: `artifacts/demos/ship_spares/`
  - Selected tools: `retrieve_documents`, `inspect_csv_records`, `calculate_math`, `list_corpus_documents`
- **Assignment Compliance Audit:** [artifacts/FINAL_ASSIGNMENT_AUDIT.md](artifacts/FINAL_ASSIGNMENT_AUDIT.md)

---

## Verification Scope & Known Limitations

- **Deterministic mocked verification:** All unit tests and unseen-brief end-to-end checks pass deterministically with mocked LLM responses, exercising real PDF/CSV parsing, ChromaDB retrieval, hybrid judge deterministic checks, SHA-256 suite immutability, Critic–Optimizer loop routing, and artifact persistence.
- **Live Gemini status:** During verification (`2026-11-26`), `GEMINI_MODEL=gemini-3.8-flash` returned `503 UNAVAILABLE` (*"This model is currently experiencing high demand"*). Therefore, live Gemini LLM responses have **not** been confirmed at runtime, and the committed demonstration artifacts in `artifacts/demos/` explicitly record `"execution_mode": "deterministic_mocked_llm_due_to_live_503"` and `"live_gemini_verified": false`.
- **Dense retrieval on exact SKU IDs:** In `ship_spares`, pure dense vector retrieval (`all-MiniLM-L6-v2`) missed 2 exact-SKU queries (`sh_01`, `sh_12`) even at `top_k=8`, which is recorded honestly in `final_report.json` and underscores why `inspect_csv_records` is selected for exact SKU lookups.
