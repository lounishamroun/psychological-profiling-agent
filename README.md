# ARIA — Adaptive Retrieval-Augmented Interrogation Agent

A multi-agent AI interrogation simulator with RAG-enhanced tactical reasoning and dynamic psychological profiling.

## Overview

ARIA simulates realistic police interrogations using 3 LLM-powered agents orchestrated by LangGraph:

- **Inspector Agent** — Asks strategic questions guided by RAG-retrieved interrogation tactics (PEACE, Reid, SUE)
- **Suspect Agent** — Responds according to a psychological profile (guilty, accomplice, or innocent)
- **Profiler Agent** — Analyzes each exchange, detects inconsistencies, evaluates credibility using behavioral examples

The system runs in a loop: `Retrieve → Inspector → Suspect → Profiler → [loop or report]`.

## Architecture

```
                    ┌──────────────┐
                    │   ChromaDB   │
                    │ (embeddings) │
                    └──┬───────┬───┘
                  tactics    examples
                  (k=3)      (k=3)
                    ▼           ▼
START → Retrieve → Inspector → Suspect → Profiler → [continue?] → Final Report → END
                                                         │
                                                         └── loop back to Retrieve
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Gemini 2.5 Flash |
| Orchestration | LangGraph |
| Vector Store | ChromaDB (ephemeral) |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 (384d) |
| Chunking | langchain-text-splitters (MarkdownHeaderTextSplitter) |
| UI | Streamlit |
| Observability | Langfuse |

## Project Structure

```
src/
├── agents.py       # Inspector, Suspect, Profiler, Final Report agents
├── graph.py        # LangGraph workflow definition
├── rag.py          # ChromaDB index building + retrieval functions
├── prompts.py      # Prompt templates for all agents
├── state.py        # LangGraph shared state schema
├── utils.py        # LLM wrapper (call_llm), helpers
├── data_prep.py    # PDF→TXT→MD pipeline, CSV→JSONL enrichment
data/
├── cases/          # Case files (e.g. case_001.json)
├── suspects/       # Suspect profiles (personality, hidden truth, strategy)
├── rag_docs/
│   ├── raw/        # Source PDFs + CSV
│   └── processed/  # .txt, .md (generated), .jsonl (enriched)
app.py              # Streamlit interface
```

## RAG Pipeline

**Sources:**
- 6 reference documents on interrogation techniques (PEACE, Reid, SUE, UNODC, Mendez, Investigative Interviewing)
- 1 behavioral dataset of 1600 Q/R pairs enriched with PEACE phases and SUE tactics

**Data preparation:**
```
PDF → PyPDF2 → .txt (cleaned) → Gemini → .md (structured)
CSV → heuristic enrichment (PEACE/SUE classification) → .jsonl
```

**Differentiated access:**
- Inspector receives tactical chunks from .md documents
- Profiler receives behavioral examples from .jsonl (filtered via `where={"source": "dataset"}`)
- Suspect receives nothing (preserves information asymmetry)

## Setup

```bash
# Clone
git clone https://github.com/lounishamroun/psychological-profiling-agent.git
cd psychological-profiling-agent

# Install
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Configure
cp .env.exemple .env
# Edit .env with your API keys:
#   GOOGLE_API_KEY=...
#   LANGFUSE_SECRET_KEY=...
#   LANGFUSE_PUBLIC_KEY=...
#   LANGFUSE_BASE_URL=...

# Run
streamlit run app.py
```

## Usage

1. Select a suspect from the sidebar
2. Toggle RAG ON/OFF
3. Set the number of interrogation turns
4. Click "Start Interrogation"
5. View the transcript, profiler analysis, and final report

## Data Preparation (optional)

```bash
# Extract PDFs to .txt
python -m src.data_prep extract

# Generate .md from .txt via Gemini
python -m src.data_prep generate-md

# Transform CSV to enriched JSONL
python -m src.data_prep transform
```

## Key Design Decisions

- **Ephemeral ChromaDB** — Index is small (~1800 chunks), rebuilds in seconds. No persistence needed.
- **Dynamic RAG queries** — Query changes every turn based on last question + profiler analysis. Not static context.
- **No RAG for Suspect** — A real suspect doesn't know the interrogator's playbook.
- **Temperature control** — Low for Profiler (deterministic analysis), high for Suspect (unpredictable behavior).

## Testing

Tests use **pytest** with **pytest-cov** for coverage. All LLM calls are mocked — no API key required.

```bash
# Run all tests with coverage
pytest

# Run a specific test file
pytest tests/test_utils.py -v

# Run with HTML coverage report
pytest --cov-report=html
```

**Test structure:**

| File | What it tests |
|------|--------------|
| `test_utils.py` | JSON parsing (7 edge cases), `format_conversation`, `load_json`, `call_llm` mock, comparison mode |
| `test_agents.py` | All 5 agents return correct state keys, prompt content, malformed output handling |
| `test_graph.py` | `_should_continue` logic (5 cases), graph compilation, node wiring |
| `test_prompts.py` | Placeholder presence, `.format()` success, rubric keywords |
| `test_state.py` | Required fields, `operator.add` on append fields, plain list on overwrite fields |

**Coverage (43 tests):**

```
src/agents.py      100%
src/prompts.py     100%
src/state.py       100%
src/utils.py        84%
src/graph.py        51%
```

## License

MIT
