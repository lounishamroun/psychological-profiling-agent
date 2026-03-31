# Architecture — ARIA

## Overview

ARIA is a multi-agent interrogation simulator. Three LLM-powered agents interact in a loop orchestrated by LangGraph, with a RAG pipeline providing contextual knowledge to the right agents at the right time.

---

## Agent Loop

```
START
  │
  ▼
┌─────────────────┐
│ Retrieve Context │  ← builds a search query from the current state
└────────┬────────┘     injects tactical chunks → Inspector
         │              injects behavioral examples → Profiler
         ▼
┌─────────────────┐
│ Inspector Agent  │  ← asks the next strategic question
└────────┬────────┘     guided by RAG tactical docs
         ▼
┌─────────────────┐
│  Suspect Agent   │  ← answers according to psychological profile
└────────┬────────┘     no RAG access (preserves information asymmetry)
         ▼
┌─────────────────┐
│  Profiler Agent  │  ← analyzes the exchange, evaluates credibility
└────────┬────────┘     guided by RAG behavioral examples
         │
   turn < max_turns?
         │
    YES  │  NO
    ◄────┘   │
             ▼
    ┌──────────────────┐
    │  Final Report     │  ← synthesizes the full interrogation
    └────────┬─────────┘
             ▼
            END
```

---

## Agents

### Inspector Agent
- **Role**: Interrogator. Formulates questions to extract the truth.
- **RAG input**: Top-3 chunks from tactical `.md` documents (PEACE, Reid, SUE, UNODC, Mendez, Investigative Interviewing).
- **Query**: Built from `last_question` + `profiler_output.reason`.
- **Temperature**: 0.7 — strategic but not robotic.

### Suspect Agent
- **Role**: Simulates a suspect (guilty / accomplice / innocent).
- **RAG input**: None. The suspect must not know the interrogator's playbook.
- **Context**: Receives the full conversation history + their own psychological profile (personality, hidden truth, defense strategy).
- **Temperature**: 1.0 — unpredictable and realistic.

### Profiler Agent
- **Role**: Behavioral analyst. Evaluates credibility and detects inconsistencies.
- **RAG input**: Top-3 behavioral examples from the JSONL dataset (filtered: `source="dataset"`).
- **Query**: Built from `last_answer` (the suspect's most recent response).
- **Temperature**: 0.2 — deterministic, fact-based analysis.

### Final Report Agent
- **Role**: Generates a structured debrief of the interrogation.
- **Input**: Full conversation history + all profiler outputs.
- **Temperature**: 0.3 — factual and reproducible.

---

## RAG Pipeline

### Data Sources

| Source | Format | Content | Used by |
|--------|--------|---------|---------|
| 6 reference PDFs | `.md` (generated) | Interrogation tactics (PEACE, Reid, SUE, UNODC, Mendez, Investigative Interviewing) | Inspector |
| Behavioral dataset | `.jsonl` (enriched) | 1600 Q/R pairs annotated with PEACE phase, SUE tactic, veracity | Profiler |

### Data Preparation

```
PDF → PyPDF2 → .txt (regex-cleaned) → Gemini 2.5 Flash → .md (structured)
CSV → heuristic enrichment (_classify_phase_peace, _classify_tactique_sue) → .jsonl
```

### Indexing (`build_index()`)

1. Load `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions, multilingual)
2. Split `.md` files by headers (`#`, `##`, `###`) → ~200 chunks with hierarchical metadata
3. Each `.jsonl` line → 1 chunk: `"Question: ... | Réponse: ... | Tactique: ..."` + metadata
4. All chunks vectorized and stored in ChromaDB collection `"interrogation_vault"`

### Retrieval (each turn)

```
Inspector query  = last_question + profiler_output.reason
Profiler query   = last_answer

retrieve(query, k=3)                         → any source (Inspector)
retrieve_behavioral_examples(query, k=3)     → source="dataset" only (Profiler)
```

ChromaDB uses cosine similarity to return the top-k closest chunks.

---

## State (LangGraph `InterrogationState`)

| Field | Type | Description |
|-------|------|-------------|
| `case_data` | dict | Case summary, key facts, suspect list |
| `suspect_profile` | dict | Personality, hidden truth, defense strategy |
| `conversation_history` | list | All Inspector/Suspect exchanges |
| `retrieved_context` | list[str] | RAG chunks for Inspector (current turn) |
| `profiler_context` | list[str] | RAG examples for Profiler (current turn) |
| `rag_history` | list | Full RAG trace across all turns (query + retrieved chunks) |
| `last_question` | str | Last question asked by Inspector |
| `last_answer` | str | Last answer given by Suspect |
| `profiler_output` | dict | Latest profiler analysis (score, reason, signals) |
| `profiler_history` | list | All profiler outputs across turns |
| `turn` | int | Current turn number |
| `max_turns` | int | Maximum number of turns before forcing final report |
| `final_report` | str | Generated at the end of the loop |

---

## Observability

All agents and the `call_llm` function are decorated with Langfuse `@observe`:

```
call_llm         → @observe(as_type="generation")  — tokens, latency, prompt/response
inspector_agent  → @observe
suspect_agent    → @observe
profiler_agent   → @observe
final_report_agent → @observe
```

Traces are flushed at the end of each Streamlit run (`langfuse_client.flush()`).

---

## RAG Toggle (A/B Testing)

The Streamlit UI exposes a RAG ON/OFF toggle. When OFF, `rag_collection=None` is passed to `build_graph()` and all retrieval returns empty lists.

For rigorous A/B comparison:
- Set **all agent temperatures to 0.0** — the only variable is the RAG
- Run the same case with RAG ON and RAG OFF
- Use the **Judge Agent** (temperature 0.0) to score both runs on the same criteria

This isolates the RAG's contribution from LLM randomness.

---

## File Map

```
src/
├── agents.py       # Agent functions (LangGraph nodes)
├── graph.py        # Graph definition + retrieve_context node
├── rag.py          # build_index(), retrieve(), retrieve_behavioral_examples()
├── prompts.py      # Prompt templates (INSPECTOR_PROMPT, SUSPECT_PROMPT, ...)
├── state.py        # InterrogationState TypedDict
├── utils.py        # call_llm(), get_model(), parse_json_response(), format_conversation()
├── data_prep.py    # extract_and_clean_pdfs(), generate_md_from_txt(), transform_csv_to_jsonl()
app.py              # Streamlit UI
data/
├── cases/          # case_XXX.json
├── suspects/       # suspect_XXX.json
└── rag_docs/
    ├── raw/        # Source PDFs + CSV
    └── processed/  # .txt, .md, .jsonl
```
