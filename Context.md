PROJECT CONTEXT PROMPT
======================

I will do one of the task tagged as “Person X”, since this same prompt will be sent to team members we need to split responsibilities,I will be <PERSON X>, as a reminder here’s the context of the project:

Project: Multi-Agent AI Interrogation Simulator with Dynamic Profiling
Repo: psychological-profiling-agent
Stack: Python, Streamlit, LangGraph, ChromaDB, sentence-transformers, Gemini 2.0 Flash

GOAL:


Build a Streamlit app where 3 AI agents run a simulated police interrogation:
- Inspector agent asks strategic questions
- Suspect agent answers in character based on a hidden profile
- Profiler agent scores each answer (stress, evasion, consistency, suspicion)
- A RAG layer retrieves relevant interrogation tactics from local PDF/text docs
- LangGraph orchestrates a fixed turn loop (retrieve → ask → answer → profile → repeat)
- The app ends with a final interrogation report

Project goal:
Build a Streamlit app where:
- an Inspector agent asks questions,
- a Suspect agent answers based on a hidden suspect profile and case facts,
- a Profiler agent analyzes each answer and outputs structured suspicion metrics,
- a small RAG layer retrieves relevant interrogation tactics / case context,
- LangGraph orchestrates the loop for a fixed number of turns,
- the app ends with a final report.

Hard constraints:
- We are beginners.
- We have only 1 day.
- Team of 4.
- This is a POC / demo, not production software.
- Avoid overengineering.
- Every implementation choice must maximize speed, clarity, and demo reliability.

Non-goals:
- No authentication
- No database server
- No Docker unless strictly necessary
- No live external data ingestion from APIs
- No advanced agent autonomy
- No complex memory systems
- No heavy observability stack unless the core app already works
- No unnecessary abstractions
- No enterprise architecture
- No test suite unless I explicitly ask
- No premature optimization

Core stack:
- Python
- Streamlit
- LangGraph
- ChromaDB
- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- Gemini 2.0 Flash (or easily swappable LLM call layer)
- python-dotenv
- local JSON/TXT files as dataset

Architecture to follow:
1. Local handcrafted dataset
- case files in JSON
- suspect profiles in JSON
- interrogation tactics / reference docs in TXT or MD

2. RAG
- chunk docs with RecursiveCharacterTextSplitter
- embed with paraphrase-multilingual-MiniLM-L12-v2
- store in ChromaDB
- retrieve top-k relevant chunks
- optionally reranking later, but not in the initial MVP

3. Agents
- Inspector agent:
asks one focused question at a time using conversation history, retrieved context, and profiler feedback
- Suspect agent:
answers according to hidden suspect profile and case facts
- Profiler agent:
outputs structured JSON with fields like:
stress_level
evasion_score
consistency_score
suspicion_score
reason

4. LangGraph orchestration
Fixed graph, minimal complexity:
START
-> retrieve_context
-> inspector_agent
-> suspect_agent
-> profiler_agent
-> if turn < max_turns go back to retrieve_context
-> else final_report
-> END

5. UI
Streamlit app with:
- conversation panel
- live profiler metrics
- retrieved context display
- final verdict/report

Coding principles:
- Prefer simple Python functions over deep class hierarchies.
- Use small, readable modules.
- Keep state explicit and easy to inspect.
- Use typed dicts / dataclasses only if they simplify things.
- Use JSON-compatible structures where possible.
- Make code easy for beginners to understand and modify.
- When in doubt, choose the simpler implementation.
- Do not add features I did not ask for.
- Do not redesign the app unless necessary.
- Do not use LangChain everywhere just because it exists.
- Keep integrations minimal and practical.


LangGraph flow:
START → retrieve_context → inspector_agent → suspect_agent → profiler_agent
  → if turn < max_turns: loop back to retrieve_context
  → else: final_report → END

==================================================
WHAT HAS BEEN DONE (Branches exist for each step)
==================================================

✅ Branch: feature/sample-data (committed)
   - data/cases/case_001.json — "The Belmont Jewelry Heist" case scenario
   - data/suspects/suspect_001.json — "Marc Duval" suspect profile with hidden truth, strategy, vulnerabilities

✅ Branch: feature/state-and-utils (committed)
   - src/state.py — InterrogationState TypedDict (shared state for LangGraph)
     Fields: case_data, suspect_profile, conversation_history, retrieved_context,
     last_question, last_answer, profiler_output, profiler_history, turn, max_turns, final_report
     Uses Annotated[list, operator.add] for conversation_history and profiler_history (auto-append)
   - src/utils.py — LLM wrapper and helpers
     Functions: call_llm(prompt), load_json(path), parse_json_response(text), format_conversation(history)
     Uses google-generativeai with Gemini 2.0 Flash, reads GOOGLE_API_KEY from .env

✅ Branch: feature/prompts-and-agents (in progress, committed prompts.py and agents.py)
   - src/prompts.py — 4 prompt templates with {placeholders}:
     INSPECTOR_PROMPT, SUSPECT_PROMPT, PROFILER_PROMPT, FINAL_REPORT_PROMPT
   - src/agents.py — 4 agent functions:
     inspector_agent(state), suspect_agent(state), profiler_agent(state), final_report_agent(state)
     Each reads state → builds prompt → calls call_llm() → returns state updates dict

✅ Already in data/rag_docs/ (7 PDFs):
   - Interview_and_interrogation_methods_and_their_effe.pdf
   - UNODC_2024_Practical_Guide_on_the_Investigation_of_Corruption_Cases.pdf
   - fr_apt_paper_on_principles.pdf
   - investigative-interviewing.pdf
   - peace_method.pdf
   - raid_technique.pdf
   - us_justice_dept.pdf

==================================================
WHAT STILL NEEDS TO BE DONE
==================================================

Step 4: src/graph.py — LangGraph orchestration - Build a StateGraph with InterrogationState. - Add nodes: retrieve_context, inspector_agent, suspect_agent, profiler_agent, final_report. - Add edges: linear flow + conditional edge (loop or finish based on turn count). - Export: build_graph(rag_collection) function.

 Step 5: src/rag.py — Optimized RAG pipeline - Load documents from data/rag_docs/ (Prioritize .md files). - Chunking: Use MarkdownHeaderTextSplitter (split by #, ##, ###) to keep tactical concepts (e.g., "Phase A: Clarification") as single, coherent units. - Embed: sentence-transformers "paraphrase-multilingual-MiniLM-L12-v2". - Store: ChromaDB (in-memory ephemeral client). - Export: build_index() and retrieve(collection, query, k=3). - Query Logic: The retrieval query should be based on the current interview phase or the suspect's last behavioral state. 

Step 6: app.py — Streamlit UI - Main area: Chat transcript with distinct styles for Inspector and Suspect. - Side panel: Real-time "Behavioral Dashboard" showing stress/evasion levels. - Charts: Use st.line_chart to track suspicion and stress evolution over turns. - Final Report: Comprehensive summary of the interview's effectiveness. t

Step 7: .env file
   - Must contain: GOOGLE_API_KEY=your_gemini_api_key

Step 8: End-to-end test
   - Run: streamlit run app.py
   - Verify the full loop works for 5 turns and produces a final report

==================================================
PYPROJECT.TOML DEPENDENCIES (already set)
==================================================

streamlit, langgraph, langchain-text-splitters, chromadb,
sentence-transformers, google-generativeai, python-dotenv, datasets

==================================================
FOLDER STRUCTURE
==================================================

project/
  app.py              ← Streamlit UI (TODO)
  .env                ← API key (TODO)
  pyproject.toml      ← dependencies (DONE)
  data/
    cases/case_001.json       (DONE)
    suspects/suspect_001.json (DONE)
    rag_docs/                 (7 PDFs already there)
  src/
    _init_.py
    state.py     (DONE)
    utils.py     (DONE)
    prompts.py   (DONE)
    agents.py    (DONE)
    rag.py       (TODO)
    graph.py     (TODO)

==================================================
TASK ASSIGNMENT SUGGESTIONS FOR TEAMMATES
==================================================

Person A: src/rag.py
Fichier cible : src/rag.py

Objectif : Transformer les manuels de doctrine en une base de données vectorielle capable de fournir des conseils tactiques en temps réel à l'Inspecteur et des critères d'évaluation au Profiler.
1. Ingestion et Nettoyage du Corpus

La priorité est de garantir la qualité de la donnée source pour éviter les hallucinations de l'IA.

    Sélection des sources : Charger exclusivement les documents au format Markdown (.md) : le cadre PEACE, la méthode SUE, les principes de Méndez et le guide de l'UNODC.

    Nettoyage structurel : S'assurer que le texte ne contient plus de bruits (numéros de pages, en-têtes de PDF mal convertis) qui pourraient fausser la recherche sémantique.

2. Découpage Sémantique (Smart Chunking)

Le découpage est l'étape critique qui permet à l'IA de comprendre le contexte.

    Utilisation du MarkdownHeaderTextSplitter : Segmenter les documents en fonction de leur hiérarchie (#, ##, ###).

    Cohérence des blocs : Chaque "chunk" (fragment) doit représenter une unité tactique complète (par exemple, la définition entière d'une phase de l'entretien ou une technique de questionnement spécifique).

    Héritage des métadonnées : Chaque fragment doit conserver l'étiquette de sa source (ex: source: mendez) et de sa section pour permettre des recherches filtrées par les autres agents.

3. Pipeline de Vectorisation

Il s'agit de transformer le texte en langage mathématique pour la recherche.

    Modèle d'Embedding : Implémenter paraphrase-multilingual-MiniLM-L12-v2 via la bibliothèque sentence-transformers. Ce modèle multilingue gère correctement le français, essentiel pour nos documents RAG.

    Base de Données Vectorielle : Configurer ChromaDB en mode éphémère (InMemory). L'index doit être reconstruit à chaque lancement pour garantir que les agents travaillent toujours sur la version la plus à jour des documents.

4. Interface de Récupération (Retrieval API)

La Personne A doit exposer deux fonctions essentielles utilisables par l'orchestrateur (LangGraph) :

    build_index() : Fonction qui automatise le chargement, le découpage et l'indexation de tous les fichiers du dossier data/rag_docs/.

    retrieve(query, context_filter, k=3) : Fonction de recherche qui accepte une requête textuelle (ex: "stratégie d'évasion détectée") et renvoie les 3 fragments les plus pertinents. Elle doit permettre de filtrer par métadonnées si l'agent a besoin d'un type de conseil spécifique (ex: uniquement des conseils éthiques).

5. Intégration et Support Multi-Agents

    Lien avec l'Inspecteur : Fournir les "scripts" et types de questions (TED, ouvertes) à poser selon la phase PEACE en cours.

    Lien avec le Profiler : Fournir les indicateurs de crédibilité issus de la méthode SUE pour que le Profiler puisse scorer la réponse du suspect.

    Validation : Implémenter une fonction de vérification simple pour afficher un échantillon des données indexées au démarrage de l'application Streamlit.

Person B: src/graph.py
  - Wire up LangGraph StateGraph with all nodes and edges
  - Needs: langgraph, src/state.py, src/agents.py, src/rag.py

Person C: app.py
  - Build the Streamlit UI
  - Needs: streamlit, all src/ modules working

Person D: Testing + polish
  - Create .env, run end-to-end, fix prompt issues, improve UI display

==================================================
CODING RULES
==================================================

- RAG Content: Only use cleaned Markdown documents to avoid "noise" (page numbers, footers).
- Semantic Splitting: Never split in the middle of a paragraph; use headers as delimiters.
- Structured Output: Profiler Agent MUST return structured JSON for chart compatibility.
- Model usage: Use response_mime_type="application/json" for the Profiler and Report agents.
- Functions over classes: Keep the logic flat and easy to debug.
- Single LLM Entrypoint: All agents must use call_llm() from utils.py.