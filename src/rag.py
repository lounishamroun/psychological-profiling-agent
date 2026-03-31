"""
RAG pipeline: index interrogation docs + dataset, retrieve relevant chunks.

Two data sources:
- Markdown tactical docs (.md) → split by headers → for Inspector tactics
- JSONL behavioral dataset → one chunk per example → for Profiler patterns

Uses ChromaDB (ephemeral) + sentence-transformers multilingual embeddings.
"""

import os
import json
import glob

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import MarkdownHeaderTextSplitter


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAG_DOCS_DIR = os.path.join("data", "rag_docs", "processed")
JSONL_PATH = os.path.join(RAG_DOCS_DIR, "interrogation_dataset_enriched.jsonl")


# ---------------------------------------------------------------------------
# Embedding wrapper for ChromaDB
# ---------------------------------------------------------------------------
class LocalEmbeddingFunction(EmbeddingFunction[Documents]):
    """Wraps sentence-transformers so ChromaDB can call it directly."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------
def build_index() -> chromadb.Collection:
    """Build the ChromaDB collection from .md docs and the JSONL dataset.

    Returns the collection object (to be passed to retrieve()).
    """
    embedding_fn = LocalEmbeddingFunction()
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name="interrogation_vault",
        embedding_function=embedding_fn,
    )

    # --- 1. Index Markdown tactical docs ---
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )

    md_files = sorted(glob.glob(os.path.join(RAG_DOCS_DIR, "*.md")))
    doc_ids = []
    doc_texts = []
    doc_metas = []

    for filepath in md_files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            doc_id = f"md-{filename}-{i}"
            text = chunk.page_content.strip()
            if not text:
                continue

            # Build metadata from header hierarchy
            meta = {"source": filename}
            for key in ("h1", "h2", "h3"):
                if key in chunk.metadata:
                    meta[key] = chunk.metadata[key]

            doc_ids.append(doc_id)
            doc_texts.append(text)
            doc_metas.append(meta)

    if doc_texts:
        collection.add(ids=doc_ids, documents=doc_texts, metadatas=doc_metas)
        print(f"[RAG] Indexed {len(doc_texts)} chunks from {len(md_files)} .md files")

    # --- 2. Index JSONL behavioral dataset ---
    jsonl_ids = []
    jsonl_texts = []
    jsonl_metas = []

    if os.path.exists(JSONL_PATH):
        with open(JSONL_PATH, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)

                meta_entry = entry.get("metadata", {})
                text = (
                    f"Question: {entry.get('question', '')} | "
                    f"Réponse: {entry.get('reponse', '')} | "
                    f"Tactique: {meta_entry.get('tactique', 'N/A')}"
                )

                jsonl_ids.append(f"jsonl-{entry.get('id', line_num)}")
                jsonl_texts.append(text)
                jsonl_metas.append({
                    "source": "dataset",
                    "veracite": entry.get("veracite", "unknown"),
                    "phase": meta_entry.get("phase", "unknown"),
                    "tactique": meta_entry.get("tactique", "N/A"),
                    "type_q": meta_entry.get("type_q", "N/A"),
                })

        if jsonl_texts:
            collection.add(ids=jsonl_ids, documents=jsonl_texts, metadatas=jsonl_metas)
            print(f"[RAG] Indexed {len(jsonl_texts)} examples from JSONL dataset")
    else:
        print(f"[RAG] Warning: JSONL not found at {JSONL_PATH}, skipping dataset")

    return collection


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------
def retrieve(collection: chromadb.Collection, query: str, k: int = 3) -> list[str]:
    """Query the collection and return top-k chunks as a flat list of strings.

    Compatible with agents.py which does: "\\n---\\n".join(retrieved_context)
    """
    results = collection.query(query_texts=[query], n_results=k)
    # ChromaDB returns {"documents": [[str, str, ...]]} — flatten the nested list
    documents = results.get("documents", [[]])
    return documents[0] if documents else []


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Building index...")
    col = build_index()
    print(f"\nTotal documents in collection: {col.count()}")

    test_query = "technique de questionnement pour confronter un suspect évasif"
    print(f"\nTest query: '{test_query}'")
    results = retrieve(col, test_query, k=3)
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(r[:300])
