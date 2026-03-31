"""Data prep: generate .txt and .jsonl files from raw sources.

Consolidates data preparation into one module:
- extract_and_clean_pdfs()  -> PDF pages -> cleaned .txt in processed/
- transform_csv_to_jsonl()  -> CSV -> enriched JSONL with PEACE/SUE metadata
- load_json(path)           -> load a JSON file (case or suspect)

Note: The .md files in processed/ are hand-curated and versioned directly.
This script only generates .txt and .jsonl files.
"""

import os
import csv
import json
import re

import PyPDF2
from .utils import call_llm


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "rag_docs", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "rag_docs", "processed")
INPUT_CSV = os.path.join(RAW_DIR, "interrogation_dataset.csv")
OUTPUT_JSONL = os.path.join(PROCESSED_DIR, "interrogation_dataset_enriched.jsonl")


# ---------------------------------------------------------------------------
# PDF extraction & cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean raw text extracted from a PDF for RAG usage."""
    # Document-specific headers/footers
    text = re.sub(r"(?m)^\d+\s*GUIDE PRATIQUE SUR LES ENQUÊTES.*$", "", text)
    text = re.sub(r"(?m)^ENQUÊTE SUR LES AFFAIRES DE CORRUPT\s*ION\s*\d+\s*$", "", text)
    text = re.sub(r"(?m)^The Méndez Principles on Effective Interviewing.*$", "", text)
    text = re.sub(r"(?m)^\d{2}/\d{2}/\d{4}\s+Investigative interviewing.*$", "", text)
    text = re.sub(r"(?m)^Credibility Assessment,.*$", "", text)
    text = re.sub(
        r"(?m)^(SUE: T\s*HEORETICAL\s*PRINCIPLES|TRANSLATING\s*PSyCHOLOGICAL\s*THEOR\s*y|META\s*-ANAL\s*yTIC\s*REvIEw)\s*\d+\s*$",
        "", text,
    )
    text = re.sub(r"(?m)^\d+\.\s*STRATEGIC USE OF EVIDENCE\s*\d+\s*$", "", text)
    text = re.sub(r"(?m)^https?://\S+$", "", text)
    text = re.sub(r"interviewing/investigative-interviewingPage\s*\d+", "", text)
    text = re.sub(r"(?m)^Rachel E Sadler.*$", "", text)
    text = re.sub(r"\[Type here\]", "", text)

    # Isolated page numbers
    text = re.sub(r"(?m)^\s*\d{1,3}\s*$", "", text)

    # Footnotes
    text = re.sub(r"\d{1,3}\s+(?:Voir|See|Ibid|Cf\.)[^\n]*", "", text)
    text = re.sub(r"(?m)^\d{1,3}\s+(?:La |Le |Les |L'|Cette |Ce |En |Dans ).*?(?:\.|$)", "", text)

    # URLs
    text = re.sub(r"https?://\S+", "", text)

    # Publication dates / read time
    text = re.sub(r"(?:First published|Updated|Written by|Publié en|mis à jour en)[^\n]*", "", text)
    text = re.sub(r"\d+ mins? read", "", text)

    # Academic references in parentheses
    text = re.sub(r"\([A-Z]{2,}[^)]*\d{4}[^)]*\)", "", text)

    # "For further information" lines
    text = re.sub(r"(?m)^For further information see.*$", "", text)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _extract_and_clean_pages(pdf_name: str, page_ranges: list[tuple[int, int]], output_name: str) -> str | None:
    """Extract pages from a raw PDF, clean, and save as .txt in processed/."""
    pdf_path = os.path.join(RAW_DIR, pdf_name)
    if not os.path.exists(pdf_path):
        print(f"[SKIP] {pdf_name} — not found in {RAW_DIR}")
        return None

    reader = PyPDF2.PdfReader(pdf_path)
    extracted_text = []

    for start, end in page_ranges:
        for i in range(start - 1, min(end, len(reader.pages))):
            text = reader.pages[i].extract_text()
            if text and text.strip():
                extracted_text.append(text.strip())

    raw = "\n\n".join(extracted_text)
    cleaned = clean_text(raw)

    output_path = os.path.join(PROCESSED_DIR, output_name)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"[OK] {output_name} — {len(cleaned):,} chars")
    return output_path


def extract_and_clean_pdfs():
    """Extract and clean all relevant PDF chapters into processed/ .txt files."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    pdf_jobs = [
        (
            "UNODC_2024_Practical_Guide_on_the_Investigation_of_Corruption_Cases_FRE.pdf",
            [(60, 65)],
            "UNODC_conduite_entretien.txt",
        ),
        (
            "fr_apt_paper_on_principles_and_uncat_updated_-_layout.pdf",
            [(1, 1), (3, 15)],
            "Mendez_principes_entretien_efficace.txt",
        ),
        (
            "investigative-interviewing--1774906911.pdf",
            [(1, 19)],
            "InvestigativeInterviewing_PEACE_framework.txt",
        ),
        (
            "SUEChapterRaskinbook.pdf",
            [(9, 23), (29, 31)],
            "SUE_utilisation_strategique_preuve.txt",
        ),
        (
            "peace_method.pdf",
            [(1, 100)],
            "peace_method.txt",
        ),
        (
            "raid_technique.pdf",
            [(1, 6)],
            "Reid_technique_entretien_interrogatoire.txt",
        ),
    ]

    print(f"[data_prep] Extracting PDFs → .txt in {PROCESSED_DIR}")
    for pdf_name, pages, out_name in pdf_jobs:
        _extract_and_clean_pages(pdf_name, pages, out_name)


# ---------------------------------------------------------------------------
# CSV -> JSONL transformation (PEACE/SUE enrichment)
# ---------------------------------------------------------------------------

def _classify_phase_peace(description: str) -> str:
    desc = description.lower()
    if any(kw in desc for kw in [
        "direct accusation", "confrontational", "accusation",
        "pressure", "high stress", "high stakes",
        "strategic probe", "confrontation",
    ]):
        return "Confronter"
    if any(kw in desc for kw in [
        "unconventional", "psychological", "feigned empathy",
        "feigned casual", "deceptive calm", "loyalty test",
        "deceptive probing",
    ]):
        return "Clarifier"
    if any(kw in desc for kw in [
        "intermediate", "probe", "cross-verification",
        "testing", "evidence", "financial", "conspiracy",
        "witness probe", "cyber", "technical questioning",
        "involvement",
    ]):
        return "Rendre compte"
    return "Engager"


def _classify_tactique_sue(tag: str, answer: str, description: str) -> str:
    if tag != "culprit":
        return "N/A"

    answer_lower = answer.lower()

    denial_markers = [
        r"\bno\b", r"\bnon\b", r"\bnever\b", r"\bdidn'?t\b",
        r"\bdon'?t\b", r"\bwasn'?t\b", r"\bdeny\b", r"\bnot\b",
        r"\bi wasn'?t\b", r"\bi haven'?t\b", r"\bi couldn'?t\b",
        r"\bi don'?t know\b",
    ]
    has_denial = any(re.search(pat, answer_lower) for pat in denial_markers)

    avoidance_markers = [
        r"\bjust\b", r"\bonly\b", r"\bmaybe\b", r"\bperhaps\b",
        r"\bi think\b", r"\bi'?m not sure\b", r"\bi can'?t recall\b",
        r"\bi forgot\b", r"\bi was told\b", r"\bsomeone\b",
        r"\bhonest mistake\b", r"\bgot lost\b", r"\bpersonal\b",
        r"\broutine\b", r"\bstandard\b", r"\bfollowed\b",
    ]
    has_avoidance = any(re.search(pat, answer_lower) for pat in avoidance_markers)

    if has_denial and not has_avoidance:
        return "Déni"
    if has_avoidance and not has_denial:
        return "Évitement"
    if has_denial and has_avoidance:
        return "Déni"
    return "Évitement" if len(answer) < 40 else "Déni"


def _classify_type_question(question: str) -> str:
    q = question.lower().strip()

    suggestive = [
        r"isn'?t it", r"don'?t you", r"didn'?t you",
        r"wouldn'?t you", r"aren'?t you", r"wasn'?t it",
        r"you saw", r"you were", r"you did",
        r"right\?$", r"correct\?$",
    ]
    if any(re.search(pat, q) for pat in suggestive):
        return "Suggestive"

    open_patterns = [
        r"^what", r"^who", r"^why", r"^where", r"^how",
        r"^tell\b", r"^explain\b", r"^describe\b",
        r"^can you (explain|describe|tell|walk)",
    ]
    if any(re.search(pat, q) for pat in open_patterns):
        return "Ouverte"

    closed_patterns = [
        r"^did\b", r"^were\b", r"^are\b", r"^do\b",
        r"^is\b", r"^have\b", r"^was\b", r"^has\b",
        r"^can you (confirm|verify|identify|account)",
    ]
    if any(re.search(pat, q) for pat in closed_patterns):
        return "Fermée spécifique"

    if re.search(r"^can you\b", q):
        return "Ouverte"

    return "Fermée spécifique"


def _classify_statut_preuve(description: str) -> str:
    desc = description.lower()
    if any(kw in desc for kw in [
        "direct accusation", "confrontational",
        "evidence probe", "evidence integrity",
        "evidence tampering", "cross-verification",
        "high stakes", "high stress",
        "strategic probe",
    ]):
        return "Divulgation tardive"
    return "Rétention"


def _transform_row(row: dict) -> dict:
    description = row.get("description", "")
    tag = row.get("tag", "")
    answer = row.get("answer", "")
    question = row.get("question", "")

    return {
        "id": row["id"],
        "question": question,
        "reponse": answer,
        "veracite": tag,
        "metadata": {
            "phase": _classify_phase_peace(description),
            "tactique": _classify_tactique_sue(tag, answer, description),
            "type_q": _classify_type_question(question),
            "preuve": _classify_statut_preuve(description),
        },
    }


def transform_csv_to_jsonl():
    """Transform the raw CSV dataset into enriched JSONL with PEACE/SUE metadata."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if not os.path.exists(INPUT_CSV):
        print(f"[data_prep] CSV not found: {INPUT_CSV}, skipping")
        return

    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [_transform_row(row) for row in reader]

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[data_prep] {len(rows)} rows exported → {OUTPUT_JSONL}")


# ---------------------------------------------------------------------------
# PDF .txt → structured .md via LLM
# ---------------------------------------------------------------------------

_MD_GENERATION_PROMPT = """
Tu es un expert en techniques d'interrogatoire et d'entretien investigatif.
Tu reçois un texte brut extrait d'un document académique ou professionnel sur les techniques d'entretien.

Ta tâche : restructure ce contenu en un document Markdown bien organisé, en français, destiné à alimenter un système RAG pour aider un inspecteur lors d'un interrogatoire.

Règles strictes :
- Utilise des headers Markdown : # pour le titre principal, ## pour les sections, ### pour les sous-sections
- Garde uniquement le contenu tactique et opérationnel utile (phases, techniques, signaux comportementaux, stratégies)
- Supprime les références bibliographiques, introductions académiques, remerciements, tables des matières
- Rédige en français clair et concis, même si le texte source est en anglais
- N'ajoute PAS de commentaire introductif ou de conclusion — commence directement par le titre `#`
- Les sections doivent être autonomes et compréhensibles sans contexte supplémentaire
- Conserve l'intégralité du contenu tactique et opérationnel — ne résume pas, ne tronque pas

Texte source :
{txt_content}

Markdown structuré :
"""


def generate_md_from_txt(force: bool = False) -> None:
    """Generate structured .md files from .txt files in processed/ using the LLM.

    For each .txt file, if the corresponding .md does not exist (or force=True),
    sends the text to Gemini and saves the structured Markdown output.

    Args:
        force: If True, regenerate even if .md already exists.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    txt_files = sorted([
        f for f in os.listdir(PROCESSED_DIR)
        if f.endswith(".txt")
    ])

    if not txt_files:
        print("[data_prep] No .txt files found in processed/, run extract-pdfs first.")
        return

    for txt_filename in txt_files:
        md_filename = txt_filename.replace(".txt", ".md")
        txt_path = os.path.join(PROCESSED_DIR, txt_filename)
        md_path = os.path.join(PROCESSED_DIR, md_filename)

        if os.path.exists(md_path) and not force:
            print(f"[SKIP] {md_filename} already exists (use --force to regenerate)")
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            txt_content = f.read()

        print(f"[LLM]  Generating {md_filename} ({len(txt_content):,} chars) ...")
        prompt = _MD_GENERATION_PROMPT.format(txt_content=txt_content)
        md_content = call_llm(prompt)

        # Strip potential markdown code fences if the LLM wraps the output
        md_content = re.sub(r"^```(?:markdown)?\n?", "", md_content.strip())
        md_content = re.sub(r"\n?```$", "", md_content.strip())

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content.strip() + "\n")

        print(f"[OK]   {md_filename} — {len(md_content):,} chars")


# ---------------------------------------------------------------------------
# JSON loader (cases, suspects)
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    """Load and return a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    usage = (
        "Usage: python -m src.data_prep "
        "[extract-pdfs | transform-jsonl | generate-md [--force] | all [--force]]"
    )

    args = sys.argv[1:]
    cmd = args[0] if args else "all"
    force = "--force" in args

    if cmd == "extract-pdfs":
        extract_and_clean_pdfs()
    elif cmd == "transform-jsonl":
        transform_csv_to_jsonl()
    elif cmd == "generate-md":
        generate_md_from_txt(force=force)
    elif cmd == "all":
        extract_and_clean_pdfs()
        print()
        transform_csv_to_jsonl()
        print()
        generate_md_from_txt(force=force)
    else:
        print(usage)
        sys.exit(1)