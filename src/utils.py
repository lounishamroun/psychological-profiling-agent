"""
Utility functions: LLM wrapper, JSON loading, response parsing.

This module keeps all the "plumbing" in one place:
- call_llm(prompt) -> str : the ONE function all agents use to talk to Gemini
- load_json(path) -> dict : load a JSON file
- parse_json_response(text) -> dict : extract JSON from LLM output
- format_conversation(history) -> str : format chat history for prompts
"""

import os
import json
import re
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env file so we can read GOOGLE_API_KEY
load_dotenv()

# Module-level cache for the model (created once, reused)
_model = None
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def get_model():
    """Initialize the Gemini model (once) and return it."""
    global _model
    if _model is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found! "
                "Create a .env file with: GOOGLE_API_KEY=your_key_here"
            )
        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        _model = genai.GenerativeModel(model_name)
    return _model


def call_llm(prompt: str) -> str:
    """Send a prompt to Gemini and return the text response.

    This is the ONLY function that talks to the LLM.
    All agents call this — so if you ever want to swap to a different
    model (OpenAI, Claude, Ollama...), you only change this function.
    """
    model = get_model()
    response = model.generate_content(prompt)
    return response.text


def load_json(path: str) -> dict:
    """Load and return a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def parse_json_response(text: str) -> dict:
    """Extract a JSON object from an LLM response.

    LLMs often wrap JSON in ```json ... ``` markdown blocks,
    or add extra text around it. This function handles that.
    """
    text = text.strip()

    # Strip markdown code block if present (```json ... ```)
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object {...} anywhere in the text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # If nothing works, return safe fallback values
    return {
        "stress_level": 0.5,
        "evasion_score": 0.5,
        "consistency_score": 0.5,
        "suspicion_score": 0.5,
        "reason": "Could not parse profiler output",
    }


def format_conversation(history: list) -> str:
    """Format conversation history into a readable string for prompts.

    Takes: [{"role": "inspector", "content": "..."}, ...]
    Returns: "Inspector: ...\nSuspect: ..."
    """
    if not history:
        return "(No conversation yet)"
    lines = []
    for msg in history:
        role = msg["role"].capitalize()
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)
