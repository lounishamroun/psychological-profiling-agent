"""
Streamlit app for the interrogation simulator.

Person C scope:
- present the interrogation transcript,
- show profiler metrics and charts,
- display retrieved context and final report,
- integrate with LangGraph/RAG when those modules are available.
"""

from __future__ import annotations

import html
from pathlib import Path
import importlib
from typing import Any, Callable

import pandas as pd
import streamlit as st

from src.agents import final_report_agent, inspector_agent, profiler_agent, suspect_agent
from src.utils import load_json

BASE_DIR = Path(__file__).resolve().parent
CASE_PATH = BASE_DIR / "data" / "cases" / "case_001.json"
SUSPECT_PATH = BASE_DIR / "data" / "suspects" / "suspect_001.json"

ChartData = pd.DataFrame


def configure_page() -> None:
    """Set page metadata and inject a small visual system."""
    st.set_page_config(
        page_title="Psychological Profiling Agent",
        page_icon=":male-detective:",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(29, 78, 216, 0.10), transparent 32%),
                radial-gradient(circle at top right, rgba(234, 88, 12, 0.10), transparent 28%),
                #f5f7fb;
        }
        .role-card {
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.85rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }
        .role-card.inspector {
            background: linear-gradient(135deg, #eff6ff, #dbeafe);
            border-left: 6px solid #1d4ed8;
        }
        .role-card.suspect {
            background: linear-gradient(135deg, #fff7ed, #ffedd5);
            border-left: 6px solid #ea580c;
        }
        .role-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            color: #334155;
            margin-bottom: 0.3rem;
        }
        .role-content {
            color: #0f172a;
            line-height: 1.55;
        }
        .context-box {
            border-radius: 14px;
            padding: 0.9rem 1rem;
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.08);
            margin-bottom: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_initial_state(max_turns: int) -> dict[str, Any]:
    """Create the initial interrogation state used by both execution modes."""
    return {
        "case_data": load_json(str(CASE_PATH)),
        "suspect_profile": load_json(str(SUSPECT_PATH)),
        "conversation_history": [],
        "retrieved_context": [],
        "last_question": "",
        "last_answer": "",
        "profiler_output": {},
        "profiler_history": [],
        "turn": 0,
        "max_turns": max_turns,
        "final_report": "",
    }


def merge_agent_updates(state: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Reproduce LangGraph's list-append behavior in the local fallback runner."""
    merged = dict(state)
    for key, value in updates.items():
        if key in {"conversation_history", "profiler_history"}:
            existing = list(merged.get(key, []))
            existing.extend(value)
            merged[key] = existing
        else:
            merged[key] = value
    return merged


def load_backend_functions() -> tuple[Callable[..., Any] | None, Callable[..., Any] | None]:
    """Best-effort import of graph and RAG builders from teammate modules."""
    build_graph = None
    build_index = None

    try:
        graph_module = importlib.import_module("src.graph")
        build_graph = getattr(graph_module, "build_graph", None)
    except Exception:
        build_graph = None

    try:
        rag_module = importlib.import_module("src.rag")
        build_index = getattr(rag_module, "build_index", None)
    except Exception:
        build_index = None

    return build_graph, build_index


def run_with_langgraph(max_turns: int) -> tuple[dict[str, Any], str]:
    """Run the interrogation through the official graph if it is available."""
    build_graph, build_index = load_backend_functions()
    if not callable(build_graph) or not callable(build_index):
        raise RuntimeError("LangGraph or RAG backend not available yet.")

    rag_collection = build_index()
    graph = build_graph(rag_collection)
    initial_state = build_initial_state(max_turns)
    result = graph.invoke(initial_state)
    return result, "LangGraph + RAG"


def run_with_local_fallback(max_turns: int) -> tuple[dict[str, Any], str]:
    """Run a simple sequential loop so the UI stays usable before full integration."""
    state = build_initial_state(max_turns)

    for _ in range(max_turns):
        state["retrieved_context"] = []
        state = merge_agent_updates(state, inspector_agent(state))
        state = merge_agent_updates(state, suspect_agent(state))
        state = merge_agent_updates(state, profiler_agent(state))

    state = merge_agent_updates(state, final_report_agent(state))
    return state, "Sequential fallback"


def run_interrogation(max_turns: int) -> tuple[dict[str, Any], str]:
    """Prefer the shared graph, otherwise fall back to a simple local execution mode."""
    try:
        return run_with_langgraph(max_turns)
    except Exception:
        return run_with_local_fallback(max_turns)


def history_to_dataframe(profiler_history: list[dict[str, Any]]) -> ChartData:
    """Convert profiler history into a chart-friendly dataframe."""
    rows = []
    for index, entry in enumerate(profiler_history, start=1):
        rows.append(
            {
                "turn": index,
                "stress_level": float(entry.get("stress_level", 0.0)),
                "evasion_score": float(entry.get("evasion_score", 0.0)),
                "consistency_score": float(entry.get("consistency_score", 0.0)),
                "suspicion_score": float(entry.get("suspicion_score", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def render_message(role: str, content: str) -> None:
    """Render one transcript message with a role-specific card."""
    css_role = "inspector" if role == "inspector" else "suspect"
    title = "Inspector" if role == "inspector" else "Suspect"
    safe_content = html.escape(content).replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="role-card {css_role}">
            <div class="role-label">{title}</div>
            <div class="role-content">{safe_content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(state: dict[str, Any], backend_name: str) -> None:
    """Render controls and live profiler metrics."""
    with st.sidebar:
        st.header("Behavioral Dashboard")
        st.caption(f"Execution mode: {backend_name}")

        latest = state.get("profiler_output") or {}
        st.metric("Turn", f"{state.get('turn', 0)} / {state.get('max_turns', 0)}")
        st.metric("Stress", f"{float(latest.get('stress_level', 0.0)):.2f}")
        st.metric("Evasion", f"{float(latest.get('evasion_score', 0.0)):.2f}")
        st.metric("Consistency", f"{float(latest.get('consistency_score', 0.0)):.2f}")
        st.metric("Suspicion", f"{float(latest.get('suspicion_score', 0.0)):.2f}")

        profiler_history = state.get("profiler_history", [])
        if profiler_history:
            chart_df = history_to_dataframe(profiler_history)
            st.subheader("Suspicion Over Time")
            st.line_chart(chart_df, x="turn", y="suspicion_score", height=180)
            st.subheader("Stress Over Time")
            st.line_chart(chart_df, x="turn", y="stress_level", height=180)
        else:
            st.info("Run the simulation to populate live metrics.")

        with st.expander("Latest profiler reason", expanded=bool(latest.get("reason"))):
            st.write(latest.get("reason", "No profiler analysis yet."))

        with st.expander("Retrieved context", expanded=False):
            context_chunks = state.get("retrieved_context", [])
            if context_chunks:
                for chunk in context_chunks:
                    safe_chunk = html.escape(str(chunk)).replace("\n", "<br>")
                    st.markdown(
                        f'<div class="context-box">{safe_chunk}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.write("No retrieved context available in the current run.")


def render_header(case_data: dict[str, Any]) -> None:
    """Render the top-of-page case context."""
    st.title("Multi-Agent AI Interrogation Simulator")
    st.caption("Inspector vs Suspect with live behavioral profiling.")

    facts = case_data.get("key_facts", [])
    with st.container():
        st.subheader(case_data.get("title", "Case Overview"))
        st.write(case_data.get("summary", ""))
        if facts:
            with st.expander("Key case facts", expanded=False):
                for fact in facts:
                    st.write(f"- {fact}")


def render_transcript(state: dict[str, Any]) -> None:
    """Render the interrogation transcript in the main area."""
    st.subheader("Transcript")
    history = state.get("conversation_history", [])
    if not history:
        st.info("Click 'Run interrogation' to generate the conversation.")
        return

    for message in history:
        render_message(message.get("role", "suspect"), message.get("content", ""))


def render_final_report(state: dict[str, Any]) -> None:
    """Render the final report when available."""
    report = state.get("final_report", "")
    st.subheader("Final Report")
    if report:
        st.markdown(report)
    else:
        st.info("The final report will appear here after the simulation completes.")


def initialize_session() -> None:
    """Ensure required session keys exist."""
    if "simulation_state" not in st.session_state:
        st.session_state.simulation_state = build_initial_state(max_turns=5)
    if "backend_name" not in st.session_state:
        st.session_state.backend_name = "Not run yet"


def main() -> None:
    """Run the Streamlit app."""
    configure_page()
    initialize_session()

    case_data = st.session_state.simulation_state["case_data"]
    render_header(case_data)

    with st.sidebar:
        st.subheader("Controls")
        max_turns = st.slider("Number of turns", min_value=1, max_value=8, value=5)
        run_clicked = st.button("Run interrogation", type="primary", use_container_width=True)
        reset_clicked = st.button("Reset", use_container_width=True)

    if reset_clicked:
        st.session_state.simulation_state = build_initial_state(max_turns=max_turns)
        st.session_state.backend_name = "Not run yet"

    if run_clicked:
        with st.spinner("Running interrogation..."):
            try:
                state, backend_name = run_interrogation(max_turns=max_turns)
            except Exception as exc:
                st.error(f"Simulation failed: {exc}")
            else:
                st.session_state.simulation_state = state
                st.session_state.backend_name = backend_name

    state = st.session_state.simulation_state
    backend_name = st.session_state.backend_name
    render_sidebar(state, backend_name)

    transcript_col, report_col = st.columns([1.35, 1.0], gap="large")
    with transcript_col:
        render_transcript(state)
    with report_col:
        render_final_report(state)


if __name__ == "__main__":
    main()
