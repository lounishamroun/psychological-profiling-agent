"""
Streamlit app for the interrogation simulator.
"""

from __future__ import annotations

import html
from pathlib import Path
import importlib
from typing import Any, Callable

import streamlit as st
from langfuse import observe, Langfuse

from src.agents import final_report_agent, inspector_agent, profiler_agent, suspect_agent, judge_agent
from src.utils import load_json

BASE_DIR = Path(__file__).resolve().parent
CASE_PATH = BASE_DIR / "data" / "cases" / "case_001.json"
SUSPECTS_DIR = BASE_DIR / "data" / "suspects"


def get_case_suffix(case_path: Path = CASE_PATH) -> str:
    """Extract the numeric suffix from a case filename like case_001.json."""
    return case_path.stem.split("_")[-1]


def suspect_sort_key(suspect_path: Path) -> tuple[int, str]:
    """Sort suspects by numeric index first, then by filename."""
    parts = suspect_path.stem.split("_")
    if len(parts) > 1 and parts[1].isdigit():
        return int(parts[1]), suspect_path.stem
    return 9999, suspect_path.stem

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
        :root {
            color-scheme: dark;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(59, 130, 246, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(249, 115, 22, 0.14), transparent 24%),
                linear-gradient(180deg, #030712 0%, #020617 48%, #000000 100%);
            color: #e5e7eb;
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(2, 6, 23, 0.98));
            border-right: 1px solid rgba(148, 163, 184, 0.18);
        }
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp p,
        .stApp label,
        .stApp span,
        .stApp div {
            color: inherit;
        }
        [data-testid="stMetric"],
        [data-testid="stAlert"],
        .stExpander,
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(15, 23, 42, 0.62);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 16px;
            box-shadow: 0 12px 34px rgba(0, 0, 0, 0.28);
        }
        [data-testid="stMetric"] {
            padding: 0.75rem 0.9rem;
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {
            color: #f8fafc;
        }
        [data-baseweb="base-input"],
        [data-baseweb="select"] > div,
        .stSlider {
            background: rgba(15, 23, 42, 0.66);
            color: #f8fafc;
        }
        .role-card {
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.85rem;
            border: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.24);
        }
        .role-card.inspector {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.92));
            border-left: 6px solid #3b82f6;
        }
        .role-card.suspect {
            background: linear-gradient(135deg, rgba(41, 21, 12, 0.96), rgba(28, 25, 23, 0.92));
            border-left: 6px solid #f97316;
        }
        .role-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            color: #cbd5e1;
            margin-bottom: 0.3rem;
        }
        .role-content {
            color: #f8fafc;
            line-height: 1.55;
        }
        .context-box {
            border-radius: 14px;
            padding: 0.9rem 1rem;
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.18);
            margin-bottom: 0.75rem;
        }
        .stButton > button {
            background: linear-gradient(135deg, #1d4ed8, #2563eb);
            color: #f8fafc;
            border: 1px solid rgba(96, 165, 250, 0.28);
        }
        .stButton > button[kind="secondary"] {
            background: rgba(15, 23, 42, 0.78);
            color: #e5e7eb;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_available_suspects(case_path: Path = CASE_PATH) -> dict[str, dict[str, Any]]:
    """Load suspects that belong to the current case and build labels."""
    case_suffix = get_case_suffix(case_path)
    suspects: dict[str, dict[str, Any]] = {}
    pattern = f"suspect_*_{case_suffix}_*.json"
    for suspect_path in sorted(SUSPECTS_DIR.glob(pattern), key=suspect_sort_key):
        profile = load_json(str(suspect_path))
        suspects[str(suspect_path)] = {
            "path": suspect_path,
            "profile": profile,
            "label": (
                f"{profile.get('name', suspect_path.stem)}"
                f" - {profile.get('occupation', 'Unknown role')}"
            ),
        }
    return suspects


def build_initial_state(max_turns: int, suspect_path: Path | None = None) -> dict[str, Any]:
    """Create the initial interrogation state used by both execution modes."""
    if suspect_path is None:
        available_suspects = get_available_suspects()
        if not available_suspects:
            raise FileNotFoundError("No suspect files found for the current case.")
        selected_suspect_path = next(iter(available_suspects.values()))["path"]
    else:
        selected_suspect_path = suspect_path
    return {
        "case_data": load_json(str(CASE_PATH)),
        "suspect_profile": load_json(str(selected_suspect_path)),
        "conversation_history": [],
        "retrieved_context": [],
        "profiler_context": [],
        "rag_history": [],
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
        if key in {"conversation_history", "profiler_history", "rag_history"}:
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


@observe(name="run_with_langgraph")
def run_with_langgraph(
    max_turns: int,
    suspect_path: Path | None = None,
    use_rag: bool = True,
) -> tuple[dict[str, Any], str]:
    """Run the interrogation through the official graph if it is available."""
    build_graph, build_index = load_backend_functions()
    if not callable(build_graph):
        raise RuntimeError("LangGraph backend not available yet.")

    rag_collection = build_index() if (callable(build_index) and use_rag) else None
    graph = build_graph(rag_collection)
    initial_state = build_initial_state(max_turns, suspect_path=suspect_path)
    result = graph.invoke(initial_state)
    if not use_rag:
        return result, "LangGraph (RAG OFF)"
    if rag_collection is None:
        return result, "LangGraph (no RAG available)"
    return result, "LangGraph + RAG"


@observe(name="run_with_local_fallback")
def run_with_local_fallback(
    max_turns: int,
    suspect_path: Path | None = None,
    use_rag: bool = True,
) -> tuple[dict[str, Any], str]:
    """Run a simple sequential loop so the UI stays usable before full integration."""
    state = build_initial_state(max_turns, suspect_path=suspect_path)

    for _ in range(max_turns):
        state["retrieved_context"] = []
        state["profiler_context"] = []
        state = merge_agent_updates(state, inspector_agent(state))
        state = merge_agent_updates(state, suspect_agent(state))
        state = merge_agent_updates(state, profiler_agent(state))

    state = merge_agent_updates(state, final_report_agent(state))
    if use_rag:
        return state, "Sequential fallback (RAG unavailable)"
    return state, "Sequential fallback (RAG OFF)"


@observe(name="interrogation_session")
def run_interrogation(
    max_turns: int,
    suspect_path: Path | None = None,
    use_rag: bool = True,
) -> tuple[dict[str, Any], str]:
    """Prefer the shared graph, otherwise fall back to a simple local execution mode.

    Langfuse traces this as the top-level trace for the entire interrogation.
    """
    try:
        return run_with_langgraph(max_turns, suspect_path=suspect_path, use_rag=use_rag)
    except Exception:
        return run_with_local_fallback(max_turns, suspect_path=suspect_path, use_rag=use_rag)


def history_to_chart_rows(profiler_history: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Convert profiler history into chart-friendly rows."""
    rows: list[dict[str, float]] = []
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
    return rows


def shorten_text(text: Any, max_chars: int = 220) -> str:
    """Collapse whitespace and trim long UI strings."""
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= max_chars:
        return collapsed
    return f"{collapsed[: max_chars - 3].rstrip(' ,;:')}..."


def summarize_rag_chunk(chunk: Any, context_type: str) -> str:
    """Turn raw RAG chunks into short, readable highlights."""
    raw_text = str(chunk).strip()
    if not raw_text:
        return ""

    if context_type == "profiler_context":
        fields: dict[str, str] = {}
        for part in raw_text.split("|"):
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            fields[key.strip().lower()] = value.strip()

        tactic = fields.get("tactique")
        question = fields.get("question")
        answer = fields.get("reponse") or fields.get("réponse") or fields.get("response")

        pieces = []
        if tactic and tactic != "N/A":
            pieces.append(f"Tactic: {tactic}")
        if question:
            pieces.append(f"Question pattern: {question}")
        if answer:
            pieces.append(f"Response pattern: {answer}")
        if pieces:
            return shorten_text(" | ".join(pieces))

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if lines:
        candidate = lines[0]
        if len(candidate) < 90 and len(lines) > 1:
            candidate = f"{candidate} {lines[1]}"
        return shorten_text(candidate)

    return shorten_text(raw_text)


def collect_rag_highlights(rag_history: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Extract the main RAG points used across the run for the UI."""
    grouped = {
        "Questioning guidance": [],
        "Behavioral comparisons": [],
    }
    limits = {
        "Questioning guidance": 3,
        "Behavioral comparisons": 3,
    }
    seen: set[str] = set()

    for entry in rag_history:
        for chunk in entry.get("retrieved_context", []):
            summary = summarize_rag_chunk(chunk, "retrieved_context")
            summary_key = summary.lower()
            if not summary or summary_key in seen:
                continue
            if len(grouped["Questioning guidance"]) < limits["Questioning guidance"]:
                grouped["Questioning guidance"].append(summary)
                seen.add(summary_key)

        for chunk in entry.get("profiler_context", []):
            summary = summarize_rag_chunk(chunk, "profiler_context")
            summary_key = summary.lower()
            if not summary or summary_key in seen:
                continue
            if len(grouped["Behavioral comparisons"]) < limits["Behavioral comparisons"]:
                grouped["Behavioral comparisons"].append(summary)
                seen.add(summary_key)

        if all(len(grouped[group]) >= limits[group] for group in grouped):
            break

    return grouped


def rag_was_used(state: dict[str, Any], backend_name: str) -> bool:
    """Return True when the finished run actually used RAG."""
    return backend_name == "LangGraph + RAG" and bool(state.get("rag_history"))


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
            chart_rows = history_to_chart_rows(profiler_history)
            st.subheader("Suspicion Over Time")
            st.line_chart(chart_rows, x="turn", y="suspicion_score", height=180)
            st.subheader("Stress Over Time")
            st.line_chart(chart_rows, x="turn", y="stress_level", height=180)
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


def render_header(case_data: dict[str, Any], suspect_profile: dict[str, Any]) -> None:
    """Render the top-of-page case context."""
    st.title("Multi-Agent AI Interrogation Simulator")
    st.caption("Inspector vs Suspect with live behavioral profiling.")
    st.markdown(
        "**Selected suspect:** "
        f"{suspect_profile.get('name', 'Unknown suspect')} "
        f"({suspect_profile.get('occupation', 'Unknown occupation')})"
    )

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


def render_rag_support(state: dict[str, Any]) -> None:
    """Render the main retrieved RAG points that informed the run."""
    rag_history = state.get("rag_history", [])
    highlights = collect_rag_highlights(rag_history)
    latest_entry = rag_history[-1] if rag_history else {}

    st.subheader("RAG Support")
    st.caption("Key retrieved points that supported the questioning and truthfulness assessment.")

    if not rag_history:
        st.info("No RAG support was captured for this run.")
        return

    any_highlight = False
    for heading, items in highlights.items():
        if not items:
            continue
        any_highlight = True
        st.markdown(f"**{heading}**")
        for item in items:
            st.markdown(f"- {item}")

    if not any_highlight:
        st.info("RAG was enabled, but no readable supporting points were retained.")

    with st.expander("Queries used for retrieval", expanded=False):
        st.write(f"Inspector query: {latest_entry.get('query', 'N/A')}")
        st.write(f"Profiler query: {latest_entry.get('profiler_query', 'N/A')}")


def initialize_session(suspect_options: dict[str, dict[str, Any]]) -> None:
    """Ensure required session keys exist."""
    default_suspect_path = next(iter(suspect_options))
    if "selected_suspect_path" not in st.session_state:
        st.session_state.selected_suspect_path = default_suspect_path
    elif st.session_state.selected_suspect_path not in suspect_options:
        st.session_state.selected_suspect_path = default_suspect_path
    if "simulation_state" not in st.session_state:
        st.session_state.simulation_state = build_initial_state(
            max_turns=5,
            suspect_path=Path(st.session_state.selected_suspect_path),
        )
    if "backend_name" not in st.session_state:
        st.session_state.backend_name = "Not run yet"
    if "use_rag" not in st.session_state:
        st.session_state.use_rag = True


def main() -> None:
    """Run the Streamlit app."""
    configure_page()

    suspect_options = get_available_suspects()
    if not suspect_options:
        st.error("No suspect files found for the current case in data/suspects.")
        st.stop()
    initialize_session(suspect_options)

    selected_suspect_path = st.session_state.selected_suspect_path
    selected_suspect_profile = suspect_options[selected_suspect_path]["profile"]
    case_data = st.session_state.simulation_state["case_data"]
    render_header(case_data, selected_suspect_profile)

    with st.sidebar:
        st.subheader("Controls")
        selected_suspect_path = st.selectbox(
            "Suspect to interrogate",
            options=list(suspect_options),
            index=list(suspect_options).index(st.session_state.selected_suspect_path),
            format_func=lambda suspect_key: suspect_options[suspect_key]["label"],
            key="selected_suspect_path",
        )
        max_turns = st.slider("Number of turns", min_value=1, max_value=8, value=5)
        use_rag = st.toggle(
            "RAG enabled",
            key="use_rag",
            help="Toggle retrieval-augmented generation on/off to compare results.",
        )
        comparison = st.toggle(
            "Comparison mode (temp=0)",
            key="comparison_mode",
            help="Forces all agents to temperature=0 for deterministic, reproducible A/B comparison.",
        )
        run_clicked = st.button("Run interrogation", type="primary", use_container_width=True)
        reset_clicked = st.button("Reset", use_container_width=True)
        st.caption("The selected suspect and RAG mode are applied on the next run or reset.")

    if reset_clicked:
        st.session_state.simulation_state = build_initial_state(
            max_turns=max_turns,
            suspect_path=Path(selected_suspect_path),
        )
        st.session_state.backend_name = "Not run yet"

    if run_clicked:
        # Set comparison mode before running so call_llm sees it
        from src import utils as _utils_module
        _utils_module.comparison_mode = comparison

        with st.spinner("Running interrogation..."):
            try:
                state, backend_name = run_interrogation(
                    max_turns=max_turns,
                    suspect_path=Path(selected_suspect_path),
                    use_rag=use_rag,
                )
            except Exception as exc:
                st.error(f"Simulation failed: {exc}")
            else:
                st.session_state.simulation_state = state
                st.session_state.backend_name = backend_name

                # Run the Judge on the completed simulation
                try:
                    judge_result = judge_agent(state)
                    st.session_state.judge_output = judge_result
                    # Store per-mode for RAG comparison
                    mode_key = "judge_rag_on" if use_rag else "judge_rag_off"
                    st.session_state[mode_key] = judge_result
                except Exception:
                    st.session_state.judge_output = None
            finally:
                # Flush Langfuse so all traces are sent before the page rerenders
                try:
                    Langfuse().flush()
                except Exception:
                    pass

    state = st.session_state.simulation_state
    backend_name = st.session_state.backend_name
    render_sidebar(state, backend_name)

    transcript_col, report_col = st.columns([1.35, 1.0], gap="large")
    with transcript_col:
        render_transcript(state)
    with report_col:
        render_final_report(state)
        if rag_was_used(state, backend_name):
            st.divider()
            render_rag_support(state)

    # --- Judge evaluation section ---
    judge_output = st.session_state.get("judge_output")
    if judge_output:
        st.divider()
        st.subheader("LLM-as-Judge Evaluation")
        st.caption("Rubric-based scoring — every number is traced to countable facts from the transcript.")

        j1, j2, j3, j4 = st.columns(4)
        j1.metric("Inspector Quality", f"{float(judge_output.get('inspector_quality', 0)):.0%}")
        j2.metric("Suspect Realism", f"{float(judge_output.get('suspect_realism', 0)):.0%}")
        j3.metric("Profiler Accuracy", f"{float(judge_output.get('profiler_accuracy', 0)):.0%}")
        j4.metric("Overall Effectiveness", f"{float(judge_output.get('overall_effectiveness', 0)):.0%}")

        # Detailed breakdowns
        with st.expander("Inspector details", expanded=False):
            d = judge_output.get("inspector_details", {})
            st.markdown(
                f"- **Questions asked:** {d.get('total_questions', '?')}\n"
                f"- **Unique topics covered:** {d.get('unique_topics', '?')}\n"
                f"- **Repeated questions:** {d.get('repeated_questions', '?')}\n"
                f"- **Times evidence was cited:** {d.get('evidence_used', '?')}\n"
                f"- **Follow-ups on evasion:** {d.get('followups_on_evasion', '?')}"
            )

        with st.expander("Suspect details", expanded=False):
            d = judge_output.get("suspect_details", {})
            st.markdown(
                f"- **Total answers:** {d.get('total_answers', '?')}\n"
                f"- **In character:** {d.get('in_character', '?')}\n"
                f"- **Strategy followed:** {d.get('strategy_followed', '?')}\n"
                f"- **Contradicts profile:** {d.get('contradicts_profile', '?')}\n"
                f"- **Shows vulnerabilities:** {d.get('shows_vulnerabilities', '?')}"
            )

        with st.expander("Profiler details", expanded=False):
            d = judge_output.get("profiler_details", {})
            st.markdown(
                f"- **Total turns scored:** {d.get('total_turns', '?')}\n"
                f"- **Accurate assessments:** {d.get('accurate_assessments', '?')}\n"
                f"- **Final suspicion aligned with truth:** {d.get('final_suspicion_alignment', '?')}"
            )

        with st.expander("Effectiveness details", expanded=False):
            d = judge_output.get("effectiveness_details", {})
            st.markdown(
                f"- **Correct verdict:** {d.get('correct_verdict', '?')}\n"
                f"- **Hidden truth elements surfaced:** {d.get('truth_elements_surfaced', '?')}\n"
                f"- **Progressive pressure applied:** {d.get('progressive_pressure', '?')}"
            )

        with st.expander("Judge reasoning", expanded=True):
            st.write(judge_output.get("reasoning", "No reasoning provided."))

    # --- RAG ON vs OFF comparison ---
    rag_on = st.session_state.get("judge_rag_on")
    rag_off = st.session_state.get("judge_rag_off")
    if rag_on and rag_off:
        st.divider()
        st.subheader("RAG ON vs RAG OFF Comparison")
        st.caption("Side-by-side Judge scores from both runs. Run once with RAG ON, once with RAG OFF.")

        criteria = [
            ("Inspector Quality", "inspector_quality"),
            ("Suspect Realism", "suspect_realism"),
            ("Profiler Accuracy", "profiler_accuracy"),
            ("Overall Effectiveness", "overall_effectiveness"),
        ]
        cols = st.columns(4)
        for col, (label, key) in zip(cols, criteria):
            on_val = float(rag_on.get(key, 0))
            off_val = float(rag_off.get(key, 0))
            delta = on_val - off_val
            col.metric(
                label,
                f"{on_val:.0%} vs {off_val:.0%}",
                delta=f"{delta:+.0%}",
                help=f"RAG ON: {on_val:.0%} | RAG OFF: {off_val:.0%}",
            )

        with st.expander("RAG ON reasoning"):
            st.write(rag_on.get("reasoning", ""))
        with st.expander("RAG OFF reasoning"):
            st.write(rag_off.get("reasoning", ""))


if __name__ == "__main__":
    main()
