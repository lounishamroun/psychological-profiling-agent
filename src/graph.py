"""
LangGraph orchestration for the interrogation simulator.

Graph flow:
  START → retrieve_context → inspector_agent → suspect_agent → profiler_agent
       → (if turn < max_turns) loop back to retrieve_context
       → (else) final_report → END

Exports:
  build_graph(rag_collection) → compiled LangGraph StateGraph
"""

from langgraph.graph import StateGraph, END
from src.state import InterrogationState
from src.agents import (
    inspector_agent,
    suspect_agent,
    profiler_agent,
    final_report_agent,
)


# ---------------------------------------------------------------------------
# Retrieve context node (wraps Person A's rag.retrieve)
# ---------------------------------------------------------------------------

def _make_retrieve_node(rag_collection):
    """Return a retrieve_context node function bound to the given collection."""

    def retrieve_context(state: InterrogationState) -> dict:
        # Build a query from  the last question + last profiler reason
        query_parts = []
        if state.get("last_question"):
            query_parts.append(state["last_question"])
        if state.get("profiler_output") and state["profiler_output"].get("reason"):
            query_parts.append(state["profiler_output"]["reason"])
        # Fallback for the very first turn
        if not query_parts:
            query_parts.append(state["case_data"].get("summary", "interrogation"))

        query = " ".join(query_parts)

        # Call RAG retrieve — gracefully degrade if collection is None
        if rag_collection is not None:
            try:
                from src.rag import retrieve
                docs = retrieve(rag_collection, query, k=3)
            except Exception:
                docs = []
        else:
            docs = []

        return {"retrieved_context": docs}

    return retrieve_context


# ---------------------------------------------------------------------------
# Conditional edge: loop or finish?
# ---------------------------------------------------------------------------

def _should_continue(state: InterrogationState) -> str:
    """Return the next node name based on the current turn count."""
    if state["turn"] < state["max_turns"]:
        return "retrieve_context"
    return "final_report"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_graph(rag_collection=None):
    """Build and compile the interrogation StateGraph.

    Parameters
    ----------
    rag_collection : chromadb Collection or None
        The ChromaDB collection to query for context.
        Pass None to run without RAG (useful for testing).

    Returns
    -------
    Compiled LangGraph application (call .invoke(initial_state) to run).
    """
    graph = StateGraph(InterrogationState)

    # --- Nodes ---
    graph.add_node("retrieve_context", _make_retrieve_node(rag_collection))
    graph.add_node("inspector_agent", inspector_agent)
    graph.add_node("suspect_agent", suspect_agent)
    graph.add_node("profiler_agent", profiler_agent)
    graph.add_node("final_report", final_report_agent)

    # --- Edges ---
    # START → retrieve_context
    graph.set_entry_point("retrieve_context")

    # Linear chain: retrieve → inspector → suspect → profiler
    graph.add_edge("retrieve_context", "inspector_agent")
    graph.add_edge("inspector_agent", "suspect_agent")
    graph.add_edge("suspect_agent", "profiler_agent")

    # Conditional: after profiler, loop or finish
    graph.add_conditional_edges(
        "profiler_agent",
        _should_continue,
        {
            "retrieve_context": "retrieve_context",
            "final_report": "final_report",
        },
    )

    # final_report → END
    graph.add_edge("final_report", END)

    return graph.compile()
