"""
LangGraph graph definition.

This wires up the interrogation loop:
  START → retrieve_context → inspector → suspect → profiler
       → loop back if we haven't hit max_turns yet
       → otherwise generate the final report → END
"""

from langgraph.graph import StateGraph, END
from src.state import InterrogationState
from src.agents import (
    inspector_agent,
    suspect_agent,
    profiler_agent,
    final_report_agent,
)


# -- Retrieve context node (closure so we can pass the RAG collection in) --

def _make_retrieve_node(rag_collection):
    """Creates the retrieve_context node with access to the RAG collection."""

    def retrieve_context(state: InterrogationState) -> dict:
        # Use last question + profiler reason as search query
        query_parts = []
        if state.get("last_question"):
            query_parts.append(state["last_question"])
        if state.get("profiler_output") and state["profiler_output"].get("reason"):
            query_parts.append(state["profiler_output"]["reason"])
        # First turn: we don't have anything yet, so just use the case summary
        if not query_parts:
            query_parts.append(state["case_data"].get("summary", "interrogation"))

        query = " ".join(query_parts)

        # Try to get docs from RAG, skip if no collection is set up yet
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


# -- After profiling: do we keep going or wrap up? --

def _should_continue(state: InterrogationState) -> str:
    """Decides whether we loop for another turn or go to the final report."""
    if state["turn"] < state["max_turns"]:
        return "retrieve_context"
    return "final_report"


# -- Main entry point --

def build_graph(rag_collection=None):
    """Build the full interrogation graph.

    Pass a ChromaDB collection for RAG, or None to skip it (handy for testing).
    Returns a compiled graph — just call .invoke(initial_state) to run it.
    """
    graph = StateGraph(InterrogationState)

    # Register all the nodes
    graph.add_node("retrieve_context", _make_retrieve_node(rag_collection))
    graph.add_node("inspector_agent", inspector_agent)
    graph.add_node("suspect_agent", suspect_agent)
    graph.add_node("profiler_agent", profiler_agent)
    graph.add_node("final_report", final_report_agent)

    # Wire them up: retrieve → inspector → suspect → profiler
    graph.set_entry_point("retrieve_context")
    graph.add_edge("retrieve_context", "inspector_agent")
    graph.add_edge("inspector_agent", "suspect_agent")
    graph.add_edge("suspect_agent", "profiler_agent")

    # After profiler: loop back or generate the final report
    graph.add_conditional_edges(
        "profiler_agent",
        _should_continue,
        {
            "retrieve_context": "retrieve_context",
            "final_report": "final_report",
        },
    )

    # Done → end
    graph.add_edge("final_report", END)

    return graph.compile()
