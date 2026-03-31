"""
Tests for src/graph.py — verify graph structure and flow logic.

LLM calls are mocked so we test the wiring, not the model.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.graph import build_graph, _should_continue


# ---------------------------------------------------------------------------
# _should_continue
# ---------------------------------------------------------------------------

class TestShouldContinue:
    """Test the conditional edge logic."""

    def test_continues_when_turns_remain(self):
        state = {"turn": 2, "max_turns": 5}
        assert _should_continue(state) == "retrieve_context"

    def test_stops_at_max_turns(self):
        state = {"turn": 5, "max_turns": 5}
        assert _should_continue(state) == "final_report"

    def test_stops_when_over_max(self):
        state = {"turn": 10, "max_turns": 3}
        assert _should_continue(state) == "final_report"

    def test_single_turn(self):
        state = {"turn": 0, "max_turns": 1}
        assert _should_continue(state) == "retrieve_context"

    def test_zero_turns_means_immediate_report(self):
        state = {"turn": 0, "max_turns": 0}
        assert _should_continue(state) == "final_report"


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------

class TestBuildGraph:
    """Test that the graph compiles and has the expected nodes."""

    def test_compiles_without_rag(self):
        graph = build_graph(rag_collection=None)
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        graph = build_graph(rag_collection=None)
        node_names = set(graph.get_graph().nodes.keys())
        expected = {"retrieve_context", "inspector_agent", "suspect_agent", "profiler_agent", "final_report"}
        assert expected.issubset(node_names)
