"""
Tests for src/state.py — verify the state schema has expected fields.
"""

from typing import get_type_hints, Annotated
import operator

from src.state import InterrogationState


class TestInterrogationState:

    def test_has_all_required_fields(self):
        hints = get_type_hints(InterrogationState, include_extras=True)
        expected_fields = [
            "case_data", "suspect_profile", "conversation_history",
            "retrieved_context", "profiler_context", "rag_history",
            "last_question", "last_answer", "profiler_output",
            "profiler_history", "turn", "max_turns", "final_report",
        ]
        for field in expected_fields:
            assert field in hints, f"Missing field: {field}"

    def test_append_fields_use_operator_add(self):
        """Fields that grow across turns must use Annotated[list, operator.add]."""
        hints = get_type_hints(InterrogationState, include_extras=True)
        append_fields = ["conversation_history", "profiler_history", "rag_history"]
        for field in append_fields:
            hint = hints[field]
            assert hasattr(hint, "__metadata__"), f"{field} should be Annotated"
            assert operator.add in hint.__metadata__, f"{field} should use operator.add"

    def test_non_append_fields_are_plain(self):
        """Fields that are overwritten each turn should NOT use operator.add."""
        hints = get_type_hints(InterrogationState, include_extras=True)
        overwrite_fields = ["retrieved_context", "profiler_context"]
        for field in overwrite_fields:
            hint = hints[field]
            # Plain `list` has no __metadata__
            assert not hasattr(hint, "__metadata__"), f"{field} should be plain list, not Annotated"
