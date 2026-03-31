"""
Tests for src/agents.py — verify each agent builds the right prompt
and returns correct state update keys.

All LLM calls are mocked.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from src.utils import load_json


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_state():
    """Minimal InterrogationState for testing agents."""
    return {
        "case_data": {"title": "Test Case", "summary": "A test heist", "key_facts": ["fact1"]},
        "suspect_profile": {
            "name": "John Doe",
            "role": "coupable",
            "occupation": "Engineer",
            "personality": "Calm",
            "hidden_truth": "Did it",
            "strategy": "Deny everything",
            "vulnerabilities": ["Nervous when pressed on timeline"],
        },
        "conversation_history": [],
        "retrieved_context": ["Use open-ended questions to start."],
        "profiler_context": ["Question: Where were you? | Réponse: I was home."],
        "rag_history": [],
        "last_question": "",
        "last_answer": "",
        "profiler_output": {},
        "profiler_history": [],
        "turn": 0,
        "max_turns": 3,
        "final_report": "",
    }


# ---------------------------------------------------------------------------
# Inspector agent
# ---------------------------------------------------------------------------

class TestInspectorAgent:

    @patch("src.agents.call_llm", return_value="Where were you on the night of the heist?")
    def test_returns_question_and_history(self, mock_llm, sample_state):
        from src.agents import inspector_agent
        result = inspector_agent(sample_state)

        assert "last_question" in result
        assert result["last_question"] == "Where were you on the night of the heist?"
        assert len(result["conversation_history"]) == 1
        assert result["conversation_history"][0]["role"] == "inspector"

    @patch("src.agents.call_llm", return_value="Follow-up question?")
    def test_prompt_contains_suspect_name(self, mock_llm, sample_state):
        from src.agents import inspector_agent
        inspector_agent(sample_state)

        prompt = mock_llm.call_args[0][0]
        assert "John Doe" in prompt


# ---------------------------------------------------------------------------
# Suspect agent
# ---------------------------------------------------------------------------

class TestSuspectAgent:

    @patch("src.agents.call_llm", return_value="I was at home all evening.")
    def test_returns_answer_and_history(self, mock_llm, sample_state):
        from src.agents import suspect_agent
        sample_state["last_question"] = "Where were you?"
        result = suspect_agent(sample_state)

        assert "last_answer" in result
        assert result["last_answer"] == "I was at home all evening."
        assert result["conversation_history"][0]["role"] == "suspect"

    @patch("src.agents.call_llm", return_value="I don't know.")
    def test_prompt_contains_hidden_profile(self, mock_llm, sample_state):
        from src.agents import suspect_agent
        sample_state["last_question"] = "Tell me about your alibi."
        suspect_agent(sample_state)

        prompt = mock_llm.call_args[0][0]
        assert "hidden_truth" in prompt
        assert "strategy" in prompt


# ---------------------------------------------------------------------------
# Profiler agent
# ---------------------------------------------------------------------------

class TestProfilerAgent:

    @patch("src.agents.call_llm", return_value='{"stress_level":0.6,"evasion_score":0.3,"consistency_score":0.9,"suspicion_score":0.4,"reason":"Calm"}')
    def test_returns_metrics_and_increments_turn(self, mock_llm, sample_state):
        from src.agents import profiler_agent
        sample_state["last_question"] = "Where were you?"
        sample_state["last_answer"] = "At home."
        result = profiler_agent(sample_state)

        assert "profiler_output" in result
        assert result["profiler_output"]["stress_level"] == 0.6
        assert result["turn"] == 1  # was 0, now incremented
        assert len(result["profiler_history"]) == 1

    @patch("src.agents.call_llm", return_value="not valid json")
    def test_handles_malformed_llm_output(self, mock_llm, sample_state):
        from src.agents import profiler_agent
        sample_state["last_question"] = "Q"
        sample_state["last_answer"] = "A"
        result = profiler_agent(sample_state)

        # Should get fallback values instead of crashing
        assert "profiler_output" in result
        assert "stress_level" in result["profiler_output"]


# ---------------------------------------------------------------------------
# Final report agent
# ---------------------------------------------------------------------------

class TestFinalReportAgent:

    @patch("src.agents.call_llm", return_value="## Summary\nThe suspect was evasive.")
    def test_returns_report(self, mock_llm, sample_state):
        from src.agents import final_report_agent
        result = final_report_agent(sample_state)

        assert "final_report" in result
        assert "Summary" in result["final_report"]


# ---------------------------------------------------------------------------
# Judge agent
# ---------------------------------------------------------------------------

class TestJudgeAgent:

    @patch("src.agents.call_llm", return_value='{"inspector_quality":0.8,"suspect_realism":0.7,"profiler_accuracy":0.6,"overall_effectiveness":0.5,"reasoning":"Good session."}')
    def test_returns_scores(self, mock_llm, sample_state):
        from src.agents import judge_agent
        sample_state["final_report"] = "## Summary\nDone."
        result = judge_agent(sample_state)

        assert result["inspector_quality"] == 0.8
        assert "reasoning" in result
