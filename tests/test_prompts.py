"""
Tests for src/prompts.py — verify templates have all required placeholders
and produce valid formatted strings.
"""

from src.prompts import (
    INSPECTOR_PROMPT,
    SUSPECT_PROMPT,
    PROFILER_PROMPT,
    FINAL_REPORT_PROMPT,
    JUDGE_PROMPT,
)


class TestInspectorPrompt:

    def test_has_required_placeholders(self):
        for placeholder in ["suspect_name", "case_data", "retrieved_context", "conversation_history", "profiler_output"]:
            assert f"{{{placeholder}}}" in INSPECTOR_PROMPT

    def test_formats_without_error(self):
        result = INSPECTOR_PROMPT.format(
            suspect_name="John",
            case_data="case info",
            retrieved_context="context",
            conversation_history="Q&A",
            profiler_output="{}",
        )
        assert "John" in result


class TestSuspectPrompt:

    def test_has_required_placeholders(self):
        for placeholder in ["suspect_name", "suspect_profile", "case_data", "conversation_history", "last_question"]:
            assert f"{{{placeholder}}}" in SUSPECT_PROMPT

    def test_formats_without_error(self):
        result = SUSPECT_PROMPT.format(
            suspect_name="Jane",
            suspect_profile="profile data",
            case_data="case info",
            conversation_history="history",
            last_question="Where were you?",
        )
        assert "Jane" in result
        assert "Where were you?" in result


class TestProfilerPrompt:

    def test_has_required_placeholders(self):
        for placeholder in ["case_data", "conversation_history", "last_question", "last_answer", "profiler_context"]:
            assert f"{{{placeholder}}}" in PROFILER_PROMPT

    def test_formats_without_error(self):
        result = PROFILER_PROMPT.format(
            case_data="case",
            conversation_history="convo",
            last_question="Q",
            last_answer="A",
            profiler_context="examples",
        )
        assert "Q" in result


class TestFinalReportPrompt:

    def test_has_required_placeholders(self):
        for placeholder in ["case_data", "conversation_history", "all_profiler_outputs"]:
            assert f"{{{placeholder}}}" in FINAL_REPORT_PROMPT


class TestJudgePrompt:

    def test_has_required_placeholders(self):
        for placeholder in ["case_data", "suspect_profile", "conversation_history", "all_profiler_outputs", "final_report"]:
            assert f"{{{placeholder}}}" in JUDGE_PROMPT

    def test_contains_rubric_keywords(self):
        assert "inspector_quality" in JUDGE_PROMPT
        assert "suspect_realism" in JUDGE_PROMPT
        assert "profiler_accuracy" in JUDGE_PROMPT
        assert "overall_effectiveness" in JUDGE_PROMPT
