"""
Tests for src/utils.py — parse_json_response, format_conversation, load_json.

"""

import json
import os
import pytest
from unittest.mock import patch, MagicMock

from src.utils import parse_json_response, format_conversation, load_json


# ---------------------------------------------------------------------------
# parse_json_response
# ---------------------------------------------------------------------------

class TestParseJsonResponse:
    """Tests for the JSON extraction helper."""

    def test_plain_json(self):
        raw = '{"stress_level": 0.7, "reason": "nervous"}'
        result = parse_json_response(raw)
        assert result["stress_level"] == 0.7
        assert result["reason"] == "nervous"

    def test_json_in_code_block(self):
        raw = '```json\n{"stress_level": 0.5}\n```'
        result = parse_json_response(raw)
        assert result["stress_level"] == 0.5

    def test_json_with_surrounding_text(self):
        raw = 'Here is the analysis:\n{"evasion_score": 0.3}\nEnd of analysis.'
        result = parse_json_response(raw)
        assert result["evasion_score"] == 0.3

    def test_malformed_returns_fallback(self):
        raw = "This is not JSON at all."
        result = parse_json_response(raw)
        assert result["reason"] == "Could not parse profiler output"
        assert result["stress_level"] == 0.5

    def test_empty_string_returns_fallback(self):
        result = parse_json_response("")
        assert "stress_level" in result

    def test_nested_json(self):
        raw = '{"score": 0.8, "details": {"a": 1, "b": 2}}'
        result = parse_json_response(raw)
        assert result["score"] == 0.8
        assert result["details"]["a"] == 1

    def test_code_block_without_lang_tag(self):
        raw = '```\n{"stress_level": 0.9}\n```'
        result = parse_json_response(raw)
        assert result["stress_level"] == 0.9


# ---------------------------------------------------------------------------
# format_conversation
# ---------------------------------------------------------------------------

class TestFormatConversation:
    """Tests for conversation history formatting."""

    def test_empty_history(self):
        result = format_conversation([])
        assert isinstance(result, str)

    def test_single_message(self):
        history = [{"role": "inspector", "content": "Where were you?"}]
        result = format_conversation(history)
        assert "inspector" in result.lower()
        assert "Where were you?" in result

    def test_multiple_messages(self):
        history = [
            {"role": "inspector", "content": "Question 1"},
            {"role": "suspect", "content": "Answer 1"},
            {"role": "inspector", "content": "Question 2"},
        ]
        result = format_conversation(history)
        assert "Question 1" in result
        assert "Answer 1" in result
        assert "Question 2" in result

    def test_preserves_order(self):
        history = [
            {"role": "inspector", "content": "First"},
            {"role": "suspect", "content": "Second"},
        ]
        result = format_conversation(history)
        assert result.index("First") < result.index("Second")


# ---------------------------------------------------------------------------
# load_json
# ---------------------------------------------------------------------------

class TestLoadJson:
    """Tests for JSON file loading."""

    def test_load_case_file(self):
        path = os.path.join("data", "cases", "case_001.json")
        data = load_json(path)
        assert "title" in data or "summary" in data
        assert isinstance(data, dict)

    def test_load_suspect_file(self):
        path = os.path.join("data", "suspects", "suspect_1_001_martin.json")
        data = load_json(path)
        assert "name" in data
        assert "hidden_truth" in data

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_json("data/cases/nonexistent.json")


# ---------------------------------------------------------------------------
# call_llm (mocked)
# ---------------------------------------------------------------------------

class TestCallLlm:
    """Tests for the LLM wrapper with mocked Gemini."""

    @patch("src.utils.get_model")
    @patch("src.utils.Langfuse")
    def test_call_returns_text(self, mock_langfuse_cls, mock_get_model):
        mock_response = MagicMock()
        mock_response.text = "This is a test response"
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_get_model.return_value = mock_model
        mock_langfuse_cls.return_value = MagicMock()

        from src.utils import call_llm
        result = call_llm("Hello", temperature=0.5)

        assert result == "This is a test response"
        mock_model.generate_content.assert_called_once()

    @patch("src.utils.get_model")
    @patch("src.utils.Langfuse")
    def test_comparison_mode_forces_temp_zero(self, mock_langfuse_cls, mock_get_model):
        import src.utils as utils_module

        mock_response = MagicMock()
        mock_response.text = "response"
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_get_model.return_value = mock_model
        mock_langfuse_cls.return_value = MagicMock()

        original = utils_module.comparison_mode
        try:
            utils_module.comparison_mode = True
            utils_module.call_llm("test", temperature=0.9)

            # Check that generate_content was called with temperature=0.0
            call_args = mock_model.generate_content.call_args
            gen_config = call_args[1].get("generation_config") or call_args[0][1]
            assert gen_config["temperature"] == 0.0
        finally:
            utils_module.comparison_mode = original
