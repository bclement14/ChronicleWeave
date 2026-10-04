# tests/test_llm_processor.py
"""Step 9 must fail, without writing, when the model returns no text (final review F9). No real API call."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from chronicleweave.modules.llm_processor import LLMConfig, _call_gemini_api, process_with_llm


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    script = tmp_path / "final_script.txt"
    script.write_text("[GM] bonjour", encoding="utf-8")
    out = tmp_path / "final_outputs"
    out.mkdir()
    return script, out


@pytest.mark.parametrize("reply", ["", "  \n"])
def test_empty_summary_fails_and_keeps_previous_output(tmp_path, monkeypatch, reply):
    script, out = _setup(tmp_path, monkeypatch)
    cfg = LLMConfig(enable_narrative=False, enable_cumulative_summary=False)
    previous = out / cfg.output_summary_filename
    previous.write_text("previous summary", encoding="utf-8")
    with patch("chronicleweave.modules.llm_processor._call_gemini_api", return_value=reply):
        with pytest.raises(RuntimeError, match="empty"):
            process_with_llm(script, out, cfg, SimpleNamespace(base_path=tmp_path))
    assert previous.read_text(encoding="utf-8") == "previous summary"


def test_empty_narrative_fails_without_writing(tmp_path, monkeypatch):
    script, out = _setup(tmp_path, monkeypatch)
    cfg = LLMConfig(enable_summary=False, enable_cumulative_summary=False)
    with patch("chronicleweave.modules.llm_processor._call_gemini_api", return_value=""):
        with pytest.raises(RuntimeError, match="empty"):
            process_with_llm(script, out, cfg, SimpleNamespace(base_path=tmp_path))
    assert not (out / cfg.output_narrative_filename).exists()


def test_empty_cumulative_summary_keeps_previous(tmp_path, monkeypatch):
    script, out = _setup(tmp_path, monkeypatch)
    cfg = LLMConfig(enable_narrative=False)
    cumulative = out / cfg.output_cumulative_meta_summary_filename
    cumulative.write_text("campaign so far", encoding="utf-8")
    with patch("chronicleweave.modules.llm_processor._call_gemini_api", side_effect=["session summary", ""]):
        with pytest.raises(RuntimeError, match="empty"):
            process_with_llm(script, out, cfg, SimpleNamespace(base_path=tmp_path))
    assert cumulative.read_text(encoding="utf-8") == "campaign so far"


def test_safety_blocked_candidate_raises():
    response = SimpleNamespace(text=None, prompt_feedback=None,
                               candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="SAFETY"))])
    with patch("chronicleweave.modules.llm_processor.genai.Client") as client:
        client.return_value.models.generate_content.return_value = response
        with pytest.raises(RuntimeError, match="SAFETY"):
            _call_gemini_api("test-key-not-real", "gemini-test", "prompt", 10, 0.5, None)
