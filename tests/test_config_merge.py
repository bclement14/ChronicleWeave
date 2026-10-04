# tests/test_config_merge.py
"""Tests for the run_pipeline config-merge helper."""

from pathlib import Path

import pytest

from chronicleweave.pipeline import _build_pipeline_config, PipelineConfig
from chronicleweave.modules.llm_processor import LLMConfig


def _explicit(**overrides):
    """Default explicit_kwargs (all None) with selected overrides applied."""
    base = {
        "steps_to_run": None, "log_level": None,
        "run_whisperx": None, "diarize": None,
        "input_audio_folderName": None,
        "script_chunk_size": None, "script_chunk_filename_template": None,
        "script_chunks_in_final_folder": None, "script_chunks_folderName": None,
        "script_chunks_compact_mode": None,
    }
    base.update(overrides)
    return base


def test_defaults_when_nothing_provided(tmp_path):
    cfg = _build_pipeline_config(
        config_obj=None, base_path=str(tmp_path),
        explicit_kwargs=_explicit(), llm_config_overrides=None, extra_kwargs={},
    )
    assert isinstance(cfg, PipelineConfig)
    assert cfg.base_path == tmp_path.resolve()
    assert cfg.input_audio_folderName == "tracks"  # default preserved
    assert cfg.diarize is False


def test_explicit_kwargs_override_defaults(tmp_path):
    cfg = _build_pipeline_config(
        config_obj=None, base_path=str(tmp_path),
        explicit_kwargs=_explicit(diarize=True, log_level="DEBUG"),
        llm_config_overrides=None, extra_kwargs={},
    )
    assert cfg.diarize is True
    assert cfg.log_level == "DEBUG"


def test_none_explicit_does_not_override_config_obj(tmp_path):
    base = PipelineConfig(diarize=True, log_level="WARNING")
    cfg = _build_pipeline_config(
        config_obj=base, base_path=str(tmp_path),
        explicit_kwargs=_explicit(),  # all None
        llm_config_overrides=None, extra_kwargs={},
    )
    assert cfg.diarize is True
    assert cfg.log_level == "WARNING"


def test_explicit_overrides_config_obj(tmp_path):
    base = PipelineConfig(diarize=True)
    cfg = _build_pipeline_config(
        config_obj=base, base_path=str(tmp_path),
        explicit_kwargs=_explicit(diarize=False),
        llm_config_overrides=None, extra_kwargs={},
    )
    assert cfg.diarize is False


def test_extra_kwargs_override_explicit(tmp_path):
    cfg = _build_pipeline_config(
        config_obj=None, base_path=str(tmp_path),
        explicit_kwargs=_explicit(diarize=False),
        llm_config_overrides=None,
        extra_kwargs={"diarize": True, "auto_prepare_tracks": False},
    )
    assert cfg.diarize is True
    assert cfg.auto_prepare_tracks is False


def test_unknown_kwargs_warned_and_ignored(tmp_path, caplog):
    import logging
    caplog.set_level(logging.WARNING, logger="chronicleweave.pipeline")
    cfg = _build_pipeline_config(
        config_obj=None, base_path=str(tmp_path),
        explicit_kwargs=_explicit(),
        llm_config_overrides=None,
        extra_kwargs={"nonexistent_field": 42},
    )
    assert isinstance(cfg, PipelineConfig)
    assert any("nonexistent_field" in r.message for r in caplog.records)


def test_base_path_resolved(tmp_path):
    cfg = _build_pipeline_config(
        config_obj=None, base_path=str(tmp_path / "sub"),
        explicit_kwargs=_explicit(), llm_config_overrides=None, extra_kwargs={},
    )
    assert cfg.base_path == (tmp_path / "sub").resolve()


def test_base_path_extra_kwargs_wins(tmp_path):
    other = tmp_path / "other"
    cfg = _build_pipeline_config(
        config_obj=None, base_path=str(tmp_path),
        explicit_kwargs=_explicit(),
        llm_config_overrides=None,
        extra_kwargs={"base_path": str(other)},
    )
    assert cfg.base_path == other.resolve()


def test_llm_config_dict_override(tmp_path):
    cfg = _build_pipeline_config(
        config_obj=None, base_path=str(tmp_path),
        explicit_kwargs=_explicit(),
        llm_config_overrides={"language": "fr", "temperature": 0.1},
        extra_kwargs={},
    )
    assert cfg.llm_config.language == "fr"
    assert cfg.llm_config.temperature == 0.1
    # Untouched fields keep defaults
    assert cfg.llm_config.api_provider == "gemini"


def test_llm_config_full_replace(tmp_path):
    custom = LLMConfig(language="de", temperature=0.0)
    cfg = _build_pipeline_config(
        config_obj=None, base_path=str(tmp_path),
        explicit_kwargs=_explicit(),
        llm_config_overrides=custom, extra_kwargs={},
    )
    assert cfg.llm_config is custom


def test_invalid_config_obj_ignored(tmp_path, caplog):
    import logging
    caplog.set_level(logging.WARNING, logger="chronicleweave.pipeline")
    cfg = _build_pipeline_config(
        config_obj="not a config",  # bogus
        base_path=str(tmp_path),
        explicit_kwargs=_explicit(),
        llm_config_overrides=None, extra_kwargs={},
    )
    assert isinstance(cfg, PipelineConfig)
    assert any("config_obj" in r.message for r in caplog.records)
