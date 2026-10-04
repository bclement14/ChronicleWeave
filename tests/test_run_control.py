# tests/test_run_control.py
import json
from pathlib import Path

import pytest

from chronicleweave.modules.prepare_tracks import parse_speaker_config
from chronicleweave.pipeline import (
    DONE_MARKER_NAME,
    PipelineConfig,
    PipelineError,
    StepRange,
    check_coverage,
    check_tracks_folder,
    check_final_script,
    check_step_inputs,
    cleanup_outputs,
    expected_chunk_stems,
    parse_steps,
    step_output_paths,
)

SPK = parse_speaker_config("player_one:Titar,gamemaster:GM", "Spoticord")


@pytest.mark.parametrize("spec,expected", [
    (None, (1, 8)), ("1-7", (1, 7)), ("5", (5, 5)), (3, (3, 3)),
    (slice(1, 8), (1, 7)), ([2, 3, 4], (2, 4)), (StepRange(2, 6), (2, 6)),
])
def test_parse_steps(spec, expected):
    assert tuple(parse_steps(spec)) == expected


@pytest.mark.parametrize("spec", ["0-3", "3-10", "5-2", [1, 3], "a-b", []])
def test_parse_steps_invalid(spec):
    with pytest.raises(ValueError):
        parse_steps(spec)


def _layout(tmp_path):
    for d in ("tracks", "chunked_tracks", "wx_output", "json_files", "srt_files", "final_outputs/script_chunks"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    (tmp_path / "tracks" / "GM.flac").write_bytes(b"x")
    (tmp_path / "chunked_tracks" / "GM-01.flac").write_bytes(b"x")
    for name in ("merged_audio.flac", "cut_points.txt", "merged_transcript.srt",
                 "cleaned_transcript.srt", "final_script.txt"):
        (tmp_path / "final_outputs" / name).write_text("x")
    (tmp_path / "craig-A.flac.zip").write_bytes(b"zip")
    return PipelineConfig(base_path=tmp_path)


def test_cleanup_from_step_2_keeps_step_1_outputs(tmp_path):
    cfg = _layout(tmp_path)
    cleanup_outputs(cfg, 2)
    assert (tmp_path / "chunked_tracks").exists()
    assert (tmp_path / "final_outputs" / "cut_points.txt").exists()
    for gone in ("wx_output", "json_files", "srt_files", "final_outputs/script_chunks",
                 "final_outputs/final_script.txt", "final_outputs/merged_transcript.srt"):
        assert not (tmp_path / gone).exists()
    assert (tmp_path / "tracks" / "GM.flac").exists()
    assert (tmp_path / "craig-A.flac.zip").exists()


def test_cleanup_manual_mode_keeps_wx_output(tmp_path):
    cfg = _layout(tmp_path)
    cfg = PipelineConfig(base_path=tmp_path, run_whisperx=False)
    cleanup_outputs(cfg, 2)
    assert (tmp_path / "wx_output").exists()


def test_cleanup_guard_refuses_tracks_overlap(tmp_path):
    _layout(tmp_path)
    cfg = PipelineConfig(base_path=tmp_path, script_chunks_in_final_folder=False, script_chunks_folderName="tracks")
    with pytest.raises(PipelineError, match="tracks"):
        cleanup_outputs(cfg, 1)
    assert (tmp_path / "tracks" / "GM.flac").exists()
    assert (tmp_path / "chunked_tracks").exists()  # nothing deleted before the guard


def test_cleanup_guard_refuses_outside_session(tmp_path):
    session = tmp_path / "s"
    session.mkdir()
    cfg = PipelineConfig(base_path=session, srt_folderName="../elsewhere")
    with pytest.raises(PipelineError, match="outside"):
        cleanup_outputs(cfg, 4)


def test_step_inputs_missing_fails_before_cleanup(tmp_path):
    cfg = PipelineConfig(base_path=tmp_path)
    (tmp_path / "chunked_tracks").mkdir()
    with pytest.raises(PipelineError, match="step 2"):
        check_step_inputs(cfg, 2)


def test_coverage_detects_missing_and_extra(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "GM-01.json").write_text("{}")
    (tmp_path / "out" / "Old-01.json").write_text("{}")
    with pytest.raises(PipelineError, match="Pen-01"):
        check_coverage(3, tmp_path / "out", ["GM-01", "Pen-01"], ".json")


def test_coverage_uses_existing_chunk_files(tmp_path):
    cfg = PipelineConfig(base_path=tmp_path)
    chunks = tmp_path / "chunked_tracks"
    chunks.mkdir()
    for n in ("GM-01", "GM-02", "Pen-01"):  # Pen's track ended before chunk 2
        (chunks / f"{n}.flac").write_bytes(b"x")
    assert expected_chunk_stems(cfg) == ["GM-01", "GM-02", "Pen-01"]


def test_expected_chunk_stems_reports_calling_step(tmp_path):
    cfg = PipelineConfig(base_path=tmp_path)
    with pytest.raises(PipelineError) as err:
        expected_chunk_stems(cfg, 4)
    assert err.value.step == 4


def _put(tmp_path, *names):
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")


def test_step_inputs_start_3_needs_chunks_for_coverage(tmp_path):
    cfg = PipelineConfig(base_path=tmp_path)
    _put(tmp_path, "wx_output/GM-01.json")
    with pytest.raises(PipelineError, match="chunked_tracks"):
        check_step_inputs(cfg, 3, 3)


def test_step_inputs_cut_points_only_when_step_5_in_range(tmp_path):
    cfg = PipelineConfig(base_path=tmp_path)
    _put(tmp_path, "wx_output/GM-01.json", "chunked_tracks/GM-01.flac")
    check_step_inputs(cfg, 3, 4)
    with pytest.raises(PipelineError, match="cut_points"):
        check_step_inputs(cfg, 3, 5)


def test_step_inputs_start_7_needs_json_and_chunks(tmp_path):
    cfg = PipelineConfig(base_path=tmp_path)
    _put(tmp_path, "final_outputs/cleaned_transcript.srt", "chunked_tracks/GM-01.flac")
    with pytest.raises(PipelineError, match="json_files"):
        check_step_inputs(cfg, 7, 7)
    _put(tmp_path, "json_files/GM-01.json")
    check_step_inputs(cfg, 7, 8)


def test_step_inputs_start_5_with_step_7_needs_json(tmp_path):
    cfg = PipelineConfig(base_path=tmp_path)
    _put(tmp_path, "srt_files/GM-01.srt", "final_outputs/cut_points.txt", "chunked_tracks/GM-01.flac")
    check_step_inputs(cfg, 5, 6)
    with pytest.raises(PipelineError, match="json_files"):
        check_step_inputs(cfg, 5, 7)


def test_step_inputs_manual_mode_uses_wx_output_as_expected_set(tmp_path):
    cfg = PipelineConfig(base_path=tmp_path, run_whisperx=False)
    _put(tmp_path, "json_files/GM-01.json")
    with pytest.raises(PipelineError, match="wx_output"):
        check_step_inputs(cfg, 4, 4)
    _put(tmp_path, "wx_output/GM-01.json")
    check_step_inputs(cfg, 4, 4)


def test_step_inputs_step_9_needs_final_script(tmp_path):
    cfg = PipelineConfig(base_path=tmp_path)
    with pytest.raises(PipelineError, match="final_script"):
        check_step_inputs(cfg, 9, 9)


@pytest.mark.parametrize("in_final,folder", [(False, "final_outputs"), (True, ".")])
def test_cleanup_guard_refuses_final_outputs_folder(tmp_path, in_final, folder):
    _layout(tmp_path)
    cfg = PipelineConfig(base_path=tmp_path, script_chunks_in_final_folder=in_final, script_chunks_folderName=folder)
    with pytest.raises(PipelineError, match="final outputs"):
        cleanup_outputs(cfg, 1)
    assert (tmp_path / "final_outputs" / "final_script.txt").exists()
    assert (tmp_path / "chunked_tracks").exists()  # nothing deleted before the guard


def _final(tmp_path, script, jsons):
    (tmp_path / "final_outputs").mkdir(exist_ok=True)
    (tmp_path / "final_outputs" / "final_script.txt").write_text(script, encoding="utf-8")
    (tmp_path / "json_files").mkdir(exist_ok=True)
    for stem, segs in jsons.items():
        (tmp_path / "json_files" / f"{stem}.json").write_text(json.dumps({"segments": segs}))
    return PipelineConfig(base_path=tmp_path)


SEG = [{"start": 0, "end": 1, "text": "salut"}]


def test_final_gate_ok(tmp_path):
    cfg = _final(tmp_path, "[GM] salut\n\n[Titar] oui", {"GM-01": SEG, "Titar-01": SEG})
    check_final_script(cfg, ["GM-01", "Titar-01"], SPK)


def test_final_gate_empty_script(tmp_path):
    cfg = _final(tmp_path, "  \n", {"GM-01": SEG})
    with pytest.raises(PipelineError, match="empty"):
        check_final_script(cfg, ["GM-01"], SPK)


def test_final_gate_unknown_tag(tmp_path):
    cfg = _final(tmp_path, "[1-player_one] salut", {"GM-01": SEG})
    with pytest.raises(PipelineError, match="1-player_one"):
        check_final_script(cfg, ["GM-01"], SPK)


def test_final_gate_missing_speaker_with_speech(tmp_path):
    cfg = _final(tmp_path, "[GM] salut", {"GM-01": SEG, "Titar-01": SEG})
    with pytest.raises(PipelineError, match="Titar"):
        check_final_script(cfg, ["GM-01", "Titar-01"], SPK)


def test_final_gate_silent_speaker_allowed(tmp_path):
    cfg = _final(tmp_path, "[GM] salut", {"GM-01": SEG, "Titar-01": []})
    check_final_script(cfg, ["GM-01", "Titar-01"], SPK)


# --- final review F1: the chunk template is a plain file name, checked when the config is built ---

@pytest.mark.parametrize("template", ["../../tracks/GM.flac", "sub/x_{:02d}.txt", "..\\x_{:02d}.txt",
                                      "x..{:02d}.txt", ""])
def test_chunk_template_rejected_at_config_time(tmp_path, template):
    with pytest.raises(ValueError, match="template"):
        PipelineConfig(base_path=tmp_path, script_chunk_filename_template=template)


def test_chunk_template_default_accepted(tmp_path):
    assert PipelineConfig(base_path=tmp_path).script_chunk_filename_template == "script_chunk_{:02d}.txt"


@pytest.mark.parametrize("folder", ["../other/tracks", ".", "..", "/tmp"])
def test_tracks_folder_must_be_inside_session(tmp_path, folder):
    session = tmp_path / "s"
    session.mkdir()
    with pytest.raises(PipelineError, match="tracks folder"):
        check_tracks_folder(PipelineConfig(base_path=session, input_audio_folderName=folder))


def test_tracks_folder_nested_inside_session_ok(tmp_path):
    check_tracks_folder(PipelineConfig(base_path=tmp_path, input_audio_folderName="audio/tracks"))


# --- final review F2: never delete a folder that holds a Craig archive anywhere below it ---

@pytest.mark.parametrize("archive", ["saved_archives/craig-X.flac.zip", "saved_archives/deep/Craig-Y.FLAC.ZIP"])
def test_cleanup_guard_refuses_folder_containing_an_archive(tmp_path, archive):
    _layout(tmp_path)
    _put(tmp_path, archive)
    cfg = PipelineConfig(base_path=tmp_path, script_chunks_in_final_folder=False,
                         script_chunks_folderName="saved_archives")
    with pytest.raises(PipelineError, match="archive"):
        cleanup_outputs(cfg, 1)
    assert (tmp_path / archive).exists()
    assert (tmp_path / "chunked_tracks").exists()  # nothing deleted before the guard


# --- final review F3: paths that differ only by letter case are the same folder on NTFS ---

@pytest.mark.parametrize("folder,match", [
    ("Tracks", "tracks"), ("TRACKS/sub", "tracks"), ("FINAL_OUTPUTS", "final outputs"),
])
def test_cleanup_guard_ignores_letter_case(tmp_path, folder, match):
    _layout(tmp_path)
    _put(tmp_path, f"{folder}/keep.flac")
    cfg = PipelineConfig(base_path=tmp_path, script_chunks_in_final_folder=False, script_chunks_folderName=folder)
    with pytest.raises(PipelineError, match=match):
        cleanup_outputs(cfg, 1)
    assert (tmp_path / folder / "keep.flac").exists()
    assert (tmp_path / "chunked_tracks").exists()


def test_cleanup_guard_case_variant_containing_tracks(tmp_path):
    _layout(tmp_path)
    _put(tmp_path, "audio/tracks/GM.flac")
    cfg = PipelineConfig(base_path=tmp_path, input_audio_folderName="audio/tracks",
                         script_chunks_in_final_folder=False, script_chunks_folderName="AUDIO")
    with pytest.raises(PipelineError, match="tracks"):
        cleanup_outputs(cfg, 8)
    assert (tmp_path / "audio" / "tracks" / "GM.flac").exists()
