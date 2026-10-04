# tests/test_prepare_tracks.py
import zipfile
from pathlib import Path

import pytest

from chronicleweave.modules.prepare_tracks import (
    PrepareResult,
    SpeakerConfig,
    TrackPreparationError,
    classify_track,
    find_craig_archive,
    parse_speaker_config,
    prepare_tracks,
    speaker_config_from_env,
    username_from_track,
)

SPEAKERS = "player_one:Titar,gamemaster:GM,penny42:Pen,player_three:Khalebe,playerfour:Kashkyst"


@pytest.fixture
def cfg() -> SpeakerConfig:
    return parse_speaker_config(SPEAKERS, "Spoticord")


def _zip(session: Path, entries: dict, name: str = "craig-ABC.flac.zip") -> Path:
    path = session / name
    with zipfile.ZipFile(path, "w") as zf:
        for entry, payload in entries.items():
            zf.writestr(entry, payload)
        zf.writestr("info.txt", "meta")
        zf.writestr("raw.dat", b"\x00")
    return path


# --- parsing ---

def test_parse_valid(cfg):
    assert cfg.speakers["player_one"] == "Titar"
    assert cfg.ignore == ("spoticord",)
    assert cfg.tags == frozenset({"Titar", "GM", "Pen", "Khalebe", "Kashkyst"})


def test_parse_ignores_whitespace_and_empty_tokens():
    c = parse_speaker_config(" a:A , ,b:B,", " Spoticord , ,")
    assert dict(c.speakers) == {"a": "A", "b": "B"}
    assert c.ignore == ("spoticord",)


@pytest.mark.parametrize("value", [None, "", "  ", "nocolon", "a:", ":A", "a:A,a:B", "a:Tag,b:tag"])
def test_parse_invalid_raises(value):
    with pytest.raises(TrackPreparationError):
        parse_speaker_config(value, None)


def test_from_env_reads_variables():
    c = speaker_config_from_env({"CW_SPEAKERS": "x:X", "CW_IGNORE_TRACKS": "bot"})
    assert dict(c.speakers) == {"x": "X"} and c.ignore == ("bot",)


def test_from_env_missing_raises():
    with pytest.raises(TrackPreparationError, match="CW_SPEAKERS"):
        speaker_config_from_env({})


# --- classification ---

def test_username_strips_index_and_underscores():
    assert username_from_track("4-_playerfour") == "playerfour"
    assert username_from_track("1-player_one") == "player_one"


def test_classify(cfg):
    assert classify_track("2-Spoticord_Music_4270", cfg) == ("ignored", None)
    assert classify_track("Titar", cfg) == ("tag", "Titar")
    assert classify_track("4-_PlayerFour", cfg) == ("mapped", "Kashkyst")
    assert classify_track("9-stranger", cfg) == ("unknown", None)


def test_classify_exact_match_not_substring(cfg):
    # "pen" must not match "penny42"-like substrings any more
    c = parse_speaker_config("pen:Pen", None)
    assert classify_track("3-penny42", c) == ("unknown", None)


def test_ignore_wins_over_mapping():
    c = parse_speaker_config("spoticord_music_4270:Music", "Spoticord")
    assert classify_track("1-Spoticord_Music_4270", c) == ("ignored", None)


# --- archive discovery ---

def test_find_archive_first_sorted_and_skips_dirs(tmp_path, caplog):
    (tmp_path / "craig-AAA.flac").mkdir()  # Windows-extracted folder, not an archive
    (tmp_path / "craig-ZZZ.flac.zip").write_bytes(b"x")
    (tmp_path / "craig-BBB.flac.zip").write_bytes(b"x")
    assert find_craig_archive(tmp_path).name == "craig-BBB.flac.zip"
    assert "Multiple Craig archives" in caplog.text


def test_find_archive_ignores_directory_named_like_archive(tmp_path):
    (tmp_path / "craig-AAA.flac.zip").mkdir()
    assert find_craig_archive(tmp_path) is None


# --- prepare_tracks ---

def test_extract_rename_and_skip_ignored(tmp_path, cfg):
    _zip(tmp_path, {"1-player_one.flac": b"aa", "2-Spoticord_Music_4270.flac": b"bb", "3-penny42.flac": b"cc"})
    result = prepare_tracks(tmp_path, cfg)
    tracks = tmp_path / "tracks"
    assert sorted(p.name for p in tracks.iterdir()) == ["Pen.flac", "Titar.flac"]
    assert (tracks / "Titar.flac").read_bytes() == b"aa"
    assert isinstance(result, PrepareResult)
    assert sorted(result.tags) == ["Pen", "Titar"]


def test_idempotent(tmp_path, cfg):
    _zip(tmp_path, {"1-player_one.flac": b"aa"})
    prepare_tracks(tmp_path, cfg)
    second = prepare_tracks(tmp_path, cfg)
    assert second.extracted == [] and second.renamed == {} and second.deleted == []


def test_already_renamed_tracks_accepted(tmp_path, cfg):
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    (tracks / "GM.flac").write_bytes(b"gm")
    (tracks / "info.txt").write_text("kept")
    result = prepare_tracks(tmp_path, cfg)
    assert result.tags == ["GM"]
    assert (tracks / "info.txt").exists()


def test_raw_names_in_tracks_plus_archive_no_duplicates(tmp_path, cfg):
    _zip(tmp_path, {"1-player_one.flac": b"aa", "2-gamemaster.flac": b"bb"})
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    (tracks / "1-player_one.flac").write_bytes(b"aa")
    prepare_tracks(tmp_path, cfg)
    assert sorted(p.name for p in tracks.glob("*.flac")) == ["GM.flac", "Titar.flac"]


def test_ignored_track_in_tracks_is_deleted(tmp_path, cfg):
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    (tracks / "3-Spoticord_Music_4270.flac").write_bytes(b"music")
    (tracks / "GM.flac").write_bytes(b"gm")
    result = prepare_tracks(tmp_path, cfg)
    assert not (tracks / "3-Spoticord_Music_4270.flac").exists()
    assert result.deleted == ["3-Spoticord_Music_4270.flac"]


def test_archive_is_never_modified(tmp_path, cfg):
    archive = _zip(tmp_path, {"1-player_one.flac": b"aa", "2-Spoticord_Music_4270.flac": b"bb"})
    before = archive.read_bytes()
    prepare_tracks(tmp_path, cfg)
    assert archive.read_bytes() == before


def test_unknown_track_fails(tmp_path, cfg):
    _zip(tmp_path, {"7-guest.flac": b"zz"})
    with pytest.raises(TrackPreparationError, match="7-guest.flac"):
        prepare_tracks(tmp_path, cfg)


def test_unknown_in_tracks_fails(tmp_path, cfg):
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    (tracks / "7-guest.flac").write_bytes(b"zz")
    with pytest.raises(TrackPreparationError, match="CW_SPEAKERS or CW_IGNORE_TRACKS"):
        prepare_tracks(tmp_path, cfg)


def test_rename_collision_fails(tmp_path, cfg):
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    (tracks / "1-player_one.flac").write_bytes(b"aa")
    (tracks / "Titar.flac").write_bytes(b"other")
    with pytest.raises(TrackPreparationError, match="already exists"):
        prepare_tracks(tmp_path, cfg)


def test_interrupted_extraction_recovered(tmp_path, cfg):
    _zip(tmp_path, {"1-player_one.flac": b"aa", "2-gamemaster.flac": b"bb"})
    tracks = tmp_path / "tracks"
    leftover = tracks / ".extract-tmp"
    leftover.mkdir(parents=True)
    (leftover / "2-gamemaster.flac").write_bytes(b"b")  # partial file from a crash
    prepare_tracks(tmp_path, cfg)
    assert not leftover.exists()
    assert (tracks / "GM.flac").read_bytes() == b"bb"
    assert (tracks / "Titar.flac").read_bytes() == b"aa"


def test_no_tracks_at_all_fails(tmp_path, cfg):
    with pytest.raises(TrackPreparationError, match="no speaker track"):
        prepare_tracks(tmp_path, cfg)


def test_validate_only_mode_rejects_raw_names(tmp_path, cfg):
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    (tracks / "1-player_one.flac").write_bytes(b"aa")
    with pytest.raises(TrackPreparationError, match="not renamed"):
        prepare_tracks(tmp_path, cfg, rename=False)
    assert (tracks / "1-player_one.flac").exists()


def test_validate_only_mode_accepts_tags(tmp_path, cfg):
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    (tracks / "GM.flac").write_bytes(b"gm")
    assert prepare_tracks(tmp_path, cfg, rename=False).tags == ["GM"]


def test_duplicate_tag_in_archive_fails_and_extracts_nothing(tmp_path, cfg):
    _zip(tmp_path, {"1-gamemaster.flac": b"aa", "7-gamemaster.flac": b"bb", "2-player_one.flac": b"cc"})
    with pytest.raises(TrackPreparationError, match="1-gamemaster.flac.*7-gamemaster.flac.*GM"):
        prepare_tracks(tmp_path, cfg)
    tracks = tmp_path / "tracks"
    assert not tracks.exists() or list(tracks.glob("*.flac")) == []


def test_validate_only_keeps_extract_leftover(tmp_path, cfg):
    tracks = tmp_path / "tracks"
    leftover = tracks / ".extract-tmp"
    leftover.mkdir(parents=True)
    (tracks / "GM.flac").write_bytes(b"gm")
    assert prepare_tracks(tmp_path, cfg, rename=False).tags == ["GM"]
    assert leftover.exists()
