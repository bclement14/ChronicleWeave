# tests/test_prepare_tracks.py

import json
import zipfile
from pathlib import Path

import pytest

from chronicleweave.modules.prepare_tracks import (
    DEFAULT_MAPPING_PATH,
    SpeakerRule,
    apply_speaker_mapping,
    load_speaker_mapping,
    prepare_tracks,
    resolve_display_name,
    unzip_craig_archive,
    _username_from_track,
)


# --- Fixtures ---

@pytest.fixture
def rules() -> list:
    return [
        SpeakerRule(match="player_one", name="Titar"),
        SpeakerRule(match="gamemaster", name="GM"),
        SpeakerRule(match="pen", name="Pen"),
        SpeakerRule(match="player_three", name="Khalebe"),
        SpeakerRule(match="playerfour", name="Kashkyst"),
    ]


@pytest.fixture
def mapping_file(tmp_path: Path) -> Path:
    p = tmp_path / "mapping.json"
    p.write_text(json.dumps({
        "mappings": [
            {"match": "player_one", "name": "Titar"},
            {"match": "gamemaster", "name": "GM"},
            {"match": "pen", "name": "Pen"},
            {"match": "player_three", "name": "Khalebe"},
            {"match": "playerfour", "name": "Kashkyst"},
        ]
    }), encoding="utf-8")
    return p


def _make_zip(target: Path, members: dict) -> Path:
    """Create a zip with given {name: bytes} members. Returns target path."""
    with zipfile.ZipFile(target, "w") as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return target


# --- _username_from_track ---

@pytest.mark.parametrize("stem,expected", [
    ("1-player_one", "player_one"),
    ("4-_playerfour", "playerfour"),
    ("12-gamemaster", "gamemaster"),
    ("plain", "plain"),
    ("__weird", "weird"),
])
def test_username_from_track(stem, expected):
    assert _username_from_track(stem) == expected


# --- resolve_display_name ---

def test_resolve_display_name_match(rules):
    assert resolve_display_name("1-player_one", rules) == "Titar"
    assert resolve_display_name("5-_playerfour", rules) == "Kashkyst"
    assert resolve_display_name("3-penny42", rules) == "Pen"


def test_resolve_display_name_no_match(rules):
    assert resolve_display_name("99-stranger", rules) is None


def test_resolve_display_name_case_insensitive(rules):
    assert resolve_display_name("2-GameMaster", rules) == "GM"


# --- load_speaker_mapping ---

def test_load_speaker_mapping_file(mapping_file):
    rules = load_speaker_mapping(mapping_file)
    assert len(rules) == 5
    assert rules[0] == SpeakerRule(match="player_one", name="Titar")


def test_load_speaker_mapping_missing(tmp_path):
    rules = load_speaker_mapping(tmp_path / "does_not_exist.json")
    assert rules == []


def test_load_speaker_mapping_malformed(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    assert load_speaker_mapping(p) == []


def test_load_speaker_mapping_skips_bad_entries(tmp_path):
    p = tmp_path / "mixed.json"
    p.write_text(json.dumps({
        "mappings": [
            {"match": "ok", "name": "Ok"},
            {"match": "", "name": "Empty"},  # rejected
            {"match": "missing_name"},        # rejected
            "not a dict",                      # rejected
        ]
    }), encoding="utf-8")
    rules = load_speaker_mapping(p)
    assert rules == [SpeakerRule(match="ok", name="Ok")]


def test_packaged_default_mapping_loads():
    """Sanity-check that the JSON shipped with the package is well-formed."""
    rules = load_speaker_mapping(DEFAULT_MAPPING_PATH)
    assert len(rules) >= 5
    names = {r.name for r in rules}
    assert {"Titar", "GM", "Pen", "Khalebe", "Kashkyst"} <= names


# --- apply_speaker_mapping ---

def test_apply_speaker_mapping_renames(tmp_path, rules):
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    for name in [
        "1-player_one.flac",
        "2-gamemaster.flac",
        "3-penny42.flac",
        "4-_playerfour.flac",
        "5-player_three.flac",
    ]:
        (tracks / name).write_bytes(b"")

    renamed = apply_speaker_mapping(tracks, rules)
    assert set(renamed.values()) == {"Titar", "GM", "Pen", "Kashkyst", "Khalebe"}
    final = {p.name for p in tracks.glob("*.flac")}
    assert final == {"Titar.flac", "GM.flac", "Pen.flac", "Kashkyst.flac", "Khalebe.flac"}


def test_apply_speaker_mapping_idempotent(tmp_path, rules):
    """Second run must be a no-op."""
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    (tracks / "1-player_one.flac").write_bytes(b"")
    apply_speaker_mapping(tracks, rules)
    second = apply_speaker_mapping(tracks, rules)
    assert second == {}
    assert (tracks / "Titar.flac").exists()


def test_apply_speaker_mapping_unknown_kept(tmp_path, rules, caplog):
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    (tracks / "9-stranger.flac").write_bytes(b"hi")
    renamed = apply_speaker_mapping(tracks, rules)
    assert renamed == {}
    assert (tracks / "9-stranger.flac").exists()


def test_apply_speaker_mapping_no_rules(tmp_path):
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    (tracks / "1-player_one.flac").write_bytes(b"")
    assert apply_speaker_mapping(tracks, []) == {}
    assert (tracks / "1-player_one.flac").exists()


def test_apply_speaker_mapping_collision(tmp_path, rules):
    """Two source tracks resolving to the same display name: second is left alone."""
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    (tracks / "Titar.flac").write_bytes(b"existing")
    (tracks / "1-player_one.flac").write_bytes(b"new")
    renamed = apply_speaker_mapping(tracks, rules)
    assert renamed == {}
    assert (tracks / "Titar.flac").exists()
    assert (tracks / "1-player_one.flac").exists()


# --- unzip_craig_archive ---

def test_unzip_extracts_only_flacs(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    archive = session / "craig-ABCDEF.flac.zip"
    _make_zip(archive, {
        "1-player_one.flac": b"flac1",
        "2-gamemaster.flac": b"flac2",
        "info.txt": b"meta",
        "raw.dat": b"raw",
    })

    tracks = session / "tracks"
    extracted = unzip_craig_archive(session, tracks)
    assert extracted is True
    assert {p.name for p in tracks.iterdir()} == {
        "1-player_one.flac",
        "2-gamemaster.flac",
    }


def test_unzip_skipped_when_tracks_populated(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    _make_zip(session / "craig-X.flac.zip", {"1-player_one.flac": b"x"})
    tracks = session / "tracks"
    tracks.mkdir()
    (tracks / "already.flac").write_bytes(b"existing")

    assert unzip_craig_archive(session, tracks) is False
    assert (tracks / "already.flac").exists()
    assert not (tracks / "1-player_one.flac").exists()


def test_unzip_no_archive(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    assert unzip_craig_archive(session, session / "tracks") is False


# --- prepare_tracks (integration) ---

def test_prepare_tracks_end_to_end(tmp_path, mapping_file):
    session = tmp_path / "session27"
    session.mkdir()
    _make_zip(session / "craig-XYZ.flac.zip", {
        "1-player_one.flac": b"a",
        "2-gamemaster.flac": b"b",
        "3-penny42.flac": b"c",
        "4-player_three.flac": b"d",
        "5-_playerfour.flac": b"e",
        "info.txt": b"meta",
        "raw.dat": b"raw",
    })

    extracted, renamed = prepare_tracks(session, mapping_path=mapping_file)
    assert extracted is True
    assert set(renamed.values()) == {"Titar", "GM", "Pen", "Khalebe", "Kashkyst"}

    final = {p.name for p in (session / "tracks").glob("*.flac")}
    assert final == {"Titar.flac", "GM.flac", "Pen.flac", "Khalebe.flac", "Kashkyst.flac"}


def test_prepare_tracks_idempotent(tmp_path, mapping_file):
    session = tmp_path / "session"
    session.mkdir()
    _make_zip(session / "craig-XYZ.flac.zip", {"1-player_one.flac": b"a"})

    first = prepare_tracks(session, mapping_path=mapping_file)
    second = prepare_tracks(session, mapping_path=mapping_file)
    assert first == (True, {"1-player_one": "Titar"})
    assert second == (False, {})
    assert (session / "tracks" / "Titar.flac").exists()
