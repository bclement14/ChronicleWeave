# tests/test_envfile.py
import importlib
import os
import sys
from pathlib import Path

import pytest

from chronicleweave.envfile import find_env_file, load_env


def test_explicit_env_file_wins(tmp_path):
    explicit = tmp_path / "custom.env"
    explicit.write_text("CW_TEST_A=1\n")
    (tmp_path / ".env").write_text("CW_TEST_A=2\n")
    assert find_env_file(explicit, cwd=tmp_path, repo_root=None) == explicit


def test_explicit_env_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_env_file(tmp_path / "nope.env", cwd=tmp_path, repo_root=None)


def test_walks_up_from_cwd(tmp_path):
    (tmp_path / ".env").write_text("X=1\n")
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    assert find_env_file(None, cwd=deep, repo_root=None) == tmp_path / ".env"


def test_falls_back_to_repo_root(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".env").write_text("X=1\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    # tmp_path itself has no .env, so the walk from `elsewhere` finds nothing
    assert find_env_file(None, cwd=elsewhere, repo_root=repo) == repo / ".env"


def test_none_when_nothing_found(tmp_path):
    assert find_env_file(None, cwd=tmp_path, repo_root=None) is None


def test_shell_value_wins(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("CW_TEST_SHELL=from_file\nCW_TEST_NEW=from_file\n")
    monkeypatch.setenv("CW_TEST_SHELL", "from_shell")
    monkeypatch.delenv("CW_TEST_NEW", raising=False)
    loaded = load_env(env, cwd=tmp_path, repo_root=None)
    assert loaded == env
    assert os.environ["CW_TEST_SHELL"] == "from_shell"
    assert os.environ["CW_TEST_NEW"] == "from_file"
    monkeypatch.delenv("CW_TEST_NEW", raising=False)


def test_importing_pipeline_reads_no_env(tmp_path):
    # Run in a subprocess: re-importing the module in-process would break mock.patch targets in other tests.
    import subprocess
    (tmp_path / ".env").write_text("CW_TEST_IMPORT=leaked\n")
    env = {k: v for k, v in os.environ.items() if k != "CW_TEST_IMPORT"}
    out = subprocess.run(
        [sys.executable, "-c", "import os, chronicleweave.pipeline; print(os.environ.get('CW_TEST_IMPORT'))"],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out == "None"
