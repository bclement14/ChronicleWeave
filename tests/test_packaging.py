# tests/test_packaging.py
import importlib
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_package_imports_lowercase():
    mod = importlib.import_module("chronicleweave")
    assert Path(mod.__file__).parent.name == "chronicleweave"


def test_pyproject_is_single_source():
    assert not (ROOT / "setup.py").exists()
    assert not (ROOT / "requirements.txt").exists()
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["build-system"]["build-backend"] == "setuptools.build_meta"
    project = data["project"]
    assert project["requires-python"] == ">=3.10"
    assert project["scripts"]["chronicleweave"] == "chronicleweave.cli:main"
    assert project["scripts"]["chronicleweave-batch"] == "chronicleweave.cli:batch_main"
    deps = " ".join(project["dependencies"])
    assert "python-dotenv" in deps
    assert "audioop-lts" in deps


def test_env_example_has_no_secrets():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "CW_SPEAKERS=" in text and "CW_IGNORE_TRACKS=" in text
    for forbidden in ("API_KEY", "HF_TOKEN", "TOKEN="):
        assert forbidden not in text
