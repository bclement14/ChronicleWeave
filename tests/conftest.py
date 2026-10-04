# tests/conftest.py
import os
import stat

import pytest


@pytest.fixture(autouse=True)
def _no_real_docker(tmp_path_factory, monkeypatch):
    """Put a fake `docker` first on PATH: any test that reaches the real CLI fails loudly."""
    bin_dir = tmp_path_factory.mktemp("fakebin")
    fake = bin_dir / "docker"
    fake.write_text("#!/bin/sh\necho 'fake docker: tests must not run docker' >&2\nexit 97\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))  # keeps ~/.cache/chronicleweave out of the real home
