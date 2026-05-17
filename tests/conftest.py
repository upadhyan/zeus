"""Shared pytest fixtures for the ZEUS test suite."""
import os
import pytest


@pytest.fixture
def isolated_cache_dir(tmp_path, monkeypatch):
    """Point ZEUS_CACHE_DIR at a fresh tmp_path so tests don't touch the user's real cache."""
    monkeypatch.setenv("ZEUS_CACHE_DIR", str(tmp_path))
    return tmp_path
