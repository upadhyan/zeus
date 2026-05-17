"""Weight download + cache + state-dict load semantics."""
from __future__ import annotations
import hashlib
import io
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

from zeus import weights as W


def _make_fake_state_dict(tmp_path: Path, *, wrapped: bool = True) -> Path:
    """Save a minimal fake checkpoint and return its path."""
    sd = {"foo.weight": torch.zeros(2, 2)}
    obj = {"model": sd} if wrapped else sd
    path = tmp_path / "fake.pt"
    torch.save(obj, path)
    return path


# --- get_checkpoint_path -----------------------------------------------------

def test_explicit_model_path_is_returned_verbatim(tmp_path):
    p = tmp_path / "custom.pt"
    p.write_bytes(b"")
    assert W.get_checkpoint_path(model_path=p) == p


def test_cache_dir_kwarg_overrides_env(tmp_path, monkeypatch):
    """`cache_dir=` arg beats $ZEUS_CACHE_DIR which beats platformdirs."""
    monkeypatch.setenv("ZEUS_CACHE_DIR", str(tmp_path / "env"))
    expected = tmp_path / "kwarg" / "zeus.pt"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"abc")
    sha = hashlib.sha256(b"abc").hexdigest()
    with patch.object(W, "EXPECTED_SHA256", sha):
        result = W.get_checkpoint_path(cache_dir=tmp_path / "kwarg")
    assert result == expected


def test_env_var_used_when_no_kwarg(tmp_path, monkeypatch):
    cache = tmp_path / "envcache"
    cache.mkdir()
    (cache / "zeus.pt").write_bytes(b"xyz")
    sha = hashlib.sha256(b"xyz").hexdigest()
    monkeypatch.setenv("ZEUS_CACHE_DIR", str(cache))
    with patch.object(W, "EXPECTED_SHA256", sha):
        result = W.get_checkpoint_path()
    assert result == cache / "zeus.pt"


def test_existing_file_with_wrong_checksum_triggers_redownload(isolated_cache_dir, monkeypatch):
    """If the cached file's hash differs from EXPECTED_SHA256, it's redownloaded."""
    cached = isolated_cache_dir / "zeus.pt"
    cached.write_bytes(b"corrupt")
    good_bytes = b"fresh content"

    def fake_download(url, dest_tmp):
        Path(dest_tmp).write_bytes(good_bytes)

    monkeypatch.setattr(W, "_stream_download", fake_download)
    monkeypatch.setattr(W, "EXPECTED_SHA256", hashlib.sha256(good_bytes).hexdigest())
    result = W.get_checkpoint_path()
    assert result == cached
    assert cached.read_bytes() == good_bytes


def test_existing_file_with_correct_checksum_is_not_redownloaded(isolated_cache_dir, monkeypatch):
    good_bytes = b"already-correct"
    cached = isolated_cache_dir / "zeus.pt"
    cached.write_bytes(good_bytes)
    monkeypatch.setattr(W, "EXPECTED_SHA256", hashlib.sha256(good_bytes).hexdigest())

    def boom(url, dest_tmp):
        raise AssertionError("should not be called")

    monkeypatch.setattr(W, "_stream_download", boom)
    result = W.get_checkpoint_path()
    assert result == cached


# --- load_zeus_state_dict ----------------------------------------------------

def test_load_state_dict_unwraps_model_key(tmp_path):
    path = _make_fake_state_dict(tmp_path, wrapped=True)
    sd = W._load_raw_state_dict(path, map_location="cpu")
    assert "foo.weight" in sd


def test_load_state_dict_accepts_raw(tmp_path):
    path = _make_fake_state_dict(tmp_path, wrapped=False)
    sd = W._load_raw_state_dict(path, map_location="cpu")
    assert "foo.weight" in sd
