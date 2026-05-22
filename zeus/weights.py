"""Download, cache, and load the released ZEUS checkpoint.

Resolution order for the checkpoint path (see spec §5.5):
  1. explicit `model_path=` argument
  2. `cache_dir=` argument
  3. $ZEUS_CACHE_DIR
  4. platformdirs.user_cache_dir("zeus")
"""
from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
import warnings
from pathlib import Path
from typing import Optional

import torch
from platformdirs import user_cache_dir
from tqdm import tqdm


DEFAULT_RELEASE_URL = "https://github.com/upadhyan/zeus/releases/download/v0.1.0/zeus.pt"
# TODO(release): fill in the SHA-256 of the released zeus.pt asset.
# Until the release is cut, this is a placeholder that will fail any checksum
# check — the test suite monkeypatches it.
EXPECTED_SHA256 = "eb60086459de338b1795ce2849e02e0d7e59cd95af43d01534dd84083f2b749b"

_CHUNK = 1 << 20  # 1 MiB


def _resolve_cache_dir(cache_dir: Optional[Path | str]) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    env = os.environ.get("ZEUS_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    return Path(user_cache_dir("zeus"))


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _stream_download(url: str, dest_tmp: Path) -> None:
    """Stream `url` to `dest_tmp`, showing a tqdm progress bar."""
    req = urllib.request.Request(url, headers={"User-Agent": "zeus-fork"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0)) or None
        with open(dest_tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024, desc="zeus.pt"
        ) as bar:
            while True:
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))


def get_checkpoint_path(
    model_path: Optional[Path | str] = None,
    cache_dir: Optional[Path | str] = None,
) -> Path:
    """Return a local path to a usable zeus.pt; download and cache on first use."""
    if model_path is not None:
        return Path(model_path).expanduser()

    cache = _resolve_cache_dir(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / "zeus.pt"

    if target.exists():
        if _sha256_of_file(target) == EXPECTED_SHA256:
            return target
        warnings.warn(
            f"Cached zeus.pt at {target} failed checksum; re-downloading."
        )

    print(f"Downloading ZEUS weights (~305 MB) to {target}...")
    tmp = target.with_suffix(".pt.tmp")
    try:
        _stream_download(DEFAULT_RELEASE_URL, tmp)
        actual = _sha256_of_file(tmp)
        if actual != EXPECTED_SHA256:
            raise RuntimeError(
                f"Downloaded zeus.pt failed checksum.\n"
                f"  url: {DEFAULT_RELEASE_URL}\n"
                f"  expected: {EXPECTED_SHA256}\n"
                f"  actual:   {actual}\n"
                f"  local path: {tmp}\n"
                f"You can manually place a correct zeus.pt at the cache path "
                f"({target}) and re-run."
            )
        shutil.move(str(tmp), str(target))
    finally:
        if tmp.exists():
            tmp.unlink()

    return target


def _load_raw_state_dict(path: Path, *, map_location) -> dict:
    """Load a checkpoint file and return the model state-dict, unwrapping {"model": ...}."""
    obj = torch.load(path, map_location=map_location)
    if isinstance(obj, dict) and "model" in obj and not _looks_like_state_dict(obj):
        return obj["model"]
    return obj


def _looks_like_state_dict(d: dict) -> bool:
    """Heuristic: a raw state-dict's values are tensors; a wrapper's values are mixed."""
    for v in d.values():
        if isinstance(v, torch.Tensor):
            return True
        return False
    return False


# The released checkpoint contains parameters for the upstream training-time
# `decoder`, `out_layer`, and `cluster_embedding` modules, which this fork
# strips from the model in Task 4. After loading with strict=False those keys
# show up as UNEXPECTED keys (present in the checkpoint, absent from the model).
_TOLERATED_UNEXPECTED_PREFIXES: tuple[str, ...] = (
    "decoder.",
    "out_layer.",
    "cluster_embedding.",
)


def load_zeus_state_dict(model, path: Path, *, map_location) -> None:
    """Load `path` into `model`, tolerating known-stripped checkpoint keys.

    Loads with `strict=False`. Warns once if `unexpected_keys` contains
    anything outside the known-stripped prefixes (decoder/out_layer), and
    warns once if `missing_keys` is non-empty for any model parameter (the
    released checkpoint should fill every parameter the stripped model has).
    """
    sd = _load_raw_state_dict(path, map_location=map_location)
    result = model.load_state_dict(sd, strict=False)
    surprising_unexpected = [
        k for k in result.unexpected_keys
        if not any(k.startswith(p) for p in _TOLERATED_UNEXPECTED_PREFIXES)
    ]
    if surprising_unexpected:
        warnings.warn(
            "ZEUS checkpoint has unexpected keys: "
            + ", ".join(surprising_unexpected[:10])
            + (" ..." if len(surprising_unexpected) > 10 else "")
        )
    if result.missing_keys:
        warnings.warn(
            "ZEUS checkpoint missing keys (model has parameters the checkpoint doesn't): "
            + ", ".join(result.missing_keys[:10])
            + (" ..." if len(result.missing_keys) > 10 else "")
        )
