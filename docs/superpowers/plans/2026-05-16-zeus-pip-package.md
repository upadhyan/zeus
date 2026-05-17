# ZEUS pip-installable refactor — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the current research codebase at `/home/upadhyan/zeus/` into a pip-installable inference-only library with sklearn-style estimators (`Zeus`, `ZeusClusterer`), GitHub-Releases-hosted weights, and platform-cached on-disk weights — implementing the design in `docs/superpowers/specs/2026-05-16-zeus-pip-package-design.md`.

**Architecture:** Single `zeus/` Python package containing the model (cleaned of training-only branches), a small download/cache layer, a preprocessing pipeline (DataFrame → `(N, 30)` tensor), and the two sklearn-style estimators. Training scripts, OpenML loaders, wandb logging, and synthetic-data generators are deleted. Original README content is preserved with inline "removed" notes pointing to the upstream repo.

**Tech Stack:** Python ≥3.10, PyTorch ≥2.0, scikit-learn, pandas, scipy, platformdirs, tqdm, pytest. No CI; no PyPI; install via `pip install git+https://github.com/upadhyan/zeus.git`.

---

## Conventions for this plan

- Run **all commands from the repo root** `/home/upadhyan/zeus/`.
- Install the package once in editable mode after Task 1 so tests can import it: `pip install -e .`
- Run tests with `pytest -x tests/` (or with a specific test path noted in each task). `-x` stops on first failure.
- Commit after each task. Use the commit message shown in the task's final step.
- "Expected: FAIL with ..." in a step means the test must fail with that reason at that point — if it passes, something is wrong with the test, not the code.
- Spec references like `spec §5.4` refer to sections of `docs/superpowers/specs/2026-05-16-zeus-pip-package-design.md`.

---

## Chunk 1: Foundation and model cleanup

This chunk does the no-net-new-behavior work that the new code in chunk 2 depends on: the package metadata, the frozen-constant module, surgical edits to `model/zeus.py` (so the `configs` module can be deleted later), inlining the two helpers from `model_utils.py`, and moving `inference_methods/` inside the package. After this chunk the model still works exactly as before but no longer imports anything from `configs.py` or `model_utils.py`.

### Task 1: Add `pyproject.toml`, `tests/` skeleton, and install editable

**Files:**
- Create: `pyproject.toml`
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py`
- Modify: `.gitignore` (add `.venv/`, `*.egg-info/`, `__pycache__/` if missing)

- [ ] **Step 1: Inspect existing `.gitignore` and decide what to add**

Run: `cat .gitignore`

The current ignore list covers `results/`, `slurm-*.out`, IDE stuff. Confirm `*.egg-info/`, `.venv/`, `__pycache__/`, `.pytest_cache/`, and `*.pyc` are listed; add any that are missing.

- [ ] **Step 2: Write `pyproject.toml`**

Create `/home/upadhyan/zeus/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "zeus"
version = "0.1.0"
description = "Zero-shot Embeddings for Unsupervised Separation of tabular data"
readme = "README.md"
requires-python = ">=3.10"
license = { file = "LICENSE" }
authors = [{ name = "Nakul Upadhya (fork maintainer)" }]
dependencies = [
    "torch>=2.0",
    "numpy>=1.24",
    "scipy>=1.10",
    "scikit-learn>=1.3",
    "pandas>=2.0",
    "platformdirs>=4.0",
    "tqdm>=4.60",
]

[project.optional-dependencies]
test = ["pytest>=7.0"]

[project.urls]
Homepage = "https://github.com/upadhyan/zeus"
Upstream = "https://github.com/gmum/zeus"
Paper = "https://arxiv.org/abs/2505.10704"

[tool.setuptools.packages.find]
include = ["zeus*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Create empty `tests/__init__.py` and `tests/conftest.py`**

`tests/__init__.py` is just an empty file (touch it).

`tests/conftest.py`:

```python
"""Shared pytest fixtures for the ZEUS test suite."""
import os
import pytest


@pytest.fixture
def isolated_cache_dir(tmp_path, monkeypatch):
    """Point ZEUS_CACHE_DIR at a fresh tmp_path so tests don't touch the user's real cache."""
    monkeypatch.setenv("ZEUS_CACHE_DIR", str(tmp_path))
    return tmp_path
```

- [ ] **Step 4: Confirm `LICENSE` exists at repo root**

Run: `ls LICENSE 2>/dev/null && echo "LICENSE found" || echo "MISSING — checking legal/"`

If missing at the root, the `legal/` folder has a `LICENSE` from TabPFN. Copy it: `cp legal/LICENSE LICENSE` (or whatever filename is in `legal/`). The `pyproject.toml` `license` field needs a file at the path it points to.

- [ ] **Step 5: Install editable**

Run: `pip install -e .`

Expected: build succeeds, `zeus` is importable: `python -c "import zeus; print(zeus.__file__)"` prints the path under `zeus/zeus/__init__.py`.

If torch isn't already installed in the active env, you'll see torch downloading. That's fine.

- [ ] **Step 6: Smoke test — pytest collects an empty suite**

Run: `pytest tests/ -v`

Expected: `no tests ran in 0.XXs` (exit code 5 from pytest; that's fine). This confirms pytest discovers the directory.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml tests/__init__.py tests/conftest.py .gitignore LICENSE
git commit -m "build: add pyproject.toml and pytest skeleton"
```

---

### Task 2: Create `zeus/_config.py` with frozen checkpoint constants

**Files:**
- Create: `zeus/_config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
"""Frozen checkpoint constants are stable and match the released zeus.pt."""
from zeus import _config as c


def test_frozen_constants_present_and_correct():
    assert c.EMBED_DIM == 512
    assert c.N_HEAD == 4
    assert c.HID_DIM == 1024
    assert c.N_LAYERS == 12
    assert c.NUM_GAUSSIANS == 10
    assert c.INPUT_DIM == 30
    assert c.DROPOUT == 0.0
    assert c.EFFICIENT_EVAL_MASKING is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'zeus._config'`.

- [ ] **Step 3: Write `zeus/_config.py`**

```python
"""Frozen hyperparameters for the released ZEUS checkpoint (v0.1.0 of this fork).

These are NOT user-tunable. A `model_path=` pointing at a checkpoint trained
with different hyperparameters is unsupported; the load will either fail or
produce garbage.
"""
from __future__ import annotations

EMBED_DIM: int = 512
N_HEAD: int = 4
HID_DIM: int = 1024
N_LAYERS: int = 12
NUM_GAUSSIANS: int = 10
INPUT_DIM: int = 30
DROPOUT: float = 0.0
EFFICIENT_EVAL_MASKING: bool = True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add zeus/_config.py tests/test_config.py
git commit -m "feat(config): add frozen checkpoint constants module"
```

---

### Task 3: Inline `SeqBN` and `bool_mask_to_att_mask` into `model/zeus.py`, delete `model/model_utils.py`

The two helpers from `zeus/model/model_utils.py` are dead code at inference (only reachable via `input_normalization=True` or `full_attention=True`, neither of which is ever set), but the model module currently imports them, so we need to either inline them or strip their references. We **inline** to keep the diff to the model body minimal — only one import line changes.

**Files:**
- Modify: `zeus/model/zeus.py:1-12` (imports), then add helper definitions
- Delete: `zeus/model/model_utils.py` (after the imports change is in place)

- [ ] **Step 1: Verify what `model_utils.py` contains**

Run: `cat zeus/model/model_utils.py`

Confirm it contains `get_cosine_schedule_with_warmup` (training-only — drop), `SeqBN`, `default_device` (drop — unused at inference), and `bool_mask_to_att_mask`.

- [ ] **Step 2: Edit `zeus/model/zeus.py` imports**

Open `zeus/model/zeus.py`. Replace the top imports (lines 1-12):

```python
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, TransformerEncoder

from zeus.configs import LossType
from zeus.model.layer import TransformerEncoderLayer
from zeus.model.model_utils import SeqBN, bool_mask_to_att_mask
from sklearn.preprocessing import MinMaxScaler
```

with:

```python
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, TransformerEncoder

from zeus.configs import LossType
from zeus.model.layer import TransformerEncoderLayer
from sklearn.preprocessing import MinMaxScaler
```

Then **append** after the imports, before `class ZeusTransformerModel`:

```python
class SeqBN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)
        self.d_model = d_model

    def forward(self, x):
        assert self.d_model == x.shape[-1]
        flat_x = x.view(-1, self.d_model)
        flat_x = self.bn(flat_x)
        return flat_x.view(*x.shape)


def bool_mask_to_att_mask(mask):
    return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
```

(`LossType` import stays for now — Task 4 removes it.)

- [ ] **Step 3: Sanity-import the module**

Run: `python -c "from zeus.model.zeus import ZeusTransformerModel, SeqBN, bool_mask_to_att_mask; print('ok')"`

Expected: `ok` (no `ImportError`).

- [ ] **Step 4: Delete `model_utils.py`**

Run: `git rm zeus/model/model_utils.py`

- [ ] **Step 5: Confirm no other module imports `model_utils`**

Run: `grep -RIn "model_utils" zeus/ tests/ 2>/dev/null`

Expected: no output. (`pretrain.py` and `evaluation.py` at the repo root may still reference it; they're being deleted later. Ignore those for now.)

- [ ] **Step 6: Commit**

```bash
git add zeus/model/zeus.py
git commit -m "refactor(model): inline SeqBN and bool_mask_to_att_mask, drop model_utils"
```

---

### Task 4: Remove `LossType` / `decoder` / `distance_based_logit` from `ZeusTransformerModel`

This is the surgical edit that lets us delete `zeus/configs.py` in chunk 2. The released checkpoint always runs with `loss_type=LossType.CENTER`; the `decoder` / `out_layer` modules and the `distance_based_logit` branch are unused at inference (see spec §5.3).

**Files:**
- Modify: `zeus/model/zeus.py` (constructor, forward, top imports)
- Create: `tests/test_model_smoke.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_model_smoke.py`:

```python
"""Smoke tests for ZeusTransformerModel after the loss_type / decoder refactor.

These don't load the released checkpoint — they just verify the model can be
constructed, runs a forward pass, and produces the expected shape.
"""
import torch
import pytest

from zeus import _config as c


def _build_model():
    from zeus.model.encoders import Linear
    from zeus.model.zeus import ZeusTransformerModel

    encoder = Linear(c.INPUT_DIM, c.EMBED_DIM, replace_nan_by_zero=True)
    return ZeusTransformerModel(
        encoder,
        ninp=c.EMBED_DIM,
        nhead=c.N_HEAD,
        nhid=c.HID_DIM,
        nlayers=c.N_LAYERS,
        dropout=c.DROPOUT,
        n_clusters=c.NUM_GAUSSIANS,
        efficient_eval_masking=c.EFFICIENT_EVAL_MASKING,
    )


def test_constructor_takes_no_loss_type_or_decoder_args():
    """The refactored constructor should not accept loss_type, decoder, n_out, or distance_based_logit."""
    import inspect
    from zeus.model.zeus import ZeusTransformerModel
    sig = inspect.signature(ZeusTransformerModel.__init__)
    for forbidden in ("loss_type", "decoder", "n_out", "distance_based_logit"):
        assert forbidden not in sig.parameters, (
            f"{forbidden} should have been removed in the refactor"
        )


def test_forward_returns_correct_shape():
    """Forward should return (N + NUM_GAUSSIANS, 1, EMBED_DIM)."""
    model = _build_model()
    model.eval()
    n = 5
    x = torch.randn(n, 1, c.INPUT_DIM)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (n + c.NUM_GAUSSIANS, 1, c.EMBED_DIM)


def test_model_has_no_configs_import():
    """zeus.model.zeus should no longer import from zeus.configs."""
    import zeus.model.zeus as zm
    src = open(zm.__file__).read()
    assert "from zeus.configs" not in src
    assert "import zeus.configs" not in src
```

- [ ] **Step 2: Run tests to verify failures**

Run: `pytest tests/test_model_smoke.py -v`

Expected: FAIL.
- `test_constructor_takes_no_loss_type_or_decoder_args` fails — `loss_type` is still a parameter.
- `test_forward_returns_correct_shape` may pass (the model already works) OR fail because the new test fixture passes kwargs the current constructor doesn't accept (the current signature is positional-friendly, so check).
- `test_model_has_no_configs_import` fails — the import is still there.

- [ ] **Step 3: Edit `ZeusTransformerModel.__init__`**

Open `zeus/model/zeus.py`. Replace the constructor body (currently lines ~15-48):

```python
class ZeusTransformerModel(nn.Module):
    def __init__(self, encoder, n_out, ninp, nhead, nhid, nlayers,
                 dropout=0.0, *, n_clusters=10, dist_based_logit=False,
                 loss_type=LossType.CENTER, decoder=None,
                 input_normalization=False, pre_norm=False,
                 activation='gelu', recompute_attn=False, full_attention=False,
                 all_layers_same_init=False, efficient_eval_masking=True):
        super().__init__()
        self.model_type = 'Transformer'
        encoder_layer_creator = lambda: TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation=activation,
                                                                pre_norm=pre_norm, recompute_attn=recompute_attn)
        self.transformer_encoder = TransformerEncoder(encoder_layer_creator(), nlayers)\
            if all_layers_same_init else TransformerEncoderDiffInit(encoder_layer_creator, nlayers)
        self.ninp = ninp
        self.encoder = encoder
        self.distance_based_logit = dist_based_logit
        self.loss_type = loss_type

        if not self.distance_based_logit:
            self.decoder = decoder(ninp, nhid, n_out) if decoder is not None \
                else nn.Sequential(nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, n_out))
        else:
            self.decoder = nn.Sequential(nn.Linear(ninp, nhid), nn.GELU())
            self.out_layer = nn.Linear(nhid, n_out, bias=False)

        self.input_ln = SeqBN(ninp) if input_normalization else None
        self.efficient_eval_masking = efficient_eval_masking
        self.full_attention = full_attention

        self.n_out = n_out
        self.nhid = nhid

        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, 1, ninp))

        self.init_weights()
```

with:

```python
class ZeusTransformerModel(nn.Module):
    def __init__(self, encoder, ninp, nhead, nhid, nlayers, *,
                 dropout=0.0, n_clusters=10,
                 input_normalization=False, pre_norm=False,
                 activation='gelu', recompute_attn=False, full_attention=False,
                 all_layers_same_init=False, efficient_eval_masking=True):
        super().__init__()
        self.model_type = 'Transformer'
        encoder_layer_creator = lambda: TransformerEncoderLayer(
            ninp, nhead, nhid, dropout, activation=activation,
            pre_norm=pre_norm, recompute_attn=recompute_attn,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer_creator(), nlayers) \
            if all_layers_same_init else TransformerEncoderDiffInit(encoder_layer_creator, nlayers)
        self.ninp = ninp
        self.encoder = encoder

        self.input_ln = SeqBN(ninp) if input_normalization else None
        self.efficient_eval_masking = efficient_eval_masking
        self.full_attention = full_attention

        self.nhid = nhid

        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, 1, ninp))

        self.init_weights()
```

- [ ] **Step 4: Edit `forward` to drop the loss_type branch**

Replace the current `forward` method body (the section starting with `def forward(self, x, *, k=0):`):

```python
    def forward(self, x, *, k=0):
        x_src = self.encoder(x)
        src_mask = None

        if src_mask is None:
            full_len = len(x_src) + len(self.cluster_centers)
            if self.full_attention:
                src_mask = bool_mask_to_att_mask(torch.ones((full_len, full_len), dtype=torch.bool)).to(x_src.device)
            elif self.efficient_eval_masking:
                src_mask = full_len
            else:
                src_mask = self.generate_D_q_matrix(full_len, 0).to(x_src.device)

        src = torch.cat([x_src, self.cluster_centers], 0)

        if self.input_ln is not None:
            src = self.input_ln(src)

        output = self.transformer_encoder(src, src_mask)
        if self.loss_type == LossType.CENTER:
            return output

        #output = self.decoder(output)
#
        #if self.distance_based_logit:
        #    output = -torch.linalg.norm(output.unsqueeze(-1) - self.out_layer.weight.T, dim=-2)**2

        return output
```

with:

```python
    def forward(self, x, *, k=0):
        x_src = self.encoder(x)

        full_len = len(x_src) + len(self.cluster_centers)
        if self.full_attention:
            src_mask = bool_mask_to_att_mask(
                torch.ones((full_len, full_len), dtype=torch.bool)
            ).to(x_src.device)
        elif self.efficient_eval_masking:
            src_mask = full_len
        else:
            src_mask = self.generate_D_q_matrix(full_len, 0).to(x_src.device)

        src = torch.cat([x_src, self.cluster_centers], 0)

        if self.input_ln is not None:
            src = self.input_ln(src)

        return self.transformer_encoder(src, src_mask)
```

The `k` kwarg stays (it's unused but harmless and avoids breaking callers that pass it).

- [ ] **Step 5: Drop the `LossType` import**

In `zeus/model/zeus.py`, remove the line:

```python
from zeus.configs import LossType
```

Save.

- [ ] **Step 6: Run smoke tests**

Run: `pytest tests/test_model_smoke.py -v`

Expected: all three tests PASS.

- [ ] **Step 7: Confirm nothing else in `zeus/` imports `LossType`**

Run: `grep -RIn "LossType" zeus/ tests/ 2>/dev/null`

Expected: no output. (`pretrain.py` / `evaluation.py` / `utils.py` may still reference it — they're being deleted. Ignore.)

- [ ] **Step 8: Commit**

```bash
git add zeus/model/zeus.py tests/test_model_smoke.py
git commit -m "refactor(model): drop loss_type, decoder, and distance_based_logit branches"
```

---

### Task 5: Move `inference_methods/` inside the `zeus/` package

**Files:**
- Move: `inference_methods/simple_gmm.py` → `zeus/inference_methods/simple_gmm.py`
- Move: `inference_methods/__init__.py` → `zeus/inference_methods/__init__.py`
- Modify: `zeus/utils.py:16` (only place that imports it inside `zeus/`)

- [ ] **Step 1: Confirm sibling package is otherwise unused inside `zeus/`**

Run: `grep -RIn "from inference_methods" /home/upadhyan/zeus/ 2>/dev/null`

You should see exactly two hits: `zeus/utils.py:16` and `evaluation.py:5`.

- `zeus/utils.py:16` needs to be updated in this task (Step 3 below).
- `evaluation.py:5` does **not** need updating: `evaluation.py` is deleted in its entirety in Task 10. The test suite never imports it, so leaving it temporarily broken between Tasks 5 and 10 is intentional. **Do not edit `evaluation.py` here.**

- [ ] **Step 2: Move the files**

Run:

```bash
mkdir -p zeus/inference_methods
git mv inference_methods/simple_gmm.py zeus/inference_methods/simple_gmm.py
git mv inference_methods/__init__.py zeus/inference_methods/__init__.py
rmdir inference_methods
```

- [ ] **Step 3: Update the one import in `zeus/utils.py`**

Edit `zeus/utils.py:16`:

```python
from inference_methods.simple_gmm import SimplifiedGMM
```

to:

```python
from zeus.inference_methods.simple_gmm import SimplifiedGMM
```

(`zeus/utils.py` itself is being deleted later in chunk 2, but until then `pretrain.py` / `evaluation.py` may import from it; keeping it working in the interim avoids a broken intermediate state. **Note:** if you find any other consumer when re-checking with grep, update those too.)

- [ ] **Step 4: Sanity import**

Run: `python -c "from zeus.inference_methods.simple_gmm import SimplifiedGMM; print(SimplifiedGMM)"`

Expected: `<class 'zeus.inference_methods.simple_gmm.SimplifiedGMM'>`.

- [ ] **Step 5: Run all tests so far**

Run: `pytest tests/ -v`

Expected: all tests from Tasks 2 and 4 still PASS.

- [ ] **Step 6: Commit**

```bash
git add zeus/inference_methods/ zeus/utils.py
git commit -m "refactor: move inference_methods package inside zeus/"
```

---

## Chunk 2: New package code, deletions, and documentation

Now the foundation is in place: the model is configs-free and lives under a single top-level package. Chunk 2 builds the three new modules (`weights.py`, `preprocessing.py`, `api.py`), wires them through `__init__.py`, deletes everything that's now obsolete, and rewrites the README + CLAUDE.md.

### Task 6: Build `zeus/weights.py` (download, cache, state-dict load)

**Files:**
- Create: `zeus/weights.py`
- Create: `tests/test_weights.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_weights.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_weights.py -v`

Expected: FAIL — `zeus.weights` module doesn't exist yet.

- [ ] **Step 3: Write `zeus/weights.py`**

```python
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
EXPECTED_SHA256 = "0" * 64

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
# `decoder` and (optionally) `out_layer` modules, which this fork strips from
# the model in Task 4. After loading with strict=False those keys show up as
# UNEXPECTED keys (present in the checkpoint, absent from the model).
_TOLERATED_UNEXPECTED_PREFIXES: tuple[str, ...] = ("decoder.", "out_layer.")


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
```

- [ ] **Step 4: Run weights tests to verify they pass**

Run: `pytest tests/test_weights.py -v`

Expected: all 7 tests PASS. If `test_explicit_model_path_is_returned_verbatim` fails because the path doesn't exist — re-read the test; the design is "no validation when `model_path=` is given" — that should pass.

- [ ] **Step 5: Commit**

```bash
git add zeus/weights.py tests/test_weights.py
git commit -m "feat(weights): add download, cache, and state-dict load helpers"
```

---

### Task 7: Build `zeus/preprocessing.py` (`prepare_inputs`)

**Files:**
- Create: `zeus/preprocessing.py`
- Create: `tests/test_preprocessing.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_preprocessing.py`:

```python
"""Input preprocessing: DataFrame / ndarray / Tensor -> (n, INPUT_DIM) float32 tensor."""
from __future__ import annotations
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from zeus import _config as c
from zeus.preprocessing import prepare_inputs


def test_numpy_exact_dim_passes_through_with_scaling():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, c.INPUT_DIM)).astype(np.float32)
    out = prepare_inputs(X)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    assert out.shape == (20, c.INPUT_DIM)
    assert out.min() >= -1.0 - 1e-6 and out.max() <= 1.0 + 1e-6


def test_numpy_small_dim_is_zero_padded_then_scaled():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(15, 8)).astype(np.float32)
    out = prepare_inputs(X)
    assert out.shape == (15, c.INPUT_DIM)
    # The padded columns (8..) before scaling are all zeros, but after MinMax
    # to [-1, 1] columns with constant value should become -1 (sklearn default).
    # Just check we didn't lose rows.
    assert torch.isfinite(out).all()


def test_numpy_large_dim_is_pca_then_scaled():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 64)).astype(np.float32)
    out = prepare_inputs(X)
    assert out.shape == (50, c.INPUT_DIM)
    assert out.min() >= -1.0 - 1e-6 and out.max() <= 1.0 + 1e-6


def test_tensor_input_accepted():
    X = torch.randn(20, c.INPUT_DIM)
    out = prepare_inputs(X)
    assert out.shape == (20, c.INPUT_DIM)


def test_dataframe_with_categoricals_is_one_hot_encoded():
    df = pd.DataFrame({
        "a": np.random.randn(30),
        "b": np.random.randn(30),
        "color": np.random.choice(["red", "blue", "green"], size=30),
        "flag": np.random.choice([True, False], size=30),
    })
    out = prepare_inputs(df)
    assert out.shape == (30, c.INPUT_DIM)


def test_nans_in_numeric_dataframe_are_mean_imputed():
    df = pd.DataFrame({
        "a": [1.0, np.nan, 3.0, 4.0, 5.0],
        "b": [10.0, 20.0, np.nan, 40.0, 50.0],
    })
    out = prepare_inputs(df)
    assert out.shape == (5, c.INPUT_DIM)
    assert torch.isfinite(out).all()


def test_nans_in_ndarray_emit_warning_and_are_imputed():
    X = np.array([
        [1.0, 2.0, np.nan],
        [4.0, 5.0, 6.0],
        [np.nan, 8.0, 9.0],
    ])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = prepare_inputs(X)
    assert any("NaN" in str(w.message) for w in caught)
    assert torch.isfinite(out).all()


def test_preprocess_false_path_requires_exact_dim(monkeypatch):
    """Bypass-mode validates shape and refuses anything but (n, INPUT_DIM)."""
    from zeus.preprocessing import passthrough_inputs
    good = torch.randn(5, c.INPUT_DIM)
    out = passthrough_inputs(good)
    assert out.shape == (5, c.INPUT_DIM)
    assert out.dtype == torch.float32

    bad = np.zeros((5, c.INPUT_DIM - 1), dtype=np.float32)
    with pytest.raises(ValueError, match=r"expected \(n, 30\)"):
        passthrough_inputs(bad)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_preprocessing.py -v`

Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Write `zeus/preprocessing.py`**

```python
"""Input preprocessing for the Zeus encoder.

Converts ndarray / DataFrame / Tensor inputs into a (n, INPUT_DIM) float32
tensor suitable for `Zeus.transform`. See spec §5.6.
"""
from __future__ import annotations

import warnings
from typing import Union

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from zeus._config import INPUT_DIM

ArrayLike = Union[np.ndarray, pd.DataFrame, torch.Tensor]


def _is_categorical_like(dt) -> bool:
    """True if `dt` is object, pandas Categorical, or boolean."""
    return dt == object or isinstance(dt, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(dt)


def _df_to_numeric_matrix(df: pd.DataFrame) -> np.ndarray:
    cat_mask = df.dtypes.apply(_is_categorical_like)
    cat_cols = list(df.columns[cat_mask])
    num_cols = list(df.columns[~cat_mask])

    parts = []
    if num_cols:
        nums = df[num_cols].astype(float)
        # Per-column mean impute
        col_means = nums.mean(axis=0, skipna=True)
        nums = nums.fillna(col_means)
        # If a column is entirely NaN, its mean is NaN; refuse explicitly
        if nums.isna().any().any():
            bad = nums.columns[nums.isna().any()].tolist()
            raise ValueError(f"All-NaN column(s) cannot be imputed: {bad}")
        parts.append(nums.to_numpy(dtype=np.float64))
    if cat_cols:
        ohe = pd.get_dummies(df[cat_cols], dummy_na=False)
        parts.append(ohe.to_numpy(dtype=np.float64))
    if not parts:
        raise ValueError("Empty DataFrame; nothing to preprocess.")
    return np.concatenate(parts, axis=1)


def _mean_impute_array(X: np.ndarray) -> np.ndarray:
    if not np.isnan(X).any():
        return X
    warnings.warn(
        "Input contains NaN values; mean-imputing per column. "
        "Pass a DataFrame for explicit handling.",
        stacklevel=3,
    )
    col_means = np.nanmean(X, axis=0)
    if np.isnan(col_means).any():
        bad = np.where(np.isnan(col_means))[0].tolist()
        raise ValueError(f"All-NaN column index(es) cannot be imputed: {bad}")
    inds = np.where(np.isnan(X))
    X = X.copy()
    X[inds] = np.take(col_means, inds[1])
    return X


def _adjust_dim(X: np.ndarray, target_dim: int) -> np.ndarray:
    d = X.shape[1]
    if d == target_dim:
        return X
    if d > target_dim:
        return PCA(n_components=target_dim).fit_transform(X)
    pad = np.zeros((X.shape[0], target_dim - d), dtype=X.dtype)
    return np.concatenate([X, pad], axis=1)


def prepare_inputs(X: ArrayLike, target_dim: int = INPUT_DIM) -> torch.Tensor:
    """Convert `X` to a (n, target_dim) float32 tensor.

    Order: DataFrame split + impute + one-hot, OR ndarray impute,
    then PCA-or-pad, then MinMaxScale to [-1, 1].
    Matches the existing `evaluate_model` pipeline (spec §5.6).
    """
    if isinstance(X, pd.DataFrame):
        mat = _df_to_numeric_matrix(X)
    elif isinstance(X, torch.Tensor):
        mat = _mean_impute_array(X.detach().cpu().numpy().astype(np.float64))
    elif isinstance(X, np.ndarray):
        mat = _mean_impute_array(X.astype(np.float64))
    else:
        raise TypeError(f"Unsupported input type: {type(X).__name__}")

    if mat.ndim != 2:
        raise ValueError(f"Expected 2-D input, got shape {mat.shape}")

    mat = _adjust_dim(mat, target_dim)
    mat = MinMaxScaler(feature_range=(-1, 1)).fit_transform(mat)
    return torch.tensor(mat, dtype=torch.float32)


def passthrough_inputs(X: ArrayLike, target_dim: int = INPUT_DIM) -> torch.Tensor:
    """Used when `preprocess=False`; rejects anything but a numeric (n, target_dim) array."""
    if isinstance(X, pd.DataFrame):
        raise ValueError(
            "preprocess=False requires a numeric ndarray/Tensor; got DataFrame."
        )
    if isinstance(X, torch.Tensor):
        arr = X
    elif isinstance(X, np.ndarray):
        arr = torch.tensor(X, dtype=torch.float32)
    else:
        raise TypeError(f"Unsupported input type: {type(X).__name__}")
    if arr.ndim != 2 or arr.shape[1] != target_dim:
        raise ValueError(f"expected (n, {target_dim}), got {tuple(arr.shape)}")
    return arr.to(dtype=torch.float32)
```

- [ ] **Step 4: Run preprocessing tests to verify they pass**

Run: `pytest tests/test_preprocessing.py -v`

Expected: all PASS. (The NaN warning fires on every call — no module-level flag — so the test is reliable regardless of execution order.)

- [ ] **Step 5: Commit**

```bash
git add zeus/preprocessing.py tests/test_preprocessing.py
git commit -m "feat(preprocessing): add prepare_inputs / passthrough_inputs"
```

---

### Task 8: Build `zeus/api.py` (`Zeus` and `ZeusClusterer`)

**Files:**
- Create: `zeus/api.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_api.py`:

```python
"""End-to-end sklearn-style API smoke tests.

These tests do NOT require the released zeus.pt. They build a fresh
ZeusTransformerModel with frozen-constant hyperparameters, save its
randomly-initialized state-dict to a tempfile, and point the API at it
via `model_path=`. Embeddings will be meaningless, but shape contracts
and the clusterer plumbing are exercised end to end.
"""
from __future__ import annotations
import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.exceptions import NotFittedError

from zeus import _config as c


@pytest.fixture(scope="module")
def fake_checkpoint(tmp_path_factory):
    """A randomly-initialized state-dict saved as zeus.pt-shaped checkpoint."""
    from zeus.model.encoders import Linear
    from zeus.model.zeus import ZeusTransformerModel

    encoder = Linear(c.INPUT_DIM, c.EMBED_DIM, replace_nan_by_zero=True)
    model = ZeusTransformerModel(
        encoder,
        ninp=c.EMBED_DIM,
        nhead=c.N_HEAD,
        nhid=c.HID_DIM,
        nlayers=c.N_LAYERS,
        dropout=c.DROPOUT,
        n_clusters=c.NUM_GAUSSIANS,
        efficient_eval_masking=c.EFFICIENT_EVAL_MASKING,
    )
    path = tmp_path_factory.mktemp("ckpt") / "zeus.pt"
    torch.save({"model": model.state_dict()}, path)
    return path


@pytest.fixture
def small_df():
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "x1": rng.normal(size=20),
        "x2": rng.normal(size=20),
        "color": rng.choice(["a", "b", "c"], size=20),
        "flag": rng.choice([True, False], size=20),
    })


def test_zeus_fit_transform_shape(fake_checkpoint, small_df):
    from zeus import Zeus
    emb = Zeus(model_path=fake_checkpoint, device="cpu").fit_transform(small_df)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (20, c.EMBED_DIM)


def test_zeus_fit_is_noop(fake_checkpoint, small_df):
    from zeus import Zeus
    z = Zeus(model_path=fake_checkpoint, device="cpu")
    out = z.fit(small_df)
    assert out is z


def test_zeus_transform_accepts_ndarray(fake_checkpoint):
    from zeus import Zeus
    X = np.random.randn(15, 8).astype(np.float32)
    emb = Zeus(model_path=fake_checkpoint, device="cpu").transform(X)
    assert emb.shape == (15, c.EMBED_DIM)


def test_zeus_preprocess_false_requires_exact_dim(fake_checkpoint):
    from zeus import Zeus
    bad = np.zeros((10, 20), dtype=np.float32)
    with pytest.raises(ValueError, match=r"expected \(n, 30\)"):
        Zeus(model_path=fake_checkpoint, device="cpu", preprocess=False).transform(bad)


def test_clusterer_fit_predict_returns_labels(fake_checkpoint, small_df):
    from zeus import ZeusClusterer
    labels = ZeusClusterer(
        n_clusters=3, model_path=fake_checkpoint, device="cpu", random_state=0
    ).fit_predict(small_df)
    assert labels.shape == (20,)
    assert set(np.unique(labels)).issubset({0, 1, 2})


def test_clusterer_kmeans_probabilities_raises(fake_checkpoint, small_df):
    from zeus import ZeusClusterer
    clf = ZeusClusterer(
        n_clusters=3, method="kmeans", model_path=fake_checkpoint, device="cpu",
    ).fit(small_df)
    with pytest.raises(AttributeError, match="probabilities_ requires"):
        _ = clf.probabilities_


def test_clusterer_simple_gmm_probabilities_sum_to_one(fake_checkpoint, small_df):
    from zeus import ZeusClusterer
    clf = ZeusClusterer(
        n_clusters=3, method="simple_gmm", model_path=fake_checkpoint, device="cpu",
        random_state=0, n_init=2,
    ).fit(small_df)
    p = clf.probabilities_
    assert p.shape == (20, 3)
    np.testing.assert_allclose(p.sum(axis=1), np.ones(20), atol=1e-5)


def test_clusterer_not_fitted_errors_on_probabilities(fake_checkpoint):
    """`probabilities_` calls check_is_fitted internally; raw `labels_` access
    raises plain AttributeError which is fine — we only contract the property."""
    from zeus import ZeusClusterer
    clf = ZeusClusterer(
        n_clusters=3, method="simple_gmm", model_path=fake_checkpoint, device="cpu",
    )
    with pytest.raises((NotFittedError, AttributeError)):
        _ = clf.probabilities_
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_api.py -v`

Expected: FAIL — `zeus.Zeus` / `zeus.ZeusClusterer` don't exist yet.

- [ ] **Step 3: Write `zeus/api.py`**

```python
"""sklearn-compatible estimators wrapping the ZEUS transformer (spec §5.4)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted

from zeus import _config as c
from zeus.model.encoders import Linear
from zeus.model.zeus import ZeusTransformerModel
from zeus.preprocessing import passthrough_inputs, prepare_inputs
from zeus.weights import get_checkpoint_path, load_zeus_state_dict


DeviceLike = Union[str, torch.device]
ArrayLike = Union[np.ndarray, pd.DataFrame, torch.Tensor]


def _resolve_device(device: DeviceLike) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _build_model() -> ZeusTransformerModel:
    encoder = Linear(c.INPUT_DIM, c.EMBED_DIM, replace_nan_by_zero=True)
    return ZeusTransformerModel(
        encoder,
        ninp=c.EMBED_DIM,
        nhead=c.N_HEAD,
        nhid=c.HID_DIM,
        nlayers=c.N_LAYERS,
        dropout=c.DROPOUT,
        n_clusters=c.NUM_GAUSSIANS,
        efficient_eval_masking=c.EFFICIENT_EVAL_MASKING,
    )


class Zeus(TransformerMixin, BaseEstimator):
    """Zero-shot tabular encoder.

    Embeddings are *batch-context-dependent*: every row attends to every
    other row in the same `transform` call. `fit` is therefore a no-op,
    and `fit(X_train)` followed by `transform(X_test)` is NOT equivalent
    to `fit_transform(X_test)` on its own. Use `fit_transform(X)` on the
    dataset you actually want to embed.
    """

    def __init__(
        self,
        *,
        device: DeviceLike = "auto",
        preprocess: bool = True,
        model_path: Optional[Union[Path, str]] = None,
        cache_dir: Optional[Union[Path, str]] = None,
    ):
        self.device = device
        self.preprocess = preprocess
        self.model_path = model_path
        self.cache_dir = cache_dir

    # sklearn cloning relies on `get_params` reading attributes set in __init__;
    # heavy state (model, device object) is loaded lazily in `_ensure_model`.
    def _ensure_model(self) -> None:
        if getattr(self, "_model", None) is not None:
            return
        self._device = _resolve_device(self.device)
        path = get_checkpoint_path(
            model_path=self.model_path,
            cache_dir=self.cache_dir,
        )
        model = _build_model()
        load_zeus_state_dict(model, path, map_location=self._device)
        model.to(self._device)
        model.eval()
        self._model = model

    def fit(self, X, y=None):  # noqa: D401
        """No-op (zero-shot). Returns self so the TransformerMixin contract works."""
        self._ensure_model()
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        self._ensure_model()
        prep = prepare_inputs if self.preprocess else passthrough_inputs
        x = prep(X)
        x = x.unsqueeze(1).to(self._device)
        with torch.no_grad():
            out = self._model(x)
        # Strip the NUM_GAUSSIANS learned cluster centers concatenated by the model.
        emb = out[:-c.NUM_GAUSSIANS].squeeze(1).detach().cpu().numpy()
        return emb


class ZeusClusterer(ClusterMixin, BaseEstimator):
    """End-to-end clustering on Zeus embeddings.

    Exposes only `fit` / `fit_predict` (no `predict(X_new)`), because
    embeddings depend on the batch they were computed in. For probabilistic
    methods the soft assignments are exposed as `self.probabilities_`,
    which is populated by `fit`.
    """

    def __init__(
        self,
        n_clusters: int,
        *,
        method: Literal["kmeans", "gmm", "simple_gmm"] = "kmeans",
        device: DeviceLike = "auto",
        preprocess: bool = True,
        model_path: Optional[Union[Path, str]] = None,
        cache_dir: Optional[Union[Path, str]] = None,
        random_state: Optional[int] = None,
        n_init: int = 10,
    ):
        self.n_clusters = n_clusters
        self.method = method
        self.device = device
        self.preprocess = preprocess
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.random_state = random_state
        self.n_init = n_init

    def fit(self, X, y=None):
        encoder = Zeus(
            device=self.device,
            preprocess=self.preprocess,
            model_path=self.model_path,
            cache_dir=self.cache_dir,
        )
        emb = encoder.fit_transform(X)
        emb = MinMaxScaler(feature_range=(-1, 1)).fit_transform(emb)
        self.embedding_ = emb

        if self.method == "kmeans":
            km = KMeans(
                n_clusters=self.n_clusters,
                n_init=self.n_init,
                random_state=self.random_state,
            ).fit(emb)
            self.labels_ = km.labels_
            self.cluster_centers_ = km.cluster_centers_
        elif self.method == "gmm":
            gmm = GaussianMixture(
                n_components=self.n_clusters,
                n_init=self.n_init,
                random_state=self.random_state,
            ).fit(emb)
            self.labels_ = gmm.predict(emb)
            self.cluster_centers_ = gmm.means_
            self.probabilities_ = gmm.predict_proba(emb)
        elif self.method == "simple_gmm":
            from zeus.inference_methods.simple_gmm import SimplifiedGMM
            sgmm = SimplifiedGMM(
                n_components=self.n_clusters,
                n_init=self.n_init,
                random_state=self.random_state,
            ).fit(emb)
            self.labels_ = sgmm.predict(emb)
            self.cluster_centers_ = sgmm.means_
            self.probabilities_ = sgmm.predict_proba(emb)
        else:
            raise ValueError(f"Unknown method: {self.method!r}")

        return self

    def fit_predict(self, X, y=None) -> np.ndarray:
        return self.fit(X).labels_

    # `probabilities_` is exposed as a property so we can raise a helpful error
    # when the chosen method doesn't produce probabilities.
    @property
    def probabilities_(self) -> np.ndarray:
        check_is_fitted(self, attributes=["labels_"])
        if self.method == "kmeans":
            raise AttributeError(
                "probabilities_ requires method='gmm' or 'simple_gmm'"
            )
        try:
            return self._probabilities_
        except AttributeError as e:
            raise AttributeError(
                "probabilities_ is not set; this is a bug in ZeusClusterer.fit."
            ) from e

    @probabilities_.setter
    def probabilities_(self, value: np.ndarray) -> None:
        self._probabilities_ = value
```

- [ ] **Step 4: Run API tests to verify they pass**

Run: `pytest tests/test_api.py -v`

Expected: all PASS. This is the slowest test file (each call runs a 12-layer transformer forward on CPU); the whole file should still finish in <60s for a 20-row fixture.

- [ ] **Step 5: Run the whole suite to confirm nothing regressed**

Run: `pytest tests/ -v`

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add zeus/api.py tests/test_api.py
git commit -m "feat(api): add sklearn-style Zeus and ZeusClusterer estimators"
```

---

### Task 9: Wire up `zeus/__init__.py`

**Files:**
- Modify: `zeus/__init__.py` (currently empty)
- Create: `tests/test_public_api.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_public_api.py`:

```python
"""The public surface advertised by `zeus.__init__`."""
import zeus


def test_top_level_exports():
    assert hasattr(zeus, "Zeus")
    assert hasattr(zeus, "ZeusClusterer")
    assert hasattr(zeus, "__version__")
    assert isinstance(zeus.__version__, str)


def test_no_torch_train_imports_on_module_load():
    """Importing zeus must not pull in wandb, openml, or omegaconf."""
    import sys
    for forbidden in ("wandb", "openml", "omegaconf"):
        assert forbidden not in sys.modules, (
            f"`import zeus` should not have imported {forbidden}"
        )
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_public_api.py -v`

Expected: FAIL — `zeus.Zeus` not yet exported at the top level.

- [ ] **Step 3: Write `zeus/__init__.py`**

```python
"""ZEUS — Zero-shot Embeddings for Unsupervised Separation of tabular data.

Public API:
  - Zeus           — sklearn TransformerMixin; produces row embeddings.
  - ZeusClusterer  — sklearn ClusterMixin; produces hard (and optional soft) labels.

See https://github.com/upadhyan/zeus for installation and examples;
upstream research codebase: https://github.com/gmum/zeus.
"""
from zeus.api import Zeus, ZeusClusterer

__version__ = "0.1.0"
__all__ = ["Zeus", "ZeusClusterer", "__version__"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_public_api.py -v`

Expected: PASS.

- [ ] **Step 5: Full suite check**

Run: `pytest tests/ -v`

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add zeus/__init__.py tests/test_public_api.py
git commit -m "feat: expose Zeus / ZeusClusterer from zeus.__init__"
```

---

### Task 10: Delete obsoleted files

This task is destructive but reversible (everything stays in git history). Do it in one commit so a single revert restores the old layout. **Do not skip any of the deletions** — each was confirmed unused after Tasks 1-9.

**Files:** (all to be **deleted**)
- `pretrain.py`
- `evaluation.py`
- `zeus.pt`
- `requirements.txt`
- `zeus/datasets.py`
- `zeus/wandb_logging.py`
- `zeus/visualization.py`
- `zeus/configs.py`
- `zeus/initialziation.py`
- `zeus/utils.py`
- `synthetic_datasets/` (recursive)

- [ ] **Step 1: Final confirmation no remaining imports inside `zeus/`**

Run:

```bash
grep -RIn -E "from (zeus\.(configs|datasets|wandb_logging|visualization|initialziation|utils)|inference_methods)" zeus/ tests/ 2>/dev/null
```

Expected: no output. If anything appears, fix that file's imports first — do not proceed until clean.

- [ ] **Step 2: Verify nothing in tests references the doomed files**

Run:

```bash
grep -RIn -E "(pretrain|evaluate_model|openml_ids|gmm_loss|hungarian|setup_seed|wandb_logging)" tests/ 2>/dev/null
```

Expected: no output.

- [ ] **Step 3: Delete the files via `git rm`**

```bash
git rm pretrain.py evaluation.py requirements.txt zeus.pt
git rm zeus/datasets.py zeus/wandb_logging.py zeus/visualization.py
git rm zeus/configs.py zeus/initialziation.py zeus/utils.py
git rm -r synthetic_datasets
```

If `zeus.pt` is not tracked (it was added as a binary in commit `f9a3e7b` — confirm with `git ls-files | grep zeus.pt`), use plain `rm zeus.pt` instead.

- [ ] **Step 4: Confirm the tree**

Run: `find zeus/ -type f | sort`

Expected output (exact list):

```
zeus/__init__.py
zeus/_config.py
zeus/api.py
zeus/inference_methods/__init__.py
zeus/inference_methods/simple_gmm.py
zeus/model/__init__.py
zeus/model/encoders.py
zeus/model/layer.py
zeus/model/zeus.py
zeus/preprocessing.py
zeus/weights.py
```

- [ ] **Step 5: Re-run the full test suite**

Run: `pytest tests/ -v`

Expected: every test from Tasks 1-9 still passes. Nothing in the new code path references the deleted modules; if a test breaks here, that's the regression to fix.

- [ ] **Step 6: Commit**

```bash
git commit -m "refactor: drop training/eval scripts and obsoleted modules"
```

---

### Task 11: Rewrite `README.md`

Preserve the original content verbatim below the new sections, with inline "removed" admonitions on the affected subsections (spec §5.9).

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Save the original README content**

Run: `cp README.md /tmp/zeus-original-readme.md`

(Just a scratch copy — not committed. You'll paste from this into the new README below the dividing line.)

- [ ] **Step 2: Write the new `README.md`**

Replace the entire content of `README.md` with:

````markdown
# ZEUS — pip-installable fork

> **Note:** This is a personal fork of <https://github.com/gmum/zeus>,
> repackaged for easier pip installation and a cleaner sklearn-style API.
> Please cite the original paper and refer to the upstream repo for the
> canonical research codebase.

## Installation

```bash
pip install git+https://github.com/upadhyan/zeus.git
```

The 305 MB checkpoint downloads automatically on first use and is cached
in your platform's user cache directory (override with `$ZEUS_CACHE_DIR`).

> **Tested with:** Python 3.11, PyTorch 2.5.1 + CUDA 12.1. Any
> `torch >= 2.0` build (including CPU-only) should work.

## Quick start

```python
import pandas as pd
from zeus import Zeus, ZeusClusterer

df = pd.read_csv("mydata.csv")

# 1) Embeddings only
emb = Zeus().fit_transform(df)         # (n, 512) numpy array

# 2) End-to-end clustering
labels = ZeusClusterer(n_clusters=5).fit_predict(df)

# 3) Soft assignments
clf = ZeusClusterer(n_clusters=5, method="simple_gmm").fit(df)
probs = clf.probabilities_             # (n, 5)
```

## API reference

### `zeus.Zeus(*, device="auto", preprocess=True, model_path=None, cache_dir=None)`

sklearn `TransformerMixin`. Methods:

- `fit(X)` — no-op (ZEUS is zero-shot); returns `self`.
- `transform(X)` — returns `(n, 512)` numpy embeddings.
- `fit_transform(X)` — inherited; equivalent to `transform(X)`.

Accepts `np.ndarray`, `pd.DataFrame`, or `torch.Tensor` of shape `(n, d)`.
With `preprocess=True` (default), DataFrames with mixed dtypes are
auto-encoded (one-hot for object/category/bool columns, mean-imputation
for NaNs, PCA-or-zero-pad to dim 30, MinMax-scale to `[-1, 1]`). With
`preprocess=False`, the input must already be a numeric `(n, 30)` array.

**Important:** embeddings depend on every other row in the batch. Calling
`transform(X_test)` after `fit(X_train)` is **not** equivalent to running
both at once — use `fit_transform(X)` on the dataset you want to embed.

### `zeus.ZeusClusterer(n_clusters, *, method="kmeans", device="auto", preprocess=True, model_path=None, cache_dir=None, random_state=None, n_init=10)`

sklearn `ClusterMixin`. Methods:

- `fit(X)` — runs the encoder, MinMax-scales to `[-1, 1]`, then runs the chosen clusterer.
- `fit_predict(X)` — returns `(n,)` int labels.

Fitted attributes: `labels_`, `embedding_`, `cluster_centers_`, and
(when `method != "kmeans"`) `probabilities_` with shape `(n, n_clusters)`
summing to 1 per row.

No `predict(X_new)` is provided — same context-dependence reason as `Zeus`.

## Citation

Please cite the original paper:

```bibtex
@article{zeus2025,
  title={ZEUS: Zero-shot Embeddings for Unsupervised Separation of Tabular Data},
  url={https://arxiv.org/abs/2505.10704},
  year={2025}
}
```

## License

This fork inherits the TabPFN v1 license (see `legal/`).

---

# Original README

> **Note:** This section preserves the original README from
> <https://github.com/gmum/zeus> verbatim, with inline notes on
> commands and files that have been removed from this fork.

# ZEUS: Zero-shot Embeddings for Unsupervised Separation of Tabular Data

Code repository for [https://arxiv.org/abs/2505.10704](https://arxiv.org/abs/2505.10704).

Repository is based on the first version of TabPFN. The license is located in the [legal](legal) folder. Link to TabPFN2 repository
[https://github.com/PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN).

## Abstract
Clustering tabular data remains a significant open challenge in data analysis and machine learning.
Unlike for image data, similarity between tabular records often varies across datasets,
making the definition of clusters highly dataset-dependent. Furthermore,
the absence of supervised signals complicates hyperparameter tuning in deep learning clustering methods,
frequently resulting in unstable performance. To address these issues and reduce the need for per-dataset tuning,
we adopt an emerging approach in deep learning: zero-shot learning. We propose ZEUS,
a self-contained model capable of clustering new datasets without any additional training or fine-tuning.
It operates by decomposing complex datasets into meaningful components that can then be clustered effectively.
Thanks to pre-training on synthetic datasets generated from a latent-variable prior,
it generalizes across various datasets without requiring user intervention. To the best of our knowledge,
ZEUS is the first zero-shot method capable of generating embeddings for tabular data in a fully unsupervised manner.
Experimental results demonstrate that it performs on par with or better than traditional clustering algorithms
and recent deep learning-based methods, while being significantly faster and more user-friendly.

## Setup

> **Note:** Manual conda setup is no longer needed in this fork — see the
> Installation section above. The original recipe is preserved below for
> reference.

Setup with conda environment.

```shell
conda create -n zeus python=3.11
conda activate zeus
pip install -r requirements.txt
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

## Experiments
Details of ZEUS configuration parameters can be found in the [zeus/configs.py](zeus/configs.py) file.

## Pre-training

> **Note:** `pretrain.py` has been removed from this fork (inference-only
> package). To retrain ZEUS from scratch, use the original repo at
> <https://github.com/gmum/zeus>.

Pre-training can be performed using the following command:

```shell
python pretrain.py nr_epochs=300 dim=30 use_pca=True num_test_datasets=200 num_categorical=3 pca_dim=30 learning_rate=2e-5 inf_method=KMEANS
```

## Model checkpoint

> **Note:** This fork auto-downloads weights from GitHub Releases on first
> use — no manual download required. The original Google Drive link is
> preserved below.

ZEUS checkpoint is available at [Google Drive](https://drive.google.com/file/d/1D7uikacymUnmmMxjUjBuCNIomqhBWS67/view?usp=sharing).


## Evaluation

> **Note:** `evaluation.py` has been removed from this fork. The
> sklearn-style API replaces it — see Quick start above. For the original
> OpenML evaluation harness, use <https://github.com/gmum/zeus>.

The evaluation of ZEUS can be executed as follows:

```shell
python .\evaluation.py model_path=zeus.pt inf_method=KMEANS eval_dataset=OPENML metric_type=ARI results_file=openml.csv
```
````

(End-of-fence note: the closing fence above is four backticks because the README contains its own triple-backtick code blocks. When you paste this into the file, the wrapping fence must be at least one backtick longer than any fence it contains.)

- [ ] **Step 3: Verify the diff renders correctly**

Run: `git diff README.md | head -100`

Skim the diff to confirm the new content is at the top and the original content is preserved verbatim below.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for the pip-installable sklearn-style API"
```

---

### Task 12: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md` (currently untracked but exists in the working tree)

- [ ] **Step 1: Read the current CLAUDE.md to understand what's still accurate**

Run: `cat CLAUDE.md`

Most of the architecture sections describe internals that haven't changed — keep those. The Commands, Environment, and import-path sections need updating.

- [ ] **Step 2: Rewrite the affected sections**

Replace the `## Environment`, `## Commands`, and the bullets in `## Conventions / gotchas` that no longer apply. Use this drop-in replacement for the early sections (keep everything from `## Architecture` onward, with only the import-path typo note removed):

```markdown
## Environment

Python ≥3.10. PyTorch ≥2.0 (any build — CPU, CUDA, MPS).

```shell
pip install -e .             # editable install for development
pip install -e ".[test]"     # plus pytest
```

The package is install-only via the git URL; there is no PyPI release.
A CUDA build of PyTorch is recommended for non-trivial inputs but not
required — `Zeus(device="auto")` falls back to CPU.

## Commands

The user-facing surface is sklearn-style; there are no scripts.

```python
from zeus import Zeus, ZeusClusterer

emb     = Zeus().fit_transform(df)
labels  = ZeusClusterer(n_clusters=5).fit_predict(df)
```

The checkpoint downloads on first use. Override its location via
`Zeus(model_path=...)` or the `ZEUS_CACHE_DIR` env var.

Tests:

```shell
pytest tests/ -v
```

No CI, no linter config.
```

For the `## Conventions / gotchas` section, **remove** the bullets about:

- "Batch size is always 1" (no longer relevant — the data-loader is gone)
- "Seed is fixed at 42" (no longer fixed — `setup_seed` is deleted)
- "`output_dir` defaults to `'results'`" (no `output_dir` anymore)
- "Cluster count for synthetic data is sampled in `[2, num_gaussians+1)`" (training-only)

**Add** these replacement bullets:

- **Embeddings are batch-context-dependent.** Self-attention runs across rows, so a row's embedding depends on every other row in the same `transform` call. `Zeus.fit` is therefore a no-op; `ZeusClusterer` exposes only `fit_predict`, never `predict(X_new)`.
- **Frozen checkpoint constants live in `zeus/_config.py`** (`EMBED_DIM=512`, `INPUT_DIM=30`, `NUM_GAUSSIANS=10`, `N_LAYERS=12`, `N_HEAD=4`, `HID_DIM=1024`). The released `zeus.pt` bakes these in; pointing `model_path=` at a checkpoint trained with different hyperparameters is unsupported.
- **`state_dict` is loaded with `strict=False`.** The checkpoint contains parameters from the upstream training-time `decoder`/`out_layer` branches that this fork removed; those become `missing_keys` and are tolerated silently by `zeus/weights.py`. Anything else missing emits a `warnings.warn`.

Also **remove** the entire `**Config flow.**` paragraph (no more `initialziation.py`) and the `_efficient_eval_masking_` discussion stays accurate, just delete the reference to `initialize()`.

- [ ] **Step 3: Run tests one more time as a sanity check**

Run: `pytest tests/ -v`

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for the new package layout and API"
```

---

### Task 13 (manual / human-only): Cut the GitHub release and fill in the SHA-256

This is a one-time manual step that **cannot be done by an autonomous worker** because it requires repo-owner credentials and uploading a binary asset. Surface this to the human after Task 12 is done.

**Files:**
- Modify: `zeus/weights.py` (the `EXPECTED_SHA256` constant)

- [ ] **Step 1: Compute the SHA-256 of the local checkpoint** (whoever has it on disk)

```bash
sha256sum zeus.pt   # or shasum -a 256 zeus.pt on macOS
```

- [ ] **Step 2: Create a GitHub release tagged `v0.1.0` on `upadhyan/zeus`**

Upload `zeus.pt` as a release asset. Confirm the resulting download URL matches `https://github.com/upadhyan/zeus/releases/download/v0.1.0/zeus.pt`.

- [ ] **Step 3: Replace the placeholder SHA-256 in `zeus/weights.py`**

Edit `zeus/weights.py`:

```python
EXPECTED_SHA256 = "0" * 64
```

→

```python
EXPECTED_SHA256 = "<the hash printed in step 1>"
```

- [ ] **Step 4: Manually test the download path on a fresh machine** (or by clearing the cache)

```bash
rm -rf ~/.cache/zeus
python -c "from zeus import Zeus; Zeus().fit_transform(__import__('pandas').DataFrame({'a': [1,2,3,4,5,6,7,8,9,10]*3, 'b': [0.5]*30})).shape"
```

Expected: the download progress bar appears, then the script prints `(30, 512)`.

- [ ] **Step 5: Commit**

```bash
git add zeus/weights.py
git commit -m "release: lock in v0.1.0 checkpoint SHA-256"
git tag v0.1.0
git push origin main v0.1.0
```

---

## Done

After Task 12, the package is functional end-to-end on the developer's machine (assuming they have the checkpoint locally and pass it via `model_path=`). Task 13 is required before users can `pip install git+...` and have things "just work" with no manual checkpoint download.
