# ZEUS Pip Package & sklearn-Style API — Design

- **Date:** 2026-05-16
- **Status:** Approved (pending spec review)
- **Author:** upadhyan (fork maintainer)
- **Upstream:** https://github.com/gmum/zeus (original ZEUS implementation; this is a personal fork being repackaged)

## 1. Context

The current repository is a research codebase derived from TabPFN v1. It works but is uncomfortable to use:

- Two top-level scripts (`pretrain.py`, `evaluation.py`) both unconditionally call `wandb.init()`, making inference impossible without a wandb account.
- The "API" is OmegaConf CLI `key=value` parsing through a single 75-field `GMMConfig` dataclass. There is no Python-callable entry point that takes a numpy array and returns embeddings.
- Two parallel top-level packages — `zeus/` and `inference_methods/` — meaning consumers must add both to their path.
- `zeus.pt` (305 MB checkpoint) sits in the repo root and is fetched via a manual Google Drive download.
- A spelling typo (`zeus/initialziation.py`) is load-bearing for imports.
- Heavy dependencies (`wandb`, `openml`, `matplotlib`, `tqdm`) are mandatory even for pure inference.

The goal of this work is to make this fork pip-installable and to expose a clean sklearn-style API on top of the existing pretrained checkpoint. The model itself is not changing.

## 2. Goals

- `pip install git+https://github.com/upadhyan/zeus.git` installs a working package with no manual checkpoint download and no wandb requirement.
- `from zeus import Zeus, ZeusClusterer` gives users two sklearn-compatible estimators.
- First call to either estimator transparently downloads and caches the checkpoint.
- Inference works on CPU and GPU; device default auto-detects.
- DataFrames with mixed dtypes "just work" by default.
- The original paper/research content remains discoverable in the README.

## 3. Non-goals

- Pretraining or fine-tuning. The original `pretrain.py` is being removed from this fork; anyone wanting to retrain should use the upstream repository.
- OpenML evaluation harness. Removed for the same reason.
- PyPI distribution. The package is installed via the git URL only.
- A `predict(X_new)` API on the clusterer. ZEUS embeddings are batch-context-dependent (self-attention across rows), so prediction on unseen rows is not well-defined.
- Internal batching of very large inputs. Documented as a memory caveat instead.
- CI setup. Tests exist locally; CI is a follow-up.

## 4. Decisions

| Decision | Choice |
|---|---|
| Package scope | Inference-only |
| Distribution | Git URL install (no PyPI) |
| API surface | `Zeus` (encoder, `TransformerMixin`) + `ZeusClusterer` (clusterer, `ClusterMixin`) |
| Clusterer prediction surface | `fit_predict` only; no `predict(X_new)` |
| Weight hosting | GitHub Releases asset on `upadhyan/zeus` |
| Cache location | `platformdirs.user_cache_dir("zeus")`, overridable via `ZEUS_CACHE_DIR` env var |
| Input handling | Auto-handle DataFrames (one-hot + impute + scale + PCA/pad); `preprocess=False` bypass |
| Device default | `"auto"` — CUDA if available else CPU |
| `n_init` semantics | Pass through honestly to whichever backend; default 10 |
| Training scripts | `pretrain.py`, `evaluation.py` deleted |
| Original README content | Preserved verbatim below new content, with inline "removed" notes |

## 5. Architecture

### 5.1 Frozen checkpoint constants

The released `zeus.pt` bakes in specific model hyperparameters; the inference package treats these as constants rather than configurable knobs. They live in `zeus/_config.py`:

```python
EMBED_DIM = 512        # ZeusTransformerModel ninp / output feature dim
N_HEAD = 4
HID_DIM = 1024
N_LAYERS = 12
NUM_GAUSSIANS = 10     # number of learned cluster_centers in the model
INPUT_DIM = 30         # model's expected input feature dimension
DROPOUT = 0.0
EFFICIENT_EVAL_MASKING = True
```

These are not user-tunable. Passing a `model_path` to a checkpoint trained with different hyperparameters is unsupported — the load will fail or produce garbage. (A future-proof variant would store these in the checkpoint and read them back at load time; out of scope for v0.1.0.)

### 5.2 Final package layout

```
zeus/
├── __init__.py          # re-exports: Zeus, ZeusClusterer, __version__
├── api.py               # Zeus, ZeusClusterer (sklearn-style estimators)
├── preprocessing.py     # DataFrame -> (N, dim=30) tensor pipeline
├── weights.py           # download + cache for zeus.pt
├── _config.py           # internal frozen dataclass: ModelHParams matching checkpoint
├── model/
│   ├── __init__.py
│   ├── zeus.py          # ZeusTransformerModel (unchanged module body)
│   ├── layer.py         # custom TransformerEncoderLayer
│   └── encoders.py      # Linear encoder
└── inference_methods/   # moved from sibling location
    ├── __init__.py
    └── simple_gmm.py
```

Repo root retains: `README.md` (rewritten top, original preserved below), `LICENSE`, `legal/`, `.gitignore`, `pyproject.toml` (new), `CLAUDE.md` (updated for the new layout), `docs/`.

### 5.3 Deletions

The following files/directories are removed entirely from the working tree on `main` (they remain accessible via `git log` if needed):

- `pretrain.py`
- `evaluation.py`
- `inference_methods/` (sibling — moved inside `zeus/`)
- `synthetic_datasets/` (only used by deleted `evaluation.py`)
- `zeus.pt` (replaced by GitHub Releases asset; not git-tracked)
- `zeus/datasets.py` (synthetic-data generation + OpenML loading — training-only)
- `zeus/wandb_logging.py`
- `zeus/visualization.py`
- `zeus/configs.py` (replaced by `_config.py`; `LossType`, `MetricType`, `EvalDatasetType`, `InferenceMethodType`, and the 75-field `GMMConfig` are all replaced — `LossType` was used at inference only as a no-op branch gate in `ZeusTransformerModel.forward` and `__init__`; see "Required edits to `model/zeus.py`" below)
- `zeus/initialziation.py` (the typo file — replaced by `weights.py` + new constructors)
- `zeus/model/model_utils.py` (`get_cosine_schedule_with_warmup` is training-only; the module-level `default_device` constant is unused at inference and dropped; the small helpers `SeqBN` and `bool_mask_to_att_mask` referenced from `model/zeus.py` and `model/layer.py` are inlined into those files)
- `requirements.txt` (replaced by `pyproject.toml`)

`zeus/utils.py` is gutted. The only inference-relevant logic — the post-embedding scaling + KMeans/GMM/SimpleGMM dispatch currently in `predict_clusters` — is moved into `zeus/api.py` (inside `ZeusClusterer.fit`). Everything else in `utils.py` (`gmm_loss_with_regularizes`, `hungarian_algorithm`, `evaluate_model`, `openml_ids`, `true_class_assignments`, `setup_seed`) is deleted.

#### Required edits to `model/zeus.py`

The current `ZeusTransformerModel` imports `LossType` from the deleted `zeus.configs` and uses it in two places:

```python
# constructor signature
def __init__(self, ..., loss_type=LossType.CENTER, ...):
    ...
    self.loss_type = loss_type

# forward()
if self.loss_type == LossType.CENTER:
    return output
# (the lines below this branch are already commented out)
```

At inference, only `LossType.CENTER` is ever used. The refactor drops the `loss_type` parameter from the constructor and removes the branch — `forward` simply returns `output` directly. This removes the dependency on `zeus.configs`. No other model module imports anything from `configs.py`.

Additionally, the constructor currently creates an unused `decoder` (and conditionally `out_layer`) module gated by `distance_based_logit`. These modules' weights exist in the checkpoint but are never executed at inference. The refactor drops both `distance_based_logit` and `decoder` / `out_layer` from `__init__` entirely. The `state_dict` load handles the resulting extra keys via `strict=False` (see section 5.5).

After these edits, `ZeusTransformerModel.__init__` signature becomes:

```python
def __init__(self, encoder, ninp, nhead, nhid, nlayers, *,
             dropout=0.0, n_clusters=NUM_GAUSSIANS,
             input_normalization=False, pre_norm=False,
             activation='gelu', recompute_attn=False,
             full_attention=False, all_layers_same_init=False,
             efficient_eval_masking=True):
```

The unused `n_out` and `decoder` (callable) arguments are removed. `model/zeus.py`'s `predict_embedding` helper is preserved (or deleted; not load-bearing — `Zeus.transform` performs the unsqueeze/slice itself, see section 5.4).

### 5.4 Public API

#### `zeus.Zeus`

```python
class Zeus(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        *,
        device: str | torch.device = "auto",
        preprocess: bool = True,
        model_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ): ...

    def fit(self, X, y=None) -> Self: ...        # no-op (zero-shot); validates only
    def transform(self, X) -> np.ndarray: ...    # returns (n, 512)
    # fit_transform inherited from TransformerMixin
```

- `transform` accepts `np.ndarray`, `pd.DataFrame`, or `torch.Tensor` of shape `(n, d)` and always returns a `np.ndarray` of shape `(n, embed_dim)` where `embed_dim = EMBED_DIM = 512`.
- Internal flow inside `transform`:
  1. `x = prepare_inputs(X, target_dim=INPUT_DIM)` (if `preprocess=True`) or pass `X` through after shape validation.
  2. `x = x.unsqueeze(1).to(device)` — model expects `(N, 1, INPUT_DIM)`.
  3. `with torch.no_grad(): out = model(x)`.
  4. `out = out[:-NUM_GAUSSIANS].squeeze(1)` — strip the `NUM_GAUSSIANS=10` learned cluster centers concatenated by the model; what remains is the per-row embedding.
  5. Move to CPU, `.numpy()`, return.

  The model exposes a `predict_embedding` helper that already implements steps 2–4 plus an in-graph MinMax scale to `[-1, 1]`. The refactor will **not** use that helper for `Zeus.transform` — we want raw embeddings here, leaving the `[-1, 1]` scaling to `ZeusClusterer` (which is where the existing pipeline applies it). `predict_embedding` may be deleted or kept as a convenience method; not load-bearing either way.
- `fit` is a no-op because ZEUS is zero-shot. It returns `self` after a basic shape/finite check. It exists so `Zeus` is a drop-in `TransformerMixin` (which depends on `fit`).
- **Context-dependence note** (mandatory in docstring): embeddings depend on every other row in the same `transform` call because the transformer attends across rows. `fit(X_train)` followed by `transform(X_test)` is therefore *not* equivalent to running both together; for the intended use case call `fit_transform(X)` on the dataset you want to embed.

#### `zeus.ZeusClusterer`

```python
class ZeusClusterer(ClusterMixin, BaseEstimator):
    def __init__(
        self,
        n_clusters: int,
        *,
        method: Literal["kmeans", "gmm", "simple_gmm"] = "kmeans",
        device: str | torch.device = "auto",
        preprocess: bool = True,
        model_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        random_state: int | None = None,
        n_init: int = 10,
    ): ...

    def fit(self, X, y=None) -> Self:
        # encode X via internal Zeus(), MinMax-scale to [-1, 1],
        # run chosen clusterer; populate labels_, embedding_, cluster_centers_,
        # and (for probabilistic methods) probabilities_
        ...

    def fit_predict(self, X, y=None) -> np.ndarray:   # (n,) int labels
    @property
    def probabilities_(self) -> np.ndarray:           # only for method ∈ {"gmm", "simple_gmm"}
```

- `predict(X_new)` is **not** implemented (same context-dependence reason as `Zeus`; follows sklearn precedent of DBSCAN / AgglomerativeClustering, which also expose only `fit_predict`).
- **Soft assignments are exposed as a fitted attribute, not a method.** For `method ∈ {"gmm", "simple_gmm"}`, `fit` computes per-row soft assignments alongside the hard labels and stashes them on `self.probabilities_` (shape `(n, n_clusters)`, rows sum to 1). Accessing this attribute when `method="kmeans"` raises `AttributeError("probabilities_ requires method='gmm' or 'simple_gmm'")`. Rationale: a `predict_proba(X)` method is misleading here, because re-scoring on a new `X` would re-embed and re-fit (changing the cluster identities), and re-scoring on the same `X` is a stateful lookup, not a function. The fitted attribute pattern is honest about the constraint and removes the array-equality check the earlier draft proposed.
- After `fit`, `self.labels_`, `self.embedding_`, and `self.cluster_centers_` are populated; `self.probabilities_` additionally for probabilistic methods.
- `n_init` is passed through identically to each backend (no `/10`, no hardcoded override). Default `n_init=10` matches sklearn conventions and matches the value the previous GMM path was actually receiving.

#### Usage examples

```python
from zeus import Zeus, ZeusClusterer

# Embeddings only
emb = Zeus().fit_transform(df)

# End-to-end clustering with auto-preprocessing
labels = ZeusClusterer(n_clusters=5).fit_predict(df)

# Soft assignments
clf = ZeusClusterer(n_clusters=5, method="simple_gmm").fit(df)
probs = clf.probabilities_
```

### 5.5 Weight download & caching (`zeus/weights.py`)

```python
DEFAULT_RELEASE_URL = "https://github.com/upadhyan/zeus/releases/download/v0.1.0/zeus.pt"
EXPECTED_SHA256 = "<filled in once the release is cut>"

def get_checkpoint_path(
    model_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> Path: ...
```

Resolution order:

1. `model_path` argument (if given) — return verbatim, no download, no checksum check.
2. `cache_dir` argument, else `$ZEUS_CACHE_DIR`, else `platformdirs.user_cache_dir("zeus")`.
3. If `<cache_dir>/zeus.pt` exists *and* its SHA-256 matches `EXPECTED_SHA256` → return that path.
4. Otherwise:
   a. Print `"Downloading ZEUS weights (~305 MB) to <path>..."` (once).
   b. Stream the download with a `tqdm` progress bar (stdlib `urllib.request`).
   c. Write to `<cache_dir>/zeus.pt.tmp`.
   d. Verify SHA-256.
   e. Atomic rename to `zeus.pt`.
   f. Return the path.

A corrupt file (failing checksum) is treated as a re-download trigger, not a hard error. If the download fails (network, 404, etc.) a clear exception surfaces with the URL and the local path the user can populate manually as `model_path=`.

#### State-dict load semantics

A small companion helper `load_zeus_state_dict(model, path, device)` lives next to `get_checkpoint_path` and is called from `Zeus.__init__` (and shared by `ZeusClusterer` via its internal `Zeus`). Behavior:

1. `obj = torch.load(path, map_location=device)`.
2. If `obj` is a `dict` and `"model"` is a key, use `obj["model"]`; otherwise assume `obj` is itself a state-dict. This preserves compatibility with both the existing checkpoint format (which wraps the state-dict in a `{"model": ..., "optimizer": ..., "epoch": ...}` dict, see `pretrain.py:158`) and a clean exported state-dict.
3. `model.load_state_dict(state_dict, strict=False)`.
4. The returned `_IncompatibleKeys` is inspected: if `missing_keys` is non-empty for anything other than the known-unused `decoder.*` / `out_layer.*` parameters (left over from the non-CENTER loss branch that the refactor is removing), warn via `warnings.warn` once. `unexpected_keys` is silently allowed (extra optimizer/scheduler state, etc.).

`strict=False` is preserved from the existing `initialziation.py` loader. This is necessary because: (a) the existing checkpoint contains parameters from the `decoder` / `out_layer` branches that aren't used at inference, and (b) the refactor drops those branches from `ZeusTransformerModel.__init__`, so the loaded keys won't all match anyway.

### 5.6 Preprocessing (`zeus/preprocessing.py`)

Single entry point:

```python
def prepare_inputs(X, target_dim: int = 30) -> torch.Tensor: ...
```

Behavior when `preprocess=True`:

1. **DataFrame** → split by dtype:
   - `object` / `category` / `bool` columns → one-hot via `pd.get_dummies(..., dummy_na=False)`.
   - Numeric columns → mean-impute NaN with per-column means.
   - Concatenate to a numeric matrix.
2. **ndarray / Tensor** → mean-impute NaN per column if any are present (warn once via `warnings.warn`).
3. **Dim adjust** (PCA / pad happens *before* scaling, matching existing `evaluate_model` order):
   - If `d > target_dim`: PCA to `target_dim`.
   - If `d < target_dim`: zero-pad on the right to `target_dim`.
   - If `d == target_dim`: pass through.
4. **Scale**: `MinMaxScaler(feature_range=(-1, 1))` fit on the dim-adjusted matrix.
5. Return a `torch.float32` tensor of shape `(n, target_dim)`.

Order rationale: the existing `evaluate_model` (`zeus/utils.py:138-146`) does PCA first, then MinMax-scales the PCA outputs to `[-1, 1]`. Reversing the order would scale the original features and leave the PCA components un-normalized, which the model wasn't trained against. We match the existing order for behavioral parity with the published numbers.

One behavioral *change* from the existing code: the existing code only MinMax-scales when `d > target_dim` (the PCA branch) and skips scaling when `d <= target_dim`. This was almost certainly an oversight given the synthetic training pipeline always produces normalized inputs; we apply scaling uniformly in all three branches so user-facing behavior is consistent. The "Risks & open items" section flags this as a small intentional deviation that should be empirically validated against the published OpenML numbers.

The scaler and PCA are fit fresh each call. No state is retained between calls — consistent with the existing `evaluate_model` behavior and with the context-dependence of the model.

When `preprocess=False`: input must be a 2-D numeric array/tensor with `d == target_dim`. A clear `ValueError` is raised otherwise.

### 5.7 Packaging (`pyproject.toml`)

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "zeus"
version = "0.1.0"
description = "Zero-shot Embeddings for Unsupervised Separation of tabular data"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "numpy>=1.24",
    "scipy>=1.10",
    "scikit-learn>=1.3",
    "pandas>=2.0",
    "platformdirs>=4.0",
    "tqdm>=4.60",
]
license = { file = "LICENSE" }

[project.urls]
Homepage = "https://github.com/upadhyan/zeus"
Upstream = "https://github.com/gmum/zeus"
Paper = "https://arxiv.org/abs/2505.10704"

[tool.setuptools.packages.find]
include = ["zeus*"]
```

- `torch` is unpinned. The original `torch==2.5.1+cu121` pin is moved to a README "Tested with" note. Pinning a specific CUDA build in a public package breaks Mac/CPU/different-CUDA users.
- All other deps are inference-only. No `wandb`, `openml`, `omegaconf`, `matplotlib`, or `torchmetrics`.

### 5.8 Tests (`tests/`)

A minimal pytest suite covering:

- `test_preprocessing.py`:
  - numpy / DataFrame / torch tensor inputs all produce `(n, 30)` float32 tensors.
  - NaN handling: rows with NaN are imputed; warn-on-NaN fires once.
  - Dim mismatch: input with `d=10` is zero-padded; `d=50` is PCA'd to 30; `d=30` passes through.
  - DataFrame with mixed dtypes: object/category columns become one-hot.
- `test_weights.py`:
  - Resolution order: explicit `model_path` > `cache_dir` arg > `$ZEUS_CACHE_DIR` > platform default.
  - Existing file with correct checksum is not re-downloaded.
  - Existing file with wrong checksum triggers re-download (HTTP layer monkeypatched).
- `test_api.py`:
  - `Zeus().fit_transform(small_random_df)` returns `(n, 512)`. Fixture uses `n=20`, `d=8` so the forward pass fits comfortably in CPU memory and runs in <30s without a GPU.
  - `ZeusClusterer(n_clusters=3).fit_predict(small_random_df)` returns `(n,)` int labels with at most 3 unique values.
  - `ZeusClusterer(..., method="kmeans").probabilities_` raises `AttributeError`.
  - `ZeusClusterer(..., method="simple_gmm")` produces `probabilities_` with each row summing to 1.
  - Tests requiring the checkpoint are skipped (`pytest.skip`) if it's not cached and no network is available, so the suite runs offline.

No CI configuration is included in this work. That's a follow-up.

### 5.9 README

Reorganized as:

```
# ZEUS — pip-installable fork

> Personal fork of https://github.com/gmum/zeus, repackaged for easier
> pip installation and a cleaner API. Please cite the original paper
> and refer to the upstream repo for the canonical research codebase.

## Installation
## Quick start
## API reference

---

# Original README

<all original content, verbatim, with inline "Note: removed" admonitions
on the Setup / Pre-training / Model checkpoint / Evaluation subsections>
```

Each removed feature carries a short inline note pointing readers to `https://github.com/gmum/zeus` for the original workflow.

### 5.10 CLAUDE.md updates

CLAUDE.md is updated to reflect:

- The new module layout (`zeus.api`, `zeus.preprocessing`, `zeus.weights`).
- The new install/test commands (`pip install -e .`, `pytest`).
- The fact that `pretrain.py` / `evaluation.py` no longer exist (and where to go for them).
- The typo `initialziation.py` no longer exists.
- The sklearn-style API (Zeus, ZeusClusterer).
- Cache and weight-download behavior.

The "forward pass shape convention", "loss", "transformer layer specifics", "config flow" subsections of the existing CLAUDE.md are pruned to only the parts that remain accurate after the refactor.

## 6. Edge cases & error behavior

| Situation | Behavior |
|---|---|
| `device="auto"` and no CUDA visible | Falls back to CPU; logs an info message once. |
| User passes `device="cuda"` and CUDA isn't available | `RuntimeError` from torch is surfaced unmodified. |
| `n_clusters >= n_samples` in `ZeusClusterer.fit` | Sklearn-style: raises `ValueError`. |
| Input contains all-NaN column | `prepare_inputs` raises `ValueError` (cannot mean-impute). |
| `transform(X)` with `preprocess=False` and `d != 30` | `ValueError("expected (n, 30), got (n, {d})")`. |
| Download fails mid-stream | Partial `.tmp` file is removed; original `zeus.pt` (if present) is untouched; exception surfaces. |
| Checkpoint file present but corrupt | One automatic re-download attempt; if that also fails checksum, raise with both the URL and the local path so the user can replace it manually. |
| `probabilities_` accessed with `method="kmeans"` | `AttributeError("probabilities_ requires method='gmm' or 'simple_gmm'")`. |
| `labels_` / `probabilities_` accessed before `fit` | `NotFittedError` from sklearn's `check_is_fitted`. |

## 7. Risks & open items

- **Checkpoint SHA-256 placeholder.** The exact hash isn't known until the release is cut; the implementation plan needs a step for the maintainer to fill it in.
- **First-call latency.** A 305 MB download on the first `Zeus()` or `ZeusClusterer()` call is unavoidable. The progress bar and one-time message mitigate this; the alternative (lazy init in `fit`) is worse because it hides the cost behind the user's first compute call.
- **Tests need either network or a pre-cached checkpoint.** `test_api.py` is skipped otherwise. This is acceptable for a personal fork; a follow-up could bundle a tiny test checkpoint.
- **Context-dependence of embeddings is unusual for sklearn estimators.** The docstring and README will call this out, but users will still get tripped up by it. The decision to omit `predict(X_new)` rather than provide a footgun is deliberate.

## 8. Implementation order (for the writing-plans handoff)

A sensible incremental order for the plan:

1. Add `pyproject.toml` and `tests/` skeleton.
2. Move `inference_methods/` inside `zeus/`. Adjust imports.
3. Build `zeus/weights.py` + smoke test (skippable when offline).
4. Build `zeus/preprocessing.py` + unit tests.
5. Build `zeus/api.py` (Zeus, ZeusClusterer) + integration test.
6. Rewrite `zeus/__init__.py`.
7. Inline the small `model_utils.py` helpers (`SeqBN`, `bool_mask_to_att_mask`).
8. Delete the obsoleted files (`pretrain.py`, `evaluation.py`, `datasets.py`, `wandb_logging.py`, `visualization.py`, `configs.py`, `initialziation.py`, `utils.py`, `model/model_utils.py`, `synthetic_datasets/`, `inference_methods/` sibling, `requirements.txt`, `zeus.pt`).
9. Rewrite README (new top sections + annotated original).
10. Update CLAUDE.md.
11. Cut the GitHub release with the checkpoint asset, fill in the SHA-256 constant.
