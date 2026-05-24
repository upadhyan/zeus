# Paper-Faithful Preprocessing for Zeus

**Status:** Draft
**Date:** 2026-05-23
**Author:** brainstorming session with user
**Affects:** `zeus/preprocessing.py`, `zeus/api.py`, `tests/`, `CLAUDE.md`, `README.md`

## Context

Zeus is an sklearn-style wrapper around the transformer encoder from
[arxiv:2505.10704](https://arxiv.org/abs/2505.10704). The package is at
`v0.1.0`; not yet released on PyPI (see `RELEASING.md`). The transformer
weights are frozen; only the surrounding pipeline is malleable.

Users (including the maintainer, who needs this for benchmarking) report
that `ZeusClusterer(n_clusters=k).fit_predict(X)` produces poor ARI/NMI on
real-world datasets — including the OpenML benchmark datasets the paper
itself reports on. The transformer is not the issue: the same checkpoint
plugged into the paper's original `evaluation.py` pipeline reproduces the
published numbers.

Manual trace of `evaluation.py` → `utils.evaluate_model` →
`datasets.load_real_datasets` vs. our `zeus/preprocessing.py` →
`zeus/api.py` surfaced six concrete divergences. This spec captures the
fix.

## Problem

The current `prepare_inputs` differs from the paper's preprocessing in
ways that systematically degrade clustering quality on real data:

1. **No `StandardScaler` on numerical features.** Paper does
   `StandardScaler → MinMaxScaler(-1, 1)` per numerical column inside
   `load_real_datasets` (`datasets.py:342-346`). We only `MinMaxScale`,
   so outliers compress feature ranges and the transformer sees an
   input distribution unlike anything in its synthetic training set
   (which is also `StandardScaler → MinMaxScaler` — `datasets.py:194`,
   `datasets.py:255`).

2. **One-hot columns get rescaled to `{-1, +1}`.** Paper keeps OHE
   columns as `{0, 1}` and only re-MinMaxes when PCA has actually
   transformed the column space (`evaluate_model`, utils.py:138-146).
   Our blanket `MinMaxScaler(-1, 1)` after concat shifts every binary
   OHE column from `{0, 1}` to `{-1, +1}` — double the magnitude, mean
   shifted from 0.5 to 0. Training-time synthetic categoricals are
   `np.eye(...)` rows, so they're `{0, 1}` (`datasets.py:180`).

3. **`KMeans n_init=10` vs. paper's `n_init=100`.** Small but real
   ARI variance at the size of these datasets.

4. **No `most_frequent` imputation for categoricals.** Paper uses
   `SimpleImputer(strategy='most_frequent')` then `OneHotEncoder`. We
   use `pd.get_dummies(dummy_na=False)`, which silently emits all-zero
   rows for NaN inputs (not a valid one-hot).

5. **ndarray path is silently weaker than DataFrame path.** A user
   passing the same dataset as a numpy array gets a single mean-impute
   and no num/cat separation at all — categoricals treated as
   continuous, no `StandardScaler`, blanket `MinMaxScaler`.

6. **`MinMaxScaler` re-applied even when no PCA happened.** This is
   what mechanically causes (2). Paper re-MinMaxes only after PCA.

## Goals

1. `ZeusClusterer(n_clusters=k).fit_predict(X)` reproduces the paper's
   `evaluate_model` ARI/NMI on the OpenML benchmark datasets out of the
   box — no flags or non-default kwargs required.
2. The same path works for ndarray/Tensor input via an explicit
   `categorical_indices=` argument, with no surprise heuristics.
3. End-to-end OpenML regression test runs in `pytest tests/` and pins
   the post-fix numbers, so future preprocessing changes can't silently
   regress.

## Non-goals

- Fixing the paper's `GaussianMixture(n_init=int(n_init/10))` quirk
  (utils.py:207). We match effective behavior, not the implied intent.
- Heuristic categorical detection for arrays (int-dtype, low-cardinality
  rules). Explicit `categorical_indices=` only.
- Supporting `fit(X_train).transform(X_test)`. Embeddings are
  batch-context-dependent (see CLAUDE.md "Conventions / gotchas").
- Adding a separate paper-evaluation script. The sklearn API *is* the
  benchmarking surface.
- Changes to the transformer model itself or to the checkpoint.

## Design

### Section 1 — API surface

#### `Zeus`

```python
Zeus(
    *,
    device: DeviceLike = "auto",
    categorical_indices: Sequence[int] | None = None,   # NEW
    paper_preprocess: bool = True,                       # RENAMED from `preprocess`
    model_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
)
```

- `categorical_indices=` is **silently ignored** when a DataFrame is
  passed (DataFrame dtype wins; this matches the common benchmarking
  pattern where the same kwarg is set unconditionally across mixed
  input types).
- `paper_preprocess=False` means "trust the input as-is": must already
  be `(n, 30)` float, scaled to `[-1, 1]`. Same role as the current
  `preprocess=False`, same `passthrough_inputs` implementation.

#### `ZeusClusterer`

```python
ZeusClusterer(
    n_clusters: int,
    *,
    method: Literal["kmeans", "gmm", "simple_gmm"] = "kmeans",
    device: DeviceLike = "auto",
    categorical_indices: Sequence[int] | None = None,   # NEW (passed through to Zeus)
    paper_preprocess: bool = True,                       # RENAMED
    n_init: int | None = None,                           # NEW SEMANTICS: None → method default
    random_state: int | None = None,
    model_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
)
```

- `n_init=None` resolves at `fit` time via:

  ```python
  _PAPER_N_INIT = {"kmeans": 100, "gmm": 10, "simple_gmm": 10}
  ```

  These match the paper's *effective* values (KMeans gets 100 from
  `predict_clusters`; GaussianMixture gets `int(100/10)=10` from the
  same call site).
- Explicit `n_init=k` is honored verbatim — no `int(k/10)` divisor.
  This is a **deliberate departure** for explicit overrides (see
  Non-goals #1). A user porting a benchmark that passes
  `n_init=100` to GMM expecting paper-effective behavior would get 10x
  more init runs than the paper. The fix: when migrating an existing
  paper-style call site, drop the explicit `n_init=100` for GMM and
  let the default kick in.

### Section 2 — Preprocessing flow

Rewritten `prepare_inputs` in `zeus/preprocessing.py`:

```python
def prepare_inputs(
    X,
    *,
    categorical_indices: Sequence[int] | None = None,
    target_dim: int = INPUT_DIM,   # 30
) -> torch.Tensor:
    # 1. Resolve num / cat split.
    # NOTE: For DataFrames, we use dtype-based detection (_is_categorical_like
    # matches object / pd.CategoricalDtype / bool). The paper's load_real_datasets
    # uses OpenML's `categorical_indicator` metadata instead. These agree in
    # practice because `dataset.get_data(dataset_format='dataframe')` round-trips
    # OpenML categoricals as pandas `category` dtype. The OpenML regression test
    # in Section 5.4 is what validates this equivalence empirically.
    if isinstance(X, pd.DataFrame):
        cat_cols = [c for c in X.columns if _is_categorical_like(X[c].dtype)]
        num_cols = [c for c in X.columns if c not in cat_cols]
        frame = X
    else:
        arr = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2-D input, got shape {arr.shape}")
        n_features = arr.shape[1]
        cat_idx = sorted(set(categorical_indices or []))
        if cat_idx and (min(cat_idx) < 0 or max(cat_idx) >= n_features):
            raise ValueError(f"categorical_indices out of range for {n_features} features")
        num_idx = [i for i in range(n_features) if i not in cat_idx]
        frame = pd.DataFrame(arr)
        cat_cols, num_cols = cat_idx, num_idx

    # 2. Per-block preprocessing (paper-faithful)
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("std", StandardScaler()),
            ("mm",  MinMaxScaler(feature_range=(-1, 1))),
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            # Match paper exactly: don't set sparse_output; convert below.
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols))
    if not transformers:
        raise ValueError("Empty input; nothing to preprocess.")
    mat = ColumnTransformer(transformers).fit_transform(frame)
    if issparse(mat):
        mat = mat.toarray()
    mat = np.asarray(mat, dtype=np.float64)

    # 3. PCA or zero-pad to model input dim — matches evaluate_model exactly.
    d = mat.shape[1]
    if d > target_dim:
        mat = PCA(n_components=target_dim).fit_transform(mat)
        mat = MinMaxScaler(feature_range=(-1, 1)).fit_transform(mat)
    elif d < target_dim:
        mat = np.hstack([mat, np.zeros((mat.shape[0], target_dim - d), dtype=mat.dtype)])
    # else d == target_dim: leave alone. DO NOT re-MinMax here — that would
    # remap one-hot columns from {0, 1} to {-1, +1}, which the model was
    # never trained to see. This branch is load-bearing; see Section 2 issue (2).

    return torch.tensor(mat, dtype=torch.float32)
```

**Load-bearing details:**

- The "no MinMax when `d <= target_dim`" branch is what preserves OHE
  columns as `{0, 1}`. Without it, fix (1) and fix (4) don't fully land
  because the rescale at the end re-introduces problem (2).
- `StandardScaler` runs unconditionally on numerical columns. Paper
  does the same inside `load_real_datasets`. No-op on already-
  standardized inputs.
- ndarray input is routed through an internal DataFrame so we can reuse
  `ColumnTransformer` cleanly. The one `pd.DataFrame(arr)` allocation
  is negligible.
- `validate_categorical_indices` only checks range / dedupes. No
  dtype-based second-guessing — if the user marks a float column as
  categorical, `OneHotEncoder` encodes each distinct float; their call.
- `passthrough_inputs` is preserved as-is for `paper_preprocess=False`.

### Section 3 — Clustering defaults

`ZeusClusterer.fit` after the embedding step:

```python
emb = MinMaxScaler(feature_range=(-1, 1)).fit_transform(emb)  # unchanged
self.embedding_ = emb

_PAPER_N_INIT = {"kmeans": 100, "gmm": 10, "simple_gmm": 10}
n_init = self.n_init if self.n_init is not None else _PAPER_N_INIT[self.method]

if self.method == "kmeans":
    KMeans(n_clusters=self.n_clusters, n_init=n_init, random_state=self.random_state).fit(emb)
elif self.method == "gmm":
    GaussianMixture(n_components=self.n_clusters, n_init=n_init, random_state=self.random_state).fit(emb)
elif self.method == "simple_gmm":
    SimplifiedGMM(n_components=self.n_clusters, n_init=n_init, random_state=self.random_state).fit(emb)
```

- The embedding-side `MinMaxScaler((-1, 1))` is preserved — matches
  `predict_clusters` (utils.py:203) and is independent of input
  preprocessing.
- `random_state` defaults to `None` (matches paper). Tests pass a fixed
  seed for stability.

### Section 4 — Tests

#### Unit tests (`tests/test_preprocessing.py`, `tests/test_api.py`)

| Test | What it pins down |
|---|---|
| `test_preprocessing_matches_load_real_datasets` | Tiny synthetic DataFrame with known num + cat cols; run `prepare_inputs` and an inline copy of `load_real_datasets`' `ColumnTransformer` block; `assert_allclose` on the resulting matrix (modulo column ordering). Core regression. |
| `test_ndarray_with_categorical_indices_equals_df` | Same data as DataFrame and as `(ndarray, categorical_indices=[...])`; assert identical output matrix. |
| `test_one_hot_columns_preserved_as_zero_one` | Pure-categorical input fitting under 30 dims → assert OHE-encoded columns are exactly `{0, 1}`, NOT `{-1, 1}`. The load-bearing fix. |
| `test_pca_branch_rescales` | Input with > 30 cols → output is `(n, 30)` and within `[-1, 1]`. |
| `test_pad_branch_no_rescale` | Low-dim numeric input → trailing cols are zeros, leading scaled cols match pre-pad values exactly (no second MinMax). |
| `test_n_init_defaults_per_method` | Monkeypatch `KMeans`/`GaussianMixture`/`SimplifiedGMM` constructors; assert default `n_init` is 100/10/10 for each method; explicit override honored verbatim. |
| `test_passthrough_preprocess_false` | `paper_preprocess=False` rejects DataFrames; accepts `(n, 30)` float arrays unchanged. |
| `test_categorical_indices_ignored_for_dataframe` | DataFrame input + `categorical_indices=[0]` → arg silently ignored, DataFrame dtypes win. |
| `test_categorical_indices_out_of_range` | ndarray + out-of-range indices → `ValueError`. |
| `test_empty_input_raises` | Empty DataFrame → `ValueError("Empty input...")`. |

#### OpenML end-to-end regression (`tests/test_openml_regression.py`)

Four datasets from `openml_ids` (utils.py:242), spanning all paths:

| ID  | Name                     | Path exercised |
|---|---|---|
| 61   | iris                     | small all-numerical → zero-pad |
| 1462 | banknote-authentication  | small all-numerical → zero-pad, different scale |
| 53   | heart-statlog            | mixed num+cat → OHE preservation |
| 1510 | wdbc                     | ~30 features → boundary case (provisional) |

The 1510 / boundary-case slot is **provisional** — Phase 0 measurement
verifies each ID's effective feature count after OHE expansion. If 1510
lands at exactly 30 (no PCA, no padding) or otherwise doesn't exercise
the PCA-or-pad branch we want, swap with `40496` (`mfeat-zernike`, 47
features) or `4153` (`Smartphone-Based-Recognition-of-Human-Activities`,
561 features → definitely PCA).

Test mechanics:

- Per dataset:
  ```python
  ds = openml.datasets.get_dataset(id, download_data=True)
  X, y, _, _ = ds.get_data(
      dataset_format="dataframe",
      target=ds.default_target_attribute,
  )
  y = LabelEncoder().fit_transform(y)
  n_classes = len(np.unique(y))
  labels = ZeusClusterer(
      n_clusters=n_classes, method="kmeans", random_state=42
  ).fit_predict(X)
  ari = adjusted_rand_score(y, labels)
  nmi = normalized_mutual_info_score(y, labels)
  ```
  This matches `load_real_datasets`' loading semantics (datasets.py:329-336)
  except we hand the raw DataFrame to `ZeusClusterer` and let the new
  `prepare_inputs` do the per-block preprocessing.
- Assert `ari >= post_fix_value - 0.05` per dataset. With
  `random_state=42` and `n_init=100`, KMeans is deterministic enough
  that observed run-to-run jitter is ~0.01-0.02; the 0.05 buffer is
  comfortably above the noise floor and protects against sklearn
  minor-version output drift.
- Compute NMI for logging but do not assert on it (correlated with ARI;
  pinning both is noise).
- Test only `method='kmeans'`. GMM / SimpleGMM regressions caught by
  unit tests.
- `openml.config.cache_directory = <repo>/.openml_cache/` (gitignored).
  First run downloads; subsequent runs are local.
- Mark with `@pytest.mark.openml`. Register the marker in
  `pyproject.toml` so pytest doesn't warn. Document
  `pytest tests/ -m "not openml"` to skip in environments without
  internet.

Estimated runtime: 20-40s added after first-run cache warm-up.

#### Baseline file format (`tests/openml_baselines.md`)

A single markdown table appended in Phase 0 and updated in Phase 4:

| openml_id | name | n_features (raw) | n_features (post-OHE) | branch | before_ari | before_nmi | after_ari | after_nmi |
|---|---|---|---|---|---|---|---|---|
| 61 | iris | 4 | 4 | pad | … | … | … | … |
| … | … | … | … | … | … | … | … | … |

Plus a footer recording: `random_state=42`, sklearn version,
`Zeus.pt` SHA-256 (pin from `RELEASING.md`), date measured.

## Implementation phases

Order is chosen so that empirical baselines exist before code changes,
and so the OpenML test thresholds are set from real post-fix numbers
rather than hand-guessed.

| Phase | Action |
|---|---|
| **0** | Tiny script `scripts/measure_openml.py`: for each of the 4 IDs, load via OpenML → **record post-OHE feature count and decide which branch (PCA / pad / exact) it exercises** → run **current** `ZeusClusterer(method='kmeans')` → record ARI + NMI → write to `tests/openml_baselines.md` ("before-fix" columns + the branch column). If ID 1510's post-OHE feature count equals or exceeds `INPUT_DIM=30` without triggering PCA, swap with `40496` or `4153` per Section 4 and re-record. Commit. |
| **1** | Rewrite `zeus/preprocessing.py`: new `prepare_inputs`; keep `_is_categorical_like` and `passthrough_inputs`; delete `_df_to_numeric_matrix`, `_mean_impute_array`, `_adjust_dim`. |
| **2** | Update `zeus/api.py`: rename `preprocess` → `paper_preprocess`, add `categorical_indices=`, thread through both classes, add `n_init=None` resolution logic in `ZeusClusterer.fit`. |
| **3** | Update existing unit tests; delete tests that pinned old behavior; add the 10 unit tests above. Run `pytest tests/ -v -m "not openml"` to green. |
| **4** | Re-run `scripts/measure_openml.py` with new code; append "after-fix" column to `tests/openml_baselines.md`. Commit. |
| **5** | Write `tests/test_openml_regression.py` with thresholds = `post_fix - 0.05`. Run `pytest tests/ -v` (including OpenML); all green. |
| **6** | Update `CLAUDE.md` "Conventions / gotchas" and "Architecture" sections to reflect the new preprocessing flow; update `README.md` examples to show `categorical_indices=` for ndarray inputs. |

Phases 0, 4, and 5 are not in the "writing code" sense — they're
measurement and threshold-setting. Phases 1-3 are the implementation;
phase 6 is doc cleanup.

## Migration

- Public API rename: `Zeus(preprocess=...)` →
  `Zeus(paper_preprocess=...)`. No deprecation shim; package is pre-PyPI.
  Mention in commit message and add a v0.2.0 entry to `RELEASING.md`
  (the version-bump checklist lives at the bottom of that file).
- Behavior change: `ZeusClusterer(n_clusters=k)` now uses
  `n_init=100` for kmeans (was 10). Faithful to paper. Slightly slower
  per call.
- New optional kwarg for ndarray users with categorical features:
  `categorical_indices=`. Without it, all columns are treated as
  numerical — matches old behavior for ndarray inputs, so this is not
  a silent regression for existing array users.
- **Silent behavior change for ndarray users with NaN inputs.** The
  current `_mean_impute_array` (`zeus/preprocessing.py`) emits
  `warnings.warn(...)` when it sees NaN; the new flow uses sklearn's
  `SimpleImputer`, which is silent (per sklearn convention). Mention
  in the commit message; not loud enough to need a release-note
  bullet.

## Open questions

None — all design decisions resolved during the brainstorming session:

- `pca_dim` separate from `dim`: we use the single `INPUT_DIM=30`
  value (paper code allows them to differ, but the model encoder is
  fixed at 30, so a separate `pca_dim` parameter would be dead).
- DataFrame + `categorical_indices=`: silently ignore (dtype wins).
- GMM `n_init` quirk: match effective behavior (10), don't propagate
  the `int(.../10)` divisor.
- `random_state` default: `None`, matching paper.
- OpenML test in default suite (yes, with `@pytest.mark.openml` for
  opt-out).
