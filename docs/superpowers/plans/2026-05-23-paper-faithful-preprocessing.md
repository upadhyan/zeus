# Paper-Faithful Preprocessing — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ZeusClusterer(n_clusters=k).fit_predict(X)` reproduce the paper's `evaluate_model` ARI/NMI on OpenML benchmark datasets out of the box, by aligning our preprocessing/clustering defaults with `load_real_datasets` + `evaluate_model` in `utils.py` / `datasets.py`.

**Architecture:** Rewrite `zeus/preprocessing.py:prepare_inputs` to do per-block num/cat preprocessing via sklearn `ColumnTransformer` (paper-faithful: StandardScaler-then-MinMax for numerics, most-frequent-impute-then-OneHot for cats, no blanket re-MinMax after concat). Update `zeus/api.py` to add `categorical_indices=` (ndarray entry point), rename `preprocess` → `paper_preprocess`, and resolve `n_init` per-method (kmeans 100, gmm 10, simple_gmm 10) when not explicitly set. Empirical baselines measured before/after with a small script feeding 4 OpenML datasets, post-fix numbers pinned by a `@pytest.mark.openml` regression test.

**Tech Stack:** Python 3.10+, sklearn (ColumnTransformer, Pipeline, SimpleImputer, StandardScaler, MinMaxScaler, OneHotEncoder, PCA, KMeans, GaussianMixture), pandas, numpy, torch, pytest, openml (test-only).

**Spec:** `docs/superpowers/specs/2026-05-23-paper-faithful-preprocessing-design.md`

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `scripts/measure_openml.py` | **create** | Run 4 OpenML IDs through `ZeusClusterer`, print ARI/NMI per dataset. Used in Phase 0 (before-fix) and Phase 4 (after-fix). Stateless; no persistence. |
| `tests/openml_baselines.md` | **create** | Markdown table of measured ARI/NMI per dataset, before- and after-fix. Source of truth for the regression-test thresholds. |
| `zeus/preprocessing.py` | **rewrite** | New `prepare_inputs` doing per-block num/cat preprocessing + PCA/pad to 30. Keep `_is_categorical_like` and `passthrough_inputs`; delete `_df_to_numeric_matrix`, `_mean_impute_array`, `_adjust_dim`. |
| `zeus/api.py` | **modify** | Rename `preprocess` → `paper_preprocess`; add `categorical_indices=` (both `Zeus` and `ZeusClusterer`); add `n_init=None` resolution in `ZeusClusterer.fit`. |
| `tests/test_preprocessing.py` | **rewrite** | Replace tests pinning old behavior (e.g. `test_nans_in_ndarray_emit_warning_and_are_imputed`) with the new test matrix (Section 4 of spec). |
| `tests/test_api.py` | **modify** | Rename `preprocess=False` call sites to `paper_preprocess=False`. Add tests for `categorical_indices=` passthrough and `n_init=None` resolution. |
| `tests/test_openml_regression.py` | **create** | The 4-dataset end-to-end test. `@pytest.mark.openml`. Thresholds = post-fix - 0.05. |
| `pyproject.toml` | **modify** | Register `openml` pytest marker. |
| `.gitignore` | **modify** | Add `.openml_cache/`. |
| `CLAUDE.md` | **modify** | Update "Architecture" preprocessing section + "Conventions / gotchas" to reflect new defaults. |
| `README.md` | **modify** | Update `Zeus` / `ZeusClusterer` API ref blocks; show `categorical_indices=` for ndarray inputs. |
| `RELEASING.md` | **modify** | Add v0.2.0 entry per its existing version-bump format. |

Files NOT touched: `zeus/model/*`, `zeus/inference_methods/*`, `zeus/weights.py`, `zeus/_config.py`. Transformer weights and shape contracts are unchanged.

---

## Chunk 1: Phase 0 — Baseline measurement

**Goal:** Run the 4 OpenML datasets through the CURRENT (broken) pipeline, record ARI/NMI as the "before" column of `tests/openml_baselines.md`. This locks in the empirical evidence that the fix is doing something.

**Files this chunk touches:**
- Create: `scripts/measure_openml.py`
- Create: `tests/openml_baselines.md`

### Task 1.1: Verify openml dependency is available

- [ ] **Step 1: Check if `openml` is installed**

Run: `python -c "import openml; print(openml.__version__)"`
Expected: prints a version number. If `ModuleNotFoundError`, install with `pip install openml`.

- [ ] **Step 2: If not installed, install in the dev env**

Run: `pip install openml`
Expected: install completes. (Do NOT add `openml` to `pyproject.toml` dependencies — it's test/script-only. Phase 5 will add it to the `test` optional-dep group.)

### Task 1.2: Create the measurement script

**Files:** Create: `scripts/measure_openml.py`

- [ ] **Step 1: Create the `scripts/` directory if it doesn't exist**

Run: `ls /home/upadhyan/zeus/scripts/ 2>/dev/null || mkdir -p /home/upadhyan/zeus/scripts/`
Expected: directory exists or is created.

- [ ] **Step 2: Write `scripts/measure_openml.py`**

```python
"""Measure ZeusClusterer ARI/NMI on a small set of OpenML datasets.

Used twice during the paper-faithful-preprocessing rewrite:
  1. Phase 0 (this script's first run): record "before-fix" numbers.
  2. Phase 4 (after the api/preprocessing rewrite): record "after-fix" numbers.

Both runs append a row per dataset to `tests/openml_baselines.md` so the
regression-test thresholds (Phase 5) can be set from real measurements.

Usage:
    python scripts/measure_openml.py
    # prints a markdown table to stdout; copy/paste into tests/openml_baselines.md.

Output columns: id, name, n_features_raw, n_features_post_ohe, branch
(pad|exact|pca), ari, nmi.
"""
from __future__ import annotations
import sys
import time

import numpy as np
import openml
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler,
)

from zeus import ZeusClusterer
from zeus._config import INPUT_DIM

# Provisional 4 datasets per spec Section 4. 1510 may be swapped after
# inspecting its post-OHE feature count (see Task 1.4).
OPENML_IDS = [
    (61,   "iris"),
    (1462, "banknote-authentication"),
    (53,   "heart-statlog"),
    (1510, "wdbc"),
]


def post_ohe_feature_count(X: pd.DataFrame, categorical_indicator: list[bool]) -> int:
    """Replicates load_real_datasets' ColumnTransformer block to count final features."""
    num_cols = [c for c, is_cat in zip(X.columns, categorical_indicator) if not is_cat]
    cat_cols = [c for c, is_cat in zip(X.columns, categorical_indicator) if is_cat]

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("std", StandardScaler()),
            ("mm",  MinMaxScaler((-1, 1))),
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols))
    out = ColumnTransformer(transformers).fit_transform(X)
    return out.shape[1]


def measure_one(openml_id: int, name: str) -> dict:
    ds = openml.datasets.get_dataset(openml_id, download_data=True)
    X, y, categorical_indicator, _ = ds.get_data(
        dataset_format="dataframe",
        target=ds.default_target_attribute,
    )
    y = LabelEncoder().fit_transform(y)
    n_classes = len(np.unique(y))
    n_features_raw = X.shape[1]
    n_features_post_ohe = post_ohe_feature_count(X, categorical_indicator)

    if n_features_post_ohe > INPUT_DIM:
        branch = "pca"
    elif n_features_post_ohe < INPUT_DIM:
        branch = "pad"
    else:
        branch = "exact"

    t0 = time.time()
    labels = ZeusClusterer(
        n_clusters=n_classes,
        method="kmeans",
        random_state=42,
    ).fit_predict(X)
    elapsed = time.time() - t0

    return {
        "id": openml_id,
        "name": name,
        "n_classes": n_classes,
        "n_features_raw": n_features_raw,
        "n_features_post_ohe": n_features_post_ohe,
        "branch": branch,
        "ari": adjusted_rand_score(y, labels),
        "nmi": normalized_mutual_info_score(y, labels),
        "elapsed_s": elapsed,
    }


def main() -> int:
    rows = []
    for openml_id, name in OPENML_IDS:
        try:
            row = measure_one(openml_id, name)
        except Exception as e:
            print(f"[error] id={openml_id} ({name}): {e}", file=sys.stderr)
            continue
        rows.append(row)
        print(
            f"id={row['id']:>5}  name={row['name']:<32}  "
            f"raw={row['n_features_raw']:>3}  ohe={row['n_features_post_ohe']:>3}  "
            f"branch={row['branch']:<5}  n_classes={row['n_classes']:>2}  "
            f"ari={row['ari']:.4f}  nmi={row['nmi']:.4f}  ({row['elapsed_s']:.1f}s)"
        )

    # Print markdown table
    print()
    print("| id | name | n_features (raw) | n_features (post-OHE) | branch | ari | nmi |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['id']} | {r['name']} | {r['n_features_raw']} | "
            f"{r['n_features_post_ohe']} | {r['branch']} | "
            f"{r['ari']:.4f} | {r['nmi']:.4f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### Task 1.3: Run the script and capture before-fix numbers

- [ ] **Step 1: Run from the repo root**

Run: `python scripts/measure_openml.py`
Expected: 4 lines of progress output, then a markdown table. First run downloads ~MBs (cached in `~/.cache/openml/`); subsequent runs are local.

Expected duration: 30-90 seconds depending on hardware. KMeans `n_init=10` (current default) on 512-dim embeddings is fast.

- [ ] **Step 2: Inspect the output for the `branch` column**

For each row, note which branch (`pad` / `exact` / `pca`) it lands in.
- If ID 1510 lands in `pca` (post-OHE > 30): keep it.
- If ID 1510 lands in `exact` (post-OHE == 30): keep it too — boundary case is useful.
- If ID 1510 lands in `pad` (post-OHE < 30): swap with `40496` (mfeat-zernike, 47 features → pca) or `4153` (561 features → pca). Edit `OPENML_IDS` in `scripts/measure_openml.py` and re-run.

This ensures all three branches are exercised by the 4-dataset matrix.

### Task 1.4: Write `tests/openml_baselines.md`

**Files:** Create: `tests/openml_baselines.md`

- [ ] **Step 1: Write the file with the script output**

Copy the markdown table from the script's stdout. Add columns for after-fix (initially empty — filled in Phase 4) and a footer. Template:

```markdown
# OpenML clustering baselines

Measured by `scripts/measure_openml.py` on the 4 datasets pinned by
the paper-faithful-preprocessing regression test
(`tests/test_openml_regression.py`).

| id | name | n_features (raw) | n_features (post-OHE) | branch | before_ari | before_nmi | after_ari | after_nmi |
|---|---|---|---|---|---|---|---|---|
| 61   | iris                    | 4  | 4  | pad   | <fill>   | <fill>   |   |   |
| 1462 | banknote-authentication | 4  | 4  | pad   | <fill>   | <fill>   |   |   |
| 53   | heart-statlog           | 13 | <fill> | <fill> | <fill> | <fill> |   |   |
| 1510 | wdbc                    | <fill> | <fill> | <fill> | <fill> | <fill> |   |   |

## Run conditions

- `ZeusClusterer(n_clusters=n_classes, method='kmeans', random_state=42)`
- sklearn version: <fill from `python -c "import sklearn; print(sklearn.__version__)"`>
- `zeus.pt` SHA-256: <fill from RELEASING.md "v0.1.0 checkpoint SHA-256" entry>
- Date measured (before-fix): <YYYY-MM-DD>
- Date measured (after-fix): <to be filled in Phase 4>
```

Fill the `<fill>` placeholders with the actual numbers from Step 1. Leave `after_*` columns empty.

### Task 1.5: Commit Phase 0

- [ ] **Step 1: Stage the new files**

Run: `git -C /home/upadhyan/zeus add scripts/measure_openml.py tests/openml_baselines.md`

- [ ] **Step 2: Commit**

Run:
```bash
git -C /home/upadhyan/zeus commit -m "test: add openml baseline measurement script + before-fix numbers

Phase 0 of the paper-faithful preprocessing rewrite. scripts/measure_openml.py
runs ZeusClusterer on 4 OpenML datasets and prints ARI/NMI. The before-fix
numbers committed in tests/openml_baselines.md are the baseline the regression
test in Phase 5 will compare against (post-fix - 0.05 buffer)."
```

---

## Chunk 2: Phase 1 — Rewrite `zeus/preprocessing.py`

**Goal:** Replace `prepare_inputs` with a paper-faithful per-block pipeline. Strict TDD: write each failing test first, then implement to green.

**Files this chunk touches:**
- Modify: `zeus/preprocessing.py` (rewrite `prepare_inputs`; keep `_is_categorical_like` and `passthrough_inputs`; delete dead helpers)
- Rewrite: `tests/test_preprocessing.py` (new test matrix per spec Section 4)

### Task 2.1: Delete the obsolete test that pins old warning behavior

- [ ] **Step 1: Open `tests/test_preprocessing.py` and delete the obsolete test**

Delete the entire function `test_nans_in_ndarray_emit_warning_and_are_imputed` (the new pipeline uses `SimpleImputer`, which is silent; per Migration in spec).

- [ ] **Step 2: Run the test file to confirm it still loads**

Run: `pytest tests/test_preprocessing.py -v --collect-only`
Expected: prints the remaining tests, no collection error.

### Task 2.2: Write failing test — DataFrame matches paper's ColumnTransformer block

**Files:** Modify: `tests/test_preprocessing.py`

- [ ] **Step 1: Add the new test at the end of `tests/test_preprocessing.py`**

```python
def test_preprocessing_matches_load_real_datasets():
    """Paper-faithful per-block num/cat preprocessing.

    Builds a tiny DataFrame and an inline copy of the ColumnTransformer
    block from datasets.py:340-352, asserts our prepare_inputs produces
    the same matrix (up to the PCA/pad branch and column order).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x1": rng.normal(size=30),
        "x2": rng.normal(size=30) * 100,         # different scale: tests StandardScaler
        "color": rng.choice(["red", "blue"], size=30),
        "flag": rng.choice([True, False], size=30),
    })

    # Inline paper pipeline (datasets.py:340-352)
    paper = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("std", StandardScaler()),
            ("mm",  MinMaxScaler((-1, 1))),
        ]), ["x1", "x2"]),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), ["color", "flag"]),
    ])
    expected = paper.fit_transform(df)
    if hasattr(expected, "toarray"):
        expected = expected.toarray()

    out = prepare_inputs(df).numpy()
    # Output is (n, 30) zero-padded; first `expected.shape[1]` columns must match.
    np.testing.assert_allclose(out[:, :expected.shape[1]], expected, atol=1e-6)
    # Padding columns are zero.
    assert np.allclose(out[:, expected.shape[1]:], 0.0)
```

- [ ] **Step 2: Run the new test — expect failure**

Run: `pytest tests/test_preprocessing.py::test_preprocessing_matches_load_real_datasets -v`
Expected: FAIL — current `prepare_inputs` does not match (missing StandardScaler, blanket MinMax remaps OHE).

### Task 2.3: Write failing test — OHE columns preserved as {0, 1}

- [ ] **Step 1: Add the test**

```python
def test_one_hot_columns_preserved_as_zero_one():
    """The load-bearing fix: when no PCA happens, OHE columns must stay {0, 1},
    NOT get remapped to {-1, +1} by a blanket MinMax."""
    df = pd.DataFrame({
        "color": ["red", "blue", "red", "green", "blue"],
        "flag":  [True, False, True, False, True],
    })
    out = prepare_inputs(df).numpy()
    # Pure categorical: 3 (color) + 2 (flag) = 5 OHE cols, then zero-pad to 30.
    # The first 5 cols must be exactly in {0, 1}, never {-1, +1}.
    encoded = out[:, :5]
    unique = np.unique(encoded)
    assert set(unique.tolist()).issubset({0.0, 1.0}), (
        f"OHE cols should be in {{0, 1}} but got {unique.tolist()}"
    )
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/test_preprocessing.py::test_one_hot_columns_preserved_as_zero_one -v`
Expected: FAIL — current code's blanket MinMax remaps {0, 1} → {-1, +1}.

### Task 2.4: Write failing test — ndarray + `categorical_indices` ≡ DataFrame

- [ ] **Step 1: Add the test**

```python
def test_ndarray_with_categorical_indices_equals_df():
    """Passing the same data as a DataFrame or as (ndarray, categorical_indices=...)
    must produce the same output matrix."""
    rng = np.random.default_rng(1)
    n = 20
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat0 = rng.integers(0, 3, size=n)   # 3 categories
    cat1 = rng.integers(0, 2, size=n)   # 2 categories

    # As ndarray: cols 2, 3 are categorical
    arr = np.column_stack([x1, x2, cat0, cat1]).astype(np.float64)
    out_arr = prepare_inputs(arr, categorical_indices=[2, 3]).numpy()

    # As DataFrame: same data, cats as object dtype
    df = pd.DataFrame({
        "x1": x1, "x2": x2,
        "c0": pd.Categorical(cat0),
        "c1": pd.Categorical(cat1),
    })
    out_df = prepare_inputs(df).numpy()

    np.testing.assert_allclose(out_arr, out_df, atol=1e-6)
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/test_preprocessing.py::test_ndarray_with_categorical_indices_equals_df -v`
Expected: FAIL — `categorical_indices` kwarg doesn't exist yet (TypeError).

### Task 2.5: Write failing test — pad branch no rescale

- [ ] **Step 1: Add the test**

```python
def test_pad_branch_no_rescale():
    """When d <= target_dim, the matrix is zero-padded but NOT re-MinMaxed.
    A column whose values are already in a specific range should pass through."""
    # All-numerical, 4 cols, pre-scaled by hand: x1 already in [-1, 1]
    x1 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    df = pd.DataFrame({"x1": x1, "x2": x1, "x3": x1, "x4": x1})
    out = prepare_inputs(df).numpy()
    assert out.shape == (5, c.INPUT_DIM)
    # Numerical cols go through StandardScaler→MinMax(-1, 1). For symmetric input
    # around 0, this should round-trip to the same range.
    assert out[:, :4].min() >= -1.0 - 1e-6
    assert out[:, :4].max() <= 1.0 + 1e-6
    # Padding cols are exactly zero (NOT MinMaxed, which would be -1).
    assert np.allclose(out[:, 4:], 0.0)
```

- [ ] **Step 2: Run — expect failure (pad cols would be -1, not 0)**

Run: `pytest tests/test_preprocessing.py::test_pad_branch_no_rescale -v`
Expected: FAIL — current code MinMaxes after pad, mapping the all-zero padding to -1.

### Task 2.6: Write failing test — PCA branch

- [ ] **Step 1: Add the test**

```python
def test_pca_branch_rescales():
    """When d > target_dim, PCA reduces to 30 dims then MinMax(-1, 1)."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 64)).astype(np.float32)
    out = prepare_inputs(X).numpy()
    assert out.shape == (50, c.INPUT_DIM)
    assert out.min() >= -1.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6
```

- [ ] **Step 2: Run — should already pass with current code (PCA path is unchanged)**

Run: `pytest tests/test_preprocessing.py::test_pca_branch_rescales -v`
Expected: PASS (but kept as a regression guard).

### Task 2.7: Write failing test — empty input + out-of-range indices + DataFrame ignores `categorical_indices`

- [ ] **Step 1: Add the three small tests**

```python
def test_categorical_indices_out_of_range():
    arr = np.zeros((5, 3))
    with pytest.raises(ValueError, match="out of range"):
        prepare_inputs(arr, categorical_indices=[3])
    with pytest.raises(ValueError, match="out of range"):
        prepare_inputs(arr, categorical_indices=[-1])


def test_empty_dataframe_raises():
    with pytest.raises(ValueError, match="Empty"):
        prepare_inputs(pd.DataFrame())


def test_categorical_indices_ignored_for_dataframe():
    """DataFrame dtypes are the source of truth; categorical_indices is silently ignored."""
    df = pd.DataFrame({
        "x1": [0.1, 0.2, 0.3],
        "x2": [1.0, 2.0, 3.0],
    })
    a = prepare_inputs(df).numpy()
    b = prepare_inputs(df, categorical_indices=[0]).numpy()
    np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2: Run all three — expect mostly TypeError (kwarg missing)**

Run: `pytest tests/test_preprocessing.py -v -k "categorical_indices or empty_dataframe"`
Expected: FAIL — kwarg not implemented.

### Task 2.8: Rewrite `zeus/preprocessing.py`

**Files:** Modify: `zeus/preprocessing.py`

- [ ] **Step 1: Replace the file contents with the paper-faithful implementation**

```python
"""Input preprocessing for the Zeus encoder.

Converts ndarray / DataFrame / Tensor inputs into a (n, INPUT_DIM) float32
tensor suitable for `Zeus.transform`, matching the per-block num/cat
preprocessing the paper applies in `load_real_datasets` + `evaluate_model`
(see spec docs/superpowers/specs/2026-05-23-paper-faithful-preprocessing-design.md).
"""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from zeus._config import INPUT_DIM

ArrayLike = Union[np.ndarray, pd.DataFrame, torch.Tensor]


def _is_categorical_like(dt) -> bool:
    """True if `dt` is object, pandas Categorical, or boolean."""
    return dt == object or isinstance(dt, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(dt)


def prepare_inputs(
    X: ArrayLike,
    *,
    categorical_indices: Sequence[int] | None = None,
    target_dim: int = INPUT_DIM,
) -> torch.Tensor:
    """Convert `X` to a `(n, target_dim)` float32 tensor (paper-faithful pipeline).

    Per-block: numerical cols get SimpleImputer(mean) → StandardScaler →
    MinMaxScaler(-1, 1); categorical cols get SimpleImputer(most_frequent) →
    OneHotEncoder(handle_unknown='ignore'). Then PCA-to-target if too wide,
    or zero-pad if too narrow. The `d == target_dim` branch is a deliberate
    no-rescale to preserve OHE columns as {0, 1}.

    For DataFrame input, num/cat split is by dtype. For ndarray/Tensor input,
    pass `categorical_indices=[...]` (column indices) or omit it to treat
    every column as numerical. `categorical_indices` is silently ignored
    when a DataFrame is passed (dtype wins).
    """
    # 1. Resolve num / cat split.
    # NOTE: DataFrame path uses dtype-based detection (_is_categorical_like
    # matches object / pd.CategoricalDtype / bool). The paper's
    # load_real_datasets uses OpenML's `categorical_indicator` metadata; these
    # agree in practice because `dataset.get_data(dataset_format='dataframe')`
    # round-trips OpenML categoricals as pandas `category` dtype. The OpenML
    # regression test (tests/test_openml_regression.py) validates this
    # equivalence empirically.
    if isinstance(X, pd.DataFrame):
        cat_cols = [c for c in X.columns if _is_categorical_like(X[c].dtype)]
        num_cols = [c for c in X.columns if c not in cat_cols]
        frame = X
    else:
        if isinstance(X, torch.Tensor):
            arr = X.detach().cpu().numpy()
        elif isinstance(X, np.ndarray):
            arr = X
        else:
            raise TypeError(f"Unsupported input type: {type(X).__name__}")
        if arr.ndim != 2:
            raise ValueError(f"Expected 2-D input, got shape {arr.shape}")
        n_features = arr.shape[1]
        cat_idx = sorted(set(categorical_indices or []))
        if cat_idx and (min(cat_idx) < 0 or max(cat_idx) >= n_features):
            raise ValueError(
                f"categorical_indices out of range for {n_features} features: {cat_idx}"
            )
        num_idx = [i for i in range(n_features) if i not in cat_idx]
        frame = pd.DataFrame(arr.astype(np.float64))
        cat_cols, num_cols = cat_idx, num_idx

    # 2. Per-block preprocessing (paper-faithful, datasets.py:340-352).
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
            # Match paper exactly: no sparse_output kwarg; convert below.
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols))
    if not transformers:
        raise ValueError("Empty input; nothing to preprocess.")
    mat = ColumnTransformer(transformers).fit_transform(frame)
    if issparse(mat):
        mat = mat.toarray()
    mat = np.asarray(mat, dtype=np.float64)

    # 3. PCA or zero-pad to model input dim — matches evaluate_model exactly
    # (utils.py:138-146).
    d = mat.shape[1]
    if d > target_dim:
        mat = PCA(n_components=target_dim).fit_transform(mat)
        mat = MinMaxScaler(feature_range=(-1, 1)).fit_transform(mat)
    elif d < target_dim:
        mat = np.hstack([mat, np.zeros((mat.shape[0], target_dim - d), dtype=mat.dtype)])
    # else d == target_dim: leave alone. DO NOT re-MinMax here — that would
    # remap one-hot columns from {0, 1} to {-1, +1}, which the model was
    # never trained to see. This branch is load-bearing.

    return torch.tensor(mat, dtype=torch.float32)


def passthrough_inputs(X: ArrayLike, target_dim: int = INPUT_DIM) -> torch.Tensor:
    """Used when `paper_preprocess=False`; rejects anything but a numeric (n, target_dim) array."""
    if isinstance(X, pd.DataFrame):
        raise ValueError(
            "paper_preprocess=False requires a numeric ndarray/Tensor; got DataFrame."
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

- [ ] **Step 2: Run all preprocessing tests**

Run: `pytest tests/test_preprocessing.py -v`
Expected: every test in the file PASSES (the new ones now have an implementation; the carried-over tests `test_numpy_exact_dim_passes_through_with_scaling`, `test_numpy_small_dim_is_zero_padded_then_scaled`, `test_numpy_large_dim_is_pca_then_scaled`, `test_tensor_input_accepted`, `test_dataframe_with_categoricals_is_one_hot_encoded`, `test_nans_in_numeric_dataframe_are_mean_imputed`, `test_preprocess_false_path_requires_exact_dim` all still match the new contract).

**If `test_numpy_small_dim_is_zero_padded_then_scaled` fails**: it was written against the old behavior where the small-dim ndarray went through `MinMaxScaler` after padding (so padding became -1, not 0). The test's actual assertion is just "didn't lose rows" + `torch.isfinite`, which the new code still satisfies. If it does fail, update its docstring to match new behavior — don't change the assertion.

- [ ] **Step 3: Run the wider test suite to make sure nothing else regressed**

Run: `pytest tests/ -v --ignore=tests/test_openml_regression.py`
Expected: all passes except for tests in `tests/test_api.py` that use `preprocess=False` (those are addressed in Chunk 3) — note any failures and confirm they're confined to `preprocess=` kwarg renames.

### Task 2.9: Commit Chunk 2

- [ ] **Step 1: Stage**

Run: `git -C /home/upadhyan/zeus add zeus/preprocessing.py tests/test_preprocessing.py`

- [ ] **Step 2: Commit**

```bash
git -C /home/upadhyan/zeus commit -m "refactor: paper-faithful prepare_inputs

Rewrites zeus/preprocessing.py to do per-block num/cat preprocessing
matching load_real_datasets + evaluate_model exactly: SimpleImputer →
StandardScaler → MinMaxScaler(-1, 1) on numerics, SimpleImputer(most_frequent)
→ OneHotEncoder on categoricals, PCA-or-zero-pad to 30. The d==target_dim
no-rescale branch is load-bearing: it preserves OHE columns as {0, 1}
instead of remapping to {-1, +1}, which is the root of the broken ARI/NMI
on real datasets.

Adds categorical_indices= kwarg for ndarray/Tensor inputs (silently
ignored for DataFrames, where dtype wins). Drops the now-unused
_df_to_numeric_matrix, _mean_impute_array, _adjust_dim helpers.

Tests: full new matrix per spec Section 4."
```

---

## Chunk 3: Phase 2 — Update `zeus/api.py`

**Goal:** Rename `preprocess` → `paper_preprocess`; add `categorical_indices=` (threaded through both classes); add per-method `n_init` default resolution.

**Files this chunk touches:**
- Modify: `zeus/api.py`
- Modify: `tests/test_api.py` (kwarg rename + new tests for categorical_indices + n_init)

### Task 3.1: Update existing `tests/test_api.py` for the kwarg rename

**Files:** Modify: `tests/test_api.py`

- [ ] **Step 1: Find the call site that uses `preprocess=False`**

Run: `grep -n "preprocess=" /home/upadhyan/zeus/tests/test_api.py`
Expected: at least one line: `test_zeus_preprocess_false_requires_exact_dim` uses `Zeus(..., preprocess=False)`.

- [ ] **Step 2: Edit the call site to use `paper_preprocess=False`**

Use Edit tool to replace `preprocess=False` with `paper_preprocess=False` in `tests/test_api.py`. Also rename the test function for clarity:
- `test_zeus_preprocess_false_requires_exact_dim` → `test_zeus_paper_preprocess_false_requires_exact_dim`

- [ ] **Step 3: Run the renamed test — expect failure (kwarg not yet renamed in api.py)**

Run: `pytest tests/test_api.py::test_zeus_paper_preprocess_false_requires_exact_dim -v`
Expected: FAIL — `Zeus(...)` got unexpected kwarg `paper_preprocess`.

### Task 3.2: Write failing test — `categorical_indices` passthrough on `Zeus`

- [ ] **Step 1: Add the test**

```python
def test_zeus_passes_categorical_indices_to_prepare_inputs(fake_checkpoint, monkeypatch):
    """Zeus(categorical_indices=[...]).transform(arr) should route the indices
    through to prepare_inputs."""
    from zeus import Zeus
    import zeus.api as api_mod

    captured = {}

    def fake_prepare(X, *, categorical_indices=None, target_dim=c.INPUT_DIM):
        captured["categorical_indices"] = categorical_indices
        return torch.zeros(X.shape[0], target_dim, dtype=torch.float32)

    monkeypatch.setattr(api_mod, "prepare_inputs", fake_prepare)

    arr = np.zeros((5, 3), dtype=np.float32)
    Zeus(model_path=fake_checkpoint, device="cpu",
         categorical_indices=[0, 2]).fit_transform(arr)
    assert captured["categorical_indices"] == [0, 2]
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/test_api.py::test_zeus_passes_categorical_indices_to_prepare_inputs -v`
Expected: FAIL — `categorical_indices=` kwarg not on `Zeus`.

### Task 3.3: Write failing tests — `n_init` per-method defaults

- [ ] **Step 1: Add the test**

```python
def test_n_init_defaults_per_method(fake_checkpoint, small_df, monkeypatch):
    """When n_init is None (default), ZeusClusterer.fit must use the paper value
    for the chosen method: kmeans 100, gmm 10, simple_gmm 10. Explicit override
    honored verbatim."""
    from zeus import ZeusClusterer
    import sklearn.cluster
    import sklearn.mixture
    import zeus.inference_methods.simple_gmm as sgmm_mod

    captured = {}

    real_kmeans = sklearn.cluster.KMeans
    def fake_kmeans(*args, **kwargs):
        captured["kmeans_n_init"] = kwargs.get("n_init")
        return real_kmeans(*args, **kwargs)
    monkeypatch.setattr("zeus.api.KMeans", fake_kmeans)

    real_gmm = sklearn.mixture.GaussianMixture
    def fake_gmm(*args, **kwargs):
        captured["gmm_n_init"] = kwargs.get("n_init")
        return real_gmm(*args, **kwargs)
    monkeypatch.setattr("zeus.api.GaussianMixture", fake_gmm)

    real_sgmm = sgmm_mod.SimplifiedGMM
    def fake_sgmm(*args, **kwargs):
        captured["sgmm_n_init"] = kwargs.get("n_init")
        return real_sgmm(*args, **kwargs)
    monkeypatch.setattr("zeus.inference_methods.simple_gmm.SimplifiedGMM", fake_sgmm)

    # Default (None) → paper-faithful per method
    ZeusClusterer(n_clusters=3, method="kmeans",
                  model_path=fake_checkpoint, device="cpu",
                  random_state=0).fit(small_df)
    assert captured["kmeans_n_init"] == 100

    ZeusClusterer(n_clusters=3, method="gmm",
                  model_path=fake_checkpoint, device="cpu",
                  random_state=0).fit(small_df)
    assert captured["gmm_n_init"] == 10

    ZeusClusterer(n_clusters=3, method="simple_gmm",
                  model_path=fake_checkpoint, device="cpu",
                  random_state=0).fit(small_df)
    assert captured["sgmm_n_init"] == 10

    # Explicit override honored verbatim (no /10 divisor)
    captured.clear()
    ZeusClusterer(n_clusters=3, method="gmm", n_init=42,
                  model_path=fake_checkpoint, device="cpu",
                  random_state=0).fit(small_df)
    assert captured["gmm_n_init"] == 42
```

- [ ] **Step 2: Run — expect failure (n_init=10 default still in place)**

Run: `pytest tests/test_api.py::test_n_init_defaults_per_method -v`
Expected: FAIL — current default is `n_init=10` for all methods.

### Task 3.4: Implement the api.py changes

**Files:** Modify: `zeus/api.py`

- [ ] **Step 1: Update the `Zeus` class signature and `transform` method**

Apply these edits to `zeus/api.py`:

A. **Import — add `Sequence`** (top of file, in the typing imports):
```python
from typing import Literal, Optional, Sequence, Union
```

B. **Replace `Zeus.__init__`**:
```python
    def __init__(
        self,
        *,
        device: DeviceLike = "auto",
        categorical_indices: Optional[Sequence[int]] = None,
        paper_preprocess: bool = True,
        model_path: Optional[Union[Path, str]] = None,
        cache_dir: Optional[Union[Path, str]] = None,
    ):
        self.device = device
        self.categorical_indices = categorical_indices
        self.paper_preprocess = paper_preprocess
        self.model_path = model_path
        self.cache_dir = cache_dir
```

C. **Replace `Zeus.transform`**:
```python
    def transform(self, X: ArrayLike) -> np.ndarray:
        self._ensure_model()
        if self.paper_preprocess:
            x = prepare_inputs(X, categorical_indices=self.categorical_indices)
        else:
            x = passthrough_inputs(X)
        x = x.unsqueeze(1).to(self._device)
        with torch.no_grad():
            out = self._model(x)
        # Strip the NUM_GAUSSIANS learned cluster centers concatenated by the model.
        emb = out[:-c.NUM_GAUSSIANS].squeeze(1).detach().cpu().numpy()
        return emb
```

- [ ] **Step 2: Update `ZeusClusterer.__init__` signature and `fit`**

A. **Replace `ZeusClusterer.__init__`**:
```python
    def __init__(
        self,
        n_clusters: int,
        *,
        method: Literal["kmeans", "gmm", "simple_gmm"] = "kmeans",
        device: DeviceLike = "auto",
        categorical_indices: Optional[Sequence[int]] = None,
        paper_preprocess: bool = True,
        model_path: Optional[Union[Path, str]] = None,
        cache_dir: Optional[Union[Path, str]] = None,
        random_state: Optional[int] = None,
        n_init: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.method = method
        self.device = device
        self.categorical_indices = categorical_indices
        self.paper_preprocess = paper_preprocess
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.random_state = random_state
        self.n_init = n_init
```

B. **Add the per-method default table at module level** (just below the `_build_model` definition):
```python
_PAPER_N_INIT = {"kmeans": 100, "gmm": 10, "simple_gmm": 10}
```

C. **Replace the body of `ZeusClusterer.fit`** with:
```python
    def fit(self, X, y=None):
        encoder = Zeus(
            device=self.device,
            categorical_indices=self.categorical_indices,
            paper_preprocess=self.paper_preprocess,
            model_path=self.model_path,
            cache_dir=self.cache_dir,
        )
        emb = encoder.fit_transform(X)
        emb = MinMaxScaler(feature_range=(-1, 1)).fit_transform(emb)
        self.embedding_ = emb

        n_init = self.n_init if self.n_init is not None else _PAPER_N_INIT[self.method]

        if self.method == "kmeans":
            km = KMeans(
                n_clusters=self.n_clusters,
                n_init=n_init,
                random_state=self.random_state,
            ).fit(emb)
            self.labels_ = km.labels_
            self.cluster_centers_ = km.cluster_centers_
        elif self.method == "gmm":
            gmm = GaussianMixture(
                n_components=self.n_clusters,
                n_init=n_init,
                random_state=self.random_state,
            ).fit(emb)
            self.labels_ = gmm.predict(emb)
            self.cluster_centers_ = gmm.means_
            self.probabilities_ = gmm.predict_proba(emb)
        elif self.method == "simple_gmm":
            from zeus.inference_methods.simple_gmm import SimplifiedGMM
            sgmm = SimplifiedGMM(
                n_components=self.n_clusters,
                n_init=n_init,
                random_state=self.random_state,
            ).fit(emb)
            self.labels_ = sgmm.predict(emb)
            self.cluster_centers_ = sgmm.means_
            self.probabilities_ = sgmm.predict_proba(emb)
        else:
            raise ValueError(f"Unknown method: {self.method!r}")

        return self
```

- [ ] **Step 3: Run all api tests**

Run: `pytest tests/test_api.py -v`
Expected: all PASS including the three new tests added in 3.1-3.3.

- [ ] **Step 4: Run the full non-OpenML suite**

Run: `pytest tests/ -v --ignore=tests/test_openml_regression.py`
Expected: all PASS.

### Task 3.5: Commit Chunk 3

- [ ] **Step 1: Stage**

Run: `git -C /home/upadhyan/zeus add zeus/api.py tests/test_api.py`

- [ ] **Step 2: Commit**

```bash
git -C /home/upadhyan/zeus commit -m "feat: paper-faithful defaults in Zeus / ZeusClusterer

- Rename preprocess= → paper_preprocess= (no shim; pre-PyPI).
- Add categorical_indices= kwarg to both Zeus and ZeusClusterer; threaded
  through to prepare_inputs (silently ignored for DataFrame inputs).
- ZeusClusterer.n_init now defaults to None and resolves per-method at fit
  time: kmeans 100, gmm 10, simple_gmm 10 (paper-effective values).
  Explicit n_init=k is honored verbatim — no int(k/10) divisor.

Adds unit tests for the kwarg passthrough and the n_init resolution."
```

---

## Chunk 4: Phase 3 — Sweep for stragglers and run the full pre-OpenML suite

**Goal:** Find any remaining `preprocess=` usages (in `zeus/` or `tests/`) that weren't caught by Chunks 2-3, fix them, and confirm the entire non-OpenML suite is green before moving on to re-measurement.

**Files this chunk touches:**
- Modify: any file with a residual `preprocess=` kwarg reference (likely none — Chunks 2/3 cover the known ones).
- Read-only audit elsewhere.

### Task 4.1: Audit for residual `preprocess=` references

- [ ] **Step 1: Search the whole repo**

Run:
```bash
grep -rn "preprocess=" /home/upadhyan/zeus/zeus/ /home/upadhyan/zeus/tests/ /home/upadhyan/zeus/scripts/ /home/upadhyan/zeus/README.md /home/upadhyan/zeus/CLAUDE.md 2>/dev/null
```
Expected: only matches are inside docstrings/comments or in README/CLAUDE.md (those get updated in Chunk 7). If any executable code still has `preprocess=`, fix the call site to use `paper_preprocess=`.

- [ ] **Step 2: Search for `_df_to_numeric_matrix` / `_mean_impute_array` / `_adjust_dim`**

Run:
```bash
grep -rn "_df_to_numeric_matrix\|_mean_impute_array\|_adjust_dim" /home/upadhyan/zeus/ 2>/dev/null
```
Expected: zero matches. If anything is left, the deletion in Chunk 2 was incomplete — delete the references.

### Task 4.2: Full non-OpenML pytest run

- [ ] **Step 1: Run the suite**

Run: `pytest tests/ -v --ignore=tests/test_openml_regression.py`
Expected: every test passes. Specifically, the following should all be green:
- `tests/test_preprocessing.py` — full new matrix
- `tests/test_api.py` — including new categorical_indices + n_init tests
- `tests/test_public_api.py` — unchanged
- `tests/test_model_smoke.py` — unchanged (doesn't touch preprocess)
- `tests/test_weights.py` — unchanged
- `tests/test_config.py` — unchanged

If anything fails: stop, diagnose, fix. Do NOT move on to re-measurement with a red suite.

### Task 4.3: Commit (only if any straggler fixes were made)

- [ ] **Step 1: Check for changes**

Run: `git -C /home/upadhyan/zeus status`

- [ ] **Step 2: If there are changes, stage and commit**

```bash
git -C /home/upadhyan/zeus add <files>
git -C /home/upadhyan/zeus commit -m "fix: tidy residual preprocess= / dead-helper references"
```

If `git status` shows no changes: skip the commit; this task is just verification.

---

## Chunk 5: Phase 4 — Post-fix re-measurement

**Goal:** Re-run `scripts/measure_openml.py` against the new pipeline; record the after-fix ARI/NMI numbers in `tests/openml_baselines.md` so Phase 5 can set thresholds.

**Files this chunk touches:**
- Modify: `tests/openml_baselines.md` (fill the `after_*` columns + update footer)

### Task 5.1: Re-run the script

- [ ] **Step 1: Run**

Run: `python scripts/measure_openml.py`
Expected: 4 lines of progress + a markdown table. ARI/NMI numbers should be **substantially higher** than the before-fix values (especially for `heart-statlog`, which has categoricals).

- [ ] **Step 2: Sanity check — if any after-fix ARI is *not* higher than before-fix, investigate before proceeding**

If any dataset regressed: stop and diagnose. The fix should monotonically improve real-data clustering. (For all-numerical datasets like iris/banknote with no NaN, the change is StandardScaler+KMeans-n_init=100; could be small. For mixed datasets, the gap should be large.)

### Task 5.2: Update `tests/openml_baselines.md`

- [ ] **Step 1: Open the file and fill the after-fix columns + footer**

Use Edit to fill the `after_ari` and `after_nmi` cells for each row with the numbers from Task 5.1, and update the footer with `Date measured (after-fix): <today's date>`.

### Task 5.3: Commit

- [ ] **Step 1: Stage and commit**

```bash
git -C /home/upadhyan/zeus add tests/openml_baselines.md
git -C /home/upadhyan/zeus commit -m "test: record after-fix ARI/NMI for openml baselines

Phase 4 of the paper-faithful preprocessing rewrite. With the new
prepare_inputs + paper n_init defaults, the 4 benchmark datasets show
[describe the gap in one line, e.g. 'ARI improvements of 0.X-0.Y'].
These numbers seed the regression test thresholds in Phase 5."
```

(Replace the bracketed description with a one-line summary of the actual measured gap.)

---

## Chunk 6: Phase 5 — OpenML regression test

**Goal:** Pin the after-fix numbers as a regression test. Register the `openml` pytest marker. Gitignore the cache dir.

**Files this chunk touches:**
- Create: `tests/test_openml_regression.py`
- Modify: `pyproject.toml`
- Modify: `.gitignore`

### Task 6.1: Add `.openml_cache/` to `.gitignore`

- [ ] **Step 1: Edit `.gitignore`**

Append a section at the bottom of `.gitignore`:
```
# OpenML dataset cache (populated by tests/test_openml_regression.py)
.openml_cache/
```

### Task 6.2: Register the `openml` pytest marker

- [ ] **Step 1: Edit `pyproject.toml`**

Replace the existing `[tool.pytest.ini_options]` block:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "openml: end-to-end regression test that downloads OpenML datasets (network)",
]
```

- [ ] **Step 2: Add `openml` to the test extras**

In the `[project.optional-dependencies]` block, change the `test` line from:
```toml
test = ["pytest>=7.0"]
```
to:
```toml
test = ["pytest>=7.0", "openml>=0.14"]
```

### Task 6.3: Write the regression test

**Files:** Create: `tests/test_openml_regression.py`

- [ ] **Step 1: Write the test file**

```python
"""End-to-end regression test on 4 OpenML datasets.

Pins post-fix ARI for the paper-faithful preprocessing rewrite. Thresholds
were measured by scripts/measure_openml.py and recorded in
tests/openml_baselines.md. The 0.05 buffer absorbs sklearn-version and
random-seed jitter — with random_state=42 and n_init=100, KMeans is
deterministic enough that observed jitter is ~0.01-0.02.

Marker: `openml`. Skip with `pytest -m "not openml"` in offline envs.

OpenML datasets are cached in `.openml_cache/` (gitignored); first run
downloads ~few MB total, subsequent runs are local.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import openml
import pytest
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder

from zeus import ZeusClusterer

# Cache OpenML data inside the repo so the test is hermetic.
openml.config.cache_directory = str(Path(__file__).parent.parent / ".openml_cache")

# Thresholds = post-fix ARI from tests/openml_baselines.md, minus 0.05.
# Update both this dict and openml_baselines.md if the post-fix numbers change.
# Format: openml_id → (name, min_ari)
DATASETS = {
    61:   ("iris",                    <fill from openml_baselines.md after-fix>),
    1462: ("banknote-authentication", <fill>),
    53:   ("heart-statlog",           <fill>),
    1510: ("wdbc",                    <fill>),   # or 40496 / 4153 if 1510 was swapped
}


@pytest.mark.openml
@pytest.mark.parametrize("openml_id,name,min_ari", [
    (i, n, t) for i, (n, t) in DATASETS.items()
])
def test_openml_clustering_above_threshold(openml_id, name, min_ari):
    ds = openml.datasets.get_dataset(openml_id, download_data=True)
    X, y, _, _ = ds.get_data(
        dataset_format="dataframe",
        target=ds.default_target_attribute,
    )
    y = LabelEncoder().fit_transform(y)
    n_classes = len(np.unique(y))

    labels = ZeusClusterer(
        n_clusters=n_classes,
        method="kmeans",
        random_state=42,
    ).fit_predict(X)

    ari = adjusted_rand_score(y, labels)
    nmi = normalized_mutual_info_score(y, labels)
    # Log NMI for visibility; we only assert on ARI.
    print(f"[openml id={openml_id} name={name}] ari={ari:.4f} nmi={nmi:.4f}")
    assert ari >= min_ari, (
        f"ARI regression on openml id={openml_id} ({name}): "
        f"got {ari:.4f}, expected >= {min_ari:.4f}"
    )
```

- [ ] **Step 2: Fill the threshold placeholders**

Replace each `<fill>` with `round(after_ari - 0.05, 4)` for that dataset, using the values from `tests/openml_baselines.md`. If 1510 was swapped to 40496 or 4153 in Phase 0, update the key and name here too.

### Task 6.4: Run the regression test

- [ ] **Step 1: Run (network required on first run)**

Run: `pytest tests/test_openml_regression.py -v`
Expected: 4 parameterized cases, all PASS. First run downloads to `.openml_cache/`.

- [ ] **Step 2: Run the full suite including the marker**

Run: `pytest tests/ -v`
Expected: all tests pass. The openml-marked cases are included by default.

- [ ] **Step 3: Verify the marker filter works for offline envs**

Run: `pytest tests/ -v -m "not openml"`
Expected: openml tests are skipped (or not collected), everything else passes.

### Task 6.5: Commit Chunk 6

- [ ] **Step 1: Stage**

Run: `git -C /home/upadhyan/zeus add tests/test_openml_regression.py pyproject.toml .gitignore`

- [ ] **Step 2: Commit**

```bash
git -C /home/upadhyan/zeus commit -m "test: openml regression suite pins post-fix ARI

4 datasets from the openml_ids paper benchmark list, parameterized,
asserting ARI >= post-fix - 0.05. Threshold buffer absorbs sklearn
minor-version and random-seed jitter (observed ~0.01-0.02 with
random_state=42 and n_init=100).

Adds the openml pytest marker (skip with -m 'not openml' in offline
envs) and the openml dep to the test extras. .openml_cache/ in
.gitignore so the cache is repo-local but unversioned."
```

---

## Chunk 7: Phase 6 — Documentation updates

**Goal:** Bring `CLAUDE.md`, `README.md`, and `RELEASING.md` in line with the new API and behavior.

**Files this chunk touches:**
- Modify: `CLAUDE.md`
- Modify: `README.md`
- Modify: `RELEASING.md`

### Task 7.1: Update `CLAUDE.md`

**Files:** Modify: `CLAUDE.md`

- [ ] **Step 1: Find the preprocessing-related sections**

Run: `grep -n "preprocess\|OpenML\|prepare_inputs\|MinMax" /home/upadhyan/zeus/CLAUDE.md`
Expected: list of line numbers in the Architecture and Conventions sections.

- [ ] **Step 2: Update the "OpenML evaluation" paragraph** under Architecture (`load_real_datasets` description). The current text describes the old behavior. Replace it with a description of the new paper-faithful flow:

Find this paragraph (currently in CLAUDE.md):
> **OpenML evaluation.** `load_real_datasets` downloads 34 hardcoded OpenML IDs (`openml_ids` in `zeus/utils.py`), imputes/scales/one-hot-encodes via a `ColumnTransformer`, and either returns whole datasets (current default, `return_whole_dataset=True`) or PCA-reduces to `pca_dim`. `evaluate_model` handles the dim mismatch: PCA if `x_dim > config.dim`, zero-pad if smaller.

Replace with:
> **Real-data preprocessing.** `zeus/preprocessing.py:prepare_inputs` mirrors the paper's `load_real_datasets` + `evaluate_model` pipeline: per-block `SimpleImputer→StandardScaler→MinMaxScaler(-1,1)` on numerical columns, `SimpleImputer(most_frequent)→OneHotEncoder` on categoricals, then PCA-to-30 if too wide or zero-pad if too narrow. The `d == INPUT_DIM` no-rescale branch is load-bearing: it preserves OHE columns as `{0, 1}`, which is what the model was trained on. The original (research) `load_real_datasets` / `evaluate_model` live in the untracked top-level `datasets.py` / `utils.py` files copied from the upstream repo for reference.

- [ ] **Step 3: Update the "Conventions / gotchas" section**

Add a new bullet under the existing list:
> - **`ZeusClusterer.n_init` defaults are per-method.** `kmeans→100`, `gmm→10`, `simple_gmm→10` (paper-effective values). Pass `n_init=k` explicitly to override; the override is honored verbatim (no `int(k/10)` divisor).

Update or add a bullet for `categorical_indices`:
> - **ndarray inputs need explicit `categorical_indices=`.** Without it, every column is treated as numerical. DataFrames auto-detect from dtype; the kwarg is silently ignored when both are passed.

### Task 7.2: Update `README.md`

**Files:** Modify: `README.md`

- [ ] **Step 1: Find the Zeus / ZeusClusterer API ref blocks**

Run: `grep -n "preprocess=\|ZeusClusterer\|zeus.Zeus" /home/upadhyan/zeus/README.md`

- [ ] **Step 2: Update the `Zeus` API ref signature and behavior paragraph**

Replace:
> ### `zeus.Zeus(*, device="auto", preprocess=True, model_path=None, cache_dir=None)`

With:
> ### `zeus.Zeus(*, device="auto", categorical_indices=None, paper_preprocess=True, model_path=None, cache_dir=None)`

Replace the behavior paragraph:
> With `preprocess=True` (default), DataFrames with mixed dtypes are auto-encoded (one-hot for object/category/bool columns, mean-imputation for NaNs, PCA-or-zero-pad to dim 30, MinMax-scale to `[-1, 1]`). With `preprocess=False`, the input must already be a numeric `(n, 30)` array.

With:
> With `paper_preprocess=True` (default), inputs go through the paper's per-block pipeline: `SimpleImputer→StandardScaler→MinMaxScaler(-1, 1)` for numerical columns, `SimpleImputer(most_frequent)→OneHotEncoder` for categoricals, then PCA-to-30 if too wide or zero-pad if too narrow. DataFrame inputs auto-detect categoricals from dtype; for ndarray/Tensor inputs, pass `categorical_indices=[i, j, ...]` listing the categorical column indices (otherwise all columns are treated as numerical). With `paper_preprocess=False`, the input must already be a numeric `(n, 30)` array.

- [ ] **Step 3: Update the `ZeusClusterer` API ref signature**

Replace:
> ### `zeus.ZeusClusterer(n_clusters, *, method="kmeans", device="auto", preprocess=True, model_path=None, cache_dir=None, random_state=None, n_init=10)`

With:
> ### `zeus.ZeusClusterer(n_clusters, *, method="kmeans", device="auto", categorical_indices=None, paper_preprocess=True, model_path=None, cache_dir=None, random_state=None, n_init=None)`

Add a short note below the method bullets:
> `n_init=None` resolves at fit-time to the paper-effective default for the chosen method: `kmeans→100`, `gmm→10`, `simple_gmm→10`. Explicit values are honored verbatim.

### Task 7.3: Update `RELEASING.md`

**Files:** Modify: `RELEASING.md`

- [ ] **Step 1: Read the existing structure**

Run: `cat /home/upadhyan/zeus/RELEASING.md`

- [ ] **Step 2: Add a v0.2.0 entry**

Follow the format of the existing v0.1.0 entry. Add a section above it:
```markdown
## v0.2.0 (unreleased)

**Breaking changes:**
- `Zeus(preprocess=...)` renamed to `Zeus(paper_preprocess=...)`. Same default (`True`).
- `ZeusClusterer.n_init` default is now `None`, which resolves per-method
  at fit-time (`kmeans→100`, `gmm→10`, `simple_gmm→10`). Previously a fixed
  `10` for all methods. Explicit `n_init=k` is honored verbatim.

**Features:**
- New `categorical_indices=` kwarg on `Zeus` and `ZeusClusterer` for
  ndarray/Tensor inputs with categorical columns. DataFrames continue to
  auto-detect from dtype.

**Fixes:**
- `prepare_inputs` now matches the paper's `load_real_datasets` +
  `evaluate_model` pipeline exactly. The previous implementation skipped
  `StandardScaler` on numerical columns and re-MinMaxed after concat
  (remapping OHE columns from `{0, 1}` to `{-1, +1}`), causing severely
  degraded ARI/NMI on real data. Pinned by
  `tests/test_openml_regression.py`.

**Silent behavior changes:**
- `prepare_inputs` on ndarray inputs containing NaN no longer emits a
  `warnings.warn`; sklearn's `SimpleImputer` imputes silently (sklearn
  convention).
```

### Task 7.4: Final commit

- [ ] **Step 1: Stage**

Run: `git -C /home/upadhyan/zeus add CLAUDE.md README.md RELEASING.md`

- [ ] **Step 2: Commit**

```bash
git -C /home/upadhyan/zeus commit -m "docs: update CLAUDE.md / README / RELEASING for paper-faithful pipeline

- CLAUDE.md: rewrite the Architecture preprocessing paragraph to describe
  the new per-block pipeline; add Conventions bullets for per-method
  n_init defaults and categorical_indices=.
- README.md: update Zeus / ZeusClusterer signatures and behavior text.
- RELEASING.md: add v0.2.0 entry (breaking renames + n_init default
  change + categorical_indices + the OHE-preservation fix)."
```

### Task 7.5: Final smoke test

- [ ] **Step 1: One more clean test run**

Run: `pytest tests/ -v`
Expected: full green, including openml.

- [ ] **Step 2: Confirm git log is sensible**

Run: `git -C /home/upadhyan/zeus log --oneline -10`
Expected: 6-7 new commits (Phases 0, 2, 3, 4, 5, 6, maybe 4-cleanup if it was needed), in order, each with a clear message.

---

## Done

After Chunk 7 completes successfully, the implementation is finished:

1. ✅ `Zeus` / `ZeusClusterer` reproduce the paper's preprocessing exactly.
2. ✅ ndarray users can pass `categorical_indices=` to get parity with DataFrame inputs.
3. ✅ Paper-effective `n_init` defaults per method.
4. ✅ End-to-end OpenML regression test pinning the post-fix numbers.
5. ✅ Docs updated.
6. ✅ Migration documented in RELEASING.md.

The Phase 0 baseline file (`tests/openml_baselines.md`) is the receipt: it
shows the before-fix numbers, the after-fix numbers, and the gap.
