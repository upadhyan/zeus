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
