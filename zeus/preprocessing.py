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
