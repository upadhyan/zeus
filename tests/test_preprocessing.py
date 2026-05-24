"""Input preprocessing: DataFrame / ndarray / Tensor -> (n, INPUT_DIM) float32 tensor."""
from __future__ import annotations

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


def test_pca_branch_rescales():
    """When d > target_dim, PCA reduces to 30 dims then MinMax(-1, 1)."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 64)).astype(np.float32)
    out = prepare_inputs(X).numpy()
    assert out.shape == (50, c.INPUT_DIM)
    assert out.min() >= -1.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6


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
