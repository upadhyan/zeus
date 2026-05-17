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
