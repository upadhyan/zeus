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


def test_zeus_paper_preprocess_false_requires_exact_dim(fake_checkpoint):
    from zeus import Zeus
    bad = np.zeros((10, 20), dtype=np.float32)
    with pytest.raises(ValueError, match=r"expected \(n, 30\)"):
        Zeus(model_path=fake_checkpoint, device="cpu", paper_preprocess=False).transform(bad)


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
