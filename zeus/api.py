"""sklearn-compatible estimators wrapping the ZEUS transformer (spec §5.4)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Sequence, Union

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
from zeus.inference_methods.simple_gmm import SimplifiedGMM
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


_PAPER_N_INIT = {"kmeans": 100, "gmm": 10, "simple_gmm": 10}


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
