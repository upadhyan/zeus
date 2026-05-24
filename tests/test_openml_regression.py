"""End-to-end regression test on 4 OpenML datasets.

Pins post-fix ARI for the paper-faithful preprocessing rewrite. Thresholds
are post-fix ARI from tests/openml_baselines.md minus a 0.05 buffer to
absorb sklearn-version and random-seed jitter. With random_state=42 and
n_init=100, KMeans is deterministic enough that observed jitter is
~0.01-0.02.

Heart-statlog (id 53) uses an explicit `categorical_indices=` override
because OpenML's metadata marks all 13 columns as numeric, but 8 of the
uint8 columns are semantically categorical (sex, chest_pain_type,
fasting_blood_sugar, resting_ecg, exercise_induced_angina, slope,
num_major_vessels, thal). Without the override, heart-statlog ARI drops
to ~0.05.

Marker: `openml`. Skip with `pytest -m "not openml"` in offline envs.
OpenML datasets are cached in `.openml_cache/` (gitignored).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder

from zeus import ZeusClusterer

openml = pytest.importorskip("openml", reason="openml not installed; skipping openml regression tests")

# Hermetic cache: lives inside the repo, gitignored.
openml.config.cache_directory = str(Path(__file__).parent.parent / ".openml_cache")

# Per-dataset config:
#   openml_id -> (name, categorical_indices, min_ari)
# categorical_indices=None means rely on dtype/metadata; a list means
# override and pass via the new kwarg (requires ndarray-conversion at
# call time because DataFrame inputs ignore the kwarg).
# Thresholds = round(after_fix_ari - 0.05, 4) from tests/openml_baselines.md.
DATASETS = {
    61:   ("iris",                    None,                          0.8015),
    1462: ("banknote-authentication", None,                          0.8728),
    53:   ("heart-statlog",           [1, 2, 5, 6, 8, 10, 11, 12],   0.3628),
    1510: ("wdbc",                    None,                          0.6926),
}


@pytest.mark.openml
@pytest.mark.parametrize("openml_id,name,cat_indices,min_ari", [
    (i, n, c, t) for i, (n, c, t) in DATASETS.items()
])
def test_openml_clustering_above_threshold(openml_id, name, cat_indices, min_ari):
    ds = openml.datasets.get_dataset(openml_id, download_data=True)
    X, y, _, _ = ds.get_data(
        dataset_format="dataframe",
        target=ds.default_target_attribute,
    )
    y = LabelEncoder().fit_transform(y)
    n_classes = len(np.unique(y))

    # If cat_indices is set, convert to ndarray so the kwarg lands;
    # DataFrame inputs silently ignore categorical_indices (dtype wins).
    X_input = X.to_numpy() if cat_indices is not None else X

    labels = ZeusClusterer(
        n_clusters=n_classes,
        method="kmeans",
        categorical_indices=cat_indices,
        random_state=42,
    ).fit_predict(X_input)

    ari = adjusted_rand_score(y, labels)
    nmi = normalized_mutual_info_score(y, labels)
    # Log NMI for visibility; we only assert on ARI.
    print(f"[openml id={openml_id} name={name}] ari={ari:.4f} nmi={nmi:.4f}")
    assert ari >= min_ari, (
        f"ARI regression on openml id={openml_id} ({name}): "
        f"got {ari:.4f}, expected >= {min_ari:.4f}"
    )
