"""Synthetic-data regression test.

Smoke check that Zeus + KMeans recovers labels on sklearn's moons
dataset (2-D, 2 non-convex classes). Threshold (>0.9 ARI) is a
hardcoded sanity floor specified by the project owner — Zeus is
expected to handle this trivially after the paper-faithful preprocessing
rewrite. If this regresses, something is wrong with the pipeline at a
fundamental level.

No marker — runs in the default suite, offline.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from zeus import ZeusClusterer


def test_moons_clustering_ari_above_threshold():
    X, y = make_moons(n_samples=500, noise=0.05, random_state=42)
    labels = ZeusClusterer(
        n_clusters=2,
        method="kmeans",
        random_state=42,
    ).fit_predict(X)
    ari = adjusted_rand_score(y, labels)
    nmi = normalized_mutual_info_score(y, labels)
    print(f"[moons] ari={ari:.4f} nmi={nmi:.4f}")
    assert ari > 0.9, (
        f"ARI regression on moons: got {ari:.4f}, expected > 0.9. "
        f"This is a fundamental sanity check; failing it means the "
        f"pipeline is broken at the embedding or clustering layer."
    )
