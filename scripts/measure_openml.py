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


def _low_card_uint8_indices(X: pd.DataFrame, max_card: int = 10) -> list[int]:
    """Identify uint8 columns with low cardinality — treat as categorical.

    Heart-statlog stores categorical features (sex, chest_pain_type, slope,
    thal, etc.) as uint8 alongside continuous uint8 features (age,
    resting_blood_pressure). OpenML's categorical_indicator marks all of
    them as numeric, so we override with this heuristic.
    """
    out = []
    for i, c in enumerate(X.columns):
        if str(X[c].dtype) == "uint8" and X[c].nunique() <= max_card:
            out.append(i)
    return out


# Per-dataset categorical_indices resolver. None means "let dtype/metadata decide".
# Provisional 4 datasets per spec Section 4. 1510 may be swapped after
# inspecting its post-OHE feature count (see Task 1.4).
OPENML_IDS = [
    (61,   "iris",                    None),
    (1462, "banknote-authentication", None),
    (53,   "heart-statlog",           _low_card_uint8_indices),
    (1510, "wdbc",                    None),
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


def measure_one(openml_id: int, name: str, cat_resolver) -> dict:
    ds = openml.datasets.get_dataset(openml_id, download_data=True)
    X, y, categorical_indicator, _ = ds.get_data(
        dataset_format="dataframe",
        target=ds.default_target_attribute,
    )
    y = LabelEncoder().fit_transform(y)
    n_classes = len(np.unique(y))
    n_features_raw = X.shape[1]
    n_features_post_ohe = post_ohe_feature_count(X, categorical_indicator)

    cat_indices = cat_resolver(X) if cat_resolver is not None else None

    if n_features_post_ohe > INPUT_DIM:
        branch = "pca"
    elif n_features_post_ohe < INPUT_DIM:
        branch = "pad"
    else:
        branch = "exact"

    # `categorical_indices=` is silently ignored for DataFrame inputs (dtype
    # wins), so convert to ndarray when we want the override to land. This
    # matches the realistic user workflow when OpenML's `categorical_indicator`
    # is wrong about a column's semantic role.
    X_for_zeus = X.to_numpy() if cat_indices is not None else X

    t0 = time.time()
    labels = ZeusClusterer(
        n_clusters=n_classes,
        method="kmeans",
        categorical_indices=cat_indices,
        random_state=42,
    ).fit_predict(X_for_zeus)
    elapsed = time.time() - t0

    return {
        "id": openml_id,
        "name": name,
        "n_classes": n_classes,
        "n_features_raw": n_features_raw,
        "n_features_post_ohe": n_features_post_ohe,
        "branch": branch,
        "cat_indices": cat_indices,
        "ari": adjusted_rand_score(y, labels),
        "nmi": normalized_mutual_info_score(y, labels),
        "elapsed_s": elapsed,
    }


def main() -> int:
    rows = []
    for openml_id, name, cat_resolver in OPENML_IDS:
        try:
            row = measure_one(openml_id, name, cat_resolver)
        except Exception as e:
            print(f"[error] id={openml_id} ({name}): {e}", file=sys.stderr)
            continue
        rows.append(row)
        ci_str = "-" if row["cat_indices"] is None else str(row["cat_indices"])
        print(
            f"id={row['id']:>5}  name={row['name']:<32}  "
            f"raw={row['n_features_raw']:>3}  ohe={row['n_features_post_ohe']:>3}  "
            f"branch={row['branch']:<5}  n_classes={row['n_classes']:>2}  "
            f"cat_indices={ci_str:<40}  "
            f"ari={row['ari']:.4f}  nmi={row['nmi']:.4f}  ({row['elapsed_s']:.1f}s)"
        )

    # Print markdown table
    print()
    print("| id | name | n_features (raw) | n_features (post-OHE) | branch | cat_indices | ari | nmi |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        ci_str = "-" if r["cat_indices"] is None else str(r["cat_indices"])
        print(
            f"| {r['id']} | {r['name']} | {r['n_features_raw']} | "
            f"{r['n_features_post_ohe']} | {r['branch']} | {ci_str} | "
            f"{r['ari']:.4f} | {r['nmi']:.4f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
