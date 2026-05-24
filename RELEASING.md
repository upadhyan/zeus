# Release notes

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
- `prepare_inputs` now matches the paper's per-block preprocessing
  pipeline exactly. The previous implementation skipped `StandardScaler`
  on numerical columns and re-MinMaxed after concat (remapping OHE
  columns from `{0, 1}` to `{-1, +1}` and padding columns from `0` to
  `-1`), causing severely degraded ARI/NMI on real data. Pinned by
  `tests/test_openml_regression.py` (4 OpenML datasets) and
  `tests/test_synthetic_regression.py` (moons, ARI > 0.9).

**Silent behavior changes:**
- `prepare_inputs` on ndarray inputs containing NaN no longer emits a
  `warnings.warn`; sklearn's `SimpleImputer` imputes silently (sklearn
  convention).
