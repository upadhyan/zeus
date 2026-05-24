# OpenML clustering baselines

Measured by `scripts/measure_openml.py` on the 4 datasets pinned by
the paper-faithful-preprocessing regression test
(`tests/test_openml_regression.py`).

| id | name | n_features (raw) | n_features (post-OHE) | branch | before_ari | before_nmi | after_ari | after_nmi |
|---|---|---|---|---|---|---|---|---|
| 61   | iris                    | 4  | 4  | pad   | 0.4768 | 0.5394 |   |   |
| 1462 | banknote-authentication | 4  | 4  | pad   | 0.5716 | 0.5027 |   |   |
| 53   | heart-statlog           | 13 | 13 | pad   | 0.0517 | 0.0705 |   |   |
| 1510 | wdbc                    | 30 | 30 | exact | 0.7426 | 0.6309 |   |   |

## Run conditions

- `ZeusClusterer(n_clusters=n_classes, method='kmeans', random_state=42)`
- sklearn version: 1.5.2
- `zeus.pt` SHA-256: `eb60086459de338b1795ce2849e02e0d7e59cd95af43d01534dd84083f2b749b`
  (sourced from `zeus/weights.py::EXPECTED_SHA256`; `RELEASING.md` is untracked
  on `main` and did not carry over into this feature branch.)
- Date measured (before-fix): 2026-05-23
- Date measured (after-fix): <to be filled in Phase 4>
