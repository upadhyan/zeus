# OpenML clustering baselines

Measured by `scripts/measure_openml.py` on the 4 datasets pinned by
the paper-faithful-preprocessing regression test
(`tests/test_openml_regression.py`).

| id | name | n_features (raw) | n_features (post-OHE) | branch | before_ari | before_nmi | after_ari | after_nmi |
|---|---|---|---|---|---|---|---|---|
| 61   | iris                    | 4  | 4  | pad   | 0.4768 | 0.5394 | 0.8515 | 0.8622 |
| 1462 | banknote-authentication | 4  | 4  | pad   | 0.5716 | 0.5027 | 0.9228 | 0.8791 |
| 53   | heart-statlog           | 13 | 13 | pad   | 0.0517 | 0.0705 | 0.4128 | 0.3324 |
| 1510 | wdbc                    | 30 | 30 | exact | 0.7426 | 0.6309 | 0.7426 | 0.6309 |

## Run conditions

- `ZeusClusterer(n_clusters=n_classes, method='kmeans', random_state=42)`
- sklearn version: 1.5.2
- openml version: 0.12.2
- `zeus.pt` SHA-256: `eb60086459de338b1795ce2849e02e0d7e59cd95af43d01534dd84083f2b749b`
  (sourced from `zeus/weights.py::EXPECTED_SHA256`; `RELEASING.md` is untracked
  on `main` and did not carry over into this feature branch.)
- Date measured (before-fix): 2026-05-23
- Date measured (after-fix): 2026-05-24

## Notes

- For id 53 (heart-statlog), 8 uint8 columns with <=10 unique values are
  treated as categorical via the new `categorical_indices=` kwarg
  (`[1, 2, 5, 6, 8, 10, 11, 12]` -- sex, chest_pain, fasting_blood_sugar,
  resting_ecg, exercise_induced_angina, slope, num_major_vessels, thal).
  OpenML's `categorical_indicator` marks all 13 columns as numeric, which
  doesn't match their semantic role. Heart-statlog is passed to
  `ZeusClusterer` as an ndarray (DataFrame input silently ignores
  `categorical_indices` -- dtype wins for DataFrames). The other three
  datasets are passed as DataFrames; their all-numeric dtypes mean
  dtype-detection produces the same num/cat split as
  `categorical_indices=None` would.
- iris and banknote both improved substantially (0.4768 -> 0.8515 and
  0.5716 -> 0.9228) despite having no categorical columns. The new
  pipeline is NOT bit-identical to the old one for all-numerical input:
  the old pipeline padded first and then applied `MinMaxScaler(-1, 1)` to
  the full padded matrix. sklearn's `MinMaxScaler` maps a zero-range
  column to `feature_range[0]`, so the 26 zero-padding columns became -1
  in the old pipeline. The new pipeline scales numeric columns first and
  then pads with zeros that stay 0 -- so the model sees padding at 0
  instead of -1. This is the paper-intended behaviour (zero-padding
  stays zero) and explains the iris/banknote jump.
- wdbc is bit-identical because it has exactly 30 numeric features
  (`branch=exact`, no padding): the post-padding global MinMax in the
  old pipeline and the per-column MinMax in the new pipeline collapse to
  the same per-column operation.
