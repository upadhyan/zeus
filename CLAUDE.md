# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ZEUS — Zero-shot Embeddings for Unsupervised Separation of Tabular Data ([arxiv:2505.10704](https://arxiv.org/abs/2505.10704)). A transformer pre-trained on synthetic GMM-derived datasets that produces embeddings on which standard clusterers (KMeans / GMM) recover cluster structure without per-dataset training. Codebase is derived from TabPFN v1; original license lives in `legal/`.

## Environment

Python ≥3.10. PyTorch ≥2.0 (any build — CPU, CUDA, MPS).

```shell
pip install -e .             # editable install for development
pip install -e ".[test]"     # plus pytest
```

The package is install-only via the git URL; there is no PyPI release.
A CUDA build of PyTorch is recommended for non-trivial inputs but not
required — `Zeus(device="auto")` falls back to CPU.

## Commands

The user-facing surface is sklearn-style; there are no scripts.

```python
from zeus import Zeus, ZeusClusterer

emb     = Zeus().fit_transform(df)
labels  = ZeusClusterer(n_clusters=5).fit_predict(df)
```

The checkpoint downloads on first use. Override its location via
`Zeus(model_path=...)` or the `ZEUS_CACHE_DIR` env var.

Tests:

```shell
pytest tests/ -v
```

No CI, no linter config.

## Architecture

The whole system is one transformer encoder operating on a sequence of tabular rows.

**Forward pass shape convention.** `ZeusTransformerModel.forward` (`zeus/model/zeus.py`) expects `x` shaped `(N, 1, dim)` — sequence-first, batch=1. Training/eval scripts always do `X_batch.unsqueeze(1)` before calling the model. Internally `n_clusters` learned `cluster_centers` (`nn.Parameter` of shape `(n_clusters, 1, ninp)`) are **concatenated to the sequence** before self-attention. The model returns the full concatenated output; callers split with `output[:-config.num_gaussians]` (row embeddings) and `output[-config.num_gaussians:]` (refined cluster centers). Don't forget the slice — feeding the raw output to clustering will include `num_gaussians` extra rows.

**Loss.** `gmm_loss_with_regularizes` in `zeus/utils.py` is the training objective when `loss_type=CENTER` (default). It computes per-cluster means from one-hot labels, a softmax over squared distances weighted by the *prior cluster probabilities* (`probs`, returned from the data generator and proportional to per-cluster point counts), plus two regularizers: a margin term pushing cluster means apart (capped at 0.5) and a compactness term pulling points to their assigned mean. The `dist_lambda` is a constant 1.0; the commented warmup-on-`cur_epoch` lines indicate it was once scheduled.

**Inference path.** `predict_clusters` (`zeus/utils.py`) MinMax-scales embeddings to `[-1, 1]` and runs the chosen `inf_method`. Note the `n_init` quirk for GMM: `GaussianMixture(..., n_init=int(n_init/10))` — passing `n_init<10` gives `n_init=0` which crashes sklearn. The custom `SimplifiedGMM` (`inference_methods/simple_gmm.py`) is a fixed-identity-covariance EM with cluster-collapse reinitialization (`n_resp < 10` triggers a random restart of that component). Use it when you need `predict_proba` for Brier scoring.

**Data generation (`zeus/datasets.py`).** Training data is synthesized on-the-fly. `create_gaussian_mixture` builds a GMM with random per-cluster covariances (eigenvalues sampled in `[p1, p2]`) and Wasserstein-2-thresholded means — `retry_steps *= 2` each iteration if a candidate mean is too close to an existing one. With probability `categorical_chance`, a random number of one-hot categorical dimensions are added. `dataset_generator` then with 50% probability passes the cleaned continuous block through a `RandomNetwork` (stack of spectral-norm-controlled residual MLP blocks) and PCA-projects back — this is the `gaussian_transformed` mode that the model has to learn to invert. All datasets are zero-padded to `config.dim`; `probs` (cluster point fractions) is returned for the loss.

**Real-data preprocessing.** `zeus/preprocessing.py:prepare_inputs` mirrors the paper's `load_real_datasets` + `evaluate_model` pipeline: per-block `SimpleImputer→StandardScaler→MinMaxScaler(-1,1)` on numerical columns, `SimpleImputer(most_frequent)→OneHotEncoder` on categoricals, then PCA-to-30 if too wide or zero-pad if too narrow. The `d == INPUT_DIM` no-rescale branch is load-bearing: it preserves OHE columns as `{0, 1}`, which is what the model was trained on. The original upstream `load_real_datasets` / `evaluate_model` live in the `gmum/zeus` repo, not in this fork; the spec at `docs/superpowers/specs/2026-05-23-paper-faithful-preprocessing-design.md` records the full derivation. End-to-end ARI on 4 OpenML datasets is pinned by `tests/test_openml_regression.py`.

**Transformer layer specifics (`zeus/model/layer.py`).** Custom `TransformerEncoderLayer` supports three masking regimes — tuple (global/train/eval split), int (`single_eval_position` for efficient PFN-style masking), or standard tensor. `ZeusTransformerModel.forward` passes `src_mask = full_len` (an int) when `efficient_eval_masking=True`. The `init_weights` method zeros `linear2` and `self_attn.out_proj` — this is a deliberate residual-init from the TabPFN lineage, not a bug.

## Conventions / gotchas

- **Embeddings are batch-context-dependent.** Self-attention runs across rows, so a row's embedding depends on every other row in the same `transform` call. `Zeus.fit` is therefore a no-op; `ZeusClusterer` exposes only `fit_predict`, never `predict(X_new)`.
- **Frozen checkpoint constants live in `zeus/_config.py`** (`EMBED_DIM=512`, `INPUT_DIM=30`, `NUM_GAUSSIANS=10`, `N_LAYERS=12`, `N_HEAD=4`, `HID_DIM=1024`). The released `zeus.pt` bakes these in; pointing `model_path=` at a checkpoint trained with different hyperparameters is unsupported.
- **`state_dict` is loaded with `strict=False`.** The checkpoint contains parameters from the upstream training-time `decoder`/`out_layer` branches that this fork removed; those become `unexpected_keys` and are tolerated silently by `zeus/weights.py`. Anything else missing emits a `warnings.warn`.
- **`ZeusClusterer.n_init` defaults are per-method.** `kmeans→100`, `gmm→10`, `simple_gmm→10` (paper-effective values). Pass `n_init=k` explicitly to override; the override is honored verbatim (no `int(k/10)` divisor).
- **ndarray inputs need explicit `categorical_indices=`.** Without it, every column is treated as numerical. DataFrames auto-detect from dtype; the kwarg is silently ignored when both are passed (DataFrame dtypes win).
