# ZEUS — pip-installable fork

> **Note:** This is a personal fork of <https://github.com/gmum/zeus>,
> repackaged for easier pip installation and a cleaner sklearn-style API.
> Please cite the original paper and refer to the upstream repo for the
> canonical research codebase.

## Installation

```bash
pip install git+https://github.com/upadhyan/zeus.git
```

The 305 MB checkpoint downloads automatically on first use and is cached
in your platform's user cache directory (override with `$ZEUS_CACHE_DIR`).

> **Tested with:** Python 3.11, PyTorch 2.5.1 + CUDA 12.1. Any
> `torch >= 2.0` build (including CPU-only) should work.

## Quick start

```python
import pandas as pd
from zeus import Zeus, ZeusClusterer

df = pd.read_csv("mydata.csv")

# 1) Embeddings only
emb = Zeus().fit_transform(df)         # (n, 512) numpy array

# 2) End-to-end clustering
labels = ZeusClusterer(n_clusters=5).fit_predict(df)

# 3) Soft assignments
clf = ZeusClusterer(n_clusters=5, method="simple_gmm").fit(df)
probs = clf.probabilities_             # (n, 5)
```

## API reference

### `zeus.Zeus(*, device="auto", categorical_indices=None, paper_preprocess=True, model_path=None, cache_dir=None)`

sklearn `TransformerMixin`. Methods:

- `fit(X)` — no-op (ZEUS is zero-shot); returns `self`.
- `transform(X)` — returns `(n, 512)` numpy embeddings.
- `fit_transform(X)` — inherited; equivalent to `transform(X)`.

Accepts `np.ndarray`, `pd.DataFrame`, or `torch.Tensor` of shape `(n, d)`.
With `paper_preprocess=True` (default), inputs go through the paper's
per-block pipeline: `SimpleImputer→StandardScaler→MinMaxScaler(-1, 1)`
for numerical columns, `SimpleImputer(most_frequent)→OneHotEncoder` for
categoricals, then PCA-to-30 if too wide or zero-pad if too narrow.
DataFrame inputs auto-detect categoricals from dtype; for ndarray/Tensor
inputs, pass `categorical_indices=[i, j, ...]` listing the categorical
column indices (otherwise all columns are treated as numerical). If both
a DataFrame and `categorical_indices=` are supplied, the kwarg is silently
ignored and DataFrame dtypes win. With `paper_preprocess=False`, the input
must already be a numeric `(n, 30)` array.

**Important:** embeddings depend on every other row in the batch. Calling
`transform(X_test)` after `fit(X_train)` is **not** equivalent to running
both at once — use `fit_transform(X)` on the dataset you want to embed.

### `zeus.ZeusClusterer(n_clusters, *, method="kmeans", device="auto", categorical_indices=None, paper_preprocess=True, model_path=None, cache_dir=None, random_state=None, n_init=None)`

sklearn `ClusterMixin`. Methods:

- `fit(X)` — runs the encoder, MinMax-scales to `[-1, 1]`, then runs the chosen clusterer.
- `fit_predict(X)` — returns `(n,)` int labels.

Fitted attributes: `labels_`, `embedding_`, `cluster_centers_`, and
(when `method != "kmeans"`) `probabilities_` with shape `(n, n_clusters)`
summing to 1 per row.

`n_init=None` resolves at fit-time to the paper-effective default for the
chosen method: `kmeans→100`, `gmm→10`, `simple_gmm→10`. Explicit values
are honored verbatim.

No `predict(X_new)` is provided — same context-dependence reason as `Zeus`.

## Citation

Please cite the original paper:

```bibtex
@article{zeus2025,
  title={ZEUS: Zero-shot Embeddings for Unsupervised Separation of Tabular Data},
  url={https://arxiv.org/abs/2505.10704},
  year={2025}
}
```

## License

This fork inherits the TabPFN v1 license (see `legal/`).

---

# Original README

> **Note:** This section preserves the original README from
> <https://github.com/gmum/zeus> verbatim, with inline notes on
> commands and files that have been removed from this fork.

# ZEUS: Zero-shot Embeddings for Unsupervised Separation of Tabular Data

Code repository for [https://arxiv.org/abs/2505.10704](https://arxiv.org/abs/2505.10704).

Repository is based on the first version of TabPFN. The license is located in the [legal](legal) folder. Link to TabPFN2 repository
[https://github.com/PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN).

## Abstract
Clustering tabular data remains a significant open challenge in data analysis and machine learning.
Unlike for image data, similarity between tabular records often varies across datasets,
making the definition of clusters highly dataset-dependent. Furthermore,
the absence of supervised signals complicates hyperparameter tuning in deep learning clustering methods,
frequently resulting in unstable performance. To address these issues and reduce the need for per-dataset tuning,
we adopt an emerging approach in deep learning: zero-shot learning. We propose ZEUS,
a self-contained model capable of clustering new datasets without any additional training or fine-tuning.
It operates by decomposing complex datasets into meaningful components that can then be clustered effectively.
Thanks to pre-training on synthetic datasets generated from a latent-variable prior,
it generalizes across various datasets without requiring user intervention. To the best of our knowledge,
ZEUS is the first zero-shot method capable of generating embeddings for tabular data in a fully unsupervised manner.
Experimental results demonstrate that it performs on par with or better than traditional clustering algorithms
and recent deep learning-based methods, while being significantly faster and more user-friendly.

## Setup

> **Note:** Manual conda setup is no longer needed in this fork — see the
> Installation section above. The original recipe is preserved below for
> reference.

Setup with conda environment.

```shell
conda create -n zeus python=3.11
conda activate zeus
pip install -r requirements.txt
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

## Experiments
Details of ZEUS configuration parameters can be found in the [zeus/configs.py](zeus/configs.py) file.

## Pre-training

> **Note:** `pretrain.py` has been removed from this fork (inference-only
> package). To retrain ZEUS from scratch, use the original repo at
> <https://github.com/gmum/zeus>.

Pre-training can be performed using the following command:

```shell
python pretrain.py nr_epochs=300 dim=30 use_pca=True num_test_datasets=200 num_categorical=3 pca_dim=30 learning_rate=2e-5 inf_method=KMEANS
```

## Model checkpoint

> **Note:** This fork auto-downloads weights from GitHub Releases on first
> use — no manual download required. The original Google Drive link is
> preserved below.

ZEUS checkpoint is available at [Google Drive](https://drive.google.com/file/d/1D7uikacymUnmmMxjUjBuCNIomqhBWS67/view?usp=sharing).


## Evaluation

> **Note:** `evaluation.py` has been removed from this fork. The
> sklearn-style API replaces it — see Quick start above. For the original
> OpenML evaluation harness, use <https://github.com/gmum/zeus>.

The evaluation of ZEUS can be executed as follows:

```shell
python .\evaluation.py model_path=zeus.pt inf_method=KMEANS eval_dataset=OPENML metric_type=ARI results_file=openml.csv
```
