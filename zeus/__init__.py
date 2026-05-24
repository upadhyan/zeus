"""ZEUS — Zero-shot Embeddings for Unsupervised Separation of tabular data.

Public API:
  - Zeus           — sklearn TransformerMixin; produces row embeddings.
  - ZeusClusterer  — sklearn ClusterMixin; produces hard (and optional soft) labels.

See https://github.com/upadhyan/zeus for installation and examples;
upstream research codebase: https://github.com/gmum/zeus.
"""
from zeus.api import Zeus, ZeusClusterer

__version__ = "0.2.0.dev0"
__all__ = ["Zeus", "ZeusClusterer", "__version__"]
