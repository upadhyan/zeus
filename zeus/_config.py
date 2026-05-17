"""Frozen hyperparameters for the released ZEUS checkpoint (v0.1.0 of this fork).

These are NOT user-tunable. A `model_path=` pointing at a checkpoint trained
with different hyperparameters is unsupported; the load will either fail or
produce garbage.
"""
from __future__ import annotations

EMBED_DIM: int = 512
N_HEAD: int = 4
HID_DIM: int = 1024
N_LAYERS: int = 12
NUM_GAUSSIANS: int = 10
INPUT_DIM: int = 30
DROPOUT: float = 0.0
EFFICIENT_EVAL_MASKING: bool = True
