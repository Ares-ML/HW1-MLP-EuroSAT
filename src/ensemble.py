"""Logit-averaging ensemble of multiple ``MLP`` checkpoints.

Each member can additionally use D4 test-time augmentation, in which case the
final prediction is averaged over ``num_models * 8`` forward passes.

The module is kept as a small library function so it can be exercised from
both ``scripts/ensemble.py`` (CLI) and notebooks / unit tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .evaluate import batched_predict, batched_predict_tta
from .model import MLP


def predict_ensemble(
    checkpoints: Iterable[str | Path],
    X: np.ndarray,
    use_tta: bool = False,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Average logits across all ``checkpoints``.

    Returns
    -------
    preds : (N,) int64
        Ensemble argmax.
    mean_logits : (N, K) float32
        Mean logits across members. Useful for top-k accuracy and for stacking
        per-pair error analysis on the ensemble (rather than any single model).
    member_logits : list of (N, K) arrays
        Per-member logits, in the order checkpoints were given. Lets the caller
        report per-member metrics for an ablation table.
    """
    checkpoints = [Path(p) for p in checkpoints]
    if not checkpoints:
        raise ValueError("predict_ensemble needs at least one checkpoint")

    sum_logits: np.ndarray | None = None
    member_logits: list[np.ndarray] = []
    for ckpt in checkpoints:
        model = MLP.load(ckpt)
        if use_tta:
            _, logits = batched_predict_tta(model, X, batch_size=batch_size)
        else:
            _, logits = batched_predict(model, X, batch_size=batch_size)
        member_logits.append(logits)
        sum_logits = logits if sum_logits is None else sum_logits + logits

    assert sum_logits is not None
    mean_logits = sum_logits / len(checkpoints)
    preds = mean_logits.argmax(axis=1)
    return preds, mean_logits, member_logits
