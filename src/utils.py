"""Utility helpers: seeding and numerical gradient checking."""

from __future__ import annotations

import random
from typing import Any, Iterable

import numpy as np

from .layers import softmax_cross_entropy
from .model import MLP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _data_loss_only(model: MLP, x: np.ndarray, y: np.ndarray) -> float:
    """Cross-entropy (no L2, no dropout) for use inside the grad check.

    Mirrors whatever ``label_smoothing`` setting the model is trained with —
    otherwise finite differences would measure a different loss than the
    analytic backward pass, breaking the check spuriously.
    """
    was_training = model.training
    model.eval()
    logits = model.forward(x)
    loss, _ = softmax_cross_entropy(logits, y, label_smoothing=model.label_smoothing)
    if was_training:
        model.train()
    return float(loss)


def gradient_check(
    model: MLP,
    x: np.ndarray,
    y: np.ndarray,
    params: Iterable[tuple[str, Any, str]] | None = None,
    num_checks: int = 10,
    h: float = 1e-5,
    seed: int = 0,
) -> list[dict]:
    """Centered finite-difference check against analytic gradients.

    Runs in float64 to avoid measuring numerical noise from the fp32 forward.
    With ReLU kinks the expected relative error can reach ~1e-4 even for a
    correct implementation; >1e-2 indicates a real bug.
    """
    rng = np.random.default_rng(seed)

    # Force WD=0 and dropout=0 for the check, then restore.
    saved_wd = model.weight_decay
    saved_p1 = model.drop1.p
    saved_p2 = model.drop2.p
    model.weight_decay = 0.0
    model.drop1.p = 0.0
    model.drop2.p = 0.0
    model.eval()

    # Temporarily upcast params to float64.
    originals = {}
    for name, layer, key in model.iter_params():
        originals[name] = layer.params[key]
        layer.params[key] = layer.params[key].astype(np.float64)
    x64 = x.astype(np.float64)

    # Build analytic grads once.
    _, _ = model.loss_and_grads(x64, y)
    analytic_grads = {
        name: layer.grads[key].copy()
        for name, layer, key in model.iter_params()
    }

    param_list = list(params or model.iter_params())
    results = []
    for name, layer, key in param_list:
        W = layer.params[key]
        flat_len = W.size
        picks = rng.choice(flat_len, size=min(num_checks, flat_len), replace=False)
        rel_errs = []
        for idx in picks:
            flat_idx = int(idx)
            orig_val = W.flat[flat_idx]
            W.flat[flat_idx] = orig_val + h
            loss_plus = _data_loss_only(model, x64, y)
            W.flat[flat_idx] = orig_val - h
            loss_minus = _data_loss_only(model, x64, y)
            W.flat[flat_idx] = orig_val

            numeric = (loss_plus - loss_minus) / (2 * h)
            analytic = analytic_grads[name].flat[flat_idx]
            denom = max(abs(numeric) + abs(analytic), 1e-12)
            rel_errs.append(abs(numeric - analytic) / denom)
        results.append({
            "param": name,
            "max_rel_err": max(rel_errs),
            "mean_rel_err": float(np.mean(rel_errs)),
            "num_checks": len(rel_errs),
        })

    # Restore everything.
    for name, layer, key in model.iter_params():
        layer.params[key] = originals[name]
    model.weight_decay = saved_wd
    model.drop1.p = saved_p1
    model.drop2.p = saved_p2
    return results
