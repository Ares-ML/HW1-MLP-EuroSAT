"""Evaluation metrics: accuracy, confusion matrix, per-class precision/recall/F1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
                     ) -> np.ndarray:
    """(K, K) confusion matrix. ``cm[i, j]`` = samples with true=i, pred=j."""
    k = num_classes
    flat = y_true * k + y_pred
    return np.bincount(flat, minlength=k * k).reshape(k, k)


@dataclass
class PerClass:
    precision: np.ndarray  # (K,)
    recall: np.ndarray     # (K,)
    f1: np.ndarray         # (K,)
    support: np.ndarray    # (K,)


def per_class_metrics(cm: np.ndarray) -> PerClass:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )
    support = cm.sum(axis=1)
    return PerClass(precision=precision, recall=recall, f1=f1, support=support)


def macro_f1(cm: np.ndarray) -> float:
    return float(per_class_metrics(cm).f1.mean())


def balanced_accuracy(cm: np.ndarray) -> float:
    return float(per_class_metrics(cm).recall.mean())


def cohen_kappa(cm: np.ndarray) -> float:
    """Cohen's kappa computed from a confusion matrix."""
    total = cm.sum()
    if total == 0:
        return 0.0
    po = np.diag(cm).sum() / total
    row = cm.sum(axis=1) / total
    col = cm.sum(axis=0) / total
    pe = float((row * col).sum())
    if pe == 1.0:
        return 0.0
    return float((po - pe) / (1.0 - pe))


def top_k_accuracy(logits: np.ndarray, y_true: np.ndarray, k: int = 3) -> float:
    """Fraction of rows where the true class is among the top-k logit values."""
    topk = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    return float((topk == y_true[:, None]).any(axis=1).mean())


def batched_predict(model, X: np.ndarray, batch_size: int = 512
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(preds, logits)`` without holding a giant forward cache."""
    was_training = model.training
    model.eval()
    all_logits = []
    for start in range(0, X.shape[0], batch_size):
        all_logits.append(model.forward(X[start:start + batch_size]))
    logits = np.concatenate(all_logits, axis=0)
    preds = logits.argmax(axis=1)
    if was_training:
        model.train()
    return preds, logits


def batched_predict_tta(model, X: np.ndarray, image_size: int = 64,
                        batch_size: int = 512
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Test-time augmentation: average logits over the 8 D4 transforms.

    Pairs naturally with the D4 training augmentation; typically gives
    +1–3 points over single-view inference at the cost of 8× forward passes.
    Returns ``(preds, mean_logits)`` with the same shape as ``batched_predict``.
    """
    n = X.shape[0]
    was_training = model.training
    model.eval()

    sum_logits: np.ndarray | None = None
    x_img = X.reshape(n, image_size, image_size, 3)
    for t in range(8):
        view = x_img
        if t >= 4:
            view = view[:, :, ::-1, :]
        k = t % 4
        if k:
            view = np.rot90(view, k=k, axes=(1, 2))
        view = np.ascontiguousarray(view).reshape(n, -1)

        chunks = []
        for start in range(0, n, batch_size):
            chunks.append(model.forward(view[start:start + batch_size]))
        logits_t = np.concatenate(chunks, axis=0)
        sum_logits = logits_t if sum_logits is None else sum_logits + logits_t

    assert sum_logits is not None
    mean_logits = sum_logits / 8.0
    preds = mean_logits.argmax(axis=1)
    if was_training:
        model.train()
    return preds, mean_logits


def format_per_class_table(cm: np.ndarray, class_names) -> str:
    """Pretty-printed table with precision, recall, F1, support per class."""
    pc = per_class_metrics(cm)
    lines = [f"{'class':<22} {'prec':>6} {'rec':>6} {'f1':>6} {'support':>8}"]
    for i, name in enumerate(class_names):
        lines.append(
            f"{name:<22} {pc.precision[i]:6.3f} {pc.recall[i]:6.3f} "
            f"{pc.f1[i]:6.3f} {int(pc.support[i]):8d}"
        )
    lines.append(
        f"{'macro avg':<22} {pc.precision.mean():6.3f} {pc.recall.mean():6.3f} "
        f"{pc.f1.mean():6.3f} {int(pc.support.sum()):8d}"
    )
    return "\n".join(lines)


def top_confusion_pairs(cm: np.ndarray, k: int = 3) -> list[tuple[int, int, int]]:
    """Largest off-diagonal cells as ``(true, pred, count)`` sorted descending."""
    m = cm.copy()
    np.fill_diagonal(m, 0)
    flat_idx = np.argsort(m.ravel())[::-1][:k]
    result = []
    for idx in flat_idx:
        i, j = divmod(int(idx), m.shape[1])
        count = int(m[i, j])
        if count == 0:
            break
        result.append((i, j, count))
    return result
