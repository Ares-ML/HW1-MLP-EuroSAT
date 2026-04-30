"""Internal-representation analysis: which classes does each fc1 neuron prefer?

The first hidden layer of an MLP doesn't directly correspond to classes — its
neurons learn generic low-level features that the deeper layers compose into
class decisions. But we can still ask, empirically, "which classes most strongly
activate neuron j?" This module answers that question and persists the answer
as JSON so the report can cite specific numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import MLP


def class_activation_stats(
    model: MLP,
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int = 10,
    batch_size: int = 512,
) -> np.ndarray:
    """Mean post-activation value of every fc1 neuron, grouped by class.

    Returns a ``(H1, K)`` array where ``[j, c]`` is the average value that
    ``act1(fc1(x))`` takes on samples whose label is ``c``.

    Computed in eval mode (dropout off) so the result is deterministic.
    Accumulator is float64 to avoid summation drift over ~21,600 train samples.
    """
    was_training = model.training
    model.eval()

    h1 = model.fc1.params["W"].shape[1]
    sums = np.zeros((h1, num_classes), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)

    n = X.shape[0]
    for start in range(0, n, batch_size):
        xb = X[start:start + batch_size]
        yb = y[start:start + batch_size]
        z1 = model.fc1.forward(xb)
        a1 = model.act1.forward(z1)
        for c in range(num_classes):
            mask = yb == c
            if not mask.any():
                continue
            sums[:, c] += a1[mask].sum(axis=0).astype(np.float64)
            counts[c] += int(mask.sum())

    means = sums / np.maximum(counts, 1)
    if was_training:
        model.train()
    return means.astype(np.float32)


def neuron_class_preference(activation_means: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray]:
    """For each neuron return ``(best_class, preference_strength)``.

    ``preference_strength`` is the gap between the best class's mean activation
    and the average of the other classes', divided by the overall mean — a
    dimensionless score where 0 = no preference, ~1 = strongly class-selective.
    Dead neurons (overall mean ~ 0) get a small/zero score by construction.
    """
    h1, k = activation_means.shape
    best_class = activation_means.argmax(axis=1)
    total = activation_means.sum(axis=1)
    best_act = activation_means[np.arange(h1), best_class]
    others_mean = (total - best_act) / max(k - 1, 1)
    overall_mean = total / k
    pref = (best_act - others_mean) / (overall_mean + 1e-8)
    return best_class, pref.astype(np.float32)


def save_activation_analysis(
    activation_means: np.ndarray,
    class_names,
    path: str | Path,
    top_per_class: int = 3,
) -> dict:
    """Persist the analysis as JSON so the report can quote actual numbers.

    For every class, lists the top-``top_per_class`` most strongly-preferring
    neurons, each with its full per-class mean-activation vector.
    """
    h1, k = activation_means.shape
    best_class, pref = neuron_class_preference(activation_means)
    overall_mean = activation_means.mean(axis=1)

    summary: dict = {
        "num_neurons": int(h1),
        "num_classes": int(k),
        "class_names": list(class_names),
        "fraction_dead_neurons": float((overall_mean < overall_mean.max() * 0.01).mean()),
        "preference_strength_quantiles": {
            "p50": float(np.percentile(pref, 50)),
            "p75": float(np.percentile(pref, 75)),
            "p90": float(np.percentile(pref, 90)),
            "p99": float(np.percentile(pref, 99)),
        },
        "top_neurons_per_class": {},
    }

    for c in range(k):
        cand = np.where(best_class == c)[0]
        if cand.size == 0:
            summary["top_neurons_per_class"][class_names[c]] = []
            continue
        ranking = cand[np.argsort(-pref[cand])][:top_per_class]
        items = []
        for j in ranking:
            items.append({
                "neuron_id": int(j),
                "preference_strength": float(pref[j]),
                "best_activation": float(activation_means[j, c]),
                "class_means": {
                    class_names[cc]: float(activation_means[j, cc])
                    for cc in range(k)
                },
            })
        summary["top_neurons_per_class"][class_names[c]] = items

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary
