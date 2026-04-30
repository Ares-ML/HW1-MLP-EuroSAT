"""Plots for the HW1 report: curves, confusion matrix, W1 grid, error gallery."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data import CLASS_NAMES, IMAGE_SIZE


def plot_training_curves(history: dict, save_path: str | Path) -> None:
    """Loss (log-y) and accuracy panels. Marks best-val-acc epoch."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    best_epoch = history.get("best_epoch", -1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="train", marker="o", ms=3)
    ax.plot(epochs, history["val_loss"], label="val", marker="s", ms=3)
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (log)")
    ax.set_title("cross-entropy loss"); ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="train", marker="o", ms=3)
    ax.plot(epochs, history["val_acc"], label="val", marker="s", ms=3)
    if best_epoch > 0:
        ax.axvline(best_epoch, ls="--", color="gray",
                   label=f"best val @ ep{best_epoch}")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy")
    ax.set_title("accuracy"); ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[2]
    ax.plot(epochs, history["lr"], marker="o", ms=3, color="tab:green")
    ax.set_xlabel("epoch"); ax.set_ylabel("learning rate")
    ax.set_title("LR schedule"); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140); plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names, save_path: str | Path,
                          normalize: bool = True, title: str | None = None
                          ) -> None:
    """Heatmap with per-cell annotations."""
    cm_plot = cm.astype(np.float64)
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_plot = cm_plot / cm_plot.sum(axis=1, keepdims=True)
            cm_plot = np.nan_to_num(cm_plot)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_plot, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title or ("normalized confusion matrix" if normalize
                           else "confusion matrix"))
    thresh = cm_plot.max() / 2
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            val = cm_plot[i, j]
            text = f"{val:.2f}" if normalize else str(int(cm[i, j]))
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140); plt.close(fig)


def plot_first_layer_weights(
    W1: np.ndarray,
    std: np.ndarray,
    save_path: str | Path,
    num: int = 64,
) -> None:
    """Top-N first-layer filters as joint-RGB templates with signed colormap.

    W1 has shape (12288, H1). The flatten order used by :func:`data.load_eurosat`
    is the default C-order of ``reshape(N, -1)`` applied to (N, 64, 64, 3), so
    the inverse is ``w.reshape(64, 64, 3)``.

    We rescale by ``std`` only (NOT ``+ mean``) — multiplying by std maps the
    weights back into input-pixel units, but adding the per-channel mean would
    bias every filter toward [0.345, 0.380, 0.408] and (because B has the
    largest mean and smallest std) make the joint min-max display look uniformly
    blue. Instead we use **signed normalization centered at zero**: 0 → gray,
    positive → bright color, negative → dim. That is the standard CS231n filter
    visualization and reveals true color tendencies regardless of channel
    statistics.
    """
    in_dim, h1 = W1.shape
    if in_dim != IMAGE_SIZE * IMAGE_SIZE * 3:
        raise ValueError(f"Expected W1 with {IMAGE_SIZE*IMAGE_SIZE*3} rows, got {in_dim}")

    norms = np.linalg.norm(W1, axis=0)
    order = np.argsort(norms)[::-1][:num]

    cols = 8
    rows = (num + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.3))
    axes = np.array(axes).reshape(-1)
    for i, idx in enumerate(order):
        filt = W1[:, idx].reshape(IMAGE_SIZE, IMAGE_SIZE, 3).astype(np.float32)
        # Rescale to input-pixel units (per channel) so the relative magnitude
        # of R/G/B weights reflects their effect on actual images.
        filt = filt * std
        # Signed normalization centered at zero: 0 stays gray.
        m = float(np.abs(filt).max())
        if m > 0:
            filt_disp = filt / (2.0 * m) + 0.5
        else:
            filt_disp = np.full_like(filt, 0.5)
        axes[i].imshow(np.clip(filt_disp, 0.0, 1.0))
        axes[i].set_xticks([]); axes[i].set_yticks([])
        axes[i].set_title(f"{norms[idx]:.2f}", fontsize=7)
    for j in range(len(order), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Top-{num} first-layer filters — joint RGB (signed)")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140); plt.close(fig)


def plot_first_layer_weights_per_channel(
    W1: np.ndarray,
    std: np.ndarray,
    save_path: str | Path,
    num: int = 16,
) -> None:
    """Per-channel grayscale W1 view: rows = filters, cols = R/G/B.

    Complements the joint-RGB plot by showing each channel's spatial structure
    independently. Per-channel signed normalization (centered at zero) makes
    positive vs. negative weights immediately visible: bright = positive,
    dark = negative, mid-gray = ~0.
    """
    in_dim, _ = W1.shape
    if in_dim != IMAGE_SIZE * IMAGE_SIZE * 3:
        raise ValueError(f"Expected W1 with {IMAGE_SIZE*IMAGE_SIZE*3} rows")

    norms = np.linalg.norm(W1, axis=0)
    order = np.argsort(norms)[::-1][:num]
    channel_names = ("R", "G", "B")

    fig, axes = plt.subplots(num, 3, figsize=(3 * 1.6, num * 1.4))
    if num == 1:
        axes = axes.reshape(1, 3)

    for i, idx in enumerate(order):
        filt = W1[:, idx].reshape(IMAGE_SIZE, IMAGE_SIZE, 3).astype(np.float32)
        filt = filt * std
        for c in range(3):
            chan = filt[:, :, c]
            m = float(np.abs(chan).max())
            disp = chan / (2.0 * m) + 0.5 if m > 0 else np.full_like(chan, 0.5)
            ax = axes[i, c]
            ax.imshow(np.clip(disp, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(channel_names[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"#{int(idx)}\n‖w‖={norms[idx]:.2f}",
                              fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle(f"Top-{num} first-layer filters — per-channel (signed grayscale)",
                 y=1.0)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight"); plt.close(fig)


def plot_error_gallery(
    X_raw: np.ndarray,     # uint8 (N, 64, 64, 3), ORIGINAL pixels
    y_true: np.ndarray,
    y_pred: np.ndarray,
    logits: np.ndarray,
    test_indices_in_raw: np.ndarray,  # mapping from test-row -> raw-row
    save_path: str | Path,
    class_names=CLASS_NAMES,
    top_k: int = 16,
) -> None:
    """Grid of the ``top_k`` highest-confidence misclassifications."""
    # Softmax for display confidence.
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    wrong = np.where(y_true != y_pred)[0]
    if wrong.size == 0:
        return
    confs = probs[wrong, y_pred[wrong]]
    order = wrong[np.argsort(confs)[::-1][:top_k]]

    cols = 4
    rows = (len(order) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6))
    axes = np.array(axes).reshape(-1)
    for i, test_idx in enumerate(order):
        raw_idx = test_indices_in_raw[test_idx]
        img = X_raw[raw_idx]
        axes[i].imshow(img)
        axes[i].set_xticks([]); axes[i].set_yticks([])
        true_c = class_names[y_true[test_idx]]
        pred_c = class_names[y_pred[test_idx]]
        conf = probs[test_idx, y_pred[test_idx]]
        axes[i].set_title(f"T:{true_c}\nP:{pred_c} ({conf:.2f})", fontsize=8)
    for j in range(len(order), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Top error gallery (highest-confidence mistakes)")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140); plt.close(fig)


def plot_per_pair_error_gallery(
    X_raw: np.ndarray,                  # uint8 (N, 64, 64, 3)
    y_true: np.ndarray,
    y_pred: np.ndarray,
    logits: np.ndarray,
    test_indices_in_raw: np.ndarray,
    confusion_pairs: list[tuple[int, int, int]],
    save_dir: str | Path,
    class_names=CLASS_NAMES,
    per_pair: int = 8,
) -> list[Path]:
    """For each ``(true, pred, count)`` pair, write a gallery of ``per_pair``
    examples with the highest predicted probability for the WRONG class.

    Pairs naturally with :func:`evaluate.top_confusion_pairs`. One PNG per
    pair is written under ``save_dir`` so the report can comment on each
    confusion individually.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)

    written: list[Path] = []
    for true_i, pred_j, count in confusion_pairs:
        sel = np.where((y_true == true_i) & (y_pred == pred_j))[0]
        if sel.size == 0:
            continue
        # Sort by predicted-class probability descending — highest-confidence
        # mistakes are the most informative.
        order = sel[np.argsort(-probs[sel, pred_j])][:per_pair]

        cols = min(4, len(order))
        rows = (len(order) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
        axes = np.array(axes).reshape(-1)
        for i, test_idx in enumerate(order):
            raw_idx = int(test_indices_in_raw[test_idx])
            axes[i].imshow(X_raw[raw_idx])
            axes[i].set_xticks([]); axes[i].set_yticks([])
            conf = probs[test_idx, pred_j]
            axes[i].set_title(f"p({class_names[pred_j]})={conf:.2f}", fontsize=8)
        for j in range(len(order), len(axes)):
            axes[j].axis("off")
        title = (f"True: {class_names[true_i]}  →  "
                 f"Predicted: {class_names[pred_j]}    "
                 f"(total {count} mistakes of this type)")
        fig.suptitle(title)
        fig.tight_layout()
        out = save_dir / f"errors_{class_names[true_i]}_to_{class_names[pred_j]}.png"
        fig.savefig(out, dpi=140); plt.close(fig)
        written.append(out)
    return written


def plot_neuron_class_preferences(
    W1: np.ndarray,
    std: np.ndarray,
    X_raw_uint8: np.ndarray,            # (N_total, 64, 64, 3) uint8
    y_raw: np.ndarray,                  # (N_total,) labels matching X_raw_uint8
    activation_means: np.ndarray,       # (H1, K) from class_activation_stats
    save_path: str | Path,
    class_names=CLASS_NAMES,
    top_per_class: int = 1,
    samples_per_neuron: int = 4,
    rng: np.random.Generator | None = None,
) -> list[tuple[int, int, float]]:
    """Pair each class's most-preferring fc1 neuron with samples from that class.

    Layout: one row per selected neuron. Column 0 shows the neuron's W1 template
    using signed normalization (centered at 0; bright = positive weight, dark =
    negative). Columns 1..S show ``samples_per_neuron`` random samples from the
    preferred class, drawn from ``X_raw_uint8`` (so the panel uses original
    pixels, not the z-scored tensor).

    Returns the list of ``(class_id, neuron_id, preference_strength)`` actually
    plotted, so the caller (or the report) can cross-reference numbers.
    """
    from .analysis import neuron_class_preference

    rng = rng if rng is not None else np.random.default_rng(0)
    _, k = activation_means.shape
    best_class, pref = neuron_class_preference(activation_means)

    # Filter dead-ish neurons (overall mean < 5% of the max neuron's mean)
    overall_mean = activation_means.mean(axis=1)
    alive = overall_mean > overall_mean.max() * 0.05

    selected: list[tuple[int, int, float]] = []
    for c in range(k):
        cand = np.where((best_class == c) & alive)[0]
        if cand.size == 0:
            continue
        ranking = cand[np.argsort(-pref[cand])][:top_per_class]
        for j in ranking:
            selected.append((c, int(j), float(pref[j])))

    if not selected:
        return selected

    rows = len(selected)
    cols = 1 + samples_per_neuron
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 1.7, rows * 1.6))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for r, (c, j, score) in enumerate(selected):
        # Column 0: signed-normalized W1 template.
        filt = W1[:, j].reshape(IMAGE_SIZE, IMAGE_SIZE, 3).astype(np.float32) * std
        m = float(np.abs(filt).max())
        disp = filt / (2.0 * m) + 0.5 if m > 0 else np.full_like(filt, 0.5)
        ax = axes[r, 0]
        ax.imshow(np.clip(disp, 0.0, 1.0))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"#{j}  →{class_names[c]}\nstrength={score:.2f}",
                     fontsize=8)

        # Columns 1..S: random samples from the preferred class.
        class_indices = np.where(y_raw == c)[0]
        if class_indices.size == 0:
            continue
        picks = rng.choice(class_indices,
                           size=min(samples_per_neuron, class_indices.size),
                           replace=False)
        for s in range(samples_per_neuron):
            ax = axes[r, 1 + s]
            if s < picks.size:
                ax.imshow(X_raw_uint8[picks[s]])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0 and s == 0:
                ax.set_title(f"samples of preferred class", fontsize=9,
                             loc="left")

    fig.suptitle(
        "Neuron template ↔ preferred class — top fc1 neuron per class\n"
        "(signed-normalized W1 reshape, random class samples on the right)",
        y=1.0,
    )
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return selected


def plot_activation_heatmap(
    activation_means: np.ndarray,       # (H1, K)
    save_path: str | Path,
    class_names=CLASS_NAMES,
    neurons_per_class: int = 3,
) -> None:
    """Heatmap of per-class mean activations for the most class-selective neurons.

    Rows = selected neurons grouped by their preferred class; cols = classes.
    Diagonal-like structure visualises that each row's bright cell sits in the
    column of the neuron's preferred class.
    """
    from .analysis import neuron_class_preference

    _, k = activation_means.shape
    best_class, pref = neuron_class_preference(activation_means)

    rows: list[int] = []
    row_labels: list[str] = []
    for c in range(k):
        cand = np.where(best_class == c)[0]
        if cand.size == 0:
            continue
        top = cand[np.argsort(-pref[cand])][:neurons_per_class]
        for j in top:
            rows.append(int(j))
            row_labels.append(f"#{int(j)} ({class_names[c]})")

    if not rows:
        return
    block = activation_means[rows]

    fig, ax = plt.subplots(figsize=(0.6 * k + 2.0, 0.35 * len(rows) + 1.5))
    im = ax.imshow(block, aspect="auto", cmap="viridis")
    ax.set_xticks(range(k))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("class")
    ax.set_title(f"Mean fc1 activation (top {neurons_per_class} neurons per class)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140); plt.close(fig)


def load_history(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
