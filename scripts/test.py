"""Load a trained checkpoint and produce the full evaluation package.

Outputs (all under ``--out-dir``):
  - summary.json : overall + per-class metrics
  - confusion_matrix.png (normalized)
  - w1_filters.png        (first-layer weight templates)
  - error_gallery.png     (highest-confidence mistakes)

Example::

    python -m scripts.test --checkpoint runs/baseline/best.pkl \\
        --out-dir runs/baseline/eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import class_activation_stats, save_activation_analysis
from src.data import load_eurosat
from src.evaluate import (
    accuracy, balanced_accuracy, batched_predict, batched_predict_tta,
    cohen_kappa, confusion_matrix, format_per_class_table, macro_f1,
    top_confusion_pairs, top_k_accuracy,
)
from src.model import MLP
from src.visualize import (
    plot_activation_heatmap, plot_confusion_matrix, plot_error_gallery,
    plot_first_layer_weights, plot_first_layer_weights_per_channel,
    plot_neuron_class_preferences, plot_per_pair_error_gallery,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="EuroSAT_RGB")
    p.add_argument("--cache", default="cache/eurosat_raw.npz")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", default="runs/baseline/eval")
    p.add_argument("--seed", type=int, default=42,
                   help="must match the seed used at training time")
    p.add_argument("--num-filters", type=int, default=64)
    p.add_argument("--per-channel-filters", type=int, default=16,
                   help="number of filters in the per-channel grayscale plot")
    p.add_argument("--error-top-k", type=int, default=16)
    p.add_argument("--num-confusion-pairs", type=int, default=4,
                   help="how many top off-diagonal cells get a per-pair gallery")
    p.add_argument("--per-pair", type=int, default=8,
                   help="examples per confusion-pair gallery")
    p.add_argument("--top-per-class", type=int, default=1,
                   help="how many class-preferring neurons to plot per class")
    p.add_argument("--neuron-samples", type=int, default=4,
                   help="random class samples shown next to each chosen neuron")
    p.add_argument("--tta", action="store_true",
                   help="average logits over the 8 D4 transforms at test time")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_eurosat(root=args.data_root, cache=args.cache, seed=args.seed)
    print(data.summary())

    print(f"Loading model from {args.checkpoint}")
    model = MLP.load(args.checkpoint)

    if args.tta:
        print("Running D4 test-time augmentation (8-way logit averaging)...")
        preds, logits = batched_predict_tta(model, data.X_test)
    else:
        preds, logits = batched_predict(model, data.X_test)
    cm = confusion_matrix(data.y_test, preds, num_classes=10)

    summary = {
        "checkpoint": str(args.checkpoint),
        "tta": bool(args.tta),
        "test_acc": accuracy(data.y_test, preds),
        "balanced_acc": balanced_accuracy(cm),
        "macro_f1": macro_f1(cm),
        "cohen_kappa": cohen_kappa(cm),
        "top3_acc": top_k_accuracy(logits, data.y_test, k=3),
        "top_confusions": [
            {"true": data.class_names[i], "pred": data.class_names[j], "n": n}
            for i, j, n in top_confusion_pairs(cm, k=5)
        ],
    }
    print("\n== Test metrics ==")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, list):
            print(f"  {k}:")
            for item in v:
                print(f"    {item}")
        else:
            print(f"  {k}: {v}")
    print("\n" + format_per_class_table(cm, data.class_names))

    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_confusion_matrix(cm, data.class_names, out / "confusion_matrix.png")

    W1 = model.fc1.params["W"]
    plot_first_layer_weights(W1, data.std,
                             out / "w1_filters.png", num=args.num_filters)
    plot_first_layer_weights_per_channel(
        W1, data.std, out / "w1_filters_per_channel.png",
        num=args.per_channel_filters,
    )

    # Error gallery + neuron-class preference: ORIGINAL uint8 pixels are
    # already cached on the dataclass — no second decode + 316 MB allocation.
    assert data.X_raw is not None and data.y_raw is not None
    X_raw, y_raw = data.X_raw, data.y_raw

    # ---- Neuron-class preference analysis ---------------------------------
    print("\nComputing per-class fc1 activations on the training set...")
    import numpy as _np
    act_means = class_activation_stats(
        model, data.X_train, data.y_train, num_classes=len(data.class_names),
    )
    save_activation_analysis(
        act_means, data.class_names,
        out / "neuron_analysis.json", top_per_class=args.top_per_class,
    )
    plot_activation_heatmap(
        act_means, out / "neuron_activation_heatmap.png",
        class_names=data.class_names,
        neurons_per_class=args.top_per_class,
    )
    selected = plot_neuron_class_preferences(
        W1=model.fc1.params["W"], std=data.std,
        X_raw_uint8=X_raw, y_raw=y_raw,
        activation_means=act_means,
        save_path=out / "neuron_class_preferences.png",
        class_names=data.class_names,
        top_per_class=args.top_per_class,
        samples_per_neuron=args.neuron_samples,
        rng=_np.random.default_rng(args.seed),
    )
    print(f"  selected {len(selected)} class-preferring neurons; "
          f"see neuron_analysis.json for full per-class numbers")
    assert data.test_idx is not None
    plot_error_gallery(
        X_raw=X_raw, y_true=data.y_test, y_pred=preds, logits=logits,
        test_indices_in_raw=data.test_idx,
        save_path=out / "error_gallery.png",
        class_names=data.class_names, top_k=args.error_top_k,
    )
    pairs = top_confusion_pairs(cm, k=args.num_confusion_pairs)
    pair_files = plot_per_pair_error_gallery(
        X_raw=X_raw, y_true=data.y_test, y_pred=preds, logits=logits,
        test_indices_in_raw=data.test_idx,
        confusion_pairs=pairs,
        save_dir=out / "error_pairs",
        class_names=data.class_names, per_pair=args.per_pair,
    )
    print(f"  per-pair galleries: {len(pair_files)} files in {out / 'error_pairs'}")
    print(f"\nArtifacts written to {out}")


if __name__ == "__main__":
    main()
