"""Evaluate a logit-averaging ensemble of MLP checkpoints, optionally with TTA.

Example::

    python -m scripts.ensemble \\
        --checkpoints runs/v3b_seed42/best.pkl runs/v3b_seed43/best.pkl \\
                      runs/v3b_seed44/best.pkl \\
        --tta --out-dir runs/v3_ensemble

Outputs match ``scripts/test.py``: summary.json (overall + per-member breakdown),
confusion_matrix.png, w1_filters.png (from the FIRST member, for reference),
w1_filters_per_channel.png, error_gallery.png, error_pairs/ subfolder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import class_activation_stats, save_activation_analysis
from src.data import load_eurosat
from src.ensemble import predict_ensemble
from src.evaluate import (
    accuracy, balanced_accuracy, cohen_kappa, confusion_matrix,
    format_per_class_table, macro_f1, top_confusion_pairs, top_k_accuracy,
)
from src.model import MLP
from src.visualize import (
    plot_activation_heatmap, plot_confusion_matrix, plot_error_gallery,
    plot_first_layer_weights, plot_first_layer_weights_per_channel,
    plot_neuron_class_preferences, plot_per_pair_error_gallery,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="paths to two or more MLP .pkl checkpoints")
    p.add_argument("--data-root", default="EuroSAT_RGB")
    p.add_argument("--cache", default="cache/eurosat_raw.npz")
    p.add_argument("--out-dir", default="runs/ensemble")
    p.add_argument("--seed", type=int, default=42,
                   help="must match the seed used to create the train/test split")
    p.add_argument("--tta", action="store_true",
                   help="apply 8-way D4 TTA to each member before averaging")
    p.add_argument("--num-filters", type=int, default=64)
    p.add_argument("--per-channel-filters", type=int, default=16)
    p.add_argument("--error-top-k", type=int, default=16)
    p.add_argument("--num-confusion-pairs", type=int, default=4)
    p.add_argument("--per-pair", type=int, default=8)
    p.add_argument("--top-per-class", type=int, default=1,
                   help="how many class-preferring neurons to plot per class")
    p.add_argument("--neuron-samples", type=int, default=4,
                   help="random class samples shown next to each chosen neuron")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_eurosat(root=args.data_root, cache=args.cache, seed=args.seed)
    print(data.summary())

    print(f"Ensembling {len(args.checkpoints)} checkpoint(s); TTA={args.tta}")
    for c in args.checkpoints:
        print(f"  - {c}")
    preds, mean_logits, member_logits = predict_ensemble(
        args.checkpoints, data.X_test, use_tta=args.tta,
    )
    cm = confusion_matrix(data.y_test, preds, num_classes=10)

    # Per-member single-model accuracies for the ablation table.
    per_member = []
    for path, logits in zip(args.checkpoints, member_logits):
        m_preds = logits.argmax(axis=1)
        per_member.append({
            "checkpoint": str(path),
            "test_acc": accuracy(data.y_test, m_preds),
        })

    summary = {
        "checkpoints": [str(p) for p in args.checkpoints],
        "num_members": len(args.checkpoints),
        "tta": bool(args.tta),
        "test_acc": accuracy(data.y_test, preds),
        "balanced_acc": balanced_accuracy(cm),
        "macro_f1": macro_f1(cm),
        "cohen_kappa": cohen_kappa(cm),
        "top3_acc": top_k_accuracy(mean_logits, data.y_test, k=3),
        "per_member": per_member,
        "top_confusions": [
            {"true": data.class_names[i], "pred": data.class_names[j], "n": n}
            for i, j, n in top_confusion_pairs(cm, k=5)
        ],
    }

    print("\n== Ensemble metrics ==")
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

    # Use the first member for filter visualizations (members were trained from
    # the same config — the W1 patterns differ only by seed-dependent symmetries).
    ref_model = MLP.load(args.checkpoints[0])
    plot_first_layer_weights(ref_model.fc1.params["W"], data.std,
                             out / "w1_filters.png", num=args.num_filters)
    plot_first_layer_weights_per_channel(
        ref_model.fc1.params["W"], data.std,
        out / "w1_filters_per_channel.png", num=args.per_channel_filters,
    )

    # Original uint8 pixels are cached on the dataclass — no second decode.
    assert data.X_raw is not None and data.y_raw is not None
    X_raw, y_raw = data.X_raw, data.y_raw

    # ---- Neuron-class preference analysis (uses the first member's fc1) ---
    print("\nComputing per-class fc1 activations on the training set "
          "(first ensemble member)...")
    import numpy as _np
    act_means = class_activation_stats(
        ref_model, data.X_train, data.y_train,
        num_classes=len(data.class_names),
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
        W1=ref_model.fc1.params["W"], std=data.std,
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
        X_raw=X_raw, y_true=data.y_test, y_pred=preds, logits=mean_logits,
        test_indices_in_raw=data.test_idx,
        save_path=out / "error_gallery.png",
        class_names=data.class_names, top_k=args.error_top_k,
    )
    pairs = top_confusion_pairs(cm, k=args.num_confusion_pairs)
    pair_files = plot_per_pair_error_gallery(
        X_raw=X_raw, y_true=data.y_test, y_pred=preds, logits=mean_logits,
        test_indices_in_raw=data.test_idx,
        confusion_pairs=pairs,
        save_dir=out / "error_pairs",
        class_names=data.class_names, per_pair=args.per_pair,
    )
    print(f"  per-pair galleries: {len(pair_files)} files in {out / 'error_pairs'}")
    print(f"\nArtifacts written to {out}")


if __name__ == "__main__":
    main()
