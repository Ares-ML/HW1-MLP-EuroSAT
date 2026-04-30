"""Train a baseline MLP on EuroSAT and write checkpoint + history + curves.

Run from the project root::

    python -m scripts.train --epochs 50 --lr 1e-2 --weight-decay 1e-4

All outputs (checkpoint, history JSON, plots) go under ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make ``src`` importable when launched as a file (python scripts/train.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_eurosat
from src.evaluate import (
    accuracy, cohen_kappa, confusion_matrix, format_per_class_table,
    macro_f1, balanced_accuracy,
)
from src.model import MLP
from src.trainer import TrainConfig, final_report, train
from src.utils import set_seed
from src.visualize import plot_confusion_matrix, plot_training_curves


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="EuroSAT_RGB")
    p.add_argument("--cache", default="cache/eurosat_raw.npz")
    p.add_argument("--out-dir", default="runs/baseline")
    p.add_argument("--hidden1", type=int, default=512)
    p.add_argument("--hidden2", type=int, default=256)
    p.add_argument("--activation", choices=["relu", "sigmoid", "tanh"], default="relu")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--lr-min", type=float, default=5e-5)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--label-smoothing", type=float, default=0.1,
                   help="ε for soft cross-entropy; 0 disables")
    p.add_argument("--scheduler", choices=["cosine", "step", "none"], default="cosine")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42,
                   help="model init + shuffling seed (and the data-split seed "
                        "unless --data-seed is given)")
    p.add_argument("--data-seed", type=int, default=None,
                   help="seed used ONLY for the stratified train/val/test "
                        "split. Defaults to --seed for backwards compatibility. "
                        "Set this to a fixed value across runs (e.g. 42) when "
                        "you want multiple init seeds to share the same test "
                        "set — necessary for honest ensemble evaluation.")
    p.add_argument("--augment", dest="augment", action="store_true",
                   help="enable D4-group training augmentation (default: on)")
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.set_defaults(augment=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    data_seed = args.data_seed if args.data_seed is not None else args.seed
    print(f"Loading EuroSAT (data split seed={data_seed}, init seed={args.seed})...")
    data = load_eurosat(root=args.data_root, cache=args.cache, seed=data_seed)
    print(data.summary())

    model = MLP(
        input_dim=data.X_train.shape[1],
        hidden1=args.hidden1, hidden2=args.hidden2,
        activation=args.activation, dropout=args.dropout,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing, seed=args.seed,
    )
    cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, lr_min=args.lr_min, momentum=args.momentum,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        scheduler=args.scheduler,
        patience=args.patience, seed=args.seed, augment=args.augment,
    )

    ckpt_path = out / "best.pkl"
    hist_path = out / "history.json"
    history = train(model, data, cfg, save_path=ckpt_path, history_path=hist_path)
    print(f"\nBest val acc: {history.best_val_acc:.4f} @ epoch {history.best_epoch}")

    # Evaluate best weights on the test split.
    report = final_report(model, data)
    preds = report["preds"]
    cm = confusion_matrix(data.y_test, preds, num_classes=10)

    summary = {
        "test_acc": accuracy(data.y_test, preds),
        "balanced_acc": balanced_accuracy(cm),
        "macro_f1": macro_f1(cm),
        "cohen_kappa": cohen_kappa(cm),
        "best_val_acc": history.best_val_acc,
        "best_epoch": history.best_epoch,
    }
    print("\n== Test metrics ==")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\n" + format_per_class_table(cm, data.class_names))

    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_training_curves(history.to_dict(), out / "curves.png")
    plot_confusion_matrix(cm, data.class_names, out / "confusion_matrix.png")
    print(f"\nArtifacts written to {out}")


if __name__ == "__main__":
    main()
