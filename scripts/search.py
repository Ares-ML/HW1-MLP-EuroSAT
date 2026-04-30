"""Random hyperparameter search over learning rate, hidden sizes, weight decay.

Each trial runs a short training (default 10 epochs) with early stopping and
records its best validation accuracy. After all trials finish the best
configuration is retrained for the full schedule. Results are logged to
``search_results.json`` (per-trial) and ``best_config.json`` (final pick).

Example::

    python -m scripts.search --n-trials 20 --epochs 10 --final-epochs 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_eurosat
from src.evaluate import accuracy, batched_predict, confusion_matrix
from src.model import MLP
from src.trainer import TrainConfig, train
from src.utils import set_seed
from src.visualize import plot_confusion_matrix, plot_training_curves


def log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    """Draw from log-uniform(lo, hi) — essential for LR / WD search."""
    return float(10.0 ** rng.uniform(np.log10(lo), np.log10(hi)))


def sample_config(rng: np.random.Generator) -> dict:
    """Ranges narrowed twice — v3 zooms in around v2 search winner
    ``(256, 128, lr=7.6e-3, wd=2.4e-3, drop=0.1, bs=128, relu)``.

    Top-5 of v2 all clustered in hidden ∈ {256,512}×{128,256}, lr ∈ [3e-3, 2e-2],
    wd ∈ [5e-4, 3e-3]. tanh trials all fell outside top-5; bs=256 trials too. We
    drop those and add finer architecture choices around 256/128.
    """
    return {
        "lr": log_uniform(rng, 4e-3, 1.5e-2),
        "weight_decay": log_uniform(rng, 5e-4, 5e-3),
        "hidden1": int(rng.choice([192, 256, 384])),
        "hidden2": int(rng.choice([96, 128, 192])),
        "batch_size": int(rng.choice([64, 128])),
        "dropout": float(rng.choice([0.05, 0.1, 0.15, 0.2])),
        "activation": "relu",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="EuroSAT_RGB")
    p.add_argument("--cache", default="cache/eurosat_raw.npz")
    p.add_argument("--out-dir", default="runs/search")
    p.add_argument("--n-trials", type=int, default=25)
    p.add_argument("--epochs", type=int, default=30,
                   help="per-trial training budget (was 20 — v2 winner was "
                        "still improving at epoch 19/20, so 30 lets configs "
                        "complete their cosine schedule)")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--final-epochs", type=int, default=120,
                   help="retrain the best config for this many epochs; 0 skips")
    p.add_argument("--final-patience", type=int, default=20)
    p.add_argument("--label-smoothing", type=float, default=0.1,
                   help="ε for soft cross-entropy across all trials + final retrain")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print("Loading EuroSAT once (shared across trials)...")
    data = load_eurosat(root=args.data_root, cache=args.cache, seed=args.seed)
    print(data.summary())

    rng = np.random.default_rng(args.seed)
    results: list[dict] = []
    t_global = time.time()
    for trial_i in range(args.n_trials):
        cfg_sample = sample_config(rng)
        print(f"\n=== trial {trial_i+1}/{args.n_trials} === {cfg_sample}")
        model = MLP(
            input_dim=data.X_train.shape[1],
            hidden1=cfg_sample["hidden1"], hidden2=cfg_sample["hidden2"],
            activation=cfg_sample["activation"],
            dropout=cfg_sample["dropout"],
            weight_decay=cfg_sample["weight_decay"],
            label_smoothing=args.label_smoothing,
            seed=args.seed + trial_i,
        )
        tcfg = TrainConfig(
            epochs=args.epochs,
            batch_size=cfg_sample["batch_size"],
            lr=cfg_sample["lr"], lr_min=cfg_sample["lr"] * 0.01,
            momentum=0.9, weight_decay=cfg_sample["weight_decay"],
            label_smoothing=args.label_smoothing,
            scheduler="cosine", patience=args.patience,
            seed=args.seed + trial_i, verbose=False, log_every=0,
        )
        t0 = time.time()
        hist = train(model, data, tcfg,
                     save_path=None, history_path=None)
        dt = time.time() - t0
        entry = {
            "trial": trial_i,
            "config": cfg_sample,
            "best_val_acc": hist.best_val_acc,
            "best_epoch": hist.best_epoch,
            "epochs_run": len(hist.val_acc),
            "seconds": dt,
        }
        print(f"  -> best_val_acc={hist.best_val_acc:.4f} in {dt:.1f}s")
        results.append(entry)
        with (out / "search_results.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    results.sort(key=lambda r: r["best_val_acc"], reverse=True)
    best = results[0]
    print(f"\nBest trial #{best['trial']} val_acc={best['best_val_acc']:.4f}")
    print(f"  config: {best['config']}")
    print(f"Total search time: {time.time()-t_global:.1f}s")

    with (out / "best_config.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    if args.final_epochs > 0:
        print(f"\nRetraining best config for {args.final_epochs} epochs...")
        final_dir = out / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        cfg_sample = best["config"]
        model = MLP(
            input_dim=data.X_train.shape[1],
            hidden1=cfg_sample["hidden1"], hidden2=cfg_sample["hidden2"],
            activation=cfg_sample["activation"],
            dropout=cfg_sample["dropout"],
            weight_decay=cfg_sample["weight_decay"],
            label_smoothing=args.label_smoothing,
            seed=args.seed,
        )
        tcfg = TrainConfig(
            epochs=args.final_epochs,
            batch_size=cfg_sample["batch_size"],
            lr=cfg_sample["lr"], lr_min=cfg_sample["lr"] * 0.01,
            momentum=0.9, weight_decay=cfg_sample["weight_decay"],
            label_smoothing=args.label_smoothing,
            scheduler="cosine", patience=args.final_patience, seed=args.seed,
        )
        hist = train(model, data, tcfg,
                     save_path=final_dir / "best.pkl",
                     history_path=final_dir / "history.json")
        preds, _ = batched_predict(model, data.X_test)
        cm = confusion_matrix(data.y_test, preds, num_classes=10)
        final_summary = {
            "config": cfg_sample,
            "test_acc": accuracy(data.y_test, preds),
            "best_val_acc": hist.best_val_acc,
            "best_epoch": hist.best_epoch,
        }
        with (final_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(final_summary, f, indent=2)
        plot_training_curves(hist.to_dict(), final_dir / "curves.png")
        plot_confusion_matrix(cm, data.class_names,
                              final_dir / "confusion_matrix.png")
        print(f"Final test acc: {final_summary['test_acc']:.4f}")
        print(f"Artifacts written to {final_dir}")


if __name__ == "__main__":
    main()
