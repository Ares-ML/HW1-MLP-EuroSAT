"""Three pre-training sanity checks recommended in the experiment plan.

1. Initial loss should be ~log(K) ~= 2.302 for K=10 classes (random softmax).
2. Centered finite-difference gradient check, float64, with weight decay and
   dropout disabled. Max relative error < 1e-4 is acceptable for ReLU nets.
3. Overfit a small subset (default 20) to ~zero loss / 100% accuracy.

Run from project root::

    python -m scripts.sanity_check
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_eurosat
from src.layers import softmax_cross_entropy
from src.model import MLP
from src.optim import SGD
from src.utils import gradient_check, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="EuroSAT_RGB")
    p.add_argument("--cache", default="cache/eurosat_raw.npz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overfit-n", type=int, default=20)
    p.add_argument("--overfit-steps", type=int, default=400)
    p.add_argument("--check-checkpoint", type=int, default=8,
                   help="number of finite-difference checks per param array")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("Loading EuroSAT...")
    data = load_eurosat(root=args.data_root, cache=args.cache, seed=args.seed)
    print(data.summary())

    # ---- 1) initial loss ------------------------------------------------
    print("\n[1] Initial-loss check")
    model = MLP(input_dim=data.X_train.shape[1], seed=args.seed)
    model.eval()
    sample = data.X_train[:256]
    sample_y = data.y_train[:256]
    logits = model.forward(sample)
    init_loss, _ = softmax_cross_entropy(logits, sample_y)
    expected = math.log(10)
    print(f"  initial loss: {init_loss:.4f}  expected ≈ log(10) = {expected:.4f}")
    if abs(init_loss - expected) > 0.5:
        print("  WARNING: initial loss is far from log(K). Check init scale.")

    # ---- 2) gradient check ----------------------------------------------
    print("\n[2] Gradient check (float64, WD/dropout off, 3 datapoints)")
    gc_model = MLP(input_dim=data.X_train.shape[1], hidden1=32, hidden2=16,
                   weight_decay=0.0, dropout=0.0, seed=args.seed)
    x_gc = data.X_train[:3].astype(np.float64)
    y_gc = data.y_train[:3]
    results = gradient_check(gc_model, x_gc, y_gc,
                             num_checks=args.check_checkpoint, seed=args.seed)
    worst = 0.0
    for r in results:
        worst = max(worst, r["max_rel_err"])
        print(f"  {r['param']:>10} max_rel_err={r['max_rel_err']:.2e}  "
              f"mean={r['mean_rel_err']:.2e}  ({r['num_checks']} checks)")
    print(f"  worst overall: {worst:.2e}")
    if worst < 1e-7:
        print("  EXCELLENT — gradients agree to numerical precision")
    elif worst < 1e-4:
        print("  OK — within tolerance for ReLU networks")
    else:
        print("  FAIL — likely a backprop bug; investigate before training")

    # ---- 3) overfit a tiny subset ---------------------------------------
    print(f"\n[3] Overfit-{args.overfit_n} check (no regularization)")
    of_model = MLP(input_dim=data.X_train.shape[1], hidden1=64, hidden2=32,
                   weight_decay=0.0, dropout=0.0, seed=args.seed)
    opt = SGD(of_model, lr=1e-2, momentum=0.9)
    xb = data.X_train[:args.overfit_n]
    yb = data.y_train[:args.overfit_n]
    of_model.train()
    for step in range(args.overfit_steps):
        loss, logits = of_model.loss_and_grads(xb, yb)
        opt.step()
        if step % max(args.overfit_steps // 10, 1) == 0:
            acc = float((logits.argmax(axis=1) == yb).mean())
            print(f"  step {step:4d}  loss={loss:.4f}  train_acc={acc:.3f}")
    final_logits = of_model.forward(xb)
    final_loss, _ = softmax_cross_entropy(final_logits, yb)
    final_acc = float((final_logits.argmax(axis=1) == yb).mean())
    print(f"  final loss={final_loss:.4f}  acc={final_acc:.3f}")
    if final_acc >= 0.99 and final_loss < 0.05:
        print("  PASS — model can fit a tiny subset")
    else:
        print("  WARNING — failed to overfit. Check capacity / LR / data dtype")


if __name__ == "__main__":
    main()
