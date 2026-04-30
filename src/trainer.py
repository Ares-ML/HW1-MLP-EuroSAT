"""Training loop with validation-based best-weight saving and early stopping."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data import EuroSAT, augment_batch, iter_minibatches
from .evaluate import accuracy, batched_predict
from .layers import softmax_cross_entropy
from .model import MLP
from .optim import SGD, build_scheduler


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-2
    lr_min: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0        # ε for soft CE; 0 disables
    scheduler: str = "cosine"           # "cosine" | "step" | "none"
    step_size: int = 10                 # only used if scheduler=="step"
    gamma: float = 0.5
    patience: int = 10                  # early stop if val loss stalls this many epochs
    eval_batch_size: int = 512
    seed: int = 42
    log_every: int = 100                # print batch loss every N steps
    verbose: bool = True
    augment: bool = True                # D4-group augmentation on training batches


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)   # per epoch (avg)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    best_val_acc: float = 0.0
    best_epoch: int = -1

    def to_dict(self) -> dict:
        return {
            "train_loss": self.train_loss, "train_acc": self.train_acc,
            "val_loss": self.val_loss, "val_acc": self.val_acc,
            "lr": self.lr,
            "best_val_acc": self.best_val_acc, "best_epoch": self.best_epoch,
        }


def _evaluate(model: MLP, X: np.ndarray, y: np.ndarray, batch_size: int
              ) -> tuple[float, float]:
    """Average loss + accuracy (data term only, no regularization)."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    n = X.shape[0]
    for start in range(0, n, batch_size):
        xb = X[start:start + batch_size]
        yb = y[start:start + batch_size]
        logits = model.forward(xb)
        loss, _ = softmax_cross_entropy(logits, yb)
        total_loss += float(loss) * xb.shape[0]
        total_correct += int((logits.argmax(axis=1) == yb).sum())
    if was_training:
        model.train()
    return total_loss / n, total_correct / n


def train(
    model: MLP,
    data: EuroSAT,
    cfg: TrainConfig,
    save_path: str | Path | None = "checkpoints/best.pkl",
    history_path: str | Path | None = "checkpoints/history.json",
) -> TrainHistory:
    """Train ``model`` on ``data`` using ``cfg``. Returns the recorded history.

    The best model (by validation accuracy) is saved to ``save_path`` and also
    written back into ``model`` at the end so the caller has the best weights.
    """
    model.weight_decay = cfg.weight_decay  # honor trainer-level WD override
    model.label_smoothing = cfg.label_smoothing
    optimizer = SGD(model, lr=cfg.lr, momentum=cfg.momentum)
    n_train = data.X_train.shape[0]
    steps_per_epoch = (n_train + cfg.batch_size - 1) // cfg.batch_size

    if cfg.scheduler == "cosine":
        scheduler = build_scheduler(
            "cosine", optimizer,
            lr_max=cfg.lr, lr_min=cfg.lr_min,
            total_steps=cfg.epochs * steps_per_epoch,
        )
        per_step = True
    elif cfg.scheduler == "step":
        scheduler = build_scheduler("step", optimizer,
                                    step_size=cfg.step_size, gamma=cfg.gamma)
        per_step = False
    else:
        scheduler = None
        per_step = False

    history = TrainHistory()
    best_state: dict | None = None
    epochs_since_improve = 0
    rng = np.random.default_rng(cfg.seed)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_correct = 0
        running_seen = 0
        for step_i, (xb, yb) in enumerate(iter_minibatches(
            data.X_train, data.y_train, cfg.batch_size, shuffle=True, rng=rng
        )):
            if cfg.augment:
                xb = augment_batch(xb, rng)
            loss, logits = model.loss_and_grads(xb, yb)
            optimizer.step()
            if per_step and scheduler is not None:
                scheduler.step()

            running_loss += loss * xb.shape[0]
            running_correct += int((logits.argmax(axis=1) == yb).sum())
            running_seen += xb.shape[0]

            if cfg.verbose and cfg.log_every > 0 and step_i % cfg.log_every == 0:
                print(
                    f"  ep{epoch:02d} step {step_i:04d}/{steps_per_epoch} "
                    f"loss={loss:.4f} lr={optimizer.lr:.2e}"
                )

        if not per_step and scheduler is not None:
            scheduler.step()

        train_loss = running_loss / running_seen
        train_acc = running_correct / running_seen
        val_loss, val_acc = _evaluate(model, data.X_val, data.y_val,
                                      cfg.eval_batch_size)

        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)
        history.lr.append(optimizer.lr)

        dt = time.time() - t0
        if cfg.verbose:
            print(
                f"epoch {epoch:02d}/{cfg.epochs}  "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
                f"lr={optimizer.lr:.2e}  ({dt:.1f}s)"
            )

        if val_acc > history.best_val_acc:
            history.best_val_acc = val_acc
            history.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            if save_path is not None:
                model.save(save_path)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if cfg.patience > 0 and epochs_since_improve >= cfg.patience:
                if cfg.verbose:
                    print(f"early stopping at epoch {epoch} "
                          f"(no val improvement for {cfg.patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if history_path is not None:
        p = Path(history_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(history.to_dict(), f, indent=2)

    return history


def final_report(model: MLP, data: EuroSAT, batch_size: int = 512) -> dict:
    """Compute test accuracy + predictions for the best weights currently in model."""
    preds, logits = batched_predict(model, data.X_test, batch_size)
    return {
        "test_acc": accuracy(data.y_test, preds),
        "preds": preds,
        "logits": logits,
    }
