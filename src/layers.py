"""Layer primitives with forward / backward APIs.

Every layer stores a ``cache`` dict during ``forward`` so ``backward`` can run
without recomputing anything. Parameter layers additionally expose ``params``
and ``grads`` dicts with matching keys so optimizers can iterate generically.
"""

from __future__ import annotations

import numpy as np


class Layer:
    """Base class. Non-parametric layers can leave ``params`` / ``grads`` empty."""

    def __init__(self) -> None:
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}
        self.cache: dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class Linear(Layer):
    """Affine transform ``y = x @ W + b`` with He or Xavier init."""

    def __init__(self, in_dim: int, out_dim: int, init: str = "he",
                 rng: np.random.Generator | None = None) -> None:
        super().__init__()
        rng = rng if rng is not None else np.random.default_rng()
        if init == "he":
            scale = np.sqrt(2.0 / in_dim)
        elif init == "xavier":
            scale = np.sqrt(1.0 / in_dim)
        elif init == "small":
            # Common for classifier heads: keep initial logits near zero so
            # CE starts close to log(K) and optimization is stable.
            scale = 1e-2
        else:
            raise ValueError(f"Unknown init {init!r}")
        self.params["W"] = (rng.standard_normal((in_dim, out_dim)) * scale).astype(np.float32)
        self.params["b"] = np.zeros(out_dim, dtype=np.float32)
        self.grads["W"] = np.zeros_like(self.params["W"])
        self.grads["b"] = np.zeros_like(self.params["b"])

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache["x"] = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self.cache["x"]
        self.grads["W"] = x.T @ dout
        self.grads["b"] = dout.sum(axis=0)
        return dout @ self.params["W"].T


class ReLU(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache["mask"] = x > 0
        return x * self.cache["mask"]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.cache["mask"]


class Sigmoid(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Stable sigmoid: split positive and negative cases to avoid exp overflow.
        out = np.empty_like(x)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_x = np.exp(x[~pos])
        out[~pos] = exp_x / (1.0 + exp_x)
        self.cache["out"] = out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        out = self.cache["out"]
        return dout * out * (1.0 - out)


class Tanh(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        out = np.tanh(x)
        self.cache["out"] = out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        out = self.cache["out"]
        return dout * (1.0 - out * out)


class Dropout(Layer):
    """Inverted dropout: scaling applied at train time so inference is a no-op."""

    def __init__(self, p: float = 0.0, rng: np.random.Generator | None = None) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout p must be in [0, 1)")
        self.p = p
        self.rng = rng if rng is not None else np.random.default_rng()
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0.0:
            self.cache["mask"] = None
            return x
        mask = (self.rng.random(x.shape) >= self.p).astype(x.dtype) / (1.0 - self.p)
        self.cache["mask"] = mask
        return x * mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        mask = self.cache["mask"]
        return dout if mask is None else dout * mask


def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray,
                          label_smoothing: float = 0.0
                          ) -> tuple[float, np.ndarray]:
    """Numerically stable softmax + cross-entropy returning ``(loss, dlogits)``.

    ``logits``: (N, K). ``y``: (N,) int class ids. The returned gradient is
    already averaged over the batch, so callers just add regularization terms.

    ``label_smoothing`` ε ∈ [0, 1) replaces the one-hot target with
    ``(1−ε)·onehot + ε/K``. Gradient stays simple: ``(softmax − target)/N``.
    """
    n, k = logits.shape
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    log_probs = shifted - log_sum_exp
    probs = np.exp(log_probs)

    if label_smoothing == 0.0:
        loss = -log_probs[np.arange(n), y].mean()
        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n
        return float(loss), dlogits

    eps = float(label_smoothing)
    target = np.full((n, k), eps / k, dtype=log_probs.dtype)
    target[np.arange(n), y] += 1.0 - eps
    loss = -(target * log_probs).sum(axis=1).mean()
    dlogits = (probs - target) / n
    return float(loss), dlogits
