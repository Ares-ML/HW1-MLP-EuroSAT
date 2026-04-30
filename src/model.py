"""Three-layer MLP with configurable hidden sizes, activation, and dropout."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .layers import (
    Dropout,
    Layer,
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
    softmax_cross_entropy,
)

_ACTIVATIONS = {"relu": ReLU, "sigmoid": Sigmoid, "tanh": Tanh}


class MLP:
    """Input -> Linear -> Act -> [Dropout] -> Linear -> Act -> [Dropout] -> Linear.

    Parameters
    ----------
    input_dim, hidden1, hidden2, num_classes : int
        Layer widths for the 12288 -> 512 -> 256 -> 10 reference architecture
        (any integers are accepted).
    activation : {"relu", "sigmoid", "tanh"}
    dropout : float
        Inverted-dropout probability applied after each hidden activation.
    weight_decay : float
        L2 coefficient (applied in ``regularization_loss`` and ``add_reg_grad``).
        Biases are NOT regularized.
    seed : int | None
        Forwarded to ``np.random.default_rng`` for reproducible initialization.
    """

    def __init__(
        self,
        input_dim: int = 64 * 64 * 3,
        hidden1: int = 512,
        hidden2: int = 256,
        num_classes: int = 10,
        activation: str = "relu",
        dropout: float = 0.0,
        weight_decay: float = 0.0,
        label_smoothing: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(_ACTIVATIONS)}")
        self.config = dict(
            input_dim=input_dim,
            hidden1=hidden1,
            hidden2=hidden2,
            num_classes=num_classes,
            activation=activation,
            dropout=dropout,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
        )
        rng = np.random.default_rng(seed)
        init = "he" if activation == "relu" else "xavier"
        Act = _ACTIVATIONS[activation]

        self.fc1 = Linear(input_dim, hidden1, init=init, rng=rng)
        self.act1 = Act()
        self.drop1 = Dropout(dropout, rng=rng)
        self.fc2 = Linear(hidden1, hidden2, init=init, rng=rng)
        self.act2 = Act()
        self.drop2 = Dropout(dropout, rng=rng)
        self.fc3 = Linear(hidden2, num_classes, init="small", rng=rng)

        self.layers: list[Layer] = [
            self.fc1, self.act1, self.drop1,
            self.fc2, self.act2, self.drop2,
            self.fc3,
        ]
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.training = True

    # ---- mode toggles ----------------------------------------------------
    def train(self) -> None:
        self.training = True
        self.drop1.training = True
        self.drop2.training = True

    def eval(self) -> None:
        self.training = False
        self.drop1.training = False
        self.drop2.training = False

    # ---- core forward / backward ----------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss_and_grads(self, x: np.ndarray, y: np.ndarray
                       ) -> tuple[float, np.ndarray]:
        """Return ``(loss, logits)`` and populate ``grads`` on every Linear layer."""
        logits = self.forward(x)
        data_loss, dlogits = softmax_cross_entropy(
            logits, y, label_smoothing=self.label_smoothing
        )
        reg_loss = self.regularization_loss()
        total_loss = data_loss + reg_loss

        dout = dlogits
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        self._add_reg_grad()
        return total_loss, logits

    def predict(self, x: np.ndarray) -> np.ndarray:
        was_training = self.training
        self.eval()
        logits = self.forward(x)
        if was_training:
            self.train()
        return logits.argmax(axis=1)

    # ---- regularization --------------------------------------------------
    def regularization_loss(self) -> float:
        if self.weight_decay == 0.0:
            return 0.0
        l2 = 0.0
        for linear in (self.fc1, self.fc2, self.fc3):
            w = linear.params["W"]
            l2 += float((w * w).sum())
        return 0.5 * self.weight_decay * l2

    def _add_reg_grad(self) -> None:
        if self.weight_decay == 0.0:
            return
        for linear in (self.fc1, self.fc2, self.fc3):
            linear.grads["W"] += self.weight_decay * linear.params["W"]

    # ---- optimizer introspection ----------------------------------------
    def iter_params(self):
        """Yield ``(name, layer, key)`` for every trainable parameter."""
        for idx, linear in enumerate((self.fc1, self.fc2, self.fc3), start=1):
            yield f"fc{idx}.W", linear, "W"
            yield f"fc{idx}.b", linear, "b"

    # ---- persistence -----------------------------------------------------
    def state_dict(self) -> dict:
        state = {"config": dict(self.config), "params": {}}
        for name, layer, key in self.iter_params():
            state["params"][name] = layer.params[key].copy()
        return state

    def load_state_dict(self, state: dict) -> None:
        for name, layer, key in self.iter_params():
            layer.params[key] = state["params"][name].astype(np.float32, copy=True)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.state_dict(), f)

    @classmethod
    def load(cls, path: str | Path) -> "MLP":
        with Path(path).open("rb") as f:
            state = pickle.load(f)
        model = cls(**state["config"])
        model.load_state_dict(state)
        return model
