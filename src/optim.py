"""Optimizers and learning-rate schedulers (NumPy only)."""

from __future__ import annotations

import math

import numpy as np


class SGD:
    """SGD with optional Nesterov-free classical momentum.

    Works directly on an ``MLP``: every call to :meth:`step` iterates
    ``model.iter_params()`` and applies ``v = mu*v - lr*grad; p += v``.
    """

    def __init__(self, model, lr: float = 1e-2, momentum: float = 0.9) -> None:
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.velocities: dict[str, np.ndarray] = {}
        for name, layer, key in model.iter_params():
            self.velocities[name] = np.zeros_like(layer.params[key])

    def step(self) -> None:
        lr, mu = self.lr, self.momentum
        for name, layer, key in self.model.iter_params():
            v = self.velocities[name]
            g = layer.grads[key]
            v *= mu
            v -= lr * g
            layer.params[key] += v

    def update_weight_to_grad_ratio(self) -> float:
        """Diagnostic: median ``‖lr·dW‖ / ‖W‖`` across weight matrices.

        Target ~1e-3 per cs231n. Only weight matrices are considered (biases
        often have tiny norms and dominate the ratio otherwise).
        """
        ratios = []
        for name, layer, key in self.model.iter_params():
            if key != "W":
                continue
            w_norm = np.linalg.norm(layer.params[key])
            upd_norm = self.lr * np.linalg.norm(layer.grads[key])
            if w_norm > 0:
                ratios.append(upd_norm / w_norm)
        return float(np.median(ratios)) if ratios else 0.0


class CosineLR:
    """Cosine anneal from ``lr_max`` down to ``lr_min`` over ``total_steps``."""

    def __init__(self, optimizer: SGD, lr_max: float, lr_min: float,
                 total_steps: int) -> None:
        self.opt = optimizer
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.total_steps = max(1, total_steps)
        self.step_idx = 0
        self.opt.lr = lr_max

    def step(self) -> None:
        self.step_idx += 1
        t = min(self.step_idx / self.total_steps, 1.0)
        self.opt.lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1.0 + math.cos(math.pi * t)
        )


class StepLR:
    """Multiply ``lr`` by ``gamma`` every ``step_size`` epochs."""

    def __init__(self, optimizer: SGD, step_size: int, gamma: float = 0.5) -> None:
        self.opt = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.epoch = 0

    def step(self) -> None:
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            self.opt.lr *= self.gamma


def build_scheduler(name: str, optimizer: SGD, **kwargs):
    """Factory: ``name in {"cosine", "step", "none"}``."""
    name = name.lower()
    if name == "cosine":
        return CosineLR(optimizer, **kwargs)
    if name == "step":
        return StepLR(optimizer, **kwargs)
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler {name!r}")
