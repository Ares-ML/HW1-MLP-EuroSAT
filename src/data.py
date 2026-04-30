"""EuroSAT RGB dataset loading, splitting, normalization, and batching.

The dataset folder structure is assumed to be::

    EuroSAT_RGB/
        AnnualCrop/AnnualCrop_1.jpg
        Forest/Forest_1.jpg
        ...

Loading all 27,000 images once into a ``uint8`` tensor (~317 MB) is fast and
keeps the rest of the pipeline simple. We cache the decoded tensor to a
``.npz`` file the first time so re-runs skip JPEG decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

CLASS_NAMES: tuple[str, ...] = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)

IMAGE_SIZE = 64
NUM_CLASSES = len(CLASS_NAMES)
INPUT_DIM = IMAGE_SIZE * IMAGE_SIZE * 3


@dataclass
class EuroSAT:
    """A loaded + preprocessed EuroSAT dataset ready for training."""

    X_train: np.ndarray  # (N, 12288) float32, z-scored
    y_train: np.ndarray  # (N,) int64
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    mean: np.ndarray     # (3,) float32, training-set per-channel mean on [0,1]
    std: np.ndarray      # (3,) float32
    train_idx: np.ndarray | None = None  # original row ids in the raw uint8 tensor
    val_idx: np.ndarray | None = None
    test_idx: np.ndarray | None = None
    X_raw: np.ndarray | None = None      # (N_total, 64, 64, 3) uint8 — original pixels
    y_raw: np.ndarray | None = None      # (N_total,) labels matching X_raw
    class_names: tuple[str, ...] = CLASS_NAMES

    def summary(self) -> str:
        return (
            f"train: {self.X_train.shape}  val: {self.X_val.shape}  "
            f"test: {self.X_test.shape}\n"
            f"mean: {self.mean.tolist()}  std: {self.std.tolist()}"
        )


def _load_raw(root: Path, cache: Path | None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X_uint8 (N,64,64,3), y_int64 (N,))``, caching the decoded tensor."""
    if cache is not None and cache.exists():
        with np.load(cache) as data:
            return data["X"], data["y"]

    xs: list[np.ndarray] = []
    ys: list[int] = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = root / class_name
        files = sorted(class_dir.glob("*.jpg"))
        if not files:
            raise FileNotFoundError(f"No jpg files found under {class_dir}")
        for fp in files:
            with Image.open(fp) as img:
                arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
            if arr.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
                raise ValueError(f"Unexpected shape {arr.shape} at {fp}")
            xs.append(arr)
            ys.append(class_idx)

    X = np.stack(xs, axis=0)
    y = np.asarray(ys, dtype=np.int64)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, y=y)
    return X, y


def _stratified_split(y: np.ndarray, ratios: tuple[float, float, float],
                      seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified index split. ``ratios`` must sum to 1."""
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("split ratios must sum to 1")
    rng = np.random.default_rng(seed)
    n = y.shape[0]
    train_idx, val_idx, test_idx = [], [], []
    for c in range(NUM_CLASSES):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        n_c = idx_c.shape[0]
        n_train = int(round(n_c * ratios[0]))
        n_val = int(round(n_c * ratios[1]))
        # Remaining goes to test so the three segments exactly cover idx_c.
        train_idx.append(idx_c[:n_train])
        val_idx.append(idx_c[n_train:n_train + n_val])
        test_idx.append(idx_c[n_train + n_val:])
    tr = np.concatenate(train_idx); rng.shuffle(tr)
    va = np.concatenate(val_idx);   rng.shuffle(va)
    te = np.concatenate(test_idx);  rng.shuffle(te)
    assert tr.size + va.size + te.size == n
    return tr, va, te


def load_eurosat(
    root: str | Path = "EuroSAT_RGB",
    cache: str | Path | None = "cache/eurosat_raw.npz",
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> EuroSAT:
    """Load the dataset, do a stratified split, and z-score on training stats."""
    root = Path(root)
    cache_path = Path(cache) if cache is not None else None
    X_u8, y = _load_raw(root, cache_path)

    tr_idx, va_idx, te_idx = _stratified_split(y, ratios, seed)

    # Convert to float32 in [0, 1] and flatten. Doing this per-split keeps
    # peak memory modest (we never hold two copies of the full float tensor).
    def flatten_norm(indices: np.ndarray, mean: np.ndarray, std: np.ndarray
                     ) -> np.ndarray:
        part = X_u8[indices].astype(np.float32) / 255.0
        part -= mean
        part /= std
        return part.reshape(part.shape[0], -1)

    # Compute statistics on TRAINING pixels only (per-channel).
    # Reduce in float64 — float32 sums over ~88M values saturate at ~3.5e7
    # because 1 ULP > the magnitude of new values being added, collapsing all
    # three channel means to the same wrong value.
    train_pixels = X_u8[tr_idx].astype(np.float32) / 255.0
    mean = train_pixels.mean(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    std = train_pixels.std(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)  # guard against divide-by-zero
    del train_pixels

    X_train = flatten_norm(tr_idx, mean, std)
    X_val = flatten_norm(va_idx, mean, std)
    X_test = flatten_norm(te_idx, mean, std)

    return EuroSAT(
        X_train=X_train, y_train=y[tr_idx].astype(np.int64),
        X_val=X_val,     y_val=y[va_idx].astype(np.int64),
        X_test=X_test,   y_test=y[te_idx].astype(np.int64),
        mean=mean, std=std,
        train_idx=tr_idx, val_idx=va_idx, test_idx=te_idx,
        X_raw=X_u8, y_raw=y,
    )


def load_raw_images(
    root: str | Path = "EuroSAT_RGB",
    cache: str | Path | None = "cache/eurosat_raw.npz",
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper returning the uint8 image tensor + labels.

    Useful for the error-gallery visualizer which needs the ORIGINAL pixel
    values (not the z-scored tensor held by ``EuroSAT``).
    """
    return _load_raw(Path(root), Path(cache) if cache is not None else None)


def augment_batch(x_flat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random D4-group transform per sample.

    EuroSAT patches have no canonical orientation (a 640m x 640m satellite
    tile is equally valid flipped or rotated by 90°), so D4 is a "free" 8×
    data multiplier. We pick a random element of D4 INDEPENDENTLY for each
    sample and apply each variant to the matching subset (vectorized 8 ops
    per call instead of N), which is ~8× richer than per-batch sharing while
    only ~1.4× slower per epoch.

    D4 = {identity, R90, R180, R270, H, H·R90, H·R180, H·R270} where H is
    horizontal flip. We index them as ``t = flip*4 + k``.
    """
    n = x_flat.shape[0]
    x = x_flat.reshape(n, IMAGE_SIZE, IMAGE_SIZE, 3)
    transforms = rng.integers(0, 8, size=n)
    out = np.empty_like(x)
    for t in range(8):
        mask = transforms == t
        if not mask.any():
            continue
        sub = x[mask]
        if t >= 4:
            sub = sub[:, :, ::-1, :]               # horizontal flip
        k = t % 4
        if k:
            sub = np.rot90(sub, k=k, axes=(1, 2))
        out[mask] = np.ascontiguousarray(sub)
    return out.reshape(n, -1)


def iter_minibatches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(X_batch, y_batch)`` tuples, optionally shuffling each epoch."""
    n = X.shape[0]
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, n, batch_size):
        sel = indices[start:start + batch_size]
        yield X[sel], y[sel]


def num_batches(n_samples: int, batch_size: int) -> int:
    return (n_samples + batch_size - 1) // batch_size
