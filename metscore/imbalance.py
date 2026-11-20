"""Class imbalance: downsampling of the majority class and class weights.

Both are applied to training data only. Validation and test folds keep their natural
prevalence.
"""

from __future__ import annotations

import numpy as np

from .config import ImbalanceConfig


def downsample(
    x: np.ndarray,
    y: np.ndarray,
    cfg: ImbalanceConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Subsample the majority class down to `majority_to_minority_ratio`.

    Returns the arrays and the indices of the majority samples that were kept, or
    None when nothing was removed.
    """
    x = np.asarray(x)
    y = np.asarray(y).ravel()

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError(f"expected binary labels, found {classes!r}")

    minority_label = classes[np.argmin(counts)]
    majority_label = classes[np.argmax(counts)]
    minority_idx = np.flatnonzero(y == minority_label)
    majority_idx = np.flatnonzero(y == majority_label)

    if not cfg.downsample:
        return x, y, None
    if len(minority_idx) < cfg.min_minority:
        return x, y, None
    if len(majority_idx) / len(minority_idx) <= cfg.majority_to_minority_ratio:
        return x, y, None

    n_keep = min(
        int(len(minority_idx) * cfg.majority_to_minority_ratio), len(majority_idx)
    )
    rng = np.random.default_rng(seed)
    kept = rng.choice(majority_idx, size=n_keep, replace=False)

    selected = np.concatenate([kept, minority_idx])
    rng.shuffle(selected)
    return x[selected], y[selected], np.sort(kept)


def class_weights(y: np.ndarray, cfg: ImbalanceConfig) -> dict[int, float] | None:
    """Inverse-frequency weights of the labels as given, i.e. after downsampling."""
    if not cfg.class_weights_after_downsampling:
        raise ValueError(
            "class weights are computed after downsampling; taking them from the "
            "original distribution would correct for the imbalance twice"
        )

    y = np.asarray(y).ravel()
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    if neg == 0 or pos == 0:
        return None

    total = neg + pos
    return {0: total / (2.0 * neg), 1: total / (2.0 * pos)}


def prepare_training_set(
    x: np.ndarray,
    y: np.ndarray,
    cfg: ImbalanceConfig,
    seed: int,
    majority_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, float] | None, np.ndarray | None]:
    """Downsample, then compute the class weights. This is what the arms call."""
    if majority_indices is not None:
        y_arr = np.asarray(y).ravel()
        minority_idx = np.flatnonzero(y_arr == np.unique(y_arr)[np.argmin(np.bincount(y_arr.astype(int)))])
        selected = np.concatenate([majority_indices, minority_idx])
        x_bal, y_bal, kept = np.asarray(x)[selected], y_arr[selected], majority_indices
    else:
        x_bal, y_bal, kept = downsample(x, y, cfg, seed)

    return x_bal, y_bal, class_weights(y_bal, cfg), kept


def describe(y_before: np.ndarray, y_after: np.ndarray) -> dict:
    """Class counts before and after downsampling, saved with the run."""
    def counts(y):
        y = np.asarray(y).ravel()
        return {"n": int(y.size), "pos": int((y == 1).sum()), "neg": int((y == 0).sum())}

    before, after = counts(y_before), counts(y_after)
    return {
        "before": before,
        "after": after,
        "applied": before["n"] != after["n"],
        "ratio_before": round(before["neg"] / max(before["pos"], 1), 2),
        "ratio_after": round(after["neg"] / max(after["pos"], 1), 2),
    }
