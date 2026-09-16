"""
Metrics, computed by one implementation for every arm.

The operating point maximizes Youden's index on held-out validation data and is
frozen before the test fold is scored. Test folds are never resampled, so every
metric refers to the natural prevalence of its cohort.

Reported: AUROC, PR-AUC, sensitivity, specificity, PPV, NPV, F1, balanced
accuracy, Brier score and expected calibration error, optionally within strata
(site, sex, age band).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from .config import ThresholdConfig

@dataclass(frozen=True)
class OperatingPoint:
    """A threshold, and the data it was chosen on.

    ``selected_on`` is carried into the results so a reader can verify the
    threshold never came from the data it is scoring.
    """

    threshold: float
    criterion: str
    selected_on: str
    youden_index: float = float("nan")

def select_threshold(
    y_val: np.ndarray,
    p_val: np.ndarray,
    cfg: ThresholdConfig | None = None,
) -> OperatingPoint:
    """Choose the operating point on validation data.

    ThresholdConfig refuses to be constructed with ``source="test"``, so this
    function cannot be pointed at a test fold by configuration. Callers must
    still pass validation predictions; the arms assert this.
    """
    cfg = cfg or ThresholdConfig()
    y_val = np.asarray(y_val).ravel()
    p_val = np.asarray(p_val).ravel()

    if len(np.unique(y_val)) < 2 or not np.all(np.isfinite(p_val)):
        # One class present, or a diverged model: no threshold is meaningful.
        return OperatingPoint(0.5, cfg.criterion, cfg.source)

    grid = np.linspace(cfg.min_threshold, cfg.max_threshold, cfg.resolution)
    pos, neg = y_val == 1, y_val == 0
    n_pos, n_neg = pos.sum(), neg.sum()

    predicted = p_val[None, :] >= grid[:, None]
    sensitivity = (predicted & pos[None, :]).sum(axis=1) / n_pos
    specificity = (~predicted & neg[None, :]).sum(axis=1) / n_neg
    youden = sensitivity + specificity - 1.0

    best = int(np.argmax(youden))
    return OperatingPoint(
        threshold=float(grid[best]),
        criterion=cfg.criterion,
        selected_on=cfg.source,
        youden_index=float(youden[best]),
    )

def calibration_curve(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10
) -> dict[str, list[float]]:
    """Reliability curve using quantile bins.

    Quantile rather than uniform bins because predicted probabilities on a 2%
    prevalence problem pile up near zero, leaving uniform bins empty.
    """
    y = np.asarray(y).ravel()
    p = np.asarray(p).ravel()

    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return {"predicted": [], "observed": [], "count": []}

    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    predicted, observed, count = [], [], []
    for b in range(len(edges) - 1):
        mask = idx == b
        if not mask.any():
            continue
        predicted.append(float(p[mask].mean()))
        observed.append(float(y[mask].mean()))
        count.append(int(mask.sum()))

    return {"predicted": predicted, "observed": observed, "count": count}

def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    curve = calibration_curve(y, p, n_bins)
    if not curve["count"]:
        return float("nan")
    total = sum(curve["count"])
    return float(
        sum(
            n * abs(obs - pred)
            for pred, obs, n in zip(curve["predicted"], curve["observed"], curve["count"])
        )
        / total
    )

@dataclass
class Metrics:
    """Everything reported for one (model, evaluation set) pair."""

    n: int
    n_positive: int
    prevalence: float
    threshold: float
    threshold_selected_on: str
    auroc: float
    pr_auc: float
    brier: float
    ece: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    f1: float
    balanced_accuracy: float
    tp: int
    fp: int
    tn: int
    fn: int
    calibration: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        d = dict(self.__dict__)
        d.pop("calibration", None)
        return d

def diverged(y: np.ndarray, operating_point: OperatingPoint) -> Metrics:
    """Metrics for a model whose predictions are not finite.

    Recorded rather than raised so a differential-privacy sweep reports where
    utility collapses instead of losing the run. `n` and `prevalence` stay real
    so the row still says which evaluation it belongs to.
    """
    y = np.asarray(y).ravel().astype(int)
    nan = float("nan")
    return Metrics(
        n=int(y.size), n_positive=int(y.sum()), prevalence=float(y.mean()),
        threshold=operating_point.threshold,
        threshold_selected_on=operating_point.selected_on,
        auroc=nan, pr_auc=nan, brier=nan, ece=nan,
        sensitivity=nan, specificity=nan, ppv=nan, npv=nan, f1=nan,
        balanced_accuracy=nan, tp=0, fp=0, tn=0, fn=0,
    )

def evaluate(
    y: np.ndarray,
    p: np.ndarray,
    operating_point: OperatingPoint,
    n_bins: int = 10,
) -> Metrics:
    """Score predictions at a threshold that was frozen elsewhere.

    ``y`` must retain the natural prevalence of its cohort, test folds are
    never downsampled, so PPV means what a clinical reader expects it to mean.
    """
    y = np.asarray(y).ravel().astype(int)
    p = np.asarray(p).ravel().astype(float)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y={y.shape} p={p.shape}")

    # A model can genuinely diverge (enough differential-privacy noise will do
    # it), and the record of that is a row of NaNs on the privacy-utility curve
    # rather than an exception that discards the rest of the sweep.
    if not np.all(np.isfinite(p)):
        return diverged(y, operating_point)

    both_classes = len(np.unique(y)) == 2
    predicted = (p >= operating_point.threshold).astype(int)

    tp = int(((y == 1) & (predicted == 1)).sum())
    tn = int(((y == 0) & (predicted == 0)).sum())
    fp = int(((y == 0) & (predicted == 1)).sum())
    fn = int(((y == 1) & (predicted == 0)).sum())

    def safe(num: int, den: int) -> float:
        return float(num / den) if den else float("nan")

    sensitivity = safe(tp, tp + fn)
    specificity = safe(tn, tn + fp)
    ppv = safe(tp, tp + fp)
    npv = safe(tn, tn + fn)
    f1 = (
        2 * ppv * sensitivity / (ppv + sensitivity)
        if ppv == ppv and sensitivity == sensitivity and (ppv + sensitivity) > 0
        else float("nan")
    )

    return Metrics(
        n=int(y.size),
        n_positive=int(y.sum()),
        prevalence=float(y.mean()),
        threshold=operating_point.threshold,
        threshold_selected_on=operating_point.selected_on,
        auroc=float(roc_auc_score(y, p)) if both_classes else float("nan"),
        pr_auc=float(average_precision_score(y, p)) if both_classes else float("nan"),
        brier=float(brier_score_loss(y, p)) if both_classes else float("nan"),
        ece=expected_calibration_error(y, p, n_bins) if both_classes else float("nan"),
        sensitivity=sensitivity,
        specificity=specificity,
        ppv=ppv,
        npv=npv,
        f1=f1,
        balanced_accuracy=(sensitivity + specificity) / 2,
        tp=tp, fp=fp, tn=tn, fn=fn,
        calibration=calibration_curve(y, p, n_bins) if both_classes else {},
    )

def evaluate_by_stratum(
    y: np.ndarray,
    p: np.ndarray,
    strata: np.ndarray,
    operating_point: OperatingPoint,
    min_n: int = 20,
) -> dict[str, Metrics]:
    """    Strata smaller than ``min_n`` are skipped rather than reported with
        uninterpretable intervals; the caller records which ones were dropped.
    """
    y = np.asarray(y).ravel()
    p = np.asarray(p).ravel()
    strata = np.asarray(strata).ravel()

    out = {}
    for level in np.unique(strata[strata == strata]):  # drops NaN
        mask = strata == level
        if mask.sum() < min_n or len(np.unique(y[mask])) < 2:
            continue
        out[str(level)] = evaluate(y[mask], p[mask], operating_point)
    return out

def age_bands(age: np.ndarray) -> np.ndarray:
    """The age bands used for stratified reporting."""
    edges = [18, 25, 35, 45, 55, 65, 70, 80, 105]
    labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-69", "70-79", "80+"]
    age = np.asarray(age, dtype=float)
    out = np.full(age.shape, np.nan, dtype=object)
    for lo, hi, label in zip(edges[:-1], edges[1:], labels):
        out[(age >= lo) & (age < hi)] = label
    return out
