"""
Inference for comparisons between approaches.

  delong_test               correlated AUROCs on the same subjects
  fold_difference_bootstrap mean over folds of the per-fold AUROC difference,
                            with a stratified bootstrap over subjects for the
                            interval and the p value
  holm_bonferroni           family-wise error control across comparisons

Intervals come from resampling subjects rather than folds: repeated
cross-validation folds share their subjects, so an interval taken across folds
measures partition noise and shrinks as repetitions are added.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# DeLong
# ---------------------------------------------------------------------------

def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1
        i = j + 1

    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out

def _structural_components(scores: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """V10/V01 components of the DeLong covariance, for k models at once."""
    pos = scores[:, :n_pos]
    neg = scores[:, n_pos:]
    k, m = pos.shape
    n = neg.shape[1]

    tx = np.vstack([_midrank(pos[r]) for r in range(k)])
    ty = np.vstack([_midrank(neg[r]) for r in range(k)])
    tz = np.vstack([_midrank(scores[r]) for r in range(k)])

    auc = (tz[:, :m].sum(axis=1) / (m * n)) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    return auc, v01, v10

def _delong_cov(scores: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    auc, v01, v10 = _structural_components(scores, n_pos)
    m, n = n_pos, scores.shape[1] - n_pos
    cov = np.cov(v01) / m + np.cov(v10) / n
    return auc, np.atleast_2d(cov)

@dataclass(frozen=True)
class ComparisonResult:
    auc_a: float
    auc_b: float
    difference: float
    ci_low: float
    ci_high: float
    statistic: float
    p_value: float
    n: int
    n_positive: int
    method: str

    def __str__(self) -> str:
        return (
            f"{self.method}: {self.auc_a:.4f} vs {self.auc_b:.4f}  "
            f"Delta={self.difference:+.4f} "
            f"[{self.ci_low:+.4f}, {self.ci_high:+.4f}]  p={self.p_value:.4g}"
        )

def delong_test(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, alpha: float = 0.05
) -> ComparisonResult:
    """Compare two correlated AUROCs measured on the same subjects.

        DeLong & DeLong (1988); fast implementation after Sun & Xu (2014).
    """
    y = np.asarray(y).ravel().astype(int)
    p_a = np.asarray(p_a).ravel().astype(float)
    p_b = np.asarray(p_b).ravel().astype(float)

    if len(np.unique(y)) < 2:
        raise ValueError("both classes must be present")

    order = np.argsort(-y)  # positives first
    y_sorted = y[order]
    n_pos = int(y_sorted.sum())
    scores = np.vstack([p_a[order], p_b[order]])

    auc, cov = _delong_cov(scores, n_pos)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = float(auc[0] - auc[1])

    if var <= 0:
        z, p = 0.0, 1.0
        half = 0.0
    else:
        z = diff / np.sqrt(var)
        p = float(2 * stats.norm.sf(abs(z)))
        half = float(stats.norm.ppf(1 - alpha / 2) * np.sqrt(var))

    return ComparisonResult(
        auc_a=float(auc[0]), auc_b=float(auc[1]), difference=diff,
        ci_low=diff - half, ci_high=diff + half,
        statistic=float(z), p_value=p,
        n=int(y.size), n_positive=n_pos, method="DeLong",
    )

# ---------------------------------------------------------------------------
# Bootstrap over subjects
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric=roc_auc_score,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    stratified: bool = True,
) -> tuple[float, float, float]:
    """Percentile CI obtained by resampling subjects.

    This is the interval to report. The submitted CIs were computed across
    repetition means of a repeated cross-validation, where every repetition sees
    all 20,662 subjects and only the partition changes, so they measured
    partition noise, not sampling uncertainty, and narrowed without bound as
    repetitions were added. Resampling subjects measures the quantity a clinical
    reader assumes a CI represents.
    """
    y = np.asarray(y).ravel()
    p = np.asarray(p).ravel()
    rng = np.random.default_rng(seed)

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    n = len(y)

    values = []
    for _ in range(n_resamples):
        if stratified:
            # Preserves the number of positives, so the statistic stays defined
            # at 2% prevalence where an unstratified resample can draw none.
            idx = np.concatenate([
                rng.choice(pos_idx, size=len(pos_idx), replace=True),
                rng.choice(neg_idx, size=len(neg_idx), replace=True),
            ])
        else:
            idx = rng.integers(0, n, size=n)

        if len(np.unique(y[idx])) < 2:
            continue
        values.append(metric(y[idx], p[idx]))

    if not values:
        return float("nan"), float("nan"), float("nan")

    values = np.asarray(values)
    return (
        float(metric(y, p)),
        float(np.quantile(values, alpha / 2)),
        float(np.quantile(values, 1 - alpha / 2)),
    )

def bootstrap_difference(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    metric=roc_auc_score,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> ComparisonResult:
    """Paired bootstrap of a metric difference on the same subjects."""
    y = np.asarray(y).ravel()
    p_a = np.asarray(p_a).ravel()
    p_b = np.asarray(p_b).ravel()
    rng = np.random.default_rng(seed)

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)

    diffs = []
    for _ in range(n_resamples):
        idx = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ])
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(metric(y[idx], p_a[idx]) - metric(y[idx], p_b[idx]))

    diffs = np.asarray(diffs)
    observed = float(metric(y, p_a) - metric(y, p_b))
    # Two-sided p from the proportion of resamples on the other side of zero.
    p_value = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))

    return ComparisonResult(
        auc_a=float(metric(y, p_a)), auc_b=float(metric(y, p_b)),
        difference=observed,
        ci_low=float(np.quantile(diffs, alpha / 2)),
        ci_high=float(np.quantile(diffs, 1 - alpha / 2)),
        statistic=float("nan"), p_value=min(p_value, 1.0),
        n=int(y.size), n_positive=int((y == 1).sum()),
        method=f"paired bootstrap ({n_resamples})",
    )

def _fast_auc(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """AUROC of many score vectors against one label vector, at once.

    ``scores`` is (n_models, n_subjects). Rank-based, so a 50-fold bootstrap
    stays tractable: ``roc_auc_score`` in a Python loop would be 100,000 calls
    per comparison.
    """
    order = np.argsort(scores, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype="float64")
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, scores.shape[1] + 1, dtype="float64")

    # Mid-ranks, but only for rows that actually contain ties. Network outputs
    # are continuous, so this branch is almost never taken and checking is far
    # cheaper than correcting unconditionally.
    sorted_scores = np.take_along_axis(scores, order, axis=1)
    tied = np.flatnonzero((np.diff(sorted_scores, axis=1) == 0).any(axis=1))
    for i in tied:
        _, start, count = np.unique(sorted_scores[i], return_index=True, return_counts=True)
        for st, ct in zip(start, count):
            if ct > 1:
                ranks[i, order[i, st:st + ct]] = st + (ct + 1) / 2

    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    pos_rank_sum = ranks[:, y == 1].sum(axis=1)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def fold_difference_bootstrap(
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> ComparisonResult:
    """Compare two approaches across repeated cross-validation, honestly.

        ``scores_a`` and ``scores_b`` are (n_folds, n_subjects): one row per trained
        model, every row scored on the same subjects. The statistic is the mean over
        folds of the per-fold AUROC difference, which is exactly the difference of
        the fold-averaged AUROCs reported in the results table, the same object,
        not an ensemble of the folds.
    """
    y = np.asarray(y).ravel()
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)

    observed = float((_fast_auc(y, scores_a) - _fast_auc(y, scores_b)).mean())

    diffs = []
    for _ in range(n_resamples):
        idx = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ])
        y_b = y[idx]
        if len(np.unique(y_b)) < 2:
            continue
        d = _fast_auc(y_b, scores_a[:, idx]) - _fast_auc(y_b, scores_b[:, idx])
        diffs.append(d.mean())

    diffs = np.asarray(diffs)
    p_value = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return ComparisonResult(
        auc_a=float(_fast_auc(y, scores_a).mean()),
        auc_b=float(_fast_auc(y, scores_b).mean()),
        difference=observed,
        ci_low=float(np.quantile(diffs, alpha / 2)),
        ci_high=float(np.quantile(diffs, 1 - alpha / 2)),
        statistic=float("nan"), p_value=min(p_value, 1.0),
        n=int(y.size), n_positive=int((y == 1).sum()),
        method=f"fold-level difference, subject bootstrap ({n_resamples})",
    )

# ---------------------------------------------------------------------------
# Repeated cross-validation
# ---------------------------------------------------------------------------

def nadeau_bengio_ci(
    fold_scores: np.ndarray,
    n_train: int,
    n_test: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Variance-corrected interval for repeated k-fold cross-validation.

    The naive interval across folds understates variance because the training
    sets overlap heavily. Nadeau & Bengio (2003) correct by ``1/k + n_test/n_train``
    in place of ``1/k``. Report this alongside the subject-level bootstrap if a
    fold-level interval is wanted, never instead of it.
    """
    scores = np.asarray(fold_scores, dtype=float)
    scores = scores[~np.isnan(scores)]
    k = len(scores)
    if k < 2:
        return float(scores.mean()) if k else float("nan"), float("nan"), float("nan")

    mean = float(scores.mean())
    corrected_var = scores.var(ddof=1) * (1.0 / k + n_test / n_train)
    half = float(stats.t.ppf(1 - alpha / 2, k - 1) * np.sqrt(max(corrected_var, 0.0)))
    return mean, mean - half, mean + half

def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    """Adjust a family of pairwise comparisons.

    With several strategies compared against FedAvg, the family-wise error rate
    needs controlling; Holm is uniformly more powerful than Bonferroni and just
    as easy to defend.
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted
