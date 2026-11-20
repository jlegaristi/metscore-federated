"""Statistical comparison of approaches.

  delong_test                 two correlated AUROCs on the same subjects
  fold_difference_bootstrap   mean over folds of the per-fold AUROC difference,
                              with interval and p value from a bootstrap of subjects
  held_out_bootstrap          the same idea for any per-fold statistic, also when
                              the folds hold different subjects (cross-validation)
  decompose_auroc             within-centre and between-centre AUROC
  holm_bonferroni             correction for multiple comparisons

Intervals resample subjects, not folds: the folds of a repeated cross-validation
share their subjects, so the spread across folds is not a sampling error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

# DeLong's test


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
    """V10 and V01 components of DeLong's covariance, for k models at once."""
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
    """DeLong's test for two AUROCs measured on the same subjects.

    DeLong et al. (1988), with the fast algorithm of Sun and Xu (2014).
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

# Bootstrap over subjects


def bootstrap_p(draws: np.ndarray) -> float:
    """Two-sided bootstrap p value of a difference.

    Share of resamples on the other side of zero, with one added to numerator and
    denominator, so with B resamples it is at least 2 / (B + 1). B has to be large
    enough for the smallest p value to be reported, also after a correction for
    multiple comparisons.
    """
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    tail = min(int((draws <= 0).sum()), int((draws >= 0).sum()))
    return min(1.0, 2.0 * (tail + 1) / (draws.size + 1))


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric=roc_auc_score,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    stratified: bool = True,
) -> tuple[float, float, float]:
    """Percentile interval of a metric, resampling subjects.

    Returns (point estimate, lower, upper).
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
            # resample cases and non-cases separately, so that a resample
            # always has both
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
    """Paired bootstrap of the difference in a metric between two models."""
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
    p_value = bootstrap_p(diffs)

    return ComparisonResult(
        auc_a=float(metric(y, p_a)), auc_b=float(metric(y, p_b)),
        difference=observed,
        ci_low=float(np.quantile(diffs, alpha / 2)),
        ci_high=float(np.quantile(diffs, 1 - alpha / 2)),
        statistic=float("nan"), p_value=p_value,
        n=int(y.size), n_positive=int((y == 1).sum()),
        method=f"paired bootstrap ({n_resamples})",
    )


def _tie_runs(sorted_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and last position of the run of equal values each position belongs to."""
    n = sorted_scores.shape[1]
    position = np.broadcast_to(np.arange(n), sorted_scores.shape)
    starts = np.ones(sorted_scores.shape, dtype=bool)
    starts[:, 1:] = sorted_scores[:, 1:] != sorted_scores[:, :-1]
    ends = np.ones(sorted_scores.shape, dtype=bool)
    ends[:, :-1] = starts[:, 1:]
    first = np.maximum.accumulate(np.where(starts, position, 0), axis=1)
    last = np.minimum.accumulate(np.where(ends, position, n - 1)[:, ::-1], axis=1)[:, ::-1]
    return first, last


def _fast_auc(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """AUROC of each row of `scores` (n_models, n_subjects) against `y`.

    From mid-ranks and vectorized over the rows, which is what makes the bootstrap of
    50 fold models affordable. Ties matter: a resample repeats subjects, so every
    row has tied scores.
    """
    order = np.argsort(scores, axis=1, kind="stable")
    first, last = _tie_runs(np.take_along_axis(scores, order, axis=1))

    ranks = np.empty(order.shape, dtype="float64")
    np.put_along_axis(ranks, order, (first + last) / 2.0 + 1.0, axis=1)

    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    pos_rank_sum = ranks[:, y == 1].sum(axis=1)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _sorted_once(y: np.ndarray, scores: np.ndarray):
    """Sort each row of `scores` once, for `_auc_of_resample`."""
    order = np.argsort(scores, axis=1, kind="stable")
    first, last = _tie_runs(np.take_along_axis(scores, order, axis=1))
    return order, first, last, (np.asarray(y) == 1)[order]


def _auc_of_resample(presorted, counts: np.ndarray) -> np.ndarray:
    """AUROC of each row in a resample in which subject j appears `counts[j]` times.

    Equal to `_fast_auc` on the resampled arrays, without sorting them again: a case
    is concordant with the non-cases that score lower and half of those that tie.
    """
    order, first, last, positive = presorted
    weight = counts[order].astype("float64")
    negatives = np.where(positive, 0.0, weight)
    positives = np.where(positive, weight, 0.0)

    below = np.zeros((order.shape[0], order.shape[1] + 1))
    np.cumsum(negatives, axis=1, out=below[:, 1:])
    lower = np.take_along_axis(below, first, axis=1)
    tied = np.take_along_axis(below, last + 1, axis=1) - lower
    concordant = (positives * (lower + 0.5 * tied)).sum(axis=1)
    return concordant / (positives.sum(axis=1) * negatives.sum(axis=1))


def fold_difference_bootstrap(
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> ComparisonResult:
    """Compare two approaches whose fold models were scored on the same subjects.

    `scores_a` and `scores_b` are (n_folds, n_subjects), one row per trained model.
    The statistic is the mean over folds of the per-fold AUROC difference, that is,
    the difference of the fold-averaged AUROCs of the results table. The interval
    and the p value come from resampling subjects, stratified by class.
    """
    y = np.asarray(y).ravel()
    rng = np.random.default_rng(seed)

    observed = float((_fast_auc(y, scores_a) - _fast_auc(y, scores_b)).mean())

    sorted_a, sorted_b = _sorted_once(y, scores_a), _sorted_once(y, scores_b)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        counts = _multiplicities(y, rng)
        d = _auc_of_resample(sorted_a, counts) - _auc_of_resample(sorted_b, counts)
        diffs[i] = d.mean()

    return ComparisonResult(
        auc_a=float(_fast_auc(y, scores_a).mean()),
        auc_b=float(_fast_auc(y, scores_b).mean()),
        difference=observed,
        ci_low=float(np.quantile(diffs, alpha / 2)),
        ci_high=float(np.quantile(diffs, 1 - alpha / 2)),
        statistic=float("nan"), p_value=bootstrap_p(diffs),
        n=int(y.size), n_positive=int((y == 1).sum()),
        method=f"fold-level difference, subject bootstrap ({n_resamples})",
    )


# Within and between centres, and the bootstrap for cross-validation folds


def decompose_auroc(y, scores, site) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Pooled AUROC split into pairs from the same centre and from different centres.

    `scores` is (n_models, n_subjects). Returns the pooled, within-centre and
    between-centre AUROC of each model and the share of case/non-case pairs that are
    between centres, with pooled = (1 - share) * within + share * between.
    """
    y = np.asarray(y).ravel().astype(int)
    site = np.asarray(site).ravel()
    scores = np.atleast_2d(scores)

    pooled = _fast_auc(y, scores)
    total_pairs = float(y.sum() * (y.size - y.sum()))
    concordant, within_pairs = np.zeros(scores.shape[0]), 0.0
    for centre in np.unique(site):
        mask = site == centre
        positives, negatives = int(y[mask].sum()), int((1 - y[mask]).sum())
        if positives and negatives:
            concordant += _fast_auc(y[mask], scores[:, mask]) * positives * negatives
            within_pairs += positives * negatives

    between_pairs = total_pairs - within_pairs
    within = concordant / within_pairs if within_pairs else np.full(scores.shape[0], np.nan)
    between = (
        (pooled * total_pairs - concordant) / between_pairs
        if between_pairs else np.full(scores.shape[0], np.nan)
    )
    return pooled, within, between, between_pairs / total_pairs


@dataclass(frozen=True)
class HeldOut:
    """The held-out subjects of one fold with the scores of each model on them."""

    subjects: np.ndarray
    y: np.ndarray
    site: np.ndarray
    scores: np.ndarray  # (n_models, n_subjects)


def _multiplicities(strata: np.ndarray, rng) -> np.ndarray:
    """How many times each subject is drawn in one resample, stratum by stratum."""
    counts = np.zeros(len(strata), dtype=int)
    for stratum in np.unique(strata):
        members = np.flatnonzero(strata == stratum)
        counts[members] = rng.multinomial(len(members), np.full(len(members), 1 / len(members)))
    return counts


def held_out_bootstrap(
    folds: list[HeldOut],
    strata: np.ndarray,
    statistic,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap of a per-fold statistic averaged over folds, resampling subjects.

    A subject held out in several folds (one per repetition) is drawn the same number
    of times in all of them, which keeps the dependence between those folds. `strata`
    gives the stratum of each subject, indexed by the ids in `HeldOut.subjects`, and
    `statistic` maps (y, site, scores) to an array. Returns the observed value and
    the resampled ones.
    """
    observed = np.nanmean([statistic(f.y, f.site, f.scores) for f in folds], axis=0)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        counts = _multiplicities(strata, rng)
        values = []
        for fold in folds:
            chosen = np.repeat(np.arange(len(fold.y)), counts[fold.subjects])
            if len(np.unique(fold.y[chosen])) == 2:
                values.append(statistic(fold.y[chosen], fold.site[chosen], fold.scores[:, chosen]))
        if values:
            draws.append(np.nanmean(values, axis=0))
    return observed, np.asarray(draws)


def interval_and_p(draws: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile interval and two-sided p value of a resampled difference."""
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    return (
        float(np.quantile(draws, alpha / 2)),
        float(np.quantile(draws, 1 - alpha / 2)),
        bootstrap_p(draws),
    )

# Repeated cross-validation


def nadeau_bengio_ci(
    fold_scores: np.ndarray,
    n_train: int,
    n_test: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Interval across the folds of a repeated k-fold cross-validation.

    Uses the variance correction of Nadeau and Bengio (2003), 1/k + n_test/n_train
    instead of 1/k, because the training sets overlap. Only used to compare with
    the subject-level bootstrap.
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
    """Holm's step-down adjustment of a set of p values."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted
