# Pairwise comparison of every approach with FedAvg on the external cohort, bootstrap
# intervals and the privacy budgets.
#
# Run scripts/run_experiments.py first, then this script from analysis/.

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd

from metscore import results, stats
from metscore.config import OUTPUT_DIR, RESULTS_DIR
from metscore.results import to_latex
from metscore.plotting import label_for

pd.set_option("display.width", 220)

TABLES = OUTPUT_DIR / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

REFERENCE = "federated__fedavg"
EVALUATION = "external_train_fitted"

# Resamples behind the pairwise table. Holm multiplies the smallest p value by the
# number of comparisons, 13, and a bootstrap p value cannot go below 2 / (B + 1), so
# reporting p < 0.001 after correction needs more than 26,000 resamples.
RESAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000

# One prediction per subject, averaging the 50 fold models. This is the AUROC of an
# ensemble and comes out higher than the fold means of make_tables.py; it is only
# used for the intervals further down.


def aligned(run_id, evaluation=EVALUATION):
    frame = results.load_predictions(run_id, evaluation)
    return (
        frame.groupby("subject_index")
        .agg(y=("y_true", "first"), p=("y_pred", "mean"), site=("site", "first"))
        .reset_index()
    )


reference = aligned(REFERENCE)
print(f"{REFERENCE}: {len(reference)} subjects, {int(reference.y.sum())} positive")

# Pairwise comparison with FedAvg.
#
# Each fold gives one model per approach, and every model scores the same 661
# external subjects. The statistic is the mean over folds of the per-fold AUROC
# difference, i.e. the difference of the fold-averaged AUROCs in the results table.
# The interval and the p value come from resampling subjects.

COMPARATORS = [
    "unified__centralised",
    "individual__AKRIBEA", "individual__OSARTEN", "individual__LIVERBIBLE",
    "individual__AGEPORTUGAL", "individual__METS+",
    "federated__fedavg_uniform", "federated__fedavg_cohort",
    "federated__fedmedian", "federated__fedtrimmed",
    "federated__fedprox__mu0.01", "federated__fedprox__mu0.1", "federated__fedprox__mu1.0",
]


def fold_matrix(run_id, evaluation=EVALUATION):
    """(y, scores) with one row of scores per fold, all on the same subjects."""
    frame = results.load_predictions(run_id, evaluation)
    folds = [g.sort_values("subject_index") for _, g in frame.groupby(["repetition", "fold"])]
    return folds[0].y_true.values, np.vstack([f.y_pred.values for f in folds])


y_ext, ref_scores = fold_matrix(REFERENCE)
print(f"{REFERENCE}: {len(y_ext)} subjects, {int(y_ext.sum())} positive, "
      f"{ref_scores.shape[0]} folds")

rows, raw_p = [], {}
for run in COMPARATORS:
    try:
        _, other_scores = fold_matrix(run)
    except FileNotFoundError:
        print(f"  skipping {run} (not run yet)")
        continue

    test = stats.fold_difference_bootstrap(
        y_ext, other_scores, ref_scores, n_resamples=RESAMPLES
    )
    raw_p[run] = test.p_value

    # DeLong's test in each fold, to see in how many folds the difference holds
    per_fold = [
        stats.delong_test(y_ext, other_scores[k], ref_scores[k]).p_value
        for k in range(ref_scores.shape[0])
    ]
    rows.append({
        "run": run,
        "Approach": label_for(run),
        "_delta": test.difference,
        "Δ vs FedAvg": round(test.difference, 3),
        "95% CI": f"[{test.ci_low:+.3f}, {test.ci_high:+.3f}]",
        "p": test.p_value,
        "folds": f"{sum(q < 0.05 for q in per_fold)}/{len(per_fold)}",
    })

comparison = pd.DataFrame(rows)
comparison["p (Holm)"] = comparison["run"].map(stats.holm_bonferroni(raw_p))
comparison = comparison.drop(columns="run").sort_values("_delta", ascending=False)
comparison = comparison.drop(columns="_delta")
print(comparison.to_string(index=False))

resamples = f"{RESAMPLES:,}".replace(",", "{,}")
formatted = comparison.rename(columns={"Δ vs FedAvg": "Δ", "folds": "Folds"})
formatted = formatted[["Approach", "Δ", "95% CI", "p (Holm)", "Folds"]].copy()
formatted["p (Holm)"] = comparison["p (Holm)"].map(
    lambda v: "<0.001" if v < 0.001 else f"{v:.3f}"
)

to_latex(
    formatted,
    "Pairwise comparison against FedAvg on the external cohort",
    "tab:pairwise", TABLES / "pairwise_delong.tex", digits=3, index=False,
    full_width=False, tabcolsep="3pt",
    note=(
        "Each fold trains one model per approach and scores it on the same 661 "
        "external subjects. $\\Delta$ is the mean over folds of the per-fold AUROC "
        "difference with FedAvg, so it is exactly the difference of the "
        "fold-averaged AUROCs in Table~\\ref{tab:results}. Intervals and $p$ values "
        f"come from resampling subjects, not folds ({resamples} resamples), and $p$ "
        f"is Holm-corrected across the {len(raw_p)} comparisons. \\emph{{Folds}} counts "
        "the folds in which DeLong's test gives $p < 0.05$."
    ),
)
print(f"\nwrote {TABLES / 'pairwise_delong.tex'}")

# The centralized model trained on the clients' samples (sensitivity analysis),
# against FedAvg and against the centralized arm. Kept out of the family above.
try:
    _, cohort_scores = fold_matrix("unified__centralised__cohort-downsampling")
    _, pooled_scores = fold_matrix("unified__centralised")
    for name, other in [("FedAvg", ref_scores), ("Centralized", pooled_scores)]:
        print(f"cohort downsampling vs {name}:",
              stats.fold_difference_bootstrap(y_ext, cohort_scores, other, n_resamples=RESAMPLES))
except FileNotFoundError:
    pass

# Bootstrap intervals of the external AUROC (ensemble predictions).

KEY_RUNS = [
    "unified__centralised", "federated__fedavg", "federated__fedprox__mu0.01",
    "individual__AKRIBEA", "individual__OSARTEN",
    "individual__LIVERBIBLE", "individual__AGEPORTUGAL", "individual__METS+",
]

rows = []
for run in KEY_RUNS:
    try:
        frame = aligned(run)
    except FileNotFoundError:
        continue
    point, low, high = stats.bootstrap_ci(frame.y.values, frame.p.values, n_resamples=2000)
    rows.append({
        "Approach": label_for(run),
        "AUROC": f"{point:.3f}",
        "95% CI": f"{low:.3f}–{high:.3f}",
        "Width": round(high - low, 4),
    })

intervals = pd.DataFrame(rows)
print(intervals.to_string(index=False))

to_latex(
    intervals,
    "External AUROC with 95\\% confidence intervals from stratified bootstrap "
    "resampling of subjects (2{,}000 resamples).",
    "tab:bootstrap", TABLES / "bootstrap_ci.tex", index=False, full_width=False,
)
print(f"\nwrote {TABLES / 'bootstrap_ci.tex'}")

# Width of the interval for FedAvg: across folds, Nadeau-Bengio, and resampling
# subjects.

long = results.collect()
folds = long[
    (long.run_id == REFERENCE) & (long.metric == "auroc")
    & (long.evaluation == "external:train_fitted")
    & (long.site == "ALL") & (long.stratum_kind == "none")
]["value"].values

from scipy import stats as scipy_stats

half = scipy_stats.t.ppf(0.975, len(folds) - 1) * folds.std(ddof=1) / np.sqrt(len(folds))
naive = (folds.mean() - half, folds.mean() + half)
nb = stats.nadeau_bengio_ci(folds, n_train=16529, n_test=4133)
subject = stats.bootstrap_ci(reference.y.values, reference.p.values, n_resamples=2000)

print(f"naive across folds     [{naive[0]:.4f}, {naive[1]:.4f}]  width {naive[1]-naive[0]:.4f}")
print(f"Nadeau-Bengio          [{nb[1]:.4f}, {nb[2]:.4f}]  width {nb[2]-nb[1]:.4f}")
print(f"subject-level bootstrap[{subject[1]:.4f}, {subject[2]:.4f}]  width {subject[2]-subject[1]:.4f}")

# Paired bootstrap for PR-AUC, where DeLong does not apply.

from sklearn.metrics import average_precision_score

rows = []
for run in ["unified__centralised", "individual__AKRIBEA", "individual__METS+"]:
    try:
        other = aligned(run)
    except FileNotFoundError:
        continue
    merged = reference.merge(other, on="subject_index", suffixes=("_ref", "_oth"))
    test = stats.bootstrap_difference(
        merged.y_ref.values, merged.p_ref.values, merged.p_oth.values,
        metric=average_precision_score, n_resamples=1000,
    )
    rows.append({
        "Comparison": f"FedAvg vs {label_for(run)}",
        "PR-AUC (FedAvg)": round(test.auc_a, 4),
        "PR-AUC (other)": round(test.auc_b, 4),
        "Δ": round(test.difference, 4),
        "95% CI": f"[{test.ci_low:+.3f}, {test.ci_high:+.3f}]",
        "p": round(test.p_value, 4),
    })

print(pd.DataFrame(rows).to_string(index=False))

# Privacy budgets of the differential-privacy runs.

import json

from metscore.privacy import compute_epsilon


def private_runs():
    """(noise multiplier, run id, budget) of each private run, sorted by noise."""
    found = []
    for path in RESULTS_DIR.glob("federated__dp_fedavg*/profile.json"):
        dp = json.loads(path.read_text()).get("differential_privacy")
        if dp:
            found.append((dp["noise_multiplier"], path.parent.name, dp))
    return sorted(found, key=lambda item: item[0])


def mean_auroc(run, evaluation):
    return long[
        (long.run_id == run) & (long.metric == "auroc") & (long.evaluation == evaluation)
        & (long.site == "ALL") & (long.stratum_kind == "none")
    ]["value"].mean()


def shown_epsilon(value):
    return "$\\infty$" if value == float("inf") else f"{value:.1f}"


rows = []
for noise, run, dp in private_runs():
    rows.append({
        "Noise": "reference" if run.endswith("__reference") else f"{noise:g}",
        "ε": shown_epsilon(dp["epsilon"]),
        "ε (no sec. agg.)": shown_epsilon(dp["epsilon_without_secure_aggregation"]),
        "CV AUROC": round(mean_auroc(run, "cv"), 3),
        "Ext. AUROC": round(mean_auroc(run, "external:train_fitted"), 3),
        "Clipped": f"{dp['fraction_clipped']:.0%}",
    })

privacy = pd.DataFrame(rows)
print(privacy.to_string(index=False))

print("\nNoise required for a meaningful guarantee, 20 rounds, δ=1e-5:")
from scipy.optimize import brentq

for target in (10.0, 3.0, 1.0):
    sigma = brentq(lambda s: compute_epsilon(s, 20).epsilon - target, 0.5, 500, xtol=1e-3)
    print(f"  ε = {target:<5} requires noise multiplier {sigma:.2f}")

to_latex(
    privacy,
    r"Privacy--utility trade-off for client-level differential privacy over 20 "
    r"rounds of full participation ($\delta = 10^{-5}$), with $\varepsilon$ from "
    r"the R\'enyi accountant. The reference run has the same architecture and "
    r"weighting with no noise and no active clipping bound.",
    "tab:privacy", TABLES / "privacy_utility.tex", index=False, full_width=False,
)
print(f"\nwrote {TABLES / 'privacy_utility.tex'}")
