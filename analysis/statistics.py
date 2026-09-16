# Statistical comparison between approaches.
#
#   DeLong           correlated AUROCs measured on the same subjects
#   Bootstrap        intervals and p values, resampling subjects
#   Holm             family-wise error control across the comparisons
#
# Run scripts/run_experiments.py first.

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd

from metscore import results, stats
from metscore.results import to_latex
from metscore.plotting import label_for

pd.set_option("display.width", 220)

TABLES = Path.cwd().parent / "tables"
TABLES.mkdir(exist_ok=True)

REFERENCE = "federated__fedavg"
EVALUATION = "external_train_fitted"

# Aligning predictions. Averaging a subject's score across the folds in which
# they were held out gives one prediction per subject. This is an ensemble
# quantity and reads higher than the fold-level means of make_tables.py.

def aligned(run_id, evaluation=EVALUATION):
    frame = results.load_predictions(run_id, evaluation)
    return (
        frame.groupby("subject_index")
        .agg(y=("y_true", "first"), p=("y_pred", "mean"), site=("site", "first"))
        .reset_index()
    )

reference = aligned(REFERENCE)
print(f"{REFERENCE}: {len(reference)} subjects, {int(reference.y.sum())} positive")

# Pairwise comparison against the federated model.
#
# The unit compared is one trained model, not an ensemble of the fifty. Every
# fold produces a model, every model scores the same 661 external subjects, and
# the statistic is the mean over folds of the per-fold AUROC difference, which
# equals the difference of the fold-averaged AUROCs written by make_tables.py.
# Averaging the fifty models' predictions first would measure an ensemble that
# no deployment assembles, and favours the least stable models.
#
# Uncertainty comes from resampling subjects. Folds share their subjects and
# most of their training data, so resampling folds measures partition noise.

COMPARATORS = [
    "unified__centralised",
    "individual__AKRIBEA", "individual__OSARTEN", "individual__LIVERBIBLE",
    "individual__AGEPORTUGAL", "individual__METS+",
    "federated__fedavg_uniform", "federated__fedmedian", "federated__fedtrimmed",
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

    test = stats.fold_difference_bootstrap(y_ext, other_scores, ref_scores)
    raw_p[run] = test.p_value

    # How often the difference holds up, run by run. A mean difference can come
    # from every fold or from a few, and that distinction is what a site
    # deciding whether to federate actually cares about.
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

# Federated learning is statistically indistinguishable from the AKRIBEA and
# OSARTEN models and from the other aggregation rules, ahead of centralized
# pooling, and ahead of the models of the three small cohorts.

formatted = comparison.rename(columns={"Δ vs FedAvg": "Δ", "folds": "Folds"})
formatted = formatted[["Approach", "Δ", "95% CI", "p (Holm)", "Folds"]].copy()
# A bootstrap p of zero means no resample of 1,000 fell on the other side of
# zero, which is a bound, not a value: reporting "0.000" would claim a precision
# the procedure cannot deliver.
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
        "come from resampling subjects, not folds, and $p$ is Holm-corrected across "
        "the twelve comparisons. \\emph{Folds} counts the folds in which DeLong's "
        "test gives $p < 0.05$."
    ),
)
print(f"\nwrote {TABLES / 'pairwise_delong.tex'}")

# Confidence intervals by resampling subjects. Every repetition of the
# cross-validation sees all 20,662 subjects and only the partition changes, so
# an interval taken across repetitions measures partition noise and narrows
# without bound as repetitions are added.

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
    "External AUROC with 95\% confidence intervals from stratified bootstrap "
    "resampling of subjects (2{,}000 resamples).",
    "tab:bootstrap", TABLES / "bootstrap_ci.tex", index=False, full_width=False,
)
print(f"\nwrote {TABLES / 'bootstrap_ci.tex'}")

# How much the interval changes: the fold-level interval against the
# subject-level one, for the same federated model.

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

# Paired comparison on the metrics DeLong cannot handle, PR-AUC and PPV, by
# resampling subjects.

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

# Privacy budgets: the noise multiplier expressed as (epsilon, delta), which
# is a privacy statement rather than a hyperparameter.

import json

from metscore.privacy import compute_epsilon

rows = []
for path in sorted(
    (Path.cwd().parent / "results").glob("federated__dp_fedavg*/profile.json"),
    key=lambda p: float(p.parent.name.rsplit("noise_multiplier", 1)[1]),
):
    dp = json.loads(path.read_text()).get("differential_privacy")
    if not dp:
        continue
    run = path.parent.name
    auroc = long[
        (long.run_id == run) & (long.metric == "auroc") & (long.evaluation == "cv")
        & (long.site == "ALL") & (long.stratum_kind == "none")
    ]["value"].mean()
    rows.append({
        "Noise multiplier": dp["noise_multiplier"],
        "ε": round(dp["epsilon"], 1),
        "CV AUROC": round(auroc, 3),
        "Clipped": f"{dp['fraction_clipped']:.0%}",
    })

privacy = pd.DataFrame(rows)
print(privacy.to_string(index=False))

print("\nNoise required for a meaningful guarantee, 20 rounds, δ=1e-5:")
from scipy.optimize import brentq

for target in (10.0, 3.0, 1.0):
    sigma = brentq(lambda s: compute_epsilon(s, 20).epsilon - target, 0.5, 500, xtol=1e-3)
    print(f"  ε = {target:<5} requires noise multiplier {sigma:.2f}")
print("\nUtility has collapsed by a multiplier of 0.5, so the two ranges never meet.")

to_latex(
    privacy,
    r"Privacy--utility trade-off for client-level differential privacy over 20 "
    r"rounds of full participation ($\delta = 10^{-5}$), with $\varepsilon$ from "
    r"the R\'enyi accountant.",
    "tab:privacy", TABLES / "privacy_utility.tex", index=False, full_width=False,
)
print(f"\nwrote {TABLES / 'privacy_utility.tex'}")
