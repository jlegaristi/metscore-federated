# Tables of the paper, written as LaTeX from the metrics saved in results/.
#
# Run scripts/run_experiments.py first, then this script from analysis/.

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd

from metscore import results
from metscore.config import COHORTS, OUTPUT_DIR, RESULTS_DIR
from metscore.evaluation import OperatingPoint
from metscore.evaluation import evaluate as score
from metscore.results import to_latex, to_latex_grouped
from metscore.plotting import label_for

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

TABLES = OUTPUT_DIR / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

long = results.collect()
print(f"{len(long):,} measurements across {long.run_id.nunique()} runs")
print(f"evaluations: {sorted(long.evaluation.unique())}")

# Every run should have 50 folds (10 repetitions x 5 folds) per evaluation.

overall = long[(long.site == "ALL") & (long.stratum_kind == "none") & (long.metric == "auroc")]
completeness = overall.groupby(["run_id", "evaluation"]).size().unstack()
print(completeness.to_string())

incomplete = completeness[(completeness < 50).any(axis=1)]
if len(incomplete):
    print(f"\nINCOMPLETE, do not report these:\n{incomplete.to_string()}")
else:
    print("\nAll runs complete.")

# Mean and SD across folds. Confidence intervals are in statistics.py and
# per_centre.py, from resampling subjects.

METRICS = ["auroc", "pr_auc", "sensitivity", "specificity", "ppv", "ece"]

ORDER = [
    "unified__centralised",
    "federated__fedavg", "federated__fedavg_uniform", "federated__fedavg_cohort",
    "federated__fedprox__mu0.01", "federated__fedprox__mu0.1", "federated__fedprox__mu1.0",
    "federated__fedmedian", "federated__fedtrimmed",
    "individual__AKRIBEA", "individual__OSARTEN", "individual__LIVERBIBLE",
    "individual__AGEPORTUGAL", "individual__METS+",
]


def summarise(evaluation, metrics=METRICS, runs=None, digits=3, include_dp=False):
    """Mean ± SD across folds, one row per run.

    The differential-privacy runs and the persistent-Adam variants are left out
    unless asked for; they are reported separately.
    """
    sel = long[
        (long.evaluation == evaluation)
        & (long.site == "ALL")
        & (long.stratum_kind == "none")
        & (long.metric.isin(metrics))
    ]
    if not include_dp:
        sel = sel[~sel.run_id.str.contains("dp_fedavg|adam-persistent")]
    if runs is not None:
        sel = sel[sel.run_id.isin(runs)]

    stats = sel.groupby(["run_id", "metric"])["value"].agg(["mean", "std"])
    shown = stats.apply(
        lambda r: "—" if np.isnan(r["mean"]) else f"{r['mean']:.{digits}f} ± {r['std']:.{digits}f}",
        axis=1,
    ).unstack()

    ordered = [r for r in ORDER if r in shown.index] + [
        r for r in shown.index if r not in ORDER
    ]
    table = shown.loc[ordered, [m for m in metrics if m in shown.columns]]
    table.index = [label_for(r) for r in table.index]
    return table

# Main results table: cross-validation and external validation, same metrics.


FEDERATED = [
    "federated__fedavg", "federated__fedavg_uniform", "federated__fedavg_cohort",
    "federated__fedprox__mu0.01", "federated__fedprox__mu0.1",
    "federated__fedprox__mu1.0", "federated__fedmedian", "federated__fedtrimmed",
]
INDIVIDUAL = [
    "individual__AKRIBEA", "individual__OSARTEN", "individual__LIVERBIBLE",
    "individual__AGEPORTUGAL", "individual__METS+",
]

# The "cv" records of an institution-specific model cover its own cohort only. The
# model is also scored on the held-out patients of every cohort in the same fold, and
# together those are the pooled test fold of the centralized and federated rows, so
# the cross-validation block can compare all of them on the same patients.


def pooled_cross_validation(run_id, reference):
    """Per-fold metrics of an institution-specific model on the pooled test folds."""
    scores = pd.concat(
        [results.load_predictions(run_id, f"cv_cross_{name}") for name in COHORTS.values()],
        ignore_index=True,
    )
    own = long[
        (long.run_id == run_id) & (long.evaluation == "cv") & (long.site == "ALL")
        & (long.stratum_kind == "none") & (long.metric == "threshold")
    ].set_index(["repetition", "fold"])["value"]
    rows = []
    for (repetition, fold), block in scores.groupby(["repetition", "fold"]):
        block = block.sort_values("subject_index")
        expected = reference[(repetition, fold)]
        if not (np.array_equal(block.subject_index.to_numpy(), expected.subject_index.to_numpy())
                and np.array_equal(block.y_true.to_numpy(), expected.y_true.to_numpy())):
            raise ValueError(f"{run_id}, repetition {repetition} fold {fold}: not the pooled test fold")
        point = OperatingPoint(float(own[(repetition, fold)]), "youden", "validation")
        metrics = score(block.y_true.to_numpy(), block.y_pred.to_numpy(), point)
        rows += [
            {"run_id": run_id, "evaluation": "cv:pooled", "site": "ALL", "stratum_kind": "none",
             "repetition": repetition, "fold": fold, "metric": m, "value": getattr(metrics, m)}
            for m in METRICS
        ]
    return pd.DataFrame(rows)


centralised_folds = {
    key: block.sort_values("subject_index")[["subject_index", "y_true"]]
    for key, block in results.load_predictions("unified__centralised", "cv").groupby(["repetition", "fold"])
}
long = pd.concat(
    [long] + [pooled_cross_validation(run, centralised_folds) for run in INDIVIDUAL],
    ignore_index=True,
)


def blocks(evaluation, individual=None):
    parts = [
        (None, summarise(evaluation, runs=["unified__centralised"])),
        ("Federated aggregation strategies", summarise(evaluation, runs=FEDERATED)),
    ]
    if individual:
        parts.append(("Institution-specific models", summarise(individual, runs=INDIVIDUAL)))
    return parts


cv = summarise("cv")
ext = summarise("external:train_fitted")
print(cv.to_string(), "\n\n", ext.to_string())

_ = to_latex_grouped(
    [
        ("Cross-validation (50 folds, pooled DISCOVERY)", blocks("cv", individual="cv:pooled")),
        ("External validation (VALIDATION cohort, n = 661)",
         blocks("external:train_fitted", individual="external:train_fitted")),
    ],
    "Performance of every approach under cross-validation and on the independent "
    "external cohort. Mean $\\pm$ SD across 50 folds.",
    "tab:results",
    TABLES / "results_combined.tex",
    note=(
        "The dispersion quoted is the standard deviation across folds, not a "
        "confidence interval. Operating points maximize Youden's index on "
        "held-out validation data and are frozen before scoring; test folds "
        "retain the natural prevalence of their cohort. Within each block every "
        "row is scored on the same individuals, in cross-validation the pooled "
        "test folds of all five cohorts (Section~\\ref{subsec:validation})."
    ),
)

# Where the centralized arm downsamples: the pooled training data (the arm of the
# main table) or each cohort on its own, which gives it the samples the federated
# clients train on.
DOWNSAMPLING = [
    "unified__centralised", "unified__centralised__cohort-downsampling", "federated__fedavg",
]
if DOWNSAMPLING[1] in set(long.run_id):
    parts = []
    for name, evaluation in [("CV", "cv"), ("External", "external:train_fitted")]:
        part = summarise(evaluation, metrics=["auroc", "ece"], runs=DOWNSAMPLING)
        parts.append(part.rename(columns={"auroc": f"{name} AUROC", "ece": f"{name} ECE"}))
    sensitivity = pd.concat(parts, axis=1)
    print(sensitivity.to_string())
    _ = to_latex(
        sensitivity,
        "The centralized model with downsampling over the pooled training data and "
        "within each cohort, next to FedAvg. Mean $\\pm$ SD across 50 folds.",
        "tab:downsampling", TABLES / "downsampling_sensitivity.tex",
    )

# Scaling contracts on the external cohort. Site-adapted scaling cannot score the
# reference subjects, so it is compared with train-fitted scaling on the subjects
# both can score (train_fitted_common).
contract = (
    long[
        (long.metric == "auroc") & (long.site == "ALL")
        & (long.stratum_kind == "none")
        & (long.evaluation.isin(["external:site_adapted",
                                 "external:train_fitted_common"]))
    ]
    .groupby(["run_id", "evaluation"])["value"].mean().unstack()
)
contract.columns = ["Site-adapted", "Train-fitted"]
contract["Δ"] = contract["Site-adapted"] - contract["Train-fitted"]
contract = contract.loc[[r for r in ORDER if r in contract.index]]
contract.index = [label_for(r) for r in contract.index]
print(contract.round(3).to_string())

_ = to_latex(
    contract.round(3),
    "External AUROC under each scaling contract, pooled across the three "
    "validation sites. Site-adapted normalization re-standardizes each target "
    "site against a held-out reference split from that site. The pooled penalty "
    "is an artifact of aggregation, not a loss of discrimination: it is larger "
    "than the penalty at any individual site, because re-centering removes the "
    "between-site offset that a pooled ranking exploits when prevalence spans "
    "2.9\\% to 77.6\\%. Within site the two contracts differ little.",
    "tab:scaling_contract",
    TABLES / "scaling_contract.tex",
)

# External AUROC at each site (prevalence 9.6%, 2.9% and 77.6%).

per_site = (
    long[
        (long.metric == "auroc") & (long.evaluation == "external:train_fitted")
        & (long.site != "ALL") & (long.stratum_kind == "none")
    ]
    .groupby(["run_id", "site"])["value"].mean().unstack()
)
per_site = per_site.loc[[r for r in ORDER if r in per_site.index]]
per_site.index = [label_for(r) for r in per_site.index]
print(per_site.round(3).to_string())

_ = to_latex(
    per_site.round(3),
    "External AUROC by validation sub-cohort. Prevalence is 9.6\\% in BPH, "
    "2.9\\% in GRAZ and 77.6\\% in T\\\"ubingen.",
    "tab:per_site",
    TABLES / "per_site.tex",
)

# Cross-validation AUROC by sex and by age band. Only DISCOVERY has both for
# every subject.

for kind, caption, label in [
    ("sex", "Cross-validation AUROC stratified by sex.", "tab:strat_sex"),
    ("age_band", "Cross-validation AUROC stratified by age band.", "tab:strat_age"),
]:
    strat = (
        long[
            (long.metric == "auroc") & (long.evaluation == "cv")
            & (long.stratum_kind == kind)
            & (long.run_id.isin(["unified__centralised", "federated__fedavg"]))
        ]
        .groupby(["run_id", "stratum"])["value"].mean().unstack()
    )
    if strat.empty:
        continue
    strat.index = [label_for(r) for r in strat.index]
    print(f"\n{kind}:")
    print(strat.round(3).to_string())
    _ = to_latex(strat.round(3), caption, label, TABLES / f"strat_{kind}.tex")

# Cost of each paradigm over the 50 folds.


import json

rows = []
for run in ["unified__centralised", "federated__fedavg", "federated__fedprox__mu0.01"]:
    profile = RESULTS_DIR / run / "profile.json"
    if not profile.exists():
        continue
    data = json.loads(profile.read_text())
    comms = data.get("communication") or {}
    rows.append({
        "Approach": label_for(run),
        "Wall clock (h)": round(data["wall_clock_s"] / 3600, 2),
        "Client compute (h)": round(data["client_compute_s"] / 3600, 2),
        "Communication (MB)": comms.get("total_mb", 0.0),
        "Rounds": data.get("n_rounds", "—"),
    })

cost = pd.DataFrame(rows).set_index("Approach")
print(cost.to_string())
_ = to_latex(
    cost,
    "Computational and communication cost for 50 folds. Communication follows from "
    "the protocol: every transmitted value (trainable weights and batch normalization "
    "statistics) $\\times$ 4 bytes $\\times$ 2 directions $\\times$ 20 rounds "
    "$\\times$ 5 clients.",
    "tab:cost",
    TABLES / "cost.tex",
)

print(f"\nAll tables written to {TABLES}")
for path in sorted(TABLES.glob("*.tex")):
    print(f"  \\input{{tables/{path.name}}}")
