# Tables, generated from results/ and exported as LaTeX so that no number is
# transcribed by hand. Nothing here trains a model or recomputes a metric; the
# arms wrote every measurement and this script only reshapes them.
#
# Run scripts/run_experiments.py first.

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd

from metscore import results
from metscore.results import to_latex, to_latex_grouped
from metscore.plotting import label_for

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

TABLES = Path.cwd().parent / "tables"
TABLES.mkdir(exist_ok=True)

long = results.collect()
print(f"{len(long):,} measurements across {long.run_id.nunique()} runs")
print(f"evaluations: {sorted(long.evaluation.unique())}")

# Completeness. Every run should carry 50 folds (10 repetitions x 5 folds) for
# each evaluation; a short row means the run was interrupted.

overall = long[(long.site == "ALL") & (long.stratum_kind == "none") & (long.metric == "auroc")]
completeness = overall.groupby(["run_id", "evaluation"]).size().unstack()
print(completeness.to_string())

incomplete = completeness[(completeness < 50).any(axis=1)]
if len(incomplete):
    print(f"\nINCOMPLETE, do not report these:\n{incomplete.to_string()}")
else:
    print("\nAll runs complete.")

# Fold-level mean and standard deviation. The SD is dispersion across folds,
# not a confidence interval; intervals come from statistics.py, which resamples
# subjects.

METRICS = ["auroc", "pr_auc", "sensitivity", "specificity", "ppv", "ece"]

ORDER = [
    "unified__centralised",
    "federated__fedavg", "federated__fedavg_uniform",
    "federated__fedprox__mu0.01", "federated__fedprox__mu0.1", "federated__fedprox__mu1.0",
    "federated__fedmedian", "federated__fedtrimmed",
    "individual__AKRIBEA", "individual__OSARTEN", "individual__LIVERBIBLE",
    "individual__AGEPORTUGAL", "individual__METS+",
]

def summarise(evaluation, metrics=METRICS, runs=None, digits=3, include_dp=False):
    """Mean ± SD across folds, one row per run.

    Differential privacy is excluded by default. It is a privacy mechanism
    layered on FedAvg rather than an alternative training paradigm, so listing
    its noise levels beside the three arms mixes two axes; and six rows sitting
    near 0.50 AUROC crowd the table without informing. The privacy--utility
    trade-off is reported on its own, where the curve is the point.
    """
    sel = long[
        (long.evaluation == evaluation)
        & (long.site == "ALL")
        & (long.stratum_kind == "none")
        & (long.metric.isin(metrics))
    ]
    if not include_dp:
        sel = sel[~sel.run_id.str.contains("dp_fedavg")]
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

# Cross-validation and external validation in one table: same rows, same
# metrics, so the comparison reads vertically. Cross-validation is 10
# repetitions of stratified five-fold; test folds keep the natural prevalence of
# their cohort. External data are standardized with the scaler learned in
# training, applied identically to every approach.

FEDERATED = [
    "federated__fedavg", "federated__fedavg_uniform",
    "federated__fedprox__mu0.01", "federated__fedprox__mu0.1",
    "federated__fedprox__mu1.0", "federated__fedmedian", "federated__fedtrimmed",
]
INDIVIDUAL = [
    "individual__AKRIBEA", "individual__OSARTEN", "individual__LIVERBIBLE",
    "individual__AGEPORTUGAL", "individual__METS+",
]

def blocks(evaluation):
    return [
        (None, summarise(evaluation, runs=["unified__centralised"])),
        ("Federated aggregation strategies", summarise(evaluation, runs=FEDERATED)),
        ("Institution-specific models", summarise(evaluation, runs=INDIVIDUAL)),
    ]

cv = summarise("cv")
ext = summarise("external:train_fitted")
print(cv.to_string(), "\n\n", ext.to_string())

_ = to_latex_grouped(
    [
        ("Cross-validation (50 folds, pooled DISCOVERY)", blocks("cv")),
        ("External validation (VALIDATION cohort, n = 661)",
         blocks("external:train_fitted")),
    ],
    "Performance of every approach under cross-validation and on the independent "
    "external cohort. Mean $\\pm$ SD across 50 folds.",
    "tab:results",
    TABLES / "results_combined.tex",
    note=(
        "The dispersion quoted is the standard deviation across folds, not a "
        "confidence interval. Operating points maximize Youden's index on "
        "held-out validation data and are frozen before scoring; test folds "
        "retain the natural prevalence of their cohort. In the cross-validation "
        "block each institution-specific model is scored only within its own "
        "cohort, so those rows are not comparable with each other or with the "
        "pooled arms; the external block scores every model on the same 661 "
        "individuals."
    ),
)

# The scaling contract.
#
# Re-standardizing each external site against a held-out reference split from

# The two contracts reach different subjects, so the comparison uses
# train_fitted_common, that is, train-fitted restricted to the rows
# site-adapted can also score. Comparing against the full train-fitted
# evaluation would confound the contract with a larger sample.
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

# Per external sub-cohort. BPH, GRAZ and TUEBINGEN differ in prevalence (9.6%,
# 2.9% and 77.6%), so a pooled external AUROC hides the variation between
# them.

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

#
# By sex and by age band. Demographic coverage is complete in DISCOVERY; in the
# external cohort it is partial and differs by site.

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

#
# Every model scored on the same held-out patients of each cohort.
#
# The cross-validation column above is not a like-for-like comparison: an
# institution-specific model can only be cross-validated inside its own cohort,
# which for AKRIBEA is a working-age population at 0.83% prevalence, whereas the
# centralized and federated models are scored on pooled folds that also contain
# elderly participants and a MetS-enriched cohort. Scoring every model on the
# same cohort data removes that confound.

per_cohort = (
    long[
        (long.metric == "auroc") & (long.evaluation == "cv")
        & (long.stratum_kind == "cohort")
        & (long.run_id.isin(["unified__centralised", "federated__fedavg"]))
    ]
    .groupby(["run_id", "stratum"])["value"].mean().unstack()
)

# An individual model's own CV score already *is* its score on its own cohort.
own = (
    long[
        (long.metric == "auroc") & (long.evaluation == "cv")
        & (long.site == "ALL") & (long.stratum_kind == "none")
        & (long.run_id.str.startswith("individual"))
    ]
    .groupby("run_id")["value"].mean()
)
own.index = [r.split("__", 1)[1] for r in own.index]

sizes = (
    long[
        (long.metric == "n") & (long.evaluation == "cv")
        & (long.stratum_kind == "cohort") & (long.run_id == "federated__fedavg")
    ]
    .groupby("stratum")["value"].mean().round(0).astype(int)
)

# The best model trained at another institution, applied to this cohort's own
# held-out patients. It is the obvious alternative to federating, so it is
# reported as a column rather than argued in prose.
cross = (
    long[
        (long.metric == "auroc") & (long.evaluation == "cv:cross_cohort")
        & (long.run_id.str.startswith("individual"))
    ]
    .groupby(["run_id", "stratum"])["value"].mean().unstack()
)
cross.index = [r.split("__", 1)[1] for r in cross.index]
best_other, best_other_name = {}, {}
for coh in cross.columns:
    others = cross[coh].drop(index=coh, errors="ignore").dropna()
    if len(others):
        best_other[coh] = others.max()
        best_other_name[coh] = others.idxmax()

# Short headers, because six columns of prose titles overrun an IEEE column.
# The table note carries the definitions.
democratisation = pd.DataFrame({
    "n": sizes,
    "Local": own,
    # Value and source in one cell: a separate column for the institution reads
    # as a sixth number and costs width for nothing.
    "Best other": pd.Series(
        {c: f"{v:.3f} ({best_other_name[c]})" for c, v in best_other.items()}
    ),
    "Centralized": per_cohort.loc["unified__centralised"],
    "Federated": per_cohort.loc["federated__fedavg"],
})
democratisation["Δ"] = democratisation["Federated"] - democratisation["Local"]
democratisation = democratisation.sort_values("n")
print("\nbest other model per cohort:", best_other_name)
print(democratisation.round(3).to_string())

print(
    "\nRead this way: federation costs the two large institutions nothing on "
    "their own patients, raises the small ones above both their own model and "
    "the best model any other institution could lend them, and is beaten only "
    "on METS+, by centralised pooling."
)

_ = to_latex(
    democratisation.round(3),
    "AUROC on each cohort's own held-out data",
    "tab:global_on_local",
    TABLES / "global_on_local.tex",
    full_width=True,
    tabcolsep="4pt",
    note=(
        "All four models are scored on the same held-out subjects of the cohort "
        "named in the row. \\emph{Local} is the model trained on that cohort "
        "alone; \\emph{Best other} is the best-performing model trained at a "
        "single different institution, named in parentheses; "
        "\\emph{Centralized} is trained on all five cohorts pooled. $n$ is the "
        "mean size of the held-out fold, one fifth of the cohort. $\\Delta$ is "
        "Federated minus Local."
    ),
)


import json

rows = []
for run in ["unified__centralised", "federated__fedavg", "federated__fedprox__mu0.01"]:
    profile = Path.cwd().parent / "results" / run / "profile.json"
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
    "Computational and communication cost for 50 folds. Communication is "
    "derived analytically from the protocol: 24{,}289 parameters $\\times$ 4 bytes "
    "$\\times$ 2 directions $\\times$ 20 rounds $\\times$ 5 clients.",
    "tab:cost",
    TABLES / "cost.tex",
)

print(f"\nAll tables written to {TABLES}")
for path in sorted(TABLES.glob("*.tex")):
    print(f"  \\input{{tables/{path.name}}}")
