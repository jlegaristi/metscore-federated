# Results per centre and per institution, with bootstrap intervals over subjects:
#
#   - the aggregation weights applied to each client
#   - AUROC and calibration at each external site
#   - within-centre and between-centre AUROC
#   - paired differences on each cohort's own held-out patients
#
# Run scripts/run_experiments.py first:  python per_centre.py [resamples [paired]]

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from metscore import results, stats
from metscore.config import OUTPUT_DIR, RESULTS_DIR
from metscore.plotting import label_for
from metscore.results import to_latex, to_latex_grouped

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

TABLES = OUTPUT_DIR / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

# Resamples for the intervals, and for the paired differences. The latter also give
# p values, Holm-corrected over five cohorts; a bootstrap p value is at least
# 2 / (B + 1), so they need more.
RESAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
RESAMPLES_PAIRED = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000
EXTERNAL = "external_train_fitted"
SITES = ["BPH", "GRAZ", "TUEBINGEN"]
COHORTS = ["AKRIBEA", "OSARTEN", "LIVERBIBLE", "AGEPORTUGAL", "METS+"]
FEDERATED = "federated__fedavg"
CENTRALIZED = "unified__centralised"
APPROACHES = [CENTRALIZED, FEDERATED, "federated__fedprox__mu0.1", "federated__fedavg_uniform"]


def exists(run):
    return (RESULTS_DIR / run / "metrics.csv").exists()


def cell(point, low, high, digits=3):
    return f"{point:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def signed(point, low, high):
    return f"{point:+.3f} [{low:+.3f}, {high:+.3f}]"


def shown_p(value):
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def thousands(n):
    """20000 -> 20{,}000, as LaTeX wants it."""
    return f"{n:,}".replace(",", "{,}")

# Aggregation weights


def applied(run):
    return (
        pd.read_csv(RESULTS_DIR / run / "aggregation_weights.csv")
        .groupby("cohort")[["n_training", "n_after_downsampling", "weight_applied"]].mean()
    )


base = applied(FEDERATED)
weights = pd.DataFrame({
    "Training n": base.n_training.round(0).astype(int),
    "After downsampling": base.n_after_downsampling.round(0).astype(int),
    "FedAvg": (100 * base.weight_applied).round(1),
})
for run, name in [("federated__fedavg_cohort", "Cohort size"), ("federated__fedavg_uniform", "Uniform")]:
    if exists(run):
        weights[name] = (100 * applied(run).weight_applied).round(1)
weights = weights.loc[COHORTS]
print(weights.to_string())
two_largest = weights.loc[["AKRIBEA", "OSARTEN"], "FedAvg"].sum()
print(f"AKRIBEA + OSARTEN under FedAvg: {two_largest:.1f}%")
to_latex(
    weights, "Share of the aggregation received by each client (\\%), mean over folds.",
    "tab:weights", TABLES / "aggregation_weights.tex", digits=1, full_width=False, tabcolsep="3pt",
)

# Predictions of several runs on the same subjects


def external_folds(runs):
    """One HeldOut per fold, with the scores of each run on the external subjects."""
    frames = [results.load_predictions(run, EXTERNAL) for run in runs]
    folds = []
    for key, first in frames[0].groupby(["repetition", "fold"]):
        first = first.sort_values("subject_index")
        scores = [first.y_pred.to_numpy()]
        for other in frames[1:]:
            match = other[(other.repetition == key[0]) & (other.fold == key[1])].sort_values("subject_index")
            if not np.array_equal(match.subject_index.to_numpy(), first.subject_index.to_numpy()):
                raise ValueError(f"misaligned external subjects at {key}")
            scores.append(match.y_pred.to_numpy())
        folds.append(stats.HeldOut(
            first.subject_index.to_numpy().astype(int), first.y_true.to_numpy().astype(int),
            first.site.to_numpy(), np.vstack(scores),
        ))
    return folds


def cv_folds(runs):
    """One HeldOut per fold, with the scores of each run on the pooled test fold."""
    frames = [results.load_predictions(run, "cv") for run in runs]
    folds = []
    for key, first in frames[0].groupby(["repetition", "fold"]):
        first = first.sort_values("subject_index")
        scores = [first.y_pred.to_numpy()]
        for other in frames[1:]:
            match = other[(other.repetition == key[0]) & (other.fold == key[1])].sort_values("subject_index")
            if not np.array_equal(match.subject_index.to_numpy(), first.subject_index.to_numpy()):
                raise ValueError(f"misaligned cross-validation subjects at {key}")
            scores.append(match.y_pred.to_numpy())
        folds.append(stats.HeldOut(
            first.subject_index.to_numpy().astype(int), first.y_true.to_numpy().astype(int),
            first.site.to_numpy(), np.vstack(scores),
        ))
    return folds


def strata_of(folds):
    """Stratum (centre and label) of each subject, indexed by subject id."""
    size = 1 + max(int(f.subjects.max()) for f in folds)
    strata = np.empty(size, dtype=object)
    for fold in folds:
        strata[fold.subjects] = np.char.add(np.char.add(fold.site.astype(str), "|"), fold.y.astype(str))
    strata[strata == None] = "unused"  # noqa: E711
    return strata.astype(str)


def restrict(folds, site):
    out = []
    for fold in folds:
        mask = fold.site == site
        out.append(stats.HeldOut(fold.subjects[mask], fold.y[mask], fold.site[mask], fold.scores[:, mask]))
    return out


def expected_calibration(y, scores):
    from metscore.evaluation import expected_calibration_error

    return np.array([expected_calibration_error(y, row) for row in scores])

# External cohort, site by site


runs = [r for r in APPROACHES + [f"individual__{c}" for c in COHORTS] if exists(r)]
folds = external_folds(runs)
strata = strata_of(folds)

rows, headings = {}, {}
for site in SITES:
    site_folds = restrict(folds, site)
    site_strata = strata
    for metric, statistic in (
        ("AUROC", lambda y, s, x: stats._fast_auc(y, x)),
        ("ECE", lambda y, s, x: expected_calibration(y, x)),
        ("Mean pred.", lambda y, s, x: x.mean(axis=1)),
    ):
        observed, draws = stats.held_out_bootstrap(site_folds, site_strata, statistic, RESAMPLES)
        for i, run in enumerate(runs):
            low, high = np.quantile(draws[:, i], [0.025, 0.975])
            rows.setdefault(run, {})[f"{site} {metric}"] = cell(observed[i], low, high)
    y_site = site_folds[0].y
    print(f"{site}: n={len(y_site)}, observed prevalence {y_site.mean():.3f}")
    name = 'T\\"ubingen' if site == "TUEBINGEN" else site
    headings[site] = f"{name} ($n = {len(y_site)}$, prevalence {100 * y_site.mean():.1f}\\%)"

per_site = pd.DataFrame(rows).T
per_site.index = [label_for(r) for r in per_site.index]
print(per_site.to_string())
per_site.to_csv(TABLES / "external_per_site_all.csv")

shown = ["Centralized", "FedAvg", "AKRIBEA", "OSARTEN"]
blocks = []
for site in SITES:
    block = per_site.loc[shown, [f"{site} AUROC", f"{site} ECE"]]
    block.columns = ["AUROC [95\\% CI]", "ECE [95\\% CI]"]
    blocks.append((headings[site], [(None, block)]))
to_latex_grouped(
    blocks,
    "Discrimination and calibration at each external center",
    "tab:external_per_site", TABLES / "external_per_site.tex", full_width=False,
    note=(
        "Mean over the 50 fold models, with 95\\% intervals from resampling subjects "
        f"({thousands(RESAMPLES)} resamples). "
        "AKRIBEA and OSARTEN are the institution-specific models that transfer best "
        "(Table~\\ref{tab:results})."
    ),
)

# Within-centre and between-centre AUROC


def decomposition_table(all_folds, label):
    strata_local = strata_of(all_folds)
    names = [r for r in APPROACHES if exists(r)]

    def statistic(y, site, scores):
        pooled, within, between, _ = stats.decompose_auroc(y, scores, site)
        return np.concatenate([pooled, within, between])

    observed, draws = stats.held_out_bootstrap(all_folds, strata_local, statistic, RESAMPLES)
    k = len(names)
    share = np.mean([stats.decompose_auroc(f.y, f.scores, f.site)[3] for f in all_folds])
    out = {}
    for i, name in enumerate(names):
        low_w, high_w = np.quantile(draws[:, k + i], [0.025, 0.975])
        low_b, high_b = np.quantile(draws[:, 2 * k + i], [0.025, 0.975])
        out[label_for(name)] = {
            f"{label} pooled": f"{observed[i]:.3f}",
            f"{label} within": cell(observed[k + i], low_w, high_w),
            f"{label} between": cell(observed[2 * k + i], low_b, high_b),
        }
    print(f"{label}: {100 * share:.1f}% of the case/non-case pairs join different centres")
    return pd.DataFrame(out).T


keep = [r for r in APPROACHES if exists(r)]
index = [runs.index(r) for r in keep]
external_for_decomposition = [
    stats.HeldOut(f.subjects, f.y, f.site, f.scores[index]) for f in folds
]
decomposition = pd.concat(
    [decomposition_table(cv_folds(keep), "CV"), decomposition_table(external_for_decomposition, "External")],
    axis=1,
)
print(decomposition.to_string())
to_latex(
    decomposition,
    "AUROC separated by whether a case and a non-case come from the same center "
    "(\\emph{within}) or from different centers (\\emph{between}), with 95\\% intervals "
    "from resampling subjects.",
    "tab:decomposition", TABLES / "auroc_decomposition.tex", full_width=True, tabcolsep="3pt",
)

# Differences on each cohort's own held-out patients


def load_cohort(run, evaluation, cohort):
    frame = results.load_predictions(run, evaluation)
    return frame[frame.site == cohort]


def aligned_cohort(cohort, candidates):
    """HeldOut folds of one cohort; the federated model is the first row of scores."""
    frames = {name: load_cohort(*spec, cohort) for name, spec in candidates.items()}
    reference = frames[FEDERATED]
    ids = np.unique(reference.subject_index.to_numpy())
    folds = []
    for key, first in reference.groupby(["repetition", "fold"]):
        first = first.sort_values("subject_index")
        scores = [first.y_pred.to_numpy()]
        for name, frame in frames.items():
            if name == FEDERATED:
                continue
            match = frame[(frame.repetition == key[0]) & (frame.fold == key[1])].sort_values("subject_index")
            if not np.array_equal(match.y_true.to_numpy(), first.y_true.to_numpy()):
                raise ValueError(f"{cohort}: misaligned subjects for {name} at {key}")
            scores.append(match.y_pred.to_numpy())
        folds.append(stats.HeldOut(
            np.searchsorted(ids, first.subject_index.to_numpy()), first.y_true.to_numpy().astype(int),
            first.site.to_numpy(), np.vstack(scores),
        ))
    return folds, ids


gains, raw_local, raw_central, raw_other = [], {}, {}, {}
for cohort in COHORTS:
    long = results.collect("individual__*")
    cross = long[(long.metric == "auroc") & (long.evaluation == "cv:cross_cohort") & (long.stratum == cohort)]
    means = cross[cross.run_id != f"individual__{cohort}"].groupby("run_id")["value"].mean()
    best_other = means.idxmax()

    candidates = {
        FEDERATED: (FEDERATED, "cv"),
        "local": (f"individual__{cohort}", "cv"),
        CENTRALIZED: (CENTRALIZED, "cv"),
        "other": (best_other, f"cv_cross_{cohort}"),
    }
    cohort_folds, ids = aligned_cohort(cohort, candidates)
    y_by_id = np.empty(len(ids), dtype=object)
    for fold in cohort_folds:
        y_by_id[fold.subjects] = fold.y.astype(str)

    def statistic(y, site, scores):
        aucs = stats._fast_auc(y, scores)
        return np.concatenate([aucs, aucs[0] - aucs[1:]])

    observed, draws = stats.held_out_bootstrap(
        cohort_folds, y_by_id.astype(str), statistic, RESAMPLES_PAIRED
    )
    row = {"Cohort": cohort, "n": int(np.mean([len(f.y) for f in cohort_folds]))}
    names = ["Local", "Centralized", "Best other"]
    row.update({"Federated": f"{observed[0]:.3f}"})
    row.update({name: f"{observed[1 + i]:.3f}" for i, name in enumerate(names)})
    for i, (name, store) in enumerate(zip(names, (raw_local, raw_central, raw_other))):
        low, high, p_value = stats.interval_and_p(draws[:, 4 + i])
        store[cohort] = p_value
        row[f"Δ {name}"] = signed(observed[4 + i], low, high)
    row["Best other model"] = best_other.split("__", 1)[1]
    gains.append(row)

gains = pd.DataFrame(gains).set_index("Cohort")
for name, store in zip(["Local", "Centralized", "Best other"], (raw_local, raw_central, raw_other)):
    adjusted = stats.holm_bonferroni(store)
    gains[f"p {name}"] = [shown_p(adjusted[c]) for c in gains.index]
gains = gains.loc[COHORTS]
print(gains.to_string())
gains.to_csv(TABLES / "institution_differences_all.csv")

to_latex(
    gains[["n", "Local", "Federated", "Δ Local", "p Local", "Δ Centralized", "p Centralized"]].rename_axis(None),
    "AUROC on each cohort's own held-out patients",
    "tab:global_on_local", TABLES / "institution_differences.tex",
    full_width=True, tabcolsep="3pt",
    note=(
        "\\emph{Local} is the model trained only on that cohort and \\emph{Federated} is "
        "FedAvg, both scored on the same held-out patients; $n$ is the mean size of the "
        "held-out fold, one fifth of the cohort. Each $\\Delta$ is the difference in AUROC, "
        f"with a paired 95\\% interval from resampling subjects ({thousands(RESAMPLES_PAIRED)} "
        "resamples), not adjusted for multiplicity, and $p$ is Holm-corrected across the five "
        "cohorts."
    ),
)
print(f"\nAll tables written to {TABLES}")
