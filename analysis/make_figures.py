# Figures, saved as PDF and PNG. The style is in metscore/plotting.py.
#
# Run scripts/run_experiments.py first, then this script from analysis/.

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metscore import plotting, results, stats
from metscore.config import OUTPUT_DIR, RESULTS_DIR
from metscore.plotting import (
    AQUA, BLUE, CRITICAL, GRID, INK, INK_SOFT, MUTED, ORANGE, VIOLET,
    DOUBLE_COLUMN, SINGLE_COLUMN, label_for,
)

plotting.apply_style()

FIGURES = OUTPUT_DIR / "figures"
long = results.collect()
print(f"{long.run_id.nunique()} runs loaded; figures -> {FIGURES}")


def aligned(run_id, evaluation="external_train_fitted"):
    frame = results.load_predictions(run_id, evaluation)
    return (
        frame.groupby("subject_index")
        .agg(y=("y_true", "first"), p=("y_pred", "mean"), site=("site", "first"))
        .reset_index()
    )


def declutter(values, min_gap):
    """Move labels apart so that they do not overlap, keeping their order."""
    order = np.argsort(values)
    placed = np.array(values, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        if placed[b] - placed[a] < min_gap:
            placed[b] = placed[a] + min_gap
    return placed


def fold_means(metric="auroc", evaluation="cv"):
    sel = long[
        (long.metric == metric) & (long.evaluation == evaluation)
        & (long.site == "ALL") & (long.stratum_kind == "none")
    ]
    return sel.groupby("run_id")["value"]

# Forest plot: every approach against FedAvg (ensemble predictions, DeLong, Holm).


REFERENCE = "federated__fedavg"
COMPARATORS = [
    "individual__METS+", "individual__AGEPORTUGAL", "individual__LIVERBIBLE",
    "unified__centralised", "individual__AKRIBEA", "individual__OSARTEN",
]

reference = aligned(REFERENCE)
rows, raw_p = [], {}
for run in COMPARATORS:
    other = aligned(run)
    merged = reference.merge(other, on="subject_index", suffixes=("_r", "_o"))
    test = stats.delong_test(merged.y_r.values, merged.p_o.values, merged.p_r.values)
    raw_p[run] = test.p_value
    rows.append({"run": run, "delta": test.difference,
                 "lo": test.ci_low, "hi": test.ci_high, "auc": test.auc_a})

forest = pd.DataFrame(rows)
forest["p_holm"] = forest["run"].map(stats.holm_bonferroni(raw_p))
forest["significant"] = forest["p_holm"] < 0.05

fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN * 0.62, 2.5))
y = np.arange(len(forest))

ax.axvline(0, color=MUTED, linewidth=0.8, zorder=1)
for i, row in forest.iterrows():
    colour = ORANGE if row.significant else MUTED
    ax.plot([row.lo, row.hi], [i, i], color=colour, linewidth=2, solid_capstyle="round", zorder=2)
    ax.plot(row.delta, i, "o", color=colour, markersize=6,
            markeredgecolor="white", markeredgewidth=1.2, zorder=3)

ax.set_yticks(y)
ax.set_yticklabels([label_for(r) for r in forest.run])
ax.set_xlabel("Δ AUROC versus federated model  (95% CI)")
ax.set_ylim(-0.6, len(forest) - 0.4)
ax.grid(axis="y", visible=False)

# AUROC next to each interval, with an asterisk when significant
for i, row in forest.iterrows():
    ax.annotate(
        f"{row.auc:.3f}" + ("  *" if row.significant else ""),
        xy=(row.hi, i), xytext=(6, 0), textcoords="offset points",
        va="center", fontsize=7, color=INK_SOFT,
    )

# differences are comparator minus federated
ax.annotate("← federated better", xy=(0, -0.45), xytext=(-6, 0),
            textcoords="offset points", fontsize=6.5, color=MUTED, ha="right")
ax.annotate("comparator better →", xy=(0, -0.45), xytext=(6, 0),
            textcoords="offset points", fontsize=6.5, color=MUTED, ha="left")
ax.set_title("* Holm-corrected p < 0.05", fontsize=7, color=MUTED, loc="right", pad=6)

plotting.save(fig, "fig_pairwise_comparison")
plt.show()

# Privacy-utility curve: cross-validation AUROC against epsilon (log scale). The
# band marks epsilon below 10.

import json

dp_rows = []
reference_auroc = None
for path in RESULTS_DIR.glob("federated__dp_fedavg*/profile.json"):
    dp = json.loads(path.read_text()).get("differential_privacy")
    if not dp:
        continue
    run = path.parent.name
    if run.endswith("__reference"):
        reference_auroc = fold_means().mean().get(run, np.nan)
        continue
    dp_rows.append({
        "noise": dp["noise_multiplier"],
        "epsilon": dp["epsilon"],
        "auroc": fold_means().mean().get(run, np.nan),
    })

dp = pd.DataFrame(dp_rows).sort_values("epsilon")
baseline = reference_auroc

fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, 2.4))

ax.axhspan(0.45, 0.55, color=GRID, alpha=0.6, zorder=0)
ax.axhline(baseline, color=MUTED, linestyle=":", linewidth=1.2, zorder=1)
ax.axvspan(dp.epsilon.min() * 0.5, 10, color=AQUA, alpha=0.10, zorder=0)

ax.plot(dp.epsilon, dp.auroc, "-o", color=VIOLET, markeredgecolor="white",
        markeredgewidth=1.2, zorder=3)
for _, row in dp.iterrows():
    ax.annotate(f"σ={row.noise:g}", xy=(row.epsilon, row.auroc), xytext=(0, 8),
                textcoords="offset points", fontsize=6.5, color=INK_SOFT, ha="center")

ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel("Privacy budget ε  (δ = 10⁻⁵, stronger →)")
ax.set_ylabel("Cross-validation AUROC")
ax.set_ylim(0.42, 0.98)
ax.annotate("no differential privacy", xy=(dp.epsilon.max(), baseline), xytext=(0, 5),
            textcoords="offset points", fontsize=6.5, color=MUTED)
ax.annotate("chance", xy=(dp.epsilon.max(), 0.50), xytext=(0, -10),
            textcoords="offset points", fontsize=6.5, color=MUTED)
ax.annotate("meaningful\nguarantee", xy=(4, 0.93), fontsize=6.5, color=AQUA,
            ha="center", va="top", fontweight="medium")

plotting.save(fig, "fig_privacy_utility")
plt.show()

# External AUROC at each validation site.

SHOWN = ["unified__centralised", "federated__fedavg",
         "individual__AKRIBEA", "individual__METS+"]
COLOURS = [BLUE, ORANGE, AQUA, VIOLET]
NAMES = ["Centralized", "Federated (FedAvg)", "AKRIBEA only", "METS+ only"]

per_site = (
    long[
        (long.metric == "auroc") & (long.evaluation == "external:train_fitted")
        & (long.site != "ALL") & (long.stratum_kind == "none")
        & (long.run_id.isin(SHOWN))
    ]
    .groupby(["run_id", "site"])["value"].mean().unstack()
    .loc[SHOWN, ["BPH", "GRAZ", "TUEBINGEN"]]
)

fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, 2.5))
sites = per_site.columns
x = np.arange(len(sites))
width = 0.2

for i, (run, colour, name) in enumerate(zip(SHOWN, COLOURS, NAMES)):
    offset = (i - 1.5) * (width + 0.015)
    ax.bar(x + offset, per_site.loc[run], width, label=name,
           color=colour, edgecolor="white", linewidth=0.8, zorder=2)

ax.set_xticks(x)
ax.set_xticklabels(["BPH", "GRAZ", "TÜBINGEN"])

# size and prevalence of each site, above its bars
for pos, n, prev in zip(x, [324, 279, 58], ["9.6%", "2.9%", "77.6%"]):
    ax.annotate(f"n={n}\n{prev} MetS+", xy=(pos, 0.995), ha="center", va="top",
                fontsize=6.5, color=MUTED, linespacing=1.3)
ax.set_ylabel("External AUROC")
ax.set_ylim(0.5, 1.0)
ax.axhline(0.5, color=MUTED, linewidth=0.8)
ax.grid(axis="x", visible=False)
ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.28), columnspacing=1.2)

plotting.save(fig, "fig_per_site")
plt.show()

# Train-fitted against site-adapted scaling, as a slope chart.

contract = (
    long[
        (long.metric == "auroc") & (long.site == "ALL") & (long.stratum_kind == "none")
        & (long.evaluation.str.startswith("external"))
    ]
    .groupby(["run_id", "evaluation"])["value"].mean().unstack()
)
SLOPE = ["unified__centralised", "federated__fedavg",
         "individual__AKRIBEA", "individual__OSARTEN"]

fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, 2.6))
left = [contract.loc[r, "external:train_fitted"] for r in SLOPE]
right = [contract.loc[r, "external:site_adapted"] for r in SLOPE]
left_at = declutter(left, 0.013)
right_at = declutter(right, 0.013)

for run, colour, lo, hi, la, ra in zip(SLOPE, COLOURS, left, right, left_at, right_at):
    ax.plot([0, 1], [lo, hi], "-o", color=colour, markeredgecolor="white",
            markeredgewidth=1.1, zorder=3)
    ax.annotate(label_for(run), xy=(0, la), xytext=(-6, 0), textcoords="offset points",
                ha="right", va="center", fontsize=7, color=colour, fontweight="medium")
    ax.annotate(f"−{lo - hi:.2f}", xy=(1, ra), xytext=(6, 0), textcoords="offset points",
                ha="left", va="center", fontsize=7, color=INK_SOFT)

ax.set_xticks([0, 1])
ax.set_xticklabels(["Train-fitted\n(deployable)", "Site-adapted\n(re-standardized)"])
ax.set_xlim(-0.75, 1.4)
ax.set_ylabel("External AUROC")
ax.grid(axis="x", visible=False)

plotting.save(fig, "fig_scaling_contract")
plt.show()

# Reliability curves on the external cohort, quantile bins.

from metscore.evaluation import calibration_curve

CALIB = ["unified__centralised", "federated__fedavg", "federated__fedavg_cohort"]

fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, 2.6))
ax.plot([0, 1], [0, 1], linestyle=":", color=MUTED, linewidth=1, zorder=1)

curves = {run: calibration_curve(aligned(run).y.values, aligned(run).p.values, n_bins=8)
          for run in CALIB}
ends = declutter([curves[r]["observed"][-1] for r in CALIB], 0.075)

for run, colour, marker, end in zip(CALIB, [BLUE, ORANGE, VIOLET], ["o", "s", "D"], ends):
    curve = curves[run]
    ax.plot(curve["predicted"], curve["observed"], marker=marker, color=colour,
            markeredgecolor="white", markeredgewidth=1, label=label_for(run), zorder=3)
    plotting.annotate_last(ax, curve["predicted"][-1], end, label_for(run), colour)

ax.set_xlabel("Predicted probability")
ax.set_ylabel("Observed frequency")
ax.set_xlim(0, 1.45)
ax.set_ylim(0, 1.05)
# the axis extends past 1 to leave room for the labels
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.annotate("perfect calibration", xy=(0.62, 0.62), rotation=45, fontsize=6.5,
            color=MUTED, ha="center", va="bottom")

plotting.save(fig, "fig_calibration")
plt.show()

# Cross-validation AUROC by age band.

BANDS = ["25-34", "35-44", "45-54", "55-64", "65-69", "70-79", "80+"]
age = (
    long[
        (long.metric == "auroc") & (long.evaluation == "cv")
        & (long.stratum_kind == "age_band")
        & (long.run_id.isin(["unified__centralised", "federated__fedavg"]))
    ]
    .groupby(["run_id", "stratum"])["value"].mean().unstack()
)
age = age[[b for b in BANDS if b in age.columns]]

fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN * 0.5, 2.3))
for run, colour, marker in [("unified__centralised", BLUE, "o"),
                            ("federated__fedavg", ORANGE, "s")]:
    if run not in age.index:
        continue
    ax.plot(range(age.shape[1]), age.loc[run], marker=marker, color=colour,
            markeredgecolor="white", markeredgewidth=1, label=label_for(run), zorder=3)
    plotting.annotate_last(ax, age.shape[1] - 1, age.loc[run].iloc[-1],
                           label_for(run), colour)

ax.set_xticks(range(age.shape[1]))
ax.set_xticklabels(age.columns)
ax.set_xlabel("Age band (years)")
ax.set_ylabel("Cross-validation AUROC")
ax.set_xlim(-0.3, age.shape[1] + 0.9)
ax.set_ylim(0.7, 1.02)

plotting.save(fig, "fig_age_stratification")
plt.show()

print("Figures written:")
for path in sorted(FIGURES.glob("*.pdf")):
    print(f"  \\includegraphics[width=\\columnwidth]{{figures/{path.name}}}")

# Calibration error and training time of each strategy, as two dot plots sharing
# the y axis.

import json

FAMILY = {
    "unified__centralised": BLUE,
    "federated__fedavg": ORANGE, "federated__fedavg_uniform": ORANGE,
    "federated__fedmedian": ORANGE, "federated__fedtrimmed": ORANGE,
    "federated__fedprox__mu0.01": VIOLET, "federated__fedprox__mu0.1": VIOLET,
    "federated__fedprox__mu1.0": VIOLET,
}


def _ext(run, metric):
    sel = long[
        (long.run_id == run) & (long.metric == metric)
        & (long.evaluation == "external:train_fitted")
        & (long.site == "ALL") & (long.stratum_kind == "none")
    ]
    return sel.value.mean()


cost = pd.DataFrame([
    {
        "run": run,
        "name": label_for(run),
        "colour": colour,
        "hours": json.loads(
            (results.RESULTS_DIR / run / "profile.json").read_text()
        )["wall_clock_s"] / 3600,
        "ece": _ext(run, "ece"),
        "auroc": _ext(run, "auroc"),
    }
    for run, colour in FAMILY.items()
]).sort_values("ece", ascending=False).reset_index(drop=True)

fig, axes = plt.subplots(
    1, 2, figsize=(SINGLE_COLUMN, 2.5), sharey=True,
    gridspec_kw={"wspace": 0.0},
)
y = np.arange(len(cost))

# calibration grows to the left and time to the right, from a common axis
for ax, column, label, limit, fmt, side, ticks in (
    (axes[0], "ece", "Calibration error", (0.10, 0.46), "{:.3f}", -1, [0.2, 0.3, 0.4]),
    (axes[1], "hours", "Training time (h)", (0.0, 3.4), "{:.2f}", +1, [1, 2, 3]),
):
    for k, row in cost.iterrows():
        ax.plot([limit[0], row[column]], [k, k], color=row.colour,
                linewidth=1.1, alpha=0.35, solid_capstyle="round", zorder=2)
        ax.plot(row[column], k, "o", color=row.colour, markersize=5,
                markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        ax.annotate(fmt.format(row[column]), xy=(row[column], k),
                    xytext=(5 * side, 0), textcoords="offset points",
                    va="center", ha="left" if side > 0 else "right",
                    fontsize=5.4, color=INK_SOFT)
    ax.set_xlim(*limit)
    # no tick at the common edge
    ax.set_xticks(ticks)
    ax.set_xlabel(label)
    ax.grid(axis="y", visible=False)

axes[0].invert_xaxis()

# the two panels touch; the spine between them is the common axis
axes[0].spines["right"].set_visible(True)
axes[0].spines["right"].set_color(INK_SOFT)
axes[0].spines["right"].set_linewidth(1.1)
axes[1].spines["left"].set_visible(False)
for k, row in cost.iterrows():
    # tick on the common axis, in the colour of the model
    axes[0].plot(0.10, k, "|", color=row.colour, markersize=7,
                 markeredgewidth=1.4, clip_on=False, zorder=4)
axes[0].set_yticks(y)
axes[0].set_yticklabels(cost.name)
for tick, colour in zip(axes[0].get_yticklabels(), cost.colour):
    tick.set_color(colour)
axes[0].set_ylim(-0.7, len(cost) - 0.3)
axes[0].tick_params(axis="y", length=0, labelsize=6.2)
axes[1].tick_params(axis="y", length=0)

plotting.save(fig, "fig_cost_calibration")
plt.show()
