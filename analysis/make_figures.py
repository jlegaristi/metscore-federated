# Figures, as vector PDF for LaTeX with a PNG alongside for quick viewing.
# Style lives in metscore.plotting, so every figure shares one palette and one
# set of defaults. Each series carries a marker and a line style as well as a
# colour, so the figures survive greyscale printing.

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metscore import plotting, results, stats
from metscore.plotting import (
    AQUA, BLUE, CRITICAL, GRID, INK, INK_SOFT, MUTED, ORANGE, VIOLET,
    DOUBLE_COLUMN, SINGLE_COLUMN, label_for,
)

plotting.apply_style()

FIGURES = Path.cwd().parent / "figures"
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
    """Nudge labels apart while keeping their order, so none overlap."""
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

# Pairwise comparison against the federated model, with Holm-corrected
# significance. One series, so no legend is needed.

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

# Direct labels: significance is stated in words, never by colour alone.
for i, row in forest.iterrows():
    ax.annotate(
        f"{row.auc:.3f}" + ("  *" if row.significant else ""),
        xy=(row.hi, i), xytext=(6, 0), textcoords="offset points",
        va="center", fontsize=7, color=INK_SOFT,
    )

# The difference is comparator minus federated, so negative favours federation.
ax.annotate("← federated better", xy=(0, -0.45), xytext=(-6, 0),
            textcoords="offset points", fontsize=6.5, color=MUTED, ha="right")
ax.annotate("comparator better →", xy=(0, -0.45), xytext=(6, 0),
            textcoords="offset points", fontsize=6.5, color=MUTED, ha="left")
ax.set_title("* Holm-corrected p < 0.05", fontsize=7, color=MUTED, loc="right", pad=6)

plotting.save(fig, "fig_pairwise_comparison")
plt.show()

# Privacy and utility: epsilon against the noise the model survives, on a
# log-scaled axis. The shaded band marks the range in which a guarantee is
# conventionally considered meaningful.

import json

dp_rows = []
for path in sorted(
    (Path.cwd().parent / "results").glob("federated__dp_fedavg*/profile.json"),
    key=lambda p: float(p.parent.name.rsplit("noise_multiplier", 1)[1]),
):
    dp = json.loads(path.read_text()).get("differential_privacy")
    if not dp:
        continue
    run = path.parent.name
    dp_rows.append({
        "noise": dp["noise_multiplier"],
        "epsilon": dp["epsilon"],
        "auroc": fold_means().mean().get(run, np.nan),
    })

dp = pd.DataFrame(dp_rows).sort_values("epsilon")
baseline = fold_means().mean()["federated__fedavg"]

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

# External performance by sub-cohort. The three validation sites differ in
# prevalence: 9.6%, 2.9% and 77.6%.

SHOWN = ["unified__centralised", "federated__fedavg",
         "individual__AKRIBEA", "individual__METS+"]
COLOURS = [BLUE, ORANGE, AQUA, VIOLET]
# No hatching: four hues already separate the series. The labels name the
# model, since "AKRIBEA" alone is ambiguous in a chart whose x axis is also
# cohorts.
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

# Cohort size and prevalence sit above each group: as tick sublabels they are
# wider than the group spacing and collide.
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

# The scaling contract, as a slope chart, so that the pairing between the two
# contracts is what the reader sees.

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

# Calibration. Averaging models trained on cohorts whose prevalence ranges from
# 0.83% to 63.95% produces probabilities that are not on a common scale.
# Quantile bins, because predictions pile up near zero at 2% prevalence.

from metscore.evaluation import calibration_curve

CALIB = ["unified__centralised", "federated__fedavg", "federated__fedprox__mu0.1"]

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
# Space to the right is for the direct labels, not for impossible probabilities.
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.annotate("perfect calibration", xy=(0.62, 0.62), rotation=45, fontsize=6.5,
            color=MUTED, ha="center", va="bottom")

plotting.save(fig, "fig_calibration")
plt.show()

# Performance by age band.
#
# bands, and the elderly are the population the model would most often serve.

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

# What each aggregation rule costs and what it buys. A two-panel dot plot
# rather than a scatter, because eight labelled points would need eight
# hand-placed annotations and the ordering reads better as a column.

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
        # Shortened for the gutter: the full names do not fit between the
        # panels at single-column width, and the caption carries the detail.
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

# Both metrics radiate from the shared inner edge, so each model reads as one
# line. The names sit outside the pair, on the far left, which keeps the centre
# free and lets the whole thing fit a single column.
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
    # No tick at the shared edge: the two zeros would print on top of each other.
    ax.set_xticks(ticks)
    ax.set_xlabel(label)
    ax.grid(axis="y", visible=False)

axes[0].invert_xaxis()

# The panels are flush, so each model's two stems meet and read as one line.
# The join is marked by the shared spine, which is the only vertical rule in
# the figure and so needs no legend: left of it is calibration, right is time.
axes[0].spines["right"].set_visible(True)
axes[0].spines["right"].set_color(INK_SOFT)
axes[0].spines["right"].set_linewidth(1.1)
axes[1].spines["left"].set_visible(False)
for k, row in cost.iterrows():
    # A small tick on the centre line, in the model's colour, so the row is
    # traceable across the join.
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
