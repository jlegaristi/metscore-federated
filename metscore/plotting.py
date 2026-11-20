"""Figure style shared by the analysis scripts.

Figures are saved as PDF at IEEE column widths. The four colours are a
colour-blind-safe set, and every series also has its own marker and line style so
that the figures can be read in greyscale.
"""

from __future__ import annotations

# colours of the series
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"
CATEGORICAL = (BLUE, ORANGE, AQUA, VIOLET)

# colours with a meaning, not used for series
GOOD = "#1baf7a"
WARNING = "#eda100"
CRITICAL = "#c0392b"

# text, axes and grid
INK = "#1a1a1a"
INK_SOFT = "#4a4a4a"
MUTED = "#8a8a8a"
GRID = "#e2e2e2"
SURFACE = "#ffffff"

# One colour per approach, the same in every figure.
APPROACH_COLOR = {
    "unified": BLUE,
    "federated": ORANGE,
    "individual": AQUA,
    "dp": VIOLET,
}

APPROACH_MARKER = {"unified": "o", "federated": "s", "individual": "^", "dp": "D"}
APPROACH_LINESTYLE = {"unified": "-", "federated": "--", "individual": ":", "dp": "-."}

# IEEE column widths, in inches
SINGLE_COLUMN = 3.5
DOUBLE_COLUMN = 7.16


def apply_style() -> None:
    """Set the matplotlib defaults."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "text.color": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })


def arm_of(run_id: str) -> str:
    """The approach a run belongs to."""
    if run_id.startswith("unified"):
        return "unified"
    if run_id.startswith("individual"):
        return "individual"
    if "dp_fedavg" in run_id:
        return "dp"
    return "federated"


def style_for(run_id: str) -> dict:
    """Colour, marker and line style of a run."""
    arm = arm_of(run_id)
    return {
        "color": APPROACH_COLOR[arm],
        "marker": APPROACH_MARKER[arm],
        "linestyle": APPROACH_LINESTYLE[arm],
    }


def label_for(run_id: str) -> str:
    """Name of a run as shown in tables and figures."""
    names = {
        "unified__centralised": "Centralized",
        "unified__centralised__cohort-downsampling": "Centralized (cohort downsampling)",
        "federated__fedavg": "FedAvg",
        "federated__fedavg_uniform": "FedAvg (uniform)",
        "federated__fedavg_cohort": "FedAvg (cohort size)",
        "federated__fedmedian": "FedMedian",
        "federated__fedtrimmed": "FedTrimmedAvg",
    }
    if run_id in names:
        return names[run_id]
    if run_id.startswith("individual__"):
        return run_id.split("__", 1)[1]
    base, _, variant = run_id.partition("__adam-persistent")
    suffix = " (persistent Adam)" if "adam-persistent" in run_id else ""
    if base == "federated__fedavg":
        return f"FedAvg{suffix}"
    if "fedprox" in base:
        return f"FedProx μ={base.rsplit('mu', 1)[1]}{suffix}"
    if run_id.endswith("__reference"):
        return "Private architecture, no noise"
    if "dp_fedavg" in run_id:
        return f"DP σ={run_id.rsplit('noise_multiplier', 1)[1]}"
    return run_id


def save(fig, name: str, directory=None) -> str:
    """Save a figure as PDF and PNG."""
    from pathlib import Path

    from .config import OUTPUT_DIR

    directory = Path(directory or OUTPUT_DIR / "figures")
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(directory / f"{name}.{suffix}")
    return str(directory / f"{name}.pdf")


def annotate_last(ax, x, y, text, color, dx=0.01, **kwargs):
    """Label a line at its last point."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(4, 0),
        textcoords="offset points",
        color=color,
        va="center",
        fontsize=7.5,
        fontweight="medium",
        **kwargs,
    )
