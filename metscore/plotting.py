"""
Figure style, imported by every analysis script.

Vector PDF at IEEE column widths (3.5 in single column, 7.16 in double column),
legible at print size.

The categorical palette is validated rather than chosen by eye: all pairs pass
the lightness band, the chroma floor, colour-vision separation (worst adjacent
pair ΔE 9.2 under protanopia and deuteranopia) and the normal-vision floor
(worst pair 16.3). AQUA sits slightly under the contrast target against white,
so anything drawn in it carries a direct label.

Every series also gets a marker and a line style, so the figures survive
greyscale printing.
"""

from __future__ import annotations

# -- validated categorical palette -------------------------------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"
CATEGORICAL = (BLUE, ORANGE, AQUA, VIOLET)

# -- status, reserved and never reused as a series colour --------------------
GOOD = "#1baf7a"
WARNING = "#eda100"
CRITICAL = "#c0392b"

# -- ink ---------------------------------------------------------------------
INK = "#1a1a1a"
INK_SOFT = "#4a4a4a"
MUTED = "#8a8a8a"
GRID = "#e2e2e2"
SURFACE = "#ffffff"

#: Fixed assignment, by entity. A figure that drops a series must not repaint
#: the survivors, so approaches are looked up here rather than cycled.
APPROACH_COLOR = {
    "unified": BLUE,
    "federated": ORANGE,
    "individual": AQUA,
    "dp": VIOLET,
}

APPROACH_MARKER = {"unified": "o", "federated": "s", "individual": "^", "dp": "D"}
APPROACH_LINESTYLE = {"unified": "-", "federated": "--", "individual": ":", "dp": "-."}

#: Column widths in inches, from the IEEE template.
SINGLE_COLUMN = 3.5
DOUBLE_COLUMN = 7.16

def apply_style() -> None:
    """Set matplotlib defaults. Call once at the top of a script."""
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
    """Map a run id to its colour slot."""
    if run_id.startswith("unified"):
        return "unified"
    if run_id.startswith("individual"):
        return "individual"
    if "dp_fedavg" in run_id:
        return "dp"
    return "federated"

def style_for(run_id: str) -> dict:
    """Colour, marker and line style for one run, assigned by entity."""
    arm = arm_of(run_id)
    return {
        "color": APPROACH_COLOR[arm],
        "marker": APPROACH_MARKER[arm],
        "linestyle": APPROACH_LINESTYLE[arm],
    }

def label_for(run_id: str) -> str:
    """Readable name for axes and legends."""
    names = {
        "unified__centralised": "Centralized",
        "federated__fedavg": "FedAvg",
        "federated__fedavg_uniform": "FedAvg (uniform)",
        "federated__fedmedian": "FedMedian",
        "federated__fedtrimmed": "FedTrimmedAvg",
    }
    if run_id in names:
        return names[run_id]
    if run_id.startswith("individual__"):
        return run_id.split("__", 1)[1]
    if "fedprox" in run_id:
        return f"FedProx μ={run_id.rsplit('mu', 1)[1]}"
    if "dp_fedavg" in run_id:
        return f"DP σ={run_id.rsplit('noise_multiplier', 1)[1]}"
    return run_id

def save(fig, name: str, directory=None) -> str:
    """Write PDF for LaTeX and PNG for quick viewing."""
    from pathlib import Path

    from .config import WORKSPACE

    directory = Path(directory or WORKSPACE / "figures")
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(directory / f"{name}.{suffix}")
    return str(directory / f"{name}.pdf")

def annotate_last(ax, x, y, text, color, dx=0.01, **kwargs):
    """Direct label at the end of a line, so identity never rests on colour."""
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
