"""Where results go, and how they come back out.

Comparing approaches and pulling numbers into the paper should each be one
line, and neither should require knowing how a particular arm was implemented. Every run writes the same shape, whichever arm, strategy and
scaling contract produced it:

    results/
      index.csv                       one row per run: what it was, when, how long
      <run_id>/
        config.json                   the full ExperimentConfig, for provenance
        metrics.csv                   LONG format, one row per measurement
        profile.json                  wall-clock, compute, communication
        predictions/rep{r}_fold{f}.npz  y_true, y_pred, subject metadata

`metrics.csv` is long rather than wide on purpose. Every measurement is one row
tagged with arm, strategy, scaling contract, evaluation set, site, stratum,
repetition and fold, so a new axis (another strategy, another noise level,
another stratification) adds rows and breaks nothing downstream. Wide tables
for the manuscript are produced by pivoting at the end, in `paper_table`.

Predictions are kept per fold because the pairwise tests need subject-level
scores from two approaches on the same subjects.
Aggregated metrics cannot support DeLong.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RESULTS_DIR, ExperimentConfig
from .evaluation import Metrics

#: Column order for the long metrics table.
LONG_COLUMNS = [
    "run_id", "arm", "strategy", "scaling", "evaluation", "site",
    "stratum_kind", "stratum", "repetition", "fold", "metric", "value",
]

def make_run_id(cfg: ExperimentConfig) -> str:
    """Deterministic, readable, and stable across re-runs.

    Re-running the same configuration overwrites its folder rather than
    accumulating timestamped duplicates, which leave several near-identical
    folders and no way to tell which produced the reported numbers.
    """
    # The scaling contract is deliberately not part of the id: every run records
    # both contracts, so it identifies an evaluation, not a run.
    parts = [cfg.arm, cfg.strategy]
    if cfg.strategy_params:
        parts.append(
            "_".join(f"{k}{v}" for k, v in sorted(cfg.strategy_params.items()))
        )
    return "__".join(str(p) for p in parts)

@dataclass
class RunWriter:
    """Accumulates one run's output, then flushes it atomically."""

    config: ExperimentConfig
    run_id: str = ""
    _rows: list[dict] = None
    _started: datetime = None

    def __post_init__(self):
        self.run_id = self.run_id or make_run_id(self.config)
        self._rows = []
        self._started = datetime.now()
        self.path.mkdir(parents=True, exist_ok=True)

        # Start clean. A re-run that produces fewer folds than its predecessor
        # would otherwise leave stale .npz files behind, and `load_predictions`
        # globs the directory, so a later analysis would silently mix results
        # from two different versions of the code. metrics.csv is overwritten
        # wholesale at close, but predictions are written file by file.
        predictions = self.path / "predictions"
        if predictions.exists():
            for stale in predictions.glob("*.npz"):
                stale.unlink()
        predictions.mkdir(exist_ok=True)

        # An interrupted run must not look finished: remove the completion
        # marker up front and let close() write it again.
        metrics = self.path / "metrics.csv"
        if metrics.exists():
            metrics.unlink()

    @property
    def path(self) -> Path:
        return RESULTS_DIR / self.run_id

    # -- recording ---------------------------------------------------------

    def record(
        self,
        metrics: Metrics,
        evaluation: str,
        repetition: int,
        fold: int,
        site: str = "ALL",
        stratum_kind: str = "none",
        stratum: str = "ALL",
    ) -> None:
        """Add one evaluation's metrics as long-format rows."""
        base = {
            "run_id": self.run_id,
            "arm": self.config.arm,
            "strategy": self.config.strategy,
            "scaling": self.config.scaling.value,
            "evaluation": evaluation,
            "site": site,
            "stratum_kind": stratum_kind,
            "stratum": stratum,
            "repetition": repetition,
            "fold": fold,
        }
        for name, value in metrics.to_row().items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                self._rows.append({**base, "metric": name, "value": float(value)})

    def record_predictions(
        self,
        repetition: int,
        fold: int,
        evaluation: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        site: np.ndarray | None = None,
        subject_index: np.ndarray | None = None,
    ) -> None:
        """Persist subject-level scores so pairwise tests remain possible later.

        ``subject_index`` is the row index into the discovery/external table. It
        is what lets two runs be aligned to the same subjects for DeLong without
        assuming both iterated in the same order.
        """
        np.savez_compressed(
            self.path / "predictions" / f"{evaluation}_rep{repetition}_fold{fold}.npz",
            y_true=np.asarray(y_true),
            y_pred=np.asarray(y_pred),
            site=np.asarray(site if site is not None else []),
            subject_index=np.asarray(
                subject_index if subject_index is not None else []
            ),
        )

    # -- flushing ----------------------------------------------------------

    def close(self, profile: dict | None = None, notes: str = "") -> Path:
        if not self._rows:
            raise RuntimeError(f"{self.run_id}: nothing recorded")

        frame = pd.DataFrame(self._rows)[LONG_COLUMNS]
        frame.to_csv(self.path / "metrics.csv", index=False)
        self.config.save(self.path / "config.json")

        elapsed = (datetime.now() - self._started).total_seconds()
        if profile is not None:
            (self.path / "profile.json").write_text(json.dumps(profile, indent=2))

        _update_index(
            {
                "run_id": self.run_id,
                "arm": self.config.arm,
                "strategy": self.config.strategy,
                "scaling": self.config.scaling.value,
                "params": json.dumps(self.config.strategy_params, sort_keys=True),
                "finished": datetime.now().isoformat(timespec="seconds"),
                "elapsed_s": round(elapsed, 1),
                "n_measurements": len(frame),
                "notes": notes,
            }
        )
        return self.path

def _update_index(row: dict) -> None:
    """Keep one row per run_id, most recent wins."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    index_path = RESULTS_DIR / "index.csv"

    frame = pd.DataFrame([row])
    if index_path.exists():
        existing = pd.read_csv(index_path)
        existing = existing[existing["run_id"] != row["run_id"]]
        frame = pd.concat([existing, frame], ignore_index=True)

    frame.sort_values(["arm", "strategy", "scaling"]).to_csv(index_path, index=False)

# ---------------------------------------------------------------------------
# Reading back
# ---------------------------------------------------------------------------

def collect(pattern: str = "*") -> pd.DataFrame:
    """Every measurement from every run, in one long table.

    This is the entry point for the analysis scripts: they read this and
    nothing else, and never reload a model or recompute a metric.
    """
    frames = [
        pd.read_csv(path)
        for path in sorted(RESULTS_DIR.glob(f"{pattern}/metrics.csv"))
    ]
    if not frames:
        raise FileNotFoundError(f"no runs matching {pattern!r} under {RESULTS_DIR}")
    return pd.concat(frames, ignore_index=True)

def load_predictions(run_id: str, evaluation: str = "external") -> pd.DataFrame:
    """Subject-level scores for one run, concatenated across folds.

    Feed two of these to `stats.delong_test` after aligning on
    ``subject_index`` to compare approaches on identical subjects.
    """
    rows = []
    for path in sorted((RESULTS_DIR / run_id / "predictions").glob(f"{evaluation}_rep*.npz")):
        # Split from the right: evaluation names contain underscores of their
        # own ("external_train_fitted"), so splitting from the left mis-parses.
        _, rep, fold = path.stem.rsplit("_", 2)
        payload = np.load(path, allow_pickle=True)
        n = len(payload["y_true"])
        rows.append(
            pd.DataFrame({
                "repetition": int(rep.removeprefix("rep")),
                "fold": int(fold.removeprefix("fold")),
                "y_true": payload["y_true"],
                "y_pred": payload["y_pred"],
                "site": payload["site"] if payload["site"].size == n else "ALL",
                "subject_index": (
                    payload["subject_index"]
                    if payload["subject_index"].size == n
                    else np.arange(n)
                ),
            })
        )
    if not rows:
        raise FileNotFoundError(f"no {evaluation} predictions for run {run_id!r}")
    return pd.concat(rows, ignore_index=True)

def summarise(
    long: pd.DataFrame,
    metrics: tuple[str, ...] = ("auroc", "pr_auc", "sensitivity", "specificity", "ppv"),
    by: tuple[str, ...] = ("arm", "strategy", "scaling", "evaluation"),
) -> pd.DataFrame:
    """Mean and SD across folds, ready to read."""
    subset = long[
        (long["metric"].isin(metrics))
        & (long["stratum_kind"] == "none")
        & (long["site"] == "ALL")
    ]
    out = (
        subset.groupby([*by, "metric"])["value"]
        .agg(mean="mean", sd="std", n_folds="count")
        .reset_index()
    )
    out["display"] = out.apply(lambda r: f"{r['mean']:.3f} ± {r['sd']:.3f}", axis=1)
    return out

def paper_table(
    long: pd.DataFrame,
    evaluation: str = "cv",
    metrics: tuple[str, ...] = ("auroc", "pr_auc", "sensitivity", "specificity", "ppv"),
) -> pd.DataFrame:
    """Pivot into the wide shape used in the manuscript's results tables."""
    summary = summarise(long, metrics=metrics)
    summary = summary[summary["evaluation"] == evaluation]
    wide = summary.pivot_table(
        index=["arm", "strategy", "scaling"],
        columns="metric",
        values="display",
        aggfunc="first",
    )
    return wide.reindex(columns=[m for m in metrics if m in wide.columns]).reset_index()

#: pdfLaTeX with inputenc cannot typeset these code points, and pandas passes
#: them through untouched because they are not LaTeX special characters. The
#: substitution therefore runs after `to_latex`, leaving escape=True to protect
#: everything else.
UNICODE_TO_LATEX = {
    "μ": r"$\mu$", "σ": r"$\sigma$", "ε": r"$\varepsilon$",
    "Δ": r"$\Delta$", "±": r"$\pm$",
    "−": "$-$", "–": "--", "—": "---",
}

#: Column headers as they should read in print.
HEADERS = {
    "auroc": "AUROC", "pr_auc": "PR-AUC", "sensitivity": "Sens.",
    "specificity": "Spec.", "ppv": "PPV", "npv": "NPV",
    "brier": "Brier", "ece": "ECE", "metric": "",
}

def latex_safe(text: str) -> str:
    for char, replacement in UNICODE_TO_LATEX.items():
        text = text.replace(char, replacement)
    return text

def _check_latex(tex: str, label: str) -> None:
    """Two escaping slips that a compiler log reports only as damage downstream.

    Both have shipped once already, so they are caught where they are made.
    """
    # In LaTeX a double backslash is a row separator; before a letter it is
    # always an escaping slip from generated text, and it silently destroys the
    # surrounding table.
    slips = re.findall(r"\\\\[A-Za-z]+", tex)
    if slips:
        raise ValueError(
            f"{label}: double backslash before a command, {sorted(set(slips))}. "
            "Check the caption for over-escaped LaTeX."
        )
    # The mirror image: a caption written as a plain (non-raw) Python string
    # turns \v, \t, \b, \a, \f into control characters, so \varepsilon reaches
    # the file as VT + "arepsilon". Invisible in an editor, fatal to pdfLaTeX.
    # pandas escapes with escape=True, so a header written as "$\\Delta$" arrives
    # as "\\$\\textbackslash Delta\\$" and typesets as literal characters. Greek and
    # symbols belong in the column name as Unicode, which UNICODE_TO_LATEX
    # converts after escaping.
    escaped_math = re.findall(r"\\\\\$|\\\\textbackslash", tex)
    if escaped_math:
        raise ValueError(
            f"{label}: escaped maths in the output, {sorted(set(escaped_math))}. "
            "Write the column name with a Unicode symbol instead of LaTeX maths."
        )
    control = {f"0x{ord(c):02x}" for c in tex if ord(c) < 0x20 and c != "\n"}
    if control:
        raise ValueError(
            f"{label}: control characters in the output, {sorted(control)}. "
            "The caption is probably a non-raw string containing \\v, \\t, \\b, "
            "\\a or \\f; prefix it with r."
        )

def to_latex(
    table: pd.DataFrame,
    caption: str,
    label: str,
    path=None,
    digits: int = 3,
    full_width: bool = True,
    index: bool = True,
    tabcolsep: str = None,
    note: str = None,
) -> str:
    """    One definition for every analysis script: a second copy drifted once and
        emitted raw Greek letters that pdfLaTeX refused to compile.
    """
    table = table.rename(columns=HEADERS)
    table.columns.name = None
    body = table.to_latex(
        index=index, escape=True,
        column_format=("l" if index else "l") + "c" * (table.shape[1] - (0 if index else 1)),
        na_rep="---",
        float_format=lambda v: f"{v:.{digits}f}",
    )
    env = "table*" if full_width else "table"
    # A single IEEE column is about 252pt; six columns of prose headers overrun
    # it and spill into the neighbouring column. Tightening the gutter buys
    # roughly 36pt without touching the font size.
    gutter = f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}\n" if tabcolsep else ""

    if note:
        # Outside the tabular, not as a \multicolumn row inside it. A spanning
        # cell wider than the columns it covers forces the table to that width
        # and dumps the surplus into the last column, which visibly detaches it.
        width = r"\textwidth" if full_width else r"\columnwidth"
        body += r"\par\vspace{2pt}\parbox{%s}{\footnotesize %s}" % (width, note) + "\n"
    tex = (
        f"\\begin{{{env}}}[!htbp]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\small\n{gutter}"
        f"{latex_safe(body)}\\end{{{env}}}\n"
    )
    _check_latex(tex, label)

    if path:
        Path(path).write_text(tex)
        print(f"wrote {path}")
    return tex

def to_latex_grouped(
    sections,
    caption: str,
    label: str,
    path=None,
    note: str = None,
    full_width: bool = True,
) -> str:
    """Emit one booktabs table whose rows are grouped under headings.

    `sections` is [(heading, [(subheading_or_None, DataFrame), ...]), ...]. The
    frames must already hold formatted strings and share their columns, which is
    the condition that makes the grouping meaningful: cross-validation and
    external validation are the same models and the same metrics, so putting
    them in one table turns a comparison between two floats into a vertical
    read.
    """
    columns = list(sections[0][1][0][1].columns)
    ncol = len(columns) + 1
    header = " & ".join([""] + [HEADERS.get(c, c) for c in columns])

    lines = [r"\toprule", header + r" \\", r"\midrule"]
    for s, (heading, groups) in enumerate(sections):
        if s:
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{%d}{l}{\textbf{%s}} \\" % (ncol, heading))
        lines.append(r"\midrule")
        for g, (subheading, frame) in enumerate(groups):
            if subheading:
                if g:
                    lines.append(r"\addlinespace")
                lines.append(
                    r"\multicolumn{%d}{l}{\textit{%s}} \\" % (ncol, subheading)
                )
            indent = r"\quad " if subheading else ""
            for name, row in frame.iterrows():
                cells = " & ".join(str(v) for v in row)
                lines.append(f"{indent}{name} & {cells} " + r"\\")
    lines.append(r"\bottomrule")

    env = "table*" if full_width else "table"
    tex = (
        f"\\begin{{{env}}}[!t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\small\n"
        f"\\begin{{tabular}}{{l{'c' * len(columns)}}}\n"
        + "\n".join(lines)
        + "\n\\end{tabular}\n"
        # Outside the tabular: a spanning cell wider than the columns it covers
        # forces the table to that width and pushes the surplus into the last
        # column, detaching it visibly from the rest.
        + (r"\par\vspace{2pt}\parbox{%s}{\footnotesize %s}" % (
              r"\textwidth" if full_width else r"\columnwidth", note) + "\n"
           if note else "")
        + f"\\end{{{env}}}\n"
    )
    tex = latex_safe(tex)
    _check_latex(tex, label)

    if path:
        Path(path).write_text(tex)
        print(f"wrote {path}")
    return tex
