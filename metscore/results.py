"""Writing the results of a run and reading them back.

Every run, whatever the arm, writes the same files:

    results/
      index.csv                 one row per run
      <run_id>/
        config.json             the ExperimentConfig
        metrics.csv             long format, one row per measurement
        profile.json            wall-clock, compute, communication
        code_version.json       commit of the code
        aggregation_weights.csv, round_log.csv   (federated runs)
        predictions/<evaluation>_rep{r}_fold{f}.npz   y_true, y_pred, site, subject

`metrics.csv` has one row per (evaluation, site, stratum, repetition, fold, metric).
The predictions of every fold are kept because the paired tests need the scores of
two approaches on the same subjects.
"""

from __future__ import annotations

import fcntl
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RESULTS_DIR, WORKSPACE, ExperimentConfig
from .evaluation import Metrics

# columns of metrics.csv
LONG_COLUMNS = [
    "run_id", "arm", "strategy", "scaling", "evaluation", "site",
    "stratum_kind", "stratum", "repetition", "fold", "metric", "value",
]


def code_version() -> dict:
    """Commit of the code, and whether metscore/ or scripts/ had uncommitted changes.

    Both are None when the code is not the root of a git repository of its own (a plain
    copy inside another one, for instance): the commit found would not be this code's.
    """

    def git(*args):
        done = subprocess.run(
            ["git", "-C", str(WORKSPACE), *args], capture_output=True, text=True
        )
        return done.stdout.strip() if done.returncode == 0 else None

    top = git("rev-parse", "--show-toplevel")
    if top is None or Path(top).resolve() != WORKSPACE.resolve():
        return {"git_commit": None, "git_dirty": None}

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "metscore", "scripts")),
    }


def make_run_id(cfg: ExperimentConfig) -> str:
    """Folder name of a run: arm, strategy, parameters and variant.

    The same configuration always gets the same name, so a re-run overwrites it.
    The scaling contract is not part of the name because every run saves both.
    """
    parts = [cfg.arm, cfg.strategy]
    if cfg.strategy_params:
        parts.append(
            "_".join(f"{k}{v}" for k, v in sorted(cfg.strategy_params.items()))
        )
    if cfg.variant:
        parts.append(cfg.variant)
    return "__".join(str(p) for p in parts)


@dataclass
class RunWriter:
    """Collects the output of a run and writes it when the run ends."""

    config: ExperimentConfig
    run_id: str = ""
    _rows: list[dict] = None
    _started: datetime = None
    _weights: list[dict] = None
    _rounds: list[dict] = None

    def __post_init__(self):
        self.run_id = self.run_id or make_run_id(self.config)
        self._rows = []
        self._weights = []
        self._rounds = []
        self._started = datetime.now()
        self.path.mkdir(parents=True, exist_ok=True)

        # Remove the predictions of a previous run with the same name, or
        # load_predictions would mix old and new folds.
        predictions = self.path / "predictions"
        if predictions.exists():
            for stale in predictions.glob("*.npz"):
                stale.unlink()
        predictions.mkdir(exist_ok=True)

        # metrics.csv marks a finished run and is written by close()
        metrics = self.path / "metrics.csv"
        if metrics.exists():
            metrics.unlink()

    @property
    def path(self) -> Path:
        return RESULTS_DIR / self.run_id

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
        """Add the metrics of one evaluation."""
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
        """Save the scores of one fold.

        `subject_index` is the row of each subject in the discovery or external
        table, used later to align two runs on the same subjects.
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

    def record_aggregation_weights(self, repetition: int, fold: int, rows: list[dict]) -> None:
        """Weight applied to each client in this fold."""
        self._weights.extend({"repetition": repetition, "fold": fold, **row} for row in rows)

    def record_round(self, repetition: int, fold: int, round_index: int, score: float) -> None:
        """Federated validation score of one round."""
        self._rounds.append(
            {"repetition": repetition, "fold": fold, "round": round_index, "validation_auroc": score}
        )

    def close(self, profile: dict | None = None, notes: str = "") -> Path:
        if not self._rows:
            raise RuntimeError(f"{self.run_id}: nothing recorded")

        frame = pd.DataFrame(self._rows)[LONG_COLUMNS]
        frame.to_csv(self.path / "metrics.csv", index=False)
        self.config.save(self.path / "config.json")
        if self._weights:
            pd.DataFrame(self._weights).to_csv(self.path / "aggregation_weights.csv", index=False)
        if self._rounds:
            pd.DataFrame(self._rounds).to_csv(self.path / "round_log.csv", index=False)
        (self.path / "code_version.json").write_text(json.dumps(code_version(), indent=2))

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
    """One row per run in index.csv, the latest one."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    index_path = RESULTS_DIR / "index.csv"

    # Several jobs of a grid can finish at the same moment, so the file is locked
    # while it is read and rewritten.
    with open(RESULTS_DIR / ".index.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        frame = pd.DataFrame([row])
        if index_path.exists() and index_path.stat().st_size:
            existing = pd.read_csv(index_path)
            existing = existing[existing["run_id"] != row["run_id"]]
            frame = pd.concat([existing, frame], ignore_index=True)
        frame.sort_values(["arm", "strategy", "scaling"]).to_csv(index_path, index=False)

# Reading results


def collect(pattern: str = "*") -> pd.DataFrame:
    """The metrics of every run matching `pattern`, as one long table."""
    frames = [
        pd.read_csv(path)
        for path in sorted(RESULTS_DIR.glob(f"{pattern}/metrics.csv"))
    ]
    if not frames:
        raise FileNotFoundError(f"no runs matching {pattern!r} under {RESULTS_DIR}")
    return pd.concat(frames, ignore_index=True)


def load_predictions(run_id: str, evaluation: str = "external") -> pd.DataFrame:
    """Scores of one run on one evaluation set, all folds."""
    rows = []
    for path in sorted((RESULTS_DIR / run_id / "predictions").glob(f"{evaluation}_rep*.npz")):
        # from the right, the evaluation name has underscores too
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
    """Mean and SD across folds."""
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
    """Wide table of mean ± SD, one row per run."""
    summary = summarise(long, metrics=metrics)
    summary = summary[summary["evaluation"] == evaluation]
    wide = summary.pivot_table(
        index=["arm", "strategy", "scaling"],
        columns="metric",
        values="display",
        aggfunc="first",
    )
    return wide.reindex(columns=[m for m in metrics if m in wide.columns]).reset_index()


# Characters pdfLaTeX cannot print and pandas does not escape; replaced after
# DataFrame.to_latex.
UNICODE_TO_LATEX = {
    "μ": r"$\mu$", "σ": r"$\sigma$", "ε": r"$\varepsilon$",
    "Δ": r"$\Delta$", "±": r"$\pm$",
    "−": "$-$", "–": "--", "—": "---",
}

# column headers in the tables
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
    """Catch escaping mistakes in a generated table before it is written."""
    # a double backslash before a letter is an over-escaped command
    slips = re.findall(r"\\\\[A-Za-z]+", tex)
    if slips:
        raise ValueError(
            f"{label}: double backslash before a command, {sorted(set(slips))}. "
            "Check the caption for over-escaped LaTeX."
        )
    # maths in a column name gets escaped by pandas; use the Unicode symbol,
    # which UNICODE_TO_LATEX converts afterwards
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
    """Write a DataFrame as a booktabs table. Returns the LaTeX source."""
    table = table.rename(columns=HEADERS)
    table.columns.name = None
    body = table.to_latex(
        index=index, escape=True,
        column_format=("l" if index else "l") + "c" * (table.shape[1] - (0 if index else 1)),
        na_rep="---",
        float_format=lambda v: f"{v:.{digits}f}",
    )
    env = "table*" if full_width else "table"
    # a narrower column gap for tables that have to fit one IEEE column
    gutter = f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}\n" if tabcolsep else ""

    if note:
        # the note goes under the tabular; inside it, as a \multicolumn row,
        # it would stretch the last column
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
    """Write one table with its rows grouped under headings.

    `sections` is [(heading, [(subheading or None, DataFrame), ...]), ...]. The
    frames hold formatted strings and have the same columns.
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
