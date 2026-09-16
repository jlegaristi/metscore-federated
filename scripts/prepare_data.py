"""
Build the pinned data snapshot used by every experiment.

Outputs under ``data/``:

  discovery.csv  20,662 rows: 22 biomarkers, label, cohort and the
                     demographics used for stratified reporting (age, sex, bmi)
  external.csv   661 rows from three validation sites, same columns

Both tables are derived from the primary sources, so the chain from raw file to
model input is auditable.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import date
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metscore.config import (  # noqa: E402
    BIOMARKERS,
    COHORT_SIZES,
    COHORTS,
    DATA_DIR,
    DISCOVERY_N,
    EXTERNAL_N,
    EXTERNAL_SITE_SIZES,
    EXTERNAL_SITES,
    SELECTION_FILE,
    SOURCE_DATA,
    TARGET,
)

REPO_ROOT = Path(
    subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
)
VALIDATION_IN_GIT = (
    "data-analysis/cicbiogune/metSCORE/data/combined_validation_data.csv"
)
DEMOGRAPHICS = ["age", "sex", "bmi", "metS_profile"]

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

def _binarise(series: pd.Series) -> pd.Series:
    """Labels are 'yes'/'no' in every primary source, 0/1 in the derived files."""
    if series.dtype == object:
        return (series.astype(str).str.lower() == "yes").astype("int8")
    return series.astype("int8")

def load_panel_definition() -> dict[str, str]:
    """Map raw NMR variable -> reported biomarker name, from the selection file.

    Ten of the 22 carry a cluster label that differs from the raw variable name
    (``L5TG`` is reported as ``LDLs 4-5``); the rest keep their own name, which
    the spreadsheet encodes as an empty ``cluster_name_final``.
    """
    sel = pd.read_excel(SOURCE_DATA / SELECTION_FILE)
    chosen = sel[sel["selected"].astype(str).str.lower() == "yes"]

    mapping = {
        row.variable: (
            row.cluster_name_final
            if pd.notna(row.cluster_name_final)
            else row.variable
        )
        for row in chosen.itertuples()
    }

    if sorted(mapping.values()) != sorted(BIOMARKERS):
        unexpected = sorted(set(mapping.values()) ^ set(BIOMARKERS))
        raise SystemExit(f"panel does not match config.BIOMARKERS: {unexpected}")

    renamed = sum(1 for k, v in mapping.items() if k != v)
    print(f"  panel: {len(mapping)} variables, {renamed} renamed to cluster labels")
    return mapping

def derive_panel(df: pd.DataFrame, mapping: dict[str, str], label: str) -> pd.DataFrame:
    missing = [v for v in mapping if v not in df.columns]
    if missing:
        raise SystemExit(f"{label}: source is missing raw variables {missing}")

    panel = df[list(mapping)].rename(columns=mapping)
    panel[TARGET] = _binarise(df[TARGET]).to_numpy()
    return panel[BIOMARKERS + [TARGET]]

def _columns_match(a: pd.DataFrame, b: pd.DataFrame, columns: list[str]) -> list[str]:
    """Compare two tables row-order-independently, column by column."""
    differing = []
    for col in columns:
        x = np.sort(a[col].to_numpy(dtype=float))
        y = np.sort(b[col].to_numpy(dtype=float))
        if not np.allclose(x, y, atol=1e-9, equal_nan=True):
            differing.append(col)
    return differing

def build_discovery(mapping: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    src = SOURCE_DATA / "metS_data.csv"
    raw = pd.read_csv(src, low_memory=False)
    provenance = {src.name: _sha256(src)}

    frames = []
    for stem, cohort in COHORTS.items():
        block = raw.loc[raw["Cohort"] == cohort]
        expected = COHORT_SIZES[cohort]
        if len(block) != expected:
            raise SystemExit(
                f"{cohort}: {len(block)} rows, manuscript Table II says {expected}"
            )

        panel = derive_panel(block, mapping, cohort)
        for col in DEMOGRAPHICS:
            panel[col] = block[col].to_numpy() if col in block.columns else np.nan
        panel["cohort"] = cohort
        panel["site"] = cohort

        parts = []
        for split in ("train", "test"):
            path = SOURCE_DATA / f"{stem}_filt_{split}.csv"
            parts.append(pd.read_csv(path))
            provenance[path.name] = _sha256(path)
        reference = pd.concat(parts, ignore_index=True)

        differing = _columns_match(panel, reference, BIOMARKERS + [TARGET])
        if differing:
            raise SystemExit(f"{cohort}: derived panel differs from *_filt_* in {differing}")

        frames.append(panel)
        pos = int(panel[TARGET].sum())
        print(
            f"  {cohort:<12} n={len(panel):>6}  MetS+={pos:>4} "
            f"({pos / len(panel) * 100:5.2f}%)   matches *_filt_*"
        )

    discovery = pd.concat(frames, ignore_index=True)
    if len(discovery) != DISCOVERY_N:
        raise SystemExit(f"DISCOVERY: {len(discovery)} rows, expected {DISCOVERY_N}")

    provenance[SELECTION_FILE] = _sha256(SOURCE_DATA / SELECTION_FILE)
    return discovery, provenance

def build_external(mapping: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    frames, provenance = [], {}

    for site, filename in EXTERNAL_SITES.items():
        path = SOURCE_DATA / filename
        raw = pd.read_excel(path)
        provenance[filename] = _sha256(path)

        expected = EXTERNAL_SITE_SIZES[site]
        if len(raw) != expected:
            raise SystemExit(f"{site}: {len(raw)} rows, manuscript says {expected}")

        panel = derive_panel(raw, mapping, site)
        for col in DEMOGRAPHICS:
            panel[col] = raw[col].to_numpy() if col in raw.columns else np.nan
        panel["cohort"] = "VALIDATION"
        panel["site"] = site

        frames.append(panel)
        pos = int(panel[TARGET].sum())
        have = [c for c in DEMOGRAPHICS if panel[c].notna().any()]
        print(
            f"  {site:<12} n={len(panel):>6}  MetS+={pos:>4} "
            f"({pos / len(panel) * 100:5.2f}%)   demographics: {', '.join(have)}"
        )

    external = pd.concat(frames, ignore_index=True)
    if len(external) != EXTERNAL_N:
        raise SystemExit(f"VALIDATION: {len(external)} rows, expected {EXTERNAL_N}")

    # The working-tree copy of this file has been overwritten with a different
    # export, so the reference is read from git rather than from disk.
    blob = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"HEAD:{VALIDATION_IN_GIT}"],
        capture_output=True, text=True, check=True,
    ).stdout
    rev = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    reference = pd.read_csv(StringIO(blob))
    reference[TARGET] = _binarise(reference[TARGET])
    differing = _columns_match(external, reference, BIOMARKERS + [TARGET])
    if differing:
        raise SystemExit(f"VALIDATION: differs from the submitted table in {differing}")

    print(f"  reconstruction matches the 661 x 23 table in git HEAD ({rev[:8]})")
    provenance["verified_against"] = f"{VALIDATION_IN_GIT}@{rev[:16]}"
    return external, provenance

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("PANEL")
    mapping = load_panel_definition()

    print("\nDISCOVERY")
    discovery, disc_prov = build_discovery(mapping)

    print("\nEXTERNAL")
    external, ext_prov = build_external(mapping)

    discovery.to_csv(DATA_DIR / "discovery.csv", index=False)
    external.to_csv(DATA_DIR / "external.csv", index=False)

    (DATA_DIR / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "built": date.today().isoformat(),
                "panel": {"n": len(BIOMARKERS), "raw_to_reported": mapping},
                "discovery": {
                    "rows": len(discovery),
                    "positives": int(discovery[TARGET].sum()),
                    "by_cohort": discovery.groupby("cohort")[TARGET]
                    .agg(["size", "sum"])
                    .to_dict("index"),
                    "sources": disc_prov,
                },
                "external": {
                    "rows": len(external),
                    "positives": int(external[TARGET].sum()),
                    "by_site": external.groupby("site")[TARGET]
                    .agg(["size", "sum"])
                    .to_dict("index"),
                    "sources": ext_prov,
                },
            },
            indent=2,
            default=str,
        )
    )

    print(
        f"\nWrote {DATA_DIR / 'discovery.csv'}"
        f"\n      {DATA_DIR / 'external.csv'}"
        f"\n      {DATA_DIR / 'PROVENANCE.json'}"
    )

if __name__ == "__main__":
    main()
