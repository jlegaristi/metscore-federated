"""
Data loading and fold generation.

Folds are cut once, stratified within cohort, and consumed identically by every
arm, so the unified arm's fold *k* is exactly the union of each client's fold
*k*.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import BIOMARKERS, DATA_DIR, METADATA, TARGET

def load_discovery() -> pd.DataFrame:
    """The 20,662 DISCOVERY subjects with biomarkers, label and demographics."""
    path = DATA_DIR / "discovery.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found, run `python scripts/prepare_data.py` first."
        )
    return pd.read_csv(path)

def load_external() -> pd.DataFrame:
    """The 661-subject external validation cohort."""
    path = DATA_DIR / "external.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found, run `python scripts/prepare_data.py` first."
        )
    return pd.read_csv(path)

def assign_folds(df: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    """Stratified fold labels, computed within each cohort.

    Stratifying within cohort keeps every client's local class balance stable
    across folds (necessary because METS+ has 94 positives out of 147 while
    AKRIBEA has 93 out of 11,159) and still yields a valid stratified partition
    of the pooled data for the unified arm.
    """
    folds = pd.Series(-1, index=df.index, dtype="int8")

    for cohort, block in df.groupby("cohort", sort=True):
        y = block[TARGET]
        if y.nunique() < 2:
            raise ValueError(f"{cohort}: only one class present, cannot stratify")

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (_, test_idx) in enumerate(skf.split(block, y)):
            folds.loc[block.index[test_idx]] = fold

    if (folds < 0).any():
        raise RuntimeError("some subjects were not assigned to a fold")
    return folds

def split_fold(
    df: pd.DataFrame,
    folds: pd.Series,
    test_fold: int,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, validation, test) for one fold.

    The validation split is drawn from the training folds and stratified within
    cohort. Everything that must not see the test fold (early stopping,
    federated checkpoint selection, the operating threshold) reads from
    ``validation``.
    """
    test = df.loc[folds == test_fold]
    pool = df.loc[folds != test_fold]

    val_parts = []
    for cohort, block in pool.groupby("cohort", sort=True):
        n_val = max(1, int(round(len(block) * validation_fraction)))
        pos = block.loc[block[TARGET] == 1]
        neg = block.loc[block[TARGET] == 0]
        share = n_val / len(block)

        for part in (pos, neg):
            if len(part) == 0:
                continue
            take = min(len(part) - 1, max(1, int(round(len(part) * share))))
            if take > 0:
                val_parts.append(part.sample(n=take, random_state=seed))

    validation = pd.concat(val_parts) if val_parts else pool.iloc[:0]
    train = pool.drop(index=validation.index)
    return train, validation, test

def features(df: pd.DataFrame) -> pd.DataFrame:
    """The 22 biomarkers, in canonical order. Metadata never reaches the model."""
    return df[BIOMARKERS]

def labels(df: pd.DataFrame) -> pd.Series:
    return df[TARGET]

def metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Columns carried for stratified reporting."""
    return df[[c for c in METADATA if c in df.columns]]
