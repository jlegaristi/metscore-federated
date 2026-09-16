"""
Feature standardization under an explicit contract.

Two contracts are available and a third is deliberately unrepresentable:

  TRAIN_FITTED  (primary)    statistics from training data only, so it may
                             score every subject at the target site
  SITE_ADAPTED  (secondary)  statistics from a reference split held out at the
                             target site, so it may score only the remainder
  pooled/transductive        not available; fitting on the scored samples
                             themselves lets a subject contribute to the
                             baseline against which it is judged

Whichever contract is used, it is used for every arm. The two reach different
numbers of subjects, so they are also compared on the subset both can score,
recorded as ``external:train_fitted_common``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import BIOMARKERS, SITE_REFERENCE_FRACTION, ScalingContract

def fit_training_scaler(x_train: pd.DataFrame | np.ndarray) -> StandardScaler:
    """Fit on training features only. Never sees validation or test data."""
    scaler = StandardScaler()
    scaler.fit(np.asarray(x_train, dtype="float64"))
    return scaler

@dataclass
class EvaluationSet:
    """A target cohort partitioned by what each contract is allowed to see.

        TRAIN_FITTED takes nothing from the target site, so it is under no such
        restriction and scores ``everyone``. Withholding the reference rows from it
        too would cost 30% of the primary external result in order to keep a
        secondary comparison tidy. The contract comparison is instead made on
        ``scored``, which both contracts can reach.
    """

    scored: pd.DataFrame
    reference: pd.DataFrame
    everyone: pd.DataFrame
    site: str

    @property
    def n_scored(self) -> int:
        return len(self.scored)

    def subjects(self, contract) -> pd.DataFrame:
        """The rows a contract is permitted to score."""
        return self.everyone if contract is ScalingContract.TRAIN_FITTED else self.scored

def partition_site(
    df: pd.DataFrame,
    site: str,
    seed: int,
    reference_fraction: float = SITE_REFERENCE_FRACTION,
) -> EvaluationSet:
    """Split a target site into a reference set and a scored set.

    Stratified on the label so the reference set is representative. Done once,
    up front, independently of which contract will be applied.
    """
    from .config import TARGET

    ref_parts = []
    for _, part in df.groupby(TARGET, sort=True):
        take = int(round(len(part) * reference_fraction))
        take = min(max(take, 1), len(part) - 1) if len(part) > 1 else 0
        if take > 0:
            ref_parts.append(part.sample(n=take, random_state=seed))

    reference = pd.concat(ref_parts) if ref_parts else df.iloc[:0]
    return EvaluationSet(
        scored=df.drop(index=reference.index), reference=reference,
        everyone=df, site=site,
    )

def transform(
    contract: ScalingContract,
    evaluation: EvaluationSet,
    training_scaler: StandardScaler,
    subset: pd.DataFrame = None,
) -> np.ndarray:
    """Standardise under the given contract the rows it is allowed to score.

    ``subset`` overrides that choice, so TRAIN_FITTED can also be applied to the
    smaller set SITE_ADAPTED reaches, which is what makes the two comparable.
    """
    rows = evaluation.subjects(contract) if subset is None else subset
    x_scored = np.asarray(rows[BIOMARKERS], dtype="float64")

    if contract is ScalingContract.TRAIN_FITTED:
        return training_scaler.transform(x_scored)

    if contract is ScalingContract.SITE_ADAPTED:
        if len(evaluation.reference) < 2:
            raise ValueError(
                f"site {evaluation.site!r} has {len(evaluation.reference)} "
                "reference samples; cannot estimate site statistics"
            )
        site_scaler = StandardScaler().fit(
            np.asarray(evaluation.reference[BIOMARKERS], dtype="float64")
        )
        return site_scaler.transform(x_scored)

    raise ValueError(f"unknown contract: {contract!r}")

def aggregate_scalers(scalers: list[StandardScaler]) -> StandardScaler:
    """Combine per-client scalers into one, exactly, from summary statistics.

            pooled_mean = sum(n_i * mean_i) / sum(n_i)
            pooled_var  = sum(n_i * (var_i + mean_i^2)) / sum(n_i) - pooled_mean^2
    """
    if not scalers:
        raise ValueError("no scalers to aggregate")

    counts = np.array([s.n_samples_seen_ for s in scalers], dtype="float64")
    if counts.ndim > 1:
        counts = counts[:, 0]
    means = np.vstack([s.mean_ for s in scalers])
    variances = np.vstack([s.var_ for s in scalers])
    total = counts.sum()

    pooled_mean = (counts[:, None] * means).sum(axis=0) / total
    pooled_second = (counts[:, None] * (variances + means**2)).sum(axis=0) / total
    pooled_var = np.maximum(pooled_second - pooled_mean**2, 0.0)

    pooled = StandardScaler()
    pooled.mean_ = pooled_mean
    pooled.var_ = pooled_var
    # Guard against a constant feature producing a zero divisor.
    pooled.scale_ = np.sqrt(np.where(pooled_var > 0, pooled_var, 1.0))
    pooled.n_features_in_ = scalers[0].n_features_in_
    pooled.n_samples_seen_ = int(total)
    return pooled

def transform_internal(
    training_scaler: StandardScaler, x: pd.DataFrame | np.ndarray
) -> np.ndarray:
    """Standardise validation or test folds drawn from the training distribution.

    Cross-validation folds always use the training scaler: the site is the same,
    so there is nothing to adapt to and the contract question does not arise.
    """
    return training_scaler.transform(np.asarray(x, dtype="float64"))
