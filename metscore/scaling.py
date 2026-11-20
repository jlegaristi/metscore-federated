"""Feature standardization.

External data can be standardized in two ways (`ScalingContract`):

  TRAIN_FITTED   with the scaler fitted on the training data; every external subject
                 can be scored. This is the primary analysis.
  SITE_ADAPTED   with statistics estimated at the target site on a reference split,
                 which is then left out of the scoring.

The two score different numbers of subjects, so TRAIN_FITTED is also applied to the
subjects SITE_ADAPTED can score (saved as `external:train_fitted_common`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import BIOMARKERS, SITE_REFERENCE_FRACTION, ScalingContract


def fit_training_scaler(x_train: pd.DataFrame | np.ndarray) -> StandardScaler:
    """Scaler fitted on training features."""
    scaler = StandardScaler()
    scaler.fit(np.asarray(x_train, dtype="float64"))
    return scaler


@dataclass
class EvaluationSet:
    """One external site, split into a reference part and the part to score.

    TRAIN_FITTED needs nothing from the site, so it scores `everyone`; SITE_ADAPTED
    only `scored`.
    """

    scored: pd.DataFrame
    reference: pd.DataFrame
    everyone: pd.DataFrame
    site: str

    @property
    def n_scored(self) -> int:
        return len(self.scored)

    def subjects(self, contract) -> pd.DataFrame:
        """The rows a contract scores."""
        return self.everyone if contract is ScalingContract.TRAIN_FITTED else self.scored


def partition_site(
    df: pd.DataFrame,
    site: str,
    seed: int,
    reference_fraction: float = SITE_REFERENCE_FRACTION,
) -> EvaluationSet:
    """Split a site into reference and scored subjects, stratified by label."""
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
    """Standardize the rows of an external site under one contract.

    `subset` replaces the rows the contract would score by default.
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
    """Pool per-client scalers into one, from their counts, means and variances.

        mean = sum(n_i * mean_i) / sum(n_i)
        var  = sum(n_i * (var_i + mean_i^2)) / sum(n_i) - mean^2

    which is the scaler one would fit on the clients' data put together.
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
    # a constant feature would give a zero scale
    pooled.scale_ = np.sqrt(np.where(pooled_var > 0, pooled_var, 1.0))
    pooled.n_features_in_ = scalers[0].n_features_in_
    pooled.n_samples_seen_ = int(total)
    return pooled


def transform_internal(
    training_scaler: StandardScaler, x: pd.DataFrame | np.ndarray
) -> np.ndarray:
    """Standardize validation or test folds with the training scaler."""
    return training_scaler.transform(np.asarray(x, dtype="float64"))
