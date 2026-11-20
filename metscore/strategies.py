"""Aggregation strategies.

The aggregation itself is Flower's (`flwr.server.strategy.aggregate`), called directly
in the training loop of `arms/federated.py` instead of through Flower's simulation
engine, which starts a Ray cluster per fold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from flwr.server.strategy.aggregate import (
    aggregate,
    aggregate_median,
    aggregate_trimmed_avg,
)

# A client update: its weights and the count the server weights it by.
Update = tuple[list[np.ndarray], int]

# Clipping bound of the private runs. The L2 norm of a client update without noise,
# on the same architecture, has a median of 2.3 and a 90th percentile of 2.6; every
# private run saves the norms it observed and the share of updates clipped.
DEFAULT_CLIPPING_NORM = 2.5


def fedavg(updates: list[Update]) -> list[np.ndarray]:
    """Mean weighted by the count of each update (see `StrategySpec.weighting`)."""
    return aggregate(updates)


def fedavg_uniform(updates: list[Update]) -> list[np.ndarray]:
    """Unweighted mean."""
    return aggregate([(weights, 1) for weights, _ in updates])


def fedmedian(updates: list[Update]) -> list[np.ndarray]:
    """Coordinate-wise median."""
    return aggregate_median(updates)


def fedtrimmed(updates: list[Update], proportion_to_cut: float = 0.2) -> list[np.ndarray]:
    """Coordinate-wise trimmed mean."""
    return aggregate_trimmed_avg(updates, proportiontocut=proportion_to_cut)


@dataclass(frozen=True)
class PrivacySpec:
    """Noise multiplier and clipping bound of the mechanism in `privacy`."""

    noise_multiplier: float
    clipping_norm: float


@dataclass(frozen=True)
class StrategySpec:
    name: str
    aggregate: Callable
    proximal_mu: float = 0.0
    # What the server weights each client by:
    #   "samples"  the samples it trains on, after downsampling
    #   "cohort"   its training cohort, before downsampling
    #   "uniform"  the same for every client
    #   "none"     the rule uses no weights (median, trimmed mean)
    weighting: str = "samples"
    # only for the private strategies; applied on the client
    privacy: PrivacySpec | None = None
    purpose: str = ""
    params: dict = field(default_factory=dict)

    def client_weight(self, client) -> int:
        if self.weighting == "cohort":
            return client.n_cohort
        if self.weighting == "uniform":
            return 1
        return client.n


def build(name: str, **params) -> StrategySpec:
    """The spec of a strategy, from its name and parameters."""
    if name == "fedavg":
        return StrategySpec("fedavg", fedavg, purpose="baseline")

    if name == "fedavg_uniform":
        return StrategySpec(
            "fedavg_uniform", fedavg_uniform, weighting="uniform",
            purpose="ablation of sample-size weighting",
        )

    if name == "fedavg_cohort":
        return StrategySpec(
            "fedavg_cohort", fedavg, weighting="cohort",
            purpose="weighting by cohort size, before downsampling",
        )

    if name == "fedprox":
        mu = float(params.get("mu", 0.1))
        # same aggregation as FedAvg, mu enters the local loss
        return StrategySpec(
            "fedprox", fedavg, proximal_mu=mu,
            purpose="heterogeneity-aware aggregation",
            params={"mu": mu},
        )

    if name == "fedmedian":
        return StrategySpec(
            "fedmedian", fedmedian, weighting="none", purpose="robust aggregation",
        )

    if name == "fedtrimmed":
        cut = float(params.get("proportion_to_cut", 0.2))
        return StrategySpec(
            "fedtrimmed",
            lambda updates: fedtrimmed(updates, cut),
            weighting="none",
            purpose="robust aggregation",
            params={"proportion_to_cut": cut},
        )

    if name == "dp_fedavg":
        noise = float(params.get("noise_multiplier", 1.0))
        clip = float(params.get("clipping_norm", DEFAULT_CLIPPING_NORM))
        return StrategySpec(
            "dp_fedavg",
            fedavg_uniform,
            weighting="uniform",
            privacy=PrivacySpec(noise, clip),
            purpose="formal privacy guarantees",
            params={"noise_multiplier": noise, "clipping_norm": clip},
        )

    raise ValueError(f"unknown strategy {name!r}; known: {', '.join(AVAILABLE)}")


AVAILABLE = (
    "fedavg", "fedavg_uniform", "fedavg_cohort", "fedprox", "fedmedian", "fedtrimmed",
    "dp_fedavg",
)
