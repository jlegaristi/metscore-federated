"""
Aggregation strategies, built on Flower's own implementations.

The arithmetic is Flower's: `flwr.server.strategy.aggregate` and
`flwr.common.differential_privacy` are imported and called directly. This module
adds a uniform interface, so that the runner can sweep strategies and a new
aggregator is a registry entry rather than a new script.

Flower's simulation engine is not used. Its `run_simulation` spawns a Ray
cluster per fold; driving the same aggregation functions in-process instead is
deterministic and parallelizes cleanly across folds, with identical semantics
because the code called is the same.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from flwr.common.differential_privacy import (
    add_gaussian_noise_inplace,
    compute_clip_model_update,
)
from flwr.server.strategy.aggregate import (
    aggregate,
    aggregate_median,
    aggregate_trimmed_avg,
)

#: A client update: its weights and the number of samples behind them.
Update = tuple[list[np.ndarray], int]

#: Clipping bound for differential privacy, calibrated to the L2 norms actually
#: observed here: a client update after ten local epochs has median norm ~4.5 in
#: the low-noise regime. An uncalibrated bound is the classic way to make DP look
#: worse than it is, clipping at 1.0 bound 100% of updates and discarded most
#: of the signal before any noise was added, costing 0.12 AUROC at a noise
#: multiplier of 0.1. Every run reports its observed norm distribution and how
#: often the bound actually bound, so the choice is auditable.
DEFAULT_CLIPPING_NORM = 5.0

# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------

def fedavg(updates: list[Update]) -> list[np.ndarray]:
    """Weighted by local sample count. McMahan et al. (2017)."""
    return aggregate(updates)

def fedavg_uniform(updates: list[Update]) -> list[np.ndarray]:
    """Every client counts equally, regardless of how much data it holds."""
    return aggregate([(weights, 1) for weights, _ in updates])

def fedmedian(updates: list[Update]) -> list[np.ndarray]:
    """Coordinate-wise median. Robust to a client whose update is an outlier."""
    return aggregate_median(updates)

def fedtrimmed(updates: list[Update], proportion_to_cut: float = 0.2) -> list[np.ndarray]:
    """Coordinate-wise trimmed mean."""
    return aggregate_trimmed_avg(updates, proportiontocut=proportion_to_cut)

# ---------------------------------------------------------------------------
# Differential privacy
# ---------------------------------------------------------------------------

@dataclass
class DifferentialPrivacy:
    """Client-side clipping with server-side Gaussian noise.

            sigma = noise_multiplier * clipping_norm / num_clients
    """

    inner: Callable[[list[Update]], list[np.ndarray]]
    noise_multiplier: float
    clipping_norm: float = 1.0
    #: how the clipping norm was chosen; the honest answer is to report the
    #: distribution it was calibrated against, and how often it bound.
    observed_norms: list[float] = field(default_factory=list)

    def __call__(self, updates: list[Update], global_weights=None) -> list[np.ndarray]:
        if global_weights is None:
            raise ValueError("differential privacy needs the previous global weights")

        clipped: list[Update] = []
        for weights, n in updates:
            update = [np.asarray(w, dtype="float64").copy() for w in weights]
            self.observed_norms.append(
                float(np.sqrt(sum(
                    np.sum((u - np.asarray(g, dtype="float64")) ** 2)
                    for u, g in zip(update, global_weights)
                )))
            )
            # Clips the *delta* from the global model, then restores absolute
            # weights, which is what Flower's DP strategies do internally.
            compute_clip_model_update(
                param1=update,
                param2=[np.asarray(g, dtype="float64") for g in global_weights],
                clipping_norm=self.clipping_norm,
            )
            clipped.append((update, n))

        aggregated = self.inner(clipped)
        add_gaussian_noise_inplace(
            aggregated, self.noise_multiplier * self.clipping_norm / len(updates)
        )
        return aggregated

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategySpec:
    """What an aggregator needs from the client, and what it is for."""

    name: str
    aggregate: Callable
    proximal_mu: float = 0.0
    needs_global_weights: bool = False
    purpose: str = ""
    params: dict = field(default_factory=dict)

def build(name: str, **params) -> StrategySpec:
    """Resolve a strategy name and its parameters into a runnable spec."""
    if name == "fedavg":
        return StrategySpec("fedavg", fedavg, purpose="baseline")

    if name == "fedavg_uniform":
        return StrategySpec(
            "fedavg_uniform", fedavg_uniform,
            purpose="ablation of sample-size weighting",
        )

    if name == "fedprox":
        mu = float(params.get("mu", 0.1))
        # FedProx aggregates exactly like FedAvg; mu acts in the client's loss.
        return StrategySpec(
            "fedprox", fedavg, proximal_mu=mu,
            purpose="heterogeneity-aware aggregation",
            params={"mu": mu},
        )

    if name == "fedmedian":
        return StrategySpec("fedmedian", fedmedian, purpose="robust aggregation")

    if name == "fedtrimmed":
        cut = float(params.get("proportion_to_cut", 0.2))
        return StrategySpec(
            "fedtrimmed",
            lambda updates: fedtrimmed(updates, cut),
            purpose="robust aggregation",
            params={"proportion_to_cut": cut},
        )

    if name == "dp_fedavg":
        noise = float(params.get("noise_multiplier", 1.0))
        clip = float(params.get("clipping_norm", DEFAULT_CLIPPING_NORM))
        return StrategySpec(
            "dp_fedavg",
            DifferentialPrivacy(fedavg, noise, clip),
            needs_global_weights=True,
            purpose="formal privacy guarantees",
            params={"noise_multiplier": noise, "clipping_norm": clip},
        )

    raise ValueError(f"unknown strategy {name!r}; known: {', '.join(AVAILABLE)}")

AVAILABLE = (
    "fedavg", "fedavg_uniform", "fedprox", "fedmedian", "fedtrimmed", "dp_fedavg",
)
