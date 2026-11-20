"""Client-level differential privacy: the mechanism and its accounting.

Mechanism. Before sending its update, each client clips the change in its weights to
an L2 norm of C and adds Gaussian noise of standard deviation z*C/sqrt(K). The server
takes the unweighted mean of the K updates, which then carries noise of standard
deviation z*C/K. A weighted mean is not used because its sensitivity would be
C*max(w_i) and would depend on the client sizes.

Protected unit. The whole contribution of one institution, and with it any record of
any of its patients.

Neighbouring relation. Two federations are neighbours if one has an additional
institution (its update replaced by zero, K fixed and public). The sum of the updates
has sensitivity C and noise z*C, so every round is a Gaussian mechanism with noise
multiplier z. All clients take part in every round, so there is no amplification by
subsampling. Under the replace-one relation the sensitivity is 2C.

Trust assumptions.
  with secure aggregation      the server only sees the noisy sum and the guarantee
                               is `epsilon`. Secure aggregation is assumed, it is not
                               simulated here.
  without secure aggregation   the server sees every noisy update, with noise
                               z*C/sqrt(K) against a sensitivity of C, which gives
                               `epsilon_without_secure_aggregation`.

Outside the budget: the standardization statistics sent once before training, the
validation counts used for the operating threshold, and the choice of C, which was
made on the data. Private runs keep the model of the last round, since choosing a
round with validation scores would not be private either.

Epsilon is computed with Google's `dp_accounting` (RDP accountant).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# delta is usually taken below 1/n for n units; with five institutions it is simply
# set to a small value.
DEFAULT_DELTA = 1e-5


@dataclass(frozen=True)
class PrivacyBudget:
    epsilon: float
    delta: float
    noise_multiplier: float
    clipping_norm: float
    rounds: int
    clients: int
    sampling_probability: float
    epsilon_without_secure_aggregation: float
    epsilon_replacement: float

    def __str__(self) -> str:
        return (
            f"(eps={self.epsilon:.2f}, delta={self.delta:g}) after "
            f"{self.rounds} rounds at noise multiplier {self.noise_multiplier:g}"
        )

    def to_dict(self) -> dict:
        return {
            "epsilon": round(self.epsilon, 4),
            "epsilon_without_secure_aggregation": round(self.epsilon_without_secure_aggregation, 4),
            "epsilon_replacement": round(self.epsilon_replacement, 4),
            "delta": self.delta,
            "noise_multiplier": self.noise_multiplier,
            "clipping_norm": self.clipping_norm,
            "rounds": self.rounds,
            "clients": self.clients,
            "sampling_probability": self.sampling_probability,
            "unit": "client-level",
            "neighbouring_relation": "add or remove one client, fixed denominator",
            "mechanism": "Gaussian, client-side clipping and noise, uniform mean",
        }


def client_noise(arrays, noise_multiplier, clipping_norm, n_clients, rng) -> list[np.ndarray]:
    """The noise one client adds, with the shapes of `arrays`."""
    std = noise_multiplier * clipping_norm / math.sqrt(n_clients)
    return [rng.normal(0.0, std, np.shape(a)) if std > 0 else np.zeros(np.shape(a)) for a in arrays]


def privatise_update(
    weights, global_weights, clipping_norm, noise_multiplier, n_clients, rng,
) -> tuple[list[np.ndarray], float]:
    """Clip a client's update and add its noise (runs on the client).

    Returns the weights to send and the norm of the update before clipping.
    """
    reference = [np.asarray(g, dtype="float64") for g in global_weights]
    delta = [np.asarray(w, dtype="float64") - g for w, g in zip(weights, reference)]
    norm = float(math.sqrt(sum(float(np.sum(d**2)) for d in delta)))
    scale = min(1.0, clipping_norm / norm) if norm > 0 else 1.0
    noise = client_noise(delta, noise_multiplier, clipping_norm, n_clients, rng)
    return [g + d * scale + n for g, d, n in zip(reference, delta, noise)], norm


def _epsilon(sigma: float, rounds: int, delta: float, sampling_probability: float) -> float:
    if sigma <= 0:
        return float("inf")

    from dp_accounting import dp_event, rdp

    orders = (
        [1 + x / 10.0 for x in range(1, 100)]
        + list(range(11, 64))
        + [128, 256, 512, 1024]
    )
    accountant = rdp.rdp_privacy_accountant.RdpAccountant(orders)
    event = dp_event.GaussianDpEvent(sigma)
    if sampling_probability < 1.0:
        event = dp_event.PoissonSampledDpEvent(sampling_probability, event)
    accountant.compose(event, rounds)
    return float(accountant.get_epsilon(delta))


def compute_epsilon(
    noise_multiplier: float,
    rounds: int,
    clients: int = 5,
    clipping_norm: float = 2.5,
    delta: float = DEFAULT_DELTA,
    sampling_probability: float = 1.0,
) -> PrivacyBudget:
    """Client-level epsilon after `rounds` rounds of the mechanism described above.

    The clipping norm scales the noise and the sensitivity alike, so it does not
    change epsilon. The number of clients only enters the figure without secure
    aggregation.
    """
    return PrivacyBudget(
        epsilon=_epsilon(noise_multiplier, rounds, delta, sampling_probability),
        delta=delta,
        noise_multiplier=noise_multiplier,
        clipping_norm=clipping_norm,
        rounds=rounds,
        clients=clients,
        sampling_probability=sampling_probability,
        epsilon_without_secure_aggregation=_epsilon(
            noise_multiplier / math.sqrt(clients), rounds, delta, sampling_probability
        ),
        epsilon_replacement=_epsilon(noise_multiplier / 2.0, rounds, delta, sampling_probability),
    )


def sweep(
    noise_multipliers,
    rounds: int,
    clients: int = 5,
    clipping_norm: float = 2.5,
    delta: float = DEFAULT_DELTA,
) -> list[PrivacyBudget]:
    """Budgets for a list of noise multipliers."""
    return [
        compute_epsilon(nm, rounds, clients, clipping_norm, delta)
        for nm in noise_multipliers
    ]
