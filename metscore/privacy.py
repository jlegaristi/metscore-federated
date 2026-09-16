"""
Formal privacy accounting for the differential-privacy runs.

Accounting uses Google's `dp_accounting` (the RDP accountant) rather than a
hand-rolled composition.

What is quantified is client-level (not record-level) differential privacy under
an honest-but-curious server: each site clips its update to a bounded L2 norm
before it leaves the institution, and calibrated Gaussian noise is added at
aggregation. The guarantee is that the presence or absence of any single
participating institution changes the distribution of everything the server
observes by at most a factor of e^epsilon, except with probability delta.

Not covered: a malicious server that deviates from the protocol, and collusion
between the server and other clients.

All clients participate in every round, so the sampling probability is 1 and
there is no privacy amplification by subsampling.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Standard convention for delta: below 1/n for n participating units. With five
#: institutions, client-level delta is set far below 1/5 to be meaningful.
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

    def __str__(self) -> str:
        return (
            f"(eps={self.epsilon:.2f}, delta={self.delta:g}) after "
            f"{self.rounds} rounds at noise multiplier {self.noise_multiplier:g}"
        )

    def to_dict(self) -> dict:
        return {
            "epsilon": round(self.epsilon, 4),
            "delta": self.delta,
            "noise_multiplier": self.noise_multiplier,
            "clipping_norm": self.clipping_norm,
            "rounds": self.rounds,
            "clients": self.clients,
            "sampling_probability": self.sampling_probability,
            "unit": "client-level",
            "mechanism": "Gaussian, client-side clipping",
        }

def compute_epsilon(
    noise_multiplier: float,
    rounds: int,
    clients: int = 5,
    clipping_norm: float = 5.0,
    delta: float = DEFAULT_DELTA,
    sampling_probability: float = 1.0,
) -> PrivacyBudget:
    """Client-level epsilon after `rounds` of the Gaussian mechanism.

    What the accountant needs is noise relative to sensitivity, and for an
    *average* the two scale together. Flower adds noise with
    sigma = noise_multiplier * clipping_norm / clients; replacing one client
    moves the average of clipped updates by at most clipping_norm / clients. The
    ratio is therefore `noise_multiplier` exactly, independent of how many
    clients there are and of the clipping norm.

    Client count does not enter here. It matters through a different channel:
    with all five clients in every round there is no privacy amplification by
    subsampling, which is the mechanism a cross-device federation of millions of
    devices relies on to make differential privacy affordable. Pass
    `sampling_probability < 1` to see that effect.
    """
    if noise_multiplier <= 0:
        return PrivacyBudget(
            float("inf"), delta, noise_multiplier, clipping_norm,
            rounds, clients, sampling_probability,
        )

    effective_sigma = noise_multiplier

    from dp_accounting import dp_event, rdp

    orders = (
        [1 + x / 10.0 for x in range(1, 100)]
        + list(range(11, 64))
        + [128, 256, 512, 1024]
    )
    accountant = rdp.rdp_privacy_accountant.RdpAccountant(orders)

    event = dp_event.GaussianDpEvent(effective_sigma)
    if sampling_probability < 1.0:
        event = dp_event.PoissonSampledDpEvent(sampling_probability, event)

    accountant.compose(event, rounds)
    return PrivacyBudget(
        epsilon=float(accountant.get_epsilon(delta)),
        delta=delta,
        noise_multiplier=noise_multiplier,
        clipping_norm=clipping_norm,
        rounds=rounds,
        clients=clients,
        sampling_probability=sampling_probability,
    )

def sweep(
    noise_multipliers,
    rounds: int,
    clients: int = 5,
    clipping_norm: float = 5.0,
    delta: float = DEFAULT_DELTA,
) -> list[PrivacyBudget]:
    """The privacy axis of the privacy-utility curve."""
    return [
        compute_epsilon(nm, rounds, clients, clipping_norm, delta)
        for nm in noise_multipliers
    ]
