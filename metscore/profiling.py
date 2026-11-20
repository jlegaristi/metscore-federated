"""Cost of a run: communication, client compute and wall-clock time.

Communication is computed from the protocol (values sent x 4 bytes x 2 directions x
rounds x clients) rather than measured, because in this simulation all clients run
in one process.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

BYTES_PER_FLOAT32 = 4


@dataclass
class CommunicationCost:
    n_parameters: int
    n_clients: int
    n_rounds: int
    bytes_per_parameter: int = BYTES_PER_FLOAT32

    @property
    def per_client_per_round_bytes(self) -> int:
        """Global model down plus local update up."""
        return 2 * self.n_parameters * self.bytes_per_parameter

    @property
    def total_bytes(self) -> int:
        return self.per_client_per_round_bytes * self.n_clients * self.n_rounds

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024**2)

    def summary(self) -> dict:
        return {
            "n_parameters": self.n_parameters,
            "n_clients": self.n_clients,
            "n_rounds": self.n_rounds,
            "per_client_per_round_kb": round(self.per_client_per_round_bytes / 1024, 2),
            "per_client_total_mb": round(
                self.per_client_per_round_bytes * self.n_rounds / (1024**2), 3
            ),
            "total_mb": round(self.total_mb, 3),
        }

    def transfer_seconds(self, mbps: float) -> float:
        """Time the whole exchange would take on a link of `mbps`."""
        return self.total_bytes * 8 / (mbps * 1e6)


@dataclass
class Timer:
    """Named durations, accumulated over a run."""

    durations: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    @contextmanager
    def measure(self, label: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.durations[label] = self.durations.get(label, 0.0) + elapsed
            self.counts[label] = self.counts.get(label, 0) + 1

    def summary(self) -> dict:
        return {
            label: {
                "total_s": round(total, 4),
                "calls": self.counts[label],
                "mean_s": round(total / self.counts[label], 4),
            }
            for label, total in sorted(self.durations.items())
        }


@dataclass
class RunProfile:
    """What is saved in profile.json, for every arm."""

    arm: str
    wall_clock_s: float
    client_compute_s: float = 0.0
    n_rounds: int = 0
    n_clients: int = 1
    communication: dict | None = None
    per_round_s: list[float] = field(default_factory=list)
    timers: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["wall_clock_s"] = round(self.wall_clock_s, 3)
        d["client_compute_s"] = round(self.client_compute_s, 3)
        if self.per_round_s:
            rounds = self.per_round_s
            d["round_s_mean"] = round(sum(rounds) / len(rounds), 3)
            d["round_s_max"] = round(max(rounds), 3)
        return d


def compare(federated: RunProfile, centralised: RunProfile) -> dict:
    """Wall-clock and communication of a federated run next to a centralized one."""
    ratio = (
        federated.wall_clock_s / centralised.wall_clock_s
        if centralised.wall_clock_s
        else float("nan")
    )
    return {
        "wall_clock_ratio": round(ratio, 2),
        "federated_wall_clock_s": round(federated.wall_clock_s, 1),
        "centralised_wall_clock_s": round(centralised.wall_clock_s, 1),
        "communication_mb": (federated.communication or {}).get("total_mb", 0.0),
        "centralised_communication_mb": 0.0,
    }
