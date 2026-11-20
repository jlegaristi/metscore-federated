"""Time one training epoch on CPU (1, 2, 8 threads, unconstrained) and on GPU.

Run:  python scripts/benchmark.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent

WORKER = WORKSPACE / "scripts" / "_benchmark_worker.py"


def run(device: str, threads: int) -> float | None:
    """ms/epoch for one configuration, or None if the worker failed."""
    result = subprocess.run(
        [sys.executable, str(WORKER), device, str(threads)],
        capture_output=True, text=True,
    )
    lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip().isdigit()]
    return float(lines[-1]) if lines else None


def main() -> None:
    configurations = [
        ("CPU, 1 thread", "cpu", 1),
        ("CPU, 2 threads", "cpu", 2),
        ("CPU, 8 threads", "cpu", 8),
        ("CPU, unconstrained", "cpu", 0),
        ("GPU", "gpu", 0),
    ]

    print(f"{'configuration':<22} {'ms/epoch':>10} {'relative':>10}")
    print("-" * 44)

    measurements = []
    for label, device, threads in configurations:
        ms = run(device, threads)
        measurements.append((label, ms))
        print(f"{label:<22} {ms:>10.0f}" if ms else f"{label:<22} {'failed':>10}")

    baseline = min(ms for _, ms in measurements if ms)
    print()
    for label, ms in measurements:
        if ms:
            print(f"{label:<22} {ms:>10.0f} {ms / baseline:>9.2f}x")

    print(
        "\nPin each runner to one thread and parallelise across jobs:\n"
        '  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 '
        "TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1"
    )


if __name__ == "__main__":
    main()
