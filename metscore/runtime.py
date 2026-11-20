"""Process setup (threads, GPU, seeds). Must be imported before TensorFlow.

The network is small (24,289 trainable parameters, batches of 16) and trains faster
on one CPU thread than on a GPU. Measured with `scripts/benchmark.py` on the server
used for the experiments:

  CPU, 1 thread        314 ms/epoch
  CPU, unconstrained   716 ms/epoch
  GPU (RTX A6000)     1060 ms/epoch

So every job runs on one thread with the GPU hidden, and the grid runs many jobs in
parallel.
"""

from __future__ import annotations

import os
from pathlib import Path

# share of the CPU quota used by default; the server is shared
DEFAULT_UTILISATION = 0.75


def configure_worker(threads: int = 1, use_gpu: bool = False) -> None:
    """Limit the threads of this process. Call it before importing TensorFlow."""
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    os.environ.update(
        TF_CPP_MIN_LOG_LEVEL="3",
        OMP_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads),
        OPENBLAS_NUM_THREADS=str(threads),
        TF_NUM_INTRAOP_THREADS=str(threads),
        TF_NUM_INTEROP_THREADS="1",
    )


def apply_tensorflow_threading(threads: int = 1) -> None:
    """Same limits through the TensorFlow API, in case it was already imported."""
    import tensorflow as tf

    try:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        # already initialised, the environment variables apply
        pass

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def cpu_quota() -> float:
    """Cores available to this container (the cgroup quota, not the host's cores)."""
    for path, parse in (
        ("/sys/fs/cgroup/cpu.max", _parse_cgroup_v2),
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", _parse_cgroup_v1),
    ):
        try:
            value = parse(Path(path))
            if value:
                return value
        except (OSError, ValueError):
            continue

    return float(len(os.sched_getaffinity(0)))


def _parse_cgroup_v2(path: Path) -> float | None:
    quota, period = path.read_text().split()
    return None if quota == "max" else float(quota) / float(period)


def _parse_cgroup_v1(path: Path) -> float | None:
    quota = float(path.read_text().strip())
    if quota <= 0:
        return None
    period = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
    return quota / period


def recommended_workers(utilisation: float = DEFAULT_UTILISATION) -> int:
    """Number of single-threaded jobs to run at once (override with --workers)."""
    return max(1, int(cpu_quota() * utilisation))


def describe() -> dict:
    quota = cpu_quota()
    return {
        "visible_cores": len(os.sched_getaffinity(0)),
        "cpu_quota": quota,
        "recommended_workers": recommended_workers(),
        "note": (
            f"container sees {len(os.sched_getaffinity(0))} cores but may use "
            f"{quota:g}; workers are single-threaded and the GPU is hidden"
        ),
    }


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow.

    Weight initialization, dropout and batch shuffling use the global generators,
    so without this a run would not reproduce.
    """
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
