"""
Process setup and scheduling. Import before TensorFlow, always.

Measured on the machine used for these experiments (see `scripts/benchmark.py`):

  CPU, 1 thread        314 ms/epoch
  CPU, unconstrained   716 ms/epoch
  GPU (RTX A6000)     1060 ms/epoch

  concurrent throughput      N=4    N=8   N=16
  CPU (jobs/min)            14.1   27.1   38.6
  GPU (jobs/min)             7.5   14.0   19.9

The network has 24,289 trainable parameters and trains on batches of 16, which
is too little work to amortize a GPU's per-batch kernel launches. Throughput
comes from running many single-threaded jobs rather than from accelerating one,
so each job pins itself to one thread and hides the GPU.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Leave headroom on a shared host rather than claiming the whole quota.
DEFAULT_UTILISATION = 0.75

def configure_worker(threads: int = 1, use_gpu: bool = False) -> None:
    """Pin this process. Must run before TensorFlow is imported.

    Every runner calls this first. Arms never configure threading themselves --
    a job that quietly grabs 16 threads starves the other fifteen jobs.
    """
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
    """Belt and braces: TF ignores the env vars if it was imported first."""
    import tensorflow as tf

    try:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        # Already initialised; the environment variables did the work.
        pass

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

def cpu_quota() -> float:
    """CPU cores this container may actually use.

    ``nproc`` reports the host's 200 cores; the cgroup quota is 16. Scheduling
    against the former would oversubscribe by 12x and slow everything down.
    """
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
    """How many single-threaded jobs to run at once.

    Efficiency measured 0.96x at eight concurrent workers and 0.69x at sixteen,
    so filling the quota exactly buys throughput at the cost of being a bad
    neighbour on a shared machine. The default backs off; override it with
    `--workers` when the host is quiet.
    """
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
    """Make a fold reproducible.

    The CV seed governs fold assignment and downsampling, but weight
    initialisation, dropout and mini-batch shuffling draw from the global
    Python/NumPy/TensorFlow generators. Without seeding those, the same
    configuration trains a different network on every run: re-running the grid
    moved pooled AUROCs by up to 0.005 and per-cohort figures by 0.07.
    """
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
