"""
Launch the experiment grid.

Every job is one (arm, strategy, scaling contract) run, executed in its own
single-threaded process with the GPU hidden; see `metscore/runtime.py`.

    python scripts/run_experiments.py --list
    python scripts/run_experiments.py --smoke
    python scripts/run_experiments.py --only unified,federated
    python scripts/run_experiments.py --workers 12

Results land in `results/<run_id>/`; `results/index.csv` tracks what has run.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

from metscore import runtime  # noqa: E402
from metscore.config import (  # noqa: E402
    COHORTS, RESULTS_DIR, CVConfig, ScalingContract,
)

#: Runs record every contract; the field survives only to name the primary one
#: in the saved config, since both are present in the metrics table.
BOTH_CONTRACTS = ScalingContract.TRAIN_FITTED.value

@dataclass(frozen=True)
class Job:
    arm: str
    strategy: str
    scaling: str
    params: dict
    cohort: str | None = None

    @property
    def run_id(self) -> str:
        parts = [self.arm, self.strategy]
        if self.params:
            parts.append("_".join(f"{k}{v}" for k, v in sorted(self.params.items())))
        return "__".join(parts)

    @property
    def label(self) -> str:
        extra = f" {self.params}" if self.params else ""
        return f"{self.arm}/{self.strategy}{extra}"

def build_grid() -> list[Job]:
    """The full set of runs behind the revision."""
    federated_strategies = [
        ("fedavg", {}),
        ("fedavg_uniform", {}),
        ("fedprox", {"mu": 0.01}),
        ("fedprox", {"mu": 0.1}),
        ("fedprox", {"mu": 1.0}),
        ("fedmedian", {}),
        ("fedtrimmed", {}),
        # The noise sweep spans the trade-off deliberately. Utility has
        # collapsed by a multiplier of 1.0, while a meaningful guarantee
        # (epsilon <= 10 at delta = 1e-5, over 20 rounds of full participation)
        # needs about 2.4, so the two ends do not overlap. With five silos there
        # is no amplification by subsampling, which is what makes differential
        # privacy affordable in cross-device federations.
        ("dp_fedavg", {"noise_multiplier": 0.1}),
        ("dp_fedavg", {"noise_multiplier": 0.5}),
        ("dp_fedavg", {"noise_multiplier": 1.0}),
        ("dp_fedavg", {"noise_multiplier": 2.0}),
        ("dp_fedavg", {"noise_multiplier": 4.0}),
        ("dp_fedavg", {"noise_multiplier": 8.0}),
    ]

    jobs: list[Job] = [Job("unified", "centralised", BOTH_CONTRACTS, {})]
    jobs += [
        Job("individual", cohort, BOTH_CONTRACTS, {}, cohort=cohort)
        for cohort in COHORTS.values()
    ]
    jobs += [
        Job("federated", strategy, BOTH_CONTRACTS, params)
        for strategy, params in federated_strategies
    ]
    return jobs

CHILD = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
spec = json.loads(sys.argv[2])

from metscore import runtime
runtime.configure_worker(threads=1, use_gpu=False)
runtime.apply_tensorflow_threading(1)

from metscore.config import (
    CVConfig, ExperimentConfig, FederatedConfig, ScalingContract, TrainConfig,
)
from metscore.arms import centralised, federated

overrides = spec.get("overrides", {})
cfg = ExperimentConfig(
    name=spec["run_id"],
    arm=spec["arm"],
    strategy=spec["strategy"],
    strategy_params=spec["params"],
    scaling=ScalingContract(spec["scaling"]),
    cv=CVConfig(**overrides.get("cv", {})),
    train=TrainConfig(**overrides.get("train", {})),
    federated=FederatedConfig(**overrides.get("federated", {})),
)

reps = overrides.get("repetitions")
if spec["arm"] == "federated":
    federated.run(cfg, repetitions=reps)
elif spec["arm"] == "individual":
    centralised.run_individual(cfg, cohort=spec["cohort"], repetitions=reps)
else:
    centralised.run_unified(cfg, repetitions=reps)
'''

def launch(job: Job, overrides: dict) -> subprocess.Popen:
    import json

    payload = {
        "run_id": job.run_id, "arm": job.arm, "strategy": job.strategy,
        "scaling": job.scaling, "params": job.params, "cohort": job.cohort,
        "overrides": overrides,
    }
    log = RESULTS_DIR / "logs" / f"{job.run_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("w")
    return subprocess.Popen(
        [sys.executable, "-c", CHILD, str(WORKSPACE), json.dumps(payload)],
        stdout=handle, stderr=subprocess.STDOUT,
    )

def already_done(job: Job, expected_folds: int | None = None) -> bool:
    """A run counts as done only if it is *complete*.

    Checking for the file alone is not enough: a short verification run writes a
    perfectly valid metrics.csv with one repetition, and the grid then skips the
    real job silently. Comparing the recorded fold count against what was asked
    for catches that, and also catches a run truncated by an OOM kill.
    """
    path = RESULTS_DIR / job.run_id / "metrics.csv"
    if not path.exists():
        return False
    if expected_folds is None:
        return True

    try:
        import pandas as pd

        frame = pd.read_csv(path, usecols=["repetition", "fold", "evaluation"])
    except Exception:
        return False

    observed = frame.loc[frame["evaluation"] == "cv", ["repetition", "fold"]]
    return len(observed.drop_duplicates()) >= expected_folds

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=runtime.recommended_workers())
    parser.add_argument("--only", default="", help="comma-separated arms to run")
    parser.add_argument(
        "--repetitions", type=int, default=None,
        help="override CV repetitions, use 3 for a fast first pass",
    )
    parser.add_argument("--list", action="store_true", help="show the grid and exit")
    parser.add_argument(
        "--status", action="store_true",
        help="report which runs are finished and exit, safe while a grid runs",
    )
    parser.add_argument("--force", action="store_true", help="re-run completed jobs")
    parser.add_argument(
        "--smoke", action="store_true",
        help="1 repetition, few rounds and epochs, verifies the grid runs",
    )
    args = parser.parse_args()

    if args.status:
        grid = build_grid()
        expected = CVConfig().repetitions * CVConfig().folds
        done = [j for j in grid if already_done(j, expected)]
        print(f"{len(done)}/{len(grid)} runs complete ({expected} folds each)")
        for job in grid:
            if already_done(job, expected):
                mark = "done"
            elif already_done(job):
                mark = "PART"
            else:
                mark = "    "
            print(f"  {mark}  {job.label}")
        index = RESULTS_DIR / "index.csv"
        if index.exists():
            print(f"\n{index}")
        return

    missing = [
        name for name in ("discovery.csv", "external.csv")
        if not (WORKSPACE / "data" / name).exists()
    ]
    if missing:
        sys.exit(
            f"Missing data snapshot: {', '.join(missing)}\n"
            "Run `python scripts/prepare_data.py` first, it builds and verifies\n"
            "both tables from the primary sources. It takes about a minute and\n"
            "only needs doing once."
        )

    jobs = build_grid()
    if args.only:
        wanted = {a.strip() for a in args.only.split(",")}
        jobs = [j for j in jobs if j.arm in wanted]

    overrides = {}
    if args.repetitions is not None:
        overrides["repetitions"] = args.repetitions
    if args.smoke:
        overrides = {
            "repetitions": 1,
            "train": {"epochs": 30, "early_stopping_patience": 8, "reduce_lr_patience": 4},
            "federated": {"num_rounds": 3, "local_epochs": 2},
        }

    expected_folds = None
    if not args.smoke:
        reps = args.repetitions if args.repetitions is not None else CVConfig().repetitions
        expected_folds = reps * CVConfig().folds

    if not args.force:
        pending = [j for j in jobs if not already_done(j, expected_folds)]
        skipped = len(jobs) - len(pending)
        jobs = pending
    else:
        skipped = 0

    info = runtime.describe()
    print(
        f"{len(jobs)} jobs ({skipped} already done)  "
        f"workers={args.workers}  quota={info['cpu_quota']:g} CPUs"
        + ("  [SMOKE]" if args.smoke else "")
    )
    if args.list or not jobs:
        for job in jobs:
            print(f"  {job.run_id:<62} {job.label}")
        return

    started = time.perf_counter()
    queue, running, failed = list(jobs), {}, []
    total = len(jobs)
    stopping = False

    def stop(signum, _frame):
        """Ctrl-C or SIGTERM: take the workers down with us.

        Without this the children outlive the parent and keep burning CPU quota
        on a shared host with nothing collecting their results.
        """
        nonlocal stopping
        if stopping:
            return
        stopping = True
        print(
            f"\nsignal {signum}, stopping {len(running)} running jobs. "
            "Finished runs are kept; relaunch to resume.",
            flush=True,
        )
        for process in running:
            process.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    last_heartbeat = time.perf_counter()

    while (queue or running) and not stopping:
        while queue and len(running) < args.workers and not stopping:
            job = queue.pop(0)
            running[launch(job, overrides)] = job

        time.sleep(1.0)
        for process in [p for p in running if p.poll() is not None]:
            job = running.pop(process)
            elapsed = time.perf_counter() - started
            done = total - len(queue) - len(running)
            if process.returncode == 0:
                print(
                    f"  [{elapsed / 60:6.1f} min] done   {done:>2}/{total}  {job.label}",
                    flush=True,
                )
            else:
                failed.append(job)
                print(
                    f"  [{elapsed / 60:6.1f} min] FAILED {done:>2}/{total}  {job.label}"
                    f"  -> results/logs/{job.run_id}.log",
                    flush=True,
                )

        # A long grid can go a while with nothing finishing; say so, otherwise a
        # detached log looks indistinguishable from a hung process.
        if time.perf_counter() - last_heartbeat > 600:
            last_heartbeat = time.perf_counter()
            print(
                f"  [{(time.perf_counter() - started) / 60:6.1f} min] "
                f"running {len(running)}, queued {len(queue)}: "
                + ", ".join(sorted(j.label for j in running.values())[:4]),
                flush=True,
            )

    for process in running:
        process.wait()

    completed = sum(1 for j in jobs if already_done(j, expected_folds))
    print(
        f"\n{completed}/{len(jobs)} succeeded "
        f"in {(time.perf_counter() - started) / 60:.1f} min"
    )
    if failed:
        print("failed:")
        for job in failed:
            print(f"  {job.run_id}")
        sys.exit(1)

if __name__ == "__main__":
    main()
