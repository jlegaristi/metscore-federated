"""One timing measurement, in a process of its own.

TensorFlow reads the threading and device environment variables once, when it is
imported, so a single process cannot measure more than one configuration. This
worker is therefore launched once per configuration by `benchmark.py`.

Usage:  python scripts/_benchmark_worker.py {cpu|gpu} <threads>

Prints the mean milliseconds per epoch.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

DEVICE = sys.argv[1]
THREADS = int(sys.argv[2])

if DEVICE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.update(
    TF_CPP_MIN_LOG_LEVEL="3",
    OMP_NUM_THREADS=str(THREADS),
    TF_NUM_INTRAOP_THREADS=str(THREADS),
    TF_NUM_INTEROP_THREADS="1",
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf  # noqa: E402

if THREADS:
    tf.config.threading.set_intra_op_parallelism_threads(THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(1)
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

from metscore import data, imbalance, model, scaling  # noqa: E402
from metscore.config import BIOMARKERS, CVConfig, ImbalanceConfig  # noqa: E402

EPOCHS = 30


def main() -> None:
    cv = CVConfig()
    discovery = data.load_discovery()
    folds = data.assign_folds(discovery, cv.folds, cv.seed)
    train, _, _ = data.split_fold(discovery, folds, 0, cv.validation_fraction, cv.seed)

    scaler = scaling.fit_training_scaler(data.features(train))
    x = scaling.transform_internal(scaler, data.features(train))
    y = data.labels(train).to_numpy()
    x, y, weights, _ = imbalance.prepare_training_set(
        x, y, ImbalanceConfig(), seed=cv.seed
    )

    # One untimed epoch first, so that graph tracing is not counted.
    model.build(len(BIOMARKERS)).fit(x, y, epochs=1, batch_size=16, verbose=0)

    start = time.perf_counter()
    model.build(len(BIOMARKERS)).fit(
        x, y, epochs=EPOCHS, batch_size=16, class_weight=weights, verbose=0
    )
    print(f"{(time.perf_counter() - start) / EPOCHS * 1000:.0f}")


if __name__ == "__main__":
    main()
