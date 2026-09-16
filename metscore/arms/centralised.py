"""
Centralized training arms.

The unified arm pools every cohort; the individual arm trains on one cohort at a
time. Both are built from this module, so the only difference between them is
the data each one sees.

Order of operations per fold:

  1. cut the fold        shared assignment, identical for every arm
  2. hold out validation from the training folds only
  3. fit the scaler      on training data only
  4. rebalance           downsampling and class weights, training data only
  5. train
  6. pick the threshold  on validation data, then freeze it
  7. score               the test fold, and the external cohort
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import data as data_mod
from .. import evaluation, imbalance, model, profiling, results, runtime, scaling
from ..config import BIOMARKERS, TARGET, ExperimentConfig, ScalingContract

#: Fixed seed for partitioning external sites into reference/scored. Held
#: constant across every run so all approaches are scored on identical subjects.
EXTERNAL_PARTITION_SEED = 20261003

def _external_sets(external: pd.DataFrame) -> dict[str, scaling.EvaluationSet]:
    """Partition each external site once, identically for every run."""
    return {
        site: scaling.partition_site(block, site, seed=EXTERNAL_PARTITION_SEED)
        for site, block in external.groupby("site", sort=True)
    }

def _record_strata(
    writer: results.RunWriter,
    frame: pd.DataFrame,
    y: np.ndarray,
    p: np.ndarray,
    point: evaluation.OperatingPoint,
    evaluation_name: str,
    repetition: int,
    fold: int,
    site: str,
) -> None:
    """Per-cohort, per-sex and per-age-band breakdowns."""
    strata = {"cohort": frame["cohort"].to_numpy()}
    if "sex" in frame and frame["sex"].notna().any():
        strata["sex"] = frame["sex"].astype(str).to_numpy()
    if "age" in frame and frame["age"].notna().any():
        strata["age_band"] = evaluation.age_bands(frame["age"].to_numpy())

    for kind, values in strata.items():
        for level, metrics in evaluation.evaluate_by_stratum(y, p, values, point).items():
            writer.record(
                metrics, evaluation_name, repetition, fold,
                site=site, stratum_kind=kind, stratum=level,
            )

def train_fold(
    cfg: ExperimentConfig,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    seed: int,
):
    """Steps 3-7. Returns the fitted model, scaler, threshold and imbalance log."""
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    x_train = data_mod.features(train).to_numpy()
    y_train = data_mod.labels(train).to_numpy()
    x_val = data_mod.features(validation).to_numpy()
    y_val = data_mod.labels(validation).to_numpy()

    scaler = scaling.fit_training_scaler(x_train)
    x_train_scaled = scaling.transform_internal(scaler, x_train)
    x_val_scaled = scaling.transform_internal(scaler, x_val)

    x_balanced, y_balanced, weights, _ = imbalance.prepare_training_set(
        x_train_scaled, y_train, cfg.imbalance, seed=seed
    )
    log = imbalance.describe(y_train, y_balanced)

    net = model.build(len(BIOMARKERS), cfg.model)
    net.fit(
        x_balanced, y_balanced,
        validation_data=(x_val_scaled, y_val),
        epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        class_weight=weights,
        verbose=0,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=cfg.train.early_stopping_patience,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=cfg.train.reduce_lr_factor,
                patience=cfg.train.reduce_lr_patience,
                min_lr=cfg.train.min_lr,
            ),
        ],
    )

    # The operating point is chosen here, on validation, and never revisited.
    point = evaluation.select_threshold(
        y_val, net.predict(x_val_scaled, verbose=0).ravel(), cfg.threshold
    )
    return net, scaler, point, log

def run(
    cfg: ExperimentConfig,
    cohort: str | None = None,
    repetitions: int | None = None,
) -> str:
    """Execute one centralised arm end to end and write its results.

    ``cohort`` selects the individual arm's institution; ``None`` pools
    everything, which is the unified arm.
    """
    discovery = data_mod.load_discovery()
    external = data_mod.load_external()
    external_sets = _external_sets(external)

    # An individual arm trains on its own cohort. It keeps a handle on the full
    # DISCOVERY frame so that, after training, it can also be scored on the other
    # cohorts' held-out patients, the question of whether a large institution's
    # model would serve a small one without federating at all. That extra pass is
    # recorded under its own evaluation tag and changes nothing already reported.
    # Named distinctly on purpose: `pooled` is reused further down for the
    # concatenated external frames, and shadowing it silently made this
    # evaluation score the validation cohort from the second fold onward.
    all_cohorts = discovery
    if cohort is not None:
        discovery = discovery.loc[discovery["cohort"] == cohort].reset_index(drop=True)

    writer = results.RunWriter(cfg)
    timer = profiling.Timer()
    n_reps = repetitions if repetitions is not None else cfg.cv.repetitions
    started = time.perf_counter()
    imbalance_log = None

    for repetition in range(n_reps):
        seed = cfg.cv.seed + repetition
        folds = data_mod.assign_folds(discovery, cfg.cv.folds, seed)

        for fold in range(cfg.cv.folds):
            train, validation, test = data_mod.split_fold(
                discovery, folds, fold, cfg.cv.validation_fraction, seed
            )
            if train[TARGET].nunique() < 2 or test[TARGET].nunique() < 2:
                continue  # a tiny cohort can land a fold with only one class

            with timer.measure("train"):
                # Seeded per fold, so folds stay independent draws while the
                # run as a whole is reproducible.
                runtime.seed_everything(seed * 100 + fold)
                net, scaler, point, imbalance_log = train_fold(
                    cfg, train, validation, seed * 100 + fold
                )

            # --- cross-validation: the untouched test fold, natural prevalence
            with timer.measure("evaluate_cv"):
                x_test = scaling.transform_internal(scaler, data_mod.features(test).to_numpy())
                y_test = data_mod.labels(test).to_numpy()
                p_test = net.predict(x_test, verbose=0).ravel()

                writer.record(
                    evaluation.evaluate(y_test, p_test, point), "cv", repetition, fold
                )
                _record_strata(
                    writer, test, y_test, p_test, point, "cv", repetition, fold, "ALL"
                )
                writer.record_predictions(
                    repetition, fold, "cv", y_test, p_test,
                    site=test["cohort"].to_numpy(), subject_index=test.index.to_numpy(),
                )

            # --- an individual model applied to every other cohort's test fold
            if cohort is not None:
                with timer.measure("evaluate_cross_cohort"):
                    every_fold = data_mod.assign_folds(all_cohorts, cfg.cv.folds, seed)
                    held_out = all_cohorts.loc[every_fold == fold]
                    for other, block in held_out.groupby("cohort", sort=True):
                        y_o = data_mod.labels(block).to_numpy()
                        if len(np.unique(y_o)) < 2:
                            continue
                        x_o = scaling.transform_internal(
                            scaler, data_mod.features(block).to_numpy()
                        )
                        p_o = net.predict(x_o, verbose=0).ravel()
                        writer.record(
                            evaluation.evaluate(y_o, p_o, point),
                            "cv:cross_cohort", repetition, fold,
                            site=other, stratum_kind="cohort", stratum=other,
                        )

            # --- external: same subjects under whichever contract is declared
            with timer.measure("evaluate_external"):
                # Both contracts, one trained model. Scaling affects only how
                # external features are standardised, so re-training per
                # contract would burn half the compute for identical models.
                # Three passes from two contracts. TRAIN_FITTED needs nothing
                # from the target site, so it scores the whole cohort; the third
                # pass repeats it on the subset SITE_ADAPTED can reach, which is
                # where the two contracts may fairly be compared.
                passes = [
                    ("external:%s" % ScalingContract.TRAIN_FITTED.value,
                     ScalingContract.TRAIN_FITTED, False),
                    ("external:%s" % ScalingContract.SITE_ADAPTED.value,
                     ScalingContract.SITE_ADAPTED, False),
                    ("external:train_fitted_common",
                     ScalingContract.TRAIN_FITTED, True),
                ]
                for tag, contract, common_only in passes:
                    y_parts, p_parts, frames = [], [], []
                    for site, evaluation_set in external_sets.items():
                        rows = (evaluation_set.scored if common_only
                                else evaluation_set.subjects(contract))
                        x_site = scaling.transform(
                            contract, evaluation_set, scaler, subset=rows
                        )
                        y_site = data_mod.labels(rows).to_numpy()
                        p_site = net.predict(x_site, verbose=0).ravel()

                        writer.record(
                            evaluation.evaluate(y_site, p_site, point),
                            tag, repetition, fold, site=site,
                        )
                        y_parts.append(y_site)
                        p_parts.append(p_site)
                        frames.append(rows)

                    y_ext, p_ext = np.concatenate(y_parts), np.concatenate(p_parts)
                    pooled = pd.concat(frames)
                    writer.record(
                        evaluation.evaluate(y_ext, p_ext, point), tag, repetition, fold
                    )
                    _record_strata(
                        writer, pooled, y_ext, p_ext, point, tag, repetition, fold, "ALL"
                    )
                    writer.record_predictions(
                        repetition, fold, tag.replace(":", "_"), y_ext, p_ext,
                        site=pooled["site"].to_numpy(),
                        subject_index=pooled.index.to_numpy(),
                    )

    profile = profiling.RunProfile(
        arm=cfg.arm,
        wall_clock_s=time.perf_counter() - started,
        client_compute_s=timer.durations.get("train", 0.0),
        timers=timer.summary(),
    )
    path = writer.close(
        profile={**profile.to_dict(), "imbalance": imbalance_log},
        notes=f"cohort={cohort or 'ALL'}",
    )
    return str(path)

def run_unified(cfg: ExperimentConfig, **kwargs) -> str:
    return run(cfg, cohort=None, **kwargs)

def run_individual(cfg: ExperimentConfig, cohort: str, **kwargs) -> str:
    """The cohort must be part of the run id, or the five institutions collide."""
    if cfg.strategy != cohort:
        raise ValueError(
            f"individual runs must set strategy to the cohort so each gets its "
            f"own results folder; got strategy={cfg.strategy!r}, cohort={cohort!r}"
        )
    return run(cfg, cohort=cohort, **kwargs)
