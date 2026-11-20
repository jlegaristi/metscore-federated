"""Centralized training: one model on the pooled cohorts (unified arm) or on a single
cohort (individual arm). Both use the same code and differ only in the data.

Per fold: split into train, validation and test; fit the scaler on train; downsample
and weight the classes; train with early stopping on validation; choose the threshold
on validation; score the test fold and the external cohort.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import data as data_mod
from .. import evaluation, imbalance, model, profiling, results, runtime, scaling
from ..config import BIOMARKERS, TARGET, ExperimentConfig, ScalingContract

# Seed of the reference/scored split of the external sites, the same in every run
# so that all approaches are scored on the same subjects.
EXTERNAL_PARTITION_SEED = 20261003


def _external_sets(external: pd.DataFrame) -> dict[str, scaling.EvaluationSet]:
    """Reference/scored split of each external site."""
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
    """Metrics by cohort, sex and age band."""
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


def _downsample_by_cohort(x, y, cohort, cfg, seed):
    """Downsample every cohort separately and pool the result.

    This is the training data of the federated clients put together, so the
    centralized model can be trained on the samples they hold.
    """
    parts = [
        imbalance.downsample(x[cohort == name], y[cohort == name], cfg, seed)
        for name in sorted(np.unique(cohort))
    ]
    x_all = np.concatenate([part[0] for part in parts])
    y_all = np.concatenate([part[1] for part in parts])
    return x_all, y_all, imbalance.class_weights(y_all, cfg)


def train_fold(
    cfg: ExperimentConfig,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    seed: int,
):
    """Train one model. Returns it with its scaler, threshold and imbalance log."""
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    x_train = data_mod.features(train).to_numpy()
    y_train = data_mod.labels(train).to_numpy()
    x_val = data_mod.features(validation).to_numpy()
    y_val = data_mod.labels(validation).to_numpy()

    scaler = scaling.fit_training_scaler(x_train)
    x_train_scaled = scaling.transform_internal(scaler, x_train)
    x_val_scaled = scaling.transform_internal(scaler, x_val)

    if cfg.imbalance.within_cohort:
        x_balanced, y_balanced, weights = _downsample_by_cohort(
            x_train_scaled, y_train, train["cohort"].to_numpy(), cfg.imbalance, seed
        )
    else:
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

    # threshold from the validation split
    point = evaluation.select_threshold(
        y_val, net.predict(x_val_scaled, verbose=0).ravel(), cfg.threshold
    )
    return net, scaler, point, log


def run(
    cfg: ExperimentConfig,
    cohort: str | None = None,
    repetitions: int | None = None,
) -> str:
    """Run a centralized arm over all folds and write its results.

    `cohort` selects one institution (individual arm); None pools them all.
    """
    discovery = data_mod.load_discovery()
    external = data_mod.load_external()
    external_sets = _external_sets(external)

    # An individual model is also scored on the held-out patients of the other
    # cohorts (evaluation "cv:cross_cohort"), so the full frame is kept.
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
                continue  # a small cohort can leave a fold with one class

            with timer.measure("train"):
                # one seed per fold
                runtime.seed_everything(seed * 100 + fold)
                net, scaler, point, imbalance_log = train_fold(
                    cfg, train, validation, seed * 100 + fold
                )

            # cross-validation: the test fold
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

            # an individual model on the test fold of every other cohort
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
                        writer.record_predictions(
                            repetition, fold, f"cv_cross_{other}", y_o, p_o,
                            site=block["cohort"].to_numpy(),
                            subject_index=block.index.to_numpy(),
                        )

            # external cohort
            with timer.measure("evaluate_external"):
                # The same model under both scaling contracts, plus TRAIN_FITTED
                # on the subjects SITE_ADAPTED can score, to compare the two.
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
    """Individual arm. The cohort is the strategy, so each one has its own folder."""
    if cfg.strategy != cohort:
        raise ValueError(
            f"individual runs must set strategy to the cohort so each gets its "
            f"own results folder; got strategy={cfg.strategy!r}, cohort={cohort!r}"
        )
    return run(cfg, cohort=cohort, **kwargs)
