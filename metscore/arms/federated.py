"""
Federated training arm.

Each cohort is a client holding its own data, its own scaler and its own
validation split. Rounds proceed as in FedAvg: the server broadcasts weights,
clients train locally, and the server aggregates through `strategies`.

The checkpoint kept is the one with the best federated validation score, the
sample-weighted mean of each client's local validation AUROC. Batch
normalization statistics travel with the weights and are averaged with them.
The global scaler is pooled exactly from each client's (mean, variance, count),
so no record-level value is exchanged.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import data as data_mod
from .. import (
    evaluation, imbalance, model, privacy, profiling, results, scaling, strategies, runtime,
)
from ..config import BIOMARKERS, TARGET, ExperimentConfig, ScalingContract
from .centralised import _external_sets, _record_strata

class Client:
    """One participating institution."""

    def __init__(self, name, train, validation, cfg, seed):
        self.name = name
        self.cfg = cfg
        self.seed = seed

        x_train = data_mod.features(train).to_numpy()
        y_train = data_mod.labels(train).to_numpy()

        # Per-site normalisation: statistics never leave this client except as
        # the aggregate used to build the global scaler.
        self.scaler = scaling.fit_training_scaler(x_train)
        x_scaled = scaling.transform_internal(self.scaler, x_train)

        self.x, self.y, self.class_weight, _ = imbalance.prepare_training_set(
            x_scaled, y_train, cfg.imbalance, seed=seed
        )
        self.imbalance_log = imbalance.describe(y_train, self.y)

        self.x_val = scaling.transform_internal(
            self.scaler, data_mod.features(validation).to_numpy()
        )
        self.y_val = data_mod.labels(validation).to_numpy()
        self.n = len(self.x)

        # Built once. Compiling a Keras model costs far more than a local epoch
        # at this size, and a client is asked to train once per round for every
        # round of every fold, rebuilding it each time dominated the runtime.
        self.net = model.build(len(BIOMARKERS), self.cfg.model)
        self.proximal = None

    def fit(self, global_weights, spec: strategies.StrategySpec):
        """One round of local training. Returns weights, sample count, seconds."""
        net = self.net
        net.set_weights(global_weights)

        started = time.perf_counter()
        if spec.proximal_mu > 0:
            if self.proximal is None:
                self.proximal = ProximalTrainer(
                    net, self.x, self.y, spec.proximal_mu,
                    self.cfg.train.batch_size, self.class_weight,
                )
            self.proximal.run(self.cfg.federated.local_epochs)
        else:
            net.fit(
                self.x, self.y,
                epochs=self.cfg.federated.local_epochs,
                batch_size=self.cfg.train.batch_size,
                class_weight=self.class_weight,
                verbose=0,
            )
        elapsed = time.perf_counter() - started
        return net.get_weights(), self.n, elapsed

    def validation_score(self, weights) -> tuple[float, int]:
        """Local validation AUROC. The only signal used to pick a checkpoint.

        Returns NaN if the model has diverged. That is not an error condition to
        be swallowed: under enough differential-privacy noise the global model
        does blow up, and the run should record the collapse as a point on the
        privacy-utility curve rather than crash and lose the whole sweep.
        """
        from sklearn.metrics import roc_auc_score

        if len(np.unique(self.y_val)) < 2:
            return float("nan"), 0

        self.net.set_weights(weights)
        scores = self.net(self.x_val, training=False).numpy().ravel()
        if not np.all(np.isfinite(scores)):
            return float("nan"), 0
        return float(roc_auc_score(self.y_val, scores)), len(self.y_val)

class ProximalTrainer:
    """FedProx local training: local loss plus (mu/2) * ||w - w_global||^2.

    Li et al. (2020). The term restrains clients from drifting far from the
    global model during local training, which is what makes it worth testing on
    data as heterogeneous as this: MetS prevalence spans 0.83% to 63.95% across
    clients, so unconstrained local epochs pull hard in five directions.

    Built once per client and reused across rounds. Tracing a fresh
    `tf.function` on every call (which is what happens if the decorator sits
    inside the training function) retains one traced graph per call. Over
    10 repetitions x 5 folds x 20 rounds x 5 clients that is 5,000 retained
    graphs, and it exhausted a 96 GB container part-way through an overnight
    run. The global weights are held in reassignable variables so each round
    updates values rather than triggering a retrace.
    """

    def __init__(self, net, x, y, mu, batch_size, class_weight):
        import tensorflow as tf

        self.net = net
        self.mu = float(mu)

        sample_weight = np.ones(len(y), dtype="float32")
        if class_weight:
            sample_weight = np.where(
                y == 1, class_weight.get(1, 1.0), class_weight.get(0, 1.0)
            ).astype("float32")

        self.dataset = (
            tf.data.Dataset.from_tensor_slices(
                (x.astype("float32"), y.astype("float32"), sample_weight)
            )
            .shuffle(len(x), seed=0, reshuffle_each_iteration=True)
            .batch(batch_size)
        )

        # Trainable weights only: the proximal term is defined over parameters
        # the optimiser moves, not over batch-norm running statistics.
        self.reference = [
            tf.Variable(w, trainable=False, dtype=tf.float32)
            for w in net.trainable_weights
        ]
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.step = tf.function(self._step, reduce_retracing=True)

    def _step(self, xb, yb, wb):
        import tensorflow as tf

        with tf.GradientTape() as tape:
            predictions = tf.squeeze(self.net(xb, training=True), axis=-1)
            loss = self.loss_fn(yb, predictions, sample_weight=wb)
            proximal = tf.add_n([
                tf.reduce_sum(tf.square(w - g))
                for w, g in zip(self.net.trainable_weights, self.reference)
            ])
            loss = loss + (self.mu / 2.0) * proximal
        self.net.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.net.trainable_weights), self.net.trainable_weights)
        )
        return loss

    def run(self, epochs):
        """One round of local training against the current global weights."""
        for variable, current in zip(self.reference, self.net.trainable_weights):
            variable.assign(current)  # net already holds the broadcast weights
        for _ in range(epochs):
            for xb, yb, wb in self.dataset:
                self.step(xb, yb, wb)

def run(cfg: ExperimentConfig, repetitions: int | None = None) -> str:
    """Execute the federated arm end to end and write its results."""
    spec = strategies.build(cfg.strategy, **cfg.strategy_params)
    discovery = data_mod.load_discovery()
    external_sets = _external_sets(data_mod.load_external())

    writer = results.RunWriter(cfg)
    timer = profiling.Timer()
    n_reps = repetitions if repetitions is not None else cfg.cv.repetitions
    started = time.perf_counter()
    round_times: list[float] = []
    client_compute = 0.0

    # One compiled model reused for every scoring call in every fold.
    evaluator = model.build(len(BIOMARKERS), cfg.model)

    for repetition in range(n_reps):
        seed = cfg.cv.seed + repetition
        folds = data_mod.assign_folds(discovery, cfg.cv.folds, seed)

        for fold in range(cfg.cv.folds):
            train, validation, test = data_mod.split_fold(
                discovery, folds, fold, cfg.cv.validation_fraction, seed
            )

            # Seeded per fold, so folds stay independent draws while the run
            # as a whole is reproducible. Set before any client is built, since
            # each one initialises its own network.
            runtime.seed_everything(seed * 100 + fold)
            clients = [
                Client(
                    name,
                    train.loc[train["cohort"] == name],
                    validation.loc[validation["cohort"] == name],
                    cfg, seed * 100 + fold,
                )
                for name in sorted(discovery["cohort"].unique())
                if train.loc[train["cohort"] == name, TARGET].nunique() >= 2
            ]
            if len(clients) < 2:
                continue

            # --- federated rounds
            # Fresh initialisation per fold, matching the centralised arms.
            # Once per fold is cheap; once per round was what made this slow.
            weights = [
                np.asarray(w) for w in model.build(len(BIOMARKERS), cfg.model).get_weights()
            ]
            best = {"score": -np.inf, "weights": weights, "round": 0}

            for round_index in range(cfg.federated.num_rounds):
                round_started = time.perf_counter()
                updates = []
                for client in clients:
                    local_weights, n, seconds = client.fit(weights, spec)
                    updates.append((local_weights, n))
                    client_compute += seconds

                with timer.measure("aggregate"):
                    if spec.needs_global_weights:
                        weights = spec.aggregate(updates, global_weights=weights)
                    else:
                        weights = spec.aggregate(updates)

                scores = [c.validation_score(weights) for c in clients]
                valid = [(s, n) for s, n in scores if s == s and n]
                federated_score = (
                    sum(s * n for s, n in valid) / sum(n for _, n in valid)
                    if valid else float("nan")
                )
                if (
                    cfg.federated.checkpoint_selection == "federated_validation"
                    and federated_score == federated_score
                    and federated_score > best["score"]
                ):
                    best = {
                        "score": federated_score,
                        "weights": [np.array(w) for w in weights],
                        "round": round_index,
                    }
                round_times.append(time.perf_counter() - round_started)

            final = (
                best["weights"]
                if cfg.federated.checkpoint_selection == "federated_validation"
                else weights
            )

            # --- the global model, and the federated scaler that goes with it
            net = evaluator
            net.set_weights(final)
            global_scaler = scaling.aggregate_scalers([c.scaler for c in clients])

            # Operating point from pooled client validation, still not test.
            x_val = scaling.transform_internal(
                global_scaler, data_mod.features(validation).to_numpy()
            )
            y_val = data_mod.labels(validation).to_numpy()
            point = evaluation.select_threshold(
                y_val, net.predict(x_val, verbose=0).ravel(), cfg.threshold
            )

            # --- cross-validation on the untouched test fold
            with timer.measure("evaluate_cv"):
                x_test = scaling.transform_internal(
                    global_scaler, data_mod.features(test).to_numpy()
                )
                y_test = data_mod.labels(test).to_numpy()
                p_test = net.predict(x_test, verbose=0).ravel()

                writer.record(
                    evaluation.evaluate(y_test, p_test, point), "cv", repetition, fold
                )
                _record_strata(
                    writer, test, y_test, p_test, point, "cv", repetition, fold, "ALL"
                )
                for name, block in test.groupby("cohort", sort=True):
                    mask = test["cohort"].to_numpy() == name
                    if len(np.unique(y_test[mask])) < 2:
                        continue
                    writer.record(
                        evaluation.evaluate(y_test[mask], p_test[mask], point),
                        "cv", repetition, fold, site=name,
                    )
                writer.record_predictions(
                    repetition, fold, "cv", y_test, p_test,
                    site=test["cohort"].to_numpy(), subject_index=test.index.to_numpy(),
                )

            # --- external, under the declared contract
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
                            contract, evaluation_set, global_scaler, subset=rows
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

    # Report the update-norm distribution so the clipping norm is defensible.
    clipping_report = None
    aggregator = getattr(spec, "aggregate", None)
    if isinstance(aggregator, strategies.DifferentialPrivacy) and aggregator.observed_norms:
        norms = np.asarray(aggregator.observed_norms)
        budget = privacy.compute_epsilon(
            noise_multiplier=aggregator.noise_multiplier,
            rounds=cfg.federated.num_rounds,
            clients=len(clients),
            clipping_norm=aggregator.clipping_norm,
        )
        clipping_report = {
            **budget.to_dict(),
            "update_norm_median": round(float(np.median(norms)), 4),
            "update_norm_p90": round(float(np.quantile(norms, 0.9)), 4),
            "fraction_clipped": round(float((norms > aggregator.clipping_norm).mean()), 4),
        }

    communication = profiling.CommunicationCost(
        n_parameters=model.transmitted_parameter_count(len(BIOMARKERS), cfg.model),
        n_clients=len(discovery["cohort"].unique()),
        n_rounds=cfg.federated.num_rounds,
    )
    profile = profiling.RunProfile(
        arm=cfg.arm,
        wall_clock_s=time.perf_counter() - started,
        client_compute_s=client_compute,
        n_rounds=cfg.federated.num_rounds,
        n_clients=len(discovery["cohort"].unique()),
        communication=communication.summary(),
        per_round_s=round_times,
        timers=timer.summary(),
    )
    payload = profile.to_dict()
    if clipping_report:
        payload["differential_privacy"] = clipping_report
    return str(
        writer.close(profile=payload, notes=f"strategy={spec.name} {spec.params}")
    )
