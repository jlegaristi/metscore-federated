"""Federated training.

Each cohort is a client with its own training and validation split. Before the first
round every client sends the count, mean and variance of its training features; the
server pools them into one scaler that all clients use for training and validation,
and that the global model uses afterwards. Then, in every round, the server sends the
weights, the clients train locally and the server aggregates their updates with the
rule in `strategies`.

The model kept is that of the round with the best federated validation score: the
mean of the clients' validation AUROCs, weighted by their validation sizes. The
running statistics of batch normalization are sent and averaged with the weights.
Clients only send weights and aggregate statistics, never records.
"""

from __future__ import annotations

import time
import zlib

import numpy as np
import pandas as pd

from .. import data as data_mod
from .. import (
    evaluation, imbalance, model, privacy, profiling, results, scaling, strategies, runtime,
)
from ..config import BIOMARKERS, TARGET, ExperimentConfig, ScalingContract
from .centralised import _external_sets, _record_strata


class Client:
    """One institution."""

    def __init__(self, name, train, validation, scaler, cfg, seed, n_clients):
        self.name = name
        self.cfg = cfg
        self.seed = seed
        self.n_clients = n_clients
        self.noise_rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
        self.update_norms: list[float] = []
        # the federated scaler, the same for every client
        self.scaler = scaler

        x_train = data_mod.features(train).to_numpy()
        y_train = data_mod.labels(train).to_numpy()
        x_scaled = scaling.transform_internal(self.scaler, x_train)

        self.x, self.y, self.class_weight, _ = imbalance.prepare_training_set(
            x_scaled, y_train, cfg.imbalance, seed=seed
        )
        self.imbalance_log = imbalance.describe(y_train, self.y)

        self.x_val = scaling.transform_internal(
            self.scaler, data_mod.features(validation).to_numpy()
        )
        self.y_val = data_mod.labels(validation).to_numpy()
        self.n = len(self.x)  # samples it trains on, after downsampling
        self.n_cohort = len(y_train)  # training cohort, before downsampling

        # built once per client: compiling the model takes longer than a round
        self.net = model.build(len(BIOMARKERS), self.cfg.model)
        self.trainer = None

    def fit(self, global_weights, spec: strategies.StrategySpec):
        """One round of local training. Returns the weights, n and the seconds it took."""
        net = self.net
        net.set_weights(global_weights)

        started = time.perf_counter()
        if self.trainer is None:
            self.trainer = LocalTrainer(
                net, self.x, self.y, spec.proximal_mu,
                self.cfg.train.batch_size, self.class_weight, self.seed,
            )
        if self.cfg.federated.reset_optimizer_each_round:
            self.trainer.reset_optimizer()
        self.trainer.run(self.cfg.federated.local_epochs)
        elapsed = time.perf_counter() - started

        weights = net.get_weights()
        if spec.privacy is not None:
            if net.non_trainable_weights:
                raise ValueError(
                    "differential privacy needs a model without running statistics; "
                    "use normalization='layer'"
                )
            weights, norm = privacy.privatise_update(
                weights, global_weights,
                spec.privacy.clipping_norm, spec.privacy.noise_multiplier,
                self.n_clients, self.noise_rng,
            )
            self.update_norms.append(norm)
        return weights, self.n, elapsed

    def validation_score(self, weights) -> tuple[float, int]:
        """AUROC of `weights` on the client's validation split, and its size.

        NaN if the predictions are not finite, which happens with strong
        differential-privacy noise.
        """
        from sklearn.metrics import roc_auc_score

        if len(np.unique(self.y_val)) < 2:
            return float("nan"), 0

        self.net.set_weights(weights)
        scores = self.net(self.x_val, training=False).numpy().ravel()
        if not np.all(np.isfinite(scores)):
            return float("nan"), 0
        return float(roc_auc_score(self.y_val, scores)), len(self.y_val)


class LocalTrainer:
    """Local training loop, the same for every aggregation strategy.

    Minimizes the class-weighted binary cross-entropy plus the L1/L2 penalties of
    the network plus (mu/2) * ||w - w_global||^2. mu is zero except for FedProx
    (Li et al., 2020), so the strategies share the rest of the objective.

    One trainer per client, reused in every round: the training step is a
    `tf.function` traced once, and the global weights are kept in variables that are
    reassigned each round.
    """

    def __init__(self, net, x, y, mu, batch_size, class_weight, shuffle_seed):
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
            .shuffle(len(x), seed=shuffle_seed, reshuffle_each_iteration=True)
            .batch(batch_size)
        )

        # the proximal term is over the trainable weights, not over the running
        # statistics of batch normalization
        self.reference = [
            tf.Variable(w, trainable=False, dtype=tf.float32)
            for w in net.trainable_weights
        ]
        self.step = tf.function(self._step, reduce_retracing=True)

        optimizer = net.optimizer
        if not optimizer.built:
            optimizer.build(net.trainable_variables)
        self._initial_optimizer_state = [v.numpy() for v in optimizer.variables]

    def reset_optimizer(self):
        """Put Adam back in its initial state (moments, step count, learning rate)."""
        for variable, initial in zip(self.net.optimizer.variables, self._initial_optimizer_state):
            variable.assign(initial)

    def objective(self, xb, yb, wb, training):
        """The local loss of a batch.

        `compute_loss` is what `Model.fit` minimizes in the centralized arms: the
        compiled loss with per-sample weights plus the L1/L2 penalties. (Calling the
        Keras loss directly on 1-D tensors averages over the batch before weighting.)
        """
        import tensorflow as tf

        loss = self.net.compute_loss(
            y=yb, y_pred=self.net(xb, training=training), sample_weight=wb, training=training,
        )
        if self.mu > 0:
            proximal = tf.add_n([
                tf.reduce_sum(tf.square(w - g))
                for w, g in zip(self.net.trainable_weights, self.reference)
            ])
            loss = loss + (self.mu / 2.0) * proximal
        return loss

    def _step(self, xb, yb, wb):
        import tensorflow as tf

        with tf.GradientTape() as tape:
            loss = self.objective(xb, yb, wb, training=True)
        self.net.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.net.trainable_weights), self.net.trainable_weights)
        )
        return loss

    def sync_reference(self):
        """Use the current weights of the network as w_global."""
        for variable, current in zip(self.reference, self.net.trainable_weights):
            variable.assign(current)

    def run(self, epochs):
        """Train for `epochs` epochs from the global weights the network holds."""
        self.sync_reference()
        for _ in range(epochs):
            for xb, yb, wb in self.dataset:
                self.step(xb, yb, wb)


def federated_scaler(train, cohorts):
    """The scaler of the federation.

    Each client fits a scaler on its own training split, which amounts to sending a
    count, a mean and a variance per feature. Pooled, they give the scaler the
    centralized arm fits on all the training splits together.
    """
    return scaling.aggregate_scalers([
        scaling.fit_training_scaler(
            data_mod.features(train.loc[train["cohort"] == name]).to_numpy()
        )
        for name in cohorts
    ])


def build_clients(train, validation, cfg, seed, cohorts=None):
    """The clients (cohorts with both classes in training) and the federated scaler."""
    names = sorted(train["cohort"].unique()) if cohorts is None else cohorts
    names = [n for n in names if train.loc[train["cohort"] == n, TARGET].nunique() >= 2]
    if not names:
        return [], None

    scaler = federated_scaler(train, names)
    clients = [
        Client(
            name,
            train.loc[train["cohort"] == name],
            validation.loc[validation["cohort"] == name],
            scaler, cfg, seed, len(names),
        )
        for name in names
    ]
    return clients, scaler


def run(cfg: ExperimentConfig, repetitions: int | None = None) -> str:
    """Run the federated arm over all folds and write its results."""
    spec = strategies.build(cfg.strategy, **cfg.strategy_params)
    if spec.privacy is not None and cfg.federated.checkpoint_selection != "final_round":
        raise ValueError(
            "the privacy budget covers the noisy models themselves; choosing among them "
            "with non-private validation scores falls outside it, so use 'final_round'"
        )
    update_norms: list[float] = []
    discovery = data_mod.load_discovery()
    external_sets = _external_sets(data_mod.load_external())

    writer = results.RunWriter(cfg)
    timer = profiling.Timer()
    n_reps = repetitions if repetitions is not None else cfg.cv.repetitions
    started = time.perf_counter()
    round_times: list[float] = []
    client_compute = 0.0

    # one model for all the scoring
    evaluator = model.build(len(BIOMARKERS), cfg.model)

    for repetition in range(n_reps):
        seed = cfg.cv.seed + repetition
        folds = data_mod.assign_folds(discovery, cfg.cv.folds, seed)

        for fold in range(cfg.cv.folds):
            train, validation, test = data_mod.split_fold(
                discovery, folds, fold, cfg.cv.validation_fraction, seed
            )

            # one seed per fold, set before the clients build their networks
            runtime.seed_everything(seed * 100 + fold)
            clients, global_scaler = build_clients(
                train, validation, cfg, seed * 100 + fold,
                cohorts=sorted(discovery["cohort"].unique()),
            )
            if len(clients) < 2:
                continue

            counts = [spec.client_weight(c) for c in clients]
            writer.record_aggregation_weights(repetition, fold, [
                {
                    "cohort": c.name,
                    "n_training": c.n_cohort,
                    "n_after_downsampling": c.n,
                    "weight_applied": (
                        count / sum(counts) if spec.weighting != "none" else float("nan")
                    ),
                }
                for c, count in zip(clients, counts)
            ])

            # federated rounds, from a fresh initialization
            weights = [
                np.asarray(w) for w in model.build(len(BIOMARKERS), cfg.model).get_weights()
            ]
            best = {"score": -np.inf, "weights": None, "round": None}

            for round_index in range(cfg.federated.num_rounds):
                round_started = time.perf_counter()
                updates = []
                for client in clients:
                    local_weights, _, seconds = client.fit(weights, spec)
                    updates.append((local_weights, spec.client_weight(client)))
                    client_compute += seconds

                with timer.measure("aggregate"):
                    weights = spec.aggregate(updates)

                scores = [c.validation_score(weights) for c in clients]
                valid = [(s, n) for s, n in scores if s == s and n]
                federated_score = (
                    sum(s * n for s, n in valid) / sum(n for _, n in valid)
                    if valid else float("nan")
                )
                writer.record_round(repetition, fold, round_index, federated_score)
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

            # If no round had a finite validation score the run has diverged, and
            # the model of the last round is evaluated (not the initial one).
            final = (
                best["weights"]
                if cfg.federated.checkpoint_selection == "federated_validation"
                and best["weights"] is not None
                else weights
            )

            for client in clients:
                update_norms.extend(client.update_norms)

            # the global model
            net = evaluator
            net.set_weights(final)

            # threshold from the validation counts of the clients
            point = evaluation.select_threshold_from_counts(
                [
                    evaluation.threshold_counts(
                        c.y_val, net.predict(c.x_val, verbose=0).ravel(), cfg.threshold
                    )
                    for c in clients
                ],
                cfg.threshold,
            )

            # cross-validation: the test fold
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

    # privacy budget and the update norms behind the choice of clipping norm
    clipping_report = None
    if spec.privacy is not None and update_norms:
        norms = np.asarray(update_norms)
        budget = privacy.compute_epsilon(
            noise_multiplier=spec.privacy.noise_multiplier,
            rounds=cfg.federated.num_rounds,
            clients=len(clients),
            clipping_norm=spec.privacy.clipping_norm,
        )
        clipping_report = {
            **budget.to_dict(),
            "update_norm_median": round(float(np.median(norms)), 4),
            "update_norm_p90": round(float(np.quantile(norms, 0.9)), 4),
            "fraction_clipped": round(float((norms > spec.privacy.clipping_norm).mean()), 4),
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
