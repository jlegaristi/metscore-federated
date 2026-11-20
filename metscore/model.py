"""The network used by every arm.

Four hidden layers (128, 96, 64, 32) with ReLU, He initialization, L1/L2 penalties,
batch normalization and dropout, and a sigmoid output. The differentially private
runs replace batch normalization by layer normalization.
"""

from __future__ import annotations

import numpy as np

from .config import ModelConfig


def build(input_dim: int, cfg: ModelConfig | None = None):
    """Build and compile the classifier."""
    import keras
    from keras.layers import BatchNormalization, Dense, Dropout, Input, LayerNormalization
    from keras.models import Sequential
    from keras.optimizers import Adam
    from keras.regularizers import l1_l2

    cfg = cfg or ModelConfig()
    if len(cfg.hidden_units) != len(cfg.dropout):
        raise ValueError("hidden_units and dropout must have the same length")

    layers = [Input(shape=(input_dim,))]
    for units, drop in zip(cfg.hidden_units, cfg.dropout):
        layers.append(
            Dense(
                units,
                activation=cfg.activation,
                kernel_regularizer=l1_l2(l1=cfg.l1, l2=cfg.l2),
                kernel_initializer=cfg.kernel_initializer,
            )
        )
        if cfg.normalization == "batch":
            layers.append(BatchNormalization())
        elif cfg.normalization == "layer":
            layers.append(LayerNormalization())
        else:
            raise ValueError(f"unknown normalization: {cfg.normalization!r}")
        layers.append(Dropout(drop))

    layers.append(Dense(1, activation="sigmoid"))

    model = Sequential(layers)
    model.compile(
        optimizer=Adam(learning_rate=cfg.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def parameter_count(input_dim: int, cfg: ModelConfig | None = None) -> int:
    """Number of trainable parameters."""
    return int(sum(np.prod(w.shape) for w in build(input_dim, cfg).trainable_weights))


def transmitted_parameter_count(input_dim: int, cfg: ModelConfig | None = None) -> int:
    """Number of values exchanged per update, for the communication cost.

    FedAvg sends `get_weights()`, which includes the running statistics of batch
    normalization on top of the trainable weights.
    """
    net = build(input_dim, cfg)
    return int(sum(np.prod(w.shape)
                   for w in net.trainable_weights + net.non_trainable_weights))
