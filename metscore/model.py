"""
The network, built identically by every arm.

Four hidden layers of 128, 96, 64 and 32 units, each with ReLU activation, He
normal initialization, L1/L2 regularization, batch normalization and dropout
decreasing with depth, followed by a single sigmoid output.
"""

from __future__ import annotations

import numpy as np

from .config import ModelConfig

def build(input_dim: int, cfg: ModelConfig | None = None):
    """Compile the MetSCORE classifier.

    Imported by every arm. If an arm needs a different network, that is a change
    to ModelConfig, not a second definition elsewhere.
    """
    import keras
    from keras.layers import BatchNormalization, Dense, Dropout, Input
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
        if cfg.batch_norm:
            layers.append(BatchNormalization())
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
    """Trainable parameters, the figure quoted as the size of the model."""
    return int(sum(np.prod(w.shape) for w in build(input_dim, cfg).trainable_weights))

def transmitted_parameter_count(input_dim: int, cfg: ModelConfig | None = None) -> int:
    """Every value that crosses the wire, which feeds the communication cost.

    FedAvg sends `get_weights()`, so the batch normalization running statistics
    travel with the trainable weights and are averaged along with them. Counting
    only the trainable parameters understates the traffic by 2.6%.
    """
    net = build(input_dim, cfg)
    return int(sum(np.prod(w.shape)
                   for w in net.trainable_weights + net.non_trainable_weights))
