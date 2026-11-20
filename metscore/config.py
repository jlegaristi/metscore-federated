"""Experiment configuration: paths, the biomarker panel, cohorts and hyperparameters."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

# Paths. The three directories can be set through the environment, so that the data
# can live outside the clone and a grid can be written next to an existing one.
WORKSPACE = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("METSCORE_DATA_DIR", WORKSPACE / "data"))
RESULTS_DIR = Path(os.environ.get("METSCORE_RESULTS_DIR", WORKSPACE / "results"))
# tables/ and figures/ are written under this one
OUTPUT_DIR = Path(os.environ.get("METSCORE_OUTPUT_DIR", WORKSPACE))

# Where scripts/prepare_data.py looks for the source files of the cohorts.
SOURCE_DATA = (WORKSPACE.parent.parent.parent / "data").resolve()

# Spreadsheet of the original MetSCORE study (Gil-Redondo et al., 2024) listing every
# candidate NMR variable, its cluster and the 22 that were selected.
SELECTION_FILE = "clusters_of_variables_with_final_selection.xlsx"

# The panel. Each entry is one NMR variable, named after its cluster where the study
# renamed it (the raw column L5TG is "LDLs 4-5").
LIPOPROTEINS = [
    "LDLs 4-5",
    "HDLs 1-3 noTGs",
    "HDLs 1-3 TGs",
    "HDLs 4 noTGs",
    "VLDLs",
    "LDLs 1-3",
    "LDLs 6",
]

METABOLITES = [
    "Ala-Tyr",
    "Citric acid",
    "Creatine",
    "Creatinine",
    "Formic acid",
    "Glucose",
    "Glutamic acid",
    "Glycine",
    "Iso-Leu-Val",
    "Lact-Pyru.acids",
    "Methionine",
    "Phenylalanine",
    "Proline",
    "Sarcosine",
    "Trimethylamine-N-oxide",
]

BIOMARKERS = LIPOPROTEINS + METABOLITES

TARGET = "metS"

# Columns kept next to the biomarkers for the stratified results. DISCOVERY has all of
# them. In the external cohort BPH has age, sex and bmi, GRAZ only sex and TUEBINGEN
# only bmi.
METADATA = ["cohort", "site", "age", "sex", "bmi", "metS_profile"]

# Cohorts: file stem -> name used in the paper
COHORTS = {
    "akribea": "AKRIBEA",
    "osarten": "OSARTEN",
    "milan": "LIVERBIBLE",
    "portugal": "AGEPORTUGAL",
    "metsplus": "METS+",
}

# Expected sizes, checked when the data snapshot is built.
COHORT_SIZES = {
    "AKRIBEA": 11159,
    "OSARTEN": 8142,
    "LIVERBIBLE": 1057,
    "AGEPORTUGAL": 157,
    "METS+": 147,
}

DISCOVERY_N = 20662

# The three sites of the VALIDATION cohort, one spreadsheet each.
EXTERNAL_SITES = {
    "BPH": "validation_BPH_glucose110_20240411.xlsx",
    "GRAZ": "validation_GRAZ_glucose110_20240411.xlsx",
    "TUEBINGEN": "validation_TUEBINGEN_glucose110_20240411.xlsx",
}

EXTERNAL_SITE_SIZES = {"BPH": 324, "GRAZ": 279, "TUEBINGEN": 58}
EXTERNAL_N = 661


class ScalingContract(str, Enum):
    """How external features are standardized before scoring."""

    # With the scaler fitted on the training data. Works for a single patient.
    TRAIN_FITTED = "train_fitted"

    # With statistics of the target site, estimated on a reference split that is
    # then left out of the scoring.
    SITE_ADAPTED = "site_adapted"


# Share of a target site used as the reference split of SITE_ADAPTED.
SITE_REFERENCE_FRACTION = 0.3


@dataclass(frozen=True)
class ModelConfig:
    """The network, the same in every arm."""

    hidden_units: tuple[int, ...] = (128, 96, 64, 32)
    dropout: tuple[float, ...] = (0.35, 0.30, 0.25, 0.20)
    # "batch" or "layer". The differentially private runs need "layer", because the
    # running statistics of batch normalization cannot take noise.
    normalization: str = "batch"
    l1: float = 1e-4
    l2: float = 1e-4
    learning_rate: float = 1e-3
    activation: str = "relu"
    kernel_initializer: str = "he_normal"


@dataclass(frozen=True)
class ImbalanceConfig:
    downsample: bool = True
    majority_to_minority_ratio: float = 5.0
    # no downsampling below this many minority samples
    min_minority: int = 10
    class_weights_after_downsampling: bool = True
    # Centralized arm only: downsample every cohort on its own, as the federated
    # clients do, instead of the pooled training data (sensitivity analysis).
    within_cohort: bool = False


@dataclass(frozen=True)
class CVConfig:
    repetitions: int = 10
    folds: int = 5
    seed: int = 42
    # Share of the training folds held out as validation: early stopping, the choice
    # of federated round and the threshold all use it, never the test fold.
    validation_fraction: float = 0.2


@dataclass(frozen=True)
class ThresholdConfig:
    """How the operating point is chosen."""

    source: str = "validation"
    criterion: str = "youden"
    resolution: int = 1000
    min_threshold: float = 0.01
    max_threshold: float = 0.99

    def __post_init__(self) -> None:
        if self.source != "validation":
            raise ValueError(
                f"the threshold is chosen on validation data, got source={self.source!r}"
            )


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 1000
    batch_size: int = 16
    early_stopping_patience: int = 50
    reduce_lr_patience: int = 20
    reduce_lr_factor: float = 0.2
    min_lr: float = 1e-5


@dataclass(frozen=True)
class FederatedConfig:
    num_rounds: int = 20
    local_epochs: int = 10
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    checkpoint_selection: str = "federated_validation"
    # Adam starts every round from its initial state.
    reset_optimizer_each_round: bool = True

    def __post_init__(self) -> None:
        allowed = {"federated_validation", "final_round"}
        if self.checkpoint_selection not in allowed:
            raise ValueError(
                f"checkpoint_selection must be one of {sorted(allowed)}, "
                f"got {self.checkpoint_selection!r}"
            )


@dataclass(frozen=True)
class ExperimentConfig:
    """One run: an arm, a strategy and the settings above."""

    name: str
    arm: str  # "unified" | "individual" | "federated"
    scaling: ScalingContract = ScalingContract.TRAIN_FITTED
    model: ModelConfig = field(default_factory=ModelConfig)
    imbalance: ImbalanceConfig = field(default_factory=ImbalanceConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    # aggregation strategy (federated arm) and its parameters
    strategy: str = "fedavg"
    strategy_params: dict = field(default_factory=dict)
    # tells apart runs of one strategy that differ in some other setting
    variant: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["scaling"] = self.scaling.value
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
