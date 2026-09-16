"""
Experiment configuration.

Every arm imports its settings from here, so that no cohort list, fold count,
hyperparameter or path is redefined elsewhere.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

import os

WORKSPACE = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE / "data"
#: Overridable so a grid can be re-run beside an existing one instead of over
#: it. Set METSCORE_RESULTS_DIR to write elsewhere; the analysis scripts read the same
#: variable, so they follow whichever grid is selected.
RESULTS_DIR = Path(os.environ.get("METSCORE_RESULTS_DIR", WORKSPACE / "results"))

# Original (read-only) data location. The first-submission experiment folders
# under approach_analysis_cv/ are deliberately never written to.
SOURCE_DATA = (WORKSPACE.parent.parent.parent / "data").resolve()

# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

#: Spreadsheet mapping every candidate NMR variable to its hierarchical cluster
#: and marking the 22 selected representatives. This is the feature selection of
#: the original MetSCORE study (Gil-Redondo 2024); `scripts/prepare_data.py`
#: reads it directly, so the panel is derived from source rather than inherited
#: from opaque intermediate files.
SELECTION_FILE = "clusters_of_variables_with_final_selection.xlsx"

#: Lipoprotein parameters. Each is one representative NMR variable renamed to its
#: cluster label, e.g. the raw column ``L5TG`` is reported as ``LDLs 4-5``.
#: Selection and renaming both come from SELECTION_FILE; nothing is aggregated.
LIPOPROTEINS = [
    "LDLs 4-5",
    "HDLs 1-3 noTGs",
    "HDLs 1-3 TGs",
    "HDLs 4 noTGs",
    "VLDLs",
    "LDLs 1-3",
    "LDLs 6",
]

#: Metabolites and metabolite ratios.
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

#: The 22-biomarker panel validated in the original MetSCORE study. Held fixed
#: across all arms so that the comparison isolates the training paradigm.
BIOMARKERS = LIPOPROTEINS + METABOLITES

TARGET = "metS"

#: Metadata carried alongside the biomarkers. Needed for the stratified results
#:
#: Coverage is complete for DISCOVERY. In the external cohort it is partial and
#: differs by site: BPH has age, sex and bmi; GRAZ has sex only; TUEBINGEN has
#: bmi only, so external stratification is reported over the sites that carry
#: the variable, with the covered fraction stated.
METADATA = ["cohort", "site", "age", "sex", "bmi", "metS_profile"]

# --------------------------------------------------------------------------
# Cohorts
# --------------------------------------------------------------------------

COHORTS = {
    "akribea": "AKRIBEA",
    "osarten": "OSARTEN",
    "milan": "LIVERBIBLE",
    "portugal": "AGEPORTUGAL",
    "metsplus": "METS+",
}

#: that a silent change to a source file cannot pass unnoticed.
COHORT_SIZES = {
    "AKRIBEA": 11159,
    "OSARTEN": 8142,
    "LIVERBIBLE": 1057,
    "AGEPORTUGAL": 157,
    "METS+": 147,
}

DISCOVERY_N = 20662

#: The three VALIDATION sub-cohorts, each in its own source spreadsheet.
#: Prevalence varies enormously between them, 2.9% in GRAZ against 77.6% in
#: TUEBINGEN, so a single pooled external AUROC conceals more than it reports.
EXTERNAL_SITES = {
    "BPH": "validation_BPH_glucose110_20240411.xlsx",
    "GRAZ": "validation_GRAZ_glucose110_20240411.xlsx",
    "TUEBINGEN": "validation_TUEBINGEN_glucose110_20240411.xlsx",
}

EXTERNAL_SITE_SIZES = {"BPH": 324, "GRAZ": 279, "TUEBINGEN": 58}
EXTERNAL_N = 661

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

class ScalingContract(str, Enum):
    """How features are standardized at evaluation time."""

    #: PRIMARY. Apply the scaler fitted on training data. The only contract that
    #: can score a single patient without seeing the rest of the cohort.
    TRAIN_FITTED = "train_fitted"

    #: SECONDARY. Re-standardize using statistics estimated from a *reference
    #: split* held out at the target site, then applied to the scored samples.
    #: Legitimate domain adaptation; reported for every arm or for none.
    SITE_ADAPTED = "site_adapted"

    # A pooled/transductive contract (fitting on the scored samples themselves)
    # is deliberately not representable here.

#: Fraction of a target site's data reserved to estimate SITE_ADAPTED statistics.
#: These rows are excluded from scoring, so the contract stays non-transductive.
SITE_REFERENCE_FRACTION = 0.3

# --------------------------------------------------------------------------
# Experiment configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """The architecture shared by every arm."""

    hidden_units: tuple[int, ...] = (128, 96, 64, 32)
    dropout: tuple[float, ...] = (0.35, 0.30, 0.25, 0.20)
    batch_norm: bool = True
    l1: float = 1e-4
    l2: float = 1e-4
    learning_rate: float = 1e-3
    activation: str = "relu"
    kernel_initializer: str = "he_normal"

@dataclass(frozen=True)
class ImbalanceConfig:
    """One convention for the whole study."""

    downsample: bool = True
    majority_to_minority_ratio: float = 5.0
    #: Below this many minority samples, downsampling is skipped entirely.
    min_minority: int = 10
    class_weights_after_downsampling: bool = True

@dataclass(frozen=True)
class CVConfig:
    repetitions: int = 10
    folds: int = 5
    seed: int = 42
    #: Fraction of the training folds held out for validation. Model selection,
    #: early stopping and threshold selection all read from this split, never
    #: from the test fold.
    validation_fraction: float = 0.2

@dataclass(frozen=True)
class ThresholdConfig:
    """Where the operating point comes from (fix for F4)."""

    #: Selection data. "test" is intentionally not an accepted value.
    source: str = "validation"
    criterion: str = "youden"
    resolution: int = 1000
    min_threshold: float = 0.01
    max_threshold: float = 0.99

    def __post_init__(self) -> None:
        if self.source != "validation":
            raise ValueError(
                "Threshold selection must read from validation data (F4). "
                f"Got source={self.source!r}."
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

    def __post_init__(self) -> None:
        allowed = {"federated_validation", "final_round"}
        if self.checkpoint_selection not in allowed:
            raise ValueError(
                "Federated checkpoint may not be selected on the test folds (F2). "
                f"Expected one of {sorted(allowed)}, got "
                f"{self.checkpoint_selection!r}."
            )

@dataclass(frozen=True)
class ExperimentConfig:
    """Everything an arm needs, serialisable for provenance."""

    name: str
    arm: str  # "unified" | "individual" | "federated"
    scaling: ScalingContract = ScalingContract.TRAIN_FITTED
    model: ModelConfig = field(default_factory=ModelConfig)
    imbalance: ImbalanceConfig = field(default_factory=ImbalanceConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    #: Aggregation strategy, federated arm only.
    strategy: str = "fedavg"
    strategy_params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["scaling"] = self.scaling.value
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
