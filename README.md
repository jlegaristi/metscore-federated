# Federated prediction of metabolic syndrome from NMR metabolomics

Code accompanying the paper *Federated Learning for Metabolic Syndrome
Prediction: Privacy-Preserving Multi-Institutional Analysis Using NMR
Metabolomics*.

The repository trains and evaluates three paradigms on the same data and under
the same protocol:

- **centralized**, a single model trained on the pooled cohorts;
- **institution-specific**, one model per cohort, trained on that cohort alone;
- **federated**, one client per cohort, with five aggregation rules (FedAvg,
  uniformly weighted FedAvg, FedProx, coordinate-wise median and trimmed mean)
  and optional client-level differential privacy.

Aggregation uses Flower's implementations, called in process rather than through
its simulation engine.

## Requirements

Python 3.11 with the pinned versions in `requirements.txt`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

TensorFlow 2.17 constrains `protobuf < 5`, which in turn constrains Flower to
1.15.

## Data

The metabolomic data are not included. They were collected under the cohort
agreements described in the original MetSCORE study and remain under the
custody of the originating institutions and their ethics committees, so they
cannot be redistributed here. Access requires authorization from those
institutions.

With the source files in place, `scripts/prepare_data.py` writes the snapshot
the experiments consume:

```bash
python scripts/prepare_data.py
```

| file | rows | contents |
|---|---|---|
| `data/discovery.csv` | 20,662 | 22 biomarkers, label, cohort, age, sex, bmi |
| `data/external.csv` | 661 | the same columns for three validation sites |

## Running the experiments

```bash
python scripts/run_experiments.py --list        # the grid
python scripts/run_experiments.py --smoke       # one fold per arm
python scripts/run_experiments.py               # the full grid
python scripts/run_experiments.py --only federated --workers 12
```

Each job is one (arm, strategy, scaling contract) run over 10 repetitions of
five-fold cross-validation, in its own single-threaded process with the GPU
hidden. The model is small enough that this is faster than a GPU; see
`scripts/benchmark.py`.

Every run is seeded per fold, so a configuration reproduces its results exactly.

## Output layout

```
results/
  index.csv                     one row per run
  <run_id>/
    config.json                 the configuration that produced it
    metrics.csv                 long format, one row per measurement
    profile.json                wall clock, compute time, communication
    predictions/                per-fold subject-level scores
```

`metrics.csv` carries every metric for every evaluation (cross-validation, the
external cohort under each scaling contract) and, where applicable, within
strata by site, sex and age band.

## Analysis

The analysis scripts read `results/` and produce the tables and figures reported
in the paper. Tables are written as LaTeX so that no number is transcribed by
hand.

```bash
cd analysis
python make_tables.py           # tables/*.tex
python statistics.py            # pairwise tests, privacy budgets
MPLBACKEND=Agg python make_figures.py
```

## Layout

```
metscore/
  config.py        experiment configuration, imported by everything else
  data.py          loading and fold assignment
  model.py         the network
  imbalance.py     downsampling and class weights
  scaling.py       standardization contracts
  arms/            centralized, institution-specific and federated training
  strategies.py    aggregation rules, built on Flower
  privacy.py       (epsilon, delta) accounting for the differential-privacy runs
  evaluation.py    metrics and operating points
  stats.py         DeLong, bootstrap over subjects, Holm correction
  results.py       writing and reading runs
  profiling.py     communication, compute and wall-clock accounting
  runtime.py       process setup and scheduling
  plotting.py      figure style
scripts/           data preparation, experiment runner, hardware benchmark
analysis/          tables, statistics and figures
```

## License

Released under the MIT License; see `LICENSE`.

## Citation

TODO: add the reference once the paper is published.
