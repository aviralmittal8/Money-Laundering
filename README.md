# Money-Laundering

AML (Anti-Money Laundering) transaction detection system that combines an ANN, a GNN, and a soft-voting ensemble to flag suspicious transactions in highly imbalanced financial data, with a Streamlit demo UI for live predictions and evaluation.

## Overview

- **ANN** — learns tabular transaction behavior patterns
- **GNN** — learns account-to-account network structure
- **Ensemble** — soft voting over ANN/GNN probabilities with a tuned decision threshold

The original dataset is extremely imbalanced (~1:980 positive-to-negative), so the project also includes rebalanced training/evaluation datasets and threshold-tuning logic to make recall/precision tradeoffs explicit rather than relying on raw accuracy.

See [`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md) for the full writeup (datasets, feature engineering, model design, experimental runs, known issues), and [`docs/METHODOLOGY_AND_FORMULAS.md`](docs/METHODOLOGY_AND_FORMULAS.md) / [`docs/DATASETS_AND_PREPROCESSING.md`](docs/DATASETS_AND_PREPROCESSING.md) for methodology details.

## Repository structure

```
main.py              CLI entrypoint for training and prediction
app.py                Streamlit frontend
src/
  data_loader.py       dataset loading, standardization, feature engineering, graph construction
  preprocess.py         numeric scaling + categorical hashing
  ann.py                 ANN architecture
  gnn.py                 GNN architecture
  ensemble.py             soft voting ensemble + probability transform logic
  train.py                 end-to-end training/evaluation pipeline
  predict.py                batch prediction pipeline
  evaluate.py                 metric calculation
  plots.py                     ROC/PR/confusion matrix plot generation
models/               trained model artifacts
  run_parent_1_150_tuned/  the tuned run app.py uses as its default view (included)
outputs/              metrics, plots, threshold files
  run_parent_1_150_tuned/  metrics/plots for the tuned run (included)
datasets/             rebalanced training/evaluation datasets (ignored, see below)
```

> Note: `HI-Small_Trans.csv` (the original ~454MB dataset), `datasets/`, other exploratory `models/run_*`/`outputs/run_*` directories, and top-level model checkpoints (`models/*.pt`, `models/*.pkl`) are excluded from this repo via `.gitignore` since they exceed GitHub's file size limits or are superseded by the tuned run. `models/run_parent_1_150_tuned/` and `outputs/run_parent_1_150_tuned/` **are** included — they're small (~21MB combined) and are what `app.py` loads by default, so the demo UI works out of the box after cloning. Small demo CSVs (`HI-Small_Trans_1000_demo*.csv`) are included too.

## Setup

```powershell
pip install numpy pandas scikit-learn matplotlib torch streamlit
```

## Usage

### Launch the demo UI

```powershell
streamlit run app.py
```

### Train

```powershell
python main.py train --data datasets/HI-Small_Trans_train_ratio_1_10_augmented.csv --eval-data datasets/HI-Small_Trans_parent_1_150.csv --outputs outputs/run_parent_1_150_tuned --models models/run_parent_1_150_tuned --epochs-ann 6 --epochs-gnn 6 --min-precision 0.08
```

### Predict

```powershell
python main.py predict --data path/to/file.csv --models models/run_parent_1_150_tuned
```

## Metrics

Because this is a rare-event classification problem, accuracy alone is misleading. The project reports precision, recall, F1, ROC-AUC, and PR-AUC, and uses a tuned decision threshold rather than the default `0.5`.

## Status

Demo/research prototype. See section "Known Limitations" in [`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md) for what's not yet production-ready.

## Docs

- [`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md) — full project writeup
- [`docs/METHODOLOGY_AND_FORMULAS.md`](docs/METHODOLOGY_AND_FORMULAS.md) — methodology and formulas
- [`docs/DATASETS_AND_PREPROCESSING.md`](docs/DATASETS_AND_PREPROCESSING.md) — dataset and preprocessing details
