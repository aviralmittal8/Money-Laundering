# Datasets And Preprocessing

## 1. Original Dataset

- File: `HI-Small_Trans.csv` (not included in this repo — see [Repository Note](#6-repository-note))
- Rows: `5,078,345`
- Positives (`Is Laundering = 1`): `5,177`
- Negatives: `5,073,168`
- Ratio: about `1:980`

This is the raw, highly imbalanced source dataset. All other datasets in this project are derived from it.

---

## 2. Demo Datasets

Small, checked-in samples used for quick local testing (included in this repo):

| File | Ratio |
|---|---|
| `HI-Small_Trans_1000_demo_balanced.csv` | `1:1` |
| `HI-Small_Trans_1000_demo_curated.csv` | `1:4` |
| `HI-Small_Trans_1000_demo_moderate.csv` | `1:9` |
| `HI-Small_Trans_1000_demo_realistic.csv` | `1:49` |

Note: `HI-Small_Trans_1000_demo.csv` contains `0` positives and is not useful for evaluation — it exists only as an unfiltered raw sample.

---

## 3. Rebalanced Training Datasets

Generated from the original dataset to help the model learn minority-class structure (not included in this repo, see [Repository Note](#6-repository-note)):

- `datasets/HI-Small_Trans_train_ratio_1_10.csv`
- `datasets/HI-Small_Trans_train_ratio_1_10_augmented.csv`
- `datasets/HI-Small_Trans_train_ratio_1_20.csv`
- `datasets/HI-Small_Trans_train_ratio_1_20_augmented.csv`

Purpose:

- expose the model to a higher proportion of positive examples during training
- final evaluation is kept separate from rebalanced training data to avoid inflating results

---

## 4. Rebalanced Evaluation / Analysis Datasets

Generated for presentation and analysis (not included in this repo, see [Repository Note](#6-repository-note)):

- `datasets/HI-Small_Trans_parent_1_70.csv`
- `datasets/HI-Small_Trans_parent_1_150.csv`

Important: these are rebalanced evaluation datasets for analysis/presentation purposes, not the original real-world parent distribution (`~1:980`). They should not be described as production-representative.

---

## 5. Preprocessing Pipeline

Implemented in [`src/data_loader.py`](src/data_loader.py) and [`src/preprocess.py`](src/preprocess.py).

### 5.1 Column Normalization

Raw column names vary across source files, so the loader standardizes them into a common schema:

- `timestamp`
- `from_bank`, `from_account`
- `to_bank`, `to_account`
- `amount_received`
- `amount_paid`
- `receiving_currency`
- `payment_currency`
- `payment_format`
- `is_laundering`

Alternate label column names are also handled, including `Is Laundering`, `is_laundering`, and `label`.

### 5.2 Base Engineered Features

- `amount_received`
- `amount_paid`
- `amount_diff`
- `amount_ratio`
- `hour`
- `day_of_week`

Categorical fields:

- `receiving_currency`
- `payment_currency`
- `payment_format`
- `from_bank`
- `to_bank`

### 5.3 Behavioral Features

Cumulative/account-history features added to capture sender behavior over time, chosen to be cheaper to compute than exact sliding-window features on multi-million-row data:

- `tx_count_from_account`
- `tx_count_hour_bucket_from_account`
- `tx_count_day_bucket_from_account`
- `amount_sum_from_account`
- `unique_receivers_from_account`
- `repeated_amount_count_from_account`
- `send_receive_ratio_account`

### 5.4 Graph Features

Nodes are `bank:account` identifiers, edges are directed transactions. Node features are derived from account activity:

- total received
- total paid
- outgoing transaction count
- incoming transaction count
- unique banks sent to
- mean account-history behavior features
- send/receive ratio

### 5.5 Numeric And Categorical Encoding

Implemented in [`src/preprocess.py`](src/preprocess.py):

- numeric features are scaled using `StandardScaler`
- categorical features are encoded using `FeatureHasher` (avoids expensive one-hot encoding on high-cardinality fields such as bank/account identifiers)

For the underlying formulas (amount_diff, amount_ratio, send_receive_ratio, etc.), see [`METHODOLOGY_AND_FORMULAS.md`](METHODOLOGY_AND_FORMULAS.md).

---

## 6. Repository Note

The original dataset (`HI-Small_Trans.csv`, ~454MB) and the generated `datasets/` files exceed GitHub's per-file size limits and are excluded from this repo via `.gitignore`. Only the small demo CSVs (`HI-Small_Trans_1000_demo*.csv`) are checked in. To reproduce the rebalanced datasets, run the dataset-generation step against your own local copy of the original CSV.
