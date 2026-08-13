# Money Laundering Detection Project Documentation

## 1. Executive Summary

This project builds an AML (Anti-Money Laundering) transaction detection system using:

- ANN (Artificial Neural Network) for tabular transaction behavior
- GNN (Graph Neural Network) for account-to-account network structure
- Ensemble learning via soft voting over ANN and GNN probabilities
- Threshold tuning to convert probabilities into alerts

The project started on the original highly imbalanced dataset and then expanded into:

- rebalanced training datasets (`1:10`, `1:20`)
- rebalanced evaluation/demo datasets (`1:70`, `1:150`)
- a Streamlit frontend for live predictions and live evaluation

Current presentation-ready setup:

- default frontend model/output run: tuned `1:150` evaluation run
- default frontend metrics source:
  - `models/run_parent_1_150_tuned`
  - `outputs/run_parent_1_150_tuned`

---

## 2. Core Objective

Main objective:

- detect `Is Laundering = 1` transactions from highly imbalanced financial transaction data

Business reality:

- real AML datasets are rare-event problems
- overall accuracy is often misleading
- recall, precision, F1, PR-AUC, and threshold policy matter more than raw accuracy

---

## 3. Repository Structure

- `main.py`: CLI entrypoint for training and prediction
- `app.py`: Streamlit frontend
- `src/data_loader.py`: dataset loading, standardization, engineered features, graph construction
- `src/preprocess.py`: numeric scaling + categorical hashing
- `src/ann.py`: ANN architecture
- `src/gnn.py`: GNN architecture
- `src/ensemble.py`: soft voting ensemble and probability transformation logic
- `src/train.py`: end-to-end training/evaluation pipeline
- `src/predict.py`: batch prediction pipeline
- `src/evaluate.py`: metric calculation
- `src/plots.py`: ROC/PR/confusion matrix plot generation
- `models/`: trained model artifacts
- `outputs/`: metrics, plots, threshold files, metadata
- `datasets/`: rebalanced training/evaluation datasets

---

## 4. Datasets Used

## 4.1 Original Dataset

- `HI-Small_Trans.csv`
- rows: `5,078,345`
- positives: `5,177`
- negatives: `5,073,168`
- ratio: about `1:980`

## 4.2 Demo Datasets

- `HI-Small_Trans_1000_demo_balanced.csv`: `1:1`
- `HI-Small_Trans_1000_demo_curated.csv`: `1:4`
- `HI-Small_Trans_1000_demo_moderate.csv`: `1:9`
- `HI-Small_Trans_1000_demo_realistic.csv`: `1:49`

Note:

- `HI-Small_Trans_1000_demo.csv` is not useful for evaluation because it contains `0` positives.

## 4.3 Rebalanced Training Datasets

Created from the original dataset:

- `datasets/HI-Small_Trans_train_ratio_1_10.csv`
- `datasets/HI-Small_Trans_train_ratio_1_10_augmented.csv`
- `datasets/HI-Small_Trans_train_ratio_1_20.csv`
- `datasets/HI-Small_Trans_train_ratio_1_20_augmented.csv`

Purpose:

- help the model learn minority-class structure
- keep final evaluation separate from rebalanced training

## 4.4 Rebalanced Evaluation / Analysis Datasets

Created for presentation and analysis:

- `datasets/HI-Small_Trans_parent_1_70.csv`
- `datasets/HI-Small_Trans_parent_1_150.csv`

Important:

- these are not the real-world parent distribution
- they are rebalanced evaluation datasets for analysis/presentation

---

## 5. Data Processing Logic

## 5.1 Column Normalization

The loader standardizes:

- timestamp
- bank/account identifiers
- amount fields
- laundering label

It also handles alternate column names such as:

- `Is Laundering`
- `is_laundering`
- `label`

## 5.2 Base Engineered Features

Original tabular numeric features:

- `amount_received`
- `amount_paid`
- `amount_diff`
- `amount_ratio`
- `hour`
- `day_of_week`

Categorical features:

- `receiving_currency`
- `payment_currency`
- `payment_format`
- `from_bank`
- `to_bank`

## 5.3 Later Behavioral Features Added

To improve the model, additional behavior features were added:

- `tx_count_from_account`
- `tx_count_hour_bucket_from_account`
- `tx_count_day_bucket_from_account`
- `amount_sum_from_account`
- `unique_receivers_from_account`
- `repeated_amount_count_from_account`
- `send_receive_ratio_account`

These are cumulative/account-history style features, designed to be cheaper than exact sliding windows on multi-million-row data.

## 5.4 Graph Features

Node features are derived from account activity:

- total received
- total paid
- outgoing count
- incoming count
- unique banks sent to
- mean account behavior features
- send/receive ratio

Graph is built on `bank:account` nodes with directed transaction edges.

---

## 6. Models

## 6.1 ANN

Purpose:

- learn tabular transaction patterns

Structure:

- fully connected feed-forward network
- ReLU + dropout
- outputs a single probability score per transaction

## 6.2 GNN

Purpose:

- learn network patterns between accounts

Structure:

- 2-layer graph convolution network
- edge classifier uses:
  - source node embedding
  - destination node embedding
  - transaction features

## 6.3 Ensemble

Purpose:

- combine ANN and GNN outputs into a final probability

Current design:

- soft voting
- learned weights over ANN/GNN probabilities
- optional probability transformation
- tuned threshold for final classification

---

## 7. Metrics Used

- Accuracy: overall correctness
- Precision: flagged positives that are truly positive
- Recall: actual positives that were caught
- F1: balance of precision and recall
- AUC (ROC-AUC): ranking quality across thresholds
- PR-AUC / AP: precision-recall quality, especially important for rare-event AML

AML interpretation:

- accuracy is secondary
- recall and precision tradeoff is central
- PR-AUC is more meaningful than accuracy for deployment discussions

---

## 8. Major Problems Faced

## 8.1 Extreme Class Imbalance

Problem:

- original parent data is about `1:980`
- model can look accurate while missing most laundering cases

Impact:

- high accuracy but poor recall
- threshold selection became critical

Resolution:

- created rebalanced training files
- used `pos_weight`
- tuned thresholds instead of relying on `0.5`

## 8.2 GNN Training Bug

Problem:

- node embeddings were previously detached during training
- graph layers were not truly learning end-to-end

Impact:

- weak GNN contribution

Resolution:

- removed the harmful detach behavior
- added optimizer stabilization and gradient clipping

## 8.3 Confusion Matrix Bug

Problem:

- saved confusion matrix snapshot was generated with default threshold `0.5`
- but the run metrics used a tuned threshold

Impact:

- matrix contradicted precision/recall numbers

Resolution:

- fixed `src/train.py` so `save_confusion_matrix()` receives `threshold=optimal_threshold`
- regenerated the confusion matrix for the tuned `1:150` run

## 8.4 Ensemble Probability Collapse

Problem:

- some tuning runs produced near-constant probabilities
- validation threshold search had poor operating points

Impact:

- bad tuned outputs
- cases where all predictions became class `0`
- or cases with very low threshold causing almost all predictions to become positive

Resolution attempts:

- improved ensemble probability transform logic
- changed threshold search behavior
- used lower precision-floor settings on rebalanced runs

Result:

- tuned `1:150` run became presentation-usable
- some later feature-engineering runs still degraded sharply

## 8.5 Feature Engineering Runtime Explosion

Problem:

- exact 1h/24h rolling behavior features on the full multi-million-row parent file were too slow

Impact:

- full-parent feature-engineered training timed out before first output

Resolution:

- replaced expensive exact rolling windows with cheaper cumulative/account-history features

---

## 9. What Was Implemented

## 9.1 Training Pipeline Improvements

- separate train and evaluation datasets supported
- `--eval-data` added
- `--target-recall` added
- `--min-precision` added
- PR-AUC added to evaluation output

## 9.2 Individual Model Metrics

Saved per run:

- ANN metrics
- GNN metrics
- Ensemble metrics

Output file:

- `individual_model_metrics.csv`

## 9.3 Rebalanced Dataset Generation

Generated:

- `1:10`
- `1:10 augmented`
- `1:20`
- `1:20 augmented`
- `1:70`
- `1:150`

## 9.4 UI Improvements

`app.py` now supports:

- default view tied to tuned `1:150` run
- metrics table
- snapshot ROC / PR / confusion matrix
- threshold slider
- upload new CSV/XLSX for prediction
- live evaluation on uploaded labeled data
- live confusion matrix
- live ROC curve
- live PR curve
- PR-AUC display
- threshold sweep table for:
  - `0.01`
  - `0.02`
  - `0.03`
  - `0.05`
- safe handling of single-class uploads

---

## 10. Key Experimental Runs

## 10.1 Earlier Baseline on Original Parent Distribution

Representative baseline ensemble metrics:

- precision: `0.1160`
- recall: `0.0477`
- f1: `0.0676`
- auc: `0.9432`

Interpretation:

- conservative
- better precision
- weak recall

## 10.2 `1:10` Rebalanced Training, Real Parent Evaluation

Representative ensemble metrics:

- precision: `0.0263`
- recall: `0.2457`
- f1: `0.0476`
- auc: `0.9492`
- pr_auc: `0.0207`

Interpretation:

- much higher recall
- precision dropped sharply

## 10.3 `1:10_augmented` Rebalanced Training, Real Parent Evaluation

Representative ensemble metrics:

- precision: `0.0346`
- recall: `0.2024`
- f1: `0.0591`
- auc: `0.9564`
- pr_auc: `0.0263`

Interpretation:

- best balanced result among the real-parent rebalanced runs

## 10.4 Feature-Engineered Runs on Real Parent

Observed issue:

- some feature-engineered runs became too recall-heavy
- one tuned run produced:
  - precision near `0.001`
  - recall near `0.956`
  - unusable F1 and precision

Interpretation:

- feature additions were not production-ready yet
- not suitable for presentation as final result

## 10.5 Tuned `1:150` Parent Evaluation Run

This is the current presentation-facing run.

Ensemble:

- accuracy: `0.9275`
- precision: `0.0558`
- recall: `0.6246`
- f1: `0.1025`
- auc: `0.9277`
- pr_auc: `0.0512`
- threshold: about `0.0322`

Individual models:

- ANN:
  - accuracy: `0.9469`
  - precision: `0.0576`
  - recall: `0.4569`
  - f1: `0.1023`
  - auc: `0.9286`
- GNN:
  - accuracy: `0.5344`
  - precision: `0.0110`
  - recall: `0.7806`
  - f1: `0.0217`
  - auc: `0.6470`
- Ensemble:
  - accuracy: `0.9275`
  - precision: `0.0558`
  - recall: `0.6246`
  - f1: `0.1025`
  - auc: `0.9277`
  - pr_auc: `0.0512`

Interpretation:

- good for presentation as a tuned rebalanced-evaluation result
- strong recall
- low but plausible AML precision
- clear and explainable thresholded behavior

---

## 11. Presentation Strategy

Recommended narrative:

1. Explain the original imbalance problem (`1:980`)
2. Explain why accuracy alone is misleading
3. Show ANN + GNN + ensemble architecture
4. Explain rebalanced training and threshold tuning
5. Present the tuned `1:150` run as the visible demo/evaluation setup
6. Clearly state that `1:150` is a rebalanced evaluation dataset, not the original real-world parent ratio
7. Use the live threshold sweep in the UI to show operating-point tradeoffs

What not to claim:

- do not call `1:150` the original parent distribution
- do not claim production readiness
- do not present the worst feature-engineered run as final

---

## 12. UI Behavior and Demo Logic

Top section:

- fixed snapshot from `outputs/run_parent_1_150_tuned`

Live section:

- uses uploaded file
- applies the currently selected threshold
- computes:
  - alert count
  - accuracy
  - precision
  - recall
  - F1
  - AUC
  - PR-AUC
  - confusion matrix
  - ROC curve
  - PR curve
  - threshold sweep table

Important note:

- AUC / PR-AUC can be high even when thresholded recall is zero
- this is not a bug; it means ranking is good but the chosen threshold is too strict

---

## 13. Commands

## 13.1 Install dependencies

```powershell
& "C:\Program Files\PostgreSQL\18\pgAdmin 4\python\python.exe" -m pip install numpy pandas scikit-learn matplotlib torch streamlit
```

## 13.2 Launch frontend

```powershell
cd "c:\Users\pc\VS Code\Project"
& "C:\Program Files\PostgreSQL\18\pgAdmin 4\python\python.exe" -m streamlit run app.py
```

## 13.3 Train on custom data

Example:

```powershell
& "C:\Program Files\PostgreSQL\18\pgAdmin 4\python\python.exe" -c "import sys; sys.path.insert(0, r'c:\Users\pc\VS Code\Project'); import main; sys.argv=['main.py','train','--data','datasets/HI-Small_Trans_train_ratio_1_10_augmented.csv','--eval-data','datasets/HI-Small_Trans_parent_1_150.csv','--outputs','outputs/run_parent_1_150_tuned','--models','models/run_parent_1_150_tuned','--epochs-ann','6','--epochs-gnn','6','--min-precision','0.08']; main.main()"
```

---

## 14. Terminology

- AML: Anti-Money Laundering
- ANN: Artificial Neural Network
- GNN: Graph Neural Network
- Ensemble: combined model output
- Soft voting: weighted probability averaging across models
- Threshold: cutoff to convert probability into alert/non-alert
- Calibration: reshaping probabilities to better reflect true likelihood
- Precision: `TP / (TP + FP)`
- Recall: `TP / (TP + FN)`
- F1: harmonic mean of precision and recall
- ROC-AUC: overall ranking quality
- PR-AUC / AP: precision-recall quality for rare-event classification
- TN / FP / FN / TP: confusion-matrix components

---

## 15. Known Limitations

- Results depend strongly on evaluation distribution
- GNN remains weaker than ANN in several runs
- Ensemble tuning is still unstable in some feature-engineered settings
- Current feature engineering is still a simplified approximation, not full behavior-window modeling
- Real deployment should use time-based validation rather than only random split logic

---

## 16. Recommended Next Steps After Presentation

- stabilize ensemble tuning further
- revisit probability calibration logic
- add time-based train/validation/test split
- improve GNN node/edge features
- add top-K alert evaluation
- revisit exact rolling-window behavior features with optimized implementation

---

## 17. Final Conclusion

This project successfully evolved from a basic rare-event classifier into a demo-ready AML prototype with:

- ANN + GNN + ensemble architecture
- rebalanced training/evaluation datasets
- threshold-aware evaluation
- live Streamlit demonstration
- presentation-ready tuned `1:150` run

The system is best described as a strong research/demo pipeline with clear next steps toward a more stable and realistic AML detection workflow.
