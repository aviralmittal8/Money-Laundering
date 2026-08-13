# Methodology And Formulas

## 1. Project Methodology

This project follows a complete AML (Anti-Money Laundering) detection workflow:

1. Data ingestion and normalization
2. Feature engineering
3. Tabular preprocessing
4. Graph construction
5. Model training using ANN and GNN
6. Ensemble learning with soft voting
7. Threshold tuning for alert generation
8. Evaluation using AML-relevant metrics

The goal is not only to classify transactions, but to convert model outputs into a realistic suspicious-transaction alerting system.

---

## 2. Step-By-Step Methodology

## 2.1 Data Ingestion And Normalization

The raw transaction CSV is loaded and standardized in [src/data_loader.py](c:\Users\pc\VS Code\Project\src\data_loader.py).

Fields are normalized into a common schema:

- `timestamp`
- `from_bank`, `from_account`
- `to_bank`, `to_account`
- `amount_received`
- `amount_paid`
- `receiving_currency`
- `payment_currency`
- `payment_format`
- `is_laundering`

Why this is important:

- real financial data often has inconsistent column names
- normalization allows one training pipeline to work across multiple files

---

## 2.2 Base Feature Engineering

Initial transaction-level features:

- `amount_received`
- `amount_paid`
- `amount_diff`
- `amount_ratio`
- `hour`
- `day_of_week`

Purpose:

- capture transaction size, payment balance, and time-based behavior

---

## 2.3 Behavioral Feature Engineering

Additional account-behavior features were introduced later:

- `tx_count_from_account`
- `tx_count_hour_bucket_from_account`
- `tx_count_day_bucket_from_account`
- `amount_sum_from_account`
- `unique_receivers_from_account`
- `repeated_amount_count_from_account`
- `send_receive_ratio_account`

Purpose:

- represent historical behavior of the sender account
- improve anomaly detection beyond raw transaction values

These were designed as cumulative/account-history features so they could scale to large datasets.

---

## 2.4 Preprocessing

Implemented in [src/preprocess.py](c:\Users\pc\VS Code\Project\src\preprocess.py).

Two types of preprocessing are used:

- numeric features -> standardized using `StandardScaler`
- categorical features -> encoded using `FeatureHasher`

Why this was chosen:

- numeric scaling helps ANN training stability
- feature hashing avoids expensive one-hot encoding on high-cardinality categorical fields

---

## 2.5 Graph Construction

Implemented in [src/data_loader.py](c:\Users\pc\VS Code\Project\src\data_loader.py).

Graph logic:

- each account is treated as a node
- each transaction is treated as an edge
- account identifier is formed as:
  - `bank:account`

Node-level graph features include:

- total received
- total paid
- count of outgoing transactions
- count of incoming transactions
- unique banks sent to
- mean account-history behavior features
- send/receive ratio

Why graph modeling is useful:

- laundering is not only a transaction-level pattern
- it is also a network-flow problem

---

## 2.6 Model Training

### ANN

Implemented in [src/ann.py](c:\Users\pc\VS Code\Project\src\ann.py).

Purpose:

- learn tabular transaction patterns

Input:

- scaled numeric features
- hashed categorical features

Output:

- a laundering probability for each transaction

### GNN

Implemented in [src/gnn.py](c:\Users\pc\VS Code\Project\src\gnn.py).

Purpose:

- learn relational patterns between accounts

Input:

- graph adjacency structure
- node features
- transaction feature vector

Output:

- a laundering probability for each edge/transaction

---

## 2.7 Ensemble Learning

Implemented in [src/ensemble.py](c:\Users\pc\VS Code\Project\src\ensemble.py).

The project uses:

- soft voting ensemble

Soft voting means:

- ANN produces probability \(p_{ann}\)
- GNN produces probability \(p_{gnn}\)
- final ensemble probability is a weighted combination of both

This is better than hard voting for AML because threshold tuning depends on probabilities, not only labels.

---

## 2.8 Threshold Tuning

Implemented in [src/train.py](c:\Users\pc\VS Code\Project\src\train.py).

Instead of using the default threshold `0.5`, the project tunes thresholds on validation data.

Why:

- AML is a rare-event detection problem
- default thresholding often misses almost all positives
- threshold tuning lets us control precision/recall tradeoff

This is one of the most important methodological decisions in the project.

---

## 2.9 Evaluation

Implemented in [src/evaluate.py](c:\Users\pc\VS Code\Project\src\evaluate.py).

Evaluation metrics:

- accuracy
- precision
- recall
- F1
- ROC-AUC
- PR-AUC

Important AML interpretation:

- accuracy is not the main metric
- recall and precision are more important
- PR-AUC is especially important because positives are rare

---

## 3. Mathematical Formulas Used

## 3.1 Amount Difference

\[
\text{amount\_diff} = \text{amount\_received} - \text{amount\_paid}
\]

Purpose:

- identify mismatch between received and paid amounts

---

## 3.2 Amount Ratio

\[
\text{amount\_ratio} = \frac{\text{amount\_paid}}{\text{amount\_received} + 10^{-6}}
\]

Purpose:

- numerical stability
- compare paid vs received amount proportion

---

## 3.3 Send / Receive Ratio

\[
\text{send\_receive\_ratio} = \frac{\text{outgoing transaction count}}{\text{incoming transaction count} + 1}
\]

Purpose:

- capture whether an account mostly sends or mostly receives funds

---

## 3.4 Binary Prediction Rule

If the model predicts probability \(p\), then:

\[
\hat{y} =
\begin{cases}
1 & \text{if } p \ge \text{threshold} \\
0 & \text{otherwise}
\end{cases}
\]

Purpose:

- convert a risk score into an alert/non-alert decision

---

## 3.5 Accuracy

\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]

Meaning:

- overall percentage of correct predictions

Limitation:

- misleading in highly imbalanced datasets

---

## 3.6 Precision

\[
\text{Precision} = \frac{TP}{TP + FP}
\]

Meaning:

- among predicted laundering alerts, how many are truly laundering

AML interpretation:

- high precision means better alert quality

---

## 3.7 Recall

\[
\text{Recall} = \frac{TP}{TP + FN}
\]

Meaning:

- among actual laundering cases, how many were successfully caught

AML interpretation:

- high recall is very important because missing laundering is costly

---

## 3.8 F1 Score

\[
F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\]

Meaning:

- balance between precision and recall

---

## 3.9 Class Weight / Positive Weight

Used during training:

\[
\text{pos\_weight} = \frac{\text{number of negative samples}}{\text{number of positive samples}}
\]

Purpose:

- penalize missing minority class more strongly
- reduce majority-class bias

---

## 3.10 Soft Voting Formula

If ANN weight is \(w_{ann}\) and GNN weight is \(w_{gnn}\):

\[
p_{ensemble} = \frac{w_{ann} \cdot p_{ann} + w_{gnn} \cdot p_{gnn}}{w_{ann} + w_{gnn}}
\]

Purpose:

- combine two model outputs into one ensemble probability

---

## 4. Problems Faced And How They Were Resolved

## 4.1 Extreme Imbalance

Problem:

- original dataset was roughly `1:980`
- model could achieve high accuracy while detecting almost no positives

Resolution:

- generated rebalanced training datasets (`1:10`, `1:20`)
- used class weighting
- tuned thresholds instead of using `0.5`

---

## 4.2 GNN Learning Problem

Problem:

- GNN was not effectively learning end-to-end

Resolution:

- corrected training logic so graph embeddings were properly optimized
- added gradient clipping and optimizer stabilization

---

## 4.3 Confusion Matrix Bug

Problem:

- saved confusion matrix was generated at threshold `0.5`
- reported metrics used tuned threshold

Resolution:

- fixed snapshot confusion matrix generation to use `optimal_threshold`

---

## 4.4 Runtime Explosion From Rolling Features

Problem:

- exact 1h / 24h rolling-window feature engineering on the full dataset was too slow

Resolution:

- replaced exact rolling windows with cumulative/account-history approximations

---

## 4.5 Ensemble Instability

Problem:

- some tuning runs produced unstable or collapsed probability ranges

Resolution:

- refined soft voting / transform logic
- selected a stable presentation-friendly tuned run on `1:150`

---

## 5. Final Presentation Methodology

For the current presentation-facing system, the methodology is:

1. Use rebalanced training data (`1:10_augmented`)
2. Evaluate on rebalanced analysis/evaluation parent (`1:150`)
3. Train ANN and GNN separately
4. Combine them through soft voting
5. Tune threshold for practical precision/recall tradeoff
6. Present both:
   - static tuned run snapshot
   - live threshold-based evaluation on uploaded files

---

## 6. Simple Presentation Explanation

You can explain the project in this order:

1. “We standardized and cleaned the transaction dataset.”
2. “We engineered financial, temporal, behavioral, and graph-based features.”
3. “We trained two models: ANN for tabular behavior and GNN for relational patterns.”
4. “We combined their probabilities using soft voting.”
5. “Because money laundering is rare, we used rebalanced training and threshold tuning.”
6. “We evaluated using precision, recall, F1, ROC-AUC, and PR-AUC.”
7. “We fixed practical issues such as GNN training instability, confusion-matrix threshold mismatch, and expensive rolling-window feature computation.”

---

## 7. Final Note

This project is best described as:

- an AML-oriented machine learning prototype
- using ANN + GNN + soft voting
- with rebalanced training/evaluation strategy
- and threshold-aware deployment logic

It is not just a classifier; it is a thresholded risk-detection system.
