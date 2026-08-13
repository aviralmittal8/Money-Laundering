import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

from src.data_loader import load_dataset, get_feature_columns, build_graph_data
from src.ann import ANNModel
from src.gnn import GNNTransactionClassifier
from src.ensemble import train_meta_model, predict_meta
from src.evaluate import compute_metrics
from src.plots import ensure_dir, save_roc_curve, save_pr_curve, save_confusion_matrix
from src.preprocess import HasherPreprocessor


def _build_preprocessor(hash_dim):
    numeric_features, categorical_features = get_feature_columns()
    return HasherPreprocessor(numeric_features, categorical_features, hash_dim=hash_dim)


def _best_f1_threshold(y_true, y_prob):
    precision_arr, recall_arr, threshold_arr = precision_recall_curve(y_true, y_prob)
    f1_arr = (2 * precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-12)
    best_idx = int(np.nanargmax(f1_arr))
    if best_idx == 0 or len(threshold_arr) == 0:
        thr = 0.5
    else:
        thr = float(threshold_arr[best_idx - 1])
    return float(thr), float(f1_arr[best_idx]), float(precision_arr[best_idx]), float(recall_arr[best_idx])


def _threshold_for_target_recall(y_true, y_prob, target_recall):
    precision_arr, recall_arr, threshold_arr = precision_recall_curve(y_true, y_prob)
    valid_idx = np.where(recall_arr >= target_recall)[0]
    if len(valid_idx) == 0:
        return None
    idx = int(valid_idx[-1])
    if idx == 0 or len(threshold_arr) == 0:
        thr = 0.5
    else:
        thr = float(threshold_arr[idx - 1])
    f1 = (2 * precision_arr[idx] * recall_arr[idx]) / (precision_arr[idx] + recall_arr[idx] + 1e-12)
    return float(thr), float(f1), float(precision_arr[idx]), float(recall_arr[idx])


def _train_ann(model, train_loader, val_tensor, y_val, epochs, pos_weight, device):
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    for epoch_num in range(epochs):
        print(f"\nANN Epoch {epoch_num+1}/{epochs}") # Corrected f-string syntax with double escaped newline
        total_loss = 0.0
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"ANN Average Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        val_logits = model(val_tensor.to(device))
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
    return val_probs


def _train_gnn(
    model,
    adjacency,
    node_features,
    X_all,
    from_idx,
    to_idx,
    y_all,
    y_all_np,
    train_indices,
    epochs,
    pos_weight,
    device,
    edge_sample_size,
    random_state,
):
    rng = np.random.default_rng(random_state)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    model.to(device)
    adjacency = adjacency.to(device)
    node_features = node_features.to(device)

    for epoch_num in range(epochs):
        print(f"\nGNN Epoch {epoch_num+1}/{epochs}") # Corrected f-string syntax with double escaped newline
        model.train()

        train_targets = y_all_np[train_indices]
        pos_idx = train_indices[train_targets == 1]
        neg_idx = train_indices[train_targets == 0]

        if edge_sample_size > 0 and len(train_indices) > edge_sample_size:
            target_size = max(edge_sample_size, len(pos_idx))
            neg_needed = max(target_size - len(pos_idx), 0)
            neg_sample = rng.choice(neg_idx, size=min(neg_needed, len(neg_idx)), replace=False)
            batch_idx = np.concatenate([pos_idx, neg_sample])
            rng.shuffle(batch_idx)
        else:
            batch_idx = train_indices.copy()
            rng.shuffle(batch_idx)

        tx_features = X_all[batch_idx].to(device)
        batch_from = from_idx[batch_idx].to(device)
        batch_to = to_idx[batch_idx].to(device)
        labels = y_all[batch_idx].to(device)

        optimizer.zero_grad()
        node_emb = model.get_node_embeddings(adjacency, node_features)
        logits = model.classify_edges(node_emb, tx_features, batch_from, batch_to)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        print(
            f"GNN Epoch {epoch_num+1} Loss: {loss.item():.4f} "
            f"(sampled_edges={len(batch_idx)}, positives={int((labels == 1).sum().item())})"
        )


def _predict_gnn_probs_batched(model, node_emb, X_all, from_idx, to_idx, indices, batch_size, device):
    model.eval()
    probs = []
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        tx_features = X_all[batch].to(device)
        batch_from = from_idx[batch].to(device)
        batch_to = to_idx[batch].to(device)
        logits = model.classify_edges(node_emb, tx_features, batch_from, batch_to)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def run_training(
    data_path,
    outputs_dir="outputs",
    models_dir="models",
    eval_data_path=None,
    random_state=42,
    epochs_ann=12,
    epochs_gnn=8,
    batch_size=1024,
    hash_dim=128,
    gnn_edge_sample=200000,
    gnn_infer_batch=8192,
    min_precision=0.15,
    target_recall=None,
):
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    ensure_dir(outputs_dir)
    ensure_dir(models_dir)

    train_df = load_dataset(data_path)
    eval_df = train_df if eval_data_path is None else load_dataset(eval_data_path)
    numeric_features, categorical_features = get_feature_columns()
    feature_cols = numeric_features + categorical_features

    X_train_df = train_df[feature_cols]
    y_train_all = train_df["is_laundering"].astype(int).to_numpy()

    X_eval_df = eval_df[feature_cols]
    y_eval_all = eval_df["is_laundering"].astype(int).to_numpy()

    if eval_data_path is None:
        train_idx, temp_idx, y_train, y_temp = train_test_split(
            train_df.index.to_numpy(), y_train_all, test_size=0.3, random_state=random_state, stratify=y_train_all
        )
        val_idx, test_idx, y_val, y_test = train_test_split(
            temp_idx, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
        )
    else:
        train_idx = train_df.index.to_numpy()
        y_train = y_train_all
        val_idx, test_idx, y_val, y_test = train_test_split(
            eval_df.index.to_numpy(), y_eval_all, test_size=0.5, random_state=random_state, stratify=y_eval_all
        )

    print(f"\nTrain target distribution:\n{pd.Series(y_train).value_counts()}")

    print(f"\nValidation target distribution:\n{pd.Series(y_val).value_counts()}")
    print(f"\nTest target distribution:\n{pd.Series(y_test).value_counts()}")

    preprocessor = _build_preprocessor(hash_dim)
    X_train = preprocessor.fit_transform(X_train_df.loc[train_idx])
    X_val = preprocessor.transform(X_eval_df.loc[val_idx])
    X_test = preprocessor.transform(X_eval_df.loc[test_idx])
    X_train_all = preprocessor.transform(X_train_df)
    X_eval_all = preprocessor.transform(X_eval_df)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    X_train_all = torch.tensor(X_train_all, dtype=torch.float32)
    X_eval_all = torch.tensor(X_eval_all, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    y_train_all_tensor = torch.tensor(y_train_all, dtype=torch.float32)
    y_eval_all_tensor = torch.tensor(y_eval_all, dtype=torch.float32)

    pos_weight_value = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"Calculated initial pos_weight_value: {pos_weight_value:.2f}")
    # Keep class weighting meaningful for extreme imbalance while avoiding unstable gradients.
    capped_pos_weight_value = min(pos_weight_value, 200.0)
    gnn_pos_weight_value = min(pos_weight_value, 40.0)
    print(f"Capped pos_weight_value (ANN/ensemble): {capped_pos_weight_value:.2f}")
    print(f"Capped pos_weight_value (GNN): {gnn_pos_weight_value:.2f}")
    pos_weight = torch.tensor(capped_pos_weight_value, dtype=torch.float32)
    gnn_pos_weight = torch.tensor(gnn_pos_weight_value, dtype=torch.float32)

    ann_model = ANNModel(X_train.shape[1])
    ann_loader = DataLoader(
        TensorDataset(X_train, y_train_tensor), batch_size=batch_size, shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_ann_probs = _train_ann(ann_model, ann_loader, X_val, y_val_tensor, epochs_ann, pos_weight, device)

    train_adjacency, train_node_features, train_account_map, train_from_idx, train_to_idx = build_graph_data(train_df)
    train_from_idx = torch.tensor(train_from_idx, dtype=torch.long)
    train_to_idx = torch.tensor(train_to_idx, dtype=torch.long)

    gnn_model = GNNTransactionClassifier(
        node_input_dim=train_node_features.shape[1],
        node_hidden_dim=64,
        node_output_dim=32,
        tx_feature_dim=X_train_all.shape[1],
    )

    _train_gnn(
        gnn_model,
        train_adjacency,
        train_node_features,
        X_train_all,
        train_from_idx,
        train_to_idx,
        y_train_all_tensor,
        y_train_all,
        train_idx,
        epochs_gnn,
        gnn_pos_weight,
        device,
        gnn_edge_sample,
        random_state,
    )

    eval_adjacency, eval_node_features, eval_account_map, eval_from_idx, eval_to_idx = build_graph_data(eval_df)
    eval_from_idx = torch.tensor(eval_from_idx, dtype=torch.long)
    eval_to_idx = torch.tensor(eval_to_idx, dtype=torch.long)

    gnn_model.eval()
    with torch.no_grad():
        eval_node_emb = gnn_model.get_node_embeddings(eval_adjacency.to(device), eval_node_features.to(device)).cpu()
        val_gnn_probs = _predict_gnn_probs_batched(
            gnn_model,
            eval_node_emb,
            X_eval_all,
            eval_from_idx,
            eval_to_idx,
            val_idx,
            gnn_infer_batch,
            device,
        )

    meta_model = train_meta_model(val_ann_probs, val_gnn_probs, y_val, min_precision=min_precision)

    # Calculate meta-model probabilities for the validation set
    val_meta_probs = predict_meta(meta_model, val_ann_probs, val_gnn_probs)

    # Save validation predictions and true labels for threshold optimization
    np.save(os.path.join(outputs_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(outputs_dir, 'val_meta_probs.npy'), val_meta_probs)
    print(f'Saved y_val.npy and val_meta_probs.npy to {outputs_dir}')

    # --- Threshold Optimization on Validation Set ---
    print("\n--- Threshold Optimization on Validation Set ---")
    # Load saved validation predictions and true labels
    loaded_y_val = np.load(os.path.join(outputs_dir, 'y_val.npy'))
    loaded_val_meta_probs = np.load(os.path.join(outputs_dir, 'val_meta_probs.npy'))

    optimal_threshold = float(meta_model.threshold)
    if target_recall is not None:
        recall_target_result = _threshold_for_target_recall(loaded_y_val, loaded_val_meta_probs, target_recall)
        if recall_target_result is not None:
            optimal_threshold, best_f1, best_precision, best_recall = recall_target_result
            print(
                f"\nRecall-target threshold selected: target={target_recall:.3f}, threshold={optimal_threshold:.3f}, "
                f"precision={best_precision:.4f}, recall={best_recall:.4f}, f1={best_f1:.4f}"
            )
        else:
            print(f"\nNo threshold achieved target recall {target_recall:.3f}; using voting ensemble threshold.")
    ann_optimal_threshold, ann_best_f1, ann_best_precision, ann_best_recall = _best_f1_threshold(
        loaded_y_val, val_ann_probs
    )
    gnn_optimal_threshold, gnn_best_f1, gnn_best_precision, gnn_best_recall = _best_f1_threshold(
        loaded_y_val, val_gnn_probs
    )
    val_metrics = compute_metrics(loaded_y_val, loaded_val_meta_probs, threshold=optimal_threshold)
    print(f"\nOptimal Threshold from voting ensemble tuning: {optimal_threshold:.3f}")
    print(f"  Validation F1-score: {val_metrics['f1']:.4f}")
    print(f"  Validation Recall: {val_metrics['recall']:.4f}")
    print(f"  Validation Precision: {val_metrics['precision']:.4f}")
    print(
        f"ANN validation threshold={ann_optimal_threshold:.3f} -> "
        f"f1={ann_best_f1:.4f}, recall={ann_best_recall:.4f}, precision={ann_best_precision:.4f}"
    )
    print(
        f"GNN validation threshold={gnn_optimal_threshold:.3f} -> "
        f"f1={gnn_best_f1:.4f}, recall={gnn_best_recall:.4f}, precision={gnn_best_precision:.4f}"
    )

    # Save the optimal threshold for later use in test set evaluation
    np.save(os.path.join(outputs_dir, 'optimal_threshold.npy'), optimal_threshold)
    print(f"Saved optimal_threshold.npy to {outputs_dir}")

    # Load the optimal threshold
    optimal_threshold = np.load(os.path.join(outputs_dir, 'optimal_threshold.npy'))
    print(f"Using optimal threshold {optimal_threshold:.2f} for test set evaluation.")

    ann_model.eval()
    gnn_model.eval()
    with torch.no_grad():
        test_ann_logits = ann_model(X_test.to(device))
        test_ann_probs = torch.sigmoid(test_ann_logits).cpu().numpy()

        test_gnn_probs = _predict_gnn_probs_batched(
            gnn_model,
            eval_node_emb,
            X_eval_all,
            eval_from_idx,
            eval_to_idx,
            test_idx,
            gnn_infer_batch,
            device,
        )

    test_meta_probs = predict_meta(meta_model, test_ann_probs, test_gnn_probs)

    metrics = compute_metrics(y_test, test_meta_probs, threshold=optimal_threshold)
    ann_metrics = compute_metrics(y_test, test_ann_probs, threshold=ann_optimal_threshold)
    gnn_metrics = compute_metrics(y_test, test_gnn_probs, threshold=gnn_optimal_threshold)
    metrics_path = os.path.join(outputs_dir, "evaluation_summary.csv")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(",".join(metrics.keys()) + "\n" + ",".join([f"%s" % v for v in metrics.values()]) + "\n")

    individual_metrics_path = os.path.join(outputs_dir, "individual_model_metrics.csv")
    pd.DataFrame(
        [
            {"model": "ann", **ann_metrics, "threshold": float(ann_optimal_threshold)},
            {"model": "gnn", **gnn_metrics, "threshold": float(gnn_optimal_threshold)},
            {"model": "ensemble", **metrics, "threshold": float(optimal_threshold)},
        ]
    ).to_csv(individual_metrics_path, index=False)

    save_roc_curve(y_test, test_meta_probs, os.path.join(outputs_dir, "roc_curve.png"))
    save_pr_curve(y_test, test_meta_probs, os.path.join(outputs_dir, "pr_curve.png"))
    save_confusion_matrix(
        y_test,
        test_meta_probs,
        os.path.join(outputs_dir, "confusion_matrix.png"),
        threshold=optimal_threshold,
    )

    torch.save(ann_model.state_dict(), os.path.join(models_dir, "ann_model.pt"))
    torch.save(gnn_model.state_dict(), os.path.join(models_dir, "gnn_model.pt"))

    with open(os.path.join(models_dir, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)

    with open(os.path.join(models_dir, "meta_model.pkl"), "wb") as f:
        pickle.dump(meta_model, f)

    graph_data = {
        "adjacency": train_adjacency,
        "node_features": train_node_features,
        "account_map": train_account_map,
    }

    with torch.no_grad():
        train_node_emb = gnn_model.get_node_embeddings(train_adjacency.to(device), train_node_features.to(device)).cpu()
        graph_data["mean_node_emb"] = train_node_emb.mean(dim=0)

    torch.save(graph_data, os.path.join(models_dir, "graph_data.pt"))

    feature_config = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "ann_input_dim": int(X_train.shape[1]),
        "hash_dim": int(hash_dim),
    }
    with open(os.path.join(models_dir, "feature_config.json"), "w", encoding="utf-8") as f:
        json.dump(feature_config, f, indent=2)

    run_metadata = {
        "train_data_path": data_path,
        "eval_data_path": eval_data_path or data_path,
        "target_recall": target_recall,
        "min_precision": min_precision,
        "train_rows": int(len(train_idx)),
        "eval_rows": int(len(eval_df)),
        "train_positives": int(y_train.sum()),
        "eval_positives": int(y_eval_all.sum()),
    }
    with open(os.path.join(outputs_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    return {
        "metrics": metrics,
        "individual_metrics": {
            "ann": ann_metrics,
            "gnn": gnn_metrics,
            "ensemble": metrics,
        },
        "outputs_dir": outputs_dir,
        "models_dir": models_dir,
        "run_metadata": run_metadata,
    }
