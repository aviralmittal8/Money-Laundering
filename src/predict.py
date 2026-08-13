import json
import os
import pickle
import numpy as np
import pandas as pd
import torch

from src.ann import ANNModel
from src.data_loader import build_graph_data, get_feature_columns, load_dataset
from src.ensemble import predict_meta
from src.gnn import GNNTransactionClassifier


def _predict_gnn_probs_batched(model, node_emb, X_all, from_idx, to_idx, indices, batch_size, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            tx_features = X_all[batch].to(device)
            batch_from = from_idx[batch].to(device)
            batch_to = to_idx[batch].to(device)
            logits = model.classify_edges(node_emb, tx_features, batch_from, batch_to)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def predict_from_file(data_path, models_dir="models"):
    df = load_dataset(data_path)
    numeric_features, categorical_features = get_feature_columns()

    with open(os.path.join(models_dir, "feature_config.json"), "r", encoding="utf-8") as f:
        feature_cfg = json.load(f)
    with open(os.path.join(models_dir, "preprocessor.pkl"), "rb") as f:
        preprocessor = pickle.load(f)
    with open(os.path.join(models_dir, "meta_model.pkl"), "rb") as f:
        meta_model = pickle.load(f)

    X = df[numeric_features + categorical_features]
    X_all_np = preprocessor.transform(X)
    X_all = torch.tensor(X_all_np, dtype=torch.float32)

    ann_model = ANNModel(int(feature_cfg["ann_input_dim"]))
    ann_model.load_state_dict(torch.load(os.path.join(models_dir, "ann_model.pt"), map_location="cpu"))
    ann_model.eval()

    with torch.no_grad():
        ann_probs = torch.sigmoid(ann_model(X_all)).numpy()

    adjacency, node_features, _, from_idx_np, to_idx_np = build_graph_data(df)
    from_idx = torch.tensor(from_idx_np, dtype=torch.long)
    to_idx = torch.tensor(to_idx_np, dtype=torch.long)

    gnn_model = GNNTransactionClassifier(
        node_input_dim=node_features.shape[1],
        node_hidden_dim=64,
        node_output_dim=32,
        tx_feature_dim=X_all.shape[1],
    )
    gnn_model.load_state_dict(torch.load(os.path.join(models_dir, "gnn_model.pt"), map_location="cpu"))
    gnn_model.eval()

    with torch.no_grad():
        node_emb = gnn_model.get_node_embeddings(adjacency, node_features)
    all_idx = np.arange(len(df))
    gnn_probs = _predict_gnn_probs_batched(
        gnn_model, node_emb, X_all, from_idx, to_idx, all_idx, batch_size=8192, device=torch.device("cpu")
    )

    probs = predict_meta(meta_model, ann_probs, gnn_probs)
    out_df = pd.read_csv(data_path) if data_path.lower().endswith(".csv") else df.copy()
    return out_df, probs
