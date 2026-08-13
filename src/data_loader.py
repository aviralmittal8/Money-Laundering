import os
import re
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = {
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
}

COLUMN_ALIASES = {
    "timestamp": "timestamp",
    "time": "timestamp",
    "from bank": "from_bank",
    "from_bank": "from_bank",
    "to bank": "to_bank",
    "to_bank": "to_bank",
    "amount received": "amount_received",
    "amount_received": "amount_received",
    "receiving currency": "receiving_currency",
    "receiving_currency": "receiving_currency",
    "amount paid": "amount_paid",
    "amount_paid": "amount_paid",
    "payment currency": "payment_currency",
    "payment_currency": "payment_currency",
    "payment format": "payment_format",
    "payment_format": "payment_format",
    "is laundering": "is_laundering",
    "is_laundering": "is_laundering",
    "laundering": "is_laundering",
    "label": "is_laundering",
}

NUMERIC_FEATURES = [
    "amount_received",
    "amount_paid",
    "amount_diff",
    "amount_ratio",
    "hour",
    "day_of_week",
    "tx_count_from_account",
    "tx_count_hour_bucket_from_account",
    "tx_count_day_bucket_from_account",
    "amount_sum_from_account",
    "unique_receivers_from_account",
    "repeated_amount_count_from_account",
    "send_receive_ratio_account",
]

CATEGORICAL_FEATURES = [
    "receiving_currency",
    "payment_currency",
    "payment_format",
    "from_bank",
    "to_bank",
]


def _normalize_column_name(name):
    cleaned = re.sub(r"[_\s]+", " ", str(name).strip().lower())
    return cleaned


def _standardize_columns(df):
    if 'Account' in df.columns and 'Account.1' in df.columns:
        df = df.rename(columns={'Account': 'From Account', 'Account.1': 'To Account'})
    normalized = [_normalize_column_name(col) for col in df.columns]
    rename_map = {}
    account_positions = []

    for idx, col in enumerate(normalized):
        if col in ("from account", "from_account", "fromaccount"):
            rename_map[df.columns[idx]] = "from_account"
        elif col in ("to account", "to_account", "toaccount"):
            rename_map[df.columns[idx]] = "to_account"
        elif col == "account":
            account_positions.append(idx)
        elif col in COLUMN_ALIASES:
            rename_map[df.columns[idx]] = COLUMN_ALIASES[col]

    if account_positions:
        if len(account_positions) != 2:
            raise ValueError("Expected two 'Account' columns for from/to accounts.")
        rename_map[df.columns[account_positions[0]]] = "from_account"
        rename_map[df.columns[account_positions[1]]] = "to_account"

    df = df.rename(columns=rename_map)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df


def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df = _standardize_columns(df)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().all():
        df["timestamp"] = pd.Timestamp("1970-01-01")
    else:
        median_ts = df["timestamp"].dropna().median()
        df["timestamp"] = df["timestamp"].fillna(median_ts)

    for col in ("amount_received", "amount_paid"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["is_laundering"] = pd.to_numeric(df["is_laundering"], errors="coerce").fillna(0).astype(int)

    df["hour"] = df["timestamp"].dt.hour.astype(int)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(int)
    df["amount_diff"] = df["amount_received"] - df["amount_paid"]
    df["amount_ratio"] = df["amount_paid"] / (df["amount_received"] + 1e-6)

    for col in CATEGORICAL_FEATURES + ["from_account", "to_account"]:
        df[col] = df[col].astype(str).fillna("unknown")

    df = _add_behavior_features(df)

    return df.reset_index(drop=True)


def get_feature_columns():
    return NUMERIC_FEATURES, CATEGORICAL_FEATURES


def build_graph_data(df):
    account_keys = pd.concat([df["from_bank"] + ":" + df["from_account"], df["to_bank"] + ":" + df["to_account"]])
    account_keys = account_keys.unique().tolist()
    account_map = {key: idx for idx, key in enumerate(account_keys)}

    from_keys = df["from_bank"] + ":" + df["from_account"]
    to_keys = df["to_bank"] + ":" + df["to_account"]

    from_idx = from_keys.map(account_map).astype(int).to_numpy()
    to_idx = to_keys.map(account_map).astype(int).to_numpy()

    num_nodes = len(account_map)
    edge_src = np.concatenate([from_idx, to_idx])
    edge_dst = np.concatenate([to_idx, from_idx])
    self_loops = np.arange(num_nodes)
    edge_src = np.concatenate([edge_src, self_loops])
    edge_dst = np.concatenate([edge_dst, self_loops])

    degrees = np.bincount(edge_src, minlength=num_nodes).astype(np.float32)
    degrees[degrees == 0] = 1.0
    deg_inv_sqrt = np.power(degrees, -0.5)
    values = deg_inv_sqrt[edge_src] * deg_inv_sqrt[edge_dst]
    norm_adj = torch_sparse_coo(edge_src, edge_dst, values, num_nodes)

    node_features = pd.DataFrame({
        "total_received": df.groupby(to_keys)["amount_received"].sum(),
        "total_paid": df.groupby(from_keys)["amount_paid"].sum(),
        "count_out": df.groupby(from_keys)["amount_paid"].count(),
        "count_in": df.groupby(to_keys)["amount_received"].count(),
        "unique_banks_sent_to": df.groupby(from_keys)["to_bank"].nunique(),
        "mean_tx_count_day_bucket": df.groupby(from_keys)["tx_count_day_bucket_from_account"].mean(),
        "mean_amount_sum_from_account": df.groupby(from_keys)["amount_sum_from_account"].mean(),
        "mean_unique_receivers": df.groupby(from_keys)["unique_receivers_from_account"].mean(),
        "send_receive_ratio": df.groupby(from_keys)["send_receive_ratio_account"].mean(),
    }).fillna(0.0)

    node_features = node_features.reindex(account_keys).fillna(0.0)
    # Compress heavy-tailed transaction values before GNN training.
    node_features[["total_received", "total_paid", "mean_amount_sum_from_account"]] = np.log1p(
        node_features[["total_received", "total_paid", "mean_amount_sum_from_account"]]
    )
    # Standardize node features to reduce unstable GNN losses.
    mu = node_features.mean(axis=0)
    sigma = node_features.std(axis=0).replace(0.0, 1.0)
    node_features = ((node_features - mu) / sigma).astype(np.float32).to_numpy()

    return (
        norm_adj,
        torch_from_numpy(node_features),
        account_map,
        from_idx,
        to_idx,
    )


def torch_from_numpy(array):
    import torch

    return torch.from_numpy(array)


def _add_behavior_features(df):
    df = df.copy()
    df["from_key"] = df["from_bank"] + ":" + df["from_account"]
    df["to_key"] = df["to_bank"] + ":" + df["to_account"]

    out_count = df.groupby("from_key").size()
    in_count = df.groupby("to_key").size()
    df["send_receive_ratio_account"] = (
        df["from_key"].map(out_count) / (df["from_key"].map(in_count).fillna(0.0) + 1.0)
    ).astype(np.float32)

    ordered = df[["timestamp", "from_key", "to_key", "amount_paid"]].reset_index().sort_values("timestamp")
    ordered["hour_bucket"] = ordered["timestamp"].dt.floor("h")
    ordered["day_bucket"] = ordered["timestamp"].dt.floor("d")
    ordered["rounded_amount"] = np.round(ordered["amount_paid"], 2)

    ordered["tx_count_from_account"] = ordered.groupby("from_key").cumcount() + 1
    ordered["tx_count_hour_bucket_from_account"] = ordered.groupby(["from_key", "hour_bucket"]).cumcount() + 1
    ordered["tx_count_day_bucket_from_account"] = ordered.groupby(["from_key", "day_bucket"]).cumcount() + 1
    ordered["amount_sum_from_account"] = ordered.groupby("from_key")["amount_paid"].cumsum()

    ordered["is_new_receiver"] = ~ordered.duplicated(subset=["from_key", "to_key"])
    ordered["unique_receivers_from_account"] = (
        ordered.groupby("from_key")["is_new_receiver"].cumsum().astype(np.float32)
    )
    ordered["repeated_amount_count_from_account"] = (
        ordered.groupby(["from_key", "rounded_amount"]).cumcount() + 1
    )

    ordered = ordered.sort_values("index")
    feature_cols = [
        "tx_count_from_account",
        "tx_count_hour_bucket_from_account",
        "tx_count_day_bucket_from_account",
        "amount_sum_from_account",
        "unique_receivers_from_account",
        "repeated_amount_count_from_account",
    ]
    for col in feature_cols:
        df[col] = ordered[col].to_numpy(dtype=np.float32)

    df.drop(columns=["from_key", "to_key"], inplace=True)
    return df


def torch_sparse_coo(src, dst, values, size):
    import torch

    indices = np.vstack([src, dst])
    indices = torch.from_numpy(indices).long()
    values = torch.from_numpy(values.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, (size, size)).coalesce()
