import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.evaluate import compute_metrics
from src.predict import predict_from_file


st.set_page_config(page_title="AML Detection Demo", layout="wide")
st.title("Money Laundering Detection Demo")
st.caption("ANN + GNN + Soft Voting Ensemble")

MODELS_DIR = os.path.join("models", "run_parent_1_150_tuned")
OUTPUTS_DIR = os.path.join("outputs", "run_parent_1_150_tuned")


def _normalize_name(name):
    return str(name).strip().lower().replace("_", " ")


def _find_label_column(columns):
    candidates = {"is laundering", "is_laundering", "laundering", "label"}
    for col in columns:
        if _normalize_name(col) in candidates:
            return col
    return None


def _load_threshold():
    threshold_path = os.path.join(OUTPUTS_DIR, "optimal_threshold.npy")
    if os.path.exists(threshold_path):
        try:
            return float(np.load(threshold_path))
        except Exception:
            return 0.5
    return 0.5


def _show_metrics():
    st.subheader("Tuned 1:150 Evaluation")
    st.caption("Default view: tuned ensemble evaluation on the `1:150` rebalanced parent dataset.")
    metrics_path = os.path.join(OUTPUTS_DIR, "individual_model_metrics.csv")
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.warning("`outputs/individual_model_metrics.csv` not found. Run training first.")

    cols = st.columns(3)
    plot_files = ["roc_curve.png", "pr_curve.png", "confusion_matrix.png"]
    for col, name in zip(cols, plot_files):
        path = os.path.join(OUTPUTS_DIR, name)
        with col:
            if os.path.exists(path):
                st.image(path, caption=name.replace("_", " ").replace(".png", "").title())
            else:
                st.info(f"Missing {name}")


def _run_prediction(upload, threshold_override=None):
    if upload is None:
        return

    suffix = os.path.splitext(upload.name)[1].lower() or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.getbuffer())
        tmp_path = tmp.name

    try:
        df, probs = predict_from_file(tmp_path, models_dir=MODELS_DIR)
        threshold = _load_threshold() if threshold_override is None else float(threshold_override)
        pred = (probs >= threshold).astype(int)

        out = df.copy()
        out["predicted_prob"] = probs
        out["predicted_label"] = pred

        st.subheader("Prediction Output")
        st.write(f"Threshold used: `{threshold:.3f}`")

        pos_count = int((out["predicted_label"] == 1).sum())
        total = len(out)
        st.write(f"Flagged positives: `{pos_count}` / `{total}` ({(100.0 * pos_count / max(total, 1)):.2f}%)")
        st.dataframe(out.head(200), use_container_width=True)

        label_col = _find_label_column(out.columns)
        if label_col is not None:
            y_true = pd.to_numeric(out[label_col], errors="coerce").fillna(0).astype(int).to_numpy()
            y_prob = out["predicted_prob"].to_numpy()
            y_pred = out["predicted_label"].to_numpy()
            has_both_classes = len(np.unique(y_true)) == 2
            if has_both_classes:
                summary = compute_metrics(y_true, y_prob, threshold=threshold)
            else:
                summary = {
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                    "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                    "auc": None,
                    "pr_auc": None,
                }

            st.subheader("Live Evaluation On Uploaded Data")
            metric_cols = st.columns(6)
            metric_cols[0].metric("Accuracy", f"{summary['accuracy']:.4f}")
            metric_cols[1].metric("Precision", f"{summary['precision']:.4f}")
            metric_cols[2].metric("Recall", f"{summary['recall']:.4f}")
            metric_cols[3].metric("F1", f"{summary['f1']:.4f}")
            metric_cols[4].metric("AUC", "N/A" if summary["auc"] is None else f"{summary['auc']:.4f}")
            metric_cols[5].metric("PR-AUC", "N/A" if summary["pr_auc"] is None else f"{summary['pr_auc']:.4f}")

            if has_both_classes:
                st.caption("Threshold sweep on the current uploaded file")
                sweep_rows = []
                for thr in [0.01, 0.02, 0.03, 0.05]:
                    sweep_metrics = compute_metrics(y_true, y_prob, threshold=thr)
                    sweep_rows.append(
                        {
                            "threshold": thr,
                            "precision": round(sweep_metrics["precision"], 4),
                            "recall": round(sweep_metrics["recall"], 4),
                            "f1": round(sweep_metrics["f1"], 4),
                            "flagged_rate_pct": round(float((y_prob >= thr).mean() * 100.0), 2),
                        }
                    )
                st.dataframe(pd.DataFrame(sweep_rows), use_container_width=True, hide_index=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                fig_cm, ax_cm = plt.subplots(figsize=(4.5, 4))
                im = ax_cm.imshow(cm, cmap="viridis")
                ax_cm.set_title("Confusion Matrix")
                ax_cm.set_xlabel("Predicted label")
                ax_cm.set_ylabel("True label")
                ax_cm.set_xticks([0, 1])
                ax_cm.set_yticks([0, 1])
                for i in range(2):
                    for j in range(2):
                        ax_cm.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="w")
                fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
                st.pyplot(fig_cm)
                plt.close(fig_cm)

            with col2:
                fig_roc, ax_roc = plt.subplots(figsize=(4.5, 4))
                if has_both_classes:
                    fpr, tpr, _ = roc_curve(y_true, y_prob)
                    auc_val = roc_auc_score(y_true, y_prob)
                    ax_roc.plot(fpr, tpr, label=f"AUC={auc_val:.3f}")
                    ax_roc.plot([0, 1], [0, 1], "--", color="gray")
                    ax_roc.legend(loc="lower right")
                else:
                    ax_roc.text(0.5, 0.5, "ROC unavailable\n(single class)", ha="center", va="center")
                ax_roc.set_title("ROC Curve")
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                st.pyplot(fig_roc)
                plt.close(fig_roc)

            with col3:
                fig_pr, ax_pr = plt.subplots(figsize=(4.5, 4))
                if has_both_classes:
                    prec, rec, _ = precision_recall_curve(y_true, y_prob)
                    ap = average_precision_score(y_true, y_prob)
                    ax_pr.plot(rec, prec, label=f"AP={ap:.3f}")
                    ax_pr.legend(loc="lower left")
                else:
                    ax_pr.text(0.5, 0.5, "PR unavailable\n(single class)", ha="center", va="center")
                ax_pr.set_title("PR Curve")
                ax_pr.set_xlabel("Recall")
                ax_pr.set_ylabel("Precision")
                st.pyplot(fig_pr)
                plt.close(fig_pr)
        else:
            st.info("No label column found in uploaded file. Live confusion matrix/ROC/PR need `Is Laundering`.")

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Predictions CSV",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


_show_metrics()
st.subheader("Run New Prediction")
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
default_thr = _load_threshold()
threshold_ui = st.slider(
    "Decision Threshold",
    min_value=0.01,
    max_value=0.99,
    value=float(round(default_thr, 3)),
    step=0.01,
    help="Lower threshold increases alerts/recall; higher threshold increases precision.",
)
if st.button("Predict", type="primary"):
    _run_prediction(uploaded_file, threshold_override=threshold_ui)
