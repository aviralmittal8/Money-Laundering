import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_roc_curve(y_true, y_prob, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_pr_curve(y_true, y_prob, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_confusion_matrix(y_true, y_prob, out_path, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
