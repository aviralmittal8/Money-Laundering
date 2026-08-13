import numpy as np
from dataclasses import dataclass
from sklearn.metrics import average_precision_score, f1_score, precision_score
from sklearn.isotonic import IsotonicRegression


@dataclass
class ProbabilityTransform:
    method: str
    calibrator: object = None


@dataclass
class VotingEnsembleModel:
    ann_weight: float
    lgbm_weight: float
    gnn_weight: float
    threshold: float
    ann_transform: ProbabilityTransform
    lgbm_transform: ProbabilityTransform
    gnn_transform: ProbabilityTransform


def _to_1d(arr):
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def _voting_probs(ann_probs, lgbm_probs, gnn_probs, ann_weight, lgbm_weight, gnn_weight):
    total = ann_weight + lgbm_weight + gnn_weight
    if total <= 0:
        ann_weight, lgbm_weight, gnn_weight, total = 1.0, 1.0, 1.0, 3.0
    return (ann_weight * ann_probs + lgbm_weight * lgbm_probs + gnn_weight * gnn_probs) / total


def _fit_calibrator(probs, y):
    probs = np.clip(_to_1d(probs), 1e-6, 1 - 1e-6)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(probs, _to_1d(y).astype(int))
    return calibrator


def _fit_probability_transform(probs, y):
    probs = np.clip(_to_1d(probs), 1e-6, 1 - 1e-6)
    raw_ap = average_precision_score(y, probs)
    calibrator = _fit_calibrator(probs, y)
    calibrated = _to_1d(calibrator.transform(probs))
    calibrated_ap = average_precision_score(y, calibrated)

    # If isotonic collapses the score distribution or significantly hurts ranking,
    # keep the raw probabilities.
    if np.std(calibrated) < 1e-4 or calibrated_ap + 1e-6 < raw_ap:
        return ProbabilityTransform(method="identity")
    return ProbabilityTransform(method="isotonic", calibrator=calibrator)


def _apply_transform(transform, probs):
    probs = np.clip(_to_1d(probs), 1e-6, 1 - 1e-6)
    if transform.method == "identity" or transform.calibrator is None:
        return probs
    return _to_1d(transform.calibrator.transform(probs))


def _simplex_weight_grid(step=0.1):
    # All (a, b, c) with a, b, c on a `step` grid and a + b + c == 1.
    steps = np.round(np.arange(0.0, 1.0 + 1e-9, step), 2)
    grid = []
    for a in steps:
        for b in steps:
            c = round(1.0 - a - b, 2)
            if c < -1e-9 or c > 1.0 + 1e-9:
                continue
            grid.append((float(a), float(b), float(max(c, 0.0))))
    return grid


def train_meta_model(ann_probs, lgbm_probs, gnn_probs, y, min_precision=0.15):
    ann_probs = _to_1d(ann_probs)
    lgbm_probs = _to_1d(lgbm_probs)
    gnn_probs = _to_1d(gnn_probs)
    y = _to_1d(y).astype(int)

    ann_transform = _fit_probability_transform(ann_probs, y)
    lgbm_transform = _fit_probability_transform(lgbm_probs, y)
    gnn_transform = _fit_probability_transform(gnn_probs, y)
    ann_probs_cal = _apply_transform(ann_transform, ann_probs)
    lgbm_probs_cal = _apply_transform(lgbm_transform, lgbm_probs)
    gnn_probs_cal = _apply_transform(gnn_transform, gnn_probs)

    weight_grid = _simplex_weight_grid(step=0.1)
    precision_floor = float(min_precision)

    best_score = -1e12
    best_f1 = -1.0
    best_precision = -1.0
    best_recall = -1.0
    best_model = VotingEnsembleModel(
        ann_weight=1 / 3,
        lgbm_weight=1 / 3,
        gnn_weight=1 / 3,
        threshold=0.5,
        ann_transform=ann_transform,
        lgbm_transform=lgbm_transform,
        gnn_transform=gnn_transform,
    )

    for ann_w, lgbm_w, gnn_w in weight_grid:
        voted = _voting_probs(ann_probs_cal, lgbm_probs_cal, gnn_probs_cal, ann_w, lgbm_w, gnn_w)
        voted_max = float(np.max(voted))
        if voted_max <= 0:
            threshold_grid = np.array([0.5], dtype=np.float64)
        else:
            lower = max(1e-6, float(np.quantile(voted, 0.001)))
            upper = max(lower, voted_max)
            threshold_grid = np.unique(
                np.concatenate(
                    [
                        np.linspace(lower, upper, 80, dtype=np.float64),
                        np.quantile(voted, [0.90, 0.95, 0.99, 0.995, 0.999]).astype(np.float64),
                    ]
                )
            )
        for thr in threshold_grid:
            y_pred = (voted >= thr).astype(int)
            f1 = f1_score(y, y_pred, zero_division=0)
            precision = precision_score(y, y_pred, zero_division=0)
            tp = int(((y_pred == 1) & (y == 1)).sum())
            fn = int(((y_pred == 0) & (y == 1)).sum())
            recall = tp / max(tp + fn, 1)
            feasible = 1.0 if precision >= precision_floor else 0.0
            # Stage-2 objective: prioritize feasible precision, then maximize F1, then recall.
            score = 100.0 * feasible + f1 + 0.05 * recall + 0.02 * precision
            if (
                score > best_score
                or (
                    abs(score - best_score) < 1e-12
                    and (
                        f1 > best_f1
                        or (
                            abs(f1 - best_f1) < 1e-12
                            and (recall > best_recall or (abs(recall - best_recall) < 1e-12 and precision > best_precision))
                        )
                    )
                )
            ):
                best_score = float(score)
                best_f1 = float(f1)
                best_precision = float(precision)
                best_recall = float(recall)
                best_model = VotingEnsembleModel(
                    ann_weight=float(ann_w),
                    lgbm_weight=float(lgbm_w),
                    gnn_weight=float(gnn_w),
                    threshold=float(thr),
                    ann_transform=ann_transform,
                    lgbm_transform=lgbm_transform,
                    gnn_transform=gnn_transform,
                )

    print(
        f"Voting ensemble tuned: ann_weight={best_model.ann_weight:.2f}, "
        f"lgbm_weight={best_model.lgbm_weight:.2f}, gnn_weight={best_model.gnn_weight:.2f}, "
        f"threshold={best_model.threshold:.3f}"
    )
    print(
        f"Transforms -> ANN: {best_model.ann_transform.method}, "
        f"LightGBM: {best_model.lgbm_transform.method}, "
        f"GNN: {best_model.gnn_transform.method}"
    )
    print(
        f"Validation objective -> f1={best_f1:.4f}, "
        f"precision={best_precision:.4f}, recall={best_recall:.4f}, min_precision={precision_floor:.2f}"
    )
    return best_model


def predict_meta(model, ann_probs, lgbm_probs, gnn_probs):
    ann_probs = _to_1d(ann_probs)
    lgbm_probs = _to_1d(lgbm_probs)
    gnn_probs = _to_1d(gnn_probs)
    ann_probs_cal = _apply_transform(model.ann_transform, ann_probs)
    lgbm_probs_cal = _apply_transform(model.lgbm_transform, lgbm_probs)
    gnn_probs_cal = _apply_transform(model.gnn_transform, gnn_probs)
    return _voting_probs(
        ann_probs_cal, lgbm_probs_cal, gnn_probs_cal, model.ann_weight, model.lgbm_weight, model.gnn_weight
    )
