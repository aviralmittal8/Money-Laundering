from lightgbm import LGBMClassifier


def build_lgbm_model(pos_weight, n_estimators=400, random_state=42):
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        scale_pos_weight=float(pos_weight),
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )
