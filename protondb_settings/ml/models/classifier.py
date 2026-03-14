"""LightGBM classifier for compatibility prediction."""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Target mapping
TARGET_NAMES = {0: "borked", 1: "needs_tinkering", 2: "works_oob"}
TARGET_VALUES = {"borked": 0, "needs_tinkering": 1, "works_oob": 2}

# Categorical feature names (for LightGBM native handling)
CATEGORICAL_FEATURES = [
    "gpu_vendor",
    "gpu_family",
    "gpu_tier",
    "cpu_vendor",
    "os_family",
    "engine",
    "anticheat",
    "anticheat_status",
    "genre",
    "variant",
]


def compute_target(verdict: str | None, verdict_oob: str | None) -> int | None:
    """Compute multi-class target from verdict fields.

    0 = borked, 1 = needs_tinkering, 2 = works_oob
    """
    if verdict_oob == "yes":
        return 2  # works_oob
    if verdict_oob == "no" and verdict == "yes":
        return 1  # needs_tinkering
    if verdict == "no":
        return 0  # borked
    # verdict_oob is null, verdict is yes -> tinkering
    if verdict == "yes":
        return 1  # needs_tinkering
    return None


def train_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    categorical_cols: list[str] | None = None,
) -> lgb.LGBMClassifier:
    """Train a LightGBM classifier.

    Args:
        X_train: Training features.
        y_train: Training targets (0/1/2).
        X_test: Test features (for early stopping).
        y_test: Test targets.
        categorical_cols: List of categorical column names.

    Returns:
        Trained LGBMClassifier.
    """
    if categorical_cols is None:
        categorical_cols = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    # Convert categorical columns to pandas Categorical type
    for col in categorical_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
            X_test[col] = X_test[col].astype("category")

    model = lgb.LGBMClassifier(
        n_estimators=2000,
        num_leaves=63,
        learning_rate=0.03,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight={0: 2.0, 1: 1.0, 2: 3.0},
        n_jobs=-1,
        random_state=42,
        verbose=-1,
        importance_type="gain",
    )

    logger.info(
        "Training LightGBM: %d train, %d test, %d features, %d categorical",
        len(X_train),
        len(X_test),
        X_train.shape[1],
        len(categorical_cols),
    )

    eval_set = [(X_test, y_test)]

    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
        categorical_feature=categorical_cols,
    )

    logger.info("Best iteration: %d", model.best_iteration_)
    return model
