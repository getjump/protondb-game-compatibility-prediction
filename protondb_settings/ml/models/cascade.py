"""Two-stage cascade classifier: works/borked → tinkering/works_oob.

Architecture (from PLAN_ML_4.md experiments):
  Stage 1: borked(0) vs works(1) — all features, class_weight={0:3, 1:1}
  Stage 2: tinkering(0) vs works_oob(1) — drops report_age_days, class_weight={0:1, 1:2}

Key findings:
  - report_age_days useful for Stage 1 but harmful for Stage 2 (temporal bias)
  - Cascade F1=0.593 vs single 0.584 (+0.009)
  - Calibration: ECE 0.018 → 0.012 with isotonic regression
  - Confidence: at P≥0.7, accuracy=88% on 46% of data
"""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .classifier import CATEGORICAL_FEATURES

logger = logging.getLogger(__name__)

# Features to exclude from Stage 2 (temporal bias)
STAGE2_DROP_FEATURES = ["report_age_days"]


def train_stage1(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    categorical_cols: list[str] | None = None,
    class_weight: dict | None = None,
) -> lgb.LGBMClassifier:
    """Train Stage 1: borked (0) vs works (1)."""
    if class_weight is None:
        class_weight = {0: 3.0, 1: 1.0}

    if categorical_cols is None:
        categorical_cols = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    y_train_bin = (y_train > 0).astype(int)
    y_test_bin = (y_test > 0).astype(int)

    model = lgb.LGBMClassifier(
        n_estimators=2000, num_leaves=63, learning_rate=0.03,
        max_depth=-1, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
        class_weight=class_weight, n_jobs=-1, random_state=42,
        verbose=-1, importance_type="gain",
    )

    for col in categorical_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
            X_test[col] = X_test[col].astype("category")

    model.fit(
        X_train, y_train_bin,
        eval_set=[(X_test, y_test_bin)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=200),
        ],
        categorical_feature=categorical_cols,
    )

    logger.info("Stage 1 best iteration: %d", model.best_iteration_)
    return model


def train_stage2(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    categorical_cols: list[str] | None = None,
    drop_features: list[str] | None = None,
    label_smoothing: float = 0.15,
) -> tuple[lgb.Booster, list[str]]:
    """Train Stage 2: tinkering (0) vs works_oob (1), non-borked only.

    Uses cross_entropy objective with noise-robust hyperparameters and
    label smoothing (Phase 9.1 findings: +0.021 F1 over baseline).

    Returns:
        model: trained Booster (predict returns P(works_oob))
        drop_features: features dropped from input
    """
    if drop_features is None:
        drop_features = list(STAGE2_DROP_FEATURES)

    if categorical_cols is None:
        categorical_cols = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    # Filter to non-borked
    train_mask = y_train > 0
    test_mask = y_test > 0

    X_train_s2 = X_train[train_mask].reset_index(drop=True)
    y_train_s2 = (y_train[train_mask] - 1).astype(float)  # 0=tinkering, 1=oob

    X_test_s2 = X_test[test_mask].reset_index(drop=True)
    y_test_s2 = (y_test[test_mask] - 1).astype(float)

    # Drop temporal bias features
    existing_drops = [c for c in drop_features if c in X_train_s2.columns]
    if existing_drops:
        X_train_s2 = X_train_s2.drop(columns=existing_drops)
        X_test_s2 = X_test_s2.drop(columns=existing_drops)
        logger.info("Stage 2: dropped features %s", existing_drops)

    cat_cols_s2 = [c for c in categorical_cols if c in X_train_s2.columns]

    for col in cat_cols_s2:
        X_train_s2[col] = X_train_s2[col].astype("category")
        X_test_s2[col] = X_test_s2[col].astype("category")

    # Apply label smoothing: y_smooth = y*(1-α) + (1-y)*α
    y_smooth = y_train_s2.copy()
    if label_smoothing > 0:
        y_smooth = y_smooth * (1 - label_smoothing) + (1 - y_smooth) * label_smoothing
        logger.info("Stage 2: label smoothing alpha=%.2f", label_smoothing)

    ds_train = lgb.Dataset(X_train_s2, label=y_smooth, categorical_feature=cat_cols_s2)
    ds_test = lgb.Dataset(X_test_s2, label=y_test_s2, categorical_feature=cat_cols_s2)

    # Phase 9.1: cross_entropy + noise-robust params
    params = {
        "objective": "cross_entropy",
        "metric": "binary_logloss",
        "num_leaves": 63,
        "learning_rate": 0.02,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "min_split_gain": 0.05,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        ds_train,
        num_boost_round=3000,
        valid_sets=[ds_test],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )

    logger.info("Stage 2 best iteration: %d", model.best_iteration)
    return model, existing_drops


class CascadeClassifier:
    """Two-stage classifier: borked vs works → tinkering vs works_oob.

    Stage 1 uses all features (LGBMClassifier).
    Stage 2 uses cross_entropy Booster (lgb.train), drops temporal bias features.

    Supports optional post-hoc calibration (isotonic regression per class).
    """

    def __init__(
        self,
        stage1: lgb.LGBMClassifier,
        stage2: lgb.Booster | lgb.LGBMClassifier,
        stage2_drop_features: list[str],
        borked_threshold: float = 0.5,
        oob_threshold: float = 0.5,
        calibrators: dict[int, IsotonicRegression] | None = None,
    ):
        self.stage1 = stage1
        self.stage2 = stage2
        self.stage2_drop_features = stage2_drop_features
        self.borked_threshold = borked_threshold
        self.oob_threshold = oob_threshold
        self.calibrators = calibrators

    def _prepare_stage2_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop features not used by Stage 2."""
        cols_to_drop = [c for c in self.stage2_drop_features if c in X.columns]
        if cols_to_drop:
            return X.drop(columns=cols_to_drop)
        return X

    def _stage2_predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get Stage 2 probabilities: (n, 2) = [P(tinkering), P(oob)].

        Handles both lgb.Booster (cross_entropy, returns P(oob) directly)
        and LGBMClassifier (returns predict_proba (n, 2)).
        """
        if isinstance(self.stage2, lgb.Booster):
            p_oob = self.stage2.predict(X)
            return np.column_stack([1 - p_oob, p_oob])
        return self.stage2.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict 3-class: 0=borked, 1=tinkering, 2=works_oob.

        Uses uncalibrated probabilities with stage thresholds for classification.
        Calibration is for probability output only, not for hard predictions.
        """
        p_s1 = self.stage1.predict_proba(X)
        borked_mask = p_s1[:, 0] >= self.borked_threshold

        result = np.full(len(X), 1, dtype=int)  # default tinkering
        result[borked_mask] = 0

        works_mask = ~borked_mask
        if works_mask.any():
            X_s2 = self._prepare_stage2_input(X[works_mask])
            p_s2 = self._stage2_predict_proba(X_s2)
            result[works_mask] = np.where(p_s2[:, 1] >= self.oob_threshold, 2, 1)

        return result

    def predict_proba(self, X: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        """Predict probabilities: [P(borked), P(tinkering), P(works_oob)].

        Args:
            X: feature DataFrame
            calibrated: if True and calibrators are fitted, return calibrated probabilities
        """
        p_s1 = self.stage1.predict_proba(X)  # (n, 2): [P(borked), P(works)]
        X_s2 = self._prepare_stage2_input(X)
        p_s2 = self._stage2_predict_proba(X_s2)  # (n, 2): [P(tinkering), P(oob)]

        proba = np.zeros((len(X), 3))
        proba[:, 0] = p_s1[:, 0]                    # P(borked)
        proba[:, 1] = p_s1[:, 1] * p_s2[:, 0]       # P(works) × P(tinkering|works)
        proba[:, 2] = p_s1[:, 1] * p_s2[:, 1]       # P(works) × P(oob|works)

        if calibrated and self.calibrators:
            proba = self._calibrate(proba)

        return proba

    def _calibrate(self, proba: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration and renormalize."""
        calibrated = np.zeros_like(proba)
        for cls, iso in self.calibrators.items():
            calibrated[:, cls] = iso.predict(proba[:, cls])
        # Renormalize to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        return calibrated / row_sums

    def predict_with_confidence(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Predict with confidence metadata.

        Returns dict with:
            prediction: int array (0=borked, 1=tinkering, 2=works_oob)
            probabilities: float array (n, 3)
            confidence: float array — max probability per sample
            is_confident: bool array — True if confidence >= 0.7
            stage1_uncertain: bool array — P(borked) in [0.3, 0.7]
            stage2_uncertain: bool array — P(oob|works) in [0.35, 0.65] for non-borked
        """
        proba = self.predict_proba(X)
        prediction = proba.argmax(axis=1)
        confidence = proba.max(axis=1)

        # Stage-level uncertainty
        p_s1 = self.stage1.predict_proba(X)
        p_borked = p_s1[:, 0]
        stage1_uncertain = (p_borked >= 0.3) & (p_borked < 0.7)

        # Stage 2 uncertainty for non-borked predictions
        stage2_uncertain = np.zeros(len(X), dtype=bool)
        works_mask = prediction > 0
        if works_mask.any():
            X_s2 = self._prepare_stage2_input(X[works_mask])
            p_s2 = self._stage2_predict_proba(X_s2)
            p_oob = p_s2[:, 1]
            s2_unc = (p_oob >= 0.35) & (p_oob < 0.65)
            stage2_uncertain[works_mask] = s2_unc

        return {
            "prediction": prediction,
            "probabilities": proba,
            "confidence": confidence,
            "is_confident": confidence >= 0.7,
            "stage1_uncertain": stage1_uncertain,
            "stage2_uncertain": stage2_uncertain,
        }

    def fit_calibrators(
        self, X_cal: pd.DataFrame, y_cal: np.ndarray,
    ) -> None:
        """Fit isotonic calibration on a calibration set.

        Should be called with held-out data (not used for training).
        """
        proba_raw = self.predict_proba(X_cal, calibrated=False)
        self.calibrators = {}
        for cls in range(3):
            y_bin = (y_cal == cls).astype(int)
            iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
            iso.fit(proba_raw[:, cls], y_bin)
            self.calibrators[cls] = iso
        logger.info("Fitted isotonic calibrators on %d samples", len(y_cal))
