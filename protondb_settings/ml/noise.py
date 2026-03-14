"""Cleanlab-based noise detection for training data.

Phase 9.3: Removes top 3% suspected mislabels identified via
out-of-fold confident learning. Gives +0.021 F1 improvement.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def _cache_key(y_train: np.ndarray, n_features: int, frac_remove: float) -> str:
    """Compute a cache key from label distribution + feature count + frac."""
    h = hashlib.sha256()
    h.update(y_train.tobytes())
    h.update(f"{n_features}:{frac_remove}".encode())
    return h.hexdigest()[:16]


def find_noisy_samples(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_folds: int = 5,
    frac_remove: float = 0.03,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> np.ndarray:
    """Find and flag likely mislabeled samples using Cleanlab confident learning.

    Trains 5-fold OOF LightGBM to get predicted probabilities, then uses
    Cleanlab to rank samples by label quality. Returns boolean keep mask.

    Args:
        X_train: feature DataFrame
        y_train: integer labels (0=borked, 1=tinkering, 2=works_oob)
        n_folds: number of CV folds for OOF predictions
        frac_remove: fraction of samples to remove (0.03 = 3%)
        cache_dir: directory for caching the keep mask (default: data/)
        force: if True, ignore cached mask and recompute

    Returns:
        Boolean mask: True = keep, False = suspected mislabel.
    """
    # Try loading from cache
    cache_path = None
    if cache_dir is not None:
        key = _cache_key(y_train, X_train.shape[1], frac_remove)
        cache_path = Path(cache_dir) / f"cleanlab_mask_{key}.npy"
        if not force and cache_path.exists():
            keep_mask = np.load(cache_path)
            if len(keep_mask) == len(y_train):
                n_removed = (~keep_mask).sum()
                logger.info("Cleanlab: loaded cached mask from %s (%d removed)", cache_path, n_removed)
                return keep_mask
            logger.warning("Cleanlab cache size mismatch (%d vs %d), recomputing",
                           len(keep_mask), len(y_train))

    from cleanlab.filter import find_label_issues

    from .models.classifier import CATEGORICAL_FEATURES

    # Build out-of-fold predicted probabilities
    pred_proba = np.zeros((len(X_train), 3))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr = X_train.iloc[train_idx].copy()
        y_tr = y_train[train_idx]
        X_val = X_train.iloc[val_idx].copy()

        for col in cat_cols:
            X_tr[col] = X_tr[col].astype("category")
            X_val[col] = X_val[col].astype("category")

        model = lgb.LGBMClassifier(
            n_estimators=500, num_leaves=63, learning_rate=0.05,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=42, verbose=-1,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_train[val_idx])],
            callbacks=[lgb.early_stopping(30, verbose=False)],
            categorical_feature=cat_cols,
        )
        pred_proba[val_idx] = model.predict_proba(X_val)

    # Find label issues ranked by self-confidence
    issues = find_label_issues(
        labels=y_train,
        pred_probs=pred_proba,
        return_indices_ranked_by="self_confidence",
    )

    n_remove = int(len(y_train) * frac_remove)
    remove_set = set(issues[:n_remove])

    keep_mask = np.ones(len(y_train), dtype=bool)
    keep_mask[list(remove_set)] = False

    # Log stats
    removed_labels = y_train[~keep_mask]
    logger.info(
        "Cleanlab: %d issues found, removing %d (%.1f%%). "
        "Per class: borked=%d, tinkering=%d, works_oob=%d",
        len(issues), n_remove, frac_remove * 100,
        (removed_labels == 0).sum(),
        (removed_labels == 1).sum(),
        (removed_labels == 2).sum(),
    )

    # Save to cache
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, keep_mask)
        logger.info("Cleanlab: cached mask to %s", cache_path)

    return keep_mask
