"""Per-game aggregated prediction from individual report predictions.

Production pipeline:
  1. Per-report model predicts each report independently (F1=0.780)
  2. Aggregate predictions per (game) or (game, vendor) via majority vote
  3. Result: per-game prediction (F1=0.871 3-class, 0.943 binary)

Aggregation is at inference time — no leakage.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import Counter
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

TARGET_NAMES = {0: "borked", 1: "tinkering", 2: "works_oob"}


def aggregate_predictions(
    predictions: list[int],
    probabilities: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Aggregate per-report predictions into a single verdict.

    Args:
        predictions: list of per-report class predictions (0/1/2)
        probabilities: optional list of per-report probability arrays (n, 3)

    Returns:
        dict with verdict, confidence, breakdown, n_reports
    """
    n = len(predictions)
    if n == 0:
        return {
            "verdict": None,
            "verdict_binary": None,
            "confidence": 0.0,
            "n_reports": 0,
            "breakdown": {"borked": 0, "tinkering": 0, "works_oob": 0},
        }

    counts = Counter(predictions)
    majority = counts.most_common(1)[0][0]
    agreement = counts[majority] / n

    # Binary: borked vs works
    binary_counts = Counter(int(p > 0) for p in predictions)
    binary_majority = binary_counts.most_common(1)[0][0]
    binary_agreement = binary_counts[binary_majority] / n

    # Confidence from agreement + sample size
    size_factor = min(1.0, np.log1p(n) / np.log1p(20))  # saturates at ~20 reports
    confidence = 0.4 * agreement + 0.3 * size_factor

    # Add model probability confidence if available
    if probabilities is not None and len(probabilities) > 0:
        mean_proba = np.mean(probabilities, axis=0)
        model_conf = float(mean_proba.max())
        confidence += 0.3 * model_conf
    else:
        confidence += 0.3 * agreement  # fallback

    return {
        "verdict": TARGET_NAMES[majority],
        "verdict_binary": "works" if binary_majority > 0 else "borked",
        "confidence": round(float(confidence), 3),
        "agreement": round(float(agreement), 3),
        "n_reports": n,
        "breakdown": {
            "borked": counts.get(0, 0),
            "tinkering": counts.get(1, 0),
            "works_oob": counts.get(2, 0),
        },
        "breakdown_pct": {
            "borked": round(counts.get(0, 0) / n, 3),
            "tinkering": round(counts.get(1, 0) / n, 3),
            "works_oob": round(counts.get(2, 0) / n, 3),
        },
    }


def predict_for_game(
    app_id: int,
    conn: sqlite3.Connection,
    cascade,
    feature_builder,
    *,
    vendor: str | None = None,
    is_deck: bool | None = None,
) -> dict[str, Any]:
    """Predict compatibility for a game by aggregating all historical reports.

    Args:
        app_id: Steam app ID
        conn: database connection
        cascade: trained CascadeClassifier
        feature_builder: callable(conn, app_id, report_ids) → (X, report_ids)
        vendor: optional GPU vendor filter ("nvidia", "amd", "intel")
        is_deck: optional Steam Deck filter

    Returns:
        dict with aggregated prediction, confidence, breakdown
    """
    # Build filter
    where = ["r.app_id = ?"]
    params: list[Any] = [app_id]

    if vendor:
        vendor_patterns = {
            "nvidia": ("NVIDIA", "GeForce", "GTX", "RTX"),
            "amd": ("AMD", "Radeon", "RX "),
            "intel": ("Intel",),
        }
        patterns = vendor_patterns.get(vendor, ())
        if patterns:
            or_clauses = " OR ".join(f"r.gpu LIKE '%{p}%'" for p in patterns)
            where.append(f"({or_clauses})")

    if is_deck is True:
        where.append("(r.gpu LIKE '%anGogh%' OR r.gpu LIKE '%an Gogh%' OR r.battery_performance IS NOT NULL)")
    elif is_deck is False:
        where.append("(r.gpu NOT LIKE '%anGogh%' AND r.gpu NOT LIKE '%an Gogh%' AND r.battery_performance IS NULL)")

    where_clause = " AND ".join(where)

    report_ids = [r["id"] for r in conn.execute(
        f"SELECT id FROM reports r WHERE {where_clause}", params
    ).fetchall()]

    if not report_ids:
        return {
            "app_id": app_id,
            "verdict": None,
            "confidence": 0.0,
            "n_reports": 0,
            "note": "No matching reports found",
        }

    # Build features and predict
    X, valid_rids = feature_builder(conn, app_id, report_ids)

    if X is None or len(X) == 0:
        return {
            "app_id": app_id,
            "verdict": None,
            "confidence": 0.0,
            "n_reports": len(report_ids),
            "note": "Could not build features",
        }

    predictions = cascade.predict(X)
    try:
        probabilities = cascade.predict_proba(X)
    except Exception:
        probabilities = None

    agg = aggregate_predictions(
        predictions.tolist(),
        [probabilities[i] for i in range(len(probabilities))] if probabilities is not None else None,
    )

    return {
        "app_id": app_id,
        "filter": {
            "vendor": vendor,
            "is_deck": is_deck,
        },
        **agg,
    }
