"""Export trained model artifacts: model.pkl, embeddings.npz, label_maps.json."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib

from .features.embeddings import save_embeddings
from .features.encoding import LabelMaps

logger = logging.getLogger(__name__)


def export_model(model: Any, path: Path) -> None:
    """Save trained model with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)


def load_model(path: Path) -> Any:
    """Load trained model with joblib."""
    return joblib.load(path)


def export_all(
    model: Any,
    emb_data: dict[str, Any],
    label_maps: LabelMaps,
    output_dir: Path,
) -> dict[str, Path]:
    """Export all ML artifacts to the output directory.

    Returns a dict of artifact name -> path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.pkl"
    embeddings_path = output_dir / "embeddings.npz"
    label_maps_path = output_dir / "label_maps.json"

    export_model(model, model_path)
    save_embeddings(emb_data, embeddings_path)
    label_maps.save(label_maps_path)

    logger.info("All artifacts exported to %s", output_dir)
    return {
        "model": model_path,
        "embeddings": embeddings_path,
        "label_maps": label_maps_path,
    }
