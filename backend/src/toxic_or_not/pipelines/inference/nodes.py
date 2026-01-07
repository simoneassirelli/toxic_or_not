from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def tune_threshold_for_f1(
    model,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
) -> Dict:
    """
    Find the threshold in [0.05, 0.95] that maximizes F1 on validation set.
    Saves chosen threshold + summary stats.
    """
    from sklearn.metrics import f1_score

    texts = X_val["text"].astype(str).tolist()
    y_true = y_val["label"].astype(int).values
    proba = model.predict_proba(texts)[:, 1]

    best = {"threshold": 0.5, "f1": -1.0}
    for t in np.linspace(0.05, 0.95, 19):
        y_pred = (proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": float(f1)}

    best["computed_at_utc"] = datetime.now(timezone.utc).isoformat()
    best["val_positive_rate"] = float(y_true.mean())
    return best


def package_model_card(
    metrics: Dict,
    threshold_info: Dict,
) -> Dict:
    """A tiny model card / metadata blob to ship with the artifacts."""
    return {
        "model_type": "sklearn_pipeline_tfidf_logreg",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "threshold": threshold_info,
        "inputs": {"field": "text", "type": "string"},
        "outputs": {"field": "toxicity_score", "type": "float", "label_thresholded": True},
    }
