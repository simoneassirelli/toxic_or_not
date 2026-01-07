from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def split_data(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/val/test with optional stratification."""
    X = df[["text"]].copy()
    y = df[["label"]].copy()

    strat = y["label"] if stratify else None

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )

    # val_size is relative to full dataset; convert to fraction of trainval
    val_frac_of_trainval = val_size / (1.0 - test_size)
    strat_tv = y_trainval["label"] if stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_frac_of_trainval,
        random_state=random_state,
        stratify=strat_tv,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_tfidf_logreg(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    tfidf_params: Dict,
    logreg_params: Dict,
) -> Pipeline:
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                max_features=int(tfidf_params["max_features"]),
                ngram_range=tuple(tfidf_params["ngram_range"]),
                min_df=int(tfidf_params["min_df"]),
            )),
            ("clf", LogisticRegression(
                C=float(logreg_params["C"]),
                max_iter=int(logreg_params["max_iter"]),
                class_weight="balanced",  # helps with imbalance
            )),
        ]
    )

    model.fit(X_train["text"].astype(str).tolist(), y_train["label"].astype(int).values)
    return model


def evaluate_model(
    model: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Dict:
    def _eval_split(X: pd.DataFrame, y: pd.DataFrame) -> Dict:
        y_true = y["label"].astype(int).values
        y_proba = model.predict_proba(X["text"].astype(str).tolist())[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        out = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        # ROC-AUC can fail if a split accidentally has one class (rare if stratified)
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            out["roc_auc"] = None
        return out

    return {
        "val": _eval_split(X_val, y_val),
        "test": _eval_split(X_test, y_test),
    }
