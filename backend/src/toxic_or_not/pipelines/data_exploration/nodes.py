from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import pandas as pd


@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    columns: list[str]
    label_positive_rate: float
    missing_text_rows: int
    empty_text_rows: int
    avg_text_length_chars: float
    p50_text_length_chars: float
    p95_text_length_chars: float


def compute_basic_summary(df: pd.DataFrame) -> Dict:
    # Expect columns: text, label
    text = df["text"].astype(str)

    missing_text_rows = int(df["text"].isna().sum()) if "text" in df.columns else 0
    empty_text_rows = int((text.str.strip() == "").sum())

    lengths = text.str.len()

    summary = DatasetSummary(
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        columns=list(df.columns),
        label_positive_rate=float(df["label"].mean()),
        missing_text_rows=missing_text_rows,
        empty_text_rows=empty_text_rows,
        avg_text_length_chars=float(lengths.mean()),
        p50_text_length_chars=float(lengths.quantile(0.50)),
        p95_text_length_chars=float(lengths.quantile(0.95)),
    )
    return asdict(summary)


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    vc = df["label"].value_counts(dropna=False).sort_index()
    out = vc.rename_axis("label").reset_index(name="count")
    out["fraction"] = out["count"] / out["count"].sum()
    return out


def add_length_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["text_length_chars"] = out["text"].astype(str).str.len()
    out["text_length_words"] = out["text"].astype(str).str.split().str.len()
    return out
