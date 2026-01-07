from __future__ import annotations

import pandas as pd


def download_toxicity_dataset(
    data_dir: str,
    max_rows: int | None,
) -> pd.DataFrame:
    """
    Load the Jigsaw Toxic Comment dataset from a manually downloaded CSV
    and convert it to a binary toxicity classification dataset.

    Output columns:
      - text: str
      - label: int (0/1)
    """
    csv_path = f"{data_dir}/train.csv"
    df = pd.read_csv(csv_path)

    if max_rows is not None:
        df = df.head(max_rows)

    # Text column
    df_out = pd.DataFrame()
    df_out["text"] = df["comment_text"].astype(str)

    # Binary label: toxic if ANY toxicity label is 1
    toxicity_cols = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    df_out["label"] = (df[toxicity_cols].sum(axis=1) > 0).astype(int)

    # Drop empty text rows
    df_out = df_out[df_out["text"].str.strip().astype(bool)].reset_index(drop=True)

    return df_out
