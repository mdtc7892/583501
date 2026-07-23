"""
Stage 1 — Data Loading & Preprocessing
Loads the raw CICIDS2017 Friday-Afternoon-DDoS CSV, filters to
BENIGN/DDoS flows, drops identifier columns, removes inf/NaN and
duplicate rows, and writes a cleaned feature file to dataset/.

Run:  python src/data_preprocessing.py
"""
import os
import numpy as np
import pandas as pd

from config import RAW_CSV, CLEAN_CSV


def load_dataset(path: str = RAW_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place "
            "'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv' inside dataset/."
        )
    print(f"Loading dataset: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_and_filter(df: pd.DataFrame):
    df = df[df["Label"].isin(["BENIGN", "DDoS"])].copy()

    drop_cols = [c for c in ["Flow ID", "Source IP", "Destination IP",
                              "Timestamp", "Source Port"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()

    y = (df["Label"] == "DDoS").astype(int)
    X = df.drop(columns=["Label"])
    return X, y


def main():
    df_raw = load_dataset()
    X, y = clean_and_filter(df_raw)

    print("Data shape after cleaning:", X.shape)
    print("Class distribution:\n", y.value_counts())

    clean_df = X.copy()
    clean_df["Label"] = y
    clean_df.to_csv(CLEAN_CSV, index=False)
    print(f"Cleaned dataset saved to: {CLEAN_CSV}")
    return X, y


if __name__ == "__main__":
    main()
