"""Reusable utilities for the AI-driven intrusion-detection notebook."""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ProjectConfig:
    data_raw_dir: Path = Path("dataset/raw")
    random_state: int = 42
    label_column: str = "Label"
    benign_label: str = "BENIGN"
    test_size: float = 0.20
    val_size: float = 0.10
    n_jobs: int = -1
    data_processed_dir: Path = Path("dataset/processed")
    logs_dir: Path = Path("logs")
    models_dir: Path = Path("models")
    results_dir: Path = Path("results")

    def __post_init__(self) -> None:
        self.data_raw_dir = Path(self.data_raw_dir)
        self.data_processed_dir = Path(self.data_processed_dir)
        self.logs_dir = Path(self.logs_dir)
        self.models_dir = Path(self.models_dir)
        self.results_dir = Path(self.results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.metrics_dir = self.results_dir / "metrics"
        self.reports_dir = self.results_dir / "reports"

    def make_dirs(self) -> None:
        for directory in (
            self.data_raw_dir.parent if self.data_raw_dir.suffix else self.data_raw_dir,
            self.data_processed_dir,
            self.logs_dir,
            self.models_dir,
            self.figures_dir,
            self.metrics_dir,
            self.reports_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


class Timer:
    def __enter__(self):
        self.started = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *_exc):
        self.elapsed = time.perf_counter() - self.started


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def get_logger(log_dir: Path, name: str = "ids_framework") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler = logging.FileHandler(Path(log_dir) / "framework.log")
        stream_handler = logging.StreamHandler()
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


def load_dataset(path: Path, random_state: int = 42) -> pd.DataFrame:
    path = Path(path)
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    if not files or not all(file.exists() for file in files):
        raise FileNotFoundError(f"No CSV dataset found at: {path}")
    frames = [pd.read_csv(file, low_memory=False) for file in files]
    data = pd.concat(frames, ignore_index=True)
    return data.sample(frac=1, random_state=random_state).reset_index(drop=True)


def dataset_summary(df: pd.DataFrame, label_column: str = "Label") -> dict[str, Any]:
    labels = df[label_column].value_counts(dropna=False).to_dict() if label_column in df else {}
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1024**2), 2),
        "n_duplicates": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "class_distribution": {str(k): int(v) for k, v in labels.items()},
    }


def clean_dataset(df: pd.DataFrame, label_column: str = "Label", logger=None):
    clean = df.copy()
    clean.columns = clean.columns.str.strip()
    rows_before = len(clean)
    duplicates = int(clean.duplicated().sum())
    clean = clean.drop_duplicates()
    numeric = clean.select_dtypes(include=[np.number]).columns
    infinite = int(np.isinf(clean[numeric].to_numpy()).sum()) if len(numeric) else 0
    clean[numeric] = clean[numeric].replace([np.inf, -np.inf], np.nan)
    missing_before = int(clean.isna().sum().sum())
    if len(numeric):
        clean[numeric] = clean[numeric].fillna(clean[numeric].median())
    other = [c for c in clean.columns if c not in numeric and c != label_column]
    for column in other:
        mode = clean[column].mode(dropna=True)
        clean[column] = clean[column].fillna(mode.iloc[0] if not mode.empty else "UNKNOWN")
    if label_column in clean:
        clean = clean.dropna(subset=[label_column])
    report = {
        "rows_before": rows_before,
        "rows_after": int(len(clean)),
        "duplicates_removed": duplicates,
        "infinite_values_replaced": infinite,
        "missing_values_before_imputation": missing_before,
        "missing_values_after": int(clean.isna().sum().sum()),
    }
    if logger:
        logger.info("Cleaned dataset: %s", report)
    return clean.reset_index(drop=True), report


def variance_threshold_filter(df: pd.DataFrame, threshold: float = 0.0):
    numeric = df.select_dtypes(include=[np.number])
    passthrough = df.drop(columns=numeric.columns)
    selector = VarianceThreshold(threshold=threshold)
    transformed = selector.fit_transform(numeric)
    kept = numeric.columns[selector.get_support()].tolist()
    dropped = [c for c in numeric.columns if c not in kept]
    result = pd.DataFrame(transformed, columns=kept, index=df.index)
    return pd.concat([result, passthrough], axis=1), dropped


def remove_correlated_features(df: pd.DataFrame, threshold: float = 0.95):
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped = [column for column in upper.columns if (upper[column] > threshold).any()]
    return df.drop(columns=dropped), dropped


def save_json(data: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value))


def save_model(model: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def compute_classification_metrics(y_true, y_pred, y_proba=None) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "true_negative_rate": float(tn / (tn + fp)) if tn + fp else 0.0,
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        result["pr_auc"] = float(average_precision_score(y_true, y_proba))
    return result


def get_roc_pr_curve_data(y_true, y_proba) -> dict[str, Any]:
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "precision": precision,
        "recall": recall,
        "pr_thresholds": pr_thresholds,
    }
