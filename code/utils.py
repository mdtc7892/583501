"""
utils.py — shared helpers for 01_AI_Intrusion_Detection_Framework.ipynb

Implements: ProjectConfig, set_global_seed, get_logger, Timer, load_dataset,
dataset_summary, save_json, clean_dataset, variance_threshold_filter,
remove_correlated_features, save_model, compute_classification_metrics,
get_roc_pr_curve_data.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ProjectConfig:
    project_root: Path = field(default_factory=lambda: Path("."))
    data_raw_dir: Path = field(default_factory=lambda: Path("data/raw"))
    random_state: int = 42

    label_column: str = "Label"
    benign_label: str = "BENIGN"
    test_size: float = 0.15
    val_size: float = 0.15
    n_jobs: int = -1

    def __post_init__(self):
        self.project_root = Path(self.project_root)
        self.data_raw_dir = Path(self.data_raw_dir)
        self.data_processed_dir = self.project_root / "data" / "processed"
        self.models_dir = self.project_root / "models"
        self.results_dir = self.project_root / "results"
        self.figures_dir = self.project_root / "figures"
        self.metrics_dir = self.project_root / "metrics"
        self.reports_dir = self.project_root / "reports"
        self.logs_dir = self.project_root / "logs"

    def make_dirs(self):
        for d in [self.data_processed_dir, self.models_dir, self.results_dir,
                  self.figures_dir, self.metrics_dir, self.reports_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def get_logger(log_dir: Optional[Path] = None, name: str = "ids_pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(Path(log_dir) / "run.log")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


class Timer:
    """Context manager: `with Timer() as t: ...` then `t.elapsed` (seconds)."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start
        return False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(path: Path, random_state: int = 42) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in folder: {path}")
        dfs = [pd.read_csv(f, low_memory=False) for f in files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def dataset_summary(df: pd.DataFrame, label_column: str = "Label") -> dict:
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
        "n_duplicates": int(df.duplicated().sum()),
        "n_missing_values": int(df.isna().sum().sum()),
        "label_column": label_column,
        "class_counts": df[label_column].value_counts().to_dict() if label_column in df.columns else {},
    }


def save_json(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_dataset(df: pd.DataFrame, label_column: str = "Label", logger=None):
    report = {}
    n0 = len(df)

    df = df.drop_duplicates()
    report["duplicates_removed"] = n0 - len(df)

    id_cols = [c for c in ["Flow ID", "Source IP", "Destination IP", "Timestamp", "Source Port"]
               if c in df.columns]
    df = df.drop(columns=id_cols)
    report["identifier_columns_dropped"] = id_cols

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n_before_inf = len(df)
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    report["rows_dropped_inf_or_missing"] = n_before_inf - len(df)

    zero_var_cols = [c for c in numeric_cols if c in df.columns and df[c].nunique(dropna=True) <= 1]
    df = df.drop(columns=zero_var_cols)
    report["zero_variance_columns_dropped"] = zero_var_cols

    report["final_shape"] = list(df.shape)
    if logger:
        logger.info("Cleaning complete: %s", report)
    return df, report


def variance_threshold_filter(df: pd.DataFrame, threshold: float = 1e-4, label_column: str = "Label"):
    X = df.drop(columns=[label_column], errors="ignore")
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in X.columns if c not in numeric_cols]

    vt = VarianceThreshold(threshold=threshold)
    try:
        vt.fit(X[numeric_cols])
        kept = [c for c, keep in zip(numeric_cols, vt.get_support()) if keep]
        dropped = [c for c in numeric_cols if c not in kept]
    except ValueError:
        kept, dropped = numeric_cols, []

    X_out = X[kept + non_numeric_cols].copy()
    if label_column in df.columns:
        X_out[label_column] = df[label_column].values
    return X_out, dropped


def remove_correlated_features(df: pd.DataFrame, threshold: float = 0.95, label_column: str = "Label"):
    X = df.drop(columns=[label_column], errors="ignore")
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    corr = X[numeric_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped = [col for col in upper.columns if any(upper[col] > threshold)]
    X_out = X.drop(columns=dropped)
    if label_column in df.columns:
        X_out[label_column] = df[label_column].values
    return X_out, dropped


# ---------------------------------------------------------------------------
# Models & metrics
# ---------------------------------------------------------------------------
def save_model(model, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def compute_classification_metrics(y_true, y_pred, y_proba) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0,
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)
    except ValueError:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def get_roc_pr_curve_data(y_true, y_proba) -> dict:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return {"fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall}
