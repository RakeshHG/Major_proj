from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def evaluate_results_csv(results_csv: Path) -> Dict[str, float]:
    df = pd.read_csv(results_csv)
    y_true = df["true_binary"].astype(int)
    y_pred = df["pred_binary"].astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "avg_latency_ms": float(df["latency_ms"].mean()) if "latency_ms" in df.columns else 0.0,
    }
