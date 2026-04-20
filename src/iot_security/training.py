from pathlib import Path
import pickle
from typing import Dict

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .constants import LABEL_COL
from .cuckoo_filter import CuckooFilter
from .pipeline import make_signature
from .preprocessing import preprocess_dataset, save_preprocessed_artifacts


def build_cuckoo_filter(df, capacity: int = 50000) -> CuckooFilter:
    attack_df = df[df[LABEL_COL] != "Benign"]
    benign_df = df[df[LABEL_COL] == "Benign"]

    attack_signatures = set(attack_df.apply(make_signature, axis=1).unique())
    benign_signatures = set(benign_df.apply(make_signature, axis=1).unique())
    exclusive_attack_signatures = attack_signatures - benign_signatures

    cuckoo = CuckooFilter(capacity=capacity)
    for sig in exclusive_attack_signatures:
        cuckoo.insert(sig)
    return cuckoo


def train_models(x_train_scaled, y_train) -> Dict[str, object]:
    benign_mask = y_train == 0
    x_benign = x_train_scaled[benign_mask]

    iso = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(x_benign)

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(x_train_scaled, y_train)
    return {"isolation_forest": iso, "random_forest": rf}


def evaluate_binary_from_iso(iso_model, x_test_scaled, y_test) -> Dict[str, float]:
    iso_preds = iso_model.predict(x_test_scaled)
    y_true = (y_test != 0).astype(int)
    y_pred = (iso_preds == -1).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def save_models(models_dir: Path, cuckoo_filter, iso_model, rf_model) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    with (models_dir / "cuckoo_filter.pkl").open("wb") as f:
        pickle.dump(cuckoo_filter, f)
    with (models_dir / "isolation_forest.pkl").open("wb") as f:
        pickle.dump(iso_model, f)
    with (models_dir / "random_forest.pkl").open("wb") as f:
        pickle.dump(rf_model, f)


def run_training_pipeline(
    data_csv: Path,
    processed_dir: Path,
    models_dir: Path,
):
    prep = preprocess_dataset(data_csv)
    df = prep["df"]
    x_train_scaled = prep["x_train_scaled"]
    x_test_scaled = prep["x_test_scaled"]
    y_train = prep["y_train"]
    y_test = prep["y_test"]
    scaler = prep["scaler"]
    label_encoder = prep["label_encoder"]

    save_preprocessed_artifacts(
        processed_dir,
        models_dir,
        x_train_scaled,
        x_test_scaled,
        y_train,
        y_test,
        scaler,
        label_encoder,
    )

    cuckoo = build_cuckoo_filter(df)
    models = train_models(x_train_scaled, y_train)
    save_models(models_dir, cuckoo, models["isolation_forest"], models["random_forest"])

    metrics = evaluate_binary_from_iso(models["isolation_forest"], x_test_scaled, y_test)
    return {
        "cuckoo_filter": cuckoo,
        "isolation_forest": models["isolation_forest"],
        "random_forest": models["random_forest"],
        "scaler": scaler,
        "label_encoder": label_encoder,
        "iso_metrics": metrics,
    }
