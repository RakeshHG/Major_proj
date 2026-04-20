from pathlib import Path
import pickle
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .constants import FEATURE_COLS, LABEL_COL, RAW_DROP_COLS


def load_and_clean_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    drop_cols = [col for col in RAW_DROP_COLS if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))
    return df


def preprocess_dataset(
    csv_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    df = load_and_clean_dataset(csv_path)

    x = df[FEATURE_COLS].values.astype(np.float32)
    y_raw = df[LABEL_COL].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return {
        "df": df,
        "x_train_scaled": x_train_scaled,
        "x_test_scaled": x_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "label_encoder": label_encoder,
    }


def save_preprocessed_artifacts(
    output_data_dir: Path,
    output_models_dir: Path,
    x_train_scaled: np.ndarray,
    x_test_scaled: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    scaler: StandardScaler,
    label_encoder: LabelEncoder,
) -> None:
    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_models_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_data_dir / "X_train.npy", x_train_scaled)
    np.save(output_data_dir / "X_test.npy", x_test_scaled)
    np.save(output_data_dir / "y_train.npy", y_train)
    np.save(output_data_dir / "y_test.npy", y_test)

    with (output_models_dir / "scaler.pkl").open("wb") as f:
        pickle.dump(scaler, f)
    with (output_models_dir / "label_encoder.pkl").open("wb") as f:
        pickle.dump(label_encoder, f)
    with (output_models_dir / "feature_cols.pkl").open("wb") as f:
        pickle.dump(FEATURE_COLS, f)
