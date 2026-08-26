"""Dataset loading and preprocessing shared by all training scripts."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _csv_file(path: str | Path) -> Path:
    """Return a CSV file, accepting a CSV path or a directory containing one."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        csv_files = list(candidate.glob("*.csv"))
        if len(csv_files) == 1:
            return csv_files[0]
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in directory: {candidate}")
        raise ValueError(f"More than one CSV file found in directory: {candidate}")
    raise FileNotFoundError(f"Dataset file does not exist: {candidate}")


def _features_and_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    label_column = "Activity" if "Activity" in frame.columns else frame.columns[-1]
    feature_columns = [column for column in frame.columns if column not in {label_column, "subject"}]
    if not feature_columns:
        raise ValueError("Dataset contains no feature columns.")
    return frame[feature_columns], frame[label_column]


def load_and_preprocess_data(
    train_path: str | Path, test_path: str | Path | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """Load HAR data, scale features, and return one-hot encoded labels."""
    train_file = _csv_file(train_path)
    train_df = pd.read_csv(train_file)

    if test_path is not None and Path(test_path).exists():
        test_df = pd.read_csv(_csv_file(test_path))
    else:
        print("Note: separate test.csv not found; splitting training data (80% / 20%).")
        label_column = "Activity" if "Activity" in train_df.columns else train_df.columns[-1]
        train_df, test_df = train_test_split(
            train_df, test_size=0.2, random_state=42, stratify=train_df[label_column]
        )

    X_train_frame, y_train_raw = _features_and_labels(train_df)
    X_test_frame, y_test_raw = _features_and_labels(test_df)
    if list(X_train_frame.columns) != list(X_test_frame.columns):
        raise ValueError("Train and test CSV files do not have matching feature columns.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_frame)
    X_test = scaler.transform(X_test_frame)

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train_raw)
    try:
        y_test_encoded = encoder.transform(y_test_raw)
    except ValueError as error:
        raise ValueError("Test data contains an activity not present in training data.") from error

    class_count = len(encoder.classes_)
    y_train = np.eye(class_count, dtype=np.float32)[y_train_encoded]
    y_test = np.eye(class_count, dtype=np.float32)[y_test_encoded]
    return X_train, y_train, X_test, y_test, encoder
