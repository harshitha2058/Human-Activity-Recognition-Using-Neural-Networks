from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from preprocess import load_and_preprocess_data

ROOT = Path(__file__).resolve().parents[1]


def evaluate_baselines(train_path: str | Path, test_path: str | Path | None = None):
    X_train, y_train, X_test, y_test, label_encoder = load_and_preprocess_data(train_path, test_path)
    y_train_idx = np.argmax(y_train, axis=1)
    y_test_idx = np.argmax(y_test, axis=1)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train_idx)
        predictions = model.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test_idx, predictions) * 100:.2f}%")
        print(classification_report(y_test_idx, predictions, target_names=label_encoder.classes_))


if __name__ == "__main__":
    evaluate_baselines(ROOT / "data" / "train.csv", ROOT / "data" / "test.csv")
